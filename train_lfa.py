
import argparse
import copy
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import settings as C
import data as D
from alignment import VoxelAdapter, loss_fn, validation_specificity


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained(subject, n_voxels, device):
    if not C.DYNADIFF_REPO.exists():
        raise FileNotFoundError(
            f"the pretrained decoder was not found at {C.DYNADIFF_REPO}.\n"
            "It is an external dependency: clone the DynaDiff repository and obtain the\n"
            "subject checkpoints, then set DYNADIFF_REPO and DYNADIFF_CKPT in settings.py.")

    sys.path.insert(0, str(C.DYNADIFF_REPO))
    cwd = os.getcwd()
    os.chdir(C.DYNADIFF_REPO)
    try:
        from config.cfg import get_cfg
        from model.models import VersatileDiffusionConfig

        cfg = get_cfg(subject=subject, averaged_trial=False, cache=C.CACHE_DIR, seed=42,
                      vd_cache_dir=C.VD_CACHE_DIR, custom_infra=None)
        model = VersatileDiffusionConfig(**cfg["versatilediffusion_config"]).build(
            brain_n_in_channels=n_voxels, brain_temp_dim=C.T)
    finally:
        os.chdir(cwd)

    raw = torch.load(C.DYNADIFF_CKPT.format(subject=subject), map_location="cpu")
    state = raw["state_dict"] if "state_dict" in raw else raw
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    rename = {".query.": ".to_q.", ".key.": ".to_k.", ".value.": ".to_v.",
              ".proj_attn.": ".to_out.0."}
    fixed = {}
    for k, v in state.items():
        for old, new in rename.items():
            k = k.replace(old, new)
        fixed[k] = v
    missing, _ = model.load_state_dict(fixed, strict=False)
    brain = [k for k in missing if "brain" in k]
    if brain:
        print(f"  WARNING: {len(brain)} brain-module weights were not loaded, "
              f"e.g. {brain[:3]}")

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def main(subject, stim_set, partition, device, temporal_mixing=True):
    set_seed(42)
    n_vox_in = C.N_VOXELS[subject]          # equals V_vis when the input is nsdgeneral
    n_vox_out = C.N_VOXELS[subject]

    
    print(f"sub{subject} set {stim_set} partition {partition}")
    dataset = D.build_dataset(
        C.DATA_DIR / f"imagery_nsd_IMG_{stim_set}_sub{subject}.pt",
        C.AUG_DIR / f"fmri_vis_nsd_{C.K_AUG}_sub{subject}.pt",
        stim_set, seed=partition)

    to_t = lambda k: torch.tensor(dataset[k])
    train_loader = DataLoader(
        TensorDataset(to_t("X_train"), to_t("Y_train"), to_t("lab_train")),
        batch_size=C.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(to_t("X_val"), to_t("Y_val"), to_t("lab_val")),
        batch_size=C.BATCH_SIZE, shuffle=False)

    
    model = load_pretrained(subject, n_vox_out, device)
    brain_module = model.brain_modules["clip_image"].to(device)

    def embed(x):
        """Embedding predicted by the frozen brain module from a voxel tensor."""
        ids = torch.zeros(len(x), dtype=torch.long, device=device)
        return brain_module(x, subject_ids=ids)["MSELoss"]

    
    adapter = VoxelAdapter(n_vox_in, n_vox_out, temporal_mixing=temporal_mixing).to(device)
    optimiser = torch.optim.AdamW(adapter.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    print(f"  trainable parameters: {sum(p.numel() for p in adapter.parameters()) / 1e6:.1f} M")

    
    best, best_weights, waited = -np.inf, copy.deepcopy(adapter.state_dict()), 0
    for epoch in range(C.EPOCHS):
        adapter.train()
        running = 0.0
        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(device).float(), yb.to(device).float(), lb.to(device)
            with torch.no_grad():
                target = embed(yb)                  # embedding of the perceptual trial
            pred = embed(adapter(xb))               # embedding of the aligned imagery trial
            loss = loss_fn(pred, target, lb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += loss.item()

        spec = validation_specificity(adapter, val_loader, embed, device)
        star = " *" if spec > best else ""
        print(f"  epoch {epoch:2d} | loss {running / len(train_loader):.5f} | "
              f"val specificity {spec:+.4f}{star}")

        if spec > best:
            best, best_weights, waited = spec, copy.deepcopy(adapter.state_dict()), 0
        else:
            waited += 1
            if waited >= C.PATIENCE:
                print(f"  early stopping at epoch {epoch}")
                break

    adapter.load_state_dict(best_weights)
    adapter.eval()

    
    out = C.OUT_DIR / f"sub{subject}" / f"set{stim_set}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), out / f"adapter_p{partition}.pt")

    with torch.no_grad():
        X_test = torch.tensor(dataset["X_test"]).to(device).float()
        aligned = adapter(X_test).cpu()
    torch.save({"baseline": X_test.cpu(), "aligned": aligned,
                "stimuli": list(dataset["stimuli"])}, out / f"test_inputs_p{partition}.pt")
    print(f"  best validation specificity {best:+.4f} -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, required=True, choices=C.SUBJECTS)
    p.add_argument("--set", dest="stim_set", required=True, choices=["B", "C"],
                   help="A is not aligned: simple stimuli lie outside the NSD distribution")
    p.add_argument("--partition", type=int, default=0,
                   help=f"one of {C.PARTITIONS}; it seeds the trial split")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-temporal-mixing", action="store_true",
                   help="ablate the 1x1 convolution over time")
    a = p.parse_args()
    main(a.subject, a.stim_set, a.partition, a.device, not a.no_temporal_mixing)
