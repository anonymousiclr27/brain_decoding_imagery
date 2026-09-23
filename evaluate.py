"""Generation and evaluation of the reconstructions.

Two modes, one per stimulus class.

`--set B` (complex stimuli, a real target image exists)
    Eight reconstruction metrics. Two are pixel-level, PixCorr and SSIM; four are
    two-way identification scores computed on the features of pretrained networks
    (AlexNet layers 2 and 5, InceptionV3, CLIP ViT-L/14); two are distances
    (EfficientNet-B1, SwAV), where lower is better. In the identification metrics a
    reconstruction is compared with its own target and with 11 distractors, so chance
    is 50%. The distractors are the *other targets* — the six simple and six complex
    stimuli minus the correct one — and not the other reconstructions, which keeps
    the score comparable with the benchmark even though only six images are generated.

`--set C` (conceptual stimuli, no ground-truth image)
    The reconstruction is described by BLIP and the caption is compared with the
    target word in a sentence-embedding space (MPNet). The detour through language is
    deliberate: the loss, the conditioning of the diffusion model and the CLIP metric
    all live in the same CLIP encoder, whose two towers are aligned by construction,
    so anchoring to CLIP text would not break the circle. Neither BLIP nor MPNet
    appears anywhere in the pipeline.

Usage
-----
    python evaluate.py --subject 1 --set B --partition 0 --device cuda:0
    python evaluate.py --subject 1 --set C --partition 0 --device cuda:0
"""
import argparse
import json

import numpy as np
import torch

import settings as C


# ---------------------------------------------------------------------- generation
def generate(model, brain, device, seed=C.GEN_SEED):
    """One reconstruction per input, with a fixed sampling seed.

    The seed is fixed and shared by every method, so differences between conditions
    come from the conditioning embedding and not from the sampling noise.
    """
    with torch.no_grad():
        out = model(brain.to(device).float(),
                    torch.zeros(len(brain), dtype=torch.long, device=device),
                    None, is_img_gen_mode=True, set_seed=seed)
    return out.image.cpu()


# ----------------------------------------------------------------- complex stimuli
def reconstruction_metrics(targets12, generated, offset, device):
    """Eight metrics for `generated`, against the 12 targets with 11 distractors.

    `offset` is the position in `targets12` of the target of the first generated
    image (6 when the six complex stimuli follow the six simple ones).
    """
    import scipy as sp
    from skimage.color import rgb2gray
    from skimage.metrics import structural_similarity as ssim
    from torchvision.transforms import ToPILImage

    import metrics_lib

    to_pil = ToPILImage()
    trues = [to_pil(torch.as_tensor(i).clamp(0, 1)) for i in targets12]
    preds = [to_pil(i.clamp(0, 1)) for i in generated]
    feats = metrics_lib._compute_image_generation_features(trues + preds, device=device)

    n, m = len(trues), len(preds)
    result = {}
    for name, v in feats.items():
        gt, pr = v[:n].reshape(n, -1), v[n:].reshape(m, -1)
        if name.split("-")[0] in ("efficientnet", "swav"):
            # distance metrics: correlation distance to the correct target
            result[name] = float(np.mean([sp.spatial.distance.correlation(gt[offset + i], pr[i])
                                          for i in range(m)]))
        else:
            # two-way identification against the 11 distractors
            r = np.corrcoef(gt, pr)[:n, n:]
            congruent = r[np.arange(m) + offset, np.arange(m)]
            result[name] = float(np.mean((r < congruent).sum(0) / (n - 1)))

    pix, struct = [], []
    for i in range(m):
        a = np.array(preds[i].resize((425, 425))) / 255.0
        b = np.array(trues[offset + i].resize((425, 425))) / 255.0
        pix.append(np.corrcoef(b.reshape(-1), a.reshape(-1))[0, 1])
        struct.append(ssim(rgb2gray(a), rgb2gray(b), gaussian_weights=True, sigma=1.5,
                           use_sample_covariance=False, data_range=1.0))
    result["pixcorr"] = float(np.mean(pix))
    result["ssim"] = float(np.mean(struct))
    return result


def twelve_targets(subject):
    """The six simple and the six complex target images, in this order."""
    images = []
    for stim_set in ("A", "B"):
        d = torch.load(C.DATA_DIR / f"imagery_nsd_IMG_{stim_set}_sub{subject}.pt",
                       map_location="cpu", weights_only=False)
        ids, imgs = np.array(d["stim_ids"]), d["images_tensor"]
        order = C.BLOCK_ORDER["B"] if stim_set == "B" else sorted(set(ids.tolist()))
        images += [imgs[np.where(ids == s)[0][0]] for s in order]
    return torch.stack(images)


# -------------------------------------------------------------- conceptual stimuli
def caption_similarity(generated, words, device):
    """Cosine similarity between the caption of each reconstruction and its word."""
    from torchvision.transforms import ToPILImage
    from transformers import (AutoModel, AutoTokenizer, BlipForConditionalGeneration,
                              BlipProcessor)

    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large").to(device).eval()
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    sbert = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(device).eval()

    def sentence(texts):
        """Sentence embedding: mask-weighted mean of the token states, then L2."""
        b = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            h = sbert(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=-1)

    to_pil = ToPILImage()
    batch = proc(images=[to_pil(g.clamp(0, 1)) for g in generated], return_tensors="pt").to(device)
    with torch.no_grad():
        ids = blip.generate(**batch, max_new_tokens=30, num_beams=3)
    captions = [proc.decode(i, skip_special_tokens=True).strip() for i in ids]

    sim = (sentence(captions) @ sentence([f"a photo of {w}" for w in words]).T).cpu().numpy()
    return captions, sim


# ----------------------------------------------------------------------------- main
def main(subject, stim_set, partition, device, condition):
    from train_lfa import load_pretrained

    run_dir = C.OUT_DIR / f"sub{subject}" / f"set{stim_set}"
    test = torch.load(run_dir / f"test_inputs_p{partition}.pt",
                      map_location="cpu", weights_only=False)
    model = load_pretrained(subject, C.N_VOXELS[subject], device)

    generated = generate(model, test[condition], device)
    torch.save(generated, run_dir / f"generated_{condition}_p{partition}.pt")

    if stim_set == "B":
        result = reconstruction_metrics(twelve_targets(subject), generated, 6, device)
    else:
        captions, sim = caption_similarity(generated, C.BLOCK_ORDER["C"], device)
        result = {"captions": captions,
                  "similarity": float(np.mean(np.diag(sim))),
                  "per_word": {w: float(sim[i, i]) for i, w in enumerate(C.BLOCK_ORDER["C"])}}

    fp = run_dir / f"metrics_{condition}_p{partition}.json"
    json.dump(result, open(fp, "w"), indent=2)
    print(json.dumps(result, indent=2))
    print(f"-> {fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, required=True, choices=C.SUBJECTS)
    p.add_argument("--set", dest="stim_set", required=True, choices=["B", "C"])
    p.add_argument("--partition", type=int, default=0)
    p.add_argument("--condition", default="aligned", choices=["aligned", "baseline"],
                   help="'baseline' decodes the same test inputs without the alignment module")
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    main(a.subject, a.stim_set, a.partition, a.device, a.condition)
