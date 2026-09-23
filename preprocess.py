
import argparse

import nibabel
import nilearn.signal
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import settings as C


def resample_timecourse(x: np.ndarray, target_len: int) -> np.ndarray:
    """(voxels, time) -> (voxels, target_len), linear interpolation."""
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    out = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
    return out.squeeze(0).numpy()


def cue_to_target() -> dict:
    """cue letter -> target file name / concept word, from the experiment table."""
    df = pd.read_excel(C.CUE_PAIR_LIST)
    return dict(zip(df["cue"], df["target"]))


def onset_table(run: int, modality: str) -> pd.DataFrame:
    """Boolean table with one row per volume and one column per cue: True on onsets.

    A trial starts where the flag goes from 0 to 1, hence the `diff`.
    """
    df = pd.read_csv(C.DESIGN_MATRIX_DIR / f"design_matrix_{C.RUN_TASK[run]}.csv")
    if modality == "VIS":
        # perception runs: keep only the columns of trials where cue and image match
        cues = [c for c in df.columns if c.endswith("1")]
    else:
        cues = df.columns[::2] if any(c.isdigit() for c in df.columns) else df.columns
    flags = df[cues].astype(int)
    return flags.diff().fillna(flags.iloc[0]).astype(bool) & flags.astype(bool)


def clean_run(subject: int, run: int, roi: str) -> np.ndarray:
    """(voxels, time) of one run, truncated, detrended and z-scored."""
    ts = (C.NSD_IMAGERY_ROOT / f"nsddata_timeseries/ppdata/subj{subject:02d}/func1pt8mm"
          / f"timeseries/timeseries_nsdimagery_run{run:02d}.nii.gz")
    roi_fp = (C.NSD_IMAGERY_ROOT / f"nsddata/ppdata/subj{subject:02d}/func1pt8mm/roi" / roi)

    nifti = nibabel.load(ts, mmap=True).slicer[..., :C.N_VOLUMES]
    mask = nibabel.load(roi_fp, mmap=True).get_fdata()
    data = nifti.get_fdata()[mask > 0]                       # (voxels, time)

    data = data.T                                            # time first, as nilearn wants
    shape = data.shape
    data = nilearn.signal.clean(
        data.reshape(shape[0], -1),
        detrend=True,            # linear detrending
        high_pass=None,          # no cosine-drift regressors
        t_r=C.TR_IMAGERY,
        standardize="zscore_sample",
    )
    return data.reshape(shape).T                             # (voxels, time)


def build(subject: int, modality: str, stim_set: str, roi: str):
    cue2img = cue_to_target()
    start = int(round(C.OFFSET / C.TR_IMAGERY))
    end = int(round((C.OFFSET + C.DURATION) / C.TR_IMAGERY))

    brains, images, stim_ids = [], [], []
    for run in C.RUNS[(modality, stim_set)]:
        data = clean_run(subject, run, roi)
        onsets = onset_table(run, modality)

        for timestep, row in onsets.iterrows():
            active = row[row == 1].index.tolist()
            if not active:
                continue
            for cue in active:
                # perception columns are named "<letter>1"; imagery ones are the letter
                letter = cue[0] if modality == "VIS" else cue
                if letter not in cue2img:
                    raise KeyError(f"cue {letter!r} is not in the cue/target table")
                target = cue2img[letter]

                window = data[..., timestep + start: timestep + end]
                if window.shape[-1] < end - start:
                    # the run was truncated before the end of this window
                    print(f"run {run:02d}: dropping the trial at volume {timestep} "
                          f"(only {window.shape[-1]} of {end - start} volumes left)")
                    continue

                img_fp = C.NSD_IMAGERY_ROOT / f"rawtargetimages/set{stim_set}/{target}"
                image = np.array(
                    Image.open(img_fp).convert("RGB").resize((512, 512), Image.BILINEAR),
                    dtype=np.uint8,
                )
                brains.append(resample_timecourse(window, C.T))
                images.append(torch.from_numpy(image).long().permute(2, 0, 1) / 255.0)
                stim_ids.append(target)

    out = {
        "brain_tensor": torch.tensor(np.stack(brains), dtype=torch.float32),
        "images_tensor": torch.stack(images),
        "stim_ids": np.array(stim_ids),
    }
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = C.DATA_DIR / f"imagery_nsd_{modality}_{stim_set}_sub{subject}.pt"
    torch.save(out, fp)
    print(f"{fp}: {tuple(out['brain_tensor'].shape)}, "
          f"{len(set(stim_ids))} stimuli, {len(stim_ids)} trials")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", type=int, required=True, choices=C.SUBJECTS)
    p.add_argument("--modality", required=True, choices=["IMG", "VIS"])
    p.add_argument("--set", dest="stim_set", required=True, choices=["A", "B", "C"])
    p.add_argument("--roi", default="nsdgeneral.nii.gz",
                   help="ROI mask inside the subject's roi/ folder")
    a = p.parse_args()
    build(a.subject, a.modality, a.stim_set, a.roi)
