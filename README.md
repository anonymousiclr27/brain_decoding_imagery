# Latent Functional Alignment for visual imagery decoding

Code for *Bridging Vision and Imagery: Latent Functional Alignment for visual imagery
decoding from fMRI data*.

## Setup

```bash
pip install -r requirements.txt
```

Then edit `settings.py`, the only file containing paths:

* `NSD_IMAGERY_ROOT`, `DESIGN_MATRIX_DIR`, `CUE_PAIR_LIST` — the NSD / NSD-Imagery data;
* `DYNADIFF_REPO`, `DYNADIFF_CKPT` — the pretrained decoder.

Do not rename `settings.py` or `alignment.py`: the decoder ships its own `config` and
`model` packages, and modules of those names here would shadow them.

## Order of execution

**1. Preprocessing.** Once per subject, modality and stimulus set. Set A is needed as
well: its targets act as distractors in the metrics.

```bash
python preprocess.py --subject 1 --modality IMG --set A
python preprocess.py --subject 1 --modality IMG --set B
python preprocess.py --subject 1 --modality VIS --set B
```

Add `--roi <mask>.nii.gz` for regions other than `nsdgeneral`.

**2. Retrieval-based augmentation.** Not part of this repository. Training expects
`augmentation/fmri_vis_nsd_1000_sub<N>.pt`: a tensor `(6 * K, V, 6)` of NSD perception
trials, six contiguous blocks in the order of `BLOCK_ORDER`, with the three
repetitions of each NSD image consecutive.

**3. Training.** One run per partition; the ten partitions of the paper are the ten
values in `PARTITIONS`.

```bash
python train_lfa.py --subject 1 --set B --partition 0 --device cuda:0
```

Writes `runs/sub<N>/set<X>/adapter_p<P>.pt` and `test_inputs_p<P>.pt`.
`--no-temporal-mixing` ablates the 1×1 convolution over time.

**4. Evaluation.** After training, on the same subject, set and partition.

```bash
python evaluate.py --subject 1 --set B --partition 0 --condition aligned
python evaluate.py --subject 1 --set B --partition 0 --condition baseline
python evaluate.py --subject 1 --set C --partition 0 --condition aligned
```

Set B gives the eight reconstruction metrics, set C the similarity between the BLIP
caption and the target word in MPNet space. Results go to
`metrics_<condition>_p<P>.json`.

## Files

| file | |
|---|---|
| `settings.py` | paths, dataset constants, hyper-parameters |
| `preprocess.py` | NSD-Imagery time series → one tensor per subject, modality and set |
| `data.py` | trial splits, retrieval-pool splits, pairing and replication |
| `alignment.py` | alignment module, loss, model-selection criterion |
| `train_lfa.py` | training |
| `evaluate.py` | generation and evaluation |
| `metrics_lib.py` | feature extractors used by the metrics |
