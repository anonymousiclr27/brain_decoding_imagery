from pathlib import Path


NSD_IMAGERY_ROOT = Path("/path/to/nsd_data_imagery")      # nsddata_timeseries, nsddata, rawtargetimages
DESIGN_MATRIX_DIR = Path("./design_matrices")             # design_matrix_<task>.csv, one per run
CUE_PAIR_LIST = NSD_IMAGERY_ROOT / "nsd_imagery_experiments/cue_pair_list.xlsx"

DATA_DIR = Path("./data")            # output of preprocess.py
AUG_DIR = Path("./augmentation")     # retrieved NSD perception trials (see README)
OUT_DIR = Path("./runs")             # checkpoints, generated images, metrics

DYNADIFF_REPO = Path("/path/to/dynadiff")        # provides config.cfg and model.models
DYNADIFF_CKPT = "/path/to/checkpoints/model_dyna_subj{subject:02d}_final.pt"
# The configuration of the pretrained model resolves its caches relative to the
# working directory, so the model is built from inside DYNADIFF_REPO (see
# train_lfa.load_pretrained) and these two stay relative to it.
CACHE_DIR = "./cache"
VD_CACHE_DIR = "./versatile_diffusion"


SUBJECTS = (1, 2, 5, 7)
N_VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}       # nsdgeneral, per subject

# NSD-Imagery run numbering (see Kneeland et al., 2025). Imagery sets are acquired
# twice, perception once; attention runs (02, 05, 08) are not used.
RUNS = {
    ("VIS", "A"): [1],  ("IMG", "A"): [3, 10],
    ("VIS", "B"): [4],  ("IMG", "B"): [6, 11],
    ("VIS", "C"): [7],  ("IMG", "C"): [9, 12],
}
RUN_TASK = {1: "visA", 3: "imgA_1", 4: "visB", 6: "imgB_1", 7: "visC",
            9: "imgC_1", 10: "imgA_2", 11: "imgB_2", 12: "imgC_2"}


BLOCK_ORDER = {
    "B": ["shared0385_nsd28752.png", "shared0842_nsd61178.png", "shared0907_nsd65873.png",
          "shared0000_nsd00000.png", "shared0741_nsd53882.png", "shared0413_nsd30857.png"],
    "C": ["yellow", "stripes", "banana", "zebra", "fruit", "mammal"],
}


TR_IMAGERY = 1.0          # acquisition TR of NSD-Imagery, in seconds
TR_MODEL = 1.33           # TR the pretrained model expects (NSD is 4/3 s)
OFFSET = 4.6              # window start after stimulus onset, in seconds
DURATION = 8.0            # window length, in seconds
N_VOLUMES = 225           # runs are truncated here; events beyond it are dropped
T = int(round(DURATION / TR_MODEL))        # = 6 time samples per trial


SEED = 0
HIDDEN = 8000
DROPOUT = 0.25
LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 36
EPOCHS = 30
PATIENCE = 8
TAU = 0.05                # temperature of the supervised contrastive term
LAMBDA = 0.1              # its weight in the loss
NOISE = 0.02              # noise added to replicated imagery trials, in units of sigma
N_VAL = 20                # validation targets per stimulus
TEST_FRAC = 0.20          # imagery trials held out per stimulus
VAL_FRAC = 0.15           # of the remainder
K_AUG = 1000              # retrieved perception trials per stimulus

# The ten trial partitions used in the paper; each is an independent split.
PARTITIONS = [0, 127, 923, 42, 182, 191, 201, 255, 333, 957]
GEN_SEED = 1000           # sampling seed of Versatile Diffusion, fixed across methods
