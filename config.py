"""
config.py
=========
Central configuration file for the Radar Human Activity Classification project.
All other files import from here — change hyperparameters in one place only.
"""

import os

# ══════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════

# Root folder containing all 7 dataset subfolders
DEFAULT_DATASET_ROOT = os.path.join(os.getcwd(), "dataset")
DATASET_ROOT = os.getenv("RADAR_DATASET_ROOT", DEFAULT_DATASET_ROOT)

# Where to save checkpoints, logs, results
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
OUTPUT_DIR = os.getenv("RADAR_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
CHECKPOINT_DIR = os.getenv("RADAR_CHECKPOINT_DIR", os.path.join(OUTPUT_DIR, "checkpoints"))
LOG_DIR = os.getenv("RADAR_LOG_DIR", os.path.join(OUTPUT_DIR, "logs"))
RESULTS_DIR = os.getenv("RADAR_RESULTS_DIR", os.path.join(OUTPUT_DIR, "results"))

# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════

# The 7 subfolder names (in DATASET_ROOT)
DATASET_FOLDERS = [
    "1 December 2017 Dataset",
    "2 March 2017 Dataset",
    "3 June 2017 Dataset",
    "4 July 2018 Dataset",
    "5 February 2019 UoG Dataset",
    "6 February 2019 NG Homes Dataset",
    "7 March 2019 West Cumbria Dataset"]

# Activity labels  (K digit at start of filename → class index 0-based)
# K=1 walk, K=2 sit down, K=3 stand up, K=4 pick up, K=5 drink, K=6 fall
LABEL_MAP = {
    '1': 0,   # Walking
    '2': 1,   # Sitting down
    '3': 2,   # Standing up
    '4': 3,   # Pick up object
    '5': 4,   # Drink water
    '6': 5,   # Fall
}

CLASS_NAMES = ["Walking", "Sitting down", "Standing up", "Pick up object", "Drink water", "Fall"]

NUM_CLASSES = 6

# ══════════════════════════════════════════════════════════════════════
# RADAR SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════

# Fixed radar parameters (same across all files)
FC      = 5.8e9    # carrier frequency (Hz)
TSWEEP  = 1e-3     # chirp duration (s)
NTS     = 128      # samples per chirp
BW      = 400e6    # bandwidth (Hz)
C       = 3e8      # speed of light (m/s)

# MTI filter (Butterworth high-pass)
MTI_ORDER = 4
MTI_CUTOFF = 0.0075 # normalised cutoff frequency

# Spectrogram parameters
SPEC_WINDOW_LEN = 200
SPEC_OVERLAP_FACTOR = 0.95
SPEC_PAD_FACTOR = 4

# Range bins to sum over for spectrogram (adjust per dataset if needed)
RBIN_START = 10   # 1-based (matches MATLAB convention in the .m scripts)
RBIN_END = 30

# ══════════════════════════════════════════════════════════════════════
# CNN INPUT
# ══════════════════════════════════════════════════════════════════════

# All 3 representations are resized to this before entering the CNN
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

# Number of input streams / channels (spectrogram, range-time, range-doppler)
NUM_STREAMS = 3

# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

RANDOM_SEED    = 42
TRAIN_RATIO    = 0.80    # 80% train, 20% test
VAL_RATIO      = 0.10    # of the 80% train → 10% validation (optional)

BATCH_SIZE     = 32
NUM_EPOCHS     = 50
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4    # L2 regularisation

# Learning rate scheduler (StepLR)
LR_STEP_SIZE   = 10      # decay every N epochs
LR_GAMMA       = 0.5     # multiply LR by this factor

# Evaluate on validation set every N epochs
EVAL_EVERY     = 5

# Number of DataLoader workers (set to number of CPU cores available)
NUM_WORKERS    = 4

# Use GPU if available
DEVICE         = "cuda"   # will fall back to "cpu" if no GPU found

# ══════════════════════════════════════════════════════════════════════
# CNN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════

# Per-stream CNN feature size before fusion
STREAM_FEATURE_DIM = 256

# Fusion strategy: 'late' (concatenate after per-stream FC finishes) 
#                  'mid'  (concatenate after 2 conv blocks)
FUSION_TYPE = 'late' # or 'mid'
 
# Dropout probability in classifier head
DROPOUT = 0.5
