"""
Shared configuration for the NIDS project.

This assumes the commonly-used "training-set.csv" / "testing-set.csv" version
of UNSW-NB15 (the pre-split, header-included version most tutorials and
Kaggle mirrors use), NOT the raw UNSW-NB15_1.csv..4.csv flow files.

If you downloaded the raw 4-part files instead, see README.md section
"Using the raw UNSW-NB15_1..4.csv files" for how to adapt this.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
SAMPLE_DATA_DIR = os.path.join(PROJECT_ROOT, "sample_data")

TRAIN_FILE = os.path.join(DATA_RAW_DIR, "UNSW_NB15_training-set.csv")
TEST_FILE = os.path.join(DATA_RAW_DIR, "UNSW_NB15_testing-set.csv")

SAMPLE_TRAIN_FILE = os.path.join(SAMPLE_DATA_DIR, "sample_train.csv")
SAMPLE_TEST_FILE = os.path.join(SAMPLE_DATA_DIR, "sample_test.csv")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
ID_COL = "id"
BINARY_TARGET = "label"          # 0 = normal, 1 = attack
MULTICLASS_TARGET = "attack_cat"  # Normal, Exploits, Fuzzers, DoS, ...

CATEGORICAL_COLS = ["proto", "service", "state"]

# Columns to drop before modeling (identifiers / leakage-prone / targets)
DROP_COLS_BINARY = [ID_COL, MULTICLASS_TARGET]
DROP_COLS_MULTICLASS = [ID_COL, BINARY_TARGET]

RANDOM_STATE = 42
TEST_SIZE = 0.2  # only used if we need to carve a validation split out of train

ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
]
