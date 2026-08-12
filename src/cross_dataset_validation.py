"""
Cross-dataset validation: train on UNSW-NB15, test on a DIFFERENT
intrusion dataset (commonly CIC-IDS2017) to show your model generalizes
rather than just memorizing quirks of one dataset. This is the strongest
"novelty" element examiners respond well to, since most student NIDS
projects only ever evaluate on a single dataset's own test split.

IMPORTANT - this requires manual setup because CIC-IDS2017 has a
DIFFERENT set of column names/features than UNSW-NB15 does. There is no
universal auto-mapping; you (or a TA) need to decide the feature
correspondence. This script gives you:
  1. A worked example FEATURE_MAP for the features that translate cleanly
  2. Instructions for extending it
  3. The evaluation logic once the mapping is in place

Steps:
  1. Download CIC-IDS2017 (see README.md) and place the CSV(s) in
     data/raw/cicids2017/
  2. Fill in / extend FEATURE_MAP below based on the two datasets'
     feature documentation (both are on the same underlying idea --
     flow duration, packet counts, byte counts, etc. -- but named and
     scaled differently)
  3. Run this script
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR
from data_preprocessing import run_full_preprocessing

CICIDS_DIR = os.path.join(DATA_RAW_DIR, "cicids2017")

# ---------------------------------------------------------------------------
# WORKED EXAMPLE mapping: UNSW-NB15 feature name -> CIC-IDS2017 feature name
# This covers features that have a reasonably direct conceptual equivalent.
# You will likely need to add/adjust these once you've inspected the real
# CIC-IDS2017 column headers (they vary slightly between mirrors/versions --
# run `df.columns.tolist()` on the file you download and compare).
# ---------------------------------------------------------------------------
FEATURE_MAP = {
    "dur": "Flow Duration",
    "spkts": "Total Fwd Packets",
    "dpkts": "Total Backward Packets",
    "sbytes": "Total Length of Fwd Packets",
    "dbytes": "Total Length of Bwd Packets",
    "sload": "Flow Bytes/s",
    "sinpkt": "Fwd IAT Mean",
    "dinpkt": "Bwd IAT Mean",
}
# CIC-IDS2017's own label column, mapped to binary the same way UNSW-NB15 is.
CICIDS_LABEL_COL = "Label"
CICIDS_BENIGN_VALUE = "BENIGN"

# ---------------------------------------------------------------------------
# UNIT CORRECTIONS -- IMPORTANT
# UNSW-NB15 (generated with Argus) and CIC-IDS2017 (generated with
# CICFlowMeter) record some conceptually-equivalent time fields in
# DIFFERENT UNITS:
#   - UNSW-NB15 `dur` is in SECONDS; CIC-IDS2017 `Flow Duration` is in
#     MICROSECONDS (factor of 1,000,000)
#   - UNSW-NB15 `sinpkt`/`dinpkt` are in MILLISECONDS; CIC-IDS2017's
#     `Fwd/Bwd IAT Mean` are in MICROSECONDS (factor of 1,000)
# Without correcting for this, these features end up many orders of
# magnitude outside anything the model's scaler ever saw during training,
# which can dominate predictions regardless of the traffic's actual nature.
# These conversion factors are based on each tool's documented output
# units -- double check against your specific file/tool version if your
# results still look off after this correction.
# ---------------------------------------------------------------------------
UNIT_CORRECTIONS = {
    "dur": 1e-6,      # microseconds -> seconds
    "sinpkt": 1e-3,   # microseconds -> milliseconds
    "dinpkt": 1e-3,   # microseconds -> milliseconds
}


def load_cicids_binary(sample_n=None):
    """
    Loads CIC-IDS2017 CSV(s) from data/raw/cicids2017/, restricted to the
    columns we have a mapping for, and derives a binary label
    (0 = benign, 1 = attack) to match UNSW-NB15's `label` column.
    """
    if not os.path.isdir(CICIDS_DIR) or not os.listdir(CICIDS_DIR):
        raise FileNotFoundError(
            f"No CIC-IDS2017 files found in {CICIDS_DIR}. Download it first "
            "(see README.md) and place the CSV file(s) there."
        )

    frames = []
    for fname in sorted(os.listdir(CICIDS_DIR)):
        if fname.lower().endswith(".csv"):
            # low_memory=False avoids dtype-inference warnings on these large,
            # mixed-type files; CIC-IDS2017 CSVs are known to have some
            # numeric columns stored as strings with stray whitespace.
            frames.append(pd.read_csv(os.path.join(CICIDS_DIR, fname), low_memory=False))
    if not frames:
        raise FileNotFoundError(f"No .csv files found in {CICIDS_DIR}.")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    if sample_n:
        df = df.sample(min(sample_n, len(df)), random_state=42)

    missing = [v for v in FEATURE_MAP.values() if v not in df.columns]
    if missing:
        raise KeyError(
            f"These CIC-IDS2017 columns from FEATURE_MAP were not found: {missing}. "
            f"Actual columns in your file: {list(df.columns)}. "
            "Update FEATURE_MAP in this script to match your file's exact column names."
        )
    if CICIDS_LABEL_COL not in df.columns:
        raise KeyError(
            f"Expected label column '{CICIDS_LABEL_COL}' not found. "
            f"Actual columns: {list(df.columns)}. Update CICIDS_LABEL_COL."
        )

    y = (df[CICIDS_LABEL_COL].astype(str).str.strip() != CICIDS_BENIGN_VALUE).astype(int)
    X = df[list(FEATURE_MAP.values())].copy()
    X.columns = list(FEATURE_MAP.keys())  # rename to UNSW-NB15 side names
    # Coerce to numeric in case any column was read as string (common in these files)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([float("inf"), float("-inf")], 0).fillna(0)
    # Apply unit corrections BEFORE scaling -- see UNIT_CORRECTIONS comment above.
    for col, factor in UNIT_CORRECTIONS.items():
        if col in X.columns:
            X[col] = X[col] * factor
    return X, y


def inspect_cicids_columns():
    """
    Quick utility: load just the first CSV found and print its columns,
    so you can check/fix FEATURE_MAP BEFORE running the full validation
    (much faster than waiting for a KeyError after loading everything).
    """
    if not os.path.isdir(CICIDS_DIR) or not os.listdir(CICIDS_DIR):
        raise FileNotFoundError(f"No CIC-IDS2017 files found in {CICIDS_DIR}.")
    csvs = sorted(f for f in os.listdir(CICIDS_DIR) if f.lower().endswith(".csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in {CICIDS_DIR}.")
    first_file = os.path.join(CICIDS_DIR, csvs[0])
    df_head = pd.read_csv(first_file, nrows=5, low_memory=False)
    df_head.columns = [c.strip() for c in df_head.columns]
    print(f"Found {len(csvs)} CSV file(s): {csvs}")
    print(f"\nColumns in {csvs[0]} ({len(df_head.columns)} total):")
    for c in df_head.columns:
        print(f"  {c!r}")
    print(f"\nCurrent FEATURE_MAP targets: {list(FEATURE_MAP.values())}")
    missing = [v for v in FEATURE_MAP.values() if v not in df_head.columns]
    if missing:
        print(f"\n!! NOT FOUND in this file: {missing}")
        print("!! Update FEATURE_MAP in this script with the correct names from the list above.")
    else:
        print("\nAll FEATURE_MAP columns found. Safe to run full validation.")
    if CICIDS_LABEL_COL not in df_head.columns:
        print(f"\n!! Label column {CICIDS_LABEL_COL!r} not found either -- check the list above.")


def run_cross_dataset_validation(model_name="random_forest", sample_n=None):
    model_path = os.path.join(MODELS_DIR, f"{model_name}_binary.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model at {model_path}. Run train_models.py first.")
    model = joblib.load(model_path)

    # save=True guarantees the StandardScaler used during training is on disk --
    # we MUST reuse the exact same scaler (not a new one) to transform the
    # CIC-IDS2017 features consistently with what the model was trained on.
    X_train, _, _, _, meta = run_full_preprocessing(task="binary", save=True)
    trained_features = meta["feature_names"]

    scaler_path = os.path.join(DATA_PROCESSED_DIR, "scaler_binary.pkl")
    scaler = joblib.load(scaler_path)
    # StandardScaler stores per-feature means in the same column order it was fit on
    training_means = pd.Series(scaler.mean_, index=trained_features)

    X_cic, y_cic = load_cicids_binary(sample_n=sample_n)

    # Build the full raw (unscaled) feature frame the model expects:
    # - columns we can map from CIC-IDS2017 get real raw values
    # - columns with no CIC-IDS2017 equivalent (categorical fields, UNSW-specific
    #   engineered features) get the UNSW TRAINING MEAN, so that after scaling
    #   they land at 0 (neutral / "average"), not an arbitrary raw value of 0
    #   which for skewed features could correspond to an extreme, misleading
    #   z-score.
    X_full = pd.DataFrame(index=X_cic.index, columns=trained_features, dtype=float)
    for col in trained_features:
        if col in X_cic.columns:
            X_full[col] = X_cic[col].values
        else:
            X_full[col] = training_means[col]

    # CRITICAL: apply the SAME scaling used during training. Skipping this
    # (as an earlier version of this script did) feeds raw CIC-IDS2017 values
    # -- on a completely different scale than the standardized training data --
    # directly into the model, which pushes every split decision the same
    # direction and produces degenerate all-one-class predictions.
    X_full_scaled = pd.DataFrame(
        scaler.transform(X_full), columns=trained_features, index=X_full.index
    )

    y_pred = model.predict(X_full_scaled)

    metrics = {
        "accuracy": accuracy_score(y_cic, y_pred),
        "precision": precision_score(y_cic, y_pred, zero_division=0),
        "recall": recall_score(y_cic, y_pred, zero_division=0),
        "f1": f1_score(y_cic, y_pred, zero_division=0),
    }
    print(f"\n=== Cross-dataset validation: trained on UNSW-NB15, tested on CIC-IDS2017 ===")
    print(f"Model: {model_name}")
    print(pd.Series(metrics).round(4))
    print("\n" + classification_report(y_cic, y_pred, zero_division=0))
    print(
        "\nNOTE: only features present in FEATURE_MAP were carried over; "
        "categorical fields (proto/service/state) and UNSW-specific "
        "engineered features were zero-filled. Expect a meaningful accuracy "
        "drop vs in-dataset testing -- that drop, and discussing WHY it "
        "happens, is itself a legitimate and expected finding for your report."
    )
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest",
                         choices=["random_forest", "svm", "mlp", "xgboost"])
    parser.add_argument("--inspect", action="store_true",
                         help="Just print the CIC-IDS2017 file's columns and check "
                              "FEATURE_MAP against them, then exit (no training/prediction).")
    parser.add_argument("--sample_n", type=int, default=None,
                         help="Optional: randomly subsample this many CIC-IDS2017 rows "
                              "for a faster test run before doing the full file.")
    args = parser.parse_args()

    if args.inspect:
        inspect_cicids_columns()
    else:
        run_cross_dataset_validation(args.model, sample_n=args.sample_n)