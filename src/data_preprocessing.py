"""
Data loading and preprocessing for UNSW-NB15.

Handles:
- Loading train/test CSVs (real or sample/fallback)
- Encoding categorical features (proto, service, state)
- Scaling numeric features
- Producing X, y splits for both binary (label) and multiclass (attack_cat) tasks
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_FILE, TEST_FILE, SAMPLE_TRAIN_FILE, SAMPLE_TEST_FILE,
    CATEGORICAL_COLS, ID_COL, BINARY_TARGET, MULTICLASS_TARGET,
    DATA_PROCESSED_DIR, RANDOM_STATE,
)


def load_raw_data(use_sample_if_missing=True):
    """
    Loads the train/test CSVs. Falls back to the synthetic sample data
    (run generate_sample_data.py first) if the real dataset isn't present
    yet, so you can develop/test without waiting on the download.
    """
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        print(f"Loading REAL dataset from {TRAIN_FILE}")
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        return train_df, test_df, "real"

    if use_sample_if_missing and os.path.exists(SAMPLE_TRAIN_FILE):
        print("!! Real UNSW-NB15 files not found in data/raw/.")
        print("!! Falling back to SYNTHETIC sample data for pipeline testing.")
        print("!! Do NOT use these results as your actual project findings.")
        train_df = pd.read_csv(SAMPLE_TRAIN_FILE)
        test_df = pd.read_csv(SAMPLE_TEST_FILE)
        return train_df, test_df, "sample"

    raise FileNotFoundError(
        "No dataset found. Either place UNSW_NB15_training-set.csv / "
        "UNSW_NB15_testing-set.csv in data/raw/, or run "
        "src/generate_sample_data.py to create sample data."
    )


def clean_columns(df):
    """Lowercase column names and strip whitespace, common UNSW-NB15 CSV quirk."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def encode_categoricals(train_df, test_df, cat_cols=CATEGORICAL_COLS):
    """
    Label-encode categorical columns, fit on train, applied to both.
    Unseen categories in test are mapped to a new 'unseen' bucket rather
    than crashing (real-world service/state values can differ slightly
    between the two files).
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    encoders = {}

    for col in cat_cols:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

        le.fit(list(train_df[col].unique()) + ["__unseen__"])
        train_df[col] = le.transform(train_df[col])

        test_df[col] = test_df[col].apply(
            lambda v: v if v in le.classes_ else "__unseen__"
        )
        test_df[col] = le.transform(test_df[col])

        encoders[col] = le

    return train_df, test_df, encoders


def build_feature_target(df, task="binary"):
    """
    Splits a dataframe into X (features) and y (target) for the given task.
    task: "binary" -> predict `label` (0/1)
          "multiclass" -> predict `attack_cat`
    """
    df = df.copy()
    drop_cols = [c for c in [ID_COL] if c in df.columns]

    if task == "binary":
        y = df[BINARY_TARGET]
        drop_cols += [BINARY_TARGET, MULTICLASS_TARGET]
    elif task == "multiclass":
        y = df[MULTICLASS_TARGET]
        drop_cols += [BINARY_TARGET, MULTICLASS_TARGET]
    else:
        raise ValueError("task must be 'binary' or 'multiclass'")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    return X, y


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def run_full_preprocessing(task="binary", save=True):
    """
    Convenience wrapper: load -> clean -> encode -> split -> scale.
    Returns X_train, X_test, y_train, y_test, meta dict.
    """
    train_df, test_df, source = load_raw_data()
    train_df = clean_columns(train_df)
    test_df = clean_columns(test_df)

    train_df, test_df, encoders = encode_categoricals(train_df, test_df)

    X_train, y_train = build_feature_target(train_df, task=task)
    X_test, y_test = build_feature_target(test_df, task=task)

    # Align columns in case of any mismatch
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    if save:
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        joblib.dump(scaler, os.path.join(DATA_PROCESSED_DIR, f"scaler_{task}.pkl"))
        joblib.dump(encoders, os.path.join(DATA_PROCESSED_DIR, f"encoders_{task}.pkl"))

    meta = {"source": source, "n_features": X_train.shape[1], "feature_names": list(X_train.columns)}
    return X_train_scaled, X_test_scaled, y_train, y_test, meta


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, meta = run_full_preprocessing(task="binary")
    print(f"Source: {meta['source']}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Features ({meta['n_features']}): {meta['feature_names'][:5]}...")
    print(f"Train label balance:\n{y_train.value_counts(normalize=True)}")
