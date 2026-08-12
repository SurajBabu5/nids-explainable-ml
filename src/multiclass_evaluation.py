"""
Multiclass evaluation: trains a classifier on the `attack_cat` target
(Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis,
Backdoor, Shellcode, Worms) instead of just the binary label, and produces
a per-category confusion matrix + classification report.

This directly addresses the "class imbalance across attack categories"
limitation noted in the report: binary accuracy hides how well the model
does on rare attack types like Worms/Shellcode. This script surfaces that.

Run AFTER you have real UNSW-NB15 data in data/raw/ (works on sample data
too, but the categories will look meaningless since sample data is
synthetic).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- we only save PNGs, never display them;
                        # avoids a Tk/Tcl dependency that can be broken/missing on some
                        # Windows Python installs even when matplotlib itself works fine.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, FIGURES_DIR, RANDOM_STATE, ATTACK_CATEGORIES
from data_preprocessing import run_full_preprocessing


def get_multiclass_model(name="random_forest"):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced"
        )
    if name == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost not installed")
        return XGBClassifier(
            n_estimators=200, max_depth=6, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    raise ValueError(f"Unknown model: {name}")


def train_and_evaluate_multiclass(model_name="random_forest", save=True):
    X_train, X_test, y_train, y_test, meta = run_full_preprocessing(task="multiclass", save=False)
    print(f"Data source: {meta['source']} | Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Classes in train: {sorted(y_train.unique())}\n")

    model = get_multiclass_model(model_name)

    # XGBoost requires integer-encoded labels, not strings
    if model_name == "xgboost":
        classes = sorted(y_train.unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        y_train_fit = y_train.map(class_to_idx)
        model.fit(X_train, y_train_fit)
        y_pred_idx = model.predict(X_test)
        y_pred = pd.Series(y_pred_idx).map(idx_to_class).values
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    labels_present = sorted(set(y_test.unique()) | set(y_pred))
    # Order labels to match ATTACK_CATEGORIES where possible, for a readable plot
    ordered_labels = [c for c in ATTACK_CATEGORIES if c in labels_present]
    ordered_labels += [c for c in labels_present if c not in ordered_labels]

    cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # categories with 0 test samples -> 0 rather than NaN

    report_dict = classification_report(
        y_test, y_pred, labels=ordered_labels, zero_division=0, output_dict=True
    )
    report_str = classification_report(y_test, y_pred, labels=ordered_labels, zero_division=0)
    print(report_str)

    # --- Plot: raw counts ---
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=ordered_labels, yticklabels=ordered_labels)
    ax.set_title(f"Confusion Matrix by Attack Category - {model_name} (raw counts)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    raw_path = os.path.join(FIGURES_DIR, f"confusion_matrix_multiclass_{model_name}.png")
    fig.savefig(raw_path, dpi=150)
    plt.close(fig)

    # --- Plot: row-normalized (recall per class) -- easier to read with imbalanced classes ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                xticklabels=ordered_labels, yticklabels=ordered_labels, vmin=0, vmax=1)
    ax.set_title(f"Confusion Matrix by Attack Category - {model_name} (row-normalized / recall)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    norm_path = os.path.join(FIGURES_DIR, f"confusion_matrix_multiclass_normalized_{model_name}.png")
    fig.savefig(norm_path, dpi=150)
    plt.close(fig)

    # --- Per-category F1 summary table (sorted worst to best -- highlights weak categories) ---
    per_class_rows = []
    for cat in ordered_labels:
        if cat in report_dict:
            per_class_rows.append({
                "attack_cat": cat,
                "support": int(report_dict[cat]["support"]),
                "precision": round(report_dict[cat]["precision"], 3),
                "recall": round(report_dict[cat]["recall"], 3),
                "f1": round(report_dict[cat]["f1-score"], 3),
            })
    per_class_df = pd.DataFrame(per_class_rows).sort_values("f1")

    summary_path = os.path.join(FIGURES_DIR, "..", f"multiclass_metrics_{model_name}.csv")
    per_class_df.to_csv(summary_path, index=False)

    print("\n=== Per-category performance (sorted worst to best F1) ===")
    print(per_class_df.to_string(index=False))
    print(f"\nSaved raw confusion matrix -> {raw_path}")
    print(f"Saved normalized confusion matrix -> {norm_path}")
    print(f"Saved per-category metrics table -> {os.path.abspath(summary_path)}")

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{model_name}_multiclass.pkl"))
        print(f"Saved model -> {os.path.join(MODELS_DIR, f'{model_name}_multiclass.pkl')}")

    return per_class_df, cm, ordered_labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest",
                         choices=["random_forest", "xgboost"])
    args = parser.parse_args()
    train_and_evaluate_multiclass(args.model)