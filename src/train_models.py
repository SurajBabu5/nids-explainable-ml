"""
Trains and evaluates multiple models on UNSW-NB15 for the binary
(attack vs normal) intrusion detection task:
- Random Forest
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP / neural net)
- XGBoost

Saves trained models to models/ and a metrics summary to reports/.
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; avoids Tk/Tcl dependency issues
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, FIGURES_DIR, RANDOM_STATE
from data_preprocessing import run_full_preprocessing


def get_models():
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=RANDOM_STATE
        ),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        )
    return models


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"confusion_matrix_{name}.png"), dpi=150)
    plt.close(fig)

    report = classification_report(y_test, y_pred, zero_division=0)
    return metrics, report


def train_all_models(save_models=True):
    X_train, X_test, y_train, y_test, meta = run_full_preprocessing(task="binary")
    print(f"Data source: {meta['source']} | Train: {X_train.shape} | Test: {X_test.shape}\n")

    models = get_models()
    results = {}

    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        metrics, report = evaluate_model(name, model, X_test, y_test)
        metrics["train_time_sec"] = round(elapsed, 2)
        results[name] = metrics

        print(f"  done in {elapsed:.1f}s | "
              f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
              f"roc_auc={metrics.get('roc_auc', float('nan')):.3f}")
        print(report)

        if save_models:
            joblib.dump(model, os.path.join(MODELS_DIR, f"{name}_binary.pkl"))

    results_df = pd.DataFrame(results).T.sort_values("f1", ascending=False)
    os.makedirs(os.path.dirname(os.path.join(FIGURES_DIR, "..", "metrics_summary.csv")), exist_ok=True)
    results_path = os.path.join(FIGURES_DIR, "..", "metrics_summary.csv")
    results_df.to_csv(results_path)
    print("\n=== Summary (sorted by F1) ===")
    print(results_df.round(4))
    print(f"\nSaved metrics summary -> {os.path.abspath(results_path)}")
    print(f"Saved confusion matrix plots -> {FIGURES_DIR}")
    print(f"Saved trained models -> {MODELS_DIR}")

    return results_df, models, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    train_all_models()