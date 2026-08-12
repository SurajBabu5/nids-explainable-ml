"""
Explainability layer for the NIDS models.

- SHAP: global feature importance + summary plots (best for tree models
  like Random Forest / XGBoost, which is what most reviewers expect).
- LIME: local, per-prediction explanations (useful for explaining WHY one
  specific flow was flagged as an attack -- good demo material for a
  viva/presentation).

Run this AFTER train_models.py has produced saved models in models/.
"""

import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; avoids Tk/Tcl dependency issues
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, FIGURES_DIR
from data_preprocessing import run_full_preprocessing


def _select_positive_class(shap_values):
    """
    Normalizes SHAP output for the "Attack" (class 1) across shap library
    versions:
    - Older versions: list of 2 arrays (one per class) -> take index 1
    - Newer versions: single 3D array (samples, features, classes) -> take [:, :, 1]
    - Regression / already-binary output: 2D array -> use as-is
    """
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def explain_with_shap(model, X_train, X_test, model_name, max_display=15, sample_size=200):
    """
    Produces a SHAP summary (beeswarm) plot showing which features drive
    predictions overall, and a bar plot of mean |SHAP value| per feature.
    Uses a subsample of X_test for speed on larger datasets.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)

    # TreeExplainer is fast/exact for RF & XGBoost; fall back to KernelExplainer
    # (slow, so use a small background set) for anything else, e.g. SVM/MLP.
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap_values = _select_positive_class(shap_values)
    except Exception:
        background = shap.sample(X_train, 50, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        shap_values = _select_positive_class(shap_values)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"shap_summary_{model_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"shap_bar_{model_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP plots for {model_name} -> {FIGURES_DIR}")
    return shap_values


def explain_with_lime(model, X_train, X_test, y_test, model_name, instance_idx=0):
    """
    Explains ONE specific prediction (the flow at instance_idx in X_test)
    -- e.g. "this flow was flagged as an attack mainly because of high
    sbytes and an unusual state value."
    """
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=["Normal", "Attack"],
        mode="classification",
        random_state=42,
    )

    instance = X_test.iloc[instance_idx].values
    actual = y_test.iloc[instance_idx]

    exp = explainer.explain_instance(
        instance, model.predict_proba, num_features=10
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig = exp.as_pyplot_figure()
    fig.suptitle(f"LIME explanation - {model_name} - instance {instance_idx} (actual={actual})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"lime_instance_{instance_idx}_{model_name}.png"), dpi=150)
    plt.close(fig)

    print(f"Saved LIME explanation (instance {instance_idx}, actual label={actual}) -> {FIGURES_DIR}")
    return exp


def run_explainability(model_name="random_forest"):
    X_train, X_test, y_train, y_test, meta = run_full_preprocessing(task="binary", save=False)

    model_path = os.path.join(MODELS_DIR, f"{model_name}_binary.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model at {model_path}. Run train_models.py first."
        )
    model = joblib.load(model_path)

    print(f"Running SHAP explainability for {model_name}...")
    explain_with_shap(model, X_train, X_test, model_name)

    print(f"Running LIME explainability for {model_name} (one attack instance, one normal instance)...")
    attack_idx = int(np.where(y_test.values == 1)[0][0])
    normal_idx = int(np.where(y_test.values == 0)[0][0])
    explain_with_lime(model, X_train, X_test, y_test, model_name, instance_idx=attack_idx)
    explain_with_lime(model, X_train, X_test, y_test, model_name, instance_idx=normal_idx)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="random_forest",
                         choices=["random_forest", "svm", "mlp", "xgboost"])
    args = parser.parse_args()
    run_explainability(args.model)