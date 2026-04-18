"""
evaluate.py
-----------
Model evaluation utilities:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix (saved as PNG)
  - Training curves (loss & accuracy vs epoch)
  - Summary CSV of all experiments
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

RESULTS_DIR = "../results"
CURVES_DIR  = os.path.join(RESULTS_DIR, "training_curves")
CM_DIR      = os.path.join(RESULTS_DIR, "confusion_matrices")
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")

for d in [CURVES_DIR, CM_DIR]:
    os.makedirs(d, exist_ok=True)

CLASS_NAMES = ["Uninfected", "Parasitized"]


# ── Confusion matrix ────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name: str, save: bool = True):
    """Plot and optionally save a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save:
        path = os.path.join(CM_DIR, f"{model_name}_confusion_matrix.png")
        plt.savefig(path, dpi=150)
        print(f"Confusion matrix saved: {path}")
    plt.show()
    plt.close()


# ── Training curves ─────────────────────────────────────────────────────────────
def plot_training_curves(history, model_name: str, save: bool = True):
    """Plot accuracy and loss curves from a Keras History object."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy",      color="#2196F3")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", color="#FF5722")
    axes[0].set_title(f"Accuracy — {model_name}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss",      color="#4CAF50")
    axes[1].plot(history.history["val_loss"], label="Validation Loss", color="#F44336")
    axes[1].set_title(f"Loss — {model_name}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary Cross-Entropy Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(CURVES_DIR, f"{model_name}_training_curves.png")
        plt.savefig(path, dpi=150)
        print(f"Training curves saved: {path}")
    plt.show()
    plt.close()


# ── Full model evaluation ────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str = "model",
                   threshold: float = 0.50) -> dict:
    """
    Evaluate a trained model on the test set and print/save all metrics.

    Args:
        model:       Trained tf.keras.Model with sigmoid output.
        X_test:      NumPy array of test images.
        y_test:      NumPy array of ground-truth labels.
        model_name:  String identifier used in plot filenames.
        threshold:   Classification threshold (default 0.5).

    Returns:
        metrics: Dictionary of all computed metric values.
    """
    # Predict probabilities and apply threshold
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    # Compute metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    metrics = {
        "model":     model_name,
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1_score":  round(f1,   4),
        "auc_roc":   round(auc,  4),
    }

    # Print report
    print(f"\n{'='*55}")
    print(f"  Evaluation Results: {model_name.upper()}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Plots
    plot_confusion_matrix(y_test, y_pred, model_name)

    # Save to CSV
    _append_metrics_csv(metrics)

    return metrics


def _append_metrics_csv(metrics: dict):
    """Append a metrics row to the summary CSV."""
    df_new = pd.DataFrame([metrics])
    if os.path.exists(METRICS_CSV):
        df_existing = pd.read_csv(METRICS_CSV)
        # Update existing row or append
        df_existing = df_existing[df_existing["model"] != metrics["model"]]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(METRICS_CSV, index=False)
    print(f"Metrics saved to: {METRICS_CSV}")


# ── Compare all models ───────────────────────────────────────────────────────────
def plot_model_comparison(save: bool = True):
    """
    Read metrics_summary.csv and plot a grouped bar chart comparing all models.
    """
    if not os.path.exists(METRICS_CSV):
        print(f"No metrics file found at {METRICS_CSV}. Train models first.")
        return

    df = pd.read_csv(METRICS_CSV)
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    x     = np.arange(len(df))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        offset = (i - 2) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric.replace("_", " ").title(),
                      color=color, alpha=0.85)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].str.upper(), fontsize=11)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", ncol=5)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(RESULTS_DIR, "model_comparison.png")
        plt.savefig(path, dpi=150)
        print(f"Comparison chart saved: {path}")
    plt.show()
    plt.close()


# ── CLI ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_model_comparison()
