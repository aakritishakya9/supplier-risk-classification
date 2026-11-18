"""
Model Evaluation and Diagnostics Suite for Supplier Risk Classification.
Computes procurement-focused metrics: High-Risk Recall, PR-AUC, Confusion Matrix, and ROC curves.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)
import os


def compute_classification_metrics(y_true, y_pred, y_prob=None, pos_label=1):
    """
    Computes a comprehensive dictionary of classification metrics with emphasis on the positive (High Risk) class.
    """
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "High-Risk Recall (Late=1)": float(recall_score(y_true, y_pred, pos_label=pos_label)),
        "Precision (Late=1)": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "F1-Score (Late=1)": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
    }

    if y_prob is not None:
        metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_prob))
        metrics["PR-AUC (Avg Precision)"] = float(average_precision_score(y_true, y_prob))

    return metrics


def evaluate_thresholds(y_true, y_prob, thresholds=None):
    """
    Analyzes High-Risk Recall and Precision across different decision thresholds.
    """
    if thresholds is None:
        thresholds = [0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]

    records = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        rec = recall_score(y_true, y_pred_t, pos_label=1, zero_division=0)
        prec = precision_score(y_true, y_pred_t, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred_t, pos_label=1, zero_division=0)
        records.append({
            "Threshold": t,
            "High-Risk Recall": rec,
            "Precision": prec,
            "F1-Score": f1,
            "High-Risk Flagged %": float(np.mean(y_pred_t) * 100)
        })

    return pd.DataFrame(records)


def plot_model_diagnostics(y_true, y_prob, model_name="Gaussian Naive Bayes", save_dir="reports/figures"):
    """
    Generates and saves diagnostic plots:
    1. Confusion Matrix (at threshold 0.5 and 0.35)
    2. ROC Curve
    3. Precision-Recall Curve
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confusion Matrix at default 0.50
    y_pred_default = (y_prob >= 0.50).astype(int)
    cm = confusion_matrix(y_true, y_pred_default)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["On-Time (0)", "Late (1)"],
                yticklabels=["On-Time (0)", "Late (1)"])
    axes[0].set_title(f"{model_name}: Confusion Matrix (t=0.50)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    axes[1].plot(fpr, tpr, color="royalblue", lw=2, label=f"ROC (AUC = {auc_score:.3f})")
    axes[1].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[1].set_title(f"{model_name}: ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].legend(loc="lower right")

    # 3. Precision-Recall Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    axes[2].plot(rec, prec, color="darkorange", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    axes[2].set_title(f"{model_name}: Precision-Recall Curve")
    axes[2].set_xlabel("Recall (High-Risk Sensitivity)")
    axes[2].set_ylabel("Precision")
    axes[2].legend(loc="lower left")

    plt.tight_layout()
    file_slug = model_name.lower().replace(" ", "_")
    output_path = os.path.join(save_dir, f"{file_slug}_diagnostics.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


if __name__ == "__main__":
    # Self-test on mock data
    np.random.seed(42)
    y_test_mock = np.random.choice([0, 1], size=1000, p=[0.45, 0.55])
    y_prob_mock = np.clip(y_test_mock * 0.4 + np.random.uniform(0.1, 0.6, size=1000), 0.0, 1.0)
    y_pred_mock = (y_prob_mock >= 0.5).astype(int)

    metrics = compute_classification_metrics(y_test_mock, y_pred_mock, y_prob_mock)
    print("Self-test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    thresh_df = evaluate_thresholds(y_test_mock, y_prob_mock)
    print("\nThreshold Tuning Summary:")
    print(thresh_df.to_string(index=False))
