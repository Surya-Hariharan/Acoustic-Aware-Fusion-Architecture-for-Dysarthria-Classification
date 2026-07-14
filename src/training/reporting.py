"""
Per-fold and cross-fold output writers: predictions, metrics, confusion
matrices, ROC curves, and embeddings — everything train.py drops into
outputs/ for downstream phases (ablation tables, error analysis, Praat
correlation, embedding visualization).
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from src import config


def save_predictions(path: Path, speaker_ids, y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray, task: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_names = config.DETECTION_CLASS_NAMES if task == "detection" else config.SEVERITY_CLASS_NAMES
    df = pd.DataFrame({
        "speaker_id": speaker_ids,
        "y_true": y_true,
        "y_true_label": [class_names[i] for i in y_true],
        "y_pred": y_pred,
        "y_pred_label": [class_names[i] for i in y_pred],
        "correct": np.asarray(y_true) == np.asarray(y_pred),
    })
    if task == "detection":
        df["prob_positive"] = y_prob
    else:
        for i, name in enumerate(class_names):
            df[f"prob_{name.replace(' ', '_')}"] = y_prob[:, i]
    df.to_csv(path, index=False)


def save_metrics(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_confusion_matrix(path: Path, cm: np.ndarray, task: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_names = config.DETECTION_CLASS_NAMES if task == "detection" else config.SEVERITY_CLASS_NAMES
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
               yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roc_curve(path: Path, y_true: np.ndarray, y_prob: np.ndarray, task: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    drew_a_curve = False

    if task == "detection":
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax.plot(fpr, tpr, label="Dysarthric Patient")
            drew_a_curve = True
    else:
        for i, name in enumerate(config.SEVERITY_CLASS_NAMES):
            binary_true = (y_true == i).astype(int)
            if len(np.unique(binary_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
            ax.plot(fpr, tpr, label=name)
            drew_a_curve = True

    if not drew_a_curve:
        # Every class in this fold's split was single-valued (typical of a
        # tiny smoke-test slice) — nothing meaningful to plot.
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_embeddings(path: Path, embeddings: np.ndarray, y_true: np.ndarray, speaker_ids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, y_true=y_true, speaker_ids=np.asarray(speaker_ids))


def aggregate_fold_metrics(metrics_dir: Path, run_name: str,
                           fold_metrics: List[Dict]) -> pd.DataFrame:
    """Average metrics across folds (mean +/- std) and save a summary table."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(fold_metrics)
    metric_cols = [c for c in df.columns if c != "fold"]
    summary = df[metric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    df.to_csv(metrics_dir / f"{run_name}.per_fold.csv", index=False)
    summary.to_csv(metrics_dir / f"{run_name}.summary.csv")
    with open(metrics_dir / f"{run_name}.summary.json", "w") as f:
        json.dump(summary.to_dict(orient="index"), f, indent=2)
    return summary
