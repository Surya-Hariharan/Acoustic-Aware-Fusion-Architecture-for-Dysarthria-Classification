"""
Metric computation for one evaluation pass (validation epoch or fold test set).

Detection is binary (positive class = Dysarthric Patient); severity is
4-class, averaged macro so each severity class counts equally regardless of
how many utterances it contributed. Both report accuracy, precision,
recall (= sensitivity), specificity, F1, and ROC-AUC, matching the metric
set Phase 3's ablation table needs.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from src import config


def _binary_specificity(cm: np.ndarray) -> float:
    tn, fp, _fn, _tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def _macro_specificity(cm: np.ndarray) -> float:
    """Mean one-vs-rest specificity across classes, for multiclass confusion matrices."""
    total = cm.sum()
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return float(np.mean(specificities))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                    task: str) -> Dict[str, float]:
    """
    Args:
        y_true: (N,) int labels.
        y_pred: (N,) int predicted labels (argmax).
        y_prob: (N,) positive-class probability for detection, or
                (N, num_classes) softmax probabilities for severity.
        task: "detection" or "severity".
    Returns:
        Dict of scalar metrics: accuracy, precision, recall, specificity, f1, auroc, loss placeholder omitted.
    """
    num_classes = config.NUM_CLASSES[task]
    labels = list(range(num_classes))
    average = "binary" if task == "detection" else "macro"

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = _binary_specificity(cm) if task == "detection" else _macro_specificity(cm)

    # Every LOSO detection fold's held-out speaker is entirely one class (see
    # src.training.runner), so y_true is single-valued on most folds — checking
    # this up front avoids sklearn's UndefinedMetricWarning firing on every one
    # of those (otherwise expected, not a bug) instead of only computing AUROC
    # when it is actually defined.
    if len(np.unique(y_true)) < 2:
        auroc = float("nan")
    else:
        try:
            if task == "detection":
                auroc = roc_auc_score(y_true, y_prob)
            else:
                auroc = roc_auc_score(y_true, y_prob, labels=labels,
                                      multi_class="ovr", average="macro")
        except ValueError:
            # Multiclass: individual classes can still be absent even though
            # more than one is present overall — undefined, not zero.
            auroc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),          # sensitivity
        "specificity": float(specificity),
        "f1": float(f1),
        "auroc": float(auroc),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> np.ndarray:
    num_classes = config.NUM_CLASSES[task]
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
