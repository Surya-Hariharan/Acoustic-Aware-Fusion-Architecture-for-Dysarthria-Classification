"""
Severity metrics for one evaluation pass (a validation epoch, a fold's test
set, or every fold pooled).

UNDEFINED IS NOT ZERO. A severity LOSO fold holds out one speaker, and labels
are speaker-level, so a fold's y_true has a single class. Macro-F1, balanced
accuracy, precision, recall, specificity and AUROC are then undefined — they
are returned as NaN (shown as N/A), never as 0.0, and n_classes_present says
why. Macro averages run over the classes present in y_true, so a partial run
is not dragged down by a fabricated F1 of 0 for a class it never saw. Pooling
every fold's predictions is what makes these metrics meaningful.
"""

import warnings
from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from src import config

LABELS = list(range(config.NUM_CLASSES))


def _macro_specificity(cm: np.ndarray) -> float:
    """Mean one-vs-rest specificity over classes where it is defined."""
    total = cm.sum()
    values = []
    for i in range(cm.shape[0]):
        tp, fn, fp = cm[i, i], cm[i, :].sum() - cm[i, i], cm[:, i].sum() - cm[i, i]
        tn = total - tp - fn - fp
        values.append(tn / (tn + fp) if (tn + fp) > 0 else float("nan"))
    return float("nan") if np.all(np.isnan(values)) else float(np.nanmean(values))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """y_true/y_pred: (N,) class indices; y_prob: (N, 4) class probabilities.

    accuracy and ordinal MAE (mean |true rank - predicted rank|) are always
    defined; every class-sensitive metric is NaN when fewer than two classes
    are present."""
    n_classes_present = int(len(np.unique(y_true)))
    nan = float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else nan,
        "ordinal_mae": (float(np.mean(np.abs(y_true.astype(np.int64) - y_pred.astype(np.int64))))
                        if len(y_true) else nan),
        "balanced_accuracy": nan, "precision": nan, "recall": nan, "specificity": nan,
        "f1": nan, "f1_weighted": nan, "auroc": nan,
        "n_classes_present": n_classes_present, "n_samples": int(len(y_true)),
    }
    if n_classes_present < 2:
        return metrics

    present = sorted(int(c) for c in np.unique(y_true))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # UndefinedMetricWarning
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=present, average="macro", zero_division=0)
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
        metrics.update({
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "f1_weighted": float(f1_weighted),
            "specificity": _macro_specificity(confusion_matrix(y_true, y_pred, labels=LABELS)),
        })
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob, labels=LABELS,
                                                   multi_class="ovr", average="macro"))
        except ValueError:                                   # a class absent: undefined
            pass
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=LABELS)
