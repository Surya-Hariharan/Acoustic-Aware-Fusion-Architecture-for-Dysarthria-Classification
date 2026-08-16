"""
Metric computation for one evaluation pass (validation epoch or fold test set).

Detection is binary (positive class = Dysarthric Patient); severity is
4-class, averaged macro so each severity class counts equally regardless of
how many utterances it contributed. Both report accuracy, precision,
recall (= sensitivity), specificity, F1, and ROC-AUC, matching the metric
set Phase 3's ablation table needs.

UNDEFINED IS NOT ZERO
Every LOSO detection fold holds out ONE speaker, and UA-Speech detection
labels are speaker-level — so that speaker is entirely Healthy or entirely
Dysarthric and the fold's y_true is single-valued. Class-sensitive metrics
(precision, recall, F1, specificity, AUROC) are then *undefined*, not zero:
there is no positive class to have found or missed. Reporting 0.0 for them
produced the pipeline's most misleading artifact — a held-out control
speaker scoring "99.7% accuracy, F1 = 0.0", which reads as a catastrophic
model but actually means "this fold carries no evidence about detection at
all". compute_metrics() therefore returns float("nan") for each metric that
is undefined on the given labels, and reports n_classes_present so callers
can tell the two situations apart without re-deriving it. Aggregation with
pandas (.mean(), .agg) skips NaN by default, so a partially-valid set of
folds averages only over the folds where the metric existed.

Pooling across folds is what makes detection metrics meaningful again — see
src.training.runner.run_training, which concatenates every fold's predictions
before scoring. A pooled set covering both classes has all metrics defined.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from src import config


def _binary_specificity(cm: np.ndarray) -> float:
    """TN / (TN + FP). NaN when this fold held out no negative-class sample —
    there were no true negatives to correctly reject, so specificity has no
    value (as opposed to a value of zero). See the module docstring."""
    tn, fp, _fn, _tp = cm.ravel()
    return float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")


def _macro_specificity(cm: np.ndarray) -> float:
    """Mean one-vs-rest specificity across classes, for multiclass confusion
    matrices. Classes with no negative samples contribute NaN and are skipped
    by the nanmean rather than dragging the macro average toward zero; if no
    class is defined at all, the result is NaN."""
    total = cm.sum()
    specificities = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else float("nan"))
    if np.all(np.isnan(specificities)):
        return float("nan")
    return float(np.nanmean(specificities))


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
        Dict of scalar metrics: accuracy, precision, recall, specificity, f1,
        auroc — each NaN where undefined on these labels (see the module
        docstring) — plus n_classes_present and n_samples, so a caller can
        distinguish "not measurable on this fold" from "measured as zero"
        without re-deriving it from the labels.
    """
    num_classes = config.NUM_CLASSES[task]
    labels = list(range(num_classes))
    average = "binary" if task == "detection" else "macro"

    classes_present = np.unique(y_true)
    n_classes_present = int(len(classes_present))

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Accuracy is always defined (it needs no positive class); everything below
    # it is class-sensitive and undefined on a single-class fold. zero_division
    # is left at 0 only because the branch below never reads those values when
    # they would be undefined — see the module docstring on why 0.0 is the
    # wrong answer here.
    if n_classes_present < 2:
        precision = recall = f1 = specificity = auroc = float("nan")
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0)
        specificity = (_binary_specificity(cm) if task == "detection"
                       else _macro_specificity(cm))
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
        "n_classes_present": n_classes_present,
        "n_samples": int(len(y_true)),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> np.ndarray:
    num_classes = config.NUM_CLASSES[task]
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
