"""
Guards on what a reported number means: undefined metrics are NaN (never a
fabricated 0), genuine zeros stay zero, the LOSO fold order is
class-interleaved, and results rebuilt from disk report honest coverage.
No GPU, audio or trained model needed.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.training.metrics import compute_metrics
from src.training.reporting import (collect_run_results, save_metrics, save_predictions,
                                    speaker_level)


def _probs(n: int) -> np.ndarray:
    return np.full((n, config.NUM_CLASSES), 1.0 / config.NUM_CLASSES)


def test_single_class_fold_yields_nan_not_zero():
    """A LOSO fold holds out one speaker, i.e. one class: class-sensitive
    metrics are undefined there, not zero."""
    y_true = np.full(765, 1)
    y_pred = np.full(765, 1)
    y_pred[:5] = 2
    m = compute_metrics(y_true, y_pred, _probs(765))
    assert m["n_classes_present"] == 1 and m["n_samples"] == 765
    assert m["accuracy"] == pytest.approx(760 / 765)
    assert m["ordinal_mae"] == pytest.approx(5 / 765)
    for metric in ("precision", "recall", "f1", "balanced_accuracy", "specificity", "auroc"):
        assert np.isnan(m[metric]), metric


def test_genuine_zero_is_still_reported_as_zero():
    y_true = np.array([0, 0, 3, 3])
    y_pred = np.array([0, 0, 0, 0])                       # never predicts High
    m = compute_metrics(y_true, y_pred, _probs(4))
    assert m["recall"] == pytest.approx(0.5)              # macro over the 2 classes present
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["ordinal_mae"] == pytest.approx(1.5)


def test_macro_metrics_average_over_present_classes_only():
    """A partial run that has seen only two classes must not be dragged down by
    a fabricated F1 of 0 for classes it never held out."""
    y_true = np.array([0, 0, 1, 1])
    m = compute_metrics(y_true, y_true.copy(), _probs(4))
    assert m["f1"] == pytest.approx(1.0)
    assert np.isnan(m["auroc"])                           # needs every class present


def test_all_classes_present_defines_every_metric():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3])
    y_prob = np.eye(4)[y_pred] * 0.7 + 0.075
    m = compute_metrics(y_true, y_pred, y_prob)
    assert m["n_classes_present"] == 4
    for metric in ("f1", "balanced_accuracy", "precision", "recall", "specificity", "auroc"):
        assert np.isfinite(m[metric]), metric


def test_every_prefix_of_the_loso_order_covers_the_classes_early():
    order = config.SEVERITY_LOSO_ORDER
    assert sorted(order) == sorted(config.DYSARTHRIC_IDS)
    assert {config.SEVERITY_MAP[s] for s in order[:4]} == set(config.SEVERITY_CLASS_NAMES)


def test_speaker_level_decision_is_the_median_prediction():
    preds = pd.DataFrame({"speaker_id": ["A"] * 5 + ["B"] * 3,
                          "y_true": [2] * 5 + [0] * 3,
                          "y_pred": [0, 2, 2, 3, 3, 0, 1, 1],
                          "correct": [False, True, True, False, False, True, False, False]})
    table = speaker_level(preds).set_index("Speaker")
    assert table.loc["A", "Speaker-level prediction"] == "Mid" and table.loc["A", "Correct"]
    assert table.loc["B", "Speaker-level prediction"] == "Low"
    assert table.loc["B", "Rank error"] == 1


def test_results_rebuilt_from_disk_report_partial_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path / "predictions")
    run, folds = "TEST", ["M01", "M07", "M05"]
    for fold, label in (("M01", 0), ("M07", 1)):
        y = np.full(4, label)
        save_predictions(config.PREDICTIONS_DIR / run / f"{fold}.csv", [f"{fold}_{i}.wav" for i in range(4)],
                         [fold] * 4, y, y, np.eye(4)[y], y)
        save_metrics(config.METRICS_DIR / run / f"{fold}.json",
                     {"accuracy": 1.0, "ordinal_mae": 0.0, "true_label": config.SEVERITY_CLASS_NAMES[label]})
    with open(config.METRICS_DIR / run / "RUN_STATUS.json", "w") as handle:
        json.dump({"failed_folds": ["M05"]}, handle)

    results = collect_run_results(run, folds)
    coverage = results["coverage"]
    assert coverage["status"] == "PARTIAL" and coverage["completed"] == 2
    assert coverage["missing_folds"] == ["M05"] and coverage["failed_folds"] == ["M05"]
    assert results["n_utterances"] == 8
    pooled = results["pooled"].set_index("Metric")
    assert pooled.loc["Accuracy", "Value"] == 1.0
    assert np.isnan(pooled.loc["AUROC (macro one-vs-rest)", "Value"])
