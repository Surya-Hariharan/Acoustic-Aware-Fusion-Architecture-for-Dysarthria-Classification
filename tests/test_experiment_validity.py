"""
Guards for the experimental-validity repairs (findings F1, F2, F5, F8).

These tests exist because the pre-repair pipeline could not distinguish a
finished experiment from a truncated one, or an undefined metric from a zero.
Both failure modes were silent: they produced well-formed CSVs with plausible
numbers. Each test below pins one of those distinctions so it cannot regress.

None of them need a GPU, audio, or a trained model.

Run with:  pytest tests/test_experiment_validity.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.results import (TIER_EXCLUDED, TIER_FINAL, TIER_PRELIMINARY,
                         classify_result, is_non_experiment)
from src.splits import iter_loso_folds
from src.training.metrics import compute_metrics
from src.training.reporting import (FOLD_COMPLETED, FOLD_SKIPPED_DEADLINE,
                                    load_registry, record_fold, summarize_registry)


# ---------------------------------------------------------------------------
# F1 — undefined metrics must be NaN, never 0.0
# ---------------------------------------------------------------------------
def test_single_class_fold_yields_nan_not_zero():
    """The exact pre-repair failure: a held-out healthy-control speaker.

    The model is right about almost every utterance, so accuracy is high — but
    there is no positive class, so precision/recall/F1/specificity/AUROC are
    undefined. Reporting 0.0 made this look like a catastrophic model instead of
    an uninformative fold.
    """
    y_true = np.zeros(765, dtype=int)          # every held-out utterance is Healthy
    y_pred = np.zeros(765, dtype=int)
    y_pred[:2] = 1                             # two false positives
    y_prob = np.full(765, 0.02)

    m = compute_metrics(y_true, y_pred, y_prob, task="detection")

    assert m["n_classes_present"] == 1
    assert m["n_samples"] == 765
    assert m["accuracy"] == pytest.approx(763 / 765)
    for metric in ("precision", "recall", "f1", "specificity", "auroc"):
        assert np.isnan(m[metric]), f"{metric} must be NaN on a single-class fold, got {m[metric]}"


def test_genuine_zero_is_still_reported_as_zero():
    """A model that predicts the negative class for every sample of a
    two-class fold genuinely HAS zero recall. The NaN guard must not swallow a
    real zero — that would be the opposite error."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])            # never predicts positive
    y_prob = np.array([0.1, 0.1, 0.2, 0.2])

    m = compute_metrics(y_true, y_pred, y_prob, task="detection")

    assert m["n_classes_present"] == 2
    assert m["recall"] == 0.0 and not np.isnan(m["recall"])
    assert m["f1"] == 0.0 and not np.isnan(m["f1"])
    assert m["specificity"] == 1.0             # both negatives correctly rejected


def test_two_class_fold_defines_every_metric():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9])

    m = compute_metrics(y_true, y_pred, y_prob, task="detection")

    assert m["n_classes_present"] == 2
    for metric in ("accuracy", "precision", "recall", "f1", "specificity", "auroc"):
        assert not np.isnan(m[metric]), f"{metric} should be defined here"


def test_nan_metrics_are_skipped_by_pandas_aggregation():
    """Mixed valid/invalid folds must average over the valid ones only —
    the property that makes a partially-valid run's summary honest."""
    folds = pd.DataFrame([
        {"fold": "CF02", "f1": float("nan")},   # single-class fold
        {"fold": "F02", "f1": 0.80},
        {"fold": "CF03", "f1": 0.90},
    ])
    assert folds["f1"].mean() == pytest.approx(0.85)
    assert folds["f1"].count() == 2


def test_severity_macro_metrics_defined_with_all_classes():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 2, 2, 3])
    y_prob = np.full((8, 4), 0.25)
    m = compute_metrics(y_true, y_pred, y_prob, task="severity")
    assert m["n_classes_present"] == 4
    assert not np.isnan(m["f1"])


# ---------------------------------------------------------------------------
# F2 — every prefix of the LOSO fold order must be class-balanced
# ---------------------------------------------------------------------------
def test_all_speakers_is_complete_and_unique():
    assert len(config.ALL_SPEAKERS) == 28
    assert len(set(config.ALL_SPEAKERS)) == 28
    assert set(config.ALL_SPEAKERS) == set(config.CONTROL_IDS) | set(config.DYSARTHRIC_IDS)


def test_every_loso_prefix_covers_both_classes():
    """A run that stops early must still have evaluated both classes.

    Pre-repair, ALL_SPEAKERS was CONTROL_IDS + DYSARTHRIC_IDS, so the first 13
    folds were all healthy controls and any truncated run was single-class.
    """
    controls = set(config.CONTROL_IDS)
    for k in range(2, len(config.ALL_SPEAKERS) + 1):
        prefix = config.ALL_SPEAKERS[:k]
        n_control = sum(s in controls for s in prefix)
        assert 0 < n_control < k, (
            f"first {k} LOSO folds are single-class: {prefix}")


def test_loso_fold_iteration_follows_the_balanced_order():
    """iter_loso_folds must actually walk ALL_SPEAKERS — the ordering fix is
    worthless if the iterator sorts or regroups it."""
    df = pd.DataFrame({
        "Speaker_ID": config.ALL_SPEAKERS,
        "Group": ["Healthy Control" if s in set(config.CONTROL_IDS)
                  else "Dysarthric Patient" for s in config.ALL_SPEAKERS],
    })
    fold_ids = [fold_id for fold_id, _, _ in iter_loso_folds(df)]
    assert fold_ids == list(config.ALL_SPEAKERS)


# ---------------------------------------------------------------------------
# F5 — diagnostic runs must never be mistaken for experiments
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("run_name", [
    "_smoke_test",
    "_budget_bench_detection_acoustic",
    "_budget_bench_detection_attention_fusion",
    "_batch_bench_detection_fusion_32",
])
def test_diagnostic_runs_are_flagged(run_name):
    assert is_non_experiment(run_name)


@pytest.mark.parametrize("run_name", [
    "detection_attention_fusion", "primary_detection_deep_lora",
    "severity_fusion", "baseline_svm_detection_layer5",
])
def test_real_runs_are_not_flagged(run_name):
    assert not is_non_experiment(run_name)


# ---------------------------------------------------------------------------
# F8 — registry round-trip and the eligibility gate
# ---------------------------------------------------------------------------
@pytest.fixture
def registry_path(tmp_path):
    return tmp_path / "registry.csv"


def _record(registry_path, fold_id, index, status=FOLD_COMPLETED,
            labels="CF02=Healthy Control", n_classes=1, expected=28):
    record_fold(run_name="test_run", model="acoustic", task="detection",
                cv_protocol="loso", fold_id=fold_id, fold_index=index,
                expected_folds=expected, status=status,
                fold_description={"held_out_speakers": labels.split("=")[0],
                                  "speaker_labels": labels, "num_samples": 765},
                num_classes_present=n_classes, epochs_completed=7,
                runtime_s=100.0, registry_path=registry_path)


def test_registry_records_and_rolls_up(registry_path, monkeypatch):
    monkeypatch.setattr("src.training.reporting.REGISTRY_PATH", registry_path)
    _record(registry_path, "CF02", 1, labels="CF02=Healthy Control", n_classes=1)
    _record(registry_path, "F02", 2, labels="F02=Dysarthric Patient", n_classes=1)
    _record(registry_path, "CF03", 3, status=FOLD_SKIPPED_DEADLINE,
            labels="CF03=Healthy Control", n_classes=None)

    raw = load_registry(registry_path)
    assert len(raw) == 3

    summary = summarize_registry(registry_path)
    row = summary.iloc[0]
    assert row["completed_folds"] == 2
    assert row["skipped_folds"] == 1
    assert row["expected_folds"] == 28
    assert row["coverage"] == pytest.approx(2 / 28)
    assert row["status"] == "PARTIAL"
    # Neither fold alone had two classes, but the two together cover both —
    # which is exactly what makes a POOLED metric meaningful.
    assert bool(row["pooled_has_both_classes"]) is True
    assert row["valid_folds"] == 0


def test_registry_upserts_rather_than_duplicating(registry_path):
    """A resumed session re-records folds it loaded from cache. Duplicated rows
    would inflate completed_folds and hand the gate a false 100% coverage."""
    _record(registry_path, "CF02", 1)
    _record(registry_path, "CF02", 1)
    _record(registry_path, "CF02", 1)
    assert len(load_registry(registry_path)) == 1
    assert summarize_registry(registry_path).iloc[0]["completed_folds"] == 1


def test_single_class_pooled_detection_run_is_excluded():
    """The pre-repair headline result: two held-out controls, 99.7% accuracy.
    It must never reach the final table."""
    row = pd.Series({
        "run_name": "detection_deep_lora", "task": "detection",
        "completed_folds": 2, "expected_folds": 28, "coverage": 2 / 28,
        "valid_folds": 0, "pooled_has_both_classes": False, "status": "PARTIAL",
    })
    verdict = classify_result(row)
    assert verdict["tier"] == TIER_EXCLUDED
    assert "single-class" in verdict["reason"]


def test_incomplete_but_class_covering_run_is_preliminary():
    row = pd.Series({
        "run_name": "severity_deep_lora", "task": "severity",
        "completed_folds": 1, "expected_folds": 20, "coverage": 0.05,
        "valid_folds": 1, "pooled_has_both_classes": True, "status": "PARTIAL",
    })
    verdict = classify_result(row)
    assert verdict["tier"] == TIER_PRELIMINARY
    assert "1/20" in verdict["reason"]


def test_complete_run_is_final():
    row = pd.Series({
        "run_name": "detection_attention_fusion", "task": "detection",
        "completed_folds": 28, "expected_folds": 28, "coverage": 1.0,
        "valid_folds": 28, "pooled_has_both_classes": True, "status": "COMPLETED",
    })
    assert classify_result(row)["tier"] == TIER_FINAL


def test_never_executed_run_is_excluded():
    row = pd.Series({
        "run_name": "detection_attention_fusion", "task": "detection",
        "completed_folds": 0, "expected_folds": 28, "coverage": 0.0,
        "valid_folds": 0, "pooled_has_both_classes": False, "status": "FAILED",
    })
    verdict = classify_result(row)
    assert verdict["tier"] == TIER_EXCLUDED
    assert "Never executed" in verdict["reason"]


def test_full_loso_detection_run_with_degenerate_folds_is_final_not_preliminary():
    """Full LOSO detection: every fold is single-class by construction, but the
    pooled set covers both. With min_valid_folds=0 this is the intended
    base-paper protocol and must be reportable."""
    row = pd.Series({
        "run_name": "primary_detection_fusion", "task": "detection",
        "completed_folds": 28, "expected_folds": 28, "coverage": 1.0,
        "valid_folds": 0, "pooled_has_both_classes": True, "status": "COMPLETED",
    })
    assert classify_result(row, min_valid_folds=0)["tier"] == TIER_FINAL


def test_complete_loso_run_reaches_final_under_DEFAULT_settings():
    """Regression test for a real bug: build_result_tiers / notebook 6 call
    classify_result with its DEFAULT min_valid_folds=1 — they never pass the
    override the test above uses. Every LOSO fold is single-class by design
    (valid_folds stays 0 forever), so under the naive rule a complete, correct
    28/28 full-LOSO detection sweep — the actual base-paper protocol — would
    be permanently stuck at PRELIMINARY. classify_result must special-case
    cv_protocol == "loso" for detection so this reaches FINAL with no override,
    exactly as summarize_registry()'s real output (which does carry
    cv_protocol) would present it."""
    row = pd.Series({
        "run_name": "primary_detection_fusion", "task": "detection",
        "cv_protocol": "loso",
        "completed_folds": 28, "expected_folds": 28, "coverage": 1.0,
        "valid_folds": 0, "pooled_has_both_classes": True, "status": "COMPLETED",
    })
    verdict = classify_result(row)          # no override — the real call path
    assert verdict["tier"] == TIER_FINAL, verdict["reason"]


def test_incomplete_loso_run_still_preliminary_despite_the_exception():
    """The LOSO exception must only waive the valid-folds check, not coverage
    — a 2/28 LOSO run stays PRELIMINARY (or EXCLUDED if single-class), never
    FINAL, however the valid-folds rule is special-cased."""
    row = pd.Series({
        "run_name": "detection_deep_lora", "task": "detection",
        "cv_protocol": "loso",
        "completed_folds": 2, "expected_folds": 28, "coverage": 2 / 28,
        "valid_folds": 0, "pooled_has_both_classes": True, "status": "PARTIAL",
    })
    assert classify_result(row)["tier"] == TIER_PRELIMINARY


def test_screening_run_with_zero_valid_folds_stays_preliminary():
    """The LOSO exception must NOT leak into other protocols. Screening folds
    are constructed class-stratified (src.splits.build_screening_folds), so
    valid_folds == 0 there is a genuine anomaly worth flagging, not an
    expected consequence of the protocol."""
    row = pd.Series({
        "run_name": "detection_screen_acoustic", "task": "detection",
        "cv_protocol": "screening",
        "completed_folds": 8, "expected_folds": 8, "coverage": 1.0,
        "valid_folds": 0, "pooled_has_both_classes": True, "status": "COMPLETED",
    })
    verdict = classify_result(row)
    assert verdict["tier"] == TIER_PRELIMINARY
    assert "no individual fold" in verdict["reason"]
