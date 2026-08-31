"""
Fold-construction checks for the PRIMARY severity protocol
(src.splits.iter_severity_loso_folds / get_severity_loso_split): every one
of the 15 dysarthric speakers is used exactly once as the held-out test
speaker, no speaker ever appears in both train and test of the same fold,
no speaker is silently dropped (unlike the legacy balanced/3-per-class
protocol), and every fold's test set is entirely one speaker's utterances.

Run with: pytest tests/test_severity_loso_folds.py -v
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.splits import get_severity_loso_split, iter_severity_loso_folds


def _synthetic_manifest() -> pd.DataFrame:
    """A manifest-shaped DataFrame covering every dysarthric AND control
    speaker with a few synthetic utterances each — no real audio needed,
    since fold construction only reads Speaker_ID/Severity."""
    rows = []
    for speaker in config.ALL_SPEAKERS:
        severity = config.SEVERITY_MAP.get(speaker, "N/A (Control)")
        for i in range(3):
            rows.append({"Speaker_ID": speaker, "Severity": severity,
                        "Filename": f"{speaker}_{i}.wav"})
    return pd.DataFrame(rows)


def test_every_dysarthric_speaker_is_held_out_exactly_once():
    df = _synthetic_manifest()
    held_out = [speaker for speaker, _, _ in iter_severity_loso_folds(df)]
    assert sorted(held_out) == sorted(config.DYSARTHRIC_IDS)
    assert len(held_out) == len(set(held_out)) == 15


def test_no_speaker_is_dropped_unlike_the_balanced_protocol():
    # The full-population protocol must NOT exclude config.DROPPED_FOR_BALANCE
    # speakers — that exclusion is specific to the legacy balanced/3-per-class
    # protocol (src.splits.build_severity_folds), not this one.
    df = _synthetic_manifest()
    held_out = {speaker for speaker, _, _ in iter_severity_loso_folds(df)}
    for dropped_speaker in config.DROPPED_FOR_BALANCE:
        assert dropped_speaker in held_out, (
            f"{dropped_speaker} was excluded from the full-population LOSO "
            "protocol — it should only be excluded from the secondary "
            "balanced/3-per-class protocol")


def test_train_and_test_never_share_a_speaker():
    df = _synthetic_manifest()
    for speaker, train_df, test_df in iter_severity_loso_folds(df):
        train_speakers = set(train_df["Speaker_ID"].unique())
        test_speakers = set(test_df["Speaker_ID"].unique())
        assert test_speakers == {speaker}
        assert speaker not in train_speakers
        assert train_speakers.isdisjoint(test_speakers)


def test_every_fold_train_split_excludes_control_speakers():
    # Severity is defined only for dysarthric speakers — controls must never
    # appear in either split of a severity fold.
    df = _synthetic_manifest()
    for _, train_df, test_df in iter_severity_loso_folds(df):
        assert set(train_df["Speaker_ID"]).isdisjoint(config.CONTROL_IDS)
        assert set(test_df["Speaker_ID"]).isdisjoint(config.CONTROL_IDS)


def test_every_severity_class_is_covered_across_the_full_sweep():
    # Every severity class must appear as SOME fold's held-out test set,
    # across the full 15-fold sweep (even though no single fold is
    # class-balanced the way the legacy leave-one-per-class-out protocol is).
    df = _synthetic_manifest()
    covered_classes = set()
    for _, _, test_df in iter_severity_loso_folds(df):
        covered_classes.update(test_df["Severity"].unique())
    assert covered_classes == {"Very Low", "Low", "Mid", "High"}


def test_get_severity_loso_split_matches_iteration():
    df = _synthetic_manifest()
    train_df, test_df = get_severity_loso_split(df, "M01")
    assert set(test_df["Speaker_ID"].unique()) == {"M01"}
    assert "M01" not in set(train_df["Speaker_ID"].unique())
    assert set(train_df["Speaker_ID"]).issubset(set(config.DYSARTHRIC_IDS))


if __name__ == "__main__":
    test_every_dysarthric_speaker_is_held_out_exactly_once()
    test_no_speaker_is_dropped_unlike_the_balanced_protocol()
    test_train_and_test_never_share_a_speaker()
    test_every_fold_train_split_excludes_control_speakers()
    test_every_severity_class_is_covered_across_the_full_sweep()
    test_get_severity_loso_split_matches_iteration()
    print("All severity LOSO fold-construction tests passed.")
