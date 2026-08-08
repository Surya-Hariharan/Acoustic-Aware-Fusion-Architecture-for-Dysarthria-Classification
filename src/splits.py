"""
Cross-validation split generation.

Detection : Leave-One-Speaker-Out (LOSO) across all 28 speakers — the
            base-paper protocol, and also a cheap "screening" k-fold used
            to rank the six ablation variants before spending full-LOSO
            GPU time on any of them.
Severity  : balanced to 3 speakers per class (by excluding
            DROPPED_FOR_BALANCE), then leave-one-speaker-per-class-out,
            giving 3^4 = 81 iterations — the base-paper protocol. A random
            subsample of those 81 combinations is available for the same
            reason: 81 folds x every ablation variant is the single
            largest GPU-time item in the training notebook.
"""

import random
from itertools import product
from typing import Iterator, List, Optional, Tuple

import pandas as pd

from src import config
from src.console import print_header, print_subheader, print_kv, print_status


# ---------------------------------------------------------------------------
# Detection task: Leave-One-Speaker-Out
# ---------------------------------------------------------------------------
def get_loso_split(df: pd.DataFrame,
                   test_speaker_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (train, test) with one speaker held out for testing."""
    train_df = df[df["Speaker_ID"] != test_speaker_id]
    test_df = df[df["Speaker_ID"] == test_speaker_id]
    return train_df, test_df


def iter_loso_folds(df: pd.DataFrame) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield (held_out_speaker, train_df, test_df) for every speaker."""
    for speaker_id in config.ALL_SPEAKERS:
        train_df, test_df = get_loso_split(df, speaker_id)
        yield speaker_id, train_df, test_df


def _chunk_balanced(items: List[str], n_chunks: int) -> List[List[str]]:
    """Split `items` into `n_chunks` groups as evenly as possible (last
    groups get one fewer item when it doesn't divide evenly)."""
    n_chunks = min(n_chunks, len(items))
    base, remainder = divmod(len(items), n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def build_screening_folds(n_splits: int = 8,
                          seed: int = config.DEFAULT_SEED) -> List[List[str]]:
    """
    Speaker-grouped, class-stratified k-fold held-out speaker lists — a
    cheap stand-in for full 28-fold LOSO, used only to rank the six
    ablation variants (src.training.models.MODEL_NAMES) before committing
    full-LOSO GPU time to whichever wins. Every held-out group mixes
    control and dysarthric speakers so screening accuracy stays meaningful
    with fewer folds, unlike a single LOSO fold (always one class).
    """
    rng = random.Random(seed)
    controls, dysarthric = list(config.CONTROL_IDS), list(config.DYSARTHRIC_IDS)
    rng.shuffle(controls)
    rng.shuffle(dysarthric)
    control_chunks = _chunk_balanced(controls, n_splits)
    dysarthric_chunks = _chunk_balanced(dysarthric, n_splits)
    return [c + d for c, d in zip(control_chunks, dysarthric_chunks)]


def iter_screening_folds(df: pd.DataFrame, n_splits: int = 8,
                         seed: int = config.DEFAULT_SEED
                         ) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield (fold_id, train_df, test_df) for the detection screening protocol."""
    for i, held_out in enumerate(build_screening_folds(n_splits, seed), start=1):
        train_df = df[~df["Speaker_ID"].isin(held_out)]
        test_df = df[df["Speaker_ID"].isin(held_out)]
        yield f"screen{i}", train_df, test_df


def summarize_detection_splits(df: pd.DataFrame) -> None:
    """Print an overview of the detection LOSO protocol."""
    print_header("Detection Splits (Leave-One-Speaker-Out)")
    print_kv("Total LOSO folds", len(config.ALL_SPEAKERS))

    example_speaker = config.ALL_SPEAKERS[0]
    train_df, test_df = get_loso_split(df, example_speaker)
    print_subheader(f"Example fold ({example_speaker} held out)")
    print_kv("Train samples", len(train_df))
    print_kv("Test samples", len(test_df))


# ---------------------------------------------------------------------------
# Severity task: balanced leave-one-speaker-per-class-out
# ---------------------------------------------------------------------------
def build_severity_folds(df: pd.DataFrame) -> list:
    """
    Return the list of held-out speaker combinations for the severity task,
    one speaker per severity class per fold (expected: 81 combinations).
    """
    print_header("Severity Splits (Balanced Leave-One-Per-Class-Out)")
    print_kv("Speakers dropped for balance", config.DROPPED_FOR_BALANCE)

    df_severity = df[df["Speaker_ID"].isin(config.DYSARTHRIC_IDS)]
    df_balanced = df_severity[
        ~df_severity["Speaker_ID"].isin(config.DROPPED_FOR_BALANCE)]

    severity_groups = (df_balanced.groupby("Severity")["Speaker_ID"]
                       .unique().to_dict())

    print_subheader("Speakers per severity class")
    for severity, speakers in severity_groups.items():
        print_kv(severity, ", ".join(sorted(speakers)))

    balanced = all(len(v) == 3 for v in severity_groups.values())
    print_status("3 speakers per class - balanced" if balanced
                 else "Classes NOT balanced - adjust DROPPED_FOR_BALANCE",
                 ok=balanced)

    combos = list(product(*severity_groups.values()))
    print_kv("Severity LOSO iterations", f"{len(combos)} (expected 81)")
    return combos


def sample_severity_folds(combos: list, n_samples: int,
                          seed: int = config.DEFAULT_SEED) -> list:
    """
    Deterministic random subset of the 81 leave-one-per-class-out
    combinations — the severity task is 81 folds x every ablation variant
    carried forward from detection, the largest GPU-time item in the
    training notebook, so a smaller representative sample is what makes a
    severity comparison affordable at all. Returns `combos` unchanged if
    n_samples >= len(combos).
    """
    if n_samples >= len(combos):
        return combos
    return random.Random(seed).sample(combos, n_samples)


def get_severity_split(df: pd.DataFrame,
                       held_out: tuple) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split severity data into (train, test) for one held-out combination."""
    df_balanced = df[
        df["Speaker_ID"].isin(config.DYSARTHRIC_IDS) &
        ~df["Speaker_ID"].isin(config.DROPPED_FOR_BALANCE)]
    test_mask = df_balanced["Speaker_ID"].isin(held_out)
    return df_balanced[~test_mask], df_balanced[test_mask]
