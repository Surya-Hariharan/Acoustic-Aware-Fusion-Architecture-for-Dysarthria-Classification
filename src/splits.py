"""
Severity cross-validation: full-population Leave-One-Speaker-Out over the 15
dysarthric speakers. No speaker is dropped to balance classes; the 4/3/3/5
imbalance is handled by the class-weighted loss and macro metrics instead.
"""

from typing import Iterator, Tuple

import pandas as pd

from src import config


def get_severity_loso_split(df: pd.DataFrame,
                            test_speaker_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) over dysarthric speakers, with one speaker held out."""
    dysarthric = df[df["Speaker_ID"].isin(config.DYSARTHRIC_IDS)]
    return (dysarthric[dysarthric["Speaker_ID"] != test_speaker_id],
            dysarthric[dysarthric["Speaker_ID"] == test_speaker_id])


def iter_severity_loso_folds(df: pd.DataFrame) -> Iterator[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield (held_out_speaker, train_df, test_df) for all 15 speakers, in
    config.SEVERITY_LOSO_ORDER (class-interleaved, so a partial run covers
    every class early; a complete run is order-independent)."""
    for speaker_id in config.SEVERITY_LOSO_ORDER:
        train_df, test_df = get_severity_loso_split(df, speaker_id)
        yield speaker_id, train_df, test_df
