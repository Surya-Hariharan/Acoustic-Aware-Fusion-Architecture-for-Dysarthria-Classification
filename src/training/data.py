"""
Data loading for train.py: manifest access, stratified train/val split,
DataLoader construction, and class weighting.

LOSO (detection) and the balanced 81-fold split (severity) already come
from src.splits — this module only handles what happens inside a fold's
train portion (carving out a validation slice) and turning DataFrames into
PyTorch DataLoaders.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import UASpeechDataset
from src.praat import praat_standardizer
from src.scanning import (add_severity_labels, check_word_counts,
                          filter_mic_channel, scan_audio_files,
                          validate_wav_headers)

TASK_LABEL_COLUMN = {"detection": "Group", "severity": "Severity"}
TASK_LABEL_MAP = {"detection": config.GROUP_LABEL_MAP, "severity": config.SEVERITY_LABEL_MAP}


def load_manifest() -> pd.DataFrame:
    """Load the M6 manifest written by the data pipeline notebook/script.

    Regenerates it from raw audio if missing, so training is runnable
    standalone without requiring notebook 01 to have been run first. The
    regeneration path must mirror notebook 01 exactly — including
    validate_wav_headers(), without which the 39 zero-filled UA-Speech files
    (see README "Data verification note") would re-enter the manifest and
    torchaudio would then fail, or worse, silently train on silence.
    """
    if config.MANIFEST_PATH.exists():
        return pd.read_csv(config.MANIFEST_PATH)

    df_audio = scan_audio_files()
    df_m6 = filter_mic_channel(df_audio)
    df_m6 = validate_wav_headers(df_m6)
    check_word_counts(df_m6)
    df_m6 = add_severity_labels(df_m6)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_m6.to_csv(config.MANIFEST_PATH, index=False)
    return df_m6


def stratified_train_val_split(df: pd.DataFrame, label_column: str,
                               val_fraction: float, seed: int
                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split off a per-class fraction of df for validation.

    Utterance-level (not speaker-disjoint): the held-out fold speaker is
    already excluded upstream by the LOSO/severity split, so a validation
    speaker overlapping with train here does not leak test-fold identity.
    """
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []

    for _, group in df.groupby(label_column):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        n_val = min(max(n_val, 1 if len(idx) > 1 else 0), len(idx) - 1) if len(idx) > 1 else 0
        val_parts.append(group.loc[idx[:n_val]])
        train_parts.append(group.loc[idx[n_val:]])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def compute_class_weights(train_df: pd.DataFrame, task: str) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss, from the train split."""
    label_column = TASK_LABEL_COLUMN[task]
    label_map = TASK_LABEL_MAP[task]
    num_classes = config.NUM_CLASSES[task]

    labels = train_df[label_column].map(label_map)
    counts = labels.value_counts().reindex(range(num_classes), fill_value=0).astype(float)
    counts = counts.clip(lower=1.0)                 # avoid div-by-zero for absent classes
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights.to_numpy(), dtype=torch.float32)


def build_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  batch_size: int, num_workers: int, pin_memory: bool,
                  praat_table: Optional[pd.DataFrame] = None
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap the three fold DataFrames into DataLoaders.

    When praat_table is given (Phase 6's Model F), the standardization statistics
    are computed from the TRAIN SPLIT ONLY and then applied to val and test. Using
    global statistics would leak the held-out speaker's acoustic distribution into
    the normalization of the very fold that is meant to be measuring generalization
    to that speaker — a subtle leak, but a real one in a LOSO protocol, and exactly
    the kind a reviewer will ask about.
    """
    praat_stats = None
    if praat_table is not None:
        praat_stats = praat_standardizer(praat_table, train_df["Filename"])

    def dataset(df: pd.DataFrame) -> UASpeechDataset:
        return UASpeechDataset(df, praat_table=praat_table, praat_stats=praat_stats)

    train_loader = DataLoader(
        dataset(train_df), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=len(train_df) > batch_size)
    val_loader = DataLoader(
        dataset(val_df), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(
        dataset(test_df), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader
