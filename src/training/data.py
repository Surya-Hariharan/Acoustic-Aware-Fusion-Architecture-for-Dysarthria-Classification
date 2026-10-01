"""
Inside one LOSO fold: the manifest, the speaker-disjoint validation split,
class weights, the speaker map for the adversarial head, and the DataLoaders.
"""

import gc
import weakref
import zlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import UASpeechDataset
from src.preprocessing import segmental_standardizer, suprasegmental_standardizer
from src.scanning import (add_severity_labels, check_word_counts, filter_mic_channel,
                          scan_audio_files, validate_wav_headers)


def load_manifest() -> pd.DataFrame:
    """The M6 manifest (outputs/m6_manifest.csv), regenerated from the
    extracted audio if absent. Filepaths written on another machine or
    checkout are re-rooted onto this checkout's AUDIO_DIR."""
    if config.MANIFEST_PATH.exists():
        df = pd.read_csv(config.MANIFEST_PATH)
        if len(df) and not Path(df["Filepath"].iloc[0]).exists():
            rerooted = [str(config.AUDIO_DIR / s / f) for s, f in zip(df["Speaker_ID"], df["Filename"])]
            if Path(rerooted[0]).exists():
                df["Filepath"] = rerooted
        return df

    df = filter_mic_channel(scan_audio_files())
    df = validate_wav_headers(df)          # drops the corpus's zero-filled files
    check_word_counts(df)
    df = add_severity_labels(df)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.MANIFEST_PATH, index=False)
    return df


def speaker_disjoint_train_val_split(df: pd.DataFrame, seed: int, fold_id: str = "",
                                     val_speakers_per_class: int = 1,
                                     min_train_speakers_per_class: int = 2
                                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Move whole SPEAKERS from a fold's training portion to validation:
    one per severity class, but only while that class keeps at least
    `min_train_speakers_per_class` training speakers. For the 15-speaker LOSO
    that is 3-4 validation speakers per fold. Seeded per (seed, fold_id)."""
    rng = np.random.default_rng([seed, zlib.crc32(fold_id.encode("utf-8"))])
    val_speakers = []
    for _, group in sorted(df.groupby("Severity"), key=lambda item: str(item[0])):
        speakers = sorted(group["Speaker_ID"].unique().tolist())
        n_val = min(val_speakers_per_class, len(speakers) - min_train_speakers_per_class)
        if n_val > 0:
            val_speakers.extend(sorted(rng.choice(speakers, size=n_val, replace=False).tolist()))
    if not val_speakers:
        raise ValueError(f"No class has more than {min_train_speakers_per_class} training "
                         f"speakers; no speaker-disjoint validation set for fold {fold_id!r}.")
    val_mask = df["Speaker_ID"].isin(val_speakers)
    return (df[~val_mask].sample(frac=1.0, random_state=seed).reset_index(drop=True),
            df[val_mask].sample(frac=1.0, random_state=seed).reset_index(drop=True))


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency (balanced) weights over the 4 severity classes."""
    counts = (train_df["Severity"].map(config.SEVERITY_LABEL_MAP).value_counts()
              .reindex(range(config.NUM_CLASSES), fill_value=0).astype(float).clip(lower=1.0))
    return torch.tensor((counts.sum() / (config.NUM_CLASSES * counts)).to_numpy(),
                        dtype=torch.float32)


def build_speaker_label_map(train_df: pd.DataFrame) -> Dict[str, int]:
    """0..N-1 per training speaker of this fold, for the adversarial head."""
    return {speaker: i for i, speaker in enumerate(sorted(train_df["Speaker_ID"].unique()))}


def _init_worker(worker_id: int) -> None:
    """One intra-op thread per worker: items are tiny, the parallelism that
    matters is across workers, and oversubscription only adds contention."""
    torch.set_num_threads(1)


_LIVE_LOADERS: "weakref.WeakSet[DataLoader]" = weakref.WeakSet()


def shutdown_loaders() -> int:
    """Stop every worker process build_loaders started. A persistent worker
    otherwise lives until its iterator is garbage-collected, which a notebook
    kernel delays; across folds they accumulated until Windows ran out of
    commit (error 1455). Called after every fold, whatever its outcome."""
    stopped = 0
    for loader in list(_LIVE_LOADERS):
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
            stopped += 1
        loader._iterator = None
    _LIVE_LOADERS.clear()
    gc.collect()
    return stopped


def build_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  batch_size: int, pin_memory: bool, speaker_label_map: Dict[str, int],
                  num_workers: int = config.TRAIN_NUM_WORKERS,
                  eval_num_workers: int = config.EVAL_NUM_WORKERS,
                  test_num_workers: int = config.TEST_NUM_WORKERS
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Train/val/test loaders for one fold. Segmental and suprasegmental
    standardization statistics come from train_df ONLY and are reused for
    val/test, so the held-out speaker never influences its own normalization.
    Only the training set carries speaker labels: validation speakers are
    disjoint from training, so they are not in the map."""
    segmental_stats = segmental_standardizer(train_df["Filepath"])
    supra_stats = suprasegmental_standardizer(train_df["Filepath"])

    def dataset(df: pd.DataFrame, labels: bool) -> UASpeechDataset:
        return UASpeechDataset(df, speaker_label_map=speaker_label_map if labels else None,
                               segmental_stats=segmental_stats, supra_stats=supra_stats)

    def options(workers: int, persistent: bool) -> dict:
        kwargs = dict(num_workers=workers, pin_memory=pin_memory)
        if workers > 0:
            kwargs.update(persistent_workers=persistent, worker_init_fn=_init_worker,
                          prefetch_factor=config.DATALOADER_PREFETCH_FACTOR)
        return kwargs

    loaders = (
        DataLoader(dataset(train_df, labels=True), batch_size=batch_size, shuffle=True,
                   drop_last=len(train_df) > batch_size, **options(num_workers, persistent=True)),
        DataLoader(dataset(val_df, labels=False), batch_size=batch_size, shuffle=False,
                   **options(eval_num_workers, persistent=True)),
        DataLoader(dataset(test_df, labels=False), batch_size=batch_size, shuffle=False,
                   **options(test_num_workers, persistent=False)),
    )
    for loader in loaders:
        _LIVE_LOADERS.add(loader)
    return loaders
