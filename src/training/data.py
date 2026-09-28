"""
Data loading for train.py: manifest access, stratified train/val split,
DataLoader construction, and class weighting.

LOSO (detection) and the primary full-population severity LOSO split already come
from src.splits — this module only handles what happens inside a fold's
train portion (carving out a validation slice) and turning DataFrames into
PyTorch DataLoaders.
"""

import zlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import UASpeechDataset
from src.praat import praat_standardizer
from src.preprocessing import segmental_standardizer, suprasegmental_standardizer
from src.scanning import (add_severity_labels, check_word_counts,
                          filter_mic_channel, scan_audio_files,
                          validate_wav_headers)
from src.training.models import GATED_FUSION_MODELS

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
        df = pd.read_csv(config.MANIFEST_PATH)
        # Filepath is absolute and rooted wherever the manifest was written
        # (another checkout location, another machine). If those files are
        # not here but the same corpus is under this checkout's AUDIO_DIR
        # (<Speaker>/<Filename>, extraction's layout), re-root onto it.
        if len(df) and not Path(df["Filepath"].iloc[0]).exists():
            rerooted = [str(config.AUDIO_DIR / s / f)
                        for s, f in zip(df["Speaker_ID"], df["Filename"])]
            if Path(rerooted[0]).exists():
                df["Filepath"] = rerooted
        return df

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

    LEGACY (TrainingConfig.val_protocol="utterance"). Utterance-level, NOT
    speaker-disjoint: every validation speaker also contributes ~90% of its
    utterances to training. It does not leak the held-out TEST speaker, but
    it makes validation a within-speaker memorization check — for severity,
    where the label is a speaker attribute, val loss then selects checkpoints
    and stops early on a signal that cannot see cross-speaker failure (the
    audited 11-fold run: val F1 flat at ~0.36-0.44 while held-out speakers
    scored 0% or ~98%). Use speaker_disjoint_train_val_split instead.
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


def speaker_disjoint_train_val_split(df: pd.DataFrame, label_column: str, seed: int,
                                     fold_id: str = "",
                                     val_speakers_per_class: int = 1,
                                     min_train_speakers_per_class: int = 2
                                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carve whole SPEAKERS out of a fold's training portion for validation:

        train speakers -> training
        val speakers   -> model selection / early stopping / LR schedule
        test speaker   -> the outer LOSO evaluation (already removed upstream)

    so no validation utterance comes from a speaker the model trained on.
    Per class, `val_speakers_per_class` speakers move to validation, but only
    while at least `min_train_speakers_per_class` speakers of that class stay
    in training — a class is never reduced to a single training speaker (its
    "class" signal would then be one person's voice) and never vanishes.

    For the 15-speaker severity LOSO (4 Very Low / 3 Low / 3 Mid / 5 High)
    that yields 3-4 validation speakers per fold, one per eligible class, and
    10-11 training speakers with >= 2 per class. The choice is seeded per
    (seed, fold_id), so it is reproducible and differs across folds.
    """
    rng = np.random.default_rng([seed, zlib.crc32(fold_id.encode("utf-8"))])
    val_speakers = []
    for _, group in sorted(df.groupby(label_column), key=lambda item: str(item[0])):
        speakers = sorted(group["Speaker_ID"].unique().tolist())
        n_val = min(val_speakers_per_class, len(speakers) - min_train_speakers_per_class)
        if n_val <= 0:
            continue
        val_speakers.extend(sorted(rng.choice(speakers, size=n_val, replace=False).tolist()))
    if not val_speakers:
        raise ValueError(
            f"No class has more than {min_train_speakers_per_class} training speakers, "
            f"so no speaker-disjoint validation set can be formed for fold {fold_id!r}.")

    val_mask = df["Speaker_ID"].isin(val_speakers)
    train_df = df[~val_mask].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = df[val_mask].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def split_train_val(df: pd.DataFrame, label_column: str, protocol: str, seed: int,
                    fold_id: str = "", val_fraction: float = config.DEFAULT_VAL_FRACTION
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch on TrainingConfig.val_protocol: "speaker" (default,
    speaker_disjoint_train_val_split) or "utterance" (legacy
    stratified_train_val_split)."""
    if protocol == "speaker":
        return speaker_disjoint_train_val_split(df, label_column, seed, fold_id=fold_id)
    if protocol == "utterance":
        return stratified_train_val_split(df, label_column, val_fraction, seed)
    raise ValueError(f"val_protocol must be 'speaker' or 'utterance', got {protocol!r}")


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


# Variants with no acoustic (MFCC) pathway. Their Dataset skips MFCC
# extraction entirely — see UASpeechDataset(include_mfcc=...). Every other
# variant either is the MFCC CNN or fuses with it, so it needs the tensor.
MODELS_WITHOUT_MFCC = frozenset({"deep_frozen", "deep_lora"})

# The three-branch severity model and its ablations — the only ones whose
# Dataset needs the segmental/suprasegmental tensors and a per-fold speaker
# label map (see src.dataset.UASpeechDataset's include_three_branch/
# speaker_label_map and src.models.gated_fusion.GatedFusionModel).
MODELS_WITH_THREE_BRANCH = GATED_FUSION_MODELS


def build_speaker_label_map(train_df: pd.DataFrame) -> Dict[str, int]:
    """Contiguous 0..N-1 integer id per TRAINING speaker in this fold, for
    the adversarial speaker head (src.models.gated_fusion.SpeakerHead).
    Built fresh per fold (the training-speaker set changes every LOSO
    fold) — sorted for determinism, not because the ordering itself
    matters to the (discarded-at-inference) speaker head."""
    speakers = sorted(train_df["Speaker_ID"].unique().tolist())
    return {speaker: i for i, speaker in enumerate(speakers)}


def _init_worker(worker_id: int) -> None:
    """Pin each DataLoader worker to a single intra-op thread.

    torch defaults its intra-op pool to the machine's core count, and every
    worker process gets its own pool — so num_workers=4 on Kaggle's 4-vCPU
    box asks for 16 threads on 4 cores. The oversubscription costs more in
    contention and context switching than the per-item ops (MFCC/STFT on a
    4-second clip) could ever recover from parallelism, since each item is
    already small and the parallelism that matters is across workers.

    Numerically neutral: these reductions are deterministic at any thread
    count for these sizes, and tests/test_vad_span_cache.py's equality gate
    runs single-threaded against the pre-change outputs.
    """
    torch.set_num_threads(1)


def build_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  batch_size: int, num_workers: int, pin_memory: bool,
                  praat_table: Optional[pd.DataFrame] = None,
                  frozen_embedding_table: Optional[Dict[str, np.ndarray]] = None,
                  model_name: Optional[str] = None,
                  speaker_label_map: Optional[Dict[str, int]] = None,
                  eval_num_workers: Optional[int] = None,
                  test_num_workers: Optional[int] = None
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap the three fold DataFrames into DataLoaders.

    When praat_table is given (Phase 6's Model F), the standardization statistics
    are computed from the TRAIN SPLIT ONLY and then applied to val and test. Using
    global statistics would leak the held-out speaker's acoustic distribution into
    the normalization of the very fold that is meant to be measuring generalization
    to that speaker — a subtle leak, but a real one in a LOSO protocol, and exactly
    the kind a reviewer will ask about.

    frozen_embedding_table (Filepath -> 768-dim np.ndarray) is given only for
    the deep_frozen/fusion_frozen variants (see src.training.runner.run_fold) —
    the frozen wav2vec2 embedding is identical across every fold/epoch (the
    backbone never updates), so it is computed once for the whole dataset via
    src.training.baseline.extract_frozen_embeddings_masked and handed to every
    fold's Dataset here rather than recomputed on every forward pass.

    speaker_label_map (Speaker_ID -> contiguous int, see
    build_speaker_label_map) is given only for MODELS_WITH_THREE_BRANCH, and
    applied to the train Dataset, and to val only when every val speaker is
    also a training speaker (see val_has_speaker_labels below). The held-out
    TEST speaker is never a key in this map by construction (that is the
    whole point of a LOSO fold), so the test Dataset never looks it up.

    For MODELS_WITH_THREE_BRANCH, segmental/suprasegmental channel
    normalization statistics (src.preprocessing.segmental_standardizer /
    suprasegmental_standardizer) are likewise computed from train_df's
    filepaths only and reused unchanged for val/test — the identical
    leakage discipline as praat_stats above, applied to the two framewise
    branches instead of the utterance-level Praat table.
    """
    praat_stats = None
    if praat_table is not None:
        praat_stats = praat_standardizer(praat_table, train_df["Filename"])

    # Unknown/None model_name keeps MFCC on — the safe default, since a model
    # that needs it and doesn't get it fails loudly, whereas one that skips it
    # unnecessarily only costs time.
    include_mfcc = model_name not in MODELS_WITHOUT_MFCC
    include_three_branch = model_name in MODELS_WITH_THREE_BRANCH

    # Channel-wise normalization statistics for the Segmental/Suprasegmental
    # branches, computed from THIS FOLD'S TRAIN SPLIT FILEPATHS ONLY (never
    # val/test) — the same leakage discipline as praat_stats above. Reused
    # unchanged for this fold's val and test Datasets, so the held-out LOSO
    # speaker's utterances never influence their own normalization.
    segmental_stats = None
    supra_stats = None
    if include_three_branch:
        segmental_stats = segmental_standardizer(train_df["Filepath"])
        supra_stats = suprasegmental_standardizer(train_df["Filepath"])

    def dataset(df: pd.DataFrame, with_speaker_labels: bool = False) -> UASpeechDataset:
        return UASpeechDataset(
            df, praat_table=praat_table, praat_stats=praat_stats,
            frozen_embedding_table=frozen_embedding_table, include_mfcc=include_mfcc,
            include_three_branch=include_three_branch,
            speaker_label_map=(speaker_label_map if with_speaker_labels else None),
            segmental_stats=segmental_stats, supra_stats=supra_stats)

    # __getitem__ does real CPU work per utterance (torchaudio.load, resample,
    # VAD trim, MFCC + deltas) - with num_workers=0 that runs synchronously in
    # the main process, so the GPU sits idle waiting on it between every
    # batch. This is the usual reason a training run shows ~0% GPU
    # utilization even though the model itself is correctly on cuda:
    # persistent_workers + prefetch_factor let background worker processes
    # prepare the next batch while the current one trains on the GPU.
    #
    # Worker budget per split (config "Local hardware profile"): on Windows
    # each worker is a spawned process holding its own torch import, so
    # workers cost RAM, not just cores. Train gets `num_workers` persistent
    # workers; validation `eval_num_workers` persistent ones (reused every
    # epoch — respawning them per epoch would cost seconds each time); the
    # test loader runs once per fold (config.TEST_NUM_WORKERS, 0 = in the main
    # process), so it never holds worker processes in RAM during training.
    eval_num_workers = num_workers if eval_num_workers is None else eval_num_workers
    test_num_workers = (config.TEST_NUM_WORKERS if test_num_workers is None
                        else test_num_workers)

    def loader_kwargs(workers: int, persistent: bool) -> dict:
        kwargs = dict(num_workers=workers, pin_memory=pin_memory)
        if workers > 0:
            kwargs.update(persistent_workers=persistent,
                          prefetch_factor=config.DATALOADER_PREFETCH_FACTOR,
                          worker_init_fn=_init_worker)
        return kwargs

    # Speaker labels for validation only when every validation speaker is a
    # TRAINING speaker (the legacy utterance-level split). Under the
    # speaker-disjoint split none of them is — they are not keys of
    # speaker_label_map — so validation carries no speaker-adversary term and
    # val loss is the severity objective (plus the complementarity penalty).
    val_has_speaker_labels = (speaker_label_map is not None
                              and set(val_df["Speaker_ID"]).issubset(speaker_label_map))

    train_loader = DataLoader(
        dataset(train_df, with_speaker_labels=True), batch_size=batch_size, shuffle=True,
        drop_last=len(train_df) > batch_size, **loader_kwargs(num_workers, persistent=True))
    val_loader = DataLoader(
        dataset(val_df, with_speaker_labels=val_has_speaker_labels), batch_size=batch_size,
        shuffle=False, **loader_kwargs(eval_num_workers, persistent=True))
    test_loader = DataLoader(
        dataset(test_df, with_speaker_labels=False), batch_size=batch_size, shuffle=False,
        **loader_kwargs(test_num_workers, persistent=False))
    return train_loader, val_loader, test_loader
