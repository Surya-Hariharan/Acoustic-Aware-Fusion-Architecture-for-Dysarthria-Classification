"""
Audio preprocessing utilities.

Every utterance goes through the same deterministic chain:
  1. Load and mix down to mono.
  2. Resample to 16 kHz.
  3. Trim leading/trailing silence with Silero VAD (src/vad.py), preserving
     internal pauses.
  4. Pad or truncate to a fixed 4-second window.

Both the MFCC branch (Acoustic Pathway) and the raw-waveform branch (Deep
Pathway / wav2vec2) consume the SAME output of this function — see
UASpeechDataset.__getitem__ (src/dataset.py), which calls
load_and_preprocess_cached exactly once per item and derives both `waveform`
and `mfcc` from it. There is no separate VAD path per branch.

MFCC extraction produces 13 coefficients plus delta and delta-delta,
giving the 39-dimensional per-frame representation used by the
Acoustic Pathway (matching the base paper's baseline features).
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torchaudio

from src import config
from src import vad as vad_module
from src.console import print_kv, print_status, print_subheader, progress


def load_and_preprocess(filepath: str) -> Tuple[torch.Tensor, int]:
    """Load one utterance and return a (1, MAX_SAMPLES) 16 kHz mono waveform,
    plus the number of real (pre-padding) samples in it — everything past that
    index is zero-padding, not audio. Callers that need an attention mask for
    wav2vec2 (see src.models.deep_pathway) derive it from this length rather
    than from the waveform's content, since a genuinely silent stretch of real
    speech must NOT be masked out the way padding must."""
    waveform, _ = _load_resampled(filepath)
    waveform, _ = vad_module.apply_vad(waveform, config.TARGET_SR)
    return _pad_or_truncate(waveform)


def load_and_preprocess_with_stats(filepath: str) -> Tuple[torch.Tensor, int, Dict]:
    """Same as load_and_preprocess, but also returns the VAD stats dict
    (original_duration_s, speech_duration_s, speech_ratio, num_segments,
    fallback_used, ...). Uncached and slower — used only for the one-off
    batch VAD-stats pass (compute_vad_stats_batch) and validation plots, not
    the training hot path (see load_and_preprocess_cached below)."""
    waveform, _ = _load_resampled(filepath)
    waveform, stats = vad_module.apply_vad(waveform, config.TARGET_SR)
    waveform, valid_length = _pad_or_truncate(waveform)
    return waveform, valid_length, stats


def _load_resampled(filepath: str) -> Tuple[torch.Tensor, int]:
    """Load, mix down to mono, and resample to TARGET_SR. Shared by every
    preprocessing entry point so VAD always sees the same 16 kHz mono signal."""
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:                                # stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != config.TARGET_SR:                               # resample
        waveform = torchaudio.functional.resample(waveform, sr, config.TARGET_SR)
    return waveform, config.TARGET_SR


def _pad_or_truncate(waveform: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Pad/truncate to MAX_SAMPLES and report how many samples were real audio
    (the rest is zero-padding) — the length an attention mask must be built
    from downstream."""
    valid_length = min(waveform.shape[1], config.MAX_SAMPLES)
    if waveform.shape[1] < config.MAX_SAMPLES:
        pad = config.MAX_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :config.MAX_SAMPLES]
    return waveform, valid_length


def compute_vad_stats_batch(df: pd.DataFrame, cache_path: Optional[Path] = None,
                            use_cache: bool = True) -> pd.DataFrame:
    """
    Run VAD over every row of df (expects Filename, Speaker_ID, Filepath — the
    M6 manifest shape) and return a stats DataFrame joinable back onto it by
    Filename. Mirrors src.praat.extract_praat_features_batch's cache-check-
    and-batch pattern. Never raises on a single file's failure — apply_vad's
    own fallback contract guarantees a stats row for every utterance.
    """
    cache_path = cache_path or config.VAD_STATS_PATH
    if use_cache and cache_path is not None and Path(cache_path).exists():
        cached = pd.read_csv(cache_path)
        if set(df["Filename"]).issubset(set(cached["Filename"])):
            print_kv("VAD stats", f"loaded from cache ({cache_path})")
            return cached[cached["Filename"].isin(df["Filename"])].reset_index(drop=True)

    records = []
    print_subheader(f"VAD analysis — {len(df):,} utterances")
    for row in progress(df.itertuples(index=False), "Computing VAD stats",
                        total=len(df), unit="utt"):
        _, _, stats = load_and_preprocess_with_stats(row.Filepath)
        records.append({"Filename": row.Filename, "Speaker_ID": row.Speaker_ID, **stats})

    result = pd.DataFrame.from_records(records)
    n_fallback = int(result["fallback_used"].sum())
    print_status(f"{len(result) - n_fallback:,}/{len(result):,} utterances VAD-trimmed "
                f"({n_fallback} fell back to the original waveform)",
                ok=(n_fallback == 0))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        print_kv("VAD stats cached", cache_path)
    return result


def build_mfcc_transform() -> torchaudio.transforms.MFCC:
    """MFCC transform configured from src.config (13 coefficients)."""
    return torchaudio.transforms.MFCC(
        sample_rate=config.TARGET_SR,
        n_mfcc=config.N_MFCC,
        melkwargs=dict(config.MEL_KWARGS),
    )


def extract_mfcc_features(waveform: torch.Tensor,
                          mfcc_transform: torchaudio.transforms.MFCC) -> torch.Tensor:
    """MFCC + delta + delta-delta, concatenated to 39 dims per frame."""
    mfcc = mfcc_transform(waveform)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    return torch.cat([mfcc, delta, delta2], dim=1)           # (1, 39, frames)


# ---------------------------------------------------------------------------
# Memoized variants — every stage above is a pure, deterministic function of
# `filepath` alone (fixed config, no randomness), yet UASpeechDataset calls
# it from scratch on every __getitem__: every epoch of every fold of every
# ablation variant re-decodes, resamples, VAD-trims, and re-runs MFCC+delta
# on the same audio. With ~330 fold-trainings x up to 20 epochs, that is the
# single largest CPU cost in the whole training notebook and buys nothing —
# caching it changes no result, only how many times it's computed. Keyed by
# filepath (not by a mfcc_transform object) so the cache stays valid across
# Dataset instances (a new one is built per fold) and across the LRU's
# module-level lifetime. num_workers > 0 gives each DataLoader worker
# process its own independent copy of these caches (no cross-worker
# sharing) — still a large win, since persistent_workers=True keeps a
# fold's workers alive across all of that fold's epochs.
# ---------------------------------------------------------------------------
_CACHED_MFCC_TRANSFORM = None


def _shared_mfcc_transform() -> torchaudio.transforms.MFCC:
    """One MFCC transform per process, reused by every cached call — building
    a fresh one per Dataset (the uncached path) is itself needless repeated
    work, and would defeat extract_mfcc_features_cached's filepath keying if
    passed in as part of the cache key instead."""
    global _CACHED_MFCC_TRANSFORM
    if _CACHED_MFCC_TRANSFORM is None:
        _CACHED_MFCC_TRANSFORM = build_mfcc_transform()
    return _CACHED_MFCC_TRANSFORM


@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def load_and_preprocess_cached(filepath: str) -> Tuple[torch.Tensor, int]:
    """Same contract as load_and_preprocess, memoized per (process, filepath).
    Set config.PREPROCESS_CACHE_SIZE = 0 to disable (every call recomputes,
    matching the old behaviour) if a Colab session is RAM-constrained."""
    return load_and_preprocess(filepath)


@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def extract_mfcc_features_cached(filepath: str) -> torch.Tensor:
    """Same contract as extract_mfcc_features, memoized per (process, filepath)."""
    waveform, _ = load_and_preprocess_cached(filepath)
    return extract_mfcc_features(waveform, _shared_mfcc_transform())
