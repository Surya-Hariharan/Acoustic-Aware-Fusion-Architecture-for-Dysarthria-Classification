"""
Audio preprocessing utilities.

Every utterance goes through the same deterministic chain:
  1. Load and mix down to mono.
  2. Resample to 16 kHz.
  3. Trim silence with voice activity detection.
  4. Pad or truncate to a fixed 4-second window.

MFCC extraction produces 13 coefficients plus delta and delta-delta,
giving the 39-dimensional per-frame representation used by the
Acoustic Pathway (matching the base paper's baseline features).
"""

from functools import lru_cache

import torch
import torchaudio

from src import config


def load_and_preprocess(filepath: str) -> torch.Tensor:
    """Load one utterance and return a (1, MAX_SAMPLES) 16 kHz mono waveform."""
    waveform, sr = torchaudio.load(filepath)

    if waveform.shape[0] > 1:                                # stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != config.TARGET_SR:                               # resample
        waveform = torchaudio.functional.resample(waveform, sr, config.TARGET_SR)

    waveform = torchaudio.functional.vad(                    # trim silence
        waveform, sample_rate=config.TARGET_SR)

    if waveform.shape[1] < config.MAX_SAMPLES:               # pad / truncate
        pad = config.MAX_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :config.MAX_SAMPLES]

    return waveform


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
def load_and_preprocess_cached(filepath: str) -> torch.Tensor:
    """Same contract as load_and_preprocess, memoized per (process, filepath).
    Set config.PREPROCESS_CACHE_SIZE = 0 to disable (every call recomputes,
    matching the old behaviour) if a Colab session is RAM-constrained."""
    return load_and_preprocess(filepath)


@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def extract_mfcc_features_cached(filepath: str) -> torch.Tensor:
    """Same contract as extract_mfcc_features, memoized per (process, filepath)."""
    waveform = load_and_preprocess_cached(filepath)
    return extract_mfcc_features(waveform, _shared_mfcc_transform())
