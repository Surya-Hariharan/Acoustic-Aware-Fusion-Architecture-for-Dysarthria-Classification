"""
Audio preprocessing and the per-utterance model inputs.

Every utterance goes through one deterministic chain:
  1. load, mix to mono, resample to 16 kHz;
  2. trim leading/trailing non-speech with Silero VAD (internal pauses kept);
  3. pad or truncate to the fixed 4 s window, returning valid_length so the
     padding is masked downstream, never treated as silence.

Two VAD profiles differ only in the margin kept around speech: speech-focused
(VAD_SPEECH_PAD_MS, learned + segmental branches) and temporal-preserving
(SUPRA_VAD_SPEECH_PAD_MS, suprasegmental branch).

The *_cached getters read from the feature store (src.feature_store) and fall
back to computing live on a miss. Fold-scoped standardization is applied at
__getitem__ time from statistics of the fold's training split only.
"""

from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
import torchaudio

from src import config
from src import feature_store
from src import vad as vad_module
from src import vad_cache


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------
def _load_resampled(filepath: str) -> Tuple[torch.Tensor, int]:
    """Load, mix down to mono, resample to TARGET_SR."""
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != config.TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, config.TARGET_SR)
    return waveform, config.TARGET_SR


def _trim(waveform: torch.Tensor, filepath: str, *, supra: bool) -> torch.Tensor:
    """VAD-trim to the stored span for one profile (bit-identical to running
    Silero — see src.vad_cache), or run Silero live on a store miss."""
    span = vad_cache.vad_span(filepath, supra=supra)
    if span is None:
        speech_pad_ms = config.SUPRA_VAD_SPEECH_PAD_MS if supra else None
        trimmed, _ = vad_module.apply_vad(waveform, config.TARGET_SR, speech_pad_ms=speech_pad_ms)
        return trimmed
    num_samples = waveform.shape[-1]
    start = max(0, min(int(span[0]), num_samples))
    end = max(start, min(int(span[1]), num_samples))
    return waveform[:, start:end]


def _pad_or_truncate(waveform: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """(1, MAX_SAMPLES) waveform and the count of real (pre-padding) samples."""
    valid_length = min(waveform.shape[1], config.MAX_SAMPLES)
    if waveform.shape[1] < config.MAX_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, config.MAX_SAMPLES - waveform.shape[1]))
    else:
        waveform = waveform[:, :config.MAX_SAMPLES]
    return waveform, valid_length


def load_and_preprocess(filepath: str) -> Tuple[torch.Tensor, int]:
    """Speech-focused profile: (1, MAX_SAMPLES) waveform and valid_length."""
    waveform, _ = _load_resampled(filepath)
    return _pad_or_truncate(_trim(waveform, filepath, supra=False))


def load_and_preprocess_supra(filepath: str) -> Tuple[torch.Tensor, int]:
    """Temporal-preserving profile: (1, MAX_SAMPLES) waveform and valid_length."""
    waveform, _ = _load_resampled(filepath)
    return _pad_or_truncate(_trim(waveform, filepath, supra=True))


# ---------------------------------------------------------------------------
# MFCC
# ---------------------------------------------------------------------------
def build_mfcc_transform() -> torchaudio.transforms.MFCC:
    return torchaudio.transforms.MFCC(sample_rate=config.TARGET_SR, n_mfcc=config.N_MFCC,
                                      melkwargs=dict(config.MEL_KWARGS))


_MFCC_TRANSFORM = None


def _shared_mfcc_transform() -> torchaudio.transforms.MFCC:
    global _MFCC_TRANSFORM
    if _MFCC_TRANSFORM is None:
        _MFCC_TRANSFORM = build_mfcc_transform()
    return _MFCC_TRANSFORM


def mfcc_frame_count(num_samples):
    """Frames torchaudio's MFCC (center=True) yields for `num_samples` samples.
    Works on ints and LongTensors alike — the single source of truth for both
    the fixed 401-frame grid and every per-utterance valid-frame count."""
    return num_samples // config.MEL_KWARGS["hop_length"] + 1


def extract_mfcc_features(waveform: torch.Tensor, mfcc_transform: torchaudio.transforms.MFCC,
                          valid_length: int = None) -> torch.Tensor:
    """(1, 39, frames) MFCC + delta + delta-delta.

    With valid_length, the transform runs over the real-audio prefix only and
    the result is zero-padded to the full-window frame count, so the delta
    filters never smear the speech/padding boundary into valid frames."""
    total_frames = mfcc_frame_count(waveform.shape[-1])
    source = waveform
    if valid_length is not None and 0 < valid_length < waveform.shape[-1]:
        source = waveform[:, :valid_length]
    mfcc = mfcc_transform(source)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    features = torch.cat([mfcc, delta, delta2], dim=1)
    if features.shape[-1] < total_frames:
        features = torch.nn.functional.pad(features, (0, total_frames - features.shape[-1]))
    return features[..., :total_frames]


# ---------------------------------------------------------------------------
# Cached per-utterance inputs (feature store first, live computation on a miss)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def load_and_preprocess_cached(filepath: str) -> Tuple[torch.Tensor, int]:
    return load_and_preprocess(filepath)


@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def extract_segmental_features_cached(filepath: str) -> torch.Tensor:
    """(43, frames): MFCC+delta+delta-delta (39) then F1, F2, F3, HNR (4), on
    the speech-focused profile."""
    stored = feature_store.segmental_features(filepath)
    if stored is not None:
        return torch.from_numpy(stored.copy())
    from src.praat import extract_segmental_extra_sequence

    waveform, valid_length = load_and_preprocess_cached(filepath)
    mfcc = extract_mfcc_features(waveform, _shared_mfcc_transform(), valid_length).squeeze(0)
    extra = extract_segmental_extra_sequence(waveform.squeeze(0).numpy(), config.TARGET_SR,
                                             valid_length, mfcc.shape[-1])
    extra = np.stack([extra["f1_hz"], extra["f2_hz"], extra["f3_hz"], extra["hnr_db"]])
    return torch.cat([mfcc, torch.from_numpy(extra.astype(np.float32))], dim=0)


@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def extract_suprasegmental_features_cached(filepath: str) -> torch.Tensor:
    """(3, frames): F0 semitones, voicing, intensity dB, on the
    temporal-preserving profile and the same frame grid as the segmental input."""
    stored = feature_store.suprasegmental_features(filepath)
    if stored is not None:
        return torch.from_numpy(stored.copy())
    from src.praat import extract_suprasegmental_sequence

    waveform, valid_length = load_and_preprocess_supra(filepath)
    sequences = extract_suprasegmental_sequence(waveform.squeeze(0).numpy(), config.TARGET_SR,
                                                valid_length, mfcc_frame_count(waveform.shape[-1]))
    return torch.from_numpy(np.stack([sequences["f0_semitones"], sequences["voicing"],
                                      sequences["intensity_db"]]).astype(np.float32))


def valid_length(filepath: str, *, supra: bool = False) -> int:
    """Real-audio sample count for one profile, from the stored span when
    available (no file read), else by loading the audio."""
    length = vad_cache.vad_valid_length(filepath, supra=supra)
    if length is not None:
        return length
    return (load_and_preprocess_supra(filepath) if supra
            else load_and_preprocess_cached(filepath))[1]


# ---------------------------------------------------------------------------
# Fold-scoped channel standardization (statistics from TRAINING files only)
# ---------------------------------------------------------------------------
def _channel_mean_std(sum_, sum_sq, count) -> Tuple[np.ndarray, np.ndarray]:
    """mean/std from accumulators; a channel with no data or zero variance
    falls back to (0, 1). `count` is a scalar or a per-channel array."""
    count = np.asarray(count, dtype=np.float64)
    n = sum_.shape[0]
    no_data = count == 0
    safe_count = np.where(no_data, 1.0, count)
    mean = sum_ / safe_count
    std = np.sqrt(np.clip(sum_sq / safe_count - mean ** 2, 0.0, None)).astype(np.float32)
    mean = np.broadcast_to(mean.astype(np.float32), (n,)).copy()
    std = np.broadcast_to(std, (n,)).copy()
    std[np.broadcast_to(no_data | ~np.isfinite(std) | (std == 0), (n,))] = 1.0
    mean[np.broadcast_to(no_data | ~np.isfinite(mean), (n,))] = 0.0
    return mean, std


def segmental_standardizer(filepaths) -> Tuple[np.ndarray, np.ndarray]:
    """(43,) channel mean/std over the VALID frames of `filepaths` — pass a
    fold's training files only, so the held-out speaker never contributes."""
    n = config.SEGMENTAL_CHANNELS
    sum_, sum_sq, count = np.zeros(n), np.zeros(n), 0
    for filepath in filepaths:
        features = extract_segmental_features_cached(filepath).numpy()
        frames = min(mfcc_frame_count(valid_length(filepath)), features.shape[-1])
        if frames <= 0:
            continue
        valid = features[:, :frames].astype(np.float64)
        sum_ += valid.sum(axis=1)
        sum_sq += (valid ** 2).sum(axis=1)
        count += frames
    return _channel_mean_std(sum_, sum_sq, count)


def normalize_segmental(features: torch.Tensor, valid_frames: int,
                        stats: Tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
    """z-score each channel, then force the padded tail back to exact zero."""
    mean, std = stats
    normalized = ((features - torch.as_tensor(mean, dtype=features.dtype).unsqueeze(-1))
                  / torch.as_tensor(std, dtype=features.dtype).unsqueeze(-1))
    if 0 <= valid_frames < normalized.shape[-1]:
        normalized[:, valid_frames:] = 0.0
    return normalized


SUPRA_F0_CHANNEL, SUPRA_VOICING_CHANNEL, SUPRA_INTENSITY_CHANNEL = 0, 1, 2
SUPRA_CONTINUOUS_CHANNELS = (SUPRA_F0_CHANNEL, SUPRA_INTENSITY_CHANNEL)


def suprasegmental_standardizer(filepaths) -> Tuple[np.ndarray, np.ndarray]:
    """(2,) mean/std for F0 and intensity over valid frames of `filepaths`
    (training files only). F0 statistics use VOICED frames only — its 0 on
    unvoiced frames is a sentinel, not a pitch. The binary voicing channel is
    never standardized."""
    n = len(SUPRA_CONTINUOUS_CHANNELS)
    sum_, sum_sq, count = np.zeros(n), np.zeros(n), np.zeros(n)
    for filepath in filepaths:
        features = extract_suprasegmental_features_cached(filepath).numpy()
        frames = min(mfcc_frame_count(valid_length(filepath, supra=True)), features.shape[-1])
        if frames <= 0:
            continue
        voiced = features[SUPRA_VOICING_CHANNEL, :frames].astype(bool)
        for i, channel in enumerate(SUPRA_CONTINUOUS_CHANNELS):
            values = features[channel, :frames].astype(np.float64)
            if channel == SUPRA_F0_CHANNEL:
                values = values[voiced]
            sum_[i] += values.sum()
            sum_sq[i] += (values ** 2).sum()
            count[i] += values.size
    return _channel_mean_std(sum_, sum_sq, count)


def normalize_suprasegmental(features: torch.Tensor, valid_frames: int,
                             stats: Tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
    """z-score F0 and intensity (voicing untouched), zero the padded tail, and
    keep F0 at exact zero on every unvoiced frame so normalization never
    fabricates a pseudo-pitch there."""
    mean, std = stats
    normalized = features.clone()
    continuous = features[list(SUPRA_CONTINUOUS_CHANNELS), :]
    continuous = ((continuous - torch.as_tensor(mean, dtype=features.dtype).unsqueeze(-1))
                  / torch.as_tensor(std, dtype=features.dtype).unsqueeze(-1))
    if 0 <= valid_frames < continuous.shape[-1]:
        continuous[:, valid_frames:] = 0.0
    for i, channel in enumerate(SUPRA_CONTINUOUS_CHANNELS):
        normalized[channel] = continuous[i]
    normalized[SUPRA_F0_CHANNEL][features[SUPRA_VOICING_CHANNEL] == 0] = 0.0
    return normalized
