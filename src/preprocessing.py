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

import numpy as np
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


def load_and_preprocess_supra(filepath: str) -> Tuple[torch.Tensor, int]:
    """
    Temporal-preserving profile for the Suprasegmental branch ONLY (see
    src.models.suprasegmental_pathway). Same Silero VAD trim as
    load_and_preprocess (contiguous first-to-last speech, internal pauses
    already preserved by design — src/vad.py), but with a wider
    config.SUPRA_VAD_SPEECH_PAD_MS padding margin instead of
    config.VAD_SPEECH_PAD_MS, to protect the onset/offset dynamics a
    prosodic/temporal encoder needs from the tighter margin the
    speech-focused profile uses for the Learned/Segmental branches.

    Still pad/truncated to the same MAX_SAMPLES window as load_and_preprocess
    (so both profiles' outputs collate into fixed-size batches identically),
    with the real-audio length returned exactly as load_and_preprocess does,
    for the same masking discipline (see mfcc_valid_frame_mask and
    src.praat.extract_suprasegmental_sequence, which reads this value).
    """
    waveform, _ = _load_resampled(filepath)
    waveform, _ = vad_module.apply_vad(waveform, config.TARGET_SR,
                                       speech_pad_ms=config.SUPRA_VAD_SPEECH_PAD_MS)
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


def mfcc_frame_count(num_samples):
    """Number of frames torchaudio.transforms.MFCC (center=True, its default)
    produces for a waveform of `num_samples` samples, given
    config.MEL_KWARGS["hop_length"]. Works on a plain int or a LongTensor
    alike (only // and + are used), so it is the single source of truth for
    both a fixed total frame count (config.MAX_SAMPLES) and a per-sample
    valid-frame count derived from real audio length — see
    src.models.acoustic_pathway.AcousticPathway.valid_frame_count, which
    calls this instead of re-deriving the formula."""
    return num_samples // config.MEL_KWARGS["hop_length"] + 1


def mfcc_valid_frame_mask(total_frames: int, valid_length: int) -> torch.Tensor:
    """Frame-level boolean mask over an MFCC/delta/delta-delta tensor with
    `total_frames` frames: True = real audio-derived frame, False = frame
    generated only from the fixed-window's zero-padded tail. Derived purely
    from the post-VAD waveform's valid sample count (`valid_length`, see
    load_and_preprocess) — never from MFCC values themselves, since a
    genuinely low-energy voiced frame must not be mistaken for padding."""
    valid_frames = min(mfcc_frame_count(valid_length), total_frames)
    return torch.arange(total_frames) < valid_frames


def extract_mfcc_features(waveform: torch.Tensor,
                          mfcc_transform: torchaudio.transforms.MFCC,
                          valid_length: Optional[int] = None) -> torch.Tensor:
    """MFCC + delta + delta-delta, concatenated to 39 dims per frame.

    `waveform` is normally the fixed-window (config.MAX_SAMPLES), zero-padded
    tensor load_and_preprocess returns. Without `valid_length`, MFCC/delta/
    delta-delta are computed straight over that padded waveform — the default,
    for callers that only ever see the full window (e.g. the "before VAD, raw
    padded" comparison panel in notebooks/02_feature_analysis.ipynb).

    When `valid_length` is given, the transform instead runs only over the
    real-audio prefix `waveform[:, :valid_length]`, and the result is
    zero-padded back out to the same frame count the full waveform would have
    produced. This matters for compute_deltas: its filter looks a few frames
    ahead/behind, so computing it on the full padded waveform smears the real/
    padding boundary into the last few *valid* frames as a spurious edge
    artifact. Slicing to valid_length first keeps every delta true to only
    real audio; only the explicit pad step afterward introduces zeros.
    """
    total_frames = mfcc_frame_count(waveform.shape[-1])
    source = waveform
    if valid_length is not None and 0 < valid_length < waveform.shape[-1]:
        source = waveform[:, :valid_length]

    mfcc = mfcc_transform(source)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    features = torch.cat([mfcc, delta, delta2], dim=1)       # (1, 39, frames)

    if features.shape[-1] < total_frames:
        features = torch.nn.functional.pad(features, (0, total_frames - features.shape[-1]))
    elif features.shape[-1] > total_frames:
        features = features[..., :total_frames]
    return features


def validate_mfcc_output(features: torch.Tensor, valid_length: int,
                         mask: Optional[torch.Tensor] = None) -> None:
    """Sanity-check one extract_mfcc_features() output. Raises ValueError with
    a specific message on the first violation found; returns None (silently)
    when everything checks out. Not called on the training hot path — this is
    for notebook/debugging use, where a caught bug is worth the extra pass
    over the tensor.
    """
    expected_dim = 3 * config.N_MFCC
    if features.dim() not in (2, 3):
        raise ValueError(f"expected a (C, T) or (1, C, T) MFCC tensor, got shape {tuple(features.shape)}")
    channel_dim = -2
    if features.shape[channel_dim] != expected_dim:
        raise ValueError(
            f"MFCC channel dim is {features.shape[channel_dim]}, expected "
            f"3 * N_MFCC = {expected_dim} (mfcc + delta + delta-delta)")

    total_frames = features.shape[-1]
    if torch.isnan(features).any():
        raise ValueError("MFCC features contain NaN")
    if torch.isinf(features).any():
        raise ValueError("MFCC features contain Inf")

    valid_frames = mfcc_frame_count(valid_length)
    if valid_frames <= 0:
        raise ValueError(f"valid_length={valid_length} produced zero valid frames")
    if valid_frames > total_frames:
        raise ValueError(
            f"valid frame count ({valid_frames}) exceeds padded frame count "
            f"({total_frames}) — valid_length is longer than the fixed window")

    if mask is not None:
        if mask.shape[-1] != total_frames:
            raise ValueError(
                f"mask length ({mask.shape[-1]}) does not match feature frame "
                f"count ({total_frames})")
        expected_mask = mfcc_valid_frame_mask(total_frames, valid_length)
        if not torch.equal(mask.bool(), expected_mask):
            raise ValueError("mask does not match the frame count implied by valid_length")


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
    """Same contract as extract_mfcc_features, memoized per (process, filepath).
    Passes the VAD-trimmed waveform's valid_length through so delta/delta-delta
    are computed only over real audio, not smeared across the padded tail —
    see extract_mfcc_features's docstring."""
    waveform, valid_length = load_and_preprocess_cached(filepath)
    return extract_mfcc_features(waveform, _shared_mfcc_transform(), valid_length=valid_length)


@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def load_and_preprocess_supra_cached(filepath: str) -> Tuple[torch.Tensor, int]:
    """Same contract as load_and_preprocess_supra, memoized per (process, filepath)."""
    return load_and_preprocess_supra(filepath)


@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def extract_segmental_extra_features_cached(filepath: str) -> torch.Tensor:
    """
    Framewise formants F1-F3 + HNR (src.praat.extract_segmental_extra_sequence),
    on the SPEECH-FOCUSED profile (same audio MFCC uses), memoized per
    (process, filepath) exactly like extract_mfcc_features_cached — this is
    the expensive per-frame Praat query pass (see that function's runtime
    note), so caching it is what keeps repeated epochs/folds affordable.

    Returns a (4, total_frames) float32 tensor: f1_hz, f2_hz, f3_hz, hnr_db,
    aligned frame-for-frame with extract_mfcc_features_cached's output.
    """
    from src.praat import extract_segmental_extra_sequence

    waveform, valid_length = load_and_preprocess_cached(filepath)
    total_frames = mfcc_frame_count(waveform.shape[-1])
    sequences = extract_segmental_extra_sequence(
        waveform.squeeze(0).numpy(), config.TARGET_SR, valid_length, total_frames)
    return torch.from_numpy(
        np.stack([sequences["f1_hz"], sequences["f2_hz"],
                 sequences["f3_hz"], sequences["hnr_db"]], axis=0))


@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def extract_segmental_features_cached(filepath: str) -> torch.Tensor:
    """MFCC+delta+delta-delta (39 channels) concatenated with framewise
    formant+HNR (4 channels) along the channel axis -> (43, frames), the
    Segmental branch's full input (config.SEGMENTAL_CHANNELS — see
    src.models.segmental_pathway.SegmentalPathway)."""
    mfcc = extract_mfcc_features_cached(filepath).squeeze(0)         # (39, T)
    extra = extract_segmental_extra_features_cached(filepath)        # (4, T)
    return torch.cat([mfcc, extra], dim=0)                           # (43, T)


@lru_cache(maxsize=config.PREPROCESS_CACHE_SIZE)
def extract_suprasegmental_features_cached(filepath: str) -> torch.Tensor:
    """
    Framewise F0 (semitones) + voicing mask + intensity
    (src.praat.extract_suprasegmental_sequence), on the TEMPORAL-PRESERVING
    profile, memoized per (process, filepath).

    Returns a (3, total_frames) float32 tensor: f0_semitones, voicing,
    intensity_db, on the SAME frame grid as the segmental/MFCC branches
    (both use mfcc_frame_count(MAX_SAMPLES) frames) even though the two
    profiles' VAD spans differ — only the padding boundary (valid_length)
    differs between them, not the fixed total_frames axis length.
    """
    from src.praat import extract_suprasegmental_sequence

    waveform, valid_length = load_and_preprocess_supra_cached(filepath)
    total_frames = mfcc_frame_count(waveform.shape[-1])
    sequences = extract_suprasegmental_sequence(
        waveform.squeeze(0).numpy(), config.TARGET_SR, valid_length, total_frames)
    return torch.from_numpy(
        np.stack([sequences["f0_semitones"], sequences["voicing"],
                 sequences["intensity_db"]], axis=0))


# ---------------------------------------------------------------------------
# Fold-scoped channel-wise standardization for the three-branch model's
# framewise inputs (Segmental 43ch, Suprasegmental 3ch) — mirrors
# src.praat.praat_standardizer/praat_vector's leakage discipline exactly:
# statistics are computed from a fold's TRAIN split filepaths only, then
# applied unchanged to that fold's val/test items, so the held-out LOSO
# speaker never contributes to its own normalization.
#
# Raw per-file features stay in extract_segmental_features_cached /
# extract_suprasegmental_features_cached above (fold-agnostic, cached once
# and reused by every fold); only the mean/std statistics differ per fold,
# so normalization is applied separately, at __getitem__ time (see
# src.dataset.UASpeechDataset), not baked into the cache.
#
# Statistics are computed over VALID (non-padded) frames only — the
# fixed 401-frame window is ~85% zero-padding on the median utterance (see
# notebooks/01_data_pipeline.ipynb Stage 9), and letting that padding
# dominate the mean/std would both mis-center every real value and hide
# the real cross-channel scale difference (MFCC ~O(10), formant Hz ~O(1e3),
# HNR/intensity dB ~O(10)) that motivates normalizing at all.
# ---------------------------------------------------------------------------

def _channel_mean_std(sum_, sum_sq, count) -> Tuple[np.ndarray, np.ndarray]:
    """Shared mean/std-from-accumulators finish-up, with the same
    constant-feature and non-finite guards as src.praat.praat_standardizer."""
    if count == 0:
        n = sum_.shape[0]
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
    mean = (sum_ / count)
    var = np.clip(sum_sq / count - mean ** 2, 0.0, None)
    std = np.sqrt(var).astype(np.float32)
    mean = mean.astype(np.float32)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean, std


def segmental_standardizer(filepaths) -> Tuple[np.ndarray, np.ndarray]:
    """Channel-wise (43,) mean/std of extract_segmental_features_cached's
    output, accumulated over only the VALID (speech-focused-profile,
    pre-padding) frames of the given filepaths — pass a fold's TRAIN split
    filepaths only (see src.training.data.build_loaders), the same
    leakage-safe convention as praat_standardizer(table, filenames)."""
    n = config.SEGMENTAL_CHANNELS
    sum_ = np.zeros(n, dtype=np.float64)
    sum_sq = np.zeros(n, dtype=np.float64)
    count = 0
    for filepath in filepaths:
        features = extract_segmental_features_cached(filepath).numpy()   # (43, T)
        _, valid_length = load_and_preprocess_cached(filepath)
        valid_frames = min(mfcc_frame_count(valid_length), features.shape[-1])
        if valid_frames <= 0:
            continue
        valid = features[:, :valid_frames].astype(np.float64)
        sum_ += valid.sum(axis=1)
        sum_sq += (valid ** 2).sum(axis=1)
        count += valid_frames
    return _channel_mean_std(sum_, sum_sq, count)


def normalize_segmental(features: torch.Tensor, valid_frames: int,
                        stats: Tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
    """Apply segmental_standardizer's (mean, std) channel-wise to one
    (43, T) tensor: (x - mean) / std over every frame, then the padded tail
    (frames >= valid_frames) is forced back to exact zero — matching
    extract_mfcc_features's own valid-then-repad convention (see its
    docstring: "only the explicit pad step ... introduces zeros") rather
    than reintroducing a nonzero value into frames every downstream
    masked-pool already excludes."""
    mean, std = stats
    mean_t = torch.as_tensor(mean, dtype=features.dtype).unsqueeze(-1)
    std_t = torch.as_tensor(std, dtype=features.dtype).unsqueeze(-1)
    normalized = (features - mean_t) / std_t
    if 0 <= valid_frames < normalized.shape[-1]:
        normalized[:, valid_frames:] = 0.0
    return normalized


# Suprasegmental channel order, matching extract_suprasegmental_sequence's
# returned dict order (f0_semitones, voicing, intensity_db) and the (3, T)
# stacking above — kept as named indices rather than magic numbers so the
# "skip the voicing channel" logic below is self-explanatory.
SUPRA_F0_CHANNEL, SUPRA_VOICING_CHANNEL, SUPRA_INTENSITY_CHANNEL = 0, 1, 2
SUPRA_CONTINUOUS_CHANNELS = (SUPRA_F0_CHANNEL, SUPRA_INTENSITY_CHANNEL)


def suprasegmental_standardizer(filepaths) -> Tuple[np.ndarray, np.ndarray]:
    """Channel-wise (2,) mean/std for the CONTINUOUS suprasegmental channels
    only — f0_semitones and intensity_db — accumulated over VALID
    (temporal-preserving-profile) frames of the given filepaths. The binary
    voicing channel is deliberately excluded: it is already a well-scaled
    {0, 1} indicator, and standardizing it would destroy the "1 = real pitch
    estimate, 0 = unvoiced/invalid" semantics src.praat.praat.py's module
    docstring and src/vad.py's masking discipline both depend on."""
    n = len(SUPRA_CONTINUOUS_CHANNELS)
    sum_ = np.zeros(n, dtype=np.float64)
    sum_sq = np.zeros(n, dtype=np.float64)
    count = 0
    for filepath in filepaths:
        features = extract_suprasegmental_features_cached(filepath).numpy()   # (3, T)
        _, valid_length = load_and_preprocess_supra_cached(filepath)
        valid_frames = min(mfcc_frame_count(valid_length), features.shape[-1])
        if valid_frames <= 0:
            continue
        valid = features[np.ix_(SUPRA_CONTINUOUS_CHANNELS, range(valid_frames))].astype(np.float64)
        sum_ += valid.sum(axis=1)
        sum_sq += (valid ** 2).sum(axis=1)
        count += valid_frames
    return _channel_mean_std(sum_, sum_sq, count)


def normalize_suprasegmental(features: torch.Tensor, valid_frames: int,
                             stats: Tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
    """Apply suprasegmental_standardizer's (mean, std) to ONLY the
    f0_semitones and intensity_db channels of one (3, T) tensor; the
    voicing channel (index SUPRA_VOICING_CHANNEL) passes through
    untouched, preserving its {0, 1} meaning. The padded tail of the
    normalized channels is forced back to exact zero, same convention as
    normalize_segmental (voicing's own padded tail is already 0 by
    construction — see extract_suprasegmental_sequence — so it needs no
    extra handling here)."""
    mean, std = stats
    normalized = features.clone()
    continuous = features[list(SUPRA_CONTINUOUS_CHANNELS), :]
    mean_t = torch.as_tensor(mean, dtype=features.dtype).unsqueeze(-1)
    std_t = torch.as_tensor(std, dtype=features.dtype).unsqueeze(-1)
    normalized_continuous = (continuous - mean_t) / std_t
    if 0 <= valid_frames < normalized_continuous.shape[-1]:
        normalized_continuous[:, valid_frames:] = 0.0
    for i, channel in enumerate(SUPRA_CONTINUOUS_CHANNELS):
        normalized[channel] = normalized_continuous[i]
    return normalized
