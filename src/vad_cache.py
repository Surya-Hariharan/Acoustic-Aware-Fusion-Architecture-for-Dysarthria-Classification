"""
Silero VAD spans read from the feature store instead of re-running the model.

src.vad.apply_vad's only effect on the signal is waveform[:, start:end], and
every one of its fallback branches returns the untrimmed waveform, stored as
the span (0, N). Two integers per (utterance, profile) therefore describe the
trim exactly, and slicing from them is bit-identical to running Silero —
without the two neural forward passes per item per epoch that otherwise
dominate data loading.

The spans live in each feature-store chunk (src.feature_store.span_table).
Every lookup returns None on a miss and callers fall back to live Silero, so
the store is an optimization, never a correctness precondition.
verify_vad_span_cache re-runs live Silero on a sample and requires exact
equality.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src import config
from src import vad as vad_module
from src.console import print_note, print_status, progress

# Part of the feature-store signature: bump only if the span layout changes.
SPAN_SCHEMA_VERSION = 1


def span_cache_signature() -> Dict[str, object]:
    """Every input the stored spans depend on (embedded in the feature-store
    signature, so a chunk built under different VAD settings is ignored)."""
    return {
        "schema": SPAN_SCHEMA_VERSION,
        "vad_repo": vad_module.VAD_REPO,
        "target_sr": config.TARGET_SR,
        "vad_enabled": bool(config.VAD_ENABLED),
        "vad_threshold": config.VAD_THRESHOLD,
        "vad_min_speech_ms": config.VAD_MIN_SPEECH_MS,
        "vad_min_silence_ms": config.VAD_MIN_SILENCE_MS,
        "vad_speech_pad_ms": config.VAD_SPEECH_PAD_MS,
        "supra_vad_speech_pad_ms": config.SUPRA_VAD_SPEECH_PAD_MS,
    }


def _live_span(waveform, speech_pad_ms: Optional[int]) -> Tuple[int, int, bool]:
    """(start, end, fallback_used) for one profile by running Silero, derived
    from apply_vad's own outputs so the two can never drift apart."""
    n = int(waveform.shape[-1])
    trimmed, stats = vad_module.apply_vad(waveform, config.TARGET_SR,
                                          speech_pad_ms=speech_pad_ms)
    if stats["fallback_used"]:
        return 0, n, True
    start = int(round(stats["leading_trimmed_s"] * config.TARGET_SR))
    return start, start + int(trimmed.shape[-1]), False


@lru_cache(maxsize=1)
def _span_table() -> Dict[str, Tuple[int, int, int, int, int]]:
    """Filename -> (num_samples, speech_start, speech_end, supra_start,
    supra_end), loaded lazily once per process (main and each worker)."""
    from src import feature_store
    return dict(feature_store.span_table())


def clear_span_table_cache() -> None:
    """Drop the per-process memoized table (tests, and after a rebuild)."""
    from src import feature_store
    _span_table.cache_clear()
    feature_store.span_table.cache_clear()


def vad_span(filepath: str, *, supra: bool = False) -> Optional[Tuple[int, int]]:
    """(start, end) into the mono 16 kHz waveform for one profile, or None on a
    miss. supra=False: speech-focused (learned + segmental branches);
    supra=True: temporal-preserving (suprasegmental branch)."""
    entry = _span_table().get(Path(filepath).name)
    if entry is None:
        return None
    _, speech_start, speech_end, supra_start, supra_end = entry
    return (supra_start, supra_end) if supra else (speech_start, speech_end)


def vad_valid_length(filepath: str, *, supra: bool = False) -> Optional[int]:
    """min(end - start, MAX_SAMPLES) — exactly the valid_length the loaders in
    src.preprocessing return, with no file read and no VAD."""
    span = vad_span(filepath, supra=supra)
    if span is None:
        return None
    start, end = span
    return min(end - start, config.MAX_SAMPLES)


def verify_vad_span_cache(df: pd.DataFrame, n: int = 200, seed: int = 0) -> None:
    """Re-run live Silero on `n` random stored utterances and raise unless every
    span matches exactly. A wrong span is worse than an absent one: the stored
    features were cut with it."""
    from src.preprocessing import _load_resampled

    table = _span_table()
    if not table:
        raise RuntimeError(f"No VAD spans found — build the feature store under "
                           f"{config.FEATURE_STORE_DIR} first.")
    cached = df[df["Filename"].isin(table)]
    if len(cached) < len(df):
        print_note(f"{len(df) - len(cached):,} of {len(df):,} utterances have no stored span "
                   f"(they fall back to live VAD).")
    if cached.empty:
        return

    if config.VAD_ENABLED:
        vad_module.warmup_silero_vad()
    rows = cached.sample(n=min(n, len(cached)), random_state=seed)
    mismatches = []
    for filepath in progress(rows["Filepath"].tolist(), f"Verifying {len(rows):,} VAD spans",
                             total=len(rows), unit="file"):
        num_samples, speech_start, speech_end, supra_start, supra_end = table[Path(filepath).name]
        waveform, _ = _load_resampled(filepath)
        name = Path(filepath).name
        if int(waveform.shape[-1]) != num_samples:
            mismatches.append(f"{name}: num_samples {num_samples} vs live {waveform.shape[-1]}")
        live_speech = _live_span(waveform, None)[:2]
        live_supra = _live_span(waveform, config.SUPRA_VAD_SPEECH_PAD_MS)[:2]
        if live_speech != (speech_start, speech_end):
            mismatches.append(f"{name}: speech {(speech_start, speech_end)} vs live {live_speech}")
        if live_supra != (supra_start, supra_end):
            mismatches.append(f"{name}: supra {(supra_start, supra_end)} vs live {live_supra}")

    if mismatches:
        raise RuntimeError(f"Stored VAD spans disagree with live Silero on {len(mismatches)} "
                           f"check(s):\n  " + "\n  ".join(mismatches[:10]) +
                           "\nDelete the affected feature-store chunks and rebuild them.")
    print_status(f"All {len(rows):,} sampled spans match live Silero exactly", ok=True)
