"""
Voice activity detection: Silero VAD, replacing torchaudio.functional.vad.

torchaudio.functional.vad() is a forward-only energy-ramp detector — it trims
LEADING silence only, never trailing silence, and has no concept of "internal
pause vs. trailing silence". Combined with load_and_preprocess()'s fixed 4s
pad/truncate window (src/preprocessing.py), that left most of every clip a
near-constant zero-padded tail after the (short, single-word) UA-Speech
utterance — the artifact visible in MFCC plots.

Silero VAD is a small neural voice-activity model (github.com/snakers4/silero-vad,
loaded once per process via torch.hub, run in eval() with no dropout — inference
is deterministic given fixed weights). apply_vad() below:
  - finds the speech span [first_segment.start - pad, last_segment.end + pad]
    and slices to it, so LEADING and TRAILING non-speech are both removed but
    any internal pause between detected segments is preserved (the slice is
    contiguous, not a concatenation of only the speech segments);
  - never raises and never returns an empty waveform: any failure (model load
    error, zero detected segments, a trimmed span shorter than
    config.VAD_MIN_SPEECH_MS) falls back to the ORIGINAL waveform unchanged,
    with fallback_used=True recorded in the returned stats.

Reproducibility note: torch.hub.load("snakers4/silero-vad", ...) resolves the
repo's default branch unless pinned to a tag. VAD_REPO is pinned to the
"v6.2.1" release tag (github.com/snakers4/silero-vad/releases/tag/v6.2.1) so
that repeat runs across machines/dates resolve the same model code rather
than silently tracking upstream's moving default branch.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import torch

from src import config
from src.console import print_status

VAD_REPO = "snakers4/silero-vad:v6.2.1"

_MODEL = None
_UTILS = None


def load_silero_vad():
    """Load (and cache at module level) the Silero VAD JIT model + its utils tuple.

    Loaded once per process, kept in eval() (no dropout) — every call to
    get_speech_timestamps() with the same input is therefore deterministic.
    Raises on failure rather than swallowing the error here; callers (apply_vad)
    are responsible for the fallback-to-original-waveform contract.
    """
    global _MODEL, _UTILS
    if _MODEL is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, utils = torch.hub.load(
                repo_or_dir=VAD_REPO, model="silero_vad",
                force_reload=False, trust_repo=True, verbose=False)
        model.eval()
        _MODEL, _UTILS = model, utils
    return _MODEL, _UTILS


def get_speech_timestamps(waveform_1d: torch.Tensor, sr: int = config.VAD_SAMPLE_RATE,
                          speech_pad_ms: Optional[int] = None
                          ) -> List[Dict[str, int]]:
    """Speech segments (sample indices) for a single-channel 1-D waveform.

    speech_pad_ms overrides config.VAD_SPEECH_PAD_MS for this call only —
    used by src.preprocessing.load_and_preprocess_supra to apply the wider
    temporal-preserving margin (config.SUPRA_VAD_SPEECH_PAD_MS) without
    mutating shared config state (relevant once DataLoader worker processes
    call this concurrently)."""
    model, utils = load_silero_vad()
    get_ts = utils[0]
    with torch.no_grad():
        return get_ts(
            waveform_1d, model, sampling_rate=sr,
            threshold=config.VAD_THRESHOLD,
            min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
            speech_pad_ms=speech_pad_ms if speech_pad_ms is not None else config.VAD_SPEECH_PAD_MS,
        )


def _fallback_stats(waveform: torch.Tensor, sr: int, reason: str) -> Dict:
    duration_s = waveform.shape[-1] / sr
    return {
        "original_duration_s": duration_s,
        "speech_duration_s": duration_s,
        "speech_ratio": 1.0,
        "num_segments": 0,
        "fallback_used": True,
        "fallback_reason": reason,
        "leading_trimmed_s": 0.0,
        "trailing_trimmed_s": 0.0,
    }


def apply_vad(waveform: torch.Tensor, sr: int = config.VAD_SAMPLE_RATE,
             speech_pad_ms: Optional[int] = None
             ) -> Tuple[torch.Tensor, Dict]:
    """
    Trim leading/trailing non-speech from a (1, samples) mono waveform,
    preserving internal pauses. Never discards the utterance: any failure
    mode falls back to returning `waveform` unchanged.

    speech_pad_ms: see get_speech_timestamps — None (default) uses
    config.VAD_SPEECH_PAD_MS (the speech-focused profile's margin); pass
    config.SUPRA_VAD_SPEECH_PAD_MS for the temporal-preserving profile.

    Returns (processed_waveform, stats) — stats always has original_duration_s,
    speech_duration_s, speech_ratio, num_segments, fallback_used (+ reason),
    leading_trimmed_s, trailing_trimmed_s.
    """
    original_duration_s = waveform.shape[-1] / sr
    if not config.VAD_ENABLED:
        return waveform, _fallback_stats(waveform, sr, "vad_disabled")

    try:
        waveform_1d = waveform.squeeze(0) if waveform.dim() == 2 else waveform
        segments = get_speech_timestamps(waveform_1d, sr, speech_pad_ms=speech_pad_ms)
    except Exception as exc:
        print_status(f"Silero VAD failed ({exc}) — falling back to original waveform",
                     ok=False)
        return waveform, _fallback_stats(waveform, sr, f"exception: {exc}")

    if not segments:
        return waveform, _fallback_stats(waveform, sr, "no_speech_detected")

    start = int(segments[0]["start"])
    end = int(segments[-1]["end"])
    min_speech_samples = int(config.VAD_MIN_SPEECH_MS / 1000 * sr)
    if end - start < min_speech_samples:
        return waveform, _fallback_stats(waveform, sr, "trimmed_span_too_short")

    trimmed = waveform[:, start:end] if waveform.dim() == 2 else waveform_1d[start:end].unsqueeze(0)
    speech_duration_s = sum(
        (int(seg["end"]) - int(seg["start"])) for seg in segments) / sr

    stats = {
        "original_duration_s": original_duration_s,
        "speech_duration_s": speech_duration_s,
        "speech_ratio": speech_duration_s / original_duration_s if original_duration_s > 0 else 0.0,
        "num_segments": len(segments),
        "fallback_used": False,
        "fallback_reason": "",
        "leading_trimmed_s": start / sr,
        "trailing_trimmed_s": (waveform.shape[-1] - end) / sr,
    }
    return trimmed, stats
