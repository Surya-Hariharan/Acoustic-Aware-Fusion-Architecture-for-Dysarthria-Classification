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

Concurrent-download note: torch.hub.load() downloads+extracts a repo zipball
into a directory named "<owner>_<repo>_<ref>" under torch.hub.get_dir(), but
GitHub's zipball extracts to "<owner>-<repo>-<short_commit_sha>" internally,
so torch.hub renames the extracted folder after the fact. That
download-extract-rename sequence is NOT process-safe: if several processes
call torch.hub.load() for the same repo at the same moment (e.g. every
ProcessPoolExecutor worker in precompute_framewise_feature_cache lazily
loading VAD on its first file), they race on the same target directory,
"Directory not empty" is raised mid-rename, and the cache is left with the
commit-hash-named directory but no "<owner>_<repo>_<ref>/hubconf.py" — every
later load in every worker then fails identically, forever, once per file.
warmup_silero_vad() (called once, synchronously, in the main process before
workers are spawned) exists specifically to populate the on-disk cache
BEFORE any concurrency starts, so worker processes' own load_silero_vad()
calls are local reads, not downloads.
"""

import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from src import config
from src.console import print_status

VAD_REPO = "snakers4/silero-vad:v6.2.1"

_MODEL = None
_UTILS = None
# Set once a load fails in this process; short-circuits every later call so a
# broken/unreachable torch.hub cache is retried once, not once per file (see
# _ensure_loaded_or_disabled).
_INIT_ERROR: Optional[str] = None


def load_silero_vad(force_reload: bool = False):
    """Load (and cache at module level) the Silero VAD JIT model + its utils tuple.

    Loaded once per process, kept in eval() (no dropout) — every call to
    get_speech_timestamps() with the same input is therefore deterministic.
    Raises on failure rather than swallowing the error here; callers (apply_vad)
    are responsible for the fallback-to-original-waveform contract.
    """
    global _MODEL, _UTILS
    if _MODEL is None or force_reload:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, utils = torch.hub.load(
                repo_or_dir=VAD_REPO, model="silero_vad",
                force_reload=force_reload, trust_repo=True, verbose=False)
        model.eval()
        _MODEL, _UTILS = model, utils
    return _MODEL, _UTILS


def _clear_stale_silero_cache() -> None:
    """Remove any torch.hub cache entries for this repo only (matched by
    "silero-vad"/"silero_vad" in the entry name, inside torch.hub.get_dir()
    alone) — never touches the rest of ~/.cache/torch. Used to recover from
    the partial-extraction state left by the concurrent-download race
    described in the module docstring, without blindly wiping the whole
    torch hub cache."""
    hub_dir = Path(torch.hub.get_dir())
    if not hub_dir.exists():
        return
    for entry in hub_dir.iterdir():
        if "silero-vad" in entry.name or "silero_vad" in entry.name:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)


def warmup_silero_vad() -> None:
    """Load Silero VAD once, synchronously, and verify it actually runs on a
    dummy waveform — call this ONCE in the main process, before handing work
    to ProcessPoolExecutor workers (see precompute_framewise_feature_cache).

    Populating the on-disk torch.hub cache here, before any worker process
    exists, is what prevents the multi-process download race described in
    this module's docstring. On first failure, retries exactly once after
    clearing any stale/partial cache entries for this repo; if that also
    fails, raises RuntimeError with a diagnostic instead of leaving 21,000+
    per-file calls to each silently retry the same broken load.
    """
    global _INIT_ERROR
    _INIT_ERROR = None
    try:
        load_silero_vad()
    except Exception as first_exc:
        print_status(
            f"Silero VAD load failed ({first_exc}) — clearing torch.hub cache "
            f"entries for this repo and retrying once", ok=False)
        _clear_stale_silero_cache()
        try:
            load_silero_vad(force_reload=True)
        except Exception as exc:
            _INIT_ERROR = str(exc)
            raise RuntimeError(
                f"Silero VAD failed to initialize ({VAD_REPO}) after a "
                f"cache-clear-and-retry: {exc}\n"
                f"torch.hub cache dir: {Path(torch.hub.get_dir())}\n"
                f"Check network access to github.com and huggingface.co from "
                f"this environment, or set config.VAD_ENABLED = False to "
                f"deliberately run without VAD."
            ) from exc

    model, utils = _MODEL, _UTILS
    get_ts = utils[0]
    dummy = torch.zeros(config.VAD_SAMPLE_RATE)          # 1s of silence, smoke test only
    with torch.no_grad():
        get_ts(dummy, model, sampling_rate=config.VAD_SAMPLE_RATE)
    print_status(f"Silero VAD initialized and verified ({VAD_REPO})", ok=True)


def _ensure_loaded_or_disabled() -> bool:
    """True if the model is ready to use in THIS process. On first failure,
    records the reason once (_INIT_ERROR) and returns False for every later
    call in this process, so apply_vad falls back to the raw waveform without
    retrying a broken torch.hub load or re-printing a failure message per
    file — see the module docstring's concurrent-download race and item 4 of
    the fix (fail/log once, then explicit fallback, not 21,000 messages)."""
    global _INIT_ERROR
    if _MODEL is not None:
        return True
    if _INIT_ERROR is not None:
        return False
    try:
        load_silero_vad()
        return True
    except Exception as exc:
        _INIT_ERROR = str(exc)
        print_status(
            f"Silero VAD failed to initialize in this process ({exc}) — "
            f"every file processed by this process will fall back to the "
            f"original (non-VAD-trimmed) waveform", ok=False)
        return False


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

    if not _ensure_loaded_or_disabled():
        return waveform, _fallback_stats(waveform, sr, f"vad_init_failed: {_INIT_ERROR}")

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
