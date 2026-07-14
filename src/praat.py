"""
Phase 4 - Praat acoustic analysis.

Clinically meaningful voice-quality measures extracted directly from the
ORIGINAL audio (not the pipeline's VAD-trimmed, resampled, zero-padded
4-second window in src/preprocessing.py used for MFCC/wav2vec) via
parselmouth, the Python binding for Praat. These complement the MFCC and
wav2vec pathways with features a speech-pathology reader recognizes by
name, and let model behaviour be related back to acoustic characteristics
rather than only classification accuracy (Phase 5).

Every UA-Speech utterance is a single isolated word, so "speech rate" and
"pause duration" below are proxies appropriate to that scale (voiced-pulse
density, locally-unvoiced-frame duration) - not multi-word timing.

Call signatures (jitter/shimmer, formants, intensity) follow the standard
Praat "Voice report" scripting recipe; verified against Praat's own report
text on a sample UA-Speech file before writing this module.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call

from src.console import print_kv

PITCH_FLOOR = 75.0      # Hz - standard floor for adult speech
PITCH_CEILING = 600.0   # Hz - generous enough for female/pathological voices

JITTER_SHIMMER_KEYS = ("jitter_local", "jitter_rap", "jitter_ppq5",
                       "shimmer_local", "shimmer_apq3", "shimmer_apq11")
RHYTHM_KEYS = ("speech_rate", "pause_duration", "voice_breaks")

FEATURE_COLUMNS = (
    "f0_mean", "f0_max", "f0_min",
    *JITTER_SHIMMER_KEYS,
    "hnr_mean",
    "f1_mean", "f2_mean", "f3_mean",
    "intensity_mean", "intensity_max",
    *RHYTHM_KEYS,
)


def _safe_point_process(sound: parselmouth.Sound):
    """A PointProcess needs enough regular glottal pulses to define a period -
    short or heavily dysarthric clips can fail this. None on failure."""
    try:
        return call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
    except Exception:
        return None


def extract_pitch_features(sound: parselmouth.Sound,
                           pitch: Optional[parselmouth.Pitch] = None) -> Dict[str, float]:
    """F0 mean/max/min (Hz), voiced frames only."""
    if pitch is None:
        pitch = sound.to_pitch(time_step=None, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    voiced = pitch.selected_array["frequency"]
    voiced = voiced[voiced != 0]
    if len(voiced) == 0:
        return {"f0_mean": np.nan, "f0_max": np.nan, "f0_min": np.nan}
    return {"f0_mean": float(voiced.mean()), "f0_max": float(voiced.max()), "f0_min": float(voiced.min())}


def extract_jitter_shimmer_features(sound: parselmouth.Sound, point_process=None) -> Dict[str, float]:
    """Jitter (local, RAP, PPQ5) and shimmer (local, APQ3, APQ11): period-to-
    period and amplitude-to-amplitude perturbation, standard dysarthria
    biomarkers for vocal fold instability."""
    if point_process is None:
        point_process = _safe_point_process(sound)
    if point_process is None:
        return {k: np.nan for k in JITTER_SHIMMER_KEYS}
    try:
        return {
            "jitter_local": call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            "jitter_rap": call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            "jitter_ppq5": call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            "shimmer_local": call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "shimmer_apq3": call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "shimmer_apq11": call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        }
    except Exception:
        return {k: np.nan for k in JITTER_SHIMMER_KEYS}


def extract_hnr_features(sound: parselmouth.Sound) -> Dict[str, float]:
    """Harmonics-to-noise ratio (dB): voice breathiness / roughness."""
    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, PITCH_FLOOR, 0.1, 1.0)
        hnr_mean = call(harmonicity, "Get mean", 0, 0)
        return {"hnr_mean": float(hnr_mean)}
    except Exception:
        return {"hnr_mean": np.nan}


def extract_formant_features(sound: parselmouth.Sound) -> Dict[str, float]:
    """Formants F1-F3 (Hz), mean over the utterance: vowel articulation /
    vowel-space centralization, one of dysarthria's clearest acoustic signs."""
    try:
        formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                        maximum_formant=5500, window_length=0.025,
                                        pre_emphasis_from=50)
        return {
            "f1_mean": call(formant, "Get mean", 1, 0, 0, "Hertz"),
            "f2_mean": call(formant, "Get mean", 2, 0, 0, "Hertz"),
            "f3_mean": call(formant, "Get mean", 3, 0, 0, "Hertz"),
        }
    except Exception:
        return {"f1_mean": np.nan, "f2_mean": np.nan, "f3_mean": np.nan}


def extract_intensity_features(sound: parselmouth.Sound) -> Dict[str, float]:
    """Intensity mean/max (dB): loudness and a speaker's control over it."""
    try:
        intensity = sound.to_intensity(minimum_pitch=PITCH_FLOOR)
        return {
            "intensity_mean": call(intensity, "Get mean", 0, 0, "energy"),
            "intensity_max": call(intensity, "Get maximum", 0, 0, "Parabolic"),
        }
    except Exception:
        return {"intensity_mean": np.nan, "intensity_max": np.nan}


def extract_rhythm_features(sound: parselmouth.Sound, pitch=None, point_process=None) -> Dict[str, float]:
    """
    Speech-rate, pause, and voice-break proxies (see module docstring for
    why these are within-word proxies, not multi-word timing):
      speech_rate     glottal pulses per second - articulation-rate proxy
      pause_duration  seconds of locally-unvoiced frames within the clip
      voice_breaks    count of inter-pulse gaps > 1.25x the mean period,
                       Praat's own "Voice report" voice-break criterion
    """
    if pitch is None:
        pitch = sound.to_pitch(time_step=None, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    if point_process is None:
        point_process = _safe_point_process(sound)

    duration = sound.get_total_duration()
    f0 = pitch.selected_array["frequency"]
    unvoiced_fraction = float((f0 == 0).mean()) if len(f0) else np.nan
    pause_duration = unvoiced_fraction * duration if not np.isnan(unvoiced_fraction) else np.nan

    if point_process is None or duration <= 0:
        return {"speech_rate": np.nan, "pause_duration": pause_duration, "voice_breaks": np.nan}

    n_points = int(call(point_process, "Get number of points"))
    speech_rate = n_points / duration

    voice_breaks = 0
    if n_points > 1:
        times = np.array([call(point_process, "Get time from index", i) for i in range(1, n_points + 1)])
        periods = np.diff(times)
        mean_period = periods.mean()
        voice_breaks = int((periods > 1.25 * mean_period).sum())

    return {"speech_rate": speech_rate, "pause_duration": pause_duration, "voice_breaks": voice_breaks}


def extract_praat_features(filepath: str) -> Dict[str, float]:
    """
    All Praat acoustic features for one utterance's ORIGINAL audio file.
    Never raises: any feature group Praat can't compute on this clip (too
    short, unvoiced, no clear pitch periods) comes back as NaN so a full-
    dataset batch run doesn't die on one difficult file.
    """
    sound = parselmouth.Sound(filepath)
    try:
        pitch = sound.to_pitch(time_step=None, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    except Exception:
        pitch = None
    point_process = _safe_point_process(sound)

    features: Dict[str, float] = {}
    if pitch is not None:
        features.update(extract_pitch_features(sound, pitch))
    else:
        features.update({"f0_mean": np.nan, "f0_max": np.nan, "f0_min": np.nan})
    features.update(extract_jitter_shimmer_features(sound, point_process))
    features.update(extract_hnr_features(sound))
    features.update(extract_formant_features(sound))
    features.update(extract_intensity_features(sound))
    features.update(extract_rhythm_features(sound, pitch, point_process))
    return features


def extract_praat_features_batch(df: pd.DataFrame, cache_path: Optional[Path] = None,
                                 use_cache: bool = True, progress_every: int = 1000) -> pd.DataFrame:
    """
    Run extract_praat_features over every row of df (expects Filename,
    Speaker_ID, Group, Severity, Filepath columns, i.e. the M6 manifest
    shape), returning a features DataFrame joinable back onto it by
    Filename. Files Praat can't open at all (not just "some features
    undefined") are logged and skipped, not silently dropped.
    """
    if use_cache and cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if set(df["Filename"]).issubset(set(cached["Filename"])):
            print_kv("Praat features", f"loaded from cache ({cache_path})")
            return cached[cached["Filename"].isin(df["Filename"])].reset_index(drop=True)

    records = []
    failed = []
    total = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            features = extract_praat_features(row.Filepath)
        except Exception as e:
            failed.append((row.Filename, str(e)))
            features = {k: np.nan for k in FEATURE_COLUMNS}
        records.append({
            "Filename": row.Filename, "Speaker_ID": row.Speaker_ID,
            "Group": row.Group, "Severity": row.Severity, **features,
        })
        if (i + 1) % progress_every == 0:
            print_kv(f"  {i + 1}/{total}", f"{len(failed)} failed so far")

    print_kv("Praat features extracted", f"{total - len(failed)}/{total} succeeded")
    if failed:
        print_kv("Failed files", f"{len(failed)} (see returned DataFrame's NaN rows)")

    result = pd.DataFrame.from_records(records)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        print_kv("Praat features cached", cache_path)
    return result
