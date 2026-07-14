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
from scipy.stats import kruskal

from src.console import print_kv, print_status, print_subheader, progress

PITCH_FLOOR = 75.0      # Hz - standard floor for adult speech
PITCH_CEILING = 600.0   # Hz - generous enough for female/pathological voices

PITCH_KEYS = ("f0_mean", "f0_max", "f0_min", "f0_std", "f0_range")
JITTER_SHIMMER_KEYS = ("jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp",
                       "shimmer_local", "shimmer_apq3", "shimmer_apq11", "shimmer_dda")
HNR_KEYS = ("hnr_mean", "hnr_std", "hnr_min")
FORMANT_KEYS = ("f1_mean", "f2_mean", "f3_mean",
                "f1_std", "f2_std", "f3_std", "f2_f1_ratio")
INTENSITY_KEYS = ("intensity_mean", "intensity_max", "intensity_min", "intensity_std")
RHYTHM_KEYS = ("speech_rate", "pause_duration", "voice_breaks")

FEATURE_COLUMNS = (
    *PITCH_KEYS,
    *JITTER_SHIMMER_KEYS,
    *HNR_KEYS,
    *FORMANT_KEYS,
    *INTENSITY_KEYS,
    *RHYTHM_KEYS,
)

# Columns every features table carries alongside FEATURE_COLUMNS - the join key
# back onto m6_manifest.csv plus the labels the group comparison splits on.
ID_COLUMNS = ("Filename", "Speaker_ID", "Group", "Severity")

SEVERITY_GROUPS = ["Healthy", "Very Low", "Low", "Mid", "High"]


def _safe_point_process(sound: parselmouth.Sound):
    """A PointProcess needs enough regular glottal pulses to define a period -
    short or heavily dysarthric clips can fail this. None on failure."""
    try:
        return call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
    except Exception:
        return None


def extract_pitch_features(sound: parselmouth.Sound,
                           pitch: Optional[parselmouth.Pitch] = None) -> Dict[str, float]:
    """F0 mean/max/min/std/range (Hz), voiced frames only.

    std and range capture monopitch - the flattened intonation that is one of
    dysarthria's most audible prosodic signs, and which a mean alone hides."""
    if pitch is None:
        pitch = sound.to_pitch(time_step=None, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    voiced = pitch.selected_array["frequency"]
    voiced = voiced[voiced != 0]
    if len(voiced) == 0:
        return {k: np.nan for k in PITCH_KEYS}
    return {
        "f0_mean": float(voiced.mean()),
        "f0_max": float(voiced.max()),
        "f0_min": float(voiced.min()),
        "f0_std": float(voiced.std()),
        "f0_range": float(voiced.max() - voiced.min()),
    }


def extract_jitter_shimmer_features(sound: parselmouth.Sound, point_process=None) -> Dict[str, float]:
    """Jitter (local, RAP, PPQ5, DDP) and shimmer (local, APQ3, APQ11, DDA):
    period-to-period and amplitude-to-amplitude perturbation, standard
    dysarthria biomarkers for vocal fold instability."""
    if point_process is None:
        point_process = _safe_point_process(sound)
    if point_process is None:
        return {k: np.nan for k in JITTER_SHIMMER_KEYS}
    try:
        return {
            "jitter_local": call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            "jitter_rap": call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            "jitter_ppq5": call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            "jitter_ddp": call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
            "shimmer_local": call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "shimmer_apq3": call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "shimmer_apq11": call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            "shimmer_dda": call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        }
    except Exception:
        return {k: np.nan for k in JITTER_SHIMMER_KEYS}


HNR_SILENCE_FLOOR = -100.0   # dB - Praat parks silent/unvoiced frames near -200


def extract_hnr_features(sound: parselmouth.Sound) -> Dict[str, float]:
    """Harmonics-to-noise ratio (dB): voice breathiness / roughness.

    mean/std/min - a low minimum marks the worst moment of the utterance, which
    a mean over a mostly-clean clip washes out.

    hnr_min is taken from the frame array rather than Praat's "Get minimum":
    that call reports the ~-200 dB sentinel Praat writes into silent frames, so
    on a zero-padded/leading-silence clip it returns -200 for every file and
    carries no signal at all. "Get mean"/"Get standard deviation" already skip
    those frames, so only the minimum needs the guard."""
    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, PITCH_FLOOR, 0.1, 1.0)
        frames = np.asarray(harmonicity.values).ravel()
        voiced = frames[frames > HNR_SILENCE_FLOOR]
        return {
            "hnr_mean": float(call(harmonicity, "Get mean", 0, 0)),
            "hnr_std": float(call(harmonicity, "Get standard deviation", 0, 0)),
            "hnr_min": float(voiced.min()) if len(voiced) else np.nan,
        }
    except Exception:
        return {k: np.nan for k in HNR_KEYS}


def extract_formant_features(sound: parselmouth.Sound) -> Dict[str, float]:
    """
    Formants F1-F3 (Hz), mean and std over the utterance, plus the F2/F1 ratio.

    This is LINEAR PREDICTION ANALYSIS: to_formant_burg estimates an all-pole
    vocal-tract filter by the Burg method and reads the formants off its complex
    root pairs. F1/F2 locate a vowel in the vowel space, so their ratio is a
    direct measure of vowel-space centralization - dysarthric articulation
    undershoots the vowel targets, compressing the space and pulling F2/F1 toward
    1. It is one of the clearest and most-cited acoustic correlates of the
    disorder, and unlike the MFCC/wav2vec representations it is interpretable by
    name to a clinician.
    """
    try:
        formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5,
                                        maximum_formant=5500, window_length=0.025,
                                        pre_emphasis_from=50)
        means = {f"f{i}_mean": call(formant, "Get mean", i, 0, 0, "Hertz") for i in (1, 2, 3)}
        stds = {f"f{i}_std": call(formant, "Get standard deviation", i, 0, 0, "Hertz")
                for i in (1, 2, 3)}
        f1, f2 = means["f1_mean"], means["f2_mean"]
        ratio = float(f2 / f1) if f1 and not np.isnan(f1) and f1 > 0 else np.nan
        return {**means, **stds, "f2_f1_ratio": ratio}
    except Exception:
        return {k: np.nan for k in FORMANT_KEYS}


def extract_intensity_features(sound: parselmouth.Sound) -> Dict[str, float]:
    """Intensity mean/max/min/std (dB): loudness and a speaker's control over
    it - std is a direct proxy for the reduced loudness variation of hypokinetic
    dysarthria."""
    try:
        intensity = sound.to_intensity(minimum_pitch=PITCH_FLOOR)
        return {
            "intensity_mean": call(intensity, "Get mean", 0, 0, "energy"),
            "intensity_max": call(intensity, "Get maximum", 0, 0, "Parabolic"),
            "intensity_min": call(intensity, "Get minimum", 0, 0, "Parabolic"),
            "intensity_std": call(intensity, "Get standard deviation", 0, 0),
        }
    except Exception:
        return {k: np.nan for k in INTENSITY_KEYS}


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
        features.update({k: np.nan for k in PITCH_KEYS})
    features.update(extract_jitter_shimmer_features(sound, point_process))
    features.update(extract_hnr_features(sound))
    features.update(extract_formant_features(sound))
    features.update(extract_intensity_features(sound))
    features.update(extract_rhythm_features(sound, pitch, point_process))
    return features


def extract_praat_features_batch(df: pd.DataFrame, cache_path: Optional[Path] = None,
                                 use_cache: bool = True) -> pd.DataFrame:
    """
    Run extract_praat_features over every row of df (expects Filename,
    Speaker_ID, Group, Severity, Filepath columns, i.e. the M6 manifest
    shape), returning a features DataFrame joinable back onto it by
    Filename. Files Praat can't open at all (not just "some features
    undefined") are logged and skipped, not silently dropped.
    """
    if use_cache and cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        has_every_file = set(df["Filename"]).issubset(set(cached["Filename"]))
        # A cache written before FEATURE_COLUMNS was extended covers every file
        # but not every column. Checking only the filenames would silently serve
        # the stale schema forever, so the column set has to match too.
        stale_columns = [c for c in FEATURE_COLUMNS if c not in cached.columns]
        if has_every_file and not stale_columns:
            print_kv("Praat features", f"loaded from cache ({cache_path})")
            return cached[cached["Filename"].isin(df["Filename"])].reset_index(drop=True)
        if has_every_file and stale_columns:
            print_kv("Praat cache stale", f"missing {len(stale_columns)} feature column(s) "
                     f"(e.g. {', '.join(stale_columns[:3])}) — re-extracting")

    records = []
    failed = []
    total = len(df)

    print_subheader(f"Praat acoustic analysis — {len(FEATURE_COLUMNS)} features "
                    f"x {total:,} utterances")
    for row in progress(df.itertuples(index=False), "Extracting Praat features",
                        total=total, unit="utt"):
        try:
            features = extract_praat_features(row.Filepath)
        except Exception as e:
            failed.append((row.Filename, str(e)))
            features = {k: np.nan for k in FEATURE_COLUMNS}
        records.append({
            "Filename": row.Filename, "Speaker_ID": row.Speaker_ID,
            "Group": row.Group, "Severity": row.Severity, **features,
        })

    print_status(f"{total - len(failed):,}/{total:,} utterances measured",
                 ok=not failed)
    if failed:
        print_status(f"{len(failed)} file(s) Praat could not open — their rows are "
                     f"all-NaN, not dropped", ok=False)

    result = pd.DataFrame.from_records(records)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        print_kv("Praat features cached", cache_path)
    return result


def severity_group(features_df: pd.DataFrame) -> pd.Series:
    """The five-way group label the Phase 4 comparison splits on: controls carry
    Severity 'N/A (Control)' in the manifest, which reads as 'Healthy' here."""
    return features_df["Severity"].replace("N/A (Control)", "Healthy")


def praat_group_significance(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Kruskal-Wallis H per feature across Healthy / Very Low / Low / Mid / High.

    Non-parametric on purpose: jitter, shimmer and the pause/voice-break proxies
    are bounded and heavily skewed, so an ANOVA's normality assumption does not
    hold. NaNs are dropped per feature (a clip Praat could not measure should not
    count as a zero), and p is Bonferroni-corrected across the feature set, since
    testing ~28 features at once would otherwise manufacture significance.

    Returns one row per feature - H, p, p_adj, significant, n_used - sorted most
    significant first. This is what makes "these measures separate the severity
    groups" a claim rather than an eyeball of the box plots.
    """
    groups = severity_group(features_df)
    records = []

    for feature in FEATURE_COLUMNS:
        samples = [
            features_df.loc[groups == name, feature].dropna().to_numpy()
            for name in SEVERITY_GROUPS
        ]
        samples = [s for s in samples if len(s) > 0]
        n_used = int(sum(len(s) for s in samples))

        # Kruskal-Wallis needs >= 2 groups, and at least one group must vary -
        # a feature that is constant everywhere (or measurable in only one
        # group) has no between-group signal to test.
        if len(samples) < 2 or all(np.ptp(s) == 0 for s in samples):
            h_stat, p_value = np.nan, np.nan
        else:
            try:
                h_stat, p_value = kruskal(*samples)
            except ValueError:
                h_stat, p_value = np.nan, np.nan

        records.append({
            "feature": feature,
            "H": float(h_stat),
            "p": float(p_value),
            "p_adj": float(min(p_value * len(FEATURE_COLUMNS), 1.0)) if not np.isnan(p_value) else np.nan,
            "n_used": n_used,
        })

    result = pd.DataFrame.from_records(records)
    result["significant"] = result["p_adj"] < 0.05
    return result.sort_values("p_adj", na_position="last").reset_index(drop=True)


def load_praat_table(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the cached Phase 4 features CSV, indexed by Filename for O(1) lookup
    from the Dataset. Raises with an actionable message rather than a bare
    FileNotFoundError - a missing table means Phase 4 was simply never run.
    """
    from src import config

    cache_path = cache_path or config.PRAAT_FEATURES_PATH
    if not Path(cache_path).exists():
        raise FileNotFoundError(
            f"Praat features not found at {cache_path}. Run Stage 1 of "
            "notebooks/03_praat_analysis.ipynb (extract_praat_features_batch) "
            "to generate them - the Praat-fusion model cannot train without it."
        )
    table = pd.read_csv(cache_path)
    missing = [c for c in FEATURE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"{cache_path} is missing {len(missing)} feature column(s) "
            f"(e.g. {missing[:3]}). It was written by an older version of "
            "src/praat.py - delete it and re-run notebooks/03_praat_analysis.ipynb."
        )
    return table.set_index("Filename")


def praat_standardizer(table: pd.DataFrame, filenames) -> tuple:
    """
    Mean/std of each feature over *only* the given filenames.

    Callers pass a fold's train split, so the test speaker never contributes to
    the standardization statistics - the same leakage discipline the LOSO folds
    themselves enforce. NaNs are ignored here and imputed to 0 after
    standardization by praat_vector(), i.e. to the train-split mean.
    """
    subset = table.reindex(list(filenames))[list(FEATURE_COLUMNS)]
    mean = subset.mean(skipna=True).to_numpy(dtype=np.float32)
    std = subset.std(skipna=True).to_numpy(dtype=np.float32)
    std[~np.isfinite(std) | (std == 0)] = 1.0        # constant feature -> leave centred
    mean[~np.isfinite(mean)] = 0.0
    return mean, std


def praat_vector(table: pd.DataFrame, filename: str, stats: tuple) -> np.ndarray:
    """One utterance's standardized feature vector, NaN-imputed to the mean (0)."""
    mean, std = stats
    row = table.loc[filename, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
    standardized = (row - mean) / std
    return np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
