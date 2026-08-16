"""
Phase 5 - Error analysis.

Turns the aggregate accuracy number into an account of *what the model actually
gets wrong*: which utterances, which speakers, which severity groups, and - by
joining against Phase 4's Praat features - which acoustic characteristics the
failures share.

Consumes what Phase 1/2 already write:
  outputs/predictions/<run>/<fold>.csv   per-utterance predictions (keyed by filename)
  outputs/praat_features.csv             Phase 4 acoustic features (keyed by Filename)
  outputs/m6_manifest.csv                Filepath / Word / Severity metadata

The `filename` key is what makes all of this possible. Predictions written before
that column existed cannot be analysed - load_run_predictions says so explicitly
rather than failing on a missing column somewhere deep in a merge.

Like src/praat.py, this module owns its own plots: the diagnostics here are a
Phase 5 concern, not the EDA that src/visualization.py covers, and not the
embedding-space/attention/SHAP introspection that src/model_analysis.py covers.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
import seaborn as sns
from scipy.stats import mannwhitneyu

from src import config
from src.console import print_kv
from src.praat import (FEATURE_COLUMNS, PITCH_CEILING, PITCH_FLOOR,
                       severity_group)
from src.preprocessing import (build_mfcc_transform, extract_mfcc_features,
                               load_and_preprocess)
from src.style import CORRECT_COLOR, ERROR_COLOR, apply_style

apply_style()

# Praat's own default: below this many voiced frames a pitch contour is not
# worth plotting.
MIN_VOICED_FRAMES = 2


# ---------------------------------------------------------------------------
# Loading and joining
# ---------------------------------------------------------------------------
def load_run_predictions(run_name: str, predictions_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Concatenate every per-fold predictions CSV for one run, tagging each row with
    the fold it was held out in.

    Raises if the CSVs predate the `filename` column - those rows cannot be
    joined to audio or to the Praat features, so every downstream step here
    would be meaningless.
    """
    predictions_dir = predictions_dir or config.PREDICTIONS_DIR
    run_dir = Path(predictions_dir) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(
            f"No predictions for run '{run_name}' at {run_dir}. "
            "Train it first (src.training.runner.run_training)."
        )

    # ALL_FOLDS_pooled.* are cross-fold artefacts, not per-fold predictions.
    fold_files = sorted(p for p in run_dir.glob("*.csv") if not p.stem.startswith("ALL_FOLDS"))
    if not fold_files:
        raise FileNotFoundError(f"No per-fold prediction CSVs in {run_dir}.")

    frames = []
    for path in fold_files:
        fold_df = pd.read_csv(path)
        fold_df.insert(0, "fold", path.stem)
        frames.append(fold_df)
    preds = pd.concat(frames, ignore_index=True)

    if "filename" not in preds.columns:
        raise ValueError(
            f"Predictions for '{run_name}' have no 'filename' column, so a "
            "misclassified row cannot be traced back to its audio file or to "
            "outputs/praat_features.csv.\n"
            "These CSVs were written before src/training/reporting.py carried "
            f"utterance identity through. Re-run '{run_name}' to regenerate them."
        )

    print_kv(f"Predictions ({run_name})", f"{len(preds)} rows from {len(fold_files)} fold(s)")
    return preds


def attach_metadata(preds: pd.DataFrame, manifest: pd.DataFrame,
                    praat_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Join Filepath / WordCode / Severity (and, if given, every Praat feature) onto
    the predictions by filename. Left join: a prediction row must never be
    silently dropped just because a feature is missing for it.
    """
    manifest_cols = [c for c in ("Filename", "Filepath", "Speaker_ID", "Group",
                                 "Severity", "WordCode", "Block") if c in manifest.columns]
    merged = preds.merge(manifest[manifest_cols], left_on="filename",
                         right_on="Filename", how="left")

    if praat_features is not None:
        feature_cols = ["Filename"] + [c for c in FEATURE_COLUMNS if c in praat_features.columns]
        merged = merged.merge(praat_features[feature_cols], on="Filename", how="left")

    merged["Severity_Group"] = severity_group(merged)
    unmatched = int(merged["Filepath"].isna().sum()) if "Filepath" in merged else 0
    if unmatched:
        print_kv("WARNING: unmatched rows", f"{unmatched} prediction(s) not found in the manifest")
    return merged


# ---------------------------------------------------------------------------
# Aggregate error structure
# ---------------------------------------------------------------------------
def error_summary(preds: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Where the errors live: overall, and broken down by severity group, speaker,
    true class, and word. Each breakdown is sorted worst-first, because the
    interesting question is which slice the model fails on, not the mean.
    """
    def _rate(by: str) -> pd.DataFrame:
        grouped = preds.groupby(by).agg(
            n=("correct", "size"),
            n_wrong=("correct", lambda s: int((~s.astype(bool)).sum())),
        )
        grouped["error_rate"] = grouped["n_wrong"] / grouped["n"]
        return grouped.sort_values("error_rate", ascending=False)

    summary = {
        "overall": pd.DataFrame([{
            "n": len(preds),
            "n_wrong": int((~preds["correct"].astype(bool)).sum()),
            "error_rate": float((~preds["correct"].astype(bool)).mean()),
        }]),
    }
    for column, key in (("Severity_Group", "by_severity"), ("speaker_id", "by_speaker"),
                        ("y_true_label", "by_true_class"), ("WordCode", "by_word")):
        if column in preds.columns:
            summary[key] = _rate(column)

    # The confusion pairs, as counts - which class gets mistaken for which.
    summary["confusions"] = (
        preds[~preds["correct"].astype(bool)]
        .groupby(["y_true_label", "y_pred_label"]).size()
        .reset_index(name="count").sort_values("count", ascending=False)
    )
    return summary


def fp_fn_breakdown(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Per-class false positives / false negatives, one-vs-rest.

    error_summary()'s "confusions" table already carries the raw which-class-
    for-which-class counts; this collapses that into the FP/FN framing
    directly (rate of the actual class missed vs. rate of the predicted class
    that was wrong), which a which-class-for-which-class table doesn't state
    on its own. Sorted worst-first by the rate that matters clinically: how
    often an actual case of this class was missed.
    """
    classes = sorted(set(preds["y_true_label"]) | set(preds["y_pred_label"]))
    records = []
    for cls in classes:
        is_true = preds["y_true_label"] == cls
        is_pred = preds["y_pred_label"] == cls
        tp = int((is_true & is_pred).sum())
        fp = int((~is_true & is_pred).sum())
        fn = int(is_true.sum()) - tp
        n_true = int(is_true.sum())
        n_pred = int(is_pred.sum())
        records.append({
            "class": cls,
            "tp": tp, "fp": fp, "fn": fn,
            "n_true": n_true,
            "false_positive_rate": fp / n_pred if n_pred else np.nan,
            "false_negative_rate": fn / n_true if n_true else np.nan,
        })
    return (pd.DataFrame.from_records(records)
            .sort_values("false_negative_rate", ascending=False)
            .reset_index(drop=True))


def _confidence(preds: pd.DataFrame) -> pd.Series:
    """Probability the model assigned to the class it predicted.

    Detection CSVs store a single `prob_positive`; severity CSVs store one
    prob_<class> column per class."""
    if "prob_positive" in preds.columns:
        p = preds["prob_positive"].astype(float)
        return np.where(preds["y_pred"] == 1, p, 1.0 - p)

    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    probs = preds[prob_cols].to_numpy(dtype=float)
    return probs[np.arange(len(preds)), preds["y_pred"].to_numpy(dtype=int)]


def most_confident_errors(preds: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """The n misclassifications the model was *most sure* about.

    These are the informative failures: a wrong call at 0.51 is a coin flip, but
    a wrong call at 0.99 means the model has learned something wrong."""
    errors = preds[~preds["correct"].astype(bool)].copy()
    if errors.empty:
        return errors
    errors["confidence"] = _confidence(errors)
    return errors.sort_values("confidence", ascending=False).head(n)


def compare_error_vs_correct(preds: pd.DataFrame) -> pd.DataFrame:
    """
    Per Praat feature: do the misclassified utterances differ acoustically from
    the correctly-classified ones?

    Mann-Whitney U (non-parametric, same reasoning as Phase 4's Kruskal-Wallis)
    plus Cliff's delta as the effect size. Cliff's delta is reported because a
    p-value on ~21k utterances will be significant for effects far too small to
    matter - delta says *how* separated the two groups are, on a -1..+1 scale:

        |delta| < 0.15  negligible     0.15-0.33  small
        0.33-0.47       medium         > 0.47     large

    Sorted by |delta|, so the top rows answer "do errors cluster at low HNR /
    high jitter / short duration?" directly.
    """
    available = [c for c in FEATURE_COLUMNS if c in preds.columns]
    if not available:
        raise ValueError(
            "No Praat feature columns on these predictions - call attach_metadata() "
            "with praat_features first (see notebooks/02_feature_analysis.ipynb)."
        )

    is_correct = preds["correct"].astype(bool)
    records = []

    for feature in available:
        correct_vals = preds.loc[is_correct, feature].dropna().to_numpy()
        error_vals = preds.loc[~is_correct, feature].dropna().to_numpy()
        if len(correct_vals) < 2 or len(error_vals) < 2:
            continue

        u_stat, p_value = mannwhitneyu(error_vals, correct_vals, alternative="two-sided")
        # Cliff's delta straight from U: delta = 2U/(n1*n2) - 1.
        delta = 2.0 * u_stat / (len(error_vals) * len(correct_vals)) - 1.0

        records.append({
            "feature": feature,
            "mean_correct": float(correct_vals.mean()),
            "mean_error": float(error_vals.mean()),
            "delta_pct": float(100.0 * (error_vals.mean() - correct_vals.mean())
                               / correct_vals.mean()) if correct_vals.mean() else np.nan,
            "cliffs_delta": float(delta),
            "p": float(p_value),
            "p_adj": float(min(p_value * len(available), 1.0)),
        })

    result = pd.DataFrame.from_records(records)
    result["magnitude"] = pd.cut(
        result["cliffs_delta"].abs(), bins=[-0.01, 0.15, 0.33, 0.47, 1.0],
        labels=["negligible", "small", "medium", "large"])
    result["significant"] = result["p_adj"] < 0.05
    return (result.reindex(result["cliffs_delta"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Per-utterance diagnostics
# ---------------------------------------------------------------------------
def _pitch_contour(filepath: str):
    """(times, f0) over voiced frames of the ORIGINAL audio, or (None, None)."""
    try:
        sound = parselmouth.Sound(filepath)
        pitch = sound.to_pitch(time_step=None, pitch_floor=PITCH_FLOOR,
                               pitch_ceiling=PITCH_CEILING)
    except Exception:
        return None, None

    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    voiced = f0 > 0
    if voiced.sum() < MIN_VOICED_FRAMES:
        return None, None
    return times[voiced], f0[voiced]


def plot_utterance_diagnostics(row, show: bool = False,
                               out_dir: Optional[Path] = None) -> Optional[str]:
    """
    Four-panel diagnostic for one utterance: waveform, spectrogram, MFCC heatmap,
    and Praat F0 contour.

    The waveform/spectrogram/pitch panels come from the ORIGINAL audio (what a
    clinician would look at); the MFCC panel comes from the preprocessed 4-second
    window (what the Acoustic Pathway actually saw). That split is deliberate -
    it is the same distinction src/praat.py draws, and seeing both is how you
    tell a genuinely hard utterance from a preprocessing artefact.
    """
    filepath = row["Filepath"]
    if not isinstance(filepath, str) or not Path(filepath).exists():
        print_kv("Skipped", f"{row.get('filename', '?')} — audio not found")
        return None

    sound = parselmouth.Sound(filepath)
    original = sound.values[0]
    sr = sound.sampling_frequency
    times = np.arange(len(original)) / sr

    processed, valid_length = load_and_preprocess(filepath)
    mfcc = extract_mfcc_features(processed, build_mfcc_transform(),
                                 valid_length=valid_length)[0].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(times, original, linewidth=0.5, color="#3b6ea5")
    axes[0, 0].set_title("Waveform (original audio)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].specgram(original, Fs=sr, NFFT=400, noverlap=240, cmap="magma")
    axes[0, 1].set_title("Spectrogram (original audio)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")

    im = axes[1, 0].imshow(mfcc, aspect="auto", origin="lower", cmap="viridis")
    axes[1, 0].set_title("MFCC + delta + delta-delta (what the model saw)")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Coefficient (39)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046)

    pitch_times, pitch_f0 = _pitch_contour(filepath)
    if pitch_times is None:
        axes[1, 1].text(0.5, 0.5, "no voiced frames\n(Praat found no pitch)",
                        ha="center", va="center", transform=axes[1, 1].transAxes,
                        fontsize=11, color="grey")
    else:
        axes[1, 1].plot(pitch_times, pitch_f0, "o", markersize=2.5, color="#c44e52")
        axes[1, 1].set_ylim(PITCH_FLOOR, min(PITCH_CEILING, float(pitch_f0.max()) * 1.2))
    axes[1, 1].set_title("Praat F0 contour (original audio)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("F0 (Hz)")

    confidence = row.get("confidence")
    confidence_text = f" | p={confidence:.3f}" if isinstance(confidence, float) else ""
    fig.suptitle(
        f"{row['filename']}  —  true: {row['y_true_label']}  |  "
        f"predicted: {row['y_pred_label']}{confidence_text}",
        fontsize=14, color=ERROR_COLOR if not row["correct"] else CORRECT_COLOR)
    fig.tight_layout()

    out_dir = Path(out_dir or config.ERROR_FIGURE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(str(row['filename'])).stem}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out_path)


def plot_error_gallery(preds: pd.DataFrame, run_name: str, n: int = 8,
                       show: bool = False) -> List[str]:
    """Diagnostics for the n most confident misclassifications of a run."""
    errors = most_confident_errors(preds, n=n)
    if errors.empty:
        print_kv("Error gallery", "no misclassified utterances — nothing to plot")
        return []

    out_dir = config.ERROR_FIGURE_DIR / run_name
    paths = []
    for _, row in errors.iterrows():
        path = plot_utterance_diagnostics(row, show=show, out_dir=out_dir)
        if path:
            paths.append(path)
    print_kv("Error gallery", f"{len(paths)} figure(s) in {out_dir}")
    return paths


def plot_error_feature_distributions(preds: pd.DataFrame, comparison: pd.DataFrame,
                                     run_name: str, top_k: int = 6,
                                     show: bool = False) -> str:
    """Box plots, correct vs misclassified, for the top_k most separating features
    (by |Cliff's delta|, as ranked by compare_error_vs_correct)."""
    features = comparison.head(top_k)["feature"].tolist()
    df = preds.copy()
    df["Outcome"] = np.where(df["correct"].astype(bool), "Correct", "Misclassified")

    n_cols = 3
    n_rows = -(-len(features) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, feature in zip(axes, features):
        sns.boxplot(x="Outcome", y=feature, data=df, hue="Outcome",
                    order=["Correct", "Misclassified"], palette=[CORRECT_COLOR, ERROR_COLOR],
                    legend=False, ax=ax)
        delta = comparison.loc[comparison["feature"] == feature, "cliffs_delta"].iloc[0]
        ax.set_title(f"{feature}  (delta={delta:+.2f})", fontsize=11)
        ax.set_xlabel("")

    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle(f"Acoustic profile of {run_name}'s errors vs its correct predictions",
                 fontsize=15, y=1.01)
    fig.tight_layout()

    out_path = config.FIGURE_DIR / f"error_features_{run_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out_path)
