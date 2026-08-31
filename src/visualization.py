"""
Exploratory data analysis plots.

Every figure is saved into outputs/figures/ so results can be dropped straight
into the IEEE paper draft. Pass show=True (as the notebook does) to also render
the figure inline; scripts leave it False so nothing blocks on a window.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src import vad as vad_module
from src.console import print_header, print_kv, print_status, progress
from src.praat import FEATURE_COLUMNS, SEVERITY_GROUPS, severity_group
from src.style import apply_style

apply_style()


def _finish(fig, filename: str, show: bool) -> str:
    """Save a figure to outputs/figures/, optionally display it, then release it."""
    out_path = config.FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out_path)


def plot_praat_feature_comparison(features_df: pd.DataFrame, show: bool = False) -> str:
    """Grid of box plots: every Praat feature, split Healthy/Very Low/Low/Mid/High."""
    df = features_df.copy()
    df["Group_Order"] = severity_group(df)

    n_cols = 4
    n_rows = -(-len(FEATURE_COLUMNS) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten()

    for ax, feature in zip(axes, FEATURE_COLUMNS):
        sns.boxplot(x="Group_Order", y=feature, data=df, order=SEVERITY_GROUPS,
                   hue="Group_Order", palette="viridis", legend=False, ax=ax)
        ax.set_title(feature, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)

    for ax in axes[len(FEATURE_COLUMNS):]:
        ax.axis("off")

    fig.suptitle("Praat Acoustic Features by Severity Group", fontsize=18, y=1.01)
    fig.tight_layout()
    return _finish(fig, "praat_severity_comparison.png", show)


def build_praat_group_summary(features_df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std per Praat feature per severity group, ordered Healthy -> High."""
    df = features_df.copy()
    df["Group_Order"] = severity_group(df)
    summary = df.groupby("Group_Order")[list(FEATURE_COLUMNS)].agg(["mean", "std"])
    return summary.reindex(SEVERITY_GROUPS)


def plot_feature_correlation(features_df: pd.DataFrame, columns=FEATURE_COLUMNS,
                             show: bool = False) -> str:
    """Correlation heatmap across the Praat feature set.

    Groups like jitter (local/rap/ppq5/ddp) and shimmer are expected to
    correlate strongly with each other since they all measure the same
    underlying instability from slightly different formulas - this is what
    lets a reader see redundant feature groups at a glance before treating
    them as independent evidence in Phase 6's Praat pathway.
    """
    corr = features_df[list(columns)].corr()

    fig, ax = plt.subplots(figsize=(0.45 * len(columns) + 3, 0.45 * len(columns) + 2))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
               square=True, linewidths=0.3, ax=ax,
               cbar_kws={"shrink": 0.7, "label": "Pearson r"})
    ax.set_title("Praat Feature Correlation Matrix", fontsize=14)
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _finish(fig, "praat_feature_correlation.png", show)


# ---------------------------------------------------------------------------
# VAD-preprocessing validation panel — moved here from
# notebooks/02_feature_analysis.ipynb (was a ~130-line in-notebook function
# definition; notebooks should only call src/ functions, not define them —
# see the repository refactor's "no substantial logic in notebooks" rule).
# ---------------------------------------------------------------------------
def plot_vad_validation_panel(axes_row, row, category: str) -> dict:
    """Draw the 4-panel [raw+VAD region | processed waveform | MFCC valid
    frames | MFCC model input + padding mask] validation for one utterance
    into an existing 1x4 Axes row. Returns the VAD stats dict (plus frame-
    count bookkeeping), for the caller's summary printout.

    ROOT CAUSE this validates: src.preprocessing.load_and_preprocess trims
    with Silero VAD, then pads/truncates to a FIXED config.CLIP_SECONDS=4.0s
    window (config.MAX_SAMPLES samples) *before* MFCC ever runs — so a
    naive MFCC panel shows a long near-constant padded tail with no
    indication most frames are padding, not acoustic information. The fix
    (all three parts driven by the post-VAD valid SAMPLE count, never by
    inspecting MFCC values — a genuinely low-energy dysarthric frame must
    not be mistaken for padding):
      - src.preprocessing.mfcc_frame_count: one shared "samples -> frames" formula.
      - src.preprocessing.extract_mfcc_features(..., valid_length=...): deltas
        computed over the valid (pre-padding) slice only, then padded back out,
        so the padding boundary can't smear into the last real frames.
      - src.preprocessing.mfcc_valid_frame_mask: an explicit frame-level mask,
        used both to slice the "valid frames only" panel and to compute the
        color scale from valid frames only (padding must not dominate the
        color statistics of a plot that is, by construction, mostly padding).
    """
    from src.preprocessing import (_load_resampled, _pad_or_truncate, build_mfcc_transform,
                                   extract_mfcc_features, mfcc_valid_frame_mask,
                                   validate_mfcc_output)

    mfcc_transform = build_mfcc_transform()
    raw_waveform, sr = _load_resampled(row["Filepath"])
    raw_np = raw_waveform[0].numpy()
    trimmed_waveform, vad_stats = vad_module.apply_vad(raw_waveform, sr)
    processed_waveform, valid_length = _pad_or_truncate(trimmed_waveform)

    t_raw = np.arange(len(raw_np)) / sr
    axes_row[0].plot(t_raw, raw_np, linewidth=0.5, color="steelblue")
    if not vad_stats["fallback_used"]:
        speech_start_s = vad_stats["leading_trimmed_s"]
        speech_end_s = t_raw[-1] - vad_stats["trailing_trimmed_s"]
        axes_row[0].axvspan(speech_start_s, speech_end_s, color="orange", alpha=0.25)
    axes_row[0].set_title(f"[{category}] {row['Speaker_ID']} ({row['Severity']}) — raw + VAD region")
    axes_row[0].set_xlabel("s")

    t_proc = np.arange(processed_waveform.shape[1]) / sr
    axes_row[1].plot(t_proc, processed_waveform[0].numpy(), linewidth=0.5, color="seagreen")
    axes_row[1].set_title(f"VAD-processed ({vad_stats['speech_ratio']:.1%} speech, "
                          f"{vad_stats['num_segments']} segment(s))")
    axes_row[1].set_xlabel("s")

    mfcc = extract_mfcc_features(processed_waveform, mfcc_transform, valid_length=valid_length)
    total_frames = mfcc.shape[-1]
    frame_mask = mfcc_valid_frame_mask(total_frames, valid_length)
    validate_mfcc_output(mfcc, valid_length, frame_mask)
    valid_frames = int(frame_mask.sum().item())
    mfcc_np = mfcc[0].numpy()

    valid_slice = mfcc_np[:, :valid_frames] if valid_frames > 0 else mfcc_np
    vmin, vmax = np.percentile(valid_slice, [2, 98])

    axes_row[2].imshow(valid_slice, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
    axes_row[2].set_title(f"MFCC — valid frames only (0-{valid_frames})")
    axes_row[2].set_xlabel("frame")

    axes_row[3].imshow(mfcc_np, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
    if valid_frames < total_frames:
        axes_row[3].axvspan(valid_frames - 0.5, total_frames - 0.5, color="0.15", alpha=0.55,
                            hatch="///", edgecolor="white", linewidth=0)
    axes_row[3].set_title(f"MFCC — model input ({valid_frames} valid | "
                          f"{total_frames - valid_frames} padded / {total_frames})")
    axes_row[3].set_xlabel("frame")

    return {**vad_stats, "mfcc_total_frames": total_frames, "mfcc_valid_frames": valid_frames,
           "mfcc_padded_frames": total_frames - valid_frames,
           "mfcc_padding_ratio": (total_frames - valid_frames) / total_frames}


def plot_vad_validation_examples(df_m6: pd.DataFrame, seed: int = None
                                 ) -> Tuple[str, pd.DataFrame]:
    """
    Five VAD-behaviour example utterances (normal duration, long trailing
    silence, long leading silence, internal pauses, weak/low-energy
    dysarthric), each rendered by plot_vad_validation_panel — the combined
    figure plus per-example figures, and a summary table. Reuses
    outputs/vad_stats.csv (src.preprocessing.compute_vad_stats_batch, see
    notebooks/01_data_pipeline.ipynb Stage 9) if already computed; otherwise
    samples up to 15 utterances per speaker to select from.

    Returns (combined_figure_path, example_stats_df).
    """
    from src.preprocessing import load_and_preprocess_with_stats

    seed = seed if seed is not None else config.DEFAULT_SEED

    if config.VAD_STATS_PATH.exists():
        vad_stats_sample = pd.read_csv(config.VAD_STATS_PATH).merge(
            df_m6[["Filename", "Filepath", "Group", "Severity"]], on="Filename", how="inner")
    else:
        sample_df = (df_m6.groupby("Speaker_ID", group_keys=False)
                    .apply(lambda g: g.sample(min(15, len(g)), random_state=seed)))
        records = []
        for row in progress(sample_df.itertuples(index=False),
                            "Sampling VAD stats for example selection",
                            total=len(sample_df), unit="utt"):
            _, _, stats = load_and_preprocess_with_stats(row.Filepath)
            records.append({"Filename": row.Filename, "Filepath": row.Filepath,
                            "Group": row.Group, "Severity": row.Severity, **stats})
        vad_stats_sample = pd.DataFrame(records)

    not_fallback = vad_stats_sample[~vad_stats_sample["fallback_used"]]
    dysarthric = not_fallback[not_fallback["Group"] == "Dysarthric Patient"]

    category_picks = {
        "normal duration": not_fallback.iloc[
            (not_fallback["speech_ratio"] - not_fallback["speech_ratio"].median()).abs().argsort()[:1]],
        "long trailing silence": not_fallback.nlargest(1, "trailing_trimmed_s"),
        "long leading silence": not_fallback.nlargest(1, "leading_trimmed_s"),
        "internal pauses": not_fallback.nlargest(1, "num_segments"),
        "weak/low-energy dysarthric": (dysarthric if len(dysarthric) else not_fallback).nsmallest(1, "speech_ratio"),
    }
    example_rows = [(category, df_m6[df_m6["Filename"] == picks.iloc[0]["Filename"]].iloc[0])
                    for category, picks in category_picks.items() if len(picks)]

    fig, axes = plt.subplots(len(example_rows), 4, figsize=(18, 3.2 * len(example_rows)))
    example_stats = []

    for i, (category, row) in enumerate(example_rows):
        row_axes = axes[i] if len(example_rows) > 1 else axes
        vad_stats = plot_vad_validation_panel(row_axes, row, category)
        example_stats.append({"Category": category, "Speaker_ID": row["Speaker_ID"],
                              "Severity": row["Severity"], **vad_stats})

        per_example_fig, per_example_axes = plt.subplots(1, 4, figsize=(18, 3.2))
        plot_vad_validation_panel(per_example_axes, row, category)
        per_example_fig.tight_layout()
        per_example_path = (config.FIGURE_DIR /
                            f"vad_validation_{category.replace(' ', '_').replace('/', '-')}"
                            f"_{row['Speaker_ID']}_{Path(row['Filename']).stem}.png")
        per_example_fig.savefig(per_example_path, dpi=150, bbox_inches="tight")
        plt.close(per_example_fig)

    fig.legend(handles=[mpatches.Patch(facecolor="0.15", alpha=0.55, hatch="///",
                                       edgecolor="white", label="padded (not real audio)")],
              loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    vad_validation_path = config.FIGURE_DIR / "vad_validation_examples.png"
    fig.savefig(vad_validation_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print_header("VAD Validation")
    print_kv("Examples (5 VAD-behaviour categories)", len(example_rows))
    print_kv("Combined figure", vad_validation_path)
    print_kv("Per-example figures", f"{config.FIGURE_DIR}/vad_validation_<category>_<speaker>_<file>.png")
    print_status(f"{int(vad_stats_sample['fallback_used'].sum())}/{len(vad_stats_sample)} sampled "
                "utterances fell back to the original (untrimmed) waveform",
                ok=(vad_stats_sample["fallback_used"].sum() == 0))

    stats_df = pd.DataFrame(example_stats)[
        ["Category", "Speaker_ID", "Severity", "original_duration_s", "speech_duration_s",
         "speech_ratio", "num_segments", "mfcc_valid_frames", "mfcc_padded_frames",
         "mfcc_padding_ratio", "fallback_used"]]
    return str(vad_validation_path), stats_df
