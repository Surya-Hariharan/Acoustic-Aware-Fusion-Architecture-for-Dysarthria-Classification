"""
Exploratory data analysis plots.

Every figure is saved into outputs/figures/ so results can be dropped straight
into the IEEE paper draft. Pass show=True (as the notebook does) to also render
the figure inline; scripts leave it False so nothing blocks on a window.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.console import print_header, print_kv
from src.praat import FEATURE_COLUMNS, SEVERITY_GROUPS, severity_group

sns.set_style("whitegrid")


def _finish(fig, filename: str, show: bool) -> str:
    """Save a figure to outputs/figures/, optionally display it, then release it."""
    out_path = config.FIGURE_DIR / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out_path)


def plot_files_per_speaker(df: pd.DataFrame, show: bool = False) -> str:
    """Bar chart of audio files per speaker, coloured by group."""
    counts = (df.groupby(["Speaker_ID", "Group"]).size()
              .reset_index(name="File Count"))

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(x="Speaker_ID", y="File Count", hue="Group",
                data=counts, palette="viridis", ax=ax)
    ax.set_title("Number of Audio Files per Speaker, Grouped by Health Status")
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Number of Audio Files")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(title="Speaker Group")
    fig.tight_layout()

    return _finish(fig, "files_per_speaker.png", show)


def plot_dataset_dashboard(df: pd.DataFrame, show: bool = False) -> str:
    """2x2 EDA dashboard: group share, speaker counts, per-speaker files, mics."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("UA-Speech Dataset: Exploratory Data Analysis",
                 fontsize=20, y=1.02)

    # (0, 0) donut chart of the file share per group
    group_counts = df["Group"].value_counts()
    axes[0, 0].pie(group_counts, labels=group_counts.index,
                   autopct="%1.1f%%", startangle=90,
                   explode=[0.05] * len(group_counts),
                   colors=sns.color_palette("pastel")[:len(group_counts)],
                   pctdistance=0.85)
    axes[0, 0].add_artist(plt.Circle((0, 0), 0.70, fc="white"))
    axes[0, 0].set_title("Distribution of Total Audio Files by Group", fontsize=14)
    axes[0, 0].axis("equal")

    # (0, 1) unique speakers per group
    speakers = (df.groupby("Group")["Speaker_ID"].nunique()
                .reset_index(name="Unique Speakers"))
    sns.barplot(x="Group", y="Unique Speakers", hue="Group", data=speakers,
                ax=axes[0, 1], palette="viridis", legend=False)
    axes[0, 1].set_title("Number of Unique Speakers per Group", fontsize=14)
    axes[0, 1].set_xlabel("Speaker Group")
    axes[0, 1].set_ylabel("Count of Unique Speakers")
    for container in axes[0, 1].containers:
        axes[0, 1].bar_label(container, fmt="%d")

    # (1, 0) files per speaker
    per_speaker = (df.groupby(["Speaker_ID", "Group"]).size()
                   .reset_index(name="File Count"))
    sns.barplot(x="Speaker_ID", y="File Count", hue="Group", data=per_speaker,
                ax=axes[1, 0], palette="magma", dodge=False)
    axes[1, 0].set_title("Total Audio Files per Speaker", fontsize=14)
    axes[1, 0].set_xlabel("Speaker ID")
    axes[1, 0].set_ylabel("Number of Audio Files")
    axes[1, 0].tick_params(axis="x", rotation=90)
    axes[1, 0].legend(title="Speaker Group", loc="upper right")

    # (1, 1) files per microphone channel
    mic_counts = df["Microphone_Channel"].value_counts().sort_index()
    sns.barplot(x=mic_counts.index, y=mic_counts.values, hue=mic_counts.index,
                ax=axes[1, 1], palette="cubehelix", legend=False)
    axes[1, 1].set_title("Distribution of Audio Files by Microphone Channel",
                         fontsize=14)
    axes[1, 1].set_xlabel("Microphone Channel")
    axes[1, 1].set_ylabel("Number of Audio Files")
    for container in axes[1, 1].containers:
        axes[1, 1].bar_label(container, fmt="%d")

    fig.tight_layout()
    return _finish(fig, "uaspeech_dashboard.png", show)


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


def run_eda(df: pd.DataFrame, show: bool = False) -> None:
    """Generate and save all EDA figures."""
    config.ensure_directories()
    print_header("Exploratory Data Analysis")
    print_kv("Files-per-speaker plot", plot_files_per_speaker(df, show=show))
    print_kv("Dataset dashboard", plot_dataset_dashboard(df, show=show))
