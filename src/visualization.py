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
