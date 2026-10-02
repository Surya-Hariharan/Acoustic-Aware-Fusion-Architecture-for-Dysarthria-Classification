"""
Shared visual style for every figure (src/eda.py and the notebook): one
rcParams baseline and the fixed categorical palettes of the speech-processing
EDA panels.
"""

import matplotlib.pyplot as plt
import seaborn as sns

CORRECT_COLOR = "#55a868"
ERROR_COLOR = "#c44e52"

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "magma"

# Three-way silence/unvoiced/voiced segmentation (src/eda.py) — one fixed
# categorical triplet used identically everywhere this segmentation is
# shaded (waveform, spectrogram overlay, energy/ZCR panels), so a reader's
# color intuition (silence = neutral gray, unvoiced = warm/noisy, voiced =
# cool/periodic) carries across every EDA panel it appears in.
VOICING_COLORS = {
    "silence": "#8c8c8c",
    "unvoiced": "#dd8452",
    "voiced": "#4c72b0",
}

# Fixed qualitative triplet for F1/F2/F3 formant tracks — consistent across
# every panel a formant is drawn on (src/eda.py).
FORMANT_COLORS = {
    "F1": "#c44e52",
    "F2": "#ccb974",
    "F3": "#8172b3",
}

# Ordinal-aware sequential palette for anything grouped by severity
# (Very Low -> High) — sequential rather than qualitative, since severity is
# an ordered label (mirrors the CORAL head's own ordinal-regression framing
# in src.models.gated_fusion.CoralHead).
SEVERITY_SEQUENTIAL_CMAP = "viridis"


def apply_style() -> None:
    """Whitegrid + a restrained, presentation-ready rcParams baseline. Safe to
    call from multiple modules — later calls are idempotent no-ops in effect."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
    })
