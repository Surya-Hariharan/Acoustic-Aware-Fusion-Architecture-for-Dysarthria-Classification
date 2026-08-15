"""
Shared visual style: one color system and rcParams baseline so figures across
src/visualization.py, src/model_analysis.py, src/error_analysis.py, and
src/results.py read as one coherent report instead of four modules each
picking their own colormap.

apply_style() replaces each module's own `sns.set_style("whitegrid")` call —
same import-time convention they already used, just centralized. It is
idempotent, so multiple modules calling it in one notebook session is safe.

MODEL_COLORS is the point of this module: a fixed color per ablation variant
(src.training.models.MODEL_NAMES), so "attention_fusion" is the same color in
the ablation bar chart, the ROC overlay, and the SHAP comparison — instead of
each figure's default color cycle assigning it whatever slot it happens to
land in.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# One color per ablation variant. Chosen from the same qualitative family
# already used ad hoc across the codebase (seaborn's "deep" palette), just
# fixed to a name instead of a plot-order position.
MODEL_COLORS = {
    "acoustic": "#4c72b0",                 # Model A — MFCC 1D-CNN
    "deep_frozen": "#dd8452",              # Model B — frozen wav2vec2
    "deep_lora": "#55a868",                # Model C — wav2vec2 + LoRA
    "fusion_frozen": "#64b5cd",            # LoRA-off Model D (frozen-fusion ablation)
    "fusion": "#c44e52",                   # Model D — concatenation
    "attention_fusion": "#8172b3",         # Model E — cross-attention
    "attention_fusion_praat": "#937860",   # Model F — cross-attention + Praat
    "baseline_svm": "#8c8c8c",             # Phase 2 base-paper reproduction
}

CORRECT_COLOR = "#55a868"
ERROR_COLOR = "#c44e52"

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "magma"


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


def model_color(model_name: str) -> str:
    """MODEL_COLORS[model_name], or a neutral fallback for anything unrecognized
    (e.g. 'baseline_svm' variants not in the fixed set)."""
    return MODEL_COLORS.get(model_name, "#4c72b0")


def color_for_run(run_name: str) -> str:
    """Resolve a run name like 'detection_attention_fusion' or
    'severity_fusion' to its model's fixed color by matching the longest
    known model name that appears in it — longest first, since
    'attention_fusion' is itself a substring-free name but 'fusion' alone
    would otherwise match 'attention_fusion' runs too."""
    for name in sorted(MODEL_COLORS, key=len, reverse=True):
        if name in run_name:
            return MODEL_COLORS[name]
    return "#4c72b0"
