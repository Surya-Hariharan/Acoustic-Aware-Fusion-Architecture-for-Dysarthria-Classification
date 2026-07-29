"""
Publication-facing results: aggregation, statistical comparison, and export.

Where notebooks/03_training.ipynb produces one run's numbers and
notebooks/04/05's modules interpret one model at a time, this module answers
the questions that only make sense across every experiment at once: which
runs actually beat which others (not just by pooled metric, but by a paired
significance test across folds), a multi-model ROC/PR comparison, and a
single export folder to hand off when writing the paper.

Consumes what src/training/reporting.py already writes per run — nothing
here recomputes a metric, it only aggregates, tests, and reformats:
  outputs/metrics/<run>.summary.csv     mean +/- std per run (aggregate_fold_metrics)
  outputs/metrics/<run>.per_fold.csv    per-fold metrics, for paired testing
  outputs/predictions/<run>/*.csv       per-utterance predictions, for ROC/PR
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_curve)

from src import config
from src.console import print_kv
from src.error_analysis import load_run_predictions


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def load_all_experiment_summaries() -> pd.DataFrame:
    """
    Every trained run's pooled mean metrics in one table, keyed by run_name —
    reads the *.summary.csv files aggregate_fold_metrics() already writes per
    run rather than recomputing anything. Skips the pipeline sanity-check run
    ("_smoke_test"), which is not a real result.
    """
    rows = []
    for path in sorted(config.METRICS_DIR.glob("*.summary.csv")):
        run_name = path.stem
        if run_name.endswith("smoke_test"):
            continue
        summary = pd.read_csv(path, index_col=0)
        if "mean" not in summary.columns:
            continue
        row = summary["mean"].to_dict()
        row["run_name"] = run_name
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No experiment summaries found in {config.METRICS_DIR} — "
            "train at least one model first (src.training.runner.run_training, "
            "see notebooks/03_training.ipynb)."
        )
    return pd.DataFrame(rows).set_index("run_name").sort_index()


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------
def compare_models_statistically(run_a: str, run_b: str, metric: str = "f1") -> Dict:
    """
    Paired Wilcoxon signed-rank test on one metric, matched fold-by-fold
    between two runs (outputs/metrics/<run>.per_fold.csv, written by
    aggregate_fold_metrics for every run). Answers "is run_b actually better
    than run_a, or within LOSO fold noise?" — a question the pooled
    comparison table (phase2_comparison.csv) cannot answer by itself.

    Both runs must share the same fold protocol (both detection or both
    severity) for the pairing to be meaningful — folds are matched by ID.
    """
    path_a = config.METRICS_DIR / f"{run_a}.per_fold.csv"
    path_b = config.METRICS_DIR / f"{run_b}.per_fold.csv"
    if not path_a.exists() or not path_b.exists():
        raise FileNotFoundError(
            f"Missing per-fold metrics for '{run_a}' or '{run_b}' — both runs "
            "must be trained (src.training.runner.run_training) first."
        )

    df_a = pd.read_csv(path_a)[["fold", metric]].rename(columns={metric: "a"})
    df_b = pd.read_csv(path_b)[["fold", metric]].rename(columns={metric: "b"})
    merged = df_a.merge(df_b, on="fold", how="inner").dropna()
    if len(merged) < 2:
        raise ValueError(
            f"Only {len(merged)} matching fold(s) between '{run_a}' and '{run_b}' "
            "— need at least 2 for a paired test. Are they the same task/protocol?"
        )

    statistic, p_value = wilcoxon(merged["a"], merged["b"])
    return {
        "run_a": run_a, "run_b": run_b, "metric": metric,
        "n_folds": len(merged),
        "mean_a": float(merged["a"].mean()), "mean_b": float(merged["b"].mean()),
        "statistic": float(statistic), "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


# ---------------------------------------------------------------------------
# ROC / PR comparison
# ---------------------------------------------------------------------------
def plot_roc_pr_comparison(run_names: List[str], task: str = "detection",
                           show: bool = False) -> Dict[str, str]:
    """
    Overlay ROC and precision-recall curves for several runs, one figure
    each. src.training.reporting.save_roc_curve draws a single run's curve
    per fold/pooled call; this is the publication-facing, multi-model
    counterpart, plus the PR curve that has no equivalent anywhere yet.

    Detection only — a single positive-class probability column
    (`prob_positive`) is what makes one shared figure meaningful; severity's
    one-vs-rest curves are better as one small multiple per class.
    """
    if task != "detection":
        raise ValueError("plot_roc_pr_comparison currently supports task='detection' only.")

    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(6, 6))

    for run_name in run_names:
        preds = load_run_predictions(run_name)
        y_true = preds["y_true"].to_numpy()
        y_prob = preds["prob_positive"].to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_roc.plot(fpr, tpr, label=f"{run_name} (AUC={auc(fpr, tpr):.3f})")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax_pr.plot(recall, precision,
                  label=f"{run_name} (AP={average_precision_score(y_true, y_prob):.3f})")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC — model comparison")
    ax_roc.legend(loc="lower right", fontsize=9)
    fig_roc.tight_layout()

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall — model comparison")
    ax_pr.legend(loc="lower left", fontsize=9)
    fig_pr.tight_layout()

    roc_path = config.FIGURE_DIR / "roc_comparison.png"
    pr_path = config.FIGURE_DIR / "pr_comparison.png"
    fig_roc.savefig(roc_path, dpi=300, bbox_inches="tight")
    fig_pr.savefig(pr_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig_roc)
        plt.close(fig_pr)
    return {"roc": str(roc_path), "pr": str(pr_path)}


# ---------------------------------------------------------------------------
# Publication styling and export
# ---------------------------------------------------------------------------
def set_publication_style() -> None:
    """
    Consistent IEEE-draft-friendly styling (serif fonts, restrained grid,
    no top/right spines), applied once at this notebook's top so every
    figure it produces looks like one system — and so that styling stays
    scoped to this notebook rather than silently changing every other
    notebook's plots via a shared global rcParams mutation.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def export_results_for_paper(run_names: Optional[List[str]] = None,
                             out_dir: Optional[Path] = None) -> Path:
    """
    Copy the key comparison tables and figures into outputs/paper_exports/
    and write a LaTeX snippet alongside each table — the single folder to
    hand off when writing the paper, instead of hunting across
    outputs/metrics/ and outputs/figures/ for what's current.

    run_names, if given, also pulls each run's pooled confusion matrix into
    the export.
    """
    out_dir = Path(out_dir or config.OUTPUT_DIR / "paper_exports")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    exported_tables = []
    for name in ("phase2_comparison.csv", "phase3_severity_comparison.csv"):
        src_path = config.METRICS_DIR / name
        if not src_path.exists():
            continue
        df = pd.read_csv(src_path, index_col=0)
        shutil.copy2(src_path, out_dir / "tables" / name)
        tex_path = out_dir / "tables" / name.replace(".csv", ".tex")
        tex_path.write_text(df.to_latex(float_format="%.4f"))
        exported_tables.append(name)

    exported_figures = []
    for pattern in ("roc_comparison.png", "pr_comparison.png", "ablation_comparison.png",
                    "embedding_map_*.png", "shap_feature_importance.png",
                    "attention_map_*.png", "praat_feature_correlation.png"):
        for fig_path in config.FIGURE_DIR.glob(pattern):
            shutil.copy2(fig_path, out_dir / "figures" / fig_path.name)
            exported_figures.append(fig_path.name)

    for run_name in (run_names or []):
        cm_path = config.CONFUSION_MATRIX_DIR / run_name / "ALL_FOLDS_pooled.png"
        if cm_path.exists():
            dest_name = f"confusion_matrix_{run_name}.png"
            shutil.copy2(cm_path, out_dir / "figures" / dest_name)
            exported_figures.append(dest_name)

    print_kv("Tables exported", f"{len(exported_tables)} -> {out_dir / 'tables'}")
    print_kv("Figures exported", f"{len(exported_figures)} -> {out_dir / 'figures'}")
    return out_dir
