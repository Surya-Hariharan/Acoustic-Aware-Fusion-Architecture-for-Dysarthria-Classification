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
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_curve)

from src import config
from src.console import print_kv
from src.error_analysis import load_run_predictions
from src.style import apply_style, color_for_run


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


def style_comparison_table(df: pd.DataFrame, higher_is_better: bool = True,
                           cmap: str = "RdYlGn", decimals: int = 4):
    """
    A benchmark comparison table (phase2_comparison.csv,
    phase3_severity_comparison.csv, load_all_experiment_summaries()'s output,
    ...) as a per-column colour-graded pandas Styler — red-to-green heat per
    metric column plus the best value per column bolded — instead of a flat
    grid of decimals the reader has to scan by eye to find the winner.
    Notebook display only: this returns a Styler for `display()`/bare
    notebook-cell output, not a DataFrame — call `.data` on the result (or
    just keep using the original df) for anything that needs to be
    machine-read, exported to CSV, or compared programmatically.

    higher_is_better=True (the default) colours the highest value in each
    column green — true for every metric this project reports (accuracy,
    precision, recall, specificity, f1, auroc). Pass False for a metric
    where lower is better (e.g. test_loss) so the colour scale doesn't
    imply the opposite of what the number means.
    """
    numeric_cols = df.select_dtypes("number").columns
    effective_cmap = cmap if higher_is_better else f"{cmap}_r"

    return (df.style
           .format({c: f"{{:.{decimals}f}}" for c in numeric_cols})
           .background_gradient(cmap=effective_cmap, subset=numeric_cols, axis=0)
           .highlight_max(subset=numeric_cols, axis=0,
                          props="font-weight: bold; text-decoration: underline;"
                          if higher_is_better else "")
           .highlight_min(subset=numeric_cols, axis=0,
                          props="font-weight: bold; text-decoration: underline;"
                          if not higher_is_better else ""))


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


def mcnemar_test(run_a: str, run_b: str) -> Dict:
    """
    Exact McNemar test on paired per-utterance correctness between two runs —
    answers "of the utterances the two models disagree on, is one model right
    more often than the other?", which a difference in pooled accuracy alone
    cannot answer (a tie in overall accuracy can still hide one model being
    systematically better on a specific subset the other gets wrong).

    Loads outputs/predictions/<run>/*.csv via src.error_analysis.load_run_predictions
    for both runs (already written by every run — no training-pipeline change
    needed) and inner-joins on `filename`, so only utterances both runs were
    actually evaluated on are compared. Both runs must share the same held-out
    utterances for the pairing to be meaningful (true for any two of the primary
    detection sweep's variants, since all six use the same LOSO folds).

    Uses the exact binomial form on the discordant pairs (scipy.stats.binomtest)
    rather than the chi-square approximation — correct at any discordant-pair
    count, including the small ones typical of a single LOSO fold's test split,
    where the chi-square approximation is unreliable. No new dependency:
    statsmodels' contingency-table implementation is not used since scipy
    (already a project dependency) covers the same exact test directly.
    """
    from src.error_analysis import load_run_predictions

    preds_a = load_run_predictions(run_a)[["filename", "correct"]].rename(
        columns={"correct": "correct_a"})
    preds_b = load_run_predictions(run_b)[["filename", "correct"]].rename(
        columns={"correct": "correct_b"})
    merged = preds_a.merge(preds_b, on="filename", how="inner")
    if len(merged) == 0:
        raise ValueError(
            f"No overlapping utterances between '{run_a}' and '{run_b}' — "
            "they must be trained on the same fold protocol/dataset."
        )

    # Discordant pairs: utterances exactly one of the two models got right.
    a_only = int(((merged["correct_a"]) & (~merged["correct_b"])).sum())
    b_only = int(((~merged["correct_a"]) & (merged["correct_b"])).sum())
    n_discordant = a_only + b_only

    if n_discordant == 0:
        p_value = 1.0
    else:
        # Under H0 (the two models are equally likely to be the one that's
        # right on a discordant pair), a_only ~ Binomial(n_discordant, 0.5).
        p_value = float(binomtest(a_only, n_discordant, 0.5, alternative="two-sided").pvalue)

    return {
        "run_a": run_a, "run_b": run_b, "n_utterances": len(merged),
        "a_correct_b_wrong": a_only, "b_correct_a_wrong": b_only,
        "n_discordant": n_discordant, "p_value": p_value,
        "significant": bool(p_value < 0.05),
    }


def bootstrap_ci(run_name: str, metric: str = "f1", n_boot: int = 2000,
                 ci: float = 0.95, seed: int = 42) -> Dict:
    """
    Percentile bootstrap confidence interval for one run's pooled metric,
    resampled at the FOLD level (not raw utterance level) from
    outputs/metrics/<run>.per_fold.csv.

    Folds, not utterances, are this project's unit of statistical independence
    — src.results.compare_models_statistically already relies on this for the
    paired Wilcoxon test, for the same reason: utterances from the same LOSO
    fold (and often the same speaker) are not independent draws, so resampling
    individual utterances would understate the true uncertainty. Resampling
    whole folds with replacement and recomputing the mean each time gives a CI
    that respects that structure.
    """
    path = config.METRICS_DIR / f"{run_name}.per_fold.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing per-fold metrics for '{run_name}' — train it first "
            "(src.training.runner.run_training)."
        )
    values = pd.read_csv(path)[metric].dropna().to_numpy()
    if len(values) < 2:
        raise ValueError(f"Only {len(values)} fold(s) with a valid '{metric}' — "
                         "need at least 2 to bootstrap.")

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means[i] = sample.mean()

    alpha = 1 - ci
    lower, upper = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "run_name": run_name, "metric": metric, "n_folds": len(values),
        "n_boot": n_boot, "ci": ci,
        "mean": float(values.mean()),
        "ci_lower": float(lower), "ci_upper": float(upper),
    }


# ---------------------------------------------------------------------------
# Ablation gain table
# ---------------------------------------------------------------------------
def compute_ablation_gains(comparison_df: pd.DataFrame,
                           metrics: List[str] = ("accuracy", "f1", "auroc")) -> pd.DataFrame:
    """
    The five deltas the primary ablation matrix exists to answer, computed
    from the six-variant comparison table (index = model name: acoustic,
    deep_frozen, deep_lora, fusion_frozen, fusion, attention_fusion — see
    notebooks/03_training.ipynb Stage 8e):

        fusion_over_mfcc            fusion         - acoustic        (RQ3)
        fusion_over_wav2vec         fusion         - deep_lora       (RQ3)
        lora_over_frozen            deep_lora      - deep_frozen     (RQ4, standalone)
        lora_over_frozen_in_fusion  fusion         - fusion_frozen   (RQ4, inside fusion)
        proposed_over_fusion        attention_fusion - fusion        (proposed-model gain)

    Reports both the absolute gain in percentage points and the relative gain
    as a percentage of the baseline, for each requested metric (values are
    assumed to be fractions in [0, 1], matching src.training.metrics.compute_metrics'
    output). A comparison whose model or baseline row is missing from
    comparison_df is skipped (not raised) — useful while the primary sweep is
    still partially trained.
    """
    comparisons = {
        "fusion_over_mfcc": ("fusion", "acoustic"),
        "fusion_over_wav2vec": ("fusion", "deep_lora"),
        "lora_over_frozen": ("deep_lora", "deep_frozen"),
        "lora_over_frozen_in_fusion": ("fusion", "fusion_frozen"),
        "proposed_over_fusion": ("attention_fusion", "fusion"),
    }
    rows = []
    for label, (better, worse) in comparisons.items():
        if better not in comparison_df.index or worse not in comparison_df.index:
            print_kv("Ablation gain skipped", f"{label} — '{better}' or '{worse}' not yet in the table")
            continue
        row = {"comparison": label, "model": better, "baseline": worse}
        for metric in metrics:
            if metric not in comparison_df.columns:
                continue
            better_val, worse_val = comparison_df.loc[better, metric], comparison_df.loc[worse, metric]
            row[f"{metric}_abs_gain_pp"] = 100 * (better_val - worse_val)
            row[f"{metric}_rel_gain_pct"] = (
                100 * (better_val - worse_val) / worse_val if worse_val else float("nan"))
        rows.append(row)

    return pd.DataFrame(rows)


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
        color = color_for_run(run_name)
        preds = load_run_predictions(run_name)
        y_true = preds["y_true"].to_numpy()
        y_prob = preds["prob_positive"].to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_roc.plot(fpr, tpr, color=color, label=f"{run_name} (AUC={auc(fpr, tpr):.3f})")

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax_pr.plot(recall, precision, color=color,
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
    src.style.apply_style()'s baseline (the same one every other figure in
    the project now uses — see src/style.py), plus a serif override and a
    higher screen DPI for the final paper-facing figures this notebook
    produces. Call this last, after any other module's plotting functions
    have already run in the same session, since it overrides on top of
    apply_style() rather than replacing it.
    """
    apply_style()
    plt.rcParams.update({
        "font.family": "serif",
        "figure.dpi": 150,
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
