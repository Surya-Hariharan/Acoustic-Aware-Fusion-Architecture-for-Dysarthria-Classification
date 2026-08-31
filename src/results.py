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
# Run-name prefixes that are diagnostics, not experiments. Their outputs are
# structurally identical to a real run's — same summary/per-fold/prediction
# files — so nothing downstream can tell them apart by inspection. They must be
# excluded by name.
#   _smoke_test               pipeline sanity check (1 fold, 1 epoch, 24 samples)
#   _budget_bench_            ExperimentBudgetManager.benchmark (1 fold, 1 epoch)
#   _batch_bench_             benchmark_batch_sizes (1 fold, 1 epoch per size)
# Before this filter existed, the seven _budget_bench_* runs appeared in the
# results table beside real experiments with nothing marking them.
NON_EXPERIMENT_PREFIXES = ("_smoke_test", "_budget_bench_", "_batch_bench_")


def is_non_experiment(run_name: str) -> bool:
    """True for diagnostic/benchmark runs that must never reach a results table."""
    return (run_name.startswith(NON_EXPERIMENT_PREFIXES)
            or run_name.endswith("smoke_test"))


def load_all_experiment_summaries(include_diagnostics: bool = False,
                                  with_status: bool = True) -> pd.DataFrame:
    """
    Every trained run's per-fold mean metrics in one table, keyed by run_name —
    reads the *.summary.csv files aggregate_fold_metrics() already writes per
    run rather than recomputing anything.

    Diagnostic runs (see NON_EXPERIMENT_PREFIXES) are excluded unless
    include_diagnostics=True. When the experiment registry is populated
    (src.training.reporting), each row is additionally annotated with
    completed/expected folds, coverage and status — so a reader can see at a
    glance that a run is PARTIAL rather than inferring completeness from the
    mere presence of a number.

    Note these are PER-FOLD MEANS, not pooled metrics. For detection LOSO the
    pooled numbers (outputs/metrics/<run>/ALL_FOLDS_pooled.json) are the
    reportable ones; per-fold class-sensitive metrics are NaN on single-class
    folds by design and are skipped by the mean.
    """
    from src.training.reporting import summarize_registry

    rows = []
    for path in sorted(config.METRICS_DIR.glob("*.summary.csv")):
        # Path("detection_fusion.summary.csv").stem is "detection_fusion.summary"
        # — the double extension leaves ".summary" attached. Left unstripped, the
        # run_name never matches the registry's, so every row silently joined to
        # NaN and reported as UNREGISTERED regardless of its real status.
        run_name = path.name[:-len(".summary.csv")]
        if not include_diagnostics and is_non_experiment(run_name):
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
    table = pd.DataFrame(rows).set_index("run_name").sort_index()

    if with_status:
        registry = summarize_registry()
        if not registry.empty:
            status_cols = ["completed_folds", "expected_folds", "coverage",
                           "pooled_has_both_classes", "status"]
            table = table.join(registry.set_index("run_name")[status_cols], how="left")
            # Runs that predate the registry (every run in the pre-repair
            # session) legitimately have no status — say "UNREGISTERED" rather
            # than leaving a bare NaN that reads as a missing measurement.
            table["status"] = table["status"].fillna("UNREGISTERED")
    return table


# ---------------------------------------------------------------------------
# Final-result eligibility gate
#
# "The notebook finished" and "the experiment is valid" are different states.
# Everything below exists to keep them apart. A run reaches the FINAL table
# only by passing every check; anything else is routed to PRELIMINARY (real but
# incomplete) or EXCLUDED (cannot support a claim), always with a stated reason.
# Nothing is deleted and nothing is hidden — the brief requires incomplete work
# to remain visible, just not to masquerade as a result.
# ---------------------------------------------------------------------------
TIER_FINAL = "FINAL"
TIER_PRELIMINARY = "PRELIMINARY"
TIER_EXCLUDED = "EXCLUDED"


def classify_result(run_row: pd.Series, min_coverage: float = 1.0,
                    min_valid_folds: int = 1) -> Dict:
    """
    Sort one registry rollup row into FINAL / PRELIMINARY / EXCLUDED, with the
    reason that decided it.

    The checks, in the order they disqualify:
      1. diagnostic run                    -> EXCLUDED (benchmark, not an experiment)
      2. no completed folds                -> EXCLUDED (never ran)
      3. pooled set is single-class        -> EXCLUDED for detection; the pooled
                                              metrics are undefined, so the run
                                              carries no evidence whatever its
                                              accuracy reads
      4. coverage below min_coverage       -> PRELIMINARY (real, but incomplete)
      5. no fold with >1 held-out class    -> PRELIMINARY, WITH ONE EXCEPTION
                                              (see below)

    THE LOSO EXCEPTION (check 5)
    A detection LOSO fold holds out exactly one speaker, so it is single-class
    BY DESIGN — that is the base paper's own protocol, not a defect. Every
    complete 28-fold LOSO run would therefore have valid_folds == 0 and, absent
    this exception, would be stuck at PRELIMINARY forever regardless of
    coverage. Check 5 is skipped for task == "detection" and
    cv_protocol == "loso"; check 3 (pooled_has_both_classes) already covers the
    thing that actually matters for that protocol — whether the fold-by-fold
    single-class results pool into a real evaluation. For every other protocol
    (screening, severity's leave-one-per-class-out) multi-class folds are the
    normal, expected outcome, so an all-single-class result there is still
    treated as a sign something is wrong with the fold construction.

    min_coverage defaults to 1.0: a FINAL result must have reached every fold
    of its intended protocol. Lower it only deliberately, and say so in the
    write-up.
    """
    run_name = run_row["run_name"]
    task = run_row.get("task", "detection")
    cv_protocol = run_row.get("cv_protocol", "")
    completed = int(run_row.get("completed_folds", 0) or 0)
    expected = int(run_row.get("expected_folds", 0) or 0)
    coverage = float(run_row.get("coverage", 0.0) or 0.0)
    valid_folds = int(run_row.get("valid_folds", 0) or 0)
    both_classes = bool(run_row.get("pooled_has_both_classes", False))
    is_loso_detection = (task == "detection" and cv_protocol == "loso")

    def verdict(tier, reason):
        return {"run_name": run_name, "task": task, "tier": tier, "reason": reason,
                "completed_folds": completed, "expected_folds": expected,
                "coverage": coverage, "valid_folds": valid_folds,
                "pooled_has_both_classes": both_classes,
                "status": run_row.get("status", "UNKNOWN")}

    if is_non_experiment(run_name):
        return verdict(TIER_EXCLUDED, "Diagnostic/benchmark run, not an experiment")
    if completed == 0:
        return verdict(TIER_EXCLUDED, "Never executed — no completed folds")
    if task == "detection" and not both_classes:
        return verdict(
            TIER_EXCLUDED,
            f"Pooled held-out set is single-class across all {completed} completed "
            "fold(s) — precision/recall/F1/AUROC are undefined, so this run carries "
            "no evidence about detection")
    if coverage < min_coverage:
        return verdict(
            TIER_PRELIMINARY,
            f"Incomplete coverage: {completed}/{expected} folds ({coverage:.0%})")
    if not is_loso_detection and valid_folds < min_valid_folds:
        return verdict(
            TIER_PRELIMINARY,
            "Complete coverage but no individual fold held out more than one class — "
            "only pooled metrics are interpretable")
    return verdict(TIER_FINAL,
                   f"Complete: {completed}/{expected} folds, both classes represented"
                   + (" (LOSO: single-class per fold by design, pooled set validated)"
                      if is_loso_detection else ""))


def build_result_tiers(min_coverage: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    Every registered run sorted into the three reporting tiers.

    Returns {"final": df, "preliminary": df, "excluded": df}. Notebook 6 renders
    these as its three tables; the FINAL one is the only one a claim may rest on.
    """
    from src.training.reporting import summarize_registry

    registry = summarize_registry()
    if registry.empty:
        empty = pd.DataFrame(columns=["run_name", "task", "tier", "reason",
                                      "completed_folds", "expected_folds", "coverage",
                                      "valid_folds", "pooled_has_both_classes", "status"])
        return {"final": empty.copy(), "preliminary": empty.copy(), "excluded": empty.copy()}

    verdicts = pd.DataFrame([classify_result(row, min_coverage)
                             for _, row in registry.iterrows()])
    return {
        "final": verdicts[verdicts["tier"] == TIER_FINAL].reset_index(drop=True),
        "preliminary": verdicts[verdicts["tier"] == TIER_PRELIMINARY].reset_index(drop=True),
        "excluded": verdicts[verdicts["tier"] == TIER_EXCLUDED].reset_index(drop=True),
    }


def select_analysis_run(task: str = "detection", metric: str = "f1",
                        preferred: Optional[str] = None,
                        allow_preliminary: bool = True) -> Optional[str]:
    """
    Pick which run notebooks 4 and 5 should analyse, from the registry rather
    than from a hardcoded name.

    Both notebooks previously opened with `RUN_NAME = "detection_fusion"` — a
    run that has never existed in this project — so both raised on their first
    cell. Worse, hardcoding invites analysing whichever run the string happens
    to name regardless of whether it is valid.

    Preference order: `preferred` if it is registered and eligible → the best
    FINAL run by `metric` → the best PRELIMINARY run (only when
    allow_preliminary, since error analysis on an incomplete run is still
    informative, unlike *reporting* it) → None.

    Returns None rather than raising when nothing qualifies: an analysis
    notebook should say "nothing valid to analyse yet" and continue, not die.
    """
    tiers = build_result_tiers()
    eligible = tiers["final"]
    if allow_preliminary and not tiers["preliminary"].empty:
        eligible = pd.concat([eligible, tiers["preliminary"]], ignore_index=True)
    eligible = eligible[eligible["task"] == task]
    if eligible.empty:
        return None

    if preferred and preferred in set(eligible["run_name"]):
        return preferred

    scored = []
    for run_name in eligible["run_name"]:
        try:
            scored.append((load_pooled_metrics(run_name).get(metric), run_name))
        except FileNotFoundError:
            continue
    scored = [(value, name) for value, name in scored
              if value is not None and not pd.isna(value)]
    if not scored:
        # Nothing has a usable pooled metric — fall back to the most complete
        # run so the notebook still has something to introspect.
        return eligible.sort_values("coverage", ascending=False).iloc[0]["run_name"]
    return max(scored)[1]


def load_pooled_metrics(run_name: str) -> Dict:
    """One run's pooled-across-folds metrics, as written by run_training.

    These — not the per-fold means in load_all_experiment_summaries() — are the
    base-paper-comparable detection numbers, because pooling is what restores a
    positive class to a set of single-class LOSO folds.
    """
    path = config.METRICS_DIR / run_name / "ALL_FOLDS_pooled.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No pooled metrics for '{run_name}' at {path} — the run either never "
            "completed a fold or predates the pooled writer.")
    import json
    with open(path) as f:
        return json.load(f)


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
# Final reporting gate — tables, manifest, figures, summary
# ---------------------------------------------------------------------------
def build_final_results_table(min_coverage: float = 1.0) -> pd.DataFrame:
    """
    The FINAL results table: one row per eligible run, pooled metrics joined to
    its coverage.

    Pooled metrics, not per-fold means — pooling across folds is what restores a
    positive class to a set of single-class LOSO folds, so it is the only
    base-paper-comparable number for detection. Undefined metrics stay NaN and
    are rendered "N/A" on export; they are never coerced to zero.
    """
    tiers = build_result_tiers(min_coverage)
    rows = []
    for _, verdict in tiers["final"].iterrows():
        run_name = verdict["run_name"]
        try:
            pooled = load_pooled_metrics(run_name)
        except FileNotFoundError:
            continue
        rows.append({
            "run_name": run_name, "task": verdict["task"],
            "expected_folds": verdict["expected_folds"],
            "completed_folds": verdict["completed_folds"],
            "valid_folds": verdict["valid_folds"],
            "coverage": verdict["coverage"],
            "accuracy": pooled.get("accuracy"), "precision": pooled.get("precision"),
            "recall": pooled.get("recall"), "specificity": pooled.get("specificity"),
            "f1": pooled.get("f1"), "auroc": pooled.get("auroc"),
            "status": verdict["status"],
        })
    if not rows:
        return pd.DataFrame(columns=[
            "run_name", "task", "expected_folds", "completed_folds", "valid_folds",
            "coverage", "accuracy", "precision", "recall", "specificity", "f1",
            "auroc", "status"])
    return pd.DataFrame(rows).sort_values(["task", "f1"], ascending=[True, False])


def build_preliminary_results_table(min_coverage: float = 1.0) -> pd.DataFrame:
    """Real but incomplete runs, each carrying its coverage and the reason it
    is not FINAL. Kept visible — the brief requires incomplete work to be shown,
    just never presented as a finished result."""
    tiers = build_result_tiers(min_coverage)
    rows = []
    for _, verdict in tiers["preliminary"].iterrows():
        run_name = verdict["run_name"]
        try:
            pooled = load_pooled_metrics(run_name)
        except FileNotFoundError:
            pooled = {}
        rows.append({
            "run_name": run_name, "task": verdict["task"],
            "completed_folds": verdict["completed_folds"],
            "expected_folds": verdict["expected_folds"],
            "coverage": verdict["coverage"],
            "accuracy": pooled.get("accuracy"), "f1": pooled.get("f1"),
            "auroc": pooled.get("auroc"),
            "caveat": "PRELIMINARY — NOT FOR FINAL CLAIMS",
            "reason": verdict["reason"],
        })
    return pd.DataFrame(rows)


def build_experiment_manifest() -> pd.DataFrame:
    """
    Reproducibility manifest: everything needed to regenerate a reported result.

    Per registered run — model, task, protocol, seed, epochs, LR, batch size,
    coverage — joined with the dataset/preprocessing/feature configuration that
    lives as module constants in src/config.py rather than per-run fields. Run
    config comes from the experiment bundle's config.json where
    save_experiment_bundle wrote one; the rest falls back to the registry.
    """
    import json
    from src.training.reporting import summarize_registry

    registry = summarize_registry()
    if registry.empty:
        return pd.DataFrame()

    shared = {
        "dataset": "UA-Speech (M6 channel)",
        "target_sr": config.TARGET_SR,
        "clip_seconds": config.CLIP_SECONDS,
        "vad_enabled": config.VAD_ENABLED,
        "vad_threshold": config.VAD_THRESHOLD,
        "n_mfcc": config.N_MFCC,
        "mfcc_n_fft": config.MEL_KWARGS["n_fft"],
        "mfcc_hop_length": config.MEL_KWARGS["hop_length"],
        "mfcc_n_mels": config.MEL_KWARGS["n_mels"],
        "wav2vec2_model": config.WAV2VEC_MODEL_NAME,
        "lora_rank": config.LORA_RANK,
        "lora_alpha": config.LORA_ALPHA,
    }

    rows = []
    for _, run in registry.iterrows():
        run_config = {}
        for bundle in config.EXPERIMENTS_DIR.glob("*/config.json"):
            try:
                with open(bundle) as f:
                    candidate = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if candidate.get("run_name") == run["run_name"]:
                run_config = candidate
                break
        rows.append({
            "experiment_id": run["run_name"],
            "model": run["model"], "task": run["task"],
            "fold_protocol": run["cv_protocol"],
            "expected_folds": run["expected_folds"],
            "completed_folds": run["completed_folds"],
            "coverage": run["coverage"],
            "status": run["status"],
            "seed": run_config.get("seed", config.DEFAULT_SEED),
            "epochs": run_config.get("epochs"),
            "batch_size": run_config.get("batch_size"),
            "lr_head": run_config.get("lr_head"),
            "lr_backbone": run_config.get("lr_backbone"),
            "checkpoint_dir": str(config.CHECKPOINT_DIR / run["run_name"]),
            **shared,
        })
    return pd.DataFrame(rows)


def plot_fold_coverage(show: bool = False) -> Optional[str]:
    """
    Stacked bar of completed / failed / skipped folds per run.

    The point of this figure is that incomplete experiments become impossible to
    overlook. A leaderboard row and a coverage bar sit in the same report, so a
    reader cannot see "99.7% accuracy" without also seeing "1 of 28 folds".
    """
    from src.training.reporting import summarize_registry

    registry = summarize_registry()
    if registry.empty:
        return None

    registry = registry.sort_values(["task", "run_name"])
    labels = registry["run_name"].tolist()
    completed = registry["completed_folds"].to_numpy()
    failed = registry["failed_folds"].to_numpy()
    expected = registry["expected_folds"].to_numpy()
    remaining = np.maximum(expected - completed - failed, 0)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    ax.barh(y, completed, color="#2C6249", label="completed")
    ax.barh(y, failed, left=completed, color="#A62B22", label="failed")
    ax.barh(y, remaining, left=completed + failed, color="#D8DCE3",
            label="not run")

    for i, (done, total) in enumerate(zip(completed, expected)):
        ax.text(total + max(expected) * 0.01, i, f"{int(done)}/{int(total)}",
                va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Folds")
    ax.set_title("Evaluation coverage per experiment")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = config.RESULTS_DIR / "fold_coverage.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out_path)


def plot_baseline_comparison(reproduction_accuracy: Optional[float] = None,
                             proposed: Optional[Dict[str, float]] = None,
                             paper_accuracy: float = 0.9395,
                             task: str = "detection",
                             show: bool = False) -> Optional[str]:
    """
    Paper-reported vs. our reproduction vs. proposed models, on one axis with
    the three provenances kept visually distinct.

    The paper's number is a CITATION, not something measured here, and must
    never be shaded as though it came out of this pipeline — mixing them is how
    a reproduction gap turns into an accidental claim. Bars are labelled and
    coloured by provenance for exactly that reason.
    """
    entries, colors = [], []
    entries.append(("Paper (reported)", paper_accuracy))
    colors.append("#9AA3B0")
    if reproduction_accuracy is not None:
        entries.append(("Our reproduction", reproduction_accuracy))
        colors.append("#37516B")
    for name, value in (proposed or {}).items():
        if value is not None and not pd.isna(value):
            entries.append((name, value))
            colors.append("#0E7C86")

    if len(entries) < 2:
        return None

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(entries)), 4.5))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                f"{value:.3f}", ha="center", fontsize=9)

    ax.axhline(paper_accuracy, linestyle="--", linewidth=1, color="#9AA3B0")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Baseline comparison — {task}\n"
                 "grey = paper-reported · navy = reproduced here · teal = proposed",
                 fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    out_path = config.RESULTS_DIR / "baseline_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out_path)


def export_gated_results(min_coverage: float = 1.0) -> Dict[str, Path]:
    """
    Write the three gated tables plus the reproducibility manifest into
    outputs/results/. Undefined metrics are exported as the string "N/A", never
    as 0 — a downstream reader of the CSV must not be able to mistake
    "not measurable" for "measured as zero".
    """
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tiers = build_result_tiers(min_coverage)

    tables = {
        "final_results.csv": build_final_results_table(min_coverage),
        "preliminary_results.csv": build_preliminary_results_table(min_coverage),
        "excluded_results.csv": tiers["excluded"],
        "experiment_manifest.csv": build_experiment_manifest(),
    }

    written = {}
    for filename, table in tables.items():
        path = config.RESULTS_DIR / filename
        table.to_csv(path, index=False, na_rep="N/A")
        written[filename] = path
        print_kv(filename, f"{len(table)} row(s) -> {path}")
    return written


def print_experiment_summary(min_coverage: float = 1.0) -> None:
    """
    The end-of-notebook research summary: coverage per task, which models are
    eligible, which are not, and the explicit status of the two variants the
    project's central claim depends on.

    Every value is read from artifacts. A model that was never trained is
    reported as "not executed" rather than omitted, because an absent row is
    what let the attention-fusion variants disappear from the pre-repair report
    without anyone noticing.
    """
    from src.console import print_header, print_subheader
    from src.training.reporting import summarize_registry

    registry = summarize_registry()
    tiers = build_result_tiers(min_coverage)

    print_header("EXPERIMENT SUMMARY")

    if registry.empty:
        print_kv("Registry", "empty — no run has been executed since the registry "
                             "was added (pre-repair runs are unregistered)")
        return

    for task in sorted(registry["task"].unique()):
        subset = registry[registry["task"] == task]
        print_subheader(task.capitalize())
        print_kv("Expected folds (max across runs)", int(subset["expected_folds"].max()))
        print_kv("Completed folds (total)", int(subset["completed_folds"].sum()))
        print_kv("Runs with full coverage",
                 f"{int((subset['coverage'] >= min_coverage).sum())} / {len(subset)}")

    print_subheader("Eligibility")
    print_kv("FINAL", ", ".join(tiers["final"]["run_name"]) or "none")
    print_kv("PRELIMINARY", ", ".join(tiers["preliminary"]["run_name"]) or "none")
    print_kv("EXCLUDED", ", ".join(tiers["excluded"]["run_name"]) or "none")

    final_table = build_final_results_table(min_coverage)
    for task in ("detection", "severity"):
        subset = final_table[final_table["task"] == task]
        subset = subset[subset["f1"].notna()]
        print_kv(f"Best valid {task} model",
                 f"{subset.iloc[0]['run_name']} (F1={subset.iloc[0]['f1']:.4f})"
                 if len(subset) else "none — no run passed the eligibility gate")

    # The project's central contribution, called out by name. If these are not
    # in the registry at all, say so explicitly — silence reads as success.
    print_subheader("Central contribution")
    for model in ("attention_fusion", "attention_fusion_praat"):
        matches = registry[registry["model"] == model]
        if matches.empty:
            print_kv(model, "NOT EXECUTED — no registered run")
            continue
        row = matches.sort_values("coverage", ascending=False).iloc[0]
        print_kv(model, f"{row['status']} — {int(row['completed_folds'])}/"
                        f"{int(row['expected_folds'])} folds ({row['coverage']:.0%})")


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


# ---------------------------------------------------------------------------
# Three-branch severity architecture — paper-ready tables (Tables 1-8 + a
# limitations table). Every function here reads real, already-computed
# artifacts (the manifest, a run's pooled metrics/predictions, or a
# DataFrame the caller already produced via src.model_analysis) — nothing
# is hardcoded, so a table always reflects the actual run. Tables that
# depend on the trained checkpoint (branch ablation, gate contribution,
# explainability) take that DataFrame as an argument rather than
# recomputing it, keeping this module decoupled from src.model_analysis's
# heavier machinery (SHAP/UMAP/branch-ablation forward passes).
# ---------------------------------------------------------------------------
def build_dataset_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Table 1 — Speaker | Severity | Number of utterances, for the 15
    dysarthric speakers the primary severity protocol evaluates."""
    df = manifest[manifest["Speaker_ID"].isin(config.DYSARTHRIC_IDS)]
    table = (df.groupby(["Speaker_ID", "Severity"]).size()
            .reset_index(name="num_utterances")
            .sort_values(["Severity", "Speaker_ID"]))
    return table.reset_index(drop=True)


def build_feature_architecture_table() -> pd.DataFrame:
    """Table 2 — Branch | Input | Feature category | Channels | Encoder |
    Bottleneck, read from config.py constants and src.training.reporting.
    feature_audit() — never hardcoded."""
    from src.training.reporting import feature_audit

    audit = feature_audit(num_classes=4)
    return pd.DataFrame([
        {"branch": "Learned", "input": "raw 16kHz waveform (speech-focused profile)",
         "feature_category": "wav2vec2 contextual representation",
         "channels": config.WAV2VEC_EMBED_DIM, "encoder": "wav2vec2-base-960h + LoRA",
         "bottleneck": audit["learned_branch"]["dimensions"]},
        {"branch": "Segmental", "input": "MFCC+delta+delta-delta + framewise formants + HNR",
         "feature_category": "spectral / resonance / voice-quality (framewise)",
         "channels": audit["segmental_branch"]["input_channels"], "encoder": "3-layer 1D-CNN",
         "bottleneck": audit["segmental_branch"]["dimensions"]},
        {"branch": "Suprasegmental", "input": "F0 + voicing mask + intensity (temporal-preserving profile)",
         "feature_category": "pitch / energy / voicing (framewise)",
         "channels": audit["suprasegmental_branch"]["input_channels"], "encoder": "2-layer 1D-CNN",
         "bottleneck": audit["suprasegmental_branch"]["dimensions"]},
    ])


def build_model_dimensions_table() -> pd.DataFrame:
    """Table 3 — the exact tensor-dimension chain per branch, from
    feature_audit()'s real tensors."""
    from src.training.reporting import feature_audit

    audit = feature_audit(num_classes=4)
    return pd.DataFrame([
        {"stage": "Learned", "transform": f"{config.WAV2VEC_EMBED_DIM} -> {audit['learned_branch']['dimensions']}"},
        {"stage": "Segmental", "transform": f"{audit['segmental_branch']['input_channels']} x T -> {audit['segmental_branch']['dimensions']}"},
        {"stage": "Suprasegmental", "transform": f"{audit['suprasegmental_branch']['input_channels']} x T -> {audit['suprasegmental_branch']['dimensions']}"},
        {"stage": "Fusion", "transform": str(audit["fusion"]["fused_dim"])},
    ])


def build_final_metrics_table(run_name: str) -> pd.DataFrame:
    """Table 4 — pooled Accuracy / Balanced Accuracy / Macro-F1 /
    Weighted-F1 / Ordinal MAE for one run, from its pooled metrics JSON."""
    metrics = load_pooled_metrics(run_name)
    rows = [("Accuracy", metrics.get("accuracy")), ("Balanced Accuracy", metrics.get("balanced_accuracy")),
           ("Macro-F1", metrics.get("f1")), ("Weighted-F1", metrics.get("f1_weighted")),
           ("Ordinal MAE", metrics.get("ordinal_mae")), ("AUROC", metrics.get("auroc"))]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_per_class_metrics_table(run_name: str, task: str = "severity") -> pd.DataFrame:
    """Table 5 — per-class precision/recall/F1/support, pooled across every
    fold's held-out predictions."""
    from sklearn.metrics import precision_recall_fscore_support

    preds = load_run_predictions(run_name)
    class_names = config.SEVERITY_CLASS_NAMES if task == "severity" else config.DETECTION_CLASS_NAMES
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        preds["y_true"], preds["y_pred"], labels=labels, zero_division=np.nan)
    return pd.DataFrame({"class": class_names, "precision": precision, "recall": recall,
                         "f1": f1, "support": support})


def build_branch_ablation_table(ablation_df: pd.DataFrame) -> pd.DataFrame:
    """Table 6 — thin passthrough of
    src.model_analysis.evaluate_branch_ablation()'s output, reset to a plain
    column (not index) for CSV export."""
    return ablation_df.reset_index().rename(columns={"index": "variant"})


def build_gate_contribution_table(gate_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Table 7 — thin passthrough of
    src.model_analysis.summarize_gate_values()'s output."""
    return gate_summary_df.reset_index().rename(columns={"index": "branch"})


def build_explainability_table(shap_group_df: pd.DataFrame,
                               permutation_group_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Table 8 — feature-group SHAP importance, optionally merged with
    feature-group permutation importance for a side-by-side comparison
    (src.model_analysis.aggregate_shap_by_group /
    aggregate_permutation_importance_by_group)."""
    table = shap_group_df.rename(columns={"mean_abs_shap": "shap_importance"})
    if permutation_group_df is not None:
        table = table.merge(
            permutation_group_df.rename(columns={"importance_mean": "permutation_importance"}),
            on="group", how="outer")
    return table.sort_values(table.columns[1], ascending=False).reset_index(drop=True)


LIMITATIONS = [
    "UA-Speech has a limited number of dysarthric speakers.",
    "Vocabulary overlap exists between speaker partitions.",
    "Speaker-independent does not mean unseen-word evaluation.",
    "F0 extraction can be unreliable in dysarthric speech.",
    "CORAL imposes an ordinal latent structure.",
    "SHAP uses a surrogate model and should not be interpreted as causal attribution.",
    "Gate values indicate learned reliance, not causal importance.",
    "Branch ablation is inference-time contribution analysis, not causal proof.",
    "UA-Speech severity categories are intelligibility-derived and should not be "
    "described as direct clinical motor-severity measurements.",
    "A single training run prevents extensive hyperparameter optimization.",
]


def build_limitations_table() -> pd.DataFrame:
    """The final results notebook's limitations table — fixed, scientifically
    reviewed statements (not derived from run data, so this reads the same
    regardless of which run is being reported on)."""
    return pd.DataFrame({"limitation": LIMITATIONS})


def export_paper_tables(run_name: str, manifest: Optional[pd.DataFrame] = None,
                        ablation_df: Optional[pd.DataFrame] = None,
                        gate_summary_df: Optional[pd.DataFrame] = None,
                        shap_group_df: Optional[pd.DataFrame] = None,
                        permutation_group_df: Optional[pd.DataFrame] = None,
                        task: str = "severity", out_dir: Optional[Path] = None
                        ) -> Dict[str, Path]:
    """
    Write every available paper table (1-8 + limitations) as CSV under
    config.TABLES_DIR. Table 2/3 (architecture, no run needed) and the
    limitations table are always written; table 1 (dataset) is written when
    `manifest` is given; tables 4/5 (run-dependent) and 6-8 (checkpoint-
    dependent, via the caller's precomputed DataFrames) are each skipped —
    not an error — when the underlying run/DataFrame doesn't exist yet, so
    this is safe to call before the one-shot training run has produced any
    results (post-hoc analysis, and this export, can be run incrementally).
    """
    out_dir = Path(out_dir or config.TABLES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    tables = {"table2_feature_architecture": build_feature_architecture_table(),
             "table3_model_dimensions": build_model_dimensions_table(),
             "limitations": build_limitations_table()}
    try:
        tables["table4_final_metrics"] = build_final_metrics_table(run_name)
    except FileNotFoundError as e:
        print_kv("table4_final_metrics skipped", str(e))
    try:
        tables["table5_per_class_metrics"] = build_per_class_metrics_table(run_name, task=task)
    except FileNotFoundError as e:
        print_kv("table5_per_class_metrics skipped", str(e))
    if manifest is not None:
        tables["table1_dataset"] = build_dataset_table(manifest)
    if ablation_df is not None:
        tables["table6_branch_ablation"] = build_branch_ablation_table(ablation_df)
    if gate_summary_df is not None:
        tables["table7_gate_contribution"] = build_gate_contribution_table(gate_summary_df)
    if shap_group_df is not None:
        tables["table8_explainability"] = build_explainability_table(shap_group_df, permutation_group_df)

    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        written[name] = path

    print_kv("Paper tables exported", f"{len(written)} -> {out_dir}")
    return written
