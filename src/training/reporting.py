"""
Per-fold and cross-fold output writers: predictions, metrics, confusion
matrices, ROC curves, and embeddings — everything train.py drops into
outputs/ for downstream phases (ablation tables, error analysis, Praat
correlation, embedding visualization).

Also home to the EXPERIMENT REGISTRY (see below), the single record of what
was actually evaluated — as opposed to what merely left a file behind.
"""

import contextlib
import hashlib
import io
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, roc_curve

from src import config
from src.training.checkpoint import replace_with_retry


def save_predictions(path: Path, filenames, speaker_ids, y_true: np.ndarray,
                     y_pred: np.ndarray, y_prob: np.ndarray, task: str,
                     y_pred_argmax: Optional[np.ndarray] = None) -> None:
    """Per-utterance predictions for one fold.

    `filename` is the first column and is what makes Phase 5 possible: it joins
    a row back to its audio file (for spectrograms) and to outputs/praat_features.csv
    (for the error/feature correlation). speaker_id alone cannot do either.

    `y_pred_argmax`, when given (ordinal models, whose y_pred is the CORAL
    median decode), is written as an extra column so the decode choice can be
    audited from the CSV alone.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    class_names = config.DETECTION_CLASS_NAMES if task == "detection" else config.SEVERITY_CLASS_NAMES
    df = pd.DataFrame({
        "filename": filenames,
        "speaker_id": speaker_ids,
        "y_true": y_true,
        "y_true_label": [class_names[i] for i in y_true],
        "y_pred": y_pred,
        "y_pred_label": [class_names[i] for i in y_pred],
        "correct": np.asarray(y_true) == np.asarray(y_pred),
    })
    if y_pred_argmax is not None:
        df["y_pred_argmax"] = y_pred_argmax
    if task == "detection":
        df["prob_positive"] = y_prob
    else:
        for i, name in enumerate(class_names):
            df[f"prob_{name.replace(' ', '_')}"] = y_prob[:, i]
    temp_path = path.with_name(path.name + ".tmp")
    df.to_csv(temp_path, index=False)
    replace_with_retry(temp_path, path)


def save_metrics(path: Path, metrics: Dict[str, float]) -> None:
    """Atomic (temp file + retried replace): together with the predictions
    CSV this file is the fold-finished marker, so it must never be left
    half-written, and a sync client holding the old copy open must not fail
    a fold whose training already finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "w") as f:
        json.dump(metrics, f, indent=2)
    replace_with_retry(temp_path, path)


def save_confusion_matrix(path: Path, cm: np.ndarray, task: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    class_names = config.DETECTION_CLASS_NAMES if task == "detection" else config.SEVERITY_CLASS_NAMES
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
               yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roc_curve(path: Path, y_true: np.ndarray, y_prob: np.ndarray, task: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    drew_a_curve = False

    if task == "detection":
        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            ax.plot(fpr, tpr, label="Dysarthric Patient")
            drew_a_curve = True
    else:
        for i, name in enumerate(config.SEVERITY_CLASS_NAMES):
            binary_true = (y_true == i).astype(int)
            if len(np.unique(binary_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
            ax.plot(fpr, tpr, label=name)
            drew_a_curve = True

    if not drew_a_curve:
        # Every class in this fold's split was single-valued (typical of a
        # tiny smoke-test slice) — nothing meaningful to plot.
        plt.close(fig)
        return

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_embeddings(path: Path, embeddings: np.ndarray, y_true: np.ndarray,
                    speaker_ids, filenames=None,
                    branch_embeddings: Optional[Dict[str, np.ndarray]] = None,
                    gate_weights: Optional[np.ndarray] = None) -> None:
    """Test-fold embeddings, keyed by filename so Phase 5's embedding map can be
    coloured by whether each point was classified correctly.

    branch_embeddings ({"learned":, "segmental":, "supra":} arrays) and
    gate_weights ((N,3) array, columns [learned, segmental, supra]) are
    written as extra keys in the SAME .npz — populated only for
    GatedFusionModel (see src.training.engine.EpochResult) — rather than a
    second file, so a consumer that only wants the fused `embeddings` array
    is unaffected and one file per fold stays the on-disk contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "embeddings": embeddings,
        "y_true": y_true,
        "speaker_ids": np.asarray(speaker_ids),
    }
    if filenames is not None:
        arrays["filenames"] = np.asarray(filenames)
    if branch_embeddings is not None:
        for name, values in branch_embeddings.items():
            arrays[f"branch_{name}"] = values
    if gate_weights is not None:
        arrays["gate_weights"] = gate_weights
    np.savez(path, **arrays)


def aggregate_fold_metrics(metrics_dir: Path, run_name: str,
                           fold_metrics: List[Dict]) -> pd.DataFrame:
    """Average metrics across folds (mean +/- std) and save a summary table."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(fold_metrics)
    # Numeric scalars only — fold records also carry labels, speaker lists and
    # per-class prediction counts, which have no mean.
    metric_cols = [c for c in df.columns if c != "fold"
                   and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    summary = df[metric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    df.to_csv(metrics_dir / f"{run_name}.per_fold.csv", index=False)
    summary.to_csv(metrics_dir / f"{run_name}.summary.csv")
    with open(metrics_dir / f"{run_name}.summary.json", "w") as f:
        json.dump(summary.to_dict(orient="index"), f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Experiment registry
#
# THE PROBLEM IT SOLVES
# Before this existed, every consumer inferred "this experiment finished" from
# "a CSV exists". That is how a single held-out control speaker — one fold of
# an intended 28, carrying no positive class at all — became a headline
# "99.7% accuracy" row in the final comparison table. A run that stops at the
# time budget writes exactly the same files as one that completes; nothing on
# disk distinguished them.
#
# The registry records what was actually evaluated: which folds ran, which
# speakers they held out, which classes those speakers covered, and whether the
# run reached its intended fold count. Downstream reporting reads STATUS from
# here rather than guessing from filenames.
#
# It deliberately lives beside save_experiment_bundle (which already owns
# outputs/experiments/) and is written from the same run_fold/run_training path
# that already computes every one of these values — it is an extension of the
# existing bundle system, not a second bookkeeping mechanism.
# ---------------------------------------------------------------------------
REGISTRY_PATH = config.EXPERIMENTS_DIR / "registry.csv"

# Per-fold outcomes.
FOLD_COMPLETED = "COMPLETED"        # trained and evaluated in this session
FOLD_CACHED = "CACHED"              # loaded from a previous session's output
FOLD_FAILED = "FAILED"              # raised; excluded from pooling
FOLD_SKIPPED_DEADLINE = "SKIPPED_DEADLINE"   # budget ran out before it started
FOLD_INTERRUPTED = "INTERRUPTED"    # started, stopped between epochs at the hard
                                    # deadline; resumes from latest.pt next session

# Run-level rollups.
RUN_COMPLETED = "COMPLETED"         # every expected fold has a result
RUN_PARTIAL = "PARTIAL"             # some folds ran, some did not
RUN_FAILED = "FAILED"               # folds ran but none produced a result
RUN_NOT_STARTED = "NOT_STARTED"

REGISTRY_COLUMNS = [
    "run_name", "model", "task", "cv_protocol", "fold_id", "fold_index",
    "expected_folds", "held_out_speakers", "speaker_labels", "num_samples",
    "num_classes_present", "epochs_completed", "runtime_s", "status", "recorded_at",
]


def describe_fold(test_df: pd.DataFrame) -> Dict:
    """Held-out composition of one fold, straight from its test split.

    `speaker_labels` is what makes a single-class fold self-evident in the
    registry without re-reading predictions: for a detection LOSO fold it reads
    e.g. "CF02=Healthy Control", and num_classes_present is 1.
    """
    speakers = sorted(test_df["Speaker_ID"].unique().tolist())
    label_column = "Severity" if "Severity" in test_df.columns else "Group"
    pairs = (test_df[["Speaker_ID", label_column]].drop_duplicates()
             .sort_values("Speaker_ID"))
    return {
        "held_out_speakers": ";".join(speakers),
        "speaker_labels": ";".join(f"{r.Speaker_ID}={getattr(r, label_column)}"
                                   for r in pairs.itertuples(index=False)),
        "num_samples": int(len(test_df)),
    }


def record_fold(run_name: str, model: str, task: str, cv_protocol: str,
                fold_id: str, fold_index: int, expected_folds: int,
                status: str, fold_description: Optional[Dict] = None,
                num_classes_present: Optional[int] = None,
                epochs_completed: Optional[int] = None,
                runtime_s: Optional[float] = None,
                registry_path: Optional[Path] = None) -> None:
    """Upsert one (run_name, fold_id) row into the registry.

    Upsert, not append: re-running a fold (after a crash, or a resumed session
    re-reading it from cache) must update its row rather than accumulate
    duplicates that would inflate the completed-fold count and hand the
    eligibility gate a false 100% coverage.
    """
    registry_path = Path(registry_path or REGISTRY_PATH)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "run_name": run_name, "model": model, "task": task,
        "cv_protocol": cv_protocol, "fold_id": fold_id, "fold_index": fold_index,
        "expected_folds": expected_folds,
        "held_out_speakers": "", "speaker_labels": "", "num_samples": np.nan,
        "num_classes_present": num_classes_present,
        "epochs_completed": epochs_completed,
        "runtime_s": runtime_s, "status": status,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(fold_description or {}),
    }

    existing = load_registry(registry_path)
    new_row = pd.DataFrame([row])
    if existing.empty:
        # Concatenating onto an all-NA placeholder frame lets pandas infer
        # column dtypes from the empty side and warns about it; the first row
        # should simply define the schema.
        updated = new_row
    else:
        keep = ~((existing["run_name"] == run_name) & (existing["fold_id"] == fold_id))
        # All-NA columns dropped before concat (pandas deprecates letting them
        # decide dtypes); the reindex below restores the full column set.
        frames = [frame.dropna(axis=1, how="all") for frame in (existing[keep], new_row)]
        updated = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    temp_path = registry_path.with_name(registry_path.name + ".tmp")
    updated.reindex(columns=REGISTRY_COLUMNS).to_csv(temp_path, index=False)
    replace_with_retry(temp_path, registry_path)


def load_registry(registry_path: Optional[Path] = None) -> pd.DataFrame:
    """The raw per-fold registry, or an empty frame with the right columns."""
    registry_path = Path(registry_path or REGISTRY_PATH)
    if not registry_path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    return pd.read_csv(registry_path)


def summarize_registry(registry_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Per-run rollup: how much of each experiment actually happened.

    Columns:
      expected_folds / completed_folds  intended vs. produced a result
      valid_folds                       folds whose held-out set had >1 class,
                                        i.e. folds on which class-sensitive
                                        metrics are defined at all
      failed_folds / skipped_folds      excluded, with the reason distinguished
      coverage                          completed / expected
      pooled_has_both_classes           whether the union of completed folds
                                        covers more than one class — the thing
                                        that decides if a POOLED detection
                                        metric means anything
      status                            COMPLETED / PARTIAL / FAILED

    A run can have completed_folds > 0 and pooled_has_both_classes == False —
    that is precisely the pre-repair failure mode (two held-out controls), and
    it is why coverage alone is not a sufficient gate.
    """
    registry = load_registry(registry_path)
    if registry.empty:
        return pd.DataFrame(columns=[
            "run_name", "model", "task", "cv_protocol", "expected_folds",
            "completed_folds", "valid_folds", "failed_folds", "skipped_folds",
            "coverage", "pooled_has_both_classes", "total_runtime_s", "status"])

    rows = []
    for run_name, group in registry.groupby("run_name", sort=True):
        done = group[group["status"].isin([FOLD_COMPLETED, FOLD_CACHED])]
        expected = int(group["expected_folds"].max())
        completed = int(len(done))
        classes = pd.to_numeric(done["num_classes_present"], errors="coerce")
        rows.append({
            "run_name": run_name,
            "model": group["model"].iloc[0],
            "task": group["task"].iloc[0],
            "cv_protocol": group["cv_protocol"].iloc[0],
            "expected_folds": expected,
            "completed_folds": completed,
            "valid_folds": int((classes > 1).sum()),
            "failed_folds": int((group["status"] == FOLD_FAILED).sum()),
            "skipped_folds": int(group["status"].isin(
                [FOLD_SKIPPED_DEADLINE, FOLD_INTERRUPTED]).sum()),
            "coverage": completed / expected if expected else 0.0,
            # Distinct held-out labels across every completed fold. One fold of
            # Healthy plus one of Dysarthric pools to both classes even though
            # neither fold alone is valid — which is exactly why this is
            # computed over the union rather than per fold.
            "pooled_has_both_classes": _pooled_class_count(done) > 1,
            "total_runtime_s": float(pd.to_numeric(
                done["runtime_s"], errors="coerce").sum()),
            "status": (RUN_COMPLETED if completed and completed >= expected
                       else RUN_PARTIAL if completed
                       else RUN_FAILED),
        })
    return pd.DataFrame(rows).sort_values(["task", "run_name"]).reset_index(drop=True)


def _pooled_class_count(completed_folds: pd.DataFrame) -> int:
    """Distinct held-out class labels across every completed fold of one run,
    parsed from the `speaker_labels` column ("CF02=Healthy Control;...")."""
    labels = set()
    for entry in completed_folds["speaker_labels"].dropna():
        for pair in str(entry).split(";"):
            if "=" in pair:
                labels.add(pair.split("=", 1)[1])
    return len(labels)


# ---------------------------------------------------------------------------
# Run-level console reports (src.training.runner.run_training)
#
# Every count below is derived from the fold list the run was actually
# configured with — never a hard-coded protocol size — and every metric that
# is undefined on the data it would be computed from is printed as N/A with
# the reason, never as a number.
# ---------------------------------------------------------------------------
RUN_STATUS_COMPLETE = "COMPLETE"
RUN_STATUS_PARTIAL = "PARTIAL"
RUN_STATUS_FAILED = "FAILED"


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def _fmt_metric(value) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.4f}"


def run_coverage(expected_fold_ids: List[str], fold_status: Dict[str, str]) -> Dict:
    """Coverage of one run_training call over its configured folds: which
    folds produced a result (trained now or loaded from disk), which failed,
    which were skipped by the runtime guard or interrupted, and which are
    therefore missing — plus the COMPLETE / PARTIAL / FAILED status."""
    expected = list(expected_fold_ids)
    done = [f for f in expected if fold_status.get(f) in (FOLD_COMPLETED, FOLD_CACHED)]
    failed = [f for f in expected if fold_status.get(f) == FOLD_FAILED]
    interrupted = [f for f in expected if fold_status.get(f) == FOLD_INTERRUPTED]
    skipped = [f for f in expected
               if fold_status.get(f) in (FOLD_SKIPPED_DEADLINE, FOLD_INTERRUPTED)]
    missing = [f for f in expected if f not in done]
    if expected and len(done) == len(expected):
        status = RUN_STATUS_COMPLETE
    elif done:
        status = RUN_STATUS_PARTIAL
    else:
        status = RUN_STATUS_FAILED
    return {
        "expected": len(expected), "completed": len(done),
        "trained_this_session": sum(fold_status.get(f) == FOLD_COMPLETED for f in expected),
        "loaded_from_disk": sum(fold_status.get(f) == FOLD_CACHED for f in expected),
        "completion_pct": round(100.0 * len(done) / len(expected), 1) if expected else 0.0,
        "status": status,
        "expected_folds": expected, "completed_folds": done, "missing_folds": missing,
        "failed_folds": failed, "skipped_folds": skipped, "interrupted_folds": interrupted,
    }


def print_run_coverage(coverage: Dict, run_name: str) -> None:
    from src.console import print_header, print_kv, print_note, print_status

    print_header(f"Run status — {coverage['status']}")
    print_kv("Folds with a result", f"{coverage['completed']} / {coverage['expected']}  "
             f"({coverage['trained_this_session']} trained now, "
             f"{coverage['loaded_from_disk']} loaded from disk)")
    if coverage["failed_folds"]:
        print_kv("Failed", ", ".join(coverage["failed_folds"]))
    if coverage["interrupted_folds"]:
        print_kv("Interrupted (resumes mid-fold)", ", ".join(coverage["interrupted_folds"]))
    not_started = [f for f in coverage["skipped_folds"] if f not in coverage["interrupted_folds"]]
    if not_started:
        print_kv("Not started", ", ".join(not_started))
    if coverage["status"] == RUN_STATUS_COMPLETE:
        print_status(f"All {coverage['expected']} folds of {run_name} are complete.", ok=True)
    else:
        print_note(f"Re-run the training cell to finish the "
                   f"{len(coverage['missing_folds'])} missing fold(s) — finished folds load "
                   "from disk. Pooled numbers until then are partial, not final.")


def print_fold_report(record: Dict, fold_index: int, n_folds: int, max_epochs: int,
                      task: str) -> None:
    """The end-of-fold summary: two lines. Only metrics DEFINED on one
    held-out speaker are shown — a severity LOSO fold has a single true
    class, so macro-F1 / balanced accuracy / AUROC exist only pooled (Section
    7 of the notebook), never per fold."""
    from src.console import print_note

    best_epoch = record.get("best_epoch")
    parts = [f"accuracy {_fmt_metric(record.get('accuracy'))[:5]}"]
    if task == "severity":
        parts.append(f"ordinal MAE {_fmt_metric(record.get('ordinal_mae'))[:5]}")
    parts.append(f"best epoch {best_epoch} of {record.get('epochs_completed')} run"
                 if best_epoch is not None else "no improving epoch")
    parts.append(f"fold time {_fmt_duration(record.get('fold_time_s'))}")
    print(f"  Result     {record['fold']} ({record.get('true_label')}): " + " · ".join(parts))
    distribution = record.get("pred_distribution") or {}
    print("  Predicted  " + " · ".join(f"{name} {count}" for name, count in distribution.items()))
    if record.get("n_classes_present", 0) >= 2:
        print(f"  Macro-F1 {_fmt_metric(record.get('f1'))} · balanced accuracy "
              f"{_fmt_metric(record.get('balanced_accuracy'))} · AUROC "
              f"{_fmt_metric(record.get('auroc'))}")
    if "coral_thresholds" in record and not record.get("coral_thresholds_ordered"):
        print_note(f"CORAL threshold biases {record['coral_thresholds']} are not rank-ordered; "
                   "the median decode may differ from the raw threshold count here.")


def print_runtime_status(folds_done: int, n_folds: int, remaining_folds: int,
                         run_elapsed_s: float, fold_estimate_s: Optional[float],
                         n_timed_folds: int, safety_factor: float,
                         session_elapsed_s: Optional[float] = None,
                         session_budget_s: Optional[float] = None,
                         deadline_in_s: Optional[float] = None) -> None:
    """After every fold, ONE line: folds accounted for, elapsed time, mean
    fold time and the estimated time to finish — plus, only when a deadline
    is set, whether that finish fits it (ON TRACK = with the safety factor,
    AT RISK = only without it, WILL NOT FIT = the runtime guard will skip)."""
    parts = [f"{folds_done}/{n_folds} folds done", f"elapsed {_fmt_duration(run_elapsed_s)}"]
    if session_elapsed_s is not None and session_budget_s:
        parts.append(f"session {_fmt_duration(session_elapsed_s)} of "
                     f"{_fmt_duration(session_budget_s)}")
    if fold_estimate_s is not None and remaining_folds:
        remaining_s = remaining_folds * fold_estimate_s
        parts.append(f"mean fold {_fmt_duration(fold_estimate_s)}")
        parts.append(f"about {_fmt_duration(remaining_s)} to go")
        if deadline_in_s is not None:
            margin = deadline_in_s - remaining_s * safety_factor
            parts.append("ON TRACK" if margin >= 0
                         else "AT RISK" if deadline_in_s - remaining_s >= 0
                         else "WILL NOT FIT — later folds will be skipped")
    print("  Progress   " + " · ".join(parts))


def print_pooled_evaluation(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                            task: str, coverage: Dict,
                            speakers: Optional[List[str]] = None,
                            verbose: bool = True) -> Dict:
    """Pooled evaluation over every completed held-out speaker: headline
    metrics, confusion matrix, true/predicted class distributions, and
    per-class precision/recall/F1 — each marked N/A where undefined (a class
    never predicted has no precision; a class absent from the pooled set has
    no recall or AUROC). Labelled PARTIAL whenever coverage is incomplete.
    Returns the per-class table and distributions as a JSON-able dict;
    verbose=False computes the same report without printing it."""
    if not verbose:
        with contextlib.redirect_stdout(io.StringIO()):
            return print_pooled_evaluation(y_true, y_pred, y_prob, task, coverage,
                                           speakers=speakers, verbose=True)
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix
    from sklearn.metrics import roc_auc_score

    from src.console import print_header, print_kv, print_note, print_subheader, print_table
    from src.training.metrics import compute_metrics

    class_names = (config.SEVERITY_CLASS_NAMES if task == "severity"
                   else config.DETECTION_CLASS_NAMES)
    labels = list(range(len(class_names)))
    status = coverage["status"]
    tag = "" if status == RUN_STATUS_COMPLETE else f" — {status}, NOT A FINAL RESULT"
    print_header(f"Pooled held-out evaluation ({coverage['completed']}/{coverage['expected']} "
                 f"folds, {len(y_true):,} utterances){tag}")
    if status != RUN_STATUS_COMPLETE:
        print_note(f"Missing folds: {', '.join(coverage['missing_folds'])}. Pooled numbers "
                   "from a partial LOSO run are not comparable to the full protocol.")

    metrics = compute_metrics(y_true, y_pred, y_prob, task)
    print_subheader("Headline metrics")
    absent = [name for i, name in enumerate(class_names) if not (y_true == i).any()]
    if absent:
        print_note(f"No held-out utterances of {', '.join(absent)} yet: macro F1 / balanced "
                   f"accuracy average over the {len(class_names) - len(absent)} present "
                   f"classes only, and macro AUROC is undefined.")
    for key, label in (("accuracy", "Accuracy"), ("f1", "Macro F1"),
                       ("f1_weighted", "Weighted F1"),
                       ("balanced_accuracy", "Balanced accuracy"),
                       ("ordinal_mae", "Ordinal MAE"), ("auroc", "AUROC (macro one-vs-rest)")):
        if key == "ordinal_mae" and task != "severity":
            continue
        value = metrics.get(key)
        reason = ""
        if value is None or not np.isfinite(value):
            reason = ("  — undefined: fewer than 2 true classes pooled"
                      if metrics["n_classes_present"] < 2
                      else "  — undefined: not every class is present in the pooled set")
        print_kv(label, _fmt_metric(value) + reason)

    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    print_subheader("Confusion matrix (rows = true, columns = predicted)")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print_table(cm_df.reset_index().rename(columns={"index": "true \\ predicted"}))

    true_counts = cm.sum(axis=1)
    pred_counts = cm.sum(axis=0)
    per_class = per_class_table(y_true, y_pred, y_prob, task)
    print_subheader("Per-class (N/A = undefined: never predicted / absent from pooled set)")
    shown = per_class.copy()
    for column in ("precision", "recall", "f1", "auroc_ovr"):
        shown[column] = shown[column].map(_fmt_metric)
    print_table(shown)

    print_subheader("Class distributions")
    print_kv("True", " | ".join(f"{n} {int(c)}" for n, c in zip(class_names, true_counts)))
    print_kv("Predicted", " | ".join(f"{n} {int(c)}" for n, c in zip(class_names, pred_counts)))
    never = [n for n, c in zip(class_names, pred_counts) if c == 0]
    if never:
        print_note(f"Never predicted: {', '.join(never)}.")

    report = {"confusion_matrix": cm.tolist(), "class_names": class_names,
              "per_class": per_class.to_dict(orient="records"),
              "true_distribution": dict(zip(class_names, true_counts.tolist())),
              "pred_distribution": dict(zip(class_names, pred_counts.tolist()))}

    if speakers is not None and task == "severity":
        # One decision per speaker — the median of that speaker's utterance
        # predictions — since severity is a speaker-level clinical label.
        frame = pd.DataFrame({"speaker": speakers, "y_true": y_true, "y_pred": y_pred})
        per_speaker = frame.groupby("speaker").agg(
            true=("y_true", "first"), pred=("y_pred", lambda s: int(np.floor(np.median(s)))))
        speaker_acc = float((per_speaker["true"] == per_speaker["pred"]).mean())
        speaker_mae = float((per_speaker["true"] - per_speaker["pred"]).abs().mean())
        print_subheader("Speaker-level (median of each speaker's utterance predictions)")
        print_kv("Speakers correct", f"{int((per_speaker['true'] == per_speaker['pred']).sum())}"
                 f" / {len(per_speaker)}  (accuracy {speaker_acc:.4f}, MAE {speaker_mae:.4f})")
        report["speaker_level"] = {"accuracy": speaker_acc, "ordinal_mae": speaker_mae,
                                   "predictions": {s: {"true": int(r.true), "pred": int(r.pred)}
                                                   for s, r in per_speaker.iterrows()}}
    return report


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                    task: str) -> pd.DataFrame:
    """One row per class: true/predicted counts, precision, recall, F1 and
    one-vs-rest AUROC — NaN wherever undefined (precision of a class never
    predicted; recall/F1/AUROC of a class absent from y_true)."""
    from sklearn.metrics import roc_auc_score

    class_names = (config.SEVERITY_CLASS_NAMES if task == "severity"
                   else config.DETECTION_CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    true_counts, pred_counts = cm.sum(axis=1), cm.sum(axis=0)
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        precision = tp / pred_counts[i] if pred_counts[i] > 0 else float("nan")
        recall = tp / true_counts[i] if true_counts[i] > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
              else (0.0 if np.isfinite(recall) and true_counts[i] > 0 else float("nan")))
        binary = (y_true == i).astype(int)
        if 0 < binary.sum() < len(binary):
            prob_i = y_prob if (task == "detection" and i == 1) else (
                1 - y_prob if task == "detection" else y_prob[:, i])
            auroc = float(roc_auc_score(binary, prob_i))
        else:
            auroc = float("nan")
        rows.append({"class": name, "true_n": int(true_counts[i]), "pred_n": int(pred_counts[i]),
                     "precision": precision, "recall": recall, "f1": f1, "auroc_ovr": auroc})
    return pd.DataFrame(rows)


def collect_run_results(run_name: str, expected_folds: List[str],
                        task: str = "severity") -> Dict[str, object]:
    """Everything a results cell needs, rebuilt from the per-fold files on
    disk (outputs/metrics/<run>/<fold>.json + outputs/predictions/<run>/<fold>.csv)
    — so it is correct after a complete run, a partial one, or a run
    interrupted mid-fold, and never depends on the in-memory return value of
    run_training or on a RUN_STATUS file only a finished call writes.

    Returns display-ready DataFrames plus the coverage summary:
      coverage   expected / completed / missing / failed folds, status
      per_fold   one row per completed fold (only per-fold-defined metrics)
      pooled     headline pooled metrics, NaN (with a reason) where undefined
      per_class  per-class precision / recall / F1 / AUROC
      confusion  pooled confusion matrix (rows true, columns predicted)
      speakers   speaker-level decision (median of utterance predictions)
    """
    from src.training.metrics import compute_metrics

    class_names = (config.SEVERITY_CLASS_NAMES if task == "severity"
                   else config.DETECTION_CLASS_NAMES)
    rows, frames = [], []
    for fold_id in expected_folds:
        metrics_path = config.METRICS_DIR / run_name / f"{fold_id}.json"
        predictions_path = config.PREDICTIONS_DIR / run_name / f"{fold_id}.csv"
        if not (metrics_path.exists() and predictions_path.exists()):
            continue
        with open(metrics_path) as handle:
            m = json.load(handle)
        preds = pd.read_csv(predictions_path)
        counts = np.bincount(preds["y_pred"], minlength=len(class_names))
        row = {"Fold": fold_id,
               "True class": m.get("true_label") or class_names[int(preds["y_true"].iloc[0])],
               "Utterances": len(preds),
               "Accuracy": m.get("accuracy")}
        if task == "severity":
            row["Ordinal MAE"] = m.get("ordinal_mae")
        row.update({f"Pred {name}": int(c) for name, c in zip(class_names, counts)})
        row.update({"Best epoch": m.get("best_epoch"),
                    "Epochs run": m.get("epochs_completed"),
                    "Train (min)": (m.get("train_time_s") or float("nan")) / 60,
                    "Val speakers": str(m.get("val_speakers", "")).replace(";", ", ")})
        rows.append(row)
        frames.append(preds)

    registry = load_registry()
    completed = [r["Fold"] for r in rows]
    failed = []
    if not registry.empty:
        mine = registry[(registry["run_name"] == run_name) & (registry["status"] == FOLD_FAILED)]
        failed = [f for f in mine["fold_id"] if f not in completed]
    missing = [f for f in expected_folds if f not in completed]
    status = (RUN_STATUS_COMPLETE if expected_folds and len(completed) == len(expected_folds)
              else RUN_STATUS_PARTIAL if completed else RUN_STATUS_FAILED)
    coverage = {"expected": len(expected_folds), "completed": len(completed),
                "completion_pct": round(100.0 * len(completed) / max(len(expected_folds), 1), 1),
                "missing_folds": missing, "failed_folds": failed, "status": status}
    results: Dict[str, object] = {"coverage": coverage, "per_fold": pd.DataFrame(rows)}
    if not frames:
        return results

    preds = pd.concat(frames, ignore_index=True)
    y_true = preds["y_true"].to_numpy()
    y_pred = preds["y_pred"].to_numpy()
    y_prob = (preds["prob_positive"].to_numpy() if task == "detection" else
              preds[[f"prob_{n.replace(' ', '_')}" for n in class_names]].to_numpy())
    metrics = compute_metrics(y_true, y_pred, y_prob, task)
    absent = [n for i, n in enumerate(class_names) if not (y_true == i).any()]
    reason = ("undefined: fewer than 2 true classes pooled" if metrics["n_classes_present"] < 2
              else f"undefined: no held-out {', '.join(absent)} yet" if absent else "")
    pooled_rows = []
    for key, label in (("accuracy", "Accuracy"), ("f1", "Macro F1"), ("f1_weighted", "Weighted F1"),
                       ("balanced_accuracy", "Balanced accuracy"), ("ordinal_mae", "Ordinal MAE"),
                       ("auroc", "AUROC (macro one-vs-rest)")):
        if key == "ordinal_mae" and task != "severity":
            continue
        value = metrics.get(key)
        defined = value is not None and np.isfinite(value)
        note = "" if defined else reason
        if defined and absent and key in ("f1", "balanced_accuracy"):
            note = f"over the {len(class_names) - len(absent)} classes present"
        pooled_rows.append({"Metric": label, "Value": value if defined else float("nan"),
                            "Note": note})
    results["pooled"] = pd.DataFrame(pooled_rows)
    results["n_utterances"] = int(len(y_true))
    results["per_class"] = per_class_table(y_true, y_pred, y_prob, task)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    results["confusion"] = pd.DataFrame(cm, index=[f"True {n}" for n in class_names],
                                        columns=[f"Pred {n}" for n in class_names])
    if task == "severity":
        per_speaker = preds.groupby("speaker_id").agg(
            true=("y_true", "first"), pred=("y_pred", lambda v: int(np.floor(np.median(v)))),
            accuracy=("correct", "mean"))
        results["speakers"] = pd.DataFrame({
            "Speaker": per_speaker.index,
            "True class": [class_names[int(t)] for t in per_speaker["true"]],
            "Speaker-level prediction": [class_names[int(v)] for v in per_speaker["pred"]],
            "Correct": per_speaker["true"].to_numpy() == per_speaker["pred"].to_numpy(),
            "Utterance accuracy": per_speaker["accuracy"].to_numpy(),
        }).reset_index(drop=True)
    return results


def redecode_saved_predictions(predictions_dir: Path) -> pd.DataFrame:
    """Re-score a finished severity run's saved per-fold prediction CSVs
    under both decodings — argmax (what runs before the ordinal-decoding fix
    reported) and the CORAL median (src.losses.coral_rank_from_class_probs) —
    from the stored class probabilities alone, no retraining. For auditing an
    old run; its validation protocol is unchanged by this, so it is NOT a
    substitute for re-running under the speaker-disjoint protocol."""
    from src.losses import coral_rank_from_class_probs
    from src.training.metrics import compute_metrics

    frames = [pd.read_csv(p) for p in sorted(Path(predictions_dir).glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"No prediction CSVs under {predictions_dir}")
    preds = pd.concat(frames, ignore_index=True)
    prob_cols = [f"prob_{name.replace(' ', '_')}" for name in config.SEVERITY_CLASS_NAMES]
    probs = preds[prob_cols].to_numpy()
    y_true = preds["y_true"].to_numpy()
    decodes = {"argmax": probs.argmax(axis=1),
               "coral_median": coral_rank_from_class_probs(torch.from_numpy(probs)).numpy()}
    rows = []
    for name, y_pred in decodes.items():
        metrics = compute_metrics(y_true, y_pred, probs, "severity")
        rows.append({"decode": name, **{k: metrics[k] for k in (
            "accuracy", "f1", "f1_weighted", "balanced_accuracy", "ordinal_mae", "auroc")},
            "pred_distribution": np.bincount(y_pred, minlength=len(prob_cols)).tolist()})
    return pd.DataFrame(rows)


def save_experiment_bundle(experiment_name: str, model_name: str, task: str,
                           cfg, pooled_metrics: Dict, summary: pd.DataFrame,
                           num_classes: int) -> Path:
    """
    Repackage one budget-managed primary-detection experiment's already-written
    outputs into outputs/experiments/<experiment_name>/{config.json,metrics.json,
    predictions.csv,timing.json,checkpoint/} — additive to (not a replacement
    for) the flat outputs/{metrics,predictions,checkpoints,...}/<run_name>/
    layout that run_training()/run_fold() already wrote via save_metrics/
    save_predictions/save_checkpoint. Call once after run_training() returns.

    `cfg` is the TrainingConfig used for the run; `pooled_metrics` and `summary`
    are run_training()'s two return values.
    """
    from src.training.metrics import compute_confusion_matrix
    from src.training.models import build_model, parameter_counts

    run_name = cfg.run_name or f"{task}_{model_name}"
    bundle_dir = config.EXPERIMENTS_DIR / experiment_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # --- predictions.csv: concatenate every fold's already-saved predictions ---
    pred_dir = config.PREDICTIONS_DIR / run_name
    fold_files = sorted(p for p in pred_dir.glob("*.csv") if p.stem != "ALL_FOLDS_pooled")
    predictions_df = pd.concat([pd.read_csv(p) for p in fold_files], ignore_index=True)
    predictions_df = predictions_df.rename(columns={
        "speaker_id": "speaker", "y_true_label": "true_label",
        "y_pred_label": "predicted_label"})
    predictions_df.to_csv(bundle_dir / "predictions.csv", index=False)

    # --- timing.json: per-fold mean/std + summed totals, from run_fold's
    #     fold_time_s/train_time_s/inference_time_s (see runner.py::run_fold) ---
    timing_fields = [c for c in ("fold_time_s", "train_time_s", "inference_time_s")
                     if c in summary.index]
    per_fold_path = config.METRICS_DIR / f"{run_name}.per_fold.csv"
    per_fold_df = pd.read_csv(per_fold_path) if per_fold_path.exists() else None
    timing_out = {
        field: {"mean_s": float(summary.loc[field, "mean"]),
               "std_s": float(summary.loc[field, "std"]),
               "total_s": float(per_fold_df[field].sum()) if per_fold_df is not None else None}
        for field in timing_fields
    }
    with open(bundle_dir / "timing.json", "w") as f:
        json.dump(timing_out, f, indent=2)

    # --- metrics.json: pooled metrics + confusion matrix + parameter counts +
    #     training/inference time (also mirrored in timing.json above; kept
    #     here too since requirement 6 asks metrics.json itself to carry them) ---
    cm = compute_confusion_matrix(
        predictions_df["y_true"].to_numpy(), predictions_df["y_pred"].to_numpy(), task)
    params = parameter_counts(build_model(model_name, num_classes))
    metrics_out = {
        **pooled_metrics,
        "confusion_matrix": cm.tolist(),
        **params,
        "training_time_s": timing_out.get("train_time_s", {}).get("total_s"),
        "inference_time_s": timing_out.get("inference_time_s", {}).get("total_s"),
    }
    with open(bundle_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # --- config.json: the run's TrainingConfig plus everything else needed to
    #     reproduce it that TrainingConfig itself doesn't carry — VAD/MFCC/LoRA
    #     settings live as module-level constants in src/config.py, not per-run
    #     fields, and the VAD fallback rate is a property of the *dataset*, not
    #     the run, computed once by src.preprocessing.compute_vad_stats_batch
    #     into outputs/vad_stats.csv (Stage 9). ---
    vad_diagnostics = None
    if config.VAD_STATS_PATH.exists():
        vad_stats = pd.read_csv(config.VAD_STATS_PATH)
        vad_diagnostics = {
            "n_utterances": int(len(vad_stats)),
            "vad_fallback_count": int(vad_stats["fallback_used"].sum()),
            "vad_fallback_rate": float(vad_stats["fallback_used"].mean()),
            "mean_speech_ratio": float(vad_stats["speech_ratio"].mean()),
            "median_speech_ratio": float(vad_stats["speech_ratio"].median()),
        }

    cfg_dict = {
        **asdict(cfg), "model_name": model_name, "task": task,
        "vad": {
            "enabled": config.VAD_ENABLED, "backend": "Silero VAD (torch.hub)",
            "threshold": config.VAD_THRESHOLD, "min_speech_ms": config.VAD_MIN_SPEECH_MS,
            "min_silence_ms": config.VAD_MIN_SILENCE_MS, "speech_pad_ms": config.VAD_SPEECH_PAD_MS,
            "sample_rate": config.VAD_SAMPLE_RATE, "diagnostics": vad_diagnostics,
        },
        "audio": {
            "target_sr": config.TARGET_SR, "clip_seconds": config.CLIP_SECONDS,
            "max_samples": config.MAX_SAMPLES,
        },
        "mfcc": {
            "n_mfcc": config.N_MFCC, "mel_kwargs": config.MEL_KWARGS,
            "deltas": "delta + delta-delta", "output_dim": 3 * config.N_MFCC,
        },
        "lora": {
            "rank": config.LORA_RANK, "alpha": config.LORA_ALPHA,
            "dropout": config.LORA_DROPOUT, "target_modules": config.LORA_TARGET_MODULES,
            # deep_frozen/fusion_frozen use_lora=False; acoustic has no wav2vec2
            # pathway at all; every other variant runs with use_lora=True.
            "applies_to_this_model": model_name not in
                ("acoustic", "deep_frozen", "fusion_frozen"),
        },
        "wav2vec2_model": config.WAV2VEC_MODEL_NAME,
    }
    with open(bundle_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    # --- checkpoint/: copy every fold's best.pt (already saved by run_fold) ---
    ckpt_src_dir = config.CHECKPOINT_DIR / run_name
    ckpt_dst_dir = bundle_dir / "checkpoint"
    ckpt_dst_dir.mkdir(parents=True, exist_ok=True)
    for ckpt_file in ckpt_src_dir.glob("*/best.pt"):
        shutil.copy2(ckpt_file, ckpt_dst_dir / f"{ckpt_file.parent.name}_best.pt")

    return bundle_dir


# ---------------------------------------------------------------------------
# One-shot run freezing: the FEATURE AUDIT and FINAL RUN CONFIGURATION blocks
# (architecture plan Part 2, Component 16), plus a config-hash guard.
#
# Every number below is derived from a real, instantiated GatedFusionModel's
# tensors (via one dummy forward pass) or from src.config constants actually
# read by that model — never hardcoded — so this output is guaranteed to
# reflect the code that will actually run, not a comment that can drift from
# it (see the same failure mode this project already fixed once for
# DROPPED_FOR_BALANCE — a hand-maintained description of a value the code
# had since changed).
# ---------------------------------------------------------------------------
FROZEN_CONFIG_PATH = config.RESULTS_DIR / "frozen_config.json"


def _dummy_three_branch_batch(batch_size: int = 2, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Minimal, correctly-shaped tensors for one GatedFusionModel forward
    pass — no real audio needed, only used to read off tensor shapes."""
    from src.preprocessing import mfcc_frame_count

    total_frames = mfcc_frame_count(config.MAX_SAMPLES)
    return {
        "waveform": torch.zeros(batch_size, config.MAX_SAMPLES, device=device),
        "mfcc": torch.zeros(batch_size, config.SEGMENTAL_CHANNELS, total_frames, device=device),
        "supra": torch.zeros(batch_size, config.SUPRA_CHANNELS, total_frames, device=device),
        "attention_mask": torch.ones(batch_size, config.MAX_SAMPLES, dtype=torch.bool, device=device),
        "supra_valid_frames": torch.full((batch_size,), total_frames, dtype=torch.long, device=device),
    }


def feature_audit(model=None, num_classes: int = 4) -> Dict:
    """
    Derive the brief's "FEATURE AUDIT" block from a real model instance and
    one dummy forward pass — every dimension is read off the actual tensors,
    every feature name is read off src.praat's column lists, nothing here is
    a hardcoded number that could silently drift from the code.
    """
    from src.praat import SEGMENTAL_FEATURE_COLUMNS, SUPRASEGMENTAL_FEATURE_COLUMNS
    from src.training.models import build_model, SEVERITY_MODEL_NAME

    owns_model = model is None
    if owns_model:
        model = build_model(SEVERITY_MODEL_NAME, num_classes, num_speakers=2)
    model.eval()

    batch = _dummy_three_branch_batch(device=next(model.parameters()).device)
    with torch.no_grad():
        z_learned, z_segmental, z_supra = model.encode_branches(
            batch["waveform"], batch["mfcc"], batch["supra"],
            batch["attention_mask"], batch["supra_valid_frames"])
        z_unified, gates = model.fuse(z_learned, z_segmental, z_supra)

    audit = {
        "learned_branch": {
            "raw_hidden_shape": [batch["waveform"].shape[0], "T", config.WAV2VEC_EMBED_DIM],
            "projected_shape": list(z_learned.shape),
            "dimensions": z_learned.shape[-1],
            "wav2vec2_model": config.WAV2VEC_MODEL_NAME,
            "lora_target_modules": config.LORA_TARGET_MODULES,
            "lora_rank": config.LORA_RANK, "lora_alpha": config.LORA_ALPHA,
            "lora_dropout": config.LORA_DROPOUT,
        },
        "segmental_branch": {
            "input_channels": config.SEGMENTAL_CHANNELS,
            "input_shape": list(batch["mfcc"].shape),
            "projected_shape": list(z_segmental.shape),
            "dimensions": z_segmental.shape[-1],
            "engineered_feature_families": [
                "MFCC (13)", "delta-MFCC (13)", "delta-delta-MFCC (13)",
                "framewise F1/F2/F3 (3)", "framewise HNR (1)"],
            "shap_surrogate_feature_names": list(SEGMENTAL_FEATURE_COLUMNS),
        },
        "suprasegmental_branch": {
            "input_channels": config.SUPRA_CHANNELS,
            "input_shape": list(batch["supra"].shape),
            "projected_shape": list(z_supra.shape),
            "dimensions": z_supra.shape[-1],
            "engineered_feature_families": [
                "F0 (semitones; zero when unvoiced)", "voicing mask", "intensity (dB)"],
            "shap_surrogate_feature_names": list(SUPRASEGMENTAL_FEATURE_COLUMNS),
        },
        "fusion": {
            "learned_dim": z_learned.shape[-1], "segmental_dim": z_segmental.shape[-1],
            "supra_dim": z_supra.shape[-1], "fused_dim": z_unified.shape[-1],
            "gate_weights_this_dummy_batch": gates.mean(dim=0).tolist(),
        },
    }

    if owns_model:
        del model
    return audit


def print_feature_audit(model=None, num_classes: int = 4) -> Dict:
    """Print the brief's FEATURE AUDIT block (Section 13) and return the same
    dict feature_audit() computes."""
    from src.console import print_kv, print_subheader

    audit = feature_audit(model=model, num_classes=num_classes)
    learned, segmental = audit["learned_branch"], audit["segmental_branch"]
    supra, fusion = audit["suprasegmental_branch"], audit["fusion"]

    def shape(values) -> str:
        return " x ".join(str(v) for v in values)

    print_subheader("Feature audit (shapes read from a dummy forward pass)")
    print_kv("Learned", f"[B x T x {config.WAV2VEC_EMBED_DIM}] -> pooled -> "
                        f"[{shape(learned['projected_shape'])}]")
    print_kv("Segmental", f"[{shape(segmental['input_shape'])}] -> "
                          f"[{shape(segmental['projected_shape'])}]")
    print_kv("  channels", ", ".join(segmental["engineered_feature_families"]))
    print_kv("Suprasegmental", f"[{shape(supra['input_shape'])}] -> "
                               f"[{shape(supra['projected_shape'])}]")
    print_kv("  channels", ", ".join(supra["engineered_feature_families"]))
    print_kv("Fused representation", f"{fusion['learned_dim']} + {fusion['segmental_dim']} + "
                                     f"{fusion['supra_dim']} = {fusion['fused_dim']}")
    return audit


def _git_commit_hash() -> Optional[str]:
    """Best-effort `git rev-parse HEAD` for reproducibility (brief Section
    43) — never raises; returns None if git isn't available, this isn't a
    repo, or anything else goes wrong. Not load-bearing for training itself,
    only for the frozen-config provenance record."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
            capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _software_versions() -> Dict[str, Optional[str]]:
    """Package/runtime versions actually installed in THIS process — read
    from the imported modules themselves, not requirements.txt, so this
    reflects what the run actually executed with. Any single missing
    package degrades to None rather than aborting the whole block."""
    import platform

    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for module_name in ("torch", "torchaudio", "transformers", "peft", "numpy", "pandas"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except Exception:
            versions[module_name] = None
    try:
        import torch
        versions["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    except Exception:
        versions["cuda"] = None
    return versions


def _ablation_switches(model_name: str) -> Optional[Dict]:
    """The GatedFusionModel switches a named severity ablation uses, or None
    for the primary model / legacy variants."""
    from src.training.models import _ABLATION_DEFAULTS, SEVERITY_ABLATIONS
    if model_name not in SEVERITY_ABLATIONS:
        return None
    return {key: (list(value) if isinstance(value, tuple) else value)
            for key, value in {**_ABLATION_DEFAULTS, **SEVERITY_ABLATIONS[model_name]}.items()}


def build_final_run_configuration(cfg, df: pd.DataFrame, num_speakers_total: int) -> Dict:
    """
    Everything the brief's Section 23 "FINAL RUN CONFIGURATION" block needs,
    read from `cfg` (a src.training.runner.TrainingConfig) and `df` (the
    manifest) — never hardcoded. Call ONCE, before the real training run,
    and pass the result to write_frozen_config() to freeze it.
    """
    from src import splits as splits_module

    audit = feature_audit(num_classes=config.NUM_CLASSES[cfg.task])
    fold_speakers = (config.SEVERITY_LOSO_ORDER if cfg.severity_protocol == "full_loso"
                     else sorted(set(config.DYSARTHRIC_IDS) - set(config.DROPPED_FOR_BALANCE)))
    # cfg.max_folds truncates full_loso the same way src.training.runner.run_training
    # does (fold_iter[:cfg.max_folds]) — reflected here too so a budget-reduced run's
    # frozen config records which speakers it ACTUALLY evaluated, not the full 15.
    if cfg.severity_protocol == "full_loso" and cfg.max_folds is not None:
        fold_speakers = fold_speakers[:cfg.max_folds]

    return {
        "dataset": {
            "speakers_total": num_speakers_total,
            "severity_classes": config.SEVERITY_CLASS_NAMES,
            "severity_protocol": cfg.severity_protocol,
            "fold_speakers": fold_speakers,
            "num_folds": (len(fold_speakers) if cfg.severity_protocol == "full_loso" else 81),
        },
        "audio": {
            "sampling_rate": config.TARGET_SR,
            "clip_seconds": config.CLIP_SECONDS,
            "max_samples": config.MAX_SAMPLES,
            "speech_focused_vad_pad_ms": config.VAD_SPEECH_PAD_MS,
            "temporal_preserving_vad_pad_ms": config.SUPRA_VAD_SPEECH_PAD_MS,
        },
        "learned_branch": audit["learned_branch"],
        "segmental_branch": {k: v for k, v in audit["segmental_branch"].items()
                             if k != "shap_surrogate_feature_names"},
        "suprasegmental_branch": {k: v for k, v in audit["suprasegmental_branch"].items()
                                  if k != "shap_surrogate_feature_names"},
        "complementarity": {"method": "batch cross-covariance (Barlow-Twins-style, mean-normalized)",
                            "lambda": config.LAMBDA_COMP},
        "speaker_invariance": {"method": "gradient-reversal adversarial speaker head",
                               "lambda": config.LAMBDA_SPEAKER,
                               "grl_lambda": config.GRL_LAMBDA},
        "fusion": {"method": "learned softmax gate over 3 branch embeddings",
                  "fused_dim": audit["fusion"]["fused_dim"]},
        "severity_head": {"method": "CORAL ordinal regression",
                          "loss": "class-weighted CORAL binary cross-entropy sum",
                          "decoding": "median of the CORAL distribution (threshold count)"},
        "validation": {"protocol": getattr(cfg, "val_protocol", "utterance"),
                       "monitored": "validation ordinal (CORAL) loss"},
        "ablation": _ablation_switches(cfg.model),
        "gradient_checkpointing": getattr(cfg, "gradient_checkpointing", None),
        "optimizer": {"type": "AdamW", "lr_head": cfg.lr_head, "lr_backbone": cfg.lr_backbone,
                     "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
                     "epochs": cfg.epochs, "patience": cfg.patience,
                     "grad_clip_norm": cfg.grad_clip, "seed": cfg.seed},
        "run_name": cfg.run_name,
        "model": cfg.model,
        "task": cfg.task,
        "provenance": {
            "git_commit": _git_commit_hash(),
            "software_versions": _software_versions(),
        },
    }


def print_final_run_configuration(cfg, df: pd.DataFrame) -> Dict:
    """Print the brief's Section 23 FINAL RUN CONFIGURATION block. Call
    once, immediately before the real training run — everything printed is
    also what write_frozen_config() persists."""
    from src.console import print_header, print_kv, print_subheader

    num_speakers_total = int(df["Speaker_ID"].nunique()) if "Speaker_ID" in df.columns else 0
    final_config = build_final_run_configuration(cfg, df, num_speakers_total)

    print_header("Final run configuration")
    print_kv("Run name", final_config["run_name"])
    print_kv("Model / task", f"{final_config['model']} / {final_config['task']}")
    prov = final_config["provenance"]
    versions = prov["software_versions"]
    print_kv("Git commit", (prov["git_commit"] or "unavailable")[:12])
    print_kv("Software", ", ".join(f"{name} {version}" for name, version in versions.items()
                                   if version))

    d = final_config["dataset"]
    print_subheader("Data and protocol")
    print_kv("Severity protocol", f"{d['severity_protocol']}, {d['num_folds']} folds")
    print_kv("Fold order", ", ".join(d["fold_speakers"]))
    print_kv("Classes", ", ".join(d["severity_classes"]))
    a = final_config["audio"]
    print_kv("Audio", f"{a['sampling_rate']} Hz, {a['clip_seconds']:g} s window "
                      f"({a['max_samples']:,} samples)")
    print_kv("VAD padding (speech / temporal)", f"{a['speech_focused_vad_pad_ms']} ms / "
                                                f"{a['temporal_preserving_vad_pad_ms']} ms")
    v = final_config["validation"]
    print_kv("Validation", f"{v['protocol']}-disjoint; early stopping on {v['monitored']}")

    print_subheader("Architecture")
    lb = final_config["learned_branch"]
    print_kv("Learned branch", f"{lb['wav2vec2_model']} + LoRA r={lb['lora_rank']}, "
                               f"alpha={lb['lora_alpha']}, dropout={lb['lora_dropout']}")
    print_kv("  LoRA targets / projection", f"{', '.join(lb['lora_target_modules'])} / "
                                           f"{config.WAV2VEC_EMBED_DIM} -> {lb['dimensions']}")
    sb = final_config["segmental_branch"]
    print_kv("Segmental branch", f"{sb['input_channels']} ch -> {sb['dimensions']}")
    pb = final_config["suprasegmental_branch"]
    print_kv("Suprasegmental branch", f"{pb['input_channels']} ch -> {pb['dimensions']}")
    f = final_config["fusion"]
    print_kv("Fusion", f"{f['method']} ({f['fused_dim']}-dim)")
    sh = final_config["severity_head"]
    print_kv("Severity head", f"{sh['method']}; decode = {sh['decoding']}")
    c, s = final_config["complementarity"], final_config["speaker_invariance"]
    print_kv("Complementarity penalty", f"lambda {c['lambda']}")
    print_kv("Speaker adversary (GRL)", f"lambda {s['lambda']}, GRL strength {s['grl_lambda']}")
    if final_config["ablation"]:
        print_kv("Ablation switches", final_config["ablation"])

    o = final_config["optimizer"]
    print_subheader("Optimisation")
    print_kv("Optimizer", f"{o['type']}, lr {o['lr_head']:g} (head) / {o['lr_backbone']:g} "
                          f"(backbone), weight decay {o['weight_decay']:g}")
    print_kv("Batch size / max epochs / patience", f"{o['batch_size']} / {o['epochs']} / "
                                                   f"{o['patience']}")
    print_kv("Grad clip / seed / grad checkpointing",
             f"{o['grad_clip_norm']} / {o['seed']} / {final_config['gradient_checkpointing']}")
    return final_config


def _config_hash(final_config: Dict) -> str:
    """Stable hash of a FINAL RUN CONFIGURATION dict — used only to detect
    "this run_name was already frozen with a DIFFERENT configuration", not
    for anything security-sensitive.

    Provenance (git commit, package versions) is recorded but NOT hashed: it
    describes the environment, not the experiment, and hashing it made every
    commit between sessions refuse to resume a half-finished run."""
    experiment = {key: value for key, value in final_config.items() if key != "provenance"}
    payload = json.dumps(experiment, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_frozen_config(final_config: Dict, path: Optional[Path] = None) -> Path:
    """
    Persist the FINAL RUN CONFIGURATION, with its hash, to
    outputs/results/frozen_config.json — call once, after
    print_final_run_configuration() and before the real training run.
    Architecture plan Part 4, step 4 / brief Section 23's "freeze the
    configuration" instruction.
    """
    path = Path(path or FROZEN_CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": final_config, "config_hash": _config_hash(final_config),
              "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def check_frozen_config_guard(final_config: Dict, path: Optional[Path] = None) -> None:
    """
    Minimal code-level backstop for the one-shot-training rule (brief
    Section 23/25): if a frozen_config.json already exists for this
    run_name's configuration and its hash does NOT match `final_config`,
    raise — refusing to silently retrain a FINAL-tier run under an unchanged
    run_name with different hyperparameters. A no-op if no frozen config has
    been written yet (the normal case, before the one real run) or if the
    hash matches exactly (a legitimate resume of the SAME configuration).

    This is deliberately narrow: it protects against the specific failure
    mode of quietly re-running the one-shot experiment with a tweaked
    config under the same name, not a general hyperparameter-search guard —
    see the architecture plan's Part 2, Component 16 for why a heavier
    mechanism was judged unnecessary on top of the existing experiment
    registry (src.results / tests/test_experiment_validity.py).
    """
    path = Path(path or FROZEN_CONFIG_PATH)
    if not path.exists():
        return
    with open(path) as f:
        frozen = json.load(f)
    if frozen.get("config", {}).get("run_name") != final_config.get("run_name"):
        return
    if frozen.get("config_hash") != _config_hash(final_config):
        raise RuntimeError(
            f"A frozen configuration already exists at {path} for run_name "
            f"'{final_config.get('run_name')}' with a DIFFERENT configuration "
            "hash. The one-shot training rule forbids re-running this "
            "experiment under the same name with changed hyperparameters — "
            "use a new run_name if this is a deliberate, separate run, or "
            "delete the frozen config file if the earlier freeze was itself "
            "a mistake made before any real training happened.")
