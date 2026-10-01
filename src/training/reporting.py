"""
Artifacts and reports of a training run.

  per fold   predictions CSV, metrics JSON (the fold-finished marker), confusion
             matrix, embeddings (.npz: fused, per-branch, gate weights)
  per run    RUN_STATUS.json, pooled metrics, pooled confusion matrix + ROC,
             and the frozen configuration with its guard
  notebook   collect_run_results() rebuilds every table from disk, so it is
             correct after a complete, partial or interrupted run.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                    # file-only; the notebook switches back to inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src import config
from src.console import print_header, print_kv, print_note, print_status, print_subheader
from src.training.checkpoint import replace_with_retry
from src.training.metrics import compute_confusion_matrix, compute_metrics

CLASS_NAMES = config.SEVERITY_CLASS_NAMES
PROB_COLUMNS = [f"prob_{name.replace(' ', '_')}" for name in CLASS_NAMES]

# Per-fold outcomes.
FOLD_COMPLETED = "COMPLETED"                # trained in this session
FOLD_CACHED = "CACHED"                      # finished in an earlier session, loaded from disk
FOLD_FAILED = "FAILED"                      # raised on every attempt
FOLD_SKIPPED_DEADLINE = "SKIPPED_DEADLINE"  # not started: the deadline would not allow it
FOLD_INTERRUPTED = "INTERRUPTED"            # stopped between epochs; resumes from latest.pt

RUN_STATUS_COMPLETE, RUN_STATUS_PARTIAL, RUN_STATUS_FAILED = "COMPLETE", "PARTIAL", "FAILED"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    write(temp_path)
    replace_with_retry(temp_path, path)


def save_predictions(path: Path, filenames, speaker_ids, y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray, y_pred_argmax: np.ndarray) -> None:
    """One row per utterance; `filename` joins it back to its audio."""
    frame = pd.DataFrame({
        "filename": filenames, "speaker_id": speaker_ids,
        "y_true": y_true, "y_true_label": [CLASS_NAMES[i] for i in y_true],
        "y_pred": y_pred, "y_pred_label": [CLASS_NAMES[i] for i in y_pred],
        "correct": np.asarray(y_true) == np.asarray(y_pred),
        "y_pred_argmax": y_pred_argmax,
        **{column: y_prob[:, i] for i, column in enumerate(PROB_COLUMNS)},
    })
    _atomic_write(path, lambda temp: frame.to_csv(temp, index=False))


def save_metrics(path: Path, metrics: Dict) -> None:
    """Atomic: with the predictions CSV, this file marks a fold as finished."""
    def write(temp):
        with open(temp, "w") as handle:
            json.dump(metrics, handle, indent=2, default=str)
    _atomic_write(path, write)


def save_confusion_matrix(path: Path, cm: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_curve(path: Path, y_true: np.ndarray, y_prob: np.ndarray, title: str) -> None:
    """One-vs-rest ROC per class present with both outcomes (pooled sets only:
    a single LOSO fold holds one class, so it has no curve)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    drawn = False
    for i, name in enumerate(CLASS_NAMES):
        binary = (y_true == i).astype(int)
        if 0 < binary.sum() < len(binary):
            fpr, tpr, _ = roc_curve(binary, y_prob[:, i])
            ax.plot(fpr, tpr, label=name)
            drawn = True
    if drawn:
        path.parent.mkdir(parents=True, exist_ok=True)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title, fontsize=9)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    plt.close(fig)


def save_embeddings(path: Path, embeddings: Dict[str, np.ndarray], y_true: np.ndarray,
                    speaker_ids, filenames) -> None:
    """Test-fold embeddings: "embeddings" (fused 256-D), "gate_weights" (N, 3)
    and "branch_<name>" per present branch, keyed by filename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"embeddings": embeddings["fused"], "gate_weights": embeddings["gates"],
              "y_true": y_true, "speaker_ids": np.asarray(speaker_ids),
              "filenames": np.asarray(filenames)}
    arrays.update({f"branch_{name}": values for name, values in embeddings.items()
                   if name not in ("fused", "gates")})
    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# Coverage and console reports
# ---------------------------------------------------------------------------
def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    hours, remainder = divmod(int(round(max(0.0, float(seconds)))), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def _fmt_metric(value) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.3f}"


def run_coverage(expected_fold_ids: List[str], fold_status: Dict[str, str]) -> Dict:
    """Which of the configured folds produced a result, failed, were skipped
    or interrupted — and the COMPLETE / PARTIAL / FAILED status."""
    expected = list(expected_fold_ids)
    done = [f for f in expected if fold_status.get(f) in (FOLD_COMPLETED, FOLD_CACHED)]
    interrupted = [f for f in expected if fold_status.get(f) == FOLD_INTERRUPTED]
    status = (RUN_STATUS_COMPLETE if expected and len(done) == len(expected)
              else RUN_STATUS_PARTIAL if done else RUN_STATUS_FAILED)
    return {
        "expected": len(expected), "completed": len(done), "status": status,
        "trained_this_session": sum(fold_status.get(f) == FOLD_COMPLETED for f in expected),
        "loaded_from_disk": sum(fold_status.get(f) == FOLD_CACHED for f in expected),
        "completion_pct": round(100.0 * len(done) / len(expected), 1) if expected else 0.0,
        "expected_folds": expected, "completed_folds": done,
        "missing_folds": [f for f in expected if f not in done],
        "failed_folds": [f for f in expected if fold_status.get(f) == FOLD_FAILED],
        "skipped_folds": [f for f in expected
                          if fold_status.get(f) in (FOLD_SKIPPED_DEADLINE, FOLD_INTERRUPTED)],
        "interrupted_folds": interrupted,
    }


def print_run_coverage(coverage: Dict, run_name: str) -> None:
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
        print_note(f"Re-run the training cell to finish the {len(coverage['missing_folds'])} "
                   "missing fold(s); finished folds load from disk. Pooled numbers are partial "
                   "until then.")


def print_fold_report(record: Dict) -> None:
    """Two lines per finished fold. One held-out speaker has one true class,
    so only accuracy and ordinal MAE are defined per fold."""
    best = record.get("best_epoch")
    print(f"  Result     {record['fold']} ({record.get('true_label')}): "
          f"accuracy {_fmt_metric(record.get('accuracy'))} · ordinal MAE "
          f"{_fmt_metric(record.get('ordinal_mae'))} · "
          + (f"best epoch {best} of {record.get('epochs_completed')} run" if best is not None
             else "no improving epoch")
          + f" · fold time {_fmt_duration(record.get('fold_time_s'))}")
    print("  Predicted  " + " · ".join(f"{name} {count}" for name, count
                                       in (record.get("pred_distribution") or {}).items()))
    if "coral_thresholds" in record and not record.get("coral_thresholds_ordered"):
        print_note(f"CORAL threshold biases {record['coral_thresholds']} are not rank-ordered; "
                   "the median decode may differ from the raw threshold count here.")


def print_runtime_status(folds_done: int, n_folds: int, remaining_folds: int,
                         elapsed_s: float, fold_estimate_s: Optional[float],
                         deadline_in_s: Optional[float] = None,
                         safety_factor: float = 1.15) -> None:
    """One progress line after every fold."""
    parts = [f"{folds_done}/{n_folds} folds done", f"elapsed {_fmt_duration(elapsed_s)}"]
    if fold_estimate_s is not None and remaining_folds:
        remaining_s = remaining_folds * fold_estimate_s
        parts += [f"mean fold {_fmt_duration(fold_estimate_s)}",
                  f"about {_fmt_duration(remaining_s)} to go"]
        if deadline_in_s is not None:
            parts.append("ON TRACK" if deadline_in_s >= remaining_s * safety_factor
                         else "AT RISK" if deadline_in_s >= remaining_s
                         else "WILL NOT FIT — later folds will be skipped")
    print("  Progress   " + " · ".join(parts))


# ---------------------------------------------------------------------------
# Pooled results, rebuilt from disk
# ---------------------------------------------------------------------------
def speaker_level(predictions: pd.DataFrame) -> pd.DataFrame:
    """One decision per speaker — the median of its utterance predictions,
    since severity is a speaker-level clinical label."""
    grouped = predictions.groupby("speaker_id").agg(
        true=("y_true", "first"), pred=("y_pred", lambda v: int(np.floor(np.median(v)))),
        accuracy=("correct", "mean"))
    return pd.DataFrame({
        "Speaker": grouped.index,
        "True class": [CLASS_NAMES[int(t)] for t in grouped["true"]],
        "Speaker-level prediction": [CLASS_NAMES[int(p)] for p in grouped["pred"]],
        "Correct": grouped["true"].to_numpy() == grouped["pred"].to_numpy(),
        "Rank error": (grouped["pred"] - grouped["true"]).abs().to_numpy(),
        "Utterance accuracy": grouped["accuracy"].to_numpy(),
    }).reset_index(drop=True)


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Per class: counts, precision, recall, F1, one-vs-rest AUROC — NaN where
    undefined (never predicted, or absent from the pooled set)."""
    cm = compute_confusion_matrix(y_true, y_pred)
    true_counts, pred_counts = cm.sum(axis=1), cm.sum(axis=0)
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        precision = cm[i, i] / pred_counts[i] if pred_counts[i] else float("nan")
        recall = cm[i, i] / true_counts[i] if true_counts[i] else float("nan")
        if not true_counts[i]:
            f1 = float("nan")
        elif np.isfinite(precision) and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        binary = (y_true == i).astype(int)
        auroc = (float(roc_auc_score(binary, y_prob[:, i])) if 0 < binary.sum() < len(binary)
                 else float("nan"))
        rows.append({"class": name, "true_n": int(true_counts[i]), "pred_n": int(pred_counts[i]),
                     "precision": precision, "recall": recall, "f1": f1, "auroc_ovr": auroc})
    return pd.DataFrame(rows)


def collect_run_results(run_name: str, expected_folds: List[str]) -> Dict[str, object]:
    """Every results table, from the per-fold files on disk:
    coverage, per_fold, pooled, per_class, confusion, speakers, n_utterances."""
    rows, frames = [], []
    for fold_id in expected_folds:
        metrics_path = config.METRICS_DIR / run_name / f"{fold_id}.json"
        predictions_path = config.PREDICTIONS_DIR / run_name / f"{fold_id}.csv"
        if not (metrics_path.exists() and predictions_path.exists()):
            continue
        with open(metrics_path) as handle:
            m = json.load(handle)
        preds = pd.read_csv(predictions_path)
        counts = np.bincount(preds["y_pred"], minlength=len(CLASS_NAMES))
        rows.append({"Fold": fold_id, "True class": m.get("true_label"),
                     "Utterances": len(preds), "Accuracy": m.get("accuracy"),
                     "Ordinal MAE": m.get("ordinal_mae"),
                     **{f"Pred {name}": int(c) for name, c in zip(CLASS_NAMES, counts)},
                     "Best epoch": m.get("best_epoch"), "Epochs run": m.get("epochs_completed"),
                     "Train (min)": (m.get("train_time_s") or float("nan")) / 60,
                     "Val speakers": str(m.get("val_speakers", "")).replace(";", ", ")})
        frames.append(preds)

    completed = [r["Fold"] for r in rows]
    status_path = config.METRICS_DIR / run_name / "RUN_STATUS.json"
    failed = []
    if status_path.exists():
        with open(status_path) as handle:
            failed = [f for f in json.load(handle).get("failed_folds", []) if f not in completed]
    coverage = {"expected": len(expected_folds), "completed": len(completed),
                "completion_pct": round(100.0 * len(completed) / max(len(expected_folds), 1), 1),
                "missing_folds": [f for f in expected_folds if f not in completed],
                "failed_folds": failed,
                "status": (RUN_STATUS_COMPLETE if expected_folds and len(completed) == len(expected_folds)
                           else RUN_STATUS_PARTIAL if completed else RUN_STATUS_FAILED)}
    results: Dict[str, object] = {"coverage": coverage, "per_fold": pd.DataFrame(rows)}
    if not frames:
        return results

    preds = pd.concat(frames, ignore_index=True)
    y_true, y_pred = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    y_prob = preds[PROB_COLUMNS].to_numpy()
    metrics = compute_metrics(y_true, y_pred, y_prob)
    absent = [n for i, n in enumerate(CLASS_NAMES) if not (y_true == i).any()]
    reason = ("undefined: fewer than 2 classes pooled" if metrics["n_classes_present"] < 2
              else f"undefined: no held-out {', '.join(absent)} yet" if absent else "")
    pooled = []
    for key, label in (("accuracy", "Accuracy"), ("f1", "Macro F1"), ("f1_weighted", "Weighted F1"),
                       ("balanced_accuracy", "Balanced accuracy"), ("ordinal_mae", "Ordinal MAE"),
                       ("auroc", "AUROC (macro one-vs-rest)")):
        value = metrics[key]
        defined = np.isfinite(value)
        note = "" if defined else reason
        if defined and absent and key in ("f1", "balanced_accuracy"):
            note = f"over the {len(CLASS_NAMES) - len(absent)} classes present"
        pooled.append({"Metric": label, "Value": value if defined else float("nan"), "Note": note})
    cm = compute_confusion_matrix(y_true, y_pred)
    results.update({
        "pooled": pd.DataFrame(pooled), "n_utterances": int(len(y_true)),
        "per_class": per_class_table(y_true, y_pred, y_prob),
        "confusion": pd.DataFrame(cm, index=[f"True {n}" for n in CLASS_NAMES],
                                  columns=[f"Pred {n}" for n in CLASS_NAMES]),
        "speakers": speaker_level(preds),
    })
    return results


# ---------------------------------------------------------------------------
# Feature audit and the frozen run configuration
# ---------------------------------------------------------------------------
def print_feature_audit(model) -> Dict[str, List[int]]:
    """Every branch's input -> embedding shape, read from one dummy forward
    pass of the real model rather than restated by hand."""
    from src.preprocessing import mfcc_frame_count

    frames = mfcc_frame_count(config.MAX_SAMPLES)
    device = next(model.parameters()).device
    batch = 2
    model.eval()
    with torch.no_grad():
        branches = model.encode_branches(
            torch.zeros(batch, config.MAX_SAMPLES, device=device),
            torch.zeros(batch, config.SEGMENTAL_CHANNELS, frames, device=device),
            torch.zeros(batch, config.SUPRA_CHANNELS, frames, device=device),
            torch.ones(batch, config.MAX_SAMPLES, dtype=torch.bool, device=device),
            torch.full((batch,), frames, dtype=torch.long, device=device))
        fused, _ = model.fuse(*branches)
    inputs = {"learned": f"[B x {config.MAX_SAMPLES}] waveform -> wav2vec2 [B x T x "
                         f"{config.WAV2VEC_EMBED_DIM}] -> pooled",
              "segmental": f"[B x {config.SEGMENTAL_CHANNELS} x {frames}] MFCC+d+dd, F1-F3, HNR",
              "supra": f"[B x {config.SUPRA_CHANNELS} x {frames}] F0, voicing, intensity"}
    shapes = {}
    print_subheader("Feature audit (shapes from a dummy forward pass)")
    for name, z in zip(("learned", "segmental", "supra"), branches):
        if z is not None:
            shapes[name] = list(z.shape)
            print_kv(name.capitalize(), f"{inputs[name]} -> [{' x '.join(map(str, z.shape))}]")
    shapes["fused"] = list(fused.shape)
    print_kv("Fused representation", " + ".join(str(s[-1]) for k, s in shapes.items() if k != "fused")
             + f" = {fused.shape[-1]}")
    return shapes


def _git_commit_hash() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT,
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def _software_versions() -> Dict[str, Optional[str]]:
    import platform
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for module_name in ("torch", "torchaudio", "transformers", "peft", "numpy", "pandas"):
        try:
            versions[module_name] = getattr(__import__(module_name), "__version__", None)
        except Exception:
            versions[module_name] = None
    versions["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    return versions


def build_final_run_configuration(cfg) -> Dict:
    """Everything that defines the experiment, from `cfg` (a TrainingConfig)
    and src.config. Provenance (git commit, versions) is recorded but not
    hashed by the guard."""
    from src.training.models import model_switches

    fold_speakers = list(config.SEVERITY_LOSO_ORDER)
    if cfg.max_folds is not None:
        fold_speakers = fold_speakers[:cfg.max_folds]
    return {
        "run_name": cfg.run_name, "model": cfg.model,
        "ablation_switches": {k: list(v) if isinstance(v, tuple) else v
                              for k, v in model_switches(cfg.model).items()} or None,
        "dataset": {"severity_classes": CLASS_NAMES, "protocol": "severity LOSO",
                    "fold_speakers": fold_speakers, "num_folds": len(fold_speakers),
                    "limit_samples": cfg.limit_samples},
        "audio": {"sampling_rate": config.TARGET_SR, "clip_seconds": config.CLIP_SECONDS,
                  "speech_focused_vad_pad_ms": config.VAD_SPEECH_PAD_MS,
                  "temporal_preserving_vad_pad_ms": config.SUPRA_VAD_SPEECH_PAD_MS},
        "architecture": {
            "wav2vec2_model": config.WAV2VEC_MODEL_NAME,
            "lora": {"rank": config.LORA_RANK, "alpha": config.LORA_ALPHA,
                     "dropout": config.LORA_DROPOUT, "targets": config.LORA_TARGET_MODULES},
            "embedding_dims": {"learned": config.LEARNED_EMBED_DIM,
                               "segmental": config.SEGMENTAL_EMBED_DIM,
                               "supra": config.SUPRA_EMBED_DIM},
            "input_channels": {"segmental": config.SEGMENTAL_CHANNELS,
                               "supra": config.SUPRA_CHANNELS},
            "lambda_comp": config.LAMBDA_COMP, "lambda_speaker": config.LAMBDA_SPEAKER,
            "grl_lambda": config.GRL_LAMBDA,
            "severity_head": "CORAL, median decode",
        },
        "validation": {"protocol": "speaker-disjoint, one speaker per eligible class",
                       "monitored": "validation ordinal (CORAL) loss"},
        "optimizer": {"type": "AdamW", "lr_head": cfg.lr_head, "lr_lora": cfg.lr_lora,
                      "weight_decay": cfg.weight_decay, "batch_size": cfg.batch_size,
                      "epochs": cfg.epochs, "patience": cfg.patience,
                      "grad_clip_norm": cfg.grad_clip, "seed": cfg.seed,
                      "gradient_checkpointing": cfg.gradient_checkpointing},
        "provenance": {"git_commit": _git_commit_hash(), "software_versions": _software_versions()},
    }


def print_final_run_configuration(cfg) -> Dict:
    final = build_final_run_configuration(cfg)
    a, o, d = final["architecture"], final["optimizer"], final["dataset"]
    print_header("Final run configuration")
    print_kv("Run name / model", f"{final['run_name']} / {final['model']}")
    if final["ablation_switches"]:
        print_kv("Ablation switches", final["ablation_switches"])
    print_kv("Git commit", (final["provenance"]["git_commit"] or "unavailable")[:12])
    print_kv("Software", ", ".join(f"{k} {v}" for k, v in
                                   final["provenance"]["software_versions"].items() if v))
    print_kv("Folds", f"{d['num_folds']}: {', '.join(d['fold_speakers'])}")
    print_kv("Learned branch", f"{a['wav2vec2_model']} + LoRA r={a['lora']['rank']}, "
                               f"alpha={a['lora']['alpha']} on {', '.join(a['lora']['targets'])}")
    print_kv("Loss weights", f"complementarity {a['lambda_comp']}, speaker {a['lambda_speaker']} "
                             f"(GRL {a['grl_lambda']})")
    print_kv("Optimizer", f"AdamW, lr {o['lr_head']:g} (head) / {o['lr_lora']:g} (LoRA), "
                          f"weight decay {o['weight_decay']:g}")
    print_kv("Batch / max epochs / patience", f"{o['batch_size']} / {o['epochs']} / {o['patience']}")
    print_kv("Validation", f"{final['validation']['protocol']}; early stopping on "
                           f"{final['validation']['monitored']}")
    return final


def frozen_config_path(run_name: str) -> Path:
    return config.RESULTS_DIR / run_name / "frozen_config.json"


def _config_hash(final_config: Dict) -> str:
    experiment = {k: v for k, v in final_config.items() if k != "provenance"}
    return hashlib.sha256(json.dumps(experiment, sort_keys=True, default=str).encode()).hexdigest()


def write_frozen_config(final_config: Dict, path: Optional[Path] = None) -> Path:
    """Persist the configuration and its hash, per run name."""
    path = Path(path or frozen_config_path(final_config["run_name"]))
    payload = {"config": final_config, "config_hash": _config_hash(final_config),
               "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    def write(temp):
        with open(temp, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
    _atomic_write(path, write)
    return path


def check_frozen_config_guard(final_config: Dict, path: Optional[Path] = None) -> None:
    """Raise if this run name was already frozen with a DIFFERENT
    configuration: resuming is allowed, silently changing a run is not. Use a
    new run name for a deliberate change."""
    path = Path(path or frozen_config_path(final_config["run_name"]))
    if not path.exists():
        return
    with open(path) as handle:
        frozen = json.load(handle)
    if frozen.get("config_hash") != _config_hash(final_config):
        raise RuntimeError(
            f"{path} froze run '{final_config['run_name']}' with a different configuration. "
            "Use a new run name for a deliberate change, or delete that file if the earlier "
            "freeze was a mistake made before any fold finished.")
