"""
Per-fold and cross-fold output writers: predictions, metrics, confusion
matrices, ROC curves, and embeddings — everything train.py drops into
outputs/ for downstream phases (ablation tables, error analysis, Praat
correlation, embedding visualization).
"""

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve

from src import config


def save_predictions(path: Path, filenames, speaker_ids, y_true: np.ndarray,
                     y_pred: np.ndarray, y_prob: np.ndarray, task: str) -> None:
    """Per-utterance predictions for one fold.

    `filename` is the first column and is what makes Phase 5 possible: it joins
    a row back to its audio file (for spectrograms) and to outputs/praat_features.csv
    (for the error/feature correlation). speaker_id alone cannot do either.
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
    if task == "detection":
        df["prob_positive"] = y_prob
    else:
        for i, name in enumerate(class_names):
            df[f"prob_{name.replace(' ', '_')}"] = y_prob[:, i]
    df.to_csv(path, index=False)


def save_metrics(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


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
                    speaker_ids, filenames=None) -> None:
    """Test-fold embeddings, keyed by filename so Phase 5's embedding map can be
    coloured by whether each point was classified correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "embeddings": embeddings,
        "y_true": y_true,
        "speaker_ids": np.asarray(speaker_ids),
    }
    if filenames is not None:
        arrays["filenames"] = np.asarray(filenames)
    np.savez(path, **arrays)


def aggregate_fold_metrics(metrics_dir: Path, run_name: str,
                           fold_metrics: List[Dict]) -> pd.DataFrame:
    """Average metrics across folds (mean +/- std) and save a summary table."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(fold_metrics)
    metric_cols = [c for c in df.columns if c != "fold"]
    summary = df[metric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    df.to_csv(metrics_dir / f"{run_name}.per_fold.csv", index=False)
    summary.to_csv(metrics_dir / f"{run_name}.summary.csv")
    with open(metrics_dir / f"{run_name}.summary.json", "w") as f:
        json.dump(summary.to_dict(orient="index"), f, indent=2)
    return summary


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
