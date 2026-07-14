"""
Training orchestration: fold iteration, per-fold train/validate/checkpoint,
and cross-fold aggregation.

This is the function notebooks/02_training.ipynb calls — it is reusable
logic (the same loop drives every model variant and both tasks), not a
one-off analysis step, so it lives in src/ rather than the notebook. The
notebook supplies a TrainingConfig and reads results; it never contains
training logic itself.
"""

import json
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from src import config
from src.console import (V, print_architecture, print_banner, print_fold_progress,
                        print_header, print_kv, print_metrics, print_note,
                        print_signal_chain, print_status, print_subheader, print_table)
from src.praat import FEATURE_COLUMNS as PRAAT_FEATURE_COLUMNS
from src.praat import load_praat_table
from src.splits import build_severity_folds, get_severity_split, iter_loso_folds
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.data import (TASK_LABEL_COLUMN, build_loaders,
                               compute_class_weights, stratified_train_val_split)
from src.training.early_stopping import EarlyStopping
from src.training.engine import EpochResult, build_optimizer, run_epoch
from src.training.metrics import compute_confusion_matrix, compute_metrics
from src.training.models import (MODEL_DESCRIPTIONS, MODELS_REQUIRING_PRAAT,
                                 build_model)
from src.training.reporting import (aggregate_fold_metrics, save_confusion_matrix,
                                    save_embeddings, save_metrics, save_predictions,
                                    save_roc_curve)
from src.training.utils import resolve_device, set_seed


@dataclass
class TrainingConfig:
    """Everything one run_training() call needs. Defaults come from src.config."""
    task: str = "detection"                        # "detection" or "severity"
    model: str = "fusion"                           # see src.training.models.MODEL_NAMES
    run_name: Optional[str] = None                  # defaults to "<task>_<model>"

    epochs: int = config.DEFAULT_EPOCHS
    batch_size: int = config.DEFAULT_BATCH_SIZE
    lr_head: float = config.DEFAULT_LR_HEAD
    lr_backbone: float = config.DEFAULT_LR_BACKBONE
    weight_decay: float = config.DEFAULT_WEIGHT_DECAY
    patience: int = config.DEFAULT_PATIENCE          # early stopping, epochs w/o val-loss improvement
    grad_clip: float = config.DEFAULT_GRAD_CLIP_NORM
    val_fraction: float = config.DEFAULT_VAL_FRACTION
    seed: int = config.DEFAULT_SEED

    amp: Optional[bool] = None                       # None = on iff device is CUDA
    device: Optional[str] = None                     # None = auto (cuda if available)
    num_workers: int = 0

    max_folds: Optional[int] = None                  # only run the first N folds
    folds: Optional[List[str]] = None                # only run these fold IDs
    limit_samples: Optional[int] = None               # cap rows per split — smoke testing only


def build_folds(df: pd.DataFrame, task: str):
    """Yield (fold_id, train_df, test_df) for the requested task's protocol."""
    if task == "detection":
        yield from iter_loso_folds(df)
    else:
        for combo in build_severity_folds(df):
            train_df, test_df = get_severity_split(df, combo)
            yield "-".join(combo), train_df, test_df


def _limit_samples(df: pd.DataFrame, n: Optional[int], label_column: str) -> pd.DataFrame:
    """Cap a split to ~n rows for smoke testing, keeping every class present."""
    if n is None or len(df) <= n:
        return df
    per_class = max(1, n // max(df[label_column].nunique(), 1))
    parts = [group.head(per_class) for _, group in df.groupby(label_column)]
    return pd.concat(parts).reset_index(drop=True)


def _load_completed_fold(run_name: str, fold_id: str, task: str
                         ) -> Optional[Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    If `fold_id` already has metrics + predictions on disk from a prior
    (interrupted) run of this exact `run_name`, load them instead of
    retraining — this is what lets a multi-hour/multi-day run resume after a
    crash without redoing already-finished folds. Returns None if either
    file is missing or unreadable, in which case the caller retrains normally.
    """
    metrics_path = config.METRICS_DIR / run_name / f"{fold_id}.json"
    predictions_path = config.PREDICTIONS_DIR / run_name / f"{fold_id}.csv"
    if not (metrics_path.exists() and predictions_path.exists()):
        return None

    try:
        with open(metrics_path) as f:
            metrics_dict = json.load(f)
        preds = pd.read_csv(predictions_path)
        y_true = preds["y_true"].to_numpy()
        y_pred = preds["y_pred"].to_numpy()
        if task == "detection":
            y_prob = preds["prob_positive"].to_numpy()
        else:
            class_names = config.SEVERITY_CLASS_NAMES
            prob_cols = [f"prob_{name.replace(' ', '_')}" for name in class_names]
            y_prob = preds[prob_cols].to_numpy()
        speakers = preds["speaker_id"].tolist()
    except Exception:
        return None

    return metrics_dict, y_true, y_pred, y_prob, speakers


def run_fold(fold_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
            cfg: TrainingConfig, device: torch.device, run_name: str,
            praat_table: Optional[pd.DataFrame] = None,
            fold_index: int = 1, n_folds: int = 1
            ) -> Tuple[Dict, EpochResult]:
    """Train, validate, checkpoint, and test-evaluate one fold. Returns
    (metrics_dict, test_result) — the caller pools test_result across
    folds for the cross-fold metrics."""
    label_column = TASK_LABEL_COLUMN[cfg.task]
    num_classes = config.NUM_CLASSES[cfg.task]

    train_df, val_df = stratified_train_val_split(
        train_df, label_column, cfg.val_fraction, cfg.seed)

    if cfg.limit_samples is not None:
        train_df = _limit_samples(train_df, cfg.limit_samples, label_column)
        val_df = _limit_samples(val_df, max(2, cfg.limit_samples // 4), label_column)
        test_df = _limit_samples(test_df, cfg.limit_samples, label_column)

    train_loader, val_loader, test_loader = build_loaders(
        train_df, val_df, test_df, cfg.batch_size, cfg.num_workers,
        pin_memory=(device.type == "cuda"), praat_table=praat_table)

    model = build_model(cfg.model, num_classes).to(device)
    optimizer = build_optimizer(model, cfg.lr_head, cfg.lr_backbone, cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=max(1, cfg.patience // 2))
    class_weights = compute_class_weights(train_df, cfg.task).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = cfg.amp if cfg.amp is not None else (device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    early_stopping = EarlyStopping(patience=cfg.patience, mode="min")

    log_dir = config.LOG_DIR / run_name / fold_id
    writer = SummaryWriter(log_dir=str(log_dir))
    best_ckpt_path = config.CHECKPOINT_DIR / run_name / fold_id / "best.pt"

    print_fold_progress(fold_id, fold_index, n_folds,
                        len(train_df), len(val_df), len(test_df))
    if fold_index == 1:
        print_architecture(model, cfg.model)
        print()

    for epoch in range(cfg.epochs):
        train_result = run_epoch(model, train_loader, criterion, optimizer, device,
                                 scaler, cfg.grad_clip, cfg.task, train=True,
                                 description=f"epoch {epoch + 1}/{cfg.epochs} train")
        val_result = run_epoch(model, val_loader, criterion, None, device,
                               scaler, cfg.grad_clip, cfg.task, train=False,
                               description=f"epoch {epoch + 1}/{cfg.epochs} val")
        scheduler.step(val_result.loss)

        writer.add_scalar("Loss/train", train_result.loss, epoch)
        writer.add_scalar("Loss/val", val_result.loss, epoch)
        for name, value in train_result.metrics.items():
            writer.add_scalar(f"Train/{name}", value, epoch)
        for name, value in val_result.metrics.items():
            writer.add_scalar(f"Val/{name}", value, epoch)
        writer.add_scalar("LR", optimizer.param_groups[-1]["lr"], epoch)

        is_best = early_stopping.step(val_result.loss)
        if is_best:
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, scaler,
                            epoch, val_result.loss)

        print(f"    epoch {epoch + 1:>3}/{cfg.epochs}  {V}  "
             f"train  loss {train_result.loss:.4f}  acc {train_result.metrics['accuracy']:.3f}  {V}  "
             f"val  loss {val_result.loss:.4f}  acc {val_result.metrics['accuracy']:.3f}  "
             f"f1 {val_result.metrics['f1']:.3f}" + ("   <-- best" if is_best else ""))

        if early_stopping.should_stop:
            print(f"    early stopping at epoch {epoch + 1} "
                 f"(no val-loss improvement for {cfg.patience} epochs)")
            break

    if best_ckpt_path.exists():
        load_checkpoint(best_ckpt_path, model, map_location=str(device))

    test_result = run_epoch(model, test_loader, criterion, None, device, scaler,
                            cfg.grad_clip, cfg.task, train=False, collect_embeddings=True,
                            description=f"held-out test ({fold_id})")

    save_predictions(config.PREDICTIONS_DIR / run_name / f"{fold_id}.csv",
                     test_result.filenames, test_result.speaker_ids, test_result.y_true,
                     test_result.y_pred, test_result.y_prob, cfg.task)
    save_metrics(config.METRICS_DIR / run_name / f"{fold_id}.json",
                {"fold": fold_id, "test_loss": test_result.loss, **test_result.metrics})
    save_confusion_matrix(
        config.CONFUSION_MATRIX_DIR / run_name / f"{fold_id}.png",
        compute_confusion_matrix(test_result.y_true, test_result.y_pred, cfg.task),
        cfg.task, title=f"{run_name} — fold {fold_id}")
    save_roc_curve(config.ROC_DIR / run_name / f"{fold_id}.png",
                   test_result.y_true, test_result.y_prob, cfg.task,
                   title=f"{run_name} — fold {fold_id}")
    save_embeddings(config.EMBEDDINGS_DIR / run_name / f"{fold_id}.npz",
                    test_result.embeddings, test_result.y_true, test_result.speaker_ids,
                    test_result.filenames)

    writer.close()
    print_kv(f"Fold {fold_id} held-out test", ", ".join(
        f"{k}={v:.3f}" for k, v in test_result.metrics.items()))

    return {"fold": fold_id, "test_loss": test_result.loss, **test_result.metrics}, test_result


def run_training(df: pd.DataFrame, cfg: TrainingConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run every requested fold for cfg.task/cfg.model, then aggregate.

    Returns (per_fold_summary_df, pooled_metrics_dict). Every fold's
    checkpoint/log/predictions/metrics/confusion-matrix/ROC/embedding is
    written under outputs/ as a side effect. Pooled metrics — predictions
    concatenated across every fold before scoring — are what's comparable
    to the base paper's LOSO numbers; per-fold precision/recall/AUROC are
    degenerate for detection, since each LOSO fold's held-out speaker is
    entirely one class.
    """
    set_seed(cfg.seed)
    config.ensure_directories()
    device = resolve_device(cfg.device)
    run_name = cfg.run_name or f"{cfg.task}_{cfg.model}"

    # Phase 6 Model F only. Loaded once here rather than per fold — the features
    # do not depend on which speaker is held out; only their standardization does,
    # and build_loaders recomputes that from each fold's train split.
    praat_table = None
    if cfg.model in MODELS_REQUIRING_PRAAT:
        praat_table = load_praat_table()

    protocol = ("Leave-One-Speaker-Out" if cfg.task == "detection"
                else "balanced leave-one-speaker-per-class-out")

    print_banner("UA-Speech Dysarthria Classification",
                 f"{MODEL_DESCRIPTIONS.get(cfg.model, cfg.model)}")
    print_subheader("Run configuration")
    print_kv("Task", f"{cfg.task} ({config.NUM_CLASSES[cfg.task]}-class)")
    print_kv("Model", f"{cfg.model} — {MODEL_DESCRIPTIONS.get(cfg.model, '')}")
    print_kv("Cross-validation protocol", protocol)
    print_kv("Run name", run_name)
    print_kv("Device", device)
    print_kv("Epochs / batch size", f"{cfg.epochs} / {cfg.batch_size}")
    print_kv("LR (head / wav2vec backbone)", f"{cfg.lr_head} / {cfg.lr_backbone}")
    print_kv("Early stopping patience", f"{cfg.patience} epochs on validation loss")
    if praat_table is not None:
        print_kv("Praat pathway", f"{len(PRAAT_FEATURE_COLUMNS)} features, "
                 f"standardized per fold from the train split only")

    print_signal_chain()

    fold_iter = build_folds(df, cfg.task)
    if cfg.folds:
        wanted = set(cfg.folds)
        fold_iter = (f for f in fold_iter if f[0] in wanted)
    fold_iter = list(fold_iter)
    if cfg.max_folds is not None:
        fold_iter = fold_iter[:cfg.max_folds]

    n_folds = len(fold_iter)
    if cfg.limit_samples is not None or (cfg.max_folds is not None and cfg.max_folds < 28):
        print()
        print_note("REDUCED SCALE — this is a pipeline check, not a reportable result "
                   "(max_folds / limit_samples are set).")

    fold_metrics = []
    pooled_true, pooled_pred, pooled_prob, pooled_speakers = [], [], [], []
    failed_folds = []
    for i, (fold_id, train_df, test_df) in enumerate(fold_iter, start=1):
        # Resume support: a long unattended run (full 28-fold LOSO across six
        # model variants is realistically hours-to-days) can be interrupted
        # and restarted without redoing folds that already finished.
        cached = _load_completed_fold(run_name, fold_id, cfg.task)
        if cached is not None:
            metrics_dict, y_true, y_pred, y_prob, speakers = cached
            print_kv(f"Fold {fold_id}", "already completed — loaded from disk, skipping retrain")
        else:
            try:
                metrics_dict, test_result = run_fold(fold_id, train_df, test_df, cfg, device,
                                                     run_name, praat_table,
                                                     fold_index=i, n_folds=n_folds)
            except Exception:
                # One fold's OOM/transient failure should not abort a run that
                # may have already spent hours on earlier folds — log it, free
                # whatever CUDA memory the failed attempt held, and move on.
                print_status(f"Fold {fold_id} failed — skipping (see traceback below)", ok=False)
                print(traceback.format_exc())
                failed_folds.append(fold_id)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            y_true, y_pred, y_prob = test_result.y_true, test_result.y_pred, test_result.y_prob
            speakers = test_result.speaker_ids

        fold_metrics.append(metrics_dict)
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        pooled_prob.append(y_prob)
        pooled_speakers.extend(speakers)

    if failed_folds:
        print_note(f"{len(failed_folds)} fold(s) failed and were excluded from pooling: "
                  f"{failed_folds}")

    if not fold_metrics:
        print_kv("Result", "No folds matched cfg.folds/cfg.max_folds; nothing was trained.")
        return pd.DataFrame(), {}

    summary = aggregate_fold_metrics(config.METRICS_DIR, run_name, fold_metrics)

    y_true = np.concatenate(pooled_true)
    y_pred = np.concatenate(pooled_pred)
    y_prob = np.concatenate(pooled_prob)
    pooled_metrics = compute_metrics(y_true, y_pred, y_prob, cfg.task)

    save_metrics(config.METRICS_DIR / run_name / "ALL_FOLDS_pooled.json", pooled_metrics)
    save_confusion_matrix(
        config.CONFUSION_MATRIX_DIR / run_name / "ALL_FOLDS_pooled.png",
        compute_confusion_matrix(y_true, y_pred, cfg.task),
        cfg.task, title=f"{run_name} — all folds pooled")
    save_roc_curve(config.ROC_DIR / run_name / "ALL_FOLDS_pooled.png",
                   y_true, y_prob, cfg.task, title=f"{run_name} — all folds pooled")

    print_header(f"Results — {run_name}  ({len(fold_metrics)} fold(s), "
                 f"{len(y_true):,} held-out utterances)")

    print_subheader("Per-fold mean +/- std")
    print_table(summary.reset_index().rename(columns={"index": "metric"}))
    if cfg.task == "detection":
        print()
        print_note("Every LOSO fold holds out ONE speaker, who is entirely one class, so "
                   "per-fold precision / recall / specificity / AUROC above are")
        print_note("degenerate — only 'accuracy' is meaningful per fold. The pooled "
                   "numbers below are the ones comparable to the base paper.")

    print_metrics(pooled_metrics,
                  title="Pooled across all folds (base-paper-style LOSO reporting)")

    return summary, pooled_metrics
