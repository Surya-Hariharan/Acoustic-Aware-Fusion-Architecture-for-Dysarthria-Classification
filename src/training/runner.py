"""
Training orchestration: fold iteration, per-fold train/validate/checkpoint,
and cross-fold aggregation.

This is the function notebooks/02_training.ipynb calls — it is reusable
logic (the same loop drives every model variant and both tasks), not a
one-off analysis step, so it lives in src/ rather than the notebook. The
notebook supplies a TrainingConfig and reads results; it never contains
training logic itself.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from src import config
from src.console import print_header, print_kv, print_subheader, print_table
from src.splits import build_severity_folds, get_severity_split, iter_loso_folds
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.data import (TASK_LABEL_COLUMN, build_loaders,
                               compute_class_weights, stratified_train_val_split)
from src.training.early_stopping import EarlyStopping
from src.training.engine import EpochResult, build_optimizer, run_epoch
from src.training.metrics import compute_confusion_matrix, compute_metrics
from src.training.models import build_model
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


def run_fold(fold_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
            cfg: TrainingConfig, device: torch.device, run_name: str
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
        pin_memory=(device.type == "cuda"))

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

    print_subheader(f"Fold: {fold_id}  (train={len(train_df)} val={len(val_df)} test={len(test_df)})")

    for epoch in range(cfg.epochs):
        train_result = run_epoch(model, train_loader, criterion, optimizer, device,
                                 scaler, cfg.grad_clip, cfg.task, train=True)
        val_result = run_epoch(model, val_loader, criterion, None, device,
                               scaler, cfg.grad_clip, cfg.task, train=False)
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

        print(f"    epoch {epoch + 1:>3}/{cfg.epochs} | "
             f"train_loss {train_result.loss:.4f} acc {train_result.metrics['accuracy']:.3f} | "
             f"val_loss {val_result.loss:.4f} acc {val_result.metrics['accuracy']:.3f} "
             f"f1 {val_result.metrics['f1']:.3f}" + ("  * best" if is_best else ""))

        if early_stopping.should_stop:
            print(f"    early stopping at epoch {epoch + 1} "
                 f"(no val-loss improvement for {cfg.patience} epochs)")
            break

    if best_ckpt_path.exists():
        load_checkpoint(best_ckpt_path, model, map_location=str(device))

    test_result = run_epoch(model, test_loader, criterion, None, device, scaler,
                            cfg.grad_clip, cfg.task, train=False, collect_embeddings=True)

    save_predictions(config.PREDICTIONS_DIR / run_name / f"{fold_id}.csv",
                     test_result.speaker_ids, test_result.y_true, test_result.y_pred,
                     test_result.y_prob, cfg.task)
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
                    test_result.embeddings, test_result.y_true, test_result.speaker_ids)

    writer.close()
    print_kv(f"  Fold {fold_id} test", ", ".join(
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

    print_header("UA-Speech Training Pipeline")
    print_kv("Task", cfg.task)
    print_kv("Model", cfg.model)
    print_kv("Run name", run_name)
    print_kv("Device", device)
    print_kv("Epochs / batch size", f"{cfg.epochs} / {cfg.batch_size}")
    print_kv("Patience (early stopping)", cfg.patience)

    fold_iter = build_folds(df, cfg.task)
    if cfg.folds:
        wanted = set(cfg.folds)
        fold_iter = (f for f in fold_iter if f[0] in wanted)
    if cfg.max_folds is not None:
        fold_iter = list(fold_iter)[:cfg.max_folds]

    fold_metrics = []
    pooled_true, pooled_pred, pooled_prob, pooled_speakers = [], [], [], []
    for fold_id, train_df, test_df in fold_iter:
        metrics_dict, test_result = run_fold(fold_id, train_df, test_df, cfg, device, run_name)
        fold_metrics.append(metrics_dict)
        pooled_true.append(test_result.y_true)
        pooled_pred.append(test_result.y_pred)
        pooled_prob.append(test_result.y_prob)
        pooled_speakers.extend(test_result.speaker_ids)

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

    print_header(f"Summary — {run_name} ({len(fold_metrics)} fold(s))")
    print_subheader("Per-fold mean +/- std")
    print_table(summary.reset_index().rename(columns={"index": "metric"}))
    if cfg.task == "detection":
        print_kv("Note", "each LOSO fold's test speaker is entirely one class, so "
                 "per-fold precision/recall/specificity/AUROC above are degenerate "
                 "(only 'accuracy' is meaningful per fold) — see pooled metrics below.")
    print_subheader("Pooled across all folds (base-paper-style LOSO reporting)")
    for name, value in pooled_metrics.items():
        print_kv(name, f"{value:.4f}")

    return summary, pooled_metrics
