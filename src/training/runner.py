"""
Training orchestration: fold iteration, per-fold train/validate/checkpoint,
and cross-fold aggregation.

This is the function notebooks/03_training.ipynb calls — it is reusable
logic (the same loop drives every model variant and both tasks), not a
one-off analysis step, so it lives in src/ rather than the notebook. The
notebook supplies a TrainingConfig and reads results; it never contains
training logic itself.
"""

import json
import shutil
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from src import config
from src.console import H_LIGHT, LINE_WIDTH, V, print_header, print_kv, print_note, print_status
from src.praat import FEATURE_COLUMNS as PRAAT_FEATURE_COLUMNS
from src.praat import load_praat_table
from src.splits import (build_severity_folds, get_severity_split, iter_loso_folds,
                        iter_screening_folds, iter_severity_loso_folds, sample_severity_folds)
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.data import (TASK_LABEL_COLUMN, build_loaders, build_speaker_label_map,
                               compute_class_weights, shutdown_loaders, split_train_val)
from src.training.early_stopping import EarlyStopping
from src.training.engine import EpochResult, build_optimizer, run_epoch
from src.training.metrics import compute_confusion_matrix, compute_metrics
from src.training.models import (MODEL_DESCRIPTIONS, SEVERITY_MODEL_NAME,
                                 MODELS_WITH_CACHEABLE_FROZEN_EMBEDDING,
                                 MODELS_REQUIRING_PRAAT, build_model)
from src.training.reporting import (FOLD_CACHED, FOLD_COMPLETED, FOLD_FAILED,
                                    FOLD_INTERRUPTED, FOLD_SKIPPED_DEADLINE,
                                    aggregate_fold_metrics, describe_fold,
                                    print_fold_report, print_pooled_evaluation,
                                    print_run_coverage, print_runtime_status,
                                    record_fold, run_coverage, save_confusion_matrix,
                                    save_embeddings, save_metrics, save_predictions,
                                    save_roc_curve)
from src.training.utils import (affordable_workers, configure_local_runtime, memory_status,
                                resolve_amp_dtype, resolve_device, set_seed)


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
    # 4, not 0: UASpeechDataset.__getitem__ does real CPU work per utterance
    # (audio load, resample, VAD trim, MFCC) - at 0 workers that runs
    # synchronously in the main process and starves the GPU between batches,
    # which is the usual reason training looks like it isn't using the GPU
    # at all even though the model is correctly placed on cuda. Set to 0 to
    # fall back to the old synchronous behaviour (e.g. for step-by-step
    # debugging where worker processes make tracebacks harder to read).
    num_workers: int = config.TRAIN_NUM_WORKERS      # training loader
    eval_num_workers: int = config.EVAL_NUM_WORKERS  # validation loader
    test_num_workers: int = config.TEST_NUM_WORKERS  # test loader (0 = main process)

    max_folds: Optional[int] = None                  # only run the first N folds
    folds: Optional[List[str]] = None                # only run these fold IDs
    limit_samples: Optional[int] = None               # cap rows per split — smoke testing only

    # Detection: "loso" is the base-paper's full 28-fold protocol (expensive
    # x 6 ablation variants); "screening" is a cheap speaker-grouped,
    # class-stratified k-fold used to rank variants before spending full-LOSO
    # GPU time on the winner(s). See src.splits.iter_screening_folds.
    cv_protocol: str = "loso"
    screening_folds: int = 8

    # Severity: None runs all 81 leave-one-per-class-out combinations (the
    # base-paper protocol); an int randomly subsamples that many combos
    # (src.splits.sample_severity_folds) — 81 folds x every ablation variant
    # is the single largest GPU-time item in the training notebook. Only
    # consulted when severity_protocol == "balanced_lopco" below.
    severity_fold_sample: Optional[int] = None

    # Severity protocol switch (architecture plan Part 2, Component 1-2):
    # "full_loso" (default, matches config.SEVERITY_PRIMARY_PROTOCOL) — the
    # one-shot run's PRIMARY protocol, Leave-One-Speaker-Out across all 15
    # dysarthric speakers, no speaker dropped for balance (src.splits.
    # iter_severity_loso_folds). "balanced_lopco" — the legacy base-paper
    # 3-per-class, 81 (or severity_fold_sample-subsampled) leave-one-per-
    # class-out protocol (build_severity_folds/get_severity_split above),
    # run only as an explicitly-labeled SECONDARY sanity check.
    severity_protocol: str = config.SEVERITY_PRIMARY_PROTOCOL

    # 1 = every batch steps the optimizer (unchanged default behaviour). >1
    # accumulates that many batches' gradients before stepping, simulating a
    # larger effective batch size (batch_size x grad_accum_steps) at
    # batch_size's actual memory footprint — raise this instead of
    # batch_size itself if a fold OOMs on a smaller GPU than the one
    # DEFAULT_BATCH_SIZE was tuned for.
    grad_accum_steps: int = 1

    # False (default): no per-batch progress bar inside run_epoch, only the one
    # table row per epoch run_fold prints — a 15-fold run then reads as a
    # compact report. Set True for interactive, step-by-step debugging of a
    # single batch/epoch.
    show_batch_progress: bool = False

    # How each fold's validation set is carved from its training portion —
    # see src.training.data.split_train_val. "speaker" (default): whole
    # speakers, disjoint from training, so model selection and early stopping
    # see cross-speaker generalization. "utterance": the legacy 10% per-class
    # utterance split, whose validation speakers are also training speakers.
    val_protocol: str = "speaker"

    # None = config.WAV2VEC_GRADIENT_CHECKPOINTING. Numerically identical
    # either way; only speed/memory differ (see benchmark_batch_sizes).
    gradient_checkpointing: Optional[bool] = None

    # Runtime guard (see run_training): a fold is started only if
    # now + fold_estimate * safety_factor stays inside the deadline. Before any
    # fold has run in this session the estimate is fold_time_estimate_s (e.g.
    # from src.training.session.project_runtime), or no estimate at all.
    fold_time_estimate_s: Optional[float] = None
    safety_factor: float = 1.15

    # A fold that raises is retried this many times — resuming from its
    # latest.pt — before it is recorded as FAILED. A CUDA out-of-memory retry
    # halves the batch and doubles gradient accumulation, so the effective
    # batch (and the optimisation) stays the same at half the activation memory.
    fold_retries: int = 2


def build_folds(df: pd.DataFrame, task: str, cfg: Optional["TrainingConfig"] = None):
    """Yield (fold_id, train_df, test_df) for the requested task's protocol."""
    cfg = cfg or TrainingConfig()
    if task == "detection":
        if cfg.cv_protocol == "screening":
            yield from iter_screening_folds(df, cfg.screening_folds, cfg.seed)
        else:
            yield from iter_loso_folds(df)
    elif cfg.severity_protocol == "full_loso":
        yield from iter_severity_loso_folds(df)
    else:
        combos = build_severity_folds(df)
        if cfg.severity_fold_sample is not None:
            combos = sample_severity_folds(combos, cfg.severity_fold_sample, cfg.seed)
        for combo in combos:
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
    except Exception as exc:
        # Deliberately swallowed, not re-raised: a corrupted/truncated resume
        # file must not abort an unattended multi-hour run — falling back to
        # retraining this one fold is the safe behaviour (see docstring). The
        # note is printed so the corruption itself doesn't go unnoticed.
        print_note(f"Could not load a prior result for fold {fold_id} ({exc}) — retraining it.")
        return None

    return metrics_dict, y_true, y_pred, y_prob, speakers


class FoldInterrupted(RuntimeError):
    """Raised by run_fold between epochs when the next epoch would cross the
    session's hard deadline. The fold's latest.pt is already on disk, so a
    later session resumes it mid-fold instead of losing it to Kaggle's kill."""


class FoldDiverged(RuntimeError):
    """Raised by run_fold when no epoch produced a finite validation loss. Its
    checkpoints hold diverged weights, so the retry restarts the fold from
    scratch in full float32 instead of resuming them."""


# Column layout shared by the header below and run_fold's per-epoch rows.
EPOCH_TABLE_HEADER = (f"  {'epoch':<7}{'train loss':>11}{'val loss':>10}{'val ordinal':>13}"
                      f"{'val acc':>9}{'val F1':>9}{'val MAE':>9}{'time':>8}")


def _fmt_cell(value: float, width: int = 9) -> str:
    """A metric table cell; undefined metrics (NaN — e.g. macro-F1 on a
    validation set with one class) read as 'n/a', not 'nan'."""
    return f"{value:>{width}.3f}" if np.isfinite(value) else f"{'n/a':>{width}}"


def _format_hm(seconds: float) -> str:
    """0 <= seconds -> 'XhYYm', for the elapsed/ETA lines below."""
    seconds = max(0.0, seconds)
    hours, remainder = divmod(int(seconds), 3600)
    minutes = remainder // 60
    return f"{hours}h{minutes:02d}m"


def _monitored_value(result: EpochResult) -> float:
    """What early stopping, best-checkpoint selection and the LR scheduler
    track: the validation ORDINAL loss when the model reports one (the
    GatedFusionModel family), else the total validation loss. The ordinal
    term alone keeps model selection comparable across ablations that do and
    do not add a complementarity penalty to their total loss."""
    extras = result.extras or {}
    value = extras.get("ordinal_loss")
    return float(value) if value is not None and np.isfinite(value) else float(result.loss)


def _is_out_of_memory(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _is_host_memory_failure(exc: BaseException) -> bool:
    """Windows running out of commit inside the DataLoader: "error code
    1455" (the paging file is too small), a failed shared-memory mapping, or a
    worker process killed for it."""
    text = str(exc).lower()
    return any(marker in text for marker in ("1455", "paging file", "shared file mapping",
                                             "dataloader worker", "couldn't open shared"))


def _adapt_after_failure(exc: BaseException, fold_cfg: TrainingConfig, run_name: str,
                         fold_id: str) -> Tuple[TrainingConfig, str]:
    """The configuration a failed fold is retried with, and a one-line
    description of the change. Every retry resumes from latest.pt except a
    diverged fold, whose checkpoints hold the diverged weights."""
    if _is_out_of_memory(exc) and fold_cfg.batch_size >= 8:
        adapted = replace(fold_cfg, batch_size=fold_cfg.batch_size // 2,
                          grad_accum_steps=fold_cfg.grad_accum_steps * 2)
        return adapted, (f"CUDA out of memory — batch {adapted.batch_size} x "
                         f"{adapted.grad_accum_steps} accumulation (same effective batch), "
                         "resuming from the last saved epoch.")
    if _is_host_memory_failure(exc):
        adapted = replace(fold_cfg, num_workers=0, eval_num_workers=0)
        return adapted, ("system memory ran out in the data loader — loading in-process "
                         "(no worker processes), resuming from the last saved epoch.")
    if isinstance(exc, FoldDiverged):
        shutil.rmtree(config.CHECKPOINT_DIR / run_name / fold_id, ignore_errors=True)
        return (replace(fold_cfg, amp=False),
                "training diverged — restarting this fold from scratch in float32.")
    return fold_cfg, "resuming from the last saved epoch."


def _save_fold_traceback(run_name: str, fold_id: str, attempt: int) -> Path:
    """The full traceback of the exception being handled, written next to the
    fold's TensorBoard logs rather than into the notebook output."""
    path = config.LOG_DIR / run_name / f"{fold_id}_attempt{attempt}_error.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(traceback.format_exc(), encoding="utf-8")
    return path


def _warn_if_memory_low(fold_id: str) -> None:
    """One note before a fold when this machine is short of memory — the
    fold still starts (with fewer workers), but paging would slow it badly."""
    memory = memory_status()
    if (memory["commit_available"] < config.MIN_FREE_COMMIT_GB_PER_FOLD
            or memory["ram_available"] < config.RAM_RESERVE_GB):
        print_note(f"Low memory before fold {fold_id}: {memory['ram_available']:.1f} GiB RAM / "
                   f"{memory['commit_available']:.1f} GiB commit free — close other "
                   "applications (browsers, extra editor windows) to avoid paging.")


def run_fold(fold_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
            cfg: TrainingConfig, device: torch.device, run_name: str,
            praat_table: Optional[pd.DataFrame] = None,
            frozen_embedding_table: Optional[Dict[str, np.ndarray]] = None,
            fold_index: int = 1, n_folds: int = 1,
            run_start_time: Optional[float] = None,
            total_epochs_planned: Optional[int] = None,
            epochs_completed_before: int = 0,
            hard_deadline: Optional[float] = None
            ) -> Tuple[Dict, EpochResult]:
    """Train, validate, checkpoint, and test-evaluate one fold. Returns
    (metrics_dict, test_result) — the caller pools test_result across
    folds for the cross-fold metrics.

    `run_start_time`/`total_epochs_planned`/`epochs_completed_before` are
    optional whole-run bookkeeping (set by run_training) used only to print
    an "elapsed / estimated remaining" line after each epoch — a fold
    trained standalone (e.g. from a notebook cell or the budget-benchmark
    path) simply omits them and gets the per-epoch line without the ETA.

    `hard_deadline` (time.monotonic()), if given, is checked before every
    epoch: when the next epoch — estimated from the slowest one so far —
    would end past it, the fold raises FoldInterrupted instead of starting it.
    latest.pt is already saved at that point, so the fold resumes mid-way in
    a later session."""
    fold_start = time.monotonic()
    label_column = TASK_LABEL_COLUMN[cfg.task]
    num_classes = config.NUM_CLASSES[cfg.task]

    train_df, val_df = split_train_val(train_df, label_column, cfg.val_protocol, cfg.seed,
                                       fold_id=fold_id, val_fraction=cfg.val_fraction)
    val_speakers = sorted(val_df["Speaker_ID"].unique().tolist())
    if cfg.val_protocol == "speaker":
        overlap = set(val_speakers) & set(train_df["Speaker_ID"])
        if overlap:
            raise RuntimeError(f"Speaker leakage: {sorted(overlap)} in both train and val.")
    if set(test_df["Speaker_ID"]) & (set(train_df["Speaker_ID"]) | set(val_speakers)):
        raise RuntimeError(f"Speaker leakage: held-out speaker(s) of fold {fold_id} "
                           "also appear in train or val.")

    if cfg.limit_samples is not None:
        train_df = _limit_samples(train_df, cfg.limit_samples, label_column)
        if cfg.val_protocol == "utterance":
            # Restrict val to train's (now-limited) speakers BEFORE capping it —
            # under the utterance protocol val speakers are training speakers
            # and get speaker labels (src.training.data.build_loaders), so a val
            # speaker with zero rows left in the capped train_df would crash
            # UASpeechDataset.__getitem__'s speaker_index lookup (KeyError) —
            # reproduced by the notebook's own SMOKE_RUN (limit_samples=16).
            val_df = val_df[val_df["Speaker_ID"].isin(train_df["Speaker_ID"])]
        val_df = _limit_samples(val_df, max(2, cfg.limit_samples // 4), label_column)
        test_df = _limit_samples(test_df, cfg.limit_samples, label_column)

    # Only meaningful for the GatedFusionModel family (see build_model's
    # adversarial speaker head) — harmless to build unconditionally for every
    # other model, since build_loaders only forwards it when the model actually
    # needs the three-branch Dataset fields.
    speaker_label_map = build_speaker_label_map(train_df)

    train_loader, val_loader, test_loader = build_loaders(
        train_df, val_df, test_df, cfg.batch_size, cfg.num_workers,
        eval_num_workers=cfg.eval_num_workers, test_num_workers=cfg.test_num_workers,
        pin_memory=(device.type == "cuda"), praat_table=praat_table,
        frozen_embedding_table=frozen_embedding_table, model_name=cfg.model,
        speaker_label_map=speaker_label_map)

    model = build_model(cfg.model, num_classes, num_speakers=len(speaker_label_map),
                        gradient_checkpointing=cfg.gradient_checkpointing).to(device)
    optimizer = build_optimizer(model, cfg.lr_head, cfg.lr_backbone, cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=max(1, cfg.patience // 2))
    class_weights = compute_class_weights(train_df, cfg.task).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = cfg.amp if cfg.amp is not None else (device.type == "cuda")
    # config.AMP_DTYPE (float16 by default, with a live GradScaler below);
    # bfloat16 only where the GPU runs it natively — see resolve_amp_dtype.
    amp_dtype = resolve_amp_dtype(device)
    scaler = torch.amp.GradScaler(device=device.type,
                                  enabled=use_amp and amp_dtype == torch.float16)
    early_stopping = EarlyStopping(patience=cfg.patience, mode="min")

    log_dir = config.LOG_DIR / run_name / fold_id
    writer = SummaryWriter(log_dir=str(log_dir))
    best_ckpt_path = config.CHECKPOINT_DIR / run_name / fold_id / "best.pt"
    # Saved after EVERY epoch (unlike best.pt, which only updates on a
    # val-loss improvement) so an interrupted session loses at most one
    # in-progress epoch, not everything back to this fold's last improving
    # epoch. best.pt remains what test evaluation reloads below — this file
    # exists purely to let a fresh process resume mid-fold.
    latest_ckpt_path = config.CHECKPOINT_DIR / run_name / fold_id / "latest.pt"

    true_label = ", ".join(sorted(test_df[label_column].unique()))
    print()
    print(H_LIGHT * LINE_WIDTH)
    print(f"  FOLD {fold_index}/{n_folds}  {V}  held-out {fold_id} ({true_label})  {V}  "
          f"train {len(train_df):,} · val {len(val_df):,} · test {len(test_df):,}")
    print(f"  {train_df['Speaker_ID'].nunique()} training speakers; validation speakers "
          f"{', '.join(val_speakers)}"
          + (f"; batch {cfg.batch_size} x {cfg.grad_accum_steps} accumulation"
             if cfg.grad_accum_steps > 1 else ""))
    print(H_LIGHT * LINE_WIDTH)

    epochs_completed = 0
    best_epoch, best_val_f1, best_monitored = None, None, None
    start_epoch = 0
    slowest_epoch_s = 0.0
    if latest_ckpt_path.exists():
        checkpoint = load_checkpoint(latest_ckpt_path, model, optimizer, scheduler, scaler,
                                     map_location=str(device))
        start_epoch = checkpoint["epoch"] + 1
        es_state = checkpoint.get("early_stopping") or {}
        early_stopping.best = es_state.get("best", early_stopping.best)
        early_stopping.num_bad_epochs = es_state.get("num_bad_epochs", 0)
        early_stopping.should_stop = es_state.get("should_stop", False)
        best_epoch = checkpoint.get("best_epoch")
        best_val_f1 = checkpoint.get("best_val_f1")
        best_monitored = checkpoint.get("best_monitored")
        slowest_epoch_s = float(checkpoint.get("slowest_epoch_s") or 0.0)
        epochs_completed = start_epoch
        print(f"  Resumed from {latest_ckpt_path.name}: {start_epoch} epoch(s) already done "
              f"(best so far: epoch {best_epoch}).")

    if not early_stopping.should_stop and start_epoch < cfg.epochs:
        print(EPOCH_TABLE_HEADER)
    for epoch in range(start_epoch, cfg.epochs) if not early_stopping.should_stop else ():
        if (hard_deadline is not None and slowest_epoch_s > 0
                and time.monotonic() + slowest_epoch_s > hard_deadline):
            writer.close()
            raise FoldInterrupted(
                f"fold {fold_id}: epoch {epoch + 1} (~{_format_hm(slowest_epoch_s)}) would end "
                f"past the session's hard deadline — stopped after {epochs_completed} epoch(s); "
                f"{latest_ckpt_path.name} resumes it next session.")
        epoch_start = time.monotonic()
        train_desc = (f"epoch {epoch + 1}/{cfg.epochs} train" if cfg.show_batch_progress else "")
        val_desc = (f"epoch {epoch + 1}/{cfg.epochs} val" if cfg.show_batch_progress else "")
        train_result = run_epoch(model, train_loader, criterion, optimizer, device,
                                 scaler, cfg.grad_clip, cfg.task, train=True,
                                 description=train_desc,
                                 amp_dtype=amp_dtype, amp_enabled=use_amp,
                                 grad_accum_steps=cfg.grad_accum_steps)
        val_result = run_epoch(model, val_loader, criterion, None, device,
                               scaler, cfg.grad_clip, cfg.task, train=False,
                               description=val_desc,
                               amp_dtype=amp_dtype, amp_enabled=use_amp)
        monitored = _monitored_value(val_result)
        scheduler.step(monitored)

        writer.add_scalar("Loss/train", train_result.loss, epoch)
        writer.add_scalar("Loss/val", val_result.loss, epoch)
        writer.add_scalar("Loss/val_monitored", monitored, epoch)
        for name, value in train_result.metrics.items():
            writer.add_scalar(f"Train/{name}", value, epoch)
        for name, value in val_result.metrics.items():
            writer.add_scalar(f"Val/{name}", value, epoch)
        writer.add_scalar("LR", optimizer.param_groups[-1]["lr"], epoch)

        epochs_completed = epoch + 1
        is_best = early_stopping.step(monitored)
        if is_best:
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, scaler,
                            epoch, monitored)
            best_epoch, best_val_f1, best_monitored = epoch + 1, val_result.metrics["f1"], monitored

        epoch_time_s = time.monotonic() - epoch_start
        slowest_epoch_s = max(slowest_epoch_s, epoch_time_s)

        # Every epoch, improving or not -- see latest_ckpt_path's comment
        # above. Written after best.pt so an interrupted save never leaves
        # latest.pt claiming an epoch whose best.pt update didn't land.
        save_checkpoint(latest_ckpt_path, model, optimizer, scheduler, scaler,
                        epoch, monitored,
                        extra={"early_stopping": {"best": early_stopping.best,
                                                  "num_bad_epochs": early_stopping.num_bad_epochs,
                                                  "should_stop": early_stopping.should_stop},
                              "best_epoch": best_epoch, "best_val_f1": best_val_f1,
                              "best_monitored": best_monitored,
                              "slowest_epoch_s": slowest_epoch_s,
                              "fold_id": fold_id})

        val_metrics = val_result.metrics
        print(f"  {epoch + 1:>3}/{cfg.epochs:<3}{train_result.loss:>11.4f}{val_result.loss:>10.4f}"
              f"{monitored:>13.4f}{val_metrics['accuracy']:>9.3f}{_fmt_cell(val_metrics['f1'])}"
              f"{val_metrics['ordinal_mae']:>9.3f}{epoch_time_s / 60:>7.1f}m"
              + ("   * best" if is_best else ""))
        if not (np.isfinite(train_result.loss) and np.isfinite(monitored)):
            print_note(f"Non-finite loss in epoch {epoch + 1} (train {train_result.loss}, "
                       f"val {monitored}) — this epoch cannot become the best checkpoint.")

        if early_stopping.should_stop:
            print(f"  Early stop: validation ordinal loss did not improve for {cfg.patience} "
                  f"epochs (best epoch {best_epoch}).")
            break

    if not best_ckpt_path.exists():
        # Every epoch's validation loss was non-finite: testing the last
        # weights would report a diverged model as if it were a result.
        writer.close()
        raise FoldDiverged(f"fold {fold_id}: no epoch produced a finite validation loss, "
                           "so there is no checkpoint to evaluate.")
    load_checkpoint(best_ckpt_path, model, map_location=str(device))

    train_time_s = time.monotonic() - fold_start
    inference_start = time.monotonic()
    test_result = run_epoch(model, test_loader, criterion, None, device, scaler,
                            cfg.grad_clip, cfg.task, train=False, collect_embeddings=True,
                            description=(f"held-out test ({fold_id})"
                                        if cfg.show_batch_progress else ""),
                            amp_dtype=amp_dtype, amp_enabled=use_amp)
    inference_time_s = time.monotonic() - inference_start

    class_names = (config.SEVERITY_CLASS_NAMES if cfg.task == "severity"
                   else config.DETECTION_CLASS_NAMES)
    pred_counts = np.bincount(test_result.y_pred, minlength=num_classes)
    argmax_counts = np.bincount(test_result.y_pred_argmax, minlength=num_classes)
    diagnostics = (model.coral_threshold_diagnostics()
                   if hasattr(model, "coral_threshold_diagnostics") else {})
    fold_record = {
        "fold": fold_id, "test_loss": test_result.loss, **test_result.metrics,
        "true_label": ";".join(sorted(test_df[label_column].unique())),
        "pred_distribution": {name: int(n) for name, n in zip(class_names, pred_counts)},
        "argmax_pred_distribution": {name: int(n) for name, n in zip(class_names, argmax_counts)},
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch, "best_val_f1": best_val_f1,
        "best_val_monitored": best_monitored,
        "val_protocol": cfg.val_protocol, "val_speakers": ";".join(val_speakers),
        "n_train_speakers": int(train_df["Speaker_ID"].nunique()),
        "n_train": int(len(train_df)), "n_val": int(len(val_df)),
        "batch_size": cfg.batch_size,
        "train_time_s": train_time_s, "inference_time_s": inference_time_s,
        **diagnostics,
    }

    save_predictions(config.PREDICTIONS_DIR / run_name / f"{fold_id}.csv",
                     test_result.filenames, test_result.speaker_ids, test_result.y_true,
                     test_result.y_pred, test_result.y_prob, cfg.task,
                     y_pred_argmax=test_result.y_pred_argmax)
    save_confusion_matrix(
        config.CONFUSION_MATRIX_DIR / run_name / f"{fold_id}.png",
        compute_confusion_matrix(test_result.y_true, test_result.y_pred, cfg.task),
        cfg.task, title=f"{run_name} — fold {fold_id}")
    save_roc_curve(config.ROC_DIR / run_name / f"{fold_id}.png",
                   test_result.y_true, test_result.y_prob, cfg.task,
                   title=f"{run_name} — fold {fold_id}")
    save_embeddings(config.EMBEDDINGS_DIR / run_name / f"{fold_id}.npz",
                    test_result.embeddings, test_result.y_true, test_result.speaker_ids,
                    test_result.filenames, branch_embeddings=test_result.branch_embeddings,
                    gate_weights=test_result.gate_weights)
    writer.close()

    # Timed after every artifact is written, so fold_time_s is the real cost of
    # a fold — what the runtime guard in run_training projects from.
    fold_record["fold_time_s"] = time.monotonic() - fold_start
    # Metrics last: _load_completed_fold treats metrics + predictions as the
    # "fold finished" marker, so it must not exist before everything else does.
    save_metrics(config.METRICS_DIR / run_name / f"{fold_id}.json",
                 {**fold_record, **(test_result.extras or {})})

    print_fold_report(fold_record, fold_index, n_folds, cfg.epochs, cfg.task)
    return fold_record, test_result


def _registry_kwargs(cfg: TrainingConfig, run_name: str, expected_folds: int) -> Dict:
    """The run-identifying fields every record_fold() call in run_training shares."""
    if cfg.task == "detection":
        cv_protocol = cfg.cv_protocol
    else:
        cv_protocol = ("severity_loso" if cfg.severity_protocol == "full_loso"
                       else "severity_lopco")
    return {"run_name": run_name, "model": cfg.model, "task": cfg.task,
            "cv_protocol": cv_protocol, "expected_folds": expected_folds}


def run_training(df: pd.DataFrame, cfg: TrainingConfig,
                 deadline: Optional[float] = None,
                 hard_deadline: Optional[float] = None,
                 session_start: Optional[float] = None,
                 session_budget_s: Optional[float] = None
                 ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run every requested fold for cfg.task/cfg.model, then aggregate.

    Runtime safety (all time.monotonic() timestamps, all optional):
      deadline       the SAFE budget end. A fold is started only if its
                     projected end — now + fold estimate x cfg.safety_factor —
                     falls before it; the estimate is the mean wall time of the
                     folds trained so far in this session, or
                     cfg.fold_time_estimate_s before the first one. Otherwise
                     that fold and every later one are recorded as skipped and
                     the run stops cleanly.
      hard_deadline  checked between epochs inside a fold (see run_fold): a
                     fold whose next epoch would cross it is interrupted, not
                     killed mid-write — it resumes from latest.pt next session.
      session_start / session_budget_s  only for the per-fold runtime report
                     (elapsed session time, projected finish, safety margin).

    Re-calling run_training() later resumes: finished folds load instantly
    from disk via _load_completed_fold, an interrupted one resumes mid-fold.

    Returns (per_fold_summary_df, pooled_metrics_dict). pooled_metrics also
    carries the run's coverage ("run_status" COMPLETE / PARTIAL / FAILED,
    "expected_folds", "completed_folds"), which is written to
    outputs/metrics/<run_name>/RUN_STATUS.json as well. Every fold's
    checkpoint/log/predictions/metrics/confusion-matrix/ROC/embedding is
    written under outputs/ as a side effect. Pooled metrics — predictions
    concatenated across every fold before scoring — are what's comparable
    to the base paper's LOSO numbers; per-fold class-sensitive metrics are
    undefined, since each LOSO fold's held-out speaker is entirely one class.
    """
    set_seed(cfg.seed)
    config.ensure_directories()
    device = resolve_device(cfg.device)
    configure_local_runtime(device)
    run_name = cfg.run_name or f"{cfg.task}_{cfg.model}"

    # Phase 6 Model F only. Loaded once here rather than per fold — the features
    # do not depend on which speaker is held out; only their standardization does,
    # and build_loaders recomputes that from each fold's train split.
    praat_table = None
    if cfg.model in MODELS_REQUIRING_PRAAT:
        praat_table = load_praat_table()

    # deep_frozen/fusion_frozen only. The frozen wav2vec2 embedding is the
    # same vector for a given file in every fold and every epoch (the
    # backbone never updates), so it is extracted once for the whole dataset
    # here — a cached, batched pass — rather than recomputed on every
    # training-loop forward pass (see MODELS_WITH_CACHEABLE_FROZEN_EMBEDDING).
    frozen_embedding_table = None
    if cfg.model in MODELS_WITH_CACHEABLE_FROZEN_EMBEDDING:
        from src.training.baseline import extract_frozen_embeddings_masked
        embeddings = extract_frozen_embeddings_masked(df, device=device)
        frozen_embedding_table = dict(zip(df["Filepath"], embeddings))

    if cfg.task == "detection":
        protocol = ("Leave-One-Speaker-Out" if cfg.cv_protocol != "screening"
                    else f"screening ({cfg.screening_folds}-fold, speaker-grouped)")
    elif cfg.severity_protocol == "full_loso":
        protocol = "severity Leave-One-Speaker-Out"
    else:
        protocol = "balanced leave-one-speaker-per-class-out (SECONDARY, 3/class)"
        if cfg.severity_fold_sample is not None:
            protocol += f" (subsampled to {cfg.severity_fold_sample} of 81)"

    fold_iter = build_folds(df, cfg.task, cfg)
    if cfg.folds:
        wanted = set(cfg.folds)
        fold_iter = (f for f in fold_iter if f[0] in wanted)
    fold_iter = list(fold_iter)
    if cfg.max_folds is not None:
        fold_iter = fold_iter[:cfg.max_folds]

    n_folds = len(fold_iter)
    # 28 for detection (config.ALL_SPEAKERS), 15 for severity
    # (config.DYSARTHRIC_IDS) — was hardcoded to 28 for both tasks, which
    # meant a severity run truncated to, say, max_folds=14 (< 15 but not <
    # 28) would silently NOT be flagged as reduced scale.
    full_fold_count = (len(config.ALL_SPEAKERS) if cfg.task == "detection"
                       else len(config.DYSARTHRIC_IDS))
    is_reduced_scale = (cfg.limit_samples is not None
                        or (cfg.max_folds is not None and cfg.max_folds < full_fold_count)
                        or cfg.cv_protocol == "screening")

    memory = memory_status()
    amp_on = cfg.amp if cfg.amp is not None else device.type == "cuda"
    print_header(f"Training — {run_name}")
    print_kv("Model", cfg.model)
    print_kv("Protocol", f"{protocol}, {n_folds} fold{'s' * (n_folds != 1)}; "
                         + ("speaker-disjoint validation" if cfg.val_protocol == "speaker"
                            else "utterance-level validation (LEGACY)"))
    print_kv("Schedule", f"max {cfg.epochs} epochs, patience {cfg.patience}, batch "
                         f"{cfg.batch_size}, lr {cfg.lr_head:g}/{cfg.lr_backbone:g}, "
                         + (f"AMP {str(resolve_amp_dtype(device)).replace('torch.', '')}"
                            if amp_on else "float32"))
    print_kv("Data loading", f"{cfg.num_workers} train worker{'s' * (cfg.num_workers != 1)}; "
                             "validation "
                             + (f"{cfg.eval_num_workers} worker(s)" if cfg.eval_num_workers
                                else "in-process"))
    print_kv("Memory free at start", f"{memory['ram_available']:.1f} GiB RAM, "
                                     f"{memory['commit_available']:.1f} GiB commit")
    print_kv("Fold order", ", ".join(fold_id for fold_id, _, _ in fold_iter))
    if praat_table is not None:
        print_kv("Praat pathway", f"{len(PRAAT_FEATURE_COLUMNS)} features, "
                 f"standardized per fold from the train split only")
    if is_reduced_scale:
        if cfg.limit_samples is not None:
            print_note("REDUCED SCALE — a pipeline check, not a reportable result "
                       "(limit_samples is set).")
        elif cfg.cv_protocol == "screening":
            print_note("SCREENING PROTOCOL — cheap speaker-grouped k-fold for ranking "
                       "ablation variants, not the base-paper's full LOSO result.")
        elif cfg.max_folds is not None and cfg.max_folds < full_fold_count:
            print_note(
                f"BUDGET-REDUCED FOLD COUNT — running {n_folds} of {full_fold_count} "
                f"{'speakers' if cfg.task == 'detection' else 'dysarthric speakers'}, "
                "a deliberate compute-budget concession. This is NOT the full-"
                "population primary protocol result and carries less statistical "
                "power than the complete sweep.")
        else:
            print_note("REDUCED SCALE — this is a pipeline check, not a reportable result "
                       "(max_folds / limit_samples are set).")

    registry_base = _registry_kwargs(cfg, run_name, n_folds)
    fold_metrics = []
    fold_status: Dict[str, str] = {}
    pooled_true, pooled_pred, pooled_prob, pooled_speakers = [], [], [], []
    # Whole-run bookkeeping for the per-epoch "Elapsed" line inside run_fold
    # and for the runtime guard: only folds TRAINED in this session inform the
    # time estimate (cache-loaded ones cost nothing).
    run_start_time = time.monotonic()
    total_epochs_planned = n_folds * cfg.epochs
    epochs_completed_running = 0
    session_fold_times: List[float] = []

    def estimated_fold_seconds() -> Optional[float]:
        if session_fold_times:
            return float(np.mean(session_fold_times))
        return cfg.fold_time_estimate_s

    def skip_remaining(from_index: int, status: str, reason: str) -> None:
        print_note(reason)
        for j, (skipped_id, _, skipped_test_df) in enumerate(fold_iter[from_index - 1:],
                                                             start=from_index):
            if skipped_id in fold_status:
                continue
            fold_status[skipped_id] = status if j == from_index else FOLD_SKIPPED_DEADLINE
            record_fold(**registry_base, fold_id=skipped_id, fold_index=j,
                        status=fold_status[skipped_id],
                        fold_description=describe_fold(skipped_test_df))

    for i, (fold_id, train_df, test_df) in enumerate(fold_iter, start=1):
        # Held-out composition is known before training and is recorded whatever
        # the fold's outcome — so a fold that never ran still leaves a registry
        # row saying which speakers it WOULD have covered. That is what lets
        # summarize_registry() report honest coverage instead of silently
        # shrinking the denominator to whatever happened to finish.
        fold_description = describe_fold(test_df)

        # Resume support: a long unattended run can be interrupted and
        # restarted without redoing folds that already finished.
        cached = _load_completed_fold(run_name, fold_id, cfg.task)

        if cached is None and deadline is not None:
            now = time.monotonic()
            estimate = estimated_fold_seconds()
            projected_end = now + (estimate or 0.0) * cfg.safety_factor
            if now >= deadline or projected_end > deadline:
                skip_remaining(i, FOLD_SKIPPED_DEADLINE, (
                    f"Runtime guard: not starting fold {i}/{n_folds} ({fold_id}) — "
                    + (f"projected to need {_format_hm(estimate * cfg.safety_factor)} "
                       f"(estimate x{cfg.safety_factor:.2f}) but only "
                       f"{_format_hm(max(0.0, deadline - now))} of safe budget remain."
                       if estimate else "the safe budget is already spent.")
                    + " Remaining folds are recorded as SKIPPED; re-run to resume."))
                break

        if cached is not None:
            metrics_dict, y_true, y_pred, y_prob, speakers = cached
            print(f"  Fold {i:>2}/{n_folds}  {fold_id:<4} ({metrics_dict.get('true_label')}) — "
                  f"already complete, loaded from disk: accuracy "
                  f"{metrics_dict.get('accuracy', float('nan')):.3f}, ordinal MAE "
                  f"{metrics_dict.get('ordinal_mae', float('nan')):.3f}")
            fold_status[fold_id] = FOLD_CACHED
            record_fold(**registry_base, fold_id=fold_id, fold_index=i,
                        status=FOLD_CACHED, fold_description=fold_description,
                        num_classes_present=int(len(np.unique(y_true))),
                        epochs_completed=metrics_dict.get("epochs_completed"),
                        runtime_s=metrics_dict.get("fold_time_s"))
        else:
            fold_started = time.monotonic()
            outcome = None
            fold_cfg = cfg
            for attempt in range(1 + max(0, cfg.fold_retries)):
                # Workers sized to the memory free NOW, fold by fold: another
                # application opened mid-run shrinks the pool (down to
                # in-process loading) instead of pushing Windows into paging
                # or "error 1455" mid-epoch.
                worker_budget = dict(ram_per_worker_gb=config.DATALOADER_WORKER_RAM_GB,
                                     commit_per_worker_gb=config.DATALOADER_WORKER_COMMIT_GB,
                                     reserve_commit_gb=config.TRAINING_COMMIT_RESERVE_GB)
                fold_cfg = replace(
                    fold_cfg,
                    num_workers=affordable_workers(fold_cfg.num_workers, **worker_budget),
                    eval_num_workers=affordable_workers(fold_cfg.eval_num_workers,
                                                        **worker_budget))
                _warn_if_memory_low(fold_id)
                try:
                    metrics_dict, test_result = run_fold(
                        fold_id, train_df, test_df, fold_cfg, device, run_name, praat_table,
                        frozen_embedding_table, fold_index=i, n_folds=n_folds,
                        run_start_time=run_start_time,
                        total_epochs_planned=total_epochs_planned,
                        epochs_completed_before=epochs_completed_running,
                        hard_deadline=hard_deadline)
                    outcome = "ok"
                except FoldInterrupted as exc:
                    outcome = ("deadline", str(exc))
                except KeyboardInterrupt:
                    outcome = ("user", None)
                except Exception as exc:
                    # One fold's failure should not abort a run that may have
                    # spent hours on earlier folds: free what the attempt held,
                    # adapt the configuration to the failure, and retry.
                    outcome = "failed"
                    fold_cfg, remedy = _adapt_after_failure(exc, fold_cfg, run_name, fold_id)
                    log_path = _save_fold_traceback(run_name, fold_id, attempt + 1)
                    print_status(f"Fold {fold_id} attempt {attempt + 1} failed: "
                                 f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}  "
                                 f"(traceback: {log_path.name})", ok=False)
                finally:
                    # Always stop this fold's DataLoader workers explicitly —
                    # see src.training.data.shutdown_loaders for why waiting
                    # for garbage collection exhausted RAM in a notebook.
                    shutdown_loaders()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                if outcome != "failed":
                    break
                if attempt < cfg.fold_retries:
                    print_note(f"Retrying fold {fold_id}: {remedy}")

            if isinstance(outcome, tuple) and outcome[0] == "deadline":
                skip_remaining(i, FOLD_INTERRUPTED,
                               f"Hard deadline: {outcome[1]} Later folds are recorded as SKIPPED.")
                break
            if isinstance(outcome, tuple) and outcome[0] == "user":
                skip_remaining(i, FOLD_INTERRUPTED,
                               f"Interrupted by the user during fold {fold_id}. Every finished "
                               f"epoch is saved — re-run to resume this fold from its latest.pt "
                               f"and continue with the remaining folds.")
                break
            if outcome == "failed":
                print_status(f"Fold {fold_id} FAILED after {1 + cfg.fold_retries} attempts — "
                             f"excluded from pooling; re-running the cell retries it.", ok=False)
                fold_status[fold_id] = FOLD_FAILED
                record_fold(**registry_base, fold_id=fold_id, fold_index=i,
                            status=FOLD_FAILED, fold_description=fold_description)
                continue
            session_fold_times.append(time.monotonic() - fold_started)
            fold_status[fold_id] = FOLD_COMPLETED
            record_fold(**registry_base, fold_id=fold_id, fold_index=i,
                        status=FOLD_COMPLETED, fold_description=fold_description,
                        num_classes_present=metrics_dict.get("n_classes_present"),
                        epochs_completed=metrics_dict.get("epochs_completed"),
                        runtime_s=metrics_dict.get("fold_time_s"))
            y_true, y_pred, y_prob = test_result.y_true, test_result.y_pred, test_result.y_prob
            speakers = test_result.speaker_ids

        epochs_completed_running += metrics_dict.get("epochs_completed") or cfg.epochs

        fold_metrics.append(metrics_dict)
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)
        pooled_prob.append(y_prob)
        pooled_speakers.extend(speakers)

        remaining = [fid for fid, _, _ in fold_iter if fid not in fold_status]
        print_runtime_status(
            folds_done=len(fold_status), n_folds=n_folds, remaining_folds=len(remaining),
            run_elapsed_s=time.monotonic() - run_start_time,
            fold_estimate_s=estimated_fold_seconds(), n_timed_folds=len(session_fold_times),
            safety_factor=cfg.safety_factor,
            session_elapsed_s=(time.monotonic() - session_start) if session_start else None,
            session_budget_s=session_budget_s,
            deadline_in_s=(deadline - time.monotonic()) if deadline is not None else None)

    coverage = run_coverage([fid for fid, _, _ in fold_iter], fold_status)
    print_run_coverage(coverage, run_name)
    save_metrics(config.METRICS_DIR / run_name / "RUN_STATUS.json", coverage)

    if not fold_metrics:
        print_kv("Result", "No fold produced a result; nothing to pool.")
        return pd.DataFrame(), {"run_status": coverage["status"],
                                "expected_folds": coverage["expected"],
                                "completed_folds": 0}

    summary = aggregate_fold_metrics(config.METRICS_DIR, run_name, fold_metrics)

    y_true = np.concatenate(pooled_true)
    y_pred = np.concatenate(pooled_pred)
    y_prob = np.concatenate(pooled_prob)
    pooled_metrics = compute_metrics(y_true, y_pred, y_prob, cfg.task)

    save_confusion_matrix(
        config.CONFUSION_MATRIX_DIR / run_name / "ALL_FOLDS_pooled.png",
        compute_confusion_matrix(y_true, y_pred, cfg.task),
        cfg.task, title=f"{run_name} — {coverage['completed']}/{coverage['expected']} "
                        f"folds pooled ({coverage['status']})")
    save_roc_curve(config.ROC_DIR / run_name / "ALL_FOLDS_pooled.png",
                   y_true, y_prob, cfg.task, title=f"{run_name} — all folds pooled")

    # The detailed pooled tables are the notebook's Section 7 (read back from
    # disk); here only the headline, so the training cell stays a compact log.
    pooled_report = print_pooled_evaluation(y_true, y_pred, y_prob, cfg.task, coverage,
                                            speakers=pooled_speakers, verbose=False)
    headline = [f"accuracy {pooled_metrics['accuracy']:.3f}"]
    for key, label in (("f1", "macro-F1"), ("balanced_accuracy", "balanced accuracy"),
                       ("ordinal_mae", "ordinal MAE")):
        value = pooled_metrics.get(key)
        if value is not None and np.isfinite(value):
            headline.append(f"{label} {value:.3f}")
    speaker_level = pooled_report.get("speaker_level")
    if speaker_level:
        n_correct = sum(p["true"] == p["pred"] for p in speaker_level["predictions"].values())
        headline.append(f"speakers correct {n_correct}/{len(speaker_level['predictions'])}")
    print_kv(f"Pooled ({len(y_true):,} utterances)", " · ".join(headline))
    pooled_metrics.update({"run_status": coverage["status"],
                           "expected_folds": coverage["expected"],
                           "completed_folds": coverage["completed"]})
    save_metrics(config.METRICS_DIR / run_name / "ALL_FOLDS_pooled.json",
                 {**pooled_metrics, **pooled_report})
    return summary, pooled_metrics
