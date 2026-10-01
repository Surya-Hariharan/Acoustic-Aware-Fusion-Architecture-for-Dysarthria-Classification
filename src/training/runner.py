"""
The training loop: 15-fold severity LOSO, one GatedFusionModel per fold.

Per fold: speaker-disjoint validation split, fold-scoped standardization,
train with early stopping on the validation ordinal loss, test the best
checkpoint on the held-out speaker, write artifacts. Across folds: pooled
metrics over every held-out utterance.

Resumable at two levels — a fold with metrics + predictions on disk is loaded,
not retrained; a fold interrupted mid-way resumes from its latest.pt — and
self-healing: a failing fold is retried with an adapted configuration.
"""

import json
import shutil
import time
import traceback
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from src import config
from src.console import H_LIGHT, LINE_WIDTH, V, print_header, print_kv, print_note, print_status
from src.splits import iter_severity_loso_folds
from src.training.checkpoint import load_checkpoint, save_checkpoint
from src.training.data import (build_loaders, build_speaker_label_map, compute_class_weights,
                               shutdown_loaders, speaker_disjoint_train_val_split)
from src.training.early_stopping import EarlyStopping
from src.training.engine import EpochResult, build_optimizer, run_epoch
from src.training.metrics import compute_confusion_matrix, compute_metrics
from src.training.models import build_model
from src.training.reporting import (FOLD_CACHED, FOLD_COMPLETED, FOLD_FAILED, FOLD_INTERRUPTED,
                                    FOLD_SKIPPED_DEADLINE, PROB_COLUMNS, print_fold_report,
                                    print_run_coverage, print_runtime_status, run_coverage,
                                    save_confusion_matrix, save_embeddings, save_metrics,
                                    save_predictions, save_roc_curve, speaker_level)
from src.training.utils import (THERMAL_GUARD, affordable_workers, configure_local_runtime,
                                memory_status, prevent_sleep, resolve_amp_dtype, resolve_device,
                                set_seed, wait_for_ac_power)


@dataclass
class TrainingConfig:
    """Everything one run_training() call needs. Defaults come from src.config."""
    model: str = "gated_fusion_three_branch"        # or an ablation (src.training.models)
    run_name: Optional[str] = None                  # default: "sev_<model>"

    epochs: int = config.DEFAULT_EPOCHS
    batch_size: int = config.DEFAULT_BATCH_SIZE
    lr_head: float = config.DEFAULT_LR_HEAD
    lr_lora: float = config.DEFAULT_LR_LORA
    weight_decay: float = config.DEFAULT_WEIGHT_DECAY
    patience: int = config.DEFAULT_PATIENCE
    grad_clip: float = config.DEFAULT_GRAD_CLIP_NORM
    grad_accum_steps: int = 1                       # >1: same effective batch, less VRAM
    seed: int = config.DEFAULT_SEED
    gradient_checkpointing: Optional[bool] = None   # None = config.WAV2VEC_GRADIENT_CHECKPOINTING

    amp: Optional[bool] = None                      # None = on iff CUDA
    device: Optional[str] = None
    num_workers: int = config.TRAIN_NUM_WORKERS
    eval_num_workers: int = config.EVAL_NUM_WORKERS
    test_num_workers: int = config.TEST_NUM_WORKERS

    max_folds: Optional[int] = None                 # only the first N folds (smoke tests)
    folds: Optional[List[str]] = None               # only these held-out speakers
    limit_samples: Optional[int] = None             # cap rows per split (smoke tests only)
    show_batch_progress: bool = False               # per-batch bars inside each epoch

    # Optional wall-clock cap for this session. A fold is not started unless
    # its projected end (mean fold time x safety_factor) fits; an epoch is not
    # started if it would end past the cap. Everything resumes next session.
    session_hours: Optional[float] = None
    fold_time_estimate_s: Optional[float] = None    # before any fold has been timed
    safety_factor: float = 1.15

    # A failing fold is retried this many times with an adapted configuration
    # (see _adapt_after_failure) before it is recorded as FAILED.
    fold_retries: int = 2

    def __post_init__(self):
        self.run_name = self.run_name or f"sev_{self.model}"


def build_folds(df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame, pd.DataFrame]]:
    """(held_out_speaker, train_df, test_df) for every severity LOSO fold."""
    return list(iter_severity_loso_folds(df))


def _limit_samples(df: pd.DataFrame, n: Optional[int]) -> pd.DataFrame:
    """Cap a split to ~n rows, keeping every class present (smoke tests)."""
    if n is None or len(df) <= n:
        return df
    per_class = max(1, n // max(df["Severity"].nunique(), 1))
    return pd.concat([g.head(per_class) for _, g in df.groupby("Severity")]).reset_index(drop=True)


def _load_completed_fold(run_name: str, fold_id: str):
    """(metrics, y_true, y_pred, y_prob, speakers) of a fold finished in an
    earlier session, or None — a corrupt file means "retrain", not "abort"."""
    metrics_path = config.METRICS_DIR / run_name / f"{fold_id}.json"
    predictions_path = config.PREDICTIONS_DIR / run_name / f"{fold_id}.csv"
    if not (metrics_path.exists() and predictions_path.exists()):
        return None
    try:
        with open(metrics_path) as handle:
            metrics = json.load(handle)
        preds = pd.read_csv(predictions_path)
        return (metrics, preds["y_true"].to_numpy(), preds["y_pred"].to_numpy(),
                preds[PROB_COLUMNS].to_numpy(), preds["speaker_id"].tolist())
    except Exception as exc:
        print_note(f"Could not load the saved result of fold {fold_id} ({exc}) — retraining it.")
        return None


class FoldInterrupted(RuntimeError):
    """The next epoch would cross the session cap; latest.pt is saved."""


class FoldDiverged(RuntimeError):
    """No epoch produced a finite validation loss; the retry restarts in float32."""


def _is_out_of_memory(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _is_host_memory_failure(exc: BaseException) -> bool:
    """Windows out of commit inside the DataLoader (error 1455, a failed
    shared-memory mapping, or a worker killed for it)."""
    text = str(exc).lower()
    return any(marker in text for marker in ("1455", "paging file", "shared file mapping",
                                             "dataloader worker", "couldn't open shared"))


def _adapt_after_failure(exc: BaseException, cfg: TrainingConfig, fold_id: str
                         ) -> Tuple[TrainingConfig, str]:
    """The configuration a failed fold retries with, and what changed. Every
    retry resumes from latest.pt except a diverged fold, which restarts."""
    if _is_out_of_memory(exc) and cfg.batch_size >= 8:
        adapted = replace(cfg, batch_size=cfg.batch_size // 2,
                          grad_accum_steps=cfg.grad_accum_steps * 2)
        return adapted, (f"CUDA out of memory — batch {adapted.batch_size} x "
                         f"{adapted.grad_accum_steps} accumulation (same effective batch).")
    if _is_host_memory_failure(exc):
        return (replace(cfg, num_workers=0, eval_num_workers=0),
                "system memory ran out in the data loader — loading in-process.")
    if isinstance(exc, FoldDiverged):
        shutil.rmtree(config.CHECKPOINT_DIR / cfg.run_name / fold_id, ignore_errors=True)
        return replace(cfg, amp=False), "training diverged — restarting the fold in float32."
    return cfg, "resuming from the last saved epoch."


EPOCH_HEADER = (f"  {'epoch':<7}{'train loss':>11}{'val loss':>10}{'val ordinal':>13}"
                f"{'val acc':>9}{'val F1':>9}{'val MAE':>9}{'time':>8}")


def _cell(value: float) -> str:
    return f"{value:>9.3f}" if np.isfinite(value) else f"{'n/a':>9}"


def _monitored(result: EpochResult) -> float:
    """Validation ORDINAL loss — comparable across ablations with and without
    the complementarity term — falling back to the total loss."""
    value = result.extras.get("ordinal_loss")
    return float(value) if value is not None and np.isfinite(value) else float(result.loss)


def run_fold(fold_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: TrainingConfig,
             device: torch.device, fold_index: int = 1, n_folds: int = 1,
             deadline: Optional[float] = None) -> Tuple[Dict, EpochResult]:
    """Train, select, test and save one fold. Returns (fold_record, test_result)."""
    fold_start = time.monotonic()
    # Per-fold seed: a fold's initialization and shuffling do not depend on
    # which folds ran before it in this session (resuming changes nothing).
    set_seed(cfg.seed + zlib.crc32(fold_id.encode("utf-8")) % 10_000)

    train_df, val_df = speaker_disjoint_train_val_split(train_df, cfg.seed, fold_id=fold_id)
    val_speakers = sorted(val_df["Speaker_ID"].unique())
    train_speakers = set(train_df["Speaker_ID"])
    if set(val_speakers) & train_speakers or set(test_df["Speaker_ID"]) & (train_speakers | set(val_speakers)):
        raise RuntimeError(f"Speaker leakage in fold {fold_id}.")
    if cfg.limit_samples is not None:
        train_df = _limit_samples(train_df, cfg.limit_samples)
        val_df = _limit_samples(val_df, max(2, cfg.limit_samples // 4))
        test_df = _limit_samples(test_df, cfg.limit_samples)

    speaker_label_map = build_speaker_label_map(train_df)
    train_loader, val_loader, test_loader = build_loaders(
        train_df, val_df, test_df, cfg.batch_size, pin_memory=device.type == "cuda",
        speaker_label_map=speaker_label_map, num_workers=cfg.num_workers,
        eval_num_workers=cfg.eval_num_workers, test_num_workers=cfg.test_num_workers)

    model = build_model(cfg.model, num_speakers=len(speaker_label_map),
                        gradient_checkpointing=cfg.gradient_checkpointing).to(device)
    optimizer = build_optimizer(model, cfg.lr_head, cfg.lr_lora, cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(1, cfg.patience // 2))
    class_weights = compute_class_weights(train_df).to(device)
    use_amp = cfg.amp if cfg.amp is not None else device.type == "cuda"
    amp_dtype = resolve_amp_dtype(device)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp and amp_dtype == torch.float16)
    common = dict(device=device, class_weights=class_weights, amp_dtype=amp_dtype,
                  amp_enabled=use_amp)
    early_stopping = EarlyStopping(patience=cfg.patience, mode="min")

    fold_dir = config.CHECKPOINT_DIR / cfg.run_name / fold_id
    best_path, latest_path = fold_dir / "best.pt", fold_dir / "latest.pt"
    writer = SummaryWriter(log_dir=str(config.LOG_DIR / cfg.run_name / fold_id))

    true_label = ", ".join(sorted(test_df["Severity"].unique()))
    print()
    print(H_LIGHT * LINE_WIDTH)
    print(f"  FOLD {fold_index}/{n_folds}  {V}  held-out {fold_id} ({true_label})  {V}  "
          f"train {len(train_df):,} · val {len(val_df):,} · test {len(test_df):,}")
    print(f"  {len(speaker_label_map)} training speakers; validation speakers "
          f"{', '.join(val_speakers)}"
          + (f"; batch {cfg.batch_size} x {cfg.grad_accum_steps} accumulation"
             if cfg.grad_accum_steps > 1 else ""))
    print(H_LIGHT * LINE_WIDTH)

    start_epoch, best_epoch, best_monitored, slowest_epoch_s = 0, None, None, 0.0
    if latest_path.exists():
        state = load_checkpoint(latest_path, model, optimizer, scheduler, scaler,
                                map_location=str(device))
        start_epoch = state["epoch"] + 1
        early_stopping.load_state_dict(state.get("early_stopping") or {})
        best_epoch, best_monitored = state.get("best_epoch"), state.get("best_monitored")
        slowest_epoch_s = float(state.get("slowest_epoch_s") or 0.0)
        print(f"  Resumed from {latest_path.name}: {start_epoch} epoch(s) done "
              f"(best so far: epoch {best_epoch}).")

    epochs_completed = start_epoch
    try:
        if not early_stopping.should_stop and start_epoch < cfg.epochs:
            print(EPOCH_HEADER)
        for epoch in range(start_epoch, cfg.epochs):
            if early_stopping.should_stop:
                break
            if deadline is not None and slowest_epoch_s and time.monotonic() + slowest_epoch_s > deadline:
                raise FoldInterrupted(f"fold {fold_id}: epoch {epoch + 1} would end past the session "
                                      f"cap; {latest_path.name} resumes it next session.")
            epoch_start = time.monotonic()
            label = f"epoch {epoch + 1}/{cfg.epochs}" if cfg.show_batch_progress else ""
            train_result = run_epoch(model, train_loader, optimizer=optimizer, scaler=scaler,
                                     grad_clip_norm=cfg.grad_clip,
                                     grad_accum_steps=cfg.grad_accum_steps,
                                     description=label and f"{label} train", **common)
            val_result = run_epoch(model, val_loader, description=label and f"{label} val",
                                   **common)
            monitored = _monitored(val_result)
            scheduler.step(monitored)

            for name, value in (("Loss/train", train_result.loss), ("Loss/val", val_result.loss),
                                ("Loss/val_monitored", monitored),
                                ("LR/head", optimizer.param_groups[-1]["lr"]),
                                *((f"Train/{k}", v) for k, v in {**train_result.metrics,
                                                                **train_result.extras}.items()),
                                *((f"Val/{k}", v) for k, v in {**val_result.metrics,
                                                              **val_result.extras}.items())):
                writer.add_scalar(name, value, epoch)

            epochs_completed = epoch + 1
            is_best = early_stopping.step(monitored)
            if is_best:
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, monitored)
                best_epoch, best_monitored = epoch + 1, monitored
            epoch_s = time.monotonic() - epoch_start
            slowest_epoch_s = max(slowest_epoch_s, epoch_s)
            # Every epoch, after best.pt, so a crash loses at most one epoch.
            save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, monitored,
                            extra={"early_stopping": early_stopping.state_dict(),
                                   "best_epoch": best_epoch, "best_monitored": best_monitored,
                                   "slowest_epoch_s": slowest_epoch_s, "fold_id": fold_id})

            m = val_result.metrics
            print(f"  {epoch + 1:>3}/{cfg.epochs:<3}{train_result.loss:>11.4f}{val_result.loss:>10.4f}"
                  f"{monitored:>13.4f}{m['accuracy']:>9.3f}{_cell(m['f1'])}{_cell(m['ordinal_mae'])}"
                  f"{epoch_s / 60:>7.1f}m" + ("   * best" if is_best else ""))
            if not (np.isfinite(train_result.loss) and np.isfinite(monitored)):
                print_note(f"Non-finite loss in epoch {epoch + 1} — it cannot become the best checkpoint.")
            if early_stopping.should_stop:
                print(f"  Early stop: no validation improvement for {cfg.patience} epochs "
                      f"(best epoch {best_epoch}).")

        if not best_path.exists():
            raise FoldDiverged(f"fold {fold_id}: no epoch produced a finite validation loss.")
        load_checkpoint(best_path, model, map_location=str(device))
        train_time_s = time.monotonic() - fold_start
        test_start = time.monotonic()
        test = run_epoch(model, test_loader, collect_embeddings=True,
                         description=f"held-out test ({fold_id})" if cfg.show_batch_progress else "",
                         **common)
        inference_time_s = time.monotonic() - test_start
    finally:
        writer.close()

    record = {
        "fold": fold_id, "true_label": ";".join(sorted(test_df["Severity"].unique())),
        "test_loss": test.loss, **test.metrics,
        "pred_distribution": dict(zip(config.SEVERITY_CLASS_NAMES,
                                      np.bincount(test.y_pred, minlength=config.NUM_CLASSES).tolist())),
        "argmax_pred_distribution": dict(zip(config.SEVERITY_CLASS_NAMES, np.bincount(
            test.y_pred_argmax, minlength=config.NUM_CLASSES).tolist())),
        "epochs_completed": epochs_completed, "best_epoch": best_epoch,
        "best_val_monitored": best_monitored, "val_speakers": ";".join(val_speakers),
        "n_train_speakers": len(speaker_label_map), "n_train": len(train_df), "n_val": len(val_df),
        "batch_size": cfg.batch_size, "grad_accum_steps": cfg.grad_accum_steps,
        "train_time_s": train_time_s, "inference_time_s": inference_time_s,
        **{f"test_{k}": v for k, v in test.extras.items()},
        **model.coral_threshold_diagnostics(),
    }
    save_predictions(config.PREDICTIONS_DIR / cfg.run_name / f"{fold_id}.csv", test.filenames,
                     test.speaker_ids, test.y_true, test.y_pred, test.y_prob, test.y_pred_argmax)
    save_confusion_matrix(config.CONFUSION_MATRIX_DIR / cfg.run_name / f"{fold_id}.png",
                          compute_confusion_matrix(test.y_true, test.y_pred),
                          f"{cfg.run_name} — held-out {fold_id}")
    save_embeddings(config.EMBEDDINGS_DIR / cfg.run_name / f"{fold_id}.npz", test.embeddings,
                    test.y_true, test.speaker_ids, test.filenames)
    record["fold_time_s"] = time.monotonic() - fold_start
    # Last: metrics + predictions on disk is the "fold finished" marker.
    save_metrics(config.METRICS_DIR / cfg.run_name / f"{fold_id}.json", record)
    print_fold_report(record)
    return record, test


def _train_fold_with_retries(fold_id, train_df, test_df, cfg, device, fold_index, n_folds, deadline):
    """-> ("ok", (record, test)) | ("deadline", message) | ("user", None) | ("failed", None)."""
    fold_cfg = cfg
    for attempt in range(1 + max(0, cfg.fold_retries)):
        # Workers sized to the memory free now: an application opened mid-run
        # shrinks the pool instead of pushing Windows into paging.
        budget = dict(ram_per_worker_gb=config.DATALOADER_WORKER_RAM_GB,
                      commit_per_worker_gb=config.DATALOADER_WORKER_COMMIT_GB,
                      reserve_commit_gb=config.TRAINING_COMMIT_RESERVE_GB)
        fold_cfg = replace(fold_cfg, num_workers=affordable_workers(fold_cfg.num_workers, **budget),
                           eval_num_workers=affordable_workers(fold_cfg.eval_num_workers, **budget))
        memory = memory_status()
        if (memory["commit_available"] < config.MIN_FREE_COMMIT_GB_PER_FOLD
                or memory["ram_available"] < config.RAM_RESERVE_GB):
            print_note(f"Low memory before fold {fold_id}: {memory['ram_available']:.1f} GiB RAM / "
                       f"{memory['commit_available']:.1f} GiB commit free — close browsers and "
                       "other heavy applications to avoid paging.")
        try:
            return "ok", run_fold(fold_id, train_df, test_df, fold_cfg, device, fold_index, n_folds,
                                  deadline)
        except FoldInterrupted as exc:
            return "deadline", str(exc)
        except KeyboardInterrupt:
            return "user", None
        except Exception as exc:
            fold_cfg, remedy = _adapt_after_failure(exc, fold_cfg, fold_id)
            log_path = config.LOG_DIR / cfg.run_name / f"{fold_id}_attempt{attempt + 1}_error.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            print_status(f"Fold {fold_id} attempt {attempt + 1} failed: {type(exc).__name__}: "
                         f"{str(exc).splitlines()[0][:160]}  (traceback: {log_path.name})", ok=False)
            if attempt < cfg.fold_retries:
                print_note(f"Retrying fold {fold_id}: {remedy}")
        finally:
            shutdown_loaders()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return "failed", None


def run_training(df: pd.DataFrame, cfg: TrainingConfig) -> Tuple[pd.DataFrame, Dict]:
    """Run every configured fold, then pool. Re-calling resumes.

    Returns (per-fold records, pooled metrics incl. run_status). Writes
    RUN_STATUS.json and ALL_FOLDS_pooled.{json,png} under the run's dirs."""
    config.ensure_directories()
    device = resolve_device(cfg.device)
    configure_local_runtime(device)
    run_start = time.monotonic()
    deadline = run_start + cfg.session_hours * 3600 if cfg.session_hours else None

    folds = build_folds(df)
    if cfg.folds:
        folds = [f for f in folds if f[0] in set(cfg.folds)]
    if cfg.max_folds is not None:
        folds = folds[:cfg.max_folds]
    n_folds = len(folds)

    memory = memory_status()
    print_header(f"Training — {cfg.run_name}")
    print_kv("Model", cfg.model)
    print_kv("Protocol", f"severity LOSO, {n_folds} fold(s), speaker-disjoint validation")
    print_kv("Schedule", f"max {cfg.epochs} epochs, patience {cfg.patience}, batch {cfg.batch_size}, "
                         f"lr {cfg.lr_head:g} (head) / {cfg.lr_lora:g} (LoRA), "
                         + (f"AMP {str(resolve_amp_dtype(device)).replace('torch.', '')}"
                            if (cfg.amp if cfg.amp is not None else device.type == "cuda")
                            else "float32"))
    print_kv("Memory free at start", f"{memory['ram_available']:.1f} GiB RAM, "
                                     f"{memory['commit_available']:.1f} GiB commit")
    if config.THERMAL_GUARD_ENABLED:
        print_kv("Thermal guard", f"pause at {config.GPU_TEMP_PAUSE_C} C, resume at "
                                  f"{config.GPU_TEMP_RESUME_C} C")
    if cfg.session_hours:
        print_kv("Session cap", f"{cfg.session_hours:g} h")
    if cfg.limit_samples is not None or n_folds < len(config.DYSARTHRIC_IDS):
        print_note("REDUCED SCALE — a pipeline check, not a reportable result.")

    fold_status: Dict[str, str] = {}
    records, pooled_true, pooled_pred, pooled_prob, pooled_speakers = [], [], [], [], []
    fold_times: List[float] = []

    def skip_from(index: int, first_status: str, reason: str) -> None:
        print_note(reason)
        for j, (fid, _, _) in enumerate(folds[index - 1:], start=index):
            fold_status.setdefault(fid, first_status if j == index else FOLD_SKIPPED_DEADLINE)

    with prevent_sleep():
        for i, (fold_id, train_df, test_df) in enumerate(folds, start=1):
            cached = _load_completed_fold(cfg.run_name, fold_id)
            if cached is not None:
                record, y_true, y_pred, y_prob, speakers = cached
                fold_status[fold_id] = FOLD_CACHED
                print(f"  Fold {i:>2}/{n_folds}  {fold_id:<4} ({record.get('true_label')}) — already "
                      f"complete: accuracy {record.get('accuracy', float('nan')):.3f}, ordinal MAE "
                      f"{record.get('ordinal_mae', float('nan')):.3f}")
            else:
                estimate = float(np.mean(fold_times)) if fold_times else cfg.fold_time_estimate_s
                if deadline is not None and time.monotonic() + (estimate or 0) * cfg.safety_factor > deadline:
                    skip_from(i, FOLD_SKIPPED_DEADLINE,
                              f"Session cap: not starting fold {i}/{n_folds} ({fold_id}); "
                              "re-run to continue.")
                    break
                wait_for_ac_power()
                if fold_times and config.THERMAL_GUARD_ENABLED and config.FOLD_COOLDOWN_S:
                    THERMAL_GUARD.cool_down(max_wait_s=config.FOLD_COOLDOWN_S,
                                            reason="still warm from the previous fold")
                fold_started = time.monotonic()
                outcome, payload = _train_fold_with_retries(fold_id, train_df, test_df, cfg, device,
                                                            i, n_folds, deadline)
                if outcome == "deadline":
                    skip_from(i, FOLD_INTERRUPTED, f"{payload} Later folds are not started.")
                    break
                if outcome == "user":
                    skip_from(i, FOLD_INTERRUPTED, f"Interrupted during fold {fold_id}. Every "
                              "finished epoch is saved — re-run to resume.")
                    break
                if outcome == "failed":
                    print_status(f"Fold {fold_id} FAILED after {1 + cfg.fold_retries} attempts — "
                                 "excluded from pooling; re-running retries it.", ok=False)
                    fold_status[fold_id] = FOLD_FAILED
                    continue
                record, test = payload
                fold_times.append(time.monotonic() - fold_started)
                fold_status[fold_id] = FOLD_COMPLETED
                y_true, y_pred, y_prob, speakers = test.y_true, test.y_pred, test.y_prob, test.speaker_ids

            records.append(record)
            pooled_true.append(y_true)
            pooled_pred.append(y_pred)
            pooled_prob.append(y_prob)
            pooled_speakers.extend(speakers)
            print_runtime_status(
                folds_done=len(fold_status), n_folds=n_folds,
                remaining_folds=sum(fid not in fold_status for fid, _, _ in folds),
                elapsed_s=time.monotonic() - run_start,
                fold_estimate_s=float(np.mean(fold_times)) if fold_times else cfg.fold_time_estimate_s,
                deadline_in_s=(deadline - time.monotonic()) if deadline else None,
                safety_factor=cfg.safety_factor)

    coverage = run_coverage([fid for fid, _, _ in folds], fold_status)
    print_run_coverage(coverage, cfg.run_name)
    if THERMAL_GUARD.pauses:
        print_kv("Thermal pauses", f"{THERMAL_GUARD.pauses} ({THERMAL_GUARD.paused_seconds / 60:.1f} "
                                   "min in total)")
    save_metrics(config.METRICS_DIR / cfg.run_name / "RUN_STATUS.json", coverage)
    summary = pd.DataFrame(records)
    if not records:
        return summary, {"run_status": coverage["status"], "completed_folds": 0}

    y_true, y_pred = np.concatenate(pooled_true), np.concatenate(pooled_pred)
    y_prob = np.concatenate(pooled_prob)
    pooled = compute_metrics(y_true, y_pred, y_prob)
    title = f"{cfg.run_name} — {coverage['completed']}/{coverage['expected']} folds pooled"
    save_confusion_matrix(config.CONFUSION_MATRIX_DIR / cfg.run_name / "ALL_FOLDS_pooled.png",
                          compute_confusion_matrix(y_true, y_pred), title)
    save_roc_curve(config.ROC_DIR / cfg.run_name / "ALL_FOLDS_pooled.png", y_true, y_prob, title)
    speakers = speaker_level(pd.DataFrame({"speaker_id": pooled_speakers, "y_true": y_true,
                                           "y_pred": y_pred, "correct": y_true == y_pred}))
    pooled.update({"run_status": coverage["status"], "expected_folds": coverage["expected"],
                   "completed_folds": coverage["completed"],
                   "speaker_accuracy": float(speakers["Correct"].mean()),
                   "speaker_ordinal_mae": float(speakers["Rank error"].mean())})
    save_metrics(config.METRICS_DIR / cfg.run_name / "ALL_FOLDS_pooled.json", pooled)

    headline = [f"accuracy {pooled['accuracy']:.3f}"]
    headline += [f"{label} {pooled[key]:.3f}" for key, label in
                 (("f1", "macro-F1"), ("balanced_accuracy", "balanced accuracy"),
                  ("ordinal_mae", "ordinal MAE")) if np.isfinite(pooled[key])]
    headline.append(f"speakers correct {int(speakers['Correct'].sum())}/{len(speakers)}")
    print_kv(f"Pooled ({len(y_true):,} utterances)", " · ".join(headline))
    return summary, pooled
