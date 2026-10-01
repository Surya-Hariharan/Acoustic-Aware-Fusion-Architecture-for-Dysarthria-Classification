"""
Measure this machine's real throughput and project the run time from it.

calibrate_throughput runs the REAL training step (src.training.engine.run_epoch
on a slice of fold 1 — collate, pinning, host-to-device copies, autocast,
backward, clipping) after an untimed warm-up, so the projection rests on a
measurement, not an assumption. benchmark_batch_sizes repeats that across
batch size x gradient checkpointing to choose a configuration.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src import config
from src.console import print_kv, print_note, print_status, print_subheader, print_table


@dataclass
class ThroughputMeasurement:
    train_samples_per_s: float
    eval_samples_per_s: float
    batch_size: int
    peak_memory_mb: float = 0.0
    gradient_checkpointing: Optional[bool] = None

    def epoch_seconds(self, n_train: int, n_val: int) -> float:
        return n_train / self.train_samples_per_s + n_val / self.eval_samples_per_s


def subset_blocks(df: pd.DataFrame, blocks: Sequence[str]) -> pd.DataFrame:
    """Restrict the manifest to the given UA-Speech recording blocks."""
    return df[df["Block"].isin(list(blocks))].reset_index(drop=True)


def _fold_splits(df: pd.DataFrame, seed: int):
    """(fold_id, train, val, test) for every fold, split exactly as run_fold does."""
    from src.training.data import speaker_disjoint_train_val_split
    from src.training.runner import build_folds

    for fold_id, train_df, test_df in build_folds(df):
        train_df, val_df = speaker_disjoint_train_val_split(train_df, seed, fold_id=fold_id)
        yield fold_id, train_df, val_df, test_df


def _measure(df: pd.DataFrame, model_name: str, batch_size: int, n_train: int, n_eval: int,
             seed: int, gradient_checkpointing: Optional[bool]) -> ThroughputMeasurement:
    """Time one training pass and one evaluation pass on a sample of fold 1,
    after an untimed warm-up of both (CUDA context, cuDNN autotuning, worker
    spawn, wav2vec2 load)."""
    from src.training.data import (build_loaders, build_speaker_label_map, compute_class_weights,
                                   shutdown_loaders)
    from src.training.engine import build_optimizer, run_epoch
    from src.training.models import build_model
    from src.training.utils import configure_local_runtime, resolve_amp_dtype, resolve_device, set_seed

    _, train_df, val_df, _ = next(_fold_splits(df, seed))
    train_slice = train_df.sample(n=min(n_train, len(train_df)), random_state=seed)
    eval_slice = val_df.sample(n=min(n_eval, len(val_df)), random_state=seed)

    set_seed(seed)
    device = resolve_device(None)
    configure_local_runtime(device)
    speaker_map = build_speaker_label_map(train_slice)
    train_loader, eval_loader, _ = build_loaders(train_slice, eval_slice, eval_slice.iloc[:0],
                                                 batch_size, pin_memory=device.type == "cuda",
                                                 speaker_label_map=speaker_map)
    model = build_model(model_name, num_speakers=len(speaker_map),
                        gradient_checkpointing=gradient_checkpointing).to(device)
    optimizer = build_optimizer(model, config.DEFAULT_LR_HEAD, config.DEFAULT_LR_LORA,
                                config.DEFAULT_WEIGHT_DECAY)
    amp_dtype = resolve_amp_dtype(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp and amp_dtype == torch.float16)
    common = dict(device=device, class_weights=compute_class_weights(train_slice).to(device),
                  amp_dtype=amp_dtype, amp_enabled=use_amp)

    def timed(fn) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.monotonic()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return max(time.monotonic() - start, 1e-9)

    def train_pass():
        run_epoch(model, train_loader, optimizer=optimizer, scaler=scaler, **common)

    def eval_pass():
        run_epoch(model, eval_loader, **common)

    try:
        train_pass()
        eval_pass()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_rate = len(train_slice) / timed(train_pass)
        eval_rate = len(eval_slice) / timed(eval_pass)
        peak_mb = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.0
    finally:
        shutdown_loaders()
        del model, optimizer, train_loader, eval_loader
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
    return ThroughputMeasurement(train_rate, eval_rate, batch_size, peak_mb, gradient_checkpointing)


def calibrate_throughput(df: pd.DataFrame, model_name: str,
                         batch_size: int = config.DEFAULT_BATCH_SIZE,
                         n_train_samples: int = 640, n_eval_samples: int = 320,
                         seed: int = config.DEFAULT_SEED,
                         gradient_checkpointing: Optional[bool] = None) -> ThroughputMeasurement:
    """Measured train/eval throughput at the training configuration (~1 min)."""
    print_subheader("Throughput calibration")
    print_kv("Calibration slice", f"{n_train_samples:,} train / {n_eval_samples:,} eval at batch "
                                  f"{batch_size} (after an untimed warm-up)")
    m = _measure(df, model_name, batch_size, n_train_samples, n_eval_samples, seed,
                 gradient_checkpointing)
    print_kv("Train / eval throughput", f"{m.train_samples_per_s:,.1f} / "
                                        f"{m.eval_samples_per_s:,.1f} samples/s")
    if m.peak_memory_mb:
        print_kv("Peak GPU memory", f"{m.peak_memory_mb:,.0f} MB")
    if m.train_samples_per_s < 15.0:
        print_note(f"{m.train_samples_per_s:.1f} samples/s means the data loader is starving the "
                   "GPU — check that the feature store is complete (Section 3).")
    return m


def benchmark_batch_sizes(df: pd.DataFrame, model_name: str,
                          batch_sizes: Sequence[int] = (32, 64),
                          checkpointing_options: Sequence[bool] = (False, True),
                          timed_batches: int = 6, min_gain: float = 0.10,
                          max_memory_fraction: float = 0.85, seed: int = config.DEFAULT_SEED
                          ) -> Tuple[ThroughputMeasurement, pd.DataFrame]:
    """Throughput and peak memory over batch size x gradient checkpointing.
    A configuration is stable if it neither OOMs nor peaks above
    max_memory_fraction of VRAM; a larger batch is chosen only if it is stable
    AND at least min_gain faster (it also changes the optimization)."""
    print_subheader("Batch-size benchmark")
    total_mb = (torch.cuda.get_device_properties(0).total_memory / 2 ** 20
                if torch.cuda.is_available() else float("inf"))
    rows: List[Dict] = []
    for checkpointing in checkpointing_options:
        for batch_size in sorted(batch_sizes):
            row = {"batch_size": batch_size, "grad_checkpointing": checkpointing}
            try:
                m = _measure(df, model_name, batch_size, batch_size * timed_batches,
                             2 * batch_size, seed, checkpointing)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                rows.append({**row, "train_samples_s": np.nan, "peak_mb": np.nan,
                             "stable": False, "measurement": None})
                break
            stable = m.peak_memory_mb <= max_memory_fraction * total_mb
            rows.append({**row, "train_samples_s": round(m.train_samples_per_s, 1),
                         "peak_mb": round(m.peak_memory_mb), "stable": stable, "measurement": m})
            if not stable:
                break
    candidates = sorted((r for r in rows if r["stable"]), key=lambda r: r["batch_size"])
    if not candidates:
        raise RuntimeError("No stable configuration — every candidate ran out of memory.")
    chosen = None
    for r in candidates:
        if chosen is None or r["train_samples_s"] >= chosen["train_samples_s"] * (1 + min_gain):
            chosen = r
    table = pd.DataFrame(rows).drop(columns="measurement")
    table["chosen"] = [r is chosen for r in rows]
    print_table(table)
    m = chosen["measurement"]
    print_status(f"Chosen: batch {m.batch_size}, gradient checkpointing "
                 f"{'on' if m.gradient_checkpointing else 'off'} — {m.train_samples_per_s:,.1f} "
                 f"samples/s, peak {m.peak_memory_mb:,.0f} MB", ok=True)
    return m, table


def project_runtime(df: pd.DataFrame, measurement: ThroughputMeasurement,
                    epochs: int = config.DEFAULT_EPOCHS, fold_overhead_seconds: float = 90.0,
                    seed: int = config.DEFAULT_SEED) -> Dict[str, float]:
    """Upper-bound wall clock for the full run at the measured rate, from the
    mean train/val/test sizes of the real folds. fold_overhead_seconds covers
    the standardizer pass, model build, test pass and artifact writes."""
    sizes = [(len(tr), len(va), len(te)) for _, tr, va, te in _fold_splits(df, seed)]
    n_train, n_val, n_test = (int(round(x)) for x in np.mean(sizes, axis=0))
    epoch_s = measurement.epoch_seconds(n_train, n_val)
    fold_s = epochs * epoch_s + n_test / measurement.eval_samples_per_s + fold_overhead_seconds
    return {"n_train": n_train, "n_val": n_val, "n_test": n_test, "n_folds": len(sizes),
            "epochs": epochs, "epoch_seconds": epoch_s, "fold_seconds": fold_s,
            "total_seconds": len(sizes) * fold_s}
