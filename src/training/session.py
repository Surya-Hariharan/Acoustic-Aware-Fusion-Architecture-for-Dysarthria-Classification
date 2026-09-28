"""
Session planning: measure this machine's real throughput, then size the run to
fit a wall-clock budget.

WHY THIS REPLACES THE OLD BENCHMARK STAGE
-----------------------------------------
The previous Kaggle run spent 9,764 s — 2 h 43 m of a 12 h session — on four
benchmarks, each a genuine one-fold-one-epoch training pass:

    batch_size=16 .... 2,425.6 s/epoch,  8.5 samples/s, 2,482 MB peak
    batch_size=24 .... 2,442.3 s/epoch,  8.5 samples/s, 3,348 MB peak
    batch_size=32 .... 2,460.5 s/epoch,  8.4 samples/s, 4,192 MB peak
    ExperimentBudgetManager .. 2,435.4 s   (a duplicate of the third)

Three of those measured the same constant, and the fourth re-measured it a
fourth time. Worse, the constant they found — throughput flat across a 2x batch
range — was itself the diagnosis: a starved GPU, fixed since in src/vad_cache.py.

calibrate_throughput below answers the same question from ~2,000 samples
instead of ~21,000, in roughly a minute instead of forty. It runs the REAL
training path (src.training.engine.run_epoch on a truncated loader), not a
synthetic estimate, so collate, pinning, host-to-device copies, autocast,
backward and gradient clipping are all in the measurement.

The point is not merely to be faster. It is that a measured rate plus a budget
determines how much data the run can afford, so the session is SIZED rather
than started and hoped for — see plan_session.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config
from src.console import (format_duration, print_header, print_kv, print_note,
                         print_status, print_subheader, print_table)

# Candidate data sizes, smallest first. UA-Speech records each speaker's 765
# words as three blocks of 255, and every block carries the SAME 100 common
# words, 10 digits, 26 radio-alphabet letters and 19 computer commands — only
# its 100 uncommon words are unique to it. So a block is a self-contained
# sample of every word category in the corpus, which makes "one block" a
# principled unit to cut to and an easy one to describe in a paper, unlike an
# arbitrary random subsample of utterances.
BLOCK_LADDER: Tuple[Tuple[str, ...], ...] = (("B1",), ("B1", "B2"), ("B1", "B2", "B3"))


@dataclass
class ThroughputMeasurement:
    """What one calibration pass learned about this machine."""
    train_samples_per_s: float
    eval_samples_per_s: float
    batch_size: int
    device: str
    amp_dtype: str
    n_train_measured: int
    n_eval_measured: int
    peak_memory_mb: float = 0.0
    gradient_checkpointing: Optional[bool] = None   # None = config default

    def epoch_seconds(self, n_train: int, n_val: int) -> float:
        """Projected cost of one fold-epoch: a training pass over n_train plus
        a forward-only validation pass over n_val."""
        return n_train / self.train_samples_per_s + n_val / self.eval_samples_per_s


def _calibration_slices(df: pd.DataFrame, model_name: str, task: str, n_train_samples: int,
                        n_eval_samples: int, seed: int
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train/eval slices of fold 1, split exactly as a real fold is
    (src.training.data.split_train_val, default speaker-disjoint protocol).
    Sampled rather than head(): a contiguous slice of a fold's train split is
    ordered by speaker and would measure one speaker's utterance lengths."""
    from src.training.data import split_train_val
    from src.training.runner import TrainingConfig, build_folds

    cfg = TrainingConfig(task=task, model=model_name, seed=seed)
    fold_id, train_df, _ = next(iter(build_folds(df, task, cfg)))
    label_column = "Group" if task == "detection" else "Severity"
    train_df, val_df = split_train_val(train_df, label_column, cfg.val_protocol, cfg.seed,
                                       fold_id=fold_id, val_fraction=cfg.val_fraction)
    train_slice = train_df.sample(n=min(n_train_samples, len(train_df)),
                                  random_state=seed).reset_index(drop=True)
    eval_slice = val_df.sample(n=min(n_eval_samples, len(val_df)),
                               random_state=seed).reset_index(drop=True)
    return train_slice, eval_slice


def _measure(train_slice: pd.DataFrame, eval_slice: pd.DataFrame, model_name: str,
             task: str, batch_size: int, num_workers: int, seed: int,
             gradient_checkpointing: Optional[bool] = None) -> ThroughputMeasurement:
    """One timed measurement of the REAL training step at one configuration.

    Deliberately reuses src.training.engine.run_epoch rather than hand-rolling
    a timing loop: a hand-rolled loop would drift from the real step (autocast
    dtype, gradient clipping, the model's own multi-term training_step) and
    would then be measuring something the run never does.

    A warm-up pass over EACH loader runs first and is discarded — together they
    pay for CUDA context creation, cuDNN autotuning, both loaders' worker
    spawn, the wav2vec2 weight load and the first touch of the disk caches.
    Folding those one-time costs into a per-sample rate is exactly the error
    that made the old benchmark unusable for projection. BOTH loaders, not
    just the training one: each spawns its own persistent workers on first
    iteration, and leaving that cold start inside the eval measurement once
    reported eval as SLOWER than training.
    """
    from src.training.data import build_loaders, build_speaker_label_map, compute_class_weights
    from src.training.engine import build_optimizer, run_epoch
    from src.training.models import build_model
    from src.training.utils import resolve_device, set_seed

    set_seed(seed)
    device = resolve_device(None)
    speaker_label_map = build_speaker_label_map(train_slice)
    train_loader, eval_loader, _ = build_loaders(
        train_slice, eval_slice, eval_slice.iloc[:0], batch_size, num_workers,
        pin_memory=(device.type == "cuda"), model_name=model_name,
        speaker_label_map=speaker_label_map)

    num_classes = config.NUM_CLASSES[task]
    model = build_model(model_name, num_classes, num_speakers=len(speaker_label_map),
                        gradient_checkpointing=gradient_checkpointing).to(device)
    optimizer = build_optimizer(model, config.DEFAULT_LR_HEAD, config.DEFAULT_LR_BACKBONE,
                                config.DEFAULT_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_slice, task).to(device))

    use_amp = device.type == "cuda"
    supports_bf16 = use_amp and torch.cuda.get_device_capability(device)[0] >= 8
    amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    scaler = torch.amp.GradScaler(device=device.type,
                                  enabled=use_amp and amp_dtype == torch.float16)
    common = dict(criterion=criterion, device=device, task=task, scaler=scaler,
                  grad_clip_norm=config.DEFAULT_GRAD_CLIP_NORM, amp_dtype=amp_dtype,
                  amp_enabled=use_amp)

    try:
        # -- warm-up (discarded) ---------------------------------------------
        run_epoch(model, train_loader, optimizer=optimizer, train=True, **common)
        run_epoch(model, eval_loader, optimizer=None, train=False, **common)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        # -- timed training pass ---------------------------------------------
        start = time.monotonic()
        run_epoch(model, train_loader, optimizer=optimizer, train=True, **common)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_rate = len(train_slice) / max(time.monotonic() - start, 1e-9)

        # -- timed eval pass -------------------------------------------------
        start = time.monotonic()
        run_epoch(model, eval_loader, optimizer=None, train=False, **common)
        if device.type == "cuda":
            torch.cuda.synchronize()
        eval_rate = len(eval_slice) / max(time.monotonic() - start, 1e-9)

        peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2
                   if device.type == "cuda" else 0.0)
    finally:
        del model, optimizer, train_loader, eval_loader
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except RuntimeError:        # the CUDA error that brought us here, re-raised
                pass

    return ThroughputMeasurement(
        train_samples_per_s=train_rate, eval_samples_per_s=eval_rate,
        batch_size=batch_size, device=str(device),
        amp_dtype=str(amp_dtype).replace("torch.", ""),
        n_train_measured=len(train_slice), n_eval_measured=len(eval_slice),
        peak_memory_mb=peak_mb, gradient_checkpointing=gradient_checkpointing)


def calibrate_throughput(df: pd.DataFrame, model_name: str, task: str = "severity",
                         batch_size: int = config.DEFAULT_BATCH_SIZE,
                         num_workers: int = 4,
                         n_train_samples: int = 960,
                         n_eval_samples: int = 320,
                         seed: int = config.DEFAULT_SEED,
                         gradient_checkpointing: Optional[bool] = None) -> ThroughputMeasurement:
    """Measure real train and eval throughput on a truncated slice of fold 1
    at one configuration (see _measure). Writes nothing to outputs/."""
    print_subheader("Throughput calibration")
    train_slice, eval_slice = _calibration_slices(df, model_name, task, n_train_samples,
                                                  n_eval_samples, seed)
    print_kv("Calibration slice", f"{len(train_slice):,} train / {len(eval_slice):,} eval "
                                  f"at batch size {batch_size}")
    print_status("Warm-up passes over both loaders (CUDA context, cuDNN autotune, "
                 "worker spawn, wav2vec2 load) — not timed...", ok=True)
    measurement = _measure(train_slice, eval_slice, model_name, task, batch_size,
                           num_workers, seed, gradient_checkpointing=gradient_checkpointing)
    print_kv("Device", measurement.device)
    print_kv("AMP dtype", measurement.amp_dtype)
    print_kv("Train throughput", f"{measurement.train_samples_per_s:,.1f} samples/s")
    print_kv("Eval throughput", f"{measurement.eval_samples_per_s:,.1f} samples/s")
    if measurement.peak_memory_mb:
        print_kv("Peak GPU memory", f"{measurement.peak_memory_mb:,.0f} MB")
    _warn_if_still_input_bound(measurement.train_samples_per_s)
    return measurement


def benchmark_batch_sizes(df: pd.DataFrame, model_name: str, task: str = "severity",
                          batch_sizes: Sequence[int] = (32, 64, 96, 128),
                          checkpointing_options: Sequence[bool] = (True, False),
                          num_workers: int = 4, timed_batches: int = 6,
                          min_gain: float = 0.10, max_memory_fraction: float = 0.85,
                          seed: int = config.DEFAULT_SEED
                          ) -> Tuple[ThroughputMeasurement, pd.DataFrame]:
    """Short throughput/memory benchmark over batch size x gradient
    checkpointing, then pick the configuration to train with.

    Each configuration trains `timed_batches` batches after a discarded
    warm-up of the same size (a few minutes in total, not an epoch per
    candidate). A configuration is STABLE if it neither runs out of memory nor
    peaks above `max_memory_fraction` of the card — headroom for the longest
    utterances and allocator fragmentation over 15 folds.

    Selection: per batch size, the faster checkpointing setting (the two are
    numerically identical). Then, walking batch sizes upward, a larger batch
    replaces the current choice only if it is stable AND at least `min_gain`
    faster — "largest stable batch that buys meaningful throughput", not the
    largest that fits. Larger batches also mean fewer optimizer steps per
    epoch at an unchanged learning rate, a real optimization change, so a
    marginal speedup is not worth it.

    Returns (chosen measurement, table of every configuration).
    """
    print_subheader("Batch-size benchmark")
    device_total_mb = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
                       if torch.cuda.is_available() else float("inf"))
    rows: List[Dict] = []
    measurements: Dict[Tuple[int, bool], ThroughputMeasurement] = {}

    for gradient_checkpointing in checkpointing_options:
        previous_rate = 0.0
        for batch_size in sorted(batch_sizes):
            train_slice, eval_slice = _calibration_slices(
                df, model_name, task, n_train_samples=batch_size * timed_batches,
                n_eval_samples=2 * batch_size, seed=seed)
            row = {"batch_size": batch_size, "grad_checkpointing": gradient_checkpointing}
            try:
                m = _measure(train_slice, eval_slice, model_name, task, batch_size,
                             num_workers, seed, gradient_checkpointing=gradient_checkpointing)
            except RuntimeError as exc:          # torch.cuda.OutOfMemoryError is a subclass
                if "out of memory" not in str(exc):
                    raise
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
                rows.append({**row, "train_samples_s": float("nan"),
                             "eval_samples_s": float("nan"), "peak_mb": float("nan"),
                             "stable": False, "note": "OOM"})
                print_kv(f"bs={batch_size} ckpt={gradient_checkpointing}", "out of memory")
                break                       # larger batches at this setting will OOM too
            stable = m.peak_memory_mb <= max_memory_fraction * device_total_mb
            measurements[(batch_size, gradient_checkpointing)] = m
            rows.append({**row, "train_samples_s": round(m.train_samples_per_s, 1),
                         "eval_samples_s": round(m.eval_samples_per_s, 1),
                         "peak_mb": round(m.peak_memory_mb),
                         "stable": stable,
                         "note": "" if stable else f">{max_memory_fraction:.0%} of GPU memory"})
            print_kv(f"bs={batch_size} ckpt={gradient_checkpointing}",
                     f"{m.train_samples_per_s:6.1f} train/s  {m.eval_samples_per_s:6.1f} eval/s  "
                     f"peak {m.peak_memory_mb:,.0f} MB")
            if not stable:
                # Over the memory ceiling: a larger batch only goes further over
                # (and on Windows/WDDM spills into system RAM instead of raising
                # OOM, collapsing throughput rather than failing).
                break
            if m.train_samples_per_s < previous_rate:
                break                       # slower than the smaller batch: stop escalating
            previous_rate = m.train_samples_per_s

    table = pd.DataFrame(rows)
    best_per_size: Dict[int, ThroughputMeasurement] = {}
    for (batch_size, _), m in measurements.items():
        stable_row = table[(table["batch_size"] == batch_size)
                           & (table["grad_checkpointing"] == m.gradient_checkpointing)]
        if not bool(stable_row["stable"].iloc[0]):
            continue
        current = best_per_size.get(batch_size)
        if current is None or m.train_samples_per_s > current.train_samples_per_s:
            best_per_size[batch_size] = m
    if not best_per_size:
        raise RuntimeError("No stable batch-size configuration — every candidate ran out of "
                           "memory or exceeded the memory ceiling.")

    chosen = None
    for batch_size in sorted(best_per_size):
        candidate = best_per_size[batch_size]
        if chosen is None or candidate.train_samples_per_s >= chosen.train_samples_per_s * (1 + min_gain):
            chosen = candidate
    table["chosen"] = [(r.batch_size == chosen.batch_size
                        and r.grad_checkpointing == chosen.gradient_checkpointing)
                       for r in table.itertuples()]
    print()
    print_table(table)
    print_status(f"Chosen: batch size {chosen.batch_size}, gradient checkpointing "
                 f"{'on' if chosen.gradient_checkpointing else 'off'} — "
                 f"{chosen.train_samples_per_s:,.1f} train samples/s, "
                 f"peak {chosen.peak_memory_mb:,.0f} MB "
                 f"(a larger batch had to be >= {min_gain:.0%} faster to be chosen)", ok=True)
    return chosen, table


def _warn_if_still_input_bound(train_rate: float) -> None:
    """The previous run's signature failure was 8.5 samples/s with the GPU at
    28% memory and flat across batch size. If calibration lands anywhere near
    that, the dataloader is still starving the GPU and no amount of budget
    arithmetic will save the session — say so before it starts, not after."""
    if train_rate < 15.0:
        print_note(
            f"Measured {train_rate:,.1f} samples/s, which is close to the 8.5 "
            f"samples/s of the input-bound run this pipeline was fixed to avoid. "
            f"Check that outputs/feature_cache/vad_spans.parquet exists and "
            f"loads (src.vad_cache.load_span_table), and that the framewise "
            f".npy caches are warm — a cold or stale cache puts two Silero "
            f"forward passes back on every item."
        )


# ---------------------------------------------------------------------------
# Sizing the run to the budget
# ---------------------------------------------------------------------------
@dataclass
class SessionPlan:
    """The configuration that fits the budget, and the arithmetic behind it."""
    blocks: Tuple[str, ...]
    n_folds: int
    epochs: int
    projected_seconds: float
    budget_seconds: float
    projection: pd.DataFrame = field(repr=False)
    fits: bool = True
    min_headroom: float = 1.25
    # The speakers this plan actually evaluates, in evaluation order — the
    # full config.SEVERITY_LOSO_ORDER unless max_folds truncated it. Kept
    # here (not re-derived from n_folds) so print_session_plan and
    # print_data_coverage_statement can report real coverage instead of
    # assuming "all 15" whenever n_folds happens to look plausible.
    fold_speakers: Tuple[str, ...] = ()

    @property
    def headroom(self) -> float:
        """Budget / projection — how wrong the measurement can be before the
        session overruns. Below ~1.3 the plan is not really a plan."""
        return self.budget_seconds / max(self.projected_seconds, 1e-9)


def subset_blocks(df: pd.DataFrame, blocks: Sequence[str]) -> pd.DataFrame:
    """Restrict the manifest to the given UA-Speech blocks."""
    return df[df["Block"].isin(list(blocks))].reset_index(drop=True)


def project_runtime(df: pd.DataFrame, measurement: ThroughputMeasurement,
                    task: str = "severity", model_name: str = "gated_fusion_three_branch",
                    n_folds: Optional[int] = None, epochs: int = config.DEFAULT_EPOCHS,
                    fold_setup_seconds: float = 30.0,
                    fold_overhead_seconds: float = 60.0,
                    seed: int = config.DEFAULT_SEED,
                    max_folds: Optional[int] = None) -> Dict[str, float]:
    """Projected wall-clock for a full run over `df`, at the measured rate.

    Fold sizes are taken from the ACTUAL first fold of `df` rather than from
    len(df) arithmetic, so a block subset, an uneven speaker or a dropped file
    is reflected instead of assumed.

    fold_setup_seconds covers build_loaders' per-fold standardizer passes,
    which the span cache reduced from ~1,250 s to a table lookup;
    fold_overhead_seconds covers the test pass, embedding collection and
    artifact writes that run_fold does once per fold (measured at ~80 s).

    `max_folds`, if given, truncates the materialized fold list exactly as
    src.training.runner.run_training does (fold_iter[:cfg.max_folds]) — so a
    budget-reduced session's projection reflects the folds that will
    actually run, not the full protocol size. build_folds() itself ignores
    TrainingConfig.max_folds (that truncation is run_training's job), which
    is why it is applied here explicitly rather than via `cfg`.
    """
    from src.training.data import split_train_val
    from src.training.runner import TrainingConfig, build_folds

    cfg = TrainingConfig(task=task, model=model_name, seed=seed)
    # build_folds yields lazily for the larger protocols, so materialize it —
    # the count is the whole point here.
    folds = list(build_folds(df, task, cfg))
    if max_folds is not None:
        folds = folds[:max_folds]
    n_folds = n_folds if n_folds is not None else len(folds)

    # Mean train/val size over EVERY fold, split exactly as run_fold splits it
    # (speaker-disjoint validation: 3 or 4 validation speakers depending on the
    # held-out speaker's class, so fold sizes genuinely differ).
    label_column = "Group" if task == "detection" else "Severity"
    sizes = []
    for fold_id, train_df, test_df in folds:
        fold_train, fold_val = split_train_val(train_df, label_column, cfg.val_protocol,
                                               cfg.seed, fold_id=fold_id,
                                               val_fraction=cfg.val_fraction)
        sizes.append((len(fold_train), len(fold_val), len(test_df)))
    n_train, n_val, n_test = (int(round(x)) for x in np.mean(sizes, axis=0))

    epoch_s = measurement.epoch_seconds(n_train, n_val)
    fold_s = fold_setup_seconds + epochs * epoch_s + fold_overhead_seconds
    return {
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "n_folds": n_folds, "epochs": epochs,
        "epoch_seconds": epoch_s, "fold_seconds": fold_s,
        "total_seconds": n_folds * fold_s,
        "fold_speakers": tuple(fold_id for fold_id, _, _ in folds),
    }


def plan_session(df: pd.DataFrame, measurement: ThroughputMeasurement,
                 budget_seconds: float, task: str = "severity",
                 model_name: str = "gated_fusion_three_branch",
                 epochs: int = config.DEFAULT_EPOCHS,
                 ladder: Sequence[Sequence[str]] = BLOCK_LADDER,
                 reserve_seconds: float = 0.0,
                 min_headroom: float = 1.25,
                 seed: int = config.DEFAULT_SEED,
                 max_folds: Optional[int] = None) -> SessionPlan:
    """Choose the largest data size on `ladder` that fits the remaining budget.

    By default, all folds are kept: the severity protocol's claim is at n = 15
    speakers, so dropping folds would not shrink the experiment, it would
    invalidate it; utterances per speaker is the dimension that can give.

    `max_folds`, if given, is a DELIBERATE, EXPLICIT override of that default
    — e.g. a session whose budget cannot fit even one block at 15 folds.
    It is threaded into every rung's projection so the table and the chosen
    plan both reflect the reduced fold count, and into the returned
    SessionPlan.fold_speakers so callers can report exactly which speakers
    were (and were not) evaluated, rather than assuming full coverage.

    min_headroom is the margin the projection must clear, not merely meet. The
    measurement behind it is a one-minute sample of a ten-hour run, and the run
    itself is not stationary: later folds contend with a filling disk, thermal
    throttling and Kaggle's shared hosts. A rung projected at exactly 100% of
    the budget is a coin flip, so 1.25 means "choose the larger rung only when
    the measured rate could be 25% optimistic and the session would still
    land". Early stopping (patience 3) usually returns more slack than this on
    top, since folds rarely run all `epochs`.

    Returns the smallest rung with a note when even that overruns, rather than
    refusing to plan — run_training's deadline still stops cleanly at a fold
    boundary, so an honest "this will be partial" beats an exception.
    """
    available = budget_seconds - reserve_seconds
    rows, chosen = [], None

    for blocks in ladder:
        subset = subset_blocks(df, blocks)
        projection = project_runtime(subset, measurement, task=task,
                                     model_name=model_name, epochs=epochs, seed=seed,
                                     max_folds=max_folds)
        fits = projection["total_seconds"] * min_headroom <= available
        rows.append({
            "blocks": "+".join(blocks),
            "utterances_per_speaker": len(subset) // subset["Speaker_ID"].nunique(),
            "train_per_fold": projection["n_train"],
            "epoch": format_duration(projection["epoch_seconds"]),
            "per_fold": format_duration(projection["fold_seconds"]),
            f"total_x{projection['n_folds']}_folds": format_duration(projection["total_seconds"]),
            "headroom": f"{available / max(projection['total_seconds'], 1e-9):.2f}x",
            "fits": "yes" if fits else "no",
        })
        if fits:
            chosen = (blocks, projection)

    table = pd.DataFrame(rows)
    if chosen is None:
        smallest = ladder[0]
        projection = project_runtime(subset_blocks(df, smallest), measurement,
                                     task=task, model_name=model_name,
                                     epochs=epochs, seed=seed, max_folds=max_folds)
        print_note(
            f"Even {'+'.join(smallest)} projects "
            f"{format_duration(projection['total_seconds'])}, over the "
            f"{format_duration(available)} available. Running it anyway — the "
            f"deadline wired into run_training stops cleanly at a fold boundary, "
            f"so this session will finish some folds and a later one can resume "
            f"the rest."
        )
        return SessionPlan(blocks=tuple(smallest), n_folds=projection["n_folds"],
                           epochs=epochs, projected_seconds=projection["total_seconds"],
                           budget_seconds=available, projection=table, fits=False,
                           min_headroom=min_headroom, fold_speakers=projection["fold_speakers"])

    blocks, projection = chosen
    return SessionPlan(blocks=tuple(blocks), n_folds=projection["n_folds"],
                       epochs=epochs, projected_seconds=projection["total_seconds"],
                       budget_seconds=available, projection=table, fits=True,
                       min_headroom=min_headroom, fold_speakers=projection["fold_speakers"])


def print_session_plan(plan: SessionPlan, measurement: ThroughputMeasurement) -> None:
    """Report the plan and the arithmetic that produced it."""
    print_header("Session plan")
    print_kv("Measured train throughput", f"{measurement.train_samples_per_s:,.1f} samples/s")
    print_kv("Measured eval throughput", f"{measurement.eval_samples_per_s:,.1f} samples/s")
    print()
    print_table(plan.projection)
    print()
    total_dysarthric = len(config.DYSARTHRIC_IDS)
    is_full_coverage = len(plan.fold_speakers) >= total_dysarthric
    print_kv("Blocks selected", "+".join(plan.blocks))
    print_kv("Folds", f"{plan.n_folds} (all dysarthric speakers — never reduced)"
             if is_full_coverage
             else f"{plan.n_folds} of {total_dysarthric} dysarthric speakers "
                  "(BUDGET-REDUCED, see note below)")
    print_kv("Epochs per fold (max)", plan.epochs)
    print_kv("Projected total", format_duration(plan.projected_seconds))
    print_kv("Budget", format_duration(plan.budget_seconds))
    if plan.fits:
        print_status(f"Fits with {plan.headroom:.2f}x headroom — the measured rate can be "
                     f"that far optimistic before the session overruns "
                     f"(minimum required: {plan.min_headroom:.2f}x)", ok=True)
    else:
        print_status("Does NOT fit — this session will be partial and resumable", ok=False)
    if not is_full_coverage:
        missing = sorted(set(config.DYSARTHRIC_IDS) - set(plan.fold_speakers))
        print_note(
            f"This session evaluates only {len(plan.fold_speakers)} of "
            f"{total_dysarthric} dysarthric speakers — a deliberate compute-budget "
            f"concession (max_folds), NOT the full-population primary protocol. "
            f"Speakers never held out this session: {missing}. Report any result "
            f"from this run as a reduced-population check, not the primary "
            f"15-speaker severity result."
        )
    print_data_coverage_statement(plan)


def print_data_coverage_statement(plan: SessionPlan) -> None:
    """State, in one place, exactly what a block-limited (and, if max_folds
    was set, fold-limited) run does and does not cut.

    At the DEFAULT (max_folds=None), nothing is ever cut at the speaker/fold
    level: every one of config.DYSARTHRIC_IDS (all 15 dysarthric speakers,
    all 4 severity classes) is both trained on and held out as the LOSO test
    speaker in some fold — see src.splits.iter_severity_loso_folds and
    plan_session's own docstring ("dropping folds would not shrink the
    experiment, it would invalidate it"). A Kaggle time budget then can only
    shrink UTTERANCES PER SPEAKER, via which blocks are included.

    Why that specific cut is principled rather than an arbitrary truncation:
    UA-Speech's three blocks each carry the SAME common words, digits,
    radio-alphabet letters and computer commands (see BLOCK_LADDER's own
    comment) — only each block's uncommon words are unique to it. So "trained
    on block B1" means "trained on one full, balanced pass over every word
    CATEGORY in the corpus, at one third of the utterances-per-speaker" — not
    a random subsample that could have over- or under-represented any
    category by chance.

    If plan.fold_speakers is a PROPER subset of config.DYSARTHRIC_IDS (i.e.
    plan_session was called with an explicit max_folds), this instead reports
    the reduced speaker/severity-class coverage honestly, rather than
    asserting the never-reduced claim above.
    """
    words_per_speaker = len(plan.blocks) * (config.WORDS_PER_SPEAKER // 3)
    coverage_pct = 100 * words_per_speaker / config.WORDS_PER_SPEAKER
    fold_speakers = plan.fold_speakers or tuple(config.DYSARTHRIC_IDS)
    total_dysarthric = len(config.DYSARTHRIC_IDS)
    is_full_coverage = len(fold_speakers) >= total_dysarthric

    severity_counts: Dict[str, int] = {}
    for speaker in fold_speakers:
        severity_counts[config.SEVERITY_MAP[speaker]] = (
            severity_counts.get(config.SEVERITY_MAP[speaker], 0) + 1)

    print()
    print_subheader("Data coverage — what this run's block subset does and does not cut")
    if is_full_coverage:
        print_kv("Dysarthric speakers", f"{total_dysarthric} of {total_dysarthric} "
                 f"(100% — every LOSO protocol always trains on, and separately "
                 f"holds out, every speaker)")
    else:
        print_kv("Dysarthric speakers", f"{len(fold_speakers)} of {total_dysarthric} "
                 f"(BUDGET-REDUCED via max_folds — NOT full LOSO coverage)")
    print_kv("Severity classes covered", ", ".join(
        f"{sev} x{n}" for sev, n in sorted(severity_counts.items())))
    print_kv("Utterances per speaker used", f"{words_per_speaker} of "
             f"{config.WORDS_PER_SPEAKER} ({coverage_pct:.0f}%, block(s) "
             f"{'+'.join(plan.blocks)})")
    print_note(
        "Every UA-Speech block carries the same common-word/digit/letter/"
        "command categories — only its uncommon words are block-specific — so "
        "this is a balanced reduction in utterances per speaker" +
        (" at full speaker and severity-class coverage, not a random or "
         "biased subsample." if is_full_coverage else
         ", independent of the max_folds speaker reduction noted above.")
    )
