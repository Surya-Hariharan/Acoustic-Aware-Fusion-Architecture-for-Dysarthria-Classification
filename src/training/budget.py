"""
Experiment compute-budget manager.

Every training-loop cell in notebooks/03_training.ipynb previously hand-rolled
the same pattern: pick SESSION_BUDGET_HOURS, compute a time.monotonic() deadline,
pass it into run_training(..., deadline=...), and hope it fits. Nothing measured
the actual per-variant cost first.

ExperimentBudgetManager replaces the guess with a measurement: it benchmarks one
LOSO fold x one epoch per model variant (real wall-clock, via the existing
run_training/run_fold path — no separate training code), projects each variant's
full-run cost, and allocates a slice of a hard wall-clock cap to each variant
proportional to its measured cost. The resulting per-variant deadline is fed into
the SAME run_training(..., deadline=...) call every screening/LOSO cell already
uses — src.training.runner.run_training already stops cleanly at a fold boundary
(the deadline check happens before run_fold is called, never mid-fold), so this
module adds no new stopping logic, only a measured way to set the deadline.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import torch

from src import config
from src.console import print_header, print_kv, print_note, print_status, print_subheader, print_table
from src.training.models import MODELS_WITH_CACHEABLE_FROZEN_EMBEDDING
from src.training.runner import TrainingConfig, run_training
from src.training.utils import resolve_device


@dataclass
class ExperimentBudgetManager:
    """
    Usage:
        budget = ExperimentBudgetManager(models=PRIMARY_DETECTION_MODELS,
                                         hard_cap_hours=6.0)
        budget.benchmark(df_m6, task="detection")
        for name in budget.models:
            deadline = budget.deadline_for(name)
            if deadline is None:
                continue  # budget exhausted before this variant could start
            cfg = TrainingConfig(model=name, ..., cv_protocol="loso")
            run_training(df_m6, cfg, deadline=deadline)
            budget.record_actual(name, elapsed_seconds=...)
    """
    models: List[str]
    hard_cap_hours: float = 6.0
    n_folds: int = 28                          # full LOSO fold count
    epochs_per_fold_estimate: Optional[int] = None   # None -> derive from patience
    # Distinguishes this manager's log from any other manager alive in the same
    # notebook. notebooks/03_training.ipynb builds two — a screening manager
    # (Stage 5) and the primary-sweep manager (Stage 8d) — and with a single
    # shared default path the second silently overwrote the first's benchmarks,
    # destroying the screening measurements. Any distinct string works; the
    # name only has to differ between concurrent managers.
    name: str = "default"
    log_path: Optional[str] = None             # defaults to outputs/metrics/budget_manager_<name>_log.json

    benchmarks: Dict[str, float] = field(default_factory=dict)     # variant -> seconds/(fold*epoch)
    estimates: Dict[str, float] = field(default_factory=dict)      # variant -> projected total seconds
    allocations: Dict[str, float] = field(default_factory=dict)    # variant -> allotted seconds
    actuals: Dict[str, float] = field(default_factory=dict)        # variant -> measured seconds spent
    _session_start: float = field(default_factory=time.monotonic, repr=False)

    def __post_init__(self):
        if self.log_path is None:
            self.log_path = config.METRICS_DIR / f"budget_manager_{self.name}_log.json"
        if self.epochs_per_fold_estimate is None:
            # Early stopping usually fires a few epochs past its best, not at
            # DEFAULT_EPOCHS — patience+2 is a documented, adjustable guess, not a
            # measurement; benchmark() only measures ONE epoch (unavoidable — an
            # honest per-fold estimate would require running a whole fold to
            # convergence, defeating the point of a cheap benchmark).
            self.epochs_per_fold_estimate = config.DEFAULT_PATIENCE + 2

    # -----------------------------------------------------------------
    def benchmark(self, df: pd.DataFrame, task: str = "detection",
                  cfg_overrides: Optional[Dict] = None) -> Dict[str, float]:
        """One LOSO fold x one epoch per variant, real wall-clock via run_training
        (max_folds=1, epochs=1) — not a synthetic estimate. Skips a variant already
        benchmarked in a prior call (idempotent across notebook re-runs)."""
        cfg_overrides = cfg_overrides or {}
        print_header("Experiment Budget Manager — benchmarking")

        # deep_frozen/fusion_frozen precompute (and disk-cache) every file's
        # frozen wav2vec2 embedding once per dataset (see
        # src.training.runner.run_training and
        # src.training.baseline.extract_frozen_embeddings_masked). That first
        # extraction is a one-time cost the real full run also pays exactly
        # once — but run inside the loop below, it would land entirely inside
        # this benchmark's "1 fold x 1 epoch" timer, which _estimate() then
        # multiplies by n_folds x epochs_per_fold_estimate, wildly
        # overestimating these two variants' budget. Warming the cache here,
        # outside the timed section, keeps the benchmark measuring only what
        # every other variant's benchmark measures: actual training compute.
        if any(name in MODELS_WITH_CACHEABLE_FROZEN_EMBEDDING for name in self.models):
            from src.training.baseline import extract_frozen_embeddings_masked
            from src.training.utils import resolve_device
            print_kv("Frozen embedding cache", "warming (one-time, outside the timed benchmark)")
            extract_frozen_embeddings_masked(df, device=resolve_device())

        for name in self.models:
            if name in self.benchmarks:
                print_kv(name, f"already benchmarked — {self.benchmarks[name]:.1f}s/fold/epoch")
                continue
            cfg = TrainingConfig(task=task, model=name, epochs=1, max_folds=1,
                                 run_name=f"_budget_bench_{task}_{name}", **cfg_overrides)
            print_status(f"Benchmarking {name} -- 1 fold x 1 epoch, real wall-clock "
                        "(not a silent step; a per-epoch summary prints below)...", ok=True)
            start = time.monotonic()
            run_training(df, cfg)
            elapsed = time.monotonic() - start
            self.benchmarks[name] = elapsed
            print_kv(name, f"{elapsed:.1f}s for 1 fold x 1 epoch")
        self._estimate()
        self._save_log()
        return self.benchmarks

    def _estimate(self) -> None:
        for name, per_fold_epoch in self.benchmarks.items():
            self.estimates[name] = per_fold_epoch * self.epochs_per_fold_estimate * self.n_folds

    def allocate(self) -> Dict[str, float]:
        """Split hard_cap_hours across variants proportional to estimated cost.
        Call after benchmark(). Re-callable if hard_cap_hours changes."""
        if not self.estimates:
            raise RuntimeError("Call benchmark() before allocate() — no measurements yet.")
        hard_cap_s = self.hard_cap_hours * 3600
        total_estimate = sum(self.estimates.values())
        if total_estimate <= hard_cap_s:
            # Everything comfortably fits — give each variant its full estimate
            # plus headroom, capped by the remaining session time.
            self.allocations = dict(self.estimates)
        else:
            self.allocations = {
                name: hard_cap_s * (est / total_estimate)
                for name, est in self.estimates.items()
            }
        self._print_plan(hard_cap_s, total_estimate)
        self._save_log()
        return self.allocations

    def _print_plan(self, hard_cap_s: float, total_estimate: float) -> None:
        print_subheader("Budget allocation")
        rows = pd.DataFrame({
            "model": list(self.estimates.keys()),
            "bench_s_per_fold_epoch": [self.benchmarks[m] for m in self.estimates],
            "estimated_total_h": [self.estimates[m] / 3600 for m in self.estimates],
            "allocated_h": [self.allocations[m] / 3600 for m in self.estimates],
        })
        print_table(rows)
        print_kv("Hard cap", f"{hard_cap_s / 3600:.2f}h")
        print_kv("Total estimated (unconstrained)", f"{total_estimate / 3600:.2f}h")
        if total_estimate > hard_cap_s:
            print_note(f"Estimated cost ({total_estimate/3600:.1f}h) exceeds the "
                      f"{self.hard_cap_hours}h hard cap — allocations scaled down "
                      "proportionally. Early stopping usually finishes well under a "
                      "variant's full allocation; this is a ceiling, not a target.")

    # -----------------------------------------------------------------
    def projection_for(self, model_name: str, remaining_folds: Optional[int] = None) -> Dict:
        """
        What finishing `model_name` would actually cost, from its measured
        benchmark — reported BEFORE committing, which the previous
        implementation never did.

        `remaining_folds` defaults to the full fold count; pass the number still
        outstanding (total minus whatever the on-disk resume cache already
        holds) to project only the work left to do.
        """
        remaining_folds = self.n_folds if remaining_folds is None else remaining_folds
        per_fold_epoch = self.benchmarks.get(model_name)
        if per_fold_epoch is None:
            return {"model": model_name, "projected_s": None,
                    "remaining_folds": remaining_folds}
        projected = per_fold_epoch * self.epochs_per_fold_estimate * remaining_folds
        session_remaining = max(
            0.0, self._session_start + self.hard_cap_hours * 3600 - time.monotonic())
        return {
            "model": model_name,
            "remaining_folds": remaining_folds,
            "s_per_fold_epoch": per_fold_epoch,
            "projected_s": projected,
            "projected_h": projected / 3600,
            "session_remaining_h": session_remaining / 3600,
            "fits_in_session": projected <= session_remaining,
            # One fold is the true granularity: the deadline is only checked
            # BETWEEN folds (src.training.runner.run_training), so a fold that
            # starts always runs to completion. An allocation smaller than this
            # cannot be honoured and will overrun the cap.
            "min_useful_s": per_fold_epoch * self.epochs_per_fold_estimate,
        }

    def deadline_for(self, model_name: str, remaining_folds: Optional[int] = None,
                     allow_partial: bool = True) -> Optional[float]:
        """
        A time.monotonic() deadline for `model_name`, sized from its allocation
        and anchored to the remaining session time.

        Returns None — and says why — when the variant cannot usefully run.
        Three distinct refusals, where the previous version had one silent skip:

          1. budget_exhausted        the session cap is already spent.
          2. insufficient_for_one_fold
             There is not even time for a single fold. Because run_training
             only checks the deadline between folds, handing back a deadline
             shorter than one fold's measured cost does not yield a smaller
             result — it yields a full-length fold that overruns the cap. That
             is exactly how the pre-repair primary sweep spent 2.58h against a
             2.5h cap and then dropped its last three variants, including the
             proposed architecture.
          3. cannot_complete         (only when allow_partial=False)
             It cannot finish all remaining folds, so a FINAL run refuses to
             start it rather than manufacture a partial leaderboard.

        Leave allow_partial=True for screening/development, where a partial
        ranking is still useful — it is now labelled PARTIAL in the registry
        rather than passing as a finished result.
        """
        if model_name not in self.allocations:
            raise KeyError(f"No allocation for '{model_name}' — call allocate() first.")

        projection = self.projection_for(model_name, remaining_folds)
        session_deadline = self._session_start + self.hard_cap_hours * 3600
        remaining = session_deadline - time.monotonic()

        if remaining <= 0:
            print_note(f"Skipping '{model_name}' [PARTIAL: budget_exhausted] — the "
                      f"{self.hard_cap_hours}h session budget is spent. Re-run later "
                      "to resume; completed folds load from disk, not retrained.")
            return None

        min_useful = projection.get("min_useful_s")
        if min_useful is not None and remaining < min_useful:
            print_note(
                f"Skipping '{model_name}' [PARTIAL: insufficient_for_one_fold] — "
                f"{remaining / 60:.0f} min left but one fold measures "
                f"~{min_useful / 60:.0f} min. Starting it would overrun the cap, "
                "since the deadline is only enforced between folds.")
            return None

        if not allow_partial and not projection.get("fits_in_session", False):
            print_note(
                f"Skipping '{model_name}' [PARTIAL: cannot_complete] — needs "
                f"~{projection['projected_h']:.1f}h for "
                f"{projection['remaining_folds']} fold(s) but only "
                f"{projection['session_remaining_h']:.1f}h remain. allow_partial=False, "
                "so it is not started rather than producing an incomplete result.")
            return None

        print_kv(f"{model_name} projection",
                f"~{projection['projected_h']:.1f}h for {projection['remaining_folds']} "
                f"fold(s); {projection['session_remaining_h']:.1f}h left in session"
                + ("" if projection["fits_in_session"] else "   [will be PARTIAL]"))

        slice_s = min(self.allocations[model_name], remaining)
        return time.monotonic() + slice_s

    def preflight(self, remaining_folds: Optional[Dict[str, int]] = None) -> pd.DataFrame:
        """
        The "before you press go" table: projected cost per variant against the
        session budget, as one view rather than something discovered variant by
        variant as the budget drains.

        Call this before a FINAL run. If the total exceeds the cap, the run WILL
        be partial — decide that deliberately here instead of learning it from a
        truncated leaderboard afterwards.
        """
        remaining_folds = remaining_folds or {}
        rows = [self.projection_for(m, remaining_folds.get(m)) for m in self.models]
        table = pd.DataFrame([r for r in rows if r.get("projected_s") is not None])

        print_header("Pre-flight — projected cost")
        if table.empty:
            print_note("No benchmarks yet — call benchmark() first.")
            return table

        print_table(table[["model", "remaining_folds", "s_per_fold_epoch",
                           "projected_h", "fits_in_session"]])
        total_h = table["projected_s"].sum() / 3600
        print_kv("Total projected", f"{total_h:.1f}h")
        print_kv("Hard cap", f"{self.hard_cap_hours:.1f}h")
        print_kv("Epochs per fold (estimate)", self.epochs_per_fold_estimate)
        print_kv("Folds per variant", self.n_folds)
        if total_h > self.hard_cap_hours:
            print_note(f"Projected {total_h:.1f}h exceeds the {self.hard_cap_hours}h cap "
                      f"by {total_h - self.hard_cap_hours:.1f}h — this run will be PARTIAL. "
                      "Raise the cap, cut folds/variants, or plan to resume across "
                      "sessions (completed folds load from disk and are never retrained).")
        else:
            print_status(f"Fits: {total_h:.1f}h projected against a "
                        f"{self.hard_cap_hours:.1f}h cap", ok=True)
        return table

    def record_actual(self, model_name: str, elapsed_seconds: float) -> None:
        self.actuals[model_name] = elapsed_seconds
        self._save_log()

    # -----------------------------------------------------------------
    def _save_log(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump({
                "hard_cap_hours": self.hard_cap_hours,
                "n_folds": self.n_folds,
                "epochs_per_fold_estimate": self.epochs_per_fold_estimate,
                "benchmarks_seconds_per_fold_epoch": self.benchmarks,
                "estimates_seconds": self.estimates,
                "allocations_seconds": self.allocations,
                "actuals_seconds": self.actuals,
            }, f, indent=2)

    @classmethod
    def load(cls, models: List[str], hard_cap_hours: float = 6.0,
             name: str = "default",
             log_path: Optional[str] = None) -> "ExperimentBudgetManager":
        """Resume a prior benchmark/allocation from this manager's log instead of
        re-benchmarking — useful across notebook restarts within one session."""
        log_path = log_path or (config.METRICS_DIR / f"budget_manager_{name}_log.json")
        manager = cls(models=models, hard_cap_hours=hard_cap_hours, name=name,
                      log_path=log_path)
        if log_path.exists():
            with open(log_path) as f:
                data = json.load(f)
            manager.benchmarks = data.get("benchmarks_seconds_per_fold_epoch", {})
            manager.estimates = data.get("estimates_seconds", {})
            manager.allocations = data.get("allocations_seconds", {})
            manager.actuals = data.get("actuals_seconds", {})
            print_status(f"Resumed budget manager state from {log_path}", ok=True)
        return manager


def benchmark_batch_sizes(df: pd.DataFrame, task: str, model_name: str,
                          batch_sizes: List[int] = (8, 16, 32),
                          epochs: int = 1) -> pd.DataFrame:
    """
    Measured (not guessed) batch-size selection: one real LOSO fold x `epochs`
    epoch(s) per candidate batch size, via the same run_training path every
    other benchmark in this module uses — no synthetic timing loop. Reports
    epoch wall-clock time, samples/sec, and peak GPU memory per candidate, and
    prints the fastest one that did not OOM.

    config.DEFAULT_BATCH_SIZE=32 was previously a comment-documented guess
    ("tuned for an 8GB RTX 4060 with AMP") rather than a measurement on the
    GPU actually running a given session. Call this once before
    ExperimentBudgetManager.benchmark() and pass the chosen batch_size into
    every TrainingConfig for the primary sweep (see
    notebooks/03_training.ipynb) — larger batches change wall-clock speed and
    GPU memory footprint, not the science, so this is safe to tune per-machine.
    """
    from src.splits import iter_loso_folds

    device = resolve_device()
    _, train_df, _ = next(iter_loso_folds(df))
    n_train = len(train_df)

    print_header("Batch-size benchmark")
    print_kv("Model", model_name)
    print_kv("Candidates", list(batch_sizes))

    rows = []
    for candidate_index, batch_size in enumerate(batch_sizes, start=1):
        print_status(f"[{candidate_index}/{len(batch_sizes)}] Starting batch_size={batch_size} "
                    f"benchmark -- 1 fold x {epochs} epoch(s), real wall-clock (a per-epoch "
                    "summary prints below)...", ok=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        cfg = TrainingConfig(task=task, model=model_name, epochs=epochs, max_folds=1,
                             batch_size=batch_size,
                             run_name=f"_batch_bench_{task}_{model_name}_{batch_size}")
        start = time.monotonic()
        oom = False
        try:
            run_training(df, cfg)
            elapsed = time.monotonic() - start
        except torch.cuda.OutOfMemoryError:
            elapsed = float("nan")
            oom = True
            if device.type == "cuda":
                torch.cuda.empty_cache()

        peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6
                       if device.type == "cuda" and not oom else float("nan"))
        samples_per_sec = n_train * epochs / elapsed if not oom and elapsed > 0 else float("nan")
        row = {"batch_size": batch_size, "epoch_time_s": elapsed / epochs if not oom else float("nan"),
              "samples_per_sec": samples_per_sec, "peak_gpu_mem_mb": peak_mem_mb, "oom": oom}
        rows.append(row)
        print_kv(f"batch_size={batch_size}",
                 f"OOM" if oom else f"{row['epoch_time_s']:.2f}s/epoch, "
                 f"{samples_per_sec:.1f} samples/sec, {peak_mem_mb:.0f} MB peak")

    result = pd.DataFrame(rows)
    stable = result[~result["oom"]]
    if len(stable):
        best = stable.loc[stable["epoch_time_s"].idxmin()]
        print_status(f"Fastest stable batch size: {int(best['batch_size'])} "
                    f"({best['epoch_time_s']:.2f}s/epoch, {best['peak_gpu_mem_mb']:.0f} MB peak)",
                    ok=True)
    else:
        print_note("Every candidate batch size OOM'd on this GPU — try smaller values.")

    result_path = config.METRICS_DIR / f"batch_size_benchmark_{task}_{model_name}.csv"
    config.ensure_directories()
    result.to_csv(result_path, index=False)
    print_kv("Batch-size benchmark saved", result_path)
    return result
