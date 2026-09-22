"""
Stall-resilient process-pool mapping, shared by the one-time precompute
passes (src.vad_cache, src.preprocessing).

WHY THIS EXISTS
---------------
Both precompute passes used to drive their ProcessPoolExecutor with
`executor.map(worker_fn, items, chunksize=N)`. map() yields results in
SUBMISSION order, not completion order. If a single item makes one worker
hang forever (a corrupt WAV, a decode stall — anything that blocks without
raising), the other n_workers-1 processes keep finishing their own chunks in
the background, but map()'s iterator cannot yield any of those results until
the stuck one resolves, because it must preserve order. The progress bar
(src.console.progress), which only prints when a result comes back, then goes
silent forever with no traceback and no further log output — exactly the
failure observed on a real Kaggle run of the VAD-span pass: steady ~30s
progress ticks, then nothing, indefinitely, with the process still alive.

resilient_process_map replaces that with completion-order collection
(concurrent.futures.wait(..., return_when=FIRST_COMPLETED)), so one stuck
task cannot block the rest, plus a stall watchdog: if NO task completes
within config.PRECOMPUTE_STALL_TIMEOUT_S while work remains, the outstanding
worker process(es) are force-killed, the still-pending items are logged and
returned as `skipped`, and the pass continues with whatever did complete.
This is safe for both call sites because both build an optimization-only
disk cache (see src.vad_cache's and src.preprocessing's module docstrings):
a missing row is just a cache miss, never a correctness problem.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, Iterable, List, Tuple, TypeVar

from src import config
from src.console import ProgressReporter, print_note

T = TypeVar("T")
R = TypeVar("R")


def resilient_process_map(
    worker_fn: Callable[[T], R],
    items: List[T],
    *,
    n_workers: int,
    max_tasks_per_child: int,
    description: str,
    unit: str = "file",
    stall_timeout_s: float = config.PRECOMPUTE_STALL_TIMEOUT_S,
) -> Tuple[List[R], List[T]]:
    """Run worker_fn over items in a process pool, resilient to one stuck task.

    Returns (results, skipped) — results in COMPLETION order (not the order of
    `items`; callers that need to re-associate a result with its input must do
    so from the result itself, as every worker_fn in this codebase already
    does by returning the item's own key). `skipped` lists the items that
    never completed — because their task stalled past stall_timeout_s, or
    because worker_fn raised for them — logged with a diagnostic so the
    specific offending file(s) are identifiable rather than a silent freeze.
    """
    if not items:
        return [], []

    results: List[R] = []
    skipped: List[T] = []

    executor = ProcessPoolExecutor(
        max_workers=n_workers, max_tasks_per_child=max_tasks_per_child)
    bar = ProgressReporter(None, description=description, total=len(items), unit=unit)
    try:
        future_to_item = {executor.submit(worker_fn, item): item for item in items}
        pending = set(future_to_item)

        while pending:
            done, pending = wait(pending, timeout=stall_timeout_s,
                                 return_when=FIRST_COMPLETED)
            if not done:
                stuck = [future_to_item[f] for f in pending]
                print_note(
                    f"No '{description}' task has completed in "
                    f"{stall_timeout_s:.0f}s while {len(stuck)} file(s) are "
                    f"still outstanding — treating them as stuck rather than "
                    f"slow. Killing the stalled worker process(es) and "
                    f"skipping these files (they are simply absent from the "
                    f"cache, which falls back to live VAD at read time): "
                    f"{stuck[:5]}{' ...' if len(stuck) > 5 else ''}"
                )
                skipped.extend(stuck)
                for proc in list(getattr(executor, "_processes", {}).values()):
                    proc.kill()
                break

            for future in done:
                item = future_to_item[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    print_note(f"Task failed for {item}: {exc} — skipping.")
                    skipped.append(item)
                bar.update(1)
    finally:
        bar.close()
        executor.shutdown(wait=False, cancel_futures=True)

    return results, skipped
