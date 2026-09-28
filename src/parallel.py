"""
Stall-resilient process-pool mapping, shared by the one-time precompute
passes (src.vad_cache, src.preprocessing).

WHY THIS EXISTS
---------------
Both precompute passes used to drive their ProcessPoolExecutor with
`executor.map(worker_fn, items, chunksize=N)`. map() yields results in
SUBMISSION order, not completion order. If a single item makes one worker
hang forever, the other n_workers-1 processes keep finishing their own chunks
in the background, but map()'s iterator cannot yield any of those results
until the stuck one resolves, because it must preserve order. The progress
bar, which only prints when a result comes back, then goes silent forever
with no traceback and no further log output.

resilient_process_map replaces that with completion-order collection
(concurrent.futures.wait(..., return_when=FIRST_COMPLETED)), so one stuck
task cannot block the rest, plus a stall watchdog: if NO task completes
within config.PRECOMPUTE_STALL_TIMEOUT_S while work remains, the outstanding
worker process(es) are force-killed, the still-pending items are logged and
returned as `skipped`, and the pass continues with whatever did complete.
This is safe for both call sites because both build an optimization-only
disk cache: a missing row is just a cache miss, never a correctness problem.

THE ACTUAL STALL MECHANISM: ProcessPoolExecutor's max_tasks_per_child RESPAWN,
NOT A HUNG FILE, AND NOT (SOLELY) THREAD OVERSUBSCRIPTION
--------------------------------------------------------------------------------
Two independent theories were tried and ruled out by direct evidence before
this one:

1. A single poisoned file hanging one worker. Ruled out: the stall watchdog
   (added specifically to catch this) instead showed ALL n_workers processes
   going silent simultaneously, which a lone stuck file cannot cause.

2. CPU thread oversubscription (torch defaulting every process's intra-op
   pool to the full core count, n_workers-fold oversubscribed). This is a
   real inefficiency and _worker_thread_init below still fixes it, but THREE
   independent Kaggle runs — one before that fix, two after — all stalled at
   EXACTLY 800 completed tasks: 4 workers x PRECOMPUTE_MAX_TASKS_PER_CHILD
   (200). That precise, repeated boundary is the signature of
   ProcessPoolExecutor's own internal worker-recycling (which replaces a
   worker in place once it hits max_tasks_per_child), not of thread
   contention, which would not care about a task-count boundary at all.

In this environment — a Kaggle kernel whose main process has already
initialized a CUDA context (cell 1.2's torch.cuda calls, well before any pool
is created) — the DEFAULT start method spawns initial workers fine, but
ProcessPoolExecutor's in-place respawn of a worker mid-run appears to hang
deterministically. The fix here does not depend on knowing the exact reason:
it simply avoids that code path. Instead of asking ProcessPoolExecutor to
recycle a worker in place, resilient_process_map processes `items` in
batches of max_tasks_per_child * n_workers, fully shutting down (wait=True)
and recreating the pool between batches. This bounds per-worker task count
identically to max_tasks_per_child (so the RSS-growth concern
PRECOMPUTE_MAX_TASKS_PER_CHILD's own docstring describes is still addressed),
but via a full, ordinary pool teardown/recreate — a far more common and
better-tested code path than an in-place mid-run respawn.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, List, Tuple, TypeVar

from src import config
from src.console import ProgressReporter, print_note

T = TypeVar("T")
R = TypeVar("R")


def _worker_thread_init() -> None:
    """ProcessPoolExecutor initializer: pin this worker to a single intra-op
    thread. Without this, n_workers processes each default to a full-core-count
    thread pool, oversubscribing the machine n_workers-fold — a real
    throughput cost even though it was not, on its own, the cause of the
    exactly-800-tasks stall this module's docstring documents."""
    import torch
    torch.set_num_threads(1)


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
    """Run worker_fn over items in a process pool, resilient to one stuck task
    AND to ProcessPoolExecutor's own max_tasks_per_child respawn hanging (see
    module docstring).

    Returns (results, skipped) — results in COMPLETION order (not the order of
    `items`; callers that need to re-associate a result with its input must do
    so from the result itself, as every worker_fn in this codebase already
    does by returning the item's own key). `skipped` lists the items that
    never completed — their task stalled past stall_timeout_s, or worker_fn
    raised for them — logged with a diagnostic so the specific offending
    file(s) are identifiable rather than a silent freeze.

    Processes `items` in batches of max_tasks_per_child * n_workers, with a
    FRESH ProcessPoolExecutor per batch (fully joined via shutdown(wait=True)
    before the next batch's pool is created) — see the module docstring for
    why this replaces passing max_tasks_per_child directly to
    ProcessPoolExecutor.
    """
    if not items:
        return [], []

    results: List[R] = []
    skipped: List[T] = []
    batch_size = max(1, max_tasks_per_child * n_workers)

    bar = ProgressReporter(None, description=description, total=len(items), unit=unit)
    try:
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start:batch_start + batch_size]
            batch_results, batch_skipped = _run_one_batch(
                worker_fn, batch, n_workers=n_workers,
                description=description, stall_timeout_s=stall_timeout_s, bar=bar)
            results.extend(batch_results)
            skipped.extend(batch_skipped)
    finally:
        bar.close()

    return results, skipped


def _run_one_batch(
    worker_fn: Callable[[T], R],
    batch: List[T],
    *,
    n_workers: int,
    description: str,
    stall_timeout_s: float,
    bar: ProgressReporter,
) -> Tuple[List[R], List[T]]:
    """One batch's worth of work through a FRESH, short-lived pool — no
    max_tasks_per_child passed to it, since bounding per-worker task count is
    now this function's caller's job (via batch sizing), not
    ProcessPoolExecutor's internal respawn (see module docstring)."""
    results: List[R] = []
    skipped: List[T] = []

    executor = ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_thread_init)
    try:
        future_to_item = {executor.submit(worker_fn, item): item for item in batch}
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
        # wait=True (unlike the old single-pool version's wait=False): this is
        # exactly the full join that a mid-run max_tasks_per_child respawn was
        # NOT doing cleanly. Any process already .kill()-ed above joins
        # immediately; anything still legitimately finishing gets to.
        executor.shutdown(wait=True, cancel_futures=True)

    return results, skipped
