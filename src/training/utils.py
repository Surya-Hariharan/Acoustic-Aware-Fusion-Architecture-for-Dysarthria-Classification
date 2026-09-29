"""Small shared helpers: seeding and device resolution."""

import random

import numpy as np
import torch

from src import config
from src.console import print_note


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp_dtype(device: torch.device) -> torch.dtype:
    """AMP dtype from config.AMP_DTYPE. "bfloat16" is honoured only on GPUs
    with NATIVE bf16 (compute capability >= 8.0) — torch.cuda.is_bf16_supported()
    also says True where bf16 is emulated, which would trade the tensor-core
    fp16 path for a slow software one; everything else runs float16 with a
    live GradScaler (see run_fold)."""
    if (config.AMP_DTYPE == "bfloat16" and device.type == "cuda"
            and torch.cuda.get_device_capability(device)[0] >= 8):
        return torch.bfloat16
    return torch.float16


_RUNTIME_CONFIGURED = False


def configure_local_runtime(device: torch.device) -> None:
    """One-time, per-process CUDA settings for the local hardware profile
    (config "Local hardware profile"):

      * cudnn.benchmark — every input is the same fixed 4 s window, so
        autotuning wav2vec2's conv feature encoder once pays off every step.
        It changes kernel selection, not the model or data.
      * a VRAM cap (config.CUDA_MEMORY_FRACTION) — on Windows/WDDM, running
        past physical VRAM does not raise OOM but spills into system RAM and
        collapses throughput ~10x; the cap turns that into a clean OOM and
        leaves headroom for the display.

    The cap is the smaller of that fraction and what is actually FREE on the
    card now (less 0.3 GiB): VRAM another application holds (a browser, the
    desktop compositor) is not available to training, and a cap above the
    free amount would let the allocator spill instead of failing cleanly.
    """
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED or device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(config.CUDNN_BENCHMARK)
    if config.CUDA_MEMORY_FRACTION:
        index = device.index if device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        fraction = min(float(config.CUDA_MEMORY_FRACTION), (free - 0.3 * 2 ** 30) / total)
        torch.cuda.set_per_process_memory_fraction(max(fraction, 0.1), index)
        if fraction < 0.75:
            print_note(f"Only {free / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GiB of GPU memory is "
                       "free — another application is using the GPU. Training needs ~6 GiB at "
                       "batch 32; close it (or restart the kernel) if a fold runs out of memory.")
    _RUNTIME_CONFIGURED = True


def memory_status() -> dict:
    """System memory right now, in GiB: available/total physical RAM and
    available/limit COMMIT (RAM + page file).

    Commit is what a Windows process actually runs out of — "The paging file
    is too small" (error 1455) — and it is consumed far faster than RAM by
    spawned workers (each maps torch's CUDA DLLs, ~2 GB committed). On
    Windows it is read with GlobalMemoryStatusEx; elsewhere RAM + swap stands
    in for it."""
    gib = 2 ** 30
    try:
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"ram_available": status.ullAvailPhys / gib, "ram_total": status.ullTotalPhys / gib,
                    "commit_available": status.ullAvailPageFile / gib,
                    "commit_limit": status.ullTotalPageFile / gib}
    except (AttributeError, OSError):
        pass
    import psutil
    ram, swap = psutil.virtual_memory(), psutil.swap_memory()
    return {"ram_available": ram.available / gib, "ram_total": ram.total / gib,
            "commit_available": (ram.available + swap.free) / gib,
            "commit_limit": (ram.total + swap.total) / gib}


def affordable_workers(requested: int, ram_per_worker_gb: float, commit_per_worker_gb: float,
                       reserve_ram_gb: float = None, reserve_commit_gb: float = None) -> int:
    """The largest worker count <= `requested` whose memory fits in what is
    free right now after the reserves — 0 if not even one does. Workers are a
    throughput aid, never a requirement: every caller also works in-process."""
    reserve_ram_gb = config.RAM_RESERVE_GB if reserve_ram_gb is None else reserve_ram_gb
    reserve_commit_gb = config.COMMIT_RESERVE_GB if reserve_commit_gb is None else reserve_commit_gb
    status = memory_status()
    by_ram = int((status["ram_available"] - reserve_ram_gb) // ram_per_worker_gb)
    by_commit = int((status["commit_available"] - reserve_commit_gb) // commit_per_worker_gb)
    return max(0, min(int(requested), by_ram, by_commit))


def resolve_device(requested: str = None) -> torch.device:
    """
    cuda if available, else cpu — but say so explicitly rather than falling
    back silently. A 28/81-fold LOSO sweep with wav2vec2 fine-tuning on CPU
    is not "slow", it is unusable, and the most common cause is not this
    function's logic but the *kernel a notebook happens to be running under*
    not being the same Python environment `pip install`/`conda` targeted —
    e.g. a fresh `pip install torch` (no CUDA index URL, see requirements.txt's
    top comment) silently installs a CPU-only wheel with the same version
    string, so `import torch; torch.__version__` alone won't reveal the
    problem — only torch.cuda.is_available() does.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print_note("CUDA is not visible to this Python environment/kernel — training "
              "will run on CPU. If a GPU is installed, this almost always means "
              "the active notebook kernel is not the environment torch+CUDA was "
              "installed into. Run `import torch; torch.cuda.is_available()` in "
              "a cell to confirm, and switch the notebook's kernel if it prints "
              "False despite the GPU being present.")
    return torch.device("cpu")
