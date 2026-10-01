"""
Runtime helpers: seeding, device and AMP selection, memory accounting, and the
laptop safeguards — GPU thermal guard, sleep prevention, AC-power check.
"""

import contextlib
import random
import subprocess
import sys
import time
from typing import Optional

import numpy as np
import torch

from src import config
from src.console import print_note, print_status


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """cuda if available, else cpu — with a note, since CPU training of this
    model is unusable and the usual cause is the wrong notebook kernel."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    print_note("CUDA is not visible to this kernel — training would run on CPU. Select the "
               "Python environment with the CUDA build of torch.")
    return torch.device("cpu")


def resolve_amp_dtype(device: torch.device) -> torch.dtype:
    """config.AMP_DTYPE; bfloat16 only where the GPU runs it natively (compute
    capability >= 8.0), otherwise float16 with a GradScaler."""
    if (config.AMP_DTYPE == "bfloat16" and device.type == "cuda"
            and torch.cuda.get_device_capability(device)[0] >= 8):
        return torch.bfloat16
    return torch.float16


_RUNTIME_CONFIGURED = False


def configure_local_runtime(device: torch.device) -> None:
    """Once per process: cuDNN autotuning (fixed input shape) and a VRAM cap —
    the smaller of config.CUDA_MEMORY_FRACTION and what is free now. On
    Windows/WDDM, running past physical VRAM spills into system RAM and
    throughput collapses ~10x; the cap turns that into a clean, retryable OOM."""
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
                       "batch 32; close it if a fold runs out of memory.")
    _RUNTIME_CONFIGURED = True


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def memory_status() -> dict:
    """Free/total physical RAM and free/limit COMMIT (RAM + page file), GiB.
    Commit is what Windows actually runs out of ("the paging file is too
    small", error 1455), and spawned workers consume it far faster than RAM."""
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
                       reserve_ram_gb: Optional[float] = None,
                       reserve_commit_gb: Optional[float] = None) -> int:
    """Largest worker count <= requested that fits in what is free now (0 if
    none does — every caller also works in-process)."""
    reserve_ram_gb = config.RAM_RESERVE_GB if reserve_ram_gb is None else reserve_ram_gb
    reserve_commit_gb = config.COMMIT_RESERVE_GB if reserve_commit_gb is None else reserve_commit_gb
    status = memory_status()
    by_ram = int((status["ram_available"] - reserve_ram_gb) // ram_per_worker_gb)
    by_commit = int((status["commit_available"] - reserve_commit_gb) // commit_per_worker_gb)
    return max(0, min(int(requested), by_ram, by_commit))


# ---------------------------------------------------------------------------
# Laptop safeguards
# ---------------------------------------------------------------------------
def gpu_temperature(index: int = 0) -> Optional[int]:
    """Core temperature in C via NVML when pynvml is installed, else
    nvidia-smi; None if neither is available."""
    try:
        return int(torch.cuda.temperature(index))
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return int(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


class ThermalGuard:
    """Pause training between batches while the GPU is too hot.

    check() is cheap to call every batch: it reads the sensor at most every
    config.GPU_TEMP_CHECK_INTERVAL_S. At or above GPU_TEMP_PAUSE_C it sleeps
    until the GPU cools to GPU_TEMP_RESUME_C (or GPU_COOLDOWN_MAX_WAIT_S
    passes). Sleeping between batches changes nothing about what is computed —
    only when."""

    def __init__(self):
        self._last_check = 0.0
        self._disabled = False
        self.pauses = 0
        self.paused_seconds = 0.0

    def check(self, force: bool = False) -> None:
        if self._disabled or not torch.cuda.is_available() or config.GPU_TEMP_PAUSE_C <= 0:
            return
        now = time.monotonic()
        if not force and now - self._last_check < config.GPU_TEMP_CHECK_INTERVAL_S:
            return
        self._last_check = now
        temperature = gpu_temperature()
        if temperature is None:
            self._disabled = True
            print_note("GPU temperature is unreadable (no nvidia-smi / pynvml) — the thermal "
                       "guard is off; watch temperatures with another tool.")
            return
        if temperature >= config.GPU_TEMP_PAUSE_C:
            self.cool_down(temperature, max_wait_s=config.GPU_COOLDOWN_MAX_WAIT_S,
                           reason=f"reached {temperature} C")

    def cool_down(self, temperature: Optional[int] = None, max_wait_s: float = 0.0,
                  reason: str = "") -> None:
        """Wait (up to max_wait_s) until the GPU is at or below the resume
        temperature. Returns at once if it already is."""
        temperature = gpu_temperature() if temperature is None else temperature
        if temperature is None or temperature <= config.GPU_TEMP_RESUME_C or max_wait_s <= 0:
            return
        start = time.monotonic()
        print_note(f"GPU {reason or f'at {temperature} C'} — pausing until it cools to "
                   f"{config.GPU_TEMP_RESUME_C} C (at most {max_wait_s / 60:.0f} min).")
        while temperature is not None and temperature > config.GPU_TEMP_RESUME_C:
            if time.monotonic() - start >= max_wait_s:
                print_note(f"GPU still at {temperature} C after {max_wait_s / 60:.0f} min — "
                           "resuming. Improve airflow (raise the laptop's rear, clean the vents).")
                break
            time.sleep(5)
            temperature = gpu_temperature()
        waited = time.monotonic() - start
        self.pauses += 1
        self.paused_seconds += waited
        self._last_check = time.monotonic()
        print_status(f"GPU at {temperature} C after a {waited:.0f} s pause — resuming.", ok=True)


THERMAL_GUARD = ThermalGuard()


@contextlib.contextmanager
def prevent_sleep():
    """Keep Windows from sleeping while training runs (the display may still
    turn off). Closing the lid follows its own power setting — set "When I
    close the lid: Do nothing" for plugged-in, or keep the lid open."""
    if sys.platform != "win32":
        yield
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


def on_battery() -> bool:
    """True when the machine has a battery and is not plugged in."""
    try:
        import psutil
        battery = psutil.sensors_battery()
    except Exception:
        return False
    return battery is not None and not battery.power_plugged


def wait_for_ac_power() -> None:
    """Block until the charger is connected (config.REQUIRE_AC_POWER). On
    battery the GPU runs at a fraction of its clocks and a fold would drain
    the battery before it finished."""
    if not config.REQUIRE_AC_POWER or not on_battery():
        return
    print_note("Running on battery — waiting for the charger before training continues "
               "(set config.REQUIRE_AC_POWER = False to override).")
    while on_battery():
        time.sleep(10)
    print_status("Charger connected — continuing.", ok=True)
