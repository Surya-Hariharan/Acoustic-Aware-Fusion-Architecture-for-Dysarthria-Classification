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
    """One-time, per-process CUDA settings for the local RTX 4060 profile
    (config "Local hardware profile"):

      * cudnn.benchmark — every input is the same fixed 4 s window, so
        autotuning wav2vec2's conv feature encoder once pays off every step.
        It changes kernel selection, not the model or data.
      * a VRAM cap (config.CUDA_MEMORY_FRACTION) — on Windows/WDDM, running
        past physical VRAM does not raise OOM but spills into system RAM and
        collapses throughput ~10x; the cap turns that into a clean OOM and
        leaves headroom for the display.
    """
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED or device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(config.CUDNN_BENCHMARK)
    if config.CUDA_MEMORY_FRACTION:
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(float(config.CUDA_MEMORY_FRACTION), index)
    _RUNTIME_CONFIGURED = True


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
