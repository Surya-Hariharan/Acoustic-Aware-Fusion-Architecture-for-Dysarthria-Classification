"""Small shared helpers: seeding and device resolution."""

import random

import numpy as np
import torch

from src.console import print_note


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
