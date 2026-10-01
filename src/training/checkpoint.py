"""
Fold checkpoints: atomic, compact, resumable.

Only trainable parameters and buffers are stored — the frozen wav2vec2
backbone is reloaded from its pretrained checkpoint when the model is built —
so a checkpoint is ~9 MB instead of ~386 MB. Writes go to a temp file and are
renamed into place (retried while OneDrive or Defender holds the old copy), so
a session killed mid-save never leaves a truncated checkpoint.
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch


COMPACT_STATE_SCOPE = "trainable_params_and_buffers"


def replace_with_retry(temp_path: Path, path: Path, attempts: int = 12,
                       delay_s: float = 0.5) -> None:
    """os.replace, retried on PermissionError. On Windows a file another
    process has open cannot be replaced — and this project lives under
    OneDrive, which opens every freshly written checkpoint to upload it (as
    does Defender, to scan it). Without the retry, the per-epoch latest.pt
    save failed intermittently and took the whole fold down with it."""
    for attempt in range(attempts):
        try:
            os.replace(temp_path, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s * (attempt + 1))


def _frozen_parameter_names(model: torch.nn.Module) -> set:
    return {name for name, param in model.named_parameters() if not param.requires_grad}


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: torch.amp.GradScaler, epoch: int,
                    monitored_value: float, extra: Optional[dict] = None) -> None:
    """`extra` is merged into the saved dict as-is (e.g. early-stopping state,
    fold id) -- used by the per-epoch "latest" resume checkpoint in
    src.training.runner.run_fold; the best-checkpoint call site simply omits it.

    Written to a temporary file and renamed into place, so a session killed
    mid-save leaves the previous checkpoint intact rather than a truncated one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen = _frozen_parameter_names(model)
    checkpoint = {
        "epoch": epoch,
        "model_state": {key: value for key, value in model.state_dict().items()
                        if key not in frozen},
        "state_scope": COMPACT_STATE_SCOPE,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "monitored_value": monitored_value,
    }
    if extra:
        checkpoint.update(extra)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temp_path)
    replace_with_retry(temp_path, path)


def load_checkpoint(path: Path, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None, scaler: Optional[torch.amp.GradScaler] = None,
                    map_location: str = "cpu") -> dict:
    """Restore a checkpoint into an already-built `model`. A compact
    checkpoint (see COMPACT_STATE_SCOPE) must supply every key except frozen
    parameters — anything else missing, or any unexpected key, raises rather
    than silently leaving part of the model at its initialization."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("state_scope") == COMPACT_STATE_SCOPE:
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        frozen = _frozen_parameter_names(model)
        missing_trainable = [key for key in missing if key not in frozen]
        if missing_trainable or unexpected:
            raise RuntimeError(
                f"Checkpoint {path} does not match this model: missing non-frozen "
                f"keys {missing_trainable[:5]}, unexpected keys {list(unexpected)[:5]}.")
    else:
        model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint
