"""Checkpoint save/load for one fold's training run."""

from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: torch.amp.GradScaler, epoch: int,
                    monitored_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "monitored_value": monitored_value,
    }, path)


def load_checkpoint(path: Path, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None, scaler: Optional[torch.amp.GradScaler] = None,
                    map_location: str = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint
