"""
Training/evaluation engine: runs one epoch (train, validation, or final
fold test) of a train.py run.

AMP autocast, gradient scaling, gradient clipping, and metric bookkeeping
all live here so train.py only orchestrates folds, optimizers, scheduling,
and I/O — it never touches a raw batch.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.console import progress
from src.training.metrics import compute_metrics


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float]
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_prob: Optional[np.ndarray] = None
    speaker_ids: Optional[List[str]] = None
    filenames: Optional[List[str]] = None     # utterance identity, for Phase 5
    embeddings: Optional[np.ndarray] = None


def run_epoch(model: nn.Module, loader, criterion: nn.Module,
              optimizer: Optional[torch.optim.Optimizer], device: torch.device,
              scaler: torch.amp.GradScaler, grad_clip_norm: float, task: str,
              train: bool, collect_embeddings: bool = False,
              description: str = "") -> EpochResult:
    """
    Run one full pass over `loader`.

    train=True updates weights (requires `optimizer`); train=False runs a
    no-grad forward pass (validation or test). Predictions/metrics are
    always collected — the extra bookkeeping is negligible next to a
    wav2vec 2.0 forward pass. Set collect_embeddings=True only for the
    final per-fold test pass that populates outputs/embeddings/.
    """
    model.train(mode=train)
    label_key = "group_label" if task == "detection" else "severity_label"
    device_type = device.type
    amp_enabled = scaler.is_enabled()

    running_loss, num_samples = 0.0, 0
    all_true, all_pred, all_prob, all_embeddings = [], [], [], []
    all_speakers, all_filenames = [], []

    grad_context = torch.enable_grad() if train else torch.no_grad()
    batches = (progress(loader, description, total=len(loader), leave=False, unit="batch")
               if description else loader)

    with grad_context:
        for batch in batches:
            waveform = batch["waveform"].squeeze(1).to(device, non_blocking=True)
            mfcc = batch["mfcc"].to(device, non_blocking=True)
            labels = batch[label_key].to(device, non_blocking=True)

            # Present only when the Dataset was built with a Praat table, i.e.
            # for Phase 6's Model F. Every other model ignores it (see
            # src/training/models.py), so the engine needs no per-model branching.
            praat = batch["praat"].to(device, non_blocking=True) if "praat" in batch else None

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device_type, enabled=amp_enabled):
                if collect_embeddings:
                    features = model.forward_features(waveform, mfcc, praat)
                    logits = model.classifier(features)
                else:
                    logits = model(waveform, mfcc, praat)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            probs = torch.softmax(logits.detach().float(), dim=1)
            preds = probs.argmax(dim=1)
            all_true.append(labels.detach().cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append((probs[:, 1] if task == "detection" else probs).cpu().numpy())
            all_speakers.extend(batch["speaker_id"])
            all_filenames.extend(batch["filename"])
            if collect_embeddings:
                all_embeddings.append(features.detach().float().cpu().numpy())

    avg_loss = running_loss / max(num_samples, 1)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    metrics = compute_metrics(y_true, y_pred, y_prob, task)
    embeddings = np.concatenate(all_embeddings) if collect_embeddings else None

    return EpochResult(loss=avg_loss, metrics=metrics, y_true=y_true, y_pred=y_pred,
                       y_prob=y_prob, speaker_ids=all_speakers,
                       filenames=all_filenames, embeddings=embeddings)


def build_optimizer(model: nn.Module, lr_head: float, lr_backbone: float,
                    weight_decay: float) -> torch.optim.AdamW:
    """AdamW with a lower LR for the wav2vec backbone than the rest of the model."""
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if "wav2vec" in name else head_params).append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head})
    if not groups:
        raise ValueError("Model has no trainable parameters.")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
