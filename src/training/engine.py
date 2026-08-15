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
              description: str = "", amp_dtype: torch.dtype = torch.float16,
              amp_enabled: Optional[bool] = None,
              grad_accum_steps: int = 1) -> EpochResult:
    """
    Run one full pass over `loader`.

    train=True updates weights (requires `optimizer`); train=False runs a
    no-grad forward pass (validation or test). Predictions/metrics are
    always collected — the extra bookkeeping is negligible next to a
    wav2vec 2.0 forward pass. Set collect_embeddings=True only for the
    final per-fold test pass that populates outputs/embeddings/.

    grad_accum_steps=1 (default) steps the optimizer every batch — identical
    to the old unconditional per-batch step. >1 accumulates that many
    batches' gradients (loss divided accordingly) before one optimizer step,
    simulating a larger effective batch size at the true batch size's memory
    footprint — a fallback for when raising cfg.batch_size directly risks
    OOM. Inert when train=False (validation/test never touch this branch).
    """
    model.train(mode=train)
    label_key = "group_label" if task == "detection" else "severity_label"
    device_type = device.type
    # Not derived from scaler.is_enabled(): the scaler is only ever enabled
    # for the fp16 path (bf16 needs no loss scaling — see runner.py), so
    # using it here would silently disable autocast itself whenever bf16 is
    # selected. Falls back to the scaler's flag only for callers that don't
    # pass amp_enabled explicitly, preserving the old behaviour for them.
    if amp_enabled is None:
        amp_enabled = scaler.is_enabled()

    running_loss, num_samples = 0.0, 0
    all_true, all_pred, all_prob, all_embeddings = [], [], [], []
    all_speakers, all_filenames = [], []

    grad_context = torch.enable_grad() if train else torch.no_grad()
    batches = (progress(loader, description, total=len(loader), leave=False, unit="batch")
               if description else loader)
    num_batches = len(loader)

    with grad_context:
        if train:
            optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(batches):
            waveform = batch["waveform"].squeeze(1).to(device, non_blocking=True)
            mfcc = batch["mfcc"].to(device, non_blocking=True)
            labels = batch[label_key].to(device, non_blocking=True)

            # Sample-level boolean mask (True = real audio, False = the
            # zero-padding _pad_or_truncate appended past waveform_length) —
            # every model threads this to DeepPathway so wav2vec2's
            # self-attention and pooling never draw on padded silence
            # (see src/preprocessing.py's load_and_preprocess docstring).
            waveform_length = batch["waveform_length"].to(device, non_blocking=True)
            attention_mask = (torch.arange(waveform.shape[1], device=device)[None, :]
                              < waveform_length[:, None])

            # Present only when the Dataset was built with a Praat table, i.e.
            # for Phase 6's Model F. Every other model ignores it (see
            # src/training/models.py), so the engine needs no per-model branching.
            praat = batch["praat"].to(device, non_blocking=True) if "praat" in batch else None
            # Present only for deep_frozen/fusion_frozen (see
            # MODELS_WITH_CACHEABLE_FROZEN_EMBEDDING) — the precomputed frozen
            # wav2vec2 vector, used instead of a live backbone forward pass.
            deep_embedding = (batch["deep_embedding"].to(device, non_blocking=True)
                              if "deep_embedding" in batch else None)

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                if collect_embeddings:
                    features = model.forward_features(waveform, mfcc, praat,
                                                       attention_mask=attention_mask,
                                                       deep_embedding=deep_embedding)
                    logits = model.classifier(features)
                else:
                    logits = model(waveform, mfcc, praat, attention_mask=attention_mask,
                                   deep_embedding=deep_embedding)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss / grad_accum_steps).backward()
                is_last_batch = (step + 1) == num_batches
                if (step + 1) % grad_accum_steps == 0 or is_last_batch:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

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
