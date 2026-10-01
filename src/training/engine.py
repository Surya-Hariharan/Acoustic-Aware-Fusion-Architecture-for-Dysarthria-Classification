"""
One pass over a DataLoader — a training epoch, a validation epoch, or a fold's
held-out test — for a GatedFusionModel: AMP autocast, gradient scaling and
accumulation, clipping, metrics, and (on the test pass) embedding collection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.console import progress
from src.training.metrics import compute_metrics
from src.training.utils import THERMAL_GUARD


@dataclass
class EpochResult:
    loss: float
    metrics: Dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray                  # CORAL median decode
    y_pred_argmax: np.ndarray           # argmax of class probabilities, for audits
    y_prob: np.ndarray                  # (N, 4) class probabilities
    speaker_ids: List[str]
    filenames: List[str]
    extras: Dict[str, float] = field(default_factory=dict)   # sample-weighted means
    # Test pass only: {"fused": (N,256), "gates": (N,3), "learned"/"segmental"/"supra": ...}
    embeddings: Optional[Dict[str, np.ndarray]] = None


def run_epoch(model: nn.Module, loader, device: torch.device, *,
              optimizer: Optional[torch.optim.Optimizer] = None,
              scaler: Optional[torch.amp.GradScaler] = None,
              class_weights: Optional[torch.Tensor] = None,
              grad_clip_norm: float = 1.0, grad_accum_steps: int = 1,
              amp_dtype: torch.dtype = torch.float16, amp_enabled: bool = True,
              collect_embeddings: bool = False, description: str = "") -> EpochResult:
    """Train when `optimizer` is given, otherwise evaluate under no_grad.

    grad_accum_steps > 1 accumulates that many batches before each optimizer
    step (same effective batch at a fraction of the activation memory — what a
    CUDA-OOM retry uses)."""
    train = optimizer is not None
    model.train(mode=train)
    if scaler is None:
        scaler = torch.amp.GradScaler(device=device.type, enabled=False)

    loss_sum, n_seen = torch.zeros((), device=device), 0
    extras_sum: Dict[str, torch.Tensor] = {}
    true, pred, pred_argmax, probs, speakers, filenames = [], [], [], [], [], []
    embeddings: Dict[str, List[np.ndarray]] = {}

    batches = (progress(loader, description, total=len(loader), leave=False, unit="batch")
               if description else loader)
    num_batches = len(loader)
    if train:
        optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad() if train else torch.no_grad():
        for step, batch in enumerate(batches):
            THERMAL_GUARD.check()
            waveform = batch["waveform"].squeeze(1).to(device, non_blocking=True)
            lengths = batch["waveform_length"].to(device, non_blocking=True)
            labels = batch["severity_label"].to(device, non_blocking=True)
            speaker_index = batch.get("speaker_index")
            inputs = dict(
                waveform=waveform,
                segmental=batch["segmental"].to(device, non_blocking=True),
                supra=batch["supra"].to(device, non_blocking=True),
                supra_valid_frames=batch["supra_valid_frames"].to(device, non_blocking=True),
                attention_mask=torch.arange(waveform.shape[1], device=device)[None, :] < lengths[:, None],
                labels=labels, class_weights=class_weights,
                speaker_index=(speaker_index.to(device, non_blocking=True)
                               if speaker_index is not None else None))

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model.training_step(**inputs, return_embeddings=collect_embeddings)
            logits, loss, extras = outputs[:3]

            if train:
                scaler.scale(loss / grad_accum_steps).backward()
                if (step + 1) % grad_accum_steps == 0 or step + 1 == num_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            batch_size = labels.size(0)
            loss_sum += loss.detach().float() * batch_size
            n_seen += batch_size
            for key, value in extras.items():
                extras_sum[key] = extras_sum.get(key, 0.0) + value * batch_size

            class_probs = torch.softmax(logits.detach().float(), dim=1)
            true.append(labels.cpu().numpy())
            pred.append(model.predict_labels(logits).cpu().numpy())
            pred_argmax.append(class_probs.argmax(dim=1).cpu().numpy())
            probs.append(class_probs.cpu().numpy())
            speakers.extend(batch["speaker_id"])
            filenames.extend(batch["filename"])
            if collect_embeddings:
                for name, tensor in outputs[3].items():
                    embeddings.setdefault(name, []).append(tensor.detach().float().cpu().numpy())
            if description:
                batches.set_postfix_str(f"loss={loss_sum.item() / n_seen:.4f}", refresh=False)

    y_true, y_pred, y_prob = np.concatenate(true), np.concatenate(pred), np.concatenate(probs)
    # NaN extras (e.g. no speaker labels on validation) stay NaN — reported as n/a.
    extras_mean = {key: float(value) / max(n_seen, 1) for key, value in extras_sum.items()}
    return EpochResult(
        loss=loss_sum.item() / max(n_seen, 1),
        metrics=compute_metrics(y_true, y_pred, y_prob),
        y_true=y_true, y_pred=y_pred, y_pred_argmax=np.concatenate(pred_argmax), y_prob=y_prob,
        speaker_ids=speakers, filenames=filenames, extras=extras_mean,
        embeddings=({name: np.concatenate(arrays) for name, arrays in embeddings.items()}
                    if collect_embeddings else None))


def build_optimizer(model: nn.Module, lr_head: float, lr_lora: float,
                    weight_decay: float) -> torch.optim.AdamW:
    """AdamW with two groups: LoRA adapters (the only trainable parameters
    inside wav2vec2) at lr_lora, everything else trainable at lr_head."""
    lora, head = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (lora if "lora_" in name else head).append(param)
    groups = [{"params": params, "lr": lr, "name": group_name}
              for group_name, params, lr in (("lora", lora, lr_lora), ("head", head, lr_head))
              if params]
    if not groups:
        raise ValueError("Model has no trainable parameters.")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
