"""
Checkpoint save/load — one function pair per model family, since a PyTorch
module and an sklearn estimator serialize completely differently and forcing
both through one format would make one of them non-idiomatic:

  PyTorch  (Deep/Acoustic/Fusion pathways)  -> save_checkpoint/load_checkpoint,
             torch.save() of a state_dict + optimizer/scheduler/scaler state
             (.pt) - the standard PyTorch format; NOT raw pickle of the whole
             module, which is what makes a checkpoint load safely across
             torch versions and lets load_checkpoint restore optimizer state
             for resumed training, not just inference.
  sklearn  (Phase 2 SVM baseline, SHAP RandomForest surrogate) ->
             save_sklearn_model/load_sklearn_model, joblib (.pkl) - the
             standard scikit-learn format, and more efficient than stdlib
             pickle for the numpy arrays inside a fitted estimator.

.h5 (Keras/TensorFlow's format) is not used anywhere in this project since
nothing here is a Keras model.
"""

import os
import time
from pathlib import Path
from typing import Optional

import joblib
import torch


# "model_state" holds every trainable parameter plus every buffer, but NOT
# the frozen parameters — i.e. the frozen wav2vec2 backbone, which is
# reloaded from its pretrained checkpoint whenever the model is built. For
# the three-branch model that is ~0.75M of ~95M parameters: a ~9 MB file (weights
# plus AdamW state)
# instead of ~386 MB, written after every epoch. The full-size files filled
# 8.5 GB of Kaggle's ~20 GB /kaggle/working after 11 of 15 folds.
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


def save_sklearn_model(path: Path, model) -> None:
    """Persist a fitted sklearn estimator (LinearSVC/CalibratedClassifierCV,
    RandomForestClassifier, ...) via joblib. Used for Phase 2's per-fold SVM
    baseline and the SHAP surrogate models — neither was saved anywhere
    before, so refitting was the only way to reuse either one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_sklearn_model(path: Path):
    """Inverse of save_sklearn_model — returns the fitted estimator as-is."""
    return joblib.load(path)
