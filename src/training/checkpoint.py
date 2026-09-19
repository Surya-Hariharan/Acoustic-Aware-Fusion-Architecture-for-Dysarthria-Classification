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

from pathlib import Path
from typing import Optional

import joblib
import torch


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, scaler: torch.amp.GradScaler, epoch: int,
                    monitored_value: float, extra: Optional[dict] = None) -> None:
    """`extra` is merged into the saved dict as-is (e.g. early-stopping state,
    fold id) -- used by the per-epoch "latest" resume checkpoint in
    src.training.runner.run_fold; the best-checkpoint call site simply omits it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "monitored_value": monitored_value,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


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
