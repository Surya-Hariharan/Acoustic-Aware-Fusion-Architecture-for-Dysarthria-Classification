"""
Model registry: the primary three-branch model and its controlled ablations.

Every entry is a GatedFusionModel with the same encoders, CORAL head, data,
folds, validation protocol and optimizer; only the listed switches differ.
Unlisted switches take the ablation baseline — gated fusion, no
complementarity penalty, no speaker adversary — so ab7/ab8 each add exactly
one regularizer to ab6, and SEVERITY_MODEL_NAME (both) is the full model.
"""

from typing import Dict, Optional

import torch.nn as nn

from src import config
from src.models.gated_fusion import BRANCH_NAMES, GatedFusionModel

SEVERITY_MODEL_NAME = "gated_fusion_three_branch"

SEVERITY_ABLATIONS = {
    "ab1_wav2vec2_only": dict(branches=("learned",)),
    "ab2_acoustic_only": dict(branches=("segmental", "supra")),
    "ab3_wav2vec2_acoustic_concat": dict(branches=BRANCH_NAMES, fusion="concat"),
    "ab4_wav2vec2_segmental": dict(branches=("learned", "segmental")),
    "ab5_wav2vec2_suprasegmental": dict(branches=("learned", "supra")),
    "ab6_full_fusion": dict(branches=BRANCH_NAMES),
    "ab7_full_complementarity": dict(branches=BRANCH_NAMES, use_complementarity=True),
    "ab8_full_speaker_grl": dict(branches=BRANCH_NAMES, use_speaker_adversary=True),
}
ABLATION_DEFAULTS = dict(fusion="gated", use_complementarity=False, use_speaker_adversary=False)
MODEL_NAMES = (SEVERITY_MODEL_NAME, *SEVERITY_ABLATIONS)


def model_switches(model_name: str) -> Dict[str, object]:
    """The GatedFusionModel constructor switches for a registry name."""
    if model_name == SEVERITY_MODEL_NAME:
        return {}
    if model_name not in SEVERITY_ABLATIONS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {MODEL_NAMES}.")
    return {**ABLATION_DEFAULTS, **SEVERITY_ABLATIONS[model_name]}


def build_model(model_name: str, num_speakers: int = 1,
                gradient_checkpointing: Optional[bool] = None) -> GatedFusionModel:
    """num_speakers sizes the adversarial speaker head (this fold's training
    speakers); it does not affect the severity path."""
    return GatedFusionModel(num_classes=config.NUM_CLASSES, num_speakers=num_speakers,
                            gradient_checkpointing=gradient_checkpointing,
                            **model_switches(model_name))


def parameter_counts(model: nn.Module) -> Dict[str, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total,
            "trainable_pct": 100 * trainable / total if total else 0.0}
