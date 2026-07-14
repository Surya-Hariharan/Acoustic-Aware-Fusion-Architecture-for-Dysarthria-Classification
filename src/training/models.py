"""
Model factory for train.py.

Every model exposes the same calling convention — forward(waveform, mfcc, praat)
and forward_features(waveform, mfcc, praat) — so the training engine never needs
to know which pathway(s) a given model actually uses. This lines up all six
ablation variants as a single --model switch instead of six bespoke scripts:

    acoustic                A  MFCC 1D-CNN only
    deep_frozen             B  frozen wav2vec 2.0 + MLP head
    deep_lora               C  wav2vec 2.0 + LoRA + MLP head
    fusion                  D  LoRA wav2vec + MFCC CNN, concatenated
    attention_fusion        E  LoRA wav2vec + MFCC CNN, cross-attended (Phase 6)
    attention_fusion_praat  F  Model E + Praat features as a third pathway

Every argument is optional and models ignore what they do not use — `praat` is
supplied by the DataLoader only for Model F, so it arrives as None everywhere
else. Widening the signature (rather than special-casing Model F in the engine)
is what keeps run_epoch free of any per-model branching.
"""

import torch
import torch.nn as nn

from src import config
from src.models.acoustic_pathway import AcousticPathway
from src.models.attention_fusion import (AttentionFusionModel,
                                         AttentionFusionPraatModel)
from src.models.deep_pathway import DeepPathway
from src.models.concat_fusion import FusionModel

MODEL_NAMES = ("acoustic", "deep_frozen", "deep_lora", "fusion",
               "attention_fusion", "attention_fusion_praat")

# One-line description per variant, used in the run banner and the ablation
# table so a reader never has to decode a bare model string.
MODEL_DESCRIPTIONS = {
    "acoustic": "Model A — MFCC 1D-CNN (cepstral features only)",
    "deep_frozen": "Model B — frozen wav2vec 2.0 + MLP head",
    "deep_lora": "Model C — wav2vec 2.0 + LoRA adapters + MLP head",
    "fusion": "Model D — LoRA wav2vec + MFCC CNN, concatenated",
    "attention_fusion": "Model E — LoRA wav2vec + MFCC CNN, cross-attended",
    "attention_fusion_praat": "Model F — Model E + Praat features (third pathway)",
}

# Models whose DataLoader must also carry Phase 4's Praat feature vector.
# src.training.runner reads this to decide whether to load praat_features.csv.
MODELS_REQUIRING_PRAAT = frozenset({"attention_fusion_praat"})


class AcousticClassifier(nn.Module):
    """Acoustic Pathway (MFCC 1D-CNN) + classification head. Ablation Model A."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.acoustic_pathway = AcousticPathway()
        self.classifier = nn.Sequential(
            nn.Linear(config.ACOUSTIC_EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward_features(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                         praat: torch.Tensor = None) -> torch.Tensor:
        return self.acoustic_pathway(mfcc)

    def forward(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                praat: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(self.forward_features(waveform, mfcc, praat))


class DeepClassifier(nn.Module):
    """Deep Pathway (wav2vec 2.0) + classification head.

    use_lora=True  -> ablation Model C (LoRA wav2vec).
    use_lora=False -> ablation Model B (frozen wav2vec, base-paper style
                       feature extractor, but with an MLP head trained on
                       top instead of the paper's SVM — see Phase 2 for the
                       literal SVM reproduction).
    """

    def __init__(self, num_classes: int, use_lora: bool = True):
        super().__init__()
        self.deep_pathway = DeepPathway(use_lora=use_lora)
        self.classifier = nn.Sequential(
            nn.Linear(config.WAV2VEC_EMBED_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                         praat: torch.Tensor = None) -> torch.Tensor:
        return self.deep_pathway(waveform)

    def forward(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                praat: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(self.forward_features(waveform, mfcc, praat))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Instantiate one of the six architecture variants by name."""
    if model_name == "acoustic":
        return AcousticClassifier(num_classes=num_classes)
    if model_name == "deep_frozen":
        return DeepClassifier(num_classes=num_classes, use_lora=False)
    if model_name == "deep_lora":
        return DeepClassifier(num_classes=num_classes, use_lora=True)
    if model_name == "fusion":
        return FusionModel(num_classes=num_classes)
    if model_name == "attention_fusion":
        return AttentionFusionModel(num_classes=num_classes)
    if model_name == "attention_fusion_praat":
        return AttentionFusionPraatModel(num_classes=num_classes)
    raise ValueError(f"Unknown model '{model_name}'. Choose from {MODEL_NAMES}.")
