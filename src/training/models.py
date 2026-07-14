"""
Model factory for train.py.

Every model exposes the same calling convention — forward(waveform, mfcc)
and forward_features(waveform, mfcc) — so the training engine never needs
to know which pathway(s) a given model actually uses. This also lines up
the four Phase 3 ablation variants (MFCC-only, frozen wav2vec, LoRA
wav2vec, fusion) as a single --model switch instead of four bespoke
scripts.
"""

import torch
import torch.nn as nn

from src import config
from src.models.acoustic_pathway import AcousticPathway
from src.models.deep_pathway import DeepPathway
from src.models.fusion import FusionModel

MODEL_NAMES = ("acoustic", "deep_frozen", "deep_lora", "fusion")


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

    def forward_features(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None) -> torch.Tensor:
        return self.acoustic_pathway(mfcc)

    def forward(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(self.forward_features(waveform, mfcc))


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

    def forward_features(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None) -> torch.Tensor:
        return self.deep_pathway(waveform)

    def forward(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(self.forward_features(waveform, mfcc))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Instantiate one of the four architecture variants by name."""
    if model_name == "acoustic":
        return AcousticClassifier(num_classes=num_classes)
    if model_name == "deep_frozen":
        return DeepClassifier(num_classes=num_classes, use_lora=False)
    if model_name == "deep_lora":
        return DeepClassifier(num_classes=num_classes, use_lora=True)
    if model_name == "fusion":
        return FusionModel(num_classes=num_classes)
    raise ValueError(f"Unknown model '{model_name}'. Choose from {MODEL_NAMES}.")
