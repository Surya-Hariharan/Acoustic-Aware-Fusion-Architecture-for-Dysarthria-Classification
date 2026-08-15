"""
Fusion Architecture (Role 3): concatenation of both pathway embeddings.

The 768-dim latent embedding (Deep Pathway) and the 128-dim acoustic
embedding (Acoustic Pathway) are physically concatenated and passed through
the final classification head — the core novelty of the team spec.
"""

from typing import Optional

import torch
import torch.nn as nn

from src import config
from src.models.acoustic_pathway import AcousticPathway
from src.models.deep_pathway import DeepPathway


class FusionModel(nn.Module):
    """
    Dual-pathway fusion network.

    num_classes = 2 for the detection task (healthy vs dysarthric),
    num_classes = 4 for the severity task (Very Low / Low / Mid / High).

    use_lora=True  (default) — LoRA-adapted wav2vec 2.0 backbone (ablation
                   Model D / "LoRA-Fusion" in the supervisor's requested
                   experiment matrix).
    use_lora=False — frozen wav2vec 2.0 backbone, giving an MFCC+frozen-
                   Wav2Vec2 baseline to compare LoRA-Fusion against
                   (registered as model name "fusion_frozen" in
                   src/training/models.py).
    """

    def __init__(self, num_classes: int = 2, use_lora: bool = True):
        super().__init__()
        self.deep_pathway = DeepPathway(use_lora=use_lora)
        self.acoustic_pathway = AcousticPathway()

        fused_dim = config.WAV2VEC_EMBED_DIM + config.ACOUSTIC_EMBED_DIM  # 896
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                         praat: torch.Tensor = None,
                         attention_mask: Optional[torch.Tensor] = None,
                         deep_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) raw audio for the Deep Pathway.
            mfcc:     (batch, 39, frames) features for the Acoustic Pathway.
            praat:    ignored — accepted so every model shares one call signature
                      (see src/training/models.py). Phase 6's Model F is the only
                      variant that consumes it.
            attention_mask: (batch, samples) real-audio mask — both pathways
                      read the same padded waveform (src/dataset.py), so one
                      mask covers both (see DeepPathway/AcousticPathway).
            deep_embedding: precomputed frozen wav2vec2 vector (fusion_frozen
                      only — see MODELS_WITH_CACHEABLE_FROZEN_EMBEDDING); used
                      in place of a live self.deep_pathway forward pass.
        Returns:
            (batch, 896) fused embedding, pre-classification-head.
        """
        latent_embedding = (deep_embedding if deep_embedding is not None
                            else self.deep_pathway(waveform, attention_mask=attention_mask))  # (B, 768)
        acoustic_embedding = self.acoustic_pathway(mfcc, attention_mask=attention_mask)  # (B, 128)
        return torch.cat([latent_embedding, acoustic_embedding], dim=1)

    def forward(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                praat: torch.Tensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                deep_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) raw audio for the Deep Pathway.
            mfcc:     (batch, 39, frames) features for the Acoustic Pathway.
            praat:    ignored — see forward_features.
        Returns:
            (batch, num_classes) classification logits.
        """
        return self.classifier(self.forward_features(
            waveform, mfcc, praat, attention_mask, deep_embedding))
