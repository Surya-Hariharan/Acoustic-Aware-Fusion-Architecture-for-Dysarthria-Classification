"""
Fusion Architecture (Role 3): concatenation of both pathway embeddings.

The 768-dim latent embedding (Deep Pathway) and the 128-dim acoustic
embedding (Acoustic Pathway) are physically concatenated and passed through
the final classification head — the core novelty of the team spec.
"""

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
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.deep_pathway = DeepPathway()
        self.acoustic_pathway = AcousticPathway()

        fused_dim = config.WAV2VEC_EMBED_DIM + config.ACOUSTIC_EMBED_DIM  # 896
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveform: torch.Tensor, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) raw audio for the Deep Pathway.
            mfcc:     (batch, 39, frames) features for the Acoustic Pathway.
        Returns:
            (batch, num_classes) classification logits.
        """
        latent_embedding = self.deep_pathway(waveform)        # (B, 768)
        acoustic_embedding = self.acoustic_pathway(mfcc)      # (B, 128)
        fused = torch.cat([latent_embedding, acoustic_embedding], dim=1)
        return self.classifier(fused)
