"""
Acoustic Pathway (Role 2): lightweight 1D-CNN over MFCC features.

Takes the 39-dimensional MFCC (+delta +delta-delta) frame sequence and
produces a dense, deterministic physical embedding that grounds the fusion
model in classical signal processing.
"""

import torch
import torch.nn as nn

from src import config


class AcousticPathway(nn.Module):
    """1D-CNN: (batch, 39, frames) MFCC input -> (batch, 128) embedding."""

    def __init__(self,
                 in_channels: int = 3 * config.N_MFCC,        # 39
                 embed_dim: int = config.ACOUSTIC_EMBED_DIM):  # 128
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)                   # global average pool

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mfcc: (batch, 39, frames) — squeeze the channel dim from the
                  dataset's (batch, 1, 39, frames) before calling if needed.
        Returns:
            (batch, embed_dim) acoustic embedding.
        """
        if mfcc.dim() == 4:                                   # (B, 1, 39, T)
            mfcc = mfcc.squeeze(1)
        features = self.conv(mfcc)
        return self.pool(features).squeeze(-1)
