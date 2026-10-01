"""
Segmental branch: short-time articulatory and voice-quality behaviour.

Input is MFCC + delta + delta-delta (39) with framewise F1-F3 and HNR (4) —
config.SEGMENTAL_CHANNELS = 43 per 10 ms frame, on the speech-focused profile.
F0, voicing and intensity belong to the suprasegmental branch instead.
A 3-layer 1D-CNN, a mean-pool over real frames only, and a linear bottleneck
to config.SEGMENTAL_EMBED_DIM (64).
"""

from typing import Optional

import torch
import torch.nn as nn

from src import config


class SegmentalPathway(nn.Module):
    """(batch, 43, frames) -> (batch, 64)."""

    def __init__(self, in_channels: int = config.SEGMENTAL_CHANNELS, hidden_dim: int = 128,
                 embed_dim: int = config.SEGMENTAL_EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.bottleneck = nn.Linear(hidden_dim, embed_dim)

    @staticmethod
    def _pool_frames(num_frames):
        """Frame count after the two kernel-2, stride-2 max-pools."""
        pooled = (num_frames - 2) // 2 + 1
        return (pooled - 2) // 2 + 1

    def valid_frame_count(self, waveform_attention_mask: torch.Tensor) -> torch.Tensor:
        """Sample-level mask -> real frames left after pooling (at least 1)."""
        from src.preprocessing import mfcc_frame_count
        mfcc_frames = mfcc_frame_count(waveform_attention_mask.sum(dim=1))
        return self._pool_frames(mfcc_frames).clamp(min=1)

    def forward(self, segmental_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """attention_mask: (batch, samples) waveform mask; frames derived from
        the zero-padded tail are excluded from the pool."""
        features = self.conv(segmental_features)
        if attention_mask is None:
            return self.bottleneck(features.mean(dim=-1))
        valid = self.valid_frame_count(attention_mask)
        frame_idx = torch.arange(features.shape[-1], device=features.device)[None, :]
        mask = (frame_idx < valid[:, None]).unsqueeze(1).to(features.dtype)
        pooled = (features * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        return self.bottleneck(pooled)
