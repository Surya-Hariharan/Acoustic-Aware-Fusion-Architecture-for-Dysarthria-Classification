"""
Suprasegmental branch of the three-branch gated-fusion severity architecture.

The one branch of the three that is genuinely new (see the architecture
plan's Part 2, Component 6) — pitch, energy, and voicing behavior, framewise,
on the TEMPORAL-PRESERVING preprocessing profile (wider VAD margin than the
speech-focused profile the Learned/Segmental branches use — see
src.preprocessing.load_and_preprocess_supra). Deliberately restricted to
what isolated-word UA-Speech recordings can support: no phrase-level
intonation or multi-word rhythm, only F0 contour, a voicing mask, and an
energy/intensity contour. Global duration and voiced/unvoiced ratio are not
fed as extra channels — they're already implicit in how much of the sequence
the voicing mask marks valid, and are logged separately as diagnostics
rather than duplicated into the input.

A smaller CNN than SegmentalPathway (3 input channels instead of 43 — this
branch carries much less raw information) + masked mean-pool + a bottleneck
projection to config.SUPRA_EMBED_DIM (64).
"""

from typing import Optional

import torch
import torch.nn as nn

from src import config


class SuprasegmentalPathway(nn.Module):
    """1D-CNN: (batch, 3, frames) F0/voicing/energy input -> (batch, 64) embedding."""

    def __init__(self,
                 in_channels: int = config.SUPRA_CHANNELS,       # 3
                 hidden_dim: int = 64,
                 embed_dim: int = config.SUPRA_EMBED_DIM):       # 64
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.bottleneck = nn.Linear(hidden_dim, embed_dim)

    @staticmethod
    def _pool_frames(num_frames):
        """One stride-2, kernel-2 MaxPool1d layer (half the depth of
        SegmentalPathway's two — this branch's signal is smoother and
        needs less temporal downsampling to summarize)."""
        return (num_frames - 2) // 2 + 1

    def forward(self, supra_features: torch.Tensor,
               valid_frames: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            supra_features: (batch, 3, frames) — f0_semitones, voicing,
                intensity_db, on the same frame grid as the segmental branch
                (see src.praat.extract_suprasegmental_sequence).
            valid_frames: (batch,) PRE-pool real-frame count on that same
                grid, computed from the temporal-preserving profile's own
                valid_length (see src.dataset — a distinct quantity from the
                speech-focused profile's mfcc_valid_frames, since the two
                preprocessing profiles trim differently). None pools the
                full window (only safe for a single, known-unpadded clip).
        Returns:
            (batch, 64) suprasegmental embedding, Z_supra.
        """
        features = self.conv(supra_features)                    # (B, hidden_dim, T')
        if valid_frames is None:
            pooled = features.mean(dim=-1)
        else:
            pooled_valid = self._pool_frames(valid_frames).clamp(min=1)  # (B,)
            frame_idx = torch.arange(features.shape[-1], device=features.device)[None, :]
            frame_mask = (frame_idx < pooled_valid[:, None]).unsqueeze(1).to(features.dtype)
            summed = (features * frame_mask).sum(dim=-1)
            counts = frame_mask.sum(dim=-1).clamp(min=1.0)
            pooled = summed / counts
        return self.bottleneck(pooled)
