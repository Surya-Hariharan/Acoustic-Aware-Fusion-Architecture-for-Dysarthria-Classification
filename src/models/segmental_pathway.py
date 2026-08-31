"""
Segmental branch of the three-branch gated-fusion severity architecture.

Local, short-time acoustic-articulatory behavior: MFCC + delta + delta-delta
(the existing 39-dim/frame Acoustic Pathway input) concatenated with
framewise formants F1-F3 (resonance/articulation) and framewise HNR (voice
quality) — config.SEGMENTAL_CHANNELS = 43 channels/frame total. Deliberately
excludes F0/energy/duration (those belong to the Suprasegmental branch) and
jitter/shimmer/CPPS (kept as utterance-level SHAP-surrogate features only —
framewise glottal-pulse-based measures do not have a natural per-frame
value the way a spectral envelope does).

Same 3-layer 1D-CNN + masked-mean-pool architecture as
src.models.acoustic_pathway.AcousticPathway (same kernel/stride, so the
pooling arithmetic is identical and directly reused), plus a final
Linear bottleneck down to config.SEGMENTAL_EMBED_DIM (64) — see the
architecture plan's Part 2, Component 5 for why a bottleneck matters here.
"""

from typing import Optional

import torch
import torch.nn as nn

from src import config


class SegmentalPathway(nn.Module):
    """1D-CNN: (batch, 43, frames) segmental input -> (batch, 64) embedding."""

    def __init__(self,
                 in_channels: int = config.SEGMENTAL_CHANNELS,     # 43
                 hidden_dim: int = 128,
                 embed_dim: int = config.SEGMENTAL_EMBED_DIM):     # 64
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
        """Same two stride-2, kernel-2 MaxPool1d layers as
        AcousticPathway._pool_frames — identical conv geometry, so the same
        length formula applies (kept as a local copy rather than a shared
        import so this module has no dependency on the Acoustic Pathway)."""
        pooled = (num_frames - 2) // 2 + 1
        return (pooled - 2) // 2 + 1

    def valid_frame_count(self, waveform_attention_mask: torch.Tensor) -> torch.Tensor:
        """Sample-level mask -> pooled valid-frame count. Segmental input
        shares the MFCC frame grid (both derived from the speech-focused
        profile's mfcc_frame_count), so this mirrors
        AcousticPathway.valid_frame_count exactly."""
        from src.preprocessing import mfcc_frame_count
        valid_samples = waveform_attention_mask.sum(dim=1)
        mfcc_frames = mfcc_frame_count(valid_samples)
        return self._pool_frames(mfcc_frames).clamp(min=1)

    def forward(self, segmental_features: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            segmental_features: (batch, 43, frames) — MFCC+delta+delta-delta
                concatenated with framewise formants+HNR along the channel
                axis (see src.training.data.build_segmental_features).
            attention_mask: (batch, samples) sample-level waveform mask, same
                convention as AcousticPathway.forward — excludes frames drawn
                from the fixed window's zero-padded tail from the pool.
        Returns:
            (batch, 64) segmental embedding, Z_segmental.
        """
        features = self.conv(segmental_features)              # (B, hidden_dim, T)
        if attention_mask is None:
            pooled = features.mean(dim=-1)
        else:
            valid_frames = self.valid_frame_count(attention_mask)      # (B,)
            frame_idx = torch.arange(features.shape[-1], device=features.device)[None, :]
            frame_mask = (frame_idx < valid_frames[:, None]).unsqueeze(1).to(features.dtype)
            summed = (features * frame_mask).sum(dim=-1)
            counts = frame_mask.sum(dim=-1).clamp(min=1.0)
            pooled = summed / counts
        return self.bottleneck(pooled)
