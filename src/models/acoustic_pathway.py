"""
Acoustic Pathway (Role 2): lightweight 1D-CNN over MFCC features.

Takes the 39-dimensional MFCC (+delta +delta-delta) frame sequence and
produces a dense, deterministic physical embedding that grounds the fusion
model in classical signal processing.
"""

from typing import Optional

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

    def forward_sequence(self, mfcc: torch.Tensor) -> torch.Tensor:
        """
        The per-frame convolutional feature map, *before* the global average pool
        that forward() applies — the acoustic counterpart to
        DeepPathway.forward_sequence, and for the same Phase 6 reason.

        Args:
            mfcc: (batch, 39, frames) — or (batch, 1, 39, frames) from the dataset.
        Returns:
            (batch, frames', embed_dim) — the two MaxPool1d(2) stages quarter the
            frame count, so a 401-frame MFCC becomes ~100 tokens of 128 dims.
            Note the (B, T, C) layout: nn.MultiheadAttention(batch_first=True)
            wants channels last, whereas Conv1d emits (B, C, T).
        """
        if mfcc.dim() == 4:                                   # (B, 1, 39, T)
            mfcc = mfcc.squeeze(1)
        return self.conv(mfcc).transpose(1, 2)                # (B, C, T) -> (B, T, C)

    @staticmethod
    def _pool_frames(num_frames):
        """Apply this module's two stride-2, kernel-2 MaxPool1d layers'
        length formula ((L - 2) // 2 + 1) twice — works on an int or a
        LongTensor alike, so it covers both a fixed total frame count and a
        per-sample valid-frame count with the same arithmetic."""
        pooled = (num_frames - 2) // 2 + 1
        return (pooled - 2) // 2 + 1

    def valid_frame_count(self, waveform_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert a sample-level waveform attention mask (see DeepPathway) into
        the number of *pooled* frames (this module's forward_sequence output)
        that fall before the padded tail — mirrors torchaudio.transforms.MFCC's
        center=True STFT framing (frames = valid_samples // hop_length + 1),
        then this module's pooling stages. Both pathways read the same padded
        waveform (src/dataset.py), so this needs no separate length input.
        """
        valid_samples = waveform_attention_mask.sum(dim=1)
        mfcc_frames = valid_samples // config.MEL_KWARGS["hop_length"] + 1
        return self._pool_frames(mfcc_frames).clamp(min=1)

    def sequence_key_padding_mask(self, mfcc: torch.Tensor,
                                  attention_mask: torch.Tensor) -> torch.Tensor:
        """
        The frame-level padding mask matching forward_sequence's output, in
        nn.MultiheadAttention's key_padding_mask convention (True = ignore
        this position) — the acoustic counterpart to
        DeepPathway.sequence_key_padding_mask, for Phase 6's cross-attention.
        `mfcc`'s raw (pre-pool) frame count is fixed (every clip is padded to
        the same MAX_SAMPLES window), so the pooled total below is the same
        for every row in the batch — only which of those pooled frames are
        valid varies per row, via valid_frame_count.
        """
        num_frames = self._pool_frames(mfcc.shape[-1])
        valid_frames = self.valid_frame_count(attention_mask)
        frame_idx = torch.arange(num_frames, device=mfcc.device)[None, :]
        return frame_idx >= valid_frames[:, None]            # True where padded

    def forward(self, mfcc: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            mfcc: (batch, 39, frames) — squeeze the channel dim from the
                  dataset's (batch, 1, 39, frames) before calling if needed.
            attention_mask: (batch, samples) sample-level waveform mask (see
                  DeepPathway.forward). When given, the pool below excludes
                  frames derived from the padded tail of the waveform;
                  otherwise every padded utterance's embedding is diluted by
                  the fixed-window silence, same issue as DeepPathway's.
        Returns:
            (batch, embed_dim) acoustic embedding.
        """
        if mfcc.dim() == 4:                                   # (B, 1, 39, T)
            mfcc = mfcc.squeeze(1)
        features = self.conv(mfcc)                             # (B, C, T)
        if attention_mask is None:
            return self.pool(features).squeeze(-1)

        valid_frames = self.valid_frame_count(attention_mask)   # (B,)
        frame_idx = torch.arange(features.shape[-1], device=features.device)[None, :]
        frame_mask = (frame_idx < valid_frames[:, None]).unsqueeze(1).to(features.dtype)  # (B,1,T)
        summed = (features * frame_mask).sum(dim=-1)
        counts = frame_mask.sum(dim=-1).clamp(min=1.0)
        return summed / counts
