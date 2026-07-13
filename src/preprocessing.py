"""
Audio preprocessing utilities.

Every utterance goes through the same deterministic chain:
  1. Load and mix down to mono.
  2. Resample to 16 kHz.
  3. Trim silence with voice activity detection.
  4. Pad or truncate to a fixed 4-second window.

MFCC extraction produces 13 coefficients plus delta and delta-delta,
giving the 39-dimensional per-frame representation used by the
Acoustic Pathway (matching the base paper's baseline features).
"""

import torch
import torchaudio

from src import config


def load_and_preprocess(filepath: str) -> torch.Tensor:
    """Load one utterance and return a (1, MAX_SAMPLES) 16 kHz mono waveform."""
    waveform, sr = torchaudio.load(filepath)

    if waveform.shape[0] > 1:                                # stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != config.TARGET_SR:                               # resample
        waveform = torchaudio.functional.resample(waveform, sr, config.TARGET_SR)

    waveform = torchaudio.functional.vad(                    # trim silence
        waveform, sample_rate=config.TARGET_SR)

    if waveform.shape[1] < config.MAX_SAMPLES:               # pad / truncate
        pad = config.MAX_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :config.MAX_SAMPLES]

    return waveform


def build_mfcc_transform() -> torchaudio.transforms.MFCC:
    """MFCC transform configured from src.config (13 coefficients)."""
    return torchaudio.transforms.MFCC(
        sample_rate=config.TARGET_SR,
        n_mfcc=config.N_MFCC,
        melkwargs=dict(config.MEL_KWARGS),
    )


def extract_mfcc_features(waveform: torch.Tensor,
                          mfcc_transform: torchaudio.transforms.MFCC) -> torch.Tensor:
    """MFCC + delta + delta-delta, concatenated to 39 dims per frame."""
    mfcc = mfcc_transform(waveform)
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)
    return torch.cat([mfcc, delta, delta2], dim=1)           # (1, 39, frames)
