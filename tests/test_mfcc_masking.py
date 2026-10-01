"""
Masked pooling in the segmental branch (MFCC + formants + HNR): the padded
tail of the fixed 4 s window must not reach the embedding, and the pool must
be exactly the mean over real frames.

The conv stack (k5 -> pool2 -> k5 -> pool2 -> k3) gives each output frame a
~26-input-frame receptive field, so padding right at the speech boundary is
unavoidably convolved into the last valid frames. What masking guarantees, and
what is pinned here, is that padding beyond that receptive field — most of a
typical window — contributes nothing.
"""

import torch

from src import config
from src.models.segmental_pathway import SegmentalPathway
from src.preprocessing import mfcc_frame_count

RECEPTIVE_FIELD_MARGIN = 48          # input frames, comfortably beyond the ~26-frame span
TOTAL_FRAMES = mfcc_frame_count(config.MAX_SAMPLES)


def _sample_mask_for(valid_frames: torch.Tensor) -> torch.Tensor:
    """Waveform mask whose implied MFCC frame count is `valid_frames`."""
    hop = config.MEL_KWARGS["hop_length"]
    return torch.arange(config.MAX_SAMPLES)[None, :] < (valid_frames[:, None] * hop)


def test_mfcc_frame_count_defines_the_401_frame_window():
    assert TOTAL_FRAMES == 401
    assert mfcc_frame_count(torch.tensor([0, 160, 64000])).tolist() == [1, 2, 401]


def test_padding_beyond_the_receptive_field_cannot_affect_the_embedding():
    torch.manual_seed(0)
    valid_frames = torch.tensor([80, 160, 240, 40])
    model = SegmentalPathway().eval()
    features = torch.randn(4, config.SEGMENTAL_CHANNELS, TOTAL_FRAMES)
    zero_padded, garbage_padded = features.clone(), features.clone()
    for i, v in enumerate(valid_frames.tolist()):
        zero_padded[i, :, v:] = 0.0
        garbage_padded[i, :, v:] = 0.0
        garbage_padded[i, :, v + RECEPTIVE_FIELD_MARGIN:] = 1e4
    mask = _sample_mask_for(valid_frames)
    with torch.no_grad():
        assert torch.allclose(model(zero_padded, attention_mask=mask),
                              model(garbage_padded, attention_mask=mask), atol=1e-4)


def test_masked_pool_equals_manual_mean_over_valid_frames_only():
    torch.manual_seed(1)
    valid_frames = torch.tensor([100, 200])
    model = SegmentalPathway().eval()
    features = torch.randn(2, config.SEGMENTAL_CHANNELS, TOTAL_FRAMES)
    for i, v in enumerate(valid_frames.tolist()):
        features[i, :, v:] = 0.0
    mask = _sample_mask_for(valid_frames)
    with torch.no_grad():
        embedding = model(features, attention_mask=mask)
        feature_map = model.conv(features)
        n_valid = model.valid_frame_count(mask)
    for i, n in enumerate(n_valid.tolist()):
        expected = model.bottleneck(feature_map[i, :, :n].mean(dim=-1))
        assert torch.allclose(embedding[i], expected, atol=1e-5)


def test_masking_actually_changes_the_result():
    """With ~86% of the window padded, the masked embedding must differ from
    the unmasked full-window mean — otherwise the mask stopped working."""
    torch.manual_seed(2)
    model = SegmentalPathway().eval()
    features = torch.randn(1, config.SEGMENTAL_CHANNELS, TOTAL_FRAMES)
    features[0, :, 55:] = 0.0
    with torch.no_grad():
        masked = model(features, attention_mask=_sample_mask_for(torch.tensor([55])))
        unmasked = model(features)
    assert not torch.allclose(masked, unmasked, atol=1e-3)
