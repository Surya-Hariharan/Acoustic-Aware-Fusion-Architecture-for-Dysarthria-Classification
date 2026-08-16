"""
Numeric proof that MFCC padding does not leak into the model representation.

Two things are checked, both without touching real audio or the network
weights that would make a slow integration test:
  1. AcousticPathway's masked pooling gives (near-)identical output whether
     the padded tail is all-zero or arbitrary garbage, as long as the mask
     correctly marks it as padding — the pooling must be reading the mask,
     not merely getting lucky because padding happens to be zero.
  2. src.preprocessing.validate_mfcc_output raises on the violations it
     claims to catch (dimension mismatch, NaN, a mask that doesn't match
     valid_length) and passes silently on a genuinely valid tensor.

Run with: pytest tests/test_mfcc_masking.py -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.models.acoustic_pathway import AcousticPathway
from src.preprocessing import mfcc_frame_count, mfcc_valid_frame_mask, validate_mfcc_output


# AcousticPathway's conv stack is Conv1d(k=5,p=2) -> MaxPool1d(2) ->
# Conv1d(k=5,p=2) -> MaxPool1d(2) -> Conv1d(k=3,p=1), so one OUTPUT frame draws
# on roughly 26 input frames. Masked pooling therefore cannot make the padded
# tail perfectly invisible: padding within a receptive field of the boundary is
# convolved into the last few VALID pooled frames before any pooling happens.
#
# That boundary smear is inherent to a convolutional encoder over a padded
# sequence and is bounded — in the real pipeline the padding is zeros, not
# adversarial values. What masking must guarantee, and what these tests pin, is
# the part that actually dominated the representation before the fix: the ~300
# padded frames of a typical 4-second window must contribute nothing to the
# pooled embedding. A generous margin is used below so the test asserts the
# real guarantee rather than an architectural impossibility.
RECEPTIVE_FIELD_MARGIN = 48          # raw MFCC frames; comfortably > the ~26-frame span


def _sample_mask_for(valid_frames: torch.Tensor) -> torch.Tensor:
    """Sample-level waveform mask whose implied MFCC frame count is
    `valid_frames` — the input AcousticPathway.valid_frame_count expects."""
    hop = config.MEL_KWARGS["hop_length"]
    return torch.arange(config.MAX_SAMPLES)[None, :] < (valid_frames[:, None] * hop)


def test_padding_beyond_the_receptive_field_cannot_affect_the_embedding():
    """The guarantee masking actually provides: whatever sits in the padded
    tail, beyond a receptive field of the boundary, is invisible to the pooled
    output. This is the property that was broken before — an unmasked mean over
    the full window let hundreds of padded frames dilute every embedding."""
    torch.manual_seed(0)
    total_frames = mfcc_frame_count(config.MAX_SAMPLES)
    channels = 3 * config.N_MFCC
    valid_frames = torch.tensor([80, 160, 240, 40])

    model = AcousticPathway().eval()
    features = torch.randn(4, channels, total_frames)

    zero_padded = features.clone()
    garbage_padded = features.clone()
    for i, v in enumerate(valid_frames.tolist()):
        zero_padded[i, :, v:] = 0.0
        garbage_padded[i, :, v:] = 0.0
        # Perturb only well past the boundary, where no valid output frame's
        # receptive field can reach.
        garbage_padded[i, :, v + RECEPTIVE_FIELD_MARGIN:] = 1e4

    sample_mask = _sample_mask_for(valid_frames)
    with torch.no_grad():
        out_zero = model(zero_padded, attention_mask=sample_mask)
        out_garbage = model(garbage_padded, attention_mask=sample_mask)

    assert torch.allclose(out_zero, out_garbage, atol=1e-5), (
        "changing the far padded tail changed the pooled embedding — padding is "
        "leaking into the representation")


def test_masked_pool_equals_manual_mean_over_valid_frames_only():
    """The masked pool must be exactly the mean over valid pooled positions —
    not an approximation, and not the full-window mean."""
    torch.manual_seed(1)
    total_frames = mfcc_frame_count(config.MAX_SAMPLES)
    valid_frames = torch.tensor([100, 200])

    model = AcousticPathway().eval()
    features = torch.randn(2, 3 * config.N_MFCC, total_frames)
    for i, v in enumerate(valid_frames.tolist()):
        features[i, :, v:] = 0.0

    sample_mask = _sample_mask_for(valid_frames)
    with torch.no_grad():
        pooled = model(features, attention_mask=sample_mask)
        feature_map = model.conv(features)                       # (B, C, T')
        n_valid = model.valid_frame_count(sample_mask)           # (B,)

    for i, n in enumerate(n_valid.tolist()):
        expected = feature_map[i, :, :n].mean(dim=-1)
        assert torch.allclose(pooled[i], expected, atol=1e-5), (
            f"row {i}: masked pool != mean over its {n} valid pooled frames")


def test_masking_actually_changes_the_result():
    """A guard against the mask silently becoming a no-op: with most of the
    window padded, the masked embedding must differ substantially from the
    unmasked full-window mean. If these ever match, masking stopped working."""
    torch.manual_seed(2)
    total_frames = mfcc_frame_count(config.MAX_SAMPLES)
    valid_frames = torch.tensor([55])                # ~0.55s of speech in a 4s window

    model = AcousticPathway().eval()
    features = torch.randn(1, 3 * config.N_MFCC, total_frames)
    features[0, :, 55:] = 0.0

    with torch.no_grad():
        masked = model(features, attention_mask=_sample_mask_for(valid_frames))
        unmasked = model(features)                   # no mask -> pools the whole window

    assert not torch.allclose(masked, unmasked, atol=1e-3), (
        "masked and unmasked pooling agree on an 86%-padded input — the mask "
        "is not being applied")


def test_validate_mfcc_output_accepts_well_formed_tensor():
    total_frames = 50
    valid_length = 4000  # samples -> mfcc_frame_count(4000) frames of real audio
    features = torch.randn(1, 3 * config.N_MFCC, total_frames)
    mask = mfcc_valid_frame_mask(total_frames, valid_length)
    validate_mfcc_output(features, valid_length, mask)  # must not raise


def test_validate_mfcc_output_rejects_nan():
    total_frames = 50
    valid_length = 4000
    features = torch.randn(1, 3 * config.N_MFCC, total_frames)
    features[0, 0, 0] = float("nan")
    try:
        validate_mfcc_output(features, valid_length)
        assert False, "expected ValueError on NaN input"
    except ValueError:
        pass


def test_validate_mfcc_output_rejects_wrong_channel_dim():
    features = torch.randn(1, 3 * config.N_MFCC + 1, 50)
    try:
        validate_mfcc_output(features, 4000)
        assert False, "expected ValueError on wrong channel dimension"
    except ValueError:
        pass


def test_validate_mfcc_output_rejects_mismatched_mask():
    total_frames = 50
    valid_length = 4000
    features = torch.randn(1, 3 * config.N_MFCC, total_frames)
    wrong_mask = torch.ones(total_frames, dtype=torch.bool)  # claims every frame valid
    try:
        validate_mfcc_output(features, valid_length, wrong_mask)
        assert False, "expected ValueError on a mask inconsistent with valid_length"
    except ValueError:
        pass


def test_mfcc_frame_count_matches_config_max_samples():
    # config.py documents MFCC shape as (1, 39, 401) for MAX_SAMPLES=64000,
    # hop_length=160 — this pins that relationship so a future config change
    # is caught here rather than only showing up as a shape mismatch deep in
    # a training run.
    assert mfcc_frame_count(config.MAX_SAMPLES) == config.MAX_SAMPLES // config.MEL_KWARGS["hop_length"] + 1


if __name__ == "__main__":
    test_padding_beyond_the_receptive_field_cannot_affect_the_embedding()
    test_masked_pool_equals_manual_mean_over_valid_frames_only()
    test_masking_actually_changes_the_result()
    test_validate_mfcc_output_accepts_well_formed_tensor()
    test_validate_mfcc_output_rejects_nan()
    test_validate_mfcc_output_rejects_wrong_channel_dim()
    test_validate_mfcc_output_rejects_mismatched_mask()
    test_mfcc_frame_count_matches_config_max_samples()
    print("All MFCC masking tests passed.")
