"""
Numeric checks for the Suprasegmental branch's masking (mirrors
tests/test_mfcc_masking.py's pattern for SuprasegmentalPathway) and for the
framewise F0/voicing/intensity and formant/HNR extraction functions'
edge-case handling (src/praat.py) — never raising, never fabricating a
"real" voiced F0 value for a frame that has none.

Run with: pytest tests/test_supra_sequence_masking.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.suprasegmental_pathway import SuprasegmentalPathway
from src.praat import (_hz_to_semitones, _interpolate_unvoiced,
                       extract_segmental_extra_sequence, extract_suprasegmental_sequence)

RECEPTIVE_FIELD_MARGIN = 16   # SuprasegmentalPathway is one Conv1d(k=5,p=2)->MaxPool1d(2)
                              # -> Conv1d(k=5,p=2): a comfortably generous margin past the
                              # ~5-frame receptive field at the boundary.


def test_supra_padding_beyond_receptive_field_cannot_affect_embedding():
    torch.manual_seed(0)
    total_frames = 401
    channels = 3
    valid_frames = torch.tensor([80, 160, 240, 40])

    model = SuprasegmentalPathway().eval()
    features = torch.randn(4, channels, total_frames)

    zero_padded = features.clone()
    garbage_padded = features.clone()
    for i, v in enumerate(valid_frames.tolist()):
        zero_padded[i, :, v:] = 0.0
        garbage_padded[i, :, v:] = 0.0
        garbage_padded[i, :, v + RECEPTIVE_FIELD_MARGIN:] = 1e4

    with torch.no_grad():
        out_zero = model(zero_padded, valid_frames=valid_frames)
        out_garbage = model(garbage_padded, valid_frames=valid_frames)

    assert torch.allclose(out_zero, out_garbage, atol=1e-4), (
        "changing the far padded tail changed the pooled embedding")


def test_supra_masked_pool_equals_manual_mean_over_valid_frames():
    torch.manual_seed(1)
    total_frames = 401
    valid_frames = torch.tensor([100, 200])

    model = SuprasegmentalPathway().eval()
    features = torch.randn(2, 3, total_frames)
    for i, v in enumerate(valid_frames.tolist()):
        features[i, :, v:] = 0.0

    with torch.no_grad():
        pooled = model(features, valid_frames=valid_frames)
        feature_map = model.conv(features)
        pooled_valid = model._pool_frames(valid_frames).clamp(min=1)

    # model(...) applies the bottleneck Linear AFTER pooling (unlike
    # AcousticPathway, whose conv stack already emits embed_dim channels) —
    # the manual reference must go through the same bottleneck to compare
    # apples to apples.
    for i, n in enumerate(pooled_valid.tolist()):
        expected_pre_bottleneck = feature_map[i, :, :n].mean(dim=-1)
        with torch.no_grad():
            expected = model.bottleneck(expected_pre_bottleneck)
        assert torch.allclose(pooled[i], expected, atol=1e-4)


def test_supra_masking_actually_changes_the_result():
    torch.manual_seed(2)
    total_frames = 401
    valid_frames = torch.tensor([50])

    model = SuprasegmentalPathway().eval()
    features = torch.randn(1, 3, total_frames)
    features[0, :, 50:] = 0.0

    with torch.no_grad():
        masked = model(features, valid_frames=valid_frames)
        unmasked = model(features)

    assert not torch.allclose(masked, unmasked, atol=1e-3)


def test_interpolate_unvoiced_fills_only_the_gaps():
    f0 = np.array([100.0, 0.0, 0.0, 200.0, 150.0], dtype=np.float32)
    interpolated = _interpolate_unvoiced(f0)
    assert interpolated[0] == 100.0
    assert interpolated[3] == 200.0
    assert interpolated[4] == 150.0
    # Interior unvoiced frames are linearly interpolated between the
    # surrounding voiced values, strictly between them.
    assert 100.0 < interpolated[1] < 200.0
    assert 100.0 < interpolated[2] < 200.0


def test_interpolate_unvoiced_all_zero_stays_unchanged():
    f0 = np.zeros(10, dtype=np.float32)
    interpolated = _interpolate_unvoiced(f0)
    assert np.array_equal(interpolated, f0)


def test_hz_to_semitones_monotonic_and_zero_at_reference():
    hz = np.array([50.0, 100.0, 200.0], dtype=np.float32)
    semitones = _hz_to_semitones(hz, reference_hz=100.0)
    assert semitones[0] < semitones[1] < semitones[2]
    assert abs(semitones[1] - 0.0) < 1e-4          # 100 Hz re 100 Hz -> 0 semitones


def test_suprasegmental_sequence_short_circuits_for_near_zero_valid_length():
    """valid_frames <= 1 must never even attempt to run parselmouth — the
    function's own documented fast path — so this is safe to test without
    real audio."""
    waveform = np.zeros(64000, dtype=np.float32)
    total_frames = 401
    result = extract_suprasegmental_sequence(waveform, sr=16000, valid_length=0,
                                             total_frames=total_frames)
    assert set(result) == {"f0_semitones", "voicing", "intensity_db"}
    for arr in result.values():
        assert arr.shape == (total_frames,)
        assert np.all(arr == 0.0)


def test_segmental_extra_sequence_short_circuits_for_near_zero_valid_length():
    waveform = np.zeros(64000, dtype=np.float32)
    total_frames = 401
    result = extract_segmental_extra_sequence(waveform, sr=16000, valid_length=0,
                                               total_frames=total_frames)
    assert set(result) == {"f1_hz", "f2_hz", "f3_hz", "hnr_db"}
    for arr in result.values():
        assert arr.shape == (total_frames,)
        assert np.all(arr == 0.0)


def test_suprasegmental_sequence_never_raises_on_garbage_audio():
    """Praat/parselmouth can fail on pathological input (pure noise, wrong
    scale) — the function's contract is to fall back to zeros, not raise,
    matching src.praat's existing utterance-level extractors' contract."""
    rng = np.random.default_rng(0)
    garbage = (rng.standard_normal(4000) * 1e6).astype(np.float32)   # absurd amplitude
    total_frames = 401
    result = extract_suprasegmental_sequence(garbage, sr=16000, valid_length=4000,
                                             total_frames=total_frames)
    for arr in result.values():
        assert arr.shape == (total_frames,)
        assert np.isfinite(arr).all()


if __name__ == "__main__":
    test_supra_padding_beyond_receptive_field_cannot_affect_embedding()
    test_supra_masked_pool_equals_manual_mean_over_valid_frames()
    test_supra_masking_actually_changes_the_result()
    test_interpolate_unvoiced_fills_only_the_gaps()
    test_interpolate_unvoiced_all_zero_stays_unchanged()
    test_hz_to_semitones_monotonic_and_zero_at_reference()
    test_suprasegmental_sequence_short_circuits_for_near_zero_valid_length()
    test_segmental_extra_sequence_short_circuits_for_near_zero_valid_length()
    test_suprasegmental_sequence_never_raises_on_garbage_audio()
    print("All suprasegmental masking/extraction tests passed.")
