"""
Integration coverage for the real UASpeechDataset -> DataLoader -> training
engine -> GatedFusionModel data path (src/dataset.py, src/training/data.py,
src/training/engine.py, src/models/gated_fusion.py).

This is deliberately NOT a shape-only unit test like test_gated_fusion_shapes.py
(whose _dummy_batch() hand-builds a correctly-shaped 43-channel `mfcc` tensor
directly, bypassing the real dataset/engine wiring entirely). It exists
because that bypass is exactly what let a real wiring bug slip past every
other test: src/training/engine.py used to read batch["mfcc"] (the legacy
39-channel Acoustic Pathway tensor) and hand it to GatedFusionModel's
Segmental branch, which needs the 43-channel batch["segmental"] tensor
instead — a mismatch invisible to any test that never builds a real batch
from a real UASpeechDataset.

Also covers the fold-scoped, leakage-safe normalization added alongside that
fix (src.preprocessing.segmental_standardizer / suprasegmental_standardizer /
normalize_segmental / normalize_suprasegmental) and the checkpoint-compatible
Wav2Vec2 waveform normalization (src.models.deep_pathway.DeepPathway.
_zero_mean_unit_var_norm).

Requires the real M6 manifest + extracted audio (data/extracted/, per
config.MANIFEST_PATH) — skipped entirely if that data is not present on this
machine, since it is not checked into the repository.

Run with: pytest tests/test_three_branch_integration.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.dataset import UASpeechDataset
from src.models.deep_pathway import DeepPathway
from src.models.gated_fusion import GatedFusionModel
from src.models.segmental_pathway import SegmentalPathway
from src.preprocessing import (normalize_segmental, normalize_suprasegmental,
                               segmental_standardizer, suprasegmental_standardizer)
from src.training.data import build_speaker_label_map

pytestmark = pytest.mark.skipif(
    not Path(config.MANIFEST_PATH).exists(),
    reason="Requires the real M6 manifest + extracted UA-Speech audio "
           "(notebooks/01_data_pipeline.ipynb); not present on this machine.")


def _small_fold(held_out_speaker="M08", utterances_per_speaker=2):
    """A tiny, real LOSO-shaped fold: a handful of utterances per training
    speaker, plus a few held-out utterances from a speaker never in
    train_df — enough to exercise the real pipeline without pulling in the
    full ~21k-utterance manifest on every test run."""
    df = pd.read_csv(config.MANIFEST_PATH)
    dysarthric = df[df["Severity"] != "N/A (Control)"]
    train_df = (dysarthric[dysarthric["Speaker_ID"] != held_out_speaker]
                .groupby("Speaker_ID").head(utterances_per_speaker).reset_index(drop=True))
    test_df = (dysarthric[dysarthric["Speaker_ID"] == held_out_speaker]
              .head(utterances_per_speaker).reset_index(drop=True))
    assert len(train_df) > 0 and len(test_df) > 0, "fixture manifest slice came back empty"
    assert held_out_speaker not in set(train_df["Speaker_ID"])
    return train_df, test_df


def test_dataset_produces_43x401_segmental_and_3x401_supra():
    train_df, _ = _small_fold()
    ds = UASpeechDataset(train_df, include_three_branch=True)
    item = ds[0]
    assert tuple(item["segmental"].shape) == (config.SEGMENTAL_CHANNELS, 401) == (43, 401)
    assert tuple(item["supra"].shape) == (config.SUPRA_CHANNELS, 401) == (3, 401)
    assert not torch.isnan(item["segmental"]).any()
    assert not torch.isinf(item["segmental"]).any()


def test_training_batch_contains_the_three_model_inputs_with_expected_shapes():
    train_df, _ = _small_fold()
    speaker_map = build_speaker_label_map(train_df)
    ds = UASpeechDataset(train_df, include_three_branch=True, speaker_label_map=speaker_map)
    loader = DataLoader(ds, batch_size=len(train_df), shuffle=False)
    batch = next(iter(loader))

    for key in ("waveform", "mfcc", "segmental", "supra", "supra_valid_frames", "speaker_index"):
        assert key in batch, f"expected '{key}' in a three-branch batch"

    batch_size = len(train_df)
    assert tuple(batch["segmental"].shape) == (batch_size, 43, 401)
    assert tuple(batch["supra"].shape) == (batch_size, 3, 401)
    # The legacy 39-channel tensor (still present for AcousticPathway/other
    # models) must NOT be what reaches SegmentalPathway — see the engine.py
    # wiring test below.
    assert batch["mfcc"].shape[-2] == 39


def test_segmental_pathway_receives_43_channels_from_a_real_batch():
    """Reproduces src/training/engine.py's actual batch handling: the tensor
    fed to SegmentalPathway must be batch["segmental"] (43ch), not
    batch["mfcc"] (39ch, plus a leading dim SegmentalPathway never squeezes)."""
    train_df, _ = _small_fold()
    ds = UASpeechDataset(train_df, include_three_branch=True)
    loader = DataLoader(ds, batch_size=len(train_df), shuffle=False)
    batch = next(iter(loader))

    segmental_pathway_input = batch["segmental"] if "segmental" in batch else batch["mfcc"]
    model = SegmentalPathway()
    out = model(segmental_pathway_input)
    assert out.shape == (len(train_df), config.SEGMENTAL_EMBED_DIM)

    with pytest.raises(RuntimeError):
        model(batch["mfcc"])   # the old (buggy) wiring must still fail loudly


def test_gated_fusion_model_full_forward_on_a_real_dataset_batch():
    train_df, _ = _small_fold()
    speaker_map = build_speaker_label_map(train_df)
    ds = UASpeechDataset(train_df, include_three_branch=True, speaker_label_map=speaker_map)
    loader = DataLoader(ds, batch_size=len(train_df), shuffle=False)
    batch = next(iter(loader))

    waveform = batch["waveform"].squeeze(1)
    waveform_length = batch["waveform_length"]
    attention_mask = (torch.arange(waveform.shape[1])[None, :] < waveform_length[:, None])
    segmental_pathway_input = batch["segmental"] if "segmental" in batch else batch["mfcc"]

    model = GatedFusionModel(num_classes=4, num_speakers=len(speaker_map), use_lora=True).eval()
    with torch.no_grad():
        logits = model(waveform=waveform, mfcc=segmental_pathway_input, attention_mask=attention_mask,
                      supra=batch["supra"], supra_valid_frames=batch["supra_valid_frames"])
    assert logits.shape == (len(train_df), 4)
    assert torch.isfinite(logits).all()


def test_training_step_forward_and_backward_on_a_real_dataset_batch():
    """The complete training path (src/training/engine.py's real wiring,
    reproduced here): a real dataset batch through GatedFusionModel.
    training_step, with a backward pass reaching the LoRA adapters."""
    train_df, _ = _small_fold()
    speaker_map = build_speaker_label_map(train_df)
    ds = UASpeechDataset(train_df, include_three_branch=True, speaker_label_map=speaker_map)
    loader = DataLoader(ds, batch_size=len(train_df), shuffle=False)
    batch = next(iter(loader))

    waveform = batch["waveform"].squeeze(1)
    waveform_length = batch["waveform_length"]
    attention_mask = (torch.arange(waveform.shape[1])[None, :] < waveform_length[:, None])
    segmental_pathway_input = batch["segmental"] if "segmental" in batch else batch["mfcc"]

    model = GatedFusionModel(num_classes=4, num_speakers=len(speaker_map), use_lora=True)
    model.train()
    logits, loss, extras = model.training_step(
        waveform=waveform, mfcc=segmental_pathway_input, supra=batch["supra"],
        attention_mask=attention_mask, labels=batch["severity_label"],
        supra_valid_frames=batch["supra_valid_frames"], speaker_index=batch["speaker_index"],
        class_weights=torch.ones(4))

    assert logits.shape == (len(train_df), 4)
    assert torch.isfinite(loss)
    loss.backward()

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    assert len(lora_params) > 0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
              for p in lora_params), "no LoRA adapter received a nonzero gradient"


def test_normalization_statistics_never_touch_the_held_out_speaker():
    train_df, test_df = _small_fold()
    train_paths = set(train_df["Filepath"])
    test_paths = set(test_df["Filepath"])
    assert train_paths.isdisjoint(test_paths)

    seen_paths = []
    from src import preprocessing as preprocessing_module
    original = preprocessing_module.extract_segmental_features_cached

    def _spy(filepath):
        seen_paths.append(filepath)
        return original(filepath)

    preprocessing_module.extract_segmental_features_cached = _spy
    try:
        segmental_standardizer(train_df["Filepath"])
    finally:
        preprocessing_module.extract_segmental_features_cached = original

    assert set(seen_paths) == train_paths
    assert test_paths.isdisjoint(set(seen_paths)), (
        "segmental_standardizer touched a held-out speaker's file")


def test_normalization_statistics_derived_only_from_the_training_partition():
    """Statistics must be a pure function of the filepaths handed in — adding
    the held-out speaker's utterances to the input changes the result,
    proving nothing outside the given argument (e.g. the full manifest, a
    cached global) silently contributes."""
    train_df, test_df = _small_fold()
    train_only_stats = segmental_standardizer(train_df["Filepath"])
    train_plus_test_stats = segmental_standardizer(
        pd.concat([train_df["Filepath"], test_df["Filepath"]]))

    mean_a, _ = train_only_stats
    mean_b, _ = train_plus_test_stats
    assert not np.allclose(mean_a, mean_b), (
        "statistics were identical whether or not the held-out speaker's "
        "data was included — segmental_standardizer is not actually "
        "scoped to its filepaths argument")


def test_suprasegmental_normalization_preserves_the_voicing_mask():
    train_df, _ = _small_fold()
    supra_stats = suprasegmental_standardizer(train_df["Filepath"])
    ds_raw = UASpeechDataset(train_df, include_three_branch=True)
    ds_norm = UASpeechDataset(train_df, include_three_branch=True, supra_stats=supra_stats)

    raw_voicing = ds_raw[0]["supra"][1]
    normalized_voicing = ds_norm[0]["supra"][1]
    assert torch.equal(raw_voicing, normalized_voicing), (
        "the voicing channel must pass through normalize_suprasegmental unchanged")
    assert set(torch.unique(normalized_voicing).tolist()) <= {0.0, 1.0}


def test_segmental_normalization_matches_manual_zscore_on_valid_frames():
    train_df, _ = _small_fold()
    stats = segmental_standardizer(train_df["Filepath"])
    mean, std = stats

    from src.preprocessing import (extract_segmental_features_cached,
                                   load_and_preprocess_cached, mfcc_frame_count)
    filepath = train_df["Filepath"].iloc[0]
    raw = extract_segmental_features_cached(filepath)
    _, valid_length = load_and_preprocess_cached(filepath)
    valid_frames = min(mfcc_frame_count(valid_length), raw.shape[-1])

    normalized = normalize_segmental(raw, valid_frames, stats)
    mean_t = torch.as_tensor(mean).unsqueeze(-1)
    std_t = torch.as_tensor(std).unsqueeze(-1)
    expected_valid = (raw[:, :valid_frames] - mean_t) / std_t

    assert torch.allclose(normalized[:, :valid_frames], expected_valid, atol=1e-5)
    if valid_frames < normalized.shape[-1]:
        assert torch.equal(normalized[:, valid_frames:],
                           torch.zeros_like(normalized[:, valid_frames:]))


def test_wav2vec2_receives_checkpoint_compatible_normalized_waveform():
    """DeepPathway's normalize_input=True path must reproduce
    facebook/wav2vec2-base-960h's own Wav2Vec2FeatureExtractor
    (do_normalize=True) semantics: per-utterance zero-mean/unit-variance
    over the valid prefix, padded tail forced to exact zero."""
    torch.manual_seed(0)
    batch, samples = 3, 1000
    waveform = torch.randn(batch, samples) * 5.0 + 2.0
    lengths = torch.tensor([1000, 400, 0])
    attention_mask = (torch.arange(samples)[None, :] < lengths[:, None])

    normalized = DeepPathway._zero_mean_unit_var_norm(waveform, attention_mask)

    for i, length in enumerate(lengths.tolist()):
        if length <= 0:
            continue
        valid = waveform[i, :length]
        expected = (valid - valid.mean()) / torch.sqrt(valid.var(unbiased=False) + 1e-7)
        assert torch.allclose(normalized[i, :length], expected, atol=1e-4)
        if length < samples:
            assert torch.equal(normalized[i, length:], torch.zeros(samples - length))

    # GatedFusionModel opts its own DeepPathway instance into this path;
    # every other consumer (legacy fusion models, the frozen-embedding
    # baseline) keeps the default off, unchanged.
    assert GatedFusionModel(num_classes=4, num_speakers=1).deep_pathway.normalize_input is True
    assert DeepPathway(use_lora=True).normalize_input is False


if __name__ == "__main__":
    test_dataset_produces_43x401_segmental_and_3x401_supra()
    test_training_batch_contains_the_three_model_inputs_with_expected_shapes()
    test_segmental_pathway_receives_43_channels_from_a_real_batch()
    test_gated_fusion_model_full_forward_on_a_real_dataset_batch()
    test_training_step_forward_and_backward_on_a_real_dataset_batch()
    test_normalization_statistics_never_touch_the_held_out_speaker()
    test_normalization_statistics_derived_only_from_the_training_partition()
    test_suprasegmental_normalization_preserves_the_voicing_mask()
    test_segmental_normalization_matches_manual_zscore_on_valid_frames()
    test_wav2vec2_receives_checkpoint_compatible_normalized_waveform()
    print("All three-branch integration tests passed.")
