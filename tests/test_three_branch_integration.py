"""
Integration over the real data path: UASpeechDataset -> build_loaders ->
run_epoch -> GatedFusionModel, on a small slice of real UA-Speech utterances,
plus the fold-scoped normalization and the wav2vec2 input normalization.

Requires the real M6 manifest and extracted audio; skipped otherwise.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src import config
from src.dataset import UASpeechDataset
from src.models.deep_pathway import DeepPathway
from src.preprocessing import (normalize_segmental, segmental_standardizer,
                               suprasegmental_standardizer)
from src.training.data import build_loaders, build_speaker_label_map, shutdown_loaders
from src.training.engine import build_optimizer, run_epoch
from src.training.models import build_model

pytestmark = pytest.mark.skipif(
    not Path(config.MANIFEST_PATH).exists(),
    reason="Requires the real M6 manifest + extracted UA-Speech audio.")


def _small_fold(held_out_speaker="M08", utterances_per_speaker=2):
    """A tiny real LOSO-shaped fold: a few utterances per training speaker and
    a few from a held-out speaker never in train_df."""
    df = pd.read_csv(config.MANIFEST_PATH)
    dysarthric = df[df["Severity"] != "N/A (Control)"]
    train_df = (dysarthric[dysarthric["Speaker_ID"] != held_out_speaker]
                .groupby("Speaker_ID").head(utterances_per_speaker).reset_index(drop=True))
    test_df = (dysarthric[dysarthric["Speaker_ID"] == held_out_speaker]
               .head(utterances_per_speaker).reset_index(drop=True))
    assert len(train_df) > 0 and len(test_df) > 0
    return train_df, test_df


def test_dataset_items_carry_the_three_branch_inputs():
    train_df, _ = _small_fold()
    item = UASpeechDataset(train_df, speaker_label_map=build_speaker_label_map(train_df))[0]
    assert tuple(item["waveform"].shape) == (1, config.MAX_SAMPLES)
    assert tuple(item["segmental"].shape) == (config.SEGMENTAL_CHANNELS, 401) == (43, 401)
    assert tuple(item["supra"].shape) == (config.SUPRA_CHANNELS, 401) == (3, 401)
    assert torch.isfinite(item["segmental"]).all() and torch.isfinite(item["supra"]).all()
    assert 0 < int(item["supra_valid_frames"]) <= 401
    assert {"severity_label", "speaker_index", "speaker_id", "filename"} <= set(item)


def test_a_real_fold_trains_and_evaluates_through_the_engine():
    """build_loaders -> run_epoch(train) -> run_epoch(test, embeddings): the
    exact path run_fold takes, on real utterances."""
    train_df, test_df = _small_fold()
    speaker_map = build_speaker_label_map(train_df)
    train_loader, _, test_loader = build_loaders(
        train_df, test_df.iloc[:0], test_df, batch_size=8, pin_memory=False,
        speaker_label_map=speaker_map, num_workers=0)
    model = build_model("gated_fusion_three_branch", num_speakers=len(speaker_map))
    optimizer = build_optimizer(model, 1e-3, 1e-4, 1e-2)
    device = torch.device("cpu")
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    try:
        train = run_epoch(model, train_loader, device, optimizer=optimizer, amp_enabled=False)
        test = run_epoch(model, test_loader, device, amp_enabled=False, collect_embeddings=True)
    finally:
        shutdown_loaders()

    assert np.isfinite(train.loss) and np.isfinite(train.extras["speaker_loss"])
    assert any(not torch.equal(p, lora_before[n]) for n, p in model.named_parameters()
               if "lora_B" in n), "no LoRA adapter was updated"
    assert len(test.y_pred) == len(test_df) and set(test.speaker_ids) == {"M08"}
    assert np.isnan(test.extras["speaker_loss"])             # no speaker labels at test time
    assert test.embeddings["fused"].shape == (len(test_df), config.FUSED_EMBED_DIM)
    assert np.allclose(test.embeddings["gates"].sum(axis=1), 1.0, atol=1e-5)
    assert {"learned", "segmental", "supra"} <= set(test.embeddings)


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
    ds_raw = UASpeechDataset(train_df)
    ds_norm = UASpeechDataset(train_df, supra_stats=supra_stats)

    raw_voicing = ds_raw[0]["supra"][1]
    normalized_voicing = ds_norm[0]["supra"][1]
    assert torch.equal(raw_voicing, normalized_voicing), (
        "the voicing channel must pass through normalize_suprasegmental unchanged")
    assert set(torch.unique(normalized_voicing).tolist()) <= {0.0, 1.0}


def test_suprasegmental_normalization_does_not_fabricate_pitch_on_unvoiced_frames():
    """Regression test: normalize_suprasegmental must leave the F0 channel's
    explicit 'no pitch estimate' sentinel (exact 0, on every frame the
    voicing channel marks unvoiced) exactly at 0 after normalization,
    instead of shifting it to (0 - mean) / std. Also checks that the
    F0 standardization statistics themselves are not pulled toward the
    unvoiced sentinel (i.e. suprasegmental_standardizer excludes unvoiced
    frames from the F0 mean/std it computes)."""
    train_df, _ = _small_fold()
    supra_stats = suprasegmental_standardizer(train_df["Filepath"])
    mean, _ = supra_stats

    ds_raw = UASpeechDataset(train_df)
    ds_norm = UASpeechDataset(train_df, supra_stats=supra_stats)

    found_unvoiced_frame = False
    for i in range(len(train_df)):
        raw_item = ds_raw[i]["supra"]
        norm_item = ds_norm[i]["supra"]
        voicing = raw_item[1]
        unvoiced = voicing == 0
        if not unvoiced.any():
            continue
        found_unvoiced_frame = True
        f0_raw_unvoiced = raw_item[0][unvoiced]
        f0_norm_unvoiced = norm_item[0][unvoiced]
        assert torch.all(f0_raw_unvoiced == 0.0), (
            "fixture assumption broken: unvoiced frames should carry the raw 0 sentinel")
        assert torch.all(f0_norm_unvoiced == 0.0), (
            "normalize_suprasegmental must not fabricate a nonzero pseudo-pitch value "
            "on unvoiced frames — the F0 sentinel must survive normalization as exact 0")

    assert found_unvoiced_frame, (
        "fixture produced no unvoiced frames to exercise this regression test on — "
        "widen _small_fold()'s sample")
    # A voiced-only F0 mean should not be pulled toward the unvoiced-frame
    # sentinel of exactly 0 semitones re 100 Hz (a real speaker's voiced
    # pitch is essentially never that close to the reference frequency).
    assert abs(float(mean[0])) > 1e-6


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
    """DeepPathway's input normalization must reproduce
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
