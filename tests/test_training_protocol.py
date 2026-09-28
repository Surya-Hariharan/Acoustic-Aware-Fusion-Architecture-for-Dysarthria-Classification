"""
Protocol-level guarantees of the training pipeline that the audited 11-fold
Kaggle run exposed:

  * validation is speaker-disjoint from training (and from the test speaker),
    with every severity class keeping >= 2 training speakers in every fold;
  * severity is decoded as the median of the CORAL distribution, not argmax —
    argmax starves the middle classes when thresholds are close;
  * fold coverage is derived from the configured folds, never hard-coded;
  * compact checkpoints round-trip every trainable weight;
  * every named ablation builds and trains a step.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src import config
from src.losses import coral_class_probs, coral_rank_from_class_probs, coral_rank_predictions
from src.splits import iter_severity_loso_folds
from src.training.data import speaker_disjoint_train_val_split
from src.training.reporting import (FOLD_CACHED, FOLD_COMPLETED, FOLD_FAILED,
                                    FOLD_INTERRUPTED, FOLD_SKIPPED_DEADLINE, run_coverage)


def _severity_manifest(utterances_per_speaker: int = 4) -> pd.DataFrame:
    rows = []
    for speaker in config.DYSARTHRIC_IDS:
        for i in range(utterances_per_speaker):
            rows.append({"Speaker_ID": speaker, "Severity": config.SEVERITY_MAP[speaker],
                         "Filename": f"{speaker}_B1_UW{i}_M6.wav"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Speaker-disjoint validation
# ---------------------------------------------------------------------------
def test_validation_is_speaker_disjoint_in_every_severity_fold():
    df = _severity_manifest()
    for fold_id, train_df, test_df in iter_severity_loso_folds(df):
        fold_train, fold_val = speaker_disjoint_train_val_split(
            train_df, "Severity", seed=config.DEFAULT_SEED, fold_id=fold_id)
        train_spk, val_spk = set(fold_train["Speaker_ID"]), set(fold_val["Speaker_ID"])
        test_spk = set(test_df["Speaker_ID"])
        assert not train_spk & val_spk, f"{fold_id}: val speakers also in train"
        assert not test_spk & (train_spk | val_spk), f"{fold_id}: test speaker leaked"
        assert len(train_spk) + len(val_spk) + len(test_spk) == len(config.DYSARTHRIC_IDS)
        assert 3 <= len(val_spk) <= 4
        per_class = fold_train.groupby("Severity")["Speaker_ID"].nunique()
        assert per_class.reindex(config.SEVERITY_CLASS_NAMES).min() >= 2, fold_id
        # At most one validation speaker per class.
        assert fold_val.groupby("Severity")["Speaker_ID"].nunique().max() == 1


def test_validation_split_is_deterministic_per_fold():
    df = _severity_manifest()
    _, train_df, _ = next(iter(iter_severity_loso_folds(df)))
    a = speaker_disjoint_train_val_split(train_df, "Severity", seed=42, fold_id="M01")[1]
    b = speaker_disjoint_train_val_split(train_df, "Severity", seed=42, fold_id="M01")[1]
    assert sorted(a["Speaker_ID"].unique()) == sorted(b["Speaker_ID"].unique())


# ---------------------------------------------------------------------------
# Ordinal decoding
# ---------------------------------------------------------------------------
def test_median_decode_equals_coral_threshold_count_for_ordered_thresholds():
    torch.manual_seed(0)
    score = torch.randn(512, 1) * 3
    biases = torch.tensor([1.0, 0.2, -0.9])                       # rank-ordered
    threshold_logits = score + biases
    median = coral_rank_from_class_probs(coral_class_probs(threshold_logits))
    assert torch.equal(median, coral_rank_predictions(threshold_logits))


def test_argmax_starves_middle_classes_that_median_decoding_recovers():
    # Cumulative P(rank > k) = 0.6 / 0.55 / 0.5 - epsilon: the distribution's
    # median is class 2 (Mid) but most of the mass sits on the extremes.
    cumulative = torch.tensor([[0.60, 0.55, 0.49]])
    class_probs = coral_class_probs(torch.logit(cumulative))
    assert int(class_probs.argmax(dim=1)) in (0, 3)
    assert int(coral_rank_from_class_probs(class_probs)) == 2


# ---------------------------------------------------------------------------
# Coverage reporting
# ---------------------------------------------------------------------------
def test_run_coverage_counts_come_from_the_configured_folds():
    folds = [f"S{i}" for i in range(15)]
    status = {f: FOLD_COMPLETED for f in folds[:9]}
    status.update({folds[9]: FOLD_CACHED, folds[10]: FOLD_FAILED,
                   folds[11]: FOLD_INTERRUPTED, folds[12]: FOLD_SKIPPED_DEADLINE,
                   folds[13]: FOLD_SKIPPED_DEADLINE})
    coverage = run_coverage(folds, status)
    assert coverage["expected"] == 15 and coverage["completed"] == 10
    assert coverage["status"] == "PARTIAL"
    assert coverage["failed_folds"] == ["S10"]
    assert coverage["skipped_folds"] == ["S11", "S12", "S13"]
    assert coverage["missing_folds"] == ["S10", "S11", "S12", "S13", "S14"]
    assert coverage["completion_pct"] == pytest.approx(66.7)

    assert run_coverage(folds[:3], {f: FOLD_COMPLETED for f in folds[:3]})["status"] == "COMPLETE"
    assert run_coverage(folds[:3], {folds[0]: FOLD_FAILED})["status"] == "FAILED"


# ---------------------------------------------------------------------------
# Compact checkpoints
# ---------------------------------------------------------------------------
def test_compact_checkpoint_round_trips_trainable_weights(tmp_path):
    from src.training.checkpoint import load_checkpoint, save_checkpoint

    def make():
        torch.manual_seed(1)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.BatchNorm1d(8),
                                    torch.nn.Linear(8, 2))
        model[0].weight.requires_grad_(False)                     # a "frozen backbone" weight
        return model

    model = make()
    with torch.no_grad():
        model[2].weight.add_(1.0)
        model[1].running_mean.add_(0.5)
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, optimizer, scheduler, scaler, epoch=3, monitored_value=0.5)

    saved = torch.load(path, weights_only=False)
    assert "0.weight" not in saved["model_state"]                 # frozen param not stored
    restored = make()
    load_checkpoint(path, restored)
    for (name, a), (_, b) in zip(model.state_dict().items(), restored.state_dict().items()):
        assert torch.equal(a, b), name


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------
def _dummy_batch(batch_size: int = 4):
    frames = 401
    return dict(
        waveform=torch.randn(batch_size, config.MAX_SAMPLES) * 0.1,
        mfcc=torch.randn(batch_size, config.SEGMENTAL_CHANNELS, frames),
        supra=torch.randn(batch_size, config.SUPRA_CHANNELS, frames),
        attention_mask=torch.ones(batch_size, config.MAX_SAMPLES, dtype=torch.bool),
        labels=torch.tensor([0, 1, 2, 3])[:batch_size],
        supra_valid_frames=torch.full((batch_size,), frames),
        speaker_index=torch.tensor([0, 1, 0, 1])[:batch_size])


def test_acoustic_only_ablation_never_builds_wav2vec2_and_trains():
    from src.training.models import build_model

    model = build_model("ab2_acoustic_only", num_classes=4, num_speakers=2)
    assert model.deep_pathway is None and model.speaker_head is None
    logits, loss, extras = model.training_step(**_dummy_batch())
    loss.backward()
    assert logits.shape == (4, 4) and torch.isfinite(loss)
    assert extras["gate_learned"] == 0.0
    assert np.isfinite(extras["complementarity_penalty"])         # reported, not optimized
    preds = model.predict_labels(logits)
    assert preds.shape == (4,) and preds.min() >= 0 and preds.max() <= 3


@pytest.mark.parametrize("name", ["ab1_wav2vec2_only", "ab3_wav2vec2_acoustic_concat",
                                  "ab4_wav2vec2_segmental", "ab5_wav2vec2_suprasegmental",
                                  "ab6_full_fusion", "ab7_full_complementarity",
                                  "ab8_full_speaker_grl"])
def test_every_wav2vec2_ablation_builds_and_steps(name):
    from src.training.models import SEVERITY_ABLATIONS, build_model

    model = build_model(name, num_classes=4, num_speakers=2)
    switches = SEVERITY_ABLATIONS[name]
    assert model.branches == tuple(b for b in ("learned", "segmental", "supra")
                                   if b in switches["branches"])
    assert (model.speaker_head is not None) == switches.get("use_speaker_adversary", False)
    _, loss, extras = model.training_step(**_dummy_batch(2))
    loss.backward()
    assert torch.isfinite(loss)
    if switches.get("use_speaker_adversary"):
        assert np.isfinite(extras["speaker_loss"])
    else:
        assert np.isnan(extras["speaker_loss"])
