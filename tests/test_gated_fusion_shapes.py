"""
End-to-end shape/gradient checks for GatedFusionModel (src/models/gated_fusion.py) —
formalizes the ad hoc scratchpad verification run during development into permanent
test coverage: 128D/64D/64D branch bottlenecks, 256D fusion, gate softmax sums to
1, forward()/forward_features()+classifier() consistency, a full backward pass
reaches the LoRA adapters, and inference-time branch ablation (`ablate()`) actually
changes the output per branch.

Requires transformers/peft and a locally cached (or downloadable)
facebook/wav2vec2-base-960h checkpoint — DeepPathway.__init__ always loads it, so
every test here builds one real GatedFusionModel. Not a large model, but not
instant either; kept in one module so the weight load happens once conceptually per
test process (each test still builds its own model instance, matching every other
model in this codebase's per-fold-fresh-model convention — see
GatedFusionModel's docstring on why a fresh instance per fold is fine here).

Run with: pytest tests/test_gated_fusion_shapes.py -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.models.gated_fusion import GatedFusionModel


def _dummy_batch(batch_size=4, device="cpu"):
    total_frames = config.MAX_SAMPLES // config.MEL_KWARGS["hop_length"] + 1
    waveform = torch.zeros(batch_size, config.MAX_SAMPLES, device=device)
    mfcc = torch.randn(batch_size, config.SEGMENTAL_CHANNELS, total_frames, device=device)
    supra = torch.randn(batch_size, config.SUPRA_CHANNELS, total_frames, device=device)
    attention_mask = torch.ones(batch_size, config.MAX_SAMPLES, dtype=torch.bool, device=device)
    supra_valid_frames = torch.full((batch_size,), total_frames, dtype=torch.long, device=device)
    return waveform, mfcc, supra, attention_mask, supra_valid_frames, total_frames


def test_branch_bottleneck_dimensions_match_config():
    model = GatedFusionModel(num_classes=4, num_speakers=3, use_lora=True).eval()
    backbone = model.deep_pathway.wav2vec.base_model.model
    assert not backbone.config.apply_spec_augment
    assert backbone.config.mask_time_prob == backbone.config.mask_feature_prob == 0.0
    assert not hasattr(backbone, "masked_spec_embed")
    waveform, mfcc, supra, attention_mask, supra_valid_frames, _ = _dummy_batch()
    with torch.no_grad():
        z_l, z_s, z_p = model.encode_branches(waveform, mfcc, supra, attention_mask, supra_valid_frames)
    assert z_l.shape == (4, config.LEARNED_EMBED_DIM) == (4, 128)
    assert z_s.shape == (4, config.SEGMENTAL_EMBED_DIM) == (4, 64)
    assert z_p.shape == (4, config.SUPRA_EMBED_DIM) == (4, 64)


def test_fused_dimension_is_256_and_gates_sum_to_one():
    model = GatedFusionModel(num_classes=4, num_speakers=3, use_lora=True).eval()
    waveform, mfcc, supra, attention_mask, supra_valid_frames, _ = _dummy_batch()
    with torch.no_grad():
        z_l, z_s, z_p = model.encode_branches(waveform, mfcc, supra, attention_mask, supra_valid_frames)
        z_unified, gates = model.fuse(z_l, z_s, z_p)
    assert z_unified.shape == (4, config.FUSED_EMBED_DIM) == (4, 256)
    assert gates.shape == (4, 3)
    assert torch.all(gates >= 0.0)
    assert torch.allclose(gates.sum(dim=1), torch.ones(4), atol=1e-5)


def test_forward_produces_valid_class_probabilities():
    model = GatedFusionModel(num_classes=4, num_speakers=3, use_lora=True).eval()
    waveform, mfcc, supra, attention_mask, supra_valid_frames, _ = _dummy_batch()
    with torch.no_grad():
        logits = model(waveform=waveform, mfcc=mfcc, attention_mask=attention_mask,
                       supra=supra, supra_valid_frames=supra_valid_frames)
    assert logits.shape == (4, 4)
    assert torch.isfinite(logits).all()
    probs = torch.softmax(logits, dim=1)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-4)


def test_forward_features_plus_classifier_matches_forward():
    model = GatedFusionModel(num_classes=4, num_speakers=3, use_lora=True).eval()
    waveform, mfcc, supra, attention_mask, supra_valid_frames, _ = _dummy_batch()
    with torch.no_grad():
        logits = model(waveform=waveform, mfcc=mfcc, attention_mask=attention_mask,
                       supra=supra, supra_valid_frames=supra_valid_frames)
        features = model.forward_features(waveform=waveform, mfcc=mfcc, attention_mask=attention_mask,
                                          supra=supra, supra_valid_frames=supra_valid_frames)
        logits_via_classifier = model.classifier(features)
    assert features.shape == (4, config.FUSED_EMBED_DIM)
    assert torch.allclose(logits, logits_via_classifier, atol=1e-5)


def test_training_step_backward_pass_reaches_lora_adapters():
    model = GatedFusionModel(num_classes=4, num_speakers=5, use_lora=True)
    model.train()
    waveform, mfcc, supra, attention_mask, supra_valid_frames, _ = _dummy_batch(batch_size=6)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    speaker_index = torch.tensor([0, 1, 2, 3, 4, 0])
    class_weights = torch.ones(4)

    logits, loss, extras = model.training_step(
        waveform=waveform, mfcc=mfcc, supra=supra, attention_mask=attention_mask,
        labels=labels, supra_valid_frames=supra_valid_frames, speaker_index=speaker_index,
        class_weights=class_weights)

    assert logits.shape == (6, 4)
    assert torch.isfinite(loss)
    assert set(extras) == {"gate_learned", "gate_segmental", "gate_supra",
                           "complementarity_penalty", "ordinal_loss",
                           "speaker_loss", "speaker_accuracy"}
    loss.backward()

    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    assert len(lora_params) > 0, "expected LoRA adapter parameters to be trainable"
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
              for p in lora_params), "no LoRA adapter received a nonzero gradient"


def test_ablate_each_branch_changes_the_output():
    model = GatedFusionModel(num_classes=4, num_speakers=3, use_lora=True).eval()
    waveform, mfcc, supra, attention_mask, supra_valid_frames, _ = _dummy_batch()
    with torch.no_grad():
        full = model(waveform=waveform, mfcc=mfcc, attention_mask=attention_mask,
                     supra=supra, supra_valid_frames=supra_valid_frames)
        for branch in ("learned", "segmental", "supra"):
            ablated = model.ablate(waveform=waveform, mfcc=mfcc, attention_mask=attention_mask,
                                   supra=supra, supra_valid_frames=supra_valid_frames,
                                   drop_branch=branch)
            assert ablated.shape == full.shape
            assert not torch.allclose(full, ablated), f"ablating '{branch}' had no effect"


def test_ablate_invalid_branch_name_raises():
    model = GatedFusionModel(num_classes=4, num_speakers=3, use_lora=True).eval()
    waveform, mfcc, supra, attention_mask, supra_valid_frames, _ = _dummy_batch()
    try:
        model.ablate(waveform=waveform, mfcc=mfcc, attention_mask=attention_mask,
                     supra=supra, supra_valid_frames=supra_valid_frames,
                     drop_branch="not_a_real_branch")
        assert False, "expected ValueError on an unknown branch name"
    except ValueError:
        pass


if __name__ == "__main__":
    test_branch_bottleneck_dimensions_match_config()
    test_fused_dimension_is_256_and_gates_sum_to_one()
    test_forward_produces_valid_class_probabilities()
    test_forward_features_plus_classifier_matches_forward()
    test_training_step_backward_pass_reaches_lora_adapters()
    test_ablate_each_branch_changes_the_output()
    test_ablate_invalid_branch_name_raises()
    print("All GatedFusionModel shape/gradient tests passed.")
