"""Gate floor, branch dropout and sub-epoch validation chunks."""

import torch

from src import config
from src.models.gated_fusion import GateNetwork, GatedFusionModel
from src.training.runner import _TrainingChunks


def test_gate_never_collapses_below_the_uniform_floor():
    gate = GateNetwork([8, 4, 4])
    with torch.no_grad():
        gate.net[-1].bias.copy_(torch.tensor([50.0, -50.0, -50.0]))   # fully saturated softmax
    weights = gate(torch.randn(5, 8), torch.randn(5, 4), torch.randn(5, 4))
    assert torch.allclose(weights.sum(dim=1), torch.ones(5), atol=1e-5)
    assert float(weights.min()) >= config.GATE_UNIFORM_FLOOR / 3 - 1e-6


def test_branch_dropout_is_training_only_and_keeps_one_branch(monkeypatch):
    monkeypatch.setattr(config, "BRANCH_DROPOUT", 0.9)
    model = GatedFusionModel(num_classes=4, branches=("segmental", "supra"),
                             use_complementarity=False, use_speaker_adversary=False)
    z = [torch.ones(64, 64), torch.ones(64, 64)]
    model.eval()
    fused_eval, _ = model.fuse(None, *z)
    assert float(fused_eval.abs().sum(dim=1).min()) > 0
    model.train()
    fused_train, _ = model.fuse(None, *z)
    assert float(fused_train.abs().sum(dim=1).min()) > 0          # never an all-zero row
    assert float(fused_train.abs().sum()) < float(fused_eval.abs().sum())


def test_training_chunks_cover_the_pass_and_restart():
    loader = list(range(10))
    chunks = _TrainingChunks(loader, 3)
    assert len(chunks) == 4
    seen = [x for _ in range(3) for x in chunks]
    assert seen[:10] == loader and len(seen) == 12 and seen[10:] == [0, 1]
