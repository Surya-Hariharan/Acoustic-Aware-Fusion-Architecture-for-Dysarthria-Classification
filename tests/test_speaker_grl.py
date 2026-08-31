"""
Focused unit test for the gradient-reversal layer (src/losses.py) in
isolation from the full GatedFusionModel — the speaker-invariance mechanism
(architecture plan Part 2, Component 10).

Run with: pytest tests/test_speaker_grl.py -v
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import GradientReversalLayer


def test_grl_forward_is_identity():
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    grl = GradientReversalLayer(lambda_=1.0)
    out = grl(x)
    assert torch.equal(out, x)


def test_grl_backward_negates_gradient():
    torch.manual_seed(1)
    x = torch.randn(8, 16, requires_grad=True)
    grl = GradientReversalLayer(lambda_=1.0)
    out = grl(x)
    upstream_grad = torch.randn_like(out)
    out.backward(upstream_grad)
    assert torch.allclose(x.grad, -upstream_grad, atol=1e-6)


def test_grl_backward_scales_by_lambda():
    torch.manual_seed(2)
    x = torch.randn(4, 8, requires_grad=True)
    grl = GradientReversalLayer(lambda_=0.37)
    out = grl(x)
    upstream_grad = torch.randn_like(out)
    out.backward(upstream_grad)
    assert torch.allclose(x.grad, -0.37 * upstream_grad, atol=1e-6)


def test_grl_zero_lambda_blocks_gradient_entirely():
    x = torch.randn(4, 8, requires_grad=True)
    grl = GradientReversalLayer(lambda_=0.0)
    out = grl(x)
    out.backward(torch.randn_like(out))
    assert torch.allclose(x.grad, torch.zeros_like(x), atol=1e-6)


def test_grl_inside_a_downstream_classifier_still_trains_normally():
    """The downstream classifier's OWN weights must still receive their
    ordinary (non-reversed) gradient — GRL only reverses what flows further
    upstream, past the point it's inserted."""
    torch.manual_seed(3)
    grl = GradientReversalLayer(lambda_=1.0)
    head = nn.Linear(8, 3)
    x = torch.randn(5, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0, 1])

    logits = head(grl(x))
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert not torch.allclose(head.weight.grad, torch.zeros_like(head.weight.grad))
    # x's gradient is the REVERSED version of what a plain (non-GRL) chain
    # would have produced — recompute the plain-chain gradient and compare.
    x2 = x.detach().clone().requires_grad_(True)
    head2 = nn.Linear(8, 3)
    head2.load_state_dict(head.state_dict())
    # Reset head2's grad state (already trained-on-once head has no grad yet
    # since it's a fresh copy) and recompute without GRL for comparison.
    logits2 = head2(x2)
    loss2 = nn.functional.cross_entropy(logits2, labels)
    loss2.backward()
    assert torch.allclose(x.grad, -x2.grad, atol=1e-5)


if __name__ == "__main__":
    test_grl_forward_is_identity()
    test_grl_backward_negates_gradient()
    test_grl_backward_scales_by_lambda()
    test_grl_zero_lambda_blocks_gradient_entirely()
    test_grl_inside_a_downstream_classifier_still_trains_normally()
    print("All speaker-GRL tests passed.")
