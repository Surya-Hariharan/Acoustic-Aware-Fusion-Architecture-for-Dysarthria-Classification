"""
Numeric checks for the cross-branch complementarity (redundancy) penalty
(src/losses.py): near-zero for uncorrelated embeddings, large and positive
for embeddings carrying identical linear structure, and the three-way sum
used by the model equals the sum of its three pairwise terms.

Run with: pytest tests/test_complementarity_loss.py -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import redundancy_penalty, total_complementarity_penalty


def test_redundancy_penalty_is_near_zero_for_independent_gaussian_noise():
    torch.manual_seed(0)
    batch = 256
    z_a = torch.randn(batch, 128)
    z_b = torch.randn(batch, 64)
    penalty = redundancy_penalty(z_a, z_b)
    # The MEAN squared cross-correlation entry under independence is ~1/batch
    # regardless of d_a/d_b (see redundancy_penalty's docstring for why this
    # is a mean, not the raw Barlow-Twins/VICReg sum) — a loose but real
    # upper bound, not zero exactly (finite-sample correlation is never
    # exactly zero).
    expected_scale = 1.0 / batch
    assert penalty.item() < expected_scale * 5


def test_redundancy_penalty_is_large_for_identical_linear_structure():
    # A larger batch keeps the independent case's penalty close to its
    # (batch-size-dependent) noise floor, so the comparison isolates real
    # shared structure rather than small-sample correlation noise.
    torch.manual_seed(1)
    batch = 4096
    shared = torch.randn(batch, 32)
    z_a = shared @ torch.randn(32, 128)     # z_a is a linear function of `shared`
    z_b = shared @ torch.randn(32, 64)      # z_b is a DIFFERENT linear function of the SAME `shared`
    independent_a = torch.randn(batch, 128)
    independent_b = torch.randn(batch, 64)

    correlated_penalty = redundancy_penalty(z_a, z_b)
    independent_penalty = redundancy_penalty(independent_a, independent_b)
    assert correlated_penalty.item() > independent_penalty.item() * 5


def test_redundancy_penalty_is_symmetric():
    torch.manual_seed(2)
    z_a = torch.randn(32, 16)
    z_b = torch.randn(32, 8)
    assert torch.isclose(redundancy_penalty(z_a, z_b), redundancy_penalty(z_b, z_a), atol=1e-4)


def test_redundancy_penalty_zero_for_batch_size_one():
    # Per-dimension std is undefined for a single sample — must degrade
    # gracefully to zero, not raise or produce NaN/Inf.
    z_a = torch.randn(1, 16)
    z_b = torch.randn(1, 8)
    penalty = redundancy_penalty(z_a, z_b)
    assert penalty.item() == 0.0


def test_total_complementarity_penalty_sums_the_three_pairs():
    torch.manual_seed(3)
    z_l = torch.randn(32, 128)
    z_s = torch.randn(32, 64)
    z_p = torch.randn(32, 64)
    total = total_complementarity_penalty(z_l, z_s, z_p)
    manual_sum = (redundancy_penalty(z_l, z_s) + redundancy_penalty(z_l, z_p)
                 + redundancy_penalty(z_s, z_p))
    assert torch.isclose(total, manual_sum, atol=1e-4)


def test_redundancy_penalty_is_differentiable():
    z_a = torch.randn(16, 8, requires_grad=True)
    z_b = torch.randn(16, 8, requires_grad=True)
    penalty = redundancy_penalty(z_a, z_b)
    penalty.backward()
    assert z_a.grad is not None and torch.isfinite(z_a.grad).all()
    assert z_b.grad is not None and torch.isfinite(z_b.grad).all()


def test_redundancy_penalty_normalizes_by_da_times_db():
    """
    Pins the exact contract requested for the three-branch architecture:
    the penalty is the MEAN squared cross-correlation entry — i.e. divided
    by d_a * d_b — not the raw Barlow-Twins/VICReg SUM. d_a != d_b here
    deliberately, so a regression to summing (or to normalizing by the
    wrong element count) is caught rather than masked by a square d_a == d_b
    case. The expected value is reproduced independently (the same
    standardize -> cross-correlate -> square steps, written out here rather
    than delegated to the function under test) so this test still catches a
    change to redundancy_penalty's internals, not just its call signature.
    """
    torch.manual_seed(4)
    batch, d_a, d_b = 5, 6, 3
    z_a = torch.randn(batch, d_a)
    z_b = torch.randn(batch, d_b)

    eps = 1e-5
    a = (z_a - z_a.mean(dim=0, keepdim=True)) / (z_a.std(dim=0, unbiased=False, keepdim=True) + eps)
    b = (z_b - z_b.mean(dim=0, keepdim=True)) / (z_b.std(dim=0, unbiased=False, keepdim=True) + eps)
    cross_corr = (a.transpose(0, 1) @ b) / batch                # (d_a, d_b)
    expected_sum = (cross_corr ** 2).sum()
    expected_mean = expected_sum / (d_a * d_b)                  # the D_A x D_B normalization

    penalty = redundancy_penalty(z_a, z_b)
    assert torch.isclose(penalty, expected_mean, atol=1e-6)
    # And explicitly NOT the un-normalized sum, so a reversion to summing
    # (rather than averaging) over the cross-correlation matrix fails loudly.
    assert not torch.isclose(penalty, expected_sum, atol=1e-6)


def test_complementarity_penalty_stable_at_realistic_branch_dims():
    """
    Numerical-stability check at the architecture's REAL branch dimensions
    and batch size (config.LEARNED_EMBED_DIM=128, config.SEGMENTAL_EMBED_DIM
    = config.SUPRA_EMBED_DIM=64, config.DEFAULT_BATCH_SIZE) — this is the
    test that licenses keeping config.LAMBDA_COMP fixed at 0.05 rather than
    tuning it against training-fold results (per the one-shot rule: lambda
    changes only in response to a demonstrated instability, never in
    response to fold performance). Checked against both a random-init-like
    case (three independent embeddings) and a deliberately adversarial case
    (all three branches sharing linear structure — the largest redundancy
    this penalty could plausibly see), so "no instability" is demonstrated
    here, not merely assumed.
    """
    from src import config

    torch.manual_seed(5)
    batch = config.DEFAULT_BATCH_SIZE
    d_l, d_s, d_p = config.LEARNED_EMBED_DIM, config.SEGMENTAL_EMBED_DIM, config.SUPRA_EMBED_DIM

    # Case 1: random-init-like — three independent embeddings.
    z_l = torch.randn(batch, d_l)
    z_s = torch.randn(batch, d_s)
    z_p = torch.randn(batch, d_p)
    weighted_random = total_complementarity_penalty(z_l, z_s, z_p) * config.LAMBDA_COMP
    assert torch.isfinite(weighted_random)
    # Nowhere near dominating a typical ~1-3 magnitude CORAL loss (see
    # src.losses.coral_loss / the GatedFusionModel smoke test that motivated
    # this normalization in the first place).
    assert weighted_random.item() < 1.0

    # Case 2: adversarial — all three branches are different linear
    # projections of the SAME underlying factors, the worst case for this
    # penalty's magnitude.
    shared = torch.randn(batch, 32)
    z_l_adv = shared @ torch.randn(32, d_l)
    z_s_adv = shared @ torch.randn(32, d_s)
    z_p_adv = shared @ torch.randn(32, d_p)
    weighted_adv = total_complementarity_penalty(z_l_adv, z_s_adv, z_p_adv) * config.LAMBDA_COMP
    assert torch.isfinite(weighted_adv)
    # Even in the worst case, the weighted penalty must stay within a modest
    # multiple of a typical ordinal-loss scale, not explode — this is what
    # "no instability" means operationally for this run.
    assert weighted_adv.item() < 5.0


if __name__ == "__main__":
    test_redundancy_penalty_is_near_zero_for_independent_gaussian_noise()
    test_redundancy_penalty_is_large_for_identical_linear_structure()
    test_redundancy_penalty_is_symmetric()
    test_redundancy_penalty_zero_for_batch_size_one()
    test_total_complementarity_penalty_sums_the_three_pairs()
    test_redundancy_penalty_is_differentiable()
    test_redundancy_penalty_normalizes_by_da_times_db()
    test_complementarity_penalty_stable_at_realistic_branch_dims()
    print("All complementarity-loss tests passed.")
