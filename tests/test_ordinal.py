"""
Numeric checks for the CORAL ordinal severity head (src/losses.py): the
class-probability decoding sums to 1 and is non-negative, the loss decreases
as predictions move toward the true rank, and coral_rank_predictions'
threshold-count decode is internally consistent with the sigmoid it reads.

Run with: pytest tests/test_ordinal.py -v
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import coral_class_probs, coral_loss, coral_rank_predictions, coral_targets


def test_coral_targets_shape_and_monotonicity():
    # rank 0 -> no threshold exceeded; rank 3 (of 4 classes) -> all 3 exceeded.
    labels = torch.tensor([0, 1, 2, 3])
    targets = coral_targets(labels, num_classes=4)
    assert targets.shape == (4, 3)
    assert torch.equal(targets[0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(targets[3], torch.tensor([1.0, 1.0, 1.0]))
    # Every row's targets are non-increasing... actually non-decreasing count of
    # 1s as rank increases: row i should have exactly i ones (i in 0..3, first 3 thresholds).
    assert targets[1].sum().item() == 1
    assert targets[2].sum().item() == 2


def test_coral_class_probs_sum_to_one_and_nonnegative():
    torch.manual_seed(0)
    threshold_logits = torch.randn(8, 3) * 3.0   # wide range, including near-degenerate cases
    probs = coral_class_probs(threshold_logits)
    assert probs.shape == (8, 4)
    assert torch.all(probs >= 0.0)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(8), atol=1e-4)


def test_coral_class_probs_recovers_confident_rank():
    # Extremely confident thresholds (large positive/negative logits) should
    # push essentially all probability mass onto the corresponding class.
    # Rank 0: every threshold strongly says "not exceeded" (large negative logits).
    confident_rank0 = torch.tensor([[-20.0, -20.0, -20.0]])
    probs0 = coral_class_probs(confident_rank0)
    assert probs0.argmax(dim=1).item() == 0
    assert probs0[0, 0].item() > 0.99

    # Rank 3 (top class): every threshold strongly exceeded (large positive logits).
    confident_rank3 = torch.tensor([[20.0, 20.0, 20.0]])
    probs3 = coral_class_probs(confident_rank3)
    assert probs3.argmax(dim=1).item() == 3
    assert probs3[0, 3].item() > 0.99


def test_coral_rank_predictions_matches_threshold_count():
    threshold_logits = torch.tensor([[5.0, 5.0, -5.0],     # 2 thresholds exceeded -> rank 2
                                     [-5.0, -5.0, -5.0]])   # 0 exceeded -> rank 0
    ranks = coral_rank_predictions(threshold_logits)
    assert ranks.tolist() == [2, 0]


def test_coral_loss_decreases_as_logits_move_toward_truth():
    labels = torch.tensor([3, 3, 3])          # true rank = top class for all
    wrong_logits = torch.full((3, 3), -10.0)   # predicts rank 0 confidently — maximally wrong
    right_logits = torch.full((3, 3), 10.0)    # predicts rank 3 confidently — correct
    loss_wrong = coral_loss(wrong_logits, labels, num_classes=4)
    loss_right = coral_loss(right_logits, labels, num_classes=4)
    assert loss_right.item() < loss_wrong.item()
    assert loss_right.item() < 0.01           # near-zero loss when confidently correct


def test_coral_loss_applies_class_weights():
    # Two samples with DIFFERENT per-sample loss magnitudes (row 0 is a
    # confident, nearly-correct prediction; row 1 is a confident, wrong one),
    # so re-weighting which sample dominates the mean actually changes the
    # result — a symmetric case (e.g. all-zero logits) would not, since every
    # per-threshold BCE term at p=0.5 is identical regardless of the target.
    labels = torch.tensor([0, 3])
    threshold_logits = torch.tensor([[-10.0, -10.0, -10.0],   # confidently predicts rank 0 (matches label 0)
                                     [-10.0, -10.0, -10.0]])  # confidently predicts rank 0 (label 3 is wrong)
    unweighted = coral_loss(threshold_logits, labels, num_classes=4)
    weights = torch.tensor([1.0, 1.0, 1.0, 100.0])   # heavily upweight the wrong (class-3) sample
    weighted = coral_loss(threshold_logits, labels, num_classes=4, class_weights=weights)
    # Upweighting the wrong sample must pull the loss up toward its (much
    # larger) individual loss value.
    assert weighted.item() > unweighted.item()


if __name__ == "__main__":
    test_coral_targets_shape_and_monotonicity()
    test_coral_class_probs_sum_to_one_and_nonnegative()
    test_coral_class_probs_recovers_confident_rank()
    test_coral_rank_predictions_matches_threshold_count()
    test_coral_loss_decreases_as_logits_move_toward_truth()
    test_coral_loss_applies_class_weights()
    print("All ordinal (CORAL) tests passed.")
