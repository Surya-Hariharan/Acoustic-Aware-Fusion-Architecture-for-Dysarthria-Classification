"""
Loss-function building blocks for the three-branch gated-fusion severity
architecture (src/models/gated_fusion.py): ordinal CORAL classification,
cross-branch complementarity (redundancy) regularization, and the gradient-
reversal layer behind the speaker-invariance objective.

Kept separate from src/training/metrics.py (evaluation) and src/training/
engine.py (the training loop that sums these terms) because all three are
pure, stateless tensor functions with no dependency on the training loop's
bookkeeping — easy to unit-test in isolation (see tests/test_ordinal.py,
tests/test_complementarity_loss.py).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CORAL ordinal severity head (Cao, Mirjalili & Raschka, 2020 — "Rank
# Consistent Ordinal Regression for Neural Networks", arXiv:1901.07884).
#
# Severity is an ordered label (Very Low < Low < Mid < High), not a bag of
# four unrelated classes — a Very-Low/High confusion is a worse mistake than
# a Very-Low/Low one, and a plain softmax head cannot express that. CORAL
# reduces K-class ordinal regression to K-1 binary "is the true rank greater
# than threshold k" tasks that share a single weight vector, differing only
# in a per-threshold bias — this is what gives the predicted probabilities
# their rank-consistency, without a hand-added monotonicity constraint.
# ---------------------------------------------------------------------------
def coral_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B,) int rank labels -> (B, num_classes - 1) binary targets, where
    target[:, k] = 1 iff the true rank is greater than threshold k."""
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def coral_loss(threshold_logits: torch.Tensor, labels: torch.Tensor,
               num_classes: int, class_weights: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
    """
    Args:
        threshold_logits: (B, num_classes - 1) raw logits, one per rank
            threshold — NOT yet a probability and NOT the same tensor as the
            (B, num_classes) class-probability output the rest of the
            pipeline (metrics, confusion matrix) reads; see coral_class_probs.
        labels: (B,) int true rank in [0, num_classes - 1].
        class_weights: optional (num_classes,) inverse-frequency weights
            (the same tensor src.training.data.compute_class_weights already
            produces for CrossEntropyLoss) — applied per-sample by indexing
            with the sample's true class, so the ordinal loss inherits the
            project's existing class-imbalance handling instead of a second,
            inconsistent weighting scheme.
    Returns:
        Scalar loss: importance-weighted mean of the K-1 per-threshold
        binary cross-entropies, summed over thresholds per sample (the
        formulation in the CORAL paper) then averaged over the batch.
    """
    targets = coral_targets(labels, num_classes)
    per_threshold = F.binary_cross_entropy_with_logits(
        threshold_logits, targets, reduction="none")          # (B, K-1)
    per_sample = per_threshold.sum(dim=1)                      # (B,)
    if class_weights is not None:
        weights = class_weights[labels]
        return (per_sample * weights).sum() / weights.sum().clamp(min=1e-8)
    return per_sample.mean()


def coral_class_probs(threshold_logits: torch.Tensor) -> torch.Tensor:
    """
    (B, num_classes - 1) threshold logits -> (B, num_classes) per-class
    probabilities, by differencing the cumulative P(rank > k) curve:
    P(class = k) = P(rank > k-1) - P(rank > k), with P(rank > -1) := 1 and
    P(rank > K-1) := 0 at the ends.

    This is what lets the rest of the pipeline (src.training.engine.run_epoch,
    src.training.metrics.compute_metrics) treat the CORAL head exactly like
    any other classifier: log(class_probs) fed back in as "logits" reproduces
    class_probs under softmax (the differences are non-negative and sum to
    1 by construction, so no information is lost), so accuracy/F1/confusion-
    matrix/AUROC all work unmodified — only the LOSS (coral_loss above, on
    the raw threshold_logits) needs special handling, not prediction.

    Clamped to a small positive floor before renormalizing: floating-point
    slack can make a difference very slightly negative when two adjacent
    thresholds' sigmoids are nearly equal, which would otherwise make the
    downstream log() produce -inf.
    """
    probs_gt = torch.sigmoid(threshold_logits)                 # (B, K-1) P(rank > k)
    batch = probs_gt.shape[0]
    ones = probs_gt.new_ones(batch, 1)
    zeros = probs_gt.new_zeros(batch, 1)
    padded = torch.cat([ones, probs_gt, zeros], dim=1)         # (B, K+1)
    class_probs = padded[:, :-1] - padded[:, 1:]                # (B, K)
    class_probs = class_probs.clamp(min=1e-6)
    return class_probs / class_probs.sum(dim=1, keepdim=True)


def coral_rank_predictions(threshold_logits: torch.Tensor) -> torch.Tensor:
    """(B, num_classes - 1) threshold logits -> (B,) predicted rank, via the
    CORAL paper's own decode rule: the number of thresholds whose predicted
    probability exceeds 0.5. Used only as a diagnostic cross-check against
    the argmax(class_probs) prediction the rest of the pipeline reports —
    the two agree except in rare, mild non-monotonicity."""
    return (torch.sigmoid(threshold_logits) > 0.5).sum(dim=1)


# ---------------------------------------------------------------------------
# Complementarity (cross-branch redundancy) penalty — Barlow-Twins/VICReg
# lineage cross-covariance term, applied between every pair of the three
# branch embeddings instead of between two augmented views of one input.
#
# Chosen over a discriminator/contrastive scheme specifically because it is
# stable at small batch sizes, needs no negative sampling, and is a single,
# well-understood mechanism appropriate for a one-shot run with no room to
# debug a fragile adversarial term (see the architecture plan's Part 2,
# Component 8).
# ---------------------------------------------------------------------------
def redundancy_penalty(z_a: torch.Tensor, z_b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Args:
        z_a: (B, d_a), z_b: (B, d_b) — two branch embeddings for the same
            batch of utterances (arbitrary, possibly different, dimensions).
    Returns:
        Scalar: MEAN (not sum) of squared entries of the (d_a, d_b) cross-
        correlation matrix between the two batch-standardized embeddings.
        Zero iff every dimension of z_a is linearly uncorrelated (over the
        batch) with every dimension of z_b; grows as the branches encode
        shared linear structure. Requires batch size > 1 (per-dimension std
        is undefined for a single sample).

        MEAN, not the raw Barlow-Twins/VICReg sum, deliberately: this
        penalty is summed over three branch PAIRS of different
        dimensionality (128x64, 128x64, 64x64 — see
        total_complementarity_penalty) and combined with a single fixed
        λ_comp. A sum-of-squared-entries penalty scales with d_a * d_b
        regardless of true redundancy — even literally independent
        embeddings produce a "noise floor" proportional to d_a * d_b / batch
        — which would make λ_comp's effective strength depend on embedding
        size rather than on how redundant the branches actually are, and
        would let this term dominate the total loss by construction rather
        than by evidence (caught by this project's own smoke test: at
        d_a=128, d_b=64, batch=4, the summed penalty was ~150x the ordinal
        loss for a freshly-initialized, effectively-random model). The mean
        keeps the penalty's scale roughly O(1/batch) under independence
        regardless of d_a/d_b, so λ_comp means the same thing for every pair.
    """
    batch = z_a.shape[0]
    if batch < 2:
        return z_a.new_zeros(())
    a = (z_a - z_a.mean(dim=0, keepdim=True)) / (z_a.std(dim=0, unbiased=False, keepdim=True) + eps)
    b = (z_b - z_b.mean(dim=0, keepdim=True)) / (z_b.std(dim=0, unbiased=False, keepdim=True) + eps)
    cross_corr = (a.transpose(0, 1) @ b) / batch               # (d_a, d_b)
    return (cross_corr ** 2).mean()


def total_complementarity_penalty(z_learned: torch.Tensor, z_segmental: torch.Tensor,
                                  z_supra: torch.Tensor) -> torch.Tensor:
    """Sum of the redundancy penalty over all three branch pairs
    (learned-segmental, learned-supra, segmental-supra) — L_comp in the
    architecture plan's total loss."""
    return (redundancy_penalty(z_learned, z_segmental)
            + redundancy_penalty(z_learned, z_supra)
            + redundancy_penalty(z_segmental, z_supra))


# ---------------------------------------------------------------------------
# Gradient-reversal layer (Ganin & Lempitsky, 2015 — DANN). Identity on the
# forward pass; negates (and scales) the gradient on the backward pass, so a
# classifier attached downstream of this layer trains normally on its own
# loss while every layer UPSTREAM of it is pushed to make the input harder
# for that classifier to solve. Used here to make the fused representation
# harder to attribute to a speaker (see gated_fusion.py) — the single,
# well-established speaker-invariance mechanism chosen for this one-shot run
# (architecture plan Part 2, Component 10).
# ---------------------------------------------------------------------------
class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)
