"""
Three-branch gated-fusion severity model.

    Learned        wav2vec2 + LoRA, pooled, 768 -> 128 ----.
    Segmental      MFCC+formant+HNR CNN -> 64 ------------+-- softmax gate --> Z_unified (256)
    Suprasegmental F0/voicing/intensity CNN -> 64 --------'          |
                                                          +----------+----------+
                                                     CORAL head (4)     speaker head (GRL,
                                                                        training only)

Loss = class-weighted CORAL + LAMBDA_COMP * cross-branch redundancy
       + LAMBDA_SPEAKER * adversarial speaker cross-entropy.

The constructor switches build every controlled ablation of the same model
(src.training.models.SEVERITY_ABLATIONS).
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.losses import (GradientReversalLayer, complementarity_penalty, coral_class_probs,
                        coral_loss, coral_rank_from_class_probs)
from src.models.deep_pathway import DeepPathway
from src.models.segmental_pathway import SegmentalPathway
from src.models.suprasegmental_pathway import SuprasegmentalPathway

# Canonical branch order: every (B, 3) gate tensor follows it.
BRANCH_NAMES = ("learned", "segmental", "supra")
BRANCH_DIMS = {"learned": config.LEARNED_EMBED_DIM,
               "segmental": config.SEGMENTAL_EMBED_DIM,
               "supra": config.SUPRA_EMBED_DIM}


class CoralHead(nn.Module):
    """CORAL ordinal head (Cao et al., 2020): one shared projection plus K-1
    threshold biases, so every threshold logit is w.z + b_k (rank-consistent)."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.dropout = nn.Dropout(config.HEAD_DROPOUT)
        self.shared = nn.Linear(in_dim, 1, bias=False)
        self.thresholds = nn.Parameter(torch.zeros(num_classes - 1))

    def threshold_logits(self, z: torch.Tensor) -> torch.Tensor:
        """(B, K-1) raw logits of P(rank > k), for coral_loss."""
        return self.shared(self.dropout(z)) + self.thresholds

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, K) log class probabilities; softmax of this returns them exactly."""
        return torch.log(coral_class_probs(self.threshold_logits(z)))


class GateNetwork(nn.Module):
    """Softmax weights over the present branches, from their concatenated
    embeddings — logged per utterance, so branch reliance is inspectable."""

    def __init__(self, dims: Sequence[int], hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sum(dims), hidden), nn.ReLU(),
                                 nn.Linear(hidden, len(dims)))

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.net(torch.cat(embeddings, dim=1)), dim=1)
        floor = config.GATE_UNIFORM_FLOOR
        return floor / weights.shape[1] + (1.0 - floor) * weights


class SpeakerHead(nn.Module):
    """Speaker classifier behind a gradient-reversal layer, on the fused
    representation: it learns to identify speakers while pushing Z_unified to
    make that harder. Discarded at inference."""

    def __init__(self, in_dim: int, num_speakers: int, grl_lambda: float):
        super().__init__()
        self.grl = GradientReversalLayer(grl_lambda)
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, num_speakers))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.grl(z))


class GatedFusionModel(nn.Module):
    """
    num_speakers: this fold's training-speaker count (sizes the speaker head).

    Ablation switches (defaults = the full model):
      branches               subset of BRANCH_NAMES; an absent branch is never
                             built (an acoustic-only model never loads wav2vec2).
      fusion                 "gated" (softmax gate) or "concat" (no gate).
      use_complementarity    add the cross-branch redundancy penalty.
      use_speaker_adversary  add the gradient-reversal speaker loss.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, num_speakers: int = 1,
                 branches: Sequence[str] = BRANCH_NAMES, fusion: str = "gated",
                 use_complementarity: bool = True, use_speaker_adversary: bool = True,
                 gradient_checkpointing: Optional[bool] = None):
        super().__init__()
        if not branches or set(branches) - set(BRANCH_NAMES):
            raise ValueError(f"branches must be a non-empty subset of {BRANCH_NAMES}, got {branches}")
        if fusion not in ("gated", "concat"):
            raise ValueError(f"fusion must be 'gated' or 'concat', got {fusion!r}")
        self.num_classes = num_classes
        self.branches = tuple(b for b in BRANCH_NAMES if b in branches)
        self.use_complementarity = use_complementarity

        self.deep_pathway = self.learned_projection = None
        self.segmental_pathway = self.suprasegmental_pathway = None
        if "learned" in self.branches:
            self.deep_pathway = DeepPathway(gradient_checkpointing=gradient_checkpointing)
            self.learned_projection = nn.Sequential(
                nn.Linear(config.WAV2VEC_EMBED_DIM, config.LEARNED_EMBED_DIM),
                nn.LayerNorm(config.LEARNED_EMBED_DIM), nn.GELU())
        if "segmental" in self.branches:
            self.segmental_pathway = SegmentalPathway()
        if "supra" in self.branches:
            self.suprasegmental_pathway = SuprasegmentalPathway()

        # One branch, or concat fusion, has nothing to weigh: no gate.
        self.gate = (GateNetwork([BRANCH_DIMS[b] for b in self.branches])
                     if fusion == "gated" and len(self.branches) > 1 else None)
        self.embed_dim = sum(BRANCH_DIMS[b] for b in self.branches)
        self.classifier = CoralHead(self.embed_dim, num_classes)
        self.speaker_head = (SpeakerHead(self.embed_dim, max(num_speakers, 1), config.GRL_LAMBDA)
                             if use_speaker_adversary else None)

    def encode_branches(self, waveform: torch.Tensor, segmental: torch.Tensor,
                        supra: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                        supra_valid_frames: Optional[torch.Tensor] = None
                        ) -> Tuple[Optional[torch.Tensor], ...]:
        """(Z_learned, Z_segmental, Z_supra) before gating; None for a branch
        this model does not have."""
        z_learned = z_segmental = z_supra = None
        if self.deep_pathway is not None:
            z_learned = self.learned_projection(self.deep_pathway(waveform, attention_mask))
        if self.segmental_pathway is not None:
            z_segmental = self.segmental_pathway(segmental, attention_mask=attention_mask)
        if self.suprasegmental_pathway is not None:
            z_supra = self.suprasegmental_pathway(supra, valid_frames=supra_valid_frames)
        return z_learned, z_segmental, z_supra

    def fuse(self, z_learned, z_segmental, z_supra) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (Z_unified, gate weights (B, 3) in BRANCH_NAMES order; 0 for an
        absent branch, 1 for every present branch when there is no gate)."""
        by_name = {"learned": z_learned, "segmental": z_segmental, "supra": z_supra}
        present = [by_name[b] for b in self.branches]
        batch = present[0].shape[0]
        if self.training and len(present) > 1 and config.BRANCH_DROPOUT > 0:
            keep = torch.rand(batch, len(present), device=present[0].device) >= config.BRANCH_DROPOUT
            keep[torch.arange(batch), torch.randint(len(present), (batch,), device=keep.device)] = True
            present = [z * keep[:, i:i + 1].to(z.dtype) for i, z in enumerate(present)]
        weights = (self.gate(*present) if self.gate is not None
                   else present[0].new_ones(batch, len(present)))
        z_unified = torch.cat([weights[:, i:i + 1] * z for i, z in enumerate(present)], dim=1)
        gates = present[0].new_zeros(batch, len(BRANCH_NAMES))
        for i, name in enumerate(self.branches):
            gates[:, BRANCH_NAMES.index(name)] = weights[:, i]
        return z_unified, gates

    def forward(self, waveform: torch.Tensor, segmental: torch.Tensor, supra: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                supra_valid_frames: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, K) log class probabilities."""
        z_unified, _ = self.fuse(*self.encode_branches(waveform, segmental, supra,
                                                       attention_mask, supra_valid_frames))
        return self.classifier(z_unified)

    def predict_labels(self, logits: torch.Tensor) -> torch.Tensor:
        """Ordinal decode: the median of the CORAL distribution, not argmax
        (see src.losses.coral_rank_from_class_probs)."""
        return coral_rank_from_class_probs(torch.softmax(logits.detach().float(), dim=1))

    def coral_threshold_diagnostics(self) -> Dict[str, object]:
        """The K-1 threshold biases and whether they are rank-ordered."""
        biases = self.classifier.thresholds.detach().float().cpu().tolist()
        return {"coral_thresholds": [round(b, 4) for b in biases],
                "coral_thresholds_ordered": all(a >= b for a, b in zip(biases, biases[1:]))}

    def training_step(self, waveform: torch.Tensor, segmental: torch.Tensor,
                      supra: torch.Tensor, attention_mask: Optional[torch.Tensor],
                      labels: torch.Tensor, supra_valid_frames: Optional[torch.Tensor] = None,
                      speaker_index: Optional[torch.Tensor] = None,
                      class_weights: Optional[torch.Tensor] = None,
                      return_embeddings: bool = False):
        """-> (logits, loss, extras[, embeddings]).

        `loss` sums every enabled term. `extras` holds per-batch tensors (gate
        means, each loss term, speaker accuracy); they stay on the GPU so the
        caller syncs once per batch, not once per scalar. The redundancy
        penalty is reported even when it is not optimized, so ablations can
        compare how redundant their branches become. With return_embeddings,
        also returns {"fused", "gates", and each present branch}."""
        z_learned, z_segmental, z_supra = self.encode_branches(
            waveform, segmental, supra, attention_mask, supra_valid_frames)
        z_unified, gates = self.fuse(z_learned, z_segmental, z_supra)

        threshold_logits = self.classifier.threshold_logits(z_unified)
        logits = torch.log(coral_class_probs(threshold_logits))
        ordinal_loss = coral_loss(threshold_logits, labels, self.num_classes, class_weights)
        total_loss = ordinal_loss

        nan = z_unified.new_tensor(float("nan"))
        present = [z for z in (z_learned, z_segmental, z_supra) if z is not None]
        comp_penalty = nan
        if len(present) > 1:
            if self.use_complementarity:
                comp_penalty = complementarity_penalty(present)
                total_loss = total_loss + config.LAMBDA_COMP * comp_penalty
            else:
                with torch.no_grad():
                    comp_penalty = complementarity_penalty(present)

        speaker_loss = speaker_accuracy = nan
        if self.speaker_head is not None and speaker_index is not None:
            speaker_logits = self.speaker_head(z_unified)
            speaker_loss = F.cross_entropy(speaker_logits, speaker_index)
            speaker_accuracy = (speaker_logits.argmax(dim=1) == speaker_index).float().mean()
            total_loss = total_loss + config.LAMBDA_SPEAKER * speaker_loss

        gate_means = gates.detach().float().mean(dim=0)
        extras = {"gate_learned": gate_means[0], "gate_segmental": gate_means[1],
                  "gate_supra": gate_means[2],
                  "complementarity_penalty": comp_penalty.detach().float(),
                  "ordinal_loss": ordinal_loss.detach().float(),
                  "speaker_loss": speaker_loss.detach().float(),
                  "speaker_accuracy": speaker_accuracy.detach().float()}
        if not return_embeddings:
            return logits, total_loss, extras
        embeddings = {"fused": z_unified, "gates": gates}
        embeddings.update({name: z for name, z in zip(BRANCH_NAMES, (z_learned, z_segmental, z_supra))
                           if z is not None})
        return logits, total_loss, extras, embeddings

    def ablate(self, waveform: torch.Tensor, segmental: torch.Tensor, supra: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               supra_valid_frames: Optional[torch.Tensor] = None,
               drop_branch: Optional[str] = None) -> torch.Tensor:
        """Inference-time branch ablation: zero one branch's embedding BEFORE
        the gate, so the gate renormalizes over the rest. (B, K) log probs."""
        if drop_branch is not None and drop_branch not in self.branches:
            raise ValueError(f"drop_branch {drop_branch!r} is not a branch of this model "
                             f"{self.branches}.")
        embeddings = list(self.encode_branches(waveform, segmental, supra, attention_mask,
                                               supra_valid_frames))
        if drop_branch is not None:
            index = BRANCH_NAMES.index(drop_branch)
            embeddings[index] = torch.zeros_like(embeddings[index])
        z_unified, _ = self.fuse(*embeddings)
        return self.classifier(z_unified)
