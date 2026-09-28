"""
Three-branch gated-fusion severity architecture — the final design from the
one-shot architecture audit (see the plan's Part 2 for the full justification
of every component below; this module is Part 3's `src/models/gated_fusion.py`).

    Learned (wav2vec2+LoRA, 128D) --.
    Segmental (MFCC+formant+HNR CNN, 64D) --+-- gated fusion --> Z_unified (256D)
    Suprasegmental (F0/voicing/energy CNN, 64D) --'                  |
                                                          +-----------+-----------+
                                                          |                       |
                                                   CORAL severity head   speaker head (GRL)

Every branch is bottlenecked BEFORE fusion (config.LEARNED_EMBED_DIM /
SEGMENTAL_EMBED_DIM / SUPRA_EMBED_DIM), a learned gate weights each branch's
contribution (logged, not just concatenated), a cross-branch redundancy
penalty discourages the three from encoding the same information, a
gradient-reversal speaker head discourages Z_unified from encoding speaker
identity, and the severity head is ordinal (CORAL) rather than a plain
4-way softmax.

Exposes forward_features/forward/classifier like every other model in
src/training/models.py, so it plugs into src.training.engine.run_epoch's
embeddings-collection path unchanged. Its multi-term loss (ordinal + λ_comp
complementarity + λ_speaker adversarial) does NOT fit run_epoch's generic
`criterion(logits, labels)` call, so it additionally exposes training_step()
— see src.training.engine.run_epoch, which calls it when present instead of
the generic path, and falls back to the generic path for every other model
unchanged.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.losses import (GradientReversalLayer, coral_class_probs, coral_loss,
                        coral_rank_from_class_probs, redundancy_penalty)
from src.models.deep_pathway import DeepPathway
from src.models.segmental_pathway import SegmentalPathway
from src.models.suprasegmental_pathway import SuprasegmentalPathway

# Canonical branch order — every (B, 3) gate tensor and every 3-tuple of
# branch embeddings in this module follows it.
BRANCH_NAMES = ("learned", "segmental", "supra")
BRANCH_DIMS = {"learned": config.LEARNED_EMBED_DIM,
               "segmental": config.SEGMENTAL_EMBED_DIM,
               "supra": config.SUPRA_EMBED_DIM}


class CoralHead(nn.Module):
    """Ordinal severity head (Cao et al., 2020): a single shared linear
    projection plus one learned bias per rank threshold, so every threshold
    logit is `w . z + b_k` — the weight-sharing that gives CORAL its rank-
    consistency (see src.losses's module docstring)."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.shared = nn.Linear(in_dim, 1, bias=False)
        self.thresholds = nn.Parameter(torch.zeros(num_classes - 1))

    def threshold_logits(self, z: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) -> (B, num_classes - 1) raw per-threshold logits, for
        src.losses.coral_loss. NOT a probability and NOT the same tensor
        forward() returns."""
        return self.shared(z) + self.thresholds

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) -> (B, num_classes) log-class-probabilities. Feeding
        this back through softmax (as src.training.engine.run_epoch already
        does for every model) exactly reproduces coral_class_probs's output,
        so accuracy/F1/confusion-matrix/AUROC all work unmodified — see
        src.losses.coral_class_probs's docstring."""
        class_probs = coral_class_probs(self.threshold_logits(z))
        return torch.log(class_probs)


class GateNetwork(nn.Module):
    """Learns (g_learned, g_segmental, g_supra), softmax-normalized, from
    the concatenated (unweighted) branch embeddings — the mechanism that
    replaces flat concatenation/cross-attention with an inspectable,
    per-batch branch-reliance signal (architecture plan Part 2, Component 9).
    One gate per entry of `dims`, so the same module serves the two-branch
    ablations (see GatedFusionModel's `branches`)."""

    def __init__(self, dims: Sequence[int], hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sum(dims), hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(dims)),
        )

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        combined = torch.cat(embeddings, dim=1)
        return torch.softmax(self.net(combined), dim=1)          # (B, len(dims))


class SpeakerHead(nn.Module):
    """Adversarial speaker classifier behind a gradient-reversal layer — the
    single speaker-invariance mechanism (architecture plan Part 2,
    Component 10), applied to the FUSED representation so speaker identity
    carried by any branch (not only the learned one) is discouraged."""

    def __init__(self, in_dim: int, num_speakers: int, grl_lambda: float):
        super().__init__()
        self.grl = GradientReversalLayer(grl_lambda)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_speakers),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.grl(z))


class GatedFusionModel(nn.Module):
    """
    The one-shot run's primary model (severity task), and — through the
    constructor switches below — every controlled ablation of it.

    num_speakers: the current LOSO fold's TRAINING speaker count (varies per
    fold — src.training.runner.run_fold builds a fresh model per fold, same
    as every other model in this codebase, and sizes the speaker head to
    that fold's training speakers). The speaker head is discarded at
    inference; a different num_speakers per fold changes nothing about how
    Z_unified -> severity is used or compared across folds.

    Ablation switches (defaults = the full primary model, unchanged):
      branches               subset of BRANCH_NAMES to build. An absent branch
                             is never constructed (so an acoustic-only model
                             never loads wav2vec2) and comes back as None from
                             encode_branches.
      fusion                 "gated" (learned softmax gate, default) or
                             "concat" (plain concatenation, no gate).
      use_complementarity    add LAMBDA_COMP * cross-branch redundancy penalty.
      use_speaker_adversary  add LAMBDA_SPEAKER * GRL speaker-head loss.
    See src.training.models.SEVERITY_ABLATIONS for the named configurations.
    """

    def __init__(self, num_classes: int = 4, num_speakers: int = 1, use_lora: bool = True,
                 branches: Sequence[str] = BRANCH_NAMES, fusion: str = "gated",
                 use_complementarity: bool = True, use_speaker_adversary: bool = True,
                 gradient_checkpointing: Optional[bool] = None):
        super().__init__()
        unknown = set(branches) - set(BRANCH_NAMES)
        if unknown or not branches:
            raise ValueError(f"branches must be a non-empty subset of {BRANCH_NAMES}, got {branches}")
        if fusion not in ("gated", "concat"):
            raise ValueError(f"fusion must be 'gated' or 'concat', got {fusion!r}")
        self.num_classes = num_classes
        self.branches = tuple(b for b in BRANCH_NAMES if b in branches)
        self.fusion = fusion
        self.use_complementarity = use_complementarity
        self.use_speaker_adversary = use_speaker_adversary

        self.deep_pathway = None
        self.learned_projection = None
        self.segmental_pathway = None
        self.suprasegmental_pathway = None
        if "learned" in self.branches:
            # normalize_input=True: this branch alone gets the checkpoint-
            # compatible zero-mean/unit-variance waveform normalization
            # facebook/wav2vec2-base-960h's own feature extractor expects (see
            # DeepPathway's docstring) — per-utterance, so it introduces no
            # cross-fold statistics and cannot leak. Every other DeepPathway
            # consumer (src/training/models.py's legacy fusion models,
            # src/training/baseline.py's frozen-embedding sweep) keeps
            # normalize_input's default False, unchanged.
            self.deep_pathway = DeepPathway(use_lora=use_lora, normalize_input=True,
                                            gradient_checkpointing=gradient_checkpointing)
            self.learned_projection = nn.Sequential(
                nn.Linear(config.WAV2VEC_EMBED_DIM, config.LEARNED_EMBED_DIM),
                nn.LayerNorm(config.LEARNED_EMBED_DIM),
                nn.GELU(),
            )
        if "segmental" in self.branches:
            self.segmental_pathway = SegmentalPathway()
        if "supra" in self.branches:
            self.suprasegmental_pathway = SuprasegmentalPathway()

        # A single branch has nothing to weigh against, so it gets no gate
        # (its weight is identically 1); likewise concat fusion.
        self.gate = (GateNetwork(dims=[BRANCH_DIMS[b] for b in self.branches])
                     if fusion == "gated" and len(self.branches) > 1 else None)

        self.embed_dim = sum(BRANCH_DIMS[b] for b in self.branches)    # 256 for the full model
        self.classifier = CoralHead(self.embed_dim, num_classes)
        self.speaker_head = (SpeakerHead(self.embed_dim, max(num_speakers, 1), config.GRL_LAMBDA)
                             if use_speaker_adversary else None)

        self._last_gate_weights: Optional[torch.Tensor] = None   # for post-hoc gate logging

    def encode_branches(self, waveform: torch.Tensor, mfcc: torch.Tensor,
                        supra: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                        supra_valid_frames: Optional[torch.Tensor] = None
                        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """The bottlenecked branch embeddings, before gating — Z_learned
        (128D), Z_segmental (64D), Z_supra (64D), or None for a branch this
        model was built without. Exposed separately (not only via
        forward_features) so branch-ablation analysis (post-hoc, no
        retraining) can zero out one branch and re-run the gate/classifier
        on the others."""
        z_learned = z_segmental = z_supra = None
        if self.deep_pathway is not None:
            z_learned_raw = self.deep_pathway(waveform, attention_mask=attention_mask)  # (B, 768)
            z_learned = self.learned_projection(z_learned_raw)                          # (B, 128)
        if self.segmental_pathway is not None:
            z_segmental = self.segmental_pathway(mfcc, attention_mask=attention_mask)   # (B, 64)
        if self.suprasegmental_pathway is not None:
            z_supra = self.suprasegmental_pathway(supra, valid_frames=supra_valid_frames)  # (B, 64)
        return z_learned, z_segmental, z_supra

    def fuse(self, z_learned: Optional[torch.Tensor], z_segmental: Optional[torch.Tensor],
            z_supra: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Branch embeddings -> (Z_unified, gate weights). Split out from
        forward_features so branch ablation can call it directly with a
        zeroed branch without re-running that branch's (possibly expensive)
        encoder.

        Gate weights are always (B, 3) in BRANCH_NAMES order so every
        consumer (gate logging, embeddings files) reads one layout: 0 for a
        branch the model does not have, 1 for every present branch when there
        is no gate (a single branch, or concat fusion)."""
        by_name = {"learned": z_learned, "segmental": z_segmental, "supra": z_supra}
        present = [by_name[b] for b in self.branches]
        batch = present[0].shape[0]
        if self.gate is not None:
            weights = self.gate(*present)                                  # (B, n_present)
        else:
            weights = present[0].new_ones(batch, len(present))
        z_unified = torch.cat([weights[:, i:i + 1] * z for i, z in enumerate(present)], dim=1)

        gates = present[0].new_zeros(batch, len(BRANCH_NAMES))
        for i, name in enumerate(self.branches):
            gates[:, BRANCH_NAMES.index(name)] = weights[:, i]
        return z_unified, gates

    def forward_features(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                         praat: torch.Tensor = None,
                         attention_mask: Optional[torch.Tensor] = None,
                         deep_embedding: Optional[torch.Tensor] = None,
                         supra: torch.Tensor = None,
                         supra_valid_frames: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            waveform, mfcc, attention_mask: as every other model in this
                codebase (src/training/models.py).
            praat, deep_embedding: ignored — accepted only so this model
                shares the generic call signature src.training.engine.run_epoch
                uses for every model. Praat's utterance-level features are
                superseded here by the split segmental/suprasegmental
                branches; this model always runs with LoRA on, so no
                cacheable frozen embedding applies (see
                MODELS_WITH_CACHEABLE_FROZEN_EMBEDDING in
                src/training/models.py, which does not include this model).
            supra: (batch, 3, frames) suprasegmental input (see
                src.praat.extract_suprasegmental_sequence).
            supra_valid_frames: (batch,) pre-pool real-frame count on the
                temporal-preserving profile (distinct from attention_mask,
                which reflects the speech-focused profile's valid_length).
        Returns:
            (batch, embed_dim) Z_unified, pre-classification-head (256 for
            the full three-branch model).
        """
        z_learned, z_segmental, z_supra = self.encode_branches(
            waveform, mfcc, supra, attention_mask, supra_valid_frames)
        z_unified, gates = self.fuse(z_learned, z_segmental, z_supra)
        self._last_gate_weights = gates.detach()
        return z_unified

    def forward(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
               praat: torch.Tensor = None, attention_mask: Optional[torch.Tensor] = None,
               deep_embedding: Optional[torch.Tensor] = None, supra: torch.Tensor = None,
               supra_valid_frames: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(batch, num_classes) log-class-probabilities — see CoralHead.forward."""
        return self.classifier(self.forward_features(
            waveform, mfcc, praat, attention_mask, deep_embedding, supra, supra_valid_frames))

    def predict_labels(self, logits: torch.Tensor) -> torch.Tensor:
        """Ordinal decode for src.training.engine.run_epoch: the median of the
        CORAL distribution (the CORAL paper's threshold-count rule), NOT
        argmax — see src.losses.coral_rank_from_class_probs for why argmax
        starves the middle severity classes."""
        return coral_rank_from_class_probs(torch.softmax(logits.detach().float(), dim=1))

    def coral_threshold_diagnostics(self) -> Dict[str, object]:
        """The K-1 learned threshold biases and whether they are rank-ordered
        (non-increasing). Ordered biases are what make the median decode
        exactly equal to CORAL's threshold count; logged per fold."""
        biases = self.classifier.thresholds.detach().float().cpu().tolist()
        ordered = all(a >= b for a, b in zip(biases, biases[1:]))
        return {"coral_thresholds": [round(b, 4) for b in biases],
                "coral_thresholds_ordered": ordered}

    def _complementarity(self, z_learned, z_segmental, z_supra) -> torch.Tensor:
        """Redundancy penalty summed over every pair of PRESENT branches —
        identical to src.losses.total_complementarity_penalty for the full
        three-branch model."""
        present = [z for z in (z_learned, z_segmental, z_supra) if z is not None]
        total = present[0].new_zeros(())
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                total = total + redundancy_penalty(present[i], present[j])
        return total

    def training_step(self, waveform: torch.Tensor, mfcc: torch.Tensor,
                      supra: torch.Tensor, attention_mask: Optional[torch.Tensor],
                      labels: torch.Tensor, supra_valid_frames: Optional[torch.Tensor] = None,
                      speaker_index: Optional[torch.Tensor] = None,
                      class_weights: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        The engine hook (see src.training.engine.run_epoch): computes this
        model's full multi-term loss and returns (logits, loss, extras) —
        `logits` is metrics-compatible (softmax reproduces the CORAL class
        probabilities, see CoralHead; predict_labels decodes them), `loss`
        already sums every ENABLED term (no further criterion() call needed
        by the caller), and `extras` is a dict of scalars (gate weights,
        complementarity penalty, speaker-head accuracy) the engine averages
        and reports per epoch.

        The complementarity penalty is still reported (computed without
        gradient) when use_complementarity is False, so the ablations can
        compare how redundant their branches end up with and without it.
        """
        z_learned, z_segmental, z_supra = self.encode_branches(
            waveform, mfcc, supra, attention_mask, supra_valid_frames)
        z_unified, gates = self.fuse(z_learned, z_segmental, z_supra)
        self._last_gate_weights = gates.detach()

        threshold_logits = self.classifier.threshold_logits(z_unified)
        logits = torch.log(coral_class_probs(threshold_logits))

        ordinal_loss = coral_loss(threshold_logits, labels, self.num_classes, class_weights)
        total_loss = ordinal_loss

        comp_penalty = None
        if len(self.branches) > 1:
            if self.use_complementarity:
                comp_penalty = self._complementarity(z_learned, z_segmental, z_supra)
                total_loss = total_loss + config.LAMBDA_COMP * comp_penalty
            else:
                with torch.no_grad():
                    comp_penalty = self._complementarity(z_learned, z_segmental, z_supra)

        speaker_loss = None
        speaker_accuracy = float("nan")
        if self.speaker_head is not None and speaker_index is not None:
            speaker_logits = self.speaker_head(z_unified)
            speaker_loss = F.cross_entropy(speaker_logits, speaker_index)
            speaker_accuracy = (speaker_logits.argmax(dim=1) == speaker_index).float().mean().item()
            total_loss = total_loss + config.LAMBDA_SPEAKER * speaker_loss

        extras = {
            "gate_learned": gates[:, 0].mean().item(),
            "gate_segmental": gates[:, 1].mean().item(),
            "gate_supra": gates[:, 2].mean().item(),
            "complementarity_penalty": (comp_penalty.item() if comp_penalty is not None
                                        else float("nan")),
            "ordinal_loss": ordinal_loss.item(),
            "speaker_loss": float(speaker_loss.item()) if speaker_loss is not None else float("nan"),
            "speaker_accuracy": speaker_accuracy,
        }
        return logits, total_loss, extras

    def ablate(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
              attention_mask: Optional[torch.Tensor] = None, supra: torch.Tensor = None,
              supra_valid_frames: Optional[torch.Tensor] = None,
              drop_branch: Optional[str] = None) -> torch.Tensor:
        """
        Inference-time branch ablation (no retraining) — architecture plan
        Part 2, Component 15 / brief Section 16. drop_branch in
        {"learned", "segmental", "supra", None}: that branch's embedding is
        zeroed BEFORE the gate runs, so the gate itself re-normalizes over
        the remaining branches exactly as it would if that branch had never
        existed, rather than just masking its contribution after a gate
        computed with all branches present. Returns (batch, num_classes)
        log-class-probabilities, same contract as forward().
        """
        if drop_branch is not None and drop_branch not in BRANCH_NAMES:
            raise ValueError(f"Unknown drop_branch '{drop_branch}' — expected "
                             "'learned', 'segmental', 'supra', or None.")
        if drop_branch is not None and drop_branch not in self.branches:
            raise ValueError(f"drop_branch '{drop_branch}' is not a branch of this model "
                             f"(branches={self.branches}).")
        z_learned, z_segmental, z_supra = self.encode_branches(
            waveform, mfcc, supra, attention_mask, supra_valid_frames)
        if drop_branch == "learned":
            z_learned = torch.zeros_like(z_learned)
        elif drop_branch == "segmental":
            z_segmental = torch.zeros_like(z_segmental)
        elif drop_branch == "supra":
            z_supra = torch.zeros_like(z_supra)
        z_unified, _ = self.fuse(z_learned, z_segmental, z_supra)
        return self.classifier(z_unified)
