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

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.losses import (GradientReversalLayer, coral_class_probs, coral_loss,
                        total_complementarity_penalty)
from src.models.deep_pathway import DeepPathway
from src.models.segmental_pathway import SegmentalPathway
from src.models.suprasegmental_pathway import SuprasegmentalPathway


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
    per-batch branch-reliance signal (architecture plan Part 2, Component 9)."""

    def __init__(self, dims: Tuple[int, int, int], hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sum(dims), hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, z_learned: torch.Tensor, z_segmental: torch.Tensor,
               z_supra: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z_learned, z_segmental, z_supra], dim=1)
        return torch.softmax(self.net(combined), dim=1)          # (B, 3)


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
    The one-shot run's only trained model (severity task).

    num_speakers: the current LOSO fold's TRAINING speaker count (varies per
    fold — src.training.runner.run_fold builds a fresh model per fold, same
    as every other model in this codebase, and sizes the speaker head to
    that fold's training speakers). The speaker head is discarded at
    inference; a different num_speakers per fold changes nothing about how
    Z_unified -> severity is used or compared across folds.
    """

    def __init__(self, num_classes: int = 4, num_speakers: int = 1, use_lora: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # normalize_input=True: this branch alone gets the checkpoint-
        # compatible zero-mean/unit-variance waveform normalization
        # facebook/wav2vec2-base-960h's own feature extractor expects (see
        # DeepPathway's docstring) — per-utterance, so it introduces no
        # cross-fold statistics and cannot leak. Every other DeepPathway
        # consumer (src/training/models.py's legacy fusion models,
        # src/training/baseline.py's frozen-embedding sweep) keeps
        # normalize_input's default False, unchanged.
        self.deep_pathway = DeepPathway(use_lora=use_lora, normalize_input=True)
        self.learned_projection = nn.Sequential(
            nn.Linear(config.WAV2VEC_EMBED_DIM, config.LEARNED_EMBED_DIM),
            nn.LayerNorm(config.LEARNED_EMBED_DIM),
            nn.GELU(),
        )
        self.segmental_pathway = SegmentalPathway()
        self.suprasegmental_pathway = SuprasegmentalPathway()

        self.gate = GateNetwork(dims=(config.LEARNED_EMBED_DIM,
                                      config.SEGMENTAL_EMBED_DIM,
                                      config.SUPRA_EMBED_DIM))

        self.embed_dim = config.FUSED_EMBED_DIM                  # 256
        self.classifier = CoralHead(self.embed_dim, num_classes)
        self.speaker_head = SpeakerHead(self.embed_dim, max(num_speakers, 1), config.GRL_LAMBDA)

        self._last_gate_weights: Optional[torch.Tensor] = None   # for post-hoc gate logging

    def encode_branches(self, waveform: torch.Tensor, mfcc: torch.Tensor,
                        supra: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                        supra_valid_frames: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three bottlenecked branch embeddings, before gating —
        Z_learned (128D), Z_segmental (64D), Z_supra (64D). Exposed
        separately (not only via forward_features) so branch-ablation
        analysis (post-hoc, no retraining) can zero out one branch and
        re-run the gate/classifier on the other two."""
        z_learned_raw = self.deep_pathway(waveform, attention_mask=attention_mask)     # (B, 768)
        z_learned = self.learned_projection(z_learned_raw)                             # (B, 128)
        z_segmental = self.segmental_pathway(mfcc, attention_mask=attention_mask)      # (B, 64)
        z_supra = self.suprasegmental_pathway(supra, valid_frames=supra_valid_frames)  # (B, 64)
        return z_learned, z_segmental, z_supra

    def fuse(self, z_learned: torch.Tensor, z_segmental: torch.Tensor,
            z_supra: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Branch embeddings -> (Z_unified, gate weights). Split out from
        forward_features so branch ablation can call it directly with a
        zeroed branch without re-running that branch's (possibly expensive)
        encoder."""
        gates = self.gate(z_learned, z_segmental, z_supra)                 # (B, 3)
        z_unified = torch.cat([
            gates[:, 0:1] * z_learned,
            gates[:, 1:2] * z_segmental,
            gates[:, 2:3] * z_supra,
        ], dim=1)
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
            (batch, 256) Z_unified, pre-classification-head.
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

    def training_step(self, waveform: torch.Tensor, mfcc: torch.Tensor,
                      supra: torch.Tensor, attention_mask: Optional[torch.Tensor],
                      labels: torch.Tensor, supra_valid_frames: Optional[torch.Tensor] = None,
                      speaker_index: Optional[torch.Tensor] = None,
                      class_weights: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        The engine hook (see src.training.engine.run_epoch): computes this
        model's full multi-term loss and returns (logits, loss, extras) —
        `logits` is metrics-compatible (softmax/argmax reproduce the CORAL
        class probabilities, see CoralHead), `loss` already sums every term
        (no further criterion() call needed by the caller), and `extras` is
        a dict of scalars (gate weights, complementarity penalty, speaker-
        head accuracy) the engine averages and reports per epoch.
        """
        z_learned, z_segmental, z_supra = self.encode_branches(
            waveform, mfcc, supra, attention_mask, supra_valid_frames)
        z_unified, gates = self.fuse(z_learned, z_segmental, z_supra)
        self._last_gate_weights = gates.detach()

        threshold_logits = self.classifier.threshold_logits(z_unified)
        logits = torch.log(coral_class_probs(threshold_logits))

        ordinal_loss = coral_loss(threshold_logits, labels, self.num_classes, class_weights)
        comp_penalty = total_complementarity_penalty(z_learned, z_segmental, z_supra)

        speaker_loss = z_unified.new_zeros(())
        speaker_accuracy = float("nan")
        if speaker_index is not None:
            speaker_logits = self.speaker_head(z_unified)
            speaker_loss = F.cross_entropy(speaker_logits, speaker_index)
            speaker_accuracy = (speaker_logits.argmax(dim=1) == speaker_index).float().mean().item()

        total_loss = (ordinal_loss + config.LAMBDA_COMP * comp_penalty
                     + config.LAMBDA_SPEAKER * speaker_loss)

        extras = {
            "gate_learned": gates[:, 0].mean().item(),
            "gate_segmental": gates[:, 1].mean().item(),
            "gate_supra": gates[:, 2].mean().item(),
            "complementarity_penalty": comp_penalty.item(),
            "ordinal_loss": ordinal_loss.item(),
            "speaker_loss": float(speaker_loss.item()) if speaker_index is not None else float("nan"),
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
        the two remaining branches exactly as it would if that branch had
        never existed, rather than just masking its contribution after a
        gate computed with all three present. Returns (batch, num_classes)
        log-class-probabilities, same contract as forward().
        """
        z_learned, z_segmental, z_supra = self.encode_branches(
            waveform, mfcc, supra, attention_mask, supra_valid_frames)
        if drop_branch == "learned":
            z_learned = torch.zeros_like(z_learned)
        elif drop_branch == "segmental":
            z_segmental = torch.zeros_like(z_segmental)
        elif drop_branch == "supra":
            z_supra = torch.zeros_like(z_supra)
        elif drop_branch is not None:
            raise ValueError(f"Unknown drop_branch '{drop_branch}' — expected "
                             "'learned', 'segmental', 'supra', or None.")
        z_unified, _ = self.fuse(z_learned, z_segmental, z_supra)
        return self.classifier(z_unified)
