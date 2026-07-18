"""
Phase 6 - Attention-based fusion. The project's novel contribution.

FusionModel (Phase 1) fuses the two pathways by *concatenating* their pooled
embeddings: a fixed, per-utterance-identical 768+128 vector in which the deep
and acoustic evidence are weighted only by whatever the classifier's first Linear
layer happens to learn, once, for every utterance alike.

That is the thing to improve on. A mildly dysarthric speaker's pathology may live
in a brief moment of vocal-fold instability that MFCCs capture crisply, while a
severe speaker's may be spread across the whole utterance in a way wav2vec's
learned representation carries better. Concatenation cannot express that: it
applies one weighting to every input.

Cross-attention can. Each pathway *queries* the other and pulls out the frames
that are relevant to it, so the fusion is computed per utterance rather than
fixed at training time.

WHY SEQUENCES, NOT THE POOLED VECTORS
Attention over a single key is arithmetically the identity - softmax over one
element is 1.0, so the "attended" output is just that element. Cross-attending
the two *pooled* embeddings would therefore be an expensive no-op. Real
cross-attention needs the frames the pooling throws away, which is why both
pathways grew a forward_sequence() (deep: ~199 x 768, acoustic: ~100 x 128).
Those live in different dimensionalities, so each is projected into a shared
FUSION_ATTN_DIM space before attention - queries and keys must agree on a
dimension.

CONTRACT WITH THE TRAINING ENGINE
Every model here exposes forward_features(waveform, mfcc, praat) and a submodule
literally named `classifier` - src/training/engine.py calls
`model.classifier(model.forward_features(...))` directly when collecting
embeddings. Keeping the wav2vec submodule under `self.deep_pathway` also
preserves build_optimizer's backbone/head learning-rate split, which keys on the
string "wav2vec" appearing in a parameter's name.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src import config
from src.models.acoustic_pathway import AcousticPathway
from src.models.deep_pathway import DeepPathway
from src.praat import FEATURE_COLUMNS


class CrossAttentionBlock(nn.Module):
    """
    One stream attends to another: pre-norm multi-head cross-attention with a
    residual, then a residual feed-forward block. Standard transformer decoder
    geometry, minus self-attention (the streams already carry their own context -
    wav2vec from its transformer, the MFCC CNN from its receptive field).

    No positional encoding is added. wav2vec2 already injects convolutional
    positional embeddings into its frames, and both streams are mean-pooled after
    attention, so absolute token order contributes little to the fused vector -
    adding a positional code here would be parameters spent for no signal.
    """

    def __init__(self, dim: int = config.FUSION_ATTN_DIM,
                 num_heads: int = config.FUSION_ATTN_HEADS,
                 dropout: float = config.FUSION_ATTN_DROPOUT):
        super().__init__()
        self.norm_query = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query:   (B, Tq, dim) the stream doing the attending.
            context: (B, Tc, dim) the stream being attended to.
        Returns:
            ((B, Tq, dim) attended query, (B, Tq, Tc) attention weights or None)
        """
        attended, weights = self.attention(
            self.norm_query(query), self.norm_context(context), self.norm_context(context),
            need_weights=need_weights, average_attn_weights=True)
        x = query + attended                       # residual 1
        x = x + self.ffn(self.norm_ffn(x))         # residual 2
        return x, weights


class AttentionFusionModel(nn.Module):
    """
    Ablation Model E: bidirectional cross-attention fusion.

        deep frames    (B, ~199, 768) --proj--> (B, ~199, 256) --.
                                                                  |-- deep attends to acoustic
        acoustic frames (B, ~100, 128) --proj--> (B, ~100, 256) --'
                                                                  '-- acoustic attends to deep

        mean-pool each attended stream -> concat -> (B, 512) -> classifier

    Bidirectional, not one-directional: making only the deep stream attend to the
    acoustic one would privilege wav2vec as the "real" representation and demote
    the MFCC pathway to a lookup table. Running it both ways lets each pathway
    reweight itself in light of the other, which is the claim the architecture is
    actually making.

    num_classes = 2 for detection, 4 for severity.
    """

    def __init__(self, num_classes: int = 2, use_lora: bool = True):
        super().__init__()
        self.deep_pathway = DeepPathway(use_lora=use_lora)
        self.acoustic_pathway = AcousticPathway()

        dim = config.FUSION_ATTN_DIM
        self.deep_proj = nn.Linear(config.WAV2VEC_EMBED_DIM, dim)       # 768 -> 256
        self.acoustic_proj = nn.Linear(config.ACOUSTIC_EMBED_DIM, dim)  # 128 -> 256

        self.deep_from_acoustic = CrossAttentionBlock(dim)
        self.acoustic_from_deep = CrossAttentionBlock(dim)

        self.embed_dim = dim * 2                                        # 512
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _project(self, waveform: torch.Tensor, mfcc: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        deep = self.deep_proj(self.deep_pathway.forward_sequence(waveform))
        acoustic = self.acoustic_proj(self.acoustic_pathway.forward_sequence(mfcc))
        return deep, acoustic

    def forward_features(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                         praat: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) raw audio.
            mfcc:     (batch, 39, frames) acoustic features.
            praat:    ignored — accepted so every model shares one call signature.
        Returns:
            (batch, 512) attention-fused embedding, pre-classification-head.
        """
        deep, acoustic = self._project(waveform, mfcc)
        deep_attended, _ = self.deep_from_acoustic(deep, acoustic)
        acoustic_attended, _ = self.acoustic_from_deep(acoustic, deep)
        return torch.cat([deep_attended.mean(dim=1), acoustic_attended.mean(dim=1)], dim=1)

    def forward(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                praat: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(self.forward_features(waveform, mfcc, praat))

    @torch.no_grad()
    def attention_weights(self, waveform: torch.Tensor, mfcc: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The two attention maps, for interpretability.

        Returns (deep_over_acoustic, acoustic_over_deep) of shape
        (B, ~199, ~100) and (B, ~100, ~199): which acoustic frames each wav2vec
        frame drew on, and vice versa. This is what makes the fusion inspectable
        rather than merely better - and it is the groundwork for ROADMAP Phase 6's
        deferred explainability item.
        """
        deep, acoustic = self._project(waveform, mfcc)
        _, deep_over_acoustic = self.deep_from_acoustic(deep, acoustic, need_weights=True)
        _, acoustic_over_deep = self.acoustic_from_deep(acoustic, deep, need_weights=True)
        return deep_over_acoustic, acoustic_over_deep


class AttentionFusionPraatModel(nn.Module):
    """
    Ablation Model F: Model E plus a third pathway carrying Phase 4's handcrafted
    Praat features (F0, jitter, shimmer, HNR, CPPS, formants, intensity, rhythm).

    The two learned pathways see only the audio. The Praat features are the
    clinically-named measurements a speech pathologist would actually reach for,
    and they are *not* recoverable from a 4-second VAD-trimmed window - they are
    computed from the original audio (see src/praat.py). So they carry genuinely
    independent evidence, which is the whole argument for a third pathway rather
    than a richer head.

    The features are encoded into a single 256-dim token (dimension fixed by
    `num_praat_features`, currently `len(FEATURE_COLUMNS)`) and appended to the
    context of *both* cross-attention blocks:

        deep     attends over [acoustic frames ; praat token]
        acoustic attends over [deep frames     ; praat token]

    so either learned pathway can pull on the handcrafted evidence when it helps.
    The praat token is also concatenated directly into the final embedding, so its
    signal reaches the classifier even if attention learns to ignore it.

        concat(pooled_deep, pooled_acoustic, praat_vector) -> (B, 768) -> classifier
    """

    def __init__(self, num_classes: int = 2, use_lora: bool = True,
                 num_praat_features: int = len(FEATURE_COLUMNS)):
        super().__init__()
        self.deep_pathway = DeepPathway(use_lora=use_lora)
        self.acoustic_pathway = AcousticPathway()

        dim = config.FUSION_ATTN_DIM
        praat_dim = config.PRAAT_EMBED_DIM
        self.num_praat_features = num_praat_features

        self.deep_proj = nn.Linear(config.WAV2VEC_EMBED_DIM, dim)
        self.acoustic_proj = nn.Linear(config.ACOUSTIC_EMBED_DIM, dim)

        # The features arrive already standardized per fold (see
        # src.praat.praat_standardizer), so this is a plain MLP - no BatchNorm,
        # which would be unstable at the small batch sizes this trains at.
        self.praat_encoder = nn.Sequential(
            nn.Linear(num_praat_features, praat_dim),
            nn.LayerNorm(praat_dim),
            nn.GELU(),
            nn.Dropout(config.FUSION_ATTN_DROPOUT),
            nn.Linear(praat_dim, dim),
        )

        self.deep_from_context = CrossAttentionBlock(dim)
        self.acoustic_from_context = CrossAttentionBlock(dim)

        self.embed_dim = dim * 3                                        # 768
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward_features(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                         praat: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) raw audio.
            mfcc:     (batch, 39, frames) acoustic features.
            praat:    (batch, num_praat_features) standardized Praat features.
        Returns:
            (batch, 768) tri-modal attention-fused embedding.
        """
        if praat is None:
            raise ValueError(
                "AttentionFusionPraatModel requires the `praat` tensor. The "
                "DataLoader supplies it only when UASpeechDataset is constructed "
                "with a praat_table — see src.training.data.build_loaders, which "
                "does this automatically for model='attention_fusion_praat'."
            )

        deep = self.deep_proj(self.deep_pathway.forward_sequence(waveform))
        acoustic = self.acoustic_proj(self.acoustic_pathway.forward_sequence(mfcc))
        praat_token = self.praat_encoder(praat).unsqueeze(1)            # (B, 1, 256)

        # Each learned stream attends over the other stream *plus* the Praat token.
        deep_context = torch.cat([acoustic, praat_token], dim=1)
        acoustic_context = torch.cat([deep, praat_token], dim=1)

        deep_attended, _ = self.deep_from_context(deep, deep_context)
        acoustic_attended, _ = self.acoustic_from_context(acoustic, acoustic_context)

        return torch.cat([
            deep_attended.mean(dim=1),
            acoustic_attended.mean(dim=1),
            praat_token.squeeze(1),
        ], dim=1)

    def forward(self, waveform: torch.Tensor = None, mfcc: torch.Tensor = None,
                praat: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(self.forward_features(waveform, mfcc, praat))
