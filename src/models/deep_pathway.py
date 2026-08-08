"""
Deep Pathway (Role 1): wav2vec 2.0 + LoRA adapters.

Loads a pre-trained wav2vec 2.0 backbone and injects LoRA adapters into the
self-attention layers so the model can adapt to pathological speech traits
without full fine-tuning. Forward pass outputs the 768-dimensional latent
embedding consumed by the fusion head.

Requires: transformers, peft
"""

import warnings

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Wav2Vec2Model

from src import config

# The public model card needs no auth; this notice ("set HF_TOKEN for higher
# rate limits") fires once per process on the first from_pretrained() call
# and is not actionable in a fixed CI/notebook run — filtered here rather than
# left to print itself into every notebook that touches the Deep Pathway.
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")


class DeepPathway(nn.Module):
    """wav2vec 2.0, mean-pooled to a 768-dim vector.

    use_lora=True  (default) injects LoRA adapters into the self-attention
                   projections and leaves the rest of the backbone frozen —
                   the team spec's adaptable Deep Pathway.
    use_lora=False loads the plain backbone with every parameter frozen —
                   reproduces the base paper's frozen wav2vec 2.0 feature
                   extractor, used as an ablation baseline.
    """

    def __init__(self, use_lora: bool = True):
        super().__init__()
        self.use_lora = use_lora
        backbone = Wav2Vec2Model.from_pretrained(
            config.WAV2VEC_MODEL_NAME, token=config.HF_TOKEN)

        if use_lora:
            # use_reentrant=False (not the older reentrant checkpoint) recomputes
            # activations during backward instead of storing them for every
            # transformer layer - the standard ~20% compute / ~40% activation-memory
            # trade-off, which is what makes batch=32 safe on an 8 GB card. Only
            # meaningful here (use_lora=True): the frozen backbone below never
            # builds a backward graph at all (none of its params require grad),
            # so checkpointing it would trade compute for memory it never spends.
            # Must happen before get_peft_model - the reentrant-free checkpoint
            # only needs *some* trainable param inside the wrapped segment (the
            # LoRA adapters, injected next), not a grad-requiring input, so no
            # enable_input_require_grads() workaround is needed (and Wav2Vec2Model
            # has no input embeddings to hook into anyway - it takes raw audio).
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            lora_config = LoraConfig(
                r=config.LORA_RANK,
                lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT,
                target_modules=config.LORA_TARGET_MODULES,
                bias="none",
            )
            self.wav2vec = get_peft_model(backbone, lora_config)
        else:
            for param in backbone.parameters():
                param.requires_grad = False
            backbone.eval()
            self.wav2vec = backbone

    def train(self, mode: bool = True):
        """
        Keep the frozen backbone (use_lora=False) in eval mode even when the
        engine calls model.train() for a training epoch — engine.run_epoch
        toggles the whole model with one model.train(mode=train) call, which
        would otherwise re-enable the backbone's internal dropout layers
        despite every backbone param having requires_grad=False. Frozen
        should mean deterministic, not "no weight updates but still noisy."
        """
        super().train(mode)
        if not self.use_lora:
            self.wav2vec.eval()
        return self

    def forward_sequence(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        The per-frame hidden states, *before* the mean-pool that forward() applies.

        Phase 6's attention fusion needs these: cross-attention over a single
        mean-pooled vector is a no-op (a softmax over one key is always 1.0), so
        attending to wav2vec's evidence requires the frames it is pooled from.

        Args:
            waveform: (batch, samples) raw 16 kHz audio.
        Returns:
            (batch, frames, 768) — ~199 frames for a 4-second clip.
        """
        return self.wav2vec(waveform).last_hidden_state

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) raw 16 kHz audio.
        Returns:
            (batch, 768) latent embedding, mean-pooled over time.
        """
        return self.forward_sequence(waveform).mean(dim=1)

    def forward_all_layers(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Every hidden-state layer (CNN feature-extractor output + all 12
        transformer layers), each mean-pooled over time — the base paper
        (Javanmardi et al., ICASSP 2023) sweeps per-layer embeddings and
        finds different layers win for detection (layer 1) vs. severity
        (layer 13/final), so forward()'s final-layer-only pooling can't
        reproduce that comparison. Only meaningful with use_lora=False,
        since LoRA fine-tunes the backbone that produces these layers.

        Args:
            waveform: (batch, samples) raw 16 kHz audio.
        Returns:
            (batch, 13, 768) mean-pooled embedding per layer.
        """
        outputs = self.wav2vec(waveform, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=1)  # (B, 13, T, 768)
        return hidden_states.mean(dim=2)

    def trainable_parameter_summary(self) -> str:
        """Human-readable count of trainable (LoRA) vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return (f"trainable: {trainable:,} / total: {total:,} "
                f"({100 * trainable / total:.2f}%)")
