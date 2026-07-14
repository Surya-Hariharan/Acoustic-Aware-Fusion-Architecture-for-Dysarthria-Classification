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
        backbone = Wav2Vec2Model.from_pretrained(config.WAV2VEC_MODEL_NAME)

        if use_lora:
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
            self.wav2vec = backbone

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

    def trainable_parameter_summary(self) -> str:
        """Human-readable count of trainable (LoRA) vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return (f"trainable: {trainable:,} / total: {total:,} "
                f"({100 * trainable / total:.2f}%)")
