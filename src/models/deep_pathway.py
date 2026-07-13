"""
Deep Pathway (Role 1): wav2vec 2.0 + LoRA adapters.

Loads a pre-trained wav2vec 2.0 backbone and injects LoRA adapters into the
self-attention layers so the model can adapt to pathological speech traits
without full fine-tuning. Forward pass outputs the 768-dimensional latent
embedding consumed by the fusion head.

Requires: transformers, peft
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Wav2Vec2Model

from src import config


class DeepPathway(nn.Module):
    """wav2vec 2.0 with LoRA on self-attention, mean-pooled to a 768-dim vector."""

    def __init__(self):
        super().__init__()
        backbone = Wav2Vec2Model.from_pretrained(config.WAV2VEC_MODEL_NAME)

        lora_config = LoraConfig(
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias="none",
        )
        self.wav2vec = get_peft_model(backbone, lora_config)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) raw 16 kHz audio.
        Returns:
            (batch, 768) latent embedding, mean-pooled over time.
        """
        outputs = self.wav2vec(waveform)
        return outputs.last_hidden_state.mean(dim=1)

    def trainable_parameter_summary(self) -> str:
        """Human-readable count of trainable (LoRA) vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return (f"trainable: {trainable:,} / total: {total:,} "
                f"({100 * trainable / total:.2f}%)")
