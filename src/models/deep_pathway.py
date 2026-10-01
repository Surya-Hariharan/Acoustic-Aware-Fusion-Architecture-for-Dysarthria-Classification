"""
Learned branch backbone: wav2vec 2.0 (facebook/wav2vec2-base-960h) with LoRA
adapters on the self-attention q/k/v projections of all 12 encoder layers. The
backbone weights stay frozen; only the adapters train. Output is the
768-dimensional hidden state, mean-pooled over real (unpadded) frames.
"""

import warnings
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Wav2Vec2Config, Wav2Vec2Model

from src import config

# "Set HF_TOKEN for higher rate limits" fires on every first load of a public
# checkpoint and is not actionable here.
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")


class DeepPathway(nn.Module):
    """(batch, samples) raw 16 kHz audio -> (batch, 768)."""

    def __init__(self, gradient_checkpointing: Optional[bool] = None):
        super().__init__()
        if gradient_checkpointing is None:
            gradient_checkpointing = config.WAV2VEC_GRADIENT_CHECKPOINTING
        backbone_config = Wav2Vec2Config.from_pretrained(config.WAV2VEC_MODEL_NAME,
                                                         token=config.HF_TOKEN)
        if not config.WAV2VEC_APPLY_SPEC_AUGMENT:
            # The CTC checkpoint has no pretrained masked_spec_embed; with a
            # positive masking probability Transformers would create a random
            # one and use it in training. Disable before construction.
            backbone_config.apply_spec_augment = False
            backbone_config.mask_time_prob = 0.0
            backbone_config.mask_feature_prob = 0.0
        backbone = Wav2Vec2Model.from_pretrained(config.WAV2VEC_MODEL_NAME, config=backbone_config,
                                                 token=config.HF_TOKEN)
        # A pure function of the conv strides; still valid after peft wraps
        # the backbone in place.
        self._feat_extract_output_lengths = backbone._get_feat_extract_output_lengths

        self.wav2vec = get_peft_model(backbone, LoraConfig(
            r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES, bias="none"))
        # After get_peft_model: enabling it first makes peft call
        # enable_input_require_grads(), which Wav2Vec2Model (raw-audio input,
        # no input embeddings) does not implement.
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

    @staticmethod
    def _zero_mean_unit_var_norm(waveform: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Per-utterance zero-mean / unit-variance over each row's valid
        samples — what the checkpoint's own Wav2Vec2FeatureExtractor does
        (do_normalize=True). Padding is forced back to exact zero. Purely
        per-example, so it cannot leak statistics across speakers or folds."""
        if attention_mask is None:
            mean = waveform.mean(dim=-1, keepdim=True)
            var = waveform.var(dim=-1, unbiased=False, keepdim=True)
            return (waveform - mean) / torch.sqrt(var + 1e-7)
        mask = attention_mask.to(waveform.dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = (waveform * mask).sum(dim=-1, keepdim=True) / count
        var = (((waveform - mean) * mask) ** 2).sum(dim=-1, keepdim=True) / count
        return (waveform - mean) / torch.sqrt(var + 1e-7) * mask

    def frame_padding_mask(self, num_samples: int, attention_mask: torch.Tensor) -> torch.Tensor:
        """(batch, frames) True where a wav2vec2 output frame is padding."""
        num_frames = int(self._feat_extract_output_lengths(num_samples))
        feat_lengths = self._feat_extract_output_lengths(attention_mask.sum(dim=1))
        frame_idx = torch.arange(num_frames, device=attention_mask.device)[None, :]
        return frame_idx >= feat_lengths[:, None]

    def forward(self, waveform: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """attention_mask: (batch, samples), True for real audio. The
        transformer ignores padded frames and the mean-pool excludes them."""
        waveform = self._zero_mean_unit_var_norm(waveform, attention_mask)
        hidden = self.wav2vec(waveform, attention_mask=attention_mask).last_hidden_state
        if attention_mask is None:
            return hidden.mean(dim=1)
        keep = (~self.frame_padding_mask(waveform.shape[1], attention_mask)).unsqueeze(-1)
        keep = keep.to(hidden.dtype)
        return (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
