"""
Deep Pathway (Role 1): wav2vec 2.0 + LoRA adapters.

Loads a pre-trained wav2vec 2.0 backbone and injects LoRA adapters into the
self-attention layers so the model can adapt to pathological speech traits
without full fine-tuning. Forward pass outputs the 768-dimensional latent
embedding consumed by the fusion head.

Requires: transformers, peft
"""

import warnings
from typing import Optional

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
        # Kept as a direct reference to the (unwrapped) backbone so the
        # sample-length -> feature-length conversion below still works after
        # get_peft_model wraps it — get_peft_model wraps this same nn.Module
        # in place rather than copying it, so the bound method stays valid
        # and correct (it's a pure function of conv strides, unaffected by
        # LoRA adapters) even when self.wav2vec becomes a PeftModel.
        self._feat_extract_output_lengths = backbone._get_feat_extract_output_lengths

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

    def forward_sequence(self, waveform: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The per-frame hidden states, *before* the mean-pool that forward() applies.

        Phase 6's attention fusion needs these: cross-attention over a single
        mean-pooled vector is a no-op (a softmax over one key is always 1.0), so
        attending to wav2vec's evidence requires the frames it is pooled from.

        Args:
            waveform: (batch, samples) raw 16 kHz audio, right-padded with
                zeros past each row's true length.
            attention_mask: (batch, samples) bool/long, True/1 for real audio,
                False/0 for padding — passed straight to Wav2Vec2Model, which
                natively converts a sample-level mask to its internal
                feature-level one. None (default) attends over every sample,
                including padding — only safe when the caller already knows
                there is no padding (e.g. a single un-batched utterance).
        Returns:
            (batch, frames, 768) — ~199 frames for a 4-second clip.
        """
        return self.wav2vec(waveform, attention_mask=attention_mask).last_hidden_state

    def sequence_key_padding_mask(self, waveform: torch.Tensor,
                                  attention_mask: torch.Tensor) -> torch.Tensor:
        """
        The frame-level padding mask matching forward_sequence's output, in
        nn.MultiheadAttention's key_padding_mask convention (True = ignore
        this position). Derived from the same sample lengths so the frame
        count lines up exactly with what forward_sequence actually returns
        for this waveform's shape.
        """
        num_frames = self._num_frames(waveform)
        lengths = attention_mask.sum(dim=1)
        feat_lengths = self._feat_extract_output_lengths(lengths)
        frame_idx = torch.arange(num_frames, device=waveform.device)[None, :]
        return frame_idx >= feat_lengths[:, None]           # True where padded

    def _num_frames(self, waveform: torch.Tensor) -> int:
        result = self._feat_extract_output_lengths(waveform.shape[1])
        return int(result.item()) if torch.is_tensor(result) else int(result)

    def forward(self, waveform: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples) raw 16 kHz audio.
            attention_mask: see forward_sequence. When given, the mean-pool
                below also excludes padded frames — otherwise every padded
                utterance's embedding is diluted by however much silence was
                appended to reach the fixed MAX_SAMPLES window.
        Returns:
            (batch, 768) latent embedding, mean-pooled over real frames only.
        """
        hidden = self.forward_sequence(waveform, attention_mask)
        if attention_mask is None:
            return hidden.mean(dim=1)

        key_padding_mask = self.sequence_key_padding_mask(waveform, attention_mask)
        frame_mask = (~key_padding_mask).unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        summed = (hidden * frame_mask).sum(dim=1)
        counts = frame_mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward_all_layers(self, waveform: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
            attention_mask: (batch, samples) bool/long, True/1 for real audio.
                When given, the per-layer mean-pool excludes padded frames,
                exactly as forward() does.

                This matters more here than anywhere else in the project.
                Utterances are pad/truncated to a fixed 4-second window and the
                median padding fraction of that window is ~86% (measured in
                notebooks/01_data_pipeline.ipynb Stage 9), so an UNMASKED mean
                is dominated by silence — roughly six parts padding to one part
                speech. Leaving the mask off was how the Phase 2 SVM layer sweep
                was originally computed, and it is the leading candidate
                explanation for that reproduction landing at 82.25% against the
                paper's 93.95%. Both variants are kept available so the
                difference can be REPORTED as a diagnostic rather than silently
                corrected — see src.training.baseline.

                None (the default) preserves the original unmasked behaviour so
                previously cached sweeps stay reproducible.
        Returns:
            (batch, 13, 768) mean-pooled embedding per layer.
        """
        outputs = self.wav2vec(waveform, attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=1)  # (B, 13, T, 768)
        if attention_mask is None:
            return hidden_states.mean(dim=2)

        # One frame mask, broadcast across all 13 layers — every layer shares
        # the same time axis, so the padded positions are identical throughout.
        key_padding_mask = self.sequence_key_padding_mask(waveform, attention_mask)
        frame_mask = (~key_padding_mask)[:, None, :, None].to(hidden_states.dtype)
        summed = (hidden_states * frame_mask).sum(dim=2)           # (B, 13, 768)
        counts = frame_mask.sum(dim=2).clamp(min=1.0)              # (B, 1, 1)
        return summed / counts

    def trainable_parameter_summary(self) -> str:
        """Human-readable count of trainable (LoRA) vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return (f"trainable: {trainable:,} / total: {total:,} "
                f"({100 * trainable / total:.2f}%)")
