"""
Frozen wav2vec 2.0 representations: for every utterance, every hidden state
(CNN feature projection + 12 transformer layers) pooled over real frames into
[mean, std] — shape (13, 2, 768), stored as float16.

The input is the learned branch's: the speech-focused VAD profile, the 4 s
window, per-utterance normalization over valid samples
(DeepPathway._zero_mean_unit_var_norm), then pooling over real frames only.

No attention mask is passed to wav2vec2-base: that checkpoint (group-norm
feature encoder) was trained on zero-padded batches without one, and passing
a mask measurably breaks it — its CTC head, which recognises isolated words
correctly without a mask, decodes them as noise with one (see
src/aaflite/asr.py). A layer-norm checkpoint would take the mask.
Nothing here depends on a fold or a label, so it is computed once and cached
per speaker (resumable).
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Config, Wav2Vec2Model

from src import config
from src.console import print_kv, print_status, progress
from src.models.deep_pathway import DeepPathway
from src.preprocessing import load_and_preprocess_cached

NUM_HIDDEN_STATES = 13


class _WaveformDataset(Dataset):
    def __init__(self, filepaths: List[str]):
        self.filepaths = filepaths

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        waveform, length = load_and_preprocess_cached(self.filepaths[idx])
        return waveform.reshape(-1), int(length)


def load_frozen_wav2vec2(device: torch.device) -> Wav2Vec2Model:
    """The learned branch's backbone without LoRA: identical weights, eval mode."""
    backbone_config = Wav2Vec2Config.from_pretrained(config.WAV2VEC_MODEL_NAME, token=config.HF_TOKEN)
    backbone_config.apply_spec_augment = False
    backbone_config.mask_time_prob = 0.0
    backbone_config.mask_feature_prob = 0.0
    model = Wav2Vec2Model.from_pretrained(config.WAV2VEC_MODEL_NAME, config=backbone_config,
                                          token=config.HF_TOKEN)
    return model.to(device).eval()


@torch.no_grad()
def layer_stats(model: Wav2Vec2Model, waveform: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """(B, samples) audio + (B,) valid lengths -> (B, 13, 2, 768) [mean, std]
    of every hidden state over the real (unpadded) frames."""
    mask = torch.arange(waveform.shape[1], device=waveform.device)[None, :] < lengths[:, None]
    waveform = DeepPathway._zero_mean_unit_var_norm(waveform, mask)      # padding stays exactly 0
    if model.config.feat_extract_norm == "layer":
        hidden = model(waveform, attention_mask=mask, output_hidden_states=True).hidden_states
    else:
        hidden = model(waveform, output_hidden_states=True).hidden_states
    n_frames = hidden[0].shape[1]
    frame_lengths = model._get_feat_extract_output_lengths(lengths).clamp(min=1)
    keep = (torch.arange(n_frames, device=waveform.device)[None, :] < frame_lengths[:, None])
    keep = keep.unsqueeze(-1).float()
    count = keep.sum(dim=1)                                         # (B, 1)
    stats = []
    for state in hidden:
        state = state.float()
        mean = (state * keep).sum(dim=1) / count
        var = (((state - mean[:, None, :]) * keep) ** 2).sum(dim=1) / count
        stats.append(torch.stack([mean, var.clamp(min=0).sqrt()], dim=1))
    return torch.stack(stats, dim=1)


def _speaker_path(speaker: str) -> Path:
    return Path(config.AAFLITE_EMBEDDING_DIR) / f"{speaker}.npz"


def _speaker_complete(speaker: str, filenames: List[str]) -> bool:
    path = _speaker_path(speaker)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return set(filenames) <= set(data["filenames"].tolist())
    except Exception:
        return False


def extract_wav2vec2_layer_stats(df: pd.DataFrame, device: torch.device,
                                 batch_size: int = config.AAFLITE_EMBED_BATCH_SIZE,
                                 num_workers: int = 2) -> None:
    """Compute and cache the layer stats of every utterance in df, one .npz
    per speaker. Speakers already complete are skipped, so an interrupted
    extraction resumes where it stopped."""
    Path(config.AAFLITE_EMBEDDING_DIR).mkdir(parents=True, exist_ok=True)
    todo = [(spk, grp) for spk, grp in df.groupby("Speaker_ID", sort=True)
            if not _speaker_complete(spk, grp["Filename"].tolist())]
    print_kv("wav2vec2 layer stats", f"{df['Speaker_ID'].nunique() - len(todo)} speakers cached, "
                                     f"{len(todo)} to extract")
    if not todo:
        return
    model = load_frozen_wav2vec2(device)
    use_amp = device.type == "cuda"
    for speaker, group in todo:
        loader = DataLoader(_WaveformDataset(group["Filepath"].tolist()), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=device.type == "cuda")
        chunks = []
        for waveform, lengths in progress(loader, f"wav2vec2 {speaker}", total=len(loader),
                                          unit="batch"):
            waveform = waveform.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                chunks.append(layer_stats(model, waveform, lengths).cpu().numpy().astype(np.float16))
        stats = np.concatenate(chunks)
        if not np.isfinite(stats).all():
            raise RuntimeError(f"Non-finite wav2vec2 statistics for speaker {speaker}.")
        path = _speaker_path(speaker)
        temp = path.with_suffix(".tmp.npz")
        np.savez(temp, filenames=group["Filename"].to_numpy(dtype=str), stats=stats)
        temp.replace(path)
    print_status(f"wav2vec2 layer stats cached in {config.AAFLITE_EMBEDDING_DIR}", ok=True)


def load_wav2vec2_layer_stats(df: pd.DataFrame) -> np.ndarray:
    """(N, 13, 2, 768) float16 in df row order; raises if any row is missing."""
    by_speaker: Dict[str, Dict[str, int]] = {}
    arrays: Dict[str, np.ndarray] = {}
    for speaker in df["Speaker_ID"].unique():
        with np.load(_speaker_path(speaker), allow_pickle=False) as data:
            arrays[speaker] = data["stats"]
            by_speaker[speaker] = {name: i for i, name in enumerate(data["filenames"].tolist())}
    out = np.empty((len(df), NUM_HIDDEN_STATES, 2, config.WAV2VEC_EMBED_DIM), dtype=np.float16)
    for row, (speaker, filename) in enumerate(zip(df["Speaker_ID"], df["Filename"])):
        out[row] = arrays[speaker][by_speaker[speaker][filename]]
    return out


def layer_group_features(stats: np.ndarray, layers) -> np.ndarray:
    """(N, 13, 2, 768) -> (N, 1536): [mean | std], each averaged over `layers`."""
    pooled = stats[:, list(layers)].astype(np.float32).mean(axis=1)    # (N, 2, 768)
    return pooled.reshape(len(stats), -1)
