"""
UA-Speech dataset: one item per utterance with the three branch inputs.

    waveform            (1, 64000)  speech-focused profile, zero-padded
    waveform_length     real samples in it (the attention mask is built from this)
    segmental           (43, 401)   MFCC+d+dd, F1-F3, HNR — fold-standardized
    supra               (3, 401)    F0, voicing, intensity — fold-standardized
    supra_valid_frames  real frames on the temporal-preserving profile
    severity_label      0..3
    speaker_index       this fold's training-speaker id (adversarial head);
                        present only when a speaker_label_map is given
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config
from src.preprocessing import (extract_segmental_features_cached,
                               extract_suprasegmental_features_cached,
                               load_and_preprocess_cached, mfcc_frame_count,
                               normalize_segmental, normalize_suprasegmental, valid_length)


class UASpeechDataset(Dataset):
    """segmental_stats / supra_stats: (mean, std) from the fold's TRAINING
    split (src.training.data.build_loaders); None leaves features raw."""

    def __init__(self, dataframe: pd.DataFrame,
                 speaker_label_map: Optional[Dict[str, int]] = None,
                 segmental_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 supra_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.df = dataframe.reset_index(drop=True)
        self.speaker_label_map = speaker_label_map
        self.segmental_stats = segmental_stats
        self.supra_stats = supra_stats

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        filepath = row["Filepath"]
        waveform, waveform_length = load_and_preprocess_cached(filepath)

        segmental = extract_segmental_features_cached(filepath)
        supra = extract_suprasegmental_features_cached(filepath)
        segmental_valid_frames = min(mfcc_frame_count(waveform_length), segmental.shape[-1])
        supra_valid_frames = min(mfcc_frame_count(valid_length(filepath, supra=True)),
                                 supra.shape[-1])
        if self.segmental_stats is not None:
            segmental = normalize_segmental(segmental, segmental_valid_frames, self.segmental_stats)
        if self.supra_stats is not None:
            supra = normalize_suprasegmental(supra, supra_valid_frames, self.supra_stats)

        item = {
            "waveform": waveform,
            "waveform_length": torch.tensor(waveform_length, dtype=torch.long),
            "segmental": segmental,
            "supra": supra,
            "supra_valid_frames": torch.tensor(supra_valid_frames, dtype=torch.long),
            "severity_label": torch.tensor(config.SEVERITY_LABEL_MAP[row["Severity"]],
                                           dtype=torch.long),
            "speaker_id": row["Speaker_ID"],
            "filename": row["Filename"],
        }
        if self.speaker_label_map is not None:
            item["speaker_index"] = torch.tensor(self.speaker_label_map[row["Speaker_ID"]],
                                                 dtype=torch.long)
        return item
