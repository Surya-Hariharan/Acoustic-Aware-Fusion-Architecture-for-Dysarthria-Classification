"""
Unified PyTorch dataset for the UA-Speech pipeline.

Each item carries everything both pathways need — the raw waveform for the
Deep Pathway (wav2vec 2.0 + LoRA) and the MFCC tensor for the Acoustic
Pathway (1D-CNN) — so both are trained on identical audio and splits.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config
from src.preprocessing import build_mfcc_transform, extract_mfcc_features, load_and_preprocess


class UASpeechDataset(Dataset):
    """Returns waveform, MFCC features, detection label, severity label, speaker."""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)
        self.mfcc_transform = build_mfcc_transform()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        waveform = load_and_preprocess(row["Filepath"])
        mfcc = extract_mfcc_features(waveform, self.mfcc_transform)

        return {
            "waveform": waveform,                                   # (1, 64000)
            "mfcc": mfcc,                                           # (1, 39, frames)
            "group_label": torch.tensor(
                config.GROUP_LABEL_MAP[row["Group"]], dtype=torch.long),
            "severity_label": torch.tensor(
                config.SEVERITY_LABEL_MAP[row["Severity"]], dtype=torch.long),
            "speaker_id": row["Speaker_ID"],
        }
