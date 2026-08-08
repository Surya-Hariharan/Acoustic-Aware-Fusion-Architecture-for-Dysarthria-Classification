"""
Unified PyTorch dataset for the UA-Speech pipeline.

Each item carries everything both pathways need — the raw waveform for the
Deep Pathway (wav2vec 2.0 + LoRA) and the MFCC tensor for the Acoustic
Pathway (1D-CNN) — so both are trained on identical audio and splits.
"""

from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config
from src.praat import praat_vector
from src.preprocessing import (extract_mfcc_features_cached,
                               load_and_preprocess_cached)


class UASpeechDataset(Dataset):
    """Returns waveform, MFCC features, detection label, severity label, speaker,
    utterance identity, and — for Phase 6's Model F — the Praat feature vector.

    praat_table/praat_stats are supplied together or not at all. When they are,
    each item gains a "praat" key and the training engine forwards it to the
    model; when they are not, the key is simply absent from the batch and every
    other model is unaffected. src.training.data.build_loaders wires this up.
    """

    def __init__(self, dataframe: pd.DataFrame,
                 praat_table: Optional[pd.DataFrame] = None,
                 praat_stats: Optional[Tuple] = None):
        self.df = dataframe.reset_index(drop=True)

        if (praat_table is None) != (praat_stats is None):
            raise ValueError(
                "praat_table and praat_stats must be supplied together — the "
                "standardization statistics are computed from a fold's train "
                "split (see src.praat.praat_standardizer), so a table without "
                "them would silently go unnormalized."
            )
        self.praat_table = praat_table
        self.praat_stats = praat_stats

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        waveform = load_and_preprocess_cached(row["Filepath"])
        mfcc = extract_mfcc_features_cached(row["Filepath"])

        item = {
            "waveform": waveform,                                   # (1, 64000)
            "mfcc": mfcc,                                           # (1, 39, frames)
            "group_label": torch.tensor(
                config.GROUP_LABEL_MAP[row["Group"]], dtype=torch.long),
            "severity_label": torch.tensor(
                config.SEVERITY_LABEL_MAP[row["Severity"]], dtype=torch.long),
            "speaker_id": row["Speaker_ID"],
            # Utterance identity, carried all the way into outputs/predictions/
            # so Phase 5 can join a misclassified row back to its audio file and
            # to the Praat features (which are keyed by Filename). A speaker_id
            # alone cannot identify *which* utterance was got wrong.
            "filename": row["Filename"],
            "filepath": row["Filepath"],
        }

        if self.praat_table is not None:
            item["praat"] = torch.from_numpy(
                praat_vector(self.praat_table, row["Filename"], self.praat_stats))

        return item
