"""
Unified PyTorch dataset for the UA-Speech pipeline.

Each item carries everything both pathways need — the raw waveform for the
Deep Pathway (wav2vec 2.0 + LoRA) and the MFCC tensor for the Acoustic
Pathway (1D-CNN) — so both are trained on identical audio and splits.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config
from src.praat import praat_vector
from src.preprocessing import (extract_mfcc_features_cached,
                               extract_segmental_features_cached,
                               extract_suprasegmental_features_cached,
                               load_and_preprocess_cached,
                               load_and_preprocess_supra_cached, mfcc_frame_count,
                               normalize_segmental, normalize_suprasegmental)


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
                 praat_stats: Optional[Tuple] = None,
                 frozen_embedding_table: Optional[Dict[str, np.ndarray]] = None,
                 include_mfcc: bool = True,
                 include_three_branch: bool = False,
                 speaker_label_map: Optional[Dict[str, int]] = None,
                 segmental_stats: Optional[Tuple] = None,
                 supra_stats: Optional[Tuple] = None):
        self.df = dataframe.reset_index(drop=True)

        # Three-branch severity architecture only (src.models.gated_fusion) —
        # adds "segmental" (43ch, speech-focused profile) and "supra" (3ch,
        # temporal-preserving profile) tensors, plus "supra_valid_frames".
        # False for every legacy model (default), which never reads these
        # keys, matching include_mfcc's "absent key, no cost" convention.
        self.include_three_branch = include_three_branch

        # (mean, std) from src.preprocessing.segmental_standardizer /
        # suprasegmental_standardizer, computed by src.training.data.build_loaders
        # from ONE FOLD'S TRAIN SPLIT ONLY and reused unchanged for that fold's
        # val/test Datasets — the same leakage discipline as praat_stats above.
        # None (default) leaves segmental/supra raw/unnormalized, e.g. for any
        # caller that builds a Dataset outside the fold machinery (notebooks,
        # tests) without first computing fold statistics.
        self.segmental_stats = segmental_stats
        self.supra_stats = supra_stats

        # Filename-keyed int speaker id for the GRL speaker head (see
        # src.models.gated_fusion.SpeakerHead) — built per-fold from that
        # fold's TRAINING speakers only (src.training.data.build_speaker_label_map),
        # since which speakers are "in the fold" changes every LOSO fold.
        # None (default, and always for the test split) means no
        # "speaker_index" key — src.training.engine.run_epoch already treats
        # a missing key as "this model doesn't need it" for every other
        # optional field (praat, deep_embedding), so the adversarial loss is
        # simply skipped wherever this is None (see GatedFusionModel.training_step).
        self.speaker_label_map = speaker_label_map

        # MFCC extraction (STFT + mel filterbank + DCT + two delta passes) runs
        # per utterance on the DataLoader worker and is pure waste for the
        # wav2vec2-only variants, which never read the tensor — deep_frozen and
        # deep_lora discard it after it has been computed, collated, pinned and
        # copied to the GPU. deep_frozen consumes a PRECOMPUTED embedding and
        # runs no backbone forward pass at all, yet still benchmarked at ~194s
        # per fold-epoch; that cost is almost entirely this.
        #
        # False emits a zero-size placeholder so the batch dict keeps its shape
        # and src.training.engine.run_epoch needs no per-model branching.
        # src.training.data.build_loaders sets it from the model name.
        self.include_mfcc = include_mfcc

        if (praat_table is None) != (praat_stats is None):
            raise ValueError(
                "praat_table and praat_stats must be supplied together — the "
                "standardization statistics are computed from a fold's train "
                "split (see src.praat.praat_standardizer), so a table without "
                "them would silently go unnormalized."
            )
        self.praat_table = praat_table
        self.praat_stats = praat_stats
        # Filepath -> precomputed 768-dim frozen wav2vec2 embedding (see
        # src.training.baseline.extract_frozen_embeddings_masked). Only ever
        # populated for the deep_frozen/fusion_frozen variants (see
        # src.training.data.build_loaders) — every other model never sees a
        # "deep_embedding" key and is unaffected.
        self.frozen_embedding_table = frozen_embedding_table

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        waveform, waveform_length = load_and_preprocess_cached(row["Filepath"])

        if self.include_mfcc:
            mfcc = extract_mfcc_features_cached(row["Filepath"])
            # Raw (pre-pool) MFCC frame count derived from the audio itself, not
            # from MFCC values — frames >= this index are the fixed-window's
            # zero-padded tail, not real speech. AcousticPathway derives the same
            # quantity internally from waveform_length (see valid_frame_count),
            # so this is metadata for validation/visualization, not a second
            # source of truth the model reads from.
            mfcc_valid_frames = min(mfcc_frame_count(waveform_length), mfcc.shape[-1])
        else:
            mfcc = torch.empty(1, 3 * config.N_MFCC, 0)
            mfcc_valid_frames = 0

        item = {
            "waveform": waveform,                                   # (1, 64000)
            "waveform_length": torch.tensor(waveform_length, dtype=torch.long),
            "mfcc": mfcc,                                           # (1, 39, frames)
            "mfcc_valid_frames": torch.tensor(mfcc_valid_frames, dtype=torch.long),
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

        if self.frozen_embedding_table is not None:
            item["deep_embedding"] = torch.from_numpy(
                self.frozen_embedding_table[row["Filepath"]]).float()

        if self.include_three_branch:
            segmental = extract_segmental_features_cached(row["Filepath"])  # (43, T)
            supra = extract_suprasegmental_features_cached(row["Filepath"])  # (3, T)

            # Segmental shares the speech-focused profile with MFCC, so its
            # valid-frame count is the same waveform_length already computed
            # above (not a second source of truth — mirrors mfcc_valid_frames).
            segmental_valid_frames = min(mfcc_frame_count(waveform_length), segmental.shape[-1])

            _, supra_valid_length = load_and_preprocess_supra_cached(row["Filepath"])
            supra_valid_frames = min(mfcc_frame_count(supra_valid_length),
                                     supra.shape[-1])

            # Fold-scoped, leakage-safe channel normalization (see
            # src.preprocessing.segmental_standardizer / suprasegmental_standardizer
            # and src.training.data.build_loaders, which computes these stats from
            # ONE FOLD'S TRAIN SPLIT ONLY). None (e.g. a Dataset built without
            # fold statistics) leaves the raw extracted values unchanged.
            if self.segmental_stats is not None:
                segmental = normalize_segmental(segmental, segmental_valid_frames, self.segmental_stats)
            if self.supra_stats is not None:
                supra = normalize_suprasegmental(supra, supra_valid_frames, self.supra_stats)

            item["segmental"] = segmental
            item["supra"] = supra
            item["supra_valid_frames"] = torch.tensor(supra_valid_frames, dtype=torch.long)

        if self.speaker_label_map is not None:
            item["speaker_index"] = torch.tensor(
                self.speaker_label_map[row["Speaker_ID"]], dtype=torch.long)

        return item
