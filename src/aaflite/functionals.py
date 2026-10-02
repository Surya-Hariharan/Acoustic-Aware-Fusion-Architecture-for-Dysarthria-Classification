"""
Utterance-level functionals of the stored frame-wise acoustic features.

Segmental (43 channels: MFCC+d+dd, F1-F3, HNR) and suprasegmental (F0
semitones, voicing, intensity) sequences are summarized over their VALID
frames only — mean, std, 10th / 50th / 90th percentile. F0 is summarized over
voiced frames only (unvoiced frames carry F0 = 0, which is "absent", not a
pitch). Timing comes from the untruncated VAD spans, so utterances longer than
the 4 s window keep their true duration — slow, effortful speech is a primary
clinical marker of dysarthria severity.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from src import config, feature_store
from src.preprocessing import mfcc_frame_count

STATS = ("mean", "std", "p10", "p50", "p90")
SEGMENTAL_CHANNEL_NAMES = ([f"mfcc{i}" for i in range(config.N_MFCC)]
                           + [f"d_mfcc{i}" for i in range(config.N_MFCC)]
                           + [f"dd_mfcc{i}" for i in range(config.N_MFCC)]
                           + ["F1", "F2", "F3", "HNR"])
SUPRA_CHANNEL_NAMES = ["F0_st", "voicing", "intensity_db"]
TIMING_NAMES = ["speech_duration_s", "supra_duration_s", "voiced_ratio", "f0_range_st"]


def _stats(x: np.ndarray) -> np.ndarray:
    """(C, T) -> (C * 5,) over the time axis; zeros when T == 0."""
    if x.shape[1] == 0:
        return np.zeros(x.shape[0] * len(STATS), dtype=np.float32)
    p10, p50, p90 = np.percentile(x, [10, 50, 90], axis=1)
    return np.stack([x.mean(axis=1), x.std(axis=1), p10, p50, p90], axis=1).reshape(-1)


def feature_names() -> Tuple[List[str], List[str]]:
    segmental = [f"{c}_{s}" for c in SEGMENTAL_CHANNEL_NAMES for s in STATS]
    supra = [f"{c}_{s}" for c in SUPRA_CHANNEL_NAMES for s in STATS] + TIMING_NAMES
    return segmental, supra


def acoustic_functionals(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """(segmental (N, 215), suprasegmental (N, 19)) float32 in df row order.
    Every row must be in the feature store."""
    spans = feature_store.span_table()
    segmental_rows, supra_rows = [], []
    for filepath, filename in zip(df["Filepath"], df["Filename"]):
        segmental = feature_store.segmental_features(filepath)
        supra = feature_store.suprasegmental_features(filepath)
        if segmental is None or supra is None or filename not in spans:
            raise KeyError(f"{filename} is not in the feature store — build it first.")
        _, s0, s1, p0, p1 = spans[filename]
        speech_samples, supra_samples = max(0, s1 - s0), max(0, p1 - p0)
        n_seg = min(mfcc_frame_count(min(speech_samples, config.MAX_SAMPLES)), segmental.shape[1])
        n_sup = min(mfcc_frame_count(min(supra_samples, config.MAX_SAMPLES)), supra.shape[1])
        segmental, supra = segmental[:, :n_seg], supra[:, :n_sup]

        voiced = supra[1] > 0.5
        f0 = supra[0][voiced]
        supra_stats = _stats(supra)
        f0_stats = _stats(f0[None, :]) if f0.size else np.zeros(len(STATS), dtype=np.float32)
        supra_stats[:len(STATS)] = f0_stats                      # F0 over voiced frames only
        timing = np.array([speech_samples / config.TARGET_SR, supra_samples / config.TARGET_SR,
                           float(voiced.mean()) if voiced.size else 0.0,
                           float(np.ptp(f0)) if f0.size else 0.0], dtype=np.float32)
        segmental_rows.append(_stats(segmental))
        supra_rows.append(np.concatenate([supra_stats, timing]))
    return (np.asarray(segmental_rows, dtype=np.float32),
            np.asarray(supra_rows, dtype=np.float32))
