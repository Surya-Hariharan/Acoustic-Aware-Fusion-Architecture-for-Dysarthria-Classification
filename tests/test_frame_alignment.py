"""
Confirms the three branches' frame-level inputs share one common frame grid
— config.MAX_SAMPLES / config.MEL_KWARGS["hop_length"] frames — rather than
relying on extractor-specific arithmetic happening to agree by accident.

Run with: pytest tests/test_frame_alignment.py -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.praat import (extract_segmental_extra_sequence, extract_suprasegmental_sequence,
                       FRAME_HOP_SECONDS)
from src.preprocessing import mfcc_frame_count


def test_mfcc_frame_count_defines_the_common_grid():
    total_frames = mfcc_frame_count(config.MAX_SAMPLES)
    assert total_frames == config.MAX_SAMPLES // config.MEL_KWARGS["hop_length"] + 1


def test_segmental_extra_sequence_matches_the_mfcc_frame_grid():
    total_frames = mfcc_frame_count(config.MAX_SAMPLES)
    waveform = np.zeros(config.MAX_SAMPLES, dtype=np.float32)
    result = extract_segmental_extra_sequence(waveform, sr=config.TARGET_SR, valid_length=0,
                                               total_frames=total_frames)
    for name, arr in result.items():
        assert arr.shape[0] == total_frames, f"{name} has {arr.shape[0]} frames, expected {total_frames}"


def test_suprasegmental_sequence_matches_the_mfcc_frame_grid():
    total_frames = mfcc_frame_count(config.MAX_SAMPLES)
    waveform = np.zeros(config.MAX_SAMPLES, dtype=np.float32)
    result = extract_suprasegmental_sequence(waveform, sr=config.TARGET_SR, valid_length=0,
                                              total_frames=total_frames)
    for name, arr in result.items():
        assert arr.shape[0] == total_frames, f"{name} has {arr.shape[0]} frames, expected {total_frames}"


def test_frame_hop_matches_mfcc_hop_length():
    # FRAME_HOP_SECONDS is the suprasegmental/segmental-extra extractors' own
    # source of truth for their time_step; it must agree with the hop_length
    # the MFCC/segmental grid is built from, or the "common frame grid" claim
    # is only true by coincidence of frame COUNT, not actual time alignment.
    expected_hop_seconds = config.MEL_KWARGS["hop_length"] / config.TARGET_SR
    assert abs(FRAME_HOP_SECONDS - expected_hop_seconds) < 1e-6


def test_segmental_channel_count_matches_config():
    total_frames = mfcc_frame_count(config.MAX_SAMPLES)
    waveform = np.zeros(config.MAX_SAMPLES, dtype=np.float32)
    result = extract_segmental_extra_sequence(waveform, sr=config.TARGET_SR, valid_length=0,
                                               total_frames=total_frames)
    # 39 (MFCC+delta+delta-delta, computed elsewhere) + these 4 = SEGMENTAL_CHANNELS.
    assert 3 * config.N_MFCC + len(result) == config.SEGMENTAL_CHANNELS == 43


def test_suprasegmental_channel_count_matches_config():
    total_frames = mfcc_frame_count(config.MAX_SAMPLES)
    waveform = np.zeros(config.MAX_SAMPLES, dtype=np.float32)
    result = extract_suprasegmental_sequence(waveform, sr=config.TARGET_SR, valid_length=0,
                                              total_frames=total_frames)
    assert len(result) == config.SUPRA_CHANNELS == 3


if __name__ == "__main__":
    test_mfcc_frame_count_defines_the_common_grid()
    test_segmental_extra_sequence_matches_the_mfcc_frame_grid()
    test_suprasegmental_sequence_matches_the_mfcc_frame_grid()
    test_frame_hop_matches_mfcc_hop_length()
    test_segmental_channel_count_matches_config()
    test_suprasegmental_channel_count_matches_config()
    print("All frame-alignment tests passed.")
