"""
Resumability and equality gate for the chunked feature store
(src/feature_store.py).

Synthetic audio written to tmp_path, named like UA-Speech files
(<Speaker>_<Block>_<Word>_<Mic>.wav) under a speaker prefix that cannot
collide with the real corpus (the store keys on basenames).

What is asserted:
  * a build computes every missing chunk, and a second build computes none;
  * deleting one chunk (a session killed mid-build) rebuilds exactly that one;
  * a leftover temp file from an interrupted write is never read as a chunk;
  * a chunk written under a different configuration is ignored and rebuilt;
  * stored tensors equal the original live pipeline bit-for-bit, and the
    cached read path (src.preprocessing / src.vad_cache) serves them.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio

from src import config, feature_store, vad_cache

SPEAKERS = ("SYNTHA", "SYNTHB")


def _write_utterance(path, seed):
    rng = np.random.default_rng(seed)
    n = int(2.0 * config.TARGET_SR)
    t = np.arange(n) / config.TARGET_SR
    signal = rng.normal(0.0, 1e-4, n)
    start, end = int(0.5 * config.TARGET_SR), int(1.4 * config.TARGET_SR)
    # A harmonic tone burst, so Praat finds pitch and formants to report.
    signal[start:end] += 0.3 * (np.sin(2 * np.pi * 140 * t[start:end])
                                + 0.5 * np.sin(2 * np.pi * 280 * t[start:end]))
    torchaudio.save(str(path), torch.from_numpy(signal.astype(np.float32)).unsqueeze(0),
                    config.TARGET_SR)


@pytest.fixture
def store_env(tmp_path, monkeypatch):
    """Isolated store directory + a 2-chunk synthetic manifest (2 files each)."""
    monkeypatch.setattr(config, "FEATURE_STORE_DIR", tmp_path / "store")
    monkeypatch.setattr(config, "FEATURE_STORE_EXTRA_DIRS", [])
    monkeypatch.delenv("FEATURE_STORE_EXTRA_DIRS", raising=False)
    rows = []
    for s, speaker in enumerate(SPEAKERS):
        for w in range(2):
            name = f"{speaker}_B1_UW{w}_M6.wav"
            _write_utterance(tmp_path / name, seed=10 * s + w)
            rows.append({"Filename": name, "Filepath": str(tmp_path / name)})
    feature_store.clear_cache()
    vad_cache.clear_span_table_cache()
    yield pd.DataFrame(rows), tmp_path / "store"
    feature_store.clear_cache()
    vad_cache.clear_span_table_cache()


def test_chunk_key_is_speaker_and_block():
    assert feature_store.chunk_key("/x/y/M05_B1_UW13_M6.wav") == "M05_B1"
    with pytest.raises(ValueError):
        feature_store.chunk_key("recording_01.wav")


def test_build_is_resumable_and_never_recomputes_a_finished_chunk(store_env):
    df, store_dir = store_env
    first = feature_store.build_feature_store(df, n_workers=2)
    assert sorted(first["built"]) == ["SYNTHA_B1", "SYNTHB_B1"]
    assert feature_store.feature_store_coverage(df)["complete"]

    second = feature_store.build_feature_store(df, n_workers=2)
    assert second["built"] == []                                  # nothing recomputed

    # Simulate a session killed after one chunk: remove the other, and leave a
    # half-written temp file behind as an interrupted write would.
    (store_dir / "SYNTHB_B1.npz").unlink()
    (store_dir / "SYNTHB_B1.npz.tmp").write_bytes(b"truncated")
    mtime_a = (store_dir / "SYNTHA_B1.npz").stat().st_mtime_ns
    assert feature_store.feature_store_coverage(df)["chunks_missing"] == ["SYNTHB_B1"]

    third = feature_store.build_feature_store(df, n_workers=2)
    assert third["built"] == ["SYNTHB_B1"]                        # only the missing chunk
    assert (store_dir / "SYNTHA_B1.npz").stat().st_mtime_ns == mtime_a


def test_chunk_from_a_different_configuration_is_ignored(store_env, monkeypatch):
    df, _ = store_env
    feature_store.build_feature_store(df, n_workers=2)
    monkeypatch.setattr(config, "SUPRA_VAD_SPEECH_PAD_MS", config.SUPRA_VAD_SPEECH_PAD_MS + 10)
    feature_store.clear_cache()
    assert feature_store.segmental_features(df["Filepath"].iloc[0]) is None
    assert not feature_store.feature_store_coverage(df)["complete"]


def test_stored_features_match_the_live_pipeline_and_are_served(store_env):
    from src import preprocessing

    df, _ = store_env
    feature_store.build_feature_store(df, n_workers=2)
    feature_store.verify_feature_store(df, n=len(df))            # raises on any mismatch

    for filepath in df["Filepath"]:
        live_segmental, live_supra = feature_store._live_reference(filepath)
        for fn in (preprocessing.extract_segmental_features_cached,
                   preprocessing.extract_suprasegmental_features_cached):
            fn.cache_clear()
        assert np.array_equal(
            preprocessing.extract_segmental_features_cached(filepath).numpy(), live_segmental)
        assert np.array_equal(
            preprocessing.extract_suprasegmental_features_cached(filepath).numpy(), live_supra)
        # Spans reach the VAD span table (the 300-sample verification's source).
        name = filepath.split("\\")[-1].split("/")[-1]
        assert name in vad_cache._span_table()
