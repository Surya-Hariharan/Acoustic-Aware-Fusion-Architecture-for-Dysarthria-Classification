"""
Equality gate for VAD spans served from the feature store (src/vad_cache.py).

Reading the trim from two stored integers instead of running Silero is only
acceptable if nothing downstream moves, so these tests assert exact equality
(torch.equal / array_equal) against the live-Silero path — on synthetic
audio, so they run without the corpus. The corpus-scale check is
verify_vad_span_cache, run from the training notebook.
"""

import contextlib

import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio

from src import config, feature_store, preprocessing, vad_cache

# Not a UA-Speech speaker, so no real store entry can answer for these files.
SPEAKERS = ("SYNTHC", "SYNTHD")


def _write_utterance(path, speech_from, speech_to, seed, seconds=2.0):
    """Near-silence with a noise burst, so Silero finds a span to trim to."""
    rng = np.random.default_rng(seed)
    n = int(seconds * config.TARGET_SR)
    signal = rng.normal(0.0, 1e-4, n).astype(np.float32)
    start, end = int(speech_from * config.TARGET_SR), int(speech_to * config.TARGET_SR)
    signal[start:end] += (0.3 * np.hanning(end - start)
                          * rng.normal(0.0, 1.0, end - start)).astype(np.float32)
    torchaudio.save(str(path), torch.from_numpy(signal).unsqueeze(0), config.TARGET_SR)


def _clear_preprocessing_caches():
    for fn in (preprocessing.load_and_preprocess_cached,
               preprocessing.extract_segmental_features_cached,
               preprocessing.extract_suprasegmental_features_cached):
        fn.cache_clear()


def _clear_caches():
    feature_store.clear_cache()
    vad_cache.clear_span_table_cache()
    _clear_preprocessing_caches()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Four synthetic utterances with their spans in a temporary store."""
    root = tmp_path_factory.mktemp("spans")
    rows = []
    for s, speaker in enumerate(SPEAKERS):
        for w in range(2):
            name = f"{speaker}_B1_UW{w}_M6.wav"
            _write_utterance(root / name, 0.3 + 0.1 * (2 * s + w), 1.2 + 0.1 * (2 * s + w),
                             seed=2 * s + w)
            rows.append({"Filename": name, "Filepath": str(root / name), "Speaker_ID": speaker})
    df = pd.DataFrame(rows)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "FEATURE_STORE_DIR", root / "store")
        patch.setattr(config, "FEATURE_STORE_EXTRA_DIRS", [])
        patch.delenv("FEATURE_STORE_EXTRA_DIRS", raising=False)
        _clear_caches()
        feature_store.build_feature_store(df, n_workers=1)
        _clear_caches()
        yield df
    _clear_caches()


@contextlib.contextmanager
def live_vad():
    """No stored spans: every caller runs Silero."""
    original = vad_cache._span_table
    vad_cache._span_table = lambda: {}
    _clear_preprocessing_caches()
    try:
        yield
    finally:
        vad_cache._span_table = original
        _clear_preprocessing_caches()


def test_waveforms_are_bit_identical_with_and_without_stored_spans(corpus):
    for filepath in corpus["Filepath"]:
        stored = (preprocessing.load_and_preprocess(filepath),
                  preprocessing.load_and_preprocess_supra(filepath))
        with live_vad():
            live = (preprocessing.load_and_preprocess(filepath),
                    preprocessing.load_and_preprocess_supra(filepath))
        for (stored_wave, stored_len), (live_wave, live_len) in zip(stored, live):
            assert torch.equal(stored_wave, live_wave) and stored_len == live_len, filepath


def test_standardizer_statistics_are_identical_with_and_without_stored_spans(corpus):
    filepaths = corpus["Filepath"].tolist()
    stored = (preprocessing.segmental_standardizer(filepaths),
              preprocessing.suprasegmental_standardizer(filepaths))
    with live_vad():
        live = (preprocessing.segmental_standardizer(filepaths),
                preprocessing.suprasegmental_standardizer(filepaths))
    for (stored_mean, stored_std), (live_mean, live_std) in zip(stored, live):
        assert np.array_equal(stored_mean, live_mean) and np.array_equal(stored_std, live_std)


def test_vad_valid_length_matches_the_loaders(corpus):
    for filepath in corpus["Filepath"]:
        assert vad_cache.vad_valid_length(filepath) == preprocessing.load_and_preprocess(filepath)[1]
        assert (vad_cache.vad_valid_length(filepath, supra=True)
                == preprocessing.load_and_preprocess_supra(filepath)[1])


def test_lookup_keys_on_the_basename_not_the_corpus_root(corpus):
    filepath = corpus["Filepath"].iloc[0]
    name = corpus["Filename"].iloc[0]
    for elsewhere in (rf"C:\Users\someone\data\extracted\{name}", f"/mnt/data/{name}"):
        assert vad_cache.vad_span(elsewhere) == vad_cache.vad_span(filepath) is not None


def test_an_unknown_file_is_a_miss_not_an_error(corpus):
    assert vad_cache.vad_span("SYNTHZ_B1_UW0_M6.wav") is None
    assert vad_cache.vad_valid_length("SYNTHZ_B1_UW0_M6.wav") is None


def test_verification_catches_a_corrupted_span(corpus):
    table = dict(vad_cache._span_table())
    name = corpus["Filename"].iloc[0]
    n, s0, s1, p0, p1 = table[name]
    table[name] = (n, s0, s1, p0 + 160, p1)
    original = vad_cache._span_table
    vad_cache._span_table = lambda: table
    try:
        with pytest.raises(RuntimeError, match="disagree with live Silero"):
            vad_cache.verify_vad_span_cache(corpus, n=len(corpus))
    finally:
        vad_cache._span_table = original
    vad_cache.verify_vad_span_cache(corpus, n=len(corpus))            # the real spans pass


def test_a_vad_fallback_is_stored_as_the_full_span(corpus, monkeypatch):
    monkeypatch.setattr(config, "VAD_ENABLED", False)
    record = feature_store.compute_utterance_features(corpus["Filepath"].iloc[0])
    for profile in ("speech", "supra"):
        assert record[f"{profile}_fallback"]
        assert record[f"{profile}_start"] == 0
        assert record[f"{profile}_end"] == record["num_samples"]
