"""
Equality gate for the VAD span cache (src/vad_cache.py).

This is the entire safety argument for reading VAD trims from disk instead of
running Silero per item. The performance claim is uninteresting if the tensors
move, so these tests assert they do not — not "close", not "within tolerance",
but torch.equal / array_equal against the live-VAD path.

Why the claim should hold at all: src.vad.apply_vad's only effect on the signal
is waveform[:, start:end], and every one of its fallback branches returns the
untrimmed waveform, which the cache stores as (0, N). Storing two integers is
therefore a lossless description of the function, with no numerical surface —
unlike a cache of derived features, which would have one.

These tests use synthetic audio written to tmp_path rather than the UA-Speech
corpus, so they run anywhere (CI, a fresh checkout) without the dataset. The
corpus-scale equivalent is src.vad_cache.verify_vad_span_cache, which re-runs
live Silero on a sample of real utterances and is called from the training
notebook after every build.
"""

import contextlib

import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio

from src import config
from src import vad_cache

# Synthetic utterances must not collide with real UA-Speech basenames. The disk
# caches key on the basename (src.preprocessing.cache_key), so a fixture file
# called e.g. "M01_B1_UW1_M6.wav" would be answered by the REAL corpus cache
# built into outputs/feature_cache/ — returning a genuine 1.8-second
# utterance's span for a 2-second synthetic tone. This prefix is not a speaker
# in config.ALL_SPEAKER_IDS, so no such entry can exist.
SYNTHETIC_SPEAKER_PREFIX = "SYNTH"


@contextlib.contextmanager
def force_live_vad():
    """Temporarily empty the span table so callers take the live Silero path.

    Deliberately not monkeypatch: pytest's monkeypatch fixture is shared with
    the fixtures a test depends on, so monkeypatch.undo() inside a test also
    reverts the patches its fixtures made — here that would silently restore
    the REAL config.VAD_SPAN_CACHE_PATH mid-test and compare against the wrong
    cache. This restores exactly one attribute, and nothing else.
    """
    original = vad_cache._span_table
    vad_cache._span_table = lambda: {}
    try:
        yield
    finally:
        vad_cache._span_table = original
        vad_cache.clear_span_table_cache()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _write_utterance(path, seconds=2.0, speech_from=0.5, speech_to=1.4, seed=0):
    """A mono 16 kHz wav: near-silence with a burst of broadband noise in the
    middle, so Silero has something to find and a real span to trim to."""
    rng = np.random.default_rng(seed)
    n = int(seconds * config.TARGET_SR)
    signal = rng.normal(0.0, 1e-4, n).astype(np.float32)
    start, end = int(speech_from * config.TARGET_SR), int(speech_to * config.TARGET_SR)
    envelope = np.hanning(end - start).astype(np.float32)
    signal[start:end] += 0.3 * envelope * rng.normal(0.0, 1.0, end - start).astype(np.float32)
    torchaudio.save(str(path), torch.from_numpy(signal).unsqueeze(0), config.TARGET_SR)
    return path


@pytest.fixture
def corpus(tmp_path):
    """A small manifest-shaped DataFrame over synthetic utterances."""
    rows = []
    for i in range(6):
        speaker = f"{SYNTHETIC_SPEAKER_PREFIX}0{i}"
        name = f"{speaker}_B1_UW{i}_M6.wav"
        path = _write_utterance(tmp_path / name, speech_from=0.3 + 0.1 * i,
                                speech_to=1.2 + 0.1 * i, seed=i)
        rows.append({"Filename": name, "Filepath": str(path), "Speaker_ID": speaker})
    return pd.DataFrame(rows)


@pytest.fixture
def isolated_feature_caches(tmp_path, monkeypatch):
    """Redirect the framewise .npy caches into tmp_path.

    Without this the synthetic fixtures would write SYNTH*.npy entries into the
    real outputs/feature_cache/, and — worse for the equality gate — the
    formant/HNR tensor computed on the first (cached-span) pass would simply be
    re-read from disk on the second (live-span) pass, so the two would agree by
    construction rather than because the trims match.
    """
    for name in ("SEGMENTAL_EXTRA_CACHE_DIR", "SUPRASEGMENTAL_CACHE_DIR"):
        directory = tmp_path / name.lower()
        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, directory)
    return tmp_path


@pytest.fixture
def built_cache(corpus, tmp_path, monkeypatch, isolated_feature_caches):
    """Build the span table for `corpus` and point the module-level reader at
    it, restoring the real cache path afterwards."""
    cache_path = tmp_path / "vad_spans.parquet"
    vad_cache.clear_span_table_cache()
    monkeypatch.setattr(config, "VAD_SPAN_CACHE_PATH", cache_path)
    # No outputs/vad_stats.csv covers synthetic files, so this computes both
    # profiles live — which is also the path we most want under test.
    table = vad_cache.precompute_vad_span_cache(
        corpus, n_workers=1, cache_path=cache_path, seed_from_vad_stats=False,
        force=True)
    vad_cache.clear_span_table_cache()
    yield table
    vad_cache.clear_span_table_cache()


# ---------------------------------------------------------------------------
# The gate: cached trims must be bit-identical to live Silero
# ---------------------------------------------------------------------------
def test_waveforms_are_bit_identical_with_and_without_cache(corpus, built_cache):
    """load_and_preprocess and load_and_preprocess_supra must return exactly
    the same tensor and valid_length whether the span came from the table or
    from a live Silero call."""
    from src import preprocessing

    for filepath in corpus["Filepath"]:
        cached_speech = preprocessing.load_and_preprocess(filepath)
        cached_supra = preprocessing.load_and_preprocess_supra(filepath)

        with force_live_vad():
            live_speech = preprocessing.load_and_preprocess(filepath)
            live_supra = preprocessing.load_and_preprocess_supra(filepath)

        assert torch.equal(cached_speech[0], live_speech[0]), filepath
        assert cached_speech[1] == live_speech[1], filepath
        assert torch.equal(cached_supra[0], live_supra[0]), filepath
        assert cached_supra[1] == live_supra[1], filepath


def test_mfcc_and_segmental_tensors_are_bit_identical(corpus, built_cache):
    """The derived features the model actually consumes must not move either.

    Covers the full chain MFCC -> delta -> delta-delta -> concat with framewise
    formants/HNR, which is where a one-sample difference in the trim would show
    up amplified.
    """
    from src import preprocessing

    def _fresh(filepath):
        # Both tiers of cache, or the second pass would read the first pass's
        # result instead of recomputing it from its own trim.
        for cached_fn in (preprocessing.extract_mfcc_features_cached,
                          preprocessing.load_and_preprocess_cached,
                          preprocessing.load_and_preprocess_supra_cached,
                          preprocessing.extract_segmental_features_cached,
                          preprocessing.extract_segmental_extra_features_cached,
                          preprocessing.extract_suprasegmental_features_cached):
            cached_fn.cache_clear()
        for directory in (config.SEGMENTAL_EXTRA_CACHE_DIR, config.SUPRASEGMENTAL_CACHE_DIR):
            for npy in directory.glob("*.npy"):
                npy.unlink()
        return (preprocessing.extract_mfcc_features_cached(filepath),
                preprocessing.extract_segmental_features_cached(filepath))

    for filepath in corpus["Filepath"]:
        cached_mfcc, cached_segmental = _fresh(filepath)

        with force_live_vad():
            live_mfcc, live_segmental = _fresh(filepath)

        assert torch.equal(cached_mfcc, live_mfcc), filepath
        assert torch.equal(cached_segmental, live_segmental), filepath


def test_standardizers_return_identical_statistics(corpus, built_cache):
    """segmental_standardizer / suprasegmental_standardizer now take
    valid_length from the table instead of decoding each file. The (mean, std)
    they produce is fold-scoped normalization applied to every branch input, so
    a drift here would move every reported number."""
    from src.preprocessing import segmental_standardizer, suprasegmental_standardizer

    filepaths = corpus["Filepath"].tolist()
    cached_seg = segmental_standardizer(filepaths)
    cached_supra = suprasegmental_standardizer(filepaths)

    with force_live_vad():
        live_seg = segmental_standardizer(filepaths)
        live_supra = suprasegmental_standardizer(filepaths)

    for cached, live in ((cached_seg, live_seg), (cached_supra, live_supra)):
        assert np.array_equal(cached[0], live[0])
        assert np.array_equal(cached[1], live[1])


def test_vad_valid_length_matches_load_and_preprocess(corpus, built_cache):
    """vad_valid_length is claimed equal BY CONSTRUCTION to the second return
    value of the loaders (_pad_or_truncate takes min(end - start, MAX_SAMPLES)).
    This pins that claim, since two call sites now trust it instead of decoding."""
    from src.preprocessing import load_and_preprocess, load_and_preprocess_supra

    for filepath in corpus["Filepath"]:
        assert vad_cache.vad_valid_length(filepath) == load_and_preprocess(filepath)[1]
        assert (vad_cache.vad_valid_length(filepath, supra=True)
                == load_and_preprocess_supra(filepath)[1])


# ---------------------------------------------------------------------------
# Cache semantics
# ---------------------------------------------------------------------------
def test_missing_cache_is_a_miss_not_an_error(corpus, tmp_path, monkeypatch):
    """The cache is an optimization, never a correctness precondition: with no
    table on disk every lookup misses and callers take the live path."""
    from src.preprocessing import load_and_preprocess

    monkeypatch.setattr(config, "VAD_SPAN_CACHE_PATH", tmp_path / "absent.parquet")
    vad_cache.clear_span_table_cache()

    assert vad_cache.vad_span(corpus["Filepath"].iloc[0]) is None
    assert vad_cache.vad_valid_length(corpus["Filepath"].iloc[0]) is None
    waveform, valid_length = load_and_preprocess(corpus["Filepath"].iloc[0])
    assert waveform.shape == (1, config.MAX_SAMPLES)
    assert 0 < valid_length <= config.MAX_SAMPLES
    vad_cache.clear_span_table_cache()


def test_cache_built_under_a_different_config_is_refused(corpus, tmp_path, monkeypatch):
    """A span table built under different VAD settings describes a different
    trim. It must be ignored, not used — a wrong span is worse than an absent
    one, because the framewise .npy caches are derived from these spans."""
    cache_path = tmp_path / "vad_spans.parquet"
    monkeypatch.setattr(config, "VAD_SPAN_CACHE_PATH", cache_path)
    vad_cache.clear_span_table_cache()
    vad_cache.precompute_vad_span_cache(corpus, n_workers=1, cache_path=cache_path,
                                        seed_from_vad_stats=False, force=True)
    vad_cache.clear_span_table_cache()
    assert vad_cache.load_span_table(cache_path) is not None

    monkeypatch.setattr(config, "VAD_THRESHOLD", config.VAD_THRESHOLD + 0.1)
    assert vad_cache.load_span_table(cache_path) is None
    vad_cache.clear_span_table_cache()
    assert vad_cache.vad_span(corpus["Filepath"].iloc[0]) is None
    vad_cache.clear_span_table_cache()


def test_verify_vad_span_cache_catches_a_corrupted_span(corpus, tmp_path, monkeypatch):
    """verify_vad_span_cache must actually fail on a wrong span — otherwise it
    is decoration rather than a check."""
    cache_path = tmp_path / "vad_spans.parquet"
    monkeypatch.setattr(config, "VAD_SPAN_CACHE_PATH", cache_path)
    vad_cache.clear_span_table_cache()
    table = vad_cache.precompute_vad_span_cache(corpus, n_workers=1, cache_path=cache_path,
                                                seed_from_vad_stats=False, force=True)

    corrupted = table.copy()
    corrupted.loc[0, "supra_start"] = int(corrupted.loc[0, "supra_start"]) + 160
    vad_cache.write_span_table(corrupted, cache_path)
    vad_cache.clear_span_table_cache()

    with pytest.raises(RuntimeError, match="disagrees with live Silero"):
        vad_cache.verify_vad_span_cache(corpus, n=len(corpus))
    vad_cache.clear_span_table_cache()


def test_duplicate_basenames_are_rejected(corpus, tmp_path):
    """src.preprocessing.cache_key keys the disk caches on the basename, so two
    manifest rows sharing one would silently share cache entries."""
    duplicated = pd.concat([corpus, corpus.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        vad_cache.precompute_vad_span_cache(duplicated, n_workers=1,
                                            cache_path=tmp_path / "x.parquet",
                                            seed_from_vad_stats=False, force=True)


def test_fallback_rows_are_stored_as_the_full_span(tmp_path, monkeypatch):
    """apply_vad's fallbacks return the untrimmed waveform; the cache must
    record that as (0, N) rather than a sentinel the read path branches on."""
    path = _write_utterance(tmp_path / "M99_B1_UW1_M6.wav", seconds=1.0,
                            speech_from=0.4, speech_to=0.5, seed=99)
    df = pd.DataFrame([{"Filename": path.name, "Filepath": str(path)}])

    cache_path = tmp_path / "vad_spans.parquet"
    monkeypatch.setattr(config, "VAD_SPAN_CACHE_PATH", cache_path)
    monkeypatch.setattr(config, "VAD_ENABLED", False)          # forces the fallback branch
    vad_cache.clear_span_table_cache()

    table = vad_cache.precompute_vad_span_cache(df, n_workers=1, cache_path=cache_path,
                                                seed_from_vad_stats=False, force=True)
    row = table.iloc[0]
    assert bool(row["speech_fallback"]) and bool(row["supra_fallback"])
    assert row["speech_start"] == 0 and row["speech_end"] == row["num_samples"]
    assert row["supra_start"] == 0 and row["supra_end"] == row["num_samples"]
    vad_cache.clear_span_table_cache()


def test_cache_key_is_independent_of_the_corpus_root():
    """The key must not depend on where the corpus lives, or a cache built on
    one machine can never be read on another — which was the state of the disk
    caches before, and the reason a shipped Kaggle cache missed 100%."""
    from src.preprocessing import _disk_cache_path, cache_key

    windows = r"C:\Users\someone\project\data\extracted\M01\M01_B1_UW3_M6.wav"
    kaggle = "/kaggle/working/project/data/extracted/M01/M01_B1_UW3_M6.wav"
    assert cache_key(windows) == cache_key(kaggle) == "M01_B1_UW3_M6.wav"
    assert (_disk_cache_path(config.SEGMENTAL_EXTRA_CACHE_DIR, windows)
            == _disk_cache_path(config.SEGMENTAL_EXTRA_CACHE_DIR, kaggle))
