"""
Disk-persisted Silero VAD spans — the one cache that makes training GPU-bound.

WHY THIS EXISTS
---------------
src.preprocessing's two profiles each run Silero VAD over the utterance:
load_and_preprocess (speech-focused, config.VAD_SPEECH_PAD_MS) and
load_and_preprocess_supra (temporal-preserving, config.SUPRA_VAD_SPEECH_PAD_MS).
Both were memoized in memory only (an lru_cache per process), so every
DataLoader worker restart — and every LRU eviction, which on a 9,639-utterance
shuffled train split is most items — paid them again. That is TWO neural
forward passes per utterance per epoch, roughly 21,400 per epoch.

Measured on the real corpus, one item's CPU cost broke down as:

    load + resample (x2)   15.7 ms    29.8%
    Silero VAD      (x2)   42.6 ms    80.8%   (includes its second file load)
    pad + MFCC              2.3 ms     4.4%

and on a Kaggle T4 the consequence was a flat 8.5 samples/sec at batch sizes
16, 24 AND 32, using 4.2 GB of the GPU's 15 GB. Throughput that does not move
with batch size is the signature of a starved GPU, not a busy one.

WHY STORING SPANS IS NOT AN APPROXIMATION
-----------------------------------------
src.vad.apply_vad's ONLY effect on the signal is waveform[:, start:end] (see
its final lines), and every one of its five fallback branches (vad_disabled,
vad_init_failed, no_speech_detected, trimmed_span_too_short, exception)
returns the original waveform unchanged — i.e. the span (0, N). So the
function is fully described by two integers per (utterance, profile), and
slicing from stored integers is BIT-IDENTICAL to re-running the model. There
is no numerical surface here at all, which is what separates this cache from
one that stores derived features in a reduced precision.

Fallbacks are therefore stored as (0, N) with a separate boolean column, kept
for diagnostics only — never as a sentinel the read path has to branch on.

CONTRACT
--------
The cache is an OPTIMIZATION, never a correctness precondition. Every lookup
returns None on a miss and every caller falls through to the live Silero path.
Delete outputs/feature_cache/vad_spans.parquet and the pipeline still produces
identical numbers, slowly.

STALENESS
---------
The stored spans depend on the Silero release and on five config values. A
cache built under different settings would be silently WRONG rather than
merely absent — and worse, the framewise .npy caches
(config.SEGMENTAL_EXTRA_CACHE_DIR / SUPRASEGMENTAL_CACHE_DIR) are themselves
derived from these spans, so a mismatch would desynchronise the formant/HNR
channels from the MFCC channels they are concatenated to in
extract_segmental_features_cached. span_cache_signature() is therefore written
into the parquet's key-value metadata and checked on load; a mismatch is
refused loudly and the cache ignored.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config
from src import vad as vad_module
from src.console import (print_kv, print_note, print_status, print_subheader,
                         progress)
from src.parallel import resilient_process_map

# Bump when the stored COLUMN LAYOUT changes (not when a VAD tunable changes —
# those travel in the signature below and are compared value-by-value).
SPAN_SCHEMA_VERSION = 1

SPAN_COLUMNS = ["Filename", "num_samples",
                "speech_start", "speech_end", "speech_fallback",
                "supra_start", "supra_end", "supra_fallback"]

_SIGNATURE_METADATA_KEY = b"vad_span_cache_signature"


def span_cache_signature() -> Dict[str, object]:
    """Every input the stored spans depend on.

    Written into the parquet metadata at build time and compared on load. If
    any of these changes, previously-cached spans describe a different trim
    than the code would now compute, and must not be used — see the module
    docstring on why a wrong span is worse than an absent one.
    """
    return {
        "schema": SPAN_SCHEMA_VERSION,
        "vad_repo": vad_module.VAD_REPO,
        "target_sr": config.TARGET_SR,
        "vad_enabled": bool(config.VAD_ENABLED),
        "vad_threshold": config.VAD_THRESHOLD,
        "vad_min_speech_ms": config.VAD_MIN_SPEECH_MS,
        "vad_min_silence_ms": config.VAD_MIN_SILENCE_MS,
        "vad_speech_pad_ms": config.VAD_SPEECH_PAD_MS,
        "supra_vad_speech_pad_ms": config.SUPRA_VAD_SPEECH_PAD_MS,
    }


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def _live_span(waveform, speech_pad_ms: Optional[int]) -> Tuple[int, int, bool]:
    """(start, end, fallback_used) for one profile, by actually running Silero.

    Derived from apply_vad's own return values rather than by reimplementing
    its segment-selection logic, so the two can never drift apart: the span is
    recovered from the trimmed length it produced plus the leading trim it
    reported.
    """
    n = int(waveform.shape[-1])
    trimmed, stats = vad_module.apply_vad(
        waveform, config.TARGET_SR, speech_pad_ms=speech_pad_ms)
    if stats["fallback_used"]:
        return 0, n, True
    start = int(round(stats["leading_trimmed_s"] * config.TARGET_SR))
    end = start + int(trimmed.shape[-1])
    return start, end, False


def _span_for_one(filepath: str) -> Dict[str, object]:
    """Both profiles' spans for one file. Module-level (and therefore
    picklable) so it is safe to hand to ProcessPoolExecutor on Windows — the
    same reason src.preprocessing._precompute_one is module-level."""
    from src.preprocessing import _load_resampled, cache_key

    waveform, _ = _load_resampled(filepath)
    speech_start, speech_end, speech_fallback = _live_span(waveform, None)
    supra_start, supra_end, supra_fallback = _live_span(
        waveform, config.SUPRA_VAD_SPEECH_PAD_MS)
    return {
        "Filename": cache_key(filepath),
        "num_samples": int(waveform.shape[-1]),
        "speech_start": speech_start, "speech_end": speech_end,
        "speech_fallback": speech_fallback,
        "supra_start": supra_start, "supra_end": supra_end,
        "supra_fallback": supra_fallback,
    }


def _supra_span_for_one(filepath: str) -> Dict[str, object]:
    """As _span_for_one, but ONLY the temporal-preserving profile — used when
    the speech-focused profile is being seeded from outputs/vad_stats.csv
    (see precompute_vad_span_cache), which halves the build."""
    from src.preprocessing import _load_resampled, cache_key

    waveform, _ = _load_resampled(filepath)
    supra_start, supra_end, supra_fallback = _live_span(
        waveform, config.SUPRA_VAD_SPEECH_PAD_MS)
    return {
        "Filename": cache_key(filepath),
        "num_samples": int(waveform.shape[-1]),
        "supra_start": supra_start, "supra_end": supra_end,
        "supra_fallback": supra_fallback,
    }


def _seed_speech_profile(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Recover the SPEECH-FOCUSED spans from outputs/vad_stats.csv, if it
    covers every row of df.

    compute_vad_stats_batch wrote that file with apply_vad's DEFAULT
    speech_pad_ms — i.e. exactly this profile — recording
    leading_trimmed_s = start/sr and trailing_trimmed_s = (N - end)/sr. Both
    invert exactly, so the spans come back integer-for-integer with no VAD
    re-run; num_samples likewise inverts from original_duration_s. The
    inversion is not taken on trust: precompute_vad_span_cache's caller is
    expected to run verify_vad_span_cache afterwards, which re-runs live
    Silero on a sample and would catch a stale or mismatched stats file.

    Returns None (and the caller computes both profiles live) if the file is
    absent or does not cover every utterance in df.
    """
    stats_path = Path(config.VAD_STATS_PATH)
    if not stats_path.exists():
        return None

    stats = pd.read_csv(stats_path)
    needed = set(df["Filename"])
    if not needed.issubset(set(stats["Filename"])):
        return None

    sr = config.TARGET_SR
    stats = stats[stats["Filename"].isin(needed)].copy()
    num_samples = (stats["original_duration_s"] * sr).round().astype(np.int64)
    leading = (stats["leading_trimmed_s"] * sr).round().astype(np.int64)
    trailing = (stats["trailing_trimmed_s"] * sr).round().astype(np.int64)
    fallback = stats["fallback_used"].astype(bool).to_numpy()

    return pd.DataFrame({
        "Filename": stats["Filename"].to_numpy(),
        "num_samples": num_samples.to_numpy(),
        # A fallback row was never trimmed, so its span is the whole waveform.
        # leading/trailing are already 0.0 for those rows (see
        # src.vad._fallback_stats), but being explicit keeps the invariant
        # readable rather than inferred.
        "speech_start": np.where(fallback, 0, leading).astype(np.int64),
        "speech_end": np.where(fallback, num_samples,
                               num_samples - trailing).astype(np.int64),
        "speech_fallback": fallback,
    })


def precompute_vad_span_cache(df: pd.DataFrame, n_workers: int = 4,
                              cache_path: Optional[Path] = None,
                              seed_from_vad_stats: bool = True,
                              force: bool = False) -> pd.DataFrame:
    """
    One-time, parallel build of the Filename -> (start, end) span table for
    both preprocessing profiles, over every row of `df` (M6 manifest shape:
    Filename, Filepath).

    Silero is lazily torch.hub-loaded on first use. If that first use happened
    independently inside each of n_workers worker processes, they would race on
    the same download-extract-rename and corrupt the shared hub cache — the
    "No such file or directory ... hubconf.py" failure documented at length in
    src/vad.py's module docstring, which then repeats once per file for all
    21,420 files. warmup_silero_vad() is therefore called HERE, synchronously,
    in the main process, BEFORE the executor is created, exactly as
    src.preprocessing.precompute_framewise_feature_cache does.

    seed_from_vad_stats=True reuses outputs/vad_stats.csv for the
    speech-focused profile (see _seed_speech_profile), leaving only the 150 ms
    temporal-preserving profile to compute — roughly half the work. Pass False
    to compute both profiles from scratch under one provenance.

    Returns the span table, and writes it atomically to `cache_path`
    (config.VAD_SPAN_CACHE_PATH by default).
    """
    cache_path = Path(cache_path or config.VAD_SPAN_CACHE_PATH)

    if not df["Filename"].is_unique:
        duplicated = df.loc[df["Filename"].duplicated(), "Filename"].unique()[:5]
        raise ValueError(
            f"src.preprocessing.cache_key assumes basenames are unique across "
            f"the manifest, but {int(df['Filename'].duplicated().sum())} are "
            f"repeated (e.g. {list(duplicated)}). The disk caches key on the "
            f"basename, so duplicates would silently share cache entries."
        )

    if not force and cache_path.exists():
        existing = load_span_table(cache_path)
        if existing is not None and set(df["Filename"]).issubset(set(existing["Filename"])):
            print_kv("VAD span cache", f"loaded from cache ({cache_path})")
            return existing

    print_subheader(f"VAD span cache — {len(df):,} utterances")
    if config.VAD_ENABLED:
        vad_module.warmup_silero_vad()

    seeded = _seed_speech_profile(df) if seed_from_vad_stats else None
    if seeded is not None:
        print_kv("Speech-focused profile",
                 f"seeded from {Path(config.VAD_STATS_PATH).name} (no VAD re-run)")
        worker_fn = _supra_span_for_one
        description = "Computing temporal-preserving VAD spans"
    else:
        if seed_from_vad_stats:
            print_note(f"{Path(config.VAD_STATS_PATH).name} is absent or does not cover "
                       f"every utterance — computing both profiles live.")
        worker_fn = _span_for_one
        description = "Computing VAD spans (both profiles)"

    filepaths = df["Filepath"].tolist()
    records, skipped = resilient_process_map(
        worker_fn, filepaths, n_workers=n_workers,
        max_tasks_per_child=config.PRECOMPUTE_MAX_TASKS_PER_CHILD,
        description=description, unit="file",
    )
    if skipped:
        print_note(
            f"{len(skipped)} of {len(filepaths):,} file(s) were skipped after "
            f"stalling or failing (see above for which) — they are simply "
            f"absent from the span cache and fall back to live VAD at read "
            f"time. Re-run precompute_vad_span_cache(df, force=True) later to "
            f"fill them in, ideally after checking those specific files play "
            f"back correctly."
        )

    table = pd.DataFrame.from_records(records)
    if seeded is not None:
        table = table.drop(columns=["num_samples"]).merge(seeded, on="Filename", how="inner")
        expected_len = len(df) - len(skipped)
        if len(table) != expected_len:
            raise RuntimeError(
                f"Seeded merge lost rows ({len(table):,} of {expected_len:,} "
                f"expected) — {config.VAD_STATS_PATH} and the manifest "
                f"disagree on Filename. Re-run with seed_from_vad_stats=False."
            )
    table = table[SPAN_COLUMNS]

    _assert_spans_wellformed(table)
    write_span_table(table, cache_path)
    clear_span_table_cache()

    n_fallback = int(table["speech_fallback"].sum())
    print_status(f"{len(table):,} spans cached "
                 f"({n_fallback} speech-profile fallbacks to the untrimmed waveform)",
                 ok=True)
    print_kv("VAD span cache written",
             f"{cache_path} ({cache_path.stat().st_size / 1024:.0f} KB)")
    return table


def _assert_spans_wellformed(table: pd.DataFrame) -> None:
    """Structural invariants that must hold before anything is written: spans
    lie inside the waveform, are non-empty, and are ordered. A violation means
    the build is broken, and writing it would poison every later read."""
    for profile in ("speech", "supra"):
        start = table[f"{profile}_start"]
        end = table[f"{profile}_end"]
        bad = (start < 0) | (end > table["num_samples"]) | (end <= start)
        if bad.any():
            columns = ["Filename", "num_samples", f"{profile}_start", f"{profile}_end"]
            raise RuntimeError(
                f"{int(bad.sum())} malformed {profile}-profile spans, e.g.\n"
                f"{table.loc[bad, columns].head()}"
            )


def write_span_table(table: pd.DataFrame, cache_path: Path) -> None:
    """Write `table` plus the current signature, atomically — a half-written
    parquet left by an interrupted Kaggle session would otherwise be read back
    as a corrupt cache on the next run."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    arrow_table = pa.Table.from_pandas(table[SPAN_COLUMNS], preserve_index=False)
    metadata = dict(arrow_table.schema.metadata or {})
    metadata[_SIGNATURE_METADATA_KEY] = json.dumps(span_cache_signature()).encode("utf-8")
    arrow_table = arrow_table.replace_schema_metadata(metadata)

    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    pq.write_table(arrow_table, temp_path, compression="zstd")
    os.replace(temp_path, cache_path)


def load_span_table(cache_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """The cached span table, or None if it is absent, unreadable, or was built
    under a different VAD configuration.

    A signature mismatch is reported and the cache IGNORED rather than
    repaired: the correct response is to rebuild it, and the framewise .npy
    caches derived from it, which is a decision for the caller and not a
    side-effect of a read.
    """
    cache_path = Path(cache_path or config.VAD_SPAN_CACHE_PATH)
    if not cache_path.exists():
        return None

    try:
        arrow_table = pq.read_table(cache_path)
    except Exception as exc:
        print_note(f"VAD span cache at {cache_path} could not be read ({exc}) — "
                   f"falling back to live VAD.")
        return None

    metadata = arrow_table.schema.metadata or {}
    stored = metadata.get(_SIGNATURE_METADATA_KEY)
    found = json.loads(stored.decode("utf-8")) if stored else None
    expected = span_cache_signature()
    if found != expected:
        print_note(
            f"VAD span cache at {cache_path} was built under a different VAD "
            f"configuration (expected {expected}, found {found}) — ignoring it "
            f"and falling back to live VAD. Rebuild it, and the framewise "
            f"feature caches derived from it, with "
            f"precompute_vad_span_cache(df, force=True)."
        )
        return None

    return arrow_table.to_pandas()


# ---------------------------------------------------------------------------
# Reading — the hot path
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _span_table() -> Dict[str, Tuple[int, int, int, int, int]]:
    """Filename -> (num_samples, speech_start, speech_end, supra_start, supra_end).

    Loaded lazily, once per process — which means once in the main process and
    once in each DataLoader worker. Deliberately NOT passed into
    UASpeechDataset.__init__: a dict handed to the Dataset is pickled into
    every worker at spawn and again on every persistent_workers respawn,
    whereas a lazy module-level load is a single local parquet read.

    Returns {} when the cache is absent or stale, so every lookup below misses
    and callers take the live-VAD path.
    """
    table = load_span_table()
    if table is None:
        return {}
    columns = ["Filename", "num_samples", "speech_start", "speech_end",
               "supra_start", "supra_end"]
    return {
        row[0]: (int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))
        for row in table[columns].itertuples(index=False, name=None)
    }


def clear_span_table_cache() -> None:
    """Drop the per-process memoized table — for tests, and after a rebuild."""
    _span_table.cache_clear()


def vad_span(filepath: str, *, supra: bool = False) -> Optional[Tuple[int, int]]:
    """(start, end) sample indices into the post-mono-mixdown, post-resample
    waveform for one profile, or None on a cache miss.

    supra=False is the speech-focused profile (config.VAD_SPEECH_PAD_MS, used
    by the Learned and Segmental branches); supra=True is the
    temporal-preserving profile (config.SUPRA_VAD_SPEECH_PAD_MS, used by the
    Suprasegmental branch).
    """
    from src.preprocessing import cache_key

    entry = _span_table().get(cache_key(filepath))
    if entry is None:
        return None
    _, speech_start, speech_end, supra_start, supra_end = entry
    return (supra_start, supra_end) if supra else (speech_start, speech_end)


def vad_valid_length(filepath: str, *, supra: bool = False) -> Optional[int]:
    """min(end - start, config.MAX_SAMPLES) — i.e. exactly the second return
    value of load_and_preprocess / load_and_preprocess_supra, with no file
    read, no decode and no VAD.

    src.preprocessing._pad_or_truncate defines valid_length as
    min(trimmed.shape[1], MAX_SAMPLES), and trimmed.shape[1] is end - start, so
    this is equal by construction rather than by approximation.

    This is the lookup that removes the two largest non-obvious costs in the
    pipeline: src.training.data.build_loaders' per-fold standardizer passes
    (19,278 decode+VAD calls per fold, on the main thread, to obtain two
    integers per file) and UASpeechDataset.__getitem__'s second audio load.
    """
    span = vad_span(filepath, supra=supra)
    if span is None:
        return None
    start, end = span
    return min(end - start, config.MAX_SAMPLES)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_vad_span_cache(df: pd.DataFrame, n: int = 200, seed: int = 0) -> None:
    """Re-run live Silero on `n` randomly chosen utterances and assert the
    cached spans match exactly.

    Not optional polish. The framewise .npy caches were built by slicing with
    these spans, so a span table that is wrong rather than absent would leave
    the formant/HNR channels describing a different waveform than the MFCC
    channels they are concatenated to in extract_segmental_features_cached —
    a corruption no shape check and no unit test would catch. Run after every
    build, and after any change to Silero or to a VAD tunable.
    """
    from src.preprocessing import _load_resampled, cache_key

    table = _span_table()
    if not table:
        raise RuntimeError(
            f"No usable VAD span cache at {config.VAD_SPAN_CACHE_PATH} — build "
            f"it with precompute_vad_span_cache(df) first."
        )

    if config.VAD_ENABLED:
        vad_module.warmup_silero_vad()

    rows = df.sample(n=min(n, len(df)), random_state=seed)
    mismatches = []
    for filepath in progress(rows["Filepath"].tolist(),
                             f"Verifying {len(rows):,} cached VAD spans",
                             total=len(rows), unit="file"):
        entry = table.get(cache_key(filepath))
        if entry is None:
            mismatches.append((filepath, "absent from cache", ""))
            continue
        num_samples, speech_start, speech_end, supra_start, supra_end = entry

        waveform, _ = _load_resampled(filepath)
        if int(waveform.shape[-1]) != num_samples:
            mismatches.append((filepath, f"num_samples {num_samples}",
                               f"live {int(waveform.shape[-1])}"))
        live_speech = _live_span(waveform, None)[:2]
        live_supra = _live_span(waveform, config.SUPRA_VAD_SPEECH_PAD_MS)[:2]
        if live_speech != (speech_start, speech_end):
            mismatches.append((filepath, f"speech {(speech_start, speech_end)}",
                               f"live {live_speech}"))
        if live_supra != (supra_start, supra_end):
            mismatches.append((filepath, f"supra {(supra_start, supra_end)}",
                               f"live {live_supra}"))

    if mismatches:
        detail = "\n".join(f"  {Path(fp).name}: cached {cached}, {live}"
                           for fp, cached, live in mismatches[:10])
        raise RuntimeError(
            f"VAD span cache disagrees with live Silero on {len(mismatches)} of "
            f"{len(rows)} sampled utterances:\n{detail}\n"
            f"The cache is WRONG, not merely stale. Rebuild it with "
            f"precompute_vad_span_cache(df, force=True), and rebuild the "
            f"framewise feature caches derived from it."
        )
    print_status(f"All {len(rows):,} sampled spans match live Silero exactly", ok=True)
