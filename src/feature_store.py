"""
Persistent, chunked, resumable store of every per-utterance model input
except the waveform itself — the cache that keeps preprocessing out of
training entirely.

WHY THIS EXISTS
---------------
The audited Kaggle run cached 800 of 21,420 utterances before its precompute
pool stalled (see src.parallel), so training fell back to live Silero VAD and
live Praat for the rest: ~3 min epochs against a calibrated ~1m45s, and a
7-12 minute gap before EVERY fold while build_loaders' standardizer passes
re-derived features one file at a time on the main thread. Two properties of
the old caches made that failure expensive: a single-file parquet span table
written only at the very end of its pass (an interrupted pass kept nothing),
and ~43,000 per-file .npy features with no MFCC among them.

WHAT IS STORED
--------------
One compressed .npz CHUNK per UA-Speech (speaker, block) — e.g. "M05_B1" —
holding, for each of that recording block's utterances:

    num_samples, speech/supra VAD spans (+ fallback flags)   exactly the
        src.vad_cache span-table columns;
    segmental  (43, T) float32   MFCC + delta + delta-delta + F1-F3 + HNR
        (src.preprocessing.extract_segmental_features_cached's output);
    supra      (3, T)  float32   F0 semitones, voicing, intensity dB
        (extract_suprasegmental_features_cached's output).

Raw (pre-normalization) values: fold-level standardization still happens at
__getitem__ time from each fold's TRAIN split only, exactly as before.

Chunking by (speaker, block) is what makes it resumable and shareable:
  * a chunk is written atomically (temp file + rename) the moment it is
    finished, so an interrupted session keeps every completed chunk and the
    next one computes only the missing ones — never a completed chunk twice;
  * the partition does not depend on which subset a run uses — a B1 severity
    run needs the 15 "<dysarthric speaker>_B1" chunks, a later B1+B2 run adds
    15 more, and none are rebuilt;
  * 84 chunk files for the whole M6 corpus — easy to commit as a Kaggle
    dataset and attach read-only to the next session (config.
    FEATURE_STORE_EXTRA_DIRS / the FEATURE_STORE_EXTRA_DIRS env var).

CORRECTNESS CONTRACT
--------------------
The store is an optimization, never a precondition: every read returns None
on a miss and callers fall through to the original live computation. A chunk
built under a different configuration (store_signature) is ignored, loudly.
verify_feature_store recomputes a random sample through the ORIGINAL live
code path (Silero via src.vad.apply_vad, no span table, no store) and
requires bit-exact equality; src.vad_cache.verify_vad_span_cache (the 300-
sample span check) runs unchanged against the spans served from here.
"""

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.console import (ProgressReporter, format_duration, print_kv, print_note,
                         print_status, print_subheader)

# Bump when the stored LAYOUT or the framewise extraction code
# (src.praat.extract_*_sequence, src.preprocessing.extract_mfcc_features)
# changes in a way the tunables in store_signature() do not capture.
STORE_SCHEMA_VERSION = 1

SPAN_ARRAYS = ("num_samples", "speech_start", "speech_end", "speech_fallback",
               "supra_start", "supra_end", "supra_fallback")


def store_signature() -> Dict[str, object]:
    """Every configuration value the stored arrays depend on. Written into
    each chunk and compared on load; a mismatch means the chunk describes
    features the current code would not produce, so it is ignored."""
    from src import praat
    from src import vad_cache
    return {
        "schema": STORE_SCHEMA_VERSION,
        "vad": vad_cache.span_cache_signature(),
        "max_samples": config.MAX_SAMPLES,
        "n_mfcc": config.N_MFCC,
        "mel_kwargs": dict(config.MEL_KWARGS),
        "segmental_channels": config.SEGMENTAL_CHANNELS,
        "supra_channels": config.SUPRA_CHANNELS,
        "praat": {"pitch_floor": praat.PITCH_FLOOR, "pitch_ceiling": praat.PITCH_CEILING,
                  "frame_hop_s": praat.FRAME_HOP_SECONDS,
                  "hnr_silence_floor": praat.HNR_SILENCE_FLOOR},
    }


def _signature_json() -> str:
    return json.dumps(store_signature(), sort_keys=True)


def _provenance() -> Dict[str, object]:
    """Library versions a chunk was built with — recorded, NOT part of the
    signature: a different torch/parselmouth build is caught empirically by
    verify_feature_store's bit-exact check rather than by refusing outright."""
    import platform
    import parselmouth
    import torch
    import torchaudio
    return {"torch": torch.__version__, "torchaudio": torchaudio.__version__,
            "parselmouth": parselmouth.__version__, "python": platform.python_version(),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


# ---------------------------------------------------------------------------
# Chunk identity and location
# ---------------------------------------------------------------------------
def chunk_key(filename_or_path: str) -> str:
    """'<Speaker>_<Block>' from a UA-Speech basename
    '<Speaker>_<Block>_<Word>_<Mic>.wav' — e.g. 'M05_B1'."""
    parts = Path(filename_or_path).name.split("_")
    if len(parts) < 4:
        raise ValueError(f"Not a UA-Speech <Speaker>_<Block>_<Word>_<Mic> name: {filename_or_path}")
    return f"{parts[0]}_{parts[1]}"


def store_dirs() -> List[Path]:
    """Writable store first, then read-only extra locations (e.g. a previous
    Kaggle session's output attached as a dataset). The env var
    FEATURE_STORE_EXTRA_DIRS (os.pathsep-separated) adds to the config list —
    read at call time so a notebook can set it after import."""
    extra = list(config.FEATURE_STORE_EXTRA_DIRS)
    extra += [p for p in os.environ.get("FEATURE_STORE_EXTRA_DIRS", "").split(os.pathsep) if p]
    return [Path(config.FEATURE_STORE_DIR)] + [Path(p) for p in extra]


def _chunk_path(key: str) -> Optional[Path]:
    for directory in store_dirs():
        path = directory / f"{key}.npz"
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Reading — the hot path
# ---------------------------------------------------------------------------
@dataclass
class _Chunk:
    index: Dict[str, int]
    spans: Dict[str, np.ndarray]
    segmental: np.ndarray        # (N, 43, T) float32
    supra: np.ndarray            # (N, 3, T) float32


_WARNED: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print_note(message)


def _signature_ok(data, path: Path) -> bool:
    found = str(data["signature"]) if "signature" in data.files else None
    if found != _signature_json():
        _warn_once(str(path), f"Feature-store chunk {path} was built under a different "
                              f"configuration — ignoring it (it will be rebuilt).")
        return False
    return True


@lru_cache(maxsize=None)
def _load_chunk(key: str) -> Optional[_Chunk]:
    """Whole chunk into memory, once per process (~19 MB per 255-utterance
    chunk). DataLoader workers forked after the main process has loaded a
    chunk share its pages copy-on-write."""
    path = _chunk_path(key)
    if path is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if not _signature_ok(data, path):
                return None
            filenames = [str(name) for name in data["filenames"]]
            return _Chunk(index={name: i for i, name in enumerate(filenames)},
                          spans={name: data[name] for name in SPAN_ARRAYS},
                          segmental=data["segmental"], supra=data["supra"])
    except Exception as exc:
        _warn_once(str(path), f"Feature-store chunk {path} is unreadable ({exc}) — ignoring it.")
        return None


def clear_cache() -> None:
    """Drop every per-process memoized chunk and span table (tests, rebuilds)."""
    _load_chunk.cache_clear()
    span_table.cache_clear()


def _locate(filepath: str) -> Optional[Tuple[_Chunk, int]]:
    name = Path(filepath).name
    try:
        chunk = _load_chunk(chunk_key(name))
    except ValueError:
        return None
    if chunk is None or name not in chunk.index:
        return None
    return chunk, chunk.index[name]


def segmental_features(filepath: str) -> Optional[np.ndarray]:
    """(43, T) float32 raw segmental input, or None on a miss."""
    hit = _locate(filepath)
    return None if hit is None else hit[0].segmental[hit[1]]


def suprasegmental_features(filepath: str) -> Optional[np.ndarray]:
    """(3, T) float32 raw suprasegmental input, or None on a miss."""
    hit = _locate(filepath)
    return None if hit is None else hit[0].supra[hit[1]]


@lru_cache(maxsize=1)
def span_table() -> Dict[str, Tuple[int, int, int, int, int]]:
    """Filename -> (num_samples, speech_start, speech_end, supra_start,
    supra_end) for every chunk in every store directory — the same tuple
    layout as src.vad_cache._span_table, which merges this in. Reads only
    the small span arrays of each chunk, not its features."""
    table: Dict[str, Tuple[int, int, int, int, int]] = {}
    seen = set()
    for directory in store_dirs():
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.npz")):
            if path.name in seen:              # the writable store shadows extra dirs
                continue
            seen.add(path.name)
            try:
                with np.load(path, allow_pickle=False) as data:
                    if not _signature_ok(data, path):
                        continue
                    columns = [data[name] for name in ("filenames", "num_samples", "speech_start",
                                                       "speech_end", "supra_start", "supra_end")]
            except Exception as exc:
                _warn_once(str(path), f"Feature-store chunk {path} is unreadable ({exc}) — ignoring it.")
                continue
            for name, n, s0, s1, p0, p1 in zip(*columns):
                table[str(name)] = (int(n), int(s0), int(s1), int(p0), int(p1))
    return table


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def _worker_init() -> None:
    """Single intra-op thread per worker (n_workers processes already use
    every core; see src.parallel._worker_thread_init) and no library chatter."""
    import warnings
    import torch
    torch.set_num_threads(1)
    warnings.filterwarnings("ignore")


def compute_utterance_features(filepath: str) -> Dict[str, object]:
    """Every stored field for one utterance, from the audio. Spans come from
    live Silero (src.vad_cache._live_span); features from the same functions
    the cached read path uses, applied to the span-sliced waveform exactly as
    src.preprocessing._trim + _pad_or_truncate would."""
    from src import praat
    from src.preprocessing import (_load_resampled, _pad_or_truncate, _shared_mfcc_transform,
                                   extract_mfcc_features, mfcc_frame_count)
    from src.vad_cache import _live_span

    waveform, _ = _load_resampled(filepath)
    num_samples = int(waveform.shape[-1])
    speech_start, speech_end, speech_fallback = _live_span(waveform, None)
    supra_start, supra_end, supra_fallback = _live_span(waveform, config.SUPRA_VAD_SPEECH_PAD_MS)

    speech_wave, valid_length = _pad_or_truncate(waveform[:, speech_start:speech_end])
    supra_wave, supra_valid_length = _pad_or_truncate(waveform[:, supra_start:supra_end])
    total_frames = mfcc_frame_count(speech_wave.shape[-1])

    mfcc = extract_mfcc_features(speech_wave, _shared_mfcc_transform(),
                                 valid_length=valid_length).squeeze(0).numpy()      # (39, T)
    extra = praat.extract_segmental_extra_sequence(
        speech_wave.squeeze(0).numpy(), config.TARGET_SR, valid_length, total_frames)
    supra = praat.extract_suprasegmental_sequence(
        supra_wave.squeeze(0).numpy(), config.TARGET_SR, supra_valid_length, total_frames)

    segmental = np.concatenate([
        mfcc.astype(np.float32),
        np.stack([extra["f1_hz"], extra["f2_hz"], extra["f3_hz"], extra["hnr_db"]],
                 axis=0).astype(np.float32)], axis=0)
    suprasegmental = np.stack([supra["f0_semitones"], supra["voicing"],
                               supra["intensity_db"]], axis=0).astype(np.float32)
    return {"filename": Path(filepath).name, "num_samples": num_samples,
            "speech_start": speech_start, "speech_end": speech_end,
            "speech_fallback": speech_fallback,
            "supra_start": supra_start, "supra_end": supra_end,
            "supra_fallback": supra_fallback,
            "segmental": segmental, "supra": suprasegmental}


def _write_chunk(path: Path, records: List[Dict[str, object]]) -> None:
    """Atomic: a killed session leaves either the previous state or the whole
    new chunk, never a truncated file under the final name."""
    arrays = {
        "filenames": np.array([r["filename"] for r in records]),
        "segmental": np.stack([r["segmental"] for r in records]),
        "supra": np.stack([r["supra"] for r in records]),
        "signature": np.array(_signature_json()),
        "provenance": np.array(json.dumps(_provenance())),
    }
    for name in SPAN_ARRAYS:
        dtype = bool if name.endswith("fallback") else np.int64
        arrays[name] = np.array([r[name] for r in records], dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with open(temp_path, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp_path, path)


def _build_chunk(task: Tuple[str, List[str], str]) -> Dict[str, object]:
    """Worker entry point (module-level, so picklable under 'spawn'): compute
    one chunk's utterances in order and write the chunk file itself, so the
    parent never has to receive ~19 MB of arrays per chunk."""
    import gc
    key, filepaths, out_dir = task
    start = time.monotonic()
    records = []
    for i, filepath in enumerate(filepaths, start=1):
        records.append(compute_utterance_features(filepath))
        if i % 100 == 0:
            gc.collect()     # parselmouth objects hold C++ state; bound RSS growth
    _write_chunk(Path(out_dir) / f"{key}.npz", records)
    return {"key": key, "n_files": len(filepaths), "seconds": time.monotonic() - start}


def _chunk_is_complete(key: str, filenames: List[str]) -> bool:
    """A chunk counts as done only if it exists, matches the current
    signature, and holds every requested utterance — a chunk built from a
    partial subset is rebuilt when a larger one asks for it."""
    path = _chunk_path(key)
    if path is None:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if not _signature_ok(data, path):
                return False
            stored = {str(name) for name in data["filenames"]}
    except Exception:
        return False
    return set(filenames).issubset(stored)


def plan_chunks(df: pd.DataFrame) -> Dict[str, List[str]]:
    """chunk key -> filepaths (sorted by filename, so a chunk's row order is
    deterministic), for every (speaker, block) present in df."""
    if not df["Filename"].is_unique:
        raise ValueError("Feature store keys on basenames, but df has duplicate Filename values.")
    keys = df["Filename"].map(chunk_key)
    return {key: group.sort_values("Filename")["Filepath"].tolist()
            for key, group in df.assign(_key=keys).groupby("_key", sort=True)}


def feature_store_coverage(df: pd.DataFrame) -> Dict[str, object]:
    """How much of df the store already covers, by chunk and by utterance."""
    chunks = plan_chunks(df)
    done = [key for key, paths in chunks.items()
            if _chunk_is_complete(key, [Path(p).name for p in paths])]
    missing = [key for key in chunks if key not in done]
    n_done_files = sum(len(chunks[key]) for key in done)
    return {"chunks_total": len(chunks), "chunks_done": len(done),
            "chunks_missing": missing, "files_total": len(df), "files_done": n_done_files,
            "complete": not missing}


def build_feature_store(df: pd.DataFrame, n_workers: Optional[int] = None,
                        chunks_per_worker_per_round: int = 2,
                        time_budget_s: Optional[float] = None,
                        stall_timeout_s: float = None) -> Dict[str, object]:
    """Compute every chunk of df that is not already in the store — and only
    those. Resumable by construction: re-running after an interruption (or in
    a new Kaggle session with the previous output attached) skips every
    finished chunk.

    Chunks run in rounds of n_workers * chunks_per_worker_per_round on a
    FRESH 'spawn' process pool per round. 'spawn', not the Linux default
    'fork': the parent is a Jupyter kernel with a CUDA context and several
    threads, and forking such a process is not safe. Fresh pools rather than
    ProcessPoolExecutor(max_tasks_per_child=...): with every task submitted up
    front, the executor retires workers at the limit without replacing them —
    the audited run stopped at exactly 4 workers x 200 tasks = 800 files (see
    src.parallel). A round also bounds how long one worker process lives.

    time_budget_s, if given, stops starting new rounds once the next round is
    projected (from the rounds so far) to overrun it; the chunks it did not
    reach stay missing for the next call. A round in which no chunk completes
    within stall_timeout_s is killed and its chunks reported as failed.
    """
    from src import vad as vad_module

    n_workers = n_workers or max(1, (os.cpu_count() or 2))
    stall_timeout_s = stall_timeout_s or config.FEATURE_STORE_STALL_TIMEOUT_S
    out_dir = Path(config.FEATURE_STORE_DIR)
    chunks = plan_chunks(df)
    todo = [key for key, paths in chunks.items()
            if not _chunk_is_complete(key, [Path(p).name for p in paths])]
    n_todo_files = sum(len(chunks[k]) for k in todo)

    print_subheader(f"Feature store — {len(df):,} utterances in {len(chunks)} chunk(s)")
    print_kv("Store directory", out_dir)
    extra = store_dirs()[1:]
    if extra:
        print_kv("Read-only store directories", ", ".join(str(p) for p in extra))
    print_kv("Chunks already complete", f"{len(chunks) - len(todo)} / {len(chunks)}")
    print_kv("Chunks to build", f"{len(todo)} ({n_todo_files:,} utterances)")
    report = {"chunks_total": len(chunks), "built": [], "failed": [], "not_started": [],
              "seconds": 0.0}
    if not todo:
        print_status("Feature store already covers every requested utterance.", ok=True)
        return report

    if config.VAD_ENABLED:
        vad_module.warmup_silero_vad()     # populate the torch.hub cache before workers load it

    start = time.monotonic()
    round_size = max(1, n_workers * chunks_per_worker_per_round)
    rounds = [todo[i:i + round_size] for i in range(0, len(todo), round_size)]
    bar = ProgressReporter(None, description="Building feature store", total=n_todo_files,
                           unit="file")
    round_seconds: List[float] = []
    try:
        for r, round_keys in enumerate(rounds):
            elapsed = time.monotonic() - start
            if time_budget_s is not None and round_seconds:
                projected = elapsed + max(round_seconds) * len(round_keys) / round_size
                if projected > time_budget_s:
                    report["not_started"] = [k for rk in rounds[r:] for k in rk]
                    print_note(f"Feature-store time budget ({format_duration(time_budget_s)}) "
                               f"reached — {len(report['not_started'])} chunk(s) left for the "
                               f"next call; every finished chunk is already saved.")
                    break
            round_start = time.monotonic()
            executor = ProcessPoolExecutor(max_workers=min(n_workers, len(round_keys)),
                                           mp_context=get_context("spawn"),
                                           initializer=_worker_init)
            try:
                futures = {executor.submit(_build_chunk, (key, chunks[key], str(out_dir))): key
                           for key in round_keys}
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, timeout=stall_timeout_s,
                                         return_when=FIRST_COMPLETED)
                    if not done:
                        stuck = sorted(futures[f] for f in pending)
                        print_note(f"No chunk finished in {format_duration(stall_timeout_s)} — "
                                   f"killing this round's workers; chunks {stuck} stay missing "
                                   f"and are retried on the next call.")
                        report["failed"].extend(stuck)
                        for process in list(getattr(executor, "_processes", {}).values()):
                            process.kill()
                        break
                    for future in done:
                        key = futures[future]
                        try:
                            result = future.result()
                            report["built"].append(key)
                            bar.update(result["n_files"])
                            bar.set_postfix_str(f"{key} {format_duration(result['seconds'])}")
                        except Exception as exc:
                            print_note(f"Chunk {key} failed ({exc!r}) — it stays missing and is "
                                       f"retried on the next call.")
                            report["failed"].append(key)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
            round_seconds.append(time.monotonic() - round_start)
    finally:
        bar.close()
        clear_cache()
        from src import vad_cache
        vad_cache.clear_span_table_cache()

    report["seconds"] = time.monotonic() - start
    coverage = feature_store_coverage(df)
    print_status(f"Feature store covers {coverage['files_done']:,} / {coverage['files_total']:,} "
                 f"utterances ({coverage['chunks_done']}/{coverage['chunks_total']} chunks) after "
                 f"{format_duration(report['seconds'])}", ok=coverage["complete"])
    return report


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _live_reference(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """(segmental, supra) through the ORIGINAL live path — src.vad.apply_vad
    trimming (what src.preprocessing._trim does on a span-table miss), never
    the span table or this store — so the comparison is independent of both."""
    from src import praat
    from src import vad as vad_module
    from src.preprocessing import (_load_resampled, _pad_or_truncate, _shared_mfcc_transform,
                                   extract_mfcc_features, mfcc_frame_count)

    waveform, _ = _load_resampled(filepath)
    speech_trim, _ = vad_module.apply_vad(waveform, config.TARGET_SR)
    supra_trim, _ = vad_module.apply_vad(waveform, config.TARGET_SR,
                                         speech_pad_ms=config.SUPRA_VAD_SPEECH_PAD_MS)
    speech_wave, valid_length = _pad_or_truncate(speech_trim)
    supra_wave, supra_valid_length = _pad_or_truncate(supra_trim)
    total_frames = mfcc_frame_count(speech_wave.shape[-1])
    mfcc = extract_mfcc_features(speech_wave, _shared_mfcc_transform(),
                                 valid_length=valid_length).squeeze(0).numpy()
    extra = praat.extract_segmental_extra_sequence(
        speech_wave.squeeze(0).numpy(), config.TARGET_SR, valid_length, total_frames)
    supra = praat.extract_suprasegmental_sequence(
        supra_wave.squeeze(0).numpy(), config.TARGET_SR, supra_valid_length, total_frames)
    segmental = np.concatenate([mfcc.astype(np.float32), np.stack(
        [extra["f1_hz"], extra["f2_hz"], extra["f3_hz"], extra["hnr_db"]]).astype(np.float32)])
    suprasegmental = np.stack([supra["f0_semitones"], supra["voicing"],
                               supra["intensity_db"]]).astype(np.float32)
    return segmental, suprasegmental


def verify_feature_store(df: pd.DataFrame, n: int = 24, seed: int = 0) -> None:
    """Recompute `n` randomly chosen STORED utterances through the original
    live path and require bit-exact equality of both feature tensors. Raises
    on any mismatch — a wrong store is worse than an absent one, since it
    silently changes what the model trains on."""
    from src import vad as vad_module
    from src.console import progress

    stored = df[df["Filepath"].map(lambda fp: _locate(fp) is not None)]
    if stored.empty:
        raise RuntimeError("Nothing to verify — the feature store covers none of df.")
    if config.VAD_ENABLED:
        vad_module.warmup_silero_vad()
    sample = stored.sample(n=min(n, len(stored)), random_state=seed)
    mismatches = []
    for filepath in progress(sample["Filepath"].tolist(), f"Verifying {len(sample)} stored "
                             f"feature tensors", total=len(sample), unit="file"):
        live_segmental, live_supra = _live_reference(filepath)
        if not np.array_equal(segmental_features(filepath), live_segmental):
            diff = np.abs(segmental_features(filepath) - live_segmental).max()
            mismatches.append((Path(filepath).name, f"segmental max |diff| {diff:.3g}"))
        if not np.array_equal(suprasegmental_features(filepath), live_supra):
            diff = np.abs(suprasegmental_features(filepath) - live_supra).max()
            mismatches.append((Path(filepath).name, f"supra max |diff| {diff:.3g}"))
    if mismatches:
        detail = "\n".join(f"  {name}: {what}" for name, what in mismatches[:10])
        raise RuntimeError(
            f"Feature store disagrees with the live pipeline on {len(mismatches)} tensor(s):\n"
            f"{detail}\nDelete the affected chunks from {config.FEATURE_STORE_DIR} (or the "
            f"attached copy) and rebuild them in THIS environment.")
    print_status(f"All {len(sample)} sampled utterances match the live pipeline bit-for-bit "
                 f"(segmental 43ch + suprasegmental 3ch)", ok=True)
