"""
Standalone terminal data-integrity check for the entire UA-Speech extracted
dataset (all microphones/blocks, not just the M6 subset used downstream).

Runs three checks per file, cheapest first:
  1. RIFF/WAVE header (12 bytes) - catches zero-filled/truncated extractions.
  2. Full decode via soundfile - catches files with a valid header that still
     fail to decode.
  3. Content sanity on the decoded samples - NaN/Inf, near-silence, and
     sample-rate/duration outliers relative to the rest of the dataset.

This is deliberately separate from src.scanning.validate_wav_headers (which
only does check #1, only on the M6-filtered subset, as part of the notebook
pipeline) - it exists to be run once, on demand, from a terminal, before any
training starts.

Usage:
    conda activate torch-gpu
    python scripts/check_data_integrity.py
"""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.console import (print_header, print_kv, print_series, print_status,
                         print_subheader, progress)
from src.scanning import scan_audio_files

SILENCE_THRESHOLD = 1e-4    # max abs sample amplitude below this -> flagged silent
MIN_DURATION_S = 0.05       # shorter than this -> flagged too-short


def _has_valid_header(filepath: str) -> bool:
    try:
        with open(filepath, "rb") as f:
            header = f.read(12)
    except OSError:
        return False
    return len(header) == 12 and header[0:4] == b"RIFF" and header[8:12] == b"WAVE"


def _check_one(filepath: str) -> dict:
    """Run every check on one file. Never raises - failures are reported, not thrown."""
    result = {
        "Filepath": filepath,
        "valid_header": False,
        "decodable": False,
        "samplerate": None,
        "channels": None,
        "duration_s": None,
        "reason": None,
    }

    if not _has_valid_header(filepath):
        result["reason"] = "bad_header"
        return result
    result["valid_header"] = True

    try:
        info = sf.info(filepath)
    except Exception as exc:
        result["reason"] = f"info_error: {exc}"
        return result

    result["samplerate"] = info.samplerate
    result["channels"] = info.channels
    result["duration_s"] = info.frames / info.samplerate if info.samplerate else None

    try:
        data, _ = sf.read(filepath, dtype="float32")
    except Exception as exc:
        result["reason"] = f"read_error: {exc}"
        return result
    result["decodable"] = True

    if not np.isfinite(data).all():
        result["reason"] = "nan_or_inf"
        return result

    if np.max(np.abs(data)) < SILENCE_THRESHOLD:
        result["reason"] = "silent"
        return result

    if result["duration_s"] is not None and result["duration_s"] < MIN_DURATION_S:
        result["reason"] = "too_short"
        return result

    return result


def main() -> None:
    print_header("Data Integrity Check - entire dataset")
    print_kv("Scanning folder", config.AUDIO_DIR)

    df = scan_audio_files(config.AUDIO_DIR)
    if df.empty:
        print_status("No audio files found - check config.AUDIO_DIR", ok=False)
        return

    filepaths = df["Filepath"].tolist()
    print_kv("Files to check", len(filepaths))

    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(_check_one, fp): fp for fp in filepaths}
        for future in progress(as_completed(futures), "Checking files",
                               total=len(futures), unit="file"):
            results.append(future.result())

    results_df = pd.DataFrame(results)
    merged = df.merge(results_df, on="Filepath", how="left")

    # Anything not already flagged but off the dataset's dominant sample rate.
    sr_counts = merged["samplerate"].value_counts()
    dominant_sr = sr_counts.index[0] if not sr_counts.empty else None
    needs_sr_flag = (merged["reason"].isna() & merged["samplerate"].notna()
                     & (merged["samplerate"] != dominant_sr))
    merged.loc[needs_sr_flag, "reason"] = "unexpected_samplerate"

    flagged = merged[merged["reason"].notna()].copy()

    print_subheader("Summary")
    print_kv("Total files checked", len(merged))
    print_kv("Flagged files", len(flagged))
    print_kv("Dominant sample rate", f"{dominant_sr:.0f} Hz" if dominant_sr else "n/a")

    if not flagged.empty:
        print_subheader("Flagged files by reason")
        print_series(flagged["reason"].value_counts())
        print_subheader("Flagged files by speaker")
        print_series(flagged.groupby("Speaker_ID").size().sort_values(ascending=False))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.OUTPUT_DIR / "data_integrity_report.csv"
    report_cols = ["Filepath", "Speaker_ID", "Block", "WordCode", "Microphone_Channel",
                  "valid_header", "decodable", "samplerate", "channels",
                  "duration_s", "reason"]
    flagged[report_cols].to_csv(report_path, index=False)
    print_kv("Report saved to", report_path)

    print_status(
        f"{len(flagged)} file(s) flagged - review {report_path.name} before proceeding"
        if not flagged.empty else "All files passed every integrity check",
        ok=flagged.empty,
    )


if __name__ == "__main__":
    main()
