"""
Dataset scanning and verification.

Walks data/extracted/, parses every UA-Speech filename
(Speaker_Block_WordCode_Mic.wav), classifies speakers against the verified
ground truth, and produces the working DataFrame for the pipeline.

Key correctness rules (see README "Data Verification Notes"):
  * Microphone channel is taken positionally from the FINAL filename token,
    never by searching for a token starting with 'M' (which would wrongly
    match male dysarthric speaker IDs like M01).
  * macOS resource-fork duplicates ('._' prefix) are skipped.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src import config
from src.console import (
    print_header, print_subheader, print_kv,
    print_series, print_status,
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
def parse_filename(filename: str) -> Optional[dict]:
    """
    Parse one UA-Speech filename into its components.

    Expected pattern: <Speaker>_<Block>_<WordCode>_<Mic>.wav
    Returns None for resource-fork duplicates, malformed names, or
    speakers outside the verified ground truth.
    """
    if not filename.endswith(".wav") or filename.startswith("._"):
        return None

    parts = filename.removesuffix(".wav").split("_")
    if len(parts) < 4:                       # need Speaker_Block_Word_Mic
        return None

    speaker_id = parts[0]
    if speaker_id not in config.ALL_SPEAKERS:
        return None

    return {
        "Filename": filename,
        "Speaker_ID": speaker_id,
        "Group": ("Healthy Control" if speaker_id in config.CONTROL_IDS
                  else "Dysarthric Patient"),
        "Block": parts[1],
        "WordCode": parts[2],
        "Microphone_Channel": parts[-1],     # positional: final token only
    }


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------
def scan_audio_files(audio_dir: Path = config.AUDIO_DIR) -> pd.DataFrame:
    """Walk the extracted dataset and return a DataFrame of valid audio files."""
    print_header("Dataset Scan")
    print_kv("Scanning folder", audio_dir)

    records, skipped = [], 0
    for filepath in audio_dir.rglob("*.wav"):
        parsed = parse_filename(filepath.name)
        if parsed is None:
            skipped += 1
            continue
        parsed["Filepath"] = str(filepath)
        records.append(parsed)

    df = pd.DataFrame(records)
    print_kv("Valid audio files", len(df))
    print_kv("Skipped files", skipped)
    if not df.empty:
        print_kv("Unique speakers", df["Speaker_ID"].nunique())
    return df


# ---------------------------------------------------------------------------
# Verification against ground truth
# ---------------------------------------------------------------------------
def verify_speakers(df: pd.DataFrame) -> bool:
    """Check that exactly the 28 ground-truth speakers are present."""
    print_subheader("Speaker Verification")

    found = set(df["Speaker_ID"].unique())
    expected = set(config.ALL_SPEAKERS)
    missing, spurious = expected - found, found - expected

    print_kv("Expected speakers", len(expected))
    print_kv("Found speakers", len(found))
    print_kv("Missing", sorted(missing) if missing else "None")
    print_kv("Spurious", sorted(spurious) if spurious else "None")

    ok = not missing and not spurious
    print_status("All 28 ground-truth speakers present, no spurious IDs"
                 if ok else "Speaker set mismatch - investigate before continuing",
                 ok=ok)
    return ok


# ---------------------------------------------------------------------------
# Microphone channel filtering (base-paper protocol: M6 only)
# ---------------------------------------------------------------------------
def filter_mic_channel(df: pd.DataFrame,
                       mic: str = config.TARGET_MIC) -> pd.DataFrame:
    """Filter to a single microphone channel, all blocks, all word types."""
    print_subheader(f"Microphone Filter ({mic} only)")

    df_mic = df[df["Microphone_Channel"] == mic].copy()
    print_kv(f"Total {mic} samples", len(df_mic))

    group_counts = df_mic["Group"].value_counts()
    for group, count in group_counts.items():
        print_kv(f"  {group}", count)

    for group, ids in (("dysarthric", config.DYSARTHRIC_IDS),
                       ("control", config.CONTROL_IDS)):
        present = set(df_mic["Speaker_ID"].unique())
        missing = set(ids) - present
        print_kv(f"Missing {group} speakers on {mic}",
                 sorted(missing) if missing else "None")
    return df_mic


def validate_wav_headers(df_mic: pd.DataFrame) -> pd.DataFrame:
    """
    Drop files whose RIFF/WAVE header is missing or malformed - a handful
    of UA-Speech files extracted zero-filled (see README "Data verification
    note"). Cheap (12-byte read, no decode) and confirmed to match a full
    soundfile.info() pass across the M6 set: same 39 files either way.
    """
    print_subheader("WAV Header Validation")

    def _has_valid_header(filepath: str) -> bool:
        try:
            with open(filepath, "rb") as f:
                header = f.read(12)
        except OSError:
            return False
        return len(header) == 12 and header[0:4] == b"RIFF" and header[8:12] == b"WAVE"

    valid_mask = df_mic["Filepath"].map(_has_valid_header)
    invalid = df_mic[~valid_mask]

    print_kv("Files checked", len(df_mic))
    print_kv("Corrupted (dropped)", len(invalid))
    if not invalid.empty:
        print_series(invalid.groupby("Speaker_ID").size())
        print_status(f"{len(invalid)} corrupted file(s) excluded from the manifest "
                     f"- see the dropped rows' Speaker_ID/WordCode above", ok=False)
    else:
        print_status("All files have valid RIFF/WAVE headers")

    return df_mic[valid_mask].copy()


def check_word_counts(df_mic: pd.DataFrame) -> pd.Series:
    """Per-speaker utterance count check against the expected 765 words."""
    print_subheader(f"Per-Speaker Word Counts (target: {config.WORDS_PER_SPEAKER})")

    counts = df_mic.groupby("Speaker_ID").size().sort_values()
    print_series(counts)

    incomplete = counts[counts < config.WORDS_PER_SPEAKER]
    if len(incomplete):
        print_status(f"{len(incomplete)} speaker(s) below "
                     f"{config.WORDS_PER_SPEAKER} words (flag, don't drop)",
                     ok=False)
        print_series(incomplete)
    else:
        print_status("All speakers complete")
    return counts


# ---------------------------------------------------------------------------
# Severity labelling
# ---------------------------------------------------------------------------
def add_severity_labels(df_mic: pd.DataFrame) -> pd.DataFrame:
    """Attach the four-class severity label to every dysarthric sample."""
    print_subheader("Severity Labels")

    df_mic = df_mic.copy()
    df_mic["Severity"] = (df_mic["Speaker_ID"]
                          .map(config.SEVERITY_MAP)
                          .fillna("N/A (Control)"))

    unmapped = df_mic[(df_mic["Group"] == "Dysarthric Patient") &
                      (df_mic["Severity"] == "N/A (Control)")]
    print_status("No unmapped dysarthric speakers" if unmapped.empty
                 else f"Unmapped dysarthric speakers: "
                      f"{sorted(unmapped['Speaker_ID'].unique())}",
                 ok=unmapped.empty)

    print_series(df_mic["Severity"].value_counts())
    return df_mic
