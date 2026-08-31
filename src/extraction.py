"""
Dataset extraction utilities.

Drop the UA-Speech archives (UASpeech_original_C.tgz and
UASpeech_original_FM.tgz) into data/raw/ and run this module to extract
them.

Each archive bundles the full UA-Speech corpus tree under one top-level
UASpeech/ folder (UASpeech/audio/, UASpeech/doc/, UASpeech/mlf/,
UASpeech/LICENSE.txt, UASpeech/readme_UASpeech.txt). This module extracts
SELECTIVELY, not the whole tree verbatim:
  - UASpeech/audio/original/<speaker>/*.wav -> config.AUDIO_DIR/<speaker>/*.wav
    (the UASpeech/audio/original/ prefix is stripped, so downstream code —
    scanning.py's audio_dir.rglob("*.wav") plus filename-only parsing —
    sees a flat <speaker>/<file>.wav layout regardless of the archive's
    own nesting).
  - UASpeech/{doc,mlf,LICENSE.txt,readme_UASpeech.txt} -> config.CORPUS_DOCS_DIR
    (kept separate from audio, matching the layout README.md documents).
  - macOS resource-fork duplicates ("._" prefix) are skipped at extraction
    time, not just at scan time (src.scanning already tolerates them if
    present, but there is no reason to extract junk).

WHY "audio/original", NOT "normalized" OR "noisereduce": the corpus's own
readme (UASpeech/readme_UASpeech.txt, extracted to
config.CORPUS_DOCS_DIR/readme_UASpeech.txt) documents three audio
releases — audio/original (recordings used in experiments prior to about
2012), audio/normalized (scaled to use the full dynamic range, used
2012-2019), and audio/noisereduce (denoised in 2020, the corpus authors'
own recommendation for lowest ASR word-error rate). Verified directly
against this project's actual downloaded archives: only audio/original is
present (zero files under audio/normalized or audio/noisereduce in
either .tgz). This pipeline therefore trains on audio/original — meaning
per-utterance loudness/dynamic range varies across speakers and recording
sessions in a way audio/normalized would have controlled for, and this
pipeline applies no corpus-level loudness normalization of its own
(src/preprocessing.py resamples and VAD-trims but never rescales
amplitude). This is a real, documented limitation, not an oversight — see
src.results.LIMITATIONS and README "Preprocessing" — and is specifically
relevant to the suprasegmental branch's intensity/energy input channel,
which reads raw per-utterance loudness.

Usage:
    python -m src.extraction
"""

import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

from src import config
from src.console import print_header, print_kv, print_status, progress

_ARCHIVE_ROOT = "UASpeech"                          # every member path starts with this
_AUDIO_VARIANT = "original"                         # see module docstring
_AUDIO_PREFIX = f"{_ARCHIVE_ROOT}/audio/{_AUDIO_VARIANT}/"
_DOC_PREFIXES = (
    f"{_ARCHIVE_ROOT}/doc/",
    f"{_ARCHIVE_ROOT}/mlf/",
    f"{_ARCHIVE_ROOT}/LICENSE.txt",
    f"{_ARCHIVE_ROOT}/readme_UASpeech.txt",
)


def _is_resource_fork(member_name: str) -> bool:
    """macOS AppleDouble resource-fork duplicate ('._' prefix on the
    basename) — junk, never real audio/doc content (see src.scanning's
    identical scan-time check)."""
    return Path(member_name).name.startswith("._")


def _select_members(tar: tarfile.TarFile) -> List[Tuple[tarfile.TarInfo, Path, str]]:
    """Every member this project actually wants extracted, as
    (member, destination_root, relative_path) triples — audio/original/*
    (bound for config.AUDIO_DIR) and doc/mlf/license/readme (bound for
    config.CORPUS_DOCS_DIR). Everything else (other audio variants if
    ever present, video/, interface/, resource forks, directory entries)
    is skipped. Materialized as a list (not a generator) so the caller can
    report an accurate total before extraction starts."""
    selected = []
    for member in tar.getmembers():
        if not member.isfile() or _is_resource_fork(member.name):
            continue
        if member.name.startswith(_AUDIO_PREFIX):
            selected.append((member, config.AUDIO_DIR, member.name[len(_AUDIO_PREFIX):]))
        elif member.name.startswith(_DOC_PREFIXES):
            selected.append((member, config.CORPUS_DOCS_DIR, member.name[len(_ARCHIVE_ROOT) + 1:]))
    return selected


def extract_tgz(archive_path: Path) -> Optional[int]:
    """Selectively extract one .tgz archive's audio/original + doc/mlf/
    license/readme members (see module docstring). Returns the number of
    files extracted, or None on failure."""
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            print_kv("Extracting", archive_path.name)
            selected = _select_members(tar)
            count = 0
            for member, dest_root, relative in progress(
                    selected, f"  {archive_path.name}", total=len(selected), unit="file"):
                member.name = relative
                tar.extract(member, path=dest_root)
                count += 1
        print_status(f"Extracted {count:,} file(s) from {archive_path.name}")
        return count
    except FileNotFoundError:
        print_status(f"Archive not found: {archive_path}", ok=False)
    except tarfile.ReadError as exc:
        print_status(f"Cannot read {archive_path.name}: {exc}", ok=False)
    except Exception as exc:
        print_status(f"Unexpected error on {archive_path.name}: {exc}", ok=False)
    return None


def extract_all_archives() -> int:
    """Extract every expected archive from config.ARCHIVE_DIR. Returns the
    count of archives (not files) successfully extracted."""
    config.ensure_directories()
    print_header("Dataset Extraction")
    print_kv("Archive folder", config.ARCHIVE_DIR)
    print_kv("Audio extract folder", config.AUDIO_DIR)
    print_kv("Corpus docs folder", config.CORPUS_DOCS_DIR)

    extracted = 0
    for archive_name in config.ARCHIVE_FILES:
        if extract_tgz(config.ARCHIVE_DIR / archive_name) is not None:
            extracted += 1

    print_kv("Archives extracted", f"{extracted} / {len(config.ARCHIVE_FILES)}")
    return extracted


if __name__ == "__main__":
    extract_all_archives()
