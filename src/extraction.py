"""
Dataset extraction utilities.

Drop the UA-Speech archives (UASpeech_normalized_C.tgz and
UASpeech_normalized_FM.tgz) into data/archives/ and run this module to
extract them into data/extracted/.

Usage:
    python -m src.extract
"""

import tarfile
from pathlib import Path

from src import config
from src.console import print_header, print_kv, print_status


def extract_tgz(archive_path: Path, extract_to: Path) -> bool:
    """Extract a single .tgz archive. Returns True on success."""
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            print_kv("Extracting", archive_path.name)
            tar.extractall(path=extract_to)
        print_status(f"Extracted {archive_path.name}")
        return True
    except FileNotFoundError:
        print_status(f"Archive not found: {archive_path}", ok=False)
    except tarfile.ReadError as exc:
        print_status(f"Cannot read {archive_path.name}: {exc}", ok=False)
    except Exception as exc:
        print_status(f"Unexpected error on {archive_path.name}: {exc}", ok=False)
    return False


def extract_all_archives() -> int:
    """Extract every expected archive from data/archives/. Returns count extracted."""
    config.ensure_directories()
    print_header("Dataset Extraction")
    print_kv("Archive folder", config.ARCHIVE_DIR)
    print_kv("Extract folder", config.AUDIO_DIR)

    extracted = 0
    for archive_name in config.ARCHIVE_FILES:
        if extract_tgz(config.ARCHIVE_DIR / archive_name, config.AUDIO_DIR):
            extracted += 1

    print_kv("Archives extracted", f"{extracted} / {len(config.ARCHIVE_FILES)}")
    return extracted


if __name__ == "__main__":
    extract_all_archives()
