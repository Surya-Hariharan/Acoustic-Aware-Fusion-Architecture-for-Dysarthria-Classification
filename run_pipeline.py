"""
UA-Speech data pipeline runner (script entry point).

Runs the same stages as notebooks/01_data_pipeline.ipynb, but headless. Use
this for reproducible end-to-end runs; use the notebook for interactive work.

Stages:
    1. Scan data/extracted/ and parse filenames.
    2. Verify the 28-speaker ground truth.
    3. Generate EDA figures into outputs/figures/.
    4. Filter to microphone channel M6.
    5. Check per-speaker word counts (765 expected).
    6. Attach severity labels.
    7. Summarize detection (LOSO) and severity (81-fold) splits.
    8. Build the unified PyTorch dataset and export the manifest CSV.

Usage:
    python -m src.extraction     # extract archives first (once dataset arrives)
    python run_pipeline.py       # run the pipeline
"""

import sys

from src import config
from src.console import print_header, print_kv, print_status
from src.scanning import (
    add_severity_labels, check_word_counts, filter_mic_channel,
    scan_audio_files, verify_speakers,
)
from src.splits import build_severity_folds, summarize_detection_splits
from src.visualization import run_eda


def main() -> int:
    config.ensure_directories()

    print_header("UA-Speech Dysarthria Pipeline")
    print_kv("Project root", config.PROJECT_ROOT)
    print_kv("Data folder", config.DATA_DIR)
    print_kv("Ground truth", f"{len(config.CONTROL_IDS)} controls + "
                             f"{len(config.DYSARTHRIC_IDS)} dysarthric = "
                             f"{len(config.ALL_SPEAKERS)} speakers")

    # 1. Scan ----------------------------------------------------------------
    df_audio = scan_audio_files()
    if df_audio.empty:
        print_status("No audio files found in data/extracted/", ok=False)
        print("\n  Next steps once the dataset arrives:")
        print(f"    1. Copy the .tgz archives into: {config.ARCHIVE_DIR}")
        print("    2. Run: python -m src.extraction")
        print("    3. Re-run: python run_pipeline.py")
        return 1

    # 2. Verify --------------------------------------------------------------
    if not verify_speakers(df_audio):
        return 1

    # 3. EDA -----------------------------------------------------------------
    run_eda(df_audio)

    # 4-6. Filter, count, label ----------------------------------------------
    df_m6 = filter_mic_channel(df_audio)
    check_word_counts(df_m6)
    df_m6 = add_severity_labels(df_m6)

    # 7. Splits --------------------------------------------------------------
    summarize_detection_splits(df_m6)
    build_severity_folds(df_m6)

    # 8. Dataset + manifest --------------------------------------------------
    print_header("Dataset Build")
    from src.dataset import UASpeechDataset          # defer the torch import
    dataset = UASpeechDataset(df_m6)
    print_kv("Total samples", len(dataset))

    manifest_path = config.OUTPUT_DIR / "m6_manifest.csv"
    df_m6.to_csv(manifest_path, index=False)
    print_kv("Manifest saved", manifest_path)

    print_header("Pipeline Complete")
    print_status("Data pipeline verified - ready for pathway training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
