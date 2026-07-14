"""
Full-scale training driver: all six ablation variants, detection task first
(the paper's primary comparison), then severity. Mirrors
notebooks/02_training.ipynb Stages 4 and 6 exactly - the notebook can be
re-executed later to regenerate clean saved outputs cheaply, since
run_training() skips any fold already completed on disk (see
src/training/runner.py::_load_completed_fold).

Intended to run unattended for a long time (a full 28-fold LOSO pass of a
wav2vec-fine-tuning variant is hours, not minutes; six variants is realistically
a multi-day job on a single GPU) - each model is trained in its own
run_training() call, and a failing fold is isolated rather than aborting the
whole run, so this script does not need babysitting.

Usage:
    conda activate torch-gpu
    python scripts/run_full_training.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.console import print_header, print_kv
from src.training.baseline import extract_frozen_embeddings, run_svm_baseline
from src.training.data import load_manifest
from src.training.models import MODEL_NAMES
from src.training.runner import TrainingConfig, run_training


def main() -> None:
    config.ensure_directories()
    df_m6 = load_manifest()

    print_header("Full-Scale Training Driver")
    print_kv("Utterances", len(df_m6))
    print_kv("Speakers", df_m6["Speaker_ID"].nunique())
    print_kv("Models (cheapest to most expensive)", ", ".join(MODEL_NAMES))

    # Phase 2 baseline - cheap (frozen embeddings cached, SVM refit per fold),
    # re-run here mainly to guarantee it's present before the comparison table.
    frozen_embeddings = extract_frozen_embeddings(df_m6, batch_size=16)
    run_svm_baseline(df_m6, task="detection", embeddings=frozen_embeddings, max_folds=None)

    print_header("Detection task - full 28-fold LOSO, all six variants")
    for model_name in MODEL_NAMES:
        cfg = TrainingConfig(task="detection", model=model_name,
                             run_name=f"detection_{model_name}")
        run_training(df_m6, cfg)

    print_header("Severity task - full 81-fold protocol, all six variants")
    for model_name in MODEL_NAMES:
        cfg = TrainingConfig(task="severity", model=model_name,
                             run_name=f"severity_{model_name}")
        run_training(df_m6, cfg)

    print_header("Full-Scale Training Driver - Done")


if __name__ == "__main__":
    main()
