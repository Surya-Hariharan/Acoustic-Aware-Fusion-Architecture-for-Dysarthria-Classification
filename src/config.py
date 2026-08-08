"""
Central configuration for the UA-Speech Dysarthria Classification project.

All paths, constants, speaker ground truth, and label mappings live here so
every module (scanning, preprocessing, dataset, splits, models) reads from a
single source of truth.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR     = PROJECT_ROOT / "data"            # root data folder
ARCHIVE_DIR  = DATA_DIR / "archives"            # place UASpeech .tgz files here
AUDIO_DIR    = DATA_DIR / "extracted"           # extracted .wav files land here
OUTPUT_DIR   = PROJECT_ROOT / "outputs"         # reports / CSVs
FIGURE_DIR   = OUTPUT_DIR / "figures"           # saved plots
ERROR_FIGURE_DIR = FIGURE_DIR / "errors"        # Phase 5: per-utterance error diagnostics
MANIFEST_PATH = OUTPUT_DIR / "m6_manifest.csv"  # scanned+filtered+labeled M6 utterances
PRAAT_FEATURES_PATH = OUTPUT_DIR / "praat_features.csv"  # Phase 4: per-utterance acoustic features

# Training pipeline outputs (train.py)
CHECKPOINT_DIR       = OUTPUT_DIR / "checkpoints"
LOG_DIR              = OUTPUT_DIR / "logs"
PREDICTIONS_DIR      = OUTPUT_DIR / "predictions"
METRICS_DIR          = OUTPUT_DIR / "metrics"
CONFUSION_MATRIX_DIR = OUTPUT_DIR / "confusion_matrix"
ROC_DIR              = OUTPUT_DIR / "roc"
EMBEDDINGS_DIR       = OUTPUT_DIR / "embeddings"

ARCHIVE_FILES = [
    "UASpeech_normalized_C.tgz",                # healthy controls
    "UASpeech_normalized_FM.tgz",               # dysarthric speakers
]

# ---------------------------------------------------------------------------
# Ground-truth speaker composition (28 speakers, verified)
# ---------------------------------------------------------------------------
CONTROL_IDS = [
    "CF02", "CF03", "CF04", "CF05",
    "CM01", "CM04", "CM05", "CM06", "CM08",
    "CM09", "CM10", "CM12", "CM13",
]

DYSARTHRIC_IDS = [
    "F02", "F03", "F04", "F05",
    "M01", "M04", "M05", "M07", "M08",
    "M09", "M10", "M11", "M12", "M14", "M16",
]

ALL_SPEAKERS = CONTROL_IDS + DYSARTHRIC_IDS

# Severity mapping for the 15 dysarthric speakers (base-paper protocol)
SEVERITY_MAP = {
    "M01": "Very Low", "M04": "Very Low", "F03": "Very Low", "M12": "Very Low",
    "M07": "Low",      "F02": "Low",      "M16": "Low",
    "M05": "Mid",      "M11": "Mid",      "F04": "Mid",
    "M09": "High",     "M14": "High",     "M10": "High",
    "M08": "High",     "F05": "High",
}

# Speakers dropped to balance severity classes at 3 speakers each.
# Matches the base paper's exclusion criterion exactly (Javanmardi et al.,
# ICASSP 2023, arXiv:2309.14107): "one male speaker from 'very low' level of
# intelligibility and two male speakers from 'high' level of intelligibility"
# were left out to reach 3-per-class / 81 (3^4) leave-one-per-class-out folds.
# M12 is the dropped Very Low male; M08 and M09 are the two dropped High
# males (the paper doesn't name which two, so this pair is still a choice,
# but — unlike the previous ["M12","M08","F05"] — it no longer contradicts
# the paper by dropping the female High speaker F05 instead of a male one).
DROPPED_FOR_BALANCE = ["M12", "M08", "M09"]     # 1 Very Low, 2 High (all male)

# ---------------------------------------------------------------------------
# Dataset protocol
# ---------------------------------------------------------------------------
TARGET_MIC        = "M6"                        # microphone channel used
WORDS_PER_SPEAKER = 765                         # expected utterances / speaker

# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------
TARGET_SR    = 16_000                           # 16 kHz mono
CLIP_SECONDS = 4.0                              # fixed analysis window
MAX_SAMPLES  = int(TARGET_SR * CLIP_SECONDS)

# MFCC settings: 13 coefficients (+ delta + delta-delta = 39-dim per frame)
N_MFCC       = 13
MEL_KWARGS   = {"n_fft": 400, "hop_length": 160, "n_mels": 40}

# Per-(process, filepath) memoization cap for src.preprocessing's cached
# loaders — see load_and_preprocess_cached / extract_mfcc_features_cached.
# Every DataLoader worker process holds its own cache up to this many
# utterances; ~256 KB/waveform + ~62 KB/MFCC at CLIP_SECONDS=4.0, so 4000
# is roughly 1.3 GB per worker (x num_workers processes — see
# TrainingConfig.num_workers). Lower this (or set to 0 to disable caching
# entirely) if a Colab session is RAM-constrained; raise it only up to the
# fold's utterance count (more than that is wasted headroom, since nothing
# beyond it will ever be re-requested within one fold).
PREPROCESS_CACHE_SIZE = 4000

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------
GROUP_LABEL_MAP = {
    "Healthy Control":    0,
    "Dysarthric Patient": 1,
}

SEVERITY_LABEL_MAP = {
    "Very Low":        0,
    "Low":             1,
    "Mid":             2,
    "High":            3,
    "N/A (Control)":  -1,
}

# ---------------------------------------------------------------------------
# Model / architecture (from the team spec)
# ---------------------------------------------------------------------------
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"
WAV2VEC_EMBED_DIM  = 768                        # latent embedding size

# Read from the environment (`setx HF_TOKEN ...` / `$env:HF_TOKEN = ...`) rather
# than hard-coded, so the token never lands in source control. Public models like
# WAV2VEC_MODEL_NAME load fine without it; passing it when present just lifts the
# unauthenticated-request rate limit and lets the same code load gated/private
# checkpoints without a separate code path.
HF_TOKEN = os.environ.get("HF_TOKEN")
ACOUSTIC_EMBED_DIM = 128                        # 1D-CNN output embedding size

LORA_RANK          = 8
LORA_ALPHA         = 16
LORA_DROPOUT       = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]   # self-attention layers

# Phase 6: attention-based fusion. The 768-dim deep and 128-dim acoustic frame
# sequences are projected into a shared FUSION_ATTN_DIM space so cross-attention
# between them is well-defined (queries and keys must share a dimension).
FUSION_ATTN_DIM     = 256
FUSION_ATTN_HEADS   = 4          # 256 / 4 = 64 dims per head
FUSION_ATTN_DROPOUT = 0.1
PRAAT_EMBED_DIM     = 256        # Praat pathway (Model F): FEATURE_COLUMNS -> one token

# ---------------------------------------------------------------------------
# Training (train.py)
# ---------------------------------------------------------------------------
NUM_CLASSES = {"detection": 2, "severity": 4}

# The manifest carries text labels; these are the fixed class orderings used
# for confusion matrices, ROC curves, and one-hot/softmax indexing.
DETECTION_CLASS_NAMES = ["Healthy Control", "Dysarthric Patient"]
SEVERITY_CLASS_NAMES  = ["Very Low", "Low", "Mid", "High"]

DEFAULT_EPOCHS        = 20
# 32, not 16: AMP is already on for every CUDA run (src/training/runner.py),
# and LoRA fine-tuning only backpropagates through a few hundred-K adapter
# params, not the frozen backbone — a smaller batch was leaving GPU
# throughput unused without buying any regularization benefit worth the
# slower 28/81-fold sweep. wav2vec2-base at CLIP_SECONDS=4.0 with AMP fits
# batch 32 comfortably on an 8 GB card (RTX 4060 and similar); drop to 16,
# then 8, if a fold OOMs on a smaller GPU.
DEFAULT_BATCH_SIZE    = 32
DEFAULT_LR_HEAD       = 1e-3     # classifier head / acoustic pathway / LoRA adapters
DEFAULT_LR_BACKBONE   = 1e-4     # wav2vec 2.0 backbone (only when fine-tuned)
DEFAULT_WEIGHT_DECAY  = 1e-2
DEFAULT_PATIENCE      = 5        # early stopping, in epochs without improvement
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_VAL_FRACTION  = 0.1      # held out from each fold's train split
DEFAULT_SEED           = 42


def ensure_directories() -> None:
    """Create every project directory that the pipeline writes to or reads from."""
    for directory in (DATA_DIR, ARCHIVE_DIR, AUDIO_DIR, OUTPUT_DIR, FIGURE_DIR,
                       ERROR_FIGURE_DIR, CHECKPOINT_DIR, LOG_DIR, PREDICTIONS_DIR,
                       METRICS_DIR, CONFUSION_MATRIX_DIR, ROC_DIR, EMBEDDINGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
