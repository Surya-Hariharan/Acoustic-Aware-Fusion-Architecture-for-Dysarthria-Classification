"""
Central configuration for the UA-Speech Dysarthria Classification project.

All paths, constants, speaker ground truth, and label mappings live here so
every module (scanning, preprocessing, dataset, splits, models) reads from a
single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR     = PROJECT_ROOT / "data"            # root data folder
ARCHIVE_DIR  = DATA_DIR / "archives"            # place UASpeech .tgz files here
AUDIO_DIR    = DATA_DIR / "extracted"           # extracted .wav files land here
OUTPUT_DIR   = PROJECT_ROOT / "outputs"         # reports / CSVs
FIGURE_DIR   = OUTPUT_DIR / "figures"           # saved plots
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

# Speakers dropped to balance severity classes at 3 speakers each
# (assumption pending team confirmation — see README "Pipeline Summary")
DROPPED_FOR_BALANCE = ["M12", "M08", "F05"]     # 1 Very Low, 2 High

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
ACOUSTIC_EMBED_DIM = 128                        # 1D-CNN output embedding size

LORA_RANK          = 8
LORA_ALPHA         = 16
LORA_DROPOUT       = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]   # self-attention layers

# ---------------------------------------------------------------------------
# Training (train.py)
# ---------------------------------------------------------------------------
NUM_CLASSES = {"detection": 2, "severity": 4}

# The manifest carries text labels; these are the fixed class orderings used
# for confusion matrices, ROC curves, and one-hot/softmax indexing.
DETECTION_CLASS_NAMES = ["Healthy Control", "Dysarthric Patient"]
SEVERITY_CLASS_NAMES  = ["Very Low", "Low", "Mid", "High"]

DEFAULT_EPOCHS        = 20
DEFAULT_BATCH_SIZE    = 8
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
                       CHECKPOINT_DIR, LOG_DIR, PREDICTIONS_DIR, METRICS_DIR,
                       CONFUSION_MATRIX_DIR, ROC_DIR, EMBEDDINGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
