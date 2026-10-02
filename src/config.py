"""
Central configuration: paths, speaker ground truth, label maps, preprocessing,
architecture and training hyperparameters, and the local hardware profile.

Every module reads from here, so a value changed here is changed everywhere.
Values that enter src.feature_store.store_signature() (VAD, MFCC, window,
channel counts) also key the on-disk feature store: changing one invalidates
it and the next run rebuilds it.
"""

import os
from itertools import zip_longest
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_DIR = DATA_DIR / "raw"                         # UA-Speech .tgz archives
AUDIO_DIR = DATA_DIR / "extracted"                     # extracted .wav (audio/original only)
CORPUS_DOCS_DIR = DATA_DIR / "uaspeech_corpus_docs"    # corpus docs/MLF/license, kept apart
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MANIFEST_PATH = OUTPUT_DIR / "m6_manifest.csv"

# Chunked feature store (src/feature_store.py): one compressed .npz per
# (speaker, block) holding VAD spans + raw segmental (43ch) + suprasegmental
# (3ch) features. FEATURE_STORE_EXTRA_DIRS are searched read-only after it
# (also settable via the FEATURE_STORE_EXTRA_DIRS env var, os.pathsep-separated).
FEATURE_CACHE_DIR = OUTPUT_DIR / "feature_cache"
FEATURE_STORE_DIR = FEATURE_CACHE_DIR / "store"
FEATURE_STORE_EXTRA_DIRS: list = []
# No chunk (255 utterances) finishing in 30 minutes means a stuck build round.
FEATURE_STORE_STALL_TIMEOUT_S = 1800

# Per-run training artifacts, each under <dir>/<run_name>/.
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"                          # TensorBoard + fold tracebacks
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
CONFUSION_MATRIX_DIR = OUTPUT_DIR / "confusion_matrix"
ROC_DIR = OUTPUT_DIR / "roc"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
RESULTS_DIR = OUTPUT_DIR / "results"                   # frozen run configurations

ARCHIVE_FILES = [
    "UASpeech_original_C.tgz",                         # healthy controls
    "UASpeech_original_FM.tgz",                        # dysarthric speakers
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

# Severity of the 15 dysarthric speakers (UA-Speech intelligibility groups).
SEVERITY_MAP = {
    "M01": "Very Low", "M04": "Very Low", "F03": "Very Low", "M12": "Very Low",
    "M07": "Low",      "F02": "Low",      "M16": "Low",
    "M05": "Mid",      "M11": "Mid",      "F04": "Mid",
    "M09": "High",     "M14": "High",     "M10": "High",
    "M08": "High",     "F05": "High",
}

SEVERITY_CLASS_NAMES = ["Very Low", "Low", "Mid", "High"]
NUM_CLASSES = len(SEVERITY_CLASS_NAMES)
SEVERITY_LABEL_MAP = {name: i for i, name in enumerate(SEVERITY_CLASS_NAMES)}
SEVERITY_LABEL_MAP["N/A (Control)"] = -1


def _interleave_by_severity(dysarthric_ids, severity_map):
    """Round-robin the dysarthric speakers across the four severity classes,
    so every prefix of the LOSO fold order is as class-balanced as possible.
    A complete 15-fold sweep is order-independent; this only makes a partial
    run (max_folds, or an interrupted session) cover every class early."""
    groups = [[s for s in dysarthric_ids if severity_map[s] == cls]
              for cls in SEVERITY_CLASS_NAMES]
    return [speaker for round_ in zip_longest(*groups) for speaker in round_
            if speaker is not None]


SEVERITY_LOSO_ORDER = _interleave_by_severity(DYSARTHRIC_IDS, SEVERITY_MAP)

# ---------------------------------------------------------------------------
# Dataset protocol and audio preprocessing
# ---------------------------------------------------------------------------
TARGET_MIC = "M6"                                      # microphone channel used
WORDS_PER_SPEAKER = 765                                # 3 blocks x 255 words

TARGET_SR = 16_000                                     # 16 kHz mono
CLIP_SECONDS = 4.0                                     # fixed analysis window
MAX_SAMPLES = int(TARGET_SR * CLIP_SECONDS)            # 64,000 samples -> 401 frames

N_MFCC = 13                                            # + delta + delta-delta = 39
MEL_KWARGS = {"n_fft": 400, "hop_length": 160, "n_mels": 40}   # 25 ms / 10 ms

# Silero VAD (src/vad.py): trims leading/trailing non-speech to the contiguous
# first..last speech span, so internal pauses are preserved.
VAD_ENABLED = True
VAD_THRESHOLD = 0.5                                    # Silero's own default
VAD_MIN_SPEECH_MS = 100                                # isolated words: below Silero's 250 ms default
VAD_MIN_SILENCE_MS = 100                               # shorter gaps stay inside the span
VAD_SPEECH_PAD_MS = 30                                 # speech-focused profile (learned + segmental)
SUPRA_VAD_SPEECH_PAD_MS = 150                          # temporal-preserving profile (suprasegmental)
VAD_SAMPLE_RATE = 16_000                               # must equal TARGET_SR

# Per-process LRU size for the preprocessing getters. The feature store serves
# every engineered feature, so these only save a ~5 ms wav read per hit.
PREPROCESS_CACHE_SIZE = 64

# ---------------------------------------------------------------------------
# Architecture (src/models/gated_fusion.py)
# ---------------------------------------------------------------------------
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"
WAV2VEC_EMBED_DIM = 768
# Optional; lifts the Hugging Face rate limit. The checkpoint is public.
HF_TOKEN = os.environ.get("HF_TOKEN")

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]  # all 12 encoder layers

# The CTC checkpoint has no pretrained masked_spec_embed; SpecAugment would
# create a random one. Disabled before model construction (deep_pathway.py).
WAV2VEC_APPLY_SPEC_AUGMENT = False
# Recompute wav2vec2 activations in backward instead of storing them. Identical
# gradients; measured on this machine: off = +32% throughput at 5.8 GB VRAM.
WAV2VEC_GRADIENT_CHECKPOINTING = False

# Minimum total share the softmax gate leaves to uniform weighting. Without it
# the gate collapsed onto the fastest-fitting branch (segmental CNN, gate
# 0.64-0.98 on the first four folds) and the other branches got ~0 gradient.
GATE_UNIFORM_FLOOR = 0.3
# Training-only regularizers against fitting the few training speakers: dropout
# on the fused embedding before the CORAL head, and per-utterance dropout of
# whole branch embeddings (at least one branch always survives).
HEAD_DROPOUT = 0.3
BRANCH_DROPOUT = 0.15

LEARNED_EMBED_DIM = 128                               # Z_learned (768 -> 128)
SEGMENTAL_EMBED_DIM = 64                               # Z_segmental
SUPRA_EMBED_DIM = 64                                   # Z_supra
FUSED_EMBED_DIM = LEARNED_EMBED_DIM + SEGMENTAL_EMBED_DIM + SUPRA_EMBED_DIM   # 256

SEGMENTAL_CHANNELS = 3 * N_MFCC + 3 + 1                # MFCC+d+dd (39) + F1-F3 + HNR = 43
SUPRA_CHANNELS = 3                                     # F0 semitones, voicing, intensity dB

# Fixed before the run; never tuned against its results.
LAMBDA_COMP = 0.05                                     # cross-branch redundancy penalty
LAMBDA_SPEAKER = 0.1                                   # adversarial speaker loss
GRL_LAMBDA = 1.0                                       # gradient-reversal strength

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS = 12                                    # ceiling; early stopping ends folds sooner
DEFAULT_BATCH_SIZE = 32                                # measured: 76 samples/s, 5.8 GB peak VRAM
# Two optimizer groups (src.training.engine.build_optimizer): LoRA adapters
# train at DEFAULT_LR_LORA; the branch CNNs, projections, gate and heads at
# DEFAULT_LR_HEAD. The wav2vec2 backbone itself is frozen.
DEFAULT_LR_HEAD = 3e-4                                 # 1e-3 overfit the training speakers within one epoch
DEFAULT_LR_LORA = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_PATIENCE = 5                                   # epochs without validation improvement
# Validation on 3-6 speakers is noisy (accuracy swings +-5 points between
# evaluations), so a lucky first evaluation used to win: several folds
# "selected" epoch 0.5, an almost untrained head. No checkpoint is eligible and
# no patience is spent before DEFAULT_MIN_EPOCHS, and the monitored value is
# the mean of its last DEFAULT_MONITOR_SMOOTHING evaluations.
DEFAULT_MIN_EPOCHS = 3
DEFAULT_MONITOR_SMOOTHING = 2
# Validation speakers drawn per severity class (still capped so every class
# keeps at least 2 training speakers).
DEFAULT_VAL_SPEAKERS_PER_CLASS = 2
# Validate this many times per training epoch: with the best model at epoch 1,
# once-per-epoch validation skipped the checkpoints that mattered. Patience is
# counted in epochs (scaled to evaluations internally).
DEFAULT_EVALS_PER_EPOCH = 2
# Early-stopping/checkpoint metric: "ordinal_mae" (robust to the confident-wrong
# loss spikes seen on the 3-4 validation speakers) or "ordinal_loss".
DEFAULT_MONITOR = "ordinal_mae"
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# Local hardware profile — RTX 4060 Laptop (8 GB, 88 W), 16 GB RAM, Windows.
#
# On Windows every DataLoader worker is a spawned process that re-imports
# torch's CUDA DLLs (~0.7 GB resident, ~2 GB commit), so worker count is
# bounded by memory, not cores. With the feature store memory-mapped, one
# worker supplies ~340 items/s against the ~76 items/s the GPU trains at.
# ---------------------------------------------------------------------------
AMP_DTYPE = "float16"                                  # "bfloat16" only on Ampere+ (no GradScaler)
TRAIN_NUM_WORKERS = 1                                  # persistent, training loader
EVAL_NUM_WORKERS = 0                                   # validation in the main process
TEST_NUM_WORKERS = 0                                   # test in the main process
DATALOADER_PREFETCH_FACTOR = 4

# Per-process memory cost, used to size worker pools to what is free right now
# (src.training.utils.affordable_workers).
DATALOADER_WORKER_RAM_GB = 0.75
DATALOADER_WORKER_COMMIT_GB = 2.0
FEATURE_STORE_WORKER_RAM_GB = 0.9                      # torch + Silero + parselmouth
FEATURE_STORE_WORKER_COMMIT_GB = 2.2
RAM_RESERVE_GB = 2.0                                   # left for the OS and the notebook
COMMIT_RESERVE_GB = 4.0
# On WDDM the ~6.3 GB of VRAM PyTorch reserves is also charged to system
# commit; worker counts are decided before a fold reserves it.
TRAINING_COMMIT_RESERVE_GB = 8.0
MIN_FREE_COMMIT_GB_PER_FOLD = 10.0                     # below this, warn before a fold

# Cap PyTorch's share of VRAM. On WDDM, exceeding physical VRAM does not raise
# OOM — it spills into system RAM and throughput collapses ~10x. The cap turns
# that into a clean, retryable OOM and leaves headroom for the display.
CUDA_MEMORY_FRACTION = 0.90
CUDNN_BENCHMARK = True                                 # fixed 4 s input -> autotuned kernels
FEATURE_STORE_WORKERS = 8                              # upper bound, shrunk to free memory

# Optional thermal guard (src.training.utils.ThermalGuard), OFF: training runs
# without pausing and the GPU's own firmware throttles clocks if it gets hot.
# True pauses between batches at GPU_TEMP_PAUSE_C until GPU_TEMP_RESUME_C, and
# between folds for up to FOLD_COOLDOWN_S.
THERMAL_GUARD_ENABLED = False
GPU_TEMP_PAUSE_C = 83
GPU_TEMP_RESUME_C = 72
GPU_TEMP_CHECK_INTERVAL_S = 15.0
GPU_COOLDOWN_MAX_WAIT_S = 600                          # resume anyway after this; the note says so
FOLD_COOLDOWN_S = 60                                   # only with THERMAL_GUARD_ENABLED
# Refuse to start training on battery power: the GPU drops to a fraction of
# its clocks and a multi-hour run would drain the battery mid-fold.
REQUIRE_AC_POWER = True

USE_TQDM = True                                        # False: throttled line log for piped output

# ---------------------------------------------------------------------------
# AAF-Lite (src/aaflite): the same three branches on FROZEN representations,
# control-referenced per word, fused late by linear models under nested LOSO.
# Every choice below is selected inside the training speakers of each outer
# fold — never against the held-out speaker.
# ---------------------------------------------------------------------------
AAFLITE_EMBEDDING_DIR = EMBEDDINGS_DIR / "wav2vec2_layer_stats"   # one .npz per speaker
AAFLITE_EMBED_BATCH_SIZE = 32
# Contiguous wav2vec2 hidden-state groups (0 = CNN feature projection). Each
# group's per-layer [mean, std] pools are averaged into one 1536-d vector.
AAFLITE_LAYER_GROUPS = {"early": (1, 2, 3, 4), "middle": (5, 6, 7, 8), "late": (9, 10, 11, 12),
                        "all": tuple(range(1, 13))}
AAFLITE_PCA_DIMS = (32, 128)                           # learned + segmental branches
AAFLITE_C_GRID = (0.003, 0.03, 0.3)                    # inverse L2 strength, LogisticRegression
AAFLITE_FUSION_STEP = 0.1                              # simplex grid for the branch weights
AAFLITE_BOOTSTRAP = 2000                               # speaker-level bootstrap resamples
AAFLITE_N_JOBS = 4                                     # outer folds in parallel (CPU)


def ensure_directories() -> None:
    """Create every directory the pipeline writes to."""
    for directory in (DATA_DIR, ARCHIVE_DIR, AUDIO_DIR, CORPUS_DOCS_DIR, OUTPUT_DIR,
                      FEATURE_CACHE_DIR, FEATURE_STORE_DIR, CHECKPOINT_DIR, LOG_DIR,
                      PREDICTIONS_DIR, METRICS_DIR, CONFUSION_MATRIX_DIR, ROC_DIR,
                      EMBEDDINGS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
