"""
Central configuration for the UA-Speech Dysarthria Classification project.

All paths, constants, speaker ground truth, and label mappings live here so
every module (scanning, preprocessing, dataset, splits, models) reads from a
single source of truth.
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

DATA_DIR     = PROJECT_ROOT / "data"            # root data folder
ARCHIVE_DIR  = DATA_DIR / "raw"                 # place UASpeech .tgz files here
AUDIO_DIR    = DATA_DIR / "extracted"           # extracted .wav files land here (audio/original only)
CORPUS_DOCS_DIR = DATA_DIR / "uaspeech_corpus_docs"  # extracted doc/mlf/license/readme, kept out of AUDIO_DIR
OUTPUT_DIR   = PROJECT_ROOT / "outputs"         # reports / CSVs
FIGURE_DIR   = OUTPUT_DIR / "figures"           # saved plots
ERROR_FIGURE_DIR = FIGURE_DIR / "errors"        # Phase 5: per-utterance error diagnostics
MANIFEST_PATH = OUTPUT_DIR / "m6_manifest.csv"  # scanned+filtered+labeled M6 utterances
PRAAT_FEATURES_PATH = OUTPUT_DIR / "praat_features.csv"  # Phase 4: per-utterance acoustic features

# Three-branch severity architecture's figure/table/diagnostic subdirectories
# (architecture plan Work Package C) — kept as separate named constants
# rather than folded into FIGURE_DIR/ERROR_FIGURE_DIR so each analysis
# module writes to an unambiguous, predictable location instead of every
# figure type landing flat in FIGURE_DIR.
SIGNAL_FIGURE_DIR         = FIGURE_DIR / "signals"          # waveform/spectrogram/MFCC/F0 panels
REPRESENTATION_FIGURE_DIR = FIGURE_DIR / "representations"  # PCA/UMAP embedding maps
EXPLAINABILITY_FIGURE_DIR = FIGURE_DIR / "explainability"   # SHAP, permutation importance
ABLATION_FIGURE_DIR       = FIGURE_DIR / "ablation"         # branch ablation, gate analysis
METRIC_FIGURE_DIR         = FIGURE_DIR / "metrics"          # confusion matrices, ROC/PR overlays
TABLES_DIR                = OUTPUT_DIR / "tables"           # paper-ready CSV tables
DIAGNOSTICS_DIR           = OUTPUT_DIR / "diagnostics"      # VAD stats, feature-audit dumps, etc.

# Training pipeline outputs (train.py)
CHECKPOINT_DIR       = OUTPUT_DIR / "checkpoints"
LOG_DIR              = OUTPUT_DIR / "logs"
PREDICTIONS_DIR      = OUTPUT_DIR / "predictions"
METRICS_DIR          = OUTPUT_DIR / "metrics"
CONFUSION_MATRIX_DIR = OUTPUT_DIR / "confusion_matrix"
ROC_DIR              = OUTPUT_DIR / "roc"
EMBEDDINGS_DIR       = OUTPUT_DIR / "embeddings"

# Per-experiment output bundles (config/metrics/predictions/timing/checkpoint in one
# folder) for the budget-managed primary-detection sweep — see src/training/reporting.py
# ::save_experiment_bundle. Additive to the flat dirs above, which every run (old and
# new) continues to use; nothing reads EXPERIMENTS_DIR except the new sweep's own
# consumers (notebooks/03_training.ipynb, requirement-7 comparison table).
EXPERIMENTS_DIR      = OUTPUT_DIR / "experiments"

# Final reporting gate output (notebooks/06_results.ipynb). Holds only what
# survived the eligibility filter in src/results.py, plus the excluded/
# preliminary registers and the reproducibility manifest — the folder to hand
# off when writing the paper. Distinct from OUTPUT_DIR/paper_exports, which is
# a copy-and-format step; this is the gated source of truth.
RESULTS_DIR          = OUTPUT_DIR / "results"

# Filenames as actually distributed to this project (data/raw/) — the
# corpus's "audio/original" release, not "audio/normalized" or
# "audio/noisereduce" (both absent from these archives; see
# src/extraction.py's module docstring for what that means for this
# pipeline's preprocessing assumptions).
ARCHIVE_FILES = [
    "UASpeech_original_C.tgz",                  # healthy controls
    "UASpeech_original_FM.tgz",                 # dysarthric speakers
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

def _interleave_by_class(controls, dysarthric):
    """Alternate control and dysarthric speakers so that EVERY PREFIX of the
    result contains both classes (from length 2 onward), with the longer list's
    leftovers appended at the end.

    This is the order src.splits.iter_loso_folds walks, and therefore the order
    a detection LOSO run evaluates speakers in. The previous plain
    `CONTROL_IDS + DYSARTHRIC_IDS` put all 13 controls first, which meant any
    run that stopped early — a time budget, a crash, an interrupted session —
    had evaluated ONLY healthy controls. Since UA-Speech detection labels are
    speaker-level, each of those folds was single-class, so precision, recall,
    F1 and AUROC were all undefined for the entire partial run (see
    src.training.metrics' module docstring). That is exactly what happened to
    every truncated detection run in the pre-repair audit.

    A COMPLETE LOSO sweep is order-independent: it visits all 28 speakers and
    pools their predictions regardless of sequence, so no finished result
    changes and this is a bug fix rather than a protocol change. What it buys
    is that a PARTIAL run is now class-balanced, and therefore honestly
    reportable as a partial result instead of a degenerate one.
    """
    interleaved = []
    for pair in zip_longest(controls, dysarthric):
        interleaved.extend(speaker for speaker in pair if speaker is not None)
    return interleaved


ALL_SPEAKERS = _interleave_by_class(CONTROL_IDS, DYSARTHRIC_IDS)

# Severity mapping for the 15 dysarthric speakers (base-paper protocol)
SEVERITY_MAP = {
    "M01": "Very Low", "M04": "Very Low", "F03": "Very Low", "M12": "Very Low",
    "M07": "Low",      "F02": "Low",      "M16": "Low",
    "M05": "Mid",      "M11": "Mid",      "F04": "Mid",
    "M09": "High",     "M14": "High",     "M10": "High",
    "M08": "High",     "F05": "High",
}

# SECONDARY-ANALYSIS ONLY. The three-branch severity architecture's PRIMARY
# protocol (src.splits.iter_severity_loso_folds) is full-population Leave-
# One-Speaker-Out across all 15 dysarthric speakers — it does NOT drop any
# speaker. Discarding 3 of 15 speakers (20% of an already-small population)
# to force a balanced 3-per-class split is a real statistical-power cost, and
# the architecture handles class imbalance instead at the loss/metric level
# (class-weighted CORAL loss, macro-F1, balanced accuracy, per-class recall).
#
# This list is kept only so the ORIGINAL base-paper-style protocol — 3-per-
# class, leave-one-speaker-per-class-out (src.splits.build_severity_folds /
# get_severity_split, 81 = 3^4 combinations) — remains available as an
# explicitly-labeled, budget-capped SECONDARY sanity check, matching
# Javanmardi et al., ICASSP 2023 (arXiv:2309.14107): "one male speaker from
# 'very low' level of intelligibility and two male speakers from 'high'
# level of intelligibility" were left out. The paper doesn't name which two
# High speakers, so M08/M09 (both male, like the paper's excluded pair) is
# still a choice, not a reproduction of an unpublished detail.
#
# NOTE ON DOCUMENTATION DRIFT: an earlier README revision listed this set as
# ["M12", "M08", "F05"] — that dropped the female High speaker instead of a
# second male one, contradicting the paper's own "two male speakers" wording.
# The current value below is the corrected one; the README has been updated
# to match (see "Severity — secondary balanced analysis").
DROPPED_FOR_BALANCE = ["M12", "M08", "M09"]     # 1 Very Low, 2 High (all male)

# Primary severity evaluation protocol for the one-shot three-branch run.
# "full_loso": Leave-One-Speaker-Out across all 15 dysarthric speakers
# (src.splits.iter_severity_loso_folds) — the default and the one used for
# the reported result. "balanced_lopco": the legacy 3-per-class, 81-fold
# leave-one-per-class-out protocol above, run only as a secondary check.
SEVERITY_PRIMARY_PROTOCOL = "full_loso"

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
# Voice activity detection (Silero VAD, replacing torchaudio.functional.vad —
# see src/vad.py). torchaudio's vad() only trims LEADING silence; trailing
# silence and the fixed 4s pad/truncate window in load_and_preprocess() were
# together the source of the large near-constant tail visible in MFCC plots.
# ---------------------------------------------------------------------------
VAD_ENABLED         = True    # False -> load_and_preprocess() skips VAD entirely
                               # (only pad/truncate), for A/B comparison or if
                               # torch.hub is unreachable in an offline session.
VAD_THRESHOLD       = 0.5     # Silero speech-probability threshold (its own default)
VAD_MIN_SPEECH_MS   = 100     # shorter detected segments are not "speech" — UA-Speech
                               # utterances are short isolated words, so this is kept
                               # low relative to Silero's usual 250ms conversational default
VAD_MIN_SILENCE_MS  = 100     # internal gaps shorter than this stay merged into the
                               # surrounding speech segment, preserving natural pauses
VAD_SPEECH_PAD_MS   = 30      # padding added around the kept [first..last] speech span
VAD_SAMPLE_RATE     = 16_000  # must match TARGET_SR — Silero only accepts 8k/16k
VAD_STATS_PATH      = OUTPUT_DIR / "vad_stats.csv"   # per-utterance VAD stats (Stage 9)

# Temporal-preserving profile, feeding ONLY the Suprasegmental branch. Same
# Silero VAD trim as the speech-focused profile above (contiguous first-to-
# last speech, internal pauses already preserved by design), but with a much
# wider padding margin around the kept span, to protect the onset/offset
# dynamics (breathiness ramp-in, voicing decay) that a prosodic/temporal
# encoder needs and a 30ms margin can clip. Still pad/truncated to the same
# MAX_SAMPLES window for batching (see src.preprocessing.load_and_preprocess_supra),
# with valid_length propagated so padding is masked, not treated as silence.
SUPRA_VAD_SPEECH_PAD_MS = 150

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
# q/k/v only for the seven legacy ablation variants (src/training/models.py),
# kept unchanged so their already-defined behaviour doesn't shift. The new
# GatedFusionModel (src/models/gated_fusion.py) uses LORA_TARGET_MODULES_WIDE
# instead — adding the attention output projection gives this one-shot run's
# only trainable model slightly deeper adaptation, at a small, budget-checked
# parameter cost (see the plan's Part 2 Component 4).
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj"]   # self-attention layers
LORA_TARGET_MODULES_WIDE = ["q_proj", "k_proj", "v_proj", "out_proj"]

# Phase 6: attention-based fusion. The 768-dim deep and 128-dim acoustic frame
# sequences are projected into a shared FUSION_ATTN_DIM space so cross-attention
# between them is well-defined (queries and keys must share a dimension).
FUSION_ATTN_DIM     = 256
FUSION_ATTN_HEADS   = 4          # 256 / 4 = 64 dims per head
FUSION_ATTN_DROPOUT = 0.1
PRAAT_EMBED_DIM     = 256        # Praat pathway (Model F): FEATURE_COLUMNS -> one token

# ---------------------------------------------------------------------------
# Three-branch gated-fusion severity architecture (src/models/gated_fusion.py)
#
# Every branch is bottlenecked to a small, fixed dimension BEFORE fusion —
# the mechanism that forces each branch to keep only decision-relevant
# information (see the architecture plan's Part 2, Component 7). Read from
# here everywhere downstream (feature-audit printer, tests) rather than
# hardcoded, so a change here is guaranteed to be reflected everywhere.
# ---------------------------------------------------------------------------
LEARNED_EMBED_DIM   = 128        # Z_learned: wav2vec2+LoRA, mean-pooled, projected 768->128
SEGMENTAL_EMBED_DIM  = 64        # Z_segmental: MFCC+formant+HNR frame-CNN, projected 128->64
SUPRA_EMBED_DIM      = 64        # Z_supra: F0/voicing/energy frame-CNN, projected ->64
FUSED_EMBED_DIM       = SEGMENTAL_EMBED_DIM + SUPRA_EMBED_DIM + LEARNED_EMBED_DIM  # 256

# Framewise segmental channels: 13 MFCC + 13 delta + 13 delta-delta (existing)
# plus 3 framewise formants (F1-F3) and framewise HNR — see src.praat's
# extract_formant_sequence / extract_hnr_sequence.
SEGMENTAL_CHANNELS = 3 * N_MFCC + 3 + 1          # 43
# Framewise suprasegmental channels: F0 (semitones, interpolated), a binary
# voicing mask (1 = real pitch estimate, 0 = interpolated/unvoiced), and
# intensity/energy (dB) — see src.praat's extract_f0_sequence /
# extract_intensity_sequence.
SUPRA_CHANNELS = 3

# Fixed BEFORE the one training run (see the plan's Part 2, Component 12) —
# never swept or tuned against results from this run.
LAMBDA_COMP    = 0.05    # complementarity (cross-branch redundancy) penalty weight
LAMBDA_SPEAKER = 0.1     # speaker-invariance (gradient-reversal) loss weight
GRL_LAMBDA     = 1.0     # gradient-reversal strength inside the GRL layer itself

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
    for directory in (DATA_DIR, ARCHIVE_DIR, AUDIO_DIR, CORPUS_DOCS_DIR, OUTPUT_DIR, FIGURE_DIR,
                       ERROR_FIGURE_DIR, CHECKPOINT_DIR, LOG_DIR, PREDICTIONS_DIR,
                       METRICS_DIR, CONFUSION_MATRIX_DIR, ROC_DIR, EMBEDDINGS_DIR,
                       EXPERIMENTS_DIR, RESULTS_DIR,
                       SIGNAL_FIGURE_DIR, REPRESENTATION_FIGURE_DIR, EXPLAINABILITY_FIGURE_DIR,
                       ABLATION_FIGURE_DIR, METRIC_FIGURE_DIR, TABLES_DIR, DIAGNOSTICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
