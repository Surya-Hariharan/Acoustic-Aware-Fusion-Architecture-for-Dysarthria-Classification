# Dysarthria Acoustic Fusion

A multi-modal acoustic-aware fusion architecture for dysarthria detection and severity classification on the UA-Speech corpus. A LoRA-adapted wav2vec 2.0 pathway is integrated with interpretable acoustic descriptors — MFCC and clinically-named Praat measures — rather than fused with them by simple concatenation, so the architecture stays grounded in deterministic acoustic physics instead of reasoning entirely inside an opaque latent space.

## Motivation

**Base paper.** Javanmardi, F., Tirronen, S., Kodali, M., Kadiri, S. R., and Alku, P. *Wav2vec-based Detection and Severity Level Classification of Dysarthria from Speech.* ICASSP 2023.

The base paper uses a **frozen** wav2vec 2.0 as a feature extractor feeding an SVM: layer 1 embeddings win for detection, final layer embeddings win for severity. Two limitations follow. The model reasons entirely in a hidden latent space, ignoring the physical acoustic correlates of pathological speech (slurred consonants, centralized vowels), which leaves it clinically uninterpretable. And because the backbone is frozen, it cannot adapt to pathological traits at all.

**Why not just wav2vec, and why not just concatenate what you add to it.** The 768-dim wav2vec embedding captures contextual phonetic information learned from a pretraining objective with no clinical grounding — it is powerful but cannot be explained to a speech pathologist by name. MFCC and the 31 Praat measures in `src/praat.py` (jitter, shimmer, HNR, CPPS, formants, intensity, rhythm) are weaker classifiers in isolation but are the deterministic spectral-envelope and voice-source descriptors the speech-pathology literature already ties to specific impairments — monopitch, vocal-fold instability, vowel-space centralization, breathy voice quality. The claim this project tests is that pairing the two is worth more than either alone (Phase 3's ablation), and that plain concatenation — the most common fusion strategy in prior work, and the weakest — is not the ceiling: an architecture that lets the two representations attend to each other should do better than one that just stacks their vectors (Phase 6). Explainability follows the same logic: rather than treating "why did the model decide this" as an afterthought, misclassifications are correlated directly against the Praat measures (`compare_error_vs_correct()`, Phase 5) and attention weights are inspected per prediction (`attention_weights()`, Phase 6) — so behaviour can be related back to acoustic characteristics a clinician recognizes, not just to an accuracy number.

## Architecture

> **This section describes the legacy detection/severity ablation ladder** (Phases 1-6),
> preserved and isolated under `notebooks/legacy/` for reference and any future detection
> work. It is **not** the architecture the one-shot severity experiment uses — see
> "Three-branch gated-fusion severity architecture" immediately below for the primary,
> currently-reported architecture.

Three pathways feed a classification head; how they combine is itself the ablation ladder this project runs (Phase 3 and 6 — see the status table below).

| Pathway | Input | Model | Output |
|---|---|---|---|
| Deep | Raw 16 kHz waveform | wav2vec 2.0, LoRA adapters on the self-attention projections | 768-dim latent embedding |
| Acoustic | 39-dim MFCC (13 + Δ + ΔΔ) | Lightweight 1D-CNN | 128-dim physical embedding |
| Praat | 31 handcrafted features from the original (non-VAD-trimmed) audio | Standardized per fold, projected to one token | 256-dim interpretable token |

| Fusion strategy | How the pathways combine | Ablation role |
|---|---|---|
| Concatenation | Deep + Acoustic embeddings stacked (896-dim) → classification head | Model D — the common baseline this project argues against |
| Attention fusion | Deep and Acoustic streams cross-attend bidirectionally over their frame sequences (not just the pooled vectors) before the head | Model E — the primary contribution |
| Attention fusion + Praat | Model E plus the Praat token appended to both attention contexts and concatenated into the final embedding | Model F — isolates what the handcrafted features add on top of learned cross-attention |

LoRA keeps the backbone adaptable without overfitting on a small clinical corpus. Routing MFCC and Praat features directly into the decision means the interpretable descriptors are load-bearing rather than decorative — and because attention fusion operates over full frame sequences rather than single pooled vectors, `attention_weights()` exposes which pathway (and, for Model F, whether the Praat token) the model actually leaned on for a given prediction.

### Three-branch gated-fusion severity architecture (one-shot run)

The seven models/two fusion strategies above are the detection/severity **ablation ladder** and remain in the codebase, but training compute for dysarthria severity is affordable only **once** (see the architecture audit in `.claude`'s plan history for the full reasoning). For that one-shot run, severity classification uses a purpose-built architecture instead, in `src/models/gated_fusion.py`:

| Branch | Input | Encoder | Output |
|---|---|---|---|
| Learned | Raw waveform (speech-focused VAD profile) | wav2vec 2.0 + LoRA (`q/k/v/out_proj` — wider than the ablation ladder's `q/k/v`), masked mean-pool, projected 768→128 | 128-dim **Z_learned** |
| Segmental | MFCC+Δ+ΔΔ (39ch) + framewise formants F1–F3 + framewise HNR (4ch) = 43ch/frame, speech-focused profile | 3-layer 1D-CNN, masked mean-pool, projected 128→64 | 64-dim **Z_segmental** |
| Suprasegmental | Framewise F0 (semitones, voicing-interpolated) + voicing mask + intensity, 3ch/frame, **temporal-preserving** VAD profile (wider padding margin — see `src.preprocessing.load_and_preprocess_supra`) | 2-layer 1D-CNN, masked mean-pool, projected →64 | 64-dim **Z_supra** |

The three bottlenecked embeddings are combined by a **learned gate** (not concatenation or cross-attention) — `GateNetwork` produces softmax weights `(g_learned, g_segmental, g_supra)` from the three embeddings, logged per batch, so branch reliance is inspectable rather than assumed. A **cross-branch redundancy penalty** (`src.losses.redundancy_penalty`, Barlow-Twins-style batch cross-covariance, mean-normalized so it means the same thing across the three differently-sized branch pairs) discourages the branches from encoding the same information. A **gradient-reversal speaker head** (`SpeakerHead`) on the fused representation discourages it from carrying speaker identity. The severity head is **ordinal** (`CoralHead`, CORAL — Cao et al. 2020) rather than a plain 4-way softmax, since Very Low < Low < Mid < High is a real ordering a nominal classifier throws away; `src.training.metrics.compute_metrics` reports `ordinal_mae` alongside the standard classification metrics. Every hyperparameter introduced by this architecture (`config.LAMBDA_COMP`, `config.LAMBDA_SPEAKER`, embedding dimensions) is fixed in `src/config.py` before the run — none of it is tuned against this run's own results.

`GatedFusionModel.ablate()` supports post-hoc, no-retraining branch ablation (zero one branch's embedding before the gate re-normalizes over the remaining two) — the primary tool for measuring each branch's actual contribution, run against the one trained checkpoint rather than by training separate model families.

## Dataset

UA-Speech: 765 isolated words per speaker across three blocks (B1–B3), captured by an eight-microphone array at 16 kHz. The audio is **not** redistributed here — obtain it from the dataset authors and place the archives in `data/raw/`. The corpus ships three audio releases (`original`, `normalized`, `noisereduce`); this project's archives contain only `audio/original`, so extraction pulls from that variant — see `src/extraction.py`'s module docstring for what that implies for preprocessing (no corpus-level loudness normalization). Corpus reference material (word-level MLF alignments, the lexicon/word list, the base-paper PDF, the corpus's own license and readme) lives in `data/uaspeech_corpus_docs/`, kept separate from the audio the pipeline actually scans.

Following the base-paper protocol, this project uses **microphone channel M6 only**, across all blocks and all word categories, with no word-type filtering.

**Verified composition: 28 speakers, 21,420 M6 utterances** (11,475 dysarthric, 9,945 control), 765 words per speaker with no incomplete speakers.

- 13 healthy controls: CF02–CF05, CM01, CM04–CM06, CM08–CM10, CM12, CM13
- 15 dysarthric speakers: F02–F05, M01, M04, M05, M07–M12, M14, M16

Severity mapping:

| Severity | Speakers |
|---|---|
| Very Low | M01, M04, F03, M12 |
| Low | M07, F02, M16 |
| Mid | M05, M11, F04 |
| High | M09, M14, M10, M08, F05 |

### Data verification note

An earlier exploratory scan reported an inconsistent speaker and microphone count due to a filename parsing defect. It extracted the microphone channel by searching for the first underscore-separated token beginning with `M`, which matched male dysarthric speaker IDs such as `M01` before ever reaching the trailing mic token. The scan now takes the channel **positionally** from the final token.

The corrected scan also skips macOS resource-fork duplicates (`._` prefix), which the extracted archive contains in equal number to the real `.wav` files and which otherwise double the apparent file count.

## Evaluation protocol

**Detection** — Leave-One-Speaker-Out across all 28 speakers (28 folds).

**Severity — PRIMARY: full-population Leave-One-Speaker-Out.** All 15 dysarthric speakers (4 Very Low / 3 Low / 3 Mid / 5 High), no speaker dropped — `src.splits.iter_severity_loso_folds`, 15 folds. Class imbalance is handled at the loss/metric level instead of by removing speakers: the severity head's class-weighted CORAL ordinal loss, plus macro-F1, balanced accuracy, and per-class recall reported alongside pooled accuracy. This is the protocol the one-shot three-branch architecture (see below) actually trains and reports against — discarding 20% of an already-small speaker population to force a balanced split was judged too large a statistical-power cost to be the primary evaluation.

**Severity — SECONDARY (base-paper-style sanity check only).** The four classes hold 4/3/3/5 speakers; `config.DROPPED_FOR_BALANCE` (`M12`, `M08`, `M09`) excludes three to reach three per class, giving 3⁴ = 81 leave-one-speaker-per-class-out iterations (`src.splits.build_severity_folds`). The base paper gives no explicit exclusion list, so this specific set of three speakers remains an **assumption** — kept here only as an explicitly-labeled, budget-capped secondary check against the primary result above, not as the reported number.

## Preprocessing

Audio is resampled to 16 kHz mono, silence-trimmed by Silero VAD (`src/vad.py` — leading/trailing non-speech removed, internal pauses preserved, deterministic, with a safe fallback to the original waveform if VAD fails), and padded or truncated to a fixed four-second window. MFCCs are 13 coefficients plus delta and delta-delta (39-dim per frame), matching the base paper's baseline features. A single dataset class returns the waveform, the MFCC tensor, both labels, and the speaker ID, so the two pathways always see identical VAD-processed audio and identical splits.

## Repository layout

All logic lives in `src/`; only functions and architecture belong there. Notebooks are the sole front end — they call `src/`, run training, and store the resulting models/metrics, so behaviour never drifts between a script and a notebook. To change behaviour, edit the module, never a notebook.

```text
requirements.txt                Python dependencies
notebooks/
  01_data_pipeline.ipynb        Manifest build (scan/verify/filter/label/split/dataset) +
                                 VAD/padding root-cause investigation + the three-branch
                                 architecture's data-pipeline audit (primary severity LOSO
                                 protocol, branch bottleneck dims, framewise segmental/
                                 suprasegmental feature audit, F0-validity diagnostics, the
                                 two VAD profiles side by side, end-to-end shape sanity check)
  02_feature_analysis.ipynb     MFCC + VAD validation, Phase 4 Praat feature extraction
                                 (still required — the SHAP-surrogate methodology's input
                                 table), EDA, feature correlation, severity-group significance
  03_training.ipynb             The one-shot three-branch severity architecture's training
                                 interface: frozen configuration, dataset/LOSO-fold summary,
                                 model instantiation + parameter/branch-dimension audit, dummy
                                 forward pass, a bounded smoke test, configuration freeze, then
                                 the MODE-gated real 15-fold run (defaults to NOT executing it)
  04_model_analysis.ipynb       Branch embedding statistics/dimensionality, inference-time
                                 branch ablation, gate analysis, PCA/UMAP (severity- and
                                 speaker-colored, every embedding type), complementarity
                                 heatmap, SHAP (feature-group + per-class), permutation
                                 importance — all against the frozen checkpoint, post-hoc
  05_error_analysis.ipynb       Confusion matrix, per-class performance, ordinal-error
                                 breakdown, correct/incorrect-prediction galleries,
                                 representative-utterance signal panels + branch embeddings
                                 + gate values, prediction-vs-actual severity
  06_results.ipynb              The paper-ready notebook: final/per-class metrics, branch
                                 ablation, gate contribution, feature-group importance
                                 (SHAP + permutation), PCA/UMAP, representative signal
                                 figures, speaker-invariance + complementarity analysis,
                                 limitations table, full paper-table CSV export
  legacy/                       The original 7-model detection/severity ablation ladder
                                 (Phases 1-6), isolated here — not part of the one-shot
                                 severity experiment, kept for reference/future detection work
    03_detection_ablation_training.ipynb    (was 03_training.ipynb)
    04_detection_model_analysis.ipynb       (was 04_model_analysis.ipynb)
    05_detection_error_analysis.ipynb       (was 05_error_analysis.ipynb)
    06_detection_results.ipynb              (was 06_results.ipynb)
data/
  raw/                          Place the UA-Speech .tgz archives here
  extracted/                    Extracted .wav files land here (one folder per speaker)
  uaspeech_corpus_docs/         Corpus reference material: mlf/, doc/ (lexicon, wordlist,
                                 base-paper PDF), readme_UASpeech.txt, UASPEECH_LICENSE.txt
outputs/                        Generated figures, manifest, and training artifacts (gitignored)
  checkpoints/ logs/ predictions/ metrics/ confusion_matrix/ roc/ embeddings/
  figures/signals/ figures/representations/ figures/explainability/
  figures/ablation/ figures/metrics/       Three-branch architecture's figure subdirectories
  tables/                       Paper-ready CSV tables (src.results.export_paper_tables)
  diagnostics/                  VAD stats, feature-audit dumps
  experiments/<name>/           Per-experiment bundle (config.json, metrics.json,
                                 predictions.csv, timing.json, checkpoint/) for the
                                 budget-managed primary-detection sweep — additive to
                                 the flat dirs above, not a replacement for them
  results/frozen_config.json    The one-shot run's frozen configuration (git commit,
                                 software versions, architecture, hyperparameters — see
                                 src.training.reporting.write_frozen_config)
src/
  config.py                     Paths, speaker ground truth, label maps, hyperparameters
                                 (legacy ablation ladder + the three-branch architecture)
  console.py                    Aligned console output + tqdm progress helpers
  extraction.py                 Archive extraction
  scanning.py                   Filename parsing, verification, mic filter, severity labels
  preprocessing.py              Resampling, Silero VAD trimming, padding, MFCC extraction,
                                 the two explicit VAD profiles (speech-focused / temporal-
                                 preserving), segmental/suprasegmental framewise extraction
  vad.py                        Silero VAD wrapper (leading/trailing trim, fallback,
                                 per-utterance stats, per-call speech-pad-ms override)
  praat.py                      Utterance-level Praat features (F0, jitter, shimmer, HNR,
                                 CPPS, formants, intensity, rhythm) + severity-group
                                 significance test + framewise F0/voicing/intensity and
                                 formant/HNR extraction for the three-branch architecture
  losses.py                     CORAL ordinal loss, cross-branch complementarity
                                 (redundancy) penalty, gradient-reversal layer
  error_analysis.py             Per-error diagnostics, Praat-correlation explainability,
                                 the representative-utterance signal-panel figure/driver
  model_analysis.py             Ablation chart, embedding projections (t-SNE/PCA/UMAP,
                                 severity/speaker-colored, every branch), attention heatmaps,
                                 SHAP (+ feature-group + per-class), permutation importance,
                                 inference-time branch ablation, gate analysis, complementarity
                                 heatmap — for both the legacy ladder and the three-branch model
  results.py                    Cross-experiment aggregation, paired significance tests,
                                 ROC/PR overlay, publication styling, paper-table export
                                 (dataset/architecture/metrics/ablation/gate/explainability
                                 tables + the fixed limitations table)
  dataset.py                    UASpeechDataset (legacy ladder + three-branch tensors)
  splits.py                     LOSO folds (detection), legacy balanced 81-fold (severity,
                                 secondary only), primary full-population severity LOSO
  visualization.py              EDA + Praat-feature figures + feature correlation heatmap +
                                 the VAD-validation panel figure/driver
  models/
    deep_pathway.py             Legacy: wav2vec 2.0 (+ optional LoRA) -> 768-dim
    acoustic_pathway.py         Legacy: 1D-CNN over MFCC -> 128-dim
    concat_fusion.py            Legacy Model D: concatenation + classification head
    attention_fusion.py         Legacy Models E/F: bidirectional cross-attention fusion,
                                 optionally with the Praat token as a third pathway
    gated_fusion.py             GatedFusionModel: the three-branch architecture — learned
                                 gate, CORAL head, speaker-adversarial head, branch ablation
    segmental_pathway.py        Segmental branch: 43ch/frame 1D-CNN -> 64-dim
    suprasegmental_pathway.py   Suprasegmental branch: 3ch/frame 1D-CNN -> 64-dim
  training/
    models.py                   Model factory: the legacy 7-variant registry +
                                 SEVERITY_MODEL_NAME ("gated_fusion_three_branch")
    data.py                     Manifest loading, stratified train/val split, DataLoaders,
                                 per-fold speaker-label map for the adversarial head
    runner.py                   TrainingConfig + run_training(): the fold loop notebooks call
                                 (legacy LOSO/balanced-81-fold + primary severity LOSO)
    baseline.py                 Phase 2: frozen wav2vec embeddings + linear SVM per fold
    engine.py                   Train/eval epoch loop: AMP, gradient clipping, optimizer,
                                 the training_step hook for multi-term losses, branch-
                                 embedding/gate-weight collection
    metrics.py                  Accuracy/balanced accuracy/precision/recall/specificity/
                                 F1/weighted-F1/AUROC/ordinal MAE, NaN-safe for undefined
    early_stopping.py           Early stopping on validation loss
    checkpoint.py                Checkpoint save/load
    reporting.py                Predictions/metrics/confusion-matrix/ROC/embeddings I/O,
                                 feature audit + final-run-configuration printers, frozen-
                                 config write/guard (git commit, software versions)
    budget.py                   ExperimentBudgetManager: measures real per-variant cost and
                                 allocates a wall-clock budget across the primary sweep
tests/                          pytest suite: architecture shapes, CORAL/complementarity/
                                 GRL numerics, severity-LOSO fold construction, suprasegmental
                                 masking, frame alignment, frozen-config round-trip, plus the
                                 pre-existing MFCC-masking and experiment-validity guards
```

## Usage

```bash
pip install -r requirements.txt
```

Copy `UASpeech_original_C.tgz` and `UASpeech_original_FM.tgz` into `data/raw/`, then run the notebooks in order.

```bash
# 1. Data pipeline + three-branch architecture data audit
jupyter notebook notebooks/01_data_pipeline.ipynb

# 2. Feature analysis (MFCC + VAD validation + Praat)
jupyter notebook notebooks/02_feature_analysis.ipynb

# 3. Training — the one-shot three-branch severity experiment
jupyter notebook notebooks/03_training.ipynb

# 4. Model analysis (branch ablation, gate analysis, embeddings, SHAP)
jupyter notebook notebooks/04_model_analysis.ipynb

# 5. Error analysis
jupyter notebook notebooks/05_error_analysis.ipynb

# 6. Results (paper-ready tables, figures, limitations)
jupyter notebook notebooks/06_results.ipynb

# Legacy detection/severity ablation ladder (isolated, not part of the one-shot run)
jupyter notebook notebooks/legacy/03_detection_ablation_training.ipynb
```

Notebook 1 scans the extracted audio, verifies the 28-speaker ground truth, filters to M6, checks per-speaker word counts, attaches severity labels, summarizes the detection LOSO and legacy balanced-severity protocols, builds the dataset, writes `outputs/m6_manifest.csv`, then audits the three-branch architecture's data-pipeline needs: the primary 15-speaker severity LOSO protocol, branch bottleneck dimensions (from a real dummy forward pass), framewise segmental/suprasegmental feature shapes on a sample utterance, F0-validity statistics, the two VAD preprocessing profiles side by side, and a final end-to-end shape sanity check against a live `GatedFusionModel`.

Notebook 2 loads that manifest, runs the dataset-wide EDA, extracts the Phase 4 Praat features (still required — this is the input table the SHAP-surrogate explainability methodology reads), and reports feature statistics/correlation/severity-group significance.

Notebook 3 is the one-shot training interface. It loads the manifest, builds and prints the frozen configuration (git commit, software versions, architecture, hyperparameters), summarizes the dataset and the 15-fold severity LOSO protocol, instantiates `GatedFusionModel` and audits its parameter counts and branch dimensions, runs a dummy forward pass, then a **bounded smoke test** (one fold, one epoch, capped samples — never treated as a scientific result), freezes the configuration (`write_frozen_config`), and only then reaches a `MODE`-gated real-training cell:

```python
MODE = "SMOKE"  # default — the real 15-fold run does NOT execute
# MODE = "FINAL"  # deliberate human action only — the one-shot experiment
```

This mirrors `notebooks/legacy/03_detection_ablation_training.ipynb`'s own `MODE` safety convention. The legacy ablation ladder is still driven the same way it always was, via `src/training/runner.py`'s `run_training(df, TrainingConfig(...))`:

```python
from src.training.runner import TrainingConfig, run_training

cfg = TrainingConfig(task="detection", model="fusion")   # LOSO, 28 folds
# cfg = TrainingConfig(task="severity", model="gated_fusion_three_branch")  # primary, 15-fold LOSO
summary, pooled = run_training(df_m6, cfg)
```

`model` is one of the legacy `acoustic`, `deep_frozen`, `deep_lora`, `fusion_frozen`, `fusion`, `attention_fusion`, `attention_fusion_praat` (see `src/training/models.py::MODEL_DESCRIPTIONS`), or `"gated_fusion_three_branch"` (`src.training.models.SEVERITY_MODEL_NAME`) for the three-branch architecture. Every fold writes a checkpoint, TensorBoard log (`tensorboard --logdir outputs/logs`), predictions CSV, metrics JSON, confusion matrix, ROC curve, and embeddings (fused, plus per-branch + gate weights for the three-branch model) under `outputs/`; run-level metrics are reported both as a per-fold mean ± std and pooled across all folds. `TrainingConfig(..., max_folds=1, epochs=1, limit_samples=24)` gives a fast pipeline sanity check before a real run.

## Status

### Three-branch severity architecture (primary, one-shot)

| Component | Status |
|---|---|
| Learned / Segmental / Suprasegmental branches, bottlenecks, gated fusion | Implemented and unit-tested (`src/models/gated_fusion.py`, `segmental_pathway.py`, `suprasegmental_pathway.py`) |
| CORAL ordinal head, complementarity penalty, speaker-adversarial GRL | Implemented and unit-tested (`src/losses.py`) |
| Data pipeline (two VAD profiles, framewise segmental/suprasegmental extraction) | Complete and verified (synthetic-audio smoke tests; no real UA-Speech audio in this checkout) |
| Training interface (`notebooks/03_training.ipynb`) | Built, syntax- and import-validated; smoke-tested via equivalent scratchpad scripts (real wav2vec2 weights, dummy tensors) |
| Post-hoc analysis (`04_model_analysis.ipynb`, `05_error_analysis.ipynb`, `06_results.ipynb`) | Built, syntax- and import-validated; every underlying function individually smoke-tested |
| **The one-shot 15-fold training run itself** | **Not executed.** `notebooks/03_training.ipynb`'s real-training cell defaults to `MODE = "SMOKE"` |

### Legacy detection/severity ablation ladder (`notebooks/legacy/`)

| Component | Owner role | Status |
|---|---|---|
| Data pipeline and preprocessing | — | Complete and verified |
| Deep Pathway (wav2vec 2.0, optional LoRA) | Deep Pathway Lead | Trainable via `run_training()`, not yet trained to convergence |
| Acoustic Pathway (1D-CNN over MFCC) | Acoustic Pathway Lead | Trainable via `run_training()`, not yet trained to convergence |
| Fusion: training loop, checkpointing, metrics, logging (Phase 1) | Fusion Architect | Complete — see `src/training/runner.py` + `notebooks/legacy/03_detection_ablation_training.ipynb` |
| Baseline reproduction (Phase 2) | — | Frozen wav2vec + SVM implemented (`src/training/baseline.py`); full-scale comparison run pending |
| Ablation study (Phase 3) | — | All six variants implemented; full 28-fold GPU run pending |
| Praat acoustic analysis (Phase 4) | — | Code complete: 31 features, significance test, feature correlation — see `src/praat.py` |
| Error analysis (Phase 5) | — | Code complete: per-error diagnostics + Praat-correlation explainability — needs a re-trained run |
| Attention fusion + Praat pathway (Phase 6, steps 1-2) | — | Code complete (Models E, F); full-scale run + multi-task heads (step 3) pending. Explainability (step 4) done: attention maps + SHAP feature importance in `notebooks/legacy/04_detection_model_analysis.ipynb` |

## License

MIT — see [LICENSE](LICENSE). The licence covers this source code only; the UA-Speech database carries its own separate licence and data use agreement (see `data/uaspeech_corpus_docs/UASPEECH_LICENSE.txt` once the corpus is downloaded — academic/government research use only, no redistribution).

## Acknowledgments

With thanks and regards to the creators and maintainers of the UA-Speech corpus — H. Kim, M. Hasegawa-Johnson, A. Perlman, J. Gunderson, T. Huang, K. Watkin, and S. Frame at the University of Illinois at Urbana-Champaign — for building and sharing this dataset for dysarthric speech research:

> H. Kim, M. Hasegawa-Johnson, A. Perlman, J. Gunderson, T. Huang, K. Watkin, and S. Frame, "Dysarthric Speech Database for Universal Access Research," *Interspeech*, 2008.

This project would not be possible without their work, or without the participants who contributed their speech recordings to advance assistive-technology research.
