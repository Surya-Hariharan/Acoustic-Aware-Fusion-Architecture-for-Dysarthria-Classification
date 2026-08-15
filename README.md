# Dysarthria Acoustic Fusion

A multi-modal acoustic-aware fusion architecture for dysarthria detection and severity classification on the UA-Speech corpus. A LoRA-adapted wav2vec 2.0 pathway is integrated with interpretable acoustic descriptors — MFCC and clinically-named Praat measures — rather than fused with them by simple concatenation, so the architecture stays grounded in deterministic acoustic physics instead of reasoning entirely inside an opaque latent space.

## Motivation

**Base paper.** Javanmardi, F., Tirronen, S., Kodali, M., Kadiri, S. R., and Alku, P. *Wav2vec-based Detection and Severity Level Classification of Dysarthria from Speech.* ICASSP 2023.

The base paper uses a **frozen** wav2vec 2.0 as a feature extractor feeding an SVM: layer 1 embeddings win for detection, final layer embeddings win for severity. Two limitations follow. The model reasons entirely in a hidden latent space, ignoring the physical acoustic correlates of pathological speech (slurred consonants, centralized vowels), which leaves it clinically uninterpretable. And because the backbone is frozen, it cannot adapt to pathological traits at all.

**Why not just wav2vec, and why not just concatenate what you add to it.** The 768-dim wav2vec embedding captures contextual phonetic information learned from a pretraining objective with no clinical grounding — it is powerful but cannot be explained to a speech pathologist by name. MFCC and the 31 Praat measures in `src/praat.py` (jitter, shimmer, HNR, CPPS, formants, intensity, rhythm) are weaker classifiers in isolation but are the deterministic spectral-envelope and voice-source descriptors the speech-pathology literature already ties to specific impairments — monopitch, vocal-fold instability, vowel-space centralization, breathy voice quality. The claim this project tests is that pairing the two is worth more than either alone (Phase 3's ablation), and that plain concatenation — the most common fusion strategy in prior work, and the weakest — is not the ceiling: an architecture that lets the two representations attend to each other should do better than one that just stacks their vectors (Phase 6). Explainability follows the same logic: rather than treating "why did the model decide this" as an afterthought, misclassifications are correlated directly against the Praat measures (`compare_error_vs_correct()`, Phase 5) and attention weights are inspected per prediction (`attention_weights()`, Phase 6) — so behaviour can be related back to acoustic characteristics a clinician recognizes, not just to an accuracy number.

## Architecture

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

## Dataset

UA-Speech: 765 isolated words per speaker across three blocks (B1–B3), captured by an eight-microphone array at 16 kHz. The audio is **not** redistributed here — obtain it from the dataset authors and place the archives in `data/archives/`. Corpus reference material (word-level MLF alignments, the lexicon/word list, the base-paper PDF, the corpus's own license and readme) lives in `data/uaspeech_corpus_docs/`, kept separate from the audio the pipeline actually scans.

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

**Severity** — the four classes hold 4/3/3/5 speakers, so `config.DROPPED_FOR_BALANCE` excludes M12, M08 and F05 to reach three per class, giving 3⁴ = 81 leave-one-speaker-per-class-out iterations.

> The base paper gives no explicit exclusion list, so that specific set of three speakers is an **assumption**. Confirm it with the team before treating any severity result as final.

## Preprocessing

Audio is resampled to 16 kHz mono, silence-trimmed by Silero VAD (`src/vad.py` — leading/trailing non-speech removed, internal pauses preserved, deterministic, with a safe fallback to the original waveform if VAD fails), and padded or truncated to a fixed four-second window. MFCCs are 13 coefficients plus delta and delta-delta (39-dim per frame), matching the base paper's baseline features. A single dataset class returns the waveform, the MFCC tensor, both labels, and the speaker ID, so the two pathways always see identical VAD-processed audio and identical splits.

## Repository layout

All logic lives in `src/`; only functions and architecture belong there. Notebooks are the sole front end — they call `src/`, run training, and store the resulting models/metrics, so behaviour never drifts between a script and a notebook. To change behaviour, edit the module, never a notebook.

```text
requirements.txt                Python dependencies
notebooks/
  01_data_pipeline.ipynb        Interactive data-pipeline driver; imports src/
  02_feature_analysis.ipynb     MFCC + VAD validation, Phase 4 Praat feature extraction,
                                 EDA, feature correlation, severity-group significance
  03_training.ipynb             Interactive training driver: pipeline sanity check,
                                 Phase 2 baseline reproduction, MFCC-only/Wav2Vec2-only/
                                 Fusion training, automatic experiment comparison,
                                 budget-managed primary-detection sweep (VAD + LoRA)
  04_model_analysis.ipynb       Phase 6 interpretability: ablation chart, embedding
                                 space (t-SNE/PCA/UMAP), attention maps, SHAP
  05_error_analysis.ipynb       Phase 5: misclassifications correlated against Praat features
  06_results.ipynb              Cross-experiment aggregation, significance tests,
                                 ROC/PR comparison, publication export
data/
  archives/                     Place the UA-Speech .tgz archives here
  extracted/                    Extracted .wav files land here (one folder per speaker)
  uaspeech_corpus_docs/         Corpus reference material: mlf/, doc/ (lexicon, wordlist,
                                 base-paper PDF), readme_UASpeech.txt, UASPEECH_LICENSE.txt
outputs/                        Generated figures, manifest, and training artifacts (gitignored)
  checkpoints/ logs/ predictions/ metrics/ confusion_matrix/ roc/ embeddings/
  experiments/<name>/           Per-experiment bundle (config.json, metrics.json,
                                 predictions.csv, timing.json, checkpoint/) for the
                                 budget-managed primary-detection sweep — additive to
                                 the flat dirs above, not a replacement for them
src/
  config.py                     Paths, speaker ground truth, label maps, hyperparameters
  console.py                    Aligned console output helpers
  extraction.py                 Archive extraction
  scanning.py                   Filename parsing, verification, mic filter, severity labels
  preprocessing.py              Resampling, Silero VAD trimming, padding, MFCC extraction
  vad.py                        Silero VAD wrapper (leading/trailing trim, fallback,
                                 per-utterance stats)
  praat.py                      Phase 4: 31 Praat features (F0, jitter, shimmer, HNR, CPPS,
                                 formants, intensity, rhythm) + severity-group significance test
  error_analysis.py             Phase 5: per-error diagnostics, Praat-correlation explainability
  model_analysis.py             Phase 6: ablation chart, embedding projections (t-SNE/PCA/UMAP),
                                 attention heatmaps, SHAP feature importance
  results.py                    Cross-experiment aggregation, paired significance tests,
                                 ROC/PR overlay, publication styling, paper export
  dataset.py                    UASpeechDataset
  splits.py                     LOSO folds (detection), balanced 81-fold (severity)
  visualization.py              EDA + Praat-feature figures + feature correlation heatmap
  models/
    deep_pathway.py             wav2vec 2.0 (+ optional LoRA) -> 768-dim
    acoustic_pathway.py         1D-CNN over MFCC -> 128-dim
    concat_fusion.py            Model D: concatenation + classification head
    attention_fusion.py         Models E/F: bidirectional cross-attention fusion,
                                 optionally with the Praat token as a third pathway
  training/
    models.py                   Model factory: acoustic / deep_frozen / deep_lora / fusion /
                                 attention_fusion / attention_fusion_praat
    data.py                     Manifest loading, stratified train/val split, DataLoaders
    runner.py                   TrainingConfig + run_training(): the fold loop notebooks call
    baseline.py                 Phase 2: frozen wav2vec embeddings + linear SVM per fold
    engine.py                   Train/eval epoch loop: AMP, gradient clipping, optimizer
    metrics.py                  Accuracy/precision/recall/specificity/F1/AUROC
    early_stopping.py           Early stopping on validation loss
    checkpoint.py                Checkpoint save/load
    reporting.py                Predictions/metrics/confusion-matrix/ROC/embeddings I/O
    budget.py                   ExperimentBudgetManager: measures real per-variant cost and
                                 allocates a wall-clock budget across the primary sweep
```

## Usage

```bash
pip install -r requirements.txt
```

Copy `UASpeech_normalized_C.tgz` and `UASpeech_normalized_FM.tgz` into `data/archives/`, then run the notebooks in order.

```bash
# 1. Data pipeline
jupyter notebook notebooks/01_data_pipeline.ipynb

# 2. Feature analysis (MFCC + VAD validation + Praat)
jupyter notebook notebooks/02_feature_analysis.ipynb

# 3. Training
jupyter notebook notebooks/03_training.ipynb

# 4. Model analysis (ablation chart, embeddings, attention, SHAP)
jupyter notebook notebooks/04_model_analysis.ipynb

# 5. Error analysis
jupyter notebook notebooks/05_error_analysis.ipynb

# 6. Results (aggregation, significance tests, publication export)
jupyter notebook notebooks/06_results.ipynb
```

Notebook 1 scans the extracted audio, verifies the 28-speaker ground truth, filters to M6, checks per-speaker word counts, attaches severity labels, summarizes both split protocols, builds the dataset, and writes `outputs/m6_manifest.csv`.

Notebook 2 loads that manifest, runs the dataset-wide EDA (moved here from notebook 1, since it's exploratory feature analysis, not pipeline construction), extracts the Phase 4 Praat features, and reports feature statistics/correlation/severity-group significance.

Notebook 3 loads the manifest (regenerating it if missing) and drives `src/training/runner.py`'s `run_training(df, TrainingConfig(...))`:

```python
from src.training.runner import TrainingConfig, run_training

cfg = TrainingConfig(task="detection", model="fusion")   # LOSO, 28 folds
# cfg = TrainingConfig(task="severity", model="acoustic")  # balanced 81-fold
summary, pooled = run_training(df_m6, cfg)
```

`model` is one of `acoustic`, `deep_frozen`, `deep_lora`, `fusion_frozen`, `fusion`, `attention_fusion`, `attention_fusion_praat` (see `src/training/models.py::MODEL_DESCRIPTIONS` for what each maps to in the ablation study — the last one is `notebooks/02_feature_analysis.ipynb`'s Praat features on top of Phase 6's cross-attention contribution). Every fold writes a checkpoint, TensorBoard log (`tensorboard --logdir outputs/logs`), predictions CSV, metrics JSON, confusion matrix, ROC curve, and fused embedding under `outputs/`; run-level metrics are reported both as a per-fold mean ± std and pooled across all folds. `TrainingConfig(..., max_folds=1, epochs=1, limit_samples=24)` gives a fast pipeline sanity check before a real run — notebook 3's Stage 1 does exactly this. Phase 2's baseline reproduction (frozen wav2vec 2.0 + linear SVM, via `src/training/baseline.py`) lives in the same notebook, which groups every ablation variant into three families (MFCC-only, Wav2Vec2-only, Fusion) so each can be run, resumed, or extended independently.

## Status

| Component | Owner role | Status |
|---|---|---|
| Data pipeline and preprocessing | — | Complete and verified |
| Deep Pathway (wav2vec 2.0, optional LoRA) | Deep Pathway Lead | Trainable via `run_training()`, not yet trained to convergence |
| Acoustic Pathway (1D-CNN over MFCC) | Acoustic Pathway Lead | Trainable via `run_training()`, not yet trained to convergence |
| Fusion: training loop, checkpointing, metrics, logging (Phase 1) | Fusion Architect | Complete — see `src/training/runner.py` + `notebooks/03_training.ipynb` |
| Baseline reproduction (Phase 2) | — | Frozen wav2vec + SVM implemented (`src/training/baseline.py`); full-scale comparison run pending |
| Ablation study (Phase 3) | — | All six variants implemented; full 28-fold GPU run pending |
| Praat acoustic analysis (Phase 4) | — | Code complete: 31 features, significance test, feature correlation — see `src/praat.py` |
| Error analysis (Phase 5) | — | Code complete: per-error diagnostics + Praat-correlation explainability — needs a re-trained run |
| Attention fusion + Praat pathway (Phase 6, steps 1-2) | — | Code complete (Models E, F); full-scale run + multi-task heads (step 3) pending. Explainability (step 4) done: attention maps + SHAP feature importance in `notebooks/04_model_analysis.ipynb` |

## License

MIT — see [LICENSE](LICENSE). The licence covers this source code only; the UA-Speech database carries its own separate licence and data use agreement (see `data/uaspeech_corpus_docs/UASPEECH_LICENSE.txt` once the corpus is downloaded — academic/government research use only, no redistribution).

## Acknowledgments

With thanks and regards to the creators and maintainers of the UA-Speech corpus — H. Kim, M. Hasegawa-Johnson, A. Perlman, J. Gunderson, T. Huang, K. Watkin, and S. Frame at the University of Illinois at Urbana-Champaign — for building and sharing this dataset for dysarthric speech research:

> H. Kim, M. Hasegawa-Johnson, A. Perlman, J. Gunderson, T. Huang, K. Watkin, and S. Frame, "Dysarthric Speech Database for Universal Access Research," *Interspeech*, 2008.

This project would not be possible without their work, or without the participants who contributed their speech recordings to advance assistive-technology research.
