# Dysarthria Acoustic Fusion

A multi-modal acoustic-aware fusion architecture for dysarthria detection and severity classification on the UA-Speech corpus. A LoRA-adapted wav2vec 2.0 pathway and a classical MFCC pathway are trained jointly and fused before the classification head, grounding a foundation model in deterministic acoustic physics.

## Motivation

**Base paper.** Javanmardi, F., Tirronen, S., Kodali, M., Kadiri, S. R., and Alku, P. *Wav2vec-based Detection and Severity Level Classification of Dysarthria from Speech.* ICASSP 2023.

The base paper uses a **frozen** wav2vec 2.0 as a feature extractor feeding an SVM: layer 1 embeddings win for detection, final layer embeddings win for severity. Two limitations follow. The model reasons entirely in a hidden latent space, ignoring the physical acoustic correlates of pathological speech (slurred consonants, centralized vowels), which leaves it clinically uninterpretable. And because the backbone is frozen, it cannot adapt to pathological traits at all.

## Architecture

A dual-pathway network replaces the single-stream frozen approach.

| Pathway | Input | Model | Output |
|---|---|---|---|
| Deep | Raw 16 kHz waveform | wav2vec 2.0, LoRA adapters on the self-attention projections | 768-dim latent embedding |
| Acoustic | 39-dim MFCC (13 + Δ + ΔΔ) | Lightweight 1D-CNN | 128-dim physical embedding |
| Fusion | Both embeddings | Concatenation (896-dim) → classification head | Detection (2-class) or severity (4-class) logits |

LoRA keeps the backbone adaptable without overfitting on a small clinical corpus, and routing MFCCs directly into the final decision means the frequency-domain features are load-bearing rather than decorative.

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

Audio is resampled to 16 kHz mono, silence-trimmed by voice activity detection, and padded or truncated to a fixed four-second window. MFCCs are 13 coefficients plus delta and delta-delta (39-dim per frame), matching the base paper's baseline features. A single dataset class returns the waveform, the MFCC tensor, both labels, and the speaker ID, so the two pathways always see identical audio and identical splits.

## Repository layout

All logic lives in `src/`; only functions and architecture belong there. Notebooks are the sole front end — they call `src/`, run training, and store the resulting models/metrics, so behaviour never drifts between a script and a notebook. To change behaviour, edit the module, never a notebook.

```text
requirements.txt                Python dependencies
ROADMAP.md                      Phases 2-6: baseline reproduction, ablation, Praat
                                 analysis, error analysis, novel contribution
notebooks/
  01_data_pipeline.ipynb        Interactive data-pipeline driver; imports src/
  02_training.ipynb             Interactive training driver: pipeline sanity check,
                                 Phase 2 baseline reproduction and comparison
data/
  archives/                     Place the UA-Speech .tgz archives here
  extracted/                    Extracted .wav files land here (one folder per speaker)
  uaspeech_corpus_docs/         Corpus reference material: mlf/, doc/ (lexicon, wordlist,
                                 base-paper PDF), readme_UASpeech.txt, UASPEECH_LICENSE.txt
outputs/                        Generated figures, manifest, and training artifacts (gitignored)
  checkpoints/ logs/ predictions/ metrics/ confusion_matrix/ roc/ embeddings/
src/
  config.py                     Paths, speaker ground truth, label maps, hyperparameters
  console.py                    Aligned console output helpers
  extraction.py                 Archive extraction
  scanning.py                   Filename parsing, verification, mic filter, severity labels
  preprocessing.py              Resampling, VAD trimming, padding, MFCC extraction
  dataset.py                    UASpeechDataset
  splits.py                     LOSO folds (detection), balanced 81-fold (severity)
  visualization.py              EDA figures
  models/
    deep_pathway.py             wav2vec 2.0 (+ optional LoRA) -> 768-dim
    acoustic_pathway.py         1D-CNN over MFCC -> 128-dim
    fusion.py                   Concatenation + classification head
  training/
    models.py                   Model factory: acoustic / deep_frozen / deep_lora / fusion
    data.py                     Manifest loading, stratified train/val split, DataLoaders
    runner.py                   TrainingConfig + run_training(): the fold loop notebooks call
    baseline.py                 Phase 2: frozen wav2vec embeddings + linear SVM per fold
    engine.py                   Train/eval epoch loop: AMP, gradient clipping, optimizer
    metrics.py                  Accuracy/precision/recall/specificity/F1/AUROC
    early_stopping.py           Early stopping on validation loss
    checkpoint.py                Checkpoint save/load
    reporting.py                Predictions/metrics/confusion-matrix/ROC/embeddings I/O
```

## Usage

```bash
pip install -r requirements.txt
```

Copy `UASpeech_normalized_C.tgz` and `UASpeech_normalized_FM.tgz` into `data/archives/`, then run the notebooks in order.

```bash
# 1. Data pipeline
jupyter notebook notebooks/01_data_pipeline.ipynb

# 2. Training
jupyter notebook notebooks/02_training.ipynb
```

Notebook 1 scans the extracted audio, verifies the 28-speaker ground truth, writes EDA figures to `outputs/figures/`, filters to M6, checks per-speaker word counts, attaches severity labels, summarizes both split protocols, builds the dataset, and writes `outputs/m6_manifest.csv`.

Notebook 2 loads that manifest (regenerating it if missing) and drives `src/training/runner.py`'s `run_training(df, TrainingConfig(...))`:

```python
from src.training.runner import TrainingConfig, run_training

cfg = TrainingConfig(task="detection", model="fusion")   # LOSO, 28 folds
# cfg = TrainingConfig(task="severity", model="acoustic")  # balanced 81-fold
summary, pooled = run_training(df_m6, cfg)
```

`model` is one of `acoustic`, `deep_frozen`, `deep_lora`, `fusion` (see [ROADMAP.md](ROADMAP.md) for what each maps to in the ablation study). Every fold writes a checkpoint, TensorBoard log (`tensorboard --logdir outputs/logs`), predictions CSV, metrics JSON, confusion matrix, ROC curve, and fused embedding under `outputs/`; run-level metrics are reported both as a per-fold mean ± std and pooled across all folds. `TrainingConfig(..., max_folds=1, epochs=1, limit_samples=24)` gives a fast pipeline sanity check before a real run — notebook 2's Stage 1 does exactly this. Phase 2's baseline reproduction (frozen wav2vec 2.0 + linear SVM, via `src/training/baseline.py`) lives in the same notebook.

## Status

| Component | Owner role | Status |
|---|---|---|
| Data pipeline and preprocessing | — | Complete and verified |
| Deep Pathway (wav2vec 2.0, optional LoRA) | Deep Pathway Lead | Trainable via `run_training()`, not yet trained to convergence |
| Acoustic Pathway (1D-CNN over MFCC) | Acoustic Pathway Lead | Trainable via `run_training()`, not yet trained to convergence |
| Fusion: training loop, checkpointing, metrics, logging (Phase 1) | Fusion Architect | Complete — see `src/training/runner.py` + `notebooks/02_training.ipynb` |
| Baseline reproduction (Phase 2) | — | Frozen wav2vec + SVM implemented (`src/training/baseline.py`); full-scale comparison run pending |
| Ablation, Praat analysis, error analysis, novel fusion (Phases 3-6) | — | Planned — see [ROADMAP.md](ROADMAP.md) |

## License

MIT — see [LICENSE](LICENSE). The licence covers this source code only; the UA-Speech database carries its own separate licence and data use agreement (see `data/uaspeech_corpus_docs/UASPEECH_LICENSE.txt` once the corpus is downloaded — academic/government research use only, no redistribution).

## Acknowledgments

With thanks and regards to the creators and maintainers of the UA-Speech corpus — H. Kim, M. Hasegawa-Johnson, A. Perlman, J. Gunderson, T. Huang, K. Watkin, and S. Frame at the University of Illinois at Urbana-Champaign — for building and sharing this dataset for dysarthric speech research:

> H. Kim, M. Hasegawa-Johnson, A. Perlman, J. Gunderson, T. Huang, K. Watkin, and S. Frame, "Dysarthric Speech Database for Universal Access Research," *Interspeech*, 2008.

This project would not be possible without their work, or without the participants who contributed their speech recordings to advance assistive-technology research.
