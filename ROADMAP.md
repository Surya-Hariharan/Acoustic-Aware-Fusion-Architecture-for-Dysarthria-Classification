# Roadmap

Phase 1 (the training pipeline) is implemented and Phase 2 (baseline
reproduction) is underway — both are notebook-driven: `src/` holds only
functions and architecture, `notebooks/02_training.ipynb` is what actually
trains and stores models, matching the convention `notebooks/01_data_pipeline.ipynb`
already established for the data pipeline. Phases 3–6 are the plan for the
remaining work, sequenced so each phase's output feeds the next. Direction
locked in for Phase 6: **attention-based fusion** first, Praat features
second.

## Phase 1 — Training pipeline (done)

`src/training/runner.py` cross-validates any of four model variants on
either task; `notebooks/02_training.ipynb` is the front end that calls it:

```python
from src.training.runner import TrainingConfig, run_training

cfg = TrainingConfig(task="detection", model="fusion")
summary, pooled = run_training(df_m6, cfg)
```

| `model` | Architecture | Ablation role (Phase 3) |
|---|---|---|
| `acoustic` | MFCC 1D-CNN only | Model A |
| `deep_frozen` | Frozen wav2vec 2.0 + MLP head | Model B |
| `deep_lora` | wav2vec 2.0 + LoRA + MLP head | Model C |
| `fusion` | LoRA wav2vec 2.0 + MFCC 1D-CNN, concatenated | Model D |

For each fold: stratified train/val split, AdamW (separate LR for the
wav2vec backbone vs. the rest), `ReduceLROnPlateau`, class-weighted
`CrossEntropyLoss`, AMP, gradient clipping, early stopping on val loss,
best-checkpoint selection, then a held-out test pass that writes
predictions/metrics/confusion-matrix/ROC/embeddings. Metrics are reported
two ways: per-fold mean ± std, and pooled across all folds — pooling
matters because in LOSO detection every fold's test speaker is entirely
one class, so per-fold precision/recall/specificity/AUROC are individually
degenerate (only accuracy is meaningful per fold; `run_training` prints a
note to this effect). Severity folds don't have this problem since each
fold holds out one speaker per class.

Useful `TrainingConfig` fields for iterating cheaply before a real run:
`max_folds`, `folds` (a list of specific fold IDs), `limit_samples` (caps
rows per split), `epochs=1` — notebook 02's Stage 1 combines all of these
into a pipeline sanity check. TensorBoard: `tensorboard --logdir outputs/logs`.

A full 28-fold LOSO run of `deep_lora`/`fusion` fine-tunes wav2vec 2.0 on
the complete training split per fold — budget real GPU time (hours, not
minutes) before kicking that off, and prefer `max_folds`/`limit_samples`
for iterating on hyperparameters first.

## Phase 2 — Reproduce the ICASSP baseline first (in progress)

Before claiming the fusion model is better than the base paper, reproduce
the base paper's own number: frozen wav2vec 2.0 → 768-dim embedding →
linear SVM. `DeepPathway(use_lora=False)` gives the frozen extractor;
`src/training/baseline.py` does the rest:

```python
from src.training.baseline import extract_frozen_embeddings, run_svm_baseline

embeddings = extract_frozen_embeddings(df_m6)          # cached, extracted once
summary, pooled = run_svm_baseline(df_m6, task="detection", embeddings=embeddings)
```

- [x] `src/training/baseline.py`: extract frozen wav2vec embeddings per
      utterance (batched through `DeepPathway(use_lora=False)`, no
      gradient, cached to `outputs/embeddings/frozen_wav2vec_base.npz`),
      fit a Platt-calibrated `LinearSVC` per LOSO fold, report pooled
      accuracy/F1/recall/precision/specificity/AUROC exactly like Phase 1's
      pooling.
- [x] `notebooks/02_training.ipynb` Stage 4 compares the baseline against
      `deep_frozen`, `deep_lora`, `acoustic`, and `fusion` in one table
      (`outputs/metrics/phase2_comparison.csv`) — at demonstration scale by
      default (`max_folds=3`, `limit_samples=300`, `epochs=5`); rerun with
      those caps removed for the real comparison once GPU time is budgeted.
- [ ] Full-scale run of Stage 4 (all 28 folds, full training splits,
      epochs=20+) to replace the demonstration-scale numbers.
- [ ] Same comparison for the severity task (`task="severity"` — both
      `baseline.py` and `runner.py` already support it unchanged).
- Acceptance: a baseline accuracy/F1/recall number sitting next to the
  paper's reported numbers, plus the three trained variants for comparison
  — this is what makes the fusion model's improvement scientifically
  defensible rather than assumed. Detection-task baseline: see
  `outputs/metrics/baseline_svm_detection.summary.csv` and
  `ALL_FOLDS_pooled.json` for the pooled numbers.

## Phase 3 — Ablation study

Mechanically now just four `run_training()` calls (`acoustic`,
`deep_frozen`, `deep_lora`, `fusion`) on the full 28-fold LOSO detection
protocol (and optionally the 81-fold severity protocol), diffed on the
metric set Phase 1 already computes: accuracy, precision, recall
(sensitivity), specificity, F1, AUROC.

- [ ] Run all four variants to completion (full folds, no `limit_samples`).
- [ ] `outputs/metrics/*.summary.csv` + pooled JSONs → one results table
      (Stage 4/5 of notebook 02 already produces the shape of this table
      at demo scale — rerun at full scale).
- Acceptance: a single table with all four models × all six metrics,
  pooled LOSO numbers plus per-fold mean ± std.

## Phase 4 — Praat acoustic analysis

New dependency: `praat-parselmouth`. Extract, per utterance: F0
(mean/max/min), jitter (local, RAP, PPQ5), shimmer (local, APQ3, APQ11),
HNR, formants (F1–F3), intensity (mean/max), speech rate, pause duration,
voice breaks.

- [ ] `src/praat.py`: one function per feature group, taking a filepath
      (reuse `config.MANIFEST_PATH` rows) and returning a flat dict —
      mirrors the shape of `src/preprocessing.py`'s MFCC extraction.
- [ ] A notebook stage that extracts all ~30–50 features per utterance and
      compares Healthy / Very Low / Low / Mid / High as group distributions
      (box plots, group means table).
- Acceptance: a features CSV keyed by `Filename`/`Speaker_ID` (joinable
  against `m6_manifest.csv`) and a severity-group comparison figure/table
  for the discussion section.

## Phase 5 — Error analysis

Consumes Phase 1's `outputs/predictions/*.csv` (which rows were
misclassified) and `outputs/embeddings/*.npz`.

- [ ] For each misclassified utterance: plot spectrogram, MFCC heatmap,
      waveform, and (once Phase 4 exists) Praat pitch contour.
- [ ] Correlate errors against Phase 4 features (e.g. do misclassified
      utterances cluster at low HNR / high jitter / short duration?).
- Acceptance: a short error-analysis section with concrete failure
  patterns, not just an aggregate accuracy number.

## Phase 6 — Novel contribution

Direction chosen: **attention-based fusion**, built on top of Phase 1's
existing `forward_features(waveform, mfcc)` contract so it drops in as a
fifth `model` choice without touching `src/training/runner.py`.

1. **Attention-based fusion** — `src/models/attention_fusion.py`:
   cross-attention between the 768-dim deep embedding and the 128-dim
   acoustic embedding (in place of `FusionModel`'s concatenation),
   producing a learned per-utterance weighting of deep vs. acoustic
   evidence.
2. **Praat features joining the fusion** — once Phase 4 exists, add the
   ~30–50 handcrafted features as a third pathway into the attention
   fusion.
3. **Multi-task heads** (detection + severity, shared encoder) — later;
   changes the training loop's label handling, so it's the one item here
   that does touch `src/training/runner.py`.
4. **Explainability** (SHAP / integrated gradients) — last, once there's a
   trained model worth explaining.

- Acceptance for step 1: `model="attention_fusion"` trains and evaluates
  through the existing `run_training()` unmodified, and its pooled LOSO
  metrics sit in the Phase 3 ablation table as Model E.
