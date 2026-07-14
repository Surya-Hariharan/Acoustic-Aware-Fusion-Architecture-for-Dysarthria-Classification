# Roadmap

Phase 1 (the training pipeline) is implemented and Phase 2 (baseline
reproduction) is underway — both are notebook-driven: `src/` holds only
functions and architecture, `notebooks/02_training.ipynb` is what actually
trains and stores models, matching the convention `notebooks/01_data_pipeline.ipynb`
already established for the data pipeline. Phases 4, 5 and 6 are now **built**;
what remains for them is GPU time, not code.

## Status at a glance

| Phase | Code | Run |
|---|---|---|
| 1 — Training pipeline | done | done |
| 2 — ICASSP baseline | done | detection done; severity + full-scale pending |
| 3 — Ablation study | done (six variants) | pending — needs the full 28-fold GPU run |
| 4 — Praat analysis | done (30 features + significance test) | **re-run needed** — see below |
| 5 — Error analysis | done | pending — needs a re-trained run (see below) |
| 6 — Attention fusion | steps 1–2 done (Models E, F) | pending — needs the full run |

**Two ordering dependencies to know about before running anything:**

1. `outputs/praat_features.csv` was generated when `src/praat.py` extracted 18
   features; it now extracts 30. The cache is detected as stale and
   re-extracted automatically, but Stage 1 of `notebooks/03_praat_analysis.ipynb`
   must be re-run once (~25–30 min) — and **Model F cannot train until it is**.
2. Predictions now carry a `filename` column, without which Phase 5 cannot join
   an error back to its audio or to the Praat features. Prediction CSVs written
   before this change (the smoke test, `baseline_svm_detection`) lack it, so
   **Phase 5 needs at least one model re-run** before `notebooks/04` will work.
   `load_run_predictions` fails with that exact message rather than silently
   half-working.

## Phase 1 — Training pipeline (done)

`src/training/runner.py` cross-validates any of six model variants on
either task; `notebooks/02_training.ipynb` is the front end that calls it:

```python
from src.training.runner import TrainingConfig, run_training

cfg = TrainingConfig(task="detection", model="attention_fusion")
summary, pooled = run_training(df_m6, cfg)
```

| `model` | Architecture | Ablation role (Phase 3) |
|---|---|---|
| `acoustic` | MFCC 1D-CNN only | Model A |
| `deep_frozen` | Frozen wav2vec 2.0 + MLP head | Model B |
| `deep_lora` | wav2vec 2.0 + LoRA + MLP head | Model C |
| `fusion` | LoRA wav2vec 2.0 + MFCC 1D-CNN, concatenated | Model D |
| `attention_fusion` | LoRA wav2vec 2.0 + MFCC 1D-CNN, cross-attended | Model E (Phase 6) |
| `attention_fusion_praat` | Model E + 30 Praat features as a third pathway | Model F (Phase 6) |

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

Mechanically now just six `run_training()` calls (`acoustic`, `deep_frozen`,
`deep_lora`, `fusion`, `attention_fusion`, `attention_fusion_praat`) on the full
28-fold LOSO detection protocol (and optionally the 81-fold severity protocol),
diffed on the metric set Phase 1 already computes: accuracy, precision, recall
(sensitivity), specificity, F1, AUROC.

- [ ] Run all six variants to completion (full folds, no `limit_samples`).
      `ABLATION_MODELS` in notebook 02's Stage 4 already lists them.
- [ ] `outputs/metrics/*.summary.csv` + pooled JSONs → one results table
      (Stage 4/5 of notebook 02 already produces the shape of this table
      at demo scale — rerun at full scale).
- Acceptance: a single table with all six models × all six metrics,
  pooled LOSO numbers plus per-fold mean ± std.

## Phase 4 — Praat acoustic analysis (code done)

`src/praat.py` extracts **30** features per utterance from the ORIGINAL audio
(not the VAD-trimmed, zero-padded 4-second window the pathways train on —
jitter, shimmer, HNR and formants are only meaningful on natural speech):

| Group | Features |
|---|---|
| Pitch | `f0_mean/max/min/std/range` — std and range capture monopitch |
| Perturbation | jitter `local/rap/ppq5/ddp`, shimmer `local/apq3/apq11/dda` |
| Noise | `hnr_mean/std/min` |
| Articulation | `f1/f2/f3` mean+std, `f2_f1_ratio` (vowel-space centralization) |
| Loudness | `intensity_mean/max/min/std` |
| Rhythm | `speech_rate`, `pause_duration`, `voice_breaks` |

- [x] `src/praat.py`: one function per feature group, taking a filepath and
      returning a flat dict — mirrors `src/preprocessing.py`'s MFCC extraction.
      Never raises: anything Praat can't compute on a clip comes back NaN.
- [x] `notebooks/03_praat_analysis.ipynb` Stage 2: box-plot grid + group-means
      table across Healthy / Very Low / Low / Mid / High.
- [x] Stage 3: `praat_group_significance()` — Kruskal–Wallis H per feature
      (non-parametric; these distributions are bounded and skewed, so ANOVA's
      normality assumption doesn't hold), Bonferroni-corrected across the 30
      features. This is what makes "these measures separate the severity
      groups" a claim rather than an eyeball of the box plots.
- [ ] **Re-run Stage 1** — the cached CSV holds the older 18-feature schema.
      It is now detected as stale and re-extracted automatically (~25–30 min).
- Acceptance: `outputs/praat_features.csv` keyed by `Filename`/`Speaker_ID`
  (joinable against `m6_manifest.csv`), plus the comparison figure,
  `praat_severity_group_summary.csv`, and `praat_significance.csv`.

## Phase 5 — Error analysis (code done)

`src/error_analysis.py` + `notebooks/04_error_analysis.ipynb`. Consumes
`outputs/predictions/*.csv`, `outputs/embeddings/*.npz`, and Phase 4's features.

This phase was **blocked by a data bug**, now fixed: predictions were keyed only
by `speaker_id`, so a misclassified row could not be traced to its audio file or
to the Praat features. `save_predictions`/`save_embeddings` now carry a
`filename` column throughout (dataset → engine → writers).

- [x] Per misclassified utterance: waveform, spectrogram, MFCC heatmap, and
      Praat F0 contour, in one 4-panel diagnostic. The gallery targets the most
      *confident* errors — a wrong call at p=0.51 is a coin flip and says
      nothing; one at p=0.99 means the model learned something wrong.
- [x] `compare_error_vs_correct()` correlates errors against the Phase 4
      features — Mann–Whitney U with **Cliff's delta** as the effect size,
      because on ~21k utterances a p-value is significant for effects far too
      small to matter. Directly answers "do errors cluster at low HNR / high
      jitter / short duration?".
- [x] Error rate broken down by severity group, speaker, class, and word; plus a
      t-SNE embedding map marking the misclassifications — scattered errors mean
      boundary ambiguity, clustered errors mean a mislabelled region of the space.
- [ ] Needs one model re-run first (see the ordering note at the top).
- Acceptance: concrete failure patterns, not just an aggregate accuracy number.

## Phase 6 — Novel contribution (steps 1–2 done)

**Attention-based fusion**, in `src/models/attention_fusion.py`. Both models
honour the existing `forward_features` / `.classifier` contract, so they drop
into `run_training()` as ordinary `model=` choices.

A design note that matters: the original sketch said "cross-attention between the
768-dim deep embedding and the 128-dim acoustic embedding". Taken literally that
is a **no-op** — both pathways mean-pool to a single vector, and attention over a
single key is the identity (softmax over one element is always 1.0). Real
cross-attention needs the frames the pooling discards, so both pathways grew a
`forward_sequence()` (deep: ~199×768, acoustic: ~100×128) and each is projected
into a shared 256-dim space before attending.

1. [x] **Attention-based fusion** — `model="attention_fusion"` (Model E).
   *Bidirectional* cross-attention: the deep stream attends to the acoustic
   frames and vice versa. One-directional would privilege wav2vec as the "real"
   representation and demote MFCC to a lookup table; running it both ways is the
   claim the architecture actually makes. `attention_weights()` exposes both maps
   (199×100 and 100×199), which is the groundwork for step 4.
   **`src/training/runner.py` was not touched by this step**, as required.
2. [x] **Praat features joining the fusion** — `model="attention_fusion_praat"`
   (Model F). The 30 features become one 256-dim token appended to the context of
   *both* attention blocks, and are also concatenated into the final embedding so
   their signal survives even if attention learns to ignore them. Standardization
   is computed **from each fold's train split only** — global statistics would
   leak the held-out speaker's acoustic distribution into the very fold measuring
   generalization to that speaker. This step necessarily widened the shared
   `forward(waveform, mfcc, praat)` signature across all models.
3. [ ] **Multi-task heads** (detection + severity, shared encoder) — deferred;
   changes the training loop's label handling.
4. [ ] **Explainability** (SHAP / integrated gradients) — deferred, once there's
   a trained model worth explaining. `attention_weights()` is the start.

- Acceptance for step 1: `model="attention_fusion"` trains and evaluates through
  the existing `run_training()` unmodified, and its pooled LOSO metrics sit in
  the ablation table as Model E. **The comparison the contribution rests on is
  `fusion` → `attention_fusion`**: same two pathways, same data, the only
  difference being concatenation vs. learned cross-attention. `attention_fusion`
  → `attention_fusion_praat` then isolates what the handcrafted features add.
