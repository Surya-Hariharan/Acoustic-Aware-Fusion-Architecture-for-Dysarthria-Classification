# Acoustic-Aware Fusion Architecture for Dysarthria Classification

> A three-branch neural architecture that fuses segmental, suprasegmental, and self-supervised speech representations to classify dysarthria severity from the UA-Speech corpus under speaker-disjoint evaluation.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.49.0-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT%20(LoRA)-0.19.1-6F42C1)](https://huggingface.co/docs/peft)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Research overview

Dysarthria is a motor speech disorder whose acoustic signature spans several levels of the speech signal at once: *articulatory precision* (consonant blurring, vowel-space centralization), *phonatory and prosodic control* (monopitch, reduced loudness variation, irregular voicing), and *higher-level contextual structure* that a handcrafted descriptor set does not capture.

This repository builds on the premise that these levels carry *complementary* information, and tests it in two forms under the same 15-fold leave-one-speaker-out protocol:

* **AAF-Lite (final model)** — four frozen branches (a *recogniser* branch added to the learned, segmental and suprasegmental ones), each expressed relative to healthy speakers saying the same word. Severity is a property of the **speaker**, so the headline decision is made per speaker (*session level*) from a few per-utterance scalars averaged over all of its recordings; a stricter utterance-level view with nested model selection is reported next to it. Deterministic; the whole evaluation takes about half an hour on a CPU. See [Final model](#final-model-aaf-lite) and [Why the first attempts plateaued](#why-the-first-attempts-plateaued).
* **The end-to-end gated-fusion network** — wav2vec 2.0 + LoRA, segmental and suprasegmental CNNs, softmax gate, CORAL head and adversarial speaker head, trained per fold (3.5–6 h per run). Kept as the comparison.

The claim under test is deliberately narrow: **whether combining the three representations outperforms any subset of them**, for a speaker the model has never heard.

## Key idea (end-to-end network)

A self-supervised encoder (wav2vec 2.0) is strong but opaque; classical descriptors (MFCC, formants, F0, intensity, HNR) are clinically interpretable but individually weaker. Rather than choosing between them or concatenating them flatly, this architecture:

1. **Bottlenecks each branch before fusion**, so every branch keeps only decision-relevant information;
2. **Fuses with a learned softmax gate** whose weights are saved per utterance, so branch reliance is *inspectable* rather than assumed;
3. **Penalizes cross-branch redundancy** with a Barlow-Twins-style cross-correlation term;
4. **Suppresses speaker identity** in the fused representation with a gradient-reversal adversarial head;
5. **Treats severity as ordinal** with a CORAL head, since Very Low < Low < Mid < High is a real ordering.

## End-to-end architecture

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1f2937", "primaryTextColor": "#f5f5f5", "primaryBorderColor": "#9aa0a6", "lineColor": "#9aa0a6", "fontSize": "14px"}}}%%
flowchart TD
    UA["UA-Speech, audio/original<br/>microphone M6 · 15 dysarthric speakers · 765 words each"]
    VAD["16 kHz mono → Silero VAD trim (internal pauses kept)"]
    UA --> VAD
    P1["Speech-focused profile<br/>30 ms margin → 4 s window"]
    P2["Temporal-preserving profile<br/>150 ms margin → 4 s window"]
    VAD --> P1
    VAD --> P2

    LRN["LEARNED<br/>raw waveform, 64,000 samples"]
    SEG["SEGMENTAL<br/>43 × 401: MFCC+Δ+ΔΔ, F1–F3, HNR"]
    SUP["SUPRASEGMENTAL<br/>3 × 401: F0 (st), voicing, intensity"]
    P1 --> LRN
    P1 --> SEG
    P2 --> SUP

    ELRN["wav2vec2-base-960h + LoRA (q/k/v, r=8)<br/>masked mean-pool · 768 → 128"]
    ESEG["1D-CNN ×3 · masked pool · → 64"]
    ESUP["1D-CNN ×2 · masked pool · → 64"]
    LRN --> ELRN
    SEG --> ESEG
    SUP --> ESUP

    GATE["GATED FUSION · softmax gate · Z = 256"]
    ELRN --> GATE
    ESEG --> GATE
    ESUP --> GATE
    HEAD["CORAL ordinal head · 4 classes<br/>median decode"]
    SPK["Speaker head via gradient reversal<br/>training only"]
    RED["Cross-branch redundancy penalty<br/>λ = 0.05"]
    GATE --> HEAD
    GATE --> SPK
    ELRN -.-> RED
    ESEG -.-> RED
    ESUP -.-> RED

    classDef branch fill:#1a3a6b,stroke:#5b9bff,stroke-width:2px,color:#eaf1ff
    classDef fusion fill:#6b1a1a,stroke:#ff6b5b,stroke-width:2px,color:#ffe8e5
    classDef aux fill:#3a3a3a,stroke:#b0b6bb,stroke-dasharray:4 3,color:#f0f0f0
    class SEG,LRN,SUP,ESEG,ELRN,ESUP branch
    class GATE,HEAD fusion
    class SPK,RED aux
```

| Branch | Input | Profile | Encoder | Output |
|---|---|---|---|---|
| **Learned** | Raw 16 kHz waveform, 64,000 samples, per-utterance normalized | Speech-focused (30 ms) | `facebook/wav2vec2-base-960h` + LoRA, masked mean-pool, projection | **128** |
| **Segmental** | 43 × 401: 13 MFCC + Δ + ΔΔ, framewise F1–F3, HNR | Speech-focused (30 ms) | 3-layer 1D-CNN, masked mean-pool, bottleneck | **64** |
| **Suprasegmental** | 3 × 401: F0 (semitones, 0 when unvoiced), voicing mask, intensity (dB) | Temporal-preserving (150 ms) | 2-layer 1D-CNN, masked mean-pool, bottleneck | **64** |
| **Fusion → head** | The three embeddings | — | Softmax gate, weighted concatenation, CORAL | **256 → 4 classes** |

Dimensions are read from `src/config.py` and pinned by `tests/test_gated_fusion_shapes.py`.

## Why the first attempts plateaued

Two complete end-to-end runs (wav2vec 2.0 + LoRA, ~3.4 h and ~6.0 h each) reached 0.453 and 0.370 accuracy (8 and 6 of 15 speakers correct), and a first frozen-feature version of AAF-Lite reached 0.369 with the same 6/15. Diagnostics on the cached features found four causes:

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| 1 | **The label belongs to the speaker, but the models classified utterances.** | Severity explains only 30–47 % of the *between-speaker* variance, which is itself only 20–31 % of the total; a 15-way speaker-ID probe on the same features scores 0.59–0.83 (chance 0.07). | Models fitted on 14 training speakers learn voices, not severity: near-perfect on a few speakers, near zero on the Low/Mid ones. |
| 2 | **A bug degraded the learned branch.** | `wav2vec2-base` was trained on zero-padded batches *without* an attention mask; the code passed one. With the mask, its own CTC head decodes isolated words as noise (speaker-level correlation of word accuracy with severity: **+0.16**); without it, **+0.94**. | Both the end-to-end network and the first AAF-Lite used a broken representation. |
| 3 | **The model never measured what the label measures.** | UA-Speech severity is *listener intelligibility* of these exact words, and the checkpoint already contains a CTC recogniser. | The recogniser's accuracy on the prompted word is a one-number automatic analogue; it was not used. |
| 4 | **Capacity far beyond the data.** | The inner loop always selected the smallest PCA on offer (32) — the edge of its grid — for 1,500-dimensional inputs and 14 training speakers. | Fewer dimensions were needed, and a scalar per speaker is enough. |

## Final model (AAF-Lite)

`src/aaflite/` — four frozen branches, nothing fitted by gradient descent:

| Branch | Representation |
|---|---|
| **Recogniser** (`asr.py`) | A frozen CTC speech recogniser scores each utterance against its prompted word (taken from the corpus' own label files): exact match, character error rate, blank ratio, CTC likelihood per character and per frame, mean confidence. Two checkpoints: `wav2vec2-base-960h` (no attention mask) and `wav2vec2-large-960h-lv60-self` (layer norm, with mask). |
| **Learned** (`embeddings.py`) | Every hidden state of frozen `wav2vec2-base-960h`, *without* an attention mask, pooled over real frames into mean + std and averaged within a layer group (early 1–4, middle 5–8, late 9–12, all 1–12). |
| **Segmental** (`functionals.py`) | Mean, std, p10, p50, p90 of the 43 stored channels (MFCC+Δ+ΔΔ, F1–F3, HNR) over valid frames. |
| **Suprasegmental** (`functionals.py`) | The same functionals of F0 (voiced frames), voicing and intensity, plus true speech duration, voiced ratio and F0 range. |

**Control-referenced normalization** (`reference.py`). Every UA-Speech speaker reads the same 765 prompts. Each feature is expressed as a z-score against the 13 healthy controls' mean for the *same prompt* (block × word), scaled by their pooled within-prompt spread, plus one scalar per branch: the RMS distance from healthy speech. It uses no labels, and control speakers are never test speakers.

**Session level — the headline** (`speaker_level.py`). Each speaker is summarized by the mean of a few per-utterance scalars over all of its recordings and classified by a Gaussian rule (class means, one pooled within-class variance per feature, equal priors; in one dimension, nearest class mean) fitted on the 14 training speakers.

*The model itself is chosen without the held-out speaker.* In `session_nested` — the headline — an inner leave-one-speaker-out over the 14 training speakers of each outer fold picks the recogniser checkpoint (base / large) and the scalar set (word-recognition rate, character error rate, CTC likelihood per character, or a pair) among eight pre-declared candidates (`run.NESTED_CANDIDATES`; ties go to the earlier one). That candidate is refit on the 14 and applied once to the held-out speaker, so the reported number answers *"if the procedure is decided without this speaker, how well does it classify a speaker it has never seen?"* The notebook shows which candidate each fold chose.

*The fixed-feature rows are exploratory.* The `session_fixed_*` models (other scalars, the base recogniser, fusion with each acoustic branch's distance from healthy speech and with speech duration, acoustic branches alone) use feature sets fixed by hand after exploring all 15 speakers, so they are optimistic by construction; they are sensitivity analyses, not the headline. Every utterance of a held-out speaker carries that speaker's decision, so pooled accuracy equals the share of speakers classified correctly.

**Utterance level — the strict single-word view** (`pipeline.py`). Each branch is `StandardScaler → (whitened PCA of 8 / 16 / 32 dimensions) → multinomial logistic regression`; branch probabilities are fused with weights on a 0.1 simplex grid and decoded by argmax or by the ordinal expected rank. For each of the 15 outer folds, an inner leave-one-speaker-out over the 14 training speakers chooses every configuration, the fusion weights and the decoder; the held-out speaker never influences a choice. Every branch subset is reported, with and without the control reference.

**Reported alongside:** a majority-class baseline (the corpus' most frequent class; refitting it per fold would exclude the held-out speaker's own class and score exactly 0), a **permuted-label sanity check** for both views (severity shuffled across speakers — must fall to chance), 95 % bootstrap intervals over speakers, within-one-level accuracy and ordinal MAE.

## Dataset

**UA-Speech** — dysarthric and control speech, 765 isolated words per speaker in three blocks (B1–B3). The corpus is **not redistributed here**; obtain it from its authors and place the archives in `data/raw/`:

```text
data/raw/UASpeech_original_C.tgz     # healthy controls
data/raw/UASpeech_original_FM.tgz    # dysarthric speakers
```

Only the `audio/original` release is used (no loudness normalization). Microphone channel **M6**, all blocks, all word categories. The severity task uses the 15 dysarthric speakers (11,475 utterances):

| Severity | n | Speakers |
|---|---|---|
| Very Low | 4 | M01, M04, F03, M12 |
| Low | 3 | M07, F02, M16 |
| Mid | 3 | M05, M11, F04 |
| High | 5 | M09, M14, M10, M08, F05 |

Filename parsing takes the microphone channel positionally from the final token (searching for a token starting with `M` would match speaker IDs like `M01`), skips macOS `._` resource forks, and drops the corpus's zero-filled files by checking their RIFF/WAVE header.

## Preprocessing

1. Load, mix to mono, resample to **16 kHz**.
2. **Silero VAD** trims leading and trailing non-speech; the kept region is the contiguous span from the first to the last speech segment, so **internal pauses are preserved**. Any failure falls back to the untrimmed waveform.
3. Pad or truncate to a fixed **4.0 s / 64,000-sample** window (401 frames at a 10 ms hop), returning `valid_length` so padding is **masked** in every pooling step, never treated as silence.

Two VAD profiles differ only in the margin kept around speech: 30 ms for the learned and segmental branches, 150 ms for the suprasegmental branch, which needs onset/offset dynamics. **F0 contract:** voiced frames carry a semitone value; unvoiced frames are exactly 0 and flagged by the voicing channel — no contour is interpolated through them, and normalization keeps them at 0.

**Fold-scoped standardization.** Segmental and suprasegmental channels are z-scored with statistics computed from each fold's *training* speakers only (over valid frames; F0 over voiced frames), so the held-out speaker never influences its own normalization.

**Feature store.** Silero and ~2,000 Praat calls per utterance are far too slow to run per epoch. `src/feature_store.py` computes each utterance's VAD spans and both feature tensors once, in parallel, into one compressed chunk per (speaker, block) under `outputs/feature_cache/store/`. Chunks are written atomically (an interrupted build keeps every finished chunk), memory-mapped at read time (one copy shared by all DataLoader workers), and keyed by a configuration signature (a chunk built under different settings is ignored). The notebook re-runs live Silero on 300 utterances and recomputes 24 feature tensors through the original live path, requiring **bit-exact** equality.

## End-to-end training

| Aspect | Setting | Source |
|---|---|---|
| Backbone | `facebook/wav2vec2-base-960h`, frozen; SpecAugment off (the CTC checkpoint has no pretrained `masked_spec_embed`) | `config.WAV2VEC_*` |
| Adaptation | LoRA on `q_proj`, `k_proj`, `v_proj` of all 12 layers — r = 8, α = 16, dropout 0.1 | `config.LORA_*` |
| Loss | Class-weighted CORAL + 0.05 × redundancy + 0.1 × adversarial speaker CE (GRL strength 1.0) | `config.LAMBDA_*` |
| Decoding | Median of the CORAL distribution — argmax starves Low/Mid when thresholds are close | `losses.coral_rank_from_class_probs` |
| Optimizer | AdamW, weight decay 1e-2 — **LoRA adapters 1e-4**, branches / projections / gate / heads 1e-3 | `config.DEFAULT_LR_*`, `engine.build_optimizer` |
| Schedule | ReduceLROnPlateau (×0.5); early stopping on validation ordinal MAE averaged over the last 2 evaluations, patience 5, no checkpoint before epoch 3 | `config.DEFAULT_PATIENCE`, `DEFAULT_MIN_EPOCHS`, `DEFAULT_MONITOR_SMOOTHING` |
| Batch / epochs | 32, fp16 AMP, gradient clipping 1.0 / at most 12 | `config.DEFAULT_*` |
| Validation | Speaker-disjoint: up to two speakers per class while every class keeps ≥ 2 training speakers (5–6 per fold), seeded per fold | `config.DEFAULT_VAL_SPEAKERS_PER_CLASS` |
| Seed | 42, re-seeded per fold so a fold's result does not depend on which folds ran before it | `runner.run_fold` |

Hyperparameters are fixed in `src/config.py` before the run. `write_frozen_config` records the configuration, git commit and software versions under `outputs/results/<run>/frozen_config.json`; `check_frozen_config_guard` refuses to re-run the same run name under a different configuration (resuming the same one is fine).

## Evaluation protocol

**Leave-one-speaker-out over all 15 dysarthric speakers** — no speaker is dropped to balance classes; the 4/3/3/5 imbalance is handled by the class-weighted loss and macro metrics. Severity labels are speaker-level, so the effective sample size for generalization is **15 speakers**, not 11,475 utterances.

A held-out speaker has a single true class, so per-fold macro-F1, balanced accuracy and AUROC are undefined (reported as N/A, never 0). The reported numbers are **pooled** over every held-out utterance: accuracy, macro-F1, balanced accuracy, ordinal MAE, macro AUROC, per-class precision/recall, the confusion matrix, and a **speaker-level** decision (median of each speaker's utterance predictions).

## Running it on a laptop

Everything runs from `notebooks/training.ipynb` (**Run All**), measured on an RTX 4060 Laptop (8 GB, 88 W) with 16 GB RAM under Windows 11. The first run builds the feature store for all 28 speakers (30–50 min), the frozen wav2vec2 statistics (~9 min) and the recogniser scores of both checkpoints (~10 min); afterwards all of it loads from disk, and the full evaluation (session-level models, every utterance-level ablation, baselines and sanity checks) takes about 30 min on the CPU. The end-to-end network (notebook section 6) is off by default (`RUN_END_TO_END = False`); the notes below apply to it.

- **Measured, not assumed.** Section 6.3 times real training steps on this machine (~1 min) and projects the run time before the run starts.
- **No thermal pauses.** Training runs straight through; the GPU's own firmware throttles clocks if it runs hot. An optional pause-and-resume guard exists behind `config.THERMAL_GUARD_ENABLED` (off).
- **Power and sleep.** Training waits for the charger if the laptop is on battery, and keeps Windows awake while it runs. Lid closing still follows its own Windows setting — keep the lid open or set *When I close the lid → Do nothing* while plugged in.
- **Memory.** On Windows every DataLoader worker is a spawned process costing ~2 GB of commit, so training uses one persistent worker and evaluates in-process; worker counts are re-sized to the memory free before every fold. PyTorch's VRAM share is capped at 90% so an overflow raises a clean OOM instead of silently spilling into system RAM (which slows training ~10×).
- **Self-healing.** A failing fold is retried with an adapted configuration: CUDA OOM → half batch × 2 accumulation (same effective batch); host memory → no workers; divergence → restart in float32.
- **Resumable.** Finished folds load from disk; an interrupted fold resumes from its `latest.pt` (saved every epoch, ~9 MB since only trainable weights are stored). Optional `SESSION_HOURS` stops cleanly before a time cap.

## Repository structure

```text
notebooks/training.ipynb   The single notebook: environment, data, feature store, final model
                           (AAF-Lite) and its results, then the optional end-to-end network
src/
  aaflite/
    asr.py                 Frozen CTC recogniser: word accuracy / likelihood vs. the prompted word
    embeddings.py          Frozen wav2vec 2.0: mean + std of every hidden state over real frames
    functionals.py         Utterance functionals of the stored segmental / supra features
    reference.py           Control-referenced (same-prompt) normalization
    pipeline.py            Nested LOSO (utterance level), late fusion, decoders, CIs, baselines
    speaker_level.py       Session-level (one decision per speaker) Gaussian classifier
    run.py                 Feature preparation and the full experiment set
  config.py                Paths, speakers, labels, hyperparameters, hardware profile
  extraction.py            Archive extraction (audio/original)
  scanning.py              Filename parsing, verification, M6 filter, severity labels
  vad.py                   Silero VAD (trim, fallback)
  vad_cache.py             VAD spans served from the feature store + live verification
  praat.py                 Framewise F1–F3/HNR and F0/voicing/intensity (parselmouth)
  preprocessing.py         Loading, trimming, padding, MFCC, fold-scoped standardization
  feature_store.py         Chunked, resumable, memory-mapped feature store + verification
  dataset.py               UASpeechDataset
  splits.py                Severity leave-one-speaker-out folds
  losses.py                CORAL, cross-branch redundancy, gradient reversal
  console.py               Console report formatting and progress bars
  eda.py  style.py         Speech-processing EDA panels and the shared figure style
  models/
    gated_fusion.py        GatedFusionModel: gate, CORAL head, speaker head, ablation switches
    deep_pathway.py        wav2vec 2.0 + LoRA
    segmental_pathway.py   43 ch → 64
    suprasegmental_pathway.py  3 ch → 64
  training/
    runner.py              TrainingConfig, the fold loop, retries, laptop safeguards
    engine.py              One epoch: AMP, accumulation, clipping, metrics, embeddings
    data.py                Manifest, validation split, class weights, DataLoaders
    models.py              Model registry: the full model and eight ablations
    metrics.py             Severity metrics (undefined → NaN)
    reporting.py           Artifacts, results rebuilt from disk, frozen config + guard
    session.py             Throughput calibration and run-time projection
    checkpoint.py  early_stopping.py  utils.py
tests/                     Shapes, losses, masking, frame alignment, folds and validation
                           split, metrics, frozen-config guard, feature-store and VAD-span
                           equality gates, a real-data end-to-end training step, and the
                           AAF-Lite protocol (no held-out speaker in any fit or selection,
                           control-only reference, determinism, permuted labels at chance)
```

Every fold writes, under `outputs/<kind>/<run_name>/`: a checkpoint, TensorBoard logs (`tensorboard --logdir outputs/logs`), a predictions CSV, a metrics JSON, a confusion matrix and an embeddings file (fused, per-branch and gate weights per utterance).

### Ablations

`MODEL` in the notebook selects the full model or one of eight controlled ablations (`src/training/models.py`). All share data, folds, validation and optimizer; each gets its own run name.

| Name | Branches | Fusion | Redundancy | Speaker GRL |
|---|---|---|---|---|
| `ab1_wav2vec2_only` | learned | — | — | — |
| `ab2_acoustic_only` | segmental + supra | gated | — | — |
| `ab3_wav2vec2_acoustic_concat` | all three | concat | — | — |
| `ab4_wav2vec2_segmental` | learned + segmental | gated | — | — |
| `ab5_wav2vec2_suprasegmental` | learned + supra | gated | — | — |
| `ab6_full_fusion` | all three | gated | — | — |
| `ab7_full_complementarity` | all three | gated | ✓ | — |
| `ab8_full_speaker_grl` | all three | gated | — | ✓ |
| `gated_fusion_three_branch` | all three | gated | ✓ | ✓ |

## Reproducibility

```bash
conda create -n torch-gpu python=3.10 && conda activate torch-gpu
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m ipykernel install --user --name torch-gpu
pytest                       # the real-data tests skip when the corpus is absent
```

Then open `notebooks/training.ipynb` with the `torch-gpu` kernel and Run All. `AAFA_SMOKE=1` runs the final model on 4 outer folds (one per class) under separate run names — a pipeline check, not a result. An optional `HF_TOKEN` in `.env` lifts the Hugging Face rate limit; the checkpoint is public.

## Current status

| Component | Status |
|---|---|
| Data pipeline, two VAD profiles, frame-wise features, feature store (all 28 speakers) | Implemented, verified bit-exact against the live pipeline |
| Final model (AAF-Lite): recogniser + three acoustic branches, control reference, session-level and nested-LOSO utterance-level evaluation, ablations, sanity checks | Implemented and unit-tested; results are produced by notebook sections 4–5 |
| End-to-end gated-fusion network | Two complete 15-fold runs before the fixes above: accuracy 0.453 / 0.370, macro-F1 0.312 / 0.296, 8 / 6 of 15 speakers correct. Its learned branch used the attention-mask path described in [Why the first attempts plateaued](#why-the-first-attempts-plateaued); it was not re-trained. |

Run results are written to `outputs/results/aaflite_summary.json` (every model, pooled metrics, per-fold values and choices) and per run to `outputs/{predictions,metrics,confusion_matrix}/aaflite_*`.

## Limitations and open questions

| # | Question | Why it matters | Planned check |
|---|---|---|---|
| 1 | **Run-to-run variance of the end-to-end network** | Its two runs differ by 8 accuracy points, and single speakers by up to 40 (M09 0.65 → 0.26) — larger than most effects one would want to measure | Reason the final model is deterministic |
| 2 | **Researcher degrees of freedom** | The recogniser idea itself was found by exploring these 15 speakers. Nested selection removes the choice of checkpoint and scalar from the held-out speaker's reach, but not the choice of the candidate *family*; the `session_fixed_*` rows are optimistic | Validate on an independent dysarthric corpus (e.g. TORGO) |
| 3 | **Two borderline speakers decide the headline** | With 15 speakers one speaker is 6.7 points; the Low speakers F02 (29 % intelligibility) and M16 (43 %) sit next to the Very Low / Mid boundaries | Report the bootstrap interval, never the point estimate alone |
| 4 | **Speaker adversary vs. speaker-level labels** | Every utterance of a speaker has the same severity, so severity is itself partly speaker information; an adversary that removes speaker identity can also remove severity cues | Compare `ab6_full_fusion` with `ab8_full_speaker_grl` before trusting the GRL |
| 5 | **Truncation at 4 s** | ~8% of utterances exceed the window and lose their tail; long utterances are more frequent at higher severity | Stratify truncation by severity; compare a longer window |
| 6 | **Padding and boundary effects** | Masked pooling excludes padding, but convolutions and BatchNorm see it first (on average 68% of the window is padding) | Measure embedding sensitivity to padding at fixed speech content |
| 7 | **Recording level** | `audio/original` keeps absolute level, partly a session property | Decompose intensity variance into speaker vs. severity components |
| 8 | **Branch contribution** | Branch value is a claim only after the ablations run across all 15 folds | Run ab1–ab8 under the same protocol |

Structural limits no experiment removes: **15 dysarthric speakers** (3 in the smallest classes) is a small population for a 4-class task; UA-Speech contains **isolated words** only, so connected-speech prosody cannot be modelled; and all findings are corpus-specific until validated on an independent dysarthric corpus.

## License

MIT — see [LICENSE](LICENSE). The licence covers this source code only. **The UA-Speech database carries its own licence and data use agreement** (academic/government research use only, no redistribution).

## Acknowledgments

With thanks to the creators and maintainers of the UA-Speech corpus at the University of Illinois at Urbana-Champaign, and to the participants who contributed their speech recordings:

> H. Kim, M. Hasegawa-Johnson, A. Perlman, J. Gunderson, T. Huang, K. Watkin, and S. Frame,
> *Dysarthric Speech Database for Universal Access Research,* Interspeech, 2008.
