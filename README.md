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

UA-Speech: 765 isolated words per speaker across three blocks (B1–B3), captured by an eight-microphone array at 16 kHz. The audio is **not** redistributed here — obtain it from the dataset authors and place the archives in `data/archives/`.

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

All logic lives in `src/`. The notebook and the script are two front ends onto the same modules and cannot drift apart — to change behaviour, edit the module, never the notebook.

```text
run_pipeline.py                Script entry point: every stage, headless
requirements.txt               Python dependencies
notebooks/
  01_data_pipeline.ipynb       Interactive driver; imports src/, no duplicated logic
data/
  archives/                    Place the UA-Speech .tgz archives here
  extracted/                   Extracted .wav files land here
outputs/                       Generated figures and the M6 manifest (gitignored)
src/
  config.py                    Paths, speaker ground truth, label maps, hyperparameters
  console.py                   Aligned console output helpers
  extraction.py                Archive extraction
  scanning.py                  Filename parsing, verification, mic filter, severity labels
  preprocessing.py             Resampling, VAD trimming, padding, MFCC extraction
  dataset.py                   UASpeechDataset
  splits.py                    LOSO folds (detection), balanced 81-fold (severity)
  visualization.py             EDA figures
  models/
    deep_pathway.py            wav2vec 2.0 + LoRA → 768-dim
    acoustic_pathway.py        1D-CNN over MFCC → 128-dim
    fusion.py                  Concatenation + classification head
```

## Usage

```bash
pip install -r requirements.txt
```

Copy `UASpeech_normalized_C.tgz` and `UASpeech_normalized_FM.tgz` into `data/archives/`, then run either front end.

```bash
# Interactive
jupyter notebook notebooks/01_data_pipeline.ipynb

# Headless, reproducible
python -m src.extraction     # extract the archives (once)
python run_pipeline.py       # scan, verify, EDA, filter, split, build dataset
```

Both paths scan the extracted audio, verify the 28-speaker ground truth, write EDA figures to `outputs/figures/`, filter to M6, check per-speaker word counts, attach severity labels, summarize both split protocols, build the dataset, and write `outputs/m6_manifest.csv`.

## Status

| Component | Owner role | Status |
|---|---|---|
| Data pipeline and preprocessing | — | Complete and verified |
| Deep Pathway (wav2vec 2.0 + LoRA) | Deep Pathway Lead | Scaffolded, untrained |
| Acoustic Pathway (1D-CNN over MFCC) | Acoustic Pathway Lead | Scaffolded, untrained |
| Fusion: joint training loop, hyperparameter search | Fusion Architect | Not started |
| Acoustic error analysis on slurred syllables | Signal Analyst | Blocked on a trained model |

## License

MIT — see [LICENSE](LICENSE). The licence covers this source code only; the UA-Speech database carries its own separate licence and data use agreement.
