"""
Console reporting.

Every stage of the pipeline prints through this module, so a run reads as a
single coherent report rather than a scroll of ad-hoc print() calls. The output
is written to be read by a speech-processing audience: stages are named for the
signal-processing operations they perform (short-time analysis, Mel filterbank,
cepstral analysis via DCT, linear-prediction formant estimation), and the
architecture summary reports each pathway's dimensionality and trainable
parameter count rather than an opaque module dump.

Unicode box-drawing is used where the terminal supports it and degrades to ASCII
where it does not (Windows consoles under cp1252), so the same code renders
correctly in a notebook and in a piped log.
"""

import sys
from typing import Dict, Iterable, Optional

import pandas as pd

LINE_WIDTH = 78
KEY_WIDTH = 40


def _supports_unicode() -> bool:
    """Whether stdout can encode box-drawing characters."""
    encoding = getattr(sys.stdout, "encoding", None) or ""
    try:
        "═╔╗╚╝║─│".encode(encoding)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


_UNICODE = _supports_unicode()

# Glyphs, with an ASCII fallback so a piped log never raises UnicodeEncodeError.
if _UNICODE:
    H_HEAVY, H_LIGHT, V = "═", "─", "│"
    TL, TR, BL, BR = "╔", "╗", "╚", "╝"
    TICK, CROSS, ARROW, BULLET = "✓", "✗", "→", "•"
    BLOCK_FULL, BLOCK_EMPTY = "█", "░"
    DELTA = "Δ"
else:
    H_HEAVY, H_LIGHT, V = "=", "-", "|"
    TL, TR, BL, BR = "+", "+", "+", "+"
    TICK, CROSS, ARROW, BULLET = "OK", "X", "->", "*"
    BLOCK_FULL, BLOCK_EMPTY = "#", "."
    DELTA = "d"


# ---------------------------------------------------------------------------
# Structural output
# ---------------------------------------------------------------------------
def print_banner(title: str, subtitle: str = "") -> None:
    """Top-of-run title box. One per notebook stage at most."""
    inner = LINE_WIDTH - 2
    print()
    print(f"{TL}{H_HEAVY * inner}{TR}")
    print(f"{V}{title.upper().center(inner)}{V}")
    if subtitle:
        print(f"{V}{subtitle.center(inner)}{V}")
    print(f"{BL}{H_HEAVY * inner}{BR}")


def print_header(title: str) -> None:
    """Top-level section banner."""
    print()
    print(H_HEAVY * LINE_WIDTH)
    print(f"  {title.upper()}")
    print(H_HEAVY * LINE_WIDTH)


def print_subheader(title: str) -> None:
    """Second-level banner inside a section."""
    print()
    print(f"{H_LIGHT * 3} {title} " + H_LIGHT * max(0, LINE_WIDTH - len(title) - 5))


def print_kv(key: str, value) -> None:
    """Aligned 'key ....... value' line."""
    key = str(key)
    dots = "." * max(1, KEY_WIDTH - len(key))
    print(f"  {key} {dots} {value}")


def print_status(message: str, ok: bool = True) -> None:
    """Single-line pass/fail status."""
    tag = f"[ {TICK} ]" if ok else f"[ {CROSS} ]"
    print(f"  {tag} {message}")


def print_note(message: str) -> None:
    """An indented caveat — things the reader must not misread the numbers without."""
    print(f"  {BULLET} {message}")


def print_table(df: pd.DataFrame, index: bool = False, indent: int = 2,
                float_format: str = "{:.4f}") -> None:
    """Print a DataFrame as an aligned text table."""
    text = df.to_string(index=index, float_format=float_format.format)
    pad = " " * indent
    print("\n".join(pad + line for line in text.splitlines()))


def print_series(series: pd.Series, indent: int = 2) -> None:
    """Print a Series with aligned values."""
    pad = " " * indent
    print("\n".join(pad + line for line in series.to_string().splitlines()))


# ---------------------------------------------------------------------------
# Domain-specific reporting
# ---------------------------------------------------------------------------
def print_metrics(metrics: Dict[str, float], title: str = "Metrics",
                  highlight: Iterable[str] = ("accuracy", "f1")) -> None:
    """
    A metric block with a bar per value, so relative performance is legible at a
    glance instead of requiring the reader to compare decimals.

    All six metrics here are in [0, 1] (accuracy, precision, recall/sensitivity,
    specificity, F1, AUROC), which is what makes a shared bar scale meaningful.
    """
    print_subheader(title)
    highlight = set(highlight)

    for name, value in metrics.items():
        if value is None or not isinstance(value, (int, float)):
            continue
        if value != value:                                  # NaN
            print(f"  {name:<14} {'':<22}   undefined (a class was absent)")
            continue

        filled = int(round(max(0.0, min(1.0, float(value))) * 20))
        bar = BLOCK_FULL * filled + BLOCK_EMPTY * (20 - filled)
        marker = f" {ARROW}" if name in highlight else "  "
        print(f"  {name:<14} {bar}   {value:.4f}{marker}")


def print_signal_chain() -> None:
    """
    The deterministic front end, named for what each stage actually is.

    This is the short-time analysis path every utterance takes before it reaches
    either pathway, and it is worth printing once per run: the numbers below are
    the analysis parameters every downstream result depends on.
    """
    from src import config

    frame_ms = config.MEL_KWARGS["n_fft"] / config.TARGET_SR * 1000
    hop_ms = config.MEL_KWARGS["hop_length"] / config.TARGET_SR * 1000
    n_frames = config.MAX_SAMPLES // config.MEL_KWARGS["hop_length"] + 1

    print_subheader("Front end — short-time analysis")
    print_kv("Sampling rate", f"{config.TARGET_SR} Hz, mono")
    print_kv("Voice activity detection", "silence trimmed (torchaudio VAD)")
    print_kv("Analysis window", f"{config.CLIP_SECONDS:.0f} s "
                                f"({config.MAX_SAMPLES} samples, pad/truncate)")
    print_kv("Frame length", f"{config.MEL_KWARGS['n_fft']} samples ({frame_ms:.0f} ms)")
    print_kv("Frame shift", f"{config.MEL_KWARGS['hop_length']} samples ({hop_ms:.0f} ms)")
    print_kv("Mel filterbank", f"{config.MEL_KWARGS['n_mels']} triangular filters")
    print_kv("Cepstral coefficients", f"{config.N_MFCC} (DCT of the log-Mel spectrum)")
    print_kv("Dynamic features", f"{DELTA} + {DELTA}{DELTA} {ARROW} {3 * config.N_MFCC}-dim per frame")
    print_kv("Frames per utterance", f"~{n_frames}")
    print()
    print(f"  waveform {ARROW} VAD {ARROW} STFT ({frame_ms:.0f} ms / {hop_ms:.0f} ms) "
          f"{ARROW} Mel filterbank {ARROW} log {ARROW} DCT")
    print(f"           {ARROW} {config.N_MFCC} MFCC + {DELTA} + {DELTA}{DELTA} "
          f"{ARROW} {3 * config.N_MFCC}-dim  [Acoustic Pathway]")
    print(f"  waveform {ARROW} wav2vec 2.0 (self-supervised, LoRA-adapted) "
          f"{ARROW} {config.WAV2VEC_EMBED_DIM}-dim  [Deep Pathway]")


def print_architecture(model, model_name: str = "") -> None:
    """
    Per-pathway dimensionality and trainable-parameter accounting.

    The trainable/total split is the point of the LoRA argument and is worth
    stating numerically: the backbone stays frozen while a small set of
    rank-decomposition adapters carries the adaptation to pathological speech.
    """
    import torch.nn as nn

    print_subheader(f"Architecture — {model_name or model.__class__.__name__}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        if module_total == 0:
            continue
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        share = 100.0 * module_trainable / module_total if module_total else 0.0
        print_kv(f"  {name}", f"{module_trainable:>11,} / {module_total:>11,} trainable "
                              f"({share:5.1f}%)")

    embed_dim = getattr(model, "embed_dim", None)
    if embed_dim is not None:
        print_kv("Fused embedding", f"{embed_dim}-dim")

    print_kv("TOTAL trainable", f"{trainable:,} / {total:,} "
                                f"({100.0 * trainable / total:.2f}%)")
    if trainable < total:
        print_note(f"The frozen remainder is wav2vec 2.0's pre-trained backbone — "
                   f"only the adapters and head learn.")


def print_fold_progress(fold_id: str, index: int, total: int,
                        n_train: int, n_val: int, n_test: int) -> None:
    """One fold's header inside a cross-validation run."""
    print()
    print(f"{H_LIGHT * LINE_WIDTH}")
    print(f"  FOLD {index}/{total}  {V}  held-out speaker: {fold_id}  {V}  "
          f"train {n_train:,} / val {n_val:,} / test {n_test:,}")
    print(f"{H_LIGHT * LINE_WIDTH}")


def progress(iterable, description: str, total: Optional[int] = None,
             leave: bool = True, unit: str = "it"):
    """
    A tqdm progress bar with the project's house style.

    Wrapped in one place so every long-running loop (feature extraction, epochs,
    embedding passes) reports identically, and so tqdm stays a single import to
    swap if it is ever unavailable.
    """
    from tqdm.auto import tqdm

    return tqdm(iterable, desc=f"  {description}", total=total, leave=leave,
                unit=unit, ncols=90, bar_format=(
                    "{desc:<34} {percentage:3.0f}%|{bar}| "
                    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"))
