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
import time
from typing import Dict, Iterable, Optional

import pandas as pd

LINE_WIDTH = 78
KEY_WIDTH = 40

# How often progress() is allowed to emit a line. A multi-hour run over 21,420
# files should leave a handful of readable checkpoints in the log, not one line
# per file and not a carriage-return animation — see the note on progress().
PROGRESS_INTERVAL_S = 30.0


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
    print_kv("Voice activity detection", "Silero VAD (neural) — leading/trailing "
             "silence trimmed, internal pauses preserved")
    print_kv("Analysis window", f"{config.CLIP_SECONDS:.0f} s "
                                f"({config.MAX_SAMPLES} samples, pad/truncate)")
    print_kv("Frame length", f"{config.MEL_KWARGS['n_fft']} samples ({frame_ms:.0f} ms)")
    print_kv("Frame shift", f"{config.MEL_KWARGS['hop_length']} samples ({hop_ms:.0f} ms)")
    print_kv("Mel filterbank", f"{config.MEL_KWARGS['n_mels']} triangular filters")
    print_kv("Cepstral coefficients", f"{config.N_MFCC} (DCT of the log-Mel spectrum)")
    print_kv("Dynamic features", f"{DELTA} + {DELTA}{DELTA} {ARROW} {3 * config.N_MFCC}-dim per frame")
    print_kv("Frames per utterance", f"~{n_frames}")
    print()
    print(f"  waveform {ARROW} Silero VAD {ARROW} STFT ({frame_ms:.0f} ms / {hop_ms:.0f} ms) "
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
        print_note("The frozen remainder is wav2vec 2.0's pre-trained backbone — "
                   "only the adapters and head learn.")


def architecture_table(model, model_name: str = "") -> pd.DataFrame:
    """Same per-submodule trainable/total parameter accounting as
    print_architecture, as a DataFrame instead of console output — for
    notebooks/02_feature_analysis.ipynb's parameter-count table (research-
    paper tabular form), computed from the real instantiated model rather
    than re-derived by hand."""
    rows = []
    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        if module_total == 0:
            continue
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append({
            "submodule": name,
            "trainable_params": module_trainable,
            "total_params": module_total,
            "frozen_params": module_total - module_trainable,
            "pct_trainable": 100.0 * module_trainable / module_total,
        })
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows.append({
        "submodule": "TOTAL",
        "trainable_params": trainable,
        "total_params": total,
        "frozen_params": total - trainable,
        "pct_trainable": 100.0 * trainable / total if total else 0.0,
    })
    return pd.DataFrame(rows)


def print_fold_progress(fold_id: str, index: int, total: int,
                        n_train: int, n_val: int, n_test: int) -> None:
    """One fold's header inside a cross-validation run."""
    print()
    print(f"{H_LIGHT * LINE_WIDTH}")
    print(f"  FOLD {index}/{total}  {V}  held-out speaker: {fold_id}  {V}  "
          f"train {n_train:,} / val {n_val:,} / test {n_test:,}")
    print(f"{H_LIGHT * LINE_WIDTH}")


def format_duration(seconds: float) -> str:
    """Human-readable duration, scaled to its own magnitude: '42s', '7m13s',
    '2h05m'. A cache build that takes 40 seconds and a fold that takes four
    hours both get a figure worth reading, rather than everything short
    collapsing to '0h00m'."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


class ProgressReporter:
    """A throttled, line-oriented progress report. No bars, no carriage returns.

    Replaces the tqdm bar this project used to wrap. tqdm renders by rewriting
    one line with '\\r', which is unreadable in any context that does not replay
    carriage returns — and a saved Kaggle notebook is exactly such a context:
    the previous run's log came back 14,862 lines long, the large majority of it
    frozen partial bars, several hundred of them from a single model load. A
    multi-hour unattended run needs a log that can be read after the fact, so
    this emits one complete line at most every PROGRESS_INTERVAL_S seconds:

        Computing VAD spans                 4,800/21,420   22%  elapsed 0h02m  eta 0h07m

    Supports the small slice of the tqdm API this codebase actually used —
    iteration, update(), set_postfix_str(), close(), and the context-manager
    protocol — so every existing call site is unchanged.

    `leave=False` suppresses the completion line, for short inner loops (see
    src.training.engine.run_epoch) whose enclosing stage reports its own summary.
    """

    def __init__(self, iterable=None, description: str = "", total: Optional[int] = None,
                 leave: bool = True, unit: str = "it",
                 interval_s: float = PROGRESS_INTERVAL_S):
        self.iterable = iterable
        self.description = description
        self.total = total if total is not None else _length_or_none(iterable)
        self.leave = leave
        self.unit = unit
        self.interval_s = interval_s
        self.n = 0
        self.postfix = ""
        self._start = time.monotonic()
        self._last_emit = self._start
        self._closed = False

    # -- tqdm-compatible surface --------------------------------------------
    def update(self, n: int = 1) -> None:
        self.n += n
        self._maybe_emit()

    def set_postfix_str(self, postfix: str) -> None:
        self.postfix = postfix

    def set_description(self, description: str) -> None:
        self.description = description

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.leave:
            self._emit(final=True)

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    # -- rendering -----------------------------------------------------------
    def _maybe_emit(self) -> None:
        now = time.monotonic()
        if now - self._last_emit >= self.interval_s:
            self._emit()
            self._last_emit = now

    def _emit(self, final: bool = False) -> None:
        elapsed = time.monotonic() - self._start
        parts = [f"  {self.description:<36}"]

        if self.total:
            parts.append(f"{self.n:>7,}/{self.total:<7,}")
            fraction = self.n / self.total
            parts.append(f"{100 * fraction:3.0f}%")
        else:
            parts.append(f"{self.n:>7,} {self.unit}")

        parts.append(f" elapsed {format_duration(elapsed)}")
        if final:
            # Below a second the elapsed time is mostly measurement noise, so a
            # derived rate would be a made-up number rather than a slow one.
            if elapsed >= 1.0:
                parts.append(f" ({self.n / elapsed:,.1f} {self.unit}/s)")
        elif self.total and self.n:
            remaining = elapsed * (self.total - self.n) / self.n
            parts.append(f" eta {format_duration(remaining)}")
        if self.postfix:
            parts.append(f"  {self.postfix}")

        print("".join(parts), flush=True)


def _length_or_none(iterable) -> Optional[int]:
    """len(iterable) where it is cheap, else None — generators and
    ProcessPoolExecutor.map results have no length, and asking is not an error."""
    try:
        return len(iterable)
    except TypeError:
        return None


def progress(iterable, description: str, total: Optional[int] = None,
             leave: bool = True, unit: str = "it") -> ProgressReporter:
    """Wrap a long-running loop in a ProgressReporter.

    One entry point so every stage (feature extraction, epochs, embedding
    passes, fold loops) reports identically. Signature is unchanged from the
    tqdm-backed version it replaces.
    """
    return ProgressReporter(iterable, description=description, total=total,
                            leave=leave, unit=unit)


def silence_library_progress() -> None:
    """Turn off third-party progress bars and demote Transformers to errors.

    Transformers 5.x prints a per-parameter "Loading weights" bar on every
    from_pretrained call. The three-branch model is rebuilt once per fold, so a
    15-fold run emitted several thousand log lines of it — and the accompanying
    'lm_head.* UNEXPECTED' report is expected here anyway (the checkpoint's
    discarded CTC head, see src/models/deep_pathway.py). Safe to call before the
    libraries are installed or importable: each block degrades to a no-op.
    """
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
