"""
Console reporting: section banners, aligned key/value lines, status and note
lines, tables, and progress bars, so a run reads as one coherent report.
Unicode box-drawing where the console supports it, ASCII otherwise.
"""

import sys
import time
from typing import Optional

import pandas as pd

LINE_WIDTH = 78
KEY_WIDTH = 40

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
    TICK, CROSS, BULLET = "✓", "✗", "•"
else:
    H_HEAVY, H_LIGHT, V = "=", "-", "|"
    TL, TR, BL, BR = "+", "+", "+", "+"
    TICK, CROSS, BULLET = "OK", "X", "*"


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
    """A throttled, line-oriented progress report for piped logs: one
    complete line at most every PROGRESS_INTERVAL_S, no carriage returns.
    Supports the slice of the tqdm API this project uses (iteration, update,
    set_postfix_str, set_description, close, context manager). leave=False
    suppresses the completion line for short inner loops."""

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

    def set_postfix_str(self, postfix: str, refresh: bool = True) -> None:
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
             leave: bool = True, unit: str = "it"):
    """Progress for a long loop (or, with iterable=None, a manual counter):
    a text tqdm bar when config.USE_TQDM (redrawn in place in Jupyter and
    saved in its final state), else the throttled ProgressReporter."""
    from src import config
    if getattr(config, "USE_TQDM", False):
        try:
            from tqdm import tqdm
            return tqdm(iterable, desc=f"  {description}", total=total, leave=leave, unit=unit,
                        file=sys.stdout, ncols=100, mininterval=0.5, smoothing=0.1,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                   "[{elapsed}<{remaining}, {rate_fmt}]{postfix}")
        except ImportError:
            pass
    return ProgressReporter(iterable, description=description, total=total,
                            leave=leave, unit=unit)


def silence_library_progress() -> None:
    """Silence Hugging Face progress bars and demote Transformers logging to
    errors — the model is rebuilt every fold, and the checkpoint's discarded
    CTC head ('lm_head' unexpected keys) is expected."""
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
