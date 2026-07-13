"""
Console formatting helpers so every stage of the pipeline prints
in a consistent, neatly aligned style.
"""

import pandas as pd

LINE_WIDTH = 72
KEY_WIDTH = 38


def print_header(title: str) -> None:
    """Top-level section banner."""
    print()
    print("=" * LINE_WIDTH)
    print(f"  {title.upper()}")
    print("=" * LINE_WIDTH)


def print_subheader(title: str) -> None:
    """Second-level banner inside a section."""
    print()
    print(f"--- {title} " + "-" * max(0, LINE_WIDTH - len(title) - 5))


def print_kv(key: str, value) -> None:
    """Aligned 'key ....... value' line."""
    dots = "." * max(1, KEY_WIDTH - len(key))
    print(f"  {key} {dots} {value}")


def print_table(df: pd.DataFrame, index: bool = False, indent: int = 2) -> None:
    """Print a DataFrame as an aligned text table."""
    text = df.to_string(index=index)
    pad = " " * indent
    print("\n".join(pad + line for line in text.splitlines()))


def print_series(series: pd.Series, indent: int = 2) -> None:
    """Print a Series with aligned values."""
    pad = " " * indent
    print("\n".join(pad + line for line in series.to_string().splitlines()))


def print_status(message: str, ok: bool = True) -> None:
    """Single-line pass/fail status."""
    tag = "[ OK ]" if ok else "[FAIL]"
    print(f"  {tag} {message}")
