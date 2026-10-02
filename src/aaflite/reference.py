"""
Control-referenced normalization: describe every utterance by how far it is
from healthy speakers saying the SAME word.

UA-Speech gives every speaker the same 765 prompts (block x word code). A raw
utterance vector mixes what was said (the word) with how it was said (the
speaker's motor control). Subtracting the healthy-control mean for that exact
prompt removes most of the lexical part; dividing by the pooled within-prompt
spread of the controls puts every feature on a "healthy variability" scale.
One extra scalar per branch — the RMS of that z-vector — is the utterance's
overall distance from healthy speech.

Leak-free by construction: fit() only accepts control speakers (who are never
test speakers and carry no severity label), and uses no labels at all.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src import config


def prompt_keys(df: pd.DataFrame) -> np.ndarray:
    """Block + word code, e.g. 'B2_UW37'. Uncommon words differ per block,
    so the block is part of the prompt identity."""
    return (df["Block"].astype(str) + "_" + df["WordCode"].astype(str)).to_numpy()


class ControlReference:
    """fit(control rows) -> transform(any rows): (N, D) -> (N, D + 1)."""

    def __init__(self, min_std: float = 1e-3):
        self.min_std = min_std
        self.means: Dict[str, np.ndarray] = {}
        self.global_mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame, X: np.ndarray) -> "ControlReference":
        speakers = set(df["Speaker_ID"])
        not_control = speakers - set(config.CONTROL_IDS)
        if not_control:
            raise ValueError(f"ControlReference.fit received non-control speakers {sorted(not_control)}.")
        X = np.asarray(X, dtype=np.float64)
        keys = prompt_keys(df)
        residuals = np.empty_like(X)
        for key in np.unique(keys):
            rows = keys == key
            mean = X[rows].mean(axis=0)
            self.means[key] = mean
            residuals[rows] = X[rows] - mean
        self.global_mean = X.mean(axis=0)
        self.scale = np.maximum(residuals.std(axis=0), self.min_std)
        return self

    def transform(self, df: pd.DataFrame, X: np.ndarray) -> np.ndarray:
        if self.scale is None:
            raise RuntimeError("ControlReference is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        reference = np.stack([self.means.get(key, self.global_mean) for key in prompt_keys(df)])
        z = (X - reference) / self.scale
        distance = np.sqrt(np.mean(z ** 2, axis=1, keepdims=True))
        return np.hstack([z, distance]).astype(np.float32)
