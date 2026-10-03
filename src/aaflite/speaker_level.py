"""
Session-level severity: one decision per speaker.

A severity label belongs to the SPEAKER — every one of a speaker's 765
utterances carries the same class — so the statistical unit is the 15 speakers,
not 11,475 utterances. Utterance-level classifiers fitted on 14 training
speakers learn voices, not severity (the features identify the speaker far
more easily than the severity). Here each speaker is summarized by the MEAN of
a few per-utterance scalars over all of its held-out recordings — a clinician's
assessment session — and classified by a Gaussian rule over the 14 training
speakers' summaries: class means, one pooled within-class variance per feature,
equal class priors. In one dimension that is nearest-class-mean.

Nothing is tuned: the feature set of every model is fixed in advance (see
src.aaflite.run.SPEAKER_MODELS), the held-out speaker's label is never used,
and its own recordings are only averaged (labels are not needed to do that).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src import config
from src.aaflite.pipeline import _finalize
from src.splits import iter_severity_loso_folds


class SpeakerGaussian:
    """Equal-prior Gaussian classifier on speaker-level feature vectors."""

    def __init__(self, min_variance: float = 1e-12):
        self.min_variance = min_variance

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpeakerGaussian":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        resid = X - self.means_[np.searchsorted(self.classes_, y)]
        dof = len(X) - len(self.classes_)
        variance = (resid ** 2).sum(axis=0) / dof if dof > 0 else X.var(axis=0)
        floor = np.maximum(self.min_variance, 1e-6 * X.var(axis=0))
        self.variance_ = np.maximum(variance, floor)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, NUM_CLASSES); classes never seen in training get probability 0."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        d2 = (((X[:, None, :] - self.means_[None]) ** 2) / self.variance_).sum(axis=2)
        logit = -0.5 * d2
        logit -= logit.max(axis=1, keepdims=True)
        prob = np.exp(logit)
        prob /= prob.sum(axis=1, keepdims=True)
        full = np.zeros((len(X), config.NUM_CLASSES))
        full[:, self.classes_] = prob
        return full


def speaker_means(speaker_ids: np.ndarray, X: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """(speakers in sorted order, (n_speakers, d)) — the mean of each speaker's rows."""
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    names = sorted(set(speaker_ids.tolist()))
    return names, np.stack([X[speaker_ids == s].mean(axis=0) for s in names])


def evaluate_speaker_level(df: pd.DataFrame, scalars: Dict[str, np.ndarray],
                           models: Dict[str, Sequence[str]], run_prefix: str,
                           y: Optional[np.ndarray] = None, folds: Optional[List[str]] = None,
                           save: bool = True) -> Dict[str, Dict]:
    """Leave-one-speaker-out for every model in `models`.

    scalars: name -> (N,) per-utterance values, rows aligned with df.
    models: run tag -> the scalar names it uses (fixed in advance).
    y: label override (the permuted-label sanity check).
    Returns tag -> the same dict as src.aaflite.pipeline.evaluate; every
    utterance of a held-out speaker carries that speaker's decision, so the
    pooled accuracy equals the share of speakers classified correctly."""
    df = df.reset_index(drop=True)
    y = (df["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy() if y is None else np.asarray(y))
    speakers = df["Speaker_ID"].to_numpy()
    index = pd.Series(np.arange(len(df)), index=df["Filename"])
    outputs = {}
    for tag, names in models.items():
        X = np.stack([np.asarray(scalars[n], dtype=np.float64) for n in names], axis=1)
        spk_names, spk_X = speaker_means(speakers, X)
        spk_y = np.array([int(y[speakers == s][0]) for s in spk_names])
        position = {s: i for i, s in enumerate(spk_names)}
        frames, rows = [], []
        for fold_id, _, test_df in iter_severity_loso_folds(df):
            if folds is not None and fold_id not in folds:
                continue
            train = np.array([s != fold_id for s in spk_names])
            model = SpeakerGaussian().fit(spk_X[train], spk_y[train])
            prob_speaker = model.predict_proba(spk_X[position[fold_id]][None])[0]
            te = index[test_df["Filename"]].to_numpy()
            prob = np.tile(prob_speaker, (len(te), 1))
            pred = np.full(len(te), int(prob_speaker.argmax()))
            frames.append((fold_id, pd.DataFrame({"filename": df["Filename"].to_numpy()[te],
                                                  "speaker_id": speakers[te], "y_true": y[te],
                                                  "y_pred": pred}), prob))
            rows.append({"fold": fold_id,
                         "true_label": config.SEVERITY_CLASS_NAMES[int(y[te][0])],
                         "accuracy": float((pred == y[te]).mean()),
                         "ordinal_mae": float(np.abs(pred - y[te]).mean()),
                         "features": list(names),
                         "speaker_values": {n: float(v) for n, v in zip(names, spk_X[position[fold_id]])}})
        outputs[tag] = _finalize(f"{run_prefix}_{tag}", frames, rows, save=save)
    return outputs


def _inner_loso_score(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Leave-one-speaker-out inside the training speakers: (speakers classified
    correctly, ordinal error summed) — selection prefers more correct, then
    fewer rank errors."""
    correct, error = 0, 0.0
    for v in range(len(X)):
        keep = np.arange(len(X)) != v
        pred = int(SpeakerGaussian().fit(X[keep], y[keep]).predict_proba(X[v][None])[0].argmax())
        correct += pred == y[v]
        error += abs(pred - y[v])
    return correct, error


def evaluate_speaker_level_nested(df: pd.DataFrame, scalars: Dict[str, np.ndarray],
                                  candidates: Dict[str, Sequence[str]], run_prefix: str, tag: str,
                                  y: Optional[np.ndarray] = None, folds: Optional[List[str]] = None,
                                  save: bool = True) -> Dict[str, Dict]:
    """Session-level evaluation with the model itself chosen by nested
    leave-one-speaker-out: which recogniser / checkpoint / scalar set to use.

    For each outer fold (one held-out speaker), every candidate is scored by an
    inner leave-one-speaker-out over the OTHER 14 speakers; the best one (most
    speakers correct, then smallest ordinal error, then declaration order) is
    refit on all 14 and applied once to the held-out speaker. The held-out
    speaker affects neither the choice nor any fit — this is the number that
    answers "how well does the procedure classify a speaker it has never seen,
    when the procedure itself is decided without that speaker?".

    Limits of what it can show: the candidate FAMILY was designed by looking at
    these 15 speakers, so only an independent corpus removes that last bias."""
    df = df.reset_index(drop=True)
    y = (df["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy() if y is None else np.asarray(y))
    speakers = df["Speaker_ID"].to_numpy()
    index = pd.Series(np.arange(len(df)), index=df["Filename"])
    names = list(candidates)
    spk_names = sorted(set(speakers.tolist()))
    position = {s: i for i, s in enumerate(spk_names)}
    spk_y = np.array([int(y[speakers == s][0]) for s in spk_names])
    spk_X = {c: speaker_means(speakers, np.stack([np.asarray(scalars[n], dtype=np.float64)
                                                  for n in candidates[c]], axis=1))[1] for c in names}
    frames, rows = [], []
    for fold_id, _, test_df in iter_severity_loso_folds(df):
        if folds is not None and fold_id not in folds:
            continue
        train = np.array([s != fold_id for s in spk_names])
        scores = {c: _inner_loso_score(spk_X[c][train], spk_y[train]) for c in names}
        chosen = max(names, key=lambda c: (scores[c][0], -scores[c][1], -names.index(c)))
        model = SpeakerGaussian().fit(spk_X[chosen][train], spk_y[train])
        prob_speaker = model.predict_proba(spk_X[chosen][position[fold_id]][None])[0]
        te = index[test_df["Filename"]].to_numpy()
        pred = np.full(len(te), int(prob_speaker.argmax()))
        frames.append((fold_id, pd.DataFrame({"filename": df["Filename"].to_numpy()[te],
                                              "speaker_id": speakers[te], "y_true": y[te],
                                              "y_pred": pred}), np.tile(prob_speaker, (len(te), 1))))
        rows.append({"fold": fold_id, "true_label": config.SEVERITY_CLASS_NAMES[int(y[te][0])],
                     "accuracy": float((pred == y[te]).mean()),
                     "ordinal_mae": float(np.abs(pred - y[te]).mean()),
                     "chosen": chosen, "inner_correct": int(scores[chosen][0]),
                     "inner_scores": {c: int(scores[c][0]) for c in names},
                     "features": list(candidates[chosen]),
                     "speaker_values": {n: float(v) for n, v in
                                        zip(candidates[chosen], spk_X[chosen][position[fold_id]])}})
    return {tag: _finalize(f"{run_prefix}_{tag}", frames, rows, save=save)}
