"""
Nested leave-one-speaker-out evaluation of AAF-Lite.

Outer loop: the 15 severity LOSO folds (src.splits), one held-out speaker each.
Inner loop: leave-one-speaker-out over the outer fold's 14 training speakers.

Inside every outer fold, for every branch, each configuration in a fixed grid
(feature variant e.g. wav2vec2 layer group, PCA size, L2 strength) gets
out-of-fold probabilities for all 14 training speakers. The best configuration
per branch is chosen on those (balanced accuracy, ties broken by log-loss),
refit on all 14 speakers, and applied to the held-out speaker. Late-fusion
weights for every branch subset, and the decision rule (argmax or the
ordinal expected rank), are chosen the same way, on the inner out-of-fold
probabilities. The held-out speaker therefore never influences
any choice — the reported numbers are honest estimates for an unseen speaker.

Each branch model is StandardScaler -> (whitened PCA) -> multinomial
LogisticRegression with balanced class weights: deterministic, and minutes
for the whole protocol on a CPU.
"""

import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from src import config
from src.splits import iter_severity_loso_folds
from src.training.metrics import compute_confusion_matrix, compute_metrics
from src.training.reporting import (save_confusion_matrix, save_metrics, save_predictions,
                                    speaker_level)

LABELS = list(range(config.NUM_CLASSES))


@dataclass
class Branch:
    """One fusion branch: candidate feature matrices (variant -> (N, D), rows
    aligned with the evaluation dataframe) and the PCA sizes to try
    (None = no PCA)."""
    name: str
    variants: Dict[str, np.ndarray]
    pca_dims: Tuple[Optional[int], ...] = (None,)
    c_grid: Tuple[float, ...] = field(default_factory=lambda: tuple(config.AAFLITE_C_GRID))

    def configurations(self) -> List[Tuple[str, Optional[int], float]]:
        return [(v, n, c) for v in self.variants for n in self.pca_dims for c in self.c_grid]


# ---------------------------------------------------------------------------
# One branch model
# ---------------------------------------------------------------------------
def _fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                 pca_dims: Sequence[Optional[int]], c_grid: Sequence[float]
                 ) -> Dict[Tuple[Optional[int], float], np.ndarray]:
    """Probabilities on X_test for every (pca size, C) — the scaler and PCA
    are fit once on X_train and reused (PCA components are nested)."""
    scaler = StandardScaler().fit(X_train)
    A, B = scaler.transform(X_train), scaler.transform(X_test)
    max_dims = max((n for n in pca_dims if n is not None), default=None)
    if max_dims is not None:
        max_dims = min(max_dims, A.shape[1], A.shape[0] - 1)
        pca = PCA(n_components=max_dims, whiten=True, svd_solver="randomized",
                  random_state=config.DEFAULT_SEED).fit(A)
        A_pca, B_pca = pca.transform(A), pca.transform(B)
    out = {}
    for n in pca_dims:
        if n is None:
            a, b = A, B
        else:
            k = min(n, A_pca.shape[1])
            a, b = A_pca[:, :k], B_pca[:, :k]
        for c in c_grid:
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
            model.fit(a, y_train)
            prob = np.zeros((len(b), config.NUM_CLASSES))
            prob[:, model.classes_] = model.predict_proba(b)
            out[(n, c)] = prob
    return out


def selection_score(y: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """Higher is better: (balanced accuracy, -log-loss) compared lexicographically."""
    clipped = np.clip(prob, 1e-7, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return (float(balanced_accuracy_score(y, prob.argmax(axis=1))),
            -float(log_loss(y, clipped, labels=LABELS)))


def _inner_oof(branch: Branch, X_rows: Dict[str, np.ndarray], y: np.ndarray,
               speakers: np.ndarray) -> Dict[Tuple[str, Optional[int], float], np.ndarray]:
    """Out-of-fold probabilities over the training speakers, per configuration."""
    oof = {cfg: np.zeros((len(y), config.NUM_CLASSES)) for cfg in branch.configurations()}
    for held in np.unique(speakers):
        val = speakers == held
        for variant, X in X_rows.items():
            probs = _fit_predict(X[~val], y[~val], X[val], branch.pca_dims, branch.c_grid)
            for (n, c), prob in probs.items():
                oof[(variant, n, c)][val] = prob
    return oof


DECODERS = ("argmax", "expected_rank")


def decode(prob: np.ndarray, decoder: str) -> np.ndarray:
    """argmax, or the ordinal decision: the probability-weighted severity rank,
    rounded — it predicts a middle class when the evidence is split between
    its neighbours, where argmax jumps to an extreme."""
    if decoder == "argmax":
        return prob.argmax(axis=1)
    if decoder == "expected_rank":
        rank = prob @ np.arange(prob.shape[1]) / np.clip(prob.sum(axis=1), 1e-12, None)
        return np.clip(np.rint(rank), 0, prob.shape[1] - 1).astype(int)
    raise ValueError(f"unknown decoder {decoder!r}")


def choose_decoder(y: np.ndarray, prob: np.ndarray) -> str:
    """The decoder with the higher balanced accuracy on out-of-fold
    probabilities (ties keep argmax)."""
    scores = {d: balanced_accuracy_score(y, decode(prob, d)) for d in DECODERS}
    return max(DECODERS, key=lambda d: (scores[d], d == "argmax"))


def fusion_weights(branch_names: Sequence[str], oof: Dict[str, np.ndarray], y: np.ndarray,
                   step: float = config.AAFLITE_FUSION_STEP) -> Dict[str, float]:
    """Simplex-grid search of late-fusion weights on out-of-fold probabilities."""
    if len(branch_names) == 1:
        return {branch_names[0]: 1.0}
    ticks = int(round(1 / step))
    best, best_score = None, None
    for combo in itertools.product(range(ticks + 1), repeat=len(branch_names)):
        if sum(combo) != ticks:
            continue
        weights = np.array(combo, dtype=float) / ticks
        prob = sum(w * oof[name] for w, name in zip(weights, branch_names))
        score = selection_score(y, prob)
        if best_score is None or score > best_score:
            best, best_score = weights, score
    return {name: float(w) for name, w in zip(branch_names, best)}


# ---------------------------------------------------------------------------
# One outer fold
# ---------------------------------------------------------------------------
def run_outer_fold(fold_id: str, train_idx: np.ndarray, test_idx: np.ndarray,
                   branches: List[Branch], y: np.ndarray, speakers: np.ndarray) -> Dict:
    """Per branch: inner-selected configuration, its training-set out-of-fold
    probabilities (for fusion) and its held-out probabilities."""
    start = time.monotonic()
    y_train, spk_train = y[train_idx], speakers[train_idx]
    result = {"fold": fold_id, "train_idx": train_idx, "test_idx": test_idx, "branches": {}}
    for branch in branches:
        rows = {v: X[train_idx] for v, X in branch.variants.items()}
        oof = _inner_oof(branch, rows, y_train, spk_train)
        best_cfg = max(oof, key=lambda cfg: selection_score(y_train, oof[cfg]))
        variant, n, c = best_cfg
        test_prob = _fit_predict(branch.variants[variant][train_idx], y_train,
                                 branch.variants[variant][test_idx], (n,), (c,))[(n, c)]
        result["branches"][branch.name] = {
            "config": {"variant": variant, "pca_dims": n, "C": c},
            "inner_score": selection_score(y_train, oof[best_cfg])[0],
            "oof": oof[best_cfg], "test": test_prob}
    result["seconds"] = time.monotonic() - start
    return result


# ---------------------------------------------------------------------------
# The whole protocol
# ---------------------------------------------------------------------------
def evaluate(df: pd.DataFrame, branches: List[Branch], subsets: Dict[str, Tuple[str, ...]],
             run_prefix: str, y: Optional[np.ndarray] = None, folds: Optional[List[str]] = None,
             n_jobs: int = config.AAFLITE_N_JOBS, save: bool = True) -> Dict[str, Dict]:
    """Nested LOSO for every branch, then late fusion for every subset.

    df: the dysarthric utterances (rows aligned with every branch matrix).
    subsets: run tag -> branch names, e.g. {"fusion": ("learned", "segmental", "supra")}.
    y: labels override (the permutation sanity check); default = df severity.
    Returns run tag -> {"predictions": DataFrame, "pooled": metrics, "folds": [...]}."""
    df = df.reset_index(drop=True)
    y = (df["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy() if y is None else np.asarray(y))
    speakers = df["Speaker_ID"].to_numpy()
    index = pd.Series(np.arange(len(df)), index=df["Filename"])
    fold_specs = []
    for fold_id, train_df, test_df in iter_severity_loso_folds(df):
        if folds is not None and fold_id not in folds:
            continue
        fold_specs.append((fold_id, index[train_df["Filename"]].to_numpy(),
                           index[test_df["Filename"]].to_numpy()))

    fold_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_outer_fold)(fid, tr, te, branches, y, speakers) for fid, tr, te in fold_specs)

    outputs = {}
    for tag, names in subsets.items():
        frames, fold_rows = [], []
        for res in fold_results:
            tr, te = res["train_idx"], res["test_idx"]
            weights = fusion_weights(names, {n: res["branches"][n]["oof"] for n in names}, y[tr])
            decoder = choose_decoder(y[tr], sum(weights[n] * res["branches"][n]["oof"] for n in names))
            prob = sum(weights[n] * res["branches"][n]["test"] for n in names)
            pred = decode(prob, decoder)
            frame = pd.DataFrame({"filename": df["Filename"].to_numpy()[te], "speaker_id": speakers[te],
                                  "y_true": y[te], "y_pred": pred})
            frames.append((res["fold"], frame, prob))
            fold_rows.append({"fold": res["fold"],
                              "true_label": config.SEVERITY_CLASS_NAMES[int(y[te][0])],
                              "accuracy": float((pred == y[te]).mean()),
                              "ordinal_mae": float(np.abs(pred - y[te]).mean()),
                              "weights": weights, "decoder": decoder,
                              "configs": {n: res["branches"][n]["config"] for n in names},
                              "train_time_s": res["seconds"]})
        outputs[tag] = _finalize(f"{run_prefix}_{tag}", frames, fold_rows, save=save)
    return outputs


def bootstrap_ci(predictions: pd.DataFrame, n_boot: int = config.AAFLITE_BOOTSTRAP,
                 seed: int = config.DEFAULT_SEED) -> Dict[str, Tuple[float, float]]:
    """95% CIs by resampling SPEAKERS (the independent unit), for utterance
    accuracy and speaker-level accuracy."""
    rng = np.random.default_rng(seed)
    by_speaker = {s: g for s, g in predictions.groupby("speaker_id")}
    names = list(by_speaker)
    speaker_correct = {s: int(np.floor(np.median(g["y_pred"]))) == int(g["y_true"].iloc[0])
                       for s, g in by_speaker.items()}
    utt, spk = [], []
    for _ in range(n_boot):
        draw = rng.choice(names, size=len(names), replace=True)
        correct = sum(int((by_speaker[s]["y_pred"] == by_speaker[s]["y_true"]).sum()) for s in draw)
        total = sum(len(by_speaker[s]) for s in draw)
        utt.append(correct / total)
        spk.append(np.mean([speaker_correct[s] for s in draw]))
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"accuracy_ci95": ci(utt), "speaker_accuracy_ci95": ci(spk)}


def _finalize(run_name: str, frames, fold_rows, save: bool) -> Dict:
    predictions = pd.concat([f for _, f, _ in frames], ignore_index=True)
    y_prob = np.concatenate([p for _, _, p in frames])
    y_true, y_pred = predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy()
    pooled = compute_metrics(y_true, y_pred, y_prob)
    speakers = speaker_level(predictions.assign(correct=y_true == y_pred))
    pooled.update({"speaker_accuracy": float(speakers["Correct"].mean()),
                   "speakers_correct": int(speakers["Correct"].sum()), "n_speakers": len(speakers),
                   "speaker_ordinal_mae": float(speakers["Rank error"].mean()),
                   "completed_folds": len(fold_rows), "expected_folds": len(config.DYSARTHRIC_IDS),
                   "run_status": "COMPLETE" if len(fold_rows) == len(config.DYSARTHRIC_IDS) else "PARTIAL",
                   **bootstrap_ci(predictions)})
    if save:
        for (fold_id, frame, prob), row in zip(frames, fold_rows):
            save_predictions(config.PREDICTIONS_DIR / run_name / f"{fold_id}.csv",
                             frame["filename"], frame["speaker_id"], frame["y_true"].to_numpy(),
                             frame["y_pred"].to_numpy(), prob, prob.argmax(axis=1))
            save_metrics(config.METRICS_DIR / run_name / f"{fold_id}.json", row)
        save_metrics(config.METRICS_DIR / run_name / "ALL_FOLDS_pooled.json", pooled)
        save_confusion_matrix(config.CONFUSION_MATRIX_DIR / run_name / "ALL_FOLDS_pooled.png",
                              compute_confusion_matrix(y_true, y_pred), f"{run_name} — pooled")
    return {"run_name": run_name, "predictions": predictions, "y_prob": y_prob,
            "pooled": pooled, "folds": fold_rows, "speakers": speakers}


def majority_baseline(df: pd.DataFrame) -> Dict:
    """Predict each fold's most frequent training class (by utterances)."""
    df = df.reset_index(drop=True)
    y = df["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy()
    frames, rows = [], []
    for fold_id, train_df, test_df in iter_severity_loso_folds(df):
        majority = int(np.bincount(train_df["Severity"].map(config.SEVERITY_LABEL_MAP),
                                   minlength=config.NUM_CLASSES).argmax())
        te = df.index[df["Filename"].isin(test_df["Filename"])].to_numpy()
        prob = np.zeros((len(te), config.NUM_CLASSES))
        prob[:, majority] = 1.0
        frame = pd.DataFrame({"filename": df["Filename"].to_numpy()[te],
                              "speaker_id": df["Speaker_ID"].to_numpy()[te],
                              "y_true": y[te], "y_pred": np.full(len(te), majority)})
        frames.append((fold_id, frame, prob))
        rows.append({"fold": fold_id, "accuracy": float((frame["y_pred"] == frame["y_true"]).mean())})
    return _finalize("aaflite_majority", frames, rows, save=False)


def permuted_labels(df: pd.DataFrame, seed: int = config.DEFAULT_SEED) -> np.ndarray:
    """Severity labels re-assigned across SPEAKERS at random (class counts
    kept). A leak-free pipeline must fall to chance on these."""
    rng = np.random.default_rng(seed)
    speakers = sorted(df["Speaker_ID"].unique())
    shuffled = rng.permutation([config.SEVERITY_MAP[s] for s in speakers])
    mapping = dict(zip(speakers, shuffled))
    return df["Speaker_ID"].map(mapping).map(config.SEVERITY_LABEL_MAP).to_numpy()


def summary_table(outputs: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for tag, out in outputs.items():
        p = out["pooled"]
        rows.append({"Model": tag, "Accuracy": p["accuracy"],
                     "Accuracy 95% CI": "[{:.3f}, {:.3f}]".format(*p["accuracy_ci95"]),
                     "Macro F1": p["f1"], "Balanced acc.": p["balanced_accuracy"],
                     "Ordinal MAE": p["ordinal_mae"], "AUROC": p["auroc"],
                     "Speakers correct": f"{p['speakers_correct']}/{p['n_speakers']}",
                     "Speaker acc. 95% CI": "[{:.2f}, {:.2f}]".format(*p["speaker_accuracy_ci95"])})
    return pd.DataFrame(rows)


def save_summary(outputs: Dict[str, Dict], path) -> None:
    payload = {tag: {"run_name": out["run_name"], "pooled": out["pooled"],
                     "folds": out["folds"]} for tag, out in outputs.items()}
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
