"""
Phase 2 — reproduce the ICASSP base paper's baseline before claiming the
fusion model beats it: a frozen wav2vec 2.0 feature extractor feeding a
linear SVM.

The frozen embedding is identical across every LOSO/severity fold (nothing
about it depends on which speaker is held out), so it is extracted once
and cached; only the SVM is refit per fold. Output layout (predictions/
metrics/confusion-matrix/ROC, pooled-across-folds metrics) mirrors
src.training.runner.run_training so the baseline and the trained models
are directly comparable.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from src import config
from src.console import (print_header, print_kv, print_metrics, print_subheader,
                         print_table, progress)
from src.dataset import UASpeechDataset
from src.models.deep_pathway import DeepPathway
from src.training.data import TASK_LABEL_COLUMN, TASK_LABEL_MAP
from src.training.metrics import compute_confusion_matrix, compute_metrics
from src.training.reporting import (aggregate_fold_metrics, save_confusion_matrix,
                                    save_metrics, save_predictions, save_roc_curve)
from src.training.runner import build_folds
from src.training.utils import resolve_device

EMBEDDING_CACHE_PATH = config.EMBEDDINGS_DIR / "frozen_wav2vec_base.npz"
ALL_LAYERS_CACHE_PATH = config.EMBEDDINGS_DIR / "frozen_wav2vec_all_layers.npz"


@torch.no_grad()
def extract_frozen_embeddings(df: pd.DataFrame, device: Optional[torch.device] = None,
                              batch_size: int = 16, num_workers: int = 0,
                              use_cache: bool = True) -> np.ndarray:
    """
    768-dim frozen wav2vec 2.0 embedding per row of df — the base paper's
    feature extractor, reused via DeepPathway(use_lora=False). Cached to
    outputs/embeddings/ keyed by Filepath, since re-extracting for every
    LOSO fold would be wasteful (the embedding doesn't depend on the fold).
    """
    config.ensure_directories()
    if use_cache and EMBEDDING_CACHE_PATH.exists():
        cached = np.load(EMBEDDING_CACHE_PATH, allow_pickle=True)
        cached_paths = cached["filepaths"]
        if set(df["Filepath"]).issubset(set(cached_paths)):
            index = {path: i for i, path in enumerate(cached_paths)}
            order = [index[p] for p in df["Filepath"]]
            print_kv("Frozen embeddings", f"loaded from cache ({EMBEDDING_CACHE_PATH})")
            return cached["embeddings"][order]

    device = device or resolve_device()
    model = DeepPathway(use_lora=False).to(device).eval()
    loader = DataLoader(UASpeechDataset(df), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    print_header("Extracting Frozen wav2vec 2.0 Embeddings")
    print_kv("Utterances", len(df))
    print_kv("Device", device)

    embeddings = []
    for batch in progress(loader, "Frozen wav2vec 2.0 forward", total=len(loader),
                          unit="batch"):
        waveform = batch["waveform"].squeeze(1).to(device, non_blocking=True)
        embeddings.append(model(waveform).cpu().numpy())
    embeddings = np.concatenate(embeddings)

    np.savez(EMBEDDING_CACHE_PATH, embeddings=embeddings,
            filepaths=df["Filepath"].to_numpy())
    print_kv("Frozen embeddings", f"extracted and cached to {EMBEDDING_CACHE_PATH}")
    return embeddings


@torch.no_grad()
def extract_frozen_embeddings_all_layers(df: pd.DataFrame, device: Optional[torch.device] = None,
                                         batch_size: int = 16, num_workers: int = 0,
                                         use_cache: bool = True) -> np.ndarray:
    """
    (N, 13, 768) frozen wav2vec 2.0 embedding per row of df — one vector per
    hidden-state layer (CNN feature-extractor output + 12 transformer
    layers), via DeepPathway.forward_all_layers(). extract_frozen_embeddings()
    only pools the final layer, which can't reproduce the base paper's
    per-layer comparison (Table 1: layer 1 wins detection; Table 3: layer 13
    wins severity) — this is what sweep_svm_baseline_layers() needs instead.
    """
    config.ensure_directories()
    if use_cache and ALL_LAYERS_CACHE_PATH.exists():
        cached = np.load(ALL_LAYERS_CACHE_PATH, allow_pickle=True)
        cached_paths = cached["filepaths"]
        if set(df["Filepath"]).issubset(set(cached_paths)):
            index = {path: i for i, path in enumerate(cached_paths)}
            order = [index[p] for p in df["Filepath"]]
            print_kv("Frozen per-layer embeddings", f"loaded from cache ({ALL_LAYERS_CACHE_PATH})")
            return cached["embeddings"][order]

    device = device or resolve_device()
    model = DeepPathway(use_lora=False).to(device).eval()
    loader = DataLoader(UASpeechDataset(df), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    print_header("Extracting Frozen wav2vec 2.0 Embeddings (All 13 Layers)")
    print_kv("Utterances", len(df))
    print_kv("Device", device)

    embeddings = []
    for batch in progress(loader, "Frozen wav2vec 2.0 forward (13 layers)",
                          total=len(loader), unit="batch"):
        waveform = batch["waveform"].squeeze(1).to(device, non_blocking=True)
        embeddings.append(model.forward_all_layers(waveform).cpu().numpy())
    embeddings = np.concatenate(embeddings)  # (N, 13, 768)

    np.savez(ALL_LAYERS_CACHE_PATH, embeddings=embeddings,
            filepaths=df["Filepath"].to_numpy())
    print_kv("Frozen per-layer embeddings", f"extracted and cached to {ALL_LAYERS_CACHE_PATH}")
    return embeddings


def run_svm_baseline(df: pd.DataFrame, task: str, embeddings: np.ndarray,
                     run_name: Optional[str] = None, max_folds: Optional[int] = None,
                     ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fit a linear SVM (LinearSVC, Platt-calibrated for probabilities) per
    fold on the frozen wav2vec embeddings — the base paper's frozen
    extractor + SVM pipeline, using the same fold protocol and pooled-
    metric convention as src.training.runner.run_training.
    """
    config.ensure_directories()
    run_name = run_name or f"baseline_svm_{task}"
    label_column = TASK_LABEL_COLUMN[task]
    label_map = TASK_LABEL_MAP[task]
    df = df.reset_index(drop=True)
    embed_index = {path: i for i, path in enumerate(df["Filepath"])}

    print_header("Phase 2 Baseline: Frozen wav2vec 2.0 + Linear SVM")
    print_kv("Task", task)
    print_kv("Run name", run_name)

    fold_iter = list(build_folds(df, task))
    if max_folds is not None:
        fold_iter = fold_iter[:max_folds]

    fold_metrics = []
    pooled_true, pooled_pred, pooled_prob = [], [], []

    for fold_id, train_df, test_df in progress(fold_iter, "Fitting SVM per LOSO fold",
                                               total=len(fold_iter), unit="fold"):
        train_idx = [embed_index[p] for p in train_df["Filepath"]]
        test_idx = [embed_index[p] for p in test_df["Filepath"]]
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train = train_df[label_column].map(label_map).to_numpy()
        y_test = test_df[label_column].map(label_map).to_numpy()

        svm = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=5000), method="sigmoid", cv=3)
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)
        probs = svm.predict_proba(X_test)
        y_prob = probs[:, 1] if task == "detection" else probs

        metrics = compute_metrics(y_test, y_pred, y_prob, task)
        speakers = test_df["Speaker_ID"].to_numpy()
        filenames = test_df["Filename"].to_numpy()

        save_predictions(config.PREDICTIONS_DIR / run_name / f"{fold_id}.csv",
                         filenames, speakers, y_test, y_pred, y_prob, task)
        save_metrics(config.METRICS_DIR / run_name / f"{fold_id}.json",
                    {"fold": fold_id, **metrics})
        save_confusion_matrix(
            config.CONFUSION_MATRIX_DIR / run_name / f"{fold_id}.png",
            compute_confusion_matrix(y_test, y_pred, task), task,
            title=f"{run_name} — fold {fold_id}")
        save_roc_curve(config.ROC_DIR / run_name / f"{fold_id}.png",
                       y_test, y_prob, task, title=f"{run_name} — fold {fold_id}")

        fold_metrics.append({"fold": fold_id, **metrics})
        pooled_true.append(y_test)
        pooled_pred.append(y_pred)
        pooled_prob.append(y_prob)

    if not fold_metrics:
        print_kv("Result", "No folds matched max_folds; nothing was fit.")
        return pd.DataFrame(), {}

    summary = aggregate_fold_metrics(config.METRICS_DIR, run_name, fold_metrics)

    y_true = np.concatenate(pooled_true)
    y_pred = np.concatenate(pooled_pred)
    y_prob = np.concatenate(pooled_prob)
    pooled_metrics = compute_metrics(y_true, y_pred, y_prob, task)

    save_metrics(config.METRICS_DIR / run_name / "ALL_FOLDS_pooled.json", pooled_metrics)
    save_confusion_matrix(
        config.CONFUSION_MATRIX_DIR / run_name / "ALL_FOLDS_pooled.png",
        compute_confusion_matrix(y_true, y_pred, task), task,
        title=f"{run_name} — all folds pooled")
    save_roc_curve(config.ROC_DIR / run_name / "ALL_FOLDS_pooled.png",
                   y_true, y_prob, task, title=f"{run_name} — all folds pooled")

    print_subheader("Per-fold mean +/- std")
    print_table(summary.reset_index().rename(columns={"index": "metric"}))
    print_metrics(pooled_metrics,
                  title="Pooled across all folds (the base-paper-comparable numbers)")

    return summary, pooled_metrics


def sweep_svm_baseline_layers(df: pd.DataFrame, task: str, all_layer_embeddings: np.ndarray,
                              max_folds: Optional[int] = None) -> pd.DataFrame:
    """
    Run the frozen-wav2vec + SVM baseline once per hidden-state layer, so the
    reproduction can be checked against the base paper's own per-layer
    result: layer 1 wins detection (93.95% acc), layer 13/final wins
    severity (44.56% acc, 4-class). all_layer_embeddings is the (N, 13, 768)
    output of extract_frozen_embeddings_all_layers(); layer 0 is the CNN
    feature-extractor output, layers 1-12 are the transformer layers.

    Writes outputs/metrics/baseline_svm_<task>_layer<i>/ per layer (same
    predictions/metrics/confusion-matrix/ROC layout as run_svm_baseline) plus
    a combined outputs/metrics/baseline_svm_<task>_layer_sweep.csv ranking
    every layer by pooled accuracy.
    """
    num_layers = all_layer_embeddings.shape[1]
    rows = []
    for layer in range(num_layers):
        run_name = f"baseline_svm_{task}_layer{layer}"
        print_subheader(f"Layer {layer} / {num_layers - 1}")
        _, pooled_metrics = run_svm_baseline(
            df, task, all_layer_embeddings[:, layer, :], run_name=run_name, max_folds=max_folds)
        if pooled_metrics:
            rows.append({"layer": layer, **pooled_metrics})

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    config.ensure_directories()
    summary.to_csv(config.METRICS_DIR / f"baseline_svm_{task}_layer_sweep.csv", index=False)

    print_subheader(f"Layer sweep summary ({task}) — best layer first")
    print_table(summary)
    if len(summary):
        best = summary.iloc[0]
        print_kv("Best layer", f"{int(best['layer'])} (accuracy={best['accuracy']:.4f})")

    return summary
