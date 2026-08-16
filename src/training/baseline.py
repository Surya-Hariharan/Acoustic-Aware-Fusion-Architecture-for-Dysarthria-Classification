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
from src.training.checkpoint import save_sklearn_model
from src.training.reporting import (FOLD_COMPLETED, aggregate_fold_metrics, describe_fold,
                                    record_fold, save_confusion_matrix, save_metrics,
                                    save_predictions, save_roc_curve)
from src.training.runner import TrainingConfig, build_folds
from src.training.utils import resolve_device

EMBEDDING_CACHE_PATH = config.EMBEDDINGS_DIR / "frozen_wav2vec_base.npz"
ALL_LAYERS_CACHE_PATH = config.EMBEDDINGS_DIR / "frozen_wav2vec_all_layers.npz"
# Separate from EMBEDDING_CACHE_PATH deliberately: that cache mean-pools over
# every frame including the zero-padded tail (kept as-is for the Phase 2 SVM
# baseline, so its numbers stay comparable across reruns) — this one excludes
# padding via the attention-mask fix (see src.models.deep_pathway), which is
# what deep_frozen/fusion_frozen training must actually consume. Conflating
# the two under one cache file would silently mix two different embeddings.
MASKED_EMBEDDING_CACHE_PATH = config.EMBEDDINGS_DIR / "frozen_wav2vec_base_masked.npz"
# All 13 layers, pooled with the attention mask. Deliberately a THIRD cache
# file rather than a flag on ALL_LAYERS_CACHE_PATH: masked and unmasked
# per-layer embeddings are different numbers, and the reproduction diagnostic
# (see sweep_svm_baseline_layers's `masked` argument) needs both side by side to
# report the difference rather than quietly replace one with the other.
ALL_LAYERS_MASKED_CACHE_PATH = config.EMBEDDINGS_DIR / "frozen_wav2vec_all_layers_masked.npz"


@torch.no_grad()
def extract_frozen_embeddings(df: pd.DataFrame, device: Optional[torch.device] = None,
                              batch_size: int = 16, num_workers: int = 4,
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
                                         batch_size: int = 16, num_workers: int = 4,
                                         use_cache: bool = True,
                                         masked: bool = False) -> np.ndarray:
    """
    (N, 13, 768) frozen wav2vec 2.0 embedding per row of df — one vector per
    hidden-state layer (CNN feature-extractor output + 12 transformer
    layers), via DeepPathway.forward_all_layers(). extract_frozen_embeddings()
    only pools the final layer, which can't reproduce the base paper's
    per-layer comparison (Table 1: layer 1 wins detection; Table 3: layer 13
    wins severity) — this is what sweep_svm_baseline_layers() needs instead.

    masked=False (default) mean-pools every frame of the fixed 4-second window,
    including the ~86% that is zero-padding on a median utterance. That is how
    the original Phase 2 sweep was computed, and it stays the default so those
    cached numbers remain reproducible.

    masked=True excludes padded frames from the pool — the more correct
    embedding, and the leading hypothesis for why this reproduction lands at
    82.25% against the paper's 93.95%. It writes to a separate cache file so
    the two can be compared in one diagnostic table instead of one silently
    replacing the other.
    """
    config.ensure_directories()
    cache_path = ALL_LAYERS_MASKED_CACHE_PATH if masked else ALL_LAYERS_CACHE_PATH
    pooling = "attention-masked" if masked else "unmasked (includes padding)"

    if use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_paths = cached["filepaths"]
        if set(df["Filepath"]).issubset(set(cached_paths)):
            index = {path: i for i, path in enumerate(cached_paths)}
            order = [index[p] for p in df["Filepath"]]
            print_kv(f"Frozen per-layer embeddings ({pooling})",
                     f"loaded from cache ({cache_path})")
            return cached["embeddings"][order]

    device = device or resolve_device()
    model = DeepPathway(use_lora=False).to(device).eval()
    loader = DataLoader(UASpeechDataset(df), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    print_header("Extracting Frozen wav2vec 2.0 Embeddings (All 13 Layers)")
    print_kv("Utterances", len(df))
    print_kv("Pooling", pooling)
    print_kv("Device", device)

    embeddings = []
    for batch in progress(loader, f"Frozen wav2vec 2.0 forward (13 layers, {pooling})",
                          total=len(loader), unit="batch"):
        waveform = batch["waveform"].squeeze(1).to(device, non_blocking=True)
        attention_mask = None
        if masked:
            waveform_length = batch["waveform_length"].to(device, non_blocking=True)
            attention_mask = (torch.arange(waveform.shape[1], device=device)[None, :]
                              < waveform_length[:, None])
        embeddings.append(
            model.forward_all_layers(waveform, attention_mask=attention_mask).cpu().numpy())
    embeddings = np.concatenate(embeddings)  # (N, 13, 768)

    np.savez(cache_path, embeddings=embeddings, filepaths=df["Filepath"].to_numpy())
    print_kv(f"Frozen per-layer embeddings ({pooling})",
             f"extracted and cached to {cache_path}")
    return embeddings


@torch.no_grad()
def extract_frozen_embeddings_masked(df: pd.DataFrame, device: Optional[torch.device] = None,
                                     batch_size: int = 16, num_workers: int = 4,
                                     use_cache: bool = True) -> np.ndarray:
    """
    768-dim frozen wav2vec 2.0 embedding per row of df, pooled with the
    attention-mask fix (padded-tail frames excluded) — what deep_frozen and
    fusion_frozen training actually need (see src.training.data.build_loaders'
    frozen_embedding_table wiring). Same one-extraction-per-file rationale and
    cache pattern as extract_frozen_embeddings, kept as a separate function/
    cache file (MASKED_EMBEDDING_CACHE_PATH) rather than a flag on that one,
    since the two are genuinely different numbers and a flag makes it too easy
    to load the wrong cache for a given caller.
    """
    config.ensure_directories()
    if use_cache and MASKED_EMBEDDING_CACHE_PATH.exists():
        cached = np.load(MASKED_EMBEDDING_CACHE_PATH, allow_pickle=True)
        cached_paths = cached["filepaths"]
        if set(df["Filepath"]).issubset(set(cached_paths)):
            index = {path: i for i, path in enumerate(cached_paths)}
            order = [index[p] for p in df["Filepath"]]
            print_kv("Frozen embeddings (masked)", f"loaded from cache ({MASKED_EMBEDDING_CACHE_PATH})")
            return cached["embeddings"][order]

    device = device or resolve_device()
    model = DeepPathway(use_lora=False).to(device).eval()
    loader = DataLoader(UASpeechDataset(df), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))

    print_header("Extracting Frozen wav2vec 2.0 Embeddings (attention-masked)")
    print_kv("Utterances", len(df))
    print_kv("Device", device)

    embeddings = []
    for batch in progress(loader, "Frozen wav2vec 2.0 forward (masked)", total=len(loader),
                          unit="batch"):
        waveform = batch["waveform"].squeeze(1).to(device, non_blocking=True)
        waveform_length = batch["waveform_length"].to(device, non_blocking=True)
        attention_mask = (torch.arange(waveform.shape[1], device=device)[None, :]
                          < waveform_length[:, None])
        embeddings.append(model(waveform, attention_mask=attention_mask).cpu().numpy())
    embeddings = np.concatenate(embeddings)

    np.savez(MASKED_EMBEDDING_CACHE_PATH, embeddings=embeddings,
            filepaths=df["Filepath"].to_numpy())
    print_kv("Frozen embeddings (masked)", f"extracted and cached to {MASKED_EMBEDDING_CACHE_PATH}")
    return embeddings


def run_svm_baseline(df: pd.DataFrame, task: str, embeddings: np.ndarray,
                     run_name: Optional[str] = None, max_folds: Optional[int] = None,
                     ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fit a linear SVM (LinearSVC, Platt-calibrated for probabilities) per
    fold on the frozen wav2vec embeddings — the base paper's frozen
    extractor + SVM pipeline, using the same fold protocol and pooled-
    metric convention as src.training.runner.run_training.

    REGISTERED, like every run_fold-based experiment. Without this, the SVM
    baseline — usually the single most complete run in the project (a plain
    28-fold LOSO / 81-fold leave-one-per-class-out sweep with no GPU budget
    to run out of) — would be invisible to src.results.build_result_tiers and
    could never appear in notebooks/06_results.ipynb's FINAL table, which
    reads the registry exclusively. model="baseline_svm" so it is
    identifiable in the registry without being mistaken for one of
    src.training.models.MODEL_NAMES.
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

    expected_folds = len(fold_iter)
    cv_protocol = "loso" if task == "detection" else "severity_lopco"

    fold_metrics = []
    pooled_true, pooled_pred, pooled_prob = [], [], []

    for fold_index, (fold_id, train_df, test_df) in enumerate(
            progress(fold_iter, "Fitting SVM per LOSO fold",
                     total=len(fold_iter), unit="fold"), start=1):
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

        # A fold's fitted SVM was previously discarded after scoring it -
        # saving it here is what lets a reviewer reload the exact estimator
        # behind a reported fold's numbers instead of only its predictions.
        save_sklearn_model(config.CHECKPOINT_DIR / run_name / f"{fold_id}_svm.pkl", svm)
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
        record_fold(run_name=run_name, model="baseline_svm", task=task,
                   cv_protocol=cv_protocol, fold_id=fold_id, fold_index=fold_index,
                   expected_folds=expected_folds, status=FOLD_COMPLETED,
                   fold_description=describe_fold(test_df),
                   num_classes_present=metrics.get("n_classes_present"))

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


def diagnose_frozen_representation(df: pd.DataFrame, task: str = "detection",
                                    n_folds: int = 8, seed: int = config.DEFAULT_SEED,
                                    mlp_epochs: int = 60, mlp_lr: float = config.DEFAULT_LR_HEAD,
                                    ) -> pd.DataFrame:
    """
    Why does `deep_frozen` (frozen wav2vec2 + MLP head) score far below a linear
    SVM on the same frozen features?

    Screening found deep_frozen at 54.9% accuracy / 0.576 AUROC while the Phase 2
    SVM baseline reached 82.25% / 0.886. Part of that is layer choice —
    deep_frozen consumes `last_hidden_state` (layer 12), and the SVM on layer 12
    scores 72.1%, not 82.25%. This function isolates the REST of the gap by
    holding the features fixed and varying only the classifier:

        LinearSVC              raw          the Phase 2 baseline's classifier
        LinearSVC              standardized does the SVM depend on scaling?
        MLP (deep_frozen head) raw          exactly what deep_frozen trains
        MLP (deep_frozen head) standardized is unnormalized input the problem?

    All four see the SAME cached masked layer-12 embeddings and the same folds,
    so any difference is attributable to the head and its optimization alone. No
    wav2vec2 forward pass and no audio decoding happen here — it runs on the
    existing embedding cache in minutes.

    Reads as: if the SVM rows beat the MLP rows on identical features, the fault
    is in the head or its training. If standardizing rescues the MLP, the fault
    is unnormalized input. If all four land near 72%, then layer choice was the
    whole story and deep_frozen has no bug at all.
    """
    from sklearn.preprocessing import StandardScaler

    device = resolve_device()
    label_column = TASK_LABEL_COLUMN[task]
    label_map = TASK_LABEL_MAP[task]
    num_classes = config.NUM_CLASSES[task]
    df = df.reset_index(drop=True)

    embeddings = extract_frozen_embeddings_masked(df, device=device)
    embed_index = {path: i for i, path in enumerate(df["Filepath"])}

    print_header("deep_frozen diagnostic — same features, four classifiers")
    print_kv("Features", "frozen wav2vec2 layer 12 (final), attention-masked pooling")
    print_kv("Protocol", f"screening {n_folds}-fold, speaker-grouped")
    print_kv("Utterances", len(df))

    cfg = TrainingConfig(task=task, cv_protocol="screening",
                         screening_folds=n_folds, seed=seed)
    folds = list(build_folds(df, task, cfg))

    def _fit_svm(X_train, y_train, X_test):
        svm = CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=5000), method="sigmoid", cv=3)
        svm.fit(X_train, y_train)
        probs = svm.predict_proba(X_test)
        return svm.predict(X_test), (probs[:, 1] if task == "detection" else probs)

    def _fit_mlp(X_train, y_train, X_test):
        """The literal deep_frozen head — src.training.models.DeepClassifier's
        classifier — trained on precomputed embeddings with the same optimizer,
        learning rate, weight decay and class weighting the real run uses."""
        torch.manual_seed(seed)
        head = torch.nn.Sequential(
            torch.nn.Linear(config.WAV2VEC_EMBED_DIM, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes),
        ).to(device)

        counts = np.bincount(y_train, minlength=num_classes).astype(float)
        weights = torch.tensor(len(y_train) / (num_classes * np.maximum(counts, 1)),
                               dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(head.parameters(), lr=mlp_lr,
                                      weight_decay=config.DEFAULT_WEIGHT_DECAY)

        Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
        yt = torch.tensor(y_train, dtype=torch.long, device=device)
        head.train()
        for _ in range(mlp_epochs):
            # Full-batch: ~19k x 768 floats is ~58 MB, trivially resident, and
            # removes minibatch noise as a confound in this comparison.
            optimizer.zero_grad(set_to_none=True)
            criterion(head(Xt), yt).backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            logits = head(torch.tensor(X_test, dtype=torch.float32, device=device))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        return preds, (probs[:, 1] if task == "detection" else probs)

    variants = {
        "LinearSVC / raw": (_fit_svm, False),
        "LinearSVC / standardized": (_fit_svm, True),
        "MLP head / raw": (_fit_mlp, False),
        "MLP head / standardized": (_fit_mlp, True),
    }

    pooled = {name: {"true": [], "pred": [], "prob": []} for name in variants}
    for fold_id, train_df, test_df in progress(folds, "Diagnostic folds",
                                               total=len(folds), unit="fold"):
        train_idx = [embed_index[p] for p in train_df["Filepath"]]
        test_idx = [embed_index[p] for p in test_df["Filepath"]]
        X_train_raw, X_test_raw = embeddings[train_idx], embeddings[test_idx]
        y_train = train_df[label_column].map(label_map).to_numpy()
        y_test = test_df[label_column].map(label_map).to_numpy()

        # Fit the scaler on the TRAIN split only — fitting on all data would
        # leak held-out speaker statistics into the comparison.
        scaler = StandardScaler().fit(X_train_raw)
        for name, (fit_fn, standardize) in variants.items():
            X_train = scaler.transform(X_train_raw) if standardize else X_train_raw
            X_test = scaler.transform(X_test_raw) if standardize else X_test_raw
            y_pred, y_prob = fit_fn(X_train, y_train, X_test)
            pooled[name]["true"].append(y_test)
            pooled[name]["pred"].append(y_pred)
            pooled[name]["prob"].append(y_prob)

    rows = []
    for name in variants:
        metrics = compute_metrics(np.concatenate(pooled[name]["true"]),
                                  np.concatenate(pooled[name]["pred"]),
                                  np.concatenate(pooled[name]["prob"]), task)
        rows.append({"classifier": name, **metrics})

    result = pd.DataFrame(rows)
    config.ensure_directories()
    out_path = config.METRICS_DIR / f"diagnostic_frozen_head_{task}.csv"
    result.to_csv(out_path, index=False)

    print_subheader("Pooled across folds — identical features, different heads")
    print_table(result[["classifier", "accuracy", "precision", "recall", "f1", "auroc"]])
    print_kv("Saved", out_path)
    return result


def sweep_svm_baseline_layers(df: pd.DataFrame, task: str, all_layer_embeddings: np.ndarray,
                              max_folds: Optional[int] = None,
                              variant: str = "") -> pd.DataFrame:
    """
    Run the frozen-wav2vec + SVM baseline once per hidden-state layer, so the
    reproduction can be checked against the base paper's own per-layer
    result: layer 1 wins detection (93.95% acc), layer 13/final wins
    severity (44.56% acc, 4-class). all_layer_embeddings is the (N, 13, 768)
    output of extract_frozen_embeddings_all_layers(); layer 0 is the CNN
    feature-extractor output, layers 1-12 are the transformer layers.

    Writes outputs/metrics/baseline_svm_<task><variant>_layer<i>/ per layer
    (same predictions/metrics/confusion-matrix/ROC layout as run_svm_baseline)
    plus a combined outputs/metrics/baseline_svm_<task><variant>_layer_sweep.csv
    ranking every layer by pooled accuracy.

    `variant` namespaces the whole output set — pass e.g. "_masked" when
    sweeping attention-masked embeddings so the run does NOT overwrite the
    original unmasked sweep. Both are needed side by side: the difference
    between them is the reproduction diagnostic (see
    build_reproduction_diagnostic), and overwriting one with the other would
    destroy the very comparison the diagnostic exists to make.
    """
    num_layers = all_layer_embeddings.shape[1]
    rows = []
    for layer in range(num_layers):
        run_name = f"baseline_svm_{task}{variant}_layer{layer}"
        print_subheader(f"Layer {layer} / {num_layers - 1}{'  ' + variant if variant else ''}")
        _, pooled_metrics = run_svm_baseline(
            df, task, all_layer_embeddings[:, layer, :], run_name=run_name, max_folds=max_folds)
        if pooled_metrics:
            rows.append({"layer": layer, **pooled_metrics})

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    config.ensure_directories()
    summary.to_csv(config.METRICS_DIR / f"baseline_svm_{task}{variant}_layer_sweep.csv",
                   index=False)

    print_subheader(f"Layer sweep summary ({task}{variant}) — best layer first")
    print_table(summary)
    if len(summary):
        best = summary.iloc[0]
        print_kv("Best layer", f"{int(best['layer'])} (accuracy={best['accuracy']:.4f})")

    return summary


def build_reproduction_diagnostic(task: str = "detection",
                                  paper_accuracy: float = 0.9395,
                                  paper_layer: int = 1) -> pd.DataFrame:
    """
    The component-by-component reproduction table the audit asked for: what
    differs between the base paper's setup and ours, and what each difference
    is worth in accuracy.

    Joins the unmasked and attention-masked layer sweeps per layer and reports
    the delta. The masked/unmasked row is the one under test — with a median
    ~86% of the fixed 4-second window being zero-padding
    (notebooks/01_data_pipeline.ipynb Stage 9), an unmasked mean-pool averages
    roughly six parts silence to one part speech, and that is the leading
    candidate for this reproduction landing at 82.25% against the paper's
    93.95%.

    This REPORTS the gap; it does not close it. Preprocessing is not tuned
    toward the paper's number — if masking does not explain the difference,
    the table says so, and the remaining causes (VAD, clip length, layer
    indexing convention, SVM hyperparameters) stay documented as unresolved.
    """
    unmasked_path = config.METRICS_DIR / f"baseline_svm_{task}_layer_sweep.csv"
    masked_path = config.METRICS_DIR / f"baseline_svm_{task}_masked_layer_sweep.csv"

    missing = [p for p in (unmasked_path, masked_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Both layer sweeps are needed for the reproduction diagnostic; missing: "
            + ", ".join(str(p) for p in missing)
            + ". Run sweep_svm_baseline_layers twice — once with the default "
              "embeddings and once with masked=True embeddings and variant='_masked'.")

    unmasked = pd.read_csv(unmasked_path).set_index("layer")
    masked = pd.read_csv(masked_path).set_index("layer")

    comparison = pd.DataFrame({
        "unmasked_accuracy": unmasked["accuracy"],
        "masked_accuracy": masked["accuracy"],
        "unmasked_f1": unmasked["f1"],
        "masked_f1": masked["f1"],
        "unmasked_auroc": unmasked["auroc"],
        "masked_auroc": masked["auroc"],
    }).sort_index()
    comparison["accuracy_delta_pp"] = 100 * (
        comparison["masked_accuracy"] - comparison["unmasked_accuracy"])
    comparison = comparison.reset_index()

    best_unmasked = unmasked["accuracy"].idxmax()
    best_masked = masked["accuracy"].idxmax()

    print_header(f"Baseline reproduction diagnostic — {task}")
    print_table(comparison)

    print_subheader("Component comparison")
    components = pd.DataFrame([
        {"component": "Reported source",
         "reference": "Javanmardi et al., ICASSP 2023",
         "ours": "this pipeline",
         "impact": "—"},
        {"component": "Best layer",
         "reference": f"layer {paper_layer}",
         "ours": f"unmasked: {best_unmasked} / masked: {best_masked}",
         "impact": "layer indexing may differ (our 0 = CNN output)"},
        {"component": "Best accuracy",
         "reference": f"{paper_accuracy:.4f}",
         "ours": f"unmasked: {unmasked['accuracy'].max():.4f} / "
                 f"masked: {masked['accuracy'].max():.4f}",
         "impact": f"gap {paper_accuracy - masked['accuracy'].max():+.4f} after masking"},
        {"component": "Temporal pooling",
         "reference": "not specified in the paper",
         "ours": "mean over frames; masked excludes ~86% padding",
         "impact": f"{100 * (masked['accuracy'].max() - unmasked['accuracy'].max()):+.2f} pp "
                   "at each sweep's best layer"},
        {"component": "VAD",
         "reference": "not specified",
         "ours": "Silero VAD, leading+trailing trim",
         "impact": "unquantified — would need an A/B with VAD_ENABLED=False"},
        {"component": "Clip window",
         "reference": "not specified",
         "ours": f"fixed {config.CLIP_SECONDS}s pad/truncate",
         "impact": "unquantified — drives the padding fraction above"},
        {"component": "Classifier",
         "reference": "linear SVM",
         "ours": "LinearSVC + Platt calibration, class_weight=balanced",
         "impact": "matched"},
    ])
    print_table(components)

    out_path = config.METRICS_DIR / f"reproduction_diagnostic_{task}.csv"
    comparison.to_csv(out_path, index=False)
    components.to_csv(config.METRICS_DIR / f"reproduction_components_{task}.csv", index=False)
    print_kv("Saved", out_path)
    return comparison
