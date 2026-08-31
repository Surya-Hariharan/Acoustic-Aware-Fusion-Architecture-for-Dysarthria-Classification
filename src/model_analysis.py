"""
Phase 6 - Model analysis and interpretability.

Where src/error_analysis.py asks "what does the model get wrong", this module
asks "what has the model actually learned": how the six ablation variants
compare, how the learned embedding space is organised, what the attention
fusion models (Model E/F) attend to, and which acoustic measurements drive a
prediction.

Consumes what Phase 1/2/4 already write:
  outputs/metrics/phase{2,3}_*.csv        ablation comparison tables
  outputs/embeddings/<run>/<fold>.npz     test-fold embeddings
  outputs/checkpoints/<run>/<fold>/best.pt  trained weights, for attention maps
  outputs/praat_features.csv              Phase 4 acoustic features

Like src/error_analysis.py and src/visualization.py, this module owns its own
plots and saves them to outputs/figures/.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src import config
from src.console import print_kv, progress
from src.praat import FEATURE_COLUMNS, FEATURE_GROUPS
from src.style import apply_style, branch_color, model_color

apply_style()


def _finish(fig, filename: str, show: bool, subdir: Optional[Path] = None) -> str:
    """Save a figure to outputs/figures/ (or a named subdirectory —
    config.{REPRESENTATION,EXPLAINABILITY,ABLATION,METRIC}_FIGURE_DIR),
    optionally display it, then release it."""
    out_dir = subdir or config.FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return str(out_path)


# ---------------------------------------------------------------------------
# Ablation comparison
# ---------------------------------------------------------------------------
def plot_ablation_comparison(comparison_df: pd.DataFrame,
                             metrics: Tuple[str, ...] = ("accuracy", "f1", "recall", "specificity"),
                             title: str = "Ablation Comparison", show: bool = False) -> str:
    """
    Grouped bar chart of every ablation variant across a metric subset - the
    visual counterpart to the phase2/phase3 comparison tables notebooks/03
    already writes, so the six-variant story in the paper doesn't rest on a
    reader scanning a table of decimals.
    """
    available = [m for m in metrics if m in comparison_df.columns]
    if not available:
        raise ValueError(f"None of {metrics} are columns of comparison_df ({list(comparison_df.columns)}).")

    id_col = comparison_df.index.name or "model"
    plot_df = (comparison_df[available].reset_index()
              .rename(columns={comparison_df.index.name or "index": id_col})
              .melt(id_vars=id_col, var_name="metric", value_name="value"))

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(comparison_df)), 6))
    sns.barplot(data=plot_df, x=id_col, y="value", hue="metric", ax=ax, palette="viridis")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="", loc="lower right", fontsize=9)
    fig.tight_layout()
    return _finish(fig, "ablation_comparison.png", show, subdir=config.ABLATION_FIGURE_DIR)


# ---------------------------------------------------------------------------
# Embedding-space visualization (t-SNE / PCA / UMAP)
# ---------------------------------------------------------------------------
BRANCH_EMBEDDING_KEYS = {
    "fused": "embeddings",
    "learned": "branch_learned",
    "segmental": "branch_segmental",
    "supra": "branch_supra",
}


def load_run_embeddings(run_name: str, embedding_type: str = "fused") -> pd.DataFrame:
    """
    Every test-fold embedding for a run, as a DataFrame with `filename`,
    `y_true`, `speaker_ids`, and an `embedding` column of vectors.

    embedding_type selects WHICH saved array to load — "fused" (Z_unified,
    every model), or "learned"/"segmental"/"supra" (GatedFusionModel only —
    see src.training.engine.EpochResult.branch_embeddings /
    src.training.reporting.save_embeddings). Raises a clear error if a
    branch-specific type is requested for a run that never saved one
    (e.g. a legacy ablation-ladder run).
    """
    if embedding_type not in BRANCH_EMBEDDING_KEYS:
        raise ValueError(f"Unknown embedding_type '{embedding_type}'. "
                         f"Choose from {list(BRANCH_EMBEDDING_KEYS)}.")
    array_key = BRANCH_EMBEDDING_KEYS[embedding_type]

    run_dir = config.EMBEDDINGS_DIR / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"No embeddings for run '{run_name}' at {run_dir}.")

    frames = []
    for path in sorted(run_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if "filenames" not in data:
            raise ValueError(
                f"{path} has no 'filenames' array — it was written before "
                f"utterance identity was carried through. Re-run '{run_name}'."
            )
        if array_key not in data:
            raise ValueError(
                f"{path} has no '{array_key}' array — embedding_type="
                f"'{embedding_type}' requires a run trained with the "
                "three-branch architecture (src.models.gated_fusion.GatedFusionModel)."
            )
        frames.append(pd.DataFrame({
            "filename": data["filenames"],
            "y_true": data["y_true"],
            "speaker_id": data["speaker_ids"] if "speaker_ids" in data else None,
            "embedding": list(data[array_key]),
        }))
    return pd.concat(frames, ignore_index=True)


def plot_embedding_map(run_name: str, preds: pd.DataFrame, task: str = "detection",
                       method: str = "tsne", embedding_type: str = "fused",
                       color_by: str = "severity", max_points: int = 3000, seed: int = 42,
                       show: bool = False) -> str:
    """
    2D projection of a test-fold embedding set.

    method="tsne" (default): local structure. "pca": linear, deterministic —
    a sanity check the nonlinear methods' layout can be compared against.
    "umap": nonlinear like t-SNE but tends to preserve more global structure
    between clusters; needs the optional `umap-learn` package.

    embedding_type: "fused" (Z_unified, works for any run) or
    "learned"/"segmental"/"supra" (GatedFusionModel branch embeddings —
    see load_run_embeddings).

    color_by: "severity" (or "detection" — the true class label,
    misclassified points marked with an X) or "speaker" (colors by
    Speaker_ID instead, with NO misclassification marker — the diagnostic
    this view is for is whether points cluster by SPEAKER rather than by
    class; visual separation here is a representation diagnostic, not proof
    of speaker leakage or its absence).
    """
    embeddings = load_run_embeddings(run_name, embedding_type=embedding_type)
    merge_cols = ["filename", "correct", "y_true_label"]
    merged = embeddings.merge(preds[merge_cols], on="filename", how="inner")
    if merged.empty:
        raise ValueError(f"No embedding rows for '{run_name}' matched its predictions.")

    if len(merged) > max_points:
        merged = merged.sample(max_points, random_state=seed).reset_index(drop=True)

    matrix = np.vstack(merged["embedding"].to_numpy())

    if method == "tsne":
        coords = TSNE(n_components=2, random_state=seed,
                      perplexity=min(30, max(5, len(merged) // 4))).fit_transform(matrix)
    elif method == "pca":
        coords = PCA(n_components=2, random_state=seed).fit_transform(matrix)
    elif method == "umap":
        import umap  # optional dependency — see requirements.txt
        coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(matrix)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'tsne', 'pca', 'umap'.")

    merged["x"], merged["y"] = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(9, 8))
    if color_by == "speaker":
        if merged["speaker_id"].isna().all():
            raise ValueError(f"Run '{run_name}' has no saved speaker_id — cannot color by speaker.")
        sns.scatterplot(data=merged, x="x", y="y", hue="speaker_id", palette="tab20",
                        s=16, alpha=0.6, linewidth=0, ax=ax, legend=False)
        ax.set_title(f"{run_name} — {method.upper()} of {embedding_type} embeddings, colored by SPEAKER",
                    fontsize=13)
    elif color_by in ("severity", "detection"):
        correct = merged[merged["correct"].astype(bool)]
        wrong = merged[~merged["correct"].astype(bool)]
        sns.scatterplot(data=correct, x="x", y="y", hue="y_true_label", palette="viridis",
                        s=14, alpha=0.45, linewidth=0, ax=ax)
        ax.scatter(wrong["x"], wrong["y"], marker="x", s=42, c="#c44e52",
                  linewidths=1.2, label=f"misclassified (n={len(wrong)})")
        ax.legend(loc="best", fontsize=9)
        ax.set_title(f"{run_name} — {method.upper()} of {embedding_type} embeddings ({task})",
                    fontsize=13)
    else:
        raise ValueError(f"Unknown color_by '{color_by}'. Choose from 'severity', 'detection', 'speaker'.")

    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    fig.tight_layout()

    return _finish(fig, f"embedding_map_{method}_{embedding_type}_{color_by}_{run_name}.png", show,
                   subdir=config.REPRESENTATION_FIGURE_DIR)


# ---------------------------------------------------------------------------
# Attention visualization (Model E only — see attention_weights() below)
# ---------------------------------------------------------------------------
def plot_attention_heatmap(run_name: str, task: str = "detection",
                           model_name: str = "attention_fusion",
                           fold_id: Optional[str] = None, sample_index: int = 0,
                           device: Optional[str] = None, show: bool = False) -> Optional[str]:
    """
    One utterance's bidirectional cross-attention map: which MFCC frames each
    wav2vec frame draws on, and vice versa.

    Only `AttentionFusionModel` (model="attention_fusion", Ablation Model E)
    exposes `attention_weights()` (src/models/attention_fusion.py) — Model F's
    tri-modal blocks don't currently expose an equivalent hook. This is the
    "if available" in the target notebook structure: for any other model_name
    it prints a note and returns None instead of raising.
    """
    from src.dataset import UASpeechDataset
    from src.training.checkpoint import load_checkpoint
    from src.training.data import load_manifest
    from src.training.models import build_model
    from src.training.utils import resolve_device

    num_classes = config.NUM_CLASSES[task]
    model = build_model(model_name, num_classes=num_classes)
    if not hasattr(model, "attention_weights"):
        print_kv("Attention visualization",
                 f"'{model_name}' has no attention_weights() — skipped")
        return None

    ckpt_dir = config.CHECKPOINT_DIR / run_name
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoints for run '{run_name}' at {ckpt_dir}.")
    if fold_id is None:
        fold_dirs = sorted(p.name for p in ckpt_dir.iterdir() if (p / "best.pt").exists())
        if not fold_dirs:
            raise FileNotFoundError(f"No completed fold checkpoints under {ckpt_dir}.")
        fold_id = fold_dirs[0]

    device = resolve_device(device)
    model = model.to(device)
    load_checkpoint(ckpt_dir / fold_id / "best.pt", model, map_location=str(device))
    model.eval()

    manifest = load_manifest()
    row = manifest.iloc[[sample_index]]
    item = UASpeechDataset(row)[0]
    waveform = item["waveform"].unsqueeze(0).to(device)
    mfcc = item["mfcc"].unsqueeze(0).to(device)

    deep_over_acoustic, acoustic_over_deep = model.attention_weights(waveform, mfcc)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.heatmap(deep_over_acoustic[0].cpu().numpy(), cmap="magma", ax=axes[0], cbar=True)
    axes[0].set_title("wav2vec frames attending over MFCC frames")
    axes[0].set_xlabel("acoustic (MFCC) frame")
    axes[0].set_ylabel("deep (wav2vec) frame")

    sns.heatmap(acoustic_over_deep[0].cpu().numpy(), cmap="magma", ax=axes[1], cbar=True)
    axes[1].set_title("MFCC frames attending over wav2vec frames")
    axes[1].set_xlabel("deep (wav2vec) frame")
    axes[1].set_ylabel("acoustic (MFCC) frame")

    fig.suptitle(f"{run_name}, fold {fold_id} — cross-attention — {item['filename']}",
                fontsize=13)
    fig.tight_layout()
    return _finish(fig, f"attention_map_{run_name}_{item['filename']}.png", show,
                   subdir=config.EXPLAINABILITY_FIGURE_DIR)


# ---------------------------------------------------------------------------
# SHAP feature importance (Praat feature space)
# ---------------------------------------------------------------------------
def _shap_surrogate_path(run_name: Optional[str], task: str) -> Path:
    """Where a SHAP surrogate for this (run, task) combination is persisted —
    one fixed convention so compute_shap_values() and anything that wants to
    reload a surrogate later agree on the path without passing it around."""
    return config.CHECKPOINT_DIR / "shap_surrogates" / f"{run_name or 'ground_truth'}_{task}.pkl"


def compute_shap_values(features_df: pd.DataFrame, task: str = "detection",
                        feature_columns=FEATURE_COLUMNS, sample_size: int = 200,
                        seed: int = 42, run_name: Optional[str] = None,
                        predictions_dir: Optional[Path] = None, save: bool = True):
    """
    SHAP feature-importance over the Phase 4 Praat feature set, via a
    RandomForestClassifier surrogate fit on those same clinically-named
    features — not SHAP's model-agnostic KernelExplainer run through the
    wav2vec 2.0 forward pass, which would be both slow and would explain a
    768-dim latent space no reader can interpret feature-by-feature. This
    gives a direct, publication-usable "which acoustic measurement drove the
    label" ranking instead.

    run_name=None (default): the surrogate is fit against the *ground-truth*
    label — "which Praat features separate Healthy from Dysarthric in
    reality". Pass a trained run's name (e.g. "detection_attention_fusion")
    to fit the surrogate against *that model's own predictions* instead —
    "which Praat features this specific model's decisions correlate with".
    That second framing is what makes a Model D vs. E vs. F SHAP comparison
    meaningful: fitting every model against the same ground-truth label
    would produce the same ranking for all three regardless of what each
    architecture actually attends to. See compare_shap_across_models() for
    running all three ablation variants this way in one call.

    task="severity" drops control speakers (Severity == "N/A (Control)")
    since severity labels apply only to the 15 dysarthric speakers.

    save=True (default) persists the fitted surrogate via
    src.training.checkpoint.save_sklearn_model to
    outputs/checkpoints/shap_surrogates/<run_name or 'ground_truth'>_<task>.pkl
    — refitting a RandomForest just to re-plot is otherwise the only option,
    since nothing else in this module keeps the surrogate around.

    Returns (explanation, X_sample, feature_columns, class_names, surrogate).
    `explanation` is a shap.Explanation (supports the modern
    shap.plots.beeswarm/waterfall API directly, not just a bare array):
    for "detection" it is already sliced to the positive class
    ("Dysarthric Patient"); for "severity" it is a list of per-class
    Explanations, in the order of `class_names`.
    """
    import shap
    from sklearn.ensemble import RandomForestClassifier

    from src.error_analysis import load_run_predictions
    from src.training.checkpoint import save_sklearn_model

    feature_columns = list(feature_columns)
    if task == "severity":
        label_map = {k: v for k, v in config.SEVERITY_LABEL_MAP.items() if v >= 0}
        df = features_df[features_df["Severity"].isin(label_map)]
        class_names = config.SEVERITY_CLASS_NAMES
    else:
        label_map = config.GROUP_LABEL_MAP
        df = features_df
        class_names = config.DETECTION_CLASS_NAMES

    if run_name is None:
        label_column = "Severity" if task == "severity" else "Group"
        df = df.dropna(subset=feature_columns)
        y = df[label_column].map(label_map).to_numpy()
    else:
        # Fit against this run's own predicted label, joined by filename —
        # every model saw the same folds, so a like-for-like SHAP comparison
        # across models only needs the y column to change.
        preds = load_run_predictions(run_name, predictions_dir=predictions_dir)
        df = df.merge(preds[["filename", "y_pred"]], left_on="Filename", right_on="filename",
                      how="inner")
        df = df.dropna(subset=feature_columns)
        y = df["y_pred"].to_numpy()

    X = df[feature_columns].to_numpy()

    surrogate = RandomForestClassifier(n_estimators=300, random_state=seed,
                                       class_weight="balanced")
    surrogate.fit(X, y)
    if save:
        surrogate_path = _shap_surrogate_path(run_name, task)
        save_sklearn_model(surrogate_path, surrogate)
        print_kv("SHAP surrogate saved", surrogate_path)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X[idx]

    # explainer(X) (not the older .shap_values(X)) returns a shap.Explanation
    # — the current shap API, and the only form shap.plots.beeswarm/waterfall
    # accept. It also sidesteps the multiclass return-shape churn between
    # shap versions (list of per-class arrays vs. one (n, features, classes)
    # ndarray) since Explanation slicing (`explanation[..., c]`) is stable
    # either way.
    explainer = shap.TreeExplainer(surrogate)
    explanation = explainer(X_sample)
    explanation.feature_names = feature_columns

    if task == "detection":
        result = explanation[..., 1]                                  # positive class
    else:
        result = [explanation[..., c] for c in range(len(class_names))]

    return result, X_sample, feature_columns, class_names, surrogate


def _shap_values_array(explanation) -> np.ndarray:
    """explanation.values if given a shap.Explanation, else pass through —
    lets every plot function below accept either the Explanation objects
    compute_shap_values() now returns or a bare ndarray."""
    return explanation.values if hasattr(explanation, "values") else np.asarray(explanation)


def plot_shap_summary(explanation, feature_columns, title: str = "SHAP Feature Importance",
                      color: str = "#4c72b0", show: bool = False,
                      filename: str = "shap_feature_importance.png") -> str:
    """
    Bar chart of mean |SHAP value| per feature — the global "which feature
    matters most, on average" view. Read directly off the values
    compute_shap_values() already produced.
    """
    shap_values = _shap_values_array(explanation)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    ranked_features = [feature_columns[i] for i in order]
    ranked_values = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(ranked_features))))
    ax.barh(ranked_features[::-1], ranked_values[::-1], color=color)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return _finish(fig, filename, show, subdir=config.EXPLAINABILITY_FIGURE_DIR)


def plot_shap_beeswarm(explanation, title: str = "SHAP Summary",
                       show: bool = False,
                       filename: str = "shap_beeswarm.png") -> str:
    """
    The standard global SHAP summary plot: one dot per utterance per
    feature, coloured by that utterance's feature value, positioned by its
    SHAP contribution. Where plot_shap_summary()'s bar chart only shows
    *how much* a feature matters on average, the beeswarm also shows
    *which direction* (does high jitter push toward "Dysarthric", or away
    from it?) and how consistent that direction is across utterances —
    the two views answer different questions and are meant to be read
    together, not as alternatives.

    Requires `explanation` to be a shap.Explanation (what compute_shap_values
    now returns) with feature_names already set, not a bare array.
    """
    import shap

    shap.plots.beeswarm(explanation, show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_size_inches(9, max(4, 0.35 * len(explanation.feature_names)))
    fig.axes[0].set_title(title, fontsize=14)
    fig.tight_layout()
    return _finish(fig, filename, show, subdir=config.EXPLAINABILITY_FIGURE_DIR)


def plot_shap_waterfall(explanation, index: int = 0,
                        title: Optional[str] = None, show: bool = False,
                        filename: str = "shap_waterfall.png") -> str:
    """
    Local (single-utterance) SHAP explanation: how each Praat feature pushed
    THIS ONE prediction away from the surrogate's base rate. The
    global bar/beeswarm plots answer "what does the model rely on in
    general" — a clinician asking "why did it flag THIS speaker" needs this
    instead, which is the per-instance explanation SHAP was built for and
    the global-only summary above cannot give.

    Requires `explanation` to be a shap.Explanation (what compute_shap_values
    now returns); `index` selects which sampled utterance to explain.
    """
    import shap

    shap.plots.waterfall(explanation[index], show=False)
    fig = plt.gcf()
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _finish(fig, filename, show, subdir=config.EXPLAINABILITY_FIGURE_DIR)


# ---------------------------------------------------------------------------
# SHAP comparison across the three fusion strategies (Models D / E / F)
# ---------------------------------------------------------------------------
def compare_shap_across_models(features_df: pd.DataFrame,
                               run_names: Dict[str, str],
                               task: str = "detection",
                               feature_columns=FEATURE_COLUMNS,
                               sample_size: int = 200, seed: int = 42,
                               top_k: int = 12,
                               predictions_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Model D (concatenation), E (cross-attention), F (cross-attention + Praat)
    differ only in how the two learned pathways combine — that is the whole
    ablation argument (see ROADMAP Phase 6). A single generic SHAP ranking
    against ground truth cannot show that difference, because it explains
    "what separates the classes", not "what each model leaned on" — the same
    number for all three regardless of architecture.

    This runs compute_shap_values(..., run_name=...) once per model against
    its *own* predictions and returns a tidy DataFrame of mean|SHAP| per
    feature per model, so a claim like "attention fusion leans more on CPPS
    and less on jitter than concatenation does" is a number, not an
    impression.

    Args:
        run_names: {model_label: trained_run_name}, e.g.
            {"fusion": "detection_fusion",
             "attention_fusion": "detection_attention_fusion",
             "attention_fusion_praat": "detection_attention_fusion_praat"}
            All three must already be trained (see notebooks/03_training.ipynb).
    Returns:
        DataFrame with columns [model, feature, mean_abs_shap], long-form —
        ready for plot_shap_comparison() or a pivot table.
    """
    records = []
    for model_label, run_name in run_names.items():
        explanation, _, cols, _, _ = compute_shap_values(
            features_df, task=task, feature_columns=feature_columns,
            sample_size=sample_size, seed=seed, run_name=run_name,
            predictions_dir=predictions_dir)
        mean_abs = np.abs(_shap_values_array(explanation)).mean(axis=0)
        for feature, value in zip(cols, mean_abs):
            records.append({"model": model_label, "feature": feature,
                            "mean_abs_shap": float(value)})
        print_kv(f"SHAP ({model_label})", f"fit against '{run_name}' predictions")

    result = pd.DataFrame.from_records(records)

    # Rank features by their max importance across models, so the comparison
    # plot highlights whichever features matter to *any* of the three,
    # rather than truncating to whatever model D happens to lean on.
    top_features = (result.groupby("feature")["mean_abs_shap"].max()
                    .sort_values(ascending=False).head(top_k).index)
    return result[result["feature"].isin(top_features)].reset_index(drop=True)


def plot_shap_comparison(comparison: pd.DataFrame,
                         title: str = "SHAP feature importance — fusion strategy comparison",
                         show: bool = False) -> str:
    """
    Grouped horizontal bar chart from compare_shap_across_models()'s output:
    one color per fusion model (src.style.MODEL_COLORS), one row per feature.
    Directly answers "does attention fusion (E) or the Praat-augmented model
    (F) rely on different acoustic evidence than plain concatenation (D)?".
    """
    order = (comparison.groupby("feature")["mean_abs_shap"].max()
            .sort_values(ascending=False).index.tolist())
    models = list(comparison["model"].unique())

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(order))))
    sns.barplot(data=comparison, y="feature", x="mean_abs_shap", hue="model",
               order=order, hue_order=models,
               palette=[model_color(m) for m in models], ax=ax)
    ax.set_xlabel("mean |SHAP value| (surrogate fit on the model's own predictions)")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=14)
    ax.legend(title="", loc="lower right", fontsize=9)
    fig.tight_layout()
    return _finish(fig, "shap_comparison_fusion_models.png", show, subdir=config.EXPLAINABILITY_FIGURE_DIR)


# ---------------------------------------------------------------------------
# Branch ablation (three-branch severity architecture only) — inference-time
# contribution analysis on the ONE trained checkpoint, not a retrained model
# family. See src.models.gated_fusion.GatedFusionModel.ablate() and the
# architecture plan's Part 2, Component 15 for why this replaces training
# separate branch-dropped models: dropping a branch changes what the FROZEN
# gate/classifier see, which is exactly "how much does this model rely on
# this branch", without spending any of the one-shot training budget.
# ---------------------------------------------------------------------------
# Plain ASCII hyphens (not a Unicode minus sign) deliberately — these labels
# are printed to Windows consoles/notebooks whose default codepage (cp1252)
# cannot encode U+2212 and raises UnicodeEncodeError on print().
BRANCH_ABLATION_VARIANTS = {"Full": None, "- Learned": "learned",
                            "- Segmental": "segmental", "- Suprasegmental": "supra"}


def evaluate_branch_ablation(model, loader, device, task: str = "severity") -> pd.DataFrame:
    """
    Runs GatedFusionModel.ablate() over one fold's test loader for each of
    {full, -learned, -segmental, -suprasegmental} and reports the resulting
    metrics plus the drop relative to the full model. Call this ONLY.
    """
    from src.training.metrics import compute_metrics

    model.eval()
    records = []
    with torch.no_grad():
        for label, drop_branch in progress(BRANCH_ABLATION_VARIANTS.items(),
                                           "Branch ablation", total=len(BRANCH_ABLATION_VARIANTS)):
            all_true, all_pred, all_prob = [], [], []
            for batch in loader:
                waveform = batch["waveform"].squeeze(1).to(device)
                mfcc = batch["mfcc"].to(device)
                supra = batch["supra"].to(device)
                waveform_length = batch["waveform_length"].to(device)
                attention_mask = (torch.arange(waveform.shape[1], device=device)[None, :]
                                  < waveform_length[:, None])
                supra_valid_frames = batch["supra_valid_frames"].to(device)
                labels = batch["severity_label"].to(device)

                logits = model.ablate(waveform=waveform, mfcc=mfcc, attention_mask=attention_mask,
                                      supra=supra, supra_valid_frames=supra_valid_frames,
                                      drop_branch=drop_branch)
                probs = torch.softmax(logits.float(), dim=1)
                all_true.append(labels.cpu().numpy())
                all_pred.append(probs.argmax(dim=1).cpu().numpy())
                all_prob.append(probs.cpu().numpy())

            y_true, y_pred, y_prob = np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_prob)
            m = compute_metrics(y_true, y_pred, y_prob, task)
            records.append({"variant": label, "macro_f1": m["f1"],
                            "balanced_accuracy": m["balanced_accuracy"],
                            "ordinal_mae": m["ordinal_mae"], "accuracy": m["accuracy"]})

    df = pd.DataFrame(records).set_index("variant")
    full = df.loc["Full"]
    df["macro_f1_drop"] = full["macro_f1"] - df["macro_f1"]
    df["balanced_accuracy_drop"] = full["balanced_accuracy"] - df["balanced_accuracy"]
    df["ordinal_mae_increase"] = df["ordinal_mae"] - full["ordinal_mae"]
    return df


def plot_branch_ablation(ablation_df: pd.DataFrame, metric: str = "macro_f1",
                         title: Optional[str] = None, show: bool = False) -> str:
    """Grouped bar chart of evaluate_branch_ablation()'s output for one
    metric — Full vs. each branch removed."""
    colors = {"Full": "#333333", "- Learned": branch_color("learned"),
             "- Segmental": branch_color("segmental"),
             "- Suprasegmental": branch_color("supra")}
    variants = list(ablation_df.index)
    values = ablation_df[metric].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(variants, values, color=[colors.get(v, "#4c72b0") for v in variants])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"Branch ablation — {metric.replace('_', ' ')}", fontsize=13)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _finish(fig, f"branch_ablation_{metric}.png", show, subdir=config.ABLATION_FIGURE_DIR)


# ---------------------------------------------------------------------------
# Gate analysis (three-branch severity architecture only). Gate weights are
# saved per-utterance alongside branch embeddings (see
# src.training.reporting.save_embeddings) — this section aggregates and
# visualizes them. A gate value reflects LEARNED reliance, not causal
# importance — see the architecture plan's Part 2, Component 9 and the
# limitations table (src.results.build_limitations_table).
# ---------------------------------------------------------------------------
def load_run_gate_weights(run_name: str) -> pd.DataFrame:
    """Every test-fold's per-utterance gate weights for a run, as a
    DataFrame with filename/y_true/gate_learned/gate_segmental/gate_supra."""
    run_dir = config.EMBEDDINGS_DIR / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"No embeddings for run '{run_name}' at {run_dir}.")

    frames = []
    for path in sorted(run_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if "gate_weights" not in data:
            raise ValueError(f"{path} has no 'gate_weights' array — gate analysis requires "
                             "a run trained with GatedFusionModel.")
        gates = data["gate_weights"]
        frames.append(pd.DataFrame({
            "filename": data["filenames"], "y_true": data["y_true"],
            "gate_learned": gates[:, 0], "gate_segmental": gates[:, 1], "gate_supra": gates[:, 2],
        }))
    return pd.concat(frames, ignore_index=True)


GATE_COLUMNS = ["gate_learned", "gate_segmental", "gate_supra"]
GATE_LABELS = ["Learned", "Segmental", "Suprasegmental"]


def summarize_gate_values(gate_df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/std gate weight per branch, over every test utterance."""
    summary = gate_df[GATE_COLUMNS].agg(["mean", "median", "std"]).T
    summary.index = GATE_LABELS
    summary.columns = ["mean", "median", "std"]
    return summary


def summarize_gate_values_by_severity(gate_df: pd.DataFrame,
                                      y_true_label: pd.Series) -> pd.DataFrame:
    """Mean gate weight per branch, per severity class — investigates
    whether gate behavior shifts with severity (e.g. the model leaning more
    on the suprasegmental branch for more severe speakers)."""
    df = gate_df[GATE_COLUMNS].copy()
    df["severity"] = y_true_label.to_numpy()
    grouped = df.groupby("severity")[GATE_COLUMNS].mean()
    grouped.columns = GATE_LABELS
    order = [c for c in config.SEVERITY_CLASS_NAMES if c in grouped.index]
    return grouped.reindex(order)


def plot_gate_distribution(gate_df: pd.DataFrame, show: bool = False) -> str:
    """Bar (mean) + violin (full distribution) of gate weights per branch."""
    colors = [branch_color(b) for b in ("learned", "segmental", "supra")]
    means = gate_df[GATE_COLUMNS].mean().to_numpy()
    long = gate_df[GATE_COLUMNS].melt(var_name="branch", value_name="gate")
    long["branch"] = long["branch"].map(dict(zip(GATE_COLUMNS, GATE_LABELS)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(GATE_LABELS, means, color=colors)
    axes[0].set_ylabel("mean gate weight")
    axes[0].set_title("Mean gate contribution")
    sns.violinplot(data=long, x="branch", y="gate", order=GATE_LABELS,
                   palette=colors, ax=axes[1])
    axes[1].set_title("Gate weight distribution")
    axes[1].set_xlabel("")
    fig.tight_layout()
    return _finish(fig, "gate_distribution.png", show, subdir=config.ABLATION_FIGURE_DIR)


def plot_gate_by_severity(gate_df: pd.DataFrame, y_true_label: pd.Series, show: bool = False) -> str:
    """Grouped bar chart of mean gate weight per branch, per severity class."""
    df = gate_df[GATE_COLUMNS].copy()
    df["severity"] = y_true_label.to_numpy()
    long = df.melt(id_vars="severity", value_vars=GATE_COLUMNS, var_name="branch", value_name="gate")
    long["branch"] = long["branch"].map(dict(zip(GATE_COLUMNS, GATE_LABELS)))
    order = [c for c in config.SEVERITY_CLASS_NAMES if c in df["severity"].unique()]
    colors = {label: branch_color(key) for key, label in zip(("learned", "segmental", "supra"), GATE_LABELS)}

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=long, x="severity", y="gate", hue="branch", order=order,
               hue_order=GATE_LABELS, palette=colors, ax=ax)
    ax.set_ylabel("mean gate weight")
    ax.set_xlabel("")
    ax.set_title("Gate contribution by severity class", fontsize=13)
    ax.legend(title="", loc="upper right", fontsize=9)
    fig.tight_layout()
    return _finish(fig, "gate_by_severity.png", show, subdir=config.ABLATION_FIGURE_DIR)


# ---------------------------------------------------------------------------
# Cross-branch complementarity diagnostic (numerical + heatmap). Reuses
# src.losses.redundancy_penalty — the SAME function/normalization the
# training loss uses — so this is not a second, inconsistent metric.
# ---------------------------------------------------------------------------
def plot_complementarity_heatmap(z_learned: np.ndarray, z_segmental: np.ndarray,
                                 z_supra: np.ndarray, show: bool = False
                                 ) -> Tuple[str, pd.DataFrame]:
    """
    3x3 branch-pair redundancy-penalty matrix (mean squared cross-
    correlation, config.LAMBDA_COMP's own units — see src.losses.
    redundancy_penalty), both as a heatmap and a small numeric table. Lower
    values support the "branches are not duplicating each other" claim;
    this is descriptive, not a hypothesis test.
    """
    from src.losses import redundancy_penalty

    branches = {"Learned": z_learned, "Segmental": z_segmental, "Suprasegmental": z_supra}
    names = list(branches)
    matrix = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            za = torch.as_tensor(branches[a], dtype=torch.float32)
            zb = torch.as_tensor(branches[b], dtype=torch.float32)
            matrix[i, j] = redundancy_penalty(za, zb).item()

    table = pd.DataFrame(matrix, index=names, columns=names)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(table, annot=True, fmt=".4f", cmap="magma", ax=ax,
               cbar_kws={"label": "mean squared cross-correlation"})
    ax.set_title("Cross-branch complementarity (redundancy) — lower is more complementary",
                fontsize=12)
    fig.tight_layout()
    path = _finish(fig, "complementarity_heatmap.png", show, subdir=config.ABLATION_FIGURE_DIR)
    return path, table


# ---------------------------------------------------------------------------
# Feature-group SHAP, per-class SHAP, and permutation importance — extends
# the existing RandomForest-surrogate SHAP methodology (compute_shap_values
# above) rather than replacing it. src.praat.FEATURE_GROUPS assigns every
# FEATURE_COLUMNS entry to exactly one of Spectral/Formant/Voice-Quality/
# Pitch/Energy/Temporal-Voicing (see the architecture plan's Part 2,
# Component 15 / brief Section 18).
# ---------------------------------------------------------------------------
def shap_summary_table(explanation, feature_columns) -> pd.DataFrame:
    """{feature, mean_abs_shap} table — the numeric form of plot_shap_summary,
    reused by both feature-group aggregation and the SHAP/permutation
    ranking comparison below."""
    shap_values = _shap_values_array(explanation)
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({"feature": list(feature_columns), "mean_abs_shap": mean_abs})


def aggregate_shap_by_group(explanation, feature_columns) -> pd.DataFrame:
    """Feature-level SHAP -> feature-GROUP SHAP, by summing mean|SHAP|
    within each src.praat.FEATURE_GROUPS group. Group-level importance is
    the more defensible research claim when many individual features are
    correlated (e.g. jitter_local/jitter_rap/jitter_ppq5/jitter_ddp all
    measuring the same underlying phenomenon)."""
    table = shap_summary_table(explanation, feature_columns)
    table["group"] = table["feature"].map(FEATURE_GROUPS)
    return (table.groupby("group")["mean_abs_shap"].sum()
           .sort_values(ascending=False).reset_index())


def plot_shap_group_importance(group_df: pd.DataFrame,
                               title: str = "SHAP Feature-Group Importance",
                               show: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(group_df["group"][::-1], group_df["mean_abs_shap"][::-1], color="#4c72b0")
    ax.set_xlabel("summed mean |SHAP value| within group")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return _finish(fig, "shap_group_importance.png", show, subdir=config.EXPLAINABILITY_FIGURE_DIR)


def plot_shap_per_class(explanations: List, feature_columns, class_names,
                        top_k: int = 12, show: bool = False) -> str:
    """Per-class SHAP bar charts, one panel per severity class — the
    per-class Explanation list compute_shap_values() already returns for
    task='severity' but this project's plotting never visualized before."""
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 6), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, explanation, name in zip(axes, explanations, class_names):
        table = shap_summary_table(explanation, feature_columns).sort_values(
            "mean_abs_shap", ascending=False).head(top_k)
        ax.barh(table["feature"][::-1], table["mean_abs_shap"][::-1], color="#4c72b0")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("mean |SHAP|")
    fig.suptitle("Per-class SHAP feature importance", fontsize=14)
    fig.tight_layout()
    return _finish(fig, "shap_per_class.png", show, subdir=config.EXPLAINABILITY_FIGURE_DIR)


def compute_permutation_importance(surrogate, X: np.ndarray, y: np.ndarray, feature_columns,
                                   seed: int = 42, n_repeats: int = 20) -> pd.DataFrame:
    """
    Permutation importance on the SAME RandomForest surrogate SHAP already
    fits (compute_shap_values) — an independent cross-check against the
    SHAP ranking, not a replacement for it (architecture plan Part 2,
    Component 15 / brief Section 30). scoring is macro-F1 for >2 classes
    (severity) or accuracy for binary (detection), matching each task's
    primary reported metric.
    """
    from sklearn.inspection import permutation_importance

    scoring = "f1_macro" if len(np.unique(y)) > 2 else "accuracy"
    result = permutation_importance(surrogate, X, y, n_repeats=n_repeats,
                                    random_state=seed, scoring=scoring)
    return pd.DataFrame({
        "feature": list(feature_columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def aggregate_permutation_importance_by_group(perm_df: pd.DataFrame) -> pd.DataFrame:
    df = perm_df.copy()
    df["group"] = df["feature"].map(FEATURE_GROUPS)
    return (df.groupby("group")["importance_mean"].sum()
           .sort_values(ascending=False).reset_index())


def plot_permutation_importance(perm_df: pd.DataFrame, top_k: int = 15,
                                title: str = "Permutation Importance", show: bool = False) -> str:
    top = perm_df.head(top_k)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top))))
    ax.barh(top["feature"][::-1], top["importance_mean"][::-1],
           xerr=top["importance_std"][::-1], color="#dd8452")
    ax.set_xlabel("mean decrease in score when the feature is permuted")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return _finish(fig, "permutation_importance.png", show, subdir=config.EXPLAINABILITY_FIGURE_DIR)


def compare_shap_and_permutation_rankings(shap_table: pd.DataFrame, perm_df: pd.DataFrame,
                                          top_k: int = 10) -> Tuple[pd.DataFrame, float, float]:
    """
    Rank-correlation comparison between the SHAP and permutation-importance
    feature rankings. Returns (top-k side-by-side rank table, Spearman rho,
    p-value). Agreement between the two methods is NOT guaranteed and is
    not claimed here — report whatever the numbers show (architecture plan
    Part 2, Component 15 / brief Section 30).
    """
    from scipy.stats import spearmanr

    shap_rank = shap_table.set_index("feature")["mean_abs_shap"].rank(ascending=False)
    perm_rank = perm_df.set_index("feature")["importance_mean"].rank(ascending=False)
    combined = pd.DataFrame({"shap_rank": shap_rank, "permutation_rank": perm_rank}).dropna()
    corr, p_value = spearmanr(combined["shap_rank"], combined["permutation_rank"])
    top = combined.sort_values("shap_rank").head(top_k)
    return top, float(corr), float(p_value)
