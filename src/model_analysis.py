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
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src import config
from src.console import print_kv
from src.praat import FEATURE_COLUMNS
from src.style import apply_style, model_color

apply_style()


def _finish(fig, filename: str, show: bool) -> str:
    """Save a figure to outputs/figures/, optionally display it, then release it."""
    out_path = config.FIGURE_DIR / filename
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
    return _finish(fig, "ablation_comparison.png", show)


# ---------------------------------------------------------------------------
# Embedding-space visualization (t-SNE / PCA / UMAP)
# ---------------------------------------------------------------------------
def load_run_embeddings(run_name: str) -> pd.DataFrame:
    """Every test-fold embedding for a run, as a DataFrame with a `filename` key
    and an `embedding` column of vectors."""
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
        frames.append(pd.DataFrame({
            "filename": data["filenames"],
            "y_true": data["y_true"],
            "embedding": list(data["embeddings"]),
        }))
    return pd.concat(frames, ignore_index=True)


def plot_embedding_map(run_name: str, preds: pd.DataFrame, task: str = "detection",
                       method: str = "tsne", max_points: int = 3000, seed: int = 42,
                       show: bool = False) -> str:
    """
    2D projection of the learned test-fold embeddings, coloured by true class,
    with misclassified points marked.

    method="tsne" (default): local structure, the same view Phase 5's error
    analysis uses to ask whether errors cluster.
    method="pca": linear, deterministic — a sanity check the nonlinear
    methods' layout can be compared against.
    method="umap": nonlinear like t-SNE but tends to preserve more global
    structure between clusters; needs the optional `umap-learn` package.
    """
    embeddings = load_run_embeddings(run_name)
    merged = embeddings.merge(preds[["filename", "correct", "y_true_label"]],
                              on="filename", how="inner")
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

    correct = merged[merged["correct"].astype(bool)]
    wrong = merged[~merged["correct"].astype(bool)]

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.scatterplot(data=correct, x="x", y="y", hue="y_true_label", palette="viridis",
                    s=14, alpha=0.45, linewidth=0, ax=ax)
    ax.scatter(wrong["x"], wrong["y"], marker="x", s=42, c="#c44e52",
               linewidths=1.2, label=f"misclassified (n={len(wrong)})")

    ax.set_title(f"{run_name} — {method.upper()} of test embeddings ({task})", fontsize=14)
    ax.set_xlabel(f"{method}-1")
    ax.set_ylabel(f"{method}-2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    return _finish(fig, f"embedding_map_{method}_{run_name}.png", show)


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
    return _finish(fig, f"attention_map_{run_name}_{item['filename']}.png", show)


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
    return _finish(fig, filename, show)


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
    return _finish(fig, filename, show)


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
    return _finish(fig, filename, show)


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
    return _finish(fig, "shap_comparison_fusion_models.png", show)
