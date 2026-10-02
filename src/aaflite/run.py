"""
End-to-end AAF-Lite driver used by notebooks/final_model.ipynb:

    prepare_features()  feature store (all 28 speakers) -> wav2vec2 layer stats
                        -> acoustic functionals -> control-referenced branches
    run_experiments()   nested LOSO: final model, every branch subset, the
                        un-referenced variant, the majority baseline and the
                        permuted-label sanity check
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src import config, feature_store
from src.aaflite.embeddings import (extract_wav2vec2_layer_stats, layer_group_features,
                                    load_wav2vec2_layer_stats)
from src.aaflite.functionals import acoustic_functionals
from src.aaflite.pipeline import (Branch, evaluate, majority_baseline, permuted_labels,
                                  save_summary, summary_table)
from src.aaflite.reference import ControlReference
from src.console import print_header, print_kv, print_status

FEATURE_CACHE = config.EMBEDDINGS_DIR / "aaflite_branch_features.npz"
SUMMARY_PATH = config.RESULTS_DIR / "aaflite_summary.json"
FINAL_TAG = "fusion_ref"
# Bump when the meaning of a cached feature changes (functionals, reference, pooling).
FEATURE_VERSION = 1

# Run tag -> branches. "_ref" = control-referenced features, "_raw" = not.
SUBSETS = {
    "learned": ("learned",), "segmental": ("segmental",), "supra": ("supra",),
    "learned+segmental": ("learned", "segmental"), "learned+supra": ("learned", "supra"),
    "segmental+supra": ("segmental", "supra"),
    "fusion": ("learned", "segmental", "supra"),
}


def _feature_signature() -> str:
    """What the cached branch features depend on besides the utterance list."""
    return json.dumps({"version": FEATURE_VERSION, "wav2vec2": config.WAV2VEC_MODEL_NAME,
                       "layer_groups": {k: list(v) for k, v in config.AAFLITE_LAYER_GROUPS.items()},
                       "store": feature_store.store_signature()}, sort_keys=True, default=str)


def prepare_features(df_full: pd.DataFrame, device=None, use_cache: bool = True
                     ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """(dysarthric df, {"ref"|"raw": {branch/variant key: (N_dys, D)}})."""
    df_full = df_full.reset_index(drop=True)
    df_dys = df_full[df_full["Speaker_ID"].isin(config.DYSARTHRIC_IDS)].reset_index(drop=True)
    if use_cache and FEATURE_CACHE.exists():
        with np.load(FEATURE_CACHE, allow_pickle=False) as data:
            fresh = ("signature" in data.files and str(data["signature"]) == _feature_signature()
                     and list(data["filenames"]) == df_dys["Filename"].tolist())
            if fresh:
                features = {"ref": {}, "raw": {}}
                for key in data.files:
                    if "__" in key:
                        kind, name = key.split("__", 1)
                        features[kind][name] = data[key]
                print_status(f"Branch features loaded from {FEATURE_CACHE}", ok=True)
                return df_dys, features

    print_header("AAF-Lite features")
    coverage = feature_store.feature_store_coverage(df_full)
    if not coverage["complete"]:
        feature_store.build_feature_store(df_full, n_workers=config.FEATURE_STORE_WORKERS)
        if not feature_store.feature_store_coverage(df_full)["complete"]:
            raise RuntimeError("Feature store incomplete — re-run to resume the build.")
    if device is None:
        from src.training.utils import resolve_device
        device = resolve_device(None)
    extract_wav2vec2_layer_stats(df_full, device)

    is_control = df_full["Speaker_ID"].isin(config.CONTROL_IDS).to_numpy()
    is_dys = df_full["Speaker_ID"].isin(config.DYSARTHRIC_IDS).to_numpy()
    df_ctrl = df_full[is_control]

    raw: Dict[str, np.ndarray] = {}
    stats = load_wav2vec2_layer_stats(df_full)
    for group, layers in config.AAFLITE_LAYER_GROUPS.items():
        raw[f"learned/{group}"] = layer_group_features(stats, layers)
    del stats
    raw["segmental/functionals"], raw["supra/functionals"] = acoustic_functionals(df_full)

    features = {"ref": {}, "raw": {}}
    for key, X in raw.items():
        reference = ControlReference().fit(df_ctrl, X[is_control])
        features["ref"][key] = reference.transform(df_dys, X[is_dys])
        features["raw"][key] = X[is_dys].astype(np.float32)
        print_kv(f"Branch {key}", f"{X.shape[1]} dims (+1 distance-from-healthy when referenced)")

    np.savez(FEATURE_CACHE, filenames=df_dys["Filename"].to_numpy(dtype=str),
             signature=np.array(_feature_signature()),
             **{f"{kind}__{key}": X for kind, d in features.items() for key, X in d.items()})
    return df_dys, features


def build_branches(features: Dict[str, np.ndarray]) -> list:
    group = lambda prefix: {k.split("/", 1)[1]: X for k, X in features.items() if k.startswith(prefix)}
    return [Branch("learned", group("learned/"), pca_dims=config.AAFLITE_PCA_DIMS),
            Branch("segmental", group("segmental/"), pca_dims=config.AAFLITE_PCA_DIMS),
            Branch("supra", group("supra/"), pca_dims=(None,))]


def run_experiments(df_dys: pd.DataFrame, features: Dict[str, Dict[str, np.ndarray]],
                    folds: Optional[list] = None, n_jobs: int = config.AAFLITE_N_JOBS,
                    sanity_check: bool = True) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """Every AAF-Lite result; also written to outputs/ and SUMMARY_PATH.
    A fold subset (smoke test) writes under separate run names and summary,
    so it can never overwrite or be mistaken for the full result."""
    outputs: Dict[str, Dict] = {}
    prefix = "aaflite" if folds is None else "aaflite_smoke"
    referenced = evaluate(df_dys, build_branches(features["ref"]), SUBSETS, prefix,
                          folds=folds, n_jobs=n_jobs)
    outputs.update({f"{tag}_ref": out for tag, out in referenced.items()})
    unreferenced = evaluate(df_dys, build_branches(features["raw"]), {"fusion": SUBSETS["fusion"]},
                            f"{prefix}_raw", folds=folds, n_jobs=n_jobs)
    outputs["fusion_raw"] = unreferenced["fusion"]
    if folds is None:
        outputs["majority_baseline"] = majority_baseline(df_dys)
        if sanity_check:
            permuted = evaluate(df_dys, build_branches(features["ref"]), {"fusion": SUBSETS["fusion"]},
                                "aaflite_permuted", y=permuted_labels(df_dys), n_jobs=n_jobs,
                                save=False)
            outputs["permuted_labels_sanity"] = permuted["fusion"]
    summary_path = SUMMARY_PATH if folds is None else SUMMARY_PATH.with_name("aaflite_smoke_summary.json")
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    save_summary(outputs, summary_path)
    return summary_table(outputs), outputs
