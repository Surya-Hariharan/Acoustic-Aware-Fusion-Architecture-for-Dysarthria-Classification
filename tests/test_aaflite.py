"""
AAF-Lite protocol guarantees:

  * the control reference only ever sees control speakers, and removes the
    word identity (same word, same speaker offset -> same z-vector);
  * in every outer fold the held-out speaker is absent from fitting, from
    inner model selection and from fusion-weight selection;
  * the protocol is deterministic, beats chance on separable synthetic data,
    and falls to chance when labels are permuted across speakers.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.aaflite import pipeline
from src.aaflite.embeddings import layer_group_features
from src.aaflite.pipeline import Branch, evaluate, fusion_weights, permuted_labels
from src.aaflite.reference import ControlReference, prompt_keys

N_WORDS = 12


def _manifest(speakers, n_words=N_WORDS):
    rows = []
    for speaker in speakers:
        for w in range(n_words):
            rows.append({"Speaker_ID": speaker, "Block": "B1", "WordCode": f"UW{w}",
                         "Filename": f"{speaker}_B1_UW{w}_M6.wav",
                         "Severity": config.SEVERITY_MAP.get(speaker, "N/A (Control)")})
    return pd.DataFrame(rows)


def _synthetic(signal: float, seed: int = 0):
    """Dysarthric manifest + 2 branches whose first dimension tracks severity."""
    rng = np.random.default_rng(seed)
    df = _manifest(config.DYSARTHRIC_IDS)
    y = df["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy()
    speaker_offset = {s: rng.normal(size=6) for s in config.DYSARTHRIC_IDS}
    base = np.stack([speaker_offset[s] for s in df["Speaker_ID"]]) + rng.normal(size=(len(df), 6))
    a = base.copy()
    a[:, 0] += signal * y
    b = rng.normal(size=(len(df), 4))
    b[:, 0] += 0.5 * signal * y
    return df, a, b


def _branches(a, b):
    return [Branch("a", {"raw": a}, pca_dims=(None, 3), c_grid=(0.1, 1.0)),
            Branch("b", {"raw": b}, c_grid=(1.0,))]


# ---------------------------------------------------------------------------
# Control reference
# ---------------------------------------------------------------------------
def test_reference_rejects_non_control_speakers():
    df = _manifest(["CF02", "M01"])
    with pytest.raises(ValueError, match="non-control"):
        ControlReference().fit(df, np.zeros((len(df), 3)))


def test_reference_removes_word_identity():
    controls = _manifest(config.CONTROL_IDS[:4])
    rng = np.random.default_rng(0)
    word_effect = {k: rng.normal(size=5) * 10 for k in np.unique(prompt_keys(controls))}
    X_controls = (np.stack([word_effect[k] for k in prompt_keys(controls)])
                  + rng.normal(scale=0.1, size=(len(controls), 5)))
    reference = ControlReference().fit(controls, X_controls)

    patient = _manifest(["M01"])
    X_patient = np.stack([word_effect[k] for k in prompt_keys(patient)]) + 3.0   # constant offset
    z = reference.transform(patient, X_patient)
    assert z.shape == (len(patient), 6)                         # + distance-from-healthy
    # Word identity removed: the 10-unit word effects no longer vary across
    # words; what remains is the constant offset, in control-spread units.
    assert z[:, :5].std(axis=0).max() < 0.05 * np.abs(z[:, :5]).mean()
    assert np.all(z[:, -1] > 5)                                 # far from healthy


def test_layer_group_features_average_requested_layers():
    stats = np.zeros((2, 13, 2, 768), dtype=np.float16)
    stats[:, 1] = 1.0
    stats[:, 2] = 3.0
    out = layer_group_features(stats, (1, 2))
    assert out.shape == (2, 1536) and np.allclose(out, 2.0)


# ---------------------------------------------------------------------------
# Nested LOSO
# ---------------------------------------------------------------------------
def test_held_out_speaker_never_reaches_any_fit(monkeypatch):
    df, a, b = _synthetic(signal=3.0)
    marker = {s: i for i, s in enumerate(config.DYSARTHRIC_IDS)}
    ids = df["Speaker_ID"].map(marker).to_numpy().astype(float)
    a = np.hstack([a, ids[:, None]])                            # last column = speaker id
    seen = []
    original = pipeline._fit_predict

    def spy(X_train, y_train, X_test, pca_dims, c_grid):
        seen.append(set(np.unique(X_train[:, -1]).astype(int)))
        return original(X_train, y_train, X_test, pca_dims, c_grid)

    monkeypatch.setattr(pipeline, "_fit_predict", spy)
    for fold_id in ("M01", "F05"):
        seen.clear()
        df_reset = df.reset_index(drop=True)
        test = (df_reset["Speaker_ID"] == fold_id).to_numpy()
        pipeline.run_outer_fold(fold_id, np.flatnonzero(~test), np.flatnonzero(test),
                                [Branch("a", {"raw": a}, c_grid=(1.0,))],
                                df_reset["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy(),
                                df_reset["Speaker_ID"].to_numpy())
        assert seen and all(marker[fold_id] not in s for s in seen)


def test_protocol_is_deterministic_and_beats_chance():
    df, a, b = _synthetic(signal=3.0)
    subsets = {"fusion": ("a", "b"), "a_only": ("a",)}
    first = evaluate(df, _branches(a, b), subsets, "test", n_jobs=1, save=False)
    second = evaluate(df, _branches(a, b), subsets, "test", n_jobs=1, save=False)
    assert np.array_equal(first["fusion"]["predictions"]["y_pred"],
                          second["fusion"]["predictions"]["y_pred"])
    assert first["fusion"]["pooled"]["accuracy"] > 0.6
    assert first["fusion"]["pooled"]["completed_folds"] == 15
    for row in first["fusion"]["folds"]:
        assert abs(sum(row["weights"].values()) - 1.0) < 1e-9


def test_permuted_labels_fall_to_chance():
    df, a, b = _synthetic(signal=3.0)
    y = permuted_labels(df)
    assert sorted(np.bincount(y)) == sorted(np.bincount(df["Severity"].map(config.SEVERITY_LABEL_MAP)))
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(len(df), 6))
    out = evaluate(df, [Branch("a", {"raw": noise}, c_grid=(1.0,))], {"a": ("a",)}, "test",
                   y=y, n_jobs=1, save=False)
    assert out["a"]["pooled"]["accuracy"] < 0.5


def test_fusion_weights_prefer_the_informative_branch():
    y = np.repeat(np.arange(4), 25)
    good = np.eye(4)[y] * 0.9 + 0.025
    bad = np.full((len(y), 4), 0.25)
    weights = fusion_weights(("good", "bad"), {"good": good, "bad": bad}, y)
    assert weights["good"] >= weights["bad"]


def test_expected_rank_decoder_predicts_the_middle_of_split_evidence():
    split = np.array([[0.45, 0.05, 0.05, 0.45]])            # torn between the extremes
    assert pipeline.decode(split, "argmax")[0] in (0, 3)
    assert pipeline.decode(split, "expected_rank")[0] in (1, 2)
    confident = np.array([[0.0, 0.0, 0.1, 0.9]])
    assert pipeline.decode(confident, "expected_rank")[0] == 3
    y = np.array([1, 2, 1, 2])
    oof = np.array([[0.5, 0.0, 0.0, 0.5], [0.4, 0.0, 0.1, 0.5]] * 2)
    assert pipeline.choose_decoder(y, oof) == "expected_rank"
