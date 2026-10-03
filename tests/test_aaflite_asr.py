"""
Recogniser-based intelligibility features and the attention-mask contract:

  * wav2vec2-base (group norm) never receives an attention mask — passing one
    breaks the checkpoint — while layer-norm checkpoints do;
  * hidden-state pooling and recogniser scoring ignore the zero padding;
  * the prompted-word mapping is speaker-independent and complete;
  * session-level decisions aggregate a speaker's utterances.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import config
from src.aaflite import asr
from src.aaflite.embeddings import layer_stats
from src.aaflite.pipeline import session_level

VOCAB = {"<pad>": 0, "|": 1, "'": 2, **{c: 3 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}}


class FakeCTC(torch.nn.Module):
    """Records the attention_mask it was called with; frames = samples // 320."""

    def __init__(self, norm: str, frames_logits=None, hidden=None):
        super().__init__()
        self.config = SimpleNamespace(feat_extract_norm=norm)
        self.calls = []
        self._logits, self._hidden = frames_logits, hidden

    def _get_feat_extract_output_lengths(self, lengths):
        return lengths // 320

    def forward(self, x, attention_mask=None, output_hidden_states=False):
        self.calls.append(attention_mask)
        batch, frames = x.shape[0], x.shape[1] // 320
        logits = self._logits if self._logits is not None else torch.zeros(batch, frames, len(VOCAB))
        hidden = self._hidden if self._hidden is not None else [torch.zeros(batch, frames, 4)] * 2
        return SimpleNamespace(logits=logits, hidden_states=hidden)


def _tokenizer():
    return SimpleNamespace(get_vocab=lambda: VOCAB, pad_token_id=0)


# ---------------------------------------------------------------------------
# Attention-mask contract
# ---------------------------------------------------------------------------
def test_group_norm_checkpoint_gets_no_attention_mask_and_layer_norm_does():
    waveform, lengths = torch.randn(2, 3200), torch.tensor([3200, 1600])
    base, large = FakeCTC("group"), FakeCTC("layer")
    asr.score_batch(base, _tokenizer(), waveform, lengths, ["AT", "GO"])
    asr.score_batch(large, _tokenizer(), waveform, lengths, ["AT", "GO"])
    assert base.calls == [None]
    assert large.calls[0] is not None and large.calls[0].dtype == torch.bool
    assert asr.uses_attention_mask(large) and not asr.uses_attention_mask(base)


def test_hidden_state_pooling_ignores_padding_and_masks_only_layer_norm_models():
    lengths = torch.tensor([3200, 1600])                       # 10 and 5 real frames
    hidden = []
    for value in (1.0, 2.0):
        h = torch.full((2, 10, 4), value)
        h[1, 5:] = 1e6                                          # padding frames must not leak in
        hidden.append(h)
    for norm in ("group", "layer"):
        model = FakeCTC(norm, hidden=hidden)
        out = layer_stats(model, torch.randn(2, 3200), lengths)
        assert out.shape == (2, 2, 2, 4)                        # (B, states, [mean, std], D)
        assert torch.allclose(out[1, 0, 0], torch.ones(4))      # mean of the 5 real frames
        assert torch.allclose(out[1, 0, 1], torch.zeros(4))     # std of identical frames
        assert (model.calls[0] is None) == (norm == "group")


# ---------------------------------------------------------------------------
# Scoring and text helpers
# ---------------------------------------------------------------------------
def test_perfect_path_scores_exact_with_zero_error():
    frames = 10
    logits = torch.full((1, frames, len(VOCAB)), -20.0)
    path = [VOCAB["A"], VOCAB["A"], 0, VOCAB["T"], VOCAB["T"], 0, 0, 0, 0, 0]
    for t, i in enumerate(path):
        logits[0, t, i] = 20.0
    model = FakeCTC("group", frames_logits=logits)
    scores = asr.score_batch(model, _tokenizer(), torch.randn(1, 3200), torch.tensor([3200]), ["AT"])
    row = dict(zip(asr.SCORE_NAMES, scores[0]))
    assert row["exact"] == 1.0 and row["cer"] == 0.0
    assert row["blank"] == pytest.approx(0.6)
    assert row["nll_char"] < 0.01


def test_all_blank_path_scores_wrong_and_unspellable_prompt_is_nan():
    model = FakeCTC("group", frames_logits=torch.cat([torch.full((2, 10, 1), 20.0),
                                                      torch.full((2, 10, len(VOCAB) - 1), -20.0)], dim=-1))
    scores = asr.score_batch(model, _tokenizer(), torch.randn(2, 3200), torch.tensor([3200, 3200]),
                             ["AT", "R2D2"])
    assert scores[0, 0] == 0.0 and scores[0, 1] == 1.0 and scores[0, 2] == 1.0
    assert np.isnan(scores[1]).all()                            # digits are not in the vocabulary


def test_text_helpers():
    assert asr.normalize_word("able-bodied") == "ABLE BODIED"
    assert asr.encode_target("ab", VOCAB) == [VOCAB["A"], VOCAB["B"]]
    assert asr.encode_target("a b", VOCAB) == [VOCAB["A"], VOCAB["|"], VOCAB["B"]]
    assert asr.encode_target("a1", VOCAB) is None
    assert asr.edit_distance("KITTEN", "SITTING") == 3 and asr.edit_distance("", "AB") == 2
    inverse = {i: t for t, i in VOCAB.items()}
    assert asr.greedy_decode([3, 3, 0, 3, 1, 4, 4], 0, inverse) == "AA B"


# ---------------------------------------------------------------------------
# Prompted words
# ---------------------------------------------------------------------------
def test_prompt_words_are_complete_and_speaker_independent():
    if not (config.CORPUS_DOCS_DIR / "mlf").exists():
        pytest.skip("corpus label files are not present")
    words = asr.load_prompt_words()                              # raises if speakers disagree
    assert len(words) == config.WORDS_PER_SPEAKER
    assert {block for block, _ in words} == {"B1", "B2", "B3"}
    assert all(isinstance(w, str) and w for w in words.values())


def test_missing_prompt_is_reported():
    import pandas as pd
    frame = pd.DataFrame({"Block": ["B1"], "WordCode": ["ZZ99"]})
    with pytest.raises(KeyError, match="No prompted word"):
        asr.prompt_word_column(frame, {("B1", "C1"): "COMMAND"})


# ---------------------------------------------------------------------------
# Session-level decision
# ---------------------------------------------------------------------------
def test_session_level_decides_per_speaker_from_all_utterances():
    speakers = np.array(["a"] * 4 + ["b"] * 4)
    y = np.array([0] * 4 + [3] * 4)
    prob = np.array([[.6, .2, .1, .1], [.4, .1, .1, .4], [.5, .2, .2, .1], [.1, .1, .1, .7]] +
                    [[.1, .1, .1, .7]] * 3 + [[.7, .1, .1, .1]])
    out = session_level(speakers, y, prob)
    assert out["correct"] == 2 and out["n"] == 2 and out["ordinal_mae"] == 0.0


# ---------------------------------------------------------------------------
# Session-level (speaker-level) classifier
# ---------------------------------------------------------------------------
def _speaker_manifest(n_per_speaker: int = 6):
    import pandas as pd
    rows = []
    for speaker in config.DYSARTHRIC_IDS:
        for i in range(n_per_speaker):
            rows.append({"Speaker_ID": speaker, "Severity": config.SEVERITY_MAP[speaker],
                         "Filename": f"{speaker}_B1_UW{i}_M6.wav"})
    return pd.DataFrame(rows)


def test_speaker_gaussian_is_nearest_class_mean_in_one_dimension():
    from src.aaflite.speaker_level import SpeakerGaussian
    X = np.array([[0.0], [0.1], [1.0], [1.1], [2.0], [2.1], [3.0], [3.1]])
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    model = SpeakerGaussian().fit(X, y)
    prob = model.predict_proba(np.array([[0.2], [1.3], [2.3], [9.0]]))
    assert prob.argmax(axis=1).tolist() == [0, 1, 2, 3]
    assert np.allclose(prob.sum(axis=1), 1.0)
    unseen = SpeakerGaussian().fit(X[:6], y[:6]).predict_proba(np.array([[3.0]]))
    assert unseen[0, 3] == 0.0                                  # a class absent from training


def test_session_level_never_fits_on_the_held_out_speaker_and_is_constant_per_speaker(monkeypatch):
    from src.aaflite import speaker_level
    df = _speaker_manifest()
    y = df["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy()
    rng = np.random.default_rng(0)
    marker = {s: 1000.0 + 100.0 * i for i, s in enumerate(config.DYSARTHRIC_IDS)}   # far apart
    scalar = np.array([marker[s] for s in df["Speaker_ID"]]) + rng.normal(size=len(df))
    fitted = []
    original = speaker_level.SpeakerGaussian.fit

    def spy(self, X, y_):
        fitted.append(np.asarray(X).copy())
        return original(self, X, y_)

    monkeypatch.setattr(speaker_level.SpeakerGaussian, "fit", spy)
    out = speaker_level.evaluate_speaker_level(df, {"s": scalar}, {"m": ["s"]}, "t", save=False)["m"]
    assert len(fitted) == len(config.DYSARTHRIC_IDS)
    from src.splits import iter_severity_loso_folds
    for (fold_id, _, _), X in zip(iter_severity_loso_folds(df), fitted):
        assert len(X) == len(config.DYSARTHRIC_IDS) - 1
        assert not np.any(np.abs(X - marker[fold_id]) < 5)     # the held-out speaker's value is absent
    preds = out["predictions"]
    assert preds.groupby("speaker_id")["y_pred"].nunique().eq(1).all()
    assert out["pooled"]["accuracy"] == pytest.approx(out["pooled"]["speakers_correct"] / out["pooled"]["n_speakers"])


def test_session_level_recovers_severity_and_falls_to_chance_on_permuted_labels():
    from src.aaflite.pipeline import permuted_labels
    from src.aaflite.speaker_level import evaluate_speaker_level
    df = _speaker_manifest(8)
    y = df["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy()
    rng = np.random.default_rng(1)
    signal = y + rng.normal(scale=0.2, size=len(y))
    good = evaluate_speaker_level(df, {"s": signal}, {"m": ["s"]}, "t", save=False)["m"]["pooled"]
    assert good["speakers_correct"] >= 14
    perm = evaluate_speaker_level(df, {"s": signal}, {"m": ["s"]}, "t", y=permuted_labels(df), save=False)["m"]["pooled"]
    assert perm["speakers_correct"] <= 8


# ---------------------------------------------------------------------------
# Nested selection of the session-level model (the clean headline)
# ---------------------------------------------------------------------------
def test_nested_selection_never_lets_the_held_out_speaker_into_any_fit_or_choice(monkeypatch):
    from src.aaflite import speaker_level
    from src.splits import iter_severity_loso_folds
    df = _speaker_manifest(6)
    y = df["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy()
    rng = np.random.default_rng(0)
    marker = {s: 1000.0 + 100.0 * i for i, s in enumerate(config.DYSARTHRIC_IDS)}
    ids = np.array([marker[s] for s in df["Speaker_ID"]])
    scalars = {"signal": ids + y + rng.normal(scale=0.1, size=len(df)), "noise": ids + rng.normal(size=len(df))}
    fits = []
    original = speaker_level.SpeakerGaussian.fit

    def spy(self, X, y_):
        fits.append(np.asarray(X).copy())
        return original(self, X, y_)

    monkeypatch.setattr(speaker_level.SpeakerGaussian, "fit", spy)
    out = speaker_level.evaluate_speaker_level_nested(
        df, scalars, {"a": ["signal"], "b": ["noise"]}, "t", "nested", save=False)["nested"]
    folds = [fold_id for fold_id, _, _ in iter_severity_loso_folds(df)]
    per_fold = len(fits) // len(folds)
    assert len(fits) == per_fold * len(folds)
    for k, fold_id in enumerate(folds):
        for X in fits[k * per_fold:(k + 1) * per_fold]:
            assert len(X) in (len(config.DYSARTHRIC_IDS) - 1, len(config.DYSARTHRIC_IDS) - 2)
            assert not np.any(np.abs(X - marker[fold_id]) < 50)     # held-out speaker absent everywhere
    assert out["pooled"]["completed_folds"] == len(config.DYSARTHRIC_IDS)


def test_nested_selection_picks_the_informative_candidate_and_is_deterministic():
    from src.aaflite.pipeline import permuted_labels
    from src.aaflite.speaker_level import evaluate_speaker_level_nested
    df = _speaker_manifest(8)
    y = df["Severity"].map(config.SEVERITY_LABEL_MAP).to_numpy()
    rng = np.random.default_rng(3)
    scalars = {"signal": y + rng.normal(scale=0.15, size=len(y)), "noise": rng.normal(size=len(y))}
    candidates = {"noise_first": ["noise"], "signal": ["signal"]}
    first = evaluate_speaker_level_nested(df, scalars, candidates, "t", "n", save=False)["n"]
    second = evaluate_speaker_level_nested(df, scalars, candidates, "t", "n", save=False)["n"]
    assert first["predictions"]["y_pred"].tolist() == second["predictions"]["y_pred"].tolist()
    assert [r["chosen"] for r in first["folds"]].count("signal") >= 13
    assert first["pooled"]["speakers_correct"] >= 14
    perm = evaluate_speaker_level_nested(df, scalars, candidates, "t", "n", y=permuted_labels(df), save=False)["n"]
    assert perm["pooled"]["speakers_correct"] <= 8
