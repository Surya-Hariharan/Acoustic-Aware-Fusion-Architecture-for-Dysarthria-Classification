"""
Recogniser-based intelligibility: how well a frozen CTC speech recogniser
recovers the word that was prompted.

UA-Speech severity classes are defined by *listener intelligibility* — the
accuracy with which naive human listeners transcribed the speaker's isolated
words. A recogniser's accuracy on the same words is the closest automatic
analogue, and it is a handful of numbers per utterance rather than a
high-dimensional embedding, which matters when only 15 speakers carry labels.

Per utterance (all from one forward pass of the CTC head, greedy decoding):

    exact       1 if the decoded string equals the prompted word
    cer         character error rate of the decoded string (edit distance / target length)
    blank       fraction of frames the greedy path emits the CTC blank
    nll_char    CTC negative log-likelihood of the prompted word, per target character
    nll_frame   the same, per frame
    conf        mean frame-wise maximum posterior

Nothing here uses a severity label. The prompted word comes from the corpus'
own label files (data/uaspeech_corpus_docs/mlf), which are speaker-independent:
a (block, word code) is the same word for every speaker.

wav2vec2-base checkpoints were trained on zero-padded batches WITHOUT an
attention mask (config.feat_extract_norm == "group"); passing one degrades them
badly (measured here: the speaker-level correlation of `exact` with severity
fell from +0.94 to +0.16). Checkpoints with layer norm ("lv60") expect it.
"""

import collections
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import config
from src.aaflite.embeddings import _WaveformDataset
from src.console import print_kv, print_status, progress
from src.models.deep_pathway import DeepPathway

SCORE_NAMES = ("exact", "cer", "blank", "nll_char", "nll_frame", "conf")
_MLF_ENTRY = re.compile(r'"\*/(\w+?)_(B\d)_(\w+?)_M\d\.lab"')


# ---------------------------------------------------------------------------
# Prompted words
# ---------------------------------------------------------------------------
def load_prompt_words(mlf_dir: Path = config.CORPUS_DOCS_DIR / "mlf") -> Dict[Tuple[str, str], str]:
    """(block, word code) -> prompted word, from every speaker's word MLF.
    Raises if two speakers disagree: the mapping must be speaker-independent."""
    seen: Dict[Tuple[str, str], set] = collections.defaultdict(set)
    paths = sorted(glob.glob(str(Path(mlf_dir) / "*" / "*_word.mlf")))
    if not paths:
        raise FileNotFoundError(f"No *_word.mlf files under {mlf_dir} — the corpus label files are required.")
    for path in paths:
        current = None
        with open(path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                match = _MLF_ENTRY.match(line)
                if match:
                    current = (match.group(2), match.group(3))
                elif line and line not in ("#!MLF!#", ".") and current is not None:
                    seen[current].add(line)
                    current = None
    conflicts = [key for key, words in seen.items() if len(words) > 1]
    if conflicts:
        raise ValueError(f"Word label files disagree across speakers for {conflicts[:5]}.")
    return {key: next(iter(words)) for key, words in seen.items()}


def prompt_word_column(df: pd.DataFrame, words: Optional[Dict[Tuple[str, str], str]] = None) -> List[str]:
    words = words or load_prompt_words()
    missing = sorted({(b, c) for b, c in zip(df["Block"], df["WordCode"]) if (b, c) not in words})
    if missing:
        raise KeyError(f"No prompted word for {missing[:5]} ({len(missing)} prompts).")
    return [words[(b, c)] for b, c in zip(df["Block"], df["WordCode"])]


# ---------------------------------------------------------------------------
# Text <-> CTC targets (pure functions)
# ---------------------------------------------------------------------------
def normalize_word(word: str) -> str:
    """Upper case, hyphens and underscores as word breaks (the vocabulary has
    letters, the apostrophe and '|' only)."""
    return re.sub(r"\s+", " ", word.upper().replace("-", " ").replace("_", " ")).strip()


def encode_target(word: str, vocab: Dict[str, int]) -> Optional[List[int]]:
    ids = [vocab["|"] if ch == " " else vocab.get(ch) for ch in normalize_word(word)]
    return None if not ids or any(i is None for i in ids) else ids


def edit_distance(a: Sequence, b: Sequence) -> int:
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        previous, row[0] = row[0], i
        for j in range(1, len(b) + 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] != b[j - 1]))
    return row[-1]


def greedy_decode(frame_ids: Sequence[int], blank_id: int, id_to_token: Dict[int, str]) -> str:
    """Collapse repeats, drop blanks, '|' becomes a space."""
    out, previous = [], None
    for i in frame_ids:
        if i != previous and i != blank_id:
            out.append(id_to_token[i])
        previous = i
    return "".join(out).replace("|", " ").strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def uses_attention_mask(model) -> bool:
    """Layer-norm checkpoints (wav2vec2-large-lv60) were trained with attention
    masks; group-norm checkpoints (wav2vec2-base) were not and must not get one."""
    return model.config.feat_extract_norm == "layer"


def load_ctc(model_name: str, device: torch.device):
    from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name, token=config.HF_TOKEN)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, token=config.HF_TOKEN).to(device).eval()
    return model, tokenizer


@torch.no_grad()
def score_batch(model, tokenizer, waveform: torch.Tensor, lengths: torch.Tensor,
                targets: Sequence[str]) -> np.ndarray:
    """(B, len(SCORE_NAMES)) float32. NaN rows for words the vocabulary cannot spell."""
    vocab = tokenizer.get_vocab()
    id_to_token = {i: t for t, i in vocab.items()}
    blank = tokenizer.pad_token_id
    sample_mask = torch.arange(waveform.shape[1], device=waveform.device)[None, :] < lengths[:, None]
    inputs = DeepPathway._zero_mean_unit_var_norm(waveform, sample_mask)     # padding stays exactly 0
    use_mask = uses_attention_mask(model)
    with torch.autocast(device_type=waveform.device.type, dtype=torch.float16,
                        enabled=waveform.device.type == "cuda"):
        logits = (model(inputs, attention_mask=sample_mask) if use_mask else model(inputs)).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    frames = model._get_feat_extract_output_lengths(lengths).clamp(min=1)

    out = np.full((len(targets), len(SCORE_NAMES)), np.nan, dtype=np.float32)
    for k, word in enumerate(targets):
        ids = encode_target(word, vocab)
        if ids is None:
            continue
        n = int(frames[k])
        lp = log_probs[k, :n]
        nll = F.ctc_loss(lp[:, None, :], torch.tensor([ids], device=lp.device),
                         torch.tensor([n]), torch.tensor([len(ids)]), blank=blank,
                         reduction="sum", zero_infinity=True).item()
        path = lp.argmax(dim=-1).cpu().numpy()
        decoded = greedy_decode(path.tolist(), blank, id_to_token)
        reference = normalize_word(word)
        out[k] = [float(decoded == reference), edit_distance(decoded, reference) / len(reference),
                  float((path == blank).mean()), nll / len(ids), nll / n,
                  float(lp.max(dim=-1).values.exp().mean())]
    return out


# ---------------------------------------------------------------------------
# Cache (one .npz per speaker, per recogniser)
# ---------------------------------------------------------------------------
def _cache_path(tag: str, speaker: str) -> Path:
    return Path(config.AAFLITE_ASR_DIR) / tag / f"{speaker}.npz"


def _complete(tag: str, speaker: str, filenames: List[str]) -> bool:
    path = _cache_path(tag, speaker)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return set(filenames) <= set(data["filenames"].tolist())
    except Exception:
        return False


def extract_asr_scores(df: pd.DataFrame, device: torch.device, model_name: str, tag: str,
                       batch_size: int = 16, num_workers: int = 2) -> None:
    """Score every utterance of df (any speakers) and cache per speaker; finished
    speakers are skipped, so an interrupted extraction resumes."""
    words = load_prompt_words()
    df = df.assign(_word=prompt_word_column(df, words))
    todo = [(spk, grp) for spk, grp in df.groupby("Speaker_ID", sort=True)
            if not _complete(tag, spk, grp["Filename"].tolist())]
    print_kv(f"Recogniser scores [{tag}]", f"{df['Speaker_ID'].nunique() - len(todo)} speakers cached, "
                                          f"{len(todo)} to score")
    if not todo:
        return
    model, tokenizer = load_ctc(model_name, device)
    for speaker, group in todo:
        loader = DataLoader(_WaveformDataset(group["Filepath"].tolist()), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
        targets, chunks, cursor = group["_word"].tolist(), [], 0
        for waveform, lengths in progress(loader, f"{tag} {speaker}", total=len(loader), unit="batch"):
            n = len(lengths)
            chunks.append(score_batch(model, tokenizer, waveform.to(device), lengths.to(device),
                                      targets[cursor:cursor + n]))
            cursor += n
        scores = np.concatenate(chunks)
        path = _cache_path(tag, speaker)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.npz")
        np.savez(temp, filenames=group["Filename"].to_numpy(dtype=str), scores=scores,
                 names=np.array(SCORE_NAMES), model=np.array(model_name))
        temp.replace(path)
    print_status(f"Recogniser scores [{tag}] cached in {Path(config.AAFLITE_ASR_DIR) / tag}", ok=True)


def load_asr_scores(df: pd.DataFrame, tag: str) -> np.ndarray:
    """(N, len(SCORE_NAMES)) float32 in df row order. Unspellable prompts (NaN)
    are filled with that column's mean so a row is never dropped."""
    out = np.empty((len(df), len(SCORE_NAMES)), dtype=np.float32)
    cache: Dict[str, Tuple[Dict[str, int], np.ndarray]] = {}
    for speaker in df["Speaker_ID"].unique():
        with np.load(_cache_path(tag, speaker), allow_pickle=False) as data:
            cache[speaker] = ({n: i for i, n in enumerate(data["filenames"].tolist())}, data["scores"])
    for row, (speaker, filename) in enumerate(zip(df["Speaker_ID"], df["Filename"])):
        index, scores = cache[speaker]
        out[row] = scores[index[filename]]
    column_means = np.nanmean(out, axis=0)
    return np.where(np.isnan(out), column_means, out).astype(np.float32)
