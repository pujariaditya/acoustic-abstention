"""Metrics, as defined in the paper.

Hallucination rate (HR): percentage of non-speech clips for which the model returns any non-whitespace text,
Calm-Whisper's definition. Punctuation and music glyphs count: Whisper emits ".", "..." and music notes on noise, and
treating those as silence would both understate the problem and reward emitting them.

WER uses Whisper's EnglishTextNormalizer. CER uses Whisper's BasicTextNormalizer, OpenCC traditional-to-simplified
conversion and no whitespace. A clip whose decoding raised counts as hallucinated on non-speech; on speech it is scored
with an empty hypothesis (every reference word deleted) but not counted in the empty-output rate, as in the harness
that produced the paper's numbers.
"""
from __future__ import annotations

import re

NONSPEECH_LEGS = ("urban", "wham", "fsd50k")
SPEECH_EN_LEGS = ("ls_clean", "ls_other", "tail", "snr5")
SPEECH_ZH_LEGS = ("fleurs_zh", "aishell")
# speech with held-out noise: the 1,000 utterances of tail and snr5, mixed with WHAM! evaluation noise
SPEECH_HELDOUT_LEGS = ("tail_heldout", "snr5_heldout")
# out-of-domain non-speech: digital silence, synthetic noise and NSynth instrument notes (aae.ood)
NONSPEECH_OOD_LEGS = ("silence", "white_noise", "pink_noise", "nsynth")
LEGS = NONSPEECH_LEGS + SPEECH_EN_LEGS + SPEECH_ZH_LEGS

# The speech-accuracy tolerance fixed before the experiments: allowed increase over the original model, in points.
# A set is beyond its tolerance when the increase exceeds it by at least 0.0005 points (``beyond_tolerance``).
TOLERANCE = {"ls_clean": 0.3, "ls_other": 0.5, "tail": 1.0, "snr5": 1.0, "fleurs_zh": 1.0, "aishell": 1.0}

_EN = _BASIC = _T2S = None


def _normalizers():
    global _EN, _BASIC, _T2S
    if _EN is None:
        import opencc
        from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

        _EN, _BASIC, _T2S = EnglishTextNormalizer(), BasicTextNormalizer(), opencc.OpenCC("t2s")
    return _EN, _BASIC, _T2S


def kind(leg: str) -> str:
    if leg in NONSPEECH_LEGS or leg in NONSPEECH_OOD_LEGS or leg == "fsd50k_all":
        return "nonspeech"
    return "speech_zh" if leg in SPEECH_ZH_LEGS else "speech_en"


def metric_of(leg: str) -> str:
    return "cer" if leg in SPEECH_ZH_LEGS else "wer"


def nonempty(text) -> bool:
    return bool((text or "").strip())


def normalize_en(t: str) -> str:
    return _normalizers()[0](t or "")


def normalize_zh(t: str) -> str:
    _, basic, t2s = _normalizers()
    return re.sub(r"\s+", "", basic(t2s.convert(t or "")))


def normalize(t: str, leg: str) -> str:
    return normalize_zh(t) if leg in SPEECH_ZH_LEGS else normalize_en(t)


def leg_metrics(leg: str, recs: list) -> dict:
    """recs: [{"text", "error", "ref"}]. HR on non-speech; WER or CER plus the empty-output rate on speech."""
    import jiwer

    n = len(recs)
    hyps = ["" if r.get("error") else (r.get("text") or "") for r in recs]
    out = {"n": n, "n_errors": sum(1 for r in recs if r.get("error"))}
    out["hr"] = 100.0 * sum(1 for r, h in zip(recs, hyps, strict=True) if r.get("error") or nonempty(h)) / max(1, n)
    k = kind(leg)
    if k != "nonspeech":
        refs = [normalize(r["ref"], leg) for r in recs]
        hyp_n = [normalize(h, leg) for h in hyps]
        out[metric_of(leg)] = 100.0 * (jiwer.cer(refs, hyp_n) if k == "speech_zh" else jiwer.wer(refs, hyp_n))
        out["empty_rate"] = 100.0 - out["hr"]
    return out


def edits(leg: str, recs: list) -> tuple[list, list]:
    """Per utterance: the word (for Mandarin, character) edits between the normalized reference and hypothesis, and
    the reference length. ``sum(edits) / sum(lengths)`` is the WER or CER of ``leg_metrics``."""
    from rapidfuzz.distance import Levenshtein

    e, n = [], []
    for r in recs:
        ref = normalize(r["ref"], leg)
        hyp = normalize("" if r.get("error") else (r.get("text") or ""), leg)
        rt, ht = (list(ref), list(hyp)) if leg in SPEECH_ZH_LEGS else (ref.split(), hyp.split())
        e.append(Levenshtein.distance(rt, ht))
        n.append(len(rt))
    return e, n


def insertions(leg: str, recs: list) -> tuple[int, int]:
    """Words (for Mandarin, characters) the hypotheses add over the normalized references, summed over ``recs``: all of
    them, and those after the last reference word (text written after the speech)."""
    import jiwer

    total = after = 0
    for r in recs:
        ref = normalize(r["ref"], leg)
        hyp = normalize("" if r.get("error") else (r.get("text") or ""), leg)
        if not ref:
            continue
        ops = jiwer.process_characters(ref, hyp) if leg in SPEECH_ZH_LEGS else jiwer.process_words(ref, hyp)
        n_ref = len(ref) if leg in SPEECH_ZH_LEGS else len(ref.split())
        for chunk in ops.alignments[0]:
            if chunk.type == "insert":
                total += chunk.hyp_end_idx - chunk.hyp_start_idx
                after += (chunk.hyp_end_idx - chunk.hyp_start_idx) if chunk.ref_start_idx >= n_ref else 0
    return total, after


def script(text: str) -> str:
    """"zh" when Chinese characters outnumber Latin letters, "en" when Latin letters do, "" for an empty output."""
    zh = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text or ""))
    en = len(re.findall(r"[A-Za-z]", text or ""))
    return "" if zh == en == 0 else ("zh" if zh >= en else "en")


def change_interval(edits_a, edits_b, lengths, resamples: int = 10000, seed: int = 0) -> tuple[float, float]:
    """The paired bootstrap 95% interval of the change in WER or CER (points) from system a to system b on the same
    utterances: utterances are drawn with replacement, and both systems are scored on every draw."""
    import numpy as np

    d_edits, n = np.asarray(edits_b) - np.asarray(edits_a), np.asarray(lengths)
    rng, out = np.random.default_rng(seed), []
    for start in range(0, resamples, 500):                 # in blocks, so that 7,176 x 10,000 draws stay small
        idx = rng.integers(0, len(n), size=(min(500, resamples - start), len(n)))
        out.append(100.0 * d_edits[idx].sum(axis=1) / n[idx].sum(axis=1))
    lo, hi = np.percentile(np.concatenate(out), [2.5, 97.5])
    return float(lo), float(hi)


def beyond_tolerance(leg: str, increase: float) -> bool:
    """Whether an increase in WER or CER over the original model (in points) exceeds the tolerance of ``leg``."""
    return round(max(0.0, increase - TOLERANCE[leg]), 3) > 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval for k successes in n trials, as fractions."""
    p = k / n
    c, h = p + z * z / (2 * n), z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - h) / (1 + z * z / n), (c + h) / (1 + z * z / n)
