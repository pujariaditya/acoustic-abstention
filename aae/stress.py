"""Low-SNR stress test on the complete LibriSpeech test-other and FLEURS Mandarin test sets.

    python -m aae.stress --model whisper-large-v3 --checkpoint whisper-large-v3/aae \\
        --out runs/stress/whisper-large-v3.json

Every utterance of both sets (2,939 + 945) is mixed with a noise clip from the training non-speech pools at -5, 5 and
15 dB SNR; ``assets/manifests/stress.jsonl`` fixes the clip of every utterance and SNR. The original model decodes the
-5 dB mixtures and the checkpoint all three. Each cell counts the utterances returned empty and scores the error rate
(WER; CER for Mandarin); at -5 dB it also counts the checkpoint transcripts that equal the original model's transcript
of the same mixture after normalization. Without added noise, the same utterances are scored by ``aae.evaluate``.

The mixtures are decoded in one list per system, sorted stably by utterance length (SNR, then manifest order, breaks
ties) and cut into batches of 8, the order in which the paper's numbers were produced. Run it in the model's
environment (``aae.models``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
import numpy as np
import torch

from aae import audio, metrics, models, weights
from aae.evaluate import DATA
from aae.prepare import read_jsonl

PLAN = {"original": (-5.0,), "checkpoint": (-5.0, 5.0, 15.0)}       # the SNRs each system decodes
LEG = {"en": "ls_other", "zh": "fleurs_zh"}                         # the normalizer and metric of each language
MAX_SECONDS = 40.0      # this test read up to 40 s of each utterance; mixing scales the noise to the whole utterance


def overlay(speech: np.ndarray, noise: np.ndarray, snr: float) -> np.ndarray:
    """Noise tiled or cut to the speech length from its start, scaled to ``snr``; peak-normalized if it clips."""
    n = noise
    if len(n) < len(speech):
        n = np.tile(n, int(np.ceil(len(speech) / max(len(n), 1))))
    n = n[: len(speech)]
    rs = float(np.sqrt(np.mean(speech ** 2)) + 1e-6)
    rn = float(np.sqrt(np.mean(n ** 2)) + 1e-6)
    out = speech + n * (rs / rn) * 10 ** (-snr / 20)
    peak = float(np.max(np.abs(out)) + 1e-6)
    return (out / peak * 0.95 if peak > 1.0 else out).astype(np.float32)


def key(utt_id: str, snr: float) -> str:
    return f"{utt_id}|{snr:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=models.MODEL_NAMES)
    ap.add_argument("--checkpoint", required=True, help="<model>/<variant> of a downloaded release, or a directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8, help="the protocol batch size (changing it changes the numbers)")
    args = ap.parse_args()

    utts, noise = [], {}            # utterances in manifest order; the noise clip of each (utterance, SNR)
    for it in read_jsonl(DATA / "manifests" / "stress.jsonl"):
        if not utts or utts[-1]["id"] != it["speech_id"]:
            utts.append({"id": it["speech_id"], "lang": it["lang"], "ref": it["ref"], "speech": it["speech"]})
        noise[(len(utts) - 1, it["snr"])] = it["noise"]
    base = [audio.load(u["speech"], max_seconds=MAX_SECONDS) for u in utts]

    torch.manual_seed(0)
    ad = models.load(args.model, models.cuda())
    texts, training = {}, {}
    for system, snrs in PLAN.items():                       # the original model first, then the checkpoint
        if system == "checkpoint":
            training = weights.apply(ad.model, weights.resolve(args.checkpoint))
        ad.model.eval()
        cells = [(i, snr) for snr in snrs for i in range(len(utts))]
        order = sorted(range(len(cells)), key=lambda c: len(base[cells[c][0]]))
        out = [None] * len(cells)
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            wavs = [overlay(base[cells[c][0]], audio.load(noise[cells[c]], max_seconds=None), cells[c][1]) for c in idx]
            with torch.no_grad():
                for c, text in zip(idx, ad.transcribe(wavs), strict=True):
                    out[c] = text
        texts[system] = {key(utts[i]["id"], snr): out[c] for c, (i, snr) in enumerate(cells)}
        print(json.dumps({"decoded": system, "mixtures": len(cells)}), flush=True)

    table = {}
    for system, snrs in PLAN.items():
        for snr in snrs:
            for lang, leg in LEG.items():
                ids = [u["id"] for u in utts if u["lang"] == lang]
                refs = [metrics.normalize(u["ref"], leg) for u in utts if u["lang"] == lang]
                hyps = [texts[system][key(i, snr)] for i in ids]
                err = (jiwer.cer if lang == "zh" else jiwer.wer)(refs, [metrics.normalize(h, leg) for h in hyps])
                cell = {"n": len(ids), "empty": sum(not h.strip() for h in hyps), "err": round(100 * err, 3)}
                if system == "checkpoint" and snr in PLAN["original"]:
                    before = [texts["original"][key(i, snr)] for i in ids]
                    cell["same"] = sum(metrics.normalize(h, leg) == metrics.normalize(b, leg)
                                       for h, b in zip(hyps, before, strict=True))
                table.setdefault(system, {})[f"{snr:g}dB_{lang}"] = cell
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": args.model, "checkpoint": args.checkpoint, "training": training,
                                          "table": table, "transcripts": texts}, ensure_ascii=False, indent=1) + "\n")
    for system, row in table.items():
        print(system, json.dumps(row))


if __name__ == "__main__":
    main()
