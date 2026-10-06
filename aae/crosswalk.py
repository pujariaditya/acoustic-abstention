"""Whisper large-v3 under the three hallucination-rate definitions in use, from one decode per clip.

    python -m aae.crosswalk --leg urban --out runs/crosswalk/urban_original.json
    python -m aae.crosswalk --leg urban --checkpoint whisper-large-v3/aae --out runs/crosswalk/urban_aae.json

Each clip keeps its text, no-speech probability p_ns and average log-probability, and is counted as:

``HR_raw``      any non-empty text                                         (Calm-Whisper; this paper)
``HR_whisper``  non-empty text that Whisper's skip rule keeps; the rule drops output with
                p_ns > 0.6 and avg_logprob <= -1                            (the rule of Aparin et al.)
``HR_nsp``      non-empty text with p_ns <= 0.6                            (HSP, Algorithm 1)

Aparin et al. count every clip the rule keeps, with or without text. For the original model that is the same count,
since it returns text for every clip. ``filter_keeps`` records that literal count as well.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from aae import audio, models, weights
from aae.evaluate import MANIFESTS
from aae.prepare import read_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--leg", default="urban", choices=("urban", "fsd50k", "wham"))
    ap.add_argument("--checkpoint", default="", help="whisper-large-v3/<variant> or a checkpoint directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    ad = models.load("whisper-large-v3", models.cuda())
    if args.checkpoint:
        weights.apply(ad.model, weights.resolve(args.checkpoint))
    ad.model.eval()
    rows = read_jsonl(MANIFESTS / f"{args.leg}.jsonl")
    items, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(rows), args.batch):              # manifest order, as the paper's crosswalk ran
            chunk = rows[i:i + args.batch]
            for r, d in zip(chunk, ad.decode([audio.load(x["staged"]) for x in chunk]), strict=True):
                items.append({"id": r["id"], "text": ad.text(d.tokens), "nsp": float(d.no_speech_prob),
                              "avg_logprob": float(d.avg_logprob)})
    n = len(items)
    text = [bool(x["text"].strip()) for x in items]
    keeps = [not (x["nsp"] > 0.6 and x["avg_logprob"] <= -1.0) for x in items]
    out = {"leg": args.leg, "checkpoint": args.checkpoint or "original", "n": n,
           "HR_raw": round(100 * sum(text) / n, 3),
           "HR_whisper": round(100 * sum(t and k for t, k in zip(text, keeps, strict=True)) / n, 3),
           "HR_nsp": round(100 * sum(t and x["nsp"] <= 0.6 for t, x in zip(text, items, strict=True)) / n, 3),
           # the rule as Aparin et al. state it (p_ns < 0.6 or avg_logprob > -1), the count the paper quotes; it differs
           # from the skip rule above only at p_ns == 0.6 exactly
           "filter_keeps": round(100 * sum(x["nsp"] < 0.6 or x["avg_logprob"] > -1.0 for x in items) / n, 3),
           "nsp_below_0.6": sum(x["nsp"] < 0.6 for x in items),
           "decode_s": round(time.time() - t0, 1), "items": items}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "items"}))


if __name__ == "__main__":
    main()
