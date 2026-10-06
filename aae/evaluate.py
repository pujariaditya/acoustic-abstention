"""Decode the complete evaluation sets with an original model or a released checkpoint, and score them.

    python -m aae.evaluate --model whisper-large-v3 --original --out runs/complete/whisper-large-v3/original
    python -m aae.evaluate --model whisper-large-v3 --checkpoint whisper-large-v3/aae \\
        --out runs/complete/whisper-large-v3/aae

Writes ``metrics.json`` (per leg: HR, WER or CER, empty-output rate, decode time) and ``transcripts.jsonl``.

The protocol is part of the result. Clips are sorted by duration (stably, so manifest order breaks ties) and decoded
in fixed per-model batches, because the audio-LLMs pad per batch; the model decodes one warm-up clip before the
checkpoint is applied; seeds are fixed and cuDNN is off. Run it in the model's environment (``aae.models``).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

from aae import audio, metrics, models, weights
from aae.prepare import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AAE_DATA", ROOT / "data"))
MANIFESTS = DATA / "manifests" / "complete"


def decode(ad, rows, batch: int, wavs=None):
    """Transcribe ``rows`` in duration order. Returns ([{"id", "text", "error"}] in manifest order, seconds, the number
    of batches split after running out of memory). ``wavs`` defaults to each row's staged clip."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["duration"])
    wavs = [audio.load(r["staged"]) for r in rows] if wavs is None else wavs
    out = [None] * len(rows)
    splits = 0

    def chunk(idx):
        """One batch; on CUDA out-of-memory, free the failed call's memory (outside the except block, whose traceback
        still holds the activations) and retry in halves down to single clips."""
        oom, err = False, None
        try:
            return ad.transcribe([wavs[i] for i in idx]), [None] * len(idx)
        except torch.OutOfMemoryError:
            oom = True
        except Exception as e:  # noqa: BLE001 (recorded per clip, scored as a failure)
            err = f"{type(e).__name__}: {e}"
        if not oom:
            return [""] * len(idx), [err] * len(idx)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(idx) == 1:
            return [""], ["OutOfMemoryError: one clip does not fit"]
        nonlocal splits
        splits += 1
        h = len(idx) // 2
        t1, e1 = chunk(idx[:h])
        t2, e2 = chunk(idx[h:])
        return t1 + t2, e1 + e2

    t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        texts, errs = chunk(idx)
        for i, t, e in zip(idx, texts, errs, strict=True):
            out[i] = {"id": rows[i]["id"], "text": t, "error": e}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.time() - t0, splits


def run(model: str, checkpoint: str | None, legs: list, out_dir, batch: int = 0, decoding: str = "protocol") -> dict:
    torch.manual_seed(0)
    device = models.cuda()
    t = time.time()
    ad = models.load(model, device)
    res = {"model": model, "checkpoint": checkpoint or "original", "legs": {},
           "timing": {"load_s": round(time.time() - t, 1)}}
    if decoding == "default":                 # the model's own default decoding instead of the paper's protocol
        if not hasattr(ad, "transcribe_default"):
            raise SystemExit(f"{model} has no default decoding other than the protocol's")
        ad.transcribe = ad.transcribe_default
        res["decoding"] = "default"
    ad.transcribe([audio.load(read_jsonl(MANIFESTS / "ls_clean.jsonl")[0]["staged"])])     # warm-up
    if checkpoint:
        res["training"] = weights.apply(ad.model, weights.resolve(checkpoint))
    ad.model.eval()
    batch = batch or models.spec(model).batch
    decode_s = audio_s = 0.0
    records = []
    for leg in legs:
        rows = read_jsonl(MANIFESTS / f"{leg}.jsonl")
        recs, dt, splits = decode(ad, rows, batch)
        if splits:
            print(f"warning: {leg}: {splits} batch(es) split after running out of GPU memory; the audio-LLMs pad per "
                  "batch, so their numbers can differ slightly from the paper's", file=sys.stderr, flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for r, rec in zip(rows, recs, strict=True):
            rec.update({"leg": leg, "ref": r["ref"]})
        m = metrics.leg_metrics(leg, recs)
        m["decode_s"] = round(dt, 2)
        m["oom_splits"] = splits
        res["legs"][leg] = m
        decode_s += dt
        audio_s += sum(r["duration"] for r in rows)
        records += recs
        rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}
        print(json.dumps({"leg": leg, **rounded}), flush=True)
    res["timing"].update({"decode_s": round(decode_s, 1), "audio_s": round(audio_s, 1),
                          "rtfx": round(audio_s / max(decode_s, 1e-6), 1)})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "transcripts.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "metrics.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=models.MODEL_NAMES)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="<model>/<variant> of a downloaded release, or a checkpoint directory")
    g.add_argument("--original", action="store_true", help="the original model, without a checkpoint")
    ap.add_argument("--legs", default="", help="comma-separated; default: every leg, and unfiltered FSD50K")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=0, help="override the protocol batch size (changes the numbers)")
    ap.add_argument("--decoding", choices=("protocol", "default"), default="protocol",
                    help="default: Whisper's transcribe() or Hojo-ASR-V1's released beam search, not the protocol")
    args = ap.parse_args()
    # fsd50k_all is the unfiltered FSD50K evaluation set (clips with speech, singing or music labels included)
    legs = [x for x in args.legs.split(",") if x] or [*metrics.LEGS, "fsd50k_all"]
    res = run(args.model, None if args.original else args.checkpoint, legs, args.out, args.batch, args.decoding)
    print(json.dumps({"timing": res["timing"]}))


if __name__ == "__main__":
    main()
