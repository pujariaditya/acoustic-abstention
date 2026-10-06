"""Transcribe your own audio with an original recognizer and with a released checkpoint, side by side.

    python -m aae.transcribe --model whisper-large-v3 street_noise.wav speech.flac
    python -m aae.transcribe --model whisper-large-v3 --checkpoint whisper-large-v3/aae *.wav

Each file goes through the evaluation's input path (mono, 16 kHz with soxr, the int16 round trip of a staged clip,
truncated to 30 s) and is decoded like an evaluation set: sorted by duration, in the model's protocol batches. Run it
in the model's environment (``aae.models``): Whisper and Qwen3-ASR in envs/main, ARK-ASR and Hojo-ASR in envs/legacy.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from aae import audio, evaluate, models, weights


def read_files(files):
    """(rows, wavs): each file exactly as the evaluation reads a staged clip of it."""
    wavs = [(audio.to_int16(audio.read_16k(f)).astype(np.float32) / 32768.0)[: int(audio.SAMPLE_RATE
                                                                                   * audio.MAX_SECONDS)]
            for f in files]
    rows = [{"id": str(f), "duration": len(w) / audio.SAMPLE_RATE} for f, w in zip(files, wavs, strict=True)]
    return rows, wavs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", required=True, choices=models.MODEL_NAMES)
    ap.add_argument("--checkpoint", default="", help="default: <model>/aae")
    args = ap.parse_args()
    rows, wavs = read_files(args.files)
    ad = models.load(args.model, "cuda:0" if torch.cuda.is_available() else "cpu")
    ad.model.eval()
    batch = models.spec(args.model).batch
    with torch.no_grad():
        original, _, _ = evaluate.decode(ad, rows, batch, wavs)
        weights.apply(ad.model, weights.resolve(args.checkpoint or f"{args.model}/aae"))
        ad.model.eval()
        ours, _, _ = evaluate.decode(ad, rows, batch, wavs)
    for f, a, b in zip(args.files, original, ours, strict=True):
        err = a["error"] or b["error"]
        print(f"{f}\n  original: {a['text']!r}\n  AAE:      {b['text']!r}" + (f"\n  error: {err}" if err else ""))


if __name__ == "__main__":
    main()
