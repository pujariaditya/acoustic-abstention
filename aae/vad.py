"""The voice-activity-detector baseline: Silero VAD in front of the original recognizers, on the complete sets.

    python -m aae.vad --out runs/vad/silero.jsonl

Silero VAD (``silero-vad`` 6.2.3, its default settings) reads every staged clip of the nine complete evaluation sets
and records the speech it finds. The gate passes a clip with at least one speech segment to the recognizer unchanged
and returns an empty transcript otherwise. The decision does not depend on the recognizer, so ``aae.report`` scores
each gated model from these decisions and that model's original transcripts of the same clips
(``runs/complete/<model>/original``), without decoding again. Runs on the CPU in the main environment.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from aae import audio, metrics

LEGS = (*metrics.NONSPEECH_LEGS, *metrics.SPEECH_EN_LEGS, *metrics.SPEECH_ZH_LEGS)
_MODEL = None


def _segments(path: str) -> list:
    """The speech segments (start and end sample) Silero VAD finds in one staged clip, with its default settings."""
    global _MODEL
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if _MODEL is None:
        torch.set_num_threads(1)
        _MODEL = load_silero_vad()
    y = torch.from_numpy(audio.load(path))
    return [(s["start"], s["end"]) for s in get_speech_timestamps(y, _MODEL, sampling_rate=audio.SAMPLE_RATE)]


def passes(row: dict) -> bool:
    """Whether the gate passes a clip of ``--out`` to the recognizer: at least one speech segment."""
    return bool(row["segments"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    from aae.evaluate import MANIFESTS
    from aae.prepare import read_jsonl

    rows = [r for leg in LEGS for r in read_jsonl(MANIFESTS / f"{leg}.jsonl")]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        segs = list(ex.map(_segments, [r["staged"] for r in rows], chunksize=16))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r, s in zip(rows, segs, strict=True):
            f.write(json.dumps({"leg": r["leg"], "id": r["id"], "segments": s}) + "\n")
    for leg in LEGS:
        n = [passes({"segments": s}) for r, s in zip(rows, segs, strict=True) if r["leg"] == leg]
        print(json.dumps({"leg": leg, "clips": len(n), "passed": sum(n)}), flush=True)


if __name__ == "__main__":
    main()
