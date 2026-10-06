"""Out-of-domain non-speech sets: audio from outside the sources of the training pools.

    python -m aae.ood --nsynth data/raw/nsynth/nsynth-test     # stages the clips, writes the four manifests

silence      1,000 clips of digital zero, 1 to 10 s long
white_noise  1,000 clips of white Gaussian noise at -60 to -20 dBFS RMS, 1 to 10 s long
pink_noise   1,000 clips of pink (1/f power) noise at the same levels and lengths
nsynth       every note of the NSynth test set (4 s, 16 kHz; Engel et al., ICML 2017; CC BY 4.0) whose instrument
             family is not "vocal": isolated instrument notes, a kind of audio absent from the training pools

The synthetic clips come from fixed seeds, so they are rebuilt bit for bit; every staged clip's digest goes into
``assets/manifests/complete/<set>.jsonl`` like the other evaluation sets. Each set is scored like the non-speech
sets of the paper (``aae.evaluate --legs silence,white_noise,pink_noise,nsynth``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aae import audio
from aae.evaluate import DATA
from aae.prepare import ROOT, write_jsonl

SYNTHETIC = {"silence": 0, "white_noise": 1, "pink_noise": 2}    # set -> seed offset
N_CLIPS, SEED = 1000, 2027


def synthetic(kind: str, rng: np.random.Generator) -> np.ndarray:
    """One clip of ``kind`` as float32 at 16 kHz."""
    n = int(round(rng.uniform(1.0, 10.0) * audio.SAMPLE_RATE))
    if kind == "silence":
        return np.zeros(n, dtype=np.float32)
    w = rng.standard_normal(n)
    if kind == "pink_noise":                       # shape the spectrum to 1/f power, then restore zero mean
        spec = np.fft.rfft(w)
        f = np.fft.rfftfreq(n, 1.0 / audio.SAMPLE_RATE)
        spec[1:] /= np.sqrt(f[1:])
        spec[0] = 0.0
        w = np.fft.irfft(spec, n)
    rms = 10 ** (rng.uniform(-60.0, -20.0) / 20)
    return (w / (np.sqrt(np.mean(w ** 2)) + 1e-12) * rms).astype(np.float32)


def stage_array(y: np.ndarray, dst: Path) -> tuple[float, str]:
    """Stage a float32 array like ``audio.stage`` stages a file: int16, the same rounding, the same digest."""
    a = audio.to_int16(y)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        np.save(dst, a)
    return len(a) / audio.SAMPLE_RATE, audio.digest(np.load(dst))


def build(nsynth: Path | None) -> dict:
    stage_dir = DATA / "stage" / "ood"
    sets = {}
    for kind, offset in SYNTHETIC.items():
        rng = np.random.default_rng(SEED + offset)
        rows = []
        for i in range(N_CLIPS):
            cid = f"{kind}_{i:04d}"
            dur, dig = stage_array(synthetic(kind, rng), stage_dir / kind / f"{cid}.npy")
            rows.append({"id": cid, "source": f"synthetic:{kind}", "duration": dur, "digest": dig,
                         "staged": str(stage_dir / kind / f"{cid}.npy")})
        sets[kind] = rows
    if nsynth:
        meta = json.loads((nsynth / "examples.json").read_text())
        rows = []
        for name in sorted(meta):
            if meta[name]["instrument_family_str"] == "vocal":
                continue
            dst = stage_dir / "nsynth" / f"{name}.wav.npy"
            dur, dig = audio.stage(nsynth / "audio" / f"{name}.wav", dst)
            rows.append({"id": f"{name}.wav", "source": "nsynth:test", "duration": dur, "digest": dig,
                         "staged": str(dst)})
        sets["nsynth"] = rows
    for leg, rows in sets.items():
        write_jsonl(ROOT / "assets" / "manifests" / "complete" / f"{leg}.jsonl",
                    [{k: r[k] for k in ("id", "source", "duration", "digest")} for r in rows])
        write_jsonl(DATA / "manifests" / "complete" / f"{leg}.jsonl",
                    [{"id": r["id"], "leg": leg, "staged": r["staged"], "duration": r["duration"], "ref": "",
                      "digest": r["digest"]} for r in rows])
        print(json.dumps({"set": leg, "clips": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 2)}))
    return sets


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nsynth", type=Path, help="the extracted nsynth-test directory (examples.json, audio/)")
    build(ap.parse_args().nsynth)


if __name__ == "__main__":
    main()
