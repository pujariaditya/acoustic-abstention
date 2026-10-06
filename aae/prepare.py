"""Build runnable manifests from the shipped ones, staging every clip and checking it against its digest.

    python -m aae.prepare --data-root /path/to/datasets            # everything: 59,244 distinct clips, 23 GB staged
    python -m aae.prepare --data-root ... --sets complete,stress   # evaluation only

Reads ``assets/manifests`` (clip ids, order, batching durations, digests) and the corpora under ``--data-root`` in the
layout of ``aae.corpora.DATA_LAYOUT``. Writes ``data/manifests/<set>/<leg>.jsonl`` and the staged audio under
``data/stage`` (``--out``, default ``$AAE_DATA`` or ``data/``). The two noisy speech sets are rebuilt from their
recipe first (``aae.noisy``).

Every staged clip is compared with the digest of the audio the paper's numbers were produced from. A mismatch means
the input audio differs (a different release of a corpus, or a different resampler build); it is reported per leg in
``data/manifests/digests.json`` and, with ``--strict``, fails the run. A clip that cannot be staged at all is left out
of the runnable manifests and always fails the run; rerunning resumes, since staged clips are kept.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from aae import audio, corpora, noisy

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AAE_DATA", ROOT / "data"))            # the default of aae.evaluate, aae.train, ...
ASSETS = ROOT / "assets" / "manifests"
SETS = ("complete", "train", "stress")
NOISY = ("tail", "snr5", "tail_heldout", "snr5_heldout")


def read_jsonl(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage_path(out: Path, source: str, clip_id: str) -> Path:
    return out / "stage" / source.replace(":", "/") / f"{clip_id}.npy"


class Catalogs:
    """Corpus readers, each built once and only when a manifest needs it."""

    def __init__(self, data_root):
        self.data_root, self.cache = Path(data_root), {}

    def __call__(self, source: str) -> dict:
        if source not in self.cache:
            self.cache[source] = corpora.catalog(self.data_root, source)
        return self.cache[source]

    def entry(self, source: str, clip_id: str) -> dict:
        try:
            return self(source)[clip_id]
        except KeyError:
            raise SystemExit(f"{source} has no clip {clip_id!r}: "
                             f"is the corpus complete under {self.data_root}?") from None


def _stage_job(job):
    src, dst = job
    try:
        return audio.stage(src, dst)
    except Exception as e:  # noqa: BLE001 (reported per clip)
        return None, f"{type(e).__name__}: {e}"


def _noisy_job(job):
    kind, speech_path, noise_path, start, wav, dst = job
    try:
        if not Path(dst).exists():
            y = noisy.MIXERS[kind](audio.read_16k(speech_path), audio.read_16k(noise_path), start)
            Path(wav).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(wav).with_name(f"{Path(wav).name}.{os.getpid()}.tmp.wav")
            noisy.write_wav(tmp, y)
            os.replace(tmp, wav)
        return audio.stage(wav, dst)
    except Exception as e:  # noqa: BLE001 (reported per clip)
        return None, f"{type(e).__name__}: {e}"


def run_jobs(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs, chunksize=32))


def prepare(data_root, out, sets, workers: int, strict: bool) -> dict:
    out = Path(out).resolve()                         # manifests record absolute paths
    cat = Catalogs(data_root)
    report = defaultdict(lambda: {"clips": 0, "matched": 0, "mismatched": [], "errors": []})

    def check(key, row_id, result, expected) -> bool:
        """Record one staged clip; False when it could not be staged."""
        dur, dig = result
        rep = report[key]
        rep["clips"] += 1
        if dur is None:
            rep["errors"].append(f"{row_id}: {dig}")
            return False
        if dig == expected:
            rep["matched"] += 1
        else:
            rep["mismatched"].append(row_id)
        return True

    for s in sets:
        if s == "complete":
            for mf in sorted((ASSETS / s).glob("*.jsonl")):
                leg, rows = mf.stem, read_jsonl(mf)
                if leg in NOISY:
                    jobs, refs = [], []
                    for r in rows:
                        sp = cat.entry(r["speech"]["source"], r["speech"]["id"])
                        nz = cat.entry(r["noise"]["source"], r["noise"]["id"])
                        wav = out / "noisy" / leg / f"{r['id']}.wav"
                        jobs.append((leg, sp["path"], nz["path"], r["noise_start"], wav,
                                     stage_path(out, f"noisy:{leg}", r["id"])))
                        refs.append(sp["ref"])
                    results = run_jobs(_noisy_job, jobs, workers)
                    staged = [j[5] for j in jobs]
                else:
                    ents = [cat.entry(r["source"], r["id"]) for r in rows]
                    staged = [stage_path(out, r["source"], r["id"]) for r in rows]
                    results = run_jobs(_stage_job, [(e["path"], p) for e, p in zip(ents, staged, strict=True)], workers)
                    refs = [e["ref"] for e in ents]
                runnable = []
                for r, p, ref, res in zip(rows, staged, refs, results, strict=True):
                    if check(f"{s}/{leg}", r["id"], res, r["digest"]):
                        runnable.append({"id": r["id"], "leg": leg, "staged": str(p), "duration": r["duration"],
                                         "ref": ref, "digest": r["digest"]})
                write_jsonl(out / "manifests" / s / f"{leg}.jsonl", runnable)
                print(f"| {s}/{leg}: {len(rows)} clips", flush=True)
        elif s == "train":
            for name in ("nonspeech", "speech"):
                rows = read_jsonl(ASSETS / "train" / f"{name}.jsonl")
                staged = [stage_path(out, r["source"], r["id"]) for r in rows]
                srcs = [cat.entry(r["source"], r["id"])["path"] for r in rows]
                results = run_jobs(_stage_job, list(zip(srcs, staged, strict=True)), workers)
                runnable = []
                for r, p, res in zip(rows, staged, results, strict=True):
                    if check(f"train/{name}", r["id"], res, r["digest"]):
                        kept = {k: v for k, v in r.items() if k != "digest"}
                        runnable.append({**kept, "staged": str(p), "digest": r["digest"]})
                write_jsonl(out / "manifests" / "train" / f"{name}.jsonl", runnable)
                print(f"| train/{name}: {len(rows)} clips", flush=True)
        elif s == "stress":
            # the low-SNR stress test: every complete test-other and FLEURS Mandarin test utterance, each paired with
            # a training-pool noise clip per SNR; all of these clips are also in the complete and training sets
            items = read_jsonl(ASSETS / "stress.jsonl")
            clips = {}
            for it in items:
                for part in (it["speech"], it["noise"]):
                    clips[(part["source"], part["id"])] = part["digest"]
            keys = sorted(clips)
            results = run_jobs(_stage_job,
                               [(cat.entry(src, cid)["path"], stage_path(out, src, cid)) for src, cid in keys], workers)
            failed = {k for k, res in zip(keys, results, strict=True)
                      if not check("stress", f"{k[0]}/{k[1]}", res, clips[k])}
            runnable = []
            for it in items:
                if any((p["source"], p["id"]) in failed for p in (it["speech"], it["noise"])):
                    continue
                sp = cat.entry(it["speech"]["source"], it["speech"]["id"])
                runnable.append({"speech_id": it["speech"]["id"], "lang": it["lang"], "snr": it["snr"],
                                 "ref": sp["ref"],
                                 "speech": str(stage_path(out, it["speech"]["source"], it["speech"]["id"])),
                                 "noise": str(stage_path(out, it["noise"]["source"], it["noise"]["id"]))})
            write_jsonl(out / "manifests" / "stress.jsonl", runnable)
            print(f"| stress: {len(items)} items", flush=True)

    summary = {k: {"clips": v["clips"], "matched": v["matched"], "mismatched": len(v["mismatched"]),
                   "errors": len(v["errors"]), "examples": (v["mismatched"] + v["errors"])[:5]}
               for k, v in report.items()}
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    report_path = out / "manifests" / "digests.json"                 # merged: one file for every set staged so far
    merged = json.loads(report_path.read_text()) if report_path.exists() else {}
    report_path.write_text(json.dumps({**merged, **summary}, indent=1) + "\n")
    bad = {k: v for k, v in summary.items() if v["mismatched"] or v["errors"]}
    total = sum(v["clips"] for v in summary.values())
    print(f"| digests: {sum(v['matched'] for v in summary.values())}/{total} clips identical to the paper's inputs")
    for k, v in bad.items():
        print(f"|   {k}: {v['mismatched']} differ, {v['errors']} failed, e.g. {v['examples'][:2]}")
    failed = sum(v["errors"] for v in summary.values())
    if failed:
        raise SystemExit(f"{failed} clips could not be staged and are left out of the manifests (see "
                         f"{out / 'manifests' / 'digests.json'}); fix the corpus and rerun, staged clips are kept")
    if bad and strict:
        raise SystemExit(f"inputs differ from the paper's (see {out / 'manifests' / 'digests.json'})")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, help="directory holding the corpora (see aae.corpora.DATA_LAYOUT)")
    ap.add_argument("--out", default=str(DATA), help="default: $AAE_DATA, else data/ in the repository")
    ap.add_argument("--sets", default=",".join(SETS), help=f"comma-separated, from {', '.join(SETS)}")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--strict", action="store_true", help="stop if any staged clip differs from its digest")
    args = ap.parse_args()
    sets = [s for s in args.sets.split(",") if s]
    unknown = set(sets) - set(SETS)
    if unknown:
        raise SystemExit(f"unknown sets {sorted(unknown)}")
    prepare(args.data_root, args.out, sets, args.workers, args.strict)


if __name__ == "__main__":
    main()
