#!/usr/bin/env python3
"""Download the public corpora into the layout ``aae.prepare`` reads (``aae.corpora.DATA_LAYOUT``).

    python scripts/fetch_data.py --data-root /path/to/datasets             # all corpora, ~95 GB of archives
    python scripts/fetch_data.py --data-root ... --only librispeech,fleurs

Nothing is redistributed by this repository; each corpus is used under its own license. The sources are the ones the
paper's inputs were built from. Two corpora came from Hugging Face parquet mirrors that carry the original audio files
as bytes (UrbanSound8K from ``danavery/urbansound8K``, AISHELL-1 test and dev from ``carlot/AIShell``); those bytes
are written out unchanged and the original directory layout is rebuilt around them (AISHELL-1 files that soundfile
cannot open as a container are decoded and rewritten as WAV; ``aae.prepare``'s digest check catches any difference).
The FSD50K development clips (a training pool only) come from the ``Fhrozen/FSD50k`` mirror as individual files, and
every other corpus from its publisher.

Every corpus is skipped once its marker file exists, so an interrupted run resumes. Needs wget, tar, unzip and 7z.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ZENODO_FSD = "https://zenodo.org/records/4060432/files"
FSD_EVAL = {"FSD50K.eval_audio.z01": "3090670eaeecc013ca1ff84fe4442aeb",
            "FSD50K.eval_audio.zip": "6fa47636c3a3ad5c7dfeba99f2637982",
            "FSD50K.ground_truth.zip": "ca27382c195e37d2269c4c866dd73485"}
ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
WHAM_URL = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
OPENSLR = "https://www.openslr.org/resources"
LIBRISPEECH = ("test-clean", "test-other", "dev-clean", "dev-other", "train-clean-100")
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
DEMAND_RECORD = "https://zenodo.org/api/records/1227121"
US8K_COLUMNS = ("slice_file_name", "fsID", "start", "end", "salience", "fold", "classID", "class")
CORPORA = ("urbansound8k", "wham", "fsd50k", "librispeech", "fleurs", "aishell", "musan", "esc50", "demand")
# Hugging Face datasets at the revisions the paper's inputs were built from
HF_REVISIONS = {"danavery/urbansound8K": "8aa9177a0c5a6949ee4ee4b7fcabb01dfd4ae466",
                "carlot/AIShell": "692ef58020d79b21f54eb25b15a4813d4f9650d7",
                "Fhrozen/FSD50k": "ccf1acaff12f3f4a4c10052dddafbb6d22152c9b",
                "google/fleurs": "70bb2e84b976b7e960aa89f1c648e09c59f894dd"}
TOOLS = {"wham": ("wget", "unzip"), "fsd50k": ("wget", "unzip", "7z"), "librispeech": ("wget", "tar"),
         "fleurs": ("tar",), "musan": ("wget", "tar"), "esc50": ("wget", "unzip"), "demand": ("wget", "unzip")}


def sh(*cmd):
    print("|   $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest: Path, md5sum: str = "") -> Path:
    ok = dest.with_name(dest.name + ".ok")
    if ok.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    sh("wget", "-c", "-nv", "--tries=20", "--timeout=120", "-O", dest, url)
    if md5sum and md5(dest) != md5sum:
        dest.unlink()
        raise SystemExit(f"md5 mismatch for {dest.name}; deleted, run again")
    ok.touch()
    return dest


def count(root: Path, ext: str) -> int:
    return sum(1 for _ in root.rglob(f"*.{ext}"))


def expect(n: int, want: int, what: str):
    if n != want:
        raise SystemExit(f"{what}: found {n} files, expected {want}")


def hf_snapshot(repo, dest: Path, patterns):
    from huggingface_hub import snapshot_download

    snapshot_download(repo, repo_type="dataset", revision=HF_REVISIONS[repo], allow_patterns=patterns,
                      local_dir=str(dest), max_workers=8)


def write_new(path: Path, data: bytes):
    """Write a file that is not there yet, atomically: an interrupted run never leaves half a file behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ corpora
def urbansound8k(root: Path, arch: Path):
    import pyarrow.parquet as pq

    out = root / "UrbanSound8K"
    src = arch / "hf_urbansound8k"
    hf_snapshot("danavery/urbansound8K", src, ["data/*.parquet"])
    rows = []
    for f in sorted((src / "data").glob("*.parquet")):
        for batch in pq.read_table(f).to_batches(max_chunksize=256):
            d = batch.to_pydict()
            for i, name in enumerate(d["slice_file_name"]):
                wav = out / "audio" / f"fold{int(d['fold'][i])}" / name
                wav.parent.mkdir(parents=True, exist_ok=True)
                if not wav.exists():
                    write_new(wav, d["audio"][i]["bytes"])        # the original file, byte for byte
                rows.append({k: d[k][i] for k in US8K_COLUMNS})
    (out / "metadata").mkdir(parents=True, exist_ok=True)
    with open(out / "metadata" / "UrbanSound8K.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    expect(count(out / "audio", "wav"), 8732, "UrbanSound8K")


def wham(root: Path, arch: Path):
    z = fetch(WHAM_URL, arch / "wham_noise.zip")
    sh("unzip", "-q", "-o", z, "-d", root / "WHAM_noise")
    base = root / "WHAM_noise" / "wham_noise"
    for split, n in (("tr", 20000), ("cv", 5000), ("tt", 3000)):
        expect(count(base / split, "wav"), n, f"WHAM! {split}")


def fsd50k(root: Path, arch: Path):
    out = root / "FSD50K"
    for name, digest in FSD_EVAL.items():
        fetch(f"{ZENODO_FSD}/{name}?download=1", arch / name, digest)
    sh("7z", "x", "-y", f"-o{out}", arch / "FSD50K.eval_audio.zip")
    sh("unzip", "-q", "-o", arch / "FSD50K.ground_truth.zip", "-d", out)
    fetch(ONTOLOGY_URL, out / "ontology.json")
    expect(count(out / "FSD50K.eval_audio", "wav"), 10231, "FSD50K eval")
    dev = arch / "hf_fsd50k"
    hf_snapshot("Fhrozen/FSD50k", dev, ["clips/dev/*"])
    target = out / "FSD50K.dev_audio"
    if not target.exists():
        shutil.move(str(dev / "clips" / "dev"), str(target))         # --archives may be on another file system
    expect(count(target, "wav"), 40966, "FSD50K dev")


def librispeech(root: Path, arch: Path):
    for split in LIBRISPEECH:
        t = fetch(f"{OPENSLR}/12/{split}.tar.gz", arch / f"librispeech-{split}.tar.gz")
        sh("tar", "-xzf", t, "-C", root)
    for split, n in zip(LIBRISPEECH, (2620, 2939, 2703, 2864, 28539), strict=True):
        expect(count(root / "LibriSpeech" / split, "flac"), n, f"LibriSpeech {split}")


def fleurs(root: Path, arch: Path):
    out = root / "FLEURS_zh"
    hf_snapshot("google/fleurs", out, ["data/cmn_hans_cn/*"])
    for split in ("test", "train"):
        (out / f"audio_{split}").mkdir(parents=True, exist_ok=True)
        sh("tar", "-xzf", out / "data" / "cmn_hans_cn" / "audio" / f"{split}.tar.gz", "-C", out / f"audio_{split}")
    for split, n in (("test", 945), ("train", 3246)):
        expect(count(out / f"audio_{split}", "wav"), n, f"FLEURS Mandarin {split}")


def aishell(root: Path, arch: Path):
    """Test and dev from the carlot/AIShell parquet mirror, rebuilt as openslr's data_aishell layout."""
    import pyarrow.parquet as pq
    import soundfile as sf

    src = arch / "hf_aishell"
    hf_snapshot("carlot/AIShell", src, ["data/test-*.parquet", "data/validation-*.parquet"])
    out = root / "AISHELL-1" / "data_aishell"
    utt_re = re.compile(r"(BAC009S\d{4}W\d{4})")
    lines, counts = [], {}
    for split, sub in (("test", "test"), ("validation", "dev")):
        n = 0
        for f in sorted((src / "data").glob(f"{split}-*.parquet")):
            for batch in pq.read_table(f).to_batches(max_chunksize=256):
                d = batch.to_pydict()
                for i, text in enumerate(d["transcription"]):
                    a = d["audio"][i]
                    utt = utt_re.search(a.get("path") or "").group(1)
                    wav = out / "wav" / sub / utt[6:11] / f"{utt}.wav"
                    wav.parent.mkdir(parents=True, exist_ok=True)
                    if not wav.exists():
                        try:
                            sf.info(io.BytesIO(a["bytes"]))
                            write_new(wav, a["bytes"])            # the original file, byte for byte
                        except Exception:  # noqa: BLE001 (not a container soundfile can read as is)
                            y, sr = sf.read(io.BytesIO(a["bytes"]))
                            tmp = wav.with_name(wav.name + ".tmp.wav")
                            sf.write(tmp, y, sr)
                            os.replace(tmp, wav)
                    lines.append(f"{utt} {' '.join(text.split())}")
                    n += 1
        counts[sub] = n
    (out / "transcript").mkdir(parents=True, exist_ok=True)
    (out / "transcript" / "aishell_transcript_v0.8.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    expect(counts.get("test", 0), 7176, "AISHELL-1 test")
    expect(counts.get("dev", 0), 14326, "AISHELL-1 dev")


def musan(root: Path, arch: Path):
    t = fetch(f"{OPENSLR}/17/musan.tar.gz", arch / "musan.tar.gz")
    (root / "MUSAN").mkdir(parents=True, exist_ok=True)
    sh("tar", "-xzf", t, "-C", root / "MUSAN")
    expect(count(root / "MUSAN" / "musan" / "noise", "wav"), 930, "MUSAN noise")
    expect(count(root / "MUSAN" / "musan" / "music", "wav"), 660, "MUSAN music")


def esc50(root: Path, arch: Path):
    z = fetch(ESC50_URL, arch / "ESC-50-master.zip")
    sh("unzip", "-q", "-o", z, "-d", root / "ESC-50")
    expect(count(root / "ESC-50", "wav"), 2000, "ESC-50")


def demand(root: Path, arch: Path):
    import urllib.request

    files = json.load(urllib.request.urlopen(DEMAND_RECORD))["files"]
    for f in files:
        if f["key"].endswith("_16k.zip"):
            z = fetch(f["links"]["self"], arch / f"DEMAND_{f['key']}", f.get("checksum", "").replace("md5:", ""))
            sh("unzip", "-q", "-o", z, "-d", root / "DEMAND")
    expect(count(root / "DEMAND", "wav"), 272, "DEMAND")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--archives", default="", help="where archives are kept (default: <data-root>/_archives)")
    ap.add_argument("--only", default="", help=f"comma-separated, from {', '.join(CORPORA)}")
    args = ap.parse_args()
    root = Path(args.data_root).resolve()
    arch = Path(args.archives).resolve() if args.archives else root / "_archives"
    arch.mkdir(parents=True, exist_ok=True)
    todo = [c for c in args.only.split(",") if c] or list(CORPORA)
    missing = sorted({t for c in todo for t in TOOLS.get(c, ()) if not shutil.which(t)})
    if missing:
        raise SystemExit(f"needs {', '.join(missing)} on PATH for {', '.join(todo)}")
    for c in todo:
        if c not in CORPORA:
            raise SystemExit(f"unknown corpus {c!r}")
        marker = root / f".{c}.done"
        if marker.exists():
            print(f"| {c}: done already")
            continue
        print(f"| {c}", flush=True)
        globals()[c](root, arch)
        marker.touch()
    print(f"| corpora ready under {root}; next: python -m aae.prepare --data-root {root}")


if __name__ == "__main__":
    sys.exit(main())
