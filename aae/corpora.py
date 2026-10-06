"""Readers for the public corpora. Each returns ``{clip id: {"path", "ref", "meta"}}`` for one split.

Clip ids are the ones ``assets/manifests`` refers to; nothing here decides which clips are used or in what order,
the shipped manifests do. ``DATA_LAYOUT`` is where each corpus is expected under the data root, which is the
layout ``scripts/fetch_data.py`` produces.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

DATA_LAYOUT = {
    "urbansound8k": "UrbanSound8K",                  # audio/fold{1..10}/*.wav, metadata/UrbanSound8K.csv
    "wham": "WHAM_noise/wham_noise",                 # {tr,cv,tt}/*.wav
    "fsd50k": "FSD50K",                              # FSD50K.{eval,dev}_audio/*.wav, FSD50K.ground_truth/*.csv
    "librispeech": "LibriSpeech",                    # {test-clean,...}/<spk>/<chapter>/*.flac
    "fleurs": "FLEURS_zh",                           # data/cmn_hans_cn/{split}.tsv, audio_{split}/{split}/*.wav
    "aishell": "AISHELL-1/data_aishell",             # wav/{test,dev}/<spk>/*.wav, transcript/*_v0.8.txt
    "musan": "MUSAN/musan",                          # {noise,music}/**/*.wav
    "esc50": "ESC-50",                               # **/meta/esc50.csv, **/audio/*.wav
    "demand": "DEMAND",                              # <ENV>/ch01.wav ... (16 kHz release)
}
ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
HUMAN_VOICE, MUSIC = "/m/09l8g", "/m/04rlf"         # AudioSet ontology roots excluded from filtered FSD50K


def _entry(path, ref="", **meta) -> dict:
    return {"path": str(path), "ref": ref, "meta": meta}


def urbansound8k(root) -> dict:
    root = Path(root)
    out = {}
    with open(root / "metadata" / "UrbanSound8K.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["slice_file_name"]] = _entry(root / "audio" / f"fold{int(r['fold'])}" / r["slice_file_name"],
                                               cls=r["class"], fsid=int(r["fsID"]))
    return out


def wham(root, split: str) -> dict:
    return {f"{split}/{p.name}": _entry(p) for p in sorted((Path(root) / split).glob("*.wav"))}


def excluded_mids(ontology_path) -> set:
    """Every AudioSet id under Human voice or Music, inclusive: the filtered-FSD50K rule."""
    nodes = {n["id"]: n for n in json.loads(Path(ontology_path).read_text(encoding="utf-8"))}
    out, stack = set(), [HUMAN_VOICE, MUSIC]
    while stack:
        mid = stack.pop()
        if mid in out or mid not in nodes:
            continue
        out.add(mid)
        stack.extend(nodes[mid].get("child_ids") or [])
    return out


def fsd50k(root, split: str, ontology_path=None) -> dict:
    """``meta.filtered`` is True when no label falls under Human voice or Music, and None without the AudioSet ontology
    (staging does not need it; the shipped manifests already list the filtered clips)."""
    root = Path(root)
    ontology = Path(ontology_path or root / "ontology.json")
    excluded = excluded_mids(ontology) if ontology.exists() else None
    out = {}
    with open(root / "FSD50K.ground_truth" / f"{split}.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            mids = [m for m in r["mids"].split(",") if m]
            out[r["fname"]] = _entry(root / f"FSD50K.{split}_audio" / f"{r['fname']}.wav",
                                     labels=[x for x in r["labels"].split(",") if x],
                                     filtered=None if excluded is None else not any(m in excluded for m in mids))
    return out


def librispeech(root, split: str) -> dict:
    out = {}
    for trans in sorted((Path(root) / split).rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt, _, text = line.strip().partition(" ")
            if utt:
                out[utt] = _entry(trans.parent / f"{utt}.flac", text.strip(), speaker=utt.split("-")[0])
    return out


def fleurs(root, split: str) -> dict:
    """Mandarin FLEURS. One sentence is read by several speakers, so the id pairs sentence id and recording."""
    root = Path(root)
    audio = root / f"audio_{split}"
    out = {}
    with open(root / "data" / "cmn_hans_cn" / f"{split}.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            path = audio / split / parts[1]
            if not path.exists():
                hits = list(audio.rglob(parts[1]))
                path = hits[0] if hits else path
            out[f"{parts[0]}_{Path(parts[1]).stem}"] = _entry(path, parts[3].strip())
    return out


def aishell(root, split: str) -> dict:
    root = Path(root)
    text = {}
    with open(root / "transcript" / "aishell_transcript_v0.8.txt", encoding="utf-8") as f:
        for line in f:
            utt, _, t = line.strip().partition(" ")
            if utt:
                text[utt] = t.replace(" ", "")
    return {w.stem: _entry(w, text[w.stem]) for w in sorted((root / "wav" / split).rglob("*.wav")) if w.stem in text}


def musan(root, kind: str) -> dict:
    return {p.stem: _entry(p) for p in sorted((Path(root) / kind).rglob("*.wav"))}


def esc50(root) -> dict:
    meta = next(Path(root).rglob("esc50.csv"))
    audio = meta.parent.parent / "audio"
    with open(meta, newline="", encoding="utf-8") as f:
        return {r["filename"]: _entry(audio / r["filename"], fsid=int(r["src_file"])) for r in csv.DictReader(f)}


def demand(root) -> dict:
    return {f"{p.parent.name}/{p.stem}": _entry(p) for p in sorted(Path(root).rglob("*.wav"))}


def catalog(data_root, source: str) -> dict:
    """Clips of one manifest source, e.g. ``librispeech:test-clean`` or ``musan:noise``."""
    corpus, _, split = source.partition(":")
    root = Path(data_root) / DATA_LAYOUT[corpus]
    readers = {"urbansound8k": lambda: urbansound8k(root), "esc50": lambda: esc50(root), "demand": lambda: demand(root),
               "wham": lambda: wham(root, split), "fsd50k": lambda: fsd50k(root, split),
               "librispeech": lambda: librispeech(root, split), "fleurs": lambda: fleurs(root, split),
               "aishell": lambda: aishell(root, split), "musan": lambda: musan(root, split)}
    return readers[corpus]()
