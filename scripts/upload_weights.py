#!/usr/bin/env python3
"""Publish the released checkpoints and the model card to the Hugging Face hub.

    HF_TOKEN=... python scripts/upload_weights.py --src REL                       # a new repository is private
    HF_TOKEN=... python scripts/upload_weights.py --src REL --card-only           # the card and licenses only
    HF_TOKEN=... python scripts/upload_weights.py --src REL --card-only --public  # ... and make it public
    python scripts/upload_weights.py --src REL --dry-run                          # print, upload nothing

Maintainer tool; users want ``download_weights.py``. ``--src`` (REL) holds ``<model>/<variant>/{model.safetensors,
training.json}``; the card reads each checkpoint's parameter count from there. Every model.safetensors is hashed
first, and the upload stops unless every digest equals the one compiled into ``aae/weights.py``, so what is published
is exactly what users verify against. ``LICENSE.md`` goes up with the card: which license each folder follows, and the
full MIT and Apache-2.0 texts (``assets/licenses``).

The token is read from the environment only; it is never written to a file or included in an upload.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aae import models, report, weights  # noqa: E402

TITLE = "Learning When to Stay Silent: Acoustic Abstention Encoding for Automatic Speech Recognition"

SPEECH_LEGS = ("ls_clean", "ls_other", "tail", "snr5", "fleurs_zh", "aishell")
ABLATION = {"aae": "AAE", "no-preservation": "λ = 0: no preservation loss",
            "transcript-ce": "CE on m: transcript cross-entropy instead",
            "independent-noise": "indep. noise: another segment in m",
            "decoder-adapter": "dec. adapters: CE-trained, in the decoder"}


def card(src) -> str:
    """The model card; every number is read from assets/paper_results.json, the numbers `aae.report` checks."""
    E = json.loads((ROOT / "assets" / "paper_results.json").read_text())
    hr_rows, sp_rows = [], []
    for m in models.MODEL_NAMES:
        t = models.spec(m).title
        hr_rows.append(f"| {t} | " + " | ".join(f"{E[f'hr/{m}/{leg}/original']} → **{E[f'hr/{m}/{leg}/aae']}**"
                                                  for leg in ("urban", "wham", "fsd50k")) + " |")
        sp_rows.append(f"| {t} | " + " | ".join(f"{E[f'speech/{m}/{leg}/original']} → {E[f'speech/{m}/{leg}/aae']}"
                                                  for leg in SPEECH_LEGS) + f" | {E[f'ablation/{m}/aae/same']} |")
    ab_rows = []
    for v, label in ABLATION.items():
        cells = []
        for m in models.MODEL_NAMES:
            ref = report.reference(m) if v == "aae" else v
            c = f"{E[f'ablation/{m}/{ref}/hr']} / {E[f'ablation/{m}/{ref}/same']}"
            beyond = E[f"ablation/{m}/{ref}/beyond"]
            cells.append(f"**{c}**" if beyond == "0" else f"{c} ({beyond})")
        ab_rows.append(f"| {label} | " + " | ".join(cells) + " |")
    ck_rows = []
    for key in weights.CHECKPOINTS:
        m, v = key.split("/")
        t = json.loads((Path(src) / key / "training.json").read_text())
        size = f"{t['trained_params'] / 1e6:.2f}M"
        ck_rows.append(f"| `{key}/` | {weights.VARIANTS[v]} | {size} |")
    titles = " | ".join(models.spec(m).title for m in models.MODEL_NAMES)
    return f"""---
license: other
license_name: mit-and-apache-2.0
license_link: LICENSE.md
library_name: pytorch
pipeline_tag: automatic-speech-recognition
language:
  - en
  - zh
base_model:
  - openai/whisper-large-v3
  - Qwen/Qwen3-ASR-1.7B-hf
  - Edge0/ARK-ASR-3B
  - HojoAI/Hojo-ASR-V1
tags:
  - speech-recognition
  - hallucination
  - non-speech
  - whisper
  - audio-llm
  - lora
  - robustness
metrics:
  - wer
  - cer
---

# acoustic-abstention

**{TITLE}**

Aditya Pujari and Ajita Rattani, University of North Texas

Released checkpoints for [acoustic-abstention](https://github.com/pujariaditya/acoustic-abstention). Acoustic
Abstention Encoding (AAE) fine-tunes a pretrained ASR model's audio encoder while its text decoder stays **frozen**: an
abstention loss makes the decoder return an empty transcript for non-speech, and a feature-preservation loss keeps the
encoder's features of speech-containing audio close to the original encoder's. Training needs no speech transcripts.
The LoRA adapters, in the last half of the encoder blocks and, for the audio-LLMs, in the audio projector, are merged
into the weights: a checkpoint has the original architecture and parameter count, and decodes with the same procedure
as the original model. Every number below is measured on complete evaluation sets.

Hallucination rate (HR, %: non-speech segments that receive any text) on the complete non-speech sets, original model →
with AAE (Whisper large-v3: 9,250 steps; Hojo-ASR-V1: the run with higher rates):

| model | UrbanSound8K (8,732) | WHAM! noise (8,000) | filtered FSD50K (6,414) |
|---|---|---|---|
{chr(10).join(hr_rows)}

Speech error rate on the complete test sets (WER; CER for the Mandarin sets), original model → with AAE, and *same*:
the share of the model's 15,680 normalized transcripts identical to the original model's:

| model | LibriSpeech clean | other | tail | 5 dB | FLEURS Mandarin | AISHELL-1 | same |
|---|---|---|---|---|---|---|---|
{chr(10).join(sp_rows)}

*tail*: a LibriSpeech test utterance, 1 s of silence, then 3 s of noise; *5 dB*: noise added at 5 dB SNR.

Ablation study on the complete sets, every row at 8,500 steps (Whisper large-v3: `aae-8500`; Hojo-ASR-V1: the other
run, `aae-run1`): mean HR over the three non-speech sets (%) / *same* (%), and in parentheses the number of the six
speech sets whose error rate exceeds the original model's by more than the tolerance fixed in advance (+0.3 WER
test-clean, +0.5 test-other, +1.0 on each noisy set, +1.0 CER on each Mandarin set); bold: every speech set within its
tolerance.

| variant | {titles} |
|---|---|---|---|---|
{chr(10).join(ab_rows)}

## Checkpoints

Twenty-four checkpoints: the released model for each ASR model, the four variants of the ablation study for each,
the two full-recipe rows of the ablation that differ from the released model (the other Hojo-ASR-V1 run, `aae-run1`,
and Whisper large-v3 at 8,500 steps, `aae-8500`), and the decoder-adaptation control: Whisper large-v3's decoder adapted
at AAE's parameter count (LoRA of rank 18 in the decoder's self-attention) with a KL term at two weights (`decoder-kl`,
lambda = 50, and `decoder-kl-1`, lambda = 1). Together they cover every evaluation behind
the paper and the repository's results.

| folder | what it is | adapter parameters |
|---|---|---|
{chr(10).join(ck_rows)}

Each folder holds `model.safetensors`, the merged float16 weights of every tensor the adapter changed (the exact
tensors that were evaluated), and `training.json`, the training arguments, loss curve and held-out curve.
`hojo-asr-v1/aae` is the second of two runs of the recipe; the first, `aae-run1`, is the full-recipe row of the
ablation study because its sampling matches the variants. `whisper-large-v3/aae` trained for 9,250 steps;
`whisper-large-v3/aae-8500` is the same run at 8,500 steps, the length of every variant, and is that model's
full-recipe row.

## Use

```bash
git clone https://github.com/pujariaditya/acoustic-abstention && cd acoustic-abstention
./setup.sh                                   # two environments + the four original ASR models, pinned
envs/main/bin/python scripts/download_weights.py     # these checkpoints, sha256-verified
```

Applying a checkpoint copies its tensors into the original model:

```python
from aae import models, weights
ad = models.load("whisper-large-v3")                           # original model, pinned revision
weights.apply(ad.model, weights.resolve("whisper-large-v3/aae"))
print(ad.transcribe([waveform_16k]))                           # [''] for non-speech
```

The Whisper checkpoints use the parameter names of openai-whisper's `large-v3.pt` (sha256 e5b1a55b...), not the
transformers layout. The other base models are pinned to Qwen/Qwen3-ASR-1.7B-hf @ bcd2b5b7, Edge0/ARK-ASR-3B @
1e28271b and HojoAI/Hojo-ASR-V1 @ a22c3818. `./scripts/reproduce.sh` re-runs every evaluation behind the paper and
checks each number against the paper's value.

## Intended use and limitations

Research artifact for reproducing the reported results. The recipe preserves each original model's speech behavior
rather than improving it. When every utterance of LibriSpeech test-other and FLEURS Mandarin test is mixed with
training-pool noise at -5 dB SNR, {E['stress/aae/-5dB/empty_min']}% to {E['stress/aae/-5dB/empty_max']}% receive an
empty output with AAE, against at most {E['stress/original/-5dB/empty_max']}% for the original models; at 5 and 15 dB,
at most {E['stress/aae/5_15dB/empty_max']}% do. The noisy speech test sets use noise from the training pools. One
training run per model and variant (two runs of the full recipe for Hojo-ASR-V1, both released).

## Licenses

Each checkpoint is a derivative of its base model and is distributed under that model's license: Whisper large-v3
(MIT), Qwen3-ASR-1.7B, ARK-ASR-3B and Hojo-ASR-V1 (Apache-2.0); `LICENSE.md` has the folder-by-folder terms and the
full texts. The checkpoints were trained on audio from WHAM! noise, FSD50K, MUSAN, ESC-50, DEMAND, LibriSpeech, FLEURS
and AISHELL-1, each under its own terms; some are non-commercial (ESC-50 is CC BY-NC 3.0, and FSD50K clips carry
per-clip Creative Commons licenses, some of them CC BY-NC). The code is MIT.
"""


def license_md() -> str:
    lic = ROOT / "assets" / "licenses"
    return f"""# Licenses of the released checkpoints

Each checkpoint is a derivative of its base model: the base model's weights with LoRA adapters merged into the
tensors listed in the folder's `model.safetensors` (the changes are described in `training.json` and in the model
card). Each folder is distributed under its base model's license:

| folders | base model | license |
|---|---|---|
| `whisper-large-v3/*` | Whisper large-v3 (OpenAI) | MIT, text below |
| `qwen3-asr-1.7b/*` | Qwen3-ASR-1.7B (Qwen) | Apache License 2.0, text below |
| `ark-asr-3b/*` | ARK-ASR-3B (Edge0) | Apache License 2.0, text below |
| `hojo-asr-v1/*` | Hojo-ASR-V1 (HojoAI) | Apache License 2.0, text below |

The code that trains and evaluates them (https://github.com/pujariaditya/acoustic-abstention) is MIT-licensed.

## MIT License (Whisper large-v3)

```
{(lic / "MIT-whisper.txt").read_text().rstrip()}
```

## Apache License 2.0 (Qwen3-ASR-1.7B, ARK-ASR-3B, Hojo-ASR-V1)

```
{(lic / "Apache-2.0.txt").read_text().rstrip()}
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True,
                    help="release directory: <model>/<variant>/{model.safetensors,training.json}")
    ap.add_argument("--repo", default=weights.HF_REPO)
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help="make the repository public")
    vis.add_argument("--private", action="store_true", help="make the repository private")
    ap.add_argument("--card-only", action="store_true",
                    help="refresh the model card and licenses, upload no checkpoints")
    ap.add_argument("--dry-run", action="store_true", help="verify digests and print the card, upload nothing")
    args = ap.parse_args()
    if not args.card_only:
        for key, want in weights.CHECKPOINTS.items():
            f = Path(args.src) / key / "model.safetensors"
            got = weights.sha256_of(f)
            if got != want:
                raise SystemExit(f"{key}: sha256 {got} differs from the pinned {want}; refusing to publish")
            print(f"| {key:36s} {f.stat().st_size / 1e9:5.2f} GB  sha256 ok")
    text, lic = card(args.src), license_md()
    if args.dry_run:
        print(text)
        print(f"| LICENSE.md: {len(lic)} characters")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN in the environment")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)    # new repositories only
    if not args.card_only:
        api.upload_large_folder(repo_id=args.repo, repo_type="model", folder_path=args.src,
                                allow_patterns=["*/*/model.safetensors", "*/*/training.json"])
    api.upload_file(path_or_fileobj=lic.encode(), path_in_repo="LICENSE.md", repo_id=args.repo, repo_type="model")
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=args.repo, repo_type="model")
    if args.public or args.private:                                  # otherwise the visibility stays as it is
        api.update_repo_settings(args.repo, private=args.private)
    private = api.repo_info(args.repo, repo_type="model").private
    print(f"| done: https://huggingface.co/{args.repo} ({'private' if private else 'public'})")


if __name__ == "__main__":
    main()
