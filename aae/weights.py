"""Released checkpoints: what they hold, where they live, and how they are applied.

A checkpoint is ``model.safetensors`` plus ``training.json``. The safetensors file holds, for every weight the adapter
changed, the merged value ``W + (alpha / r) B A`` in float16: the exact tensors that were evaluated. Applying a
checkpoint copies those tensors into the original model, so the result has the original architecture and parameter
count.

The 24 released checkpoints are listed in ``CHECKPOINTS`` with their sha256; ``scripts/download_weights.py`` fetches
them from the Hugging Face hub and refuses any model.safetensors whose digest differs from the one compiled in here
(training.json is not pinned).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

HF_REPO = os.environ.get("AAE_HF_REPO", "RootAccess4Life/acoustic-abstention")
ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = Path(os.environ.get("AAE_WEIGHTS", ROOT / "checkpoints"))

# variant -> what it changes relative to the full recipe (see configs/checkpoints.yaml for the exact arguments)
VARIANTS = {
    "aae": "the full recipe",
    "no-preservation": "preservation loss removed (lambda = 0)",
    "transcript-ce": "preservation loss replaced by cross-entropy on m toward the original model's transcript of s",
    "independent-noise": "the noise in m drawn independently of n",
    "decoder-adapter": "rank-9 LoRA on the decoder self-attention instead of the encoder, with transcript CE on m",
    "decoder-kl": "Whisper large-v3 only: rank-18 LoRA on the decoder self-attention, as many parameters as AAE's, with "
                  "the KL divergence to the original model's next-token distributions on m (lambda = 50)",
    "decoder-kl-1": "Whisper large-v3 only: the same decoder edit with the KL weight lambda = 1",
    "aae-run1": "Hojo-ASR-V1 only: the first run of the full recipe (the full-recipe row of the ablation study)",
    "aae-8500": "Whisper large-v3 only: the aae run at 8,500 steps, the length of every variant (the full-recipe "
                "row of the ablation study); aae continues the same run to 9,250 steps",
}

# "<model>/<variant>" -> sha256 of model.safetensors, as released (scripts/upload_weights.py prints these).
CHECKPOINTS = {
    "whisper-large-v3/aae":
        "4365013134ffbc048b89484f47edfdfb2ad46288e4dc3a4e371feb518e9f5ee1",
    "whisper-large-v3/aae-8500":
        "92c13a84a55c1d56e78e531a3aabb9ac08b49139f056c77f2ee8145711913463",
    "whisper-large-v3/no-preservation":
        "c442ff0b1b88c34bf4ac673f149e239d63186ce476ccc04e8cf68919e4116b4a",
    "whisper-large-v3/transcript-ce":
        "a5ba6ac176c720ce6ad615e0abf4eb4079ebf5786713351ac795478e6b16a1f3",
    "whisper-large-v3/independent-noise":
        "6ffb7da6f234e898e2486fbce01d7e59f0f50a13a726398f77b890f00c5c8857",
    "whisper-large-v3/decoder-adapter":
        "243b362e1ee48f139300bce4d6ca4d56b9d8a3c91df68651d970b57660943e83",
    "whisper-large-v3/decoder-kl":
        "9670c1bb045287b0da692c79a7c9640c3ce63f047cfc13ffbb605f75ba1c701e",
    "whisper-large-v3/decoder-kl-1":
        "edb6fd80db95194d1d4438f4ae12789832b6bd7d6a0532353edb4773f2a24a60",
    "qwen3-asr-1.7b/aae":
        "ac26c18585590525b261af7d4b5b909f5b579fdc3e0866210f63817c321c73bf",
    "qwen3-asr-1.7b/no-preservation":
        "3b5bdaa589e0e85429f49110e665358a9227494d021008ee24c1f8f02f37dc9a",
    "qwen3-asr-1.7b/transcript-ce":
        "c991ad9f6d1f3e322f19be34796cebf46ae7ca17e062f245f1f4d71f556beccb",
    "qwen3-asr-1.7b/independent-noise":
        "2984ada92685be8fb25eae865cae51f6879ca478251d023c4eda00c4da2c1749",
    "qwen3-asr-1.7b/decoder-adapter":
        "6a6313387af0a54468636230701c0f764c326d4da0193486f14f11aac00fecab",
    "ark-asr-3b/aae":
        "7c12cc33954a980accaa305e2fa28c0ebcf068b58920c33f68d76fbde6776cc4",
    "ark-asr-3b/no-preservation":
        "f18f433d800e2484707d36658d265fa3971de56dd9dd6510d1022397e15dae5b",
    "ark-asr-3b/transcript-ce":
        "40d94a07133874deb482691178d7d82e7f10dd36a54678d695d8d9114367a3ed",
    "ark-asr-3b/independent-noise":
        "bb57d66ebd9474e6c5358b3ec2d0a13f8beb504fdbc1ef7a275cc33684a009b7",
    "ark-asr-3b/decoder-adapter":
        "ad99ebefd191757e2bad0fccdba5bdedc92ca3a107565956ae07686538f3435e",
    "hojo-asr-v1/aae":
        "97028cf1fc30af68c8125e3b8c76f008f2cc911ef08f8e7eecaa08dbc1a6ac6a",
    "hojo-asr-v1/aae-run1":
        "63099f89996a49ed52173aaa14dca031a72bb0e8bdb37f9b3d8105e55c5f1f3d",
    "hojo-asr-v1/no-preservation":
        "9d059df58845b4eb29b5a297b74ab2f4482fb45444c053b49605edaffd738fa4",
    "hojo-asr-v1/transcript-ce":
        "5bb9fd76438ce2e72e48e17bcec5f12aae973d58fb337410961e9f5ebd0c84de",
    "hojo-asr-v1/independent-noise":
        "53ea0a0d85e1bb780884213a685624c80918f377b7f4e97e3c423bd2a74bf87c",
    "hojo-asr-v1/decoder-adapter":
        "47e2ccd86f5661662e87504a8894ef1ad74c972962b01833d1d44294773efdcb",
}


def sha256_of(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def local_dir(model: str, variant: str) -> Path:
    return WEIGHTS_ROOT / model / variant


def resolve(ref: str) -> Path:
    """A checkpoint reference: ``<model>/<variant>`` (downloaded release) or a directory with model.safetensors."""
    p = Path(ref)
    if (p / "model.safetensors").exists():
        return p
    model, _, variant = ref.partition("/")
    d = local_dir(model, variant)
    if (d / "model.safetensors").exists():
        return d
    raise SystemExit(f"checkpoint {ref!r} not found; fetch it with: python scripts/download_weights.py --only {ref}")


def save(out_dir, tensors: dict, training: dict) -> Path:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"format": "aae-merged", "model": training.get("model", ""), "variant": training.get("variant", "")}
    save_file({k: v.detach().to("cpu", torch.float16).contiguous() for k, v in tensors.items()},
              str(out / "model.safetensors.tmp"), metadata=meta)
    (out / "training.json.tmp").write_text(json.dumps(training, indent=1) + "\n")
    os.replace(out / "model.safetensors.tmp", out / "model.safetensors")    # a killed save never leaves half a file
    os.replace(out / "training.json.tmp", out / "training.json")
    return out


@torch.no_grad()
def apply(model: torch.nn.Module, ckpt_dir) -> dict:
    """Copy a checkpoint's tensors into an original model in place; returns the checkpoint's training record."""
    from safetensors.torch import load_file

    ckpt_dir = Path(ckpt_dir)
    tensors = load_file(str(ckpt_dir / "model.safetensors"))
    params = dict(model.named_parameters())
    unknown = [k for k in tensors if k not in params]
    if unknown:
        raise KeyError(f"{ckpt_dir}: tensors not in this model, e.g. {unknown[:3]}")
    for k, v in tensors.items():
        params[k].copy_(v.to(params[k].dtype))
    record = ckpt_dir / "training.json"
    return json.loads(record.read_text()) if record.exists() else {}
