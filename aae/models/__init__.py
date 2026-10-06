"""The four recognizers, pinned to the revisions every reported number was produced with.

Two Python environments are needed, because the models' own code needs incompatible transformers releases:

``main``    Whisper large-v3 and Qwen3-ASR-1.7B   (torch 2.9.1, transformers 5.17)
``legacy``  ARK-ASR-3B and Hojo-ASR-V1           (torch 2.5.1, transformers 4.57, hojo-asr 0.1.3)

``setup.sh`` creates both and downloads the weights into ``models/``.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

MODELS_ROOT = Path(os.environ.get("AAE_MODELS", Path(__file__).resolve().parents[2] / "models"))
_UPPER16 = r"(1[6-9]|2\d|3[01])"          # the upper 16 of 32 encoder layers
_ENC = r"(self_attn\.(q|k|v|out)_proj|fc1|fc2)$"


@dataclass(frozen=True)
class Spec:
    name: str                  # public name, used on the command line and on the Hugging Face hub
    title: str                 # name in tables
    source: str                # where the original weights come from
    revision: str              # commit (hub) or sha256 (openai-whisper) the results were produced with
    env: str                   # "main" or "legacy"
    batch: int                 # evaluation batch size; batches are part of the protocol (padding is per batch)
    encoder_locus: str         # linear layers AAE adapts: upper encoder layers and the projector
    decoder_locus: str         # decoder self-attention, for the decoder-adapter variant
    features: str              # module whose output the decoder reads, compared by the preservation loss
    lm_checkpointing: bool     # gradient checkpointing in the frozen decoder (memory)
    local_dir: str             # directory under models/


SPECS = {s.name: s for s in (
    Spec("whisper-large-v3", "Whisper large-v3", "openai-whisper:large-v3",
         "e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb", "main", 12,
         rf"^encoder\.blocks\.{_UPPER16}\.(attn\.(query|key|value|out)|mlp\.(0|2))$",
         r"^decoder\.blocks\.\d+\.attn\.(query|key|value|out)$",
         "encoder", False, "whisper"),
    Spec("qwen3-asr-1.7b", "Qwen3-ASR-1.7B", "Qwen/Qwen3-ASR-1.7B-hf",
         "bcd2b5b7f32b480ab5790554cfa8347f246a14f3", "main", 16,
         r"^model\.audio_tower\.layers\.(1[2-9]|2[0-3])\." + _ENC + r"|^model\.multi_modal_projector\.linear_(1|2)$",
         r"language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
         "model.multi_modal_projector", False, "Qwen3-ASR-1.7B-hf"),
    Spec("ark-asr-3b", "ARK-ASR-3B", "Edge0/ARK-ASR-3B",
         "1e28271b79edc97635783bea65abc89195a09ed3", "legacy", 12,
         rf"^audio_encoder\.whisper\.layers\.{_UPPER16}\." + _ENC + r"|^audio_encoder\.adapting\.(0|2)$",
         r"^model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
         "audio_encoder", True, "ARK-ASR-3B"),
    Spec("hojo-asr-v1", "Hojo-ASR-V1", "HojoAI/Hojo-ASR-V1",
         "a22c381896ce5f4b70038982bfb888fe9969ed99", "legacy", 8,
         rf"^speech_encoder\.layers\.{_UPPER16}\." + _ENC + r"|^speech_encoder\.proj(1|2)$"
         r"|^bottleneck\..*(linear_(q|k|v|out|pos)|w_1|w_2|embed\.out\.0)$",
         r"^decoder_model\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
         "ln_speech", True, "Hojo-ASR-V1"),
)}
MODEL_NAMES = tuple(SPECS)


def spec(name: str) -> Spec:
    if name not in SPECS:
        raise SystemExit(f"unknown model {name!r}; expected one of {', '.join(MODEL_NAMES)}")
    return SPECS[name]


def local_path(name: str) -> str:
    s = spec(name)
    p = MODELS_ROOT / s.local_dir
    return str(p / "large-v3.pt") if s.name == "whisper-large-v3" else str(p)


def cuda() -> str:
    """The device of every reported number. The protocol decodes in float16 on a GPU, so there is no CPU fallback.
    ``AAE_GPU_FRACTION`` caps the share of the GPU's memory this process may use (for shared GPUs)."""
    if not torch.cuda.is_available():
        raise SystemExit("this step needs a CUDA GPU (the protocol decodes in float16); none is visible")
    if os.environ.get("AAE_GPU_FRACTION"):
        torch.cuda.set_per_process_memory_fraction(float(os.environ["AAE_GPU_FRACTION"]), 0)
    return "cuda:0"


def load(name: str, device: str = "cuda:0"):
    """Load an original recognizer in inference mode. Checkpoints are applied afterwards (``aae.weights.apply``)."""
    here = "main" if sys.version_info >= (3, 12) else "legacy"
    if spec(name).env != here:
        raise SystemExit(f"{name} runs in envs/{spec(name).env}, not in this Python "
                         f"{sys.version_info.major}.{sys.version_info.minor} (envs/{here}): "
                         f"use envs/{spec(name).env}/bin/python")
    # The reported numbers were produced with cuDNN disabled (native convolution kernels); keep it that way so the
    # convolutional front ends compute exactly what they computed then.
    torch.backends.cudnn.enabled = False
    s = spec(name)
    path = local_path(name)
    if not Path(path).exists():
        raise SystemExit(f"{path} is missing: run setup.sh first")
    if s.name == "whisper-large-v3":
        from aae.models.whisper import WhisperAdapter
        return WhisperAdapter(device, path)
    if s.name == "qwen3-asr-1.7b":
        from aae.models.qwen3_asr import Qwen3ASRAdapter
        return Qwen3ASRAdapter(device, path)
    if s.name == "ark-asr-3b":
        from aae.models.ark import ARKAdapter
        return ARKAdapter(device, path)
    from aae.models.hojo import HojoAdapter
    return HojoAdapter(device, path)
