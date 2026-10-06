"""The contract every recognizer adapter implements.

    ad.model                        the torch module whose weights a checkpoint replaces
    ad.transcribe(wavs)             list of float32 16 kHz arrays -> list of str; greedy, fixed prompt, fixed limits
    ad.loss(wavs, texts, langs)     teacher-forced cross-entropy on the target text; "" is the empty transcript

Decoding is owned by the adapters and never changes with a checkpoint: a checkpoint replaces weights only.
"""
from __future__ import annotations

import numpy as np
import torch

SAMPLE_RATE = 16_000


def new_tokens_limit(n_samples: int) -> int:
    """Generation cap from audio length: generous for real speech, bounded so a looping hallucination stops."""
    return int(min(448, max(24, 12 * n_samples / SAMPLE_RATE + 16)))


def target_labels(full: dict, prompt: dict) -> torch.Tensor:
    """Labels on the target span only: the last (len(full) - len(prompt)) real tokens of each row, whichever side the
    processor padded (Qwen3-ASR's processor left-pads even when asked to right-pad)."""
    ids, mask = full["input_ids"], full["attention_mask"]
    n_target = (mask.sum(1) - prompt["attention_mask"].sum(1)).tolist()
    labels = torch.full_like(ids, -100)
    for i, n in enumerate(n_target):
        real = torch.nonzero(mask[i], as_tuple=False).flatten()
        keep = real[len(real) - int(n):]
        labels[i, keep] = ids[i, keep]
    return labels


class Adapter:
    """Base class; see the module docstring for the contract."""

    def __init__(self, device: str):
        self.device = torch.device(device)

    def transcribe(self, wavs: list[np.ndarray]) -> list[str]:
        raise NotImplementedError

    def loss(self, wavs: list[np.ndarray], texts: list[str], langs: list[str] | None = None) -> torch.Tensor:
        raise NotImplementedError
