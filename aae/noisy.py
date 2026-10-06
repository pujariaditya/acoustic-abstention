"""The noisy speech sets, rebuilt from the shipped recipe (``assets/manifests/complete/{tail,snr5}*.jsonl``).

Each item pairs one LibriSpeech test utterance with one noise clip and records where the noise segment starts, so
construction is deterministic and needs no random state. ``tail`` and ``snr5`` use noise from the training pool
(WHAM! training noise or MUSAN noise); ``tail_heldout`` and ``snr5_heldout`` mix the same 1,000 utterances with WHAM!
noise from its evaluation splits, which no model saw in training.

``tail``  the utterance, 1 s of silence, then 3 s of noise at the utterance's RMS level;
``snr5``  the utterance with noise added at 5 dB SNR.

The arithmetic below is written exactly as it ran, including its dtype promotions, and each result goes through a
16-bit WAV file before staging, as the originals did. ``python -m aae.prepare`` compares every rebuilt clip with the
digest in the recipe.
"""
from __future__ import annotations

import numpy as np

from aae.audio import SAMPLE_RATE

TAIL_GAP_S, TAIL_NOISE_S, SNR_DB = 1.0, 3.0, 5.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if len(x) else 0.0


def noise_segment(noise: np.ndarray, n: int, start: int) -> np.ndarray:
    """``n`` samples of ``noise`` from ``start``, tiling the clip first when it is shorter than ``n``."""
    if len(noise) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    return noise[start:start + n].astype(np.float32)


def segment_length(kind: str, n_speech: int) -> int:
    return int(round(TAIL_NOISE_S * SAMPLE_RATE)) if kind == "tail" else n_speech


def tail(speech: np.ndarray, noise: np.ndarray, start: int) -> np.ndarray:
    speech = np.asarray(speech, dtype=np.float32)
    gap = np.zeros(int(round(TAIL_GAP_S * SAMPLE_RATE)), dtype=np.float32)
    seg = noise_segment(np.asarray(noise, dtype=np.float32), int(round(TAIL_NOISE_S * SAMPLE_RATE)), start)
    rs, rn = _rms(speech), _rms(seg)
    if rs > 0 and rn > 0:
        seg = seg * (rs / rn)
    return np.clip(np.concatenate([speech, gap, seg]), -1.0, 1.0).astype(np.float32)


def snr5(speech: np.ndarray, noise: np.ndarray, start: int) -> np.ndarray:
    speech = np.asarray(speech, dtype=np.float32)
    n = noise_segment(np.asarray(noise, dtype=np.float32), len(speech), start)
    ps = float(np.mean(np.square(speech, dtype=np.float64)))
    pn = float(np.mean(np.square(n, dtype=np.float64)))
    if ps > 0 and pn > 0:
        n = n * np.sqrt(ps / (pn * 10 ** (SNR_DB / 10.0)))
    return np.clip(speech + n, -1.0, 1.0).astype(np.float32)


MIXERS = {"tail": tail, "snr5": snr5, "tail_heldout": tail, "snr5_heldout": snr5}


def write_wav(path, y: np.ndarray) -> None:
    import soundfile as sf

    sf.write(str(path), y, SAMPLE_RATE, subtype="PCM_16")
