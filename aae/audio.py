"""Audio I/O, fixed to the conversions the reported numbers were produced with.

Every clip is read once, mixed down to mono, resampled to 16 kHz with soxr (``soxr_hq``) and stored as an int16
``.npy``. Storing multiplies by 32767 and truncates toward zero; loading divides by 32768. The asymmetry is
deliberate: it is what produced the staged audio behind every number in the paper, and ``assets/manifests`` pins a
digest of each staged array so a different conversion is caught rather than silently scored.
"""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
MAX_SECONDS = 30.0


def read_16k(path) -> np.ndarray:
    """float32 mono at 16 kHz from any file soundfile can read."""
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    return np.ascontiguousarray(mono, dtype=np.float32)


def to_int16(y: np.ndarray) -> np.ndarray:
    return (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)


def digest(a: np.ndarray) -> str:
    """First 16 hex digits of the sha256 of an int16 array's bytes: the identity of a staged clip."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def stage(src, dst) -> tuple[float, str]:
    """Stage ``src`` as ``dst`` (.npy, int16, 16 kHz). Returns (duration in seconds, digest). Idempotent."""
    dst = Path(dst)
    if dst.exists():
        a = np.load(dst)
    else:
        a = to_int16(read_16k(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{dst.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.npy")   # unique: writers may overlap
        np.save(tmp, a)
        os.replace(tmp, dst)
    return len(a) / SAMPLE_RATE, digest(a)


def load(staged, max_seconds: float | None = MAX_SECONDS) -> np.ndarray:
    """float32 in [-1, 1) from a staged clip, truncated to ``max_seconds`` (the evaluation convention; None: whole)."""
    y = np.load(staged).astype(np.float32) / 32768.0
    return y if max_seconds is None else y[: int(SAMPLE_RATE * max_seconds)]
