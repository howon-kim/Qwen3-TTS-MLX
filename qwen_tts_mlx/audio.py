"""Audio normalization helpers shared by the UI and MLX adapter."""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

AudioTuple = tuple[np.ndarray, int]


def normalize_waveform(waveform: Any) -> np.ndarray:
    """Convert integer or floating audio to mono float32 in [-1, 1]."""

    value = np.asarray(waveform)
    if value.size == 0:
        raise ValueError("Audio is empty.")

    if np.issubdtype(value.dtype, np.integer):
        info = np.iinfo(value.dtype)
        if info.min < 0:
            value = value.astype(np.float32) / float(max(abs(info.min), info.max))
        else:
            midpoint = (info.max + 1) / 2.0
            value = (value.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(value.dtype, np.floating):
        value = value.astype(np.float32)
        peak = float(np.max(np.abs(value)))
        if peak > 1.0:
            value = value / peak
    else:
        raise TypeError(f"Unsupported audio dtype: {value.dtype}")

    if value.ndim > 1:
        value = np.mean(value, axis=-1, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError("Audio must be one-dimensional after mixing to mono.")
    return np.ascontiguousarray(np.clip(value, -1.0, 1.0), dtype=np.float32)


def from_gradio(audio: Any) -> AudioTuple | None:
    """Normalize the tuple/dict formats returned by Gradio Audio."""

    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2:
        first, second = audio
        if isinstance(first, (int, np.integer)):
            return normalize_waveform(second), int(first)
        if isinstance(second, (int, np.integer)):
            return normalize_waveform(first), int(second)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        return normalize_waveform(audio["data"]), int(audio["sampling_rate"])
    raise TypeError("Unsupported audio payload from Gradio.")


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono audio without introducing a PyTorch dependency."""

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive.")
    value = normalize_waveform(audio)
    if source_rate == target_rate:
        return value
    divisor = gcd(source_rate, target_rate)
    return np.ascontiguousarray(
        resample_poly(value, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def file_path(value: Any) -> str:
    """Resolve Gradio's file object variants to a filesystem path."""

    if isinstance(value, (str, Path)):
        return str(value)
    for attribute in ("name", "path"):
        candidate = getattr(value, attribute, None)
        if candidate:
            return str(candidate)
    raise TypeError("Could not resolve the uploaded file path.")
