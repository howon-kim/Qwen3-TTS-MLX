"""Portable, pickle-free voice profiles for the MLX app."""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import AudioTuple, file_path, normalize_waveform

PROFILE_FORMAT = "qwen3-tts-mlx-voice"
PROFILE_VERSION = 1
MAX_PROFILE_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class VoiceProfile:
    audio: np.ndarray
    sample_rate: int
    ref_text: str | None
    x_vector_only: bool

    @property
    def audio_tuple(self) -> AudioTuple:
        return self.audio, self.sample_rate


def save_voice_profile(profile: VoiceProfile, destination: str | Path | None = None) -> str:
    """Store reference audio and metadata in a portable .qvoice ZIP container."""

    if not profile.x_vector_only and not (profile.ref_text or "").strip():
        raise ValueError("Reference text is required unless x-vector-only mode is enabled.")
    waveform = normalize_waveform(profile.audio)
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, waveform, profile.sample_rate, format="WAV", subtype="PCM_16")
    metadata = {
        "format": PROFILE_FORMAT,
        "version": PROFILE_VERSION,
        "sample_rate": profile.sample_rate,
        "ref_text": profile.ref_text,
        "x_vector_only": profile.x_vector_only,
    }

    if destination is None:
        handle, destination = tempfile.mkstemp(prefix="qwen3_tts_voice_", suffix=".qvoice")
        os.close(handle)
        Path(destination).unlink(missing_ok=True)
    destination = str(destination)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        archive.writestr("reference.wav", audio_buffer.getvalue())
    return destination


def load_voice_profile(source) -> VoiceProfile:
    """Load and validate a .qvoice profile without executing serialized code."""

    path = Path(file_path(source))
    if path.stat().st_size > MAX_PROFILE_BYTES:
        raise ValueError("Voice profile is too large.")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != 2 or set(names) != {"metadata.json", "reference.wav"}:
                raise ValueError("Voice profile contains unexpected files.")
            if any(member.file_size > MAX_PROFILE_BYTES for member in members):
                raise ValueError("Voice profile contains an oversized file.")
            metadata = json.loads(archive.read("metadata.json"))
            audio_bytes = archive.read("reference.wav")
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as exc:
        raise ValueError("Invalid voice profile.") from exc

    if metadata.get("format") != PROFILE_FORMAT or metadata.get("version") != PROFILE_VERSION:
        raise ValueError("Unsupported voice profile format or version.")
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    profile = VoiceProfile(
        audio=normalize_waveform(waveform),
        sample_rate=int(sample_rate),
        ref_text=metadata.get("ref_text"),
        x_vector_only=bool(metadata.get("x_vector_only", False)),
    )
    if not profile.x_vector_only and not (profile.ref_text or "").strip():
        raise ValueError("Voice profile is missing its reference transcript.")
    return profile
