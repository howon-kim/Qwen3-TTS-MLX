"""Paragraph-aware long-form synthesis and downloadable section bundles."""

from __future__ import annotations

import json
import re
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import normalize_waveform

GenerateSection = Callable[[str], tuple[list[np.ndarray], int]]
ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class LongFormResult:
    texts: list[str]
    segments: list[np.ndarray]
    sample_rate: int
    audio: np.ndarray


def _split_oversized(text: str, max_chars: int) -> list[str]:
    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > max_chars:
        window = remaining[: max_chars + 1]
        candidates = [
            match.end()
            for match in re.finditer(r"(?:[.!?。！？；;:：]\s*|,\s+|，\s*|\s+)", window)
            if match.end() >= max_chars // 3
        ]
        cut = candidates[-1] if candidates else max_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def split_long_text(text: str, max_chars: int = 500) -> list[str]:
    """Keep paragraphs intact where possible and split oversized ones at punctuation."""

    if max_chars < 80:
        raise ValueError("max_chars must be at least 80.")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("Long-form text is empty.")
    paragraphs = [
        re.sub(r"[ \t\f\v]+", " ", paragraph.replace("\n", " ")).strip()
        for paragraph in re.split(r"\n\s*\n+", normalized)
    ]
    sections: list[str] = []
    for paragraph in paragraphs:
        if paragraph:
            sections.extend(_split_oversized(paragraph, max_chars))
    return sections


def join_segments(
    segments: list[np.ndarray], sample_rate: int, pause_ms: int = 650
) -> np.ndarray:
    if not segments:
        raise ValueError("No audio segments were generated.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if pause_ms < 0:
        raise ValueError("pause_ms cannot be negative.")
    pause = np.zeros(round(sample_rate * pause_ms / 1000), dtype=np.float32)
    parts: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        if index and pause.size:
            parts.append(pause)
        parts.append(normalize_waveform(segment))
    return np.concatenate(parts)


def synthesize_long_form(
    text: str,
    generate: GenerateSection,
    *,
    max_chars: int = 500,
    pause_ms: int = 650,
    progress: ProgressCallback | None = None,
) -> LongFormResult:
    """Generate one section at a time so long input stays bounded and recoverable."""

    texts = split_long_text(text, max_chars=max_chars)
    segments: list[np.ndarray] = []
    sample_rate: int | None = None
    for index, section in enumerate(texts, start=1):
        if progress:
            progress(index - 1, len(texts), section)
        wavs, current_rate = generate(section)
        if len(wavs) != 1:
            raise RuntimeError("Long-form section generation must return exactly one waveform.")
        if sample_rate is None:
            sample_rate = int(current_rate)
        elif sample_rate != int(current_rate):
            raise RuntimeError("Sample rate changed between long-form sections.")
        segments.append(normalize_waveform(wavs[0]))
    assert sample_rate is not None
    if progress:
        progress(len(texts), len(texts), "")
    return LongFormResult(
        texts=texts,
        segments=segments,
        sample_rate=sample_rate,
        audio=join_segments(segments, sample_rate, pause_ms=pause_ms),
    )


def save_long_form_bundle(result: LongFormResult) -> str:
    """Write combined audio, individual sections, and a manifest to a ZIP file."""

    output_dir = Path(tempfile.mkdtemp(prefix="qwen3-tts-mlx-long-form-"))
    combined_path = output_dir / "combined.wav"
    sf.write(combined_path, result.audio, result.sample_rate, subtype="PCM_16")
    section_dir = output_dir / "sections"
    section_dir.mkdir()
    manifest = {
        "format": "qwen3-tts-mlx-long-form",
        "version": 1,
        "sample_rate": result.sample_rate,
        "section_count": len(result.texts),
        "sections": [],
    }
    for index, (text, audio) in enumerate(
        zip(result.texts, result.segments, strict=True), start=1
    ):
        name = f"{index:03d}.wav"
        sf.write(section_dir / name, audio, result.sample_rate, subtype="PCM_16")
        manifest["sections"].append({"index": index, "file": f"sections/{name}", "text": text})
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    bundle_path = output_dir / "long-form-output.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(combined_path, "combined.wav")
        archive.write(manifest_path, "manifest.json")
        for path in sorted(section_dir.iterdir()):
            archive.write(path, f"sections/{path.name}")
    return str(bundle_path)
