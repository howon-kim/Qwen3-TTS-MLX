"""Official-shaped Qwen3-TTS API backed by MLX-Audio."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .audio import AudioTuple, normalize_waveform, resample
from .models import ModelKind


class MLXQwen3TTS:
    """Small compatibility facade around an MLX-Audio Qwen3-TTS model."""

    def __init__(self, model: Any, checkpoint: str):
        self.model = model
        self.checkpoint = checkpoint
        self.model_kind = self._detect_model_kind(model)

    @classmethod
    def from_pretrained(cls, checkpoint: str) -> MLXQwen3TTS:
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as exc:  # pragma: no cover - requires Apple Silicon runtime
            raise RuntimeError(
                "MLX-Audio is not installed. Run `pip install -e .` on an Apple Silicon Mac."
            ) from exc
        return cls(load_model(checkpoint), checkpoint)

    @staticmethod
    def _detect_model_kind(model: Any) -> ModelKind:
        config = getattr(model, "config", None)
        raw = getattr(config, "tts_model_type", None) or getattr(model, "tts_model_type", None)
        try:
            return ModelKind(raw)
        except ValueError as exc:
            raise ValueError(f"Unsupported Qwen3-TTS model type: {raw!r}") from exc

    def get_supported_languages(self) -> list[str]:
        method = getattr(self.model, "get_supported_languages", None)
        if callable(method):
            return list(method())
        return ["auto"]

    def get_supported_speakers(self) -> list[str]:
        method = getattr(self.model, "get_supported_speakers", None)
        if callable(method):
            return list(method())
        return []

    @staticmethod
    def _generation_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        values = dict(kwargs)
        if "max_new_tokens" in values:
            values["max_tokens"] = values.pop("max_new_tokens")
        supported = {
            "max_tokens",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "stream",
            "streaming_interval",
        }
        return {key: value for key, value in values.items() if key in supported and value is not None}

    @staticmethod
    def _collect(results: Iterable[Any], sample_rate: int) -> tuple[list[np.ndarray], int]:
        chunks: list[np.ndarray] = []
        resolved_rate = sample_rate
        for result in results:
            chunks.append(normalize_waveform(result.audio))
            resolved_rate = int(getattr(result, "sample_rate", resolved_rate))
        if not chunks:
            raise RuntimeError("The model returned no audio.")
        if len(chunks) == 1:
            return [chunks[0]], resolved_rate
        separator = np.zeros(int(resolved_rate * 0.12), dtype=np.float32)
        joined: list[np.ndarray] = []
        for index, chunk in enumerate(chunks):
            if index:
                joined.append(separator)
            joined.append(chunk)
        return [np.concatenate(joined)], resolved_rate

    def _reference_for_mlx(self, audio: AudioTuple | str) -> Any:
        if isinstance(audio, str):
            return audio
        waveform, sample_rate = audio
        target_rate = int(getattr(self.model, "sample_rate", 24000))
        waveform = resample(waveform, sample_rate, target_rate)
        try:
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - requires Apple Silicon runtime
            raise RuntimeError("MLX is not installed.") from exc
        return mx.array(waveform)

    def generate_custom_voice(
        self,
        text: str,
        language: str,
        speaker: str,
        instruct: str | None = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        if self.model_kind is not ModelKind.CUSTOM_VOICE:
            raise ValueError("The loaded checkpoint is not a CustomVoice model.")
        results = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            **self._generation_kwargs(kwargs),
        )
        return self._collect(results, self.model.sample_rate)

    def generate_voice_design(
        self,
        text: str,
        language: str,
        instruct: str,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        if self.model_kind is not ModelKind.VOICE_DESIGN:
            raise ValueError("The loaded checkpoint is not a VoiceDesign model.")
        results = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            **self._generation_kwargs(kwargs),
        )
        return self._collect(results, self.model.sample_rate)

    def generate_voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: AudioTuple | str,
        ref_text: str | None = None,
        x_vector_only_mode: bool = False,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        if self.model_kind is not ModelKind.BASE:
            raise ValueError("The loaded checkpoint is not a Base voice-cloning model.")
        if not x_vector_only_mode and not (ref_text or "").strip():
            raise ValueError("Reference text is required unless x-vector-only mode is enabled.")
        reference = self._reference_for_mlx(ref_audio)
        results = self.model.generate(
            text=text,
            lang_code=language,
            ref_audio=reference,
            ref_text=None if x_vector_only_mode else ref_text,
            **self._generation_kwargs(kwargs),
        )
        return self._collect(results, self.model.sample_rate)
