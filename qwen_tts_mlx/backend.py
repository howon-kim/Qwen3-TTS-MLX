"""Official-shaped Qwen3-TTS API backed by MLX-Audio."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

import numpy as np

from .audio import AudioTuple, normalize_waveform, resample
from .models import ModelKind

T = TypeVar("T")


@dataclass(frozen=True)
class VoiceClonePrompt:
    """Reusable, prepared reference input for voice-cloning calls."""

    ref_audio: Any
    ref_text: str | None
    x_vector_only_mode: bool = False


def _as_list(value: T | Sequence[T], name: str) -> list[T]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return [value]
    values = list(value)
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return values


def _broadcast(value: T | Sequence[T], count: int, name: str) -> list[T]:
    values = _as_list(value, name)
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} must have length 1 or match text length ({count}).")
    return values


class MLXQwen3TTS:
    """Compatibility facade around an MLX-Audio Qwen3-TTS model."""

    _COMMON_KWARGS: ClassVar[set[str]] = {
        "max_tokens",
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "stream",
        "streaming_interval",
        "verbose",
    }

    def __init__(self, model: Any, checkpoint: str):
        self.model = model
        self.checkpoint = checkpoint
        self.model_kind = self._detect_model_kind(model)

    @classmethod
    def from_pretrained(cls, checkpoint: str) -> MLXQwen3TTS:
        try:
            from mlx_audio.tts.utils import load_model
        except ModuleNotFoundError as exc:  # pragma: no cover - requires Apple Silicon runtime
            if not (exc.name or "").startswith("mlx_audio"):
                raise
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

    @classmethod
    def _generation_kwargs(
        cls, kwargs: dict[str, Any], extra_supported: Iterable[str] = ()
    ) -> dict[str, Any]:
        values = dict(kwargs)
        if "max_new_tokens" in values and "max_tokens" in values:
            raise TypeError("Pass only one of max_new_tokens or max_tokens.")
        if "max_new_tokens" in values:
            values["max_tokens"] = values.pop("max_new_tokens")

        do_sample = values.pop("do_sample", None)
        if do_sample is False:
            raise NotImplementedError("MLX-Audio does not expose greedy do_sample=False mode.")

        for name in ("speed", "pitch"):
            value = values.pop(name, None)
            if value is not None and float(value) != 1.0:
                raise NotImplementedError(f"MLX-Audio does not support {name} control.")

        if values.pop("non_streaming_mode", None) is not None:
            warnings.warn(
                "non_streaming_mode is selected internally by MLX-Audio and was ignored.",
                RuntimeWarning,
                stacklevel=2,
            )

        subtalker = sorted(key for key in values if key.startswith("subtalker_"))
        if subtalker:
            raise NotImplementedError(
                "MLX-Audio does not expose these official subtalker options: "
                + ", ".join(subtalker)
            )

        supported = cls._COMMON_KWARGS | set(extra_supported)
        unknown = sorted(set(values) - supported)
        if unknown:
            raise TypeError("Unsupported generation argument(s): " + ", ".join(unknown))
        return {key: value for key, value in values.items() if value is not None}

    @staticmethod
    def _collect(
        results: Iterable[Any],
        sample_rate: int,
        *,
        segment_pause_seconds: float = 0.0,
    ) -> tuple[list[np.ndarray], int]:
        items = list(results)
        if not items:
            raise RuntimeError("The model returned no audio.")

        resolved_rate = int(getattr(items[-1], "sample_rate", sample_rate))
        chunks = [normalize_waveform(item.audio) for item in items]
        is_stream = any(bool(getattr(item, "is_streaming_chunk", False)) for item in items)
        if is_stream or len(chunks) == 1 or segment_pause_seconds <= 0:
            return [np.concatenate(chunks)], resolved_rate

        joined: list[np.ndarray] = []
        previous_segment = getattr(items[0], "segment_idx", None)
        separator = np.zeros(int(resolved_rate * segment_pause_seconds), dtype=np.float32)
        for index, (item, chunk) in enumerate(zip(items, chunks, strict=True)):
            segment = getattr(item, "segment_idx", None)
            if index and segment is not None and segment != previous_segment:
                joined.append(separator)
            joined.append(chunk)
            previous_segment = segment
        return [np.concatenate(joined)], resolved_rate

    @staticmethod
    def _collect_batch(
        results: Iterable[Any], sample_rate: int, expected: int
    ) -> tuple[list[np.ndarray], int]:
        grouped: list[list[np.ndarray]] = [[] for _ in range(expected)]
        resolved_rate = sample_rate
        for result in results:
            index = int(getattr(result, "sequence_idx", 0))
            if index < 0 or index >= expected:
                raise RuntimeError(f"MLX-Audio returned invalid sequence index {index}.")
            grouped[index].append(normalize_waveform(result.audio))
            resolved_rate = int(getattr(result, "sample_rate", resolved_rate))
        if any(not chunks for chunks in grouped):
            raise RuntimeError("MLX-Audio did not return audio for every batch item.")
        return [np.concatenate(chunks) for chunks in grouped], resolved_rate

    @staticmethod
    def _stream(results: Iterable[Any], sample_rate: int) -> Iterator[tuple[np.ndarray, int]]:
        for result in results:
            yield normalize_waveform(result.audio), int(
                getattr(result, "sample_rate", sample_rate)
            )

    def _reference_for_mlx(self, audio: AudioTuple | str | Any) -> Any:
        if isinstance(audio, str):
            return audio
        if not isinstance(audio, tuple):
            return audio
        waveform, sample_rate = audio
        target_rate = int(getattr(self.model, "sample_rate", 24000))
        waveform = resample(waveform, sample_rate, target_rate)
        try:
            import mlx.core as mx
        except ModuleNotFoundError as exc:  # pragma: no cover - requires Apple Silicon runtime
            if not (exc.name or "").startswith("mlx"):
                raise
            raise RuntimeError("MLX is not installed.") from exc
        return mx.array(waveform)

    @staticmethod
    def _texts(text: str | Sequence[str]) -> tuple[list[str], bool]:
        texts = _as_list(text, "text")
        if any(not isinstance(item, str) or not item.strip() for item in texts):
            raise ValueError("text entries must be non-empty strings.")
        return texts, isinstance(text, str)

    def _supports_custom_instruction(self) -> bool:
        config = getattr(self.model, "config", None)
        size = str(getattr(config, "tts_model_size", "")).lower()
        return size not in {"0b6", "0.6b"} and "0.6B" not in self.checkpoint

    def generate_custom_voice(
        self,
        text: str | Sequence[str],
        language: str | Sequence[str],
        speaker: str | Sequence[str],
        instruct: str | Sequence[str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        if self.model_kind is not ModelKind.CUSTOM_VOICE:
            raise ValueError("The loaded checkpoint is not a CustomVoice model.")
        texts, _ = self._texts(text)
        languages = _broadcast(language, len(texts), "language")
        speakers = _broadcast(speaker, len(texts), "speaker")
        instructions = _broadcast("" if instruct is None else instruct, len(texts), "instruct")
        if not self._supports_custom_instruction():
            instructions = [""] * len(texts)
        options = self._generation_kwargs(kwargs)

        if len(texts) > 1 and len(set(languages)) == 1:
            results = self.model.batch_generate(
                texts=texts,
                voices=speakers,
                instructs=instructions,
                lang_code=languages[0],
                **options,
            )
            return self._collect_batch(results, self.model.sample_rate, len(texts))

        wavs: list[np.ndarray] = []
        resolved_rate = self.model.sample_rate
        for item, lang, voice, instruction in zip(
            texts, languages, speakers, instructions, strict=True
        ):
            results = self.model.generate_custom_voice(
                text=item,
                language=lang,
                speaker=voice,
                instruct=instruction or None,
                **options,
            )
            collected, resolved_rate = self._collect(results, resolved_rate)
            wavs.extend(collected)
        return wavs, resolved_rate

    def stream_custom_voice(
        self,
        *,
        text: str,
        language: str,
        speaker: str,
        instruct: str | None = None,
        **kwargs: Any,
    ) -> Iterator[tuple[np.ndarray, int]]:
        if self.model_kind is not ModelKind.CUSTOM_VOICE:
            raise ValueError("The loaded checkpoint is not a CustomVoice model.")
        options = self._generation_kwargs(kwargs)
        options["stream"] = True
        if not self._supports_custom_instruction():
            instruct = None
        return self._stream(
            self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                **options,
            ),
            self.model.sample_rate,
        )

    def generate_voice_design(
        self,
        text: str | Sequence[str],
        language: str | Sequence[str],
        instruct: str | Sequence[str],
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        if self.model_kind is not ModelKind.VOICE_DESIGN:
            raise ValueError("The loaded checkpoint is not a VoiceDesign model.")
        texts, _ = self._texts(text)
        languages = _broadcast(language, len(texts), "language")
        instructions = _broadcast(instruct, len(texts), "instruct")
        options = self._generation_kwargs(kwargs)

        if len(texts) > 1 and len(set(languages)) == 1:
            results = self.model.batch_generate(
                texts=texts,
                instructs=instructions,
                lang_code=languages[0],
                **options,
            )
            return self._collect_batch(results, self.model.sample_rate, len(texts))

        wavs: list[np.ndarray] = []
        resolved_rate = self.model.sample_rate
        for item, lang, instruction in zip(texts, languages, instructions, strict=True):
            results = self.model.generate_voice_design(
                text=item,
                language=lang,
                instruct=instruction,
                **options,
            )
            collected, resolved_rate = self._collect(results, resolved_rate)
            wavs.extend(collected)
        return wavs, resolved_rate

    def stream_voice_design(
        self,
        *,
        text: str,
        language: str,
        instruct: str,
        **kwargs: Any,
    ) -> Iterator[tuple[np.ndarray, int]]:
        if self.model_kind is not ModelKind.VOICE_DESIGN:
            raise ValueError("The loaded checkpoint is not a VoiceDesign model.")
        options = self._generation_kwargs(kwargs)
        options["stream"] = True
        return self._stream(
            self.model.generate_voice_design(
                text=text, language=language, instruct=instruct, **options
            ),
            self.model.sample_rate,
        )

    def create_voice_clone_prompt(
        self,
        ref_audio: AudioTuple | str | Any | Sequence[AudioTuple | str | Any],
        ref_text: str | None | Sequence[str | None] = None,
        x_vector_only_mode: bool = False,
    ) -> VoiceClonePrompt | list[VoiceClonePrompt]:
        if self.model_kind is not ModelKind.BASE:
            raise ValueError("The loaded checkpoint is not a Base voice-cloning model.")
        is_audio_tuple = (
            isinstance(ref_audio, tuple)
            and len(ref_audio) == 2
            and isinstance(ref_audio[1], (int, np.integer))
        )
        references = [ref_audio] if is_audio_tuple else _as_list(ref_audio, "ref_audio")
        texts = _broadcast(ref_text, len(references), "ref_text")
        prompts: list[VoiceClonePrompt] = []
        for reference, transcript in zip(references, texts, strict=True):
            clean_text = (transcript or "").strip() or None
            if not x_vector_only_mode and clean_text is None:
                raise ValueError(
                    "Reference text is required unless x-vector-only mode is enabled."
                )
            prompts.append(
                VoiceClonePrompt(
                    ref_audio=self._reference_for_mlx(reference),
                    ref_text=None if x_vector_only_mode else clean_text,
                    x_vector_only_mode=x_vector_only_mode,
                )
            )
        return prompts[0] if len(prompts) == 1 else prompts

    def generate_voice_clone(
        self,
        text: str | Sequence[str],
        language: str | Sequence[str],
        ref_audio: AudioTuple | str | Any | None = None,
        ref_text: str | None = None,
        x_vector_only_mode: bool = False,
        voice_clone_prompt: VoiceClonePrompt | Sequence[VoiceClonePrompt] | None = None,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        if self.model_kind is not ModelKind.BASE:
            raise ValueError("The loaded checkpoint is not a Base voice-cloning model.")
        texts, _ = self._texts(text)
        languages = _broadcast(language, len(texts), "language")

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("ref_audio or voice_clone_prompt is required.")
            created = self.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            prompts = _broadcast(created, len(texts), "voice_clone_prompt")
        else:
            prompts = _broadcast(voice_clone_prompt, len(texts), "voice_clone_prompt")
        if any(not isinstance(prompt, VoiceClonePrompt) for prompt in prompts):
            raise TypeError("voice_clone_prompt entries must be VoiceClonePrompt objects.")

        options = self._generation_kwargs(
            kwargs, extra_supported={"split_pattern", "streaming_context_size"}
        )
        shared_prompt = all(prompt is prompts[0] for prompt in prompts)
        if (
            len(texts) > 1
            and shared_prompt
            and not prompts[0].x_vector_only_mode
            and len(set(languages)) == 1
        ):
            results = self.model.batch_generate(
                texts=texts,
                ref_audio=prompts[0].ref_audio,
                ref_text=prompts[0].ref_text,
                lang_code=languages[0],
                **options,
            )
            return self._collect_batch(results, self.model.sample_rate, len(texts))

        wavs: list[np.ndarray] = []
        resolved_rate = self.model.sample_rate
        for item, lang, prompt in zip(texts, languages, prompts, strict=True):
            results = self.model.generate(
                text=item,
                lang_code=lang,
                ref_audio=prompt.ref_audio,
                ref_text=prompt.ref_text,
                **options,
            )
            collected, resolved_rate = self._collect(
                results, resolved_rate, segment_pause_seconds=0.12
            )
            wavs.extend(collected)
        return wavs, resolved_rate

    def stream_voice_clone(
        self,
        *,
        text: str,
        language: str,
        ref_audio: AudioTuple | str | Any | None = None,
        ref_text: str | None = None,
        x_vector_only_mode: bool = False,
        voice_clone_prompt: VoiceClonePrompt | None = None,
        **kwargs: Any,
    ) -> Iterator[tuple[np.ndarray, int]]:
        if self.model_kind is not ModelKind.BASE:
            raise ValueError("The loaded checkpoint is not a Base voice-cloning model.")
        prompt = voice_clone_prompt
        if prompt is None:
            if ref_audio is None:
                raise ValueError("ref_audio or voice_clone_prompt is required.")
            created = self.create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode)
            if isinstance(created, list):  # defensive: scalar reference always returns one prompt
                prompt = created[0]
            else:
                prompt = created
        options = self._generation_kwargs(
            kwargs, extra_supported={"split_pattern", "streaming_context_size"}
        )
        options["stream"] = True
        return self._stream(
            self.model.generate(
                text=text,
                lang_code=language,
                ref_audio=prompt.ref_audio,
                ref_text=prompt.ref_text,
                **options,
            ),
            self.model.sample_rate,
        )
