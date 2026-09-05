"""The released Qwen3-TTS model matrix and its MLX equivalents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelKind(str, Enum):
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    BASE = "base"


@dataclass(frozen=True)
class ModelSpec:
    kind: ModelKind
    size: str
    precision: str
    model_id: str
    instruction_control: bool


PRECISIONS = ("bf16", "8bit", "6bit", "5bit", "4bit")
SIZES = ("1.7B", "0.6B")

_SUFFIX = {
    ModelKind.CUSTOM_VOICE: "CustomVoice",
    ModelKind.VOICE_DESIGN: "VoiceDesign",
    ModelKind.BASE: "Base",
}


def _build_specs() -> tuple[ModelSpec, ...]:
    specs: list[ModelSpec] = []
    for kind in ModelKind:
        sizes = ("1.7B",) if kind is ModelKind.VOICE_DESIGN else SIZES
        for size in sizes:
            for precision in PRECISIONS:
                specs.append(
                    ModelSpec(
                        kind=kind,
                        size=size,
                        precision=precision,
                        model_id=(
                            f"mlx-community/Qwen3-TTS-12Hz-{size}-"
                            f"{_SUFFIX[kind]}-{precision}"
                        ),
                        instruction_control=(
                            kind is ModelKind.VOICE_DESIGN
                            or (kind is ModelKind.CUSTOM_VOICE and size == "1.7B")
                        ),
                    )
                )
    return tuple(specs)


MODEL_SPECS = _build_specs()
_MODEL_INDEX = {(spec.kind, spec.size, spec.precision): spec for spec in MODEL_SPECS}

LANGUAGES = (
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
)

SPEAKERS = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)


def resolve_model(kind: ModelKind | str, size: str, precision: str) -> ModelSpec:
    """Return a validated MLX checkpoint for an official model variant."""

    try:
        normalized_kind = kind if isinstance(kind, ModelKind) else ModelKind(kind)
    except ValueError as exc:
        raise ValueError(f"Unknown model kind: {kind}") from exc

    try:
        return _MODEL_INDEX[(normalized_kind, size, precision)]
    except KeyError as exc:
        if normalized_kind is ModelKind.VOICE_DESIGN and size == "0.6B":
            raise ValueError("Qwen3-TTS does not provide a 0.6B VoiceDesign model.") from exc
        raise ValueError(
            f"Unsupported model combination: {normalized_kind.value}/{size}/{precision}"
        ) from exc
