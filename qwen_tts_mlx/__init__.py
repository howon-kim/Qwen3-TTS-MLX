"""Qwen3-TTS on Apple Silicon through MLX-Audio."""

from .backend import MLXQwen3TTS, VoiceClonePrompt
from .long_form import LongFormResult, split_long_text, synthesize_long_form
from .models import MODEL_SPECS, ModelKind, ModelSpec, resolve_model

__all__ = [
    "MODEL_SPECS",
    "LongFormResult",
    "MLXQwen3TTS",
    "ModelKind",
    "ModelSpec",
    "VoiceClonePrompt",
    "resolve_model",
    "split_long_text",
    "synthesize_long_form",
]
__version__ = "0.2.0"
