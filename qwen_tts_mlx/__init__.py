"""Qwen3-TTS on Apple Silicon through MLX-Audio."""

from .backend import MLXQwen3TTS
from .models import MODEL_SPECS, ModelKind, ModelSpec, resolve_model

__all__ = ["MODEL_SPECS", "MLXQwen3TTS", "ModelKind", "ModelSpec", "resolve_model"]
__version__ = "0.1.0"
