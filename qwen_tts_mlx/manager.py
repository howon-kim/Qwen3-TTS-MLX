"""Lazy, single-checkpoint model manager for unified-memory Macs."""

from __future__ import annotations

import gc
import threading

from .backend import MLXQwen3TTS
from .models import ModelKind, ModelSpec, resolve_model


class ModelManager:
    """Keep only one large checkpoint resident and serialize inference calls."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._backend: MLXQwen3TTS | None = None
        self._spec: ModelSpec | None = None

    @property
    def loaded_model_id(self) -> str | None:
        return self._spec.model_id if self._spec else None

    def get(self, kind: ModelKind | str, size: str, precision: str) -> MLXQwen3TTS:
        spec = resolve_model(kind, size, precision)
        with self._lock:
            if self._spec == spec and self._backend is not None:
                return self._backend
            self.unload()
            self._backend = MLXQwen3TTS.from_pretrained(spec.model_id)
            self._spec = spec
            return self._backend

    def unload(self) -> None:
        with self._lock:
            self._backend = None
            self._spec = None
            gc.collect()
            try:
                import mlx.core as mx

                mx.clear_cache()
            except ImportError:
                pass

    def run(self, kind: ModelKind | str, size: str, precision: str, method: str, **kwargs):
        with self._lock:
            backend = self.get(kind, size, precision)
            return getattr(backend, method)(**kwargs)
