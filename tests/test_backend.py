from types import SimpleNamespace

import numpy as np
import pytest

from qwen_tts_mlx.backend import MLXQwen3TTS


class FakeResult:
    def __init__(self, values, sample_rate=24000):
        self.audio = np.asarray(values, dtype=np.float32)
        self.sample_rate = sample_rate


class FakeCustomModel:
    sample_rate = 24000
    config = SimpleNamespace(tts_model_type="custom_voice")

    def __init__(self):
        self.received = None

    def generate_custom_voice(self, **kwargs):
        self.received = kwargs
        yield FakeResult([0.1, 0.2])
        yield FakeResult([0.3])


class FakeBaseModel:
    sample_rate = 24000
    config = SimpleNamespace(tts_model_type="base")

    def __init__(self):
        self.received = None

    def generate(self, **kwargs):
        self.received = kwargs
        yield FakeResult([0.1, -0.1])


def test_custom_voice_translates_official_token_argument_and_joins_results():
    model = FakeCustomModel()
    backend = MLXQwen3TTS(model, "fake")
    wavs, sample_rate = backend.generate_custom_voice(
        text="hello",
        language="English",
        speaker="Ryan",
        max_new_tokens=100,
        ignored_parameter=True,
    )
    assert sample_rate == 24000
    assert model.received["max_tokens"] == 100
    assert "max_new_tokens" not in model.received
    assert "ignored_parameter" not in model.received
    assert wavs[0].shape == (2 + 2880 + 1,)


def test_clone_accepts_path_and_xvector_only_omits_transcript():
    model = FakeBaseModel()
    backend = MLXQwen3TTS(model, "fake")
    backend.generate_voice_clone(
        text="target",
        language="Auto",
        ref_audio="reference.wav",
        x_vector_only_mode=True,
    )
    assert model.received["ref_audio"] == "reference.wav"
    assert model.received["ref_text"] is None


def test_clone_requires_transcript_outside_xvector_mode():
    backend = MLXQwen3TTS(FakeBaseModel(), "fake")
    with pytest.raises(ValueError, match="Reference text"):
        backend.generate_voice_clone(
            text="target", language="English", ref_audio="reference.wav"
        )
