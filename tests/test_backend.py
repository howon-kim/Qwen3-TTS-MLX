from types import SimpleNamespace

import numpy as np
import pytest

from qwen_tts_mlx.backend import MLXQwen3TTS, VoiceClonePrompt


class FakeResult:
    def __init__(self, values, sample_rate=24000, **metadata):
        self.audio = np.asarray(values, dtype=np.float32)
        self.sample_rate = sample_rate
        for name, value in metadata.items():
            setattr(self, name, value)


class FakeCustomModel:
    sample_rate = 24000

    def __init__(self, size="1b7"):
        self.config = SimpleNamespace(tts_model_type="custom_voice", tts_model_size=size)
        self.received = []

    def generate_custom_voice(self, **kwargs):
        self.received.append(kwargs)
        yield FakeResult([0.1, 0.2])
        yield FakeResult([0.3])

    def batch_generate(self, **kwargs):
        self.received.append(kwargs)
        yield FakeResult([0.1], sequence_idx=1)
        yield FakeResult([0.2], sequence_idx=0)
        yield FakeResult([0.3], sequence_idx=1)


class FakeBaseModel:
    sample_rate = 24000
    config = SimpleNamespace(tts_model_type="base", tts_model_size="0b6")

    def __init__(self):
        self.received = []

    def generate(self, **kwargs):
        self.received.append(kwargs)
        yield FakeResult([0.1, -0.1])

    def batch_generate(self, **kwargs):
        self.received.append(kwargs)
        for index in range(len(kwargs["texts"])):
            yield FakeResult([index + 0.1], sequence_idx=index)


def test_custom_voice_translates_official_token_argument_without_fake_gaps():
    model = FakeCustomModel()
    backend = MLXQwen3TTS(model, "fake")
    wavs, sample_rate = backend.generate_custom_voice(
        text="hello",
        language="English",
        speaker="Ryan",
        max_new_tokens=100,
    )
    assert sample_rate == 24000
    assert model.received[0]["max_tokens"] == 100
    assert "max_new_tokens" not in model.received[0]
    np.testing.assert_allclose(wavs[0], [0.1, 0.2, 0.3])


def test_streaming_chunks_are_concatenated_without_silence():
    results = [
        FakeResult(np.ones(2400), is_streaming_chunk=True),
        FakeResult(np.ones(2400), is_streaming_chunk=True),
    ]
    wavs, _ = MLXQwen3TTS._collect(results, 24000, segment_pause_seconds=0.12)
    assert wavs[0].shape == (4800,)


def test_explicit_segments_receive_a_short_pause():
    results = [
        FakeResult([0.1], sample_rate=100, segment_idx=0),
        FakeResult([0.2], sample_rate=100, segment_idx=1),
    ]
    wavs, _ = MLXQwen3TTS._collect(results, 100, segment_pause_seconds=0.12)
    assert wavs[0].shape == (14,)
    np.testing.assert_allclose(wavs[0][1:13], 0)


def test_unknown_generation_arguments_are_rejected_instead_of_discarded():
    backend = MLXQwen3TTS(FakeCustomModel(), "fake")
    with pytest.raises(TypeError, match="ignored_parameter"):
        backend.generate_custom_voice(
            text="hello",
            language="English",
            speaker="Ryan",
            ignored_parameter=True,
        )


def test_unsupported_official_sampling_modes_are_explicit():
    backend = MLXQwen3TTS(FakeCustomModel(), "fake")
    with pytest.raises(NotImplementedError, match="do_sample=False"):
        backend.generate_custom_voice(
            text="hello", language="English", speaker="Ryan", do_sample=False
        )
    with pytest.raises(NotImplementedError, match="subtalker"):
        backend.generate_custom_voice(
            text="hello",
            language="English",
            speaker="Ryan",
            subtalker_temperature=0.8,
        )


def test_06b_custom_voice_strips_unsupported_instruction():
    model = FakeCustomModel(size="0b6")
    backend = MLXQwen3TTS(model, "mlx-community/fake-0.6B-CustomVoice")
    backend.generate_custom_voice(
        text="hello", language="English", speaker="Ryan", instruct="angry"
    )
    assert model.received[0]["instruct"] is None


def test_custom_voice_list_uses_native_batch_and_groups_sequences():
    model = FakeCustomModel()
    backend = MLXQwen3TTS(model, "fake")
    wavs, _ = backend.generate_custom_voice(
        text=["one", "two"],
        language="English",
        speaker=["Ryan", "Aiden"],
    )
    assert model.received[0]["texts"] == ["one", "two"]
    np.testing.assert_allclose(wavs[0], [0.2])
    np.testing.assert_allclose(wavs[1], [0.1, 0.3])


def test_clone_accepts_path_and_xvector_only_omits_transcript():
    model = FakeBaseModel()
    backend = MLXQwen3TTS(model, "fake")
    backend.generate_voice_clone(
        text="target",
        language="Auto",
        ref_audio="reference.wav",
        x_vector_only_mode=True,
    )
    assert model.received[0]["ref_audio"] == "reference.wav"
    assert model.received[0]["ref_text"] is None


def test_clone_requires_transcript_outside_xvector_mode():
    backend = MLXQwen3TTS(FakeBaseModel(), "fake")
    with pytest.raises(ValueError, match="Reference text"):
        backend.generate_voice_clone(
            text="target", language="English", ref_audio="reference.wav"
        )


def test_create_clone_prompt_treats_waveform_rate_tuple_as_one_reference(monkeypatch):
    backend = MLXQwen3TTS(FakeBaseModel(), "fake")
    monkeypatch.setattr(backend, "_reference_for_mlx", lambda audio: audio)
    audio = (np.ones(8, dtype=np.float32), 24000)
    prompt = backend.create_voice_clone_prompt(audio, "reference words")
    assert isinstance(prompt, VoiceClonePrompt)
    assert prompt.ref_audio is audio


def test_reusable_clone_prompt_enables_native_batch():
    model = FakeBaseModel()
    backend = MLXQwen3TTS(model, "fake")
    prompt = VoiceClonePrompt("reference.wav", "reference words")
    wavs, _ = backend.generate_voice_clone(
        text=["first", "second"],
        language="English",
        voice_clone_prompt=prompt,
    )
    assert model.received[0]["ref_audio"] == "reference.wav"
    assert model.received[0]["ref_text"] == "reference words"
    assert len(wavs) == 2


def test_stream_facade_yields_audio_rate_pairs_without_joining():
    backend = MLXQwen3TTS(FakeCustomModel(), "fake")
    chunks = list(
        backend.stream_custom_voice(
            text="hello",
            language="English",
            speaker="Ryan",
            max_new_tokens=50,
        )
    )
    assert [chunk.shape for chunk, _ in chunks] == [(2,), (1,)]
    assert all(rate == 24000 for _, rate in chunks)
