"""Opt-in Apple Silicon generation test; no model is downloaded in normal CI."""

import os

import pytest

MODEL_ID = os.environ.get("QWEN_MLX_INTEGRATION_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL_ID,
    reason="Set QWEN_MLX_INTEGRATION_MODEL to run a real MLX generation.",
)


def test_real_custom_voice_generation():
    from qwen_tts_mlx import MLXQwen3TTS

    tts = MLXQwen3TTS.from_pretrained(MODEL_ID)
    wavs, sample_rate = tts.generate_custom_voice(
        text="MLX 통합 테스트입니다.",
        language="Korean",
        speaker="Sohee",
        max_new_tokens=256,
    )
    assert sample_rate == 24000
    assert len(wavs) == 1
    assert wavs[0].size > 1000
