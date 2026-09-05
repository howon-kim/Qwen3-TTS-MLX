import pytest

from qwen_tts_mlx.models import MODEL_SPECS, ModelKind, resolve_model


def test_released_matrix_has_25_converted_variants():
    assert len(MODEL_SPECS) == 25
    assert len({spec.model_id for spec in MODEL_SPECS}) == 25


def test_resolve_model_builds_real_checkpoint_name():
    spec = resolve_model(ModelKind.CUSTOM_VOICE, "0.6B", "8bit")
    assert spec.model_id == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
    assert not spec.instruction_control


def test_17b_custom_voice_supports_instruction_control():
    assert resolve_model("custom_voice", "1.7B", "bf16").instruction_control


def test_voice_design_has_no_06b_release():
    with pytest.raises(ValueError, match="does not provide"):
        resolve_model(ModelKind.VOICE_DESIGN, "0.6B", "bf16")


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model combination"):
        resolve_model(ModelKind.BASE, "1.7B", "3bit")
