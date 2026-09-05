import numpy as np
import pytest

from qwen_tts_mlx.audio import from_gradio, normalize_waveform, resample


def test_normalize_integer_stereo_to_float_mono():
    stereo = np.array([[32767, 32767], [-32768, -32768]], dtype=np.int16)
    result = normalize_waveform(stereo)
    assert result.dtype == np.float32
    assert result.shape == (2,)
    np.testing.assert_allclose(result, [32767 / 32768, -1.0], atol=1e-6)


def test_normalize_large_float_peak():
    result = normalize_waveform(np.array([-2.0, 0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(result, [-1.0, 0.0, 0.5])


def test_gradio_sample_rate_first_tuple():
    waveform, sample_rate = from_gradio((24000, np.ones(4, dtype=np.float32)))
    assert sample_rate == 24000
    assert waveform.shape == (4,)


def test_resample_changes_expected_length():
    output = resample(np.ones(160, dtype=np.float32), 16000, 24000)
    assert output.shape == (240,)


def test_empty_audio_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        normalize_waveform([])
