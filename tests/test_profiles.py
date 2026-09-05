import zipfile

import numpy as np
import pytest

from qwen_tts_mlx.profiles import VoiceProfile, load_voice_profile, save_voice_profile


def test_profile_round_trip(tmp_path):
    path = tmp_path / "voice.qvoice"
    original = VoiceProfile(
        audio=np.linspace(-0.5, 0.5, 2400, dtype=np.float32),
        sample_rate=24000,
        ref_text="hello world",
        x_vector_only=False,
    )
    save_voice_profile(original, path)
    loaded = load_voice_profile(path)
    assert loaded.sample_rate == 24000
    assert loaded.ref_text == "hello world"
    assert not loaded.x_vector_only
    np.testing.assert_allclose(loaded.audio, original.audio, atol=4e-5)


def test_profile_requires_transcript_by_default(tmp_path):
    profile = VoiceProfile(np.ones(10, dtype=np.float32), 24000, None, False)
    with pytest.raises(ValueError, match="Reference text"):
        save_voice_profile(profile, tmp_path / "invalid.qvoice")


def test_profile_rejects_unexpected_archive_members(tmp_path):
    path = tmp_path / "bad.qvoice"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", "{}")
        archive.writestr("reference.wav", b"not wave")
        archive.writestr("surprise.txt", "no")
    with pytest.raises(ValueError, match="unexpected files"):
        load_voice_profile(path)
