import json
import zipfile

import numpy as np

from qwen_tts_mlx.long_form import (
    save_long_form_bundle,
    split_long_text,
    synthesize_long_form,
)


def test_split_long_text_preserves_paragraphs_and_bounds_sections():
    text = "First paragraph.\n\n" + "word " * 70 + "final."
    sections = split_long_text(text, max_chars=100)
    assert sections[0] == "First paragraph."
    assert len(sections) > 2
    assert all(0 < len(section) <= 100 for section in sections)


def test_synthesis_runs_each_section_and_inserts_configured_pause():
    calls = []

    def generate(section):
        calls.append(section)
        return [np.ones(10, dtype=np.float32)], 100

    result = synthesize_long_form(
        "Paragraph one.\n\nParagraph two.", generate, max_chars=100, pause_ms=200
    )
    assert calls == ["Paragraph one.", "Paragraph two."]
    assert result.audio.shape == (10 + 20 + 10,)
    np.testing.assert_allclose(result.audio[10:30], 0)


def test_long_form_bundle_contains_combined_audio_sections_and_manifest():
    result = synthesize_long_form(
        "One.\n\nTwo.",
        lambda _: ([np.ones(10, dtype=np.float32)], 24000),
        max_chars=100,
        pause_ms=0,
    )
    path = save_long_form_bundle(result)
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [
            "combined.wav",
            "manifest.json",
            "sections/001.wav",
            "sections/002.wav",
        ]
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["section_count"] == 2
    assert [section["text"] for section in manifest["sections"]] == ["One.", "Two."]
