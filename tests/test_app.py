import warnings

from qwen_tts_mlx.app import build_demo


def test_gradio_demo_builds_without_deprecated_blocks_theme_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        demo = build_demo()
    messages = [str(item.message) for item in caught]
    assert not any("Blocks" in message and "launch" in message for message in messages)
    assert len(demo.config["dependencies"]) == 9
