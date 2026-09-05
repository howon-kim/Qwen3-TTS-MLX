# Qwen3-TTS-MLX

An unofficial, Mac-first implementation of the
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) experience. It keeps the official
model families and workflows while running converted checkpoints with
[MLX-Audio](https://github.com/Blaizzy/mlx-audio) on Apple Silicon.

> This project is not affiliated with or endorsed by Qwen/Alibaba. Use generated
> voices lawfully and only with the speaker's permission.

## What works

- CustomVoice with the nine released speakers, multilingual input, and instruction
  control on the 1.7B checkpoint
- VoiceDesign from a natural-language voice description
- Voice cloning with reference audio + transcript
- x-vector-only cloning without a transcript (usually lower similarity/quality)
- VoiceDesign → Base cloning in one guided workflow
- Safe, portable `.qvoice` reference profiles (ZIP/WAV/JSON; no pickle execution)
- BF16 plus 8-bit, 6-bit, 5-bit, and 4-bit MLX checkpoints
- Lazy model download, one-checkpoint-at-a-time memory management, and advanced
  sampling controls
- MLX-Audio streaming and native batch generation remain available through the
  underlying `backend.model` API

The UI mirrors the official Gradio demo's CustomVoice, VoiceDesign, VoiceClone,
and voice save/load flows. It intentionally does not claim binary compatibility
with the official PyTorch `.pt` clone-prompt files; `.qvoice` is a safer,
project-specific interchange format.

## Requirements

- An Apple Silicon Mac (M1 or newer)
- macOS 14 or newer recommended
- Python 3.10–3.13; Python 3.12 is recommended
- Enough free disk space for the selected Hugging Face checkpoint

MLX uses the Apple GPU through Metal. It does **not** run these models on the
Apple Neural Engine.

## Install and run

```bash
git clone https://github.com/howon-kim/Qwen3-TTS-MLX.git
cd Qwen3-TTS-MLX
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
qwen-tts-mlx --inbrowser
```

You can also launch it with `python app.py`. The web app listens on
`127.0.0.1:8000` by default. Use `--host 0.0.0.0` only when you intentionally want
other devices on your network to connect.

Models are downloaded on first use to the normal Hugging Face cache. BF16 gives
the closest conversion fidelity; start with 8-bit or 6-bit when unified memory is
limited. Quantization reduces memory and disk use but can change quality.

## Released model matrix

| Workflow | Official sizes | MLX checkpoint pattern |
| --- | --- | --- |
| CustomVoice | 1.7B, 0.6B | `mlx-community/Qwen3-TTS-12Hz-{size}-CustomVoice-{precision}` |
| VoiceDesign | 1.7B | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-{precision}` |
| Voice Clone | 1.7B, 0.6B Base | `mlx-community/Qwen3-TTS-12Hz-{size}-Base-{precision}` |

There is no released 0.6B VoiceDesign checkpoint. CustomVoice instruction control
is exposed only for 1.7B, matching the official model capabilities.

Supported languages are Chinese, English, Japanese, Korean, German, French,
Russian, Portuguese, Spanish, and Italian, with automatic language selection.
Released speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna,
and Sohee.

## Python API

The facade follows the names used by the official package while returning
`(list[numpy.ndarray], sample_rate)`:

```python
import soundfile as sf

from qwen_tts_mlx import MLXQwen3TTS

tts = MLXQwen3TTS.from_pretrained(
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
)
wavs, sample_rate = tts.generate_custom_voice(
    text="안녕하세요. MLX에서 실행 중입니다.",
    language="Korean",
    speaker="Sohee",
)
sf.write("output.wav", wavs[0], sample_rate)
```

Voice cloning uses a Base checkpoint:

```python
tts = MLXQwen3TTS.from_pretrained(
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
)
wavs, sample_rate = tts.generate_voice_clone(
    text="새로 말할 문장입니다.",
    language="Korean",
    ref_audio="reference.wav",
    ref_text="참고 음성에서 실제로 말한 문장입니다.",
)
```

To use MLX-Audio's streaming or native batched generation directly:

```python
for chunk in tts.model.generate_custom_voice(
    text="Streaming from MLX.",
    speaker="Ryan",
    language="English",
    stream=True,
    streaming_interval=0.32,
):
    consume(chunk.audio)
```

## Differences from the official PyTorch package

| Area | This project |
| --- | --- |
| Runtime | MLX/Metal on Apple Silicon instead of PyTorch/CUDA |
| Weights | Community MLX conversions from Hugging Face |
| Clone profile | Safe `.qvoice` reference container, not official `.pt` prompt objects |
| Streaming/batching | Supported by MLX-Audio's native API; the Gradio UI returns a completed clip |
| Speech tokenizer API | Not wrapped as a standalone public API yet |
| Platform | macOS + Apple Silicon only |

## Development

```bash
python -m pip install -e '.[dev]' --no-deps
python -m pip install numpy scipy soundfile pytest ruff
ruff check .
pytest
```

Unit tests do not download model weights. A real generation run is the integration
test and requires an Apple Silicon Mac.

## License and credits

Code in this repository is Apache-2.0. Qwen3-TTS is developed by the Alibaba Qwen
team and released under Apache-2.0. The MLX runtime integration is provided by
MLX-Audio, licensed under MIT. Converted model repositories retain their own model
cards and license notices. See [NOTICE](NOTICE).
