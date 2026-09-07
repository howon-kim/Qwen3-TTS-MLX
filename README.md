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
- Paragraph-aware long-form generation with configurable section length and pauses
- Combined WAV playback plus a ZIP containing every section and a JSON manifest
- Safe, portable `.qvoice` reference profiles (ZIP/WAV/JSON; no pickle execution)
- Official-shaped scalar/list APIs, reusable clone prompts, and streaming iterators
- BF16 plus 8-bit, 6-bit, 5-bit, and 4-bit MLX checkpoints
- Lazy model download, one-checkpoint-at-a-time memory management, and advanced
  sampling controls
- Native MLX-Audio batching is used when compatible list inputs share a language

The UI mirrors the official Gradio demo's CustomVoice, VoiceDesign, VoiceClone,
and voice save/load flows, then adds a Mac-oriented Long Form workflow. It
intentionally does not claim binary compatibility
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

List inputs return one waveform per text and use native MLX batching when the
language and reference constraints permit it:

```python
wavs, sample_rate = tts.generate_custom_voice(
    text=["First line.", "Second line."],
    language="English",
    speaker=["Ryan", "Aiden"],
)
```

Prepare a voice reference once and reuse it across calls:

```python
prompt = tts.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="The words spoken in the reference recording.",
)
wavs, sample_rate = tts.generate_voice_clone(
    text=["First passage.", "Second passage."],
    language="English",
    voice_clone_prompt=prompt,
)
```

The streaming facade yields `(numpy.ndarray, sample_rate)` without inserting
silence between native stream chunks:

```python
for chunk, sample_rate in tts.stream_custom_voice(
    text="Streaming from MLX.",
    speaker="Ryan",
    language="English",
    streaming_interval=0.32,
):
    consume(chunk, sample_rate)
```

Unsupported official generation arguments now raise an explicit error instead of
being silently discarded. `max_new_tokens` is translated to MLX-Audio's
`max_tokens`. The 0.6B CustomVoice checkpoint ignores instructions because that
model variant does not support instruction control.

## Long-form generation

Open **Long Form (장문 제작)**, paste the full script, then choose Custom Voice,
Voice Design, or Voice Clone. The app keeps paragraph boundaries where possible,
splits oversized paragraphs near punctuation, generates one bounded section at a
time, and inserts a configurable pause. Voice Clone prepares and reuses the same
reference for every section; this is the recommended workflow for a consistent
character voice.

The ZIP output contains `combined.wav`, numbered WAV files under `sections/`, and
`manifest.json` with the corresponding text. Regular transcript-based cloning in
MLX-Audio applies an effective minimum repetition penalty of 1.5, which the UI
reports when a lower value is selected.

## Differences from the official PyTorch package

| Area | This project |
| --- | --- |
| Runtime | MLX/Metal on Apple Silicon instead of PyTorch/CUDA |
| Weights | Community MLX conversions from Hugging Face |
| Clone profile | Safe `.qvoice` reference container, not official `.pt` prompt objects |
| Reusable prompt | In-process `VoiceClonePrompt`; not binary-compatible with official prompt files |
| Streaming/batching | Public facade methods plus native MLX batching where input constraints match |
| Long text | Paragraph-aware sequential generation with combined and section outputs |
| Speech tokenizer API | Not wrapped as a standalone public API yet |
| Platform | macOS + Apple Silicon only |

## Development

```bash
python -m pip install -e '.[dev]' --no-deps
python -m pip install 'gradio>=6,<7' numpy scipy soundfile pytest ruff
ruff check .
pytest
```

Unit tests do not download model weights. A real generation run is the integration
test and requires an Apple Silicon Mac. Enable it explicitly with a CustomVoice
checkpoint or local model directory:

```bash
QWEN_MLX_INTEGRATION_MODEL="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit" \
  pytest tests/integration/test_real_generation.py -v
```

## License and credits

Code in this repository is Apache-2.0. Qwen3-TTS is developed by the Alibaba Qwen
team and released under Apache-2.0. The MLX runtime integration is provided by
MLX-Audio, licensed under MIT. Converted model repositories retain their own model
cards and license notices. See [NOTICE](NOTICE).
