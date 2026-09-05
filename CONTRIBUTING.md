# Contributing

Issues and pull requests are welcome. Keep the project Mac-first, preserve the
official Qwen3-TTS semantics where MLX-Audio supports them, and document any
intentional incompatibility.

Before opening a pull request:

```bash
ruff check .
pytest
```

Do not commit model weights, generated voices, private reference recordings, or
Hugging Face caches. Tests must not require a checkpoint download unless they are
explicitly marked as opt-in integration tests.
