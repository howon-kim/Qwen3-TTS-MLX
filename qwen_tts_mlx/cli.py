"""Command-line launcher for the Gradio application."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen-tts-mlx",
        description="Launch the Qwen3-TTS MLX demo on an Apple Silicon Mac.",
    )
    parser.add_argument("--host", "--ip", default="127.0.0.1", dest="host")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--share", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inbrowser", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-keyfile")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .app import build_demo, launch_options

    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "inbrowser": args.inbrowser,
    }
    if args.ssl_certfile:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile
    launch_kwargs.update(launch_options())
    build_demo().queue(default_concurrency_limit=1).launch(**launch_kwargs)
    return 0
