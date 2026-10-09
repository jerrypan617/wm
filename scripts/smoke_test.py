#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.models import GenerationConfig, TransformersBackend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--prompt", default="Who are you? Answer briefly.")
    args = parser.parse_args()

    backend = TransformersBackend(
        args.model_id,
        args.cache_dir,
        local_files_only=True,
    )
    response = backend.generate(
        [{"role": "user", "content": args.prompt}],
        GenerationConfig(max_new_tokens=64, enable_thinking=False),
    )
    print(response)


if __name__ == "__main__":
    main()
