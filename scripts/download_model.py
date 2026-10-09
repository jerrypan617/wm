#!/usr/bin/env python3
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--revision")
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.model_id,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    print(path)


if __name__ == "__main__":
    main()
