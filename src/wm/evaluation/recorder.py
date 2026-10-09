from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .schemas import EpisodeResult


def write_jsonl(path: str | Path, results: Iterable[EpisodeResult]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            for result in results:
                handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def read_jsonl(path: str | Path) -> list[EpisodeResult]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {input_path}")

    results: list[EpisodeResult] = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            results.append(EpisodeResult(**payload))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid checkpoint record at {input_path}:{line_number}"
            ) from exc
    return results


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
