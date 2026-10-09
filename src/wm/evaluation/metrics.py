from __future__ import annotations

from statistics import mean
from typing import Iterable

from .schemas import EpisodeResult


def summarize(results: Iterable[EpisodeResult]) -> dict[str, float | int]:
    rows = list(results)
    if not rows:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "average_steps": 0.0,
            "average_invalid_actions": 0.0,
        }
    return {
        "episodes": len(rows),
        "success_rate": mean(float(row.success) for row in rows),
        "average_steps": mean(row.steps for row in rows),
        "average_invalid_actions": mean(
            row.invalid_actions for row in rows
        ),
    }
