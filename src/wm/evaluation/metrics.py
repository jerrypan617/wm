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
            "average_success_steps": 0.0,
            "consecutive_action_repeat_rate": 0.0,
            "navigation_revisit_rate": 0.0,
            "mean_expected_information_gain": 0.0,
            "mean_realized_information_gain": 0.0,
            "completed_probes": 0,
            "probe_information_gain_mae": 0.0,
        }
    successful = [row.steps for row in rows if row.success]
    repeated_actions = sum(
        sum(left == right for left, right in zip(row.actions, row.actions[1:]))
        for row in rows
    )
    action_transitions = sum(max(0, len(row.actions) - 1) for row in rows)
    diagnostics = [
        item for row in rows for item in row.decision_diagnostics
    ]
    expected_ig = [
        float(item["expected_information_gain"])
        for item in diagnostics
        if isinstance(item.get("expected_information_gain"), (int, float))
    ]
    realized_ig = [
        float(item["realized_information_gain"])
        for item in diagnostics
        if isinstance(item.get("realized_information_gain"), (int, float))
    ]
    navigation_decisions = [
        item
        for item in diagnostics
        if item.get("navigation_action")
    ]
    navigation_revisits = sum(
        bool(item.get("destination_previously_visited"))
        or bool(item.get("destination_previously_inspected"))
        for item in navigation_decisions
    )
    completed_probes = [
        item
        for item in diagnostics
        if item.get("probe_completed")
        and isinstance(item.get("probe_expected_information_gain"), (int, float))
        and isinstance(item.get("probe_realized_information_gain"), (int, float))
    ]
    probe_mae = (
        mean(
            abs(
                float(item["probe_expected_information_gain"])
                - float(item["probe_realized_information_gain"])
            )
            for item in completed_probes
        )
        if completed_probes
        else 0.0
    )
    return {
        "episodes": len(rows),
        "success_rate": mean(float(row.success) for row in rows),
        "average_steps": mean(row.steps for row in rows),
        "average_invalid_actions": mean(
            row.invalid_actions for row in rows
        ),
        "average_success_steps": mean(successful) if successful else 0.0,
        "consecutive_action_repeat_rate": (
            repeated_actions / action_transitions if action_transitions else 0.0
        ),
        "navigation_revisit_rate": (
            navigation_revisits / len(navigation_decisions)
            if navigation_decisions
            else 0.0
        ),
        "mean_expected_information_gain": mean(expected_ig) if expected_ig else 0.0,
        "mean_realized_information_gain": mean(realized_ig) if realized_ig else 0.0,
        "completed_probes": len(completed_probes),
        "probe_information_gain_mae": probe_mae,
    }
