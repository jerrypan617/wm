from __future__ import annotations

import re


def normalize_action(action: str) -> str:
    return re.sub(r"\s+", " ", action.strip().lower())


def extract_action(response: str) -> str:
    """Turn a model response into one TextWorld command."""
    text = response or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]

    text = text.replace("```text", "").replace("```", "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "look"

    action = lines[-1]
    action = re.sub(
        r"^(?:action|next action)\s*:\s*",
        "",
        action,
        flags=re.IGNORECASE,
    )
    return action.strip().strip("\"'").rstrip(".。") or "look"


def canonicalize_against_admissible(
    action: str,
    admissible_commands: list[str] | None,
) -> str:
    if not admissible_commands:
        return action
    normalized = {
        normalize_action(command): command for command in admissible_commands
    }
    return normalized.get(normalize_action(action), action)
