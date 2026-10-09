from __future__ import annotations

from typing import Iterable


DEFAULT_SYSTEM_PROMPT = """You are the action policy for ALFWorld TextWorld.
Complete the household task by interacting with the environment one atomic
step at a time. The environment executes your response immediately, then
returns a new observation. Treat the latest observation as authoritative.
Strict output contract:
- Output exactly one executable TextWorld command on exactly one line.
- Output the command itself only: no reasoning, explanation, Markdown,
  quotation marks, labels, punctuation, or extra lines.
- Never output a sequence of commands, such as "go to ... then open ...".
- Use the exact object names and numeric identifiers from the observation or
  the allowed-command list. Never omit an identifier or invent an object.
- If an allowed-command list is provided, copy exactly one command from it.
Planning rules:
- Use the task in the initial observation as the goal and the current
  observation as the current state.
- Decompose multi-step tasks into atomic actions: navigate, open or inspect a
  container, take the exact target object, navigate to the destination, then
  put the object in/on the destination as requested.
- A failed action or "Nothing happens." means the previous action did not
  change the state; do not blindly repeat it. Choose a state-changing action
  supported by the current observation instead.
- Never repeat the immediately previous command when the environment response
  is unchanged. After arriving at a location, inspect the new contents and
  choose an available take/open/use/move action before navigating away.
- Track completed actions from the recent history: do not re-open an already
  open container, re-take an object that was moved, or oscillate between two
  locations without a new task-relevant reason.
"""


def build_user_prompt(
    initial_observation: str,
    turns: Iterable[tuple[str, str]],
    current_observation: str,
    admissible_commands: list[str] | None = None,
    *,
    max_history_turns: int = 12,
) -> str:
    all_turns = list(turns)
    recent_turns = all_turns[-max_history_turns:]
    parts = [
        "ALFWorld TextWorld action selection",
        "Task and initial state:",
        initial_observation,
    ]
    if recent_turns:
        parts.append("Recent interaction history (oldest to newest):")
        first_step = len(all_turns) - len(recent_turns) + 1
        for index, (action, observation) in enumerate(
            recent_turns, start=first_step
        ):
            parts.append(f"Step {index} - executed command:\n{action}")
            parts.append(f"Step {index} - environment response:\n{observation}")
    parts.append("Current state (use this to choose the next action):")
    parts.append(current_observation)
    if admissible_commands:
        parts.append(
            "Allowed commands — choose exactly one and copy it verbatim:\n"
            + "\n".join(f"- {command}" for command in admissible_commands)
        )
    else:
        parts.append(
            "No allowed-command list is provided. Construct one valid, atomic "
            "TextWorld command using the exact names and identifiers visible "
            "in the current state."
        )
    parts.append(
        "Before answering, silently check that the command is executable in "
        "the current state and advances the task. Return the command on one "
        "line only, for example: go to desk 1"
    )
    return "\n\n".join(parts)
