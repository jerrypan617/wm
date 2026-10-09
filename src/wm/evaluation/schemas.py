from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EpisodeResult:
    episode_id: int
    split: str
    gamefile: str | None
    success: bool
    steps: int
    invalid_actions: int
    actions: list[str] = field(default_factory=list)
    raw_responses: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    structured_states: list[dict[str, Any]] = field(default_factory=list)
    decision_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
