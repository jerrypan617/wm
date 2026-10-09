from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable


_ENTITY_RE = re.compile(r"\b([a-z][a-z0-9_]*\s+\d+)\b", re.IGNORECASE)
_TASK_RE = re.compile(r"your task is to:\s*(.+?)(?:\.|$)", re.IGNORECASE)
_ARRIVAL_RE = re.compile(
    r"you arrive at\s+([a-z][a-z0-9_]*\s+\d+)", re.IGNORECASE
)
_CLOSED_RE = re.compile(
    r"the\s+([a-z][a-z0-9_]*\s+\d+)\s+is\s+(closed|open)",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"you\s+(open|close|take|pick up|drop|clean|heat|cool|slice|toggle)\s+"
    r"(.+?)(?:\.|$)",
    re.IGNORECASE,
)
_PUT_RE = re.compile(
    r"you\s+put\s+(.+?)\s+(in|on)\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)


def normalize_entity(value: str) -> str:
    """Normalize an ALFWorld entity while retaining its instance id."""

    value = re.sub(r"\s+", " ", value.strip().lower())
    value = re.sub(r"\s+(\d+)$", r"_\1", value)
    return value.replace(" ", "_")


def normalize_fact(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def entity_type(entity: str) -> str:
    """Return an entity's type while retaining the exact entity elsewhere."""

    return re.sub(r"_\d+$", "", entity)


def _entity_tokens(text: str) -> list[str]:
    return [normalize_entity(match.group(1)) for match in _ENTITY_RE.finditer(text)]


def _facts_from_visible_list(text: str, relation: str, container: str | None) -> set[str]:
    facts: set[str] = set()
    for entity in _entity_tokens(text):
        facts.add(f"visible::{entity}")
        facts.add(f"visible_type::{entity_type(entity)}")
        if container is not None:
            facts.add(f"{relation}::{entity}::{container}")
            facts.add(
                f"{relation}_type::{entity_type(entity)}::{entity_type(container)}"
            )
    return facts


@dataclass(frozen=True)
class StructuredState:
    """A compact, serializable belief-state target.

    Facts are intentionally represented as canonical strings. This gives the
    world model a stable discrete state vocabulary while preserving the
    entity identifiers used by TextWorld. A fact can be an attribute
    (``closed::cabinet_1``) or a relation
    (``inside::apple_1::fridge_1``).
    """

    task: str | None = None
    agent_location: str | None = None
    facts: tuple[str, ...] = field(default_factory=tuple)
    unknown: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "facts",
            tuple(sorted({normalize_fact(fact) for fact in self.facts if fact})),
        )
        object.__setattr__(
            self,
            "unknown",
            tuple(sorted({normalize_fact(fact) for fact in self.unknown if fact})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "agent_location": self.agent_location,
            "facts": list(self.facts),
            "unknown": list(self.unknown),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StructuredState":
        return cls(
            task=payload.get("task"),
            agent_location=payload.get("agent_location"),
            facts=tuple(payload.get("facts", ())),
            unknown=tuple(payload.get("unknown", ())),
        )

    def serialize(self) -> str:
        parts = []
        if self.task:
            parts.append(f"task={self.task}")
        if self.agent_location:
            parts.append(f"agent_at={self.agent_location}")
        parts.extend(f"fact={fact}" for fact in self.facts)
        parts.extend(f"unknown={fact}" for fact in self.unknown)
        return "\n".join(parts) or "empty_state"


def parse_observation(observation: str) -> StructuredState:
    """Extract conservative structured facts from an ALFWorld observation.

    This parser only emits facts explicitly stated in the observation. It is
    deliberately conservative: hidden container contents are not inferred.
    The parser is useful for bootstrapping a dataset; symbolic TextWorld facts
    can later replace it as higher-quality training targets.
    """

    text = observation or ""
    lower = text.lower()
    facts: set[str] = set()
    unknown: set[str] = set()

    task_match = _TASK_RE.search(text)
    task = task_match.group(1).strip() if task_match else None
    if task:
        facts.add(f"goal::{normalize_fact(task)}")

    location: str | None = None
    arrival_match = _ARRIVAL_RE.search(text)
    if arrival_match:
        location = normalize_entity(arrival_match.group(1))
        facts.add(f"agent_at::{location}")
        facts.add(f"agent_at_type::{entity_type(location)}")

    for match in _CLOSED_RE.finditer(text):
        entity = normalize_entity(match.group(1))
        status = match.group(2).lower()
        facts.add(f"{status}::{entity}")
        facts.add(f"{status}_type::{entity_type(entity)}")

    for match in _ACTION_RE.finditer(text):
        verb = match.group(1).lower().replace(" ", "_")
        entities = _entity_tokens(match.group(2))
        if entities:
            facts.add(f"{verb}::{entities[0]}")
            facts.add(f"{verb}_type::{entity_type(entities[0])}")

    for match in _PUT_RE.finditer(text):
        source_entities = _entity_tokens(match.group(1))
        target_entities = _entity_tokens(match.group(3))
        if source_entities and target_entities:
            relation = "inside" if match.group(2).lower() == "in" else "on"
            facts.add(f"{relation}::{source_entities[0]}::{target_entities[0]}")
            facts.add(
                f"{relation}_type::{entity_type(source_entities[0])}::"
                f"{entity_type(target_entities[0])}"
            )

    if location:
        # The first visible list after an arrival describes objects on the
        # arrived-at surface. Do not attach unrelated objects elsewhere.
        arrival_end = arrival_match.end() if arrival_match else 0
        visible_tail = text[arrival_end:]
        visible_match = re.search(r"you see\s+(.+?)(?:\.|$)", visible_tail, re.I)
        if visible_match:
            facts.update(_facts_from_visible_list(visible_match.group(1), "on", location))
    else:
        visible_match = re.search(r"you see\s+(.+?)(?:\.|$)", text, re.I)
        if visible_match:
            facts.update(_facts_from_visible_list(visible_match.group(1), "visible", None))

    # Opening a container reports its contents as ``In it, you see ...``
    # rather than as an arrival at a location. Preserve the containment
    # relation so downstream planners can locate objects hidden in drawers,
    # cabinets, fridges, and microwaves.
    opened = re.search(r"you open the\s+([a-z][a-z0-9_]*\s+\d+)", text, re.I)
    if opened:
        container = normalize_entity(opened.group(1))
        visible_match = re.search(r"in it,\s+you see\s+(.+?)(?:\.|$)", text, re.I)
        if visible_match:
            facts.update(
                _facts_from_visible_list(
                    visible_match.group(1), "inside", container
                )
            )

    if "nothing happens" in lower:
        facts.add("feedback::nothing_happens")

    for phrase in ("closed", "hidden", "not visible"):
        if phrase in lower:
            unknown.add(f"observation_mentions::{phrase.replace(' ', '_')}")

    return StructuredState(
        task=task,
        agent_location=location,
        facts=tuple(facts),
        unknown=tuple(unknown),
    )


def state_facts(states: Iterable[StructuredState]) -> set[str]:
    facts: set[str] = set()
    for state in states:
        facts.update(state.facts)
        facts.update(f"unknown::{fact}" for fact in state.unknown)
    return facts
