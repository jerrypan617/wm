from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable

from wm.world_model.state import entity_type, normalize_entity, parse_observation


_ENTITY = r"[a-z][a-z0-9_]*\s+\d+"
_GO_TO_RE = re.compile(rf"^go to (?P<entity>{_ENTITY})$", re.IGNORECASE)
_OPEN_RE = re.compile(rf"^open (?P<entity>{_ENTITY})$", re.IGNORECASE)
_OPENED_INVENTORY_RE = re.compile(
    rf"you open the (?P<entity>{_ENTITY}).*?in it,\s*you see\s+(?P<items>.+?)(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
_ARRIVAL_INVENTORY_RE = re.compile(
    rf"you arrive at (?P<entity>{_ENTITY}).*?"
    rf"(?:on the (?P<surface>{_ENTITY}),\s*you see|"
    rf"in it,\s*you see)\s+(?P<items>.+?)(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
_VISIBLE_ENTITY_RE = re.compile(_ENTITY, re.IGNORECASE)

_CONTAINER_TYPES = {
    "bathtub",
    "cabinet",
    "drawer",
    "fridge",
    "laundryhamper",
    "microwave",
    "safe",
}


def location_from_action(action: str) -> tuple[str | None, int]:
    """Return the inspected location and lookahead horizon for a primitive action.

    Surface inspection happens on arrival. A closed container needs a second
    primitive action (`open`) before its contents are observed.
    """

    match = _OPEN_RE.fullmatch(action.strip())
    if match:
        return normalize_entity(match.group("entity")), 1
    match = _GO_TO_RE.fullmatch(action.strip())
    if match:
        location = normalize_entity(match.group("entity"))
        horizon = 2 if entity_type(location) in _CONTAINER_TYPES else 1
        return location, horizon
    return None, 0


def _location_inventory(observation: str) -> tuple[str, list[str]] | None:
    """Parse only observations that explicitly enumerate a surface/container."""

    text = observation or ""
    match = _OPENED_INVENTORY_RE.search(text)
    if match:
        location = normalize_entity(match.group("entity"))
        return location, _VISIBLE_ENTITY_RE.findall(match.group("items"))
    match = _ARRIVAL_INVENTORY_RE.search(text)
    if match:
        location = normalize_entity(match.group("surface") or match.group("entity"))
        return location, _VISIBLE_ENTITY_RE.findall(match.group("items"))
    return None


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(
        1.0 - probability
    )


@dataclass
class TargetLocationBelief:
    """Exchangeable belief over which uninspected locations contain goal items.

    The tractable prior assumes the remaining required items occupy distinct
    uninspected locations uniformly. It is exact for one target item and a
    useful, explicit approximation for multi-item ALFWorld tasks.
    """

    object_type: str
    required_count: int = 1
    locations: set[str] = field(default_factory=set)
    inspected_locations: set[str] = field(default_factory=set)
    visited_locations: set[str] = field(default_factory=set)
    opened_locations: set[str] = field(default_factory=set)
    found_entities: set[str] = field(default_factory=set)
    found_locations: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_actions(
        cls,
        object_type: str,
        required_count: int,
        admissible_commands: Iterable[str],
    ) -> "TargetLocationBelief":
        locations = {
            normalize_entity(match.group("entity"))
            for command in admissible_commands
            if (match := _GO_TO_RE.fullmatch(command.strip())) is not None
        }
        return cls(
            object_type=object_type.strip().lower().replace(" ", "_"),
            required_count=max(1, required_count),
            locations=locations,
        )

    @property
    def remaining_count(self) -> int:
        return max(0, self.required_count - len(self.found_entities))

    @property
    def unknown_locations(self) -> set[str]:
        return self.locations - self.inspected_locations

    def add_candidate_locations(self, commands: Iterable[str]) -> None:
        for command in commands:
            match = _GO_TO_RE.fullmatch(command.strip())
            if match is not None:
                self.locations.add(normalize_entity(match.group("entity")))

    def entropy_bits(self) -> float:
        n = len(self.unknown_locations)
        m = min(self.remaining_count, n)
        if n == 0 or m == 0:
            return 0.0
        # Uniform hypotheses over the C(n, m) possible occupied-location sets.
        return math.log2(math.comb(n, m))

    def expected_information_gain(self, location: str) -> float:
        location = normalize_entity(location)
        n = len(self.unknown_locations)
        m = min(self.remaining_count, n)
        if n == 0 or m == 0 or location not in self.unknown_locations:
            return 0.0
        # Inspecting a location reveals one Bernoulli membership event. Under
        # the uniform m-of-n prior its expected entropy reduction is h2(m/n).
        return _binary_entropy(m / n)

    def record_observation(
        self,
        observation: str,
        *,
        action: str | None = None,
    ) -> dict[str, float | int | str | None]:
        entropy_before = self.entropy_bits()
        inventory = _location_inventory(observation)
        inspected_location: str | None = None
        observed_targets: list[str] = []

        if inventory is not None:
            raw_location, raw_entities = inventory
            inspected_location = normalize_entity(raw_location)
            observed_targets = [
                normalize_entity(entity)
                for entity in raw_entities
                if entity_type(normalize_entity(entity)) == self.object_type
            ]
            self.locations.add(inspected_location)
            self.inspected_locations.add(inspected_location)

        parsed = parse_observation(observation)
        if parsed.agent_location:
            self.visited_locations.add(normalize_entity(parsed.agent_location))
        for fact in parsed.facts:
            if fact.startswith("open::"):
                self.opened_locations.add(fact.split("::", 1)[1])
            if not fact.startswith("visible::"):
                continue
            entity = fact.split("::", 1)[1]
            if entity_type(entity) == self.object_type and entity not in observed_targets:
                observed_targets.append(entity)

        # Goal objects found in an explicitly enumerated inventory are evidence
        # about their location. Other sightings still count toward the task.
        for entity in observed_targets:
            self.found_entities.add(entity)
            if inspected_location is not None:
                self.found_locations[entity] = inspected_location

        entropy_after = self.entropy_bits()
        realized_ig = max(0.0, entropy_before - entropy_after)
        return {
            "belief_entropy_before": entropy_before,
            "belief_entropy_after": entropy_after,
            "realized_information_gain": realized_ig,
            "inspected_location": inspected_location,
            "found_target_count": len(observed_targets),
        }
