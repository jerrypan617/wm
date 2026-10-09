from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from wm.world_model.state import entity_type, parse_observation

from .llm_agent import AgentResponse


_ENTITY = r"[a-z][a-z0-9_]*\s+\d+"
_GO_TO_RE = re.compile(rf"^go to (?P<entity>{_ENTITY})$", re.IGNORECASE)
_RELATION_RE = re.compile(r"^(?:on|inside)::([^:]+)::([^:]+)$")
_TASK_RE = re.compile(r"your task is to:\s*(.+?)(?:\.|$)", re.IGNORECASE)


@dataclass
class GoalSpec:
    object_type: str
    destination_type: str | None
    count: int = 1
    operation: str | None = None
    task_kind: str = "transfer"
    tool_type: str | None = None


def _clean_type(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def parse_goal(initial_observation: str) -> GoalSpec | None:
    match = _TASK_RE.search(initial_observation or "")
    if not match:
        return None
    task = match.group(1).strip().lower()
    operation = next(
        (name for name in ("clean", "heat", "cool", "slice", "toggle") if name in task),
        None,
    )
    if operation is None:
        if re.search(r"\b(?:hot|heated)\b", task):
            operation = "heat"
        elif re.search(r"\bcool\b", task):
            operation = "cool"
        elif re.search(r"\bclean\b", task):
            operation = "clean"
        elif re.search(r"\bsliced\b", task):
            operation = "slice"
    task_kind = "inspect" if re.search(r"\b(?:examine|look at)\b", task) else "transfer"
    object_match = re.search(
        r"(?:put|find|pick up|take|clean|heat|cool|slice|toggle|examine|look at)\s+"
        r"(?:(?:the|one|two|three|four|some|a|an)\s+)?"
        r"([a-z][a-z0-9_]*)",
        task,
    )
    if object_match is None:
        return None
    count_match = re.search(r"\b(two|three|four)\b", task)
    count = {"two": 2, "three": 3, "four": 4}.get(
        count_match.group(1), 1
    ) if count_match else 1
    destination_match = re.search(
        r"\b(?:in|on|into|onto)\s+(?:the\s+|a\s+|an\s+)?"
        r"([a-z][a-z0-9_]*)\b",
        task,
    )
    destination = (
        _clean_type(destination_match.group(1)) if destination_match else None
    )
    tool_match = re.search(
        r"\b(?:under|with)\s+(?:the\s+)?([a-z][a-z0-9_]*)\b", task
    )
    tool_type = _clean_type(tool_match.group(1)) if tool_match else None
    object_type = _clean_type(object_match.group(1))
    if object_type in {"clean", "cool", "hot", "heated", "sliced"}:
        adjective_match = re.search(
            rf"\b{re.escape(object_type)}\s+([a-z][a-z0-9_]*)\b", task
        )
        if adjective_match:
            object_type = _clean_type(adjective_match.group(1))
    return GoalSpec(
        object_type=object_type,
        destination_type=destination,
        count=count,
        operation=operation,
        task_kind=task_kind,
        tool_type=tool_type,
    )


class StructuredALFWorldAgent:
    """Observation-only structured planner for admissible ALFWorld actions."""

    def __init__(self) -> None:
        self._initial = ""
        self._goal: GoalSpec | None = None
        self._seen_observations: set[str] = set()
        self._known_entities: set[str] = set()
        self._sources: dict[str, str] = {}
        self._held: set[str] = set()
        self._placed: set[str] = set()
        self._completed_operations: set[tuple[str, str]] = set()
        self._used_tools: set[str] = set()
        self._visited: set[str] = set()
        self._opened: set[str] = set()
        self._actions: list[str] = []
        self._current_location: str | None = None

    def _reset(self, initial_observation: str) -> None:
        self._initial = initial_observation
        self._goal = parse_goal(initial_observation)
        self._seen_observations.clear()
        self._known_entities.clear()
        self._sources.clear()
        self._held.clear()
        self._placed.clear()
        self._completed_operations.clear()
        self._used_tools.clear()
        self._visited.clear()
        self._opened.clear()
        self._actions.clear()
        self._current_location = None

    def _ingest(self, observation: str) -> None:
        self._seen_observations.add(observation)
        state = parse_observation(observation)
        if state.agent_location:
            self._current_location = state.agent_location
            self._visited.add(state.agent_location)
        for fact in state.facts:
            if fact.startswith("visible::"):
                self._known_entities.add(fact.split("::", 1)[1])
            relation_match = _RELATION_RE.match(fact)
            if relation_match:
                entity, source = relation_match.groups()
                self._known_entities.update((entity, source))
                self._sources[entity] = source
        lower = observation.lower()
        for match in re.finditer(rf"you pick up the ({_ENTITY})", lower):
            self._held.add(match.group(1).replace(" ", "_"))
        for match in re.finditer(rf"you move the ({_ENTITY}) to the ({_ENTITY})", lower):
            entity, destination = (
                match.group(1).replace(" ", "_"),
                match.group(2).replace(" ", "_"),
            )
            self._held.discard(entity)
            self._placed.add(entity)
            self._sources[entity] = destination
        for match in re.finditer(rf"you (clean|heat|cool|slice|toggle) the ({_ENTITY})", lower):
            self._completed_operations.add((match.group(2).replace(" ", "_"), match.group(1)))
        for match in re.finditer(rf"you open the ({_ENTITY})", lower):
            self._opened.add(match.group(1).replace(" ", "_"))
        for match in re.finditer(rf"you turn on the ({_ENTITY})", lower):
            self._used_tools.add(entity_type(match.group(1).replace(" ", "_")))

    @staticmethod
    def _first(admissible: list[str], pattern: str) -> str | None:
        compiled = re.compile(pattern, re.IGNORECASE)
        return next((command for command in admissible if compiled.fullmatch(command)), None)

    def _entities_of_type(self, object_type: str, admissible: list[str]) -> list[str]:
        entities = set(self._known_entities)
        for command in admissible:
            for match in re.finditer(_ENTITY, command.lower()):
                entities.add(match.group(0).replace(" ", "_"))
        return sorted(entity for entity in entities if entity_type(entity) == object_type)

    def _go_to(self, object_type: str, admissible: list[str]) -> str | None:
        for command in admissible:
            match = _GO_TO_RE.match(command.lower())
            if not match:
                continue
            entity = match.group("entity").replace(" ", "_")
            if entity_type(entity) == object_type and entity not in self._visited:
                return command
        return None

    def _exploration_action(self, admissible: list[str]) -> str | None:
        candidates: list[tuple[int, int, str]] = []
        for index, command in enumerate(admissible):
            match = _GO_TO_RE.match(command.lower())
            if not match:
                continue
            entity = match.group("entity").replace(" ", "_")
            if entity in self._visited:
                continue
            # ALFWorld presents many drawers/cabinets before the useful
            # surfaces. Search surfaces and appliances first, then hidden
            # containers, so a late surface object still leaves budget for
            # take -> travel -> move.
            priority = 1 if entity_type(entity) in {"cabinet", "drawer", "safe"} else 0
            candidates.append((priority, index, command))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][2]

    def _choose_target(self, admissible: list[str]) -> str | None:
        if self._goal is None:
            return None
        target_type = self._goal.object_type
        candidates = self._entities_of_type(target_type, admissible)
        candidates = [entity for entity in candidates if entity not in self._placed]
        if self._held:
            return next(
                (entity for entity in self._held if entity_type(entity) == target_type),
                None,
            )
        return next(iter(candidates), None)

    def _target_action(self, target: str, admissible: list[str]) -> str | None:
        destination = self._sources.get(target)
        if target in self._held:
            return None
        target_text = target.replace("_", " ")
        if destination and self._current_location == destination:
            source_text = destination.replace("_", " ")
            if self._goal and self._goal.task_kind == "inspect":
                if self._goal.tool_type and self._goal.tool_type not in self._used_tools:
                    use_tool = self._first(
                        admissible,
                        rf"use {re.escape(self._goal.tool_type)} \d+",
                    )
                    if use_tool:
                        return use_tool
                inspected = self._first(
                    admissible, rf"(?:examine|look at) {re.escape(target_text)}"
                )
                if inspected:
                    return inspected
                if self._goal.tool_type in self._used_tools:
                    take = self._first(
                        admissible,
                        rf"take {re.escape(target_text)} from {re.escape(source_text)}",
                    )
                    if take:
                        return take
            take = self._first(admissible, rf"take {re.escape(target_text)} from {re.escape(source_text)}")
            if take:
                return take
            open_command = self._first(admissible, rf"open {re.escape(source_text)}")
            if open_command and destination not in self._opened:
                return open_command
        if destination:
            return self._first(admissible, rf"go to {re.escape(destination.replace('_', ' '))}")
        return None

    def _held_action(self, target: str, admissible: list[str]) -> str | None:
        if self._goal is None or target not in self._held:
            return None
        if self._goal.task_kind == "inspect":
            if self._goal.tool_type is None:
                return None
            if self._goal.tool_type not in self._used_tools:
                tool_type = self._goal.tool_type
                for command in admissible:
                    if re.fullmatch(rf"use {re.escape(tool_type)} \d+", command, re.I):
                        return command
                tool_entity = next(
                    (
                        entity
                        for entity in self._entities_of_type(tool_type, admissible)
                        if entity in self._sources
                    ),
                    None,
                )
                if tool_entity:
                    travel = self._first(
                        admissible,
                        rf"go to {re.escape(self._sources[tool_entity].replace('_', ' '))}",
                    )
                    if travel:
                        return travel
                travel = self._go_to(tool_type, admissible)
                if travel:
                    return travel
                return self._exploration_action(admissible)
            return None
        operation = self._goal.operation
        target_text = target.replace("_", " ")
        if operation and (target, operation) not in self._completed_operations:
            direct = self._first(admissible, rf"{operation} {re.escape(target_text)} with {_ENTITY}")
            if direct:
                return direct
            operation_location = {"clean": "sinkbasin", "heat": "microwave", "cool": "fridge"}.get(operation)
            if operation_location and self._current_location is not None:
                travel = self._first(admissible, rf"go to {operation_location} \d+")
                if travel:
                    return travel
        if self._goal.destination_type:
            destinations = self._entities_of_type(self._goal.destination_type, admissible)
            if destinations:
                destination = destinations[0]
                destination_text = destination.replace("_", " ")
                if self._current_location == destination:
                    move = self._first(
                        admissible,
                        rf"(?:move|put) {re.escape(target_text)} (?:to|in|on) {re.escape(destination_text)}",
                    )
                    if move:
                        return move
                travel = self._first(admissible, rf"go to {re.escape(destination_text)}")
                if travel:
                    return travel
        return None

    def act(
        self,
        *,
        initial_observation: str,
        current_observation: str,
        turns: list[tuple[str, str]],
        admissible_commands: list[str] | None = None,
    ) -> AgentResponse:
        if initial_observation != self._initial:
            self._reset(initial_observation)
        self._ingest(initial_observation)
        for action, observation in turns:
            if action not in self._actions:
                self._actions.append(action)
                match = _GO_TO_RE.match(action.lower())
                if match:
                    self._current_location = match.group("entity").replace(" ", "_")
                    self._visited.add(self._current_location)
            self._ingest(observation)
        self._ingest(current_observation)
        admissible = list(admissible_commands or [])
        target = self._choose_target(admissible)
        action = self._held_action(target, admissible) if target else None
        if action is None and target:
            action = self._target_action(target, admissible)
        if action is None and self._goal:
            if self._current_location:
                open_current = self._first(
                    admissible,
                    rf"open {re.escape(self._current_location.replace('_', ' '))}",
                )
                if open_current:
                    action = open_current
        if action is None and self._goal:
            # Search unexplored locations before spending a step at the final
            # destination.  This preserves enough budget for the common
            # worst case where the object is in the last drawer/sidetable
            # inspected and still needs take -> travel -> move.
            action = self._exploration_action(admissible)
        if action is None:
            action = self._first(admissible, r"look")
        if action is None and admissible:
            action = admissible[0]
        return AgentResponse(action=action or "look", raw_response=action or "look", user_prompt="")


class WorldModelGuidedStructuredAgent:
    """Conservative World-Model wrapper around the symbolic planner.

    The symbolic planner remains the safety policy.  When it would emit a
    blind ``look`` fallback, the learned transition model scores a small set
    of admissible alternatives and can replace that fallback only when the
    predicted goal-relevant state is materially better.  This makes the
    learned model part of action selection without allowing uncertain state
    predictions to override verified take/open/move operations.
    """

    def __init__(
        self,
        world_model: Any,
        *,
        max_candidates: int = 8,
        override_margin: float = 0.15,
    ) -> None:
        self.base = StructuredALFWorldAgent()
        self.world_model = world_model
        self.max_candidates = max(2, max_candidates)
        self.override_margin = override_margin
        self.wm_calls = 0
        self.wm_overrides = 0
        self.wm_errors = 0

    @staticmethod
    def _candidate_actions(
        base_action: str, admissible: Sequence[str], limit: int
    ) -> list[str]:
        candidates: list[str] = [base_action]
        for command in admissible:
            lower = command.lower()
            if command in candidates or lower == "look":
                continue
            if lower.startswith(("go to ", "open ", "take ", "pick up ")):
                candidates.append(command)
            if len(candidates) >= limit:
                break
        return candidates[:limit]

    def _score_prediction(self, prediction: Any) -> float:
        goal = self.base._goal
        if goal is None:
            return 0.0
        probabilities = prediction.predicted_next_state_probabilities
        facts = self.world_model.vocab.facts
        desired = {
            f"visible_type::{goal.object_type}": 0.35,
            f"on_type::{goal.object_type}::{goal.destination_type}": 0.8,
            f"inside_type::{goal.object_type}::{goal.destination_type}": 0.8,
        }
        if goal.operation:
            desired[f"{goal.operation}_type::{goal.object_type}"] = 0.9
        score = 0.0
        for fact, weight in desired.items():
            try:
                score += weight * float(probabilities[facts.index(fact)])
            except (ValueError, IndexError):
                continue
        try:
            score -= 0.15 * float(
                probabilities[facts.index("feedback::nothing_happens")]
            )
        except (ValueError, IndexError):
            pass
        return score

    def act(
        self,
        *,
        initial_observation: str,
        current_observation: str,
        turns: list[tuple[str, str]],
        admissible_commands: list[str] | None = None,
    ) -> AgentResponse:
        response = self.base.act(
            initial_observation=initial_observation,
            current_observation=current_observation,
            turns=turns,
            admissible_commands=admissible_commands,
        )
        admissible = list(admissible_commands or [])
        if response.action != "look" or not admissible:
            return response
        candidates = self._candidate_actions(
            response.action, admissible, self.max_candidates
        )
        if len(candidates) < 2:
            return response
        try:
            predictions = self.world_model.predict_next(
                current_observation, candidates, threshold=0.5
            )
            self.wm_calls += 1
            scores = [self._score_prediction(prediction) for prediction in predictions]
            best_index = max(range(len(scores)), key=scores.__getitem__)
            if scores[best_index] >= scores[0] + self.override_margin:
                action = candidates[best_index]
                self.wm_overrides += 1
                return AgentResponse(
                    action=action,
                    raw_response=f"{action} [world_model_score={scores[best_index]:.3f}]",
                    user_prompt="",
                )
        except Exception:
            self.wm_errors += 1
        return response
