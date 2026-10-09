from __future__ import annotations

import re
from typing import Any, Sequence

from wm.models.base import GenerationConfig, ModelBackend
from .alfworld_planner import GoalSpec, parse_goal
from .belief import TargetLocationBelief, location_from_action
from .llm_agent import AgentResponse, LLMTextAgent
from .prompt_builder import DEFAULT_SYSTEM_PROMPT


_GO_TO_RE = re.compile(r"^go to\s+(.+)$", re.IGNORECASE)
_OPEN_RE = re.compile(r"^open\s+(.+)$", re.IGNORECASE)


class EFEWorldModelAgent(LLMTextAgent):
    """Receding-horizon action selection with task progress and epistemic value.

    The policy model proposes an action. A trained action-conditioned world
    model scores a bounded shortlist, while an explicit target-location belief
    supplies a tractable expected-information-gain term. Only the first
    primitive action is executed; the agent then observes and replans.
    """

    def __init__(
        self,
        backend: ModelBackend,
        generation: GenerationConfig,
        world_model: Any,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        action_space: str = "admissible",
        max_candidates: int = 12,
        max_location_candidates: int = 5,
        progress_weight: float = 1.0,
        information_gain_weight: float = 0.35,
        action_cost_weight: float = 0.15,
        revisit_penalty: float = 0.25,
        horizon_discount: float = 0.8,
    ) -> None:
        super().__init__(
            backend,
            generation,
            system_prompt=system_prompt,
            action_space=action_space,
        )
        if max_candidates < 1 or max_location_candidates < 0:
            raise ValueError("candidate limits must be positive/non-negative")
        if not 0.0 < horizon_discount <= 1.0:
            raise ValueError("horizon_discount must be in (0, 1]")
        if min(
            progress_weight,
            information_gain_weight,
            action_cost_weight,
            revisit_penalty,
        ) < 0.0:
            raise ValueError("EFE weights and penalties must be non-negative")
        self.world_model = world_model
        self.max_candidates = max_candidates
        self.max_location_candidates = max_location_candidates
        self.progress_weight = progress_weight
        self.information_gain_weight = information_gain_weight
        self.action_cost_weight = action_cost_weight
        self.revisit_penalty = revisit_penalty
        self.horizon_discount = horizon_discount
        self.goal: GoalSpec | None = None
        self.belief: TargetLocationBelief | None = None
        self._initial_observation: str | None = None
        self._pending_probe: dict[str, Any] | None = None
        self.world_model_calls = 0
        self.world_model_errors = 0
        self.planner_decisions = 0

    def reset_episode(
        self,
        initial_observation: str,
        admissible_commands: list[str] | None = None,
    ) -> None:
        self._initial_observation = initial_observation
        self.goal = parse_goal(initial_observation)
        self.belief = (
            TargetLocationBelief.from_actions(
                self.goal.object_type,
                self.goal.count,
                admissible_commands or [],
            )
            if self.goal is not None
            else None
        )
        self._pending_probe = None
        if self.belief is not None:
            self.belief.record_observation(initial_observation)

    @staticmethod
    def _weighted_goal_score(
        probabilities: Any,
        facts: Sequence[str],
        goal: GoalSpec | None,
    ) -> float:
        if goal is None:
            return 0.0
        desired = {f"visible_type::{goal.object_type}": 0.25}
        if goal.destination_type:
            desired.update(
                {
                    f"on_type::{goal.object_type}::{goal.destination_type}": 1.0,
                    f"inside_type::{goal.object_type}::{goal.destination_type}": 1.0,
                }
            )
        if goal.operation:
            desired[f"{goal.operation}_type::{goal.object_type}"] = 1.0
        score = 0.0
        for fact, weight in desired.items():
            try:
                score += weight * float(probabilities[facts.index(fact)])
            except (ValueError, IndexError, TypeError):
                continue
        try:
            score -= 0.15 * float(
                probabilities[facts.index("feedback::nothing_happens")]
            )
        except (ValueError, IndexError, TypeError):
            pass
        return score

    def _candidate_actions(
        self,
        policy_action: str,
        admissible: Sequence[str],
        belief: TargetLocationBelief | None,
    ) -> list[str]:
        commands = list(dict.fromkeys(admissible))
        candidates: list[str] = []

        def add(command: str) -> None:
            if command in commands and command not in candidates:
                candidates.append(command)

        # Keep the policy proposal in the comparison set. The shortlist then
        # covers immediate information probes and goal-relevant interactions.
        add(policy_action)

        locations: list[tuple[float, int, int, str]] = []
        for index, command in enumerate(commands):
            match = _GO_TO_RE.fullmatch(command.strip())
            if match is None:
                continue
            location, horizon = location_from_action(command)
            if (
                location is not None
                and belief is not None
                and location in belief.opened_locations
            ):
                horizon = 1
            if location is None:
                continue
            eig = belief.expected_information_gain(location) if belief else 0.0
            discounted_eig = eig * self.horizon_discount ** max(0, horizon - 1)
            # Stable ordering: expected information first, then fewer steps,
            # then the environment's deterministic admissible-command order.
            locations.append((-discounted_eig, horizon, index, command))

        open_commands = [
            command for command in commands if _OPEN_RE.fullmatch(command.strip())
        ]
        for command in open_commands[:2]:
            add(command)

        if self.goal is not None:
            target_terms = {
                self.goal.object_type,
                self.goal.destination_type or "",
                self.goal.tool_type or "",
            }
            goal_actions = [
                command
                for command in commands
                if any(term and term in command.lower() for term in target_terms)
                and command.lower().startswith(
                    (
                        "take ",
                        "pick up ",
                        "put ",
                        "move ",
                        "clean ",
                        "heat ",
                        "cool ",
                        "slice ",
                        "toggle ",
                    )
                )
            ]
            for command in goal_actions[:3]:
                add(command)

        locations.sort()
        location_limit = min(
            self.max_location_candidates,
            max(0, self.max_candidates - len(candidates)),
        )
        for _, _, _, command in locations[:location_limit]:
            add(command)

        # Fill remaining slots deterministically so the planner can compare a
        # useful mix even when the policy proposes a non-navigation action.
        for command in commands:
            if len(candidates) >= self.max_candidates:
                break
            add(command)
        return candidates[: self.max_candidates]

    def _action_terms(
        self,
        action: str,
        belief: TargetLocationBelief | None,
    ) -> tuple[str | None, float, float, float, float]:
        location, horizon = location_from_action(action)
        if (
            location is not None
            and belief is not None
            and location in belief.opened_locations
        ):
            horizon = 1
        raw_ig = (
            belief.expected_information_gain(location)
            if belief is not None and location is not None
            else 0.0
        )
        discounted_ig = raw_ig * self.horizon_discount ** max(0, horizon - 1)
        cost = float(max(1, horizon))
        revisit_cost = 0.0
        if belief is not None and location is not None:
            if location in belief.visited_locations:
                revisit_cost += self.revisit_penalty
            if location in belief.inspected_locations:
                revisit_cost += self.revisit_penalty
        return location, raw_ig, discounted_ig, cost, revisit_cost

    def act(
        self,
        *,
        initial_observation: str,
        current_observation: str,
        turns: list[tuple[str, str]],
        admissible_commands: list[str] | None = None,
    ) -> AgentResponse:
        if self._initial_observation != initial_observation:
            self.reset_episode(initial_observation, admissible_commands)
        if self.belief is not None:
            self.belief.add_candidate_locations(admissible_commands or [])

        policy_response = super().act(
            initial_observation=initial_observation,
            current_observation=current_observation,
            turns=turns,
            admissible_commands=admissible_commands,
        )
        admissible = list(admissible_commands or [])
        candidates = self._candidate_actions(
            policy_response.action, admissible, self.belief
        )
        if not candidates:
            return AgentResponse(
                policy_response.action,
                policy_response.raw_response,
                policy_response.user_prompt,
                {"planner": "efe_world_model", "planner_status": "no_candidates"},
            )

        try:
            predictions = self.world_model.predict_next(
                current_observation, candidates, threshold=0.5
            )
            if len(predictions) != len(candidates):
                raise RuntimeError("world model returned a mismatched candidate batch")
            self.world_model_calls += 1
            self.planner_decisions += 1
            facts = self.world_model.vocab.facts
            current_score = self._weighted_goal_score(
                predictions[0].posterior_state_probabilities,
                facts,
                self.goal,
            )
            scored: list[dict[str, Any]] = []
            for index, (action, prediction) in enumerate(zip(candidates, predictions)):
                next_score = self._weighted_goal_score(
                    prediction.predicted_next_state_probabilities,
                    facts,
                    self.goal,
                )
                progress = next_score - current_score
                location, raw_ig, discounted_ig, cost, revisit_cost = (
                    self._action_terms(action, self.belief)
                )
                efe = (
                    -self.progress_weight * progress
                    - self.information_gain_weight * discounted_ig
                    + self.action_cost_weight * cost
                    + revisit_cost
                )
                scored.append(
                    {
                        "action": action,
                        "efe": efe,
                        "predicted_progress": progress,
                        "expected_information_gain": discounted_ig,
                        "raw_expected_information_gain": raw_ig,
                        "estimated_step_cost": cost,
                        "revisit_penalty": revisit_cost,
                        "probe_location": location,
                        "navigation_action": bool(
                            _GO_TO_RE.fullmatch(action.strip())
                        ),
                        "destination_previously_visited": bool(
                            belief is not None
                            and location is not None
                            and location in belief.visited_locations
                        ),
                        "destination_previously_inspected": bool(
                            belief is not None
                            and location is not None
                            and location in belief.inspected_locations
                        ),
                        "policy_proposal": action == policy_response.action,
                        "candidate_order": index,
                    }
                )
            best = min(scored, key=lambda item: item["efe"])
            self._update_pending_probe(best)
            diagnostics = {
                "planner": "efe_world_model",
                "planner_status": "scored",
                "chosen_action": best["action"],
                "policy_proposal": policy_response.action,
                "belief_entropy_bits": (
                    self.belief.entropy_bits() if self.belief is not None else 0.0
                ),
                "current_goal_score": current_score,
                "chosen_predicted_progress": best["predicted_progress"],
                "expected_information_gain": best["expected_information_gain"],
                "raw_expected_information_gain": best[
                    "raw_expected_information_gain"
                ],
                "estimated_step_cost": best["estimated_step_cost"],
                "probe_location": best["probe_location"],
                "navigation_action": best["navigation_action"],
                "destination_previously_visited": best[
                    "destination_previously_visited"
                ],
                "destination_previously_inspected": best[
                    "destination_previously_inspected"
                ],
                "efe": best["efe"],
                "candidate_scores": scored,
            }
            return AgentResponse(
                action=best["action"],
                raw_response=policy_response.raw_response,
                user_prompt=policy_response.user_prompt,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            self.world_model_errors += 1
            self._pending_probe = None
            return AgentResponse(
                action=policy_response.action,
                raw_response=policy_response.raw_response,
                user_prompt=policy_response.user_prompt,
                diagnostics={
                    "planner": "efe_world_model",
                    "planner_status": "policy_fallback",
                    "planner_error_type": type(exc).__name__,
                },
            )

    def _update_pending_probe(self, choice: dict[str, Any]) -> None:
        location = choice["probe_location"]
        if location is None or choice["raw_expected_information_gain"] <= 0.0:
            self._pending_probe = None
            return
        if self._pending_probe and self._pending_probe["location"] == location:
            return
        self._pending_probe = {
            "location": location,
            "expected_information_gain": choice["raw_expected_information_gain"],
            "planning_action": choice["action"],
            "entropy_before": (
                self.belief.entropy_bits() if self.belief is not None else 0.0
            ),
        }

    def observe_transition(
        self,
        action: str,
        observation: str,
    ) -> dict[str, Any]:
        if self.belief is None:
            return {}
        update = self.belief.record_observation(observation, action=action)
        result: dict[str, Any] = {
            "belief_entropy_before_transition": update["belief_entropy_before"],
            "belief_entropy_after_transition": update["belief_entropy_after"],
            "realized_information_gain": update["realized_information_gain"],
            "inspected_location": update["inspected_location"],
        }
        inspected_location = update["inspected_location"]
        if (
            self._pending_probe is not None
            and inspected_location == self._pending_probe["location"]
        ):
            probe_realized_ig = max(
                0.0,
                float(self._pending_probe["entropy_before"])
                - float(update["belief_entropy_after"]),
            )
            result.update(
                {
                    "probe_completed": True,
                    "probe_location": inspected_location,
                    "probe_planning_action": self._pending_probe["planning_action"],
                    "probe_expected_information_gain": self._pending_probe[
                        "expected_information_gain"
                    ],
                    "probe_realized_information_gain": probe_realized_ig,
                }
            )
            self._pending_probe = None
        else:
            result["probe_completed"] = False
        return result
