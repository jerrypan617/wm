from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from wm.models.base import GenerationConfig, ModelBackend
from wm.world_model.state import parse_observation

from .action_parser import canonicalize_against_admissible, extract_action
from .prompt_builder import DEFAULT_SYSTEM_PROMPT, build_user_prompt


@dataclass(frozen=True)
class AgentResponse:
    action: str
    raw_response: str
    user_prompt: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


class LLMTextAgent:
    def __init__(
        self,
        backend: ModelBackend,
        generation: GenerationConfig,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        action_space: str = "free",
    ) -> None:
        if action_space not in {"free", "admissible"}:
            raise ValueError("action_space must be 'free' or 'admissible'")
        self.backend = backend
        self.generation = generation
        self.system_prompt = system_prompt
        self.action_space = action_space

    def act(
        self,
        *,
        initial_observation: str,
        current_observation: str,
        turns: list[tuple[str, str]],
        admissible_commands: list[str] | None = None,
    ) -> AgentResponse:
        commands_for_prompt = (
            admissible_commands if self.action_space == "admissible" else None
        )
        user_prompt = build_user_prompt(
            initial_observation,
            turns,
            current_observation,
            commands_for_prompt,
        )
        return self._act_from_prompt(user_prompt, admissible_commands)

    def _act_from_prompt(
        self,
        user_prompt: str,
        admissible_commands: list[str] | None,
    ) -> AgentResponse:
        raw_response = self.backend.generate(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            self.generation,
        )
        action = extract_action(raw_response)
        action = canonicalize_against_admissible(action, admissible_commands)
        return AgentResponse(action, raw_response, user_prompt)


class StructuredLLMTextAgent(LLMTextAgent):
    """Qwen policy that receives the parsed symbolic state as extra context."""

    def act(
        self,
        *,
        initial_observation: str,
        current_observation: str,
        turns: list[tuple[str, str]],
        admissible_commands: list[str] | None = None,
    ) -> AgentResponse:
        commands_for_prompt = (
            admissible_commands if self.action_space == "admissible" else None
        )
        user_prompt = build_user_prompt(
            initial_observation,
            turns,
            current_observation,
            commands_for_prompt,
        )
        structured_state = parse_observation(current_observation).serialize()
        user_prompt += (
            "\n\nCanonical structured state extracted from the current observation "
            "(use it as a compact state summary; do not invent facts):\n"
            + structured_state
        )
        return self._act_from_prompt(user_prompt, admissible_commands)


class WorldModelGuidedLLMTextAgent(StructuredLLMTextAgent):
    """Qwen policy with an action-conditioned structured-state critic.

    The language model remains responsible for proposing an action.  The
    learned world model predicts the next structured state for a small set of
    admissible candidates.  It is used as a conservative tie-breaker when
    the policy is stalled (for example, it repeats an action after an
    unchanged observation), so an uncertain transition model cannot replace
    verified policy behavior on every step.
    """

    def __init__(
        self,
        backend: ModelBackend,
        generation: GenerationConfig,
        world_model: Any,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        action_space: str = "admissible",
        max_candidates: int = 8,
        override_margin: float = 0.15,
    ) -> None:
        super().__init__(
            backend,
            generation,
            system_prompt=system_prompt,
            action_space=action_space,
        )
        self.world_model = world_model
        self.max_candidates = max(2, max_candidates)
        self.override_margin = override_margin
        self.world_model_calls = 0
        self.world_model_overrides = 0
        self.world_model_errors = 0

    @staticmethod
    def _candidate_actions(
        base_action: str, admissible: Sequence[str], limit: int
    ) -> list[str]:
        candidates = [base_action] if base_action in admissible else []
        # Keep the shortlist deterministic so sharded runs are comparable.
        for command in admissible:
            if command in candidates:
                continue
            candidates.append(command)
            if len(candidates) >= limit:
                break
        return candidates[:limit]

    def _score_prediction(self, initial_observation: str, prediction: Any) -> float:
        # Import lazily to avoid a module-level cycle: alfworld_planner uses
        # AgentResponse from this module.
        from .alfworld_planner import parse_goal

        goal = parse_goal(initial_observation)
        if goal is None:
            return 0.0
        probabilities = prediction.predicted_next_state_probabilities
        facts = self.world_model.vocab.facts
        desired = {
            f"visible_type::{goal.object_type}": 0.35,
        }
        if goal.destination_type:
            desired.update(
                {
                    f"on_type::{goal.object_type}::{goal.destination_type}": 0.8,
                    f"inside_type::{goal.object_type}::{goal.destination_type}": 0.8,
                }
            )
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
        response = super().act(
            initial_observation=initial_observation,
            current_observation=current_observation,
            turns=turns,
            admissible_commands=admissible_commands,
        )
        admissible = list(admissible_commands or [])
        candidates = self._candidate_actions(
            response.action, admissible, self.max_candidates
        )
        if len(candidates) < 2:
            return response

        try:
            predictions = self.world_model.predict_next(
                current_observation, candidates, threshold=0.5
            )
            self.world_model_calls += 1
            scores = [
                self._score_prediction(initial_observation, prediction)
                for prediction in predictions
            ]
            best_index = max(range(len(scores)), key=scores.__getitem__)
            previous_action = turns[-1][0] if turns else None
            unchanged = bool(turns and turns[-1][1] == current_observation)
            stalled = unchanged or response.action == previous_action
            if (
                stalled
                and scores[best_index] >= scores[0] + self.override_margin
            ):
                action = candidates[best_index]
                self.world_model_overrides += 1
                return AgentResponse(
                    action=action,
                    raw_response=(
                        f"{action} [world_model_score="
                        f"{scores[best_index]:.3f}]"
                    ),
                    user_prompt=response.user_prompt,
                )
        except Exception:
            self.world_model_errors += 1
        return response
