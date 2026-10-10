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
_OBSERVATION_ACTION_RE = re.compile(
    r"^(?:look|examine(?:\s+.+)?|look at(?:\s+.+)?|inventory)$",
    re.IGNORECASE,
)


class EFEWorldModelAgent(LLMTextAgent):
    """Receding-horizon action selection with task progress and epistemic value.

    The policy model proposes an action. A trained action-conditioned world
    model scores a bounded shortlist, while an explicit target-location belief
    supplies a tractable expected-information-gain term. The policy remains
    the default controller; EFE is a gated tie-breaker/recovery mechanism.
    Only the first primitive action is executed; the agent then observes and
    replans.
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
        override_margin: float = 0.10,
        online_vfe: bool = False,
        online_vfe_beta: float = 1.0,
        online_vfe_state_positive_weight: float = 10.0,
        online_vfe_gradient_clip_norm: float = 1.0,
        online_vfe_steps: int = 1,
        online_vfe_update_interval: int = 1,
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
        if override_margin < 0.0:
            raise ValueError("override_margin must be non-negative")
        if online_vfe_beta < 0.0 or online_vfe_state_positive_weight <= 0.0:
            raise ValueError("invalid online VFE loss weights")
        if online_vfe_gradient_clip_norm <= 0.0 or online_vfe_steps < 1:
            raise ValueError("invalid online VFE optimization settings")
        if online_vfe_update_interval < 1:
            raise ValueError("online_vfe_update_interval must be at least 1")
        if online_vfe and not callable(getattr(world_model, "online_vfe_step", None)):
            raise ValueError("online VFE requires a World Model with online_vfe_step")
        self.world_model = world_model
        self.max_candidates = max_candidates
        self.max_location_candidates = max_location_candidates
        self.progress_weight = progress_weight
        self.information_gain_weight = information_gain_weight
        self.action_cost_weight = action_cost_weight
        self.revisit_penalty = revisit_penalty
        self.horizon_discount = horizon_discount
        self.override_margin = override_margin
        self.online_vfe = online_vfe
        self.online_vfe_beta = online_vfe_beta
        self.online_vfe_state_positive_weight = online_vfe_state_positive_weight
        self.online_vfe_gradient_clip_norm = online_vfe_gradient_clip_norm
        self.online_vfe_steps = online_vfe_steps
        self.online_vfe_update_interval = online_vfe_update_interval
        self.goal: GoalSpec | None = None
        self.belief: TargetLocationBelief | None = None
        self._initial_observation: str | None = None
        self._last_observation: str | None = None
        self._pending_probe: dict[str, Any] | None = None
        self.world_model_calls = 0
        self.world_model_errors = 0
        self.planner_decisions = 0
        self.online_vfe_updates = 0
        self.online_vfe_errors = 0
        self.online_vfe_loss_sum = 0.0
        self._online_vfe_transition_count = 0

    def reset_episode(
        self,
        initial_observation: str,
        admissible_commands: list[str] | None = None,
    ) -> None:
        self._initial_observation = initial_observation
        self._last_observation = initial_observation
        if self.online_vfe:
            reset_online_vfe = getattr(self.world_model, "reset_online_vfe", None)
            if not callable(reset_online_vfe):
                raise RuntimeError("World Model does not support online VFE reset")
            reset_online_vfe()
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
        self._online_vfe_transition_count = 0
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

    @staticmethod
    def _mentions_type(action: str, entity_type_name: str | None) -> bool:
        if not entity_type_name:
            return False
        text = action.lower().replace("_", " ")
        target = entity_type_name.lower().replace("_", " ")
        return re.search(rf"\b{re.escape(target)}(?:\s+\d+)?\b", text) is not None

    @classmethod
    def _subgoal_progress(cls, action: str, goal: GoalSpec | None) -> float:
        """Reward prerequisite actions that do not yet satisfy the terminal goal.

        The learned state score is intentionally terminal-oriented (for
        example, ``on_type::object::destination``).  ALFWorld nevertheless
        requires intermediate actions such as ``take`` before ``move``.  A
        small, admissible-action shaping term keeps the selector from
        rejecting those necessary steps when the one-step World Model score is
        negative.
        """

        if goal is None:
            return 0.0
        lower = action.strip().lower()
        if lower.startswith(("take ", "pick up ")) and cls._mentions_type(
            action, goal.object_type
        ):
            return 0.45
        if goal.operation and lower.startswith(f"{goal.operation} ") and cls._mentions_type(
            action, goal.object_type
        ):
            return 1.0
        if lower.startswith(("move ", "put ")) and cls._mentions_type(
            action, goal.object_type
        ) and cls._mentions_type(action, goal.destination_type):
            return 1.0
        if goal.task_kind == "inspect" and lower.startswith(("examine ", "look at ")) and cls._mentions_type(
            action, goal.object_type
        ):
            return 0.75
        return 0.0

    @staticmethod
    def _previous_observation(
        initial_observation: str, turns: list[tuple[str, str]]
    ) -> str | None:
        """Return the observation before the current one.

        ``turns[-1][1]`` is the current observation in EvaluationRunner, not
        the previous one.  Using it for a no-progress check would therefore
        report every step as unchanged.
        """

        if not turns:
            return None
        if len(turns) == 1:
            return initial_observation
        return turns[-2][1]

    @staticmethod
    def _is_observation_action(action: str) -> bool:
        return _OBSERVATION_ACTION_RE.fullmatch(action.strip()) is not None

    def _is_stalled(
        self,
        initial_observation: str,
        current_observation: str,
        turns: list[tuple[str, str]],
    ) -> tuple[bool, str | None]:
        previous_observation = self._previous_observation(initial_observation, turns)
        observation_unchanged = (
            previous_observation is not None
            and previous_observation == current_observation
        )
        previous_action = turns[-1][0] if turns else None
        return observation_unchanged, previous_action

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
        self._last_observation = current_observation
        if self.belief is not None:
            self.belief.add_candidate_locations(admissible_commands or [])

        policy_response = super().act(
            initial_observation=initial_observation,
            current_observation=current_observation,
            turns=turns,
            admissible_commands=admissible_commands,
        )
        admissible = list(admissible_commands or [])
        observation_unchanged, previous_action = self._is_stalled(
            initial_observation, current_observation, turns
        )
        policy_is_fallback = self._is_observation_action(policy_response.action)
        policy_repeated = previous_action == policy_response.action
        stalled = observation_unchanged or policy_is_fallback or policy_repeated
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
                model_progress = next_score - current_score
                subgoal_progress = self._subgoal_progress(action, self.goal)
                progress = model_progress + subgoal_progress
                location, raw_ig, discounted_ig, cost, revisit_cost = (
                    self._action_terms(action, self.belief)
                )
                no_progress_penalty = 0.0
                if observation_unchanged and (
                    action == previous_action
                    or self._is_observation_action(action)
                ):
                    no_progress_penalty = self.revisit_penalty * 4.0
                efe = (
                    -self.progress_weight * progress
                    - self.information_gain_weight * discounted_ig
                    + self.action_cost_weight * cost
                    + revisit_cost
                    + no_progress_penalty
                )
                scored.append(
                    {
                        "action": action,
                        "efe": efe,
                        "predicted_progress": progress,
                        "model_predicted_progress": model_progress,
                        "subgoal_progress": subgoal_progress,
                        "expected_information_gain": discounted_ig,
                        "raw_expected_information_gain": raw_ig,
                        "estimated_step_cost": cost,
                        "revisit_penalty": revisit_cost,
                        "no_progress_penalty": no_progress_penalty,
                        "probe_location": location,
                        "navigation_action": bool(
                            _GO_TO_RE.fullmatch(action.strip())
                        ),
                        "destination_previously_visited": bool(
                            self.belief is not None
                            and location is not None
                            and location in self.belief.visited_locations
                        ),
                        "destination_previously_inspected": bool(
                            self.belief is not None
                            and location is not None
                            and location in self.belief.inspected_locations
                        ),
                        "policy_proposal": action == policy_response.action,
                        "candidate_order": index,
                    }
                )
            best = min(scored, key=lambda item: item["efe"])
            policy_score = next(
                (item for item in scored if item["action"] == policy_response.action),
                None,
            )
            safe_scored = [
                item
                for item in scored
                if not (
                    observation_unchanged
                    and (
                        item["action"] == previous_action
                        or self._is_observation_action(item["action"])
                    )
                )
            ]
            safe_best = min(safe_scored or scored, key=lambda item: item["efe"])

            chosen = policy_score or safe_best
            planner_status = "policy_preserved"
            override_reason = "policy_first"
            if policy_score is None:
                chosen = safe_best
                planner_status = "scored"
                override_reason = "policy_not_in_candidates"
            elif stalled:
                policy_is_noop = (
                    policy_repeated
                    or (
                        observation_unchanged
                        and self._is_observation_action(policy_response.action)
                    )
                )
                materially_better = (
                    safe_best["efe"] + self.override_margin < policy_score["efe"]
                    and safe_best["predicted_progress"] >= 0.0
                )
                if policy_is_noop:
                    # A repeated observation/action pair must be broken even
                    # when the learned model cannot produce positive progress.
                    chosen = safe_best
                    planner_status = "stalled_recovery"
                    override_reason = "avoid_no_progress_action"
                elif materially_better:
                    chosen = safe_best
                    planner_status = "scored"
                    override_reason = "positive_margin_on_stall"
                else:
                    planner_status = "stalled_policy"
                    override_reason = "policy_action_is_safe"
            self._update_pending_probe(chosen)
            diagnostics = {
                "planner": "efe_world_model",
                "planner_status": planner_status,
                "chosen_action": chosen["action"],
                "policy_proposal": policy_response.action,
                "previous_action": previous_action,
                "observation_unchanged": observation_unchanged,
                "stalled": stalled,
                "override_reason": override_reason,
                "efe_best_action": best["action"],
                "belief_entropy_bits": (
                    self.belief.entropy_bits() if self.belief is not None else 0.0
                ),
                "current_goal_score": current_score,
                "chosen_predicted_progress": chosen["predicted_progress"],
                "chosen_model_predicted_progress": chosen[
                    "model_predicted_progress"
                ],
                "chosen_subgoal_progress": chosen["subgoal_progress"],
                "expected_information_gain": chosen["expected_information_gain"],
                "raw_expected_information_gain": chosen[
                    "raw_expected_information_gain"
                ],
                "estimated_step_cost": chosen["estimated_step_cost"],
                "probe_location": chosen["probe_location"],
                "navigation_action": chosen["navigation_action"],
                "destination_previously_visited": chosen[
                    "destination_previously_visited"
                ],
                "destination_previously_inspected": chosen[
                    "destination_previously_inspected"
                ],
                "efe": chosen["efe"],
                "efe_best": best["efe"],
                "candidate_scores": scored,
            }
            return AgentResponse(
                action=chosen["action"],
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
        result: dict[str, Any] = {}
        if self.belief is not None:
            update = self.belief.record_observation(observation, action=action)
            result.update(
                {
                    "belief_entropy_before_transition": update[
                        "belief_entropy_before"
                    ],
                    "belief_entropy_after_transition": update[
                        "belief_entropy_after"
                    ],
                    "realized_information_gain": update["realized_information_gain"],
                    "inspected_location": update["inspected_location"],
                }
            )
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
                        "probe_planning_action": self._pending_probe[
                            "planning_action"
                        ],
                        "probe_expected_information_gain": self._pending_probe[
                            "expected_information_gain"
                        ],
                        "probe_realized_information_gain": probe_realized_ig,
                    }
                )
                self._pending_probe = None
            else:
                result["probe_completed"] = False

        if self.online_vfe:
            self._online_vfe_transition_count += 1
            if (
                self._online_vfe_transition_count % self.online_vfe_update_interval
                != 0
            ):
                result["online_vfe_update_skipped"] = True
                self._last_observation = observation
                return result
            if self._last_observation is None:
                self.online_vfe_errors += 1
                result["online_vfe_error_type"] = "missing_previous_observation"
            else:
                try:
                    update_metrics: dict[str, float] = {}
                    for _ in range(self.online_vfe_steps):
                        update_metrics = self.world_model.online_vfe_step(
                            self._last_observation,
                            action,
                            observation,
                            beta=self.online_vfe_beta,
                            state_positive_weight=self.online_vfe_state_positive_weight,
                            gradient_clip_norm=self.online_vfe_gradient_clip_norm,
                        )
                        self.online_vfe_updates += 1
                        self.online_vfe_loss_sum += update_metrics["loss"]
                    result.update(
                        {
                            "online_vfe_loss": update_metrics["loss"],
                            "online_vfe_reconstruction": update_metrics[
                                "reconstruction"
                            ],
                            "online_vfe_kl": update_metrics["kl"],
                            "online_vfe_state_supervision": update_metrics[
                                "state_supervision"
                            ],
                            "online_vfe_transition_supervision": update_metrics[
                                "transition_supervision"
                            ],
                            "online_vfe_gradient_norm": update_metrics[
                                "gradient_norm"
                            ],
                        }
                    )
                except Exception as exc:
                    self.online_vfe_errors += 1
                    result["online_vfe_error_type"] = type(exc).__name__
        self._last_observation = observation
        return result
