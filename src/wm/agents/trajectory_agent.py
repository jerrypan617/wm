from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from typing import Any, Sequence

import torch

from wm.models.base import GenerationConfig, ModelBackend

from .action_parser import canonicalize_against_admissible, normalize_action
from .alfworld_planner import GoalSpec, parse_goal
from .llm_agent import AgentResponse, LLMTextAgent
from .prompt_builder import DEFAULT_SYSTEM_PROMPT


_ACTION_ENTITY_RE = re.compile(r"\b([a-z][a-z0-9_]*\s+\d+)\b", re.IGNORECASE)
_FACT_ENTITY_RE = re.compile(r"\b([a-z][a-z0-9_]*_\d+)\b", re.IGNORECASE)


def _json_object(response: str) -> dict[str, Any] | None:
    text = (response or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def _as_float_list(values: Any) -> list[float]:
    if torch.is_tensor(values):
        values = values.detach().cpu().flatten().tolist()
    return [float(value) for value in values]


class PreferenceRolloutAgent(LLMTextAgent):
    """Use multi-step imagined trajectories and an LLM preference judge.

    The policy proposes legal root actions. The learned World Model predicts
    the next state distribution; the policy backend then proposes imagined
    continuations grounded in that prediction and in the World Model's action
    vocabulary. A separate evaluation prompt scores the completed paths. Only
    the winning root action is returned to the environment. Imagined samples
    are never used as labels or as World Model training data.
    """

    def __init__(
        self,
        backend: ModelBackend,
        generation: GenerationConfig,
        world_model: Any,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        action_space: str = "admissible",
        horizon: int = 3,
        max_root_candidates: int = 5,
        max_imagined_actions: int = 12,
        state_threshold: float = 0.5,
        state_entropy_penalty_weight: float = 0.0,
        judge_max_new_tokens: int = 768,
    ) -> None:
        super().__init__(
            backend,
            generation,
            system_prompt=system_prompt,
            action_space=action_space,
        )
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if max_root_candidates < 1 or max_imagined_actions < 1:
            raise ValueError("candidate limits must be at least 1")
        if not 0.0 <= state_threshold <= 1.0:
            raise ValueError("state_threshold must be in [0, 1]")
        if state_entropy_penalty_weight < 0.0:
            raise ValueError("state_entropy_penalty_weight must be non-negative")
        if judge_max_new_tokens < 1:
            raise ValueError("judge_max_new_tokens must be at least 1")
        self.world_model = world_model
        self.horizon = horizon
        self.max_root_candidates = max_root_candidates
        self.max_imagined_actions = max_imagined_actions
        self.state_threshold = state_threshold
        self.state_entropy_penalty_weight = state_entropy_penalty_weight
        self.rollout_generation = replace(
            generation,
            max_new_tokens=min(generation.max_new_tokens, 512),
            temperature=0.0,
            do_sample=False,
        )
        self.judge_generation = replace(
            generation,
            max_new_tokens=min(generation.max_new_tokens, judge_max_new_tokens),
            temperature=0.0,
            do_sample=False,
        )
        self.world_model_calls = 0
        self.world_model_errors = 0
        self.rollout_policy_calls = 0
        self.rollout_policy_errors = 0
        self.preference_judge_calls = 0
        self.preference_judge_errors = 0

    @staticmethod
    def _mentions_type(action: str, entity_type: str | None) -> bool:
        if not entity_type:
            return False
        normalized = entity_type.lower().replace("_", " ")
        action_text = action.lower()
        return re.search(rf"\b{re.escape(normalized)}(?:\s+\d+)?\b", action_text) is not None

    @staticmethod
    def _action_priority(action: str, goal: GoalSpec | None) -> tuple[int, int, str]:
        lower = action.lower()
        task_relevance = 0
        if goal is not None:
            task_relevance += int(PreferenceRolloutAgent._mentions_type(action, goal.object_type))
            task_relevance += int(
                PreferenceRolloutAgent._mentions_type(action, goal.destination_type)
            )
        useful_verb = int(
            lower.startswith(
                ("take ", "pick up ", "open ", "put ", "move ", "clean ", "heat ", "cool ", "slice ", "toggle ", "go to ")
            )
        )
        return (-task_relevance, -useful_verb, lower)

    def _root_candidates(
        self,
        policy_action: str,
        admissible_commands: Sequence[str] | None,
        goal: GoalSpec | None,
    ) -> tuple[str, list[str]]:
        admissible = list(dict.fromkeys(admissible_commands or []))
        safe_policy_action = canonicalize_against_admissible(
            policy_action, admissible
        )
        if admissible and safe_policy_action not in admissible:
            ordered = sorted(admissible, key=lambda action: self._action_priority(action, goal))
            safe_policy_action = ordered[0]

        candidates: list[str] = []

        def add(action: str) -> None:
            if action and action not in candidates:
                candidates.append(action)

        add(safe_policy_action)
        ordered = sorted(admissible, key=lambda action: self._action_priority(action, goal))
        for action in ordered:
            add(action)
            if len(candidates) >= self.max_root_candidates:
                break
        return safe_policy_action, candidates[: self.max_root_candidates]

    @staticmethod
    def _state_entities(facts: Sequence[str]) -> set[str]:
        entities: set[str] = set()
        for fact in facts:
            for match in _FACT_ENTITY_RE.finditer(fact):
                entities.add(match.group(1).lower().replace("_", " "))
        return entities

    def _imagined_action_candidates(
        self,
        probabilities: Any,
        previous_action: str,
        goal: GoalSpec | None,
    ) -> list[str]:
        facts = list(getattr(self.world_model.vocab, "facts", []))
        actions = list(getattr(self.world_model.vocab, "actions", []))
        values = _as_float_list(probabilities)
        active_facts = [
            fact
            for fact, probability in zip(facts, values)
            if probability >= self.state_threshold
        ]
        entities = self._state_entities(active_facts)
        options: list[str] = []
        previous = normalize_action(previous_action)
        for action in actions:
            normalized = normalize_action(action)
            if not normalized or normalized == "<unk>" or normalized == previous:
                continue
            references = {
                match.group(1).lower() for match in _ACTION_ENTITY_RE.finditer(action)
            }
            normalized_references = {reference for reference in references}
            if normalized_references and not all(
                reference in entities for reference in normalized_references
            ):
                continue
            if not normalized_references and normalized in {"look", "inventory"}:
                continue
            if normalized not in options:
                options.append(normalized)
        options.sort(key=lambda action: self._action_priority(action, goal))
        return options[: self.max_imagined_actions]

    @staticmethod
    def _prediction_probabilities(prediction: Any) -> list[float]:
        return _as_float_list(prediction.predicted_next_state_probabilities)

    @staticmethod
    def _prediction_details(prediction: Any) -> tuple[list[str], list[str]]:
        raw_facts = getattr(prediction, "predicted_facts", [])
        facts = raw_facts[0] if raw_facts and isinstance(raw_facts[0], list) else raw_facts
        raw_words = getattr(prediction, "reconstructed_words", [])
        words = raw_words[0] if raw_words and isinstance(raw_words[0], list) else raw_words
        return list(facts or []), list(words or [])

    def _advance_rollouts(
        self,
        rollouts: list[dict[str, Any]],
        goal: GoalSpec | None,
        task_context: str,
    ) -> None:
        for depth in range(1, self.horizon):
            eligible: list[dict[str, Any]] = []
            for rollout in rollouts:
                if rollout.get("stopped"):
                    continue
                options = self._imagined_action_candidates(
                    rollout["state_probabilities"],
                    rollout["actions"][-1],
                    goal,
                )
                if options:
                    rollout["action_options"] = options
                    eligible.append(rollout)
                else:
                    rollout["stopped"] = True
                    rollout["stop_reason"] = "no_grounded_action_candidates"
            if not eligible:
                break

            prompt = self._continuation_prompt(eligible, task_context)
            try:
                self.rollout_policy_calls += 1
                raw = self.backend.generate(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You are an imagined rollout policy. Predicted states are hypotheses, "
                                "not observed facts. Choose only from the supplied action options. "
                                "Return valid JSON and no extra text."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    self.rollout_generation,
                )
                choices = self._parse_continuations(raw, eligible)
            except Exception as exc:
                self.rollout_policy_errors += 1
                for rollout in eligible:
                    rollout["stopped"] = True
                    rollout["stop_reason"] = f"continuation_error:{type(exc).__name__}"
                continue

            chosen: list[tuple[dict[str, Any], str]] = []
            for rollout in eligible:
                action = choices.get(rollout["trajectory_id"])
                if action is None:
                    rollout["stopped"] = True
                    rollout["stop_reason"] = "continuation_missing_or_invalid"
                    continue
                rollout["actions"].append(action)
                chosen.append((rollout, action))
            if not chosen:
                continue

            try:
                next_predictions = self.world_model.predict_from_state_probabilities(
                    [rollout["state_probabilities"] for rollout, _ in chosen],
                    [action for _, action in chosen],
                    threshold=self.state_threshold,
                )
                self.world_model_calls += 1
                if len(next_predictions) != len(chosen):
                    raise RuntimeError("world model returned a mismatched imagined batch")
            except Exception as exc:
                self.world_model_errors += 1
                for rollout, _ in chosen:
                    rollout["actions"].pop()
                    rollout["stopped"] = True
                    rollout["stop_reason"] = f"transition_error:{type(exc).__name__}"
                continue

            for (rollout, _), prediction in zip(chosen, next_predictions):
                rollout["predictions"].append(prediction)
                rollout["state_probabilities"] = self._prediction_probabilities(prediction)
                facts, words = self._prediction_details(prediction)
                rollout["states"].append({"facts": facts, "reconstructed_words": words})

    def _continuation_prompt(
        self, rollouts: Sequence[dict[str, Any]], task_context: str
    ) -> str:
        payload = []
        for rollout in rollouts:
            facts, words = rollout["states"][-1]["facts"], rollout["states"][-1]["reconstructed_words"]
            probabilities = self._prediction_probabilities(rollout["predictions"][-1])
            top_marginals = sorted(
                zip(self.world_model.vocab.facts, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
            payload.append(
                {
                    "trajectory_id": rollout["trajectory_id"],
                    "imagined_actions_so_far": rollout["actions"],
                    "predicted_state_facts": facts[:16],
                    "predicted_fact_marginals": [
                        {"fact": fact, "probability": round(float(probability), 3)}
                        for fact, probability in top_marginals
                    ],
                    "decoder_keywords": words[:12],
                    "action_options": rollout["action_options"],
                }
            )
        return (
            "Task and real context before imagined branching:\n"
            + task_context
            + "\n\nChoose imagined continuation actions for the active trajectories. "
            "Use task-relevant, state-supported actions; do not repeat the previous action. "
            "Choose exactly one action from each trajectory's action_options. "
            "Return {\"choices\":[{\"trajectory_id\":string,\"action\":string}]} .\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

    @staticmethod
    def _parse_continuations(
        response: str,
        rollouts: Sequence[dict[str, Any]],
    ) -> dict[str, str]:
        payload = _json_object(response)
        if payload is None or not isinstance(payload.get("choices"), list):
            return {}
        by_id = {rollout["trajectory_id"]: rollout for rollout in rollouts}
        parsed: dict[str, str] = {}
        for item in payload["choices"]:
            if not isinstance(item, dict):
                continue
            trajectory_id = str(item.get("trajectory_id", ""))
            rollout = by_id.get(trajectory_id)
            if rollout is None:
                continue
            proposed = normalize_action(str(item.get("action", "")))
            options = {
                normalize_action(option): option
                for option in rollout["action_options"]
            }
            if proposed in options:
                parsed[trajectory_id] = options[proposed]
        return parsed

    def _score_prompt(
        self,
        task: str,
        current_observation: str,
        rollouts: Sequence[dict[str, Any]],
    ) -> str:
        trajectories = []
        for rollout in rollouts:
            steps = []
            for index, action in enumerate(rollout["actions"]):
                state = rollout["states"][index]
                probabilities = self._prediction_probabilities(rollout["predictions"][index])
                fact_probs = list(zip(self.world_model.vocab.facts, probabilities))
                top_facts = sorted(fact_probs, key=lambda pair: pair[1], reverse=True)[:10]
                steps.append(
                    {
                        "action": action,
                        "predicted_facts": [
                            {"fact": fact, "probability": round(float(probability), 3)}
                            for fact, probability in top_facts
                        ],
                        "decoder_keywords": state["reconstructed_words"][:10],
                    }
                )
            trajectories.append(
                {
                    "trajectory_id": rollout["trajectory_id"],
                    "steps": steps,
                    "rollout_stopped_early": len(steps) < self.horizon,
                    "stop_reason": rollout.get("stop_reason"),
                }
            )
        return (
            "Score the imagined trajectories for the task. Evaluate likely task completion and "
            "meaningful progress from the predicted states; prefer coherent, efficient paths and "
            "penalize loops or actions unsupported by the predicted state. Do not treat an early-stopped "
            "rollout as efficient completion; inability to continue is evidence of incompleteness. Predictions are uncertain, "
            "so do not treat a single thresholded fact as guaranteed. Give each trajectory an integer "
            "score from 0 (very poor) to 100 (excellent), on a comparable scale. Return only JSON in "
            "the form {\"scores\":[{\"trajectory_id\":string,\"score\":number,\"reason\":string}]}.\n\n"
            + json.dumps(
                {
                    "task": task,
                    "current_observation": current_observation,
                    "trajectories": trajectories,
                },
                ensure_ascii=False,
            )
        )

    def _preference_scores(
        self,
        task: str,
        current_observation: str,
        rollouts: Sequence[dict[str, Any]],
    ) -> tuple[dict[str, float], dict[str, str], str | None]:
        prompt = self._score_prompt(task, current_observation, rollouts)
        try:
            self.preference_judge_calls += 1
            raw = self.backend.generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful trajectory preference evaluator for ALFWorld. "
                            "Compare candidate paths using only the task and supplied imagined evidence. "
                            "Return valid JSON and no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                self.judge_generation,
            )
        except Exception as exc:
            self.preference_judge_errors += 1
            return {}, {}, f"{type(exc).__name__}: {exc}"
        payload = _json_object(raw)
        if payload is None or not isinstance(payload.get("scores"), list):
            self.preference_judge_errors += 1
            return {}, {}, "judge returned invalid JSON or missing scores"
        valid_ids = {rollout["trajectory_id"] for rollout in rollouts}
        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for item in payload["scores"]:
            if not isinstance(item, dict):
                continue
            trajectory_id = str(item.get("trajectory_id", ""))
            try:
                score = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
            if trajectory_id not in valid_ids or not math.isfinite(score):
                continue
            scores[trajectory_id] = min(100.0, max(0.0, score))
            reason = item.get("reason", "")
            if isinstance(reason, str):
                reasons[trajectory_id] = reason
        if not scores:
            self.preference_judge_errors += 1
            return {}, {}, "judge did not score any known trajectory"
        return scores, reasons, None

    @staticmethod
    def _bernoulli_entropy(probabilities: Any) -> float:
        values = _as_float_list(probabilities)
        if not values:
            return 0.0
        total = 0.0
        for value in values:
            p = min(1.0 - 1e-7, max(1e-7, value))
            total -= p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)
        return total / len(values)

    def act(
        self,
        *,
        initial_observation: str,
        current_observation: str,
        turns: list[tuple[str, str]],
        admissible_commands: list[str] | None = None,
    ) -> AgentResponse:
        policy_response = super().act(
            initial_observation=initial_observation,
            current_observation=current_observation,
            turns=turns,
            admissible_commands=admissible_commands,
        )
        goal = parse_goal(initial_observation)
        safe_policy_action, candidates = self._root_candidates(
            policy_response.action,
            admissible_commands,
            goal,
        )
        if not candidates:
            return AgentResponse(
                safe_policy_action,
                policy_response.raw_response,
                policy_response.user_prompt,
                {
                    "planner": "preference_world_model_rollout",
                    "planner_status": "no_root_candidates",
                    "chosen_action": safe_policy_action,
                },
            )

        try:
            predictions = self.world_model.predict_next(
                current_observation,
                candidates,
                threshold=self.state_threshold,
            )
            self.world_model_calls += 1
            if len(predictions) != len(candidates):
                raise RuntimeError("world model returned a mismatched root batch")
        except Exception as exc:
            self.world_model_errors += 1
            return AgentResponse(
                safe_policy_action,
                policy_response.raw_response,
                policy_response.user_prompt,
                {
                    "planner": "preference_world_model_rollout",
                    "planner_status": "world_model_fallback",
                    "world_model_error": f"{type(exc).__name__}: {exc}",
                    "chosen_action": safe_policy_action,
                    "information_gain_used": False,
                },
            )

        rollouts: list[dict[str, Any]] = []
        for index, (action, prediction) in enumerate(zip(candidates, predictions)):
            probabilities = self._prediction_probabilities(prediction)
            facts, words = self._prediction_details(prediction)
            rollouts.append(
                {
                    "trajectory_id": f"traj_{index:02d}",
                    "actions": [action],
                    "predictions": [prediction],
                    "state_probabilities": probabilities,
                    "states": [{"facts": facts, "reconstructed_words": words}],
                    "root_action": action,
                    "stopped": False,
                }
            )
        self._advance_rollouts(rollouts, goal, initial_observation)
        scores, reasons, judge_error = self._preference_scores(
            initial_observation,
            current_observation,
            rollouts,
        )
        if not scores:
            return AgentResponse(
                safe_policy_action,
                policy_response.raw_response,
                policy_response.user_prompt,
                {
                    "planner": "preference_world_model_rollout",
                    "planner_status": "judge_fallback",
                    "judge_error": judge_error,
                    "chosen_action": safe_policy_action,
                    "information_gain_used": False,
                    "rollout_horizon": self.horizon,
                    "trajectories": self._diagnostic_trajectories(rollouts, {}, {}, safe_policy_action),
                },
            )

        scored: list[dict[str, Any]] = []
        for rollout in rollouts:
            trajectory_id = rollout["trajectory_id"]
            if trajectory_id not in scores:
                continue
            entropy = sum(
                self._bernoulli_entropy(
                    self._prediction_probabilities(prediction)
                )
                for prediction in rollout["predictions"]
            ) / max(1, len(rollout["predictions"]))
            adjusted = scores[trajectory_id] - (
                self.state_entropy_penalty_weight * entropy
            )
            scored.append(
                {
                    "trajectory": rollout,
                    "score": scores[trajectory_id],
                    "adjusted_score": adjusted,
                    "state_prediction_entropy": entropy,
                    "reason": reasons.get(trajectory_id, ""),
                }
            )
        if not scored:
            return AgentResponse(
                safe_policy_action,
                policy_response.raw_response,
                policy_response.user_prompt,
                {
                    "planner": "preference_world_model_rollout",
                    "planner_status": "judge_fallback",
                    "judge_error": "no scored trajectories remained",
                    "chosen_action": safe_policy_action,
                    "information_gain_used": False,
                },
            )

        # On exact ties, preserve the initial policy's admissible proposal.
        policy_scored = next(
            (item for item in scored if item["trajectory"]["root_action"] == safe_policy_action),
            None,
        )
        best = max(scored, key=lambda item: item["adjusted_score"])
        if policy_scored is not None and math.isclose(
            policy_scored["adjusted_score"], best["adjusted_score"], abs_tol=1e-9
        ):
            best = policy_scored
        chosen_action = best["trajectory"]["root_action"]
        diagnostics = {
            "planner": "preference_world_model_rollout",
            "planner_status": "preference_scored",
            "chosen_action": chosen_action,
            "policy_proposal": policy_response.action,
            "policy_proposal_was_admissible": safe_policy_action == policy_response.action,
            "override_reason": (
                "trajectory_preference" if chosen_action != safe_policy_action else "policy_or_tie"
            ),
            "rollout_horizon": self.horizon,
            "information_gain_used": False,
            "policy_action_entropy_available": False,
            "state_entropy_penalty_weight": self.state_entropy_penalty_weight,
            "trajectories": self._diagnostic_trajectories(
                rollouts,
                scores,
                reasons,
                chosen_action,
                scored_by_id={item["trajectory"]["trajectory_id"]: item for item in scored},
            ),
        }
        return AgentResponse(
            chosen_action,
            policy_response.raw_response,
            policy_response.user_prompt,
            diagnostics,
        )

    def _diagnostic_trajectories(
        self,
        rollouts: Sequence[dict[str, Any]],
        scores: dict[str, float],
        reasons: dict[str, str],
        chosen_action: str,
        *,
        scored_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        for rollout in rollouts:
            trajectory_id = rollout["trajectory_id"]
            score = scores.get(trajectory_id)
            scored = (scored_by_id or {}).get(trajectory_id, {})
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "actions": list(rollout["actions"]),
                    "root_action": rollout["root_action"],
                    "preference_score": score,
                    "adjusted_score": scored.get("adjusted_score", score),
                    "state_prediction_entropy": scored.get("state_prediction_entropy"),
                    "reason": reasons.get(trajectory_id, ""),
                    "chosen": rollout["root_action"] == chosen_action,
                    "stopped_early": len(rollout["actions"]) < self.horizon,
                    "stop_reason": rollout.get("stop_reason"),
                    "predicted_states": [
                        {
                            "facts": state["facts"][:16],
                            "decoder_keywords": state["reconstructed_words"][:12],
                        }
                        for state in rollout["states"]
                    ],
                }
            )
        return rows
