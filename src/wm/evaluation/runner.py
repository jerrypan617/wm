from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from wm.agents import LLMTextAgent
from wm.environments import AlfworldTextWorldEnv
from wm.world_model.state import parse_observation

from .metrics import summarize
from .schemas import EpisodeResult

EnvironmentFactory = Callable[[list[int]], AlfworldTextWorldEnv]
AgentFactory = Callable[[], LLMTextAgent]


@dataclass
class EvaluationRunner:
    environment: AlfworldTextWorldEnv
    agent: LLMTextAgent
    max_steps: int
    save_trajectories: bool = True
    on_episode_complete: Callable[[EpisodeResult], None] | None = None
    workers: int = 1
    environment_factory: EnvironmentFactory | None = None
    agent_factory: AgentFactory | None = None
    episode_ids: list[int] | None = None
    total_episodes: int | None = None

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

    def run(self) -> tuple[list[EpisodeResult], dict[str, Any]]:
        total_episodes = self.total_episodes or self.environment.num_episodes
        episode_ids = (
            list(self.episode_ids)
            if self.episode_ids is not None
            else list(range(self.environment.num_episodes))
        )
        if any(episode_id < 0 or episode_id >= total_episodes for episode_id in episode_ids):
            raise ValueError("episode_ids must be within [0, total_episodes)")
        if not episode_ids:
            return [], summarize([])

        if self.workers == 1:
            results = self._run_indices(
                self.environment,
                self.agent,
                episode_ids,
                total_episodes,
            )
            return results, summarize(results)

        if self.environment_factory is None or self.agent_factory is None:
            raise ValueError(
                "environment_factory and agent_factory are required when workers > 1"
            )

        return self._run_parallel(episode_ids, total_episodes)

    def _run_parallel(
        self,
        episode_ids: list[int],
        total_episodes: int,
    ) -> tuple[list[EpisodeResult], dict[str, Any]]:
        partitions = self._partition(episode_ids, min(self.workers, total_episodes))
        worker_environments: list[AlfworldTextWorldEnv] = []
        results: list[EpisodeResult] = []
        callback_lock = Lock()
        environment_lock = Lock()

        try:
            # Construct TextWorld environments before starting threads. This
            # avoids concurrent global Gym registration while still allowing
            # model requests and episode stepping to overlap.
            for partition in partitions:
                worker_environments.append(self.environment_factory(partition))
            worker_agents = [self.agent_factory() for _ in partitions]

            with ThreadPoolExecutor(
                max_workers=len(partitions),
                thread_name_prefix="alfworld-worker",
            ) as executor:
                futures = [
                    executor.submit(
                        self._run_indices,
                        environment,
                        agent,
                        partition,
                        total_episodes,
                        callback_lock,
                        environment_lock,
                    )
                    for environment, agent, partition in zip(
                        worker_environments, worker_agents, partitions
                    )
                ]
                for future in as_completed(futures):
                    results.extend(future.result())
        finally:
            for environment in worker_environments:
                environment.close()

        results.sort(key=lambda result: result.episode_id)
        return results, summarize(results)

    def _run_indices(
        self,
        environment: AlfworldTextWorldEnv,
        agent: LLMTextAgent,
        episode_ids: list[int],
        total_episodes: int,
        callback_lock: Lock | None = None,
        environment_lock: Lock | None = None,
    ) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []

        for episode_id in episode_ids:
            if environment_lock is None:
                observation, info = environment.reset()
            else:
                with environment_lock:
                    observation, info = environment.reset()
            initial_observation = observation
            reset_episode = getattr(agent, "reset_episode", None)
            if callable(reset_episode):
                reset_episode(
                    initial_observation,
                    environment.admissible_commands(info),
                )
            turns: list[tuple[str, str]] = []
            actions: list[str] = []
            raw_responses: list[str] = []
            observations: list[str] = [observation]
            decision_diagnostics: list[dict[str, Any]] = []
            structured_states: list[dict[str, Any]] = [
                parse_observation(observation).to_dict()
            ]
            invalid_actions = 0
            success = False

            for _ in range(self.max_steps):
                admissible = environment.admissible_commands(info)
                response = agent.act(
                    initial_observation=initial_observation,
                    current_observation=observation,
                    turns=turns,
                    admissible_commands=admissible,
                )

                if admissible and response.action not in admissible:
                    invalid_actions += 1

                if environment_lock is None:
                    step = environment.step(response.action)
                else:
                    with environment_lock:
                        step = environment.step(response.action)
                transition_diagnostics: dict[str, Any] = {}
                observe_transition = getattr(agent, "observe_transition", None)
                if callable(observe_transition):
                    try:
                        transition_diagnostics = observe_transition(
                            response.action, step.observation
                        )
                    except Exception as exc:
                        transition_diagnostics = {
                            "diagnostic_update_error_type": type(exc).__name__
                        }
                decision_diagnostics.append(
                    {**response.diagnostics, **transition_diagnostics}
                )
                actions.append(response.action)
                raw_responses.append(response.raw_response)
                observations.append(step.observation)
                structured_states.append(parse_observation(step.observation).to_dict())
                turns.append((response.action, step.observation))
                observation = step.observation
                info = step.info

                if step.done:
                    success = step.won
                    break

            result = EpisodeResult(
                episode_id=episode_id,
                split=environment.split,
                gamefile=(
                    str(info.get("extra.gamefile"))
                    if info.get("extra.gamefile") is not None
                    else None
                ),
                success=success,
                steps=len(actions),
                invalid_actions=invalid_actions,
                actions=actions,
                raw_responses=raw_responses,
                observations=observations,
                structured_states=structured_states,
                decision_diagnostics=decision_diagnostics,
            )
            results.append(result)
            if self.on_episode_complete is not None:
                if callback_lock is None:
                    self.on_episode_complete(result)
                else:
                    with callback_lock:
                        self.on_episode_complete(result)

            print(
                f"[{episode_id + 1}/{total_episodes}] "
                f"success={result.success} steps={result.steps} "
                f"invalid={result.invalid_actions}"
            )

        return results

    @staticmethod
    def _partition(episode_ids: list[int], workers: int) -> list[list[int]]:
        partitions: list[list[int]] = []
        base, remainder = divmod(len(episode_ids), workers)
        start = 0
        for worker_id in range(workers):
            size = base + (1 if worker_id < remainder else 0)
            if size:
                partitions.append(episode_ids[start : start + size])
            start += size
        return partitions
