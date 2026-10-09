import unittest

from wm.agents.llm_agent import AgentResponse
from wm.environments.alfworld_textworld import EnvironmentStep
from wm.evaluation.runner import EvaluationRunner


class _FakeEnvironment:
    def __init__(self, episode_ids):
        self.episode_ids = list(episode_ids)
        self.num_episodes = len(self.episode_ids)
        self.split = "valid_seen"
        self._cursor = 0
        self._current = None

    def reset(self):
        self._current = self.episode_ids[self._cursor]
        self._cursor += 1
        return f"observation-{self._current}", {}

    def step(self, action):
        return EnvironmentStep(
            observation="done",
            reward=1.0,
            done=True,
            won=True,
            info={"extra.gamefile": f"game-{self._current}"},
        )

    @staticmethod
    def admissible_commands(info):
        return []

    def close(self):
        pass


class _FakeAgent:
    def act(self, **kwargs):
        return AgentResponse("look", "look", "")


class EvaluationRunnerTest(unittest.TestCase):
    def test_explicit_episode_ids_are_resumed_subset(self):
        base = _FakeEnvironment([1, 3])
        runner = EvaluationRunner(
            environment=base,
            agent=_FakeAgent(),
            max_steps=1,
            episode_ids=[1, 3],
            total_episodes=4,
        )

        results, summary = runner.run()

        self.assertEqual([result.episode_id for result in results], [1, 3])
        self.assertEqual(summary["episodes"], 2)

    def test_parallel_workers_partition_episodes(self):
        base = _FakeEnvironment(range(4))
        created_partitions = []
        completed = []

        def make_environment(indices):
            created_partitions.append(tuple(indices))
            return _FakeEnvironment(indices)

        runner = EvaluationRunner(
            environment=base,
            agent=_FakeAgent(),
            max_steps=1,
            workers=2,
            environment_factory=make_environment,
            agent_factory=_FakeAgent,
            on_episode_complete=lambda result: completed.append(result.episode_id),
        )

        results, summary = runner.run()

        self.assertEqual([result.episode_id for result in results], [0, 1, 2, 3])
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(sorted(completed), [0, 1, 2, 3])
        self.assertEqual(sorted(created_partitions), [(0, 1), (2, 3)])


if __name__ == "__main__":
    unittest.main()
