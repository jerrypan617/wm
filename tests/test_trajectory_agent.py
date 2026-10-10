import json
import unittest
from types import SimpleNamespace

from wm.agents.trajectory_agent import PreferenceRolloutAgent
from wm.models.base import GenerationConfig


class _Backend:
    def __init__(self, judge_response=None):
        self.judge_response = judge_response or json.dumps(
            {
                "scores": [
                    {"trajectory_id": "traj_00", "score": 30},
                    {"trajectory_id": "traj_01", "score": 90},
                ]
            }
        )
        self.continuation_calls = 0
        self.prompts = []

    def generate(self, messages, generation):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        if "Choose imagined continuation actions" in prompt:
            self.continuation_calls += 1
            return json.dumps(
                {
                    "choices": [
                        {"trajectory_id": "traj_00", "action": "take mug 1 from shelf 1"},
                        {"trajectory_id": "traj_01", "action": "open cabinet 1"},
                    ]
                }
            )
        if "Score the imagined trajectories" in prompt:
            return self.judge_response
        return "go to shelf 1"


class _WorldModel:
    def __init__(self):
        self.vocab = SimpleNamespace(
            facts=[
                "agent_at::shelf_1",
                "visible::mug_1",
                "agent_at::cabinet_1",
                "visible::cabinet_1",
                "closed::cabinet_1",
                "take::mug_1",
                "inside::mug_1::cabinet_1",
            ],
            actions=[
                "go to shelf 1",
                "go to cabinet 1",
                "take mug 1 from shelf 1",
                "open cabinet 1",
            ],
        )
        self.first_step_calls = []
        self.recursive_calls = []

    def _prediction(self, probabilities, posterior=None):
        probabilities = list(probabilities)
        facts = [
            fact
            for fact, probability in zip(self.vocab.facts, probabilities)
            if probability >= 0.5
        ]
        return SimpleNamespace(
            posterior_state_probabilities=posterior or [0.0] * len(probabilities),
            predicted_next_state_probabilities=probabilities,
            predicted_facts=[facts],
            reconstructed_words=[["mug", "shelf"]],
        )

    def predict_next(self, observation, actions, threshold=0.5):
        self.first_step_calls.append((observation, list(actions)))
        result = []
        for action in actions:
            probs = [0.0] * len(self.vocab.facts)
            if action == "go to shelf 1":
                probs[0] = 0.95
                probs[1] = 0.90
            else:
                probs[2] = 0.95
                probs[3] = 0.90
                probs[4] = 0.90
            result.append(self._prediction(probs))
        return result

    def predict_from_state_probabilities(self, state_probabilities, actions, threshold=0.5):
        self.recursive_calls.append((list(actions), [list(row) for row in state_probabilities]))
        result = []
        for state, action in zip(state_probabilities, actions):
            probs = [0.0] * len(self.vocab.facts)
            if action == "take mug 1 from shelf 1":
                probs[5] = 0.95
            elif action == "open cabinet 1":
                probs[4] = 0.05
                probs[6] = 0.80
            result.append(self._prediction(probs, posterior=state))
        return result


class PreferenceRolloutAgentTest(unittest.TestCase):
    def make_agent(self, *, judge_response=None, **kwargs):
        backend = _Backend(judge_response)
        world_model = _WorldModel()
        agent = PreferenceRolloutAgent(
            backend,
            GenerationConfig(max_new_tokens=128),
            world_model,
            horizon=2,
            max_root_candidates=2,
            **kwargs,
        )
        return agent, backend, world_model

    def test_rolls_out_multiple_steps_and_executes_only_selected_root_action(self):
        agent, backend, world_model = self.make_agent()
        initial = "Your task is to: put the mug in the cabinet."
        response = agent.act(
            initial_observation=initial,
            current_observation="You arrive at hallway 1. You see a shelf 1 and cabinet 1.",
            turns=[],
            admissible_commands=["go to shelf 1", "go to cabinet 1"],
        )

        self.assertEqual(response.action, "go to cabinet 1")
        self.assertEqual(world_model.first_step_calls[0][1], ["go to shelf 1", "go to cabinet 1"])
        self.assertEqual(len(world_model.recursive_calls), 1)
        self.assertEqual(world_model.recursive_calls[0][0], [
            "take mug 1 from shelf 1",
            "open cabinet 1",
        ])
        self.assertEqual(response.diagnostics["rollout_horizon"], 2)
        self.assertEqual(response.diagnostics["planner_status"], "preference_scored")
        self.assertEqual(response.diagnostics["chosen_action"], "go to cabinet 1")
        self.assertEqual(len(response.diagnostics["trajectories"]), 2)
        self.assertTrue(all("information_gain" not in item for item in response.diagnostics["trajectories"]))
        self.assertEqual(backend.continuation_calls, 1)

    def test_invalid_policy_root_is_replaced_with_an_admissible_candidate(self):
        agent, _, _ = self.make_agent()
        agent.backend.generate = lambda messages, generation: (
            json.dumps({"scores": [{"trajectory_id": "traj_00", "score": 50}]})
            if "Score the imagined trajectories" in messages[-1]["content"]
            else (
                json.dumps({"choices": []})
                if "Choose imagined continuation actions" in messages[-1]["content"]
                else "open nonexistent 99"
            )
        )
        initial = "Your task is to: put the mug in the cabinet."
        response = agent.act(
            initial_observation=initial,
            current_observation="You are in hallway 1.",
            turns=[],
            admissible_commands=["go to shelf 1", "go to cabinet 1"],
        )
        self.assertIn(response.action, ["go to shelf 1", "go to cabinet 1"])

    def test_malformed_judge_preserves_policy_action(self):
        agent, backend, _ = self.make_agent(judge_response="not json")
        response = agent.act(
            initial_observation="Your task is to: put the mug in the cabinet.",
            current_observation="You are in hallway 1.",
            turns=[],
            admissible_commands=["go to shelf 1", "go to cabinet 1"],
        )
        self.assertEqual(response.action, "go to shelf 1")
        self.assertEqual(response.diagnostics["planner_status"], "judge_fallback")
        self.assertTrue(response.diagnostics["judge_error"])

    def test_entropy_penalty_is_optional_and_zero_by_default(self):
        agent, _, _ = self.make_agent()
        self.assertEqual(agent.state_entropy_penalty_weight, 0.0)
        self.assertAlmostEqual(agent._bernoulli_entropy([0.5]), 1.0)
        self.assertLess(agent._bernoulli_entropy([0.0, 1.0]), 1e-5)


if __name__ == "__main__":
    unittest.main()
