import unittest
from types import SimpleNamespace

from wm.agents.efe_agent import EFEWorldModelAgent
from wm.models.base import GenerationConfig


class _Backend:
    def generate(self, messages, generation):
        return "look"


class _ActionBackend:
    def __init__(self, action):
        self.action = action

    def generate(self, messages, generation):
        return self.action


class _WorldModel:
    def __init__(self):
        self.vocab = SimpleNamespace(
            facts=[
                "visible_type::mug",
                "on_type::mug::shelf",
                "inside_type::mug::cabinet",
                "feedback::nothing_happens",
            ]
        )

    def predict_next(self, observation, actions, threshold=0.5):
        predictions = []
        for action in actions:
            next_probabilities = [0.0, 0.0, 0.0, 0.0]
            if action == "take mug 1 from shelf 1":
                next_probabilities[0] = 0.8
                next_probabilities[1] = 0.8
            predictions.append(
                SimpleNamespace(
                    posterior_state_probabilities=[0.0, 0.0, 0.0, 0.0],
                    predicted_next_state_probabilities=next_probabilities,
                )
            )
        return predictions


class _OnlineWorldModel(_WorldModel):
    def __init__(self):
        super().__init__()
        self.reset_count = 0
        self.update_count = 0

    def reset_online_vfe(self):
        self.reset_count += 1

    def online_vfe_step(self, *args, **kwargs):
        self.update_count += 1
        return {
            "loss": 0.1,
            "reconstruction": 0.01,
            "kl": 0.02,
            "state_supervision": 0.03,
            "transition_supervision": 0.04,
            "gradient_norm": 0.05,
        }


class EFEWorldModelAgentTest(unittest.TestCase):
    def make_agent(self):
        return EFEWorldModelAgent(
            _Backend(),
            GenerationConfig(),
            _WorldModel(),
            information_gain_weight=0.5,
            action_cost_weight=0.05,
            horizon_discount=0.8,
        )

    def test_selects_information_seeking_admissible_action_and_logs_scores(self):
        agent = self.make_agent()
        initial = "Your task is to: put the mug in the cabinet."
        commands = ["look", "go to shelf 1", "go to cabinet 1"]
        agent.reset_episode(initial, commands)

        response = agent.act(
            initial_observation=initial,
            current_observation=initial,
            turns=[],
            admissible_commands=commands,
        )

        self.assertIn(response.action, commands)
        self.assertEqual(response.diagnostics["planner_status"], "scored")
        self.assertGreater(
            response.diagnostics["expected_information_gain"], 0.0
        )
        self.assertEqual(len(response.diagnostics["candidate_scores"]), 3)

    def test_task_progress_can_outweigh_information_gain(self):
        agent = self.make_agent()
        agent.information_gain_weight = 0.0
        initial = "Your task is to: put the mug in the cabinet."
        commands = ["look", "take mug 1 from shelf 1"]
        agent.reset_episode(initial, commands)

        response = agent.act(
            initial_observation=initial,
            current_observation=initial,
            turns=[],
            admissible_commands=commands,
        )

        self.assertEqual(response.action, "take mug 1 from shelf 1")
        self.assertGreater(response.diagnostics["chosen_predicted_progress"], 0.0)

    def test_preserves_nonstalled_policy_action(self):
        agent = EFEWorldModelAgent(
            _ActionBackend("take mug 1 from shelf 1"),
            GenerationConfig(),
            _WorldModel(),
        )
        initial = "Your task is to: put the mug in the cabinet."
        commands = ["look", "take mug 1 from shelf 1"]
        agent.reset_episode(initial, commands)

        response = agent.act(
            initial_observation=initial,
            current_observation=initial,
            turns=[],
            admissible_commands=commands,
        )

        self.assertEqual(response.action, "take mug 1 from shelf 1")
        self.assertEqual(response.diagnostics["planner_status"], "policy_preserved")
        self.assertEqual(response.diagnostics["override_reason"], "policy_first")

    def test_stalled_observation_does_not_repeat_look(self):
        agent = self.make_agent()
        initial = "Your task is to: put the mug in the cabinet."
        commands = ["look", "take mug 1 from shelf 1"]
        agent.reset_episode(initial, commands)

        response = agent.act(
            initial_observation=initial,
            current_observation=initial,
            turns=[("look", initial)],
            admissible_commands=commands,
        )

        self.assertEqual(response.action, "take mug 1 from shelf 1")
        self.assertTrue(response.diagnostics["observation_unchanged"])

    def test_delayed_probe_is_completed_when_container_contents_are_observed(self):
        agent = self.make_agent()
        initial = "Your task is to: put the mug in the cabinet."
        commands = ["look", "go to cabinet 1", "go to drawer 1"]
        agent.reset_episode(initial, commands)
        response = agent.act(
            initial_observation=initial,
            current_observation=initial,
            turns=[],
            admissible_commands=commands,
        )
        self.assertEqual(response.action, "go to cabinet 1")

        arrived = agent.observe_transition(
            response.action,
            "You arrive at cabinet 1. The cabinet 1 is closed.",
        )
        self.assertFalse(arrived["probe_completed"])
        opened = agent.observe_transition(
            "open cabinet 1",
            "You open the cabinet 1. The cabinet 1 is open. In it, you see nothing.",
        )
        self.assertTrue(opened["probe_completed"])
        self.assertEqual(opened["probe_location"], "cabinet_1")
        self.assertGreater(opened["probe_realized_information_gain"], 0.0)

    def test_online_vfe_update_interval_skips_intermediate_transitions(self):
        world_model = _OnlineWorldModel()
        agent = EFEWorldModelAgent(
            _Backend(),
            GenerationConfig(),
            world_model,
            online_vfe=True,
            online_vfe_update_interval=2,
        )
        initial = "Your task is to: put the mug in the cabinet."
        agent.reset_episode(initial, ["look"])

        first = agent.observe_transition("look", "first observation")
        second = agent.observe_transition("look", "second observation")

        self.assertTrue(first["online_vfe_update_skipped"])
        self.assertNotIn("online_vfe_loss", first)
        self.assertEqual(second["online_vfe_loss"], 0.1)
        self.assertEqual(world_model.update_count, 1)


if __name__ == "__main__":
    unittest.main()
