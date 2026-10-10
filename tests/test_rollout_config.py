import unittest

from wm.utils import load_experiment


class PreferenceRolloutConfigTest(unittest.TestCase):
    def test_seen_experiment_exposes_preference_rollout_settings(self):
        experiment = load_experiment(
            "configs/experiments/valid_seen_deepseek_rollout.yaml"
        )
        settings = experiment["preference_rollout"]
        self.assertEqual(experiment["split"], "valid_seen")
        self.assertEqual(experiment["action_space"], "admissible")
        self.assertEqual(settings["horizon"], 3)
        self.assertEqual(settings["state_entropy_penalty_weight"], 0.0)
        self.assertNotIn("information_gain_weight", settings)

    def test_seen_and_unseen_configs_share_the_same_planner_protocol(self):
        seen = load_experiment(
            "configs/experiments/valid_seen_deepseek_rollout.yaml"
        )
        unseen = load_experiment(
            "configs/experiments/valid_unseen_deepseek_rollout.yaml"
        )
        self.assertEqual(seen["preference_rollout"], unseen["preference_rollout"])
        self.assertEqual(seen["model"]["model_id"], unseen["model"]["model_id"])


if __name__ == "__main__":
    unittest.main()
