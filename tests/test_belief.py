import math
import unittest

from wm.agents.belief import TargetLocationBelief, location_from_action


class TargetLocationBeliefTest(unittest.TestCase):
    def setUp(self) -> None:
        self.belief = TargetLocationBelief.from_actions(
            "mug",
            1,
            [f"go to cabinet {index}" for index in range(1, 5)],
        )

    def test_inspection_eig_is_expected_entropy_reduction(self) -> None:
        self.assertAlmostEqual(self.belief.entropy_bits(), 2.0)
        self.assertAlmostEqual(
            self.belief.expected_information_gain("cabinet_1"),
            -0.25 * math.log2(0.25) - 0.75 * math.log2(0.75),
        )

    def test_empty_container_updates_belief_and_realized_ig(self) -> None:
        result = self.belief.record_observation(
            "You open the cabinet 1. The cabinet 1 is open. In it, you see nothing.",
            action="open cabinet 1",
        )
        self.assertIn("cabinet_1", self.belief.inspected_locations)
        self.assertAlmostEqual(self.belief.entropy_bits(), math.log2(3))
        self.assertAlmostEqual(
            float(result["realized_information_gain"]), 2.0 - math.log2(3)
        )

    def test_finding_target_collapses_single_target_belief(self) -> None:
        result = self.belief.record_observation(
            "You open the cabinet 1. The cabinet 1 is open. In it, you see a mug 1."
        )
        self.assertEqual(self.belief.found_entities, {"mug_1"})
        self.assertEqual(self.belief.found_locations["mug_1"], "cabinet_1")
        self.assertIn("cabinet_1", self.belief.opened_locations)
        self.assertEqual(self.belief.entropy_bits(), 0.0)
        self.assertAlmostEqual(float(result["realized_information_gain"]), 2.0)

    def test_multi_item_uniform_subset_prior(self) -> None:
        belief = TargetLocationBelief.from_actions(
            "knife",
            2,
            [f"go to drawer {index}" for index in range(1, 5)],
        )
        self.assertAlmostEqual(belief.entropy_bits(), math.log2(6))
        self.assertAlmostEqual(belief.expected_information_gain("drawer_1"), 1.0)

    def test_action_horizon_reflects_delayed_container_observation(self) -> None:
        self.assertEqual(location_from_action("go to cabinet 2"), ("cabinet_2", 2))
        self.assertEqual(location_from_action("go to shelf 3"), ("shelf_3", 1))
        self.assertEqual(location_from_action("open cabinet 2"), ("cabinet_2", 1))
        self.assertEqual(location_from_action("take mug 1 from cabinet 2"), (None, 0))


if __name__ == "__main__":
    unittest.main()
