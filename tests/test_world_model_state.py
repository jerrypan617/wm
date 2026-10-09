import unittest

from wm.world_model.state import StructuredState, normalize_entity, parse_observation


class WorldModelStateTest(unittest.TestCase):
    def test_normalize_entity_preserves_instance_id(self):
        self.assertEqual(normalize_entity("Shelf 1"), "shelf_1")

    def test_parser_extracts_goal_location_and_visible_object(self):
        state = parse_observation(
            "You arrive at shelf 1. On the shelf 1, you see a watch 1. "
            "Your task is to: put a vase in coffeetable."
        )
        self.assertEqual(state.agent_location, "shelf_1")
        self.assertIn("visible::watch_1", state.facts)
        self.assertIn("on::watch_1::shelf_1", state.facts)
        self.assertIn("goal::put a vase in coffeetable", state.facts)

    def test_state_is_canonical_and_serializable(self):
        state = StructuredState(facts=("closed::cabinet_1", "closed::cabinet_1"))
        restored = StructuredState.from_dict(state.to_dict())
        self.assertEqual(state, restored)
        self.assertEqual(state.facts, ("closed::cabinet_1",))

    def test_parser_keeps_open_container_contents(self):
        state = parse_observation(
            "You open the drawer 2. The drawer 2 is open. "
            "In it, you see a pencil 1."
        )
        self.assertIn("inside::pencil_1::drawer_2", state.facts)


if __name__ == "__main__":
    unittest.main()
