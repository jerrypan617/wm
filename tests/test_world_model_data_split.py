import unittest

from wm.world_model.data import TransitionRecord, split_indices_by_episode
from wm.world_model.state import StructuredState


def _record(episode_id: int, step: int) -> TransitionRecord:
    state = StructuredState(agent_location=f"room_{episode_id}")
    return TransitionRecord(
        episode_id=episode_id,
        step=step,
        observation=f"observation {episode_id}:{step}",
        action="look",
        next_observation=f"observation {episode_id}:{step + 1}",
        state=state,
        next_state=state,
    )


class EpisodeSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            _record(episode_id, step)
            for episode_id in range(6)
            for step in range(3)
        ]

    def test_keeps_each_episode_in_exactly_one_partition(self) -> None:
        train, validation = split_indices_by_episode(
            self.records, validation_ratio=0.33, seed=7
        )
        train_episodes = {self.records[index].episode_id for index in train}
        validation_episodes = {
            self.records[index].episode_id for index in validation
        }
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertFalse(train_episodes & validation_episodes)
        self.assertEqual(train_episodes | validation_episodes, set(range(6)))

    def test_split_is_deterministic_for_a_seed(self) -> None:
        first = split_indices_by_episode(
            self.records, validation_ratio=0.33, seed=11
        )
        second = split_indices_by_episode(
            self.records, validation_ratio=0.33, seed=11
        )
        self.assertEqual(first, second)

    def test_requires_valid_episode_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative episode_id"):
            split_indices_by_episode(
                [_record(-1, 0), _record(1, 0)],
                validation_ratio=0.5,
            )

    def test_requires_two_episodes_and_valid_ratio(self) -> None:
        with self.assertRaisesRegex(ValueError, "two distinct episodes"):
            split_indices_by_episode(
                [_record(0, 0), _record(0, 1)],
                validation_ratio=0.5,
            )
        with self.assertRaisesRegex(ValueError, "strictly between"):
            split_indices_by_episode(self.records, validation_ratio=1.0)


if __name__ == "__main__":
    unittest.main()
