import unittest

import torch
from torch import nn

from wm.world_model.data import WorldModelVocab
from wm.world_model.model import QwenWorldModel


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.anchor.device


def _tiny_world_model():
    model = object.__new__(QwenWorldModel)
    nn.Module.__init__(model)
    model.vocab = WorldModelVocab(
        facts=["agent_at::room_1", "visible::mug_1"],
        actions=["<unk>", "look"],
        observation_words=["<unk>", "room", "mug"],
    )
    model.encoder = _FakeEncoder()
    model.action_embedding = nn.Embedding(2, 2)
    model.transition = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2))
    model.observation_decoder = nn.Linear(2, 3)
    model.eval()
    return model


class WorldModelRolloutTest(unittest.TestCase):
    def test_predicts_next_state_from_a_batch_of_imagined_distributions(self):
        model = _tiny_world_model()
        predictions = model.predict_from_state_probabilities(
            [[0.8, 0.2], [0.1, 0.9]],
            ["look", "look"],
            threshold=0.5,
            top_words=2,
        )

        self.assertEqual(len(predictions), 2)
        self.assertEqual(tuple(predictions[0].posterior_state_probabilities.shape), (2,))
        self.assertEqual(tuple(predictions[0].predicted_next_state_probabilities.shape), (2,))
        self.assertEqual(len(predictions[0].predicted_facts), 1)
        self.assertEqual(len(predictions[0].reconstructed_words[0]), 2)

    def test_rejects_misaligned_state_and_action_batches(self):
        model = _tiny_world_model()
        with self.assertRaisesRegex(ValueError, "batch sizes must match"):
            model.predict_from_state_probabilities([[0.8, 0.2]], ["look", "look"])

    def test_rejects_a_state_with_the_wrong_fact_dimension(self):
        model = _tiny_world_model()
        with self.assertRaisesRegex(ValueError, "width must match"):
            model.predict_from_state_probabilities([[0.8]], ["look"])


if __name__ == "__main__":
    unittest.main()
