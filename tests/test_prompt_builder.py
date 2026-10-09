import unittest

from wm.agents.prompt_builder import build_user_prompt


class PromptBuilderTest(unittest.TestCase):
    def test_prompt_contains_current_observation(self) -> None:
        prompt = build_user_prompt(
            "Initial goal",
            [("look", "There is a table.")],
            "The table is open.",
        )
        self.assertIn("Initial goal", prompt)
        self.assertIn("The table is open.", prompt)

    def test_prompt_enforces_one_action(self) -> None:
        prompt = build_user_prompt(
            "The task is: put the book on the desk.",
            [],
            "You are in the room.",
            ["go to desk 1", "look"],
        )
        self.assertIn("choose exactly one and copy it verbatim", prompt)
        self.assertIn("- go to desk 1", prompt)
        self.assertIn("one line only", prompt)


if __name__ == "__main__":
    unittest.main()
