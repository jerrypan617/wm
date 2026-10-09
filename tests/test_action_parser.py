import unittest

from wm.agents.action_parser import (
    canonicalize_against_admissible,
    extract_action,
)


class ActionParserTest(unittest.TestCase):
    def test_extracts_last_action_after_thinking(self) -> None:
        response = "<think>Need to inspect the room.</think>\nlook."
        self.assertEqual(extract_action(response), "look")

    def test_removes_action_prefix(self) -> None:
        self.assertEqual(
            extract_action("Action: go to cabinet 1"),
            "go to cabinet 1",
        )

    def test_canonicalizes_admissible_command(self) -> None:
        self.assertEqual(
            canonicalize_against_admissible(
                "GO   TO cabinet 1",
                ["go to cabinet 1"],
            ),
            "go to cabinet 1",
        )


if __name__ == "__main__":
    unittest.main()
