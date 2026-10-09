# ALFWorld TextWorld benchmark

This benchmark evaluates a language model through ALFWorld's `AlfredTWEnv`.
The model receives text observations and returns one TextWorld action per turn.

The primary protocol is `free` action generation: the model does not receive
the environment's admissible-action list. The `admissible` protocol is a
diagnostic mode that separates planning errors from action-format errors.

Use `valid_seen` and `valid_unseen` as separate evaluation splits.
