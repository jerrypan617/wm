# Preference-Guided World-Model Rollouts Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a separately configurable ALFWorld agent that recursively imagines short action trajectories with the trained World Model and uses a policy-model evaluation prompt to choose the first real action, without information gain.

**Architecture:** Keep the existing one-step EFE/IG agent intact as a prior baseline. Add a batched transition API that advances a predicted factorized state distribution, then add a trajectory agent that proposes admissible root actions, alternates predicted states with policy-generated imagined continuations, and asks the same backend to score the resulting paths. Only the selected root action reaches the environment; imagined trajectories do not train the World Model. Optional normalized state-prediction entropy is a zero-default conservative penalty/diagnostic; policy action entropy is omitted because the current backend exposes no action probabilities.

**Tech Stack:** Python 3.10+, PyTorch, existing `ModelBackend` text-generation interface, ALFWorld evaluation runner, unittest/pytest.

---

### Task 1: Add recursive World Model prediction from a state distribution

**Files:**
- Modify: `src/wm/world_model/model.py`
- Test: `tests/test_world_model.py`

**Steps:**
1. Add a failing test for batched state-distribution transitions, including output shape, facts, decoded words, and unknown actions.
2. Run the focused test and confirm the method is absent.
3. Implement a no-grad API that accepts one predicted Bernoulli state vector per action and returns the existing `WorldModelPrediction` structure.
4. Run focused model tests.

### Task 2: Implement preference-scored multi-step imagined trajectories

**Files:**
- Create: `src/wm/agents/trajectory_agent.py`
- Modify: `src/wm/agents/__init__.py`
- Test: `tests/test_trajectory_agent.py`

**Steps:**
1. Add tests for root-action admissibility, recursive rollout calls, evaluator ranking, first-action-only execution, malformed evaluator fallback, and no IG term.
2. Run focused tests and confirm failures.
3. Implement policy continuation prompts grounded in predicted facts and actions from the World Model vocabulary; parse/validate generated choices.
4. Implement a separate trajectory preference-evaluation prompt returning a bounded score per trajectory; preserve the original policy action if scoring fails.
5. Add normalized Bernoulli state entropy diagnostics and a zero-default penalty option. Do not fabricate policy action entropy.
6. Run the agent tests and existing EFE tests.

### Task 3: Add an evaluation entry point and experiment config

**Files:**
- Create: `scripts/run_rollout_eval.py`
- Create: `configs/experiments/valid_seen_deepseek_rollout.yaml`
- Test: `tests/test_run_rollout_eval.py` (or a focused config/agent-construction test if CLI coverage is impractical)

**Steps:**
1. Add a test for loading the new experiment settings and constructing the trajectory agent with the configured horizon and weights.
2. Implement a runner that reuses existing model/backend/environment factories, writes episode JSONL and summary JSON, and records rollout/judge diagnostics.
3. Run CLI `--help`, config loading, and the focused test.

### Task 4: Verify compatibility and summarize usage

**Files:**
- Modify: `README.md` only if needed for a concise run command.

**Steps:**
1. Run the full test suite.
2. Run static checks/compile checks for changed modules.
3. Confirm existing EFE code and experiment configs remain unchanged.
4. Report the new workflow, limitations, and exact experiment command; do not commit or push unless requested.
