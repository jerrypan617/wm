# wm

`wm` is a lightweight local framework for evaluating language models in
interactive text environments. The first supported benchmark is ALFWorld's
TextWorld environment (`AlfredTWEnv`) with Qwen3.5-4B.

## Layout

- `src/wm/models`: local model backends.
- `src/wm/environments`: environment adapters.
- `src/wm/agents`: prompts, action parsing, and agent policy.
- `src/wm/evaluation`: episode runner, metrics, and result recording.
- `configs`: model, environment, and experiment configuration.
- `benchmarks`: benchmark protocols and notes.
- `scripts`: download, smoke-test, and evaluation entry points.
- `results` and `logs`: local outputs; ignored by Git.

## Setup

```bash
uv sync
export ALFWORLD_DATA=/absolute/path/to/alfworld_data
```

If the model is not already cached:

```bash
uv run python scripts/download_model.py \
  --model-id Qwen/Qwen3.5-4B \
  --cache-dir ./cache
```

Install/download ALFWorld data with the official ALFWorld commands if needed:

```bash
uv run alfworld-download --data-dir "$ALFWORLD_DATA"
```

## Smoke test

```bash
uv run python scripts/smoke_test.py \
  --model-id Qwen/Qwen3.5-4B \
  --cache-dir ./cache
```

## Evaluation

Run a small test first:

```bash
uv run python scripts/run_eval.py eval \
  --config configs/experiments/valid_seen.yaml \
  --num-episodes 5
```

Run the two official validation splits separately:

```bash
uv run python scripts/run_eval.py eval \
  --config configs/experiments/valid_seen.yaml

uv run python scripts/run_eval.py eval \
  --config configs/experiments/valid_unseen.yaml
```

The primary protocol is free action generation. To run the diagnostic
admissible-action protocol:

```bash
uv run python scripts/run_eval.py eval \
  --config configs/experiments/valid_seen.yaml \
  --action-space admissible \
  --num-episodes 5
```

Each run writes per-episode JSONL and a summary JSON under `results/`.

For a Qwen policy that receives the parser's canonical structured-state
summary, add `--policy-variant structured` to the evaluation command. This
keeps the policy model and action protocol unchanged while exposing facts
such as the current location, visible entities, and containment relations.

The QLoRA world-model policy experiment is run with:

```bash
uv run python scripts/run_qwen_world_model_eval.py \
  --config configs/experiments/valid_seen.yaml \
  --world-model-checkpoint checkpoints/qwen35_world_model_qlora_train200_v2_pos10.pt \
  --cache-dir cache \
  --action-space admissible \
  --output-dir results/qwen35_4b/valid_seen_world_model
```

This wrapper calls the world model for a shortlist of admissible actions at
each decision and records `world_model_calls`, `world_model_overrides`, and
`world_model_errors` in the summary and checkpoint sidecar. A completed
evaluation therefore distinguishes an actually used world model from a
checkpoint that was merely loaded.

## Belief-aware EFE experiment

The research prototype keeps the DeepSeek policy and Qwen World Model
separate. At each decision it asks the policy for a proposal, adds a bounded
set of admissible goal-relevant and inspection actions, predicts their next
fact marginals with the World Model, and minimizes:

```text
G(a) = - task_progress(a)
       - beta * discounted_expected_information_gain(a)
       + action_cost(a)
       + revisit_penalty(a)
```

It executes only the selected primitive action and replans from the next real
observation. A surface inspection is modeled as one step; inspecting a closed
container is estimated as a two-step `go to` → `open` probe (one step if the
agent already knows that container is open), so its information value is
discounted and carried until the contents are observed.

Important scope note: the current World Model predicts factorized structured
fact probabilities; those marginals are not a calibrated joint belief over
hidden environments. Therefore the first EIG term is an explicit,
task-relevant target-location belief: a uniform approximation over which
uninspected admissible locations contain the required object. Empty inventories
remove locations; observing a target updates the remaining count. This is a
useful engineering proxy, not a claim that the World Model has calibrated
parameter uncertainty. The design and assumptions are documented in
[`docs/architecture/adr-001-belief-aware-efe-planner.md`](docs/architecture/adr-001-belief-aware-efe-planner.md).

Train the World Model on the ALFWorld `train` split and keep validation splits
held out. The trainer now assigns complete episodes to either train or
validation, avoiding leakage between adjacent transitions. It also builds the
vocabulary from training episodes if `--vocab` is omitted:

```bash
uv run python scripts/train_world_model.py \
  --dataset data/world_model_train.jsonl \
  --cache-dir cache \
  --output checkpoints/qwen35_world_model.pt \
  --vocab-output checkpoints/qwen35_world_model_vocab.json
```

QLoRA is enabled by default and needs CUDA. For a full-precision frozen-backbone
run on a machine without CUDA, add `--no-qlora` (subject to available memory).

Start with a small matched comparison. The policy-only baseline and EFE system
must both use `admissible` actions so action validity is not credited to the
World Model:

```bash
# Baseline: DeepSeek policy with the same legal-action constraint
uv run wm-eval eval \
  --config configs/experiments/valid_seen_deepseek_admissible.yaml \
  --action-space admissible --num-episodes 5 \
  --output-dir results/ablation/seen_policy_only

# Full system: task progress + expected information gain + costs
uv run python scripts/run_efe_eval.py \
  --config configs/experiments/valid_seen_deepseek_efe.yaml \
  --world-model-checkpoint checkpoints/qwen35_world_model.pt \
  --cache-dir cache --num-episodes 5

# Ablation: learned task progress, but no information-seeking bonus
uv run python scripts/run_efe_eval.py \
  --config configs/experiments/valid_seen_deepseek_efe.yaml \
  --world-model-checkpoint checkpoints/qwen35_world_model.pt \
  --cache-dir cache --num-episodes 5 \
  --information-gain-weight 0
```

Matched policy-only configs are `valid_seen_deepseek_admissible.yaml` and
`valid_unseen_deepseek_admissible.yaml`. For the full validation splits, use
`valid_seen_deepseek_efe.yaml` and `valid_unseen_deepseek_efe.yaml`; add
`--resume` to continue an interrupted run. The script also accepts
`--episode-start/--episode-end` for deterministic
shards and overrides for all EFE weights. Episode JSONL records per-candidate
scores, predicted progress, expected and realized information gain; summary
JSON reports success/steps, invalid actions, consecutive action repeats, and
navigation revisits, and probe IG prediction error. Use `--progress-weight 0`
to isolate the location information term, and compare all variants on
identical episode IDs. For independent episode-range shards, give each run a
separate output directory and merge the resulting JSONL files with
`scripts/merge_eval_results.py`.

## Qwen world model

The project includes a structured, action-conditioned world-model prototype.
It parses ALFWorld observations into canonical facts, uses the local
Qwen3.5-4B backbone as a frozen observation encoder, and trains factorized
posterior, transition, and observation-reconstruction heads. The KL term is
computed over Bernoulli fact variables, so it is tractable and corresponds to
the structured state distribution rather than two JSON strings.

The initial state targets are conservative facts parsed from observations.
They do not infer hidden container contents. A later experiment can replace
these targets with TextWorld symbolic facts, but should keep those facts out
of the inference input if a non-oracle evaluation is required.

Collect exploratory transitions from the ALFWorld training split. The
collector samples from the environment's admissible action list:

```bash
uv run python scripts/collect_world_model_data.py \
  --environment-config configs/environments/alfworld_textworld.yaml \
  --split train \
  --num-episodes 1000 \
  --max-steps 50 \
  --output data/world_model_train_episodes.jsonl
```

Convert episode trajectories into transition records and build the state,
action, and observation vocabularies:

```bash
uv run python scripts/build_world_model_dataset.py \
  --input data/world_model_train_episodes.jsonl \
  --output data/world_model_train.jsonl \
  --vocab-output data/world_model_vocab.json
```

Train the Qwen-backed world model with QLoRA. This loads Qwen in 4-bit NF4
precision and trains only LoRA adapters plus the structured world-model
heads. QLoRA requires a CUDA GPU, `bitsandbytes`, and `peft`; they are project
dependencies and will be installed by `uv sync`:

```bash
uv run python scripts/train_world_model.py \
  --dataset data/world_model_train.jsonl \
  --vocab data/world_model_vocab.json \
  --cache-dir cache \
  --output checkpoints/qwen35_world_model.pt
```

Use `--no-qlora` only for a full-precision frozen-backbone smoke test. The
resulting checkpoint is not a QLoRA checkpoint.

Predict the next structured state for one or more candidate actions:

```bash
uv run python scripts/predict_world_state.py \
  --checkpoint checkpoints/qwen35_world_model.pt \
  --observation "You arrive at shelf 1. On the shelf 1, you see a watch 1." \
  --action "open shelf 1" "go to desk 1"
```

The training collector uses the `train` split. Keep `valid_seen` and
`valid_unseen` held out for world-model and downstream planning evaluation.

## DeepSeek API

Put your key in the project-root `.env` file:

```bash
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
ALFWORLD_DATA=/absolute/path/to/alfworld_data
# Optional fallback; the experiment YAML selects the model explicitly.
DEEPSEEK_MODEL=deepseek-chat
```

Run a small DeepSeek test:

```bash
uv run wm-eval eval \
  --config configs/experiments/valid_seen_deepseek.yaml \
  --num-episodes 1 \
  --max-steps 20 \
  --output-dir results/deepseek_reasoner/smoke
```

For DeepSeek API evaluation, episodes can be processed concurrently. Each
worker owns an isolated TextWorld environment:

```bash
uv run wm-eval eval \
  --config configs/experiments/valid_seen_deepseek.yaml \
  --num-episodes -1 \
  --max-steps 50 \
  --workers 4 \
  --output-dir results/deepseek_reasoner/valid_seen_w4
```

Start with 2–4 workers and increase only if the API rate limit allows it.

To resume an interrupted run, keep the same output directory and add
`--resume`. Completed episode ids are loaded from the checkpoint and skipped:

```bash
uv run wm-eval eval \
  --config configs/experiments/valid_seen_deepseek.yaml \
  --num-episodes -1 \
  --max-steps 50 \
  --workers 2 \
  --resume \
  --output-dir results/deepseek_reasoner/valid_seen_w2
```

The DeepSeek experiment configs use `deepseek-reasoner`. Its reasoning
content is not sent to TextWorld; only the final `message.content` command is
parsed and executed. The API's reasoning tokens count toward `max_tokens`, so
the reasoner config uses a larger output budget.
