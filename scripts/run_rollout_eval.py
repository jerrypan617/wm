#!/usr/bin/env python3
"""Evaluate policy-proposed actions with preference-scored WM rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.agents import PreferenceRolloutAgent  # noqa: E402
from wm.cli import _build_backend, _build_generation  # noqa: E402
from wm.environments import AlfworldTextWorldEnv  # noqa: E402
from wm.evaluation import EvaluationRunner  # noqa: E402
from wm.evaluation.metrics import summarize  # noqa: E402
from wm.evaluation.recorder import read_jsonl, write_json, write_jsonl  # noqa: E402
from wm.utils import load_experiment  # noqa: E402
from wm.world_model.data import WorldModelVocab  # noqa: E402
from wm.world_model.model import QwenWorldModel  # noqa: E402


def _load_world_model(checkpoint_path: str, cache_dir: str) -> QwenWorldModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab = WorldModelVocab.from_dict(checkpoint["vocab"])
    architecture = checkpoint["architecture"]
    model = QwenWorldModel(
        architecture["model_id"],
        vocab,
        cache_dir=cache_dir,
        local_files_only=True,
        device_map="auto",
        max_length=int(architecture["max_length"]),
        state_hidden_size=int(architecture["state_hidden_size"]),
        action_embedding_size=int(architecture["action_embedding_size"]),
        freeze_backbone=True,
        qlora=bool(architecture.get("qlora", False)),
        lora_r=int(architecture.get("lora_r", 16)),
        lora_alpha=int(architecture.get("lora_alpha", 32)),
        lora_dropout=float(architecture.get("lora_dropout", 0.05)),
        lora_target_modules=architecture.get("lora_target_modules"),
    )
    model.load_head_state_dict(checkpoint["heads"])
    model.eval()
    return model


def build_agent_settings(
    experiment: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    config = dict(experiment.get("preference_rollout", {}))
    return {
        "horizon": int(args.horizon if args.horizon is not None else config.get("horizon", 3)),
        "max_root_candidates": int(
            args.max_root_candidates
            if args.max_root_candidates is not None
            else config.get("max_root_candidates", 5)
        ),
        "max_imagined_actions": int(
            args.max_imagined_actions
            if args.max_imagined_actions is not None
            else config.get("max_imagined_actions", 12)
        ),
        "state_threshold": float(
            args.state_threshold
            if args.state_threshold is not None
            else config.get("state_threshold", 0.5)
        ),
        "state_entropy_penalty_weight": float(
            args.state_entropy_penalty_weight
            if args.state_entropy_penalty_weight is not None
            else config.get("state_entropy_penalty_weight", 0.0)
        ),
        "judge_max_new_tokens": int(
            args.judge_max_new_tokens
            if args.judge_max_new_tokens is not None
            else config.get("judge_max_new_tokens", 768)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-model-checkpoint", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--action-space", choices=["free", "admissible"], default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-root-candidates", type=int, default=None)
    parser.add_argument("--max-imagined-actions", type=int, default=None)
    parser.add_argument("--state-threshold", type=float, default=None)
    parser.add_argument("--state-entropy-penalty-weight", type=float, default=None)
    parser.add_argument("--judge-max-new-tokens", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    model_config = experiment["model"]
    environment_config = experiment["environment"]
    settings = build_agent_settings(experiment, args)
    split = args.split or experiment.get("split", "valid_seen")
    requested_episodes = (
        args.num_episodes
        if args.num_episodes is not None
        else int(experiment.get("num_episodes", -1))
    )
    action_space = args.action_space or experiment.get("action_space", "admissible")
    output_dir = Path(
        args.output_dir
        or experiment.get("output_dir", "results/preference_rollout")
    )
    run_name = f"{model_config.get('name', 'policy')}_{split}_preference_rollout"
    checkpoint_path = output_dir / f"{run_name}.checkpoint.jsonl"
    stats_path = output_dir / f"{run_name}.planner_stats.json"

    env_config = dict(environment_config)
    env_config.pop("_config_path", None)
    env_config.pop("_config_dir", None)
    environment = AlfworldTextWorldEnv(
        env_config,
        split=split,
        num_episodes=requested_episodes,
    )
    agent: PreferenceRolloutAgent | None = None
    stats = {
        "world_model_calls": 0,
        "world_model_errors": 0,
        "rollout_policy_calls": 0,
        "rollout_policy_errors": 0,
        "preference_judge_calls": 0,
        "preference_judge_errors": 0,
    }
    try:
        total_episodes = environment.num_episodes
        completed_by_id = {}
        if args.resume and checkpoint_path.is_file():
            for result in read_jsonl(checkpoint_path):
                if result.split != split:
                    raise ValueError(
                        f"Checkpoint split={result.split!r} does not match {split!r}"
                    )
                if result.episode_id in completed_by_id:
                    raise ValueError(
                        f"Duplicate episode_id in checkpoint: {result.episode_id}"
                    )
                completed_by_id[result.episode_id] = result
            if stats_path.is_file():
                stats.update(json.loads(stats_path.read_text(encoding="utf-8")))
        invalid_ids = [
            episode_id
            for episode_id in completed_by_id
            if episode_id < 0 or episode_id >= total_episodes
        ]
        if invalid_ids:
            raise ValueError("Checkpoint contains episode IDs outside this split")
        remaining_ids = [
            episode_id
            for episode_id in range(total_episodes)
            if episode_id not in completed_by_id
        ]
        if remaining_ids:
            if len(remaining_ids) != total_episodes:
                environment.close()
                environment = AlfworldTextWorldEnv(
                    env_config,
                    split=split,
                    num_episodes=len(remaining_ids),
                    episode_indices=remaining_ids,
                )
            agent = PreferenceRolloutAgent(
                _build_backend(model_config),
                _build_generation(model_config),
                _load_world_model(args.world_model_checkpoint, args.cache_dir),
                action_space=action_space,
                **settings,
            )

            def save_checkpoint(result: Any) -> None:
                completed_by_id[result.episode_id] = result
                write_jsonl(
                    checkpoint_path,
                    [completed_by_id[index] for index in sorted(completed_by_id)],
                )
                stats.update(
                    {
                        "world_model_calls": agent.world_model_calls,
                        "world_model_errors": agent.world_model_errors,
                        "rollout_policy_calls": agent.rollout_policy_calls,
                        "rollout_policy_errors": agent.rollout_policy_errors,
                        "preference_judge_calls": agent.preference_judge_calls,
                        "preference_judge_errors": agent.preference_judge_errors,
                    }
                )
                write_json(stats_path, stats)

            EvaluationRunner(
                environment=environment,
                agent=agent,
                max_steps=int(
                    args.max_steps
                    if args.max_steps is not None
                    else environment_config.get("max_steps", 50)
                ),
                on_episode_complete=save_checkpoint,
                episode_ids=remaining_ids,
                total_episodes=total_episodes,
            ).run()
            stats.update(
                {
                    "world_model_calls": agent.world_model_calls,
                    "world_model_errors": agent.world_model_errors,
                    "rollout_policy_calls": agent.rollout_policy_calls,
                    "rollout_policy_errors": agent.rollout_policy_errors,
                    "preference_judge_calls": agent.preference_judge_calls,
                    "preference_judge_errors": agent.preference_judge_errors,
                }
            )
        results = [completed_by_id[index] for index in sorted(completed_by_id)]
        if len(results) != total_episodes:
            raise RuntimeError(
                f"Evaluation ended with {len(results)}/{total_episodes} episodes"
            )
        summary = summarize(results)
    finally:
        environment.close()

    write_jsonl(output_dir / f"{run_name}.jsonl", results)
    write_json(
        output_dir / f"{run_name}.summary.json",
        {
            "model": model_config.get("name", model_config["model_id"]),
            "policy_variant": "policy_plus_preference_scored_world_model_rollout",
            "split": split,
            "action_space": action_space,
            "world_model_checkpoint": args.world_model_checkpoint,
            "preference_rollout": settings,
            "information_gain_used": False,
            "policy_action_entropy_available": False,
            **stats,
            **summary,
        },
    )
    print(json.dumps({**stats, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
