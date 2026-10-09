#!/usr/bin/env python3
"""Evaluate the Qwen policy with an action-conditioned World Model critic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.agents import WorldModelGuidedLLMTextAgent  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-model-checkpoint", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--action-space", choices=["free", "admissible"], default="admissible")
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--override-margin", type=float, default=0.15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    model_config = experiment["model"]
    environment_config = experiment["environment"]
    split = args.split or experiment.get("split", "valid_seen")
    requested_episodes = (
        args.num_episodes
        if args.num_episodes is not None
        else int(experiment.get("num_episodes", -1))
    )
    output_dir = Path(args.output_dir)
    run_name = f"{model_config.get('name', 'model')}_{split}"
    checkpoint_path = output_dir / f"{run_name}.checkpoint.jsonl"
    stats_path = output_dir / f"{run_name}.world_model_stats.json"
    checkpoint_results = read_jsonl(checkpoint_path) if args.resume else []
    completed_by_id = {result.episode_id: result for result in checkpoint_results}

    env_config = dict(environment_config)
    env_config.pop("_config_path", None)
    env_config.pop("_config_dir", None)
    environment = AlfworldTextWorldEnv(
        env_config, split=split, num_episodes=requested_episodes
    )
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else environment_config.get("max_steps", 50)
    )
    stats = {
        "world_model_calls": 0,
        "world_model_overrides": 0,
        "world_model_errors": 0,
    }
    if args.resume and stats_path.is_file():
        stats.update(json.loads(stats_path.read_text(encoding="utf-8")))

    try:
        total_episodes = environment.num_episodes
        episode_start = args.episode_start if args.episode_start is not None else 0
        episode_end = args.episode_end if args.episode_end is not None else total_episodes
        if episode_start < 0 or episode_end > total_episodes or episode_start >= episode_end:
            raise ValueError(
                f"Invalid episode range [{episode_start}, {episode_end}) for "
                f"total_episodes={total_episodes}"
            )
        episode_scope = list(range(episode_start, episode_end))
        remaining_ids = [
            episode_id for episode_id in episode_scope if episode_id not in completed_by_id
        ]
        print(
            f"World Model evaluation: {len(completed_by_id)} completed, "
            f"{len(remaining_ids)} remaining"
        )
        if not remaining_ids:
            all_results = sorted(completed_by_id.values(), key=lambda item: item.episode_id)
        else:
            environment.close()
            environment = AlfworldTextWorldEnv(
                env_config,
                split=split,
                num_episodes=len(remaining_ids),
                episode_indices=remaining_ids,
            )
            policy_backend = _build_backend(model_config)
            policy = WorldModelGuidedLLMTextAgent(
                policy_backend,
                _build_generation(model_config),
                _load_world_model(args.world_model_checkpoint, args.cache_dir),
                action_space=args.action_space,
                max_candidates=args.max_candidates,
                override_margin=args.override_margin,
            )
            # Preserve cumulative diagnostics when a partially completed
            # shard is resumed.
            policy.world_model_calls = int(stats["world_model_calls"])
            policy.world_model_overrides = int(stats["world_model_overrides"])
            policy.world_model_errors = int(stats["world_model_errors"])

            def save_checkpoint(result: Any) -> None:
                completed_by_id[result.episode_id] = result
                write_jsonl(
                    checkpoint_path,
                    [completed_by_id[index] for index in sorted(completed_by_id)],
                )
                stats.update(
                    {
                        "world_model_calls": policy.world_model_calls,
                        "world_model_overrides": policy.world_model_overrides,
                        "world_model_errors": policy.world_model_errors,
                    }
                )
                write_json(stats_path, stats)

            EvaluationRunner(
                environment=environment,
                agent=policy,
                max_steps=max_steps,
                on_episode_complete=save_checkpoint,
                episode_ids=remaining_ids,
                total_episodes=total_episodes,
            ).run()
            stats.update(
                {
                    "world_model_calls": policy.world_model_calls,
                    "world_model_overrides": policy.world_model_overrides,
                    "world_model_errors": policy.world_model_errors,
                }
            )
            all_results = sorted(completed_by_id.values(), key=lambda item: item.episode_id)
        if len(all_results) != len(episode_scope):
            raise RuntimeError(
                f"Evaluation ended with {len(all_results)}/{len(episode_scope)} episodes"
            )
        summary = summarize(all_results)
    finally:
        environment.close()

    write_jsonl(output_dir / f"{run_name}.jsonl", all_results)
    write_json(
        output_dir / f"{run_name}.summary.json",
        {
            "model": model_config.get("name", model_config["model_id"]),
            "policy_variant": "world_model_guided",
            "split": split,
            "action_space": args.action_space,
            "max_steps": max_steps,
            "world_model_checkpoint": args.world_model_checkpoint,
            "episode_start": episode_start,
            "episode_end": episode_end,
            "resumed": bool(args.resume),
            **stats,
            **summary,
        },
    )
    print(json.dumps({**stats, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
