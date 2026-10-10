#!/usr/bin/env python3
"""Evaluate a policy + learned World Model with an EFE action selector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.agents import EFEWorldModelAgent  # noqa: E402
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-model-checkpoint", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--action-space", choices=["free", "admissible"], default=None)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--progress-weight", type=float, default=None)
    parser.add_argument("--information-gain-weight", type=float, default=None)
    parser.add_argument("--action-cost-weight", type=float, default=None)
    parser.add_argument("--revisit-penalty", type=float, default=None)
    parser.add_argument("--horizon-discount", type=float, default=None)
    parser.add_argument("--override-margin", type=float, default=None)
    parser.add_argument("--online-vfe", action="store_true", default=None)
    parser.add_argument("--online-vfe-lr", type=float, default=None)
    parser.add_argument("--online-vfe-beta", type=float, default=None)
    parser.add_argument("--online-vfe-state-positive-weight", type=float, default=None)
    parser.add_argument("--online-vfe-gradient-clip-norm", type=float, default=None)
    parser.add_argument("--online-vfe-steps", type=int, default=None)
    parser.add_argument(
        "--online-vfe-scope",
        choices=["all", "heads", "transition"],
        default=None,
    )
    parser.add_argument("--online-vfe-update-interval", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    experiment = load_experiment(args.config)
    model_config = experiment["model"]
    environment_config = experiment["environment"]
    efe_config = dict(experiment.get("efe", {}))
    efe_settings = {
        "progress_weight": float(efe_config.get("progress_weight", 1.0)),
        "information_gain_weight": float(
            efe_config.get("information_gain_weight", 0.35)
        ),
        "action_cost_weight": float(efe_config.get("action_cost_weight", 0.15)),
        "revisit_penalty": float(efe_config.get("revisit_penalty", 0.25)),
        "horizon_discount": float(efe_config.get("horizon_discount", 0.8)),
        "override_margin": float(efe_config.get("override_margin", 0.10)),
    }
    for key in efe_settings:
        override = getattr(args, key)
        if override is not None:
            efe_settings[key] = override
    online_vfe_config = dict(experiment.get("online_vfe", {}))
    online_vfe_enabled = (
        bool(args.online_vfe)
        if args.online_vfe is not None
        else bool(online_vfe_config.get("enabled", False))
    )
    online_vfe_settings = {
        "enabled": online_vfe_enabled,
        "learning_rate": float(online_vfe_config.get("learning_rate", 1e-5)),
        "beta": float(online_vfe_config.get("beta", 1.0)),
        "state_positive_weight": float(
            online_vfe_config.get("state_positive_weight", 10.0)
        ),
        "gradient_clip_norm": float(
            online_vfe_config.get("gradient_clip_norm", 1.0)
        ),
        "steps": int(online_vfe_config.get("steps", 1)),
        "scope": str(online_vfe_config.get("scope", "all")),
        "update_interval": int(online_vfe_config.get("update_interval", 1)),
    }
    online_vfe_overrides = {
        "learning_rate": args.online_vfe_lr,
        "beta": args.online_vfe_beta,
        "state_positive_weight": args.online_vfe_state_positive_weight,
        "gradient_clip_norm": args.online_vfe_gradient_clip_norm,
        "steps": args.online_vfe_steps,
        "scope": args.online_vfe_scope,
        "update_interval": args.online_vfe_update_interval,
    }
    for key, override in online_vfe_overrides.items():
        if override is not None:
            online_vfe_settings[key] = override
    split = args.split or experiment.get("split", "valid_seen")
    requested_episodes = (
        args.num_episodes
        if args.num_episodes is not None
        else int(experiment.get("num_episodes", -1))
    )
    action_space = args.action_space or experiment.get("action_space", "admissible")
    output_dir = Path(args.output_dir or experiment.get("output_dir", "results/efe"))
    run_name = f"{model_config.get('name', 'policy')}_{split}_efe"
    checkpoint_path = output_dir / f"{run_name}.checkpoint.jsonl"
    stats_path = output_dir / f"{run_name}.planner_stats.json"
    checkpoint_results = read_jsonl(checkpoint_path) if args.resume else []
    completed_by_id = {}
    for result in checkpoint_results:
        if result.split != split:
            raise ValueError(
                f"Checkpoint split={result.split!r} does not match requested "
                f"split={split!r}"
            )
        if result.episode_id in completed_by_id:
            raise ValueError(
                f"Duplicate episode_id in checkpoint: {result.episode_id}"
            )
        completed_by_id[result.episode_id] = result

    env_config = dict(environment_config)
    env_config.pop("_config_path", None)
    env_config.pop("_config_dir", None)
    environment = AlfworldTextWorldEnv(
        env_config,
        split=split,
        num_episodes=requested_episodes,
    )
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else environment_config.get("max_steps", 50)
    )
    episode_start = args.episode_start if args.episode_start is not None else 0
    episode_end = args.episode_end
    stats = {
        "world_model_calls": 0,
        "world_model_errors": 0,
        "planner_decisions": 0,
        "online_vfe_updates": 0,
        "online_vfe_errors": 0,
        "online_vfe_loss_sum": 0.0,
    }
    if args.resume and stats_path.is_file():
        stats.update(json.loads(stats_path.read_text(encoding="utf-8")))

    try:
        total_episodes = environment.num_episodes
        episode_end = total_episodes if episode_end is None else episode_end
        if (
            episode_start < 0
            or episode_end > total_episodes
            or episode_start >= episode_end
        ):
            raise ValueError(
                f"Invalid episode range [{episode_start}, {episode_end}) for "
                f"total_episodes={total_episodes}"
            )
        episode_scope = list(range(episode_start, episode_end))
        out_of_range = [
            episode_id
            for episode_id in completed_by_id
            if episode_id not in episode_scope
        ]
        if out_of_range:
            raise ValueError(
                "Checkpoint contains episodes outside the requested range; "
                "use a separate output directory for a different shard."
            )
        remaining_ids = [
            episode_id
            for episode_id in episode_scope
            if episode_id not in completed_by_id
        ]
        print(
            f"EFE evaluation: {len(completed_by_id)} completed, "
            f"{len(remaining_ids)} remaining"
        )
        if remaining_ids:
            environment.close()
            environment = AlfworldTextWorldEnv(
                env_config,
                split=split,
                num_episodes=len(remaining_ids),
                episode_indices=remaining_ids,
            )
            agent = EFEWorldModelAgent(
                _build_backend(model_config),
                _build_generation(model_config),
                _load_world_model(args.world_model_checkpoint, args.cache_dir),
                action_space=action_space,
                max_candidates=int(
                    args.max_candidates
                    if args.max_candidates is not None
                    else efe_config.get("max_candidates", 12)
                ),
                max_location_candidates=int(
                    efe_config.get("max_location_candidates", 5)
                ),
                progress_weight=efe_settings["progress_weight"],
                information_gain_weight=efe_settings["information_gain_weight"],
                action_cost_weight=efe_settings["action_cost_weight"],
                revisit_penalty=efe_settings["revisit_penalty"],
                horizon_discount=efe_settings["horizon_discount"],
                override_margin=efe_settings["override_margin"],
                online_vfe=online_vfe_enabled,
                online_vfe_beta=online_vfe_settings["beta"],
                online_vfe_state_positive_weight=online_vfe_settings[
                    "state_positive_weight"
                ],
                online_vfe_gradient_clip_norm=online_vfe_settings[
                    "gradient_clip_norm"
                ],
                online_vfe_steps=online_vfe_settings["steps"],
                online_vfe_update_interval=online_vfe_settings["update_interval"],
            )
            if online_vfe_enabled:
                agent.world_model.enable_online_vfe(
                    online_vfe_settings["learning_rate"],
                    parameter_scope=online_vfe_settings["scope"],
                )
            agent.world_model_calls = int(stats["world_model_calls"])
            agent.world_model_errors = int(stats["world_model_errors"])
            agent.planner_decisions = int(stats["planner_decisions"])
            agent.online_vfe_updates = int(stats["online_vfe_updates"])
            agent.online_vfe_errors = int(stats["online_vfe_errors"])
            agent.online_vfe_loss_sum = float(stats["online_vfe_loss_sum"])

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
                        "planner_decisions": agent.planner_decisions,
                        "online_vfe_updates": agent.online_vfe_updates,
                        "online_vfe_errors": agent.online_vfe_errors,
                        "online_vfe_loss_sum": agent.online_vfe_loss_sum,
                    }
                )
                write_json(stats_path, stats)

            EvaluationRunner(
                environment=environment,
                agent=agent,
                max_steps=max_steps,
                on_episode_complete=save_checkpoint,
                episode_ids=remaining_ids,
                total_episodes=total_episodes,
            ).run()
            stats.update(
                {
                    "world_model_calls": agent.world_model_calls,
                    "world_model_errors": agent.world_model_errors,
                    "planner_decisions": agent.planner_decisions,
                    "online_vfe_updates": agent.online_vfe_updates,
                    "online_vfe_errors": agent.online_vfe_errors,
                    "online_vfe_loss_sum": agent.online_vfe_loss_sum,
                }
            )
            all_results = sorted(
                completed_by_id.values(), key=lambda result: result.episode_id
            )
        else:
            all_results = sorted(
                completed_by_id.values(), key=lambda result: result.episode_id
            )
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
            "policy_variant": "policy_plus_world_model_efe",
            "split": split,
            "action_space": action_space,
            "max_steps": max_steps,
            "world_model_checkpoint": args.world_model_checkpoint,
            "episode_start": episode_start,
            "episode_end": episode_end,
            "efe": efe_settings,
            "online_vfe": online_vfe_settings,
            "resumed": bool(args.resume),
            **stats,
            **summary,
        },
    )
    print(json.dumps({**stats, **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
