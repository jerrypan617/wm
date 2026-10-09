#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.agents import (  # noqa: E402
    StructuredALFWorldAgent,
    WorldModelGuidedStructuredAgent,
)
from wm.environments import AlfworldTextWorldEnv  # noqa: E402
from wm.evaluation import EvaluationRunner  # noqa: E402
from wm.evaluation.recorder import write_json, write_jsonl  # noqa: E402
from wm.utils.config import load_yaml  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Evaluate the structured ALFWorld planner")
    parser.add_argument("--environment-config", required=True)
    parser.add_argument("--split", default="valid_seen")
    parser.add_argument("--num-episodes", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--world-model-checkpoint",
        help="optional QLoRA checkpoint for conservative fallback guidance",
    )
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--wm-max-candidates", type=int, default=8)
    parser.add_argument("--wm-override-margin", type=float, default=0.15)
    args = parser.parse_args()

    config = load_yaml(args.environment_config)
    config.pop("_config_path", None)
    config.pop("_config_dir", None)
    env = AlfworldTextWorldEnv(config, split=args.split, num_episodes=args.num_episodes)
    world_model = None
    if args.world_model_checkpoint:
        world_model = _load_world_model(args.world_model_checkpoint, args.cache_dir)
    agent = (
        WorldModelGuidedStructuredAgent(
            world_model,
            max_candidates=args.wm_max_candidates,
            override_margin=args.wm_override_margin,
        )
        if world_model is not None
        else StructuredALFWorldAgent()
    )
    results = []
    try:
        results, summary = EvaluationRunner(
            environment=env,
            agent=agent,
            max_steps=args.max_steps,
            workers=1,
        ).run()
    finally:
        env.close()
    output_dir = Path(args.output_dir)
    run_name = f"structured_planner_{args.split}"
    write_jsonl(output_dir / f"{run_name}.jsonl", results)
    write_json(
        output_dir / f"{run_name}.summary.json",
        {
            "model": "world_model_guided_structured_planner"
            if world_model is not None
            else "structured_planner",
            "split": args.split,
            "action_space": "admissible",
            "max_steps": args.max_steps,
            **(
                {
                    "world_model_checkpoint": args.world_model_checkpoint,
                    "world_model_calls": agent.wm_calls,
                    "world_model_overrides": agent.wm_overrides,
                    "world_model_errors": agent.wm_errors,
                }
                if world_model is not None
                else {}
            ),
            **summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
