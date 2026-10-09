#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.environments import AlfworldTextWorldEnv
from wm.evaluation.recorder import write_jsonl
from wm.evaluation.schemas import EpisodeResult
from wm.utils.config import load_yaml
from wm.world_model.state import parse_observation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect admissible-action ALFWorld trajectories for the world model"
    )
    parser.add_argument("--environment-config", required=True)
    parser.add_argument("--split", choices=["train", "valid_seen", "valid_unseen"], default="train")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=["random", "expert"],
        default="random",
        help=(
            "transition collection policy; expert follows the first action "
            "from ALFWorld's train-time expert plan"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_yaml(args.environment_config)
    config.pop("_config_path", None)
    config.pop("_config_dir", None)
    environment = AlfworldTextWorldEnv(
        config,
        split=args.split,
        num_episodes=args.num_episodes,
    )
    results: list[EpisodeResult] = []
    try:
        for episode_id in range(environment.num_episodes):
            observation, info = environment.reset()
            observations = [observation]
            structured_states = [parse_observation(observation).to_dict()]
            actions: list[str] = []
            raw_responses: list[str] = []
            success = False
            invalid_actions = 0
            for _ in range(args.max_steps):
                admissible = environment.admissible_commands(info)
                if args.policy == "expert":
                    expert_plan = info.get("extra.expert_plan") or []
                    if isinstance(expert_plan, (list, tuple)) and expert_plan:
                        action = str(expert_plan[0])
                    else:
                        action = random.choice(admissible) if admissible else "look"
                else:
                    action = random.choice(admissible) if admissible else "look"
                step = environment.step(action)
                actions.append(action)
                raw_responses.append("")
                observations.append(step.observation)
                structured_states.append(parse_observation(step.observation).to_dict())
                observation, info = step.observation, step.info
                if admissible and action not in admissible:
                    invalid_actions += 1
                if step.done:
                    success = step.won
                    break
            results.append(
                EpisodeResult(
                    episode_id=episode_id,
                    split=args.split,
                    gamefile=(
                        str(info.get("extra.gamefile"))
                        if info.get("extra.gamefile") is not None
                        else None
                    ),
                    success=success,
                    steps=len(actions),
                    invalid_actions=invalid_actions,
                    actions=actions,
                    raw_responses=raw_responses,
                    observations=observations,
                    structured_states=structured_states,
                )
            )
            print(
                f"[{episode_id + 1}/{environment.num_episodes}] "
                f"success={success} steps={len(actions)}"
            )
    finally:
        environment.close()

    write_jsonl(Path(args.output), results)
    print(f"saved trajectories: {args.output}")


if __name__ == "__main__":
    main()
