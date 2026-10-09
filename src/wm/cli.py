from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wm.agents import LLMTextAgent, StructuredLLMTextAgent
from wm.environments import AlfworldTextWorldEnv
from wm.evaluation import EvaluationRunner
from wm.evaluation.metrics import summarize
from wm.evaluation.recorder import read_jsonl, write_json, write_jsonl
from wm.models import DeepSeekBackend, GenerationConfig, TransformersBackend
from wm.utils import load_experiment


def _build_generation(config: dict[str, Any]) -> GenerationConfig:
    generation = config.get("generation", {})
    return GenerationConfig(
        max_new_tokens=int(generation.get("max_new_tokens", 128)),
        temperature=float(generation.get("temperature", 0.0)),
        top_p=float(generation.get("top_p", 1.0)),
        top_k=int(generation.get("top_k", 20)),
        do_sample=bool(generation.get("do_sample", False)),
        enable_thinking=bool(generation.get("enable_thinking", False)),
    )


def _build_backend(model_config: dict[str, Any]):
    backend_name = model_config.get("backend", "transformers").lower()
    if backend_name == "deepseek":
        return DeepSeekBackend(
            model_id=model_config.get("model_id", "deepseek-chat"),
            api_key_env=model_config.get("api_key_env", "DEEPSEEK_API_KEY"),
            base_url=model_config.get("base_url"),
            timeout=float(model_config.get("timeout", 120)),
            max_retries=int(model_config.get("max_retries", 2)),
            dotenv_path=model_config.get("dotenv_path"),
        )
    if backend_name == "transformers":
        return TransformersBackend(
            model_id=model_config["model_id"],
            cache_dir=model_config.get("cache_dir"),
            device_map=model_config.get("device_map", "auto"),
            local_files_only=bool(model_config.get("local_files_only", False)),
            torch_dtype=model_config.get("torch_dtype", "auto"),
        )
    raise ValueError(f"Unsupported model backend: {backend_name}")


def _build_agent(
    model_config: dict[str, Any],
    action_space: str,
    policy_variant: str = "baseline",
) -> LLMTextAgent:
    agent_class = {
        "baseline": LLMTextAgent,
        "structured": StructuredLLMTextAgent,
    }.get(policy_variant)
    if agent_class is None:
        raise ValueError(f"Unsupported policy variant: {policy_variant}")
    return agent_class(
        _build_backend(model_config),
        _build_generation(model_config),
        action_space=action_space,
    )


def run_eval(args: argparse.Namespace) -> None:
    experiment = load_experiment(args.config)
    model_config = experiment["model"]
    environment_config = experiment["environment"]

    split = args.split or experiment.get("split", "valid_seen")
    requested_episodes = (
        args.num_episodes
        if args.num_episodes is not None
        else int(experiment.get("num_episodes", -1))
    )
    action_space = args.action_space or experiment.get("action_space", "free")
    workers = (
        args.workers
        if args.workers is not None
        else int(experiment.get("workers", 1))
    )
    if workers < 1:
        raise ValueError("--workers must be at least 1")

    backend_name = model_config.get("backend", "transformers").lower()
    if workers > 1 and backend_name != "deepseek":
        raise ValueError(
            "Parallel evaluation currently supports the DeepSeek API backend only; "
            "use --workers 1 for local Transformers inference."
        )

    output_dir = Path(args.output_dir or experiment.get("output_dir", "results"))
    run_name = f"{model_config.get('name', 'model')}_{split}"
    checkpoint_path = output_dir / f"{run_name}.checkpoint.jsonl"
    checkpoint_results = read_jsonl(checkpoint_path) if args.resume else []
    completed_by_id = {}
    for result in checkpoint_results:
        if result.split != split:
            raise ValueError(
                f"Checkpoint split={result.split!r} does not match requested split={split!r}"
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

    try:
        total_episodes = environment.num_episodes
        invalid_ids = [
            episode_id
            for episode_id in completed_by_id
            if episode_id < 0 or episode_id >= total_episodes
        ]
        if invalid_ids:
            raise ValueError(
                f"Checkpoint contains episode ids outside this evaluation: "
                f"{invalid_ids[:5]}"
            )

        episode_start = args.episode_start if args.episode_start is not None else 0
        episode_end = (
            args.episode_end if args.episode_end is not None else total_episodes
        )
        if episode_start < 0 or episode_end > total_episodes or episode_start >= episode_end:
            raise ValueError(
                f"Invalid episode range [{episode_start}, {episode_end}) for "
                f"total_episodes={total_episodes}"
            )
        episode_scope = list(range(episode_start, episode_end))
        remaining_ids = [
            episode_id
            for episode_id in episode_scope
            if episode_id not in completed_by_id
        ]

        if args.resume:
            print(
                f"Resuming from {checkpoint_path}: "
                f"{len(completed_by_id)} completed, "
                f"{len(remaining_ids)} remaining"
            )

        if not remaining_ids:
            all_results = sorted(
                completed_by_id.values(), key=lambda result: result.episode_id
            )
            summary = summarize(all_results)
            runner = None
        else:
            # For a resumed run, the base environment only contains the
            # remaining games. The runner still receives absolute episode ids.
            if args.resume or args.episode_start is not None or args.episode_end is not None:
                environment.close()
                environment = AlfworldTextWorldEnv(
                    env_config,
                    split=split,
                    num_episodes=len(remaining_ids),
                    episode_indices=remaining_ids,
                )

            agent = _build_agent(model_config, action_space, args.policy_variant)

            def save_checkpoint(result: Any) -> None:
                completed_by_id[result.episode_id] = result
                write_jsonl(
                    checkpoint_path,
                    [
                        completed_by_id[episode_id]
                        for episode_id in sorted(completed_by_id)
                    ],
                )

            runner_kwargs: dict[str, Any] = {}
            if workers > 1:
                def make_environment(indices: list[int]) -> AlfworldTextWorldEnv:
                    return AlfworldTextWorldEnv(
                        env_config,
                        split=split,
                        num_episodes=len(indices),
                        episode_indices=indices,
                    )

                def make_agent() -> LLMTextAgent:
                    return _build_agent(
                        model_config, action_space, args.policy_variant
                    )

                runner_kwargs.update(
                    environment_factory=make_environment,
                    agent_factory=make_agent,
                )

            runner = EvaluationRunner(
                environment=environment,
                agent=agent,
                max_steps=max_steps,
                on_episode_complete=save_checkpoint,
                workers=workers,
                episode_ids=remaining_ids,
                total_episodes=total_episodes,
                **runner_kwargs,
            )
            print(f"Evaluating with {workers} worker(s)")
            runner.run()

            all_results = sorted(
                completed_by_id.values(), key=lambda result: result.episode_id
            )
            if len(all_results) != len(episode_scope):
                raise RuntimeError(
                    f"Evaluation ended with {len(all_results)}/{len(episode_scope)} "
                    "episodes in the requested scope"
                )
            summary = summarize(all_results)
    finally:
        environment.close()

    jsonl_path = output_dir / f"{run_name}.jsonl"
    summary_path = output_dir / f"{run_name}.summary.json"
    write_jsonl(jsonl_path, all_results)
    write_json(
        summary_path,
        {
            "model": model_config.get("name", model_config["model_id"]),
            "split": split,
            "action_space": action_space,
            "max_steps": max_steps,
            "workers": workers,
            "policy_variant": args.policy_variant,
            "resumed": bool(args.resume),
            **summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"episodes: {jsonl_path}")
    print(f"summary:  {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="wm evaluation framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("eval", help="run an evaluation")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument(
        "--split",
        choices=["train", "valid_seen", "valid_unseen"],
    )
    evaluate.add_argument("--num-episodes", type=int)
    evaluate.add_argument("--max-steps", type=int)
    evaluate.add_argument(
        "--action-space",
        choices=["free", "admissible"],
    )
    evaluate.add_argument(
        "--policy-variant",
        choices=["baseline", "structured"],
        default="baseline",
        help="Qwen policy input variant",
    )
    evaluate.add_argument(
        "--workers",
        type=int,
        help="parallel episode workers; DeepSeek API backend only",
    )
    evaluate.add_argument(
        "--resume",
        action="store_true",
        help="resume unfinished episodes from the output checkpoint",
    )
    evaluate.add_argument(
        "--episode-start",
        type=int,
        help="inclusive absolute episode id for a local evaluation shard",
    )
    evaluate.add_argument(
        "--episode-end",
        type=int,
        help="exclusive absolute episode id for a local evaluation shard",
    )
    evaluate.add_argument("--output-dir")
    evaluate.set_defaults(handler=run_eval)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
