#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.evaluation.metrics import summarize  # noqa: E402
from wm.evaluation.recorder import write_json, write_jsonl  # noqa: E402
from wm.evaluation.schemas import EpisodeResult  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint evaluation shards")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="valid_seen")
    parser.add_argument("--action-space", default="admissible")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    records = []
    seen: set[int] = set()
    for input_path in args.input:
        for line in Path(input_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            episode_id = int(payload["episode_id"])
            if episode_id in seen:
                raise ValueError(f"Duplicate episode_id: {episode_id}")
            seen.add(episode_id)
            records.append(payload)
    records.sort(key=lambda item: int(item["episode_id"]))
    episode_results = [EpisodeResult(**record) for record in records]
    summary = summarize(episode_results)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / f"{args.run_name}.jsonl", episode_results)
    write_json(
        output_dir / f"{args.run_name}.summary.json",
        {
            "model": args.model,
            "split": args.split,
            "action_space": args.action_space,
            "max_steps": args.max_steps,
            "shards": list(args.input),
            **summary,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
