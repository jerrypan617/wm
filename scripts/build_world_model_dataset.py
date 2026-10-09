#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.world_model.data import (
    WorldModelVocab,
    build_records_from_results,
    save_transition_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ALFWorld episode JSONL results into world-model transitions"
    )
    parser.add_argument("--input", nargs="+", required=True, help="episode JSONL files")
    parser.add_argument("--output", required=True, help="transition JSONL output")
    parser.add_argument(
        "--vocab-output",
        required=True,
        help="JSON vocabulary used by the world-model heads",
    )
    parser.add_argument("--max-observation-words", type=int, default=4096)
    parser.add_argument(
        "--reparse-structured-states",
        action="store_true",
        help="reparse observations with the current state parser instead of stored states",
    )
    args = parser.parse_args()

    records = build_records_from_results(
        args.input,
        reparse_structured_states=args.reparse_structured_states,
    )
    if not records:
        raise SystemExit("No transitions were found in the input files")
    vocab = WorldModelVocab.build(
        records,
        max_observation_words=args.max_observation_words,
    )
    save_transition_records(args.output, records)
    vocab.save(args.vocab_output)
    print(f"transitions: {len(records)}")
    print(f"state facts: {len(vocab.facts)}")
    print(f"actions: {len(vocab.actions)}")
    print(f"observation words: {len(vocab.observation_words)}")
    print(f"dataset: {Path(args.output)}")
    print(f"vocab: {Path(args.vocab_output)}")


if __name__ == "__main__":
    main()
