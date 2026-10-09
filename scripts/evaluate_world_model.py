#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.world_model.data import (  # noqa: E402
    TransitionDataset,
    WorldModelVocab,
    build_records_from_results,
)
from wm.world_model.model import QwenWorldModel  # noqa: E402


def _counts(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> tuple[int, int, int]:
    predicted = torch.sigmoid(logits) >= threshold
    expected = target >= 0.5
    tp = int((predicted & expected).sum())
    fp = int((predicted & ~expected).sum())
    fn = int((~predicted & expected).sum())
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a structured ALFWorld world model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", nargs="+", required=True, help="episode JSONL files")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output",
        help="optional JSON path for persisting evaluation metrics",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = WorldModelVocab.from_dict(checkpoint["vocab"])
    records = build_records_from_results(args.input, reparse_structured_states=True)
    dataset = TransitionDataset(records, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    architecture = checkpoint["architecture"]
    model = QwenWorldModel(
        architecture["model_id"],
        vocab,
        cache_dir=args.cache_dir,
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

    totals: dict[str, float] = {}
    count = 0
    posterior_counts = [0, 0, 0]
    transition_counts = [0, 0, 0]
    for batch in loader:
        with torch.no_grad():
            outputs = model(
                batch["observation"],
                batch["next_observation"],
                batch["action"],
            )
            losses = model.free_energy(outputs, batch, beta=1.0)
        size = len(batch["observation"])
        count += size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * size
        for index, value in enumerate(
            _counts(
                outputs["next_q_logits"],
                batch["next_state"].to(model.device),
                args.threshold,
            )
        ):
            posterior_counts[index] += value
        for index, value in enumerate(
            _counts(
                outputs["prior_next_logits"],
                batch["next_state"].to(model.device),
                args.threshold,
            )
        ):
            transition_counts[index] += value

    result = {
        "transitions": count,
        **{key: value / max(1, count) for key, value in totals.items()},
        "posterior_next_state": _f1(*posterior_counts),
        "transition_next_state": _f1(*transition_counts),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
