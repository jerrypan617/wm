#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from wm.world_model.data import WorldModelVocab  # noqa: E402
from wm.world_model.model import QwenWorldModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict ALFWorld next states for candidate actions"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--action", nargs="+", required=True)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = WorldModelVocab.from_dict(checkpoint["vocab"])
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
    predictions = model.predict_next(
        args.observation,
        args.action,
        threshold=args.threshold,
    )
    output = []
    for action, prediction in zip(args.action, predictions):
        output.append(
            {
                "action": action,
                "predicted_facts": prediction.predicted_facts[0],
                "reconstructed_observation_tokens": prediction.reconstructed_words[0],
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
