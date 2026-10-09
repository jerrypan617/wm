#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wm.world_model.data import (  # noqa: E402
    TransitionDataset,
    WorldModelVocab,
    load_transition_records,
)
from wm.world_model.model import QwenWorldModel  # noqa: E402


def _run_epoch(
    model: QwenWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    beta: float,
    state_positive_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        outputs = model(
            batch["observation"],
            batch["next_observation"],
            batch["action"],
        )
        losses = model.free_energy(
            outputs,
            batch,
            beta=beta,
            state_positive_weight=state_positive_weight,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
        batch_size = len(batch["observation"])
        count += batch_size
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
    return {key: value / max(1, count) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Qwen ALFWorld world model")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--state-hidden-size", type=int, default=512)
    parser.add_argument("--action-embedding-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta-start", type=float, default=0.0)
    parser.add_argument("--beta-end", type=float, default=1.0)
    parser.add_argument(
        "--state-positive-weight",
        type=float,
        default=10.0,
        help="positive-class weight for sparse structured-state facts",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="optional deterministic cap for fast ablations; use -1 for all records",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-qlora",
        action="store_false",
        dest="qlora",
        help="disable QLoRA for a full-precision frozen-backbone smoke test",
    )
    parser.set_defaults(qlora=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-module",
        action="append",
        dest="lora_target_modules",
        help="repeat to override the default Qwen linear-attention/MLP targets",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_transition_records(args.dataset)
    if args.max_samples > 0 and len(records) > args.max_samples:
        records = random.sample(records, args.max_samples)
    if len(records) < 2:
        raise SystemExit("At least two transitions are required")
    vocab = WorldModelVocab.load(args.vocab)
    dataset = TransitionDataset(records, vocab)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False)

    model = QwenWorldModel(
        args.model_id,
        vocab,
        cache_dir=args.cache_dir,
        local_files_only=True,
        device_map=args.device_map,
        max_length=args.max_length,
        state_hidden_size=args.state_hidden_size,
        action_embedding_size=args.action_embedding_size,
        freeze_backbone=True,
        qlora=args.qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        fraction = epoch / max(1, args.epochs - 1)
        beta = args.beta_start + fraction * (args.beta_end - args.beta_start)
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            beta,
            args.state_positive_weight,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                None,
                beta,
                args.state_positive_weight,
            )
        row: dict[str, float | int] = {"epoch": epoch + 1, "beta": beta}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": model.architecture,
            "vocab": vocab.to_dict(),
            "heads": model.head_state_dict(),
            "history": history,
        },
        output,
    )
    print(f"checkpoint: {output}")


if __name__ == "__main__":
    main()
