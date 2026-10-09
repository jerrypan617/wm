"""Structured world-model components for ALFWorld trajectories."""

from .data import (
    WorldModelVocab,
    build_records_from_results,
    load_transition_records,
    save_transition_records,
)
from .state import StructuredState, parse_observation

__all__ = [
    "QwenWorldModel",
    "StructuredState",
    "WorldModelPrediction",
    "WorldModelVocab",
    "bernoulli_kl",
    "build_records_from_results",
    "load_transition_records",
    "parse_observation",
    "save_transition_records",
]


def __getattr__(name: str):
    # Keep state/data utilities importable without importing Transformers.
    if name in {"QwenWorldModel", "WorldModelPrediction", "bernoulli_kl"}:
        from .model import QwenWorldModel, WorldModelPrediction, bernoulli_kl

        return {
            "QwenWorldModel": QwenWorldModel,
            "WorldModelPrediction": WorldModelPrediction,
            "bernoulli_kl": bernoulli_kl,
        }[name]
    raise AttributeError(name)
