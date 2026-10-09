from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .state import StructuredState, parse_observation, state_facts


_WORD_RE = re.compile(r"[a-z][a-z0-9_]*", re.IGNORECASE)


def tokenize_observation(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text or "")]


@dataclass(frozen=True)
class TransitionRecord:
    episode_id: int
    step: int
    observation: str
    action: str
    next_observation: str
    state: StructuredState
    next_state: StructuredState

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "observation": self.observation,
            "action": self.action,
            "next_observation": self.next_observation,
            "state": self.state.to_dict(),
            "next_state": self.next_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransitionRecord":
        return cls(
            episode_id=int(payload["episode_id"]),
            step=int(payload["step"]),
            observation=str(payload["observation"]),
            action=str(payload["action"]),
            next_observation=str(payload["next_observation"]),
            state=StructuredState.from_dict(payload["state"]),
            next_state=StructuredState.from_dict(payload["next_state"]),
        )


def build_records_from_results(
    paths: Sequence[str | Path],
    *,
    reparse_structured_states: bool = False,
) -> list[TransitionRecord]:
    records: list[TransitionRecord] = []
    for path in paths:
        input_path = Path(path)
        for line_number, line in enumerate(
            input_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            payload = json.loads(line)
            observations = [str(item) for item in payload.get("observations", [])]
            actions = [str(item) for item in payload.get("actions", [])]
            structured_states = payload.get("structured_states")
            if len(observations) != len(actions) + 1:
                raise ValueError(
                    f"{input_path}:{line_number} has {len(observations)} observations "
                    f"for {len(actions)} actions"
                )
            for step, action in enumerate(actions):
                observation = observations[step]
                next_observation = observations[step + 1]
                if (
                    not reparse_structured_states
                    and
                    isinstance(structured_states, list)
                    and len(structured_states) == len(observations)
                ):
                    state = StructuredState.from_dict(structured_states[step])
                    next_state = StructuredState.from_dict(structured_states[step + 1])
                else:
                    state = parse_observation(observation)
                    next_state = parse_observation(next_observation)
                records.append(
                    TransitionRecord(
                        episode_id=int(payload.get("episode_id", -1)),
                        step=step,
                        observation=observation,
                        action=action,
                        next_observation=next_observation,
                        state=state,
                        next_state=next_state,
                    )
                )
    return records


def save_transition_records(path: str | Path, records: Iterable[TransitionRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_transition_records(path: str | Path) -> list[TransitionRecord]:
    input_path = Path(path)
    return [
        TransitionRecord.from_dict(json.loads(line))
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class WorldModelVocab:
    facts: list[str]
    actions: list[str]
    observation_words: list[str]

    @classmethod
    def build(
        cls,
        records: Sequence[TransitionRecord],
        *,
        max_observation_words: int = 4096,
    ) -> "WorldModelVocab":
        fact_set: set[str] = set()
        action_set: set[str] = set()
        word_counts: Counter[str] = Counter()
        for record in records:
            fact_set.update(state_facts((record.state, record.next_state)))
            action_set.add(record.action.strip().lower())
            word_counts.update(tokenize_observation(record.observation))
            word_counts.update(tokenize_observation(record.next_observation))
        words = [word for word, _ in word_counts.most_common(max_observation_words)]
        return cls(
            facts=sorted(fact_set),
            actions=["<unk>"] + sorted(action_set),
            observation_words=["<unk>"] + words,
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "facts": self.facts,
            "actions": self.actions,
            "observation_words": self.observation_words,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorldModelVocab":
        return cls(
            facts=list(payload["facts"]),
            actions=list(payload["actions"]),
            observation_words=list(payload["observation_words"]),
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "WorldModelVocab":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _index(self, values: Sequence[str]) -> dict[str, int]:
        return {value: index for index, value in enumerate(values)}

    def state_vector(self, state: StructuredState) -> torch.Tensor:
        index = self._index(self.facts)
        vector = torch.zeros(len(self.facts), dtype=torch.float32)
        for fact in state_facts((state,)):
            if fact in index:
                vector[index[fact]] = 1.0
        return vector

    def action_index(self, action: str) -> int:
        index = self._index(self.actions)
        return index.get(action.strip().lower(), 0)

    def observation_vector(self, observation: str) -> torch.Tensor:
        index = self._index(self.observation_words)
        vector = torch.zeros(len(self.observation_words), dtype=torch.float32)
        for word in set(tokenize_observation(observation)):
            if word in index:
                vector[index[word]] = 1.0
        return vector


class TransitionDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: Sequence[TransitionRecord], vocab: WorldModelVocab):
        self.records = list(records)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "observation": record.observation,
            "next_observation": record.next_observation,
            "state": self.vocab.state_vector(record.state),
            "next_state": self.vocab.state_vector(record.next_state),
            "action": torch.tensor(self.vocab.action_index(record.action), dtype=torch.long),
            "observation_bow": self.vocab.observation_vector(record.observation),
            "next_observation_bow": self.vocab.observation_vector(record.next_observation),
        }
