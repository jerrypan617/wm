from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SPLIT_TO_ALFWORLD = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


@dataclass(frozen=True)
class EnvironmentStep:
    observation: str
    reward: float | None
    done: bool
    won: bool
    info: dict[str, Any]


def _first_item(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _unwrap_infos(infos: dict[str, Any]) -> dict[str, Any]:
    return {key: _first_item(value) for key, value in infos.items()}


class AlfworldTextWorldEnv:
    """Small, single-environment adapter around ALFWorld's TextWorld API."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        split: str,
        num_episodes: int = -1,
        episode_indices: Sequence[int] | None = None,
    ) -> None:
        if split not in SPLIT_TO_ALFWORLD:
            raise ValueError(f"Unsupported split: {split}")

        try:
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        except ImportError as exc:
            raise RuntimeError(
                "ALFWorld is not installed. Run `uv sync`."
            ) from exc

        split_config_key = {
            "train": "data_path",
            "valid_seen": "eval_id_data_path",
            "valid_unseen": "eval_ood_data_path",
        }[split]
        data_path = os.path.expandvars(
            os.path.expanduser(config["dataset"][split_config_key])
        )
        if not Path(data_path).is_dir():
            raise FileNotFoundError(
                f"ALFWORLD_DATA is missing or invalid: {data_path}. "
                "Set ALFWORLD_DATA and run "
                "`uv run alfworld-download --data-dir \"$ALFWORLD_DATA\"`."
            )

        self.split = split
        self.config = config
        self._alfworld = AlfredTWEnv(config, train_eval=SPLIT_TO_ALFWORLD[split])
        # AlfredTWEnv builds ``game_files`` from os.walk(), whose traversal
        # order is not part of its API.  Episode ids are positional in this
        # adapter, so normalize the order before selecting a shard.  Without
        # this, two runs of the same split can assign different games to the
        # same episode id.
        self._alfworld.game_files = sorted(self._alfworld.game_files)
        self._alfworld.num_games = len(self._alfworld.game_files)
        available = self._alfworld.num_games
        if episode_indices is not None:
            indices = list(episode_indices)
            invalid = [index for index in indices if index < 0 or index >= available]
            if invalid:
                raise IndexError(
                    f"Episode indices out of range for {split}: {invalid[:3]}"
                )
            self._alfworld.game_files = [
                self._alfworld.game_files[index] for index in indices
            ]
            self._alfworld.num_games = len(self._alfworld.game_files)
            available = self._alfworld.num_games

        self.num_episodes = (
            min(num_episodes, available) if num_episodes > 0 else available
        )
        if self.num_episodes == 0:
            raise RuntimeError(
                f"No solvable games found in {split}: {data_path}"
            )
        self._env = self._alfworld.init_env(batch_size=1)

    def reset(self) -> tuple[str, dict[str, Any]]:
        observations, infos = self._env.reset()
        return str(observations[0]), _unwrap_infos(infos)

    def step(self, action: str) -> EnvironmentStep:
        observations, rewards, dones, infos = self._env.step([action])
        info = _unwrap_infos(infos)
        reward = _first_item(rewards)
        return EnvironmentStep(
            observation=str(observations[0]),
            reward=float(reward) if reward is not None else None,
            done=bool(_first_item(dones, False)),
            won=bool(info.get("won", False)),
            info=info,
        )

    @staticmethod
    def admissible_commands(info: dict[str, Any]) -> list[str]:
        commands = info.get("admissible_commands") or []
        return [str(command) for command in commands]

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close:
            close()
