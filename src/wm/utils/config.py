from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries before expanding YAML variables."""
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    _load_dotenv(Path.cwd() / ".env")
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = _expand(config)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_config_path(path: str | Path, base_dir: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base_dir) / candidate
    return candidate.resolve()


def load_experiment(path: str | Path) -> dict[str, Any]:
    experiment = load_yaml(path)
    base_dir = experiment["_config_dir"]
    model = load_yaml(resolve_config_path(experiment["model_config"], base_dir))
    environment = load_yaml(
        resolve_config_path(experiment["environment_config"], base_dir)
    )
    if model.get("cache_dir"):
        model["cache_dir"] = str(
            resolve_config_path(model["cache_dir"], model["_config_dir"])
        )
    experiment["model"] = model
    experiment["environment"] = environment
    return experiment
