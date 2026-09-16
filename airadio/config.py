"""Configuration loading: config.yaml for behaviour, .env for secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Read-only view over config.yaml plus environment secrets.

    Values are reached with dotted paths: ``cfg.get("planner.block_minutes")``.
    """

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, config_path: str | os.PathLike | None = None,
             root: str | os.PathLike | None = None) -> "Config":
        root_path = Path(root) if root else PROJECT_ROOT
        load_dotenv(root_path / ".env")

        path = Path(config_path) if config_path else root_path / "config" / "config.yaml"
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        # config.local.yaml lets the laptop override the committed defaults.
        local = path.with_name("config.local.yaml")
        if local.exists():
            with open(local, "r", encoding="utf-8") as handle:
                data = _deep_merge(data, yaml.safe_load(handle) or {})

        return cls(data, root_path)

    # -- access -------------------------------------------------------------

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict:
        value = self._data.get(name)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def env(name: str, default: str = "") -> str:
        return os.environ.get(name, default) or default

    def path(self, key: str) -> Path:
        """Resolve a configured directory, creating it if needed."""
        raw = self.get(f"paths.{key}") or key
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    @property
    def db_path(self) -> Path:
        return self.path("state") / "radio.db"

    @property
    def now_playing_path(self) -> Path:
        return self.path("state") / "now_playing.txt"

    def mood_for_hour(self, hour: int) -> dict:
        """Time-of-day profile for a 0-23 hour, with a safe default."""
        for entry in self.get("planner.mood_map", []) or []:
            start, end = entry.get("hours", [0, 24])
            if start <= hour < end:
                return entry
        return {"name": "default", "mood": "varied and listenable", "energy": [1, 5]}


_cached: Config | None = None


def get_config(reload: bool = False) -> Config:
    global _cached
    if _cached is None or reload:
        _cached = Config.load()
    return _cached
