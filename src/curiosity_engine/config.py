from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_routing import validate_model_config
from .policies import validate_reasoning_policy


class ConfigurationError(ValueError):
    pass


def repository_root() -> Path:
    configured = os.environ.get("CURIOSITY_REPO_ROOT")
    if configured:
        return Path(configured).resolve()
    source = Path(__file__).resolve().parents[2]
    if (source / "configs").is_dir():
        return source
    return Path.cwd().resolve()


def configuration_root() -> Path:
    source = repository_root()
    if (source / "configs").is_dir():
        return source
    installed = Path(sys.prefix) / "share" / "curiosity-engine"
    if (installed / "configs").is_dir():
        return installed
    raise ConfigurationError("cannot locate Curiosity Engine configs; set CURIOSITY_REPO_ROOT")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot load {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must contain a JSON object")
    return data


@dataclass(frozen=True)
class AppConfig:
    root: Path
    production: dict[str, Any]
    reasoning: dict[str, Any]
    context: dict[str, Any]
    autonomy: dict[str, Any]
    promotion: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path | None = None) -> AppConfig:
        repo = Path(root).resolve() if root else configuration_root()
        cfg = repo / "configs"
        app = cls(
            root=repo,
            production=_load_json(cfg / "production.json"),
            reasoning=_load_json(cfg / "reasoning-policy.json"),
            context=_load_json(cfg / "context-policy.json"),
            autonomy=_load_json(cfg / "autonomy-policy.json"),
            promotion=_load_json(cfg / "promotion-policy.json"),
        )
        errors = app.validate()
        if errors:
            raise ConfigurationError("; ".join(errors))
        return app

    def validate(self) -> list[str]:
        errors = validate_reasoning_policy(self.reasoning)
        errors.extend(validate_model_config(self.production))
        runtime = self.production.get("runtime") or {}
        for key in ("reasoning_policy", "context_policy", "autonomy_policy"):
            configured = runtime.get(key)
            if not configured:
                errors.append(f"production.runtime.{key} is required")
            elif not (self.root / configured).exists():
                errors.append(f"production.runtime.{key} does not exist: {configured}")
        return errors

    def workflow(self, name: str) -> dict[str, Any]:
        try:
            return dict(self.reasoning["workflows"][name])
        except KeyError as exc:
            raise ConfigurationError(f"unknown workflow: {name}") from exc

    @property
    def model_defaults(self) -> dict[str, Any]:
        return dict(self.production.get("models") or {})
