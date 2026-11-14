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


def family_home() -> Path:
    """Return the mutable household root, separate from the installed runtime.

    Source checkouts retain the historical ``<repo>/private`` layout unless an
    explicit home is configured. Packaged installs use ``~/.curiosity-engine``.
    """

    configured = os.environ.get("CURIOSITY_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    source = repository_root()
    if (source / ".git").exists() or (source / "private").exists():
        return source
    return (Path.home() / ".curiosity-engine").resolve()


def private_root() -> Path:
    configured = os.environ.get("CURIOSITY_PRIVATE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return family_home() / "private"


def family_workspace() -> Path:
    configured = os.environ.get("CURIOSITY_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    return family_home() / "workspace"


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
        configured_roles = set((self.production.get("models") or {}).keys())
        for workflow_name, workflow in (self.reasoning.get("workflows") or {}).items():
            required_roles = [str(workflow.get("generator_role") or "")]
            required_roles.extend(f"critic_{critic}" for critic in workflow.get("critics") or [])
            for role in required_roles:
                if role and role not in configured_roles:
                    errors.append(
                        f"reasoning.workflows.{workflow_name} requires missing production model role: {role}"
                    )
        for component_name, component in (self.production.get("components") or {}).items():
            if not isinstance(component, dict):
                continue
            required_roles = [str(component.get("model_role") or "")]
            required_roles.extend(str(role) for role in component.get("critics") or [])
            for role in required_roles:
                if role and role not in configured_roles:
                    errors.append(f"production.components.{component_name} requires missing model role: {role}")
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
