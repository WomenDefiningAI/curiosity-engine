from __future__ import annotations

import os
import stat
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import private_root, repository_root

ProviderName = Literal["openai", "anthropic", "openrouter"]
RuntimeMode = Literal["api", "coding_agent_attended"]

SECRET_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

CORE_TEXT_ROLES = (
    "reasoning",
    "critic_factual",
    "critic_pedagogy",
    "critic_context",
    "critic_parent_effort",
    "critic_epistemic",
    "judge",
)


class BrainRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    model: str = Field(min_length=1, max_length=240)
    reasoning_effort: str | None = Field(default=None, max_length=32)
    capabilities: list[str] = Field(default_factory=list)


class BrainConfig(BaseModel):
    """Non-secret, family-private selection of the runtime brain stack."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    runtime: RuntimeMode = "api"
    routes: dict[str, BrainRoute] = Field(default_factory=dict)
    disclosure_acknowledged_at: str | None = None
    recommendation_status: Literal["family_evaluating", "family_recommended", "custom"] = "custom"
    notes: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_runtime(self) -> BrainConfig:
        if self.runtime == "api" and "reasoning" not in self.routes:
            raise ValueError("API runtime requires a reasoning route")
        return self


def brain_config_path(root: str | Path | None = None) -> Path:
    configured = os.environ.get("CURIOSITY_BRAIN_CONFIG")
    if configured:
        return Path(configured).resolve()
    return (Path(root).resolve() / "private" if root else private_root()) / "setup" / "brain.json"


def model_env_path(root: str | Path | None = None) -> Path:
    configured = os.environ.get("CURIOSITY_MODEL_ENV")
    if configured:
        return Path(configured).resolve()
    return (Path(root).resolve() / "private" if root else private_root()) / "setup" / "model.env"


def brain_config_fingerprint(root: str | Path | None = None) -> str:
    path = brain_config_path(root)
    if path.is_file():
        return sha256(path.read_bytes()).hexdigest()[:16]
    backend = os.environ.get("CURIOSITY_BACKEND", "deterministic")
    model = os.environ.get("CURIOSITY_MODEL", "")
    legacy_path = model_env_path(root)
    if legacy_path.is_file():
        if stat.S_IMODE(legacy_path.stat().st_mode) & 0o077:
            backend, model = "insecure-private-route", ""
        else:
            for raw_line in legacy_path.read_text(encoding="utf-8").splitlines():
                if "=" not in raw_line or raw_line.lstrip().startswith("#"):
                    continue
                key, value = raw_line.split("=", 1)
                if key.strip() == "CURIOSITY_BACKEND" and "CURIOSITY_BACKEND" not in os.environ:
                    backend = value.strip().strip("'\"")
                elif key.strip() == "CURIOSITY_MODEL" and "CURIOSITY_MODEL" not in os.environ:
                    model = value.strip().strip("'\"")
    legacy_route = f"{backend.casefold()}:{model}"
    return "legacy-" + sha256(legacy_route.encode()).hexdigest()[:16]


def _require_owner_only(path: Path) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        relative = path.relative_to(repository_root()) if path.is_relative_to(repository_root()) else path
        raise PermissionError(f"{relative} must be owner-only; run: chmod 600 {relative}")


def load_brain_config(root: str | Path | None = None) -> BrainConfig | None:
    path = brain_config_path(root)
    if not path.is_file():
        return None
    _require_owner_only(path)
    return BrainConfig.model_validate_json(path.read_text(encoding="utf-8"))


def write_brain_config(config: BrainConfig, root: str | Path | None = None) -> Path:
    path = brain_config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def configure_api_brain(
    *,
    provider: ProviderName,
    model: str,
    vision_model: str | None = None,
    image_provider: ProviderName | None = None,
    image_model: str | None = None,
    web_search: bool = False,
    reasoning_effort: str | None = None,
    recommendation_status: Literal["family_evaluating", "family_recommended", "custom"] = "custom",
    root: str | Path | None = None,
) -> BrainConfig:
    if image_model and (image_provider or provider) == "anthropic":
        raise ValueError("the maintained Anthropic route does not provide image generation; choose openai or openrouter")
    capabilities = ["structured_output"]
    if web_search:
        capabilities.append("web_search")
    route = BrainRoute(
        provider=provider,
        model=model.strip(),
        reasoning_effort=reasoning_effort,
        capabilities=capabilities,
    )
    routes = {role: route.model_copy(deep=True) for role in CORE_TEXT_ROLES}
    routes["structured_extraction"] = BrainRoute(
        provider=provider,
        model=(vision_model or model).strip(),
        reasoning_effort=reasoning_effort,
        capabilities=["structured_output", "vision", "document_ocr"],
    )
    routes["visual_qa"] = BrainRoute(
        provider=provider,
        model=(vision_model or model).strip(),
        reasoning_effort=reasoning_effort,
        capabilities=["structured_output", "vision"],
    )
    if image_model:
        routes["image_generation"] = BrainRoute(
            provider=image_provider or provider,
            model=image_model.strip(),
            capabilities=["image_generation"],
        )
    return BrainConfig(
        runtime="api",
        routes=routes,
        disclosure_acknowledged_at=datetime.now(UTC).isoformat(),
        recommendation_status=recommendation_status,
    )


def load_secret_settings(root: str | Path | None = None) -> dict[str, str]:
    keys = (
        "CURIOSITY_BACKEND",
        "CURIOSITY_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    )
    settings = {key: os.environ.get(key, "") for key in keys}
    path = model_env_path(root)
    if path.is_file():
        _require_owner_only(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            normalized = key.strip()
            if normalized in settings and not settings[normalized]:
                settings[normalized] = value.strip().strip("'\"")
    settings["CURIOSITY_BACKEND"] = settings["CURIOSITY_BACKEND"] or "deterministic"
    return settings


def secret_is_configured(provider: str, value: str) -> bool:
    if not value or "REPLACE_ME" in value or len(value) < 20:
        return False
    prefixes = {"openai": "sk-", "anthropic": "sk-ant-", "openrouter": "sk-or-"}
    return value.startswith(prefixes.get(provider, "sk-"))


def ensure_model_env_template(providers: set[str], root: str | Path | None = None) -> Path:
    path = model_env_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    existing_keys: set[str] = set()
    if path.is_file():
        _require_owner_only(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if "=" in raw_line and not raw_line.lstrip().startswith("#"):
                key, _separator, _ignored_value = raw_line.partition("=")
                existing_keys.add(key.strip())
    missing_keys = [SECRET_KEYS[provider] for provider in sorted(providers) if SECRET_KEYS[provider] not in existing_keys]
    placeholders = [f"{key}=REPLACE_ME" for key in missing_keys]
    if not path.exists():
        lines = [
            "# Curiosity Engine provider credentials. This file is ignored by Git.",
            "# Paste keys here directly; never paste them into coding-agent chat.",
            *placeholders,
        ]
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
    elif placeholders:
        # Never rewrite an existing credential file: append only missing placeholders so a
        # crash cannot truncate a family's already-configured provider keys.
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n" + "\n".join(placeholders) + "\n")
    path.chmod(0o600)
    return path


def brain_status(root: str | Path | None = None) -> dict[str, Any]:
    config = load_brain_config(root)
    secrets = load_secret_settings(root)
    if config is None:
        legacy = secrets["CURIOSITY_BACKEND"].casefold()
        legacy_provider = legacy if legacy in SECRET_KEYS else None
        return {
            "configured": False,
            "runtime": "legacy" if legacy_provider else "deterministic",
            "providers": [legacy_provider] if legacy_provider else [],
            "credentials_present": bool(
                legacy_provider and secret_is_configured(legacy_provider, secrets[SECRET_KEYS[legacy_provider]])
            ),
            "multimodal_stack_configured": False,
            "image_generation_configured": False,
            "image_generation_runtime_ready": False,
            "blockers": ["run curiosity brain configure"],
        }
    providers = {route.provider for route in config.routes.values()}
    missing_keys = [
        provider
        for provider in sorted(providers)
        if not secret_is_configured(provider, secrets[SECRET_KEYS[provider]])
    ]
    reasoning = config.routes.get("reasoning")
    vision = config.routes.get("visual_qa")
    extraction = config.routes.get("structured_extraction")
    image = config.routes.get("image_generation")
    blockers: list[str] = []
    if not config.disclosure_acknowledged_at:
        blockers.append("provider disclosure has not been acknowledged")
    if missing_keys:
        blockers.append("credentials missing for: " + ", ".join(missing_keys))
    if not reasoning or "structured_output" not in reasoning.capabilities:
        blockers.append("reasoning route must declare structured_output")
    if not vision or "vision" not in vision.capabilities:
        blockers.append("visual QA route must declare vision")
    if not extraction or "document_ocr" not in extraction.capabilities:
        blockers.append("document extraction route must declare document_ocr")
    if not image or "image_generation" not in image.capabilities:
        blockers.append("image generation route is not configured")
    return {
        "configured": not blockers,
        "runtime": config.runtime,
        "providers": sorted(providers),
        "credentials_present": not missing_keys,
        "multimodal_stack_configured": bool(vision and extraction and image),
        "image_generation_configured": bool(
            image and "image_generation" in image.capabilities and image.provider not in missing_keys
        ),
        "image_generation_runtime_ready": bool(
            image
            and "image_generation" in image.capabilities
            and image.provider == "openai"
            and image.provider not in missing_keys
        ),
        "recommendation_status": config.recommendation_status,
        "blockers": blockers,
    }
