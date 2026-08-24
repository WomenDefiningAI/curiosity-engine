from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL_ROLES = {
    "reasoning": {"purpose": "curiosity ladders, experience design, synthesis"},
    "structured_extraction": {"purpose": "school/newsletter and signal extraction"},
    "image_generation": {"purpose": "decorative/illustrative visuals only"},
    "visual_qa": {"purpose": "multimodal inspection of final artifacts"},
    "cheap_classifier": {"purpose": "routing, attribution, low-risk classification"},
    "critic_factual": {"purpose": "adversarial factual and uncertainty review"},
    "critic_pedagogy": {"purpose": "adversarial pedagogy review"},
    "critic_context": {"purpose": "adversarial context/retrieval review"},
    "critic_parent_effort": {"purpose": "busy-parent feasibility review"},
    "critic_epistemic": {"purpose": "challenge child-state claims and evidence"},
    "critic_visual": {"purpose": "challenge visual plans before rendering"},
    "judge": {"purpose": "independent selection and evaluation"},
}


@dataclass(frozen=True)
class ModelRoute:
    role: str
    provider: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None


def resolve_model_role(config: dict[str, Any], role: str) -> ModelRoute:
    if role not in DEFAULT_MODEL_ROLES:
        raise ValueError(f"unknown model role: {role}")
    entry = (config.get("models") or {}).get(role) or {}
    return ModelRoute(
        role=role,
        provider=entry.get("provider"),
        model=entry.get("model"),
        reasoning_effort=entry.get("reasoning_effort"),
    )


def validate_model_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    models = config.get("models") or {}
    unknown = set(models) - set(DEFAULT_MODEL_ROLES)
    if unknown:
        errors.append("unknown model roles: " + ", ".join(sorted(unknown)))
    for role, entry in models.items():
        if not isinstance(entry, dict):
            errors.append(f"models.{role} must be a mapping")
            continue
        if bool(entry.get("model")) != bool(entry.get("provider")):
            errors.append(f"models.{role} should set provider and model together")
    return errors
