from __future__ import annotations

from typing import Any

TRUST_TIERS = {"A", "B", "C"}
ASSET_METHODS = {"trusted_source", "deterministic", "generative", "none"}
CERTAINTY_LEVELS = {"established", "likely", "debated", "unknown"}

# Knowledge-bearing visuals should not use generative image models as their source of truth.
FORBIDDEN_GENERATIVE_KINDS = {
    "anatomy_diagram",
    "map",
    "number_line",
    "graph",
    "chart",
    "clock",
    "coin_set",
    "counting_set",
    "periodic_table",
    "electron_configuration",
    "geometry_diagram",
    "life_cycle_sequence",
    "labeled_scientific_diagram",
}


def validate_fact_model(model: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    facts = model.get("facts")
    if not isinstance(facts, list) or not facts:
        errors.append("fact_model.facts must be a non-empty list")
    else:
        for i, fact in enumerate(facts):
            if not isinstance(fact, dict):
                errors.append(f"fact_model.facts[{i}] must be an object")
                continue
            if not fact.get("claim"):
                errors.append(f"fact_model.facts[{i}].claim is required")
            certainty = fact.get("certainty", "established")
            if certainty not in CERTAINTY_LEVELS:
                errors.append(f"fact_model.facts[{i}].certainty invalid: {certainty}")
            if certainty in {"established", "likely", "debated"} and not fact.get("source"):
                errors.append(f"fact_model.facts[{i}].source is required for {certainty} claims")
    return errors


def validate_artifact_trust(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tier = spec.get("trust_tier")
    if tier not in TRUST_TIERS:
        errors.append("trust_tier must be A, B, or C")
    assets = spec.get("assets", [])
    if not isinstance(assets, list):
        return errors + ["assets must be a list"]
    for i, asset in enumerate(assets):
        method = asset.get("method")
        kind = asset.get("kind", "illustration")
        if method not in ASSET_METHODS:
            errors.append(f"assets[{i}].method invalid: {method}")
        if method == "generative" and kind in FORBIDDEN_GENERATIVE_KINDS:
            errors.append(f"assets[{i}] uses generative imagery for knowledge-bearing kind {kind}")
        if method == "trusted_source" and not asset.get("source"):
            errors.append(f"assets[{i}].source is required for trusted_source")
        if asset.get("contains_text") and method == "generative":
            errors.append(f"assets[{i}] must not bake instructional text into generated imagery")
        if asset.get("exact_count") is not None and method == "generative":
            errors.append(f"assets[{i}] exact counts must be deterministic, not generative")
    if tier == "C":
        fm = spec.get("fact_model")
        if not isinstance(fm, dict):
            errors.append("Tier C artifacts require fact_model")
        else:
            errors.extend(validate_fact_model(fm))
        for i, asset in enumerate(assets):
            if asset.get("knowledge_bearing") and asset.get("method") not in {"trusted_source", "deterministic"}:
                errors.append(f"Tier C knowledge-bearing assets[{i}] must be trusted_source or deterministic")
    return errors


def trust_summary(spec: dict[str, Any]) -> dict[str, Any]:
    errors = validate_artifact_trust(spec)
    return {
        "trust_tier": spec.get("trust_tier"),
        "verified": not errors,
        "checks": {
            "fact_model_present": bool(spec.get("fact_model")),
            "provenance_present": any(a.get("source") for a in spec.get("assets", []) if isinstance(a, dict)),
            "generative_text_absent": not any(
                a.get("method") == "generative" and a.get("contains_text")
                for a in spec.get("assets", [])
                if isinstance(a, dict)
            ),
        },
        "errors": errors,
    }
