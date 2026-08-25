from __future__ import annotations

import re
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

PRIVATE_PROMPT_MARKERS = {
    "my child",
    "my daughter",
    "my son",
    "our family",
    "slack",
    "private resource",
    "purchased resource",
}

PRIVATE_DECORATIVE_TERMS = {
    "address",
    "classmate",
    "dad",
    "daughter",
    "doctor",
    "email",
    "family",
    "father",
    "friend",
    "home",
    "hospital",
    "mom",
    "mother",
    "parent",
    "school",
    "sister",
    "son",
    "street",
    "teacher",
    "therapy",
    "therapist",
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


def validate_response_visual_intent(intent: dict[str, Any]) -> list[str]:
    """Fail closed for response visuals that exceed the v0.1 educational trust boundary."""

    errors: list[str] = []
    kind = str(intent.get("kind") or "")
    role = str(intent.get("knowledge_role") or "")
    panels = intent.get("panels") or []
    source_refs = intent.get("source_refs") or []
    if role == "instructional":
        errors.append("instructional response visuals require the future Tier C pipeline")
    if kind == "comparison_cards" and not intent.get("not_to_scale"):
        errors.append("comparison cards must be labeled not to scale")
    if kind in {"comparison_cards", "activity_sequence"}:
        panel_text = " ".join(
            str(value)
            for panel in panels
            if isinstance(panel, dict)
            for value in (panel.get("label", ""), panel.get("detail", ""))
        )
        if any(character.isdigit() for character in panel_text):
            errors.append("MVP deterministic response cards may not teach exact numeric values")
    if kind == "decorative_illustration":
        if role != "decorative":
            errors.append("generated illustrations must be decorative")
        if panels:
            errors.append("generated illustrations may not contain instructional panels")
        if source_refs:
            errors.append("purchased or public source imagery is not copied into generated illustrations")
        subject = str(intent.get("subject") or "")
        lowered = subject.casefold()
        if not subject.strip():
            errors.append("decorative subject is required")
        if any(marker in lowered for marker in PRIVATE_PROMPT_MARKERS):
            errors.append("decorative subject may not contain family or private-resource context")
        if any(marker in subject for marker in ("<@", "@", "/Users/", "private/", "http://", "https://")):
            errors.append("decorative subject contains an identifier, path, or URL")
        if subject != lowered:
            errors.append("decorative subject must use lowercase generic nouns without proper names")
        words = set(re.findall(r"[a-z]+", lowered))
        if words & PRIVATE_DECORATIVE_TERMS or words & {"my", "our", "your", "his", "her", "their"}:
            errors.append("decorative subject contains personal relationship, school, home, or health context")
        if any(character.isdigit() for character in subject):
            errors.append("decorative subject may not contain numbers or identifiers")
    for source in source_refs:
        if not str(source).startswith(("https://", "http://")):
            errors.append("visual source references must be public HTTP(S) URLs")
    return errors
