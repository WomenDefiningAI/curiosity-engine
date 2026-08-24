from __future__ import annotations

from typing import Any

ROLE_REQUIREMENTS: dict[str, set[str]] = {
    "reasoning": {"structured_output"},
    "structured_extraction": {"structured_output", "vision", "document_ocr"},
    "visual_qa": {"structured_output", "vision"},
    "image_generation": {"image_generation"},
    "critic_factual": {"structured_output"},
    "critic_pedagogy": {"structured_output"},
    "critic_context": {"structured_output"},
    "critic_parent_effort": {"structured_output"},
    "critic_epistemic": {"structured_output"},
    "judge": {"structured_output"},
}


def capability_gaps(role: str, route: dict[str, Any]) -> list[str]:
    required = ROLE_REQUIREMENTS.get(role, {"structured_output"})
    declared = {str(item) for item in route.get("capabilities") or []}
    return sorted(required - declared)


def model_stack_requirements() -> list[dict[str, Any]]:
    """Stable product requirements, deliberately independent of volatile model names."""

    return [
        {
            "role": "reasoning",
            "required": ["structured_output"],
            "evaluation": ["elementary factuality", "age calibration", "curiosity preservation", "parent effort"],
        },
        {
            "role": "structured_extraction",
            "required": ["structured_output", "vision", "document_ocr"],
            "evaluation": ["scanned worksheet OCR", "handwriting tolerance", "layout and table comprehension"],
        },
        {
            "role": "visual_qa",
            "required": ["structured_output", "vision"],
            "evaluation": ["small text", "clipping", "misleading diagrams", "print legibility"],
        },
        {
            "role": "image_generation",
            "required": ["image_generation"],
            "evaluation": ["child-appropriate illustrations", "instruction following", "editing", "visual consistency"],
            "note": "Generated imagery is decorative; code renders all instructional text and knowledge-bearing geometry.",
        },
    ]
