from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_reasoning_policy(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_reasoning_policy(policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    workflows = policy.get("workflows")
    if not isinstance(workflows, dict) or not workflows:
        return ["workflows must be a non-empty mapping"]
    for name, cfg in workflows.items():
        depth = cfg.get("context_depth")
        if not isinstance(depth, int) or depth < 0 or depth > 4:
            errors.append(f"{name}.context_depth must be 0..4")
        if cfg.get("budget") not in {"fast", "normal", "deep", "lab"}:
            errors.append(f"{name}.budget must be fast|normal|deep|lab")
        if not cfg.get("generator_role"):
            errors.append(f"{name}.generator_role required")
    return errors
