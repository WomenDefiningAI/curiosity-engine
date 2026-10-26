from __future__ import annotations

import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Small dependency-free canonical JSON helper for policy bootstrap code."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
