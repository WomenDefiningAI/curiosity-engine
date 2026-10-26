#!/usr/bin/env python3
"""Regenerate the immutable manifest embedded in Curiosity Engine wheels."""

from __future__ import annotations

import json
from pathlib import Path

from curiosity_engine.personalization_policy import promotion_manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "release" / "release-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(promotion_manifest(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
