#!/usr/bin/env python3
"""Keep the trusted-parent beta gate small and centered on family value."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

MAX_DEFAULT_TESTS = 293


def _case_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload) if isinstance(payload, list) else 1


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if collected.returncode:
        sys.stderr.write(collected.stdout)
        sys.stderr.write(collected.stderr)
        return collected.returncode
    counts = [
        int(match.group(1))
        for match in re.finditer(r"^tests/test_[^:]+\.py:\s+(\d+)\s*$", collected.stdout, re.MULTILINE)
    ]
    total = sum(counts)
    if not counts or total > MAX_DEFAULT_TESTS:
        print(f"default pytest budget exceeded: {total} collected; limit is {MAX_DEFAULT_TESTS}")
        return 1

    safety = _case_count(root / "evals" / "safety" / "child-material.json")
    value = _case_count(root / "evals" / "family_usefulness" / "core.json") + _case_count(
        root / "evals" / "performance" / "core.json"
    )
    if value <= safety:
        print(f"family usefulness/performance cases ({value}) must exceed focused safety cases ({safety})")
        return 1
    print(f"beta test budget: {total}/{MAX_DEFAULT_TESTS}; value cases: {value}; safety cases: {safety}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
