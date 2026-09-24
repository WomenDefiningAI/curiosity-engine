"""One metered, text-only TypeSafe request. No SDK, retries, or fallback judge."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

from .model_usage import metered_call

MODEL = "jev-1.13.0"


def load_key(path: Path) -> str:
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("TypeSafe key file must exist and be owner-only")
    for line in path.read_text().splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "TYPESAFE_API_KEY" and value.strip().strip("\"'"):
            return value.strip().strip("\"'")
    raise ValueError("TYPESAFE_API_KEY is not configured")


def evaluate(state, questions, *, key: str):
    request = {"model": MODEL, "state": state, "questions": questions}
    data = json.dumps(request).encode()
    # Byte bound fits the per-question 32k token context without dropping turns.
    if len(data) > 32_000:
        raise ValueError("Conversation exceeds the text evaluation limit")

    def invoke(_budgeted):
        req = Request("https://api.typesafe.ai/v1/systemone", data=data,
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urlopen(req, timeout=15) as response:
            payload = json.load(response)
        return SimpleNamespace(usage=payload.get("usage"), payload=payload)

    return metered_call(request, invoke, role="text_judge").payload
