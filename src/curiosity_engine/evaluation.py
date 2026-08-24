from __future__ import annotations

from pydantic import ValidationError

from .contracts import PullThreadOutput


def validate_pull_thread(payload: dict) -> list[str]:
    candidate = {key: value for key, value in payload.items() if not key.startswith("_")}
    try:
        PullThreadOutput.model_validate(candidate)
    except ValidationError as exc:
        return [
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors(include_url=False)
        ]
    return []
