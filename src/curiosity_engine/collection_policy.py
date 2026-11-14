from __future__ import annotations

from pathlib import Path
from typing import Any

from .db import connect


class CollectionPausedError(ValueError):
    """A learner/profile write was rejected by the parent's collection control."""


def require_collection_enabled_in_transaction(conn: Any, child_id: str) -> None:
    control = conn.execute(
        "SELECT collection_enabled FROM learner_model_controls WHERE child_id=?",
        (child_id,),
    ).fetchone()
    if control is not None and not bool(control["collection_enabled"]):
        raise CollectionPausedError("learner-model collection is paused for this child")


def require_collection_enabled(db_path: str | Path, child_id: str) -> None:
    with connect(db_path) as conn:
        require_collection_enabled_in_transaction(conn, child_id)
