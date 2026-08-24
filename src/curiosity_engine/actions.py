from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactService
from .db import connect, jdump, jload, utcnow


def list_actions(db_path: str | Path, *, status: str | None = None) -> list[dict[str, Any]]:
    where = "WHERE a.status=?" if status else ""
    params = (status,) if status else ()
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""SELECT a.*,e.child_id FROM actions a LEFT JOIN events e ON e.id=a.event_id
                {where} ORDER BY a.created_at DESC""",
            params,
        ).fetchall()
    return [
        {
            **{key: row[key] for key in row.keys() if key not in {"payload_json", "result_json"}},
            "payload": jload(row["payload_json"]),
            "result": jload(row["result_json"]) if row["result_json"] else None,
        }
        for row in rows
    ]


def execute_action(
    db_path: str | Path,
    action_id: str,
    *,
    output_dir: str | Path,
    actor: str = "parent",
    visual_backend=None,
) -> dict[str, Any]:
    """Execute a parent-selected allowlisted action. Model output alone cannot call this."""

    if actor != "parent":
        raise PermissionError("MVP actions require a parent actor")
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT a.action_type,a.payload_json,a.status,a.result_json,e.child_id
               FROM actions a LEFT JOIN events e ON e.id=a.event_id WHERE a.id=?""",
            (action_id,),
        ).fetchone()
        if not row:
            raise KeyError(action_id)
        if row["status"] == "completed":
            return jload(row["result_json"])
        if row["status"] == "executing":
            raise RuntimeError("action is already executing")
        if row["action_type"] != "propose_artifact":
            raise ValueError(f"action type is not allowlisted: {row['action_type']}")
        if not row["child_id"]:
            raise ValueError("artifact action is missing a child")
        payload = jload(row["payload_json"])
        conn.execute("UPDATE actions SET status='executing',updated_at=? WHERE id=?", (utcnow(), action_id))
    try:
        spec = payload.get("spec") if isinstance(payload, dict) and "spec" in payload else payload
        result = ArtifactService(db_path, output_dir).create(
            child_id=row["child_id"], spec=spec, visual_backend=visual_backend
        )
    except Exception as exc:
        with connect(db_path) as conn:
            conn.execute(
                "UPDATE actions SET status='failed',updated_at=?,result_json=? WHERE id=?",
                (utcnow(), jdump({"error": repr(exc)}), action_id),
            )
        raise
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE actions SET status='completed',updated_at=?,result_json=? WHERE id=?",
            (utcnow(), jdump(result), action_id),
        )
    return result
