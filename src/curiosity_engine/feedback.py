from __future__ import annotations

from pathlib import Path

from .contracts import FeedbackInput
from .db import connect, init_db, utcnow


def record_feedback(db_path: str | Path, feedback: FeedbackInput) -> int:
    """Record parent outcome data as an observation, never as an automatic trait."""

    init_db(db_path)
    with connect(db_path) as conn:
        if feedback.experience_id:
            experience = conn.execute(
                "SELECT child_id FROM experiences WHERE id=?", (feedback.experience_id,)
            ).fetchone()
            if not experience or experience["child_id"] != feedback.child_id:
                raise ValueError("experience does not belong to child")
        if feedback.artifact_id:
            artifact = conn.execute("SELECT child_id FROM artifacts WHERE id=?", (feedback.artifact_id,)).fetchone()
            if not artifact or artifact["child_id"] != feedback.child_id:
                raise ValueError("artifact does not belong to child")
        now = utcnow()
        cur = conn.execute(
            """INSERT INTO feedback(child_id,experience_id,artifact_id,outcome,note,source,created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                feedback.child_id,
                feedback.experience_id,
                feedback.artifact_id,
                feedback.outcome,
                feedback.note,
                feedback.source,
                now,
            ),
        )
        if feedback.experience_id:
            status = "completed" if feedback.outcome not in {"not_used"} else "not_used"
            conn.execute(
                "UPDATE experiences SET status=?,completed_at=?,feedback=? WHERE id=?",
                (
                    status,
                    now if status == "completed" else None,
                    feedback.note or feedback.outcome,
                    feedback.experience_id,
                ),
            )
        conn.execute(
            """INSERT INTO observations(child_id,kind,text,source,confidence,occurred_at,metadata_json)
               VALUES(?,?,?,?,?,?,?)""",
            (
                feedback.child_id,
                "feedback",
                feedback.note or feedback.outcome,
                feedback.source,
                1.0,
                now,
                '{"epistemic_state":"observation"}',
            ),
        )
        return int(cur.lastrowid)
