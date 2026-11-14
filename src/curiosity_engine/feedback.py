from __future__ import annotations

from pathlib import Path

from .contracts import FeedbackInput
from .db import connect, init_db, jdump, utcnow
from .personalization import record_feedback_signal


def record_feedback(db_path: str | Path, feedback: FeedbackInput) -> int:
    """Record parent outcome data for audit, outside the model-facing child profile."""

    init_db(db_path)
    with connect(db_path) as conn:
        plan_bindings: set[tuple[str, str]] = set()
        if feedback.event_id:
            event = conn.execute(
                """SELECT e.child_id,r.pedagogy_plan_id,r.pedagogy_plan_hash FROM events e
                   JOIN responses r ON r.event_id=e.id
                   WHERE e.id=? AND r.status='completed'""",
                (feedback.event_id,),
            ).fetchone()
            if not event or event["child_id"] != feedback.child_id:
                raise ValueError("response does not belong to child")
            if event["pedagogy_plan_id"]:
                plan_bindings.add((str(event["pedagogy_plan_id"]), str(event["pedagogy_plan_hash"])))
        if feedback.experience_id:
            experience = conn.execute(
                "SELECT child_id,pedagogy_plan_id,pedagogy_plan_hash FROM experiences WHERE id=?",
                (feedback.experience_id,),
            ).fetchone()
            if not experience or experience["child_id"] != feedback.child_id:
                raise ValueError("experience does not belong to child")
            if experience["pedagogy_plan_id"]:
                plan_bindings.add((str(experience["pedagogy_plan_id"]), str(experience["pedagogy_plan_hash"])))
        if feedback.artifact_id:
            artifact = conn.execute(
                "SELECT child_id,pedagogy_plan_id,pedagogy_plan_hash FROM artifacts WHERE id=?",
                (feedback.artifact_id,),
            ).fetchone()
            if not artifact or artifact["child_id"] != feedback.child_id:
                raise ValueError("artifact does not belong to child")
            if artifact["pedagogy_plan_id"]:
                plan_bindings.add((str(artifact["pedagogy_plan_id"]), str(artifact["pedagogy_plan_hash"])))
        if len(plan_bindings) > 1:
            raise ValueError("feedback sources do not share one pedagogy plan")
        plan_id, plan_hash = next(iter(plan_bindings), (None, None))
        now = utcnow()
        cur = conn.execute(
            """INSERT INTO feedback(
               child_id,event_id,experience_id,artifact_id,outcome,note,source,created_at,
               pedagogy_plan_id,pedagogy_plan_hash
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                feedback.child_id,
                feedback.event_id,
                feedback.experience_id,
                feedback.artifact_id,
                feedback.outcome,
                feedback.note,
                feedback.source,
                now,
                plan_id,
                plan_hash,
            ),
        )
        feedback_id = int(cur.lastrowid)
        record_feedback_signal(
            conn,
            feedback_id=feedback_id,
            feedback=feedback,
            plan_id=plan_id,
            plan_hash=plan_hash,
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
                jdump(
                    {
                        "epistemic_state": "observation",
                        "learning_scope": "diagnostic",
                        "independence_status": "excluded",
                        "profile_eligible": False,
                        "record_role": "output_feedback",
                        **({"response_event_id": feedback.event_id} if feedback.event_id else {}),
                    }
                ),
            ),
        )
        return feedback_id
