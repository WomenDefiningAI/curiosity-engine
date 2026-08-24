from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .context_builder import build_context
from .contracts import DirectorCandidate
from .db import connect, init_db, jdump, jload, utcnow
from .reasoning import POLICIES, ReasoningEngine


class AutonomousDirector:
    """A bounded weekly recommender: at most one suggestion, and doing nothing is valid."""

    def __init__(self, db_path: str | Path, reasoning: ReasoningEngine | None = None):
        self.db_path = str(db_path)
        self.reasoning = reasoning or ReasoningEngine()
        init_db(self.db_path)

    def _proactive_enabled(self) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT proactive_enabled FROM household_settings WHERE id='default'"
            ).fetchone()
        return bool(row and row["proactive_enabled"])

    def ensure_weekly_schedule(
        self,
        child_id: str,
        *,
        start_at: datetime | None = None,
        enabled: bool | None = None,
    ) -> str:
        schedule_id = f"weekly:{child_id}"
        next_run = (start_at or datetime.now(UTC)).isoformat()
        active = self._proactive_enabled() if enabled is None else enabled
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO schedules(id,schedule_type,child_id,cadence,next_run_at,enabled,payload_json)
                   VALUES(?,?,?,?,?,?,'{}') ON CONFLICT(id) DO UPDATE SET enabled=excluded.enabled""",
                (schedule_id, "weekly_reflection", child_id, "P7D", next_run, int(active)),
            )
        return schedule_id

    def run_due(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        current = now or datetime.now(UTC)
        with connect(self.db_path) as conn:
            due = conn.execute(
                """SELECT s.id,s.child_id FROM schedules s
                   JOIN household_settings h ON h.id='default' AND h.proactive_enabled=1
                   WHERE s.enabled=1 AND s.schedule_type='weekly_reflection'
                   AND s.next_run_at<=? ORDER BY s.next_run_at""",
                (current.isoformat(),),
            ).fetchall()
        return [self.reflect_for_child(row["child_id"], schedule_id=row["id"], now=current) for row in due]

    def reflect_for_child(
        self,
        child_id: str,
        *,
        schedule_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or datetime.now(UTC)
        cooldown = (current - timedelta(days=6)).isoformat()
        with connect(self.db_path) as conn:
            recent = conn.execute(
                """SELECT * FROM opportunities WHERE kind!='do_nothing'
                   AND created_at>=? AND status IN ('suggested','accepted') ORDER BY created_at DESC LIMIT 1""",
                (cooldown,),
            ).fetchone()
        if recent:
            choice = DirectorCandidate(
                kind="do_nothing",
                title="No extra suggestion this week",
                rationale="A family opportunity is already active; avoid piling on parent work.",
                parent_effort="very_low",
                priority=0.05,
                payload={"active_opportunity_id": recent["id"]},
            )
            considered: list[dict[str, Any]] = []
        else:
            policy = POLICIES["weekly_reflection"]
            context = build_context(
                self.db_path,
                child_id,
                {"type": "scheduled_reflection", "text": "weekly reflection", "metadata": {}},
                depth=policy.context_depth,
            )
            envelope = self.reasoning.run(
                policy=policy,
                context=context,
                event={
                    "type": "scheduled_reflection",
                    "child_id": child_id,
                    "text": "Find at most one unusually timely, low-effort learning opportunity; doing nothing is valid.",
                },
            )
            choice = DirectorCandidate.model_validate(envelope.output["choice"])
            considered = envelope.output.get("considered", [])
        if choice.parent_effort not in {"very_low", "low"}:
            choice = DirectorCandidate(
                kind="do_nothing",
                title="No low-effort opportunity this week",
                rationale="The candidate required more parent work than the weekly autonomy boundary allows.",
                parent_effort="very_low",
                priority=0.05,
                payload={},
            )
        week = current.date().isocalendar()
        dedupe = sha256(f"{child_id}:{week.year}:{week.week}:{choice.kind}:{choice.title}".encode()).hexdigest()
        opportunity_id = f"opp_{uuid4().hex[:16]}"
        enabled = int(self._proactive_enabled())
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO opportunities(id,child_id,kind,title,rationale,priority,parent_effort,payload_json,
                   status,dedupe_key,created_at,expires_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(child_id,dedupe_key) DO NOTHING""",
                (
                    opportunity_id,
                    child_id,
                    choice.kind,
                    choice.title,
                    choice.rationale,
                    choice.priority,
                    choice.parent_effort,
                    jdump(choice.payload),
                    "suggested" if choice.kind != "do_nothing" else "no_action",
                    dedupe,
                    current.isoformat(),
                    (current + timedelta(days=7)).isoformat(),
                ),
            )
            stored = conn.execute(
                "SELECT id FROM opportunities WHERE child_id=? AND dedupe_key=?", (child_id, dedupe)
            ).fetchone()
            opportunity_id = stored["id"]
            target_schedule = schedule_id or f"weekly:{child_id}"
            conn.execute(
                """INSERT INTO schedules(id,schedule_type,child_id,cadence,next_run_at,last_run_at,enabled,payload_json)
                   VALUES(?,?,?,?,?,?,?,'{}') ON CONFLICT(id) DO UPDATE SET last_run_at=excluded.last_run_at,
                   next_run_at=excluded.next_run_at,enabled=excluded.enabled""",
                (
                    target_schedule,
                    "weekly_reflection",
                    child_id,
                    "P7D",
                    (current + timedelta(days=7)).isoformat(),
                    current.isoformat(),
                    enabled,
                ),
            )
        return {
            "opportunity_id": opportunity_id,
            "choice": choice.model_dump(mode="json"),
            "considered": considered,
            "autonomy": "recommendation_only",
            "next_reflection_at": (current + timedelta(days=7)).isoformat(),
        }


def list_opportunities(db_path: str | Path, child_id: str, *, active_only: bool = True) -> list[dict[str, Any]]:
    clause = " AND status='suggested' AND (expires_at IS NULL OR expires_at>?)" if active_only else ""
    params: tuple[Any, ...] = (child_id, utcnow()) if active_only else (child_id,)
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM opportunities WHERE child_id=?" + clause + " ORDER BY created_at DESC",
            params,
        ).fetchall()
    return [{**dict(row), "payload": jload(row["payload_json"])} for row in rows]
