from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from .contracts import InteractionOption, InteractionPlan, ScheduleProposal
from .db import connect, init_db, jdump, jload, utcnow
from .interaction import enqueue_delivery
from .interactions import create_interaction, interaction_blocks
from .sessions import SessionStore
from .transports.contracts import OutboundMessage

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def _parse_weekday(text: str) -> str:
    lowered = text.casefold()
    return next((name for name in WEEKDAYS if name in lowered), "sunday")


def _parse_time(text: str) -> str:
    matches = list(re.finditer(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*(am|pm)\b", text, re.IGNORECASE))
    if matches:
        match = matches[-1]
        hour = int(match.group(1)) % 12
        if match.group(3).casefold() == "pm":
            hour += 12
        return f"{hour:02d}:{int(match.group(2) or 0):02d}"
    matches = list(re.finditer(r"\b((?:[01]\d|2[0-3])):([0-5]\d)\b", text))
    if matches:
        match = matches[-1]
        return f"{int(match.group(1)):02d}:{int(match.group(2)):02d}"
    matches = list(
        re.finditer(
            r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s+(?:in\s+the\s+)?(morning|afternoon|evening)\b",
            text,
            re.IGNORECASE,
        )
    )
    if not matches:
        return "18:00"
    match = matches[-1]
    hour = int(match.group(1)) % 12
    if match.group(3).casefold() in {"afternoon", "evening"}:
        hour += 12
    return f"{hour:02d}:{int(match.group(2) or 0):02d}"


def checkin_message_from_request(text: str, child_name: str = "your child") -> str:
    """Build one bounded visible check-in message from the parent's wording."""

    compact = " ".join(text.split()).strip()
    quoted = re.findall(r"[“\"]([^”\"]{8,500}[?.!])[”\"]", compact)
    if quoted:
        candidate = quoted[-1].strip()
        if not re.search(r"nothing recurring|not scheduled|draft", candidate, re.IGNORECASE):
            return candidate
    lowered = compact.casefold()
    if re.search(r"\b(?:one|an?)\s+(?:easy|simple|quick)?\s*drawing idea\b", lowered):
        return (
            f"Drawing idea for {child_name}: draw one interesting thing you noticed this week from a surprising "
            "point of view. What will you choose?"
        )
    ask_match = re.search(r"\bask me (?:whether|if)\s+(.+?)(?:[.!?]|$)", compact, re.IGNORECASE)
    if ask_match:
        requested = ask_match.group(1).strip(" ,.;:")
        if re.match(r"^i want\b", requested, re.IGNORECASE):
            question = re.sub(r"^i want\b", "Do you want", requested, flags=re.IGNORECASE)
        elif re.match(r"^i would like\b", requested, re.IGNORECASE):
            question = re.sub(r"^i would like\b", "Would you like", requested, flags=re.IGNORECASE)
        else:
            question = re.sub(r"^i\s+", "Do you ", requested, flags=re.IGNORECASE)
            question = question[0].upper() + question[1:]
        return question + ("" if question.endswith("?") else "?")
    ask_for_match = re.search(
        r"\b(?:where\s+you\s+)?ask me for\s+(.+?)(?:[.!?]|$)",
        compact,
        re.IGNORECASE,
    )
    if ask_for_match:
        requested = ask_for_match.group(1).strip(" ,.;:")
        requested = re.sub(r"^(?:one|a)\s+(?:small|quick|simple)\s+", "one ", requested, flags=re.IGNORECASE)
        return f"What is {requested} from this week?"
    asking_match = re.search(
        r"\basking\s+(.+?)(?:\s+every\s+week|\s+that\s+week|[.!?]|$)",
        compact,
        re.IGNORECASE,
    )
    if asking_match:
        requested = asking_match.group(1).strip(" ,.;:")
        if requested.casefold().startswith("what "):
            return requested[0].upper() + requested[1:] + ("" if requested.endswith("?") else "?")
    child_topic_match = re.search(
        r"\bcheck[- ]?in\s+with\s+[A-Z][A-Za-z'’-]{1,40}\s+about\s+(.+?)(?:[.!?]|$)",
        compact,
        re.IGNORECASE,
    )
    if child_topic_match:
        topic = child_topic_match.group(1).strip(" ,.;:")
        if re.search(r"\bwhat\s+(?:she|he|they)\s+wants?\s+help\s+with\s+at\s+school\b", topic, re.IGNORECASE):
            return "What would you like help with at school this week?"
        topic = re.sub(r"\b(?:she|he|they)\b", "you", topic, flags=re.IGNORECASE)
        return f"What would you like to share about {topic}?"
    topic_match = re.search(
        r"\babout\s+(.+?)(?:,?\s+with\s+(?:just\s+)?one\s+question|[.!?]|$)",
        compact,
        re.IGNORECASE,
    )
    if topic_match:
        topic = topic_match.group(1).strip(" ,.;:")
        if re.match(r"noticing\s+", topic, re.IGNORECASE):
            subject = re.sub(r"^noticing\s+", "", topic, flags=re.IGNORECASE)
            return f"What {subject} can you notice today?"
        return f"What did {child_name} notice about {topic} this week?"
    return f"What did {child_name} notice or wonder about this week?"


def weekly_checkin_preview(text: str, child_name: str = "your child") -> str:
    """Render the exact pending cadence and message in parent-friendly form."""

    weekday = _parse_weekday(text).title()
    local_time = _parse_time(text)
    hour, minute = (int(value) for value in local_time.split(":"))
    suffix = "a.m." if hour < 12 else "p.m."
    display_hour = hour % 12 or 12
    display_time = f"{display_hour}:{minute:02d} {suffix}"
    message = checkin_message_from_request(text, child_name)
    boundary_notes: list[str] = []
    lowered = text.casefold()
    if re.search(r"\b(?:to me only|parent[- ]facing|reminder to me|not (?:a message )?to (?:her|him|them))\b", lowered):
        boundary_notes.append("This reminder is for the parent, not the child.")
    if re.search(r"\b(?:not|don't|do not)\b.{0,35}\b(?:report card|assessment|grade|mastery)\b", lowered):
        boundary_notes.append("Anything shared stays family context, not a report card or mastery claim.")
    boundary = (" " + " ".join(boundary_notes)) if boundary_notes else ""
    return (
        f"Draft: {weekday} at {display_time}: “{message}” "
        f"Nothing recurring will be scheduled until you explicitly approve it.{boundary}"
    )


def _next_occurrence(weekday: str, local_time: str, timezone: str, *, after: datetime | None = None) -> datetime:
    zone = ZoneInfo(timezone)
    current = (after or datetime.now(UTC)).astimezone(zone)
    hour, minute = (int(value) for value in local_time.split(":"))
    days = (WEEKDAYS[weekday] - current.weekday()) % 7
    candidate = current.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=days)
    if candidate <= current:
        candidate += timedelta(days=7)
    return candidate.astimezone(UTC)


class SchedulerService:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        init_db(self.db_path)
        self.sessions = SessionStore(self.db_path)

    def proposal_from_request(
        self,
        *,
        request: str,
        binding_id: str,
        channel_id: str,
        child_id: str | None = None,
    ) -> ScheduleProposal:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT timezone FROM household_settings WHERE id='default'").fetchone()
            child_row = (
                conn.execute("SELECT name FROM children WHERE id=?", (child_id,)).fetchone()
                if child_id
                else None
            )
        timezone = str(row["timezone"] if row else "UTC")
        child_name = str(child_row["name"] if child_row else "your child").split()[0]
        return ScheduleProposal(
            child_id=child_id,
            weekday=_parse_weekday(request),
            local_time=_parse_time(request),
            timezone=timezone,
            channel_id=channel_id,
            binding_id=binding_id,
            checkin_message=checkin_message_from_request(request, child_name),
        )

    def create_weekly_checkin(self, proposal: ScheduleProposal | dict[str, Any]) -> dict[str, Any]:
        parsed = proposal if isinstance(proposal, ScheduleProposal) else ScheduleProposal.model_validate(proposal)
        schedule_key = sha256(
            f"{parsed.binding_id}:{parsed.child_id or 'family'}:{parsed.weekday}:{parsed.local_time}".encode()
        ).hexdigest()[:18]
        schedule_id = f"checkin:{schedule_key}"
        next_run = _next_occurrence(parsed.weekday, parsed.local_time, parsed.timezone)
        payload = parsed.model_dump(mode="json")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO schedules(id,schedule_type,child_id,cadence,next_run_at,enabled,payload_json)
                   VALUES(?,?,?,?,?,1,?) ON CONFLICT(id) DO UPDATE SET cadence=excluded.cadence,
                   next_run_at=excluded.next_run_at,enabled=1,payload_json=excluded.payload_json""",
                (
                    schedule_id,
                    "weekly_parent_checkin",
                    parsed.child_id,
                    f"weekly:{parsed.weekday}:{parsed.local_time}",
                    next_run.isoformat(),
                    jdump(payload),
                ),
            )
            conn.execute(
                "UPDATE household_settings SET weekly_checkins_enabled=1,updated_at=? WHERE id='default'",
                (utcnow(),),
            )
        return {
            "status": "scheduled",
            "schedule_id": schedule_id,
            "workflow": parsed.workflow,
            "weekday": parsed.weekday,
            "local_time": parsed.local_time,
            "timezone": parsed.timezone,
            "next_run_at": next_run.isoformat(),
        }

    def pause(self, schedule_id: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            cursor = conn.execute("UPDATE schedules SET enabled=0 WHERE id=?", (schedule_id,))
        if cursor.rowcount != 1:
            raise KeyError(schedule_id)
        return {"status": "paused", "schedule_id": schedule_id}

    def run_due(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        current = now or datetime.now(UTC)
        with connect(self.db_path) as conn:
            due = conn.execute(
                """SELECT * FROM schedules WHERE schedule_type='weekly_parent_checkin' AND enabled=1
                   AND next_run_at<=? ORDER BY next_run_at""",
                (current.isoformat(),),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for schedule in due:
            result = self._materialize(schedule, current)
            if result:
                results.append(result)
        return results

    def _materialize(self, schedule: Any, current: datetime) -> dict[str, Any] | None:
        payload = jload(schedule["payload_json"])
        proposal = ScheduleProposal.model_validate(payload)
        due_at = str(schedule["next_run_at"])
        run_id = f"srun_{uuid4().hex[:20]}"
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT id,status FROM schedule_runs WHERE schedule_id=? AND due_at=?",
                (schedule["id"], due_at),
            ).fetchone()
            if existing:
                return None
            conn.execute(
                """INSERT INTO schedule_runs(id,schedule_id,due_at,status,created_at,updated_at)
                   VALUES(?,?,?,'running',?,?)""",
                (run_id, schedule["id"], due_at, now, now),
            )
            next_run = _next_occurrence(
                proposal.weekday,
                proposal.local_time,
                proposal.timezone,
                after=current + timedelta(minutes=1),
            )
            conn.execute(
                "UPDATE schedules SET last_run_at=?,next_run_at=? WHERE id=?",
                (current.isoformat(), next_run.isoformat(), schedule["id"]),
            )
        with connect(self.db_path) as conn:
            binding = conn.execute("SELECT * FROM transport_bindings WHERE id=? AND status='active'", (proposal.binding_id,)).fetchone()
        if not binding:
            with connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE schedule_runs SET status='failed',result_json=?,updated_at=? WHERE id=?",
                    (jdump({"reason": "binding unavailable"}), utcnow(), run_id),
                )
            return {"schedule_run_id": run_id, "status": "failed"}
        conversation_ref = sha256(f"{binding['team_id']}:{binding['channel_id']}".encode()).hexdigest()[:20]
        session = self.sessions.get_or_create(
            origin="schedule",
            transport="slack",
            binding_id=proposal.binding_id,
            conversation_ref=conversation_ref,
            thread_ref=str(schedule["id"]),
            child_id=proposal.child_id,
            capability_id="weekly_parent_checkin",
        )
        plan = InteractionPlan(
            kind="choose_one",
            title="Weekly curiosity check-in",
            prompt=(
                "What stood out this week? Choose a shortcut or reply naturally in this thread. "
                "This check-in is not treated as proof of interest or mastery."
            ),
            options=[
                InteractionOption(label="Something clicked", intent="weekly_feedback", payload={"outcome": "engaged", "schedule_id": schedule["id"]}),
                InteractionOption(label="We got stuck", intent="weekly_feedback", payload={"outcome": "not_helpful", "schedule_id": schedule["id"]}),
                InteractionOption(label="One easy next idea", intent="weekly_next_thread", payload={"schedule_id": schedule["id"]}),
                InteractionOption(label="Pause check-ins", intent="pause_schedule", payload={"schedule_id": schedule["id"]}),
            ],
        )
        presented = create_interaction(
            self.db_path,
            binding_id=proposal.binding_id,
            session_id=str(session["id"]),
            plan=plan,
        )
        text = proposal.checkin_message
        delivery_id = enqueue_delivery(
            self.db_path,
            proposal.binding_id,
            OutboundMessage(
                channel_id=proposal.channel_id,
                text=text,
                blocks=interaction_blocks(presented),
            ),
            idempotency_key=f"schedule:{schedule['id']}:{due_at}",
        )
        self.sessions.append_message(
            str(session["id"]),
            role="assistant",
            content=text,
            kind="weekly_checkin",
            metadata={"schedule_id": schedule["id"], "schedule_run_id": run_id, "delivery_id": delivery_id},
        )
        result = {"schedule_run_id": run_id, "status": "completed", "delivery_id": delivery_id, "session_id": session["id"]}
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE schedule_runs SET status='completed',session_id=?,result_json=?,updated_at=? WHERE id=?",
                (session["id"], jdump(result), utcnow(), run_id),
            )
        return result
