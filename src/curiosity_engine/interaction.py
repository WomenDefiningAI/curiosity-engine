from __future__ import annotations

import re
import secrets
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .db import connect, init_db, jdump, jload, utcnow
from .transports.contracts import InboundMessage, OutboundMessage

PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
RESOURCE_CONTEXT_MODES = {"metadata_only", "selected_excerpts"}


class TransportConflict(ValueError):
    pass


def _validate_clock(value: str | None) -> str | None:
    if value is None:
        return None
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("quiet hours must use 24-hour HH:MM")
    return value


def _validate_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown IANA timezone: {value}") from exc
    return value


def setup_household(
    db_path: str | Path,
    *,
    owner_name: str,
    timezone: str,
    quiet_start: str | None = "20:00",
    quiet_end: str | None = "07:00",
    proactive_enabled: bool = False,
    resource_context_mode: str = "metadata_only",
) -> dict[str, Any]:
    init_db(db_path)
    display_name = owner_name.strip()
    if not display_name:
        raise ValueError("owner name is required")
    timezone = _validate_timezone(timezone)
    quiet_start = _validate_clock(quiet_start)
    quiet_end = _validate_clock(quiet_end)
    if resource_context_mode not in RESOURCE_CONTEXT_MODES:
        raise ValueError("resource context mode must be metadata_only or selected_excerpts")
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        owner = conn.execute(
            "SELECT id,display_name FROM parent_principals WHERE role='owner' AND status='active'"
        ).fetchone()
        if owner:
            owner_id = owner["id"]
            if owner["display_name"] != display_name:
                conn.execute(
                    "UPDATE parent_principals SET display_name=?,updated_at=? WHERE id=?",
                    (display_name, now, owner_id),
                )
        else:
            owner_id = f"parent_{uuid4().hex[:16]}"
            conn.execute(
                "INSERT INTO parent_principals(id,display_name,role,status,created_at,updated_at) VALUES(?,?,'owner','active',?,?)",
                (owner_id, display_name, now, now),
            )
        conn.execute(
            """INSERT INTO household_settings(id,timezone,quiet_start,quiet_end,proactive_enabled,
                                               weekly_suggestion_limit,resource_context_mode,created_at,updated_at)
               VALUES('default',?,?,?,?,1,?,?,?)
               ON CONFLICT(id) DO UPDATE SET timezone=excluded.timezone,quiet_start=excluded.quiet_start,
                 quiet_end=excluded.quiet_end,proactive_enabled=excluded.proactive_enabled,
                 resource_context_mode=excluded.resource_context_mode,updated_at=excluded.updated_at""",
            (timezone, quiet_start, quiet_end, int(proactive_enabled), resource_context_mode, now, now),
        )
        conn.execute(
            "INSERT INTO interaction_audit(actor_parent_id,action,subject_type,subject_id,metadata_json,created_at) VALUES(?,?,'household','default',?,?)",
            (
                owner_id,
                "household_setup",
                jdump(
                    {
                        "timezone": timezone,
                        "proactive_enabled": proactive_enabled,
                        "resource_context_mode": resource_context_mode,
                    }
                ),
                now,
            ),
        )
        conn.execute("UPDATE schedules SET enabled=? WHERE schedule_type='weekly_reflection'", (int(proactive_enabled),))
    return {
        "status": "configured",
        "owner_id": owner_id,
        "timezone": timezone,
        "quiet_hours": {"start": quiet_start, "end": quiet_end},
        "proactive_enabled": proactive_enabled,
        "resource_context_mode": resource_context_mode,
        "weekly_suggestion_limit": 1,
    }


def household_resource_context_mode(db_path: str | Path) -> str:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT resource_context_mode FROM household_settings WHERE id='default'").fetchone()
    return str(row["resource_context_mode"]) if row else "metadata_only"


def set_household_resource_context_mode(db_path: str | Path, mode: str) -> dict[str, str]:
    if mode not in RESOURCE_CONTEXT_MODES:
        raise ValueError("resource context mode must be metadata_only or selected_excerpts")
    init_db(db_path)
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        settings = conn.execute("SELECT id FROM household_settings WHERE id='default'").fetchone()
        if not settings:
            raise ValueError("household setup is not configured")
        owner = conn.execute(
            "SELECT id FROM parent_principals WHERE role='owner' AND status='active'"
        ).fetchone()
        conn.execute(
            "UPDATE household_settings SET resource_context_mode=?,updated_at=? WHERE id='default'",
            (mode, now),
        )
        conn.execute(
            """INSERT INTO interaction_audit(
                   actor_parent_id,action,subject_type,subject_id,metadata_json,created_at
               ) VALUES(?,?,'household','default',?,?)""",
            (owner["id"] if owner else None, "resource_context_mode_changed", jdump({"mode": mode}), now),
        )
    return {"status": "configured", "resource_context_mode": mode}


def add_parent(db_path: str | Path, display_name: str) -> dict[str, Any]:
    init_db(db_path)
    name = display_name.strip()
    if not name:
        raise ValueError("parent display name is required")
    parent_id = f"parent_{uuid4().hex[:16]}"
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO parent_principals(id,display_name,role,status,created_at,updated_at) VALUES(?,?,'parent','active',?,?)",
            (parent_id, name, now, now),
        )
    return {"parent_id": parent_id, "display_name": name, "role": "parent", "status": "active"}


def onboarding_status(db_path: str | Path) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as conn:
        settings = conn.execute("SELECT * FROM household_settings WHERE id='default'").fetchone()
        parents = [
            dict(row)
            for row in conn.execute(
                "SELECT id,display_name,role,status FROM parent_principals ORDER BY role DESC,display_name"
            )
        ]
        bindings = [
            dict(row)
            for row in conn.execute(
                """SELECT b.id,b.transport,b.team_id,b.user_id,b.channel_id,b.parent_id,b.status
                   FROM transport_bindings b ORDER BY b.created_at"""
            )
        ]
        inbox_count = conn.execute("SELECT COUNT(*) FROM capture_inbox WHERE status='unassigned'").fetchone()[0]
    return {
        "configured": settings is not None and bool(parents),
        "settings": dict(settings) if settings else None,
        "parents": parents,
        "bindings": bindings,
        "unassigned_captures": inbox_count,
    }


def create_pairing_code(db_path: str | Path, parent_id: str, *, ttl_minutes: int = 15) -> dict[str, Any]:
    if not 1 <= ttl_minutes <= 60:
        raise ValueError("pairing code TTL must be between 1 and 60 minutes")
    init_db(db_path)
    code = "".join(secrets.choice(PAIRING_ALPHABET) for _ in range(8))
    code_hash = sha256(code.encode()).hexdigest()
    created = datetime.now(UTC)
    expires = created + timedelta(minutes=ttl_minutes)
    with connect(db_path) as conn:
        parent = conn.execute(
            "SELECT id FROM parent_principals WHERE id=? AND status='active'", (parent_id,)
        ).fetchone()
        if not parent:
            raise ValueError("active parent not found")
        conn.execute(
            "INSERT INTO transport_pairing_codes(code_hash,transport,parent_id,expires_at,created_at) VALUES(?,'slack',?,?,?)",
            (code_hash, parent_id, expires.isoformat(), created.isoformat()),
        )
    return {
        "transport": "slack",
        "parent_id": parent_id,
        "pairing_code": code,
        "expires_at": expires.isoformat(),
        "instruction": f"Send `pair {code}` to the Curiosity Engine Slack bot in a DM or approved family channel.",
    }


def consume_pairing_code(db_path: str | Path, code: str, message: InboundMessage) -> dict[str, Any]:
    code_hash = sha256(code.strip().upper().encode()).hexdigest()
    now = datetime.now(UTC)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT parent_id,expires_at,used_at FROM transport_pairing_codes WHERE code_hash=? AND transport=?",
            (code_hash, message.transport),
        ).fetchone()
        if not row or row["used_at"]:
            raise PermissionError("pairing code is invalid or already used")
        if datetime.fromisoformat(row["expires_at"]) <= now:
            raise PermissionError("pairing code expired; create a new one locally")
        existing = conn.execute(
            """SELECT id FROM transport_bindings
               WHERE transport=? AND team_id=? AND user_id=? AND channel_id=?""",
            (message.transport, message.team_id, message.user_id, message.channel_id),
        ).fetchone()
        binding_id = existing["id"] if existing else f"binding_{uuid4().hex[:16]}"
        if existing:
            conn.execute(
                "UPDATE transport_bindings SET parent_id=?,status='active',updated_at=? WHERE id=?",
                (row["parent_id"], now.isoformat(), binding_id),
            )
        else:
            conn.execute(
                """INSERT INTO transport_bindings(id,transport,team_id,user_id,channel_id,parent_id,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,'active',?,?)""",
                (
                    binding_id,
                    message.transport,
                    message.team_id,
                    message.user_id,
                    message.channel_id,
                    row["parent_id"],
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        conn.execute("UPDATE transport_pairing_codes SET used_at=? WHERE code_hash=?", (now.isoformat(), code_hash))
        conn.execute(
            "INSERT INTO interaction_audit(actor_parent_id,action,subject_type,subject_id,metadata_json,created_at) VALUES(?,?,'transport_binding',?, ?,?)",
            (
                row["parent_id"],
                "slack_paired",
                binding_id,
                jdump({"team_id": message.team_id, "channel_id": message.channel_id}),
                now.isoformat(),
            ),
        )
    return {"binding_id": binding_id, "parent_id": row["parent_id"], "status": "active"}


def active_binding(db_path: str | Path, message: InboundMessage) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT b.*,p.display_name,p.role FROM transport_bindings b
               JOIN parent_principals p ON p.id=b.parent_id
               WHERE b.transport=? AND b.team_id=? AND b.user_id=? AND b.channel_id=?
                 AND b.status='active' AND p.status='active'""",
            (message.transport, message.team_id, message.user_id, message.channel_id),
        ).fetchone()
    return dict(row) if row else None


def list_bindings(db_path: str | Path) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        return [
            dict(row)
            for row in conn.execute(
                """SELECT b.id,b.transport,b.team_id,b.user_id,b.channel_id,b.parent_id,b.status,b.created_at,
                          p.display_name,p.role
                   FROM transport_bindings b JOIN parent_principals p ON p.id=b.parent_id
                   ORDER BY b.created_at"""
            )
        ]


def revoke_binding(db_path: str | Path, binding_id: str) -> dict[str, Any]:
    now = utcnow()
    with connect(db_path) as conn:
        row = conn.execute("SELECT parent_id,status FROM transport_bindings WHERE id=?", (binding_id,)).fetchone()
        if not row:
            raise KeyError(binding_id)
        conn.execute("UPDATE transport_bindings SET status='revoked',updated_at=? WHERE id=?", (now, binding_id))
        conn.execute(
            "INSERT INTO interaction_audit(actor_parent_id,action,subject_type,subject_id,created_at) VALUES(?,?,'transport_binding',?,?)",
            (row["parent_id"], "slack_revoked", binding_id, now),
        )
    return {"binding_id": binding_id, "status": "revoked"}


def begin_receipt(db_path: str | Path, message: InboundMessage, binding_id: str | None) -> bool:
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT payload_hash FROM transport_receipts WHERE transport=? AND external_event_id=?",
            (message.transport, message.external_event_id),
        ).fetchone()
        if existing:
            if existing["payload_hash"] != message.payload_hash:
                raise TransportConflict("transport event ID was reused with a different payload")
            return False
        conn.execute(
            """INSERT INTO transport_receipts(transport,external_event_id,payload_hash,binding_id,status,received_at)
               VALUES(?,?,?,?, 'received',?)""",
            (message.transport, message.external_event_id, message.payload_hash, binding_id, now),
        )
    return True


def finish_receipt(
    db_path: str | Path,
    message: InboundMessage,
    *,
    status: str,
    event_id: str | None = None,
    error: str | None = None,
) -> None:
    if status not in {"completed", "rejected", "failed"}:
        raise ValueError("invalid receipt status")
    with connect(db_path) as conn:
        conn.execute(
            """UPDATE transport_receipts SET status=?,event_id=?,processed_at=?,error=?
               WHERE transport=? AND external_event_id=?""",
            (status, event_id, utcnow(), error, message.transport, message.external_event_id),
        )


def create_unassigned_capture(db_path: str | Path, message: InboundMessage, parent_id: str) -> dict[str, Any]:
    inbox_id = f"inbox_{uuid4().hex[:12]}"
    now = utcnow()
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id,status FROM capture_inbox WHERE transport=? AND external_event_id=?",
            (message.transport, message.external_event_id),
        ).fetchone()
        if existing:
            return {"inbox_id": existing["id"], "status": existing["status"], "duplicate": True}
        conn.execute(
            """INSERT INTO capture_inbox(id,parent_id,transport,external_event_id,text,status,created_at)
               VALUES(?,?,?,?,?,'unassigned',?)""",
            (inbox_id, parent_id, message.transport, message.external_event_id, message.text, now),
        )
    return {"inbox_id": inbox_id, "status": "unassigned", "duplicate": False}


def list_inbox(db_path: str | Path, *, status: str = "unassigned") -> list[dict[str, Any]]:
    init_db(db_path)
    if status not in {"unassigned", "assigned", "dismissed", "all"}:
        raise ValueError("invalid inbox status")
    query = "SELECT * FROM capture_inbox"
    params: tuple[Any, ...] = ()
    if status != "all":
        query += " WHERE status=?"
        params = (status,)
    query += " ORDER BY created_at DESC"
    with connect(db_path) as conn:
        return [dict(row) for row in conn.execute(query, params)]


def resolve_inbox(db_path: str | Path, inbox_id: str, *, child_id: str | None, dismiss: bool = False) -> dict[str, Any]:
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM capture_inbox WHERE id=?", (inbox_id,)).fetchone()
        if not row:
            raise KeyError(inbox_id)
        if row["status"] != "unassigned":
            return dict(row)
        if dismiss:
            conn.execute(
                "UPDATE capture_inbox SET status='dismissed',resolved_at=? WHERE id=?", (now, inbox_id)
            )
            return {**dict(row), "status": "dismissed", "resolved_at": now}
        child = conn.execute("SELECT id FROM children WHERE id=?", (child_id,)).fetchone()
        if not child:
            raise ValueError("child not found")
        conn.execute(
            "UPDATE capture_inbox SET status='assigned',child_id=?,resolved_at=? WHERE id=?",
            (child_id, now, inbox_id),
        )
    return {**dict(row), "status": "assigned", "child_id": child_id, "resolved_at": now}


def enqueue_delivery(
    db_path: str | Path,
    binding_id: str,
    message: OutboundMessage,
    *,
    idempotency_key: str,
    expires_in_hours: int = 24,
) -> str:
    now = datetime.now(UTC)
    delivery_id = f"delivery_{uuid4().hex[:16]}"
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM delivery_outbox WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        if existing:
            return existing["id"]
        conn.execute(
            """INSERT INTO delivery_outbox(id,transport,binding_id,payload_json,idempotency_key,status,attempts,
                                             available_at,expires_at,created_at,updated_at)
               VALUES(?,'slack',?,?,?,'queued',0,?,?,?,?)""",
            (
                delivery_id,
                binding_id,
                message.model_dump_json(),
                idempotency_key,
                now.isoformat(),
                (now + timedelta(hours=expires_in_hours)).isoformat(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
    return delivery_id


def ready_deliveries(db_path: str | Path, *, limit: int = 20) -> list[dict[str, Any]]:
    now = utcnow()
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT o.*,b.team_id,b.user_id,b.channel_id FROM delivery_outbox o
               JOIN transport_bindings b ON b.id=o.binding_id
               WHERE o.transport='slack' AND o.status IN ('queued','failed') AND o.available_at<=?
                 AND (o.expires_at IS NULL OR o.expires_at>?) AND o.attempts<5 AND b.status='active'
               ORDER BY o.created_at LIMIT ?""",
            (now, now, limit),
        ).fetchall()
    return [{**dict(row), "payload": jload(row["payload_json"])} for row in rows]


def mark_delivery(
    db_path: str | Path,
    delivery_id: str,
    *,
    status: str,
    external_message_id: str | None = None,
    error: str | None = None,
) -> None:
    if status not in {"sending", "sent", "failed", "unknown", "expired"}:
        raise ValueError("invalid delivery status")
    now = datetime.now(UTC)
    with connect(db_path) as conn:
        row = conn.execute("SELECT attempts FROM delivery_outbox WHERE id=?", (delivery_id,)).fetchone()
        if not row:
            raise KeyError(delivery_id)
        attempts = int(row["attempts"]) + (1 if status == "sending" else 0)
        available = now + timedelta(seconds=min(300, 2 ** max(attempts, 1)))
        conn.execute(
            """UPDATE delivery_outbox SET status=?,attempts=?,available_at=?,external_message_id=?,
                   last_error=?,updated_at=? WHERE id=?""",
            (
                status,
                attempts,
                available.isoformat(),
                external_message_id,
                error[:2_000] if error else None,
                now.isoformat(),
                delivery_id,
            ),
        )
