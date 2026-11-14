from __future__ import annotations

import os
import re
import secrets
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .brain_config import brain_config_fingerprint
from .config import AppConfig
from .db import connect, init_db, jdump, jload, transport_event_hash, utcnow
from .deletion_ledger import redact_parent_memory_deliveries
from .transports.contracts import InboundMessage, OutboundMessage

PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
RESOURCE_CONTEXT_MODES = {"metadata_only", "selected_excerpts"}
VISUAL_MODES = {"off", "deterministic", "decorative"}
CHECKPOINT_STATUSES = {"pending", "pass", "fail"}


class TransportConflict(ValueError):
    pass


def answer_stack_fingerprint(db_path: str | Path) -> str:
    """Bind a parent quality review to the model routes and answer-shaping configuration."""

    app = AppConfig.load()
    digest = sha256()
    digest.update(brain_config_fingerprint().encode())
    from .text_review import MODEL, POLICY

    digest.update(f"text-review:{os.environ.get('CURIOSITY_TEXT_REVIEW', 'configured')}:{MODEL}:{POLICY}".encode())
    for relative in (
        "configs/production.json",
        "configs/reasoning-policy.json",
        "configs/context-policy.json",
        "prompts/generator-v1.md",
        "prompts/critic-v1.md",
        "src/curiosity_engine/visuals.py",
        "src/curiosity_engine/trust.py",
        "src/curiosity_engine/text_review.py",
    ):
        path = app.root / relative
        if path.is_file():
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    with connect(db_path) as conn:
        family_lens = conn.execute(
            "SELECT profile_json FROM learning_preferences WHERE scope_type='household' AND scope_id='default'"
        ).fetchone()
        resource_mode = conn.execute(
            "SELECT resource_context_mode,visual_mode FROM household_settings WHERE id='default'"
        ).fetchone()
    digest.update(str(family_lens["profile_json"] if family_lens else "unconfigured").encode())
    digest.update(str(resource_mode["resource_context_mode"] if resource_mode else "metadata_only").encode())
    digest.update(str(resource_mode["visual_mode"] if resource_mode else "deterministic").encode())
    return digest.hexdigest()[:16]


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
        visual_mode = str(
            conn.execute("SELECT visual_mode FROM household_settings WHERE id='default'").fetchone()["visual_mode"]
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
                        "visual_mode": visual_mode,
                    }
                ),
                now,
            ),
        )
        context_proactivity_available = bool(
            AppConfig.load().autonomy.get("context_driven_suggestions_enabled", False)
        )
        conn.execute(
            "UPDATE schedules SET enabled=? WHERE schedule_type='weekly_reflection'",
            (int(proactive_enabled and context_proactivity_available),),
        )
    return {
        "status": "configured",
        "owner_id": owner_id,
        "timezone": timezone,
        "quiet_hours": {"start": quiet_start, "end": quiet_end},
        "proactive_enabled": proactive_enabled,
        "context_driven_suggestions_enabled": bool(
            proactive_enabled and context_proactivity_available
        ),
        "resource_context_mode": resource_context_mode,
        "visual_mode": visual_mode,
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


def household_visual_mode(db_path: str | Path) -> str:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT visual_mode FROM household_settings WHERE id='default'").fetchone()
    return str(row["visual_mode"]) if row else "deterministic"


def set_household_visual_mode(db_path: str | Path, mode: str) -> dict[str, str]:
    if mode not in VISUAL_MODES:
        raise ValueError("visual mode must be off, deterministic, or decorative")
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
        conn.execute("UPDATE household_settings SET visual_mode=?,updated_at=? WHERE id='default'", (mode, now))
        conn.execute(
            """INSERT INTO interaction_audit(
                   actor_parent_id,action,subject_type,subject_id,metadata_json,created_at
               ) VALUES(?,?,'household','default',?,?)""",
            (owner["id"] if owner else None, "visual_mode_changed", jdump({"mode": mode}), now),
        )
    return {"status": "configured", "visual_mode": mode}


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
        checkpoints = {
            row["checkpoint"]: {
                "status": row["status"],
                "verified_at": row["verified_at"],
            }
            for row in conn.execute(
                "SELECT checkpoint,status,verified_at FROM onboarding_checkpoints ORDER BY checkpoint"
            )
        }
        family_lens = conn.execute(
            "SELECT 1 FROM learning_preferences WHERE scope_type='household' AND scope_id='default'"
        ).fetchone()
    review_hash = answer_stack_fingerprint(db_path)
    with connect(db_path) as conn:
        latest_review = conn.execute(
            """SELECT decision FROM onboarding_reviews WHERE brain_config_hash=?
               ORDER BY created_at DESC,id DESC LIMIT 1""",
            (review_hash,),
        ).fetchone()
    return {
        "configured": settings is not None and bool(parents),
        "settings": dict(settings) if settings else None,
        "parents": parents,
        "bindings": bindings,
        "unassigned_captures": inbox_count,
        "checkpoints": checkpoints,
        "family_lens_configured": bool(family_lens),
        "quality_review_accepted": bool(latest_review and latest_review["decision"] == "pass"),
    }


def record_onboarding_checkpoint(
    db_path: str | Path,
    checkpoint: str,
    *,
    status: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in CHECKPOINT_STATUSES:
        raise ValueError("invalid onboarding checkpoint status")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", checkpoint):
        raise ValueError("invalid onboarding checkpoint name")
    init_db(db_path)
    now = utcnow()
    sanitized = evidence or {}
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO onboarding_checkpoints(checkpoint,status,evidence_json,verified_at,updated_at)
               VALUES(?,?,?,?,?)
               ON CONFLICT(checkpoint) DO UPDATE SET status=excluded.status,
                 evidence_json=excluded.evidence_json,verified_at=excluded.verified_at,updated_at=excluded.updated_at""",
            (checkpoint, status, jdump(sanitized), now if status == "pass" else None, now),
        )
    return {"checkpoint": checkpoint, "status": status, "verified_at": now if status == "pass" else None}


def onboarding_checkpoint(db_path: str | Path, checkpoint: str) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT checkpoint,status,evidence_json,verified_at,updated_at FROM onboarding_checkpoints WHERE checkpoint=?",
            (checkpoint,),
        ).fetchone()
    if not row:
        return None
    return {**dict(row), "evidence": jload(row["evidence_json"])}


def configure_family_lens(db_path: str | Path, profile: dict[str, Any]) -> dict[str, Any]:
    init_db(db_path)
    allowed = {
        "pedagogy",
        "themes",
        "activity_minutes",
        "parent_effort",
        "reading_load",
        "materials",
        "content_boundaries",
    }
    unknown = set(profile) - allowed
    if unknown:
        raise ValueError("unknown family-lens fields: " + ", ".join(sorted(unknown)))
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO learning_preferences(scope_type,scope_id,profile_json,source,created_at,updated_at)
               VALUES('household','default',?,'parent',?,?)
               ON CONFLICT(scope_type,scope_id) DO UPDATE SET profile_json=excluded.profile_json,
                 source=excluded.source,updated_at=excluded.updated_at""",
            (jdump(profile), now, now),
        )
    record_onboarding_checkpoint(db_path, "family_lens_ready", status="pass", evidence={"version": 1})
    return {"status": "configured", "fields": sorted(profile), "private": True}


def record_onboarding_review(
    db_path: str | Path,
    *,
    event_id: str,
    factuality: str,
    grade_fit: str,
    curiosity_value: str,
    parent_effort: str,
    note: str | None = None,
) -> dict[str, Any]:
    ratings = (factuality, grade_fit, curiosity_value, parent_effort)
    if any(value not in {"pass", "retry"} for value in ratings):
        raise ValueError("review ratings must be pass or retry")
    init_db(db_path)
    with connect(db_path) as conn:
        event = conn.execute(
            """SELECT e.id,r.workflow,r.output_json FROM events e
               JOIN responses r ON r.event_id=e.id AND r.status='completed'
               JOIN transport_receipts t ON t.event_id=e.id AND t.transport='slack' AND t.status='completed'
               JOIN delivery_outbox d ON d.binding_id=t.binding_id AND d.status='sent'
                 AND d.idempotency_key LIKE ('slack:' || t.external_event_id || ':%')
               WHERE e.id=? AND e.type='child_question' AND e.status='completed'
               ORDER BY d.updated_at DESC LIMIT 1""",
            (event_id,),
        ).fetchone()
    if not event:
        raise ValueError("review requires a completed real Slack answer with confirmed delivery")
    decision = "pass" if all(value == "pass" for value in ratings) else "retry"
    now = utcnow()
    review_hash = answer_stack_fingerprint(db_path)
    response = jload(event["output_json"])
    generated_hash = str((response.get("_reasoning") or {}).get("answer_stack_hash") or "")
    if generated_hash != review_hash:
        raise ValueError("review requires an answer generated by the current answer stack")
    response_hash = sha256(str(event["output_json"]).encode()).hexdigest()
    with connect(db_path) as conn:
        cursor = conn.execute(
            """INSERT INTO onboarding_reviews(
               event_id,brain_config_hash,response_hash,workflow,factuality,grade_fit,curiosity_value,
               parent_effort,note,decision,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id,
                review_hash,
                response_hash,
                event["workflow"],
                factuality,
                grade_fit,
                curiosity_value,
                parent_effort,
                note,
                decision,
                now,
            ),
        )
        review_id = int(cursor.lastrowid)
    record_onboarding_checkpoint(
        db_path,
        "quality_review",
        status="pass" if decision == "pass" else "fail",
        evidence={"review_id": review_id, "event_recorded": True, "config_hash": review_hash},
    )
    return {"review_id": review_id, "decision": decision, "event_recorded": True, "event_id": event_id}


def reviewable_slack_events(db_path: str | Path, *, limit: int = 5) -> list[dict[str, Any]]:
    """List sanitized IDs for delivered Slack answers without exposing child text or names."""

    if not 1 <= limit <= 20:
        raise ValueError("reviewable event limit must be 1..20")
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT e.id AS event_id,e.created_at,r.workflow,r.output_json,
                      EXISTS(SELECT 1 FROM onboarding_reviews q WHERE q.event_id=e.id) AS reviewed
               FROM events e
               JOIN responses r ON r.event_id=e.id AND r.status='completed'
               JOIN transport_receipts t ON t.event_id=e.id AND t.transport='slack' AND t.status='completed'
               JOIN delivery_outbox d ON d.binding_id=t.binding_id AND d.status='sent'
                 AND d.idempotency_key LIKE ('slack:' || t.external_event_id || ':%')
               WHERE e.type='child_question' AND e.status='completed'
               GROUP BY e.id,e.created_at,r.workflow
               ORDER BY e.created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    current_hash = answer_stack_fingerprint(db_path)
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        output = jload(item.pop("output_json"))
        item["current_answer_stack"] = (
            str((output.get("_reasoning") or {}).get("answer_stack_hash") or "") == current_hash
        )
        result.append(item)
    return result


def delivered_slack_response(
    db_path: str | Path,
    *,
    event_id: str,
    binding_id: str,
) -> dict[str, Any] | None:
    """Return a response only when this exact paired conversation received it."""

    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT e.id AS event_id,e.child_id,e.text,r.status
               FROM events e
               JOIN responses r ON r.event_id=e.id
               JOIN transport_receipts t ON t.event_id=e.id AND t.transport='slack'
                 AND t.status='completed' AND t.binding_id=?
               JOIN delivery_outbox d ON d.binding_id=t.binding_id AND d.status='sent'
                 AND d.idempotency_key LIKE ('slack:' || t.external_event_id || ':%')
               WHERE e.id=? AND e.type='child_question'
                 AND r.status IN ('completed','rejected')
               ORDER BY d.updated_at DESC LIMIT 1""",
            (binding_id, event_id),
        ).fetchone()
    return dict(row) if row else None


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
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT parent_id,status FROM transport_bindings WHERE id=?", (binding_id,)).fetchone()
        if not row:
            raise KeyError(binding_id)
        conn.execute("UPDATE transport_bindings SET status='revoked',updated_at=? WHERE id=?", (now, binding_id))
        delivery_ids = {
            str(item[0])
            for item in conn.execute(
                """SELECT id FROM delivery_outbox
                   WHERE binding_id=? AND status IN ('queued','failed','sending')""",
                (binding_id,),
            ).fetchall()
        }
        conn.execute(
            """UPDATE delivery_outbox
               SET status=CASE WHEN status='sending' THEN 'unknown' ELSE 'expired' END,
                   last_error='Slack binding revoked before delivery',updated_at=?
               WHERE binding_id=? AND status IN ('queued','failed','sending')""",
            (now, binding_id),
        )
        if delivery_ids:
            redact_parent_memory_deliveries(conn, delivery_ids=delivery_ids)
        conn.execute(
            """UPDATE slack_file_outbox
               SET status=CASE
                     WHEN status IN ('waiting_asset','queued','failed') THEN 'expired'
                     ELSE 'unknown'
                   END,
                   active_attempt_id=NULL,lease_token_hash=NULL,
                   last_error='Slack binding revoked before file delivery',updated_at=?
               WHERE binding_id=? AND status IN (
                 'waiting_asset','queued','failed','ticket_acquiring','ticket_acquired',
                 'uploading','bytes_uploaded','completing'
               )""",
            (now, binding_id),
        )
        conn.execute(
            """UPDATE artifact_file_outbox
               SET status=CASE
                     WHEN status IN ('queued','failed') THEN 'expired'
                     ELSE 'unknown'
                   END,
                   active_attempt_id=NULL,lease_token_hash=NULL,
                   last_error='Slack binding revoked before artifact delivery',updated_at=?
               WHERE binding_id=? AND status IN (
                 'queued','failed','ticket_acquiring','ticket_acquired',
                 'uploading','bytes_uploaded','completing'
               )""",
            (now, binding_id),
        )
        conn.execute(
            """UPDATE background_delivery_attempts
               SET status='unknown',error_code='binding_revoked',completed_at=?
               WHERE status='sending' AND delivery_id IN (
                 SELECT delivery_id FROM background_delivery_outbox WHERE binding_id=?
               )""",
            (now, binding_id),
        )
        conn.execute(
            """UPDATE background_delivery_outbox
               SET status=CASE WHEN status='sending' THEN 'unknown' ELSE 'cancelled' END,
                   active_attempt_id=NULL,lease_owner=NULL,leased_at=NULL,lease_token_hash=NULL,
                   updated_at=?
               WHERE binding_id=? AND status IN ('awaiting_confirmation','queued','sending')""",
            (now, binding_id),
        )
        conn.execute(
            "INSERT INTO interaction_audit(actor_parent_id,action,subject_type,subject_id,created_at) VALUES(?,?,'transport_binding',?,?)",
            (row["parent_id"], "slack_revoked", binding_id, now),
        )
    return {"binding_id": binding_id, "status": "revoked"}


def begin_receipt(
    db_path: str | Path,
    message: InboundMessage,
    binding_id: str | None,
    *,
    retry_failed: bool = False,
) -> bool:
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        deleted = conn.execute(
            """SELECT 1 FROM deleted_transport_event_tombstones
               WHERE transport_event_hash=?""",
            (transport_event_hash(message.transport, message.external_event_id),),
        ).fetchone()
        if deleted:
            raise TransportConflict("transport event was deleted and cannot be replayed")
        existing = conn.execute(
            "SELECT payload_hash,status FROM transport_receipts WHERE transport=? AND external_event_id=?",
            (message.transport, message.external_event_id),
        ).fetchone()
        if existing:
            if retry_failed and existing["status"] in {"failed", "rejected"}:
                conn.execute(
                    """UPDATE transport_receipts
                       SET payload_hash=?,binding_id=?,status='received',event_id=NULL,
                           received_at=?,processed_at=NULL,error=NULL
                       WHERE transport=? AND external_event_id=?""",
                    (
                        message.payload_hash,
                        binding_id,
                        now,
                        message.transport,
                        message.external_event_id,
                    ),
                )
                return True
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


def _invalid_parent_memory_delivery_ids(
    conn: Any,
    delivery_ids: set[str],
    *,
    now: str,
) -> set[str]:
    """Find typed deliveries whose referenced memory is no longer usable."""

    if not delivery_ids:
        return set()
    placeholders = ",".join("?" for _ in delivery_ids)
    tombstones = {
        str(row[0])
        for row in conn.execute(
            "SELECT memory_record_hash FROM parent_memory_tombstones"
        ).fetchall()
    }
    invalid: set[str] = set()
    for row in conn.execute(
        f"""SELECT m.delivery_id,m.memory_key,m.memory_version,m.content_hash,
                   p.status AS memory_status,p.content_hash AS current_content_hash,
                   p.expires_at,p.scope_type,p.scope_id,
                   COALESCE(h.enabled,1) AS household_enabled,
                   COALESCE(s.enabled,1) AS scope_enabled
            FROM delivery_outbox_parent_memories m
            LEFT JOIN parent_memories p
              ON p.memory_key=m.memory_key AND p.version=m.memory_version
            LEFT JOIN parent_memory_settings h
              ON h.scope_type='household' AND h.scope_id='default'
            LEFT JOIN parent_memory_settings s
              ON s.scope_type=p.scope_type AND s.scope_id=p.scope_id
            WHERE m.delivery_id IN ({placeholders})""",
        tuple(sorted(delivery_ids)),
    ).fetchall():
        record_hash = sha256(f"{row['memory_key']}\0{row['memory_version']}".encode()).hexdigest()
        if (
            row["memory_status"] != "active"
            or row["current_content_hash"] != row["content_hash"]
            or (row["expires_at"] is not None and str(row["expires_at"]) <= now)
            or not bool(row["household_enabled"])
            or not bool(row["scope_enabled"])
            or record_hash in tombstones
        ):
            invalid.add(str(row["delivery_id"]))
    return invalid


def binding_can_deliver(conn: Any, binding_id: str, *, channel_id: str | None = None) -> bool:
    channel_clause = "" if channel_id is None else " AND b.channel_id=?"
    params: tuple[str, ...] = (binding_id,) if channel_id is None else (binding_id, channel_id)
    return bool(
        conn.execute(
            f"""SELECT 1 FROM transport_bindings b
               JOIN parent_principals p ON p.id=b.parent_id
               WHERE b.id=? AND b.transport='slack'
                 AND b.status='active' AND p.status='active'{channel_clause}""",
            params,
        ).fetchone()
    )


def enqueue_delivery(
    db_path: str | Path,
    binding_id: str,
    message: OutboundMessage,
    *,
    idempotency_key: str,
    expires_in_hours: int = 24,
    parent_memory_refs: list[dict[str, Any]] | None = None,
) -> str:
    normalized_refs: list[tuple[str, int]] = []
    seen_refs: set[tuple[str, int]] = set()
    if parent_memory_refs is not None:
        if not isinstance(parent_memory_refs, list) or len(parent_memory_refs) > 24:
            raise ValueError("parent memory delivery references must be a bounded list")
        for ref in parent_memory_refs:
            if not isinstance(ref, dict) or set(ref) != {"memory_key", "version"}:
                raise ValueError("parent memory delivery references must be exact typed records")
            memory_key = ref.get("memory_key")
            version = ref.get("version")
            if (
                not isinstance(memory_key, str)
                or not 2 <= len(memory_key) <= 160
                or type(version) is not int
                or version < 1
            ):
                raise ValueError("parent memory delivery reference is invalid")
            normalized = (memory_key, version)
            if normalized in seen_refs:
                raise ValueError("parent memory delivery references must be unique")
            seen_refs.add(normalized)
            normalized_refs.append(normalized)
    now = datetime.now(UTC)
    delivery_id = f"delivery_{uuid4().hex[:16]}"
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not binding_can_deliver(conn, binding_id, channel_id=message.channel_id):
            raise ValueError("Slack delivery binding is unavailable")
        existing = conn.execute(
            "SELECT id FROM delivery_outbox WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        if existing:
            existing_refs = {
                (str(row[0]), int(row[1]))
                for row in conn.execute(
                    """SELECT memory_key,memory_version
                       FROM delivery_outbox_parent_memories WHERE delivery_id=?""",
                    (existing["id"],),
                ).fetchall()
            }
            if existing_refs != seen_refs:
                raise ValueError("delivery idempotency key was reused with different memory lineage")
            return str(existing["id"])
        resolved_refs: list[tuple[str, int, str]] = []
        for memory_key, version in normalized_refs:
            memory = conn.execute(
                """SELECT content_hash FROM parent_memories
                   WHERE memory_key=? AND version=? AND status='active'""",
                (memory_key, version),
            ).fetchone()
            tombstone = conn.execute(
                """SELECT 1 FROM parent_memory_tombstones
                   WHERE memory_record_hash=?""",
                (sha256(f"{memory_key}\0{version}".encode()).hexdigest(),),
            ).fetchone()
            if not memory or tombstone:
                raise ValueError("parent memory delivery reference is unavailable")
            resolved_refs.append((memory_key, version, str(memory["content_hash"])))
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
        conn.executemany(
            """INSERT INTO delivery_outbox_parent_memories(
                   delivery_id,memory_key,memory_version,content_hash,created_at
               ) VALUES(?,?,?,?,?)""",
            [
                (delivery_id, memory_key, version, content_hash, now.isoformat())
                for memory_key, version, content_hash in resolved_refs
            ],
        )
        if _invalid_parent_memory_delivery_ids(conn, {delivery_id}, now=now.isoformat()):
            raise ValueError("parent memory delivery reference is unavailable")
    return delivery_id


def ready_deliveries(db_path: str | Path, *, limit: int = 20) -> list[dict[str, Any]]:
    current = datetime.now(UTC)
    now = current.isoformat()
    stale = (current - timedelta(minutes=10)).isoformat()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        stale_ids = {
            str(row[0])
            for row in conn.execute(
                "SELECT id FROM delivery_outbox WHERE status='sending' AND updated_at<?",
                (stale,),
            ).fetchall()
        }
        conn.execute(
            """UPDATE delivery_outbox SET status='unknown',last_error='connector stopped during send',updated_at=?
               WHERE status='sending' AND updated_at<?""",
            (now, stale),
        )
        terminal_ids = {
            str(row[0])
            for row in conn.execute(
                """SELECT id FROM delivery_outbox
                   WHERE status IN ('queued','failed')
                     AND ((expires_at IS NOT NULL AND expires_at<=?) OR attempts>=5)""",
                (now,),
            ).fetchall()
        }
        if terminal_ids:
            placeholders = ",".join("?" for _ in terminal_ids)
            conn.execute(
                f"""UPDATE delivery_outbox
                    SET status='expired',last_error=COALESCE(last_error,'delivery expired'),updated_at=?
                    WHERE id IN ({placeholders})""",
                (now, *sorted(terminal_ids)),
            )
        if stale_ids or terminal_ids:
            redact_parent_memory_deliveries(
                conn,
                delivery_ids=stale_ids | terminal_ids,
            )
        unauthorized_ids = {
            str(row[0])
            for row in conn.execute(
                """SELECT o.id FROM delivery_outbox o
                   JOIN transport_bindings b ON b.id=o.binding_id
                   LEFT JOIN parent_principals p ON p.id=b.parent_id
                   WHERE o.transport='slack' AND o.status IN ('queued','failed')
                     AND (b.status!='active' OR p.id IS NULL OR p.status!='active'
                          OR COALESCE(json_extract(o.payload_json,'$.channel_id'),'')!=b.channel_id)"""
            ).fetchall()
        }
        if unauthorized_ids:
            placeholders = ",".join("?" for _ in unauthorized_ids)
            conn.execute(
                f"""UPDATE delivery_outbox
                    SET status='expired',last_error='Slack delivery authority revoked',updated_at=?
                    WHERE id IN ({placeholders})""",
                (now, *sorted(unauthorized_ids)),
            )
            redact_parent_memory_deliveries(conn, delivery_ids=unauthorized_ids)
        ready_ids = {
            str(row[0])
            for row in conn.execute(
                """SELECT o.id FROM delivery_outbox o
                   JOIN transport_bindings b ON b.id=o.binding_id
                   JOIN parent_principals p ON p.id=b.parent_id
                   WHERE o.transport='slack' AND o.status IN ('queued','failed')
                     AND o.available_at<=? AND (o.expires_at IS NULL OR o.expires_at>?)
                     AND o.attempts<5 AND b.status='active' AND p.status='active'
                     AND COALESCE(json_extract(o.payload_json,'$.channel_id'),'')=b.channel_id""",
                (now, now),
            ).fetchall()
        }
        invalid_ready_ids = _invalid_parent_memory_delivery_ids(conn, ready_ids, now=now)
        if invalid_ready_ids:
            redact_parent_memory_deliveries(conn, delivery_ids=invalid_ready_ids)
        rows = conn.execute(
            """SELECT o.*,b.team_id,b.user_id,b.channel_id FROM delivery_outbox o
               JOIN transport_bindings b ON b.id=o.binding_id
               JOIN parent_principals p ON p.id=b.parent_id
               WHERE o.transport='slack' AND o.status IN ('queued','failed') AND o.available_at<=?
                 AND (o.expires_at IS NULL OR o.expires_at>?) AND o.attempts<5
                 AND b.status='active' AND p.status='active'
                 AND COALESCE(json_extract(o.payload_json,'$.channel_id'),'')=b.channel_id
               ORDER BY o.created_at LIMIT ?""",
            (now, now, limit),
        ).fetchall()
    return [{**dict(row), "payload": jload(row["payload_json"])} for row in rows]


def claim_delivery(db_path: str | Path, delivery_id: str) -> bool:
    """Atomically claim one queued delivery so two connector processes cannot both send it."""

    now = datetime.now(UTC)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        delivery = conn.execute(
            "SELECT binding_id,status,payload_json FROM delivery_outbox WHERE id=?",
            (delivery_id,),
        ).fetchone()
        if not delivery:
            return False
        payload = jload(delivery["payload_json"])
        channel_id = payload.get("channel_id") if isinstance(payload, dict) else None
        if not isinstance(channel_id, str) or not binding_can_deliver(
            conn,
            str(delivery["binding_id"]),
            channel_id=channel_id,
        ):
            if delivery["status"] in {"queued", "failed"}:
                conn.execute(
                    """UPDATE delivery_outbox
                       SET status='expired',last_error='Slack delivery authority revoked',updated_at=?
                       WHERE id=?""",
                    (now.isoformat(), delivery_id),
                )
            redact_parent_memory_deliveries(conn, delivery_ids={delivery_id})
            return False
        if _invalid_parent_memory_delivery_ids(
            conn,
            {delivery_id},
            now=now.isoformat(),
        ):
            redact_parent_memory_deliveries(conn, delivery_ids={delivery_id})
            return False
        updated = conn.execute(
            """UPDATE delivery_outbox SET status='sending',attempts=attempts+1,updated_at=?
               WHERE id=? AND status IN ('queued','failed') AND available_at<=?
                 AND (expires_at IS NULL OR expires_at>?) AND attempts<5""",
            (now.isoformat(), delivery_id, now.isoformat(), now.isoformat()),
        )
    return updated.rowcount == 1


def mark_delivery(
    db_path: str | Path,
    delivery_id: str,
    *,
    status: str,
    external_message_id: str | None = None,
    error: str | None = None,
) -> bool:
    if status not in {"sending", "sent", "failed", "unknown", "expired"}:
        raise ValueError("invalid delivery status")
    now = datetime.now(UTC)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT attempts,binding_id,status,payload_json FROM delivery_outbox WHERE id=?",
            (delivery_id,),
        ).fetchone()
        if not row:
            raise KeyError(delivery_id)
        current_status = str(row["status"])
        payload = jload(row["payload_json"])
        channel_id = payload.get("channel_id") if isinstance(payload, dict) else None
        authority_active = isinstance(channel_id, str) and binding_can_deliver(
            conn,
            str(row["binding_id"]),
            channel_id=channel_id,
        )
        if current_status == "sending" and not authority_active:
            conn.execute(
                """UPDATE delivery_outbox
                   SET status='unknown',last_error='Slack delivery authority revoked',updated_at=?
                   WHERE id=? AND status='sending'""",
                (now.isoformat(), delivery_id),
            )
            redact_parent_memory_deliveries(conn, delivery_ids={delivery_id})
            return False
        if status == "sending":
            if current_status != "sending" or not authority_active:
                return False
            updated = conn.execute(
                "UPDATE delivery_outbox SET updated_at=? WHERE id=? AND status='sending'",
                (now.isoformat(), delivery_id),
            )
            return updated.rowcount == 1
        expected_statuses = (
            {"sending"}
            if status in {"sent", "failed", "unknown"}
            else {"queued", "failed", "sending"}
        )
        if current_status not in expected_statuses:
            return False
        if status in {"sent", "failed"} and not authority_active:
            return False
        attempts = int(row["attempts"])
        available = now + timedelta(seconds=min(300, 2 ** max(attempts, 1)))
        updated = conn.execute(
            """UPDATE delivery_outbox SET status=?,attempts=?,available_at=?,external_message_id=?,
                   last_error=?,updated_at=? WHERE id=? AND status=?""",
            (
                status,
                attempts,
                available.isoformat(),
                external_message_id,
                error[:2_000] if error else None,
                now.isoformat(),
                delivery_id,
                current_status,
            ),
        )
        changed = updated.rowcount == 1
        if changed and (
            status in {"sent", "unknown", "expired"} or (status == "failed" and attempts >= 5)
        ):
            redact_parent_memory_deliveries(conn, delivery_ids={delivery_id})
        return changed


def enqueue_visual_delivery(
    db_path: str | Path,
    *,
    visual_job_id: str,
    binding_id: str,
    depends_on_delivery_id: str,
    channel_id: str,
    thread_id: str | None,
    idempotency_key: str,
    purpose: str = "response_visual",
    expires_in_hours: int = 24,
) -> str:
    """Queue one visual by database ID; transport payloads never carry arbitrary file paths."""

    init_db(db_path)
    now = datetime.now(UTC)
    visual_delivery_id = f"visual_delivery_{uuid4().hex[:16]}"
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not binding_can_deliver(conn, binding_id, channel_id=channel_id):
            raise ValueError("Slack visual delivery binding is unavailable")
        existing = conn.execute(
            "SELECT id FROM slack_file_outbox WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        if existing:
            return str(existing["id"])
        dependency = conn.execute(
            "SELECT 1 FROM delivery_outbox WHERE id=? AND binding_id=?",
            (depends_on_delivery_id, binding_id),
        ).fetchone()
        if not dependency:
            raise ValueError("visual delivery dependency is unavailable for this binding")
        job = conn.execute("SELECT status FROM visual_jobs WHERE id=?", (visual_job_id,)).fetchone()
        if not job:
            raise ValueError("visual job not found")
        asset = conn.execute("SELECT id FROM visual_assets WHERE job_id=?", (visual_job_id,)).fetchone()
        status = "queued" if job["status"] == "completed" and asset else "waiting_asset"
        conn.execute(
            """INSERT INTO slack_file_outbox(
                 id,visual_job_id,visual_asset_id,binding_id,depends_on_delivery_id,channel_id,thread_id,
                 idempotency_key,purpose,status,attempts,available_at,expires_at,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,0,?,?,?,?)""",
            (
                visual_delivery_id,
                visual_job_id,
                asset["id"] if asset else None,
                binding_id,
                depends_on_delivery_id,
                channel_id,
                thread_id,
                idempotency_key,
                purpose,
                status,
                now.isoformat(),
                (now + timedelta(hours=expires_in_hours)).isoformat(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
    return visual_delivery_id


def ready_visual_deliveries(db_path: str | Path, *, limit: int = 10) -> list[dict[str, Any]]:
    current = datetime.now(UTC)
    now = current.isoformat()
    stale = (current - timedelta(minutes=10)).isoformat()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """UPDATE slack_file_outbox SET status='unknown',active_attempt_id=NULL,lease_token_hash=NULL,
                 last_error='connector stopped during completion',updated_at=?
               WHERE status='completing' AND updated_at<?""",
            (now, stale),
        )
        conn.execute(
            """UPDATE slack_file_outbox SET status='failed',last_error='connector stopped before completion',
                 active_attempt_id=NULL,lease_token_hash=NULL,available_at=?,updated_at=?
               WHERE status='ticket_acquiring' AND updated_at<?""",
            (now, now, stale),
        )
        conn.execute(
            """UPDATE slack_file_outbox SET status='unknown',
                 active_attempt_id=NULL,lease_token_hash=NULL,
                 last_error='connector stopped after Slack ticket acquisition',updated_at=?
               WHERE status IN ('ticket_acquired','uploading','bytes_uploaded','completing') AND updated_at<?""",
            (now, stale),
        )
        conn.execute(
            """UPDATE slack_file_outbox
               SET status='expired',active_attempt_id=NULL,lease_token_hash=NULL,
                   last_error=COALESCE(last_error,'visual delivery expired'),updated_at=?
               WHERE status IN ('queued','failed')
                 AND ((expires_at IS NOT NULL AND expires_at<=?) OR attempts>=3)""",
            (now, now),
        )
        conn.execute(
            """UPDATE slack_file_outbox AS f
               SET status='expired',last_error='Slack visual delivery authority revoked',updated_at=?
               WHERE f.status IN ('queued','failed') AND NOT EXISTS(
                 SELECT 1 FROM transport_bindings b
                 JOIN parent_principals p ON p.id=b.parent_id
                 WHERE b.id=f.binding_id AND b.transport='slack'
                   AND b.status='active' AND p.status='active' AND b.channel_id=f.channel_id
               )""",
            (now,),
        )
        rows = conn.execute(
            """SELECT f.*,a.path,a.filename,a.mime_type,a.byte_count,a.sha256,a.title,a.caption,a.alt_text,
                      d.status AS dependency_status,b.status AS binding_status
               FROM slack_file_outbox f
               JOIN visual_assets a ON a.id=f.visual_asset_id
               JOIN delivery_outbox d ON d.id=f.depends_on_delivery_id
               JOIN transport_bindings b ON b.id=f.binding_id
               JOIN parent_principals p ON p.id=b.parent_id
               WHERE f.status IN ('queued','failed') AND f.available_at<=?
                 AND (f.expires_at IS NULL OR f.expires_at>?) AND f.attempts<3
                 AND d.status='sent' AND d.binding_id=f.binding_id
                 AND b.transport='slack' AND b.status='active' AND p.status='active'
                 AND b.channel_id=f.channel_id
               ORDER BY f.created_at LIMIT ?""",
            (now, now, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def claim_visual_delivery(db_path: str | Path, delivery_id: str) -> dict[str, str] | None:
    now = datetime.now(UTC).isoformat()
    attempt_id = f"visual_attempt_{uuid4().hex[:20]}"
    lease_token = uuid4().hex + uuid4().hex
    lease_token_hash = sha256(lease_token.encode()).hexdigest()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        delivery = conn.execute(
            "SELECT binding_id,channel_id,status FROM slack_file_outbox WHERE id=?",
            (delivery_id,),
        ).fetchone()
        if not delivery:
            return None
        if not binding_can_deliver(
            conn,
            str(delivery["binding_id"]),
            channel_id=str(delivery["channel_id"]),
        ):
            if delivery["status"] in {"queued", "failed"}:
                conn.execute(
                    """UPDATE slack_file_outbox
                       SET status='expired',last_error='Slack visual delivery authority revoked',updated_at=?
                       WHERE id=?""",
                    (now, delivery_id),
                )
            return None
        updated = conn.execute(
            """UPDATE slack_file_outbox SET status='ticket_acquiring',attempts=attempts+1,
                   active_attempt_id=?,lease_token_hash=?,updated_at=?
               WHERE id=? AND status IN ('queued','failed') AND available_at<=? AND attempts<3
                 AND (expires_at IS NULL OR expires_at>?)
                 AND active_attempt_id IS NULL AND lease_token_hash IS NULL
                 AND EXISTS(
                   SELECT 1 FROM delivery_outbox d
                   WHERE d.id=slack_file_outbox.depends_on_delivery_id
                     AND d.binding_id=slack_file_outbox.binding_id AND d.status='sent'
                 )
                 AND EXISTS(
                   SELECT 1 FROM transport_bindings b
                   JOIN parent_principals p ON p.id=b.parent_id
                   WHERE b.id=slack_file_outbox.binding_id AND b.transport='slack'
                     AND b.status='active' AND p.status='active'
                     AND b.channel_id=slack_file_outbox.channel_id
                 )""",
            (attempt_id, lease_token_hash, now, delivery_id, now, now),
        )
    if updated.rowcount != 1:
        return None
    return {"attempt_id": attempt_id, "lease_token": lease_token}


def mark_visual_delivery(
    db_path: str | Path,
    delivery_id: str,
    *,
    status: str,
    attempt_id: str,
    lease_token: str,
    slack_file_id: str | None = None,
    external_message_id: str | None = None,
    error: str | None = None,
) -> bool:
    allowed = {
        "ticket_acquiring",
        "ticket_acquired",
        "uploading",
        "bytes_uploaded",
        "completing",
        "sent",
        "failed",
        "unknown",
        "expired",
    }
    if status not in allowed:
        raise ValueError("invalid visual delivery status")
    now = datetime.now(UTC)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT slack_file_id,attempts,binding_id,channel_id,status,
                      active_attempt_id,lease_token_hash
               FROM slack_file_outbox WHERE id=?""",
            (delivery_id,),
        ).fetchone()
        if not row:
            raise KeyError(delivery_id)
        if (
            str(row["active_attempt_id"] or "") != attempt_id
            or not row["lease_token_hash"]
            or not secrets.compare_digest(
                str(row["lease_token_hash"]),
                sha256(lease_token.encode()).hexdigest(),
            )
        ):
            return False
        in_flight = {
            "ticket_acquiring",
            "ticket_acquired",
            "uploading",
            "bytes_uploaded",
            "completing",
        }
        current_status = str(row["status"])
        authority_active = binding_can_deliver(
            conn,
            str(row["binding_id"]),
            channel_id=str(row["channel_id"]),
        )
        if current_status in in_flight and not authority_active:
            conn.execute(
                """UPDATE slack_file_outbox
                   SET status='unknown',active_attempt_id=NULL,lease_token_hash=NULL,
                       last_error='Slack visual delivery authority revoked',updated_at=?
                   WHERE id=? AND status=? AND active_attempt_id=? AND lease_token_hash=?""",
                (
                    now.isoformat(),
                    delivery_id,
                    current_status,
                    attempt_id,
                    str(row["lease_token_hash"]),
                ),
            )
            return False
        expected = {
            "ticket_acquiring": {"ticket_acquiring"},
            "ticket_acquired": {"ticket_acquiring"},
            "uploading": {"ticket_acquired"},
            "bytes_uploaded": {"uploading"},
            "completing": {"bytes_uploaded"},
            "sent": {"completing"},
            "failed": {"ticket_acquiring"},
            "unknown": in_flight,
            "expired": in_flight,
        }[status]
        if current_status not in expected:
            return False
        available = now + timedelta(seconds=min(300, 2 ** max(int(row["attempts"]), 1)))
        terminal = status in {"sent", "failed", "unknown", "expired"}
        updated = conn.execute(
            """UPDATE slack_file_outbox SET status=?,available_at=?,slack_file_id=?,external_message_id=?,
                 last_error=?,active_attempt_id=?,lease_token_hash=?,updated_at=?
               WHERE id=? AND status=? AND active_attempt_id=? AND lease_token_hash=?""",
            (
                status,
                available.isoformat(),
                slack_file_id if slack_file_id is not None else row["slack_file_id"],
                external_message_id,
                error[:2_000] if error else None,
                None if terminal else attempt_id,
                None if terminal else str(row["lease_token_hash"]),
                now.isoformat(),
                delivery_id,
                current_status,
                attempt_id,
                str(row["lease_token_hash"]),
            ),
        )
        return updated.rowcount == 1
