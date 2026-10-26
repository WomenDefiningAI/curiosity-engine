from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .db import connect, init_db, utcnow
from .interaction import binding_can_deliver

MAX_ARTIFACT_ATTEMPTS = 3
ARTIFACT_STALE_SECONDS = 300


def _recover_artifact_deliveries(conn: Any, *, now: str, cutoff: str) -> int:
    retryable = conn.execute(
        """UPDATE artifact_file_outbox
           SET status='queued',active_attempt_id=NULL,lease_token_hash=NULL,
               available_at=?,updated_at=?,last_error='recovered pre-ticket claim'
           WHERE status='ticket_acquiring' AND updated_at<? AND slack_file_id IS NULL""",
        (now, now, cutoff),
    )
    conn.execute(
        """UPDATE artifact_file_outbox
           SET status='unknown',active_attempt_id=NULL,lease_token_hash=NULL,
               updated_at=?,last_error='interrupted after ticket acquisition'
           WHERE status IN ('ticket_acquired','uploading','bytes_uploaded','completing')
             AND updated_at<?""",
        (now, cutoff),
    )
    return retryable.rowcount


def enqueue_artifact_delivery(
    db_path: str | Path,
    *,
    artifact: dict[str, Any],
    binding_id: str,
    channel_id: str,
    thread_id: str | None,
    idempotency_key: str,
    depends_on_delivery_id: str | None = None,
    preview: bool = False,
) -> str:
    init_db(db_path)
    path = Path(str(artifact["pdf_path"])).resolve()
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    if digest != artifact.get("sha256"):
        raise ValueError("artifact bytes changed after validation")
    if preview:
        path = path.with_suffix(".png")
        data = path.read_bytes()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("artifact preview is not a PNG")
        digest = sha256(data).hexdigest()
    delivery_id = f"afile_{uuid4().hex[:20]}"
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not binding_can_deliver(conn, binding_id, channel_id=channel_id):
            raise ValueError("Slack artifact delivery binding is unavailable")
        if depends_on_delivery_id is not None:
            dependency = conn.execute(
                "SELECT 1 FROM delivery_outbox WHERE id=? AND binding_id=?",
                (depends_on_delivery_id, binding_id),
            ).fetchone()
            if not dependency:
                raise ValueError("artifact delivery dependency is unavailable for this binding")
        conn.execute(
            """INSERT INTO artifact_file_outbox(
                   id,artifact_id,binding_id,depends_on_delivery_id,channel_id,thread_id,path,filename,mime_type,
                   byte_count,sha256,title,comment,idempotency_key,status,attempts,available_at,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,'queued',0,?,?,?)
               ON CONFLICT(idempotency_key) DO NOTHING""",
            (
                delivery_id,
                artifact["artifact_id"],
                binding_id,
                depends_on_delivery_id,
                channel_id,
                thread_id,
                str(path),
                path.name,
                "image/png" if preview else "application/pdf",
                len(data),
                digest,
                str(artifact.get("title") or "Curiosity Engine activity")[:200],
                str(artifact.get("directions") or "Look and play together here.")[:500] if preview
                else "Printable PDF — printing is optional.",
                idempotency_key,
                now,
                now,
                now,
            ),
        )
        row = conn.execute(
            "SELECT id FROM artifact_file_outbox WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
    return str(row["id"])


def enqueue_artifact_bundle(db_path: str | Path, **kwargs: Any) -> list[str]:
    """Put the usable image first, followed by the printable from the same artifact."""
    ids = []
    preview = Path(str(kwargs["artifact"]["pdf_path"])).with_suffix(".png")
    if preview.is_file() and preview.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        ids.append(enqueue_artifact_delivery(db_path, **{
            **kwargs, "preview": True, "idempotency_key": kwargs["idempotency_key"] + ":preview",
        }))
    ids.append(enqueue_artifact_delivery(db_path, **kwargs))
    return ids


def ready_artifact_deliveries(db_path: str | Path, *, limit: int = 10) -> list[dict[str, Any]]:
    init_db(db_path)
    current = datetime.now(UTC)
    now = current.isoformat()
    cutoff = (current - timedelta(seconds=ARTIFACT_STALE_SECONDS)).isoformat()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _recover_artifact_deliveries(conn, now=now, cutoff=cutoff)
        conn.execute(
            """UPDATE artifact_file_outbox
               SET status='expired',active_attempt_id=NULL,lease_token_hash=NULL,
                   last_error=COALESCE(last_error,'artifact delivery attempts exhausted'),updated_at=?
               WHERE status IN ('queued','failed') AND attempts>=?""",
            (now, MAX_ARTIFACT_ATTEMPTS),
        )
        conn.execute(
            """UPDATE artifact_file_outbox AS f
               SET status='expired',last_error='Slack artifact delivery authority revoked',updated_at=?
               WHERE f.status IN ('queued','failed') AND NOT EXISTS(
                 SELECT 1 FROM transport_bindings b
                 JOIN parent_principals p ON p.id=b.parent_id
                 WHERE b.id=f.binding_id AND b.transport='slack'
                   AND b.status='active' AND p.status='active' AND b.channel_id=f.channel_id
               )""",
            (now,),
        )
        rows = conn.execute(
            """SELECT f.* FROM artifact_file_outbox f
               LEFT JOIN delivery_outbox d ON d.id=f.depends_on_delivery_id
               JOIN transport_bindings b ON b.id=f.binding_id
               JOIN parent_principals p ON p.id=b.parent_id
               WHERE f.status IN ('queued','failed') AND f.available_at<=?
                 AND f.attempts<?
                 AND (f.depends_on_delivery_id IS NULL OR (
                   d.status='sent' AND d.binding_id=f.binding_id
                 ))
                 AND b.transport='slack' AND b.status='active' AND p.status='active'
                 AND b.channel_id=f.channel_id
               ORDER BY f.created_at LIMIT ?""",
            (now, MAX_ARTIFACT_ATTEMPTS, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def claim_artifact_delivery(db_path: str | Path, delivery_id: str) -> dict[str, str] | None:
    now = utcnow()
    attempt_id = f"artifact_attempt_{uuid4().hex[:20]}"
    lease_token = uuid4().hex + uuid4().hex
    lease_token_hash = sha256(lease_token.encode()).hexdigest()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        delivery = conn.execute(
            "SELECT binding_id,channel_id,status FROM artifact_file_outbox WHERE id=?",
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
                    """UPDATE artifact_file_outbox
                       SET status='expired',last_error='Slack artifact delivery authority revoked',updated_at=?
                       WHERE id=?""",
                    (now, delivery_id),
                )
            return None
        cursor = conn.execute(
            """UPDATE artifact_file_outbox SET status='ticket_acquiring',attempts=attempts+1,
                   active_attempt_id=?,lease_token_hash=?,updated_at=?
               WHERE id=? AND status IN ('queued','failed')
                 AND active_attempt_id IS NULL AND lease_token_hash IS NULL
                 AND available_at<=? AND attempts<?
                 AND EXISTS(
                   SELECT 1 FROM transport_bindings b
                   JOIN parent_principals p ON p.id=b.parent_id
                   WHERE b.id=artifact_file_outbox.binding_id AND b.transport='slack'
                     AND b.status='active' AND p.status='active'
                     AND b.channel_id=artifact_file_outbox.channel_id
                 )
                 AND (
                   depends_on_delivery_id IS NULL OR EXISTS(
                     SELECT 1 FROM delivery_outbox d
                     WHERE d.id=artifact_file_outbox.depends_on_delivery_id
                       AND d.binding_id=artifact_file_outbox.binding_id AND d.status='sent'
                   )
                 )""",
            (
                attempt_id,
                lease_token_hash,
                now,
                delivery_id,
                now,
                MAX_ARTIFACT_ATTEMPTS,
            ),
        )
    if cursor.rowcount != 1:
        return None
    return {"attempt_id": attempt_id, "lease_token": lease_token}


def mark_artifact_delivery(
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
        raise ValueError("invalid artifact delivery status")
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT binding_id,channel_id,status,attempts,active_attempt_id,lease_token_hash
               FROM artifact_file_outbox WHERE id=?""",
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
                """UPDATE artifact_file_outbox
                   SET status='unknown',active_attempt_id=NULL,lease_token_hash=NULL,
                       last_error='Slack artifact delivery authority revoked',updated_at=?
                   WHERE id=? AND status=? AND active_attempt_id=? AND lease_token_hash=?""",
                (
                    now,
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
        terminal = status in {"sent", "failed", "unknown", "expired"}
        available_at = (
            datetime.now(UTC)
            + timedelta(seconds=min(300, 2 ** max(int(row["attempts"]), 1)))
        ).isoformat()
        cursor = conn.execute(
            """UPDATE artifact_file_outbox SET status=?,slack_file_id=COALESCE(?,slack_file_id),
               external_message_id=COALESCE(?,external_message_id),last_error=?,
               active_attempt_id=?,lease_token_hash=?,available_at=?,updated_at=?
               WHERE id=? AND status=? AND active_attempt_id=? AND lease_token_hash=?""",
            (
                status,
                slack_file_id,
                external_message_id,
                error,
                None if terminal else attempt_id,
                None if terminal else str(row["lease_token_hash"]),
                available_at,
                now,
                delivery_id,
                current_status,
                attempt_id,
                str(row["lease_token_hash"]),
            ),
        )
        return cursor.rowcount == 1


def recover_artifact_deliveries(db_path: str | Path, *, older_than_seconds: int = 300) -> int:
    """Fail closed after interrupted side effects; only pre-ticket claims may retry."""

    current = datetime.now(UTC)
    now = current.isoformat()
    cutoff = (current - timedelta(seconds=older_than_seconds)).isoformat()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        return _recover_artifact_deliveries(conn, now=now, cutoff=cutoff)
