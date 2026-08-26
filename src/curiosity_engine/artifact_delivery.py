from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .db import connect, init_db, utcnow


def enqueue_artifact_delivery(
    db_path: str | Path,
    *,
    artifact: dict[str, Any],
    binding_id: str,
    channel_id: str,
    thread_id: str | None,
    idempotency_key: str,
    depends_on_delivery_id: str | None = None,
) -> str:
    init_db(db_path)
    path = Path(str(artifact["pdf_path"])).resolve()
    data = path.read_bytes()
    digest = sha256(data).hexdigest()
    if digest != artifact.get("sha256"):
        raise ValueError("artifact bytes changed after validation")
    delivery_id = f"afile_{uuid4().hex[:20]}"
    now = utcnow()
    with connect(db_path) as conn:
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
                "application/pdf",
                len(data),
                digest,
                str(artifact.get("title") or "Curiosity Engine activity")[:200],
                "Here is the printable. Preview it together; printing is optional.",
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


def ready_artifact_deliveries(db_path: str | Path, *, limit: int = 10) -> list[dict[str, Any]]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT f.* FROM artifact_file_outbox f
               LEFT JOIN delivery_outbox d ON d.id=f.depends_on_delivery_id
               WHERE f.status='queued' AND f.available_at<=?
                 AND (f.depends_on_delivery_id IS NULL OR d.status='sent')
               ORDER BY f.created_at LIMIT ?""",
            (utcnow(), limit),
        ).fetchall()
    return [dict(row) for row in rows]


def claim_artifact_delivery(db_path: str | Path, delivery_id: str) -> bool:
    now = utcnow()
    with connect(db_path) as conn:
        cursor = conn.execute(
            """UPDATE artifact_file_outbox SET status='ticket_acquiring',attempts=attempts+1,updated_at=?
               WHERE id=? AND status='queued'""",
            (now, delivery_id),
        )
    return cursor.rowcount == 1


def mark_artifact_delivery(
    db_path: str | Path,
    delivery_id: str,
    *,
    status: str,
    slack_file_id: str | None = None,
    external_message_id: str | None = None,
    error: str | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            """UPDATE artifact_file_outbox SET status=?,slack_file_id=COALESCE(?,slack_file_id),
               external_message_id=COALESCE(?,external_message_id),last_error=?,updated_at=? WHERE id=?""",
            (status, slack_file_id, external_message_id, error, utcnow(), delivery_id),
        )


def recover_artifact_deliveries(db_path: str | Path, *, older_than_seconds: int = 300) -> int:
    """Fail closed after interrupted side effects; only pre-ticket claims may retry."""

    cutoff = (datetime.now(UTC) - timedelta(seconds=older_than_seconds)).isoformat()
    with connect(db_path) as conn:
        cursor = conn.execute(
            """UPDATE artifact_file_outbox SET status='queued',available_at=?,updated_at=?,last_error='recovered pre-ticket claim'
               WHERE status='ticket_acquiring' AND updated_at<? AND slack_file_id IS NULL""",
            (utcnow(), utcnow(), cutoff),
        )
        conn.execute(
            """UPDATE artifact_file_outbox SET status='unknown',updated_at=?,last_error='interrupted after ticket acquisition'
               WHERE status IN ('ticket_acquired','uploading','bytes_uploaded','completing') AND updated_at<?""",
            (utcnow(), cutoff),
        )
    return cursor.rowcount
