from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .db import connect, init_db, jdump, jload, utcnow
from .transports.contracts import InboundMessage


def persist_inbound_attachments(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    binding_id: str,
    message: InboundMessage,
) -> list[dict[str, Any]]:
    """Persist only validated private attachment references from a paired transport."""

    init_db(db_path)
    allowed_root = (Path(output_dir).resolve() / "inbound").resolve()
    now = utcnow()
    with connect(db_path) as conn:
        for attachment in message.attachments:
            path: str | None = None
            if attachment.status == "ready":
                if not attachment.private_path or not attachment.sha256 or not attachment.media_type:
                    raise ValueError("ready attachment is incomplete")
                candidate = Path(attachment.private_path).resolve()
                if not candidate.is_relative_to(allowed_root) or not candidate.is_file():
                    raise ValueError("inbound attachment is outside the private input root")
                data = candidate.read_bytes()
                if len(data) != attachment.byte_count or sha256(data).hexdigest() != attachment.sha256:
                    raise ValueError("inbound attachment changed after validation")
                path = str(candidate)
            conn.execute(
                """INSERT INTO inbound_assets(
                       id,binding_id,external_event_id,external_ref_hash,status,media_type,byte_count,sha256,path,
                       observation_json,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,'{}',?,?)
                   ON CONFLICT(binding_id,external_event_id,external_ref_hash) DO NOTHING""",
                (
                    f"asset_{uuid4().hex[:20]}",
                    binding_id,
                    message.external_event_id,
                    attachment.external_ref_hash,
                    attachment.status,
                    attachment.media_type,
                    attachment.byte_count,
                    attachment.sha256,
                    path,
                    now,
                    now,
                ),
            )
    return attachments_for_event(db_path, binding_id=binding_id, external_event_id=message.external_event_id)


def attachments_for_event(
    db_path: str | Path, *, binding_id: str, external_event_id: str
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM inbound_assets
               WHERE binding_id=? AND external_event_id=? ORDER BY created_at,id""",
            (binding_id, external_event_id),
        ).fetchall()
    return [_asset(row) for row in rows]


def attachments_for_inbox(db_path: str | Path, inbox_id: str) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM inbound_assets WHERE inbox_id=? ORDER BY created_at,id", (inbox_id,)
        ).fetchall()
    return [_asset(row) for row in rows]


def attachments_for_session(
    db_path: str | Path, session_id: str, *, limit: int = 3
) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT * FROM (
                   SELECT * FROM inbound_assets WHERE session_id=? ORDER BY created_at DESC,id DESC LIMIT ?
               ) ORDER BY created_at,id""",
            (session_id, limit),
        ).fetchall()
    return [_asset(row) for row in rows]


def link_event_attachments(
    db_path: str | Path,
    *,
    binding_id: str,
    external_event_id: str,
    inbox_id: str | None = None,
    session_id: str | None = None,
) -> None:
    if not inbox_id and not session_id:
        return
    assignments: list[str] = []
    values: list[Any] = []
    if inbox_id:
        assignments.append("inbox_id=?")
        values.append(inbox_id)
    if session_id:
        assignments.append("session_id=?")
        values.append(session_id)
    assignments.append("updated_at=?")
    values.append(utcnow())
    values.extend((binding_id, external_event_id))
    with connect(db_path) as conn:
        conn.execute(
            f"UPDATE inbound_assets SET {','.join(assignments)} WHERE binding_id=? AND external_event_id=?",
            values,
        )


def link_assets_to_session(db_path: str | Path, asset_ids: list[str], session_id: str) -> None:
    if not asset_ids:
        return
    placeholders = ",".join("?" for _ in asset_ids)
    with connect(db_path) as conn:
        conn.execute(
            f"UPDATE inbound_assets SET session_id=?,updated_at=? WHERE id IN ({placeholders})",
            (session_id, utcnow(), *asset_ids),
        )


def save_attachment_observation(
    db_path: str | Path, asset_id: str, observation: dict[str, Any]
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE inbound_assets SET observation_json=?,updated_at=? WHERE id=?",
            (jdump(observation), utcnow(), asset_id),
        )


def discard_inbox_attachments(db_path: str | Path, inbox_id: str) -> None:
    """Remove dismissed private inputs when no other message references their bytes."""

    unreferenced: list[str] = []
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id,path FROM inbound_assets WHERE inbox_id=?", (inbox_id,)
        ).fetchall()
        paths = [str(row["path"]) for row in rows if row["path"]]
        conn.execute("DELETE FROM inbound_assets WHERE inbox_id=?", (inbox_id,))
        for value in paths:
            referenced = conn.execute(
                "SELECT 1 FROM inbound_assets WHERE path=? LIMIT 1", (value,)
            ).fetchone()
            if not referenced:
                unreferenced.append(value)
    for value in unreferenced:
        Path(value).unlink(missing_ok=True)


def _asset(row: Any) -> dict[str, Any]:
    return {**dict(row), "observation": jload(row["observation_json"])}
