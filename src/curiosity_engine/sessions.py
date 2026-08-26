from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .db import connect, init_db, jdump, jload, utcnow


class SessionStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        init_db(self.db_path)

    def get_or_create(
        self,
        *,
        origin: str,
        conversation_ref: str,
        thread_ref: str,
        binding_id: str | None = None,
        transport: str | None = None,
        child_id: str | None = None,
        capability_id: str = "parent_chat",
    ) -> dict[str, Any]:
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT * FROM agent_sessions
                   WHERE origin=? AND binding_id IS ? AND conversation_ref=? AND thread_ref=?""",
                (origin, binding_id, conversation_ref, thread_ref),
            ).fetchone()
            if row:
                if child_id and not row["child_id"]:
                    conn.execute(
                        "UPDATE agent_sessions SET child_id=?,updated_at=? WHERE id=?",
                        (child_id, now, row["id"]),
                    )
                    row = conn.execute("SELECT * FROM agent_sessions WHERE id=?", (row["id"],)).fetchone()
                return self._session(row)
            session_id = f"ses_{uuid4().hex[:20]}"
            conn.execute(
                """INSERT INTO agent_sessions(
                       id,origin,transport,binding_id,conversation_ref,thread_ref,child_id,capability_id,state,
                       summary_json,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,'{}',?,?)""",
                (
                    session_id,
                    origin,
                    transport,
                    binding_id,
                    conversation_ref,
                    thread_ref,
                    child_id,
                    capability_id,
                    "active",
                    now,
                    now,
                ),
            )
            return self._session(conn.execute("SELECT * FROM agent_sessions WHERE id=?", (session_id,)).fetchone())

    def session(self, session_id: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM agent_sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            raise KeyError(session_id)
        return self._session(row)

    def find(
        self,
        *,
        origin: str,
        binding_id: str | None,
        conversation_ref: str,
        thread_ref: str,
    ) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT * FROM agent_sessions
                   WHERE origin=? AND binding_id IS ? AND conversation_ref=? AND thread_ref=?""",
                (origin, binding_id, conversation_ref, thread_ref),
            ).fetchone()
        return self._session(row) if row else None

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        kind: str = "message",
        event_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        message_id = f"msg_{uuid4().hex[:20]}"
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            ordinal = int(
                conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM agent_messages WHERE session_id=?",
                    (session_id,),
                ).fetchone()[0]
            )
            conn.execute(
                """INSERT INTO agent_messages(
                       id,session_id,ordinal,role,kind,content,event_id,metadata_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (message_id, session_id, ordinal, role, kind, content, event_id, jdump(metadata or {}), now),
            )
            conn.execute("UPDATE agent_sessions SET updated_at=? WHERE id=?", (now, session_id))
        return message_id

    def history(self, session_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT * FROM (
                       SELECT * FROM agent_messages WHERE session_id=? ORDER BY ordinal DESC LIMIT ?
                   ) ORDER BY ordinal""",
                (session_id, limit),
            ).fetchall()
        return [
            {
                **dict(row),
                "metadata": jload(row["metadata_json"]),
            }
            for row in rows
        ]

    def latest_event_id(self, session_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT event_id FROM agent_messages
                   WHERE session_id=? AND event_id IS NOT NULL ORDER BY ordinal DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
        return str(row["event_id"]) if row else None

    def start_turn(self, session_id: str, request: dict[str, Any], *, policy_hash: str) -> dict[str, Any]:
        turn_id = f"turn_{uuid4().hex[:20]}"
        now = utcnow()
        request_hash = sha256(jdump(request).encode()).hexdigest()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            ordinal = int(
                conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM agent_turns WHERE session_id=?", (session_id,)
                ).fetchone()[0]
            )
            conn.execute(
                """INSERT INTO agent_turns(
                       id,session_id,ordinal,status,request_hash,policy_hash,started_at
                   ) VALUES(?,?,?,'running',?,?,?)""",
                (turn_id, session_id, ordinal, request_hash, policy_hash, now),
            )
        return {"id": turn_id, "ordinal": ordinal, "request_hash": request_hash}

    def finish_turn(
        self,
        turn_id: str,
        *,
        response: dict[str, Any],
        provider: str | None,
        model: str | None,
        status: str = "completed",
    ) -> None:
        with connect(self.db_path) as conn:
            conn.execute(
                """UPDATE agent_turns SET provider=?,model=?,status=?,response_json=?,completed_at=? WHERE id=?""",
                (provider, model, status, jdump(response), utcnow(), turn_id),
            )

    def record_tool_call(
        self,
        *,
        session_id: str,
        turn_id: str,
        ordinal: int,
        tool_name: str,
        tool_version: str,
        arguments: dict[str, Any],
        risk: str,
        decision: str,
        status: str,
        result: dict[str, Any] | None = None,
    ) -> str:
        call_id = f"tool_{uuid4().hex[:20]}"
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO tool_calls(
                       id,session_id,turn_id,ordinal,tool_name,tool_version,argument_hash,arguments_json,risk,
                       decision,status,result_json,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    call_id,
                    session_id,
                    turn_id,
                    ordinal,
                    tool_name,
                    tool_version,
                    sha256(jdump(arguments).encode()).hexdigest(),
                    jdump(arguments),
                    risk,
                    decision,
                    status,
                    jdump(result or {}),
                    now,
                    now,
                ),
            )
        return call_id

    @staticmethod
    def _session(row: Any) -> dict[str, Any]:
        return {**dict(row), "summary": jload(row["summary_json"])}
