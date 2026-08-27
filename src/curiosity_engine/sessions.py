from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import ParentSessionState, ThreadOutputRef, ThreadPreference
from .db import connect, init_db, jdump, jload, utcnow

THREAD_PREFERENCE_CATEGORIES = {
    "answer_style",
    "visual_style",
    "artifact_style",
    "activity_style",
    "interaction_style",
}


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

    def output_index(self, session_id: str, *, limit: int = 12) -> list[dict[str, Any]]:
        """Return bounded, code-owned references to outputs in one authorized thread."""

        bounded_limit = max(1, min(limit, 40))
        refs: list[ThreadOutputRef] = []
        with connect(self.db_path) as conn:
            messages = conn.execute(
                """SELECT id,kind,content,event_id FROM agent_messages
                   WHERE session_id=? AND role='assistant' ORDER BY ordinal DESC LIMIT ?""",
                (session_id, bounded_limit * 3),
            ).fetchall()
            artifact_calls = conn.execute(
                """SELECT result_json FROM tool_calls
                   WHERE session_id=? AND tool_name='create_learning_artifact' AND status='succeeded'
                   ORDER BY created_at DESC LIMIT ?""",
                (session_id, bounded_limit),
            ).fetchall()
        for row in messages:
            content = " ".join(str(row["content"]).split())
            if not content:
                continue
            refs.append(
                ThreadOutputRef(
                    ref_id=str(row["id"]),
                    kind="answer",
                    title=str(row["kind"]).replace("_", " ")[:180],
                    snippet=content[:500],
                    event_id=str(row["event_id"]) if row["event_id"] else None,
                )
            )
        artifact_refs: list[ThreadOutputRef] = []
        for row in artifact_calls:
            result = jload(row["result_json"])
            artifact = result.get("artifact") if isinstance(result, dict) else None
            if not isinstance(artifact, dict) or not artifact.get("artifact_id"):
                continue
            artifact_id = str(artifact["artifact_id"])
            title = str(artifact.get("title") or artifact.get("artifact_type") or "learning artifact")
            artifact_refs.append(
                ThreadOutputRef(
                    ref_id=artifact_id,
                    kind="artifact",
                    title=title[:180],
                    snippet=(
                        f"{artifact.get('artifact_type', 'artifact')}: {title}"
                    )[:500],
                    event_id=str(result["event_id"]) if result.get("event_id") else None,
                    artifact_id=artifact_id,
                )
            )
        refs = [*artifact_refs, *refs]
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ref in refs:
            if ref.ref_id in seen:
                continue
            seen.add(ref.ref_id)
            deduped.append(ref.model_dump(mode="json"))
            if len(deduped) >= bounded_limit:
                break
        return deduped

    def search_outputs(
        self,
        session_id: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search only the current thread's bounded output index without model calls."""

        candidates = self.output_index(session_id, limit=40)
        terms = [term for term in "".join(ch if ch.isalnum() else " " for ch in query.casefold()).split() if len(term) >= 3]
        if not terms:
            return candidates[: max(1, min(limit, 5))]
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for position, item in enumerate(candidates):
            haystack = f"{item.get('title') or ''} {item.get('snippet') or ''}".casefold()
            score = sum(1 for term in terms if term in haystack)
            if score:
                ranked.append((score, -position, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return [item for _score, _position, item in ranked[: max(1, min(limit, 5))]]

    def resolve_output_ref(self, session_id: str, ref_id: str) -> dict[str, Any]:
        for item in self.output_index(session_id, limit=40):
            if item["ref_id"] == ref_id:
                return item
        raise ValueError("output reference is not available in this thread")

    def update_thread_preference(
        self,
        session_id: str,
        *,
        operation: str,
        category: str,
        value: str | None,
        source_message_id: str,
    ) -> dict[str, Any]:
        if operation not in {"set", "clear"}:
            raise ValueError("preference operation must be set or clear")
        if category not in THREAD_PREFERENCE_CATEGORIES:
            raise ValueError("unsupported thread preference category")
        normalized = " ".join(str(value or "").split())
        if operation == "set" and not normalized:
            raise ValueError("a preference value is required")
        if len(normalized) > 400:
            raise ValueError("thread preference is too long")
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            source = conn.execute(
                "SELECT 1 FROM agent_messages WHERE id=? AND session_id=? AND role='user'",
                (source_message_id, session_id),
            ).fetchone()
            if not source:
                raise ValueError("preference source message is unavailable in this thread")
            ordinal = int(
                conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM agent_session_preference_events WHERE session_id=?",
                    (session_id,),
                ).fetchone()[0]
            )
            conn.execute(
                """INSERT INTO agent_session_preference_events(
                       id,session_id,ordinal,category,operation,value,source_message_id,created_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    f"pref_{uuid4().hex[:20]}",
                    session_id,
                    ordinal,
                    category,
                    operation,
                    normalized if operation == "set" else None,
                    source_message_id,
                    now,
                ),
            )
            rows = conn.execute(
                """SELECT category,operation,value,source_message_id
                   FROM agent_session_preference_events WHERE session_id=? ORDER BY ordinal""",
                (session_id,),
            ).fetchall()
            active: dict[str, ThreadPreference] = {}
            for row in rows:
                if row["operation"] == "clear":
                    active.pop(str(row["category"]), None)
                else:
                    active[str(row["category"])] = ThreadPreference(
                        category=str(row["category"]),
                        value=str(row["value"]),
                        source_message_id=str(row["source_message_id"]),
                    )
            session_row = conn.execute(
                "SELECT summary_json FROM agent_sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not session_row:
                raise KeyError(session_id)
            summary = jload(session_row["summary_json"])
            prior = summary.get("parent_agent") if isinstance(summary, dict) else None
            state = ParentSessionState.model_validate(
                {
                    **(prior if isinstance(prior, dict) else {}),
                    "version": 1,
                    "preferences": [item.model_dump(mode="json") for item in active.values()],
                }
            )
            summary = dict(summary) if isinstance(summary, dict) else {}
            summary["parent_agent"] = state.model_dump(mode="json")
            conn.execute(
                "UPDATE agent_sessions SET summary_json=?,updated_at=? WHERE id=?",
                (jdump(summary), now, session_id),
            )
        return {
            "status": "completed",
            "operation": operation,
            "category": category,
            "preferences": [item.model_dump(mode="json") for item in active.values()],
        }

    def active_preferences(self, session_id: str) -> list[dict[str, Any]]:
        session = self.session(session_id)
        summary = session.get("summary") or {}
        raw = summary.get("parent_agent") if isinstance(summary, dict) else None
        state = ParentSessionState.model_validate(raw or {})
        return [item.model_dump(mode="json") for item in state.preferences]

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
