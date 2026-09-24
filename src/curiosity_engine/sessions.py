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
            if event_id is not None:
                self._require_event_session(conn, session_id, event_id)
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
            if event_id is not None:
                conn.execute(
                    """INSERT INTO agent_message_event_lineage(message_id,event_id)
                       VALUES(?,?)""",
                    (message_id, event_id),
                )
            conn.execute("UPDATE agent_sessions SET updated_at=? WHERE id=?", (now, session_id))
        return message_id

    def attach_event(self, message_id: str, event_id: str) -> None:
        """Bind one already-persisted message to its deterministic source event."""

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            message = conn.execute(
                "SELECT session_id FROM agent_messages WHERE id=?",
                (message_id,),
            ).fetchone()
            if not message:
                raise ValueError("message is unavailable or already has an event")
            self._require_event_session(conn, str(message["session_id"]), event_id)
            changed = conn.execute(
                "UPDATE agent_messages SET event_id=? WHERE id=? AND event_id IS NULL",
                (event_id, message_id),
            ).rowcount
            if changed == 1:
                conn.execute(
                    """INSERT OR IGNORE INTO agent_message_event_lineage(message_id,event_id)
                       VALUES(?,?)""",
                    (message_id, event_id),
                )
        if changed != 1:
            raise ValueError("message is unavailable or already has an event")

    def bind_event_lineage(
        self,
        *,
        session_id: str,
        turn_id: str,
        capability_run_id: str,
        event_id: str,
        source_message_id: str | None = None,
        primary: bool = True,
    ) -> None:
        """Bind every durable copy produced by one parent-agent turn to its event."""

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_event_session(conn, session_id, event_id)
            ownership = conn.execute(
                """SELECT e.child_id AS event_child_id,s.child_id AS session_child_id
                   FROM events e JOIN agent_sessions s ON s.id=? WHERE e.id=?""",
                (session_id, event_id),
            ).fetchone()
            if not ownership:
                raise ValueError("event or parent-agent session is unavailable")
            if ownership["event_child_id"] != ownership["session_child_id"]:
                raise ValueError("event does not belong to the parent-agent session child")
            if source_message_id is not None:
                message = conn.execute(
                    "SELECT event_id FROM agent_messages WHERE id=? AND session_id=?",
                    (source_message_id, session_id),
                ).fetchone()
                if not message:
                    raise ValueError("parent-agent source message is unavailable")
                conn.execute(
                    """INSERT OR IGNORE INTO agent_message_event_lineage(message_id,event_id)
                       VALUES(?,?)""",
                    (source_message_id, event_id),
                )
                if primary:
                    conn.execute(
                        "UPDATE agent_messages SET event_id=? WHERE id=? AND session_id=?",
                        (event_id, source_message_id, session_id),
                    )
            turn = conn.execute(
                "SELECT 1 FROM agent_turns WHERE id=? AND session_id=?",
                (turn_id, session_id),
            ).fetchone()
            if not turn:
                raise ValueError("parent-agent turn is unavailable")
            conn.execute(
                "INSERT OR IGNORE INTO agent_turn_event_lineage(turn_id,event_id) VALUES(?,?)",
                (turn_id, event_id),
            )
            conn.execute(
                """INSERT OR IGNORE INTO tool_call_event_lineage(tool_call_id,event_id)
                   SELECT id,? FROM tool_calls WHERE turn_id=? AND session_id=?""",
                (event_id, turn_id, session_id),
            )
            run = conn.execute(
                "SELECT 1 FROM capability_runs WHERE id=? AND session_id=?",
                (capability_run_id, session_id),
            ).fetchone()
            if not run:
                raise ValueError("parent-agent capability run is unavailable")
            conn.execute(
                """INSERT OR IGNORE INTO capability_run_event_lineage(capability_run_id,event_id)
                   VALUES(?,?)""",
                (capability_run_id, event_id),
            )
            if primary:
                conn.execute(
                    "UPDATE agent_turns SET event_id=? WHERE id=? AND session_id=?",
                    (event_id, turn_id, session_id),
                )
                conn.execute(
                    "UPDATE capability_runs SET produced_event_id=? WHERE id=? AND session_id=?",
                    (event_id, capability_run_id, session_id),
                )
                release_ids = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT id FROM release_units WHERE capability_run_id=?",
                        (capability_run_id,),
                    ).fetchall()
                ]
                conn.execute(
                    "UPDATE release_units SET event_id=? WHERE capability_run_id=?",
                    (event_id, capability_run_id),
                )
                for release_id in release_ids:
                    conn.execute(
                        """INSERT OR IGNORE INTO release_unit_event_lineage(release_unit_id,event_id)
                           VALUES(?,?)""",
                        (release_id, event_id),
                    )

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

    def evidence_safe_history(self, session_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Omit parent-memory wording before thread context enters a child event."""

        history = self.history(session_id, limit=limit)
        with connect(self.db_path) as conn:
            source_rows = conn.execute(
                """SELECT DISTINCT source.source_message_id,source.memory_key,m.ordinal,
                          (SELECT MIN(n.ordinal) FROM agent_messages n
                           WHERE n.session_id=m.session_id AND n.role='user'
                             AND n.ordinal>m.ordinal) AS next_user_ordinal
                   FROM (
                     SELECT source_message_id,memory_key FROM parent_memories
                     WHERE source_message_id IS NOT NULL
                     UNION ALL
                     SELECT source_message_id,'legacy:' || id AS memory_key
                     FROM agent_session_preference_events
                   ) source
                   JOIN agent_messages m ON m.id=source.source_message_id
                   WHERE m.session_id=?""",
                (session_id,),
            ).fetchall()
            memory_sources = {
                str(row["source_message_id"]): str(row["memory_key"]) for row in source_rows
            }
            memory_reply_spans = [
                (int(row["ordinal"]), int(row["next_user_ordinal"]) if row["next_user_ordinal"] else None)
                for row in source_rows
            ]
        safe: list[dict[str, Any]] = []
        for item in history:
            message_id = str(item["id"])
            memory_key = memory_sources.get(message_id)
            ordinal = int(item["ordinal"])
            item_metadata = item.get("metadata") or {}
            is_memory_inspection = (
                item.get("kind") == "parent_memory_inspection"
                or bool(item_metadata.get("memory_refs"))
            )
            is_memory_reply = item["role"] == "assistant" and any(
                source_ordinal < ordinal
                and (next_user_ordinal is None or ordinal < next_user_ordinal)
                for source_ordinal, next_user_ordinal in memory_reply_spans
            )
            if memory_key or is_memory_reply or is_memory_inspection:
                safe.append(
                    {
                        **item,
                        "kind": "parent_memory_operation",
                        "content": "[parent memory wording omitted from child evidence]",
                        "metadata": {
                            "memory_key": memory_key,
                            "source_message_id": message_id if memory_key else None,
                            "inspection_redacted": is_memory_inspection,
                        },
                    }
                )
            else:
                safe.append(item)
        return safe

    def latest_event_id(self, session_id: str) -> str | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT m.event_id FROM agent_messages m
                   JOIN events e ON e.id=m.event_id AND e.status='completed'
                   JOIN responses r ON r.event_id=m.event_id AND r.status='completed'
                   WHERE m.session_id=? AND m.event_id IS NOT NULL
                   ORDER BY m.ordinal DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
        return str(row["event_id"]) if row else None

    def latest_source_event_id(self, session_id: str) -> str | None:
        """Return the newest authorized learning response or context-only source in this thread."""

        with connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT m.event_id FROM agent_messages m
                   JOIN events e ON e.id=m.event_id AND e.status='completed'
                   LEFT JOIN responses r ON r.event_id=e.id AND r.status='completed'
                   WHERE m.session_id=? AND m.event_id IS NOT NULL
                     AND (e.type='context_input' OR r.event_id IS NOT NULL)
                   ORDER BY m.ordinal DESC LIMIT 1""",
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
                    snippet=(f"{artifact.get('artifact_type', 'artifact')}: {title}")[:500],
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
        terms = [
            term for term in "".join(ch if ch.isalnum() else " " for ch in query.casefold()).split() if len(term) >= 3
        ]
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
            if operation == "clear":
                conn.execute(
                    """UPDATE agent_session_preference_events SET value=NULL
                       WHERE session_id=? AND category=? AND operation='set'""",
                    (session_id, category),
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
                elif row["value"]:
                    active[str(row["category"])] = ThreadPreference(
                        category=str(row["category"]),
                        value=str(row["value"]),
                        source_message_id=str(row["source_message_id"]),
                    )
                else:
                    active.pop(str(row["category"]), None)
            session_row = conn.execute("SELECT summary_json FROM agent_sessions WHERE id=?", (session_id,)).fetchone()
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
        """Return the effective legacy preferences with exact source lineage.

        Phase 4 writes new preferences to the parent-memory ledger, but old
        installations can still contain these append-only preference events.
        Reconstructing them from the event log keeps the event ID and value hash
        available for capability-run binding instead of trusting a summary copy.
        """

        with connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id,ordinal,category,operation,value,source_message_id
                   FROM agent_session_preference_events
                   WHERE session_id=? ORDER BY ordinal""",
                (session_id,),
            ).fetchall()
        active: dict[str, dict[str, Any]] = {}
        for row in rows:
            category = str(row["category"])
            if row["operation"] == "clear" or not row["value"]:
                active.pop(category, None)
                continue
            value = str(row["value"])
            active[category] = {
                "category": category,
                "value": value,
                "source_message_id": str(row["source_message_id"]),
                "preference_event_id": str(row["id"]),
                "content_hash": sha256(value.encode()).hexdigest(),
            }
        return list(active.values())

    def active_parent_memories(self, session_id: str) -> list[dict[str, Any]]:
        """Recall explicit parent memory with a visible, code-owned reason."""

        from .adaptation import ParentMemoryLedger

        return ParentMemoryLedger(self.db_path).recall(session_id=session_id)[:24]

    def compacted_context(
        self,
        session_id: str,
        *,
        compact_after: int = 16,
        keep_recent: int = 6,
    ) -> dict[str, Any]:
        """Return inspectable structured continuity plus a few raw turns."""

        from .adaptation import ThreadCompactor

        return ThreadCompactor(self.db_path).context(
            session_id,
            compact_after=compact_after,
            keep_recent=keep_recent,
        )

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
        event_id: str | None = None,
        source_message_id: str | None = None,
        capability_run_id: str | None = None,
    ) -> str:
        call_id = f"tool_{uuid4().hex[:20]}"
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if event_id is not None:
                self._require_event_session(conn, session_id, event_id)
            conn.execute(
                """INSERT INTO tool_calls(
                       id,session_id,turn_id,ordinal,tool_name,tool_version,argument_hash,arguments_json,risk,
                       decision,status,result_json,created_at,updated_at,event_id
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    event_id,
                ),
            )
            if event_id is not None:
                conn.execute(
                    """INSERT INTO tool_call_event_lineage(tool_call_id,event_id)
                       VALUES(?,?)""",
                    (call_id, event_id),
                )
                if source_message_id is not None and capability_run_id is not None:
                    message = conn.execute(
                        """SELECT 1 FROM agent_messages
                           WHERE id=? AND session_id=? AND role='user'""",
                        (source_message_id, session_id),
                    ).fetchone()
                    turn = conn.execute(
                        "SELECT 1 FROM agent_turns WHERE id=? AND session_id=?",
                        (turn_id, session_id),
                    ).fetchone()
                    run = conn.execute(
                        "SELECT 1 FROM capability_runs WHERE id=? AND session_id=?",
                        (capability_run_id, session_id),
                    ).fetchone()
                    if not message or not turn or not run:
                        raise ValueError("prebound tool lineage is unavailable")
                    conn.execute(
                        "UPDATE agent_messages SET event_id=? WHERE id=?",
                        (event_id, source_message_id),
                    )
                    conn.execute("UPDATE agent_turns SET event_id=? WHERE id=?", (event_id, turn_id))
                    conn.execute(
                        "UPDATE capability_runs SET produced_event_id=? WHERE id=?",
                        (event_id, capability_run_id),
                    )
                    for table, column, record_id in (
                        ("agent_message_event_lineage", "message_id", source_message_id),
                        ("agent_turn_event_lineage", "turn_id", turn_id),
                        ("capability_run_event_lineage", "capability_run_id", capability_run_id),
                    ):
                        conn.execute(
                            f"INSERT OR IGNORE INTO {table}({column},event_id) VALUES(?,?)",
                            (record_id, event_id),
                        )
        return call_id

    def finish_tool_call(
        self,
        call_id: str,
        *,
        status: str,
        result: dict[str, Any],
        event_id: str | None = None,
    ) -> None:
        """Finish a pre-persisted tool call without opening a lineage crash gap."""

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT session_id FROM tool_calls WHERE id=?",
                (call_id,),
            ).fetchone()
            if not row:
                raise ValueError("tool call is unavailable")
            if event_id is not None:
                self._require_event_session(conn, str(row["session_id"]), event_id)
            conn.execute(
                """UPDATE tool_calls SET status=?,result_json=?,event_id=COALESCE(?,event_id),updated_at=?
                   WHERE id=?""",
                (status, jdump(result), event_id, utcnow(), call_id),
            )
            if event_id is not None:
                conn.execute(
                    """INSERT OR IGNORE INTO tool_call_event_lineage(tool_call_id,event_id)
                       VALUES(?,?)""",
                    (call_id, event_id),
                )

    @staticmethod
    def _require_event_session(conn: Any, session_id: str, event_id: str) -> None:
        owned = conn.execute(
            """SELECT 1 FROM agent_sessions s JOIN events e ON e.id=?
               WHERE s.id=? AND s.child_id=e.child_id""",
            (event_id, session_id),
        ).fetchone()
        if not owned:
            raise ValueError("event does not belong to the parent-agent session child")

    @staticmethod
    def _session(row: Any) -> dict[str, Any]:
        return {**dict(row), "summary": jload(row["summary_json"])}
