from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import GraphMutation, GraphMutationKind
from .db import connect, jdump, jload, utcnow
from .episodes import child_episode_context


def key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def add_child(
    db: str | Path, child_id: str, name: str, birth_year: int | None = None, grade: str | None = None
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", child_id):
        raise ValueError("child_id may contain only letters, numbers, underscores, and hyphens")
    if not name.strip():
        raise ValueError("child name cannot be empty")
    with connect(db) as conn:
        now = utcnow()
        conn.execute(
            """INSERT INTO children(id,name,birth_year,grade,created_at,updated_at) VALUES(?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name,birth_year=excluded.birth_year,
               grade=excluded.grade,updated_at=excluded.updated_at""",
            (child_id, name, birth_year, grade, now, now),
        )


def upsert_node(
    db: str | Path, child_id: str, kind: str, label: str, confidence: float = 0.6, state: dict[str, Any] | None = None
) -> int:
    now = utcnow()
    canonical = key(label)
    with connect(db) as conn:
        row = conn.execute(
            "SELECT id,state_json FROM nodes WHERE child_id=? AND kind=? AND canonical_key=?",
            (child_id, kind, canonical),
        ).fetchone()
        if row:
            old_state = jload(row["state_json"])
            old_state.update(state or {})
            conn.execute(
                "UPDATE nodes SET label=?,last_seen=?,evidence_count=evidence_count+1,confidence=MAX(confidence,?),state_json=? WHERE id=?",
                (label, now, confidence, jdump(old_state), row["id"]),
            )
            return int(row["id"])
        cur = conn.execute(
            "INSERT INTO nodes(child_id,kind,label,canonical_key,state_json,first_seen,last_seen,evidence_count,confidence) VALUES(?,?,?,?,?,?,?,?,?)",
            (child_id, kind, label, canonical, jdump(state or {}), now, now, 1, confidence),
        )
        return int(cur.lastrowid)


def add_edge(
    db: str | Path, child_id: str, source: int, relation: str, target: int, metadata: dict[str, Any] | None = None
) -> None:
    now = utcnow()
    with connect(db) as conn:
        row = conn.execute(
            "SELECT id,metadata_json FROM edges WHERE child_id=? AND source_node_id=? AND relation=? AND target_node_id=?",
            (child_id, source, relation, target),
        ).fetchone()
        if row:
            old = jload(row["metadata_json"])
            old.update(metadata or {})
            conn.execute(
                "UPDATE edges SET last_seen=?,evidence_count=evidence_count+1,metadata_json=? WHERE id=?",
                (now, jdump(old), row["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO edges(child_id,source_node_id,relation,target_node_id,metadata_json,created_at,last_seen,evidence_count) VALUES(?,?,?,?,?,?,?,1)",
                (child_id, source, relation, target, jdump(metadata or {}), now, now),
            )


def add_observation(
    db: str | Path,
    child_id: str,
    kind: str,
    text: str,
    source: str = "parent",
    confidence: float = 1.0,
    metadata: dict[str, Any] | None = None,
    event_id: str | None = None,
) -> int:
    with connect(db) as conn:
        cur = conn.execute(
            "INSERT INTO observations(child_id,kind,text,source,confidence,occurred_at,metadata_json,event_id) VALUES(?,?,?,?,?,?,?,?)",
            (child_id, kind, text, source, confidence, utcnow(), jdump(metadata or {}), event_id),
        )
        return int(cur.lastrowid)


def capture_question(
    db: str | Path, child_id: str, question: str, topics: list[str] | None = None, source: str = "parent"
) -> dict[str, Any]:
    add_observation(db, child_id, "curiosity", question, source=source)
    qid = upsert_node(db, child_id, "question", question, confidence=1.0)
    topic_ids = []
    for topic in topics or []:
        tid = upsert_node(db, child_id, "topic", topic, confidence=0.7)
        add_edge(db, child_id, qid, "about", tid)
        topic_ids.append(tid)
    return {"question_node_id": qid, "topic_node_ids": topic_ids}


def add_school_signal(
    db: str | Path,
    child_id: str,
    category: str,
    value: str,
    source_ref: str | None = None,
    expires_at: str | None = None,
    confidence: float = 0.8,
) -> None:
    if expires_at:
        parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        if parsed_expiry.tzinfo is None:
            raise ValueError("expires_at must include a timezone")
        expires_at = parsed_expiry.astimezone(UTC).isoformat()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO school_signals(child_id,category,value,source_ref,observed_at,expires_at,confidence) VALUES(?,?,?,?,?,?,?)",
            (child_id, category, value, source_ref, utcnow(), expires_at, confidence),
        )
    node_kind = "teacher_language" if category == "teacher_language" else "school_concept"
    upsert_node(db, child_id, node_kind, value, confidence=confidence)


def child_context(db: str | Path, child_id: str, limit: int = 100) -> dict[str, Any]:
    with connect(db) as conn:
        child = conn.execute("SELECT * FROM children WHERE id=?", (child_id,)).fetchone()
        if not child:
            raise ValueError(f"Unknown child: {child_id}")
        nodes = conn.execute(
            "SELECT id,kind,label,first_seen,last_seen,evidence_count,confidence,state_json FROM nodes WHERE child_id=? ORDER BY last_seen DESC LIMIT ?",
            (child_id, limit),
        ).fetchall()
        observations = conn.execute(
            "SELECT kind,text,source,confidence,occurred_at,metadata_json FROM observations WHERE child_id=? ORDER BY occurred_at DESC LIMIT ?",
            (child_id, min(limit, 30)),
        ).fetchall()
        school = conn.execute(
            """SELECT category,value,source_ref,observed_at,expires_at,confidence
               FROM school_signals WHERE child_id=? AND (expires_at IS NULL OR expires_at>?)
               ORDER BY observed_at DESC LIMIT 30""",
            (child_id, utcnow()),
        ).fetchall()
        experiences = conn.execute(
            "SELECT id,experience_type,title,status,created_at,feedback FROM experiences WHERE child_id=? ORDER BY created_at DESC LIMIT 20",
            (child_id,),
        ).fetchall()
    episode_context = child_episode_context(db, child_id, limit=min(limit, 30))
    return {
        "child": dict(child),
        "nodes": [{**dict(r), "state": jload(r["state_json"])} for r in nodes],
        "observations": [{**dict(r), "metadata": jload(r["metadata_json"])} for r in observations],
        "school_signals": [dict(r) for r in school],
        "recent_experiences": [dict(r) for r in experiences],
        **episode_context,
    }


def apply_graph_mutation(conn, mutation: GraphMutation, default_child_id: str | None = None) -> dict[str, Any]:
    """Apply one validated, allowlisted graph effect inside the caller's transaction."""

    child_id = mutation.child_id or default_child_id
    if not child_id:
        raise ValueError("graph mutation requires child_id")
    now = utcnow()
    if mutation.kind == GraphMutationKind.ADD_OBSERVATION:
        if not mutation.observation_kind or not mutation.text:
            raise ValueError("add_observation requires observation_kind and text")
        cur = conn.execute(
            """INSERT INTO observations(
               child_id,kind,text,source,confidence,occurred_at,metadata_json,event_id
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                child_id,
                mutation.observation_kind,
                mutation.text,
                mutation.source,
                mutation.confidence,
                now,
                jdump(mutation.state),
                mutation.state.get("event_id"),
            ),
        )
        return {"observation_id": int(cur.lastrowid)}
    if mutation.kind == GraphMutationKind.UPSERT_NODE:
        if not mutation.node_kind or not mutation.label:
            raise ValueError("upsert_node requires node_kind and label")
        canonical = key(mutation.label)
        row = conn.execute(
            "SELECT id,state_json FROM nodes WHERE child_id=? AND kind=? AND canonical_key=?",
            (child_id, mutation.node_kind, canonical),
        ).fetchone()
        if row:
            state = jload(row["state_json"])
            state.update(mutation.state)
            conn.execute(
                """UPDATE nodes SET label=?,last_seen=?,evidence_count=evidence_count+1,
                   confidence=MAX(confidence,?),state_json=? WHERE id=?""",
                (mutation.label, now, mutation.confidence, jdump(state), row["id"]),
            )
            node_id = int(row["id"])
        else:
            cur = conn.execute(
                """INSERT INTO nodes(child_id,kind,label,canonical_key,state_json,first_seen,last_seen,evidence_count,confidence)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    child_id,
                    mutation.node_kind,
                    mutation.label,
                    canonical,
                    jdump(mutation.state),
                    now,
                    now,
                    1,
                    mutation.confidence,
                ),
            )
            node_id = int(cur.lastrowid)
        for evidence_id in mutation.evidence_ids:
            conn.execute(
                "INSERT OR IGNORE INTO node_evidence(node_id,evidence_id,stance,created_at) VALUES(?,?,?,?)",
                (node_id, evidence_id, "supports", now),
            )
        return {"node_id": node_id}
    if mutation.kind == GraphMutationKind.ADD_EDGE:
        if mutation.source_node_id is None or mutation.target_node_id is None or not mutation.relation:
            raise ValueError("add_edge requires source_node_id, target_node_id, and relation")
        conn.execute(
            """INSERT INTO edges(child_id,source_node_id,relation,target_node_id,metadata_json,created_at,last_seen,evidence_count)
               VALUES(?,?,?,?,?,?,?,1)
               ON CONFLICT(child_id,source_node_id,relation,target_node_id) DO UPDATE SET
               last_seen=excluded.last_seen,evidence_count=edges.evidence_count+1,metadata_json=excluded.metadata_json""",
            (
                child_id,
                mutation.source_node_id,
                mutation.relation,
                mutation.target_node_id,
                jdump(mutation.state),
                now,
                now,
            ),
        )
        return {"edge": "applied"}
    if mutation.kind == GraphMutationKind.SET_KNOWLEDGE_STATE:
        if mutation.node_id is None or mutation.knowledge_state is None:
            raise ValueError("set_knowledge_state requires node_id and knowledge_state")
        row = conn.execute(
            "SELECT state_json FROM nodes WHERE id=? AND child_id=?", (mutation.node_id, child_id)
        ).fetchone()
        if not row:
            raise ValueError("knowledge-state target node not found")
        state = jload(row["state_json"])
        state["knowledge_state"] = mutation.knowledge_state
        state["knowledge_state_updated_at"] = now
        conn.execute("UPDATE nodes SET state_json=?,last_seen=? WHERE id=?", (jdump(state), now, mutation.node_id))
        return {"node_id": mutation.node_id, "knowledge_state": mutation.knowledge_state}
    raise ValueError(f"unsupported graph mutation: {mutation.kind}")
