from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .claims import revalidate_established_claims
from .contracts import Event
from .db import connect, init_db, jdump, jload, utcnow

CLUSTERING_VERSION = "deterministic-v1"
VALID_CORRECTIONS = {"retry", "deepening", "new_episode", "exclude"}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "the",
    "to",
    "what",
    "why",
}


def _canonical(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _tokens(text: str) -> set[str]:
    return {token for token in _canonical(text).split() if token not in STOP_WORDS}


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 1.0 if _canonical(left) == _canonical(right) else 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _learning_scope(event: Event) -> str:
    configured = str(event.metadata.get("learning_scope") or "")
    if configured in {"family_signal", "diagnostic", "system"}:
        return configured
    source = event.source.casefold()
    if source in {"eval", "test", "diagnostic", "live_eval", "public_eval"}:
        return "diagnostic"
    if source in {"system", "weekly_director", "scheduled_reflection"}:
        return "system"
    return "family_signal"


def _new_episode(
    conn: sqlite3.Connection,
    event: Event,
    *,
    now: str,
    relation: str,
) -> dict[str, Any]:
    episode_id = f"ep_{uuid4().hex[:16]}"
    topic_key = sha256(_canonical(event.text).encode()).hexdigest()[:20]
    conn.execute(
        """INSERT INTO episodes(
           id,child_id,status,topic_key,summary,clustering_version,opened_at,last_event_at,created_at,updated_at
           ) VALUES(?,?,'active',?,?,?,?,?,?,?)""",
        (
            episode_id,
            event.child_id,
            topic_key,
            event.text,
            CLUSTERING_VERSION,
            event.created_at.isoformat(),
            event.created_at.isoformat(),
            now,
            now,
        ),
    )
    return {"episode_id": episode_id, "relation": relation, "new_episode": True}


def resolve_episode(conn: sqlite3.Connection, event: Event) -> dict[str, Any] | None:
    """Attach an event to a conservative episode before any learner-state inference.

    Time narrows candidates but never establishes semantic independence by itself. Ambiguous
    related turns remain in one episode so they cannot accidentally become repeated evidence.
    """

    if not event.child_id:
        return None
    now = utcnow()
    scope = _learning_scope(event)
    explicit = str(event.metadata.get("episode_relation") or "").casefold()
    if explicit not in {"", "new_episode", "retry", "deepening", "clarification"}:
        explicit = ""
    event_time = event.created_at.astimezone(UTC)
    cutoff = (event_time - timedelta(days=90)).isoformat()
    rows = conn.execute(
        """SELECT m.episode_id,m.relation,m.learning_scope,e.id AS event_id,e.text,e.status,e.created_at,e.metadata_json,
                  EXISTS(
                      SELECT 1 FROM runs r
                      WHERE r.event_id=e.id AND r.status IN ('failed','rejected')
                  ) AS answer_failed
           FROM episode_memberships m JOIN events e ON e.id=m.event_id
           WHERE e.child_id=? AND e.id!=? AND e.created_at>=?
           ORDER BY e.created_at DESC LIMIT 30""",
        (event.child_id, event.id, cutoff),
    ).fetchall()

    chosen: sqlite3.Row | None = None
    relation = "initial_question"
    rationale = "No sufficiently related prior event was found."
    confidence = 1.0
    current_canonical = _canonical(event.text)
    current_conversation = str(event.metadata.get("conversation_ref") or "")
    current_thread = str(event.metadata.get("thread_ref") or "")

    if explicit == "new_episode":
        relation = "parent_marked_new_episode"
        rationale = "The parent explicitly marked a new learning occasion."
    else:
        for row in rows:
            prior_time = _parse_time(str(row["created_at"]))
            age = event_time - prior_time
            if age.total_seconds() < 0:
                continue
            prior_metadata = jload(row["metadata_json"])
            prior_conversation = str(prior_metadata.get("conversation_ref") or "")
            prior_thread = str(prior_metadata.get("thread_ref") or "")
            exact = current_canonical == _canonical(str(row["text"]))
            similarity = _similarity(event.text, str(row["text"]))
            same_thread = bool(current_thread and current_thread == prior_thread)
            same_conversation = bool(current_conversation and current_conversation == prior_conversation)
            if explicit in {"retry", "deepening", "clarification"} and (same_thread or same_conversation):
                chosen = row
                relation = f"parent_marked_{explicit}"
                rationale = "The parent explicitly classified this turn relative to the conversation."
                confidence = 1.0
                break
            if exact and age <= timedelta(hours=24):
                chosen = row
                relation = (
                    "answer_repair"
                    if row["status"] in {"failed", "rejected"} or bool(row["answer_failed"])
                    else "exact_retry"
                )
                rationale = "The same question recurred within one day; it remains one evidence episode."
                confidence = 1.0
                break
            if same_thread and age <= timedelta(hours=24):
                chosen = row
                relation = "related_followup"
                rationale = "The turn continued an explicit Slack thread within one day."
                confidence = 0.95
                break
            if same_conversation and similarity >= 0.65 and age <= timedelta(hours=2):
                chosen = row
                relation = (
                    "answer_repair"
                    if row["status"] in {"failed", "rejected"} or bool(row["answer_failed"])
                    else "related_followup"
                )
                rationale = "A strongly similar turn arrived close in time in the same conversation."
                confidence = 0.85
                break
            if exact:
                chosen = row
                relation = "later_repeat_uncertain"
                rationale = (
                    "Elapsed time alone cannot distinguish renewed interest from retry or dissatisfaction; "
                    "the exact repeat remains in one episode until a parent corrects it."
                )
                confidence = 0.7
                break
            if similarity >= 0.65 and age > timedelta(hours=24):
                relation = "related_return"
                rationale = (
                    "A meaningfully developed related question returned on a later occasion; "
                    "it starts a provisional new episode."
                )
                confidence = 0.75
                break

        if chosen is None and relation == "initial_question":
            exact_prior = conn.execute(
                """SELECT m.episode_id,m.relation,m.learning_scope,e.id AS event_id,e.text,e.status,
                          e.created_at,e.metadata_json,
                          EXISTS(
                              SELECT 1 FROM runs r
                              WHERE r.event_id=e.id AND r.status IN ('failed','rejected')
                          ) AS answer_failed
                   FROM episodes ep
                   JOIN episode_memberships m ON m.episode_id=ep.id
                   JOIN events e ON e.id=m.event_id
                   WHERE ep.child_id=? AND ep.topic_key=? AND e.id!=?
                   ORDER BY e.created_at DESC LIMIT 1""",
                (event.child_id, sha256(current_canonical.encode()).hexdigest()[:20], event.id),
            ).fetchone()
            if exact_prior:
                chosen = exact_prior
                relation = "later_repeat_uncertain"
                rationale = (
                    "Elapsed time alone cannot distinguish renewed interest from retry or dissatisfaction; "
                    "the exact repeat remains in one episode until a parent corrects it."
                )
                confidence = 0.65

    if chosen is None:
        episode = _new_episode(conn, event, now=now, relation=relation)
    else:
        episode = {"episode_id": str(chosen["episode_id"]), "relation": relation, "new_episode": False}
        conn.execute(
            "UPDATE episodes SET last_event_at=?,updated_at=? WHERE id=?",
            (event.created_at.isoformat(), now, episode["episode_id"]),
        )

    if scope == "diagnostic":
        independence = "diagnostic"
    elif scope == "system":
        independence = "system"
    elif relation == "later_repeat_uncertain":
        independence = "uncertain"
    elif episode["new_episode"]:
        independence = "eligible"
    else:
        independence = "same_episode"
    conn.execute(
        """INSERT INTO episode_memberships(
           episode_id,event_id,relation,independence_status,learning_scope,confidence,rationale,
           classifier_source,classifier_version,created_at,updated_at
           ) VALUES(?,?,?,?,?,?,?,'deterministic',?,?,?)""",
        (
            episode["episode_id"],
            event.id,
            relation,
            independence,
            scope,
            confidence,
            rationale,
            CLUSTERING_VERSION,
            now,
            now,
        ),
    )
    return {
        **episode,
        "independence_status": independence,
        "learning_scope": scope,
        "confidence": confidence,
        "rationale": rationale,
    }


def episode_for_event(db_path: str | Path, event_id: str) -> dict[str, Any] | None:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT episode_id,relation,independence_status,learning_scope,confidence,rationale,
                      classifier_source,classifier_version
               FROM episode_memberships WHERE event_id=?""",
            (event_id,),
        ).fetchone()
    return dict(row) if row else None


def child_episode_context(db_path: str | Path, child_id: str, *, limit: int = 30) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as conn:
        episodes = [
            dict(row)
            for row in conn.execute(
                """SELECT id,status,summary,clustering_version,opened_at,last_event_at
                   FROM episodes WHERE child_id=? ORDER BY last_event_at DESC LIMIT ?""",
                (child_id, limit),
            )
        ]
        episode_ids = [item["id"] for item in episodes]
        members: list[dict[str, Any]] = []
        if episode_ids:
            placeholders = ",".join("?" for _ in episode_ids)
            members = [
                dict(row)
                for row in conn.execute(
                    f"""SELECT m.episode_id,m.event_id,m.relation,m.independence_status,m.learning_scope,
                               m.confidence,m.rationale,e.text,e.source,e.status,e.created_at
                        FROM episode_memberships m JOIN events e ON e.id=m.event_id
                        WHERE m.episode_id IN ({placeholders}) ORDER BY e.created_at""",
                    episode_ids,
                )
            ]
        correction_count = int(
            conn.execute("SELECT COUNT(*) FROM context_corrections WHERE child_id=?", (child_id,)).fetchone()[0]
        )
        legacy_unreviewed_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM evidence WHERE child_id=? AND event_id IS NOT NULL AND episode_id IS NULL",
                (child_id,),
            ).fetchone()[0]
        )
    by_episode: dict[str, list[dict[str, Any]]] = {item["id"]: [] for item in episodes}
    for member in members:
        by_episode[member["episode_id"]].append(member)
    for episode in episodes:
        episode["events"] = by_episode[episode["id"]]
    return {
        "episodes": episodes,
        "graph_health": {
            "episode_count": len(episodes),
            "eligible_episode_count": sum(
                member["independence_status"] == "eligible" for member in members
            ),
            "same_episode_turn_count": sum(
                member["independence_status"] == "same_episode" for member in members
            ),
            "excluded_or_non_family_count": sum(
                member["independence_status"] in {"diagnostic", "system", "excluded"} for member in members
            ),
            "uncertain_count": sum(member["independence_status"] == "uncertain" for member in members),
            "parent_correction_count": correction_count,
            "legacy_unreviewed_count": legacy_unreviewed_count,
            "frequency_is_not_interest": True,
            "durable_interest_inference_enabled": False,
            "context_driven_proactivity_enabled": False,
        },
    }


def apply_episode_correction(
    db_path: str | Path,
    *,
    child_id: str,
    event_id: str,
    action: str,
    related_event_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Append a parent correction and update only the active episode projection."""

    if action not in VALID_CORRECTIONS:
        raise ValueError("context correction must be retry, deepening, new_episode, or exclude")
    if action in {"retry", "deepening"} and not related_event_id:
        raise ValueError(f"{action} requires --related-event")
    init_db(db_path)
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT m.*,e.text,e.created_at FROM episode_memberships m JOIN events e ON e.id=m.event_id
               WHERE m.event_id=? AND e.child_id=?""",
            (event_id, child_id),
        ).fetchone()
        if not row:
            raise ValueError("event is not part of this child's context graph")
        target_episode = str(row["episode_id"])
        if related_event_id:
            related = conn.execute(
                """SELECT m.episode_id FROM episode_memberships m JOIN events e ON e.id=m.event_id
                   WHERE m.event_id=? AND e.child_id=?""",
                (related_event_id, child_id),
            ).fetchone()
            if not related:
                raise ValueError("related event is not part of this child's context graph")
            target_episode = str(related["episode_id"])
        previous = {
            "episode_id": row["episode_id"],
            "relation": row["relation"],
            "independence_status": row["independence_status"],
        }
        if action == "new_episode":
            new_id = f"ep_{uuid4().hex[:16]}"
            conn.execute(
                """INSERT INTO episodes(
                   id,child_id,status,topic_key,summary,clustering_version,opened_at,last_event_at,created_at,updated_at
                   ) VALUES(?,?,'active',?,?,?,?,?,?,?)""",
                (
                    new_id,
                    child_id,
                    sha256(_canonical(str(row["text"])).encode()).hexdigest()[:20],
                    row["text"],
                    "parent-corrected-v1",
                    row["created_at"],
                    row["created_at"],
                    now,
                    now,
                ),
            )
            target_episode = new_id
            relation = "parent_marked_new_episode"
            independence = "eligible" if row["learning_scope"] == "family_signal" else row["independence_status"]
        elif action == "exclude":
            relation = "parent_excluded"
            independence = "excluded"
        else:
            relation = f"parent_marked_{action}"
            independence = "same_episode"
        conn.execute(
            """UPDATE episode_memberships SET episode_id=?,relation=?,independence_status=?,confidence=1,
               rationale='Parent correction overrides deterministic grouping.',classifier_source='parent',
               classifier_version='parent-v1',updated_at=? WHERE event_id=?""",
            (target_episode, relation, independence, now, event_id),
        )
        conn.execute("UPDATE evidence SET episode_id=? WHERE event_id=?", (target_episode, event_id))
        observation_rows = conn.execute(
            "SELECT id,metadata_json FROM observations WHERE event_id=?", (event_id,)
        ).fetchall()
        for observation in observation_rows:
            metadata = jload(observation["metadata_json"])
            metadata.update(
                {
                    "episode_id": target_episode,
                    "relation": relation,
                    "independence_status": independence,
                    "classifier_source": "parent",
                    "classifier_version": "parent-v1",
                }
            )
            conn.execute(
                "UPDATE observations SET metadata_json=? WHERE id=?",
                (jdump(metadata), observation["id"]),
            )
        node_rows = conn.execute("SELECT id,state_json FROM nodes WHERE child_id=?", (child_id,)).fetchall()
        for node in node_rows:
            state = jload(node["state_json"])
            if state.get("source_event_id") != event_id and state.get("event_id") != event_id:
                continue
            state.update(
                {
                    "episode_id": target_episode,
                    "relation": relation,
                    "independence_status": independence,
                    "classifier_source": "parent",
                    "classifier_version": "parent-v1",
                }
            )
            conn.execute("UPDATE nodes SET state_json=? WHERE id=?", (jdump(state), node["id"]))
        conn.execute("UPDATE episodes SET status='corrected',updated_at=? WHERE id=?", (now, row["episode_id"]))
        conn.execute(
            """INSERT INTO context_corrections(
               child_id,event_id,action,related_event_id,previous_json,note,created_at
               ) VALUES(?,?,?,?,?,?,?)""",
            (child_id, event_id, action, related_event_id, jdump(previous), note, now),
        )
        correction_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        downgraded_claim_ids = revalidate_established_claims(conn, child_id)
    return {
        "correction_id": correction_id,
        "event_id": event_id,
        "action": action,
        "episode_id": target_episode,
        "relation": relation,
        "independence_status": independence,
        "downgraded_claim_ids": downgraded_claim_ids,
    }
