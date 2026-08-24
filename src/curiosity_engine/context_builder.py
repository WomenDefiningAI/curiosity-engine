from __future__ import annotations

import re
from typing import Any

from .db import connect, jload, utcnow
from .resources import search_resources


def _terms(*values: str | None) -> set[str]:
    stop = {
        "about",
        "after",
        "again",
        "because",
        "could",
        "does",
        "from",
        "have",
        "into",
        "that",
        "their",
        "then",
        "there",
        "they",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "why",
    }
    return {
        token
        for value in values
        for token in re.findall(r"[a-z0-9]+", (value or "").casefold())
        if len(token) >= 3 and token not in stop
    }


def _rank(
    rows: list[dict[str, Any]], query_terms: set[str], text_fields: tuple[str, ...], limit: int
) -> list[dict[str, Any]]:
    def score(row: dict[str, Any]) -> tuple[int, str]:
        haystack = _terms(*(str(row.get(field) or "") for field in text_fields))
        overlap = len(query_terms & haystack)
        return overlap, str(row.get("last_seen") or row.get("occurred_at") or row.get("observed_at") or "")

    relevant = [row for row in rows if score(row)[0] > 0]
    recent_fallback = rows[: max(0, min(4, limit - len(relevant)))]
    merged: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for row in sorted(relevant, key=score, reverse=True) + recent_fallback:
        identity = row.get("id") or tuple(row.get(field) for field in text_fields)
        if identity not in seen:
            seen.add(identity)
            merged.append(row)
        if len(merged) >= limit:
            break
    return merged


def build_context(
    db_path: str,
    child_id: str,
    event: dict[str, Any],
    depth: int = 2,
    topic_hint: str | None = None,
) -> dict[str, Any]:
    """Build a bounded, relevance-ranked projection; never dump the family's database."""

    if depth not in range(0, 5):
        raise ValueError("context depth must be 0..4")
    metadata = event.get("metadata") or {}
    query_text = " ".join(str(x) for x in (event.get("text"), topic_hint, metadata.get("topic")) if x)
    query_terms = _terms(query_text)
    with connect(db_path) as conn:
        child = conn.execute("SELECT id,name,birth_year,grade FROM children WHERE id=?", (child_id,)).fetchone()
        if not child:
            raise ValueError(f"Unknown child: {child_id}")
        result: dict[str, Any] = {
            "child": dict(child),
            "event": event,
            "context_depth": depth,
        }
        if depth == 0:
            return result
        result["epistemic_rules"] = [
            "Availability is not exposure; exposure is not understanding.",
            "One observation is not a durable trait.",
            "Uncertain attribution remains uncertain.",
            "Repeated turns inside one episode are not independent evidence of interest.",
            "Node evidence_count is mention frequency only, never interest confidence.",
        ]
        preference = conn.execute(
            "SELECT profile_json FROM learning_preferences WHERE scope_type='household' AND scope_id='default'"
        ).fetchone()
        if preference:
            result["family_lens"] = jload(preference["profile_json"])

        observation_fetch = 30 if depth <= 2 else 100 if depth == 3 else 240
        fetched_observations = [
            {**dict(row), "metadata": jload(row["metadata_json"])}
            for row in conn.execute(
                """SELECT id,kind,text,source,confidence,occurred_at,metadata_json
                   FROM observations WHERE child_id=? ORDER BY occurred_at DESC LIMIT ?""",
                (child_id, observation_fetch),
            ).fetchall()
        ]
        raw_observations = [
            row
            for row in fetched_observations
            if row["metadata"].get("learning_scope", "family_signal") == "family_signal"
            and row["metadata"].get("independence_status") != "excluded"
        ]
        obs_limit = {1: 8, 2: 16, 3: 30, 4: 60}[depth]
        result["observations"] = _rank(raw_observations, query_terms, ("kind", "text"), obs_limit)
        episode_limit = {1: 4, 2: 8, 3: 16, 4: 24}[depth]
        episode_rows = [
            dict(row)
            for row in conn.execute(
                """SELECT ep.id,ep.summary,ep.status,ep.clustering_version,ep.opened_at,ep.last_event_at
                   FROM episodes ep WHERE ep.child_id=? AND EXISTS(
                     SELECT 1 FROM episode_memberships m WHERE m.episode_id=ep.id
                       AND m.learning_scope='family_signal' AND m.independence_status!='excluded'
                   ) ORDER BY ep.last_event_at DESC LIMIT ?""",
                (child_id, episode_limit * 3),
            ).fetchall()
        ]
        ranked_episodes = _rank(episode_rows, query_terms, ("summary",), episode_limit)
        for episode in ranked_episodes:
            episode["turns"] = [
                dict(row)
                for row in conn.execute(
                    """SELECT m.relation,m.independence_status,m.learning_scope,m.confidence,e.text,e.status,e.created_at
                       FROM episode_memberships m JOIN events e ON e.id=m.event_id
                       WHERE m.episode_id=? AND m.learning_scope='family_signal'
                         AND m.independence_status!='excluded' ORDER BY e.created_at LIMIT 12""",
                    (episode["id"],),
                ).fetchall()
            ]
        result["episodes"] = ranked_episodes
        school_limit = 8 if depth == 1 else 16
        result["school_signals"] = [
            dict(row)
            for row in conn.execute(
                """SELECT id,category,value,source_ref,observed_at,expires_at,confidence
                   FROM school_signals WHERE child_id=? AND (expires_at IS NULL OR expires_at>?)
                   ORDER BY observed_at DESC LIMIT ?""",
                (child_id, utcnow(), school_limit),
            ).fetchall()
        ]
        node_fetch = 40 if depth <= 2 else 120 if depth == 3 else 240
        fetched_nodes = [
            {**dict(row), "state": jload(row["state_json"])}
            for row in conn.execute(
                "SELECT * FROM nodes WHERE child_id=? ORDER BY last_seen DESC LIMIT ?",
                (child_id, node_fetch),
            ).fetchall()
        ]
        raw_nodes = [
            row
            for row in fetched_nodes
            if row["state"].get("learning_scope", "family_signal") == "family_signal"
            and row["state"].get("independence_status") != "excluded"
        ]
        node_limit = {1: 10, 2: 24, 3: 50, 4: 90}[depth]
        nodes = _rank(raw_nodes, query_terms, ("kind", "label", "canonical_key"), node_limit)
        result["nodes"] = nodes

        if depth >= 2:
            result["recent_experiences"] = [
                dict(row)
                for row in conn.execute(
                    """SELECT id,experience_type,title,status,created_at,offered_at,completed_at,feedback
                       FROM experiences WHERE child_id=? ORDER BY created_at DESC LIMIT 12""",
                    (child_id,),
                ).fetchall()
            ]
        if depth >= 3:
            node_ids = [row["id"] for row in nodes]
            if node_ids:
                marks = ",".join("?" for _ in node_ids)
                result["edges"] = [
                    {**dict(row), "metadata": jload(row["metadata_json"])}
                    for row in conn.execute(
                        f"""SELECT * FROM edges WHERE child_id=? AND
                        (source_node_id IN ({marks}) OR target_node_id IN ({marks}))
                        ORDER BY last_seen DESC LIMIT 80""",
                        (child_id, *node_ids, *node_ids),
                    ).fetchall()
                ]
        if depth >= 4:
            result["claims"] = [
                {
                    **dict(row),
                    "supporting_evidence": jload(row["supporting_evidence_json"], []),
                    "contradicting_evidence": jload(row["contradicting_evidence_json"], []),
                }
                for row in conn.execute(
                    "SELECT * FROM claims WHERE child_id=? ORDER BY updated_at DESC LIMIT 60",
                    (child_id,),
                ).fetchall()
            ]
            result["feedback_summary"] = [
                dict(row)
                for row in conn.execute(
                    """SELECT outcome,COUNT(*) AS count FROM feedback WHERE child_id=?
                       GROUP BY outcome ORDER BY count DESC""",
                    (child_id,),
                ).fetchall()
            ]

    if depth >= 2 and query_terms:
        include_excerpts = metadata.get("private_resource_mode") == "selected_excerpts"
        result["private_resources"] = search_resources(
            db_path,
            query_text,
            limit=3 if depth == 2 else 6,
            include_excerpts=include_excerpts,
        )
        result["private_resource_mode"] = "selected_excerpts" if include_excerpts else "metadata_only"
    return result
