from __future__ import annotations

import json
import re
from typing import Any

from .db import connect, jload, utcnow
from .learner_model import learner_snapshot_from_conn
from .resources import search_resources

MAX_CONTEXT_CHARS = 48_000


def prepare_model_inputs(
    context: dict[str, Any], event: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Remove shadow state and budget the complete, non-duplicated model input."""

    model_context = dict(context)
    shadow_snapshot = model_context.pop("shadow_learner_snapshot", None)
    profile_projection = model_context.pop("profile_projection", None)
    model_context.pop("event", None)
    model_context.pop("context_budget", None)
    combined = fit_context_budget({"context": model_context, "event": event})
    model_context = combined["context"]
    model_event = combined["event"]
    model_context["context_budget"] = combined["context_budget"]
    diagnostic = {
        "status": (shadow_snapshot or {}).get("status", "not_available"),
        "snapshot_digest": (shadow_snapshot or {}).get("snapshot_digest"),
        "knowledge_count": len((shadow_snapshot or {}).get("knowledge", [])),
        "interest_count": len((shadow_snapshot or {}).get("interests", [])),
        "passed_to_model": False,
        "projection_summary_passed_to_model": False,
        "projection_status": (profile_projection or {}).get("status", "not_available"),
        "complete_model_input_chars": combined["context_budget"]["serialized_chars"],
    }
    return model_context, model_event, diagnostic


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


def _bound_nested(value: Any) -> Any:
    """Deterministically cap individual values before applying the total budget."""

    if isinstance(value, str):
        return value if len(value) <= 6_000 else value[:5_997] + "..."
    if isinstance(value, list):
        return [_bound_nested(item) for item in value[:100]]
    if isinstance(value, dict):
        return {
            str(key): _bound_nested(value[key])
            for key in sorted(value, key=str)[:100]
        }
    return value


def fit_context_budget(context: dict[str, Any]) -> dict[str, Any]:
    """Enforce a hard serialized context limit by dropping least-essential detail first."""

    bounded = _bound_nested(context)
    assert isinstance(bounded, dict)
    # Seed the measurement with the maximum-width value so replacing it with
    # the final size cannot make an already-fitted payload larger.
    budget = {
        "max_serialized_chars": MAX_CONTEXT_CHARS,
        "serialized_chars": MAX_CONTEXT_CHARS,
        "truncated": bounded != context,
    }
    bounded["context_budget"] = budget

    def size() -> int:
        return len(json.dumps(bounded, ensure_ascii=False, sort_keys=True, default=str))

    while size() > MAX_CONTEXT_CHARS:
        content = bounded.get("context") if isinstance(bounded.get("context"), dict) else bounded
        changed = False
        for key in ("private_resources", "episodes", "school_signals", "recent_experiences"):
            items = content.get(key)
            if isinstance(items, list) and items:
                items.pop()
                changed = True
                break
        if changed:
            budget["truncated"] = True
            continue
        event = bounded.get("event") if content is not bounded else content.get("event")
        if isinstance(event, dict) and event.get("metadata"):
            event["metadata"] = {}
            budget["truncated"] = True
            continue
        if isinstance(event, dict) and len(str(event.get("text") or "")) > 512:
            event["text"] = str(event["text"])[:512] + "..."
            budget["truncated"] = True
            continue
        snapshot = content.get("shadow_learner_snapshot")
        if isinstance(snapshot, dict):
            for key in ("knowledge", "interests"):
                items = snapshot.get(key)
                if isinstance(items, list) and items:
                    items.pop()
                    changed = True
                    break
        if changed:
            budget["truncated"] = True
            continue
        if content.get("family_lens"):
            content["family_lens"] = {}
            budget["truncated"] = True
            continue
        removable = sorted(
            key
            for key in content
            if key
            not in {
                "child",
                "context_budget",
                "context_depth",
                "epistemic_rules",
                "event",
            }
        )
        if removable:
            content.pop(removable[-1])
            budget["truncated"] = True
            continue
        raise RuntimeError("essential context exceeds hard serialization budget")
    budget["serialized_chars"] = size()
    if budget["serialized_chars"] > MAX_CONTEXT_CHARS:
        raise RuntimeError("context budget accounting exceeded hard limit")
    return bounded


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
        memory_refs = metadata.get("parent_memory_refs") or []
        if not isinstance(memory_refs, list) or len(memory_refs) > 24:
            raise ValueError("parent memory references are invalid")
        session_ids = {
            str(row[0])
            for row in conn.execute(
                """SELECT session_id FROM capability_runs WHERE produced_event_id=?
                   UNION SELECT cr.session_id FROM capability_run_event_lineage l
                     JOIN capability_runs cr ON cr.id=l.capability_run_id
                     WHERE l.event_id=?""",
                (event.get("id"), event.get("id")),
            ).fetchall()
            if row[0]
        }
        parent_working_preferences: list[dict[str, Any]] = []
        for ref in memory_refs:
            if not isinstance(ref, dict) or set(ref) != {"memory_key", "version"}:
                raise ValueError("parent memory reference shape is invalid")
            key = str(ref["memory_key"])
            try:
                version = int(ref["version"])
            except (TypeError, ValueError) as exc:
                raise ValueError("parent memory reference version is invalid") from exc
            row = conn.execute(
                """SELECT memory_key,version,scope_type,scope_id,child_id,category,value
                   FROM parent_memories m
                   WHERE memory_key=? AND version=? AND status='active'
                     AND (expires_at IS NULL OR expires_at>?)
                     AND NOT EXISTS(
                       SELECT 1 FROM parent_memory_settings s
                       WHERE s.scope_type='household' AND s.scope_id='default' AND s.enabled=0
                     )
                     AND NOT EXISTS(
                       SELECT 1 FROM parent_memory_settings s
                       WHERE s.scope_type=m.scope_type AND s.scope_id=m.scope_id AND s.enabled=0
                     )""",
                (key, version, utcnow()),
            ).fetchone()
            if not row:
                continue
            scope_type = str(row["scope_type"])
            owned = (
                (scope_type == "household" and row["scope_id"] == "default")
                or (
                    scope_type == "child_support"
                    and row["scope_id"] == child_id
                    and row["child_id"] == child_id
                )
                or (scope_type == "thread" and row["scope_id"] in session_ids)
            )
            if not owned:
                raise ValueError("parent memory reference does not belong to this event")
            parent_working_preferences.append(
                {
                    "memory_key": str(row["memory_key"]),
                    "version": int(row["version"]),
                    "category": str(row["category"]),
                    "value": str(row["value"]),
                    "why_recalled": f"Explicit {scope_type.replace('_', '-')} working preference.",
                }
            )
        if parent_working_preferences:
            result["parent_working_preferences"] = parent_working_preferences
        if depth == 0:
            return fit_context_budget(result)
        result["epistemic_rules"] = [
            "Availability is not exposure; exposure is not understanding.",
            "One observation is not a durable trait.",
            "Generated output and model annotations cannot evidence their own effectiveness.",
            "Uncertain attribution remains uncertain.",
            "Repeated turns inside one episode are not independent evidence of interest.",
            "Node evidence_count is mention frequency only, never interest confidence.",
        ]
        preference = conn.execute(
            "SELECT profile_json FROM learning_preferences WHERE scope_type='household' AND scope_id='default'"
        ).fetchone()
        if preference:
            result["family_lens"] = jload(preference["profile_json"])

        # Phase 0 quarantines the legacy graph projection wholesale. Before the
        # authority boundary existed, a model could author observation/node/edge
        # contents and confidence. Phase 1 will expose only recomputable,
        # rule-versioned projections; raw records remain locally inspectable.
        result["observations"] = []
        result["nodes"] = []
        shadow_snapshot = learner_snapshot_from_conn(conn, child_id, limit=20)
        result["profile_projection"] = {
            "status": shadow_snapshot["status"],
            "may_affect_recommendations": False,
            "snapshot_digest": shadow_snapshot["snapshot_digest"],
            "knowledge_count": len(shadow_snapshot["knowledge"]),
            "interest_count": len(shadow_snapshot["interests"]),
        }
        result["shadow_learner_snapshot"] = shadow_snapshot
        episode_limit = {1: 4, 2: 8, 3: 16, 4: 24}[depth]
        episode_rows = [
            dict(row)
            for row in conn.execute(
                """SELECT ep.id,ep.summary,ep.status,ep.clustering_version,ep.opened_at,ep.last_event_at
                   FROM episodes ep WHERE ep.child_id=? AND EXISTS(
                     SELECT 1 FROM episode_memberships m JOIN events member_event ON member_event.id=m.event_id
                     WHERE m.episode_id=ep.id AND member_event.child_id=ep.child_id
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
                       WHERE m.episode_id=? AND e.child_id=? AND m.learning_scope='family_signal'
                         AND m.independence_status!='excluded' ORDER BY e.created_at LIMIT 12""",
                    (episode["id"], child_id),
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
        if depth >= 2:
            result["recent_experiences"] = [
                dict(row)
                for row in conn.execute(
                    """SELECT id,experience_type,title,status,created_at,offered_at,completed_at
                       FROM experiences WHERE child_id=? ORDER BY created_at DESC LIMIT 12""",
                    (child_id,),
                ).fetchall()
            ]
        if depth >= 3:
            result["edges"] = []
        if depth >= 4:
            # Legacy free-text claims are locally inspectable but never model-facing.
            result["claims"] = []
            result["feedback_summary"] = []

    if depth >= 2 and query_terms:
        include_excerpts = metadata.get("private_resource_mode") == "selected_excerpts"
        result["private_resources"] = search_resources(
            db_path,
            query_text,
            limit=3 if depth == 2 else 6,
            include_excerpts=include_excerpts,
        )
        result["private_resource_mode"] = "selected_excerpts" if include_excerpts else "metadata_only"
    return fit_context_budget(result)
