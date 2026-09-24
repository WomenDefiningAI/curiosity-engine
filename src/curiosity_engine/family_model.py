from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .db import (
    connect,
    init_db,
    jdump,
    jload,
    research_connect,
    research_control_event_id,
    transport_event_hash,
    utcnow,
)
from .deletion_ledger import (
    REDACTED_PARENT_MEMORY_INSPECTION,
    event_id_hash,
    managed_data_lifecycle,
    reconcile_database_deletions,
    record_event_deletion,
    redact_parent_memory_deliveries,
)
from .learner_model import (
    correct_evidence,
    delete_learning_source,
    learner_snapshot_from_conn,
    recompute_child_projections_in_transaction,
)
from .research_authority_ledger import record_research_authority

FAMILY_MODEL_VIEW_VERSION = "phase3-v1"
ActorType = Literal["parent", "operator"]
CorrectionAction = Literal["exclude", "restore", "correct"]

_KNOWLEDGE_COPY = {
    "exposure_recorded": "The engine has seen this topic, but has not seen enough to describe understanding.",
    "practicing_with_support": "The engine has seen this idea practiced with some help.",
    "demonstrated_in_context": "The engine saw this idea demonstrated in one kind of activity.",
    "independently_demonstrated": "The engine saw this idea demonstrated independently in more than one learning moment.",
    "transfer_observed": "The engine saw this idea used independently in meaningfully different situations.",
}
_INTEREST_COPY = {
    "triggered_situational": "This topic caught the child's attention in one learning moment.",
    "maintained_situational": "The child chose to keep going or ask another question about this topic.",
    "emerging_recurrent": "The child returned to this topic across separate learning moments.",
    "sustained_parent_confirmed": "A parent explicitly confirmed that this interest has continued over time.",
}
_CHILD_KNOWLEDGE_COPY = {
    "exposure_recorded": "We have explored this idea a little.",
    "practicing_with_support": "You have been practicing this idea with some help.",
    "demonstrated_in_context": "You showed this idea in one kind of activity.",
    "independently_demonstrated": "You showed this idea on your own in more than one learning moment.",
    "transfer_observed": "You tried this idea in different ways.",
}
_CHILD_INTEREST_COPY = {
    "triggered_situational": "This topic caught your attention once.",
    "maintained_situational": "You chose to stay with this topic or ask another question.",
    "emerging_recurrent": "You have come back to this topic in different moments.",
    "sustained_parent_confirmed": "Your family says this topic has stayed interesting to you.",
}


def _target_hash(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _require_reason(reason: str) -> str:
    cleaned = " ".join(reason.split())
    if not cleaned or len(cleaned) > 500:
        raise ValueError("reason must be 1..500 characters")
    return cleaned


def _as_utc_iso(value: datetime | str) -> str:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("retention deadline must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _reference_path(value: str) -> Path:
    return Path(os.path.abspath(Path(value).expanduser()))


def _event_file_paths(conn: Any, event_id: str) -> set[Path]:
    """Collect only files exclusively owned by durable lineage to one event."""

    owners: dict[Path, set[str | None]] = {}

    def add(path_value: Any, *event_values: Any) -> None:
        if path_value is None:
            return
        path = _reference_path(str(path_value))
        event_owners = {str(value) for value in event_values if value is not None and str(value)}
        owners.setdefault(path, set()).update(event_owners or {None})

    for row in conn.execute(
        """SELECT a.path,x.source_event_id,p.event_id AS plan_event_id
           FROM artifacts a LEFT JOIN experiences x ON x.id=a.experience_id
           LEFT JOIN pedagogy_plans p ON p.plan_id=a.pedagogy_plan_id"""
    ).fetchall():
        add(row["path"], row["source_event_id"], row["plan_event_id"])
    for row in conn.execute(
        """SELECT afo.path,x.source_event_id,p.event_id AS plan_event_id
           FROM artifact_file_outbox afo JOIN artifacts a ON a.id=afo.artifact_id
           LEFT JOIN experiences x ON x.id=a.experience_id
           LEFT JOIN pedagogy_plans p ON p.plan_id=a.pedagogy_plan_id"""
    ).fetchall():
        add(row["path"], row["source_event_id"], row["plan_event_id"])
    for row in conn.execute("SELECT path,event_id FROM visual_assets").fetchall():
        add(row["path"], row["event_id"])
    for row in conn.execute(
        """SELECT ia.path,tr.event_id FROM inbound_assets ia LEFT JOIN transport_receipts tr
             ON tr.binding_id=ia.binding_id AND tr.external_event_id=ia.external_event_id
           WHERE ia.path IS NOT NULL"""
    ).fetchall():
        add(row["path"], row["event_id"])

    exclusive = {path for path, path_owners in owners.items() if path_owners == {event_id}}
    for path in tuple(exclusive):
        if path.suffix.casefold() != ".pdf":
            continue
        preview = path.with_suffix(".png")
        preview_owners = owners.get(preview)
        if preview_owners is None or preview_owners == {event_id}:
            exclusive.add(preview)
    return exclusive


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_private_file_deletion(
    db_path: str | Path,
    *,
    deletion_id: str,
    event_id: str,
    paths: set[Path],
    output_root: str | Path | None,
) -> tuple[str | None, list[tuple[Path, Path]]]:
    """Journal linked files durably before moving them out of their live paths."""

    existing = [path for path in paths if path.exists()]
    if not existing:
        return None, []
    if output_root is None:
        raise ValueError("event has linked files; an explicit private output root is required for complete deletion")
    root = Path(output_root).expanduser().resolve()
    resolved: list[Path] = []
    for candidate in existing:
        if candidate.is_symlink():
            raise ValueError("linked event output may not be a symbolic link")
        path = candidate.resolve()
        if not path.is_relative_to(root):
            raise ValueError("linked event file is outside the configured private output root")
        if not path.is_file():
            raise ValueError("linked event output is not a regular file")
        resolved.append(path)
    resolved = sorted(set(resolved))
    quarantine_dir = root / ".family-delete-journal" / deletion_id
    quarantine_dir.mkdir(parents=True, exist_ok=False)
    quarantine_dir.chmod(0o700)
    now = utcnow()
    staged = [(path, quarantine_dir / f"{ordinal:04d}.bin") for ordinal, path in enumerate(resolved)]
    try:
        with connect(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for ordinal, (original, quarantine) in enumerate(staged):
                conn.execute(
                    """INSERT INTO family_file_deletion_journal(
                       journal_id,deletion_id,event_id_hash,original_path,quarantine_path,output_root,
                       content_sha256,state,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,'planned',?,?)""",
                    (
                        f"{deletion_id}_{ordinal:04d}",
                        deletion_id,
                        _target_hash(event_id),
                        str(original),
                        str(quarantine),
                        str(root),
                        _file_sha256(original),
                        now,
                        now,
                    ),
                )
        for ordinal, (original, quarantine) in enumerate(staged):
            os.replace(original, quarantine)
            with connect(db_path) as conn:
                conn.execute(
                    """UPDATE family_file_deletion_journal SET state='staged',updated_at=?
                       WHERE journal_id=?""",
                    (utcnow(), f"{deletion_id}_{ordinal:04d}"),
                )
    except BaseException:
        # The event still exists, so initialization deterministically restores
        # every verified file and clears the durable journal.
        init_db(db_path)
        raise
    return deletion_id, staged


def _rebuild_session_preferences(conn: Any, session_ids: set[str]) -> None:
    """Recompute cached thread preferences after source-message erasure."""

    now = utcnow()
    for session_id in sorted(session_ids):
        session = conn.execute(
            "SELECT summary_json FROM agent_sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        if not session:
            continue
        active: dict[str, dict[str, str]] = {}
        for row in conn.execute(
            """SELECT category,operation,value,source_message_id
               FROM agent_session_preference_events WHERE session_id=? ORDER BY ordinal""",
            (session_id,),
        ).fetchall():
            category = str(row["category"])
            if row["operation"] == "clear":
                active.pop(category, None)
            else:
                active[category] = {
                    "category": category,
                    "value": str(row["value"]),
                    "source_message_id": str(row["source_message_id"]),
                }
        summary = jload(session["summary_json"], {})
        summary = dict(summary) if isinstance(summary, dict) else {}
        summary["parent_agent"] = {
            "version": 1,
            "preferences": list(active.values()),
            "active_output_ref": None,
        }
        conn.execute(
            "UPDATE agent_sessions SET summary_json=?,updated_at=? WHERE id=?",
            (jdump(summary), now, session_id),
        )


def _require_child(conn: Any, child_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id,name,birth_year,grade,created_at,updated_at FROM children WHERE id=?",
        (child_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"unknown child {child_id!r}")
    return dict(row)


def _controls_from_conn(conn: Any, child_id: str) -> dict[str, Any]:
    row = conn.execute(
        """SELECT collection_enabled,recommendation_use_enabled,recommendation_policy_id,
                  recommendation_policy_version,recommendation_policy_hash,personalization_epoch,
                  default_retention_class,
                  updated_by,created_at,updated_at
           FROM learner_model_controls WHERE child_id=?""",
        (child_id,),
    ).fetchone()
    return (
        dict(row)
        if row
        else {
            "collection_enabled": 1,
            "recommendation_use_enabled": 0,
            "recommendation_policy_id": None,
            "recommendation_policy_version": None,
            "recommendation_policy_hash": None,
            "personalization_epoch": 0,
            "default_retention_class": "learning_record",
            "updated_by": None,
            "created_at": None,
            "updated_at": None,
        }
    )


def _audit(
    conn: Any,
    *,
    child_id: str,
    action: str,
    target_kind: str,
    target: str,
    actor_type: ActorType,
    details: dict[str, Any] | None = None,
) -> str:
    audit_id = f"fma_{uuid4().hex[:20]}"
    conn.execute(
        """INSERT INTO family_model_audit_log(
           audit_id,child_id,action,target_kind,target_hash,actor_type,details_json,created_at
           ) VALUES(?,?,?,?,?,?,?,?)""",
        (
            audit_id,
            child_id,
            action,
            target_kind,
            _target_hash(target),
            actor_type,
            jdump(details or {}),
            utcnow(),
        ),
    )
    return audit_id


def _plain_projection(item: dict[str, Any], *, construct: str) -> dict[str, Any]:
    if construct == "knowledge":
        state = str(item["state"])
        explanation = _KNOWLEDGE_COPY.get(state, "The engine does not have a stable description for this idea.")
        why = (
            f"{item['eligible_opportunity_count']} eligible observation(s) across "
            f"{item['eligible_episode_count']} separate learning episode(s); "
            f"support level: {str(item['support_level']).replace('_', ' ')}."
        )
        evidence_ids = [
            *item.get("supporting_evidence_ids", []),
            *item.get("contradicting_evidence_ids", []),
            *item.get("context_evidence_ids", []),
        ]
    else:
        state = str(item["phase"])
        explanation = _INTEREST_COPY.get(state, "The engine has not seen a recurring interest pattern.")
        why = (
            f"{item['eligible_episode_count']} eligible episode(s), "
            f"{item['child_initiated_count']} child-initiated signal(s), and "
            f"{item['continued_or_question_count']} continuation or new-question signal(s)."
        )
        evidence_ids = [*item.get("supporting_evidence_ids", []), *item.get("context_evidence_ids", [])]
    return {
        **item,
        "construct": construct,
        "display_state": state.replace("_", " "),
        "explanation": explanation,
        "why": why,
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
        "uses_for_recommendations": False,
    }


def _effective_evidence_view(conn: Any, child_id: str, *, limit: int | None) -> list[dict[str, Any]]:
    limit_clause = "" if limit is None else " LIMIT ?"
    parameters: tuple[Any, ...] = (child_id,) if limit is None else (child_id, limit)
    rows = [
        dict(row)
        for row in conn.execute(
            """SELECT er.evidence_id,er.opportunity_id,er.concept_id,er.concept_version,
                      c.label AS concept_label,c.domain,er.construct,er.observation_kind,er.stance,
                      er.result,er.score,er.source_type,er.source_reliability,er.evidence_strength,
                      er.projection_eligible,er.episode_id,er.context_key,er.representation_key,
                      er.assistance_level,er.attempt_number,er.is_transfer,er.is_trivial,
                      er.rule_id,er.rule_version,er.retention_class,er.expires_at,er.created_at,
                      lo.event_id,lo.plan_id,lo.actor,lo.initiation,lo.response_mode,lo.scorer,
                      lo.scorer_version,lo.independence_status,lo.occurred_at,
                      lt.title AS task_title,lt.strategy,lt.difficulty,lt.difficulty_provenance,
                      substr(e.text,1,500) AS source_excerpt,e.source AS event_source,
                      (SELECT MIN(retain_until) FROM learner_retention_events re
                       WHERE re.child_id=er.child_id AND re.opportunity_id=er.opportunity_id
                      ) AS parent_retain_until
               FROM evidence_records er
               JOIN learning_opportunities lo
                 ON lo.opportunity_id=er.opportunity_id AND lo.child_id=er.child_id
               JOIN learning_tasks lt ON lt.task_id=lo.task_id AND lt.version=lo.task_version
               JOIN concepts c ON c.concept_id=er.concept_id AND c.version=er.concept_version
               JOIN events e ON e.id=lo.event_id AND e.child_id=lo.child_id
               WHERE er.child_id=? ORDER BY lo.occurred_at DESC,er.evidence_id"""
            + limit_clause,
            parameters,
        ).fetchall()
    ]
    correction_rows = conn.execute(
        """SELECT correction_id,evidence_id,action,replacement_json,actor_type,reason,created_at
           FROM evidence_corrections WHERE child_id=? ORDER BY created_at,correction_id""",
        (child_id,),
    ).fetchall()
    corrections: dict[str, list[dict[str, Any]]] = {}
    for row in correction_rows:
        item = dict(row)
        item["replacement"] = jload(item.pop("replacement_json"), {})
        corrections.setdefault(str(item["evidence_id"]), []).append(item)
    current_links = {
        str(row["evidence_id"]): str(row["stance"])
        for row in conn.execute(
            """SELECT evidence_id,stance FROM learner_projection_evidence WHERE child_id=?
               UNION ALL
               SELECT evidence_id,stance FROM interest_projection_evidence WHERE child_id=?""",
            (child_id, child_id),
        )
    }
    now = utcnow()
    result: list[dict[str, Any]] = []
    for row in rows:
        active = True
        effective_result = str(row["result"])
        effective_assistance = str(row["assistance_level"])
        history = corrections.get(str(row["evidence_id"]), [])
        for correction in history:
            if correction["action"] == "exclude":
                active = False
            elif correction["action"] == "restore":
                active = True
            elif correction["action"] == "correct":
                effective_result = str(correction["replacement"].get("result", effective_result))
                effective_assistance = str(correction["replacement"].get("assistance_level", effective_assistance))
        deadlines = [str(value) for value in (row.get("expires_at"), row.get("parent_retain_until")) if value]
        effective_deadline = min(deadlines) if deadlines else None
        expired = bool(effective_deadline and effective_deadline <= now)
        effective_stance = current_links.get(str(row["evidence_id"]))
        counted = bool(active and not expired and effective_stance in {"supports", "contradicts"})
        if not active:
            why_counted = "Not counted because a parent or operator excluded this evidence."
        elif expired:
            why_counted = "Not counted because its retention window ended."
        elif effective_stance in {None, "context_only"}:
            why_counted = "Context only: the source, attempt, task, or assistance conditions are not strong enough."
        elif effective_stance == "contradicts":
            why_counted = "Counted as contradictory evidence so uncertainty remains visible."
        else:
            why_counted = "Counted because it met the reviewed evidence rule and episode-independence checks."
        result.append(
            {
                **row,
                "active": active,
                "expired": expired,
                "counted": counted,
                "effective_result": effective_result,
                "effective_assistance_level": effective_assistance,
                "effective_stance": effective_stance,
                "effective_projection_eligible": counted,
                "effective_retain_until": effective_deadline,
                "why_counted": why_counted,
                "corrections": history,
            }
        )
    return result


def inspect_family_model(
    db_path: str | Path,
    child_id: str,
    *,
    evidence_limit: int | None = 80,
    projection_limit: int | None = 100,
    audit_limit: int | None = 50,
) -> dict[str, Any]:
    """Return the local parent inspector with evidence, inferences, and provenance."""

    for name, limit, maximum in (
        ("evidence_limit", evidence_limit, 500),
        ("projection_limit", projection_limit, 500),
        ("audit_limit", audit_limit, 500),
    ):
        if limit is not None and (limit < 1 or limit > maximum):
            raise ValueError(f"{name} must be between 1 and {maximum}, or None for a complete export")
    init_db(db_path)
    with connect(db_path) as conn:
        child = _require_child(conn, child_id)
        snapshot = learner_snapshot_from_conn(conn, child_id, limit=projection_limit)
        controls = _controls_from_conn(conn, child_id)
        evidence = _effective_evidence_view(conn, child_id, limit=evidence_limit)
        audit_limit_clause = "" if audit_limit is None else " LIMIT ?"
        audit_parameters: tuple[Any, ...] = (child_id,) if audit_limit is None else (child_id, audit_limit)
        audit = [
            {**dict(row), "details": jload(row["details_json"], {})}
            for row in conn.execute(
                """SELECT audit_id,action,target_kind,target_hash,actor_type,details_json,created_at
                   FROM family_model_audit_log WHERE child_id=?
                   ORDER BY created_at DESC,audit_id DESC"""
                + audit_limit_clause,
                audit_parameters,
            ).fetchall()
        ]
        for item in audit:
            item.pop("details_json", None)
        active_policy = bool(
            controls["recommendation_policy_id"]
            and conn.execute(
                """SELECT 1 FROM personalization_policies
                   WHERE policy_id=? AND version=? AND definition_hash=? AND status='active'""",
                (
                    controls["recommendation_policy_id"],
                    controls["recommendation_policy_version"],
                    controls["recommendation_policy_hash"],
                ),
            ).fetchone()
        )
        personalization_decisions = [
            {
                **dict(row),
                "candidate": jload(row["candidate_json"], {}),
            }
            for row in conn.execute(
                """SELECT decision_id,event_id,plan_id,policy_id,policy_version,definition_hash,
                          evaluation_id,promotion_review_id,mode,candidate_json,decision_hash,created_at
                   FROM personalization_decisions WHERE child_id=?
                   ORDER BY created_at DESC,decision_id DESC"""
                + audit_limit_clause,
                audit_parameters,
            ).fetchall()
        ]
        for item in personalization_decisions:
            item.pop("candidate_json", None)
        personalization_feedback = [
            dict(row)
            for row in conn.execute(
                """SELECT signal_id,feedback_id,plan_id,plan_hash,outcome,used,chose_to_continue,
                          made_prediction,gave_explanation,help_level,asked_new_question,
                          parent_correction,created_at
                   FROM personalization_feedback_signals WHERE child_id=?
                   ORDER BY created_at DESC,signal_id DESC"""
                + audit_limit_clause,
                audit_parameters,
            ).fetchall()
        ]
        personalization_controls = [
            dict(row)
            for row in conn.execute(
                """SELECT event_id,enabled,policy_id,policy_version,definition_hash,actor_type,
                          reason,created_at
                   FROM personalization_control_events WHERE child_id=?
                   ORDER BY created_at DESC,event_id DESC"""
                + audit_limit_clause,
                audit_parameters,
            ).fetchall()
        ]
        research_control = conn.execute(
            """SELECT enabled,policy_id,policy_version,definition_hash,release_content_hash,
                      family_authorization_epoch,
                      updated_by,reason,updated_at FROM research_track_controls WHERE child_id=?""",
            (child_id,),
        ).fetchone()
        research_control_events = [
            dict(row)
            for row in conn.execute(
                """SELECT event_id,enabled,policy_id,policy_version,definition_hash,
                          release_content_hash,family_authorization_epoch,actor_type,reason,created_at
                   FROM research_track_control_events WHERE child_id=?
                   ORDER BY created_at DESC,event_id DESC"""
                + audit_limit_clause,
                audit_parameters,
            ).fetchall()
        ]
        research_traces = [
            {
                **dict(row),
                "evidence_ids": jload(row["evidence_ids_json"], []),
                "history": jload(row["history_json"], {}),
            }
            for row in conn.execute(
                """SELECT trace_id,trace_hash,concept_id,concept_version,projection_input_digest,
                          projection_rule_id,projection_rule_version,policy_id,policy_version,
                          definition_hash,release_content_hash,evaluation_id,promotion_review_id,
                          authorization_snapshot_id,
                          family_authorization_epoch,policy_authorization_epoch,evidence_ids_json,
                          evidence_digest,history_json,posterior_mean,interval_90_lower,
                          interval_90_upper,uncertainty,
                          method_version,advisory_only,rationale,created_at
                   FROM research_uncertainty_traces WHERE child_id=?
                   ORDER BY created_at DESC,trace_id DESC"""
                + audit_limit_clause,
                audit_parameters,
            ).fetchall()
        ]
        for item in research_traces:
            item.pop("evidence_ids_json", None)
            item.pop("history_json", None)
        research_authorization_snapshots = [
            dict(row)
            for row in conn.execute(
                """SELECT s.* FROM research_track_authorization_snapshots s
                   WHERE s.snapshot_id IN (
                     SELECT authorization_snapshot_id FROM research_uncertainty_traces
                     WHERE child_id=?
                   ) OR EXISTS(
                     SELECT 1 FROM research_track_controls c
                     WHERE c.child_id=? AND c.policy_id=s.policy_id
                       AND c.policy_version=s.policy_version
                       AND c.definition_hash=s.definition_hash
                       AND c.release_content_hash=s.release_content_hash
                   ) OR EXISTS(
                     SELECT 1 FROM research_track_control_events e
                     WHERE e.child_id=? AND e.enabled=1 AND e.policy_id=s.policy_id
                       AND e.policy_version=s.policy_version
                       AND e.definition_hash=s.definition_hash
                       AND e.release_content_hash=s.release_content_hash
                   ) ORDER BY s.policy_authorization_epoch DESC,s.snapshot_id DESC"""
                + audit_limit_clause,
                (child_id, child_id, child_id)
                + (() if audit_limit is None else (audit_limit,)),
            ).fetchall()
        ]
        research_candidates = [
            {**dict(row), "payload": jload(row["payload_json"], {})}
            for row in conn.execute(
                """SELECT candidate_id,candidate_hash,trace_id,concept_id,concept_version,
                          interest_input_digest,interest_rule_id,interest_rule_version,interest_phase,
                          policy_id,policy_version,definition_hash,release_content_hash,
                          family_authorization_epoch,policy_authorization_epoch,
                          title,rationale,payload_json,status,created_at,invalidated_at
                   FROM research_proactivity_candidates WHERE child_id=?
                   ORDER BY created_at DESC,candidate_id DESC"""
                + audit_limit_clause,
                audit_parameters,
            ).fetchall()
        ]
        for item in research_candidates:
            item.pop("payload_json", None)
    recommendation_requested = bool(controls["recommendation_use_enabled"])
    recommendation_effective = bool(
        recommendation_requested and controls["collection_enabled"] and active_policy
    )
    knowledge = [_plain_projection(item, construct="knowledge") for item in snapshot["knowledge"]]
    interests = [_plain_projection(item, construct="interest") for item in snapshot["interests"]]
    for item in [*knowledge, *interests]:
        item["uses_for_recommendations"] = recommendation_effective
    return {
        "view_version": FAMILY_MODEL_VIEW_VERSION,
        "child": child,
        "controls": {
            **controls,
            "collection_enabled": bool(controls["collection_enabled"]),
            "recommendation_use_enabled": recommendation_requested,
            "recommendation_use_effective": recommendation_effective,
            "recommendation_use_reason": (
                "An operator-promoted policy may calibrate challenge and interest support for new plans."
                if recommendation_effective
                else "Personalization remains shadow-only until both family opt-in and operator promotion are present."
            ),
        },
        "inferences": {"knowledge": knowledge, "interests": interests},
        "evidence": evidence,
        "personalization": {
            "decisions": personalization_decisions,
            "feedback_signals": personalization_feedback,
            "control_events": personalization_controls,
            "feature_vectors_exposed": False,
            "freeform_feedback_exposed": False,
        },
        "research_track": {
            "control": dict(research_control) if research_control else None,
            "control_events": research_control_events,
            "authorization_snapshots": research_authorization_snapshots,
            "uncertainty_traces": research_traces,
            "proactivity_candidates": research_candidates,
            "trace_role": "advisory_only",
            "candidate_delivery_enabled": False,
        },
        "audit": audit,
        "counts": {
            "knowledge_inferences": len(knowledge),
            "interest_inferences": len(interests),
            "evidence_records": len(evidence),
            "personalization_decisions": len(personalization_decisions),
            "personalization_feedback_signals": len(personalization_feedback),
            "personalization_control_events": len(personalization_controls),
            "research_uncertainty_traces": len(research_traces),
            "research_proactivity_candidates": len(research_candidates),
            "research_control_events": len(research_control_events),
            "research_authorization_snapshots": len(research_authorization_snapshots),
        },
        "raw_evidence_model_facing": False,
        "labels_are_not_grades_or_diagnoses": True,
    }


def set_collection_enabled(
    db_path: str | Path,
    *,
    child_id: str,
    enabled: bool,
    reason: str,
    actor_type: ActorType = "parent",
) -> dict[str, Any]:
    reason = _require_reason(reason)
    init_db(db_path)
    now = utcnow()
    with research_connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_child(conn, child_id)
        current = _controls_from_conn(conn, child_id)
        if bool(current["collection_enabled"]) == enabled:
            return {"child_id": child_id, "collection_enabled": enabled, "changed": False}
        conn.execute(
            """INSERT INTO learner_model_controls(
               child_id,collection_enabled,personalization_epoch,default_retention_class,
               updated_by,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(child_id) DO UPDATE SET
                 collection_enabled=excluded.collection_enabled,
                 personalization_epoch=excluded.personalization_epoch,
                 updated_by=excluded.updated_by,updated_at=excluded.updated_at""",
            (
                child_id,
                int(enabled),
                int(current["personalization_epoch"]) + 1,
                str(current["default_retention_class"]),
                actor_type,
                current["created_at"] or now,
                now,
            ),
        )
        research_control = conn.execute(
            "SELECT child_id FROM research_track_controls WHERE child_id=?", (child_id,)
        ).fetchone()
        if research_control:
            control_event = conn.execute(
                """SELECT event_id FROM research_track_control_events
                   WHERE child_id=? AND created_at=? AND actor_type='system'
                     AND reason='Learner-model collection was paused.'
                   ORDER BY family_authorization_epoch DESC,event_id DESC LIMIT 1""",
                (child_id, now),
            ).fetchone()
            if control_event:
                record_research_authority(
                    db_path, conn, control_event_ids=(str(control_event["event_id"]),)
                )
        audit_id = _audit(
            conn,
            child_id=child_id,
            action="collection_resumed" if enabled else "collection_paused",
            target_kind="child_model",
            target=child_id,
            actor_type=actor_type,
            details={"reason": reason},
        )
    return {"child_id": child_id, "collection_enabled": enabled, "changed": True, "audit_id": audit_id}


def set_recommendation_use_enabled(
    db_path: str | Path,
    *,
    child_id: str,
    enabled: bool,
    reason: str,
    actor_type: ActorType = "parent",
) -> dict[str, Any]:
    """Record a reversible family opt-in; operator promotion remains a separate hard gate."""

    reason = _require_reason(reason)
    init_db(db_path)
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_child(conn, child_id)
        current = _controls_from_conn(conn, child_id)
        active_policy = conn.execute(
            """SELECT policy_id,version,definition_hash FROM personalization_policies
               WHERE status='active' ORDER BY activated_at DESC LIMIT 1"""
        ).fetchone()
        if enabled and not active_policy:
            raise ValueError("family opt-in requires an active operator-promoted policy")
        current_matches = bool(
            enabled
            and active_policy
            and current["recommendation_policy_id"] == active_policy["policy_id"]
            and current["recommendation_policy_version"] == active_policy["version"]
            and current["recommendation_policy_hash"] == active_policy["definition_hash"]
        )
        if bool(current["recommendation_use_enabled"]) == enabled and (not enabled or current_matches):
            return {"child_id": child_id, "recommendation_use_enabled": enabled, "changed": False}
        conn.execute(
            """INSERT INTO learner_model_controls(
                 child_id,collection_enabled,recommendation_use_enabled,recommendation_policy_id,
                 recommendation_policy_version,recommendation_policy_hash,personalization_epoch,
                 default_retention_class,updated_by,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(child_id) DO UPDATE SET
                 recommendation_use_enabled=excluded.recommendation_use_enabled,
                 recommendation_policy_id=excluded.recommendation_policy_id,
                 recommendation_policy_version=excluded.recommendation_policy_version,
                 recommendation_policy_hash=excluded.recommendation_policy_hash,
                 personalization_epoch=excluded.personalization_epoch,
                 updated_by=excluded.updated_by,updated_at=excluded.updated_at""",
            (
                child_id,
                int(current["collection_enabled"]),
                int(enabled),
                str(active_policy["policy_id"]) if enabled else None,
                int(active_policy["version"]) if enabled else None,
                str(active_policy["definition_hash"]) if enabled else None,
                int(current["personalization_epoch"]) + 1,
                str(current["default_retention_class"]),
                actor_type,
                current["created_at"] or now,
                now,
            ),
        )
        event_id = f"persctl_{uuid4().hex[:20]}"
        conn.execute(
            """INSERT INTO personalization_control_events(
                 event_id,child_id,enabled,policy_id,policy_version,definition_hash,
                 authorization_epoch,actor_type,reason,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id,
                child_id,
                int(enabled),
                str(active_policy["policy_id"]) if enabled else None,
                int(active_policy["version"]) if enabled else None,
                str(active_policy["definition_hash"]) if enabled else None,
                int(current["personalization_epoch"]) + 1,
                actor_type,
                reason,
                now,
            ),
        )
    return {
        "child_id": child_id,
        "recommendation_use_enabled": enabled,
        "policy_id": str(active_policy["policy_id"]) if enabled else None,
        "policy_version": int(active_policy["version"]) if enabled else None,
        "definition_hash": str(active_policy["definition_hash"]) if enabled else None,
        "changed": True,
        "control_event_id": event_id,
    }


def correct_family_evidence(
    db_path: str | Path,
    *,
    child_id: str,
    evidence_id: str,
    action: CorrectionAction,
    reason: str,
    replacement: dict[str, Any] | None = None,
    actor_type: ActorType = "parent",
) -> dict[str, Any]:
    reason = _require_reason(reason)
    correction_id = f"ecorr_{uuid4().hex[:20]}"
    audit_ids: list[str] = []

    def audit_correction(conn: Any) -> None:
        audit_ids.append(
            _audit(
                conn,
                child_id=child_id,
                action="evidence_corrected",
                target_kind="evidence",
                target=evidence_id,
                actor_type=actor_type,
                details={"action": action, "correction_id": correction_id},
            )
        )

    result = correct_evidence(
        db_path,
        child_id=child_id,
        evidence_id=evidence_id,
        action=action,
        reason=reason,
        replacement=replacement,
        actor_type=actor_type,
        correction_id=correction_id,
        _transaction_hook=audit_correction,
    )
    return {**result, "audit_id": audit_ids[0] if audit_ids else None}


def set_learning_source_retention(
    db_path: str | Path,
    *,
    child_id: str,
    opportunity_id: str,
    retain_until: datetime | str,
    reason: str,
    actor_type: ActorType = "parent",
) -> dict[str, Any]:
    reason = _require_reason(reason)
    deadline = _as_utc_iso(retain_until)
    if deadline <= utcnow():
        raise ValueError("retention deadline must be in the future; delete the source to remove it now")
    init_db(db_path)
    event_id = f"ret_{uuid4().hex[:20]}"
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_child(conn, child_id)
        owned = conn.execute(
            "SELECT 1 FROM learning_opportunities WHERE opportunity_id=? AND child_id=?",
            (opportunity_id, child_id),
        ).fetchone()
        if not owned:
            raise ValueError("learning opportunity does not belong to this child")
        expiry_rows = conn.execute(
            """SELECT expires_at FROM evidence_records
               WHERE opportunity_id=? AND child_id=?""",
            (opportunity_id, child_id),
        ).fetchall()
        built_in_deadline = None
        if expiry_rows and all(row["expires_at"] is not None for row in expiry_rows):
            built_in_deadline = max(_as_utc_iso(str(row["expires_at"])) for row in expiry_rows)
        prior_row = conn.execute(
            """SELECT MIN(retain_until) AS retain_until FROM learner_retention_events
               WHERE opportunity_id=? AND child_id=?""",
            (opportunity_id, child_id),
        ).fetchone()
        prior_deadline = (
            _as_utc_iso(str(prior_row["retain_until"])) if prior_row and prior_row["retain_until"] is not None else None
        )
        bounds = [value for value in (built_in_deadline, prior_deadline) if value is not None]
        current_effective_deadline = min(bounds) if bounds else None
        if current_effective_deadline is not None and deadline > current_effective_deadline:
            raise ValueError("retention deadline may only shorten the current effective window")
        if deadline == current_effective_deadline:
            return {
                "retention_event_id": None,
                "retain_until": deadline,
                "projection_digest": recompute_child_projections_in_transaction(conn, child_id)["output_digest"],
                "audit_id": None,
                "changed": False,
            }
        conn.execute(
            """INSERT INTO learner_retention_events(
               retention_event_id,child_id,opportunity_id,retain_until,actor_type,reason,created_at
               ) VALUES(?,?,?,?,?,?,?)""",
            (event_id, child_id, opportunity_id, deadline, actor_type, reason, utcnow()),
        )
        projection = recompute_child_projections_in_transaction(conn, child_id)
        audit_id = _audit(
            conn,
            child_id=child_id,
            action="retention_changed",
            target_kind="learning_opportunity",
            target=opportunity_id,
            actor_type=actor_type,
            details={
                "retain_until": deadline,
                "previous_effective_retain_until": current_effective_deadline,
                "retention_event_id": event_id,
            },
        )
    return {
        "retention_event_id": event_id,
        "retain_until": deadline,
        "projection_digest": projection["output_digest"],
        "audit_id": audit_id,
        "changed": True,
    }


def delete_family_learning_source(
    db_path: str | Path,
    *,
    child_id: str,
    opportunity_id: str,
    reason: str,
    actor_type: ActorType = "parent",
) -> dict[str, Any]:
    reason = _require_reason(reason)
    audit_ids: list[str] = []

    def audit_deletion(conn: Any) -> None:
        audit_ids.append(
            _audit(
                conn,
                child_id=child_id,
                action="source_deleted",
                target_kind="learning_opportunity",
                target=opportunity_id,
                actor_type=actor_type,
                details={"deletion_reason": "parent_deleted" if actor_type == "parent" else "operator_cleanup"},
            )
        )

    result = delete_learning_source(
        db_path,
        child_id=child_id,
        opportunity_id=opportunity_id,
        reason="parent_deleted" if actor_type == "parent" else "operator_cleanup",
        actor_type=actor_type,
        _transaction_hook=audit_deletion,
    )
    return {**result, "audit_id": audit_ids[0] if audit_ids else None}


def delete_family_event(
    db_path: str | Path,
    *,
    child_id: str,
    event_id: str,
    confirmation: str,
    reason: str,
    output_root: str | Path | None = None,
    actor_type: ActorType = "parent",
    _test_crash_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    with managed_data_lifecycle(db_path):
        return _delete_family_event_locked(
            db_path,
            child_id=child_id,
            event_id=event_id,
            confirmation=confirmation,
            reason=reason,
            output_root=output_root,
            actor_type=actor_type,
            _test_crash_hook=_test_crash_hook,
        )


def _delete_family_event_locked(
    db_path: str | Path,
    *,
    child_id: str,
    event_id: str,
    confirmation: str,
    reason: str,
    output_root: str | Path | None = None,
    actor_type: ActorType = "parent",
    _test_crash_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Delete a raw event, every directly linked local copy, and its private files."""

    if confirmation != event_id:
        raise ValueError("event deletion requires the exact event ID as confirmation")
    reason = _require_reason(reason)
    init_db(db_path)
    operation_id = f"filedel_{uuid4().hex[:24]}"
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_child(conn, child_id)
        event = conn.execute("SELECT id FROM events WHERE id=? AND child_id=?", (event_id, child_id)).fetchone()
        if not event:
            raise ValueError("event does not belong to this child")
        existing_lock = conn.execute(
            "SELECT deletion_id FROM family_event_deletion_locks WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if existing_lock:
            operation_id = str(existing_lock["deletion_id"])
        else:
            conn.execute(
                """INSERT INTO family_event_deletion_locks(event_id,child_id,deletion_id,created_at)
                   VALUES(?,?,?,?)""",
                (event_id, child_id, operation_id, utcnow()),
            )
        file_paths = _event_file_paths(conn, event_id)
    deletion_id, staged_files = _prepare_private_file_deletion(
        db_path,
        deletion_id=operation_id,
        event_id=event_id,
        paths=file_paths,
        output_root=output_root,
    )
    if _test_crash_hook is not None:
        try:
            _test_crash_hook("after_files_staged")
        except BaseException:
            init_db(db_path)
            raise
    deletion_counts: dict[str, int] = {}
    opportunity_ids: list[str] = []
    try:
        with connect(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            _require_child(conn, child_id)
            event = conn.execute("SELECT id FROM events WHERE id=? AND child_id=?", (event_id, child_id)).fetchone()
            if not event:
                raise ValueError("event does not belong to this child")
            lock = conn.execute(
                """SELECT 1 FROM family_event_deletion_locks
                   WHERE event_id=? AND child_id=? AND deletion_id=?""",
                (event_id, child_id, operation_id),
            ).fetchone()
            if not lock:
                raise RuntimeError("event deletion reservation was lost")
            staged_originals = {_reference_path(str(original)) for original, _quarantine in staged_files}
            uncovered = {
                path
                for path in _event_file_paths(conn, event_id)
                if path.exists() and _reference_path(str(path)) not in staged_originals
            }
            if uncovered:
                raise RuntimeError("new event output appeared after deletion was reserved")
            opportunity_ids = [
                str(row["opportunity_id"])
                for row in conn.execute(
                    """WITH RECURSIVE affected_plans(plan_id) AS (
                         SELECT plan_id FROM learning_episode_plans
                         WHERE child_id=? AND source_event_id=?
                         UNION
                         SELECT child.plan_id FROM learning_episode_plans child
                         JOIN affected_plans parent
                           ON child.transfer_from_plan_id=parent.plan_id
                         WHERE child.child_id=?
                       )
                       SELECT lo.opportunity_id FROM learning_opportunities lo
                       WHERE lo.child_id=?
                         AND (lo.event_id=? OR lo.plan_id IN (SELECT plan_id FROM affected_plans))""",
                    (child_id, event_id, child_id, child_id, event_id),
                ).fetchall()
            ]
            evidence_ids = [
                str(row["evidence_id"])
                for row in conn.execute(
                    """WITH RECURSIVE affected_plans(plan_id) AS (
                         SELECT plan_id FROM learning_episode_plans
                         WHERE child_id=? AND source_event_id=?
                         UNION
                         SELECT child.plan_id FROM learning_episode_plans child
                         JOIN affected_plans parent
                           ON child.transfer_from_plan_id=parent.plan_id
                         WHERE child.child_id=?
                       )
                       SELECT er.evidence_id FROM evidence_records er
                       JOIN learning_opportunities lo
                         ON lo.opportunity_id=er.opportunity_id AND lo.child_id=er.child_id
                       WHERE er.child_id=?
                         AND (lo.event_id=? OR lo.plan_id IN (SELECT plan_id FROM affected_plans))""",
                    (child_id, event_id, child_id, child_id, event_id),
                ).fetchall()
            ]
            episode_ids = [
                str(row["episode_id"])
                for row in conn.execute(
                    "SELECT episode_id FROM episode_memberships WHERE event_id=?", (event_id,)
                ).fetchall()
            ]
            legacy_evidence_ids = {
                int(row["id"])
                for row in conn.execute(
                    "SELECT id FROM evidence WHERE event_id=? AND child_id=?",
                    (event_id, child_id),
                ).fetchall()
            }
            legacy_claim_ids: list[int] = []
            if legacy_evidence_ids:
                for row in conn.execute(
                    """SELECT id,supporting_evidence_json,contradicting_evidence_json
                       FROM claims WHERE child_id=?""",
                    (child_id,),
                ).fetchall():
                    referenced = {
                        int(value)
                        for value in (
                            jload(row["supporting_evidence_json"], []) + jload(row["contradicting_evidence_json"], [])
                        )
                    }
                    if referenced & legacy_evidence_ids:
                        legacy_claim_ids.append(int(row["id"]))
            session_ids = {
                str(row[0])
                for row in conn.execute(
                    """SELECT session_id FROM agent_messages WHERE event_id=?
                       UNION SELECT session_id FROM agent_turns WHERE event_id=?
                       UNION SELECT session_id FROM tool_calls WHERE event_id=?
                       UNION SELECT session_id FROM capability_runs
                         WHERE source_event_id=? OR produced_event_id=?
                       UNION SELECT cr.session_id FROM release_units ru
                         JOIN capability_runs cr ON cr.id=ru.capability_run_id
                         WHERE ru.event_id=?
                       UNION SELECT m.session_id FROM agent_message_event_lineage l
                         JOIN agent_messages m ON m.id=l.message_id WHERE l.event_id=?
                       UNION SELECT t.session_id FROM agent_turn_event_lineage l
                         JOIN agent_turns t ON t.id=l.turn_id WHERE l.event_id=?
                       UNION SELECT t.session_id FROM tool_call_event_lineage l
                         JOIN tool_calls t ON t.id=l.tool_call_id WHERE l.event_id=?
                       UNION SELECT cr.session_id FROM capability_run_event_lineage l
                         JOIN capability_runs cr ON cr.id=l.capability_run_id WHERE l.event_id=?
                       UNION SELECT cr.session_id FROM release_unit_event_lineage l
                         JOIN release_units ru ON ru.id=l.release_unit_id
                         JOIN capability_runs cr ON cr.id=ru.capability_run_id WHERE l.event_id=?""",
                    (
                        event_id,
                        event_id,
                        event_id,
                        event_id,
                        event_id,
                        event_id,
                        event_id,
                        event_id,
                        event_id,
                        event_id,
                        event_id,
                    ),
                ).fetchall()
                if row[0] is not None
            }
            transport_rows = [
                dict(row)
                for row in conn.execute(
                    """SELECT transport,external_event_id,binding_id FROM transport_receipts
                       WHERE event_id=?""",
                    (event_id,),
                ).fetchall()
            ]
            now = utcnow()
            for opportunity_id in opportunity_ids:
                target_hash = _target_hash(opportunity_id)
                source_tombstone_id = "tomb_" + sha256(f"{child_id}|{target_hash}".encode()).hexdigest()[:24]
                conn.execute(
                    """INSERT OR IGNORE INTO evidence_source_tombstones(
                       tombstone_id,child_id,target_kind,target_id_hash,reason,actor_type,deleted_at
                       ) VALUES(?,?,'learning_opportunity',?,'parent_deleted',?,?)""",
                    (source_tombstone_id, child_id, target_hash, actor_type, now),
                )
            event_hash = _target_hash(event_id)
            tombstone_id = "evt_tomb_" + event_hash[:24]
            conn.execute(
                """INSERT OR IGNORE INTO deleted_event_tombstones(
                   tombstone_id,event_id_hash,original_child_id_hash,reason,actor_type,deleted_at
                   ) VALUES(?,?,?,?,?,?)""",
                (tombstone_id, event_hash, _target_hash(child_id), "parent_deleted", actor_type, now),
            )
            for transport in transport_rows:
                transport_hash = transport_event_hash(
                    str(transport["transport"]),
                    str(transport["external_event_id"]),
                )
                transport_tombstone_id = "transport_tomb_" + transport_hash[:24]
                conn.execute(
                    """INSERT OR IGNORE INTO deleted_transport_event_tombstones(
                       tombstone_id,transport_event_hash,original_child_id_hash,reason,actor_type,deleted_at
                       ) VALUES(?,?,?,?,?,?)""",
                    (
                        transport_tombstone_id,
                        transport_hash,
                        _target_hash(child_id),
                        "parent_deleted",
                        actor_type,
                        now,
                    ),
                )
            if evidence_ids:
                placeholders = ",".join("?" for _ in evidence_ids)
                deletion_counts["evidence_corrections"] = conn.execute(
                    f"DELETE FROM evidence_corrections WHERE child_id=? AND evidence_id IN ({placeholders})",
                    (child_id, *evidence_ids),
                ).rowcount
            if opportunity_ids:
                placeholders = ",".join("?" for _ in opportunity_ids)
                deletion_counts["learning_opportunities"] = conn.execute(
                    f"""DELETE FROM learning_opportunities
                         WHERE child_id=? AND opportunity_id IN ({placeholders})""",
                    (child_id, *opportunity_ids),
                ).rowcount

            # Remove provider-transport correlation and every queued/sent local
            # payload copied from the source turn before dropping its receipt.
            for transport in transport_rows:
                prefix = f"{transport['transport']}:{transport['external_event_id']}:"
                deletion_counts["delivery_outbox"] = (
                    deletion_counts.get("delivery_outbox", 0)
                    + conn.execute(
                        """DELETE FROM delivery_outbox
                       WHERE substr(idempotency_key,1,length(?))=?
                         AND (binding_id=? OR ? IS NULL)""",
                        (prefix, prefix, transport["binding_id"], transport["binding_id"]),
                    ).rowcount
                )
                deletion_counts["inbound_assets"] = (
                    deletion_counts.get("inbound_assets", 0)
                    + conn.execute(
                        """DELETE FROM inbound_assets WHERE external_event_id=?
                       AND (binding_id=? OR ? IS NULL)""",
                        (transport["external_event_id"], transport["binding_id"], transport["binding_id"]),
                    ).rowcount
                )
                deletion_counts["capture_inbox"] = (
                    deletion_counts.get("capture_inbox", 0)
                    + conn.execute(
                        "DELETE FROM capture_inbox WHERE transport=? AND external_event_id=?",
                        (transport["transport"], transport["external_event_id"]),
                    ).rowcount
                )

            # Artifact approvals can contain printable lineage and print attempts
            # use a restrictive FK, so erase them before artifact rows/triggers.
            deletion_counts["print_attempts"] = conn.execute(
                """DELETE FROM print_attempts WHERE artifact_id IN (
                     SELECT a.id FROM artifacts a LEFT JOIN experiences x ON x.id=a.experience_id
                     WHERE x.source_event_id=? OR a.pedagogy_plan_id IN (
                       SELECT plan_id FROM pedagogy_plans WHERE event_id=?
                     )
                   )""",
                (event_id, event_id),
            ).rowcount
            deletion_counts["artifacts"] = conn.execute(
                """DELETE FROM artifacts WHERE id IN (
                     SELECT a.id FROM artifacts a LEFT JOIN experiences x ON x.id=a.experience_id
                     WHERE x.source_event_id=? OR a.pedagogy_plan_id IN (
                       SELECT plan_id FROM pedagogy_plans WHERE event_id=?
                     )
                   )""",
                (event_id, event_id),
            ).rowcount

            if legacy_claim_ids:
                placeholders = ",".join("?" for _ in legacy_claim_ids)
                deletion_counts["claims"] = conn.execute(
                    f"DELETE FROM claims WHERE child_id=? AND id IN ({placeholders})",
                    (child_id, *legacy_claim_ids),
                ).rowcount

            # A parent-memory row sourced from a deleted message cascades with
            # that message. Its audit record intentionally remains, but any
            # caller-supplied reason is replaced with a code-owned tombstone so
            # copied parent text cannot survive the source deletion.
            sourced_memories = [
                (str(row[0]), int(row[1]), str(row[2]))
                for row in conn.execute(
                    """SELECT DISTINCT pm.memory_key,pm.version,pm.value FROM parent_memories pm
                       JOIN agent_messages am ON am.id=pm.source_message_id
                       WHERE am.event_id=? OR EXISTS(
                         SELECT 1 FROM agent_message_event_lineage l
                         WHERE l.message_id=am.id AND l.event_id=?
                       )""",
                    (event_id, event_id),
                ).fetchall()
            ]
            sourced_memory_keys = sorted(
                {memory_key for memory_key, _version, _value in sourced_memories}
            )
            if sourced_memory_keys:
                empty_hash = sha256(b"").hexdigest()
                for memory_key, version, _value in sourced_memories:
                    record_hash = event_id_hash(f"{memory_key}\0{version}")
                    conn.execute(
                        """INSERT OR IGNORE INTO parent_memory_tombstones(
                               tombstone_id,memory_record_hash,memory_key_hash,version,actor_type,deleted_at
                           ) VALUES(?,?,?,?,?,?)""",
                        (
                            f"memory_tomb_{record_hash[:24]}",
                            record_hash,
                            event_id_hash(memory_key),
                            version,
                            actor_type,
                            now,
                        ),
                    )
                    conn.execute(
                        """UPDATE parent_memories
                           SET value='',provenance='forgotten',why_saved='forgotten',content_hash=?,
                               status='forgotten',source_message_id=NULL,updated_at=?
                           WHERE memory_key=? AND version=?""",
                        (empty_hash, now, memory_key, version),
                    )
                placeholders = ",".join("?" for _ in sourced_memory_keys)
                deletion_counts["parent_memory_event_reasons_scrubbed"] = conn.execute(
                    f"""UPDATE parent_memory_events SET reason='source deleted'
                         WHERE memory_key IN ({placeholders})""",
                    tuple(sourced_memory_keys),
                ).rowcount
                deletion_counts["parent_memory_inspections_scrubbed"] = conn.execute(
                    """UPDATE agent_messages SET content=?
                       WHERE kind='parent_memory_inspection' AND content!=?""",
                    (
                        REDACTED_PARENT_MEMORY_INSPECTION,
                        REDACTED_PARENT_MEMORY_INSPECTION,
                    ),
                ).rowcount
                deletion_counts["parent_memory_deliveries_scrubbed"] = (
                    redact_parent_memory_deliveries(
                        conn,
                        memory_keys=set(sourced_memory_keys),
                    )
                )
                deletion_counts["parent_memories_scrubbed"] = len(sourced_memories)

            direct_deletes = (
                (
                    "thread_compactions",
                    f"DELETE FROM thread_compactions WHERE session_id IN ({','.join('?' for _ in session_ids)})"
                    if session_ids
                    else "DELETE FROM thread_compactions WHERE 0",
                    tuple(sorted(session_ids)),
                ),
                ("onboarding_reviews", "DELETE FROM onboarding_reviews WHERE event_id=?", (event_id,)),
                ("output_evaluations", "DELETE FROM output_evaluations WHERE event_id=?", (event_id,)),
                (
                    "context_corrections",
                    "DELETE FROM context_corrections WHERE event_id=? OR related_event_id=?",
                    (event_id, event_id),
                ),
                (
                    "interaction_instances",
                    """DELETE FROM interaction_instances WHERE id IN (
                         SELECT interaction_id FROM interaction_instance_event_lineage WHERE event_id=?
                       )""",
                    (event_id,),
                ),
                (
                    "release_units",
                    """DELETE FROM release_units WHERE event_id=? OR id IN (
                         SELECT release_unit_id FROM release_unit_event_lineage WHERE event_id=?
                       )""",
                    (event_id, event_id),
                ),
                (
                    "tool_calls",
                    """DELETE FROM tool_calls WHERE event_id=? OR id IN (
                         SELECT tool_call_id FROM tool_call_event_lineage WHERE event_id=?
                       )""",
                    (event_id, event_id),
                ),
                (
                    "agent_turns",
                    """DELETE FROM agent_turns WHERE event_id=? OR id IN (
                         SELECT turn_id FROM agent_turn_event_lineage WHERE event_id=?
                       )""",
                    (event_id, event_id),
                ),
                (
                    "agent_messages",
                    """DELETE FROM agent_messages WHERE event_id=? OR id IN (
                         SELECT message_id FROM agent_message_event_lineage WHERE event_id=?
                       )""",
                    (event_id, event_id),
                ),
                (
                    "capability_runs",
                    """DELETE FROM capability_runs
                       WHERE source_event_id=? OR produced_event_id=? OR id IN (
                         SELECT capability_run_id FROM capability_run_event_lineage WHERE event_id=?
                       )""",
                    (event_id, event_id, event_id),
                ),
                ("transport_receipts", "DELETE FROM transport_receipts WHERE event_id=?", (event_id,)),
                ("observations", "DELETE FROM observations WHERE event_id=?", (event_id,)),
                ("evidence", "DELETE FROM evidence WHERE event_id=?", (event_id,)),
                ("feedback", "DELETE FROM feedback WHERE event_id=?", (event_id,)),
                ("actions", "DELETE FROM actions WHERE event_id=?", (event_id,)),
                ("responses", "DELETE FROM responses WHERE event_id=?", (event_id,)),
                ("graph_effects", "DELETE FROM graph_effects WHERE event_id=?", (event_id,)),
                ("visual_jobs", "DELETE FROM visual_jobs WHERE event_id=?", (event_id,)),
                ("experiences", "DELETE FROM experiences WHERE source_event_id=?", (event_id,)),
                ("jobs", "DELETE FROM jobs WHERE event_id=?", (event_id,)),
                ("runs", "DELETE FROM runs WHERE event_id=?", (event_id,)),
            )
            for name, statement, parameters in direct_deletes:
                deletion_counts[name] = conn.execute(statement, parameters).rowcount
            _rebuild_session_preferences(conn, session_ids)

            # Legacy graph nodes embed lineage in JSON rather than a relational
            # FK. Removing them also cascades their edges and evidence links.
            deletion_counts["nodes"] = conn.execute(
                """DELETE FROM nodes WHERE child_id=? AND json_valid(state_json)
                   AND json_extract(state_json,'$.source_event_id')=?""",
                (child_id, event_id),
            ).rowcount
            deleted = conn.execute("DELETE FROM events WHERE id=? AND child_id=?", (event_id, child_id)).rowcount
            if deleted != 1:
                raise RuntimeError("raw event deletion did not remove exactly one event")
            deletion_counts["events"] = deleted

            # Episode summaries duplicate the first raw question. Delete empty
            # episodes; otherwise rebuild their summary/timestamps from retained
            # members so the removed text cannot survive in the aggregate.
            for episode_id in episode_ids:
                remaining = conn.execute(
                    """SELECT e.text,e.created_at FROM episode_memberships m
                       JOIN events e ON e.id=m.event_id WHERE m.episode_id=?
                       ORDER BY e.created_at,e.id""",
                    (episode_id,),
                ).fetchall()
                if not remaining:
                    conn.execute("DELETE FROM episodes WHERE id=? AND child_id=?", (episode_id, child_id))
                else:
                    conn.execute(
                        """UPDATE episodes SET summary=?,opened_at=?,last_event_at=?,updated_at=?
                           WHERE id=? AND child_id=?""",
                        (
                            str(remaining[0]["text"]),
                            str(remaining[0]["created_at"]),
                            str(remaining[-1]["created_at"]),
                            now,
                            episode_id,
                            child_id,
                        ),
                    )
            projection = recompute_child_projections_in_transaction(conn, child_id)
            audit_id = _audit(
                conn,
                child_id=child_id,
                action="event_deleted",
                target_kind="event",
                target=event_id,
                actor_type=actor_type,
                details={
                    "deletion_reason": "parent_deleted",
                    "learning_sources_deleted": len(opportunity_ids),
                    "local_files_deleted": len(staged_files),
                    "durable_rows_deleted": sum(max(0, count) for count in deletion_counts.values()),
                },
            )
            if deletion_id is not None:
                journal_rows = conn.execute(
                    """UPDATE family_file_deletion_journal
                       SET state='db_committed',updated_at=? WHERE deletion_id=?""",
                    (utcnow(), deletion_id),
                ).rowcount
                if journal_rows != len(staged_files):
                    raise RuntimeError("private-file deletion journal is incomplete")
    except BaseException:
        init_db(db_path)
        raise
    if _test_crash_hook is not None:
        _test_crash_hook("after_db_commit")
    record_event_deletion(
        db_path,
        event_hash=event_hash,
        child_hash=_target_hash(child_id),
        deleted_at=now,
        actor_type=actor_type,
        transport_hashes=tuple(
            transport_event_hash(
                str(transport["transport"]),
                str(transport["external_event_id"]),
            )
            for transport in transport_rows
        ),
    )
    backup_purge = reconcile_database_deletions(db_path)
    # Initialization is the single idempotent recovery path. Since the event no
    # longer exists it verifies and unlinks every quarantined copy, then clears
    # the journal. An unlink failure leaves the journal durable for the next run.
    init_db(db_path)
    files_deleted = len(staged_files)
    return {
        "event_tombstone_id": tombstone_id,
        "learning_sources_deleted": len(opportunity_ids),
        "local_files_deleted": files_deleted,
        "durable_rows_deleted": sum(max(0, count) for count in deletion_counts.values()),
        "projection_digest": projection["output_digest"],
        "audit_id": audit_id,
        **backup_purge,
    }


def delete_family_model(
    db_path: str | Path,
    *,
    child_id: str,
    confirmation: str,
    reason: str,
    actor_type: ActorType = "parent",
) -> dict[str, Any]:
    with managed_data_lifecycle(db_path):
        result = _delete_family_model_locked(
            db_path,
            child_id=child_id,
            confirmation=confirmation,
            reason=reason,
            actor_type=actor_type,
        )
        return {**result, **reconcile_database_deletions(db_path)}


def _delete_family_model_locked(
    db_path: str | Path,
    *,
    child_id: str,
    confirmation: str,
    reason: str,
    actor_type: ActorType = "parent",
) -> dict[str, Any]:
    """Delete child-specific learner-model data while retaining raw family events and content-free tombstones."""

    if confirmation != child_id:
        raise ValueError("model deletion requires the exact child ID as confirmation")
    reason = _require_reason(reason)
    init_db(db_path)
    with research_connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_child(conn, child_id)
        deletion_counts: dict[str, int] = {}
        opportunities = [
            str(row[0])
            for row in conn.execute(
                "SELECT opportunity_id FROM learning_opportunities WHERE child_id=?", (child_id,)
            ).fetchall()
        ]
        now = utcnow()
        research_control = conn.execute(
            "SELECT * FROM research_track_controls WHERE child_id=?", (child_id,)
        ).fetchone()
        research_event_id: str | None = None
        if research_control and bool(research_control["enabled"]):
            research_reason = "Family learner model was deleted."
            research_epoch = int(research_control["family_authorization_epoch"]) + 1
            conn.execute(
                """UPDATE research_track_controls
                   SET enabled=0,policy_id=NULL,policy_version=NULL,definition_hash=NULL,
                       release_content_hash=NULL,family_authorization_epoch=?,updated_by=?,
                       reason=?,updated_at=? WHERE child_id=?""",
                (research_epoch, actor_type, research_reason, now, child_id),
            )
            research_event_id = research_control_event_id(
                child_id,
                False,
                None,
                None,
                None,
                None,
                research_epoch,
                actor_type,
                research_reason,
                now,
            )
            conn.execute(
                """INSERT INTO research_track_control_events(
                     event_id,child_id,enabled,policy_id,policy_version,definition_hash,
                     release_content_hash,family_authorization_epoch,actor_type,reason,created_at
                   ) VALUES(?,?,0,NULL,NULL,NULL,NULL,?,?,?,?)""",
                (
                    research_event_id,
                    child_id,
                    research_epoch,
                    actor_type,
                    research_reason,
                    now,
                ),
            )
        elif research_control:
            latest_research_event = conn.execute(
                """SELECT event_id FROM research_track_control_events WHERE child_id=?
                   ORDER BY family_authorization_epoch DESC,event_id DESC LIMIT 1""",
                (child_id,),
            ).fetchone()
            if latest_research_event:
                research_event_id = str(latest_research_event["event_id"])
        if research_event_id is not None:
            record_research_authority(
                db_path,
                conn,
                erased_control_event_ids=(research_event_id,),
            )
        for opportunity_id in opportunities:
            target_hash = _target_hash(opportunity_id)
            tombstone_id = "tomb_" + sha256(f"{child_id}|{target_hash}".encode()).hexdigest()[:24]
            conn.execute(
                """INSERT OR IGNORE INTO evidence_source_tombstones(
                   tombstone_id,child_id,target_kind,target_id_hash,reason,actor_type,deleted_at
                   ) VALUES(?,?,'learning_opportunity',?,'parent_deleted',?,?)""",
                (tombstone_id, child_id, target_hash, actor_type, now),
            )
        statements = (
            ("research_proactivity_candidates", "DELETE FROM research_proactivity_candidates WHERE child_id=?"),
            ("research_uncertainty_traces", "DELETE FROM research_uncertainty_traces WHERE child_id=?"),
            ("research_track_control_events", "DELETE FROM research_track_control_events WHERE child_id=?"),
            ("research_track_controls", "DELETE FROM research_track_controls WHERE child_id=?"),
            ("personalization_feedback_signals", "DELETE FROM personalization_feedback_signals WHERE child_id=?"),
            ("personalization_decisions", "DELETE FROM personalization_decisions WHERE child_id=?"),
            ("personalization_control_events", "DELETE FROM personalization_control_events WHERE child_id=?"),
            ("evidence_corrections", "DELETE FROM evidence_corrections WHERE child_id=?"),
            ("evidence_annotations", "DELETE FROM evidence_annotations WHERE child_id=?"),
            ("learning_opportunities", "DELETE FROM learning_opportunities WHERE child_id=?"),
            ("learning_episode_plans", "DELETE FROM learning_episode_plans WHERE child_id=?"),
            ("child_reflection_responses", "DELETE FROM child_reflection_responses WHERE child_id=?"),
            ("derived_claims_v2", "DELETE FROM derived_claims_v2 WHERE child_id=?"),
            ("learner_state_projections", "DELETE FROM learner_state_projections WHERE child_id=?"),
            ("interest_projections", "DELETE FROM interest_projections WHERE child_id=?"),
            ("observations", "DELETE FROM observations WHERE child_id=?"),
            ("claims", "DELETE FROM claims WHERE child_id=?"),
            ("nodes", "DELETE FROM nodes WHERE child_id=?"),
            ("school_signals", "DELETE FROM school_signals WHERE child_id=?"),
            ("opportunities", "DELETE FROM opportunities WHERE child_id=?"),
            ("evidence", "DELETE FROM evidence WHERE child_id=?"),
            ("episodes", "DELETE FROM episodes WHERE child_id=?"),
        )
        for table, statement in statements:
            deletion_counts[table] = conn.execute(statement, (child_id,)).rowcount
        deletion_counts["graph_effects"] = conn.execute(
            "DELETE FROM graph_effects WHERE event_id IN (SELECT id FROM events WHERE child_id=?)",
            (child_id,),
        ).rowcount
        deletion_counts["context_corrections"] = conn.execute(
            "DELETE FROM context_corrections WHERE child_id=?",
            (child_id,),
        ).rowcount
        deletion_counts["runs"] = conn.execute(
            "DELETE FROM runs WHERE event_id IN (SELECT id FROM events WHERE child_id=?)",
            (child_id,),
        ).rowcount
        deletion_counts["projection_runs"] = conn.execute(
            "DELETE FROM projection_runs WHERE child_id=?",
            (child_id,),
        ).rowcount
        deletion_counts["family_model_audit_log"] = conn.execute(
            "DELETE FROM family_model_audit_log WHERE child_id=?",
            (child_id,),
        ).rowcount
        conn.execute(
            """INSERT INTO learner_model_controls(
               child_id,collection_enabled,recommendation_use_enabled,recommendation_policy_id,
               recommendation_policy_version,recommendation_policy_hash,personalization_epoch,
               default_retention_class,updated_by,created_at,updated_at
               ) VALUES(?,0,0,NULL,NULL,NULL,1,'learning_record',?,?,?)
               ON CONFLICT(child_id) DO UPDATE SET
                 collection_enabled=0,recommendation_use_enabled=0,
                 recommendation_policy_id=NULL,recommendation_policy_version=NULL,
                 recommendation_policy_hash=NULL,
                 personalization_epoch=learner_model_controls.personalization_epoch+CASE
                   WHEN learner_model_controls.collection_enabled=0
                    AND learner_model_controls.recommendation_use_enabled=0
                    AND learner_model_controls.recommendation_policy_id IS NULL
                    AND learner_model_controls.recommendation_policy_version IS NULL
                    AND learner_model_controls.recommendation_policy_hash IS NULL
                   THEN 0 ELSE 1 END,
                 updated_by=excluded.updated_by,updated_at=excluded.updated_at""",
            (child_id, actor_type, now, now),
        )
        projection = recompute_child_projections_in_transaction(conn, child_id)
        # Recompute proves that no projection can be rebuilt from retained raw
        # conversation, then remove the content-free recomputation record too.
        conn.execute("DELETE FROM projection_runs WHERE child_id=?", (child_id,))
        audit_id = _audit(
            conn,
            child_id=child_id,
            action="model_deleted",
            target_kind="child_model",
            target=child_id,
            actor_type=actor_type,
            details={
                "deletion_reason": "parent_deleted",
                "learning_sources_deleted": len(opportunities),
                "learner_rows_deleted": sum(max(0, count) for count in deletion_counts.values()),
            },
        )
    return {
        "child_id": child_id,
        "learning_sources_deleted": len(opportunities),
        "learner_rows_deleted": sum(max(0, count) for count in deletion_counts.values()),
        "collection_enabled": False,
        "raw_events_retained": True,
        "content_free_replay_tombstones_retained": True,
        "projection_digest": projection["output_digest"],
        "audit_id": audit_id,
    }


def child_reflection_card(db_path: str | Path, child_id: str, *, limit: int = 6) -> dict[str, Any]:
    """Build optional, non-evidentiary copy that a parent may review with a child."""

    if limit < 1 or limit > 12:
        raise ValueError("reflection limit must be between 1 and 12")
    inspector = inspect_family_model(db_path, child_id, evidence_limit=1)
    combined = [
        *(
            {
                "concept_id": item["concept_id"],
                "concept_version": item["concept_version"],
                "topic": item["label"],
                "message": _CHILD_KNOWLEDGE_COPY.get(item["state"], "We are still figuring out what this shows."),
                "kind": "learning",
                "uncertainty": item["uncertainty"],
            }
            for item in inspector["inferences"]["knowledge"]
        ),
        *(
            {
                "concept_id": item["concept_id"],
                "concept_version": item["concept_version"],
                "topic": item["label"],
                "message": _CHILD_INTEREST_COPY.get(item["phase"], "We are still figuring out what feels interesting."),
                "kind": "interest",
                "uncertainty": item["uncertainty"],
            }
            for item in inspector["inferences"]["interests"]
        ),
    ]
    cards = []
    seen: set[tuple[str, int, str]] = set()
    for item in combined:
        key = (str(item["concept_id"]), int(item["concept_version"]), str(item["kind"]))
        if key in seen:
            continue
        seen.add(key)
        cards.append(
            {
                **item,
                "question": "Does that feel right to you?",
                "choices": ["yes", "not_yet", "unsure", "skip"],
            }
        )
        if len(cards) == limit:
            break
    return {
        "view_version": FAMILY_MODEL_VIEW_VERSION,
        "child_name": inspector["child"]["name"],
        "title": "What we can notice together",
        "intro": "These are careful notes from a few learning moments—not grades, labels, or a test.",
        "cards": cards,
        "empty_message": "There is not enough evidence for a reflection card yet.",
        "parent_present_required": True,
        "responses_become_evidence": False,
        "recommendations_change_automatically": False,
    }


def record_child_reflection(
    db_path: str | Path,
    *,
    child_id: str,
    concept_id: str,
    concept_version: int,
    response: Literal["yes", "not_yet", "unsure", "skip"],
    note: str | None = None,
    actor_type: ActorType = "parent",
) -> dict[str, Any]:
    if note is not None and len(note) > 500:
        raise ValueError("reflection note is limited to 500 characters")
    init_db(db_path)
    response_id = f"crev_{uuid4().hex[:20]}"
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_child(conn, child_id)
        visible = conn.execute(
            """SELECT 1 FROM learner_state_projections
               WHERE child_id=? AND concept_id=? AND concept_version=?
               UNION ALL
               SELECT 1 FROM interest_projections
               WHERE child_id=? AND concept_id=? AND concept_version=? LIMIT 1""",
            (child_id, concept_id, concept_version, child_id, concept_id, concept_version),
        ).fetchone()
        if not visible:
            raise ValueError("reflection concept is not present in this child's current model")
        conn.execute(
            """INSERT INTO child_reflection_responses(
               response_id,child_id,concept_id,concept_version,response,relayed_by,note,authority,created_at
               ) VALUES(?,?,?,?,?,?,?,'review_only',?)""",
            (response_id, child_id, concept_id, concept_version, response, actor_type, note, utcnow()),
        )
        audit_id = _audit(
            conn,
            child_id=child_id,
            action="child_reflection_recorded",
            target_kind="concept",
            target=f"{concept_id}@{concept_version}",
            actor_type=actor_type,
            details={"response": response, "response_id": response_id, "authority": "review_only"},
        )
    return {
        "response_id": response_id,
        "authority": "review_only",
        "became_evidence": False,
        "projection_changed": False,
        "audit_id": audit_id,
    }


def _export_rows(conn: Any, statement: str, parameters: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Return complete rows while adding readable forms of embedded JSON fields."""

    result: list[dict[str, Any]] = []
    for row in conn.execute(statement, parameters).fetchall():
        item = dict(row)
        for key, value in tuple(item.items()):
            if key.endswith("_json") and isinstance(value, str):
                item[f"{key[:-5]}_decoded"] = jload(value, None)
        result.append(item)
    return result


def _complete_family_model_records(conn: Any, child_id: str) -> dict[str, list[dict[str, Any]]]:
    """Export every child-owned learner record and the definitions it references."""

    direct_child_tables = (
        "learner_model_controls",
        "personalization_control_events",
        "personalization_decisions",
        "personalization_feedback_signals",
        "research_track_controls",
        "research_track_control_events",
        "research_uncertainty_traces",
        "research_proactivity_candidates",
        "learning_episode_plans",
        "evidence_annotations",
        "learning_opportunities",
        "opportunity_assistance_events",
        "evidence_records",
        "evidence_corrections",
        "projection_runs",
        "learner_state_projections",
        "learner_projection_evidence",
        "interest_projections",
        "interest_projection_evidence",
        "derived_claims_v2",
        "derived_claim_evidence_v2",
        "preference_support_records",
        "learner_retention_events",
        "child_reflection_responses",
        "family_model_audit_log",
        "evidence_source_tombstones",
        "nodes",
        "edges",
        "observations",
        "school_signals",
        "claims",
        "episodes",
        "evidence",
        "opportunities",
    )
    records = {
        table: _export_rows(conn, f'SELECT * FROM "{table}" WHERE child_id=? ORDER BY rowid', (child_id,))
        for table in direct_child_tables
    }
    records["episode_memberships"] = _export_rows(
        conn,
        """SELECT m.* FROM episode_memberships m JOIN episodes e ON e.id=m.episode_id
           WHERE e.child_id=? ORDER BY m.rowid""",
        (child_id,),
    )
    records["node_evidence"] = _export_rows(
        conn,
        """SELECT ne.* FROM node_evidence ne JOIN nodes n ON n.id=ne.node_id
           WHERE n.child_id=? ORDER BY ne.rowid""",
        (child_id,),
    )
    records["learning_preferences"] = _export_rows(
        conn,
        """SELECT * FROM learning_preferences
           WHERE scope_type='child' AND scope_id=? ORDER BY rowid""",
        (child_id,),
    )
    records["context_corrections"] = _export_rows(
        conn,
        "SELECT * FROM context_corrections WHERE child_id=? ORDER BY rowid",
        (child_id,),
    )
    records["graph_effects"] = _export_rows(
        conn,
        """SELECT g.* FROM graph_effects g JOIN events e ON e.id=g.event_id
           WHERE e.child_id=? ORDER BY g.rowid""",
        (child_id,),
    )
    records["runs"] = _export_rows(
        conn,
        """SELECT r.* FROM runs r JOIN events e ON e.id=r.event_id
           WHERE e.child_id=? ORDER BY r.rowid""",
        (child_id,),
    )
    records["source_events"] = _export_rows(
        conn,
        """SELECT e.* FROM events e WHERE e.child_id=? AND (
             e.id IN (SELECT source_event_id FROM learning_episode_plans WHERE child_id=?)
             OR e.id IN (SELECT event_id FROM learning_opportunities WHERE child_id=?)
             OR e.id IN (SELECT event_id FROM evidence_annotations WHERE child_id=?)
             OR e.id IN (SELECT event_id FROM personalization_decisions WHERE child_id=?)
             OR e.id IN (SELECT event_id FROM observations WHERE child_id=? AND event_id IS NOT NULL)
             OR e.id IN (SELECT event_id FROM evidence WHERE child_id=? AND event_id IS NOT NULL)
             OR e.id IN (
               SELECT m.event_id FROM episode_memberships m JOIN episodes ep ON ep.id=m.episode_id
               WHERE ep.child_id=?
             )
           ) ORDER BY e.created_at,e.id""",
        (child_id, child_id, child_id, child_id, child_id, child_id, child_id, child_id),
    )
    policy_filter = """(
      SELECT policy_id,policy_version FROM personalization_decisions WHERE child_id=?
      UNION SELECT policy_id,policy_version FROM personalization_control_events
        WHERE child_id=? AND policy_id IS NOT NULL
    )"""
    records["personalization_policies"] = _export_rows(
        conn,
        f"""SELECT p.* FROM personalization_policies p
             WHERE (p.policy_id,p.version) IN {policy_filter} ORDER BY p.policy_id,p.version""",
        (child_id, child_id),
    )
    records["personalization_evaluations"] = _export_rows(
        conn,
        """SELECT e.* FROM personalization_evaluations e WHERE e.evaluation_id IN (
             SELECT evaluation_id FROM personalization_decisions
             WHERE child_id=? AND evaluation_id IS NOT NULL
             UNION SELECT p.evaluation_id FROM personalization_policies p
               WHERE p.evaluation_id IS NOT NULL AND (p.policy_id,p.version) IN (
                 SELECT policy_id,policy_version FROM personalization_control_events
                 WHERE child_id=? AND policy_id IS NOT NULL
               )
           ) ORDER BY e.evaluation_id""",
        (child_id, child_id),
    )
    records["personalization_promotion_reviews"] = _export_rows(
        conn,
        """SELECT r.* FROM personalization_promotion_reviews r WHERE r.review_id IN (
             SELECT promotion_review_id FROM personalization_decisions
             WHERE child_id=? AND promotion_review_id IS NOT NULL
             UNION SELECT p.promotion_review_id FROM personalization_policies p
               WHERE p.promotion_review_id IS NOT NULL AND (p.policy_id,p.version) IN (
                 SELECT policy_id,policy_version FROM personalization_control_events
                 WHERE child_id=? AND policy_id IS NOT NULL
               )
           ) ORDER BY r.review_id""",
        (child_id, child_id),
    )
    research_policy_filter = """(
      SELECT policy_id,policy_version FROM research_track_controls
        WHERE child_id=? AND policy_id IS NOT NULL
      UNION SELECT policy_id,policy_version FROM research_track_control_events
        WHERE child_id=? AND policy_id IS NOT NULL
      UNION SELECT policy_id,policy_version FROM research_uncertainty_traces WHERE child_id=?
      UNION SELECT policy_id,policy_version FROM research_proactivity_candidates WHERE child_id=?
    )"""
    records["research_track_policies"] = _export_rows(
        conn,
        f"""SELECT p.* FROM research_track_policies p
             WHERE (p.policy_id,p.version) IN {research_policy_filter}
             ORDER BY p.policy_id,p.version""",
        (child_id, child_id, child_id, child_id),
    )
    records["research_track_evaluations"] = _export_rows(
        conn,
        f"""SELECT e.* FROM research_track_evaluations e
             WHERE (e.policy_id,e.policy_version) IN {research_policy_filter}
             ORDER BY e.evaluation_id""",
        (child_id, child_id, child_id, child_id),
    )
    records["research_track_promotion_reviews"] = _export_rows(
        conn,
        f"""SELECT r.* FROM research_track_promotion_reviews r
             WHERE (r.policy_id,r.policy_version) IN {research_policy_filter}
             ORDER BY r.review_id""",
        (child_id, child_id, child_id, child_id),
    )
    records["research_track_policy_events"] = _export_rows(
        conn,
        f"""SELECT pe.* FROM research_track_policy_events pe
             WHERE (pe.policy_id,pe.policy_version) IN {research_policy_filter}
             ORDER BY pe.created_at,pe.event_id""",
        (child_id, child_id, child_id, child_id),
    )
    records["research_track_authorization_snapshots"] = _export_rows(
        conn,
        """SELECT s.* FROM research_track_authorization_snapshots s
             WHERE s.snapshot_id IN (
               SELECT authorization_snapshot_id FROM research_uncertainty_traces
               WHERE child_id=?
             ) OR EXISTS(
               SELECT 1 FROM research_track_controls c
               WHERE c.child_id=? AND c.policy_id=s.policy_id
                 AND c.policy_version=s.policy_version
                 AND c.definition_hash=s.definition_hash
                 AND c.release_content_hash=s.release_content_hash
             ) OR EXISTS(
               SELECT 1 FROM research_track_control_events e
               WHERE e.child_id=? AND e.enabled=1 AND e.policy_id=s.policy_id
                 AND e.policy_version=s.policy_version
                 AND e.definition_hash=s.definition_hash
                 AND e.release_content_hash=s.release_content_hash
             ) ORDER BY s.policy_authorization_epoch,s.snapshot_id""",
        (child_id, child_id, child_id),
    )
    task_filter = """(
      SELECT task_id,task_version FROM learning_episode_plans WHERE child_id=?
      UNION SELECT task_id,task_version FROM learning_opportunities WHERE child_id=?
    )"""
    records["learning_tasks"] = _export_rows(
        conn,
        f"""SELECT t.* FROM learning_tasks t
             WHERE (t.task_id,t.version) IN {task_filter} ORDER BY t.task_id,t.version""",
        (child_id, child_id),
    )
    records["learning_task_concepts"] = _export_rows(
        conn,
        f"""SELECT tc.* FROM learning_task_concepts tc
             WHERE (tc.task_id,tc.task_version) IN {task_filter}
             ORDER BY tc.task_id,tc.task_version,tc.concept_id,tc.role""",
        (child_id, child_id),
    )
    records["rubrics"] = _export_rows(
        conn,
        """SELECT r.* FROM rubrics r WHERE (r.rubric_id,r.version) IN (
             SELECT rubric_id,rubric_version FROM learning_opportunities WHERE child_id=?
             UNION SELECT rubric_id,rubric_version FROM evidence_records WHERE child_id=?
             UNION SELECT t.rubric_id,t.rubric_version FROM learning_episode_plans p
               JOIN learning_tasks t ON t.task_id=p.task_id AND t.version=p.task_version
               WHERE p.child_id=?
           ) ORDER BY r.rubric_id,r.version""",
        (child_id, child_id, child_id),
    )
    concept_filter = f"""(
      SELECT concept_id,concept_version FROM evidence_records WHERE child_id=?
      UNION SELECT concept_id,concept_version FROM learner_state_projections WHERE child_id=?
      UNION SELECT concept_id,concept_version FROM interest_projections WHERE child_id=?
      UNION SELECT tc.concept_id,tc.concept_version FROM learning_task_concepts tc
        WHERE (tc.task_id,tc.task_version) IN {task_filter}
    )"""
    concept_parameters = (child_id, child_id, child_id, child_id, child_id)
    records["concepts"] = _export_rows(
        conn,
        f"""SELECT c.* FROM concepts c WHERE (c.concept_id,c.version) IN {concept_filter}
             ORDER BY c.concept_id,c.version""",
        concept_parameters,
    )
    records["concept_aliases"] = _export_rows(
        conn,
        f"""SELECT a.* FROM concept_aliases a
             WHERE (a.concept_id,a.concept_version) IN {concept_filter}
             ORDER BY a.concept_id,a.concept_version,a.normalized_alias""",
        concept_parameters,
    )
    records["concept_relations"] = _export_rows(
        conn,
        f"""SELECT r.* FROM concept_relations r WHERE
             (r.source_concept_id,r.source_concept_version) IN {concept_filter}
             OR (r.target_concept_id,r.target_concept_version) IN {concept_filter}
             ORDER BY r.relation_id""",
        (*concept_parameters, *concept_parameters),
    )
    return records


def export_family_model(db_path: str | Path, child_id: str, *, actor_type: ActorType = "parent") -> dict[str, Any]:
    """Create a complete, child-scoped JSON-serializable learner-model export."""

    inspector = inspect_family_model(
        db_path,
        child_id,
        evidence_limit=None,
        projection_limit=None,
        audit_limit=None,
    )
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        audit_id = _audit(
            conn,
            child_id=child_id,
            action="model_exported",
            target_kind="export",
            target=f"{child_id}|{utcnow()}",
            actor_type=actor_type,
            details={"format": "json", "view_version": FAMILY_MODEL_VIEW_VERSION},
        )
        records = _complete_family_model_records(conn, child_id)
        tombstones = [
            {
                "target_kind": "learning_opportunity",
                "target_id_hash": str(row["target_id_hash"]),
                "reason": str(row["reason"]),
                "actor_type": str(row["actor_type"]),
                "deleted_at": str(row["deleted_at"]),
            }
            for row in records["evidence_source_tombstones"]
        ]
        tombstones.extend(
            {
                "target_kind": "event",
                "target_id_hash": str(row["event_id_hash"]),
                "reason": str(row["reason"]),
                "actor_type": str(row["actor_type"]),
                "deleted_at": str(row["deleted_at"]),
            }
            for row in conn.execute(
                """SELECT event_id_hash,reason,actor_type,deleted_at
                   FROM deleted_event_tombstones WHERE original_child_id_hash=?
                   ORDER BY deleted_at,tombstone_id""",
                (_target_hash(child_id),),
            ).fetchall()
        )
        tombstones.extend(
            {
                "target_kind": "transport_event",
                "target_id_hash": str(row["transport_event_hash"]),
                "reason": str(row["reason"]),
                "actor_type": str(row["actor_type"]),
                "deleted_at": str(row["deleted_at"]),
            }
            for row in conn.execute(
                """SELECT transport_event_hash,reason,actor_type,deleted_at
                   FROM deleted_transport_event_tombstones WHERE original_child_id_hash=?
                   ORDER BY deleted_at,tombstone_id""",
                (_target_hash(child_id),),
            ).fetchall()
        )
    audit_rows = records["family_model_audit_log"]
    inspector["audit"] = [
        {
            **item,
            "details": item.get("details_decoded") or {},
        }
        for item in reversed(audit_rows)
    ]
    return {
        "format": "curiosity-engine-family-model",
        "format_version": 2,
        "exported_at": utcnow(),
        "inspector": inspector,
        "records": records,
        "corrections": records["evidence_corrections"],
        "retention_events": records["learner_retention_events"],
        "child_reflection_responses": records["child_reflection_responses"],
        "content_free_tombstones": tombstones,
        "export_audit_id": audit_id,
        "scope": "learner model only; transport credentials and other children's data are excluded",
    }


def write_family_model_export(
    db_path: str | Path,
    child_id: str,
    destination: str | Path,
    *,
    actor_type: ActorType = "parent",
    overwrite: bool = False,
) -> dict[str, Any]:
    path = Path(destination).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = export_family_model(db_path, child_id, actor_type=actor_type)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "exported",
        "path": str(path),
        "mode": oct(path.stat().st_mode & 0o777),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "child_id": child_id,
    }


__all__ = [
    "FAMILY_MODEL_VIEW_VERSION",
    "child_reflection_card",
    "correct_family_evidence",
    "delete_family_event",
    "delete_family_learning_source",
    "delete_family_model",
    "export_family_model",
    "inspect_family_model",
    "record_child_reflection",
    "set_collection_enabled",
    "set_learning_source_retention",
    "write_family_model_export",
]
