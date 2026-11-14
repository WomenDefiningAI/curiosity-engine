from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import ActionRequest, Event, GraphMutation, RunResult
from .db import connect, init_db, jdump, jload, utcnow
from .episodes import resolve_episode
from .graph import apply_graph_mutation

logger = logging.getLogger(__name__)


class IdempotencyConflict(ValueError):
    pass


@dataclass(frozen=True)
class Submission:
    event_id: str
    job_id: str
    duplicate: bool
    result: RunResult | None = None


@dataclass(frozen=True)
class Job:
    id: str
    job_type: str
    event_id: str | None
    payload: dict[str, Any]
    attempts: int


class Repository:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        init_db(self.db_path)

    def submit_event(self, event: Event, *, agent_lineage: dict[str, str] | None = None) -> Submission:
        job_id = f"job_{event.id}"
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload_hash FROM events WHERE id=?", (event.id,)).fetchone()
            if row:
                if row["payload_hash"] and row["payload_hash"] != event.payload_hash:
                    raise IdempotencyConflict(f"event id {event.id!r} was already used for a different payload")
                if agent_lineage:
                    self._bind_agent_lineage(conn, event, agent_lineage)
                response = self._response_from_conn(conn, event.id)
                return Submission(event.id, job_id, True, response)
            deleted = conn.execute(
                "SELECT 1 FROM deleted_event_tombstones WHERE event_id_hash=?",
                (sha256(event.id.encode()).hexdigest(),),
            ).fetchone()
            if deleted:
                raise IdempotencyConflict("deleted event cannot be recreated from a stale replay")
            collection_enabled = True
            if event.child_id:
                control = conn.execute(
                    "SELECT collection_enabled FROM learner_model_controls WHERE child_id=?",
                    (event.child_id,),
                ).fetchone()
                collection_enabled = control is None or bool(control["collection_enabled"])
            conn.execute(
                """INSERT INTO events(
                   id,type,child_id,text,source,metadata_json,created_at,status,payload_hash
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    event.id,
                    event.type,
                    event.child_id,
                    event.text,
                    event.source,
                    jdump(event.metadata),
                    event.created_at.isoformat(),
                    "queued",
                    event.payload_hash,
                ),
            )
            if collection_enabled:
                episode = resolve_episode(conn, event)
                conn.execute(
                    """INSERT INTO evidence(
                       child_id,event_id,episode_id,evidence_type,content,source,confidence,metadata_json,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        event.child_id,
                        event.id,
                        episode["episode_id"] if episode else None,
                        "raw_event",
                        event.text,
                        event.source,
                        1.0,
                        jdump(event.metadata),
                        now,
                    ),
                )
            conn.execute(
                """INSERT INTO jobs(id,job_type,status,payload_json,attempts,created_at,updated_at,idempotency_key,event_id,available_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id,
                    "process_event",
                    "queued",
                    jdump({"event_id": event.id}),
                    0,
                    now,
                    now,
                    event.id,
                    event.id,
                    now,
                ),
            )
            if agent_lineage:
                self._bind_agent_lineage(conn, event, agent_lineage)
        return Submission(event.id, job_id, False)

    @staticmethod
    def _bind_agent_lineage(conn: Any, event: Event, lineage: dict[str, str]) -> None:
        """Bind pre-persisted parent-agent records in the event transaction."""

        required = {"session_id", "turn_id", "capability_run_id", "tool_call_id", "source_message_id"}
        if set(lineage) != required or not all(str(lineage[key]).strip() for key in required):
            raise ValueError("agent event lineage is incomplete")
        session_id = str(lineage["session_id"])
        turn_id = str(lineage["turn_id"])
        run_id = str(lineage["capability_run_id"])
        tool_call_id = str(lineage["tool_call_id"])
        message_id = str(lineage["source_message_id"])
        owned = conn.execute(
            """SELECT 1 FROM agent_sessions s
               JOIN agent_messages m ON m.session_id=s.id AND m.id=? AND m.role='user'
               JOIN agent_turns t ON t.session_id=s.id AND t.id=?
               JOIN capability_runs cr ON cr.session_id=s.id AND cr.id=?
               JOIN tool_calls tc ON tc.session_id=s.id AND tc.turn_id=t.id AND tc.id=?
               WHERE s.id=? AND s.child_id=?""",
            (message_id, turn_id, run_id, tool_call_id, session_id, event.child_id),
        ).fetchone()
        if not owned:
            raise ValueError("agent event lineage does not belong to the event child and turn")
        conn.execute("UPDATE agent_messages SET event_id=? WHERE id=?", (event.id, message_id))
        conn.execute("UPDATE agent_turns SET event_id=? WHERE id=?", (event.id, turn_id))
        conn.execute("UPDATE tool_calls SET event_id=? WHERE id=?", (event.id, tool_call_id))
        conn.execute("UPDATE capability_runs SET produced_event_id=? WHERE id=?", (event.id, run_id))
        lineage_rows = (
            ("agent_message_event_lineage", "message_id", message_id),
            ("agent_turn_event_lineage", "turn_id", turn_id),
            ("tool_call_event_lineage", "tool_call_id", tool_call_id),
            ("capability_run_event_lineage", "capability_run_id", run_id),
        )
        for table, column, record_id in lineage_rows:
            conn.execute(
                f"INSERT OR IGNORE INTO {table}({column},event_id) VALUES(?,?)",
                (record_id, event.id),
            )

    def claim_job(self, worker_id: str, job_id: str | None = None, lease_seconds: int = 300) -> Job | None:
        now = datetime.now(UTC)
        stale = (now - timedelta(seconds=lease_seconds)).isoformat()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """UPDATE jobs SET status='queued',lease_owner=NULL,leased_at=NULL,updated_at=?
                   WHERE status='running' AND leased_at<?""",
                (now.isoformat(), stale),
            )
            params: list[Any] = [now.isoformat()]
            where = "status='queued' AND COALESCE(available_at,created_at)<=?"
            if job_id:
                where += " AND id=?"
                params.append(job_id)
            row = conn.execute(
                f"SELECT * FROM jobs WHERE {where} ORDER BY created_at LIMIT 1",
                params,
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """UPDATE jobs SET status='running',attempts=attempts+1,leased_at=?,lease_owner=?,updated_at=?
                   WHERE id=?""",
                (now.isoformat(), worker_id, now.isoformat(), row["id"]),
            )
            return Job(
                str(row["id"]),
                str(row["job_type"]),
                row["event_id"],
                jload(row["payload_json"]),
                int(row["attempts"]) + 1,
            )

    def event(self, event_id: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
            if not row:
                raise KeyError(event_id)
            result = dict(row)
            result["metadata"] = jload(result.pop("metadata_json"))
            return result

    def start_run(self, event_id: str, workflow: str, policy: dict[str, Any], job_id: str) -> int:
        now = utcnow()
        with connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO runs(job,status,started_at,details_json,event_id,policy_json)
                   VALUES(?,?,?,?,?,?)""",
                (workflow, "running", now, jdump({"job_id": job_id}), event_id, jdump(policy)),
            )
            run_id = int(cur.lastrowid)
            conn.execute("UPDATE jobs SET run_id=?,updated_at=? WHERE id=?", (run_id, now, job_id))
            return run_id

    def pedagogy_plan(self, event_id: str, workflow: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT approved_json,status,plan_id,plan_hash,event_id,child_id,workflow,
                          registry_version,registry_digest
                   FROM pedagogy_plans
                   WHERE event_id=? AND workflow=? AND status IN ('approved','generated')
                   ORDER BY created_at DESC LIMIT 1""",
                (event_id, workflow),
            ).fetchone()
        if not row:
            return None
        from .pedagogy import ApprovedPedagogyPlan, PedagogyRegistry

        approved = ApprovedPedagogyPlan.model_validate(jload(row["approved_json"]))
        PedagogyRegistry().validate_plan(approved)
        if any(
            (
                approved.plan_id != str(row["plan_id"]),
                approved.plan_hash != str(row["plan_hash"]),
                approved.event_id != str(row["event_id"]),
                approved.child_id != str(row["child_id"]),
                approved.workflow != str(row["workflow"]),
                approved.registry_version != str(row["registry_version"]),
                approved.registry_digest != str(row["registry_digest"]),
            )
        ):
            raise ValueError("stored pedagogy plan columns do not match its canonical payload")
        return {**approved.model_dump(mode="json"), "_status": str(row["status"])}

    def persist_pedagogy_plan(
        self,
        *,
        plan: Any,
        proposal: dict[str, Any],
        run_id: int | None,
        personalization_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist once before generation; exact replays reuse the same immutable plan."""

        from .pedagogy import ApprovedPedagogyPlan, PedagogyRegistry, materialize_pedagogy_definitions

        approved = plan if isinstance(plan, ApprovedPedagogyPlan) else ApprovedPedagogyPlan.model_validate(plan)
        if approved.personalization is not None:
            binding = approved.personalization
            if not personalization_decision or personalization_decision.get("mode") != "active":
                raise ValueError("active personalization plan requires its exact decision")
            expected_binding = {
                "decision_id": personalization_decision.get("decision_id"),
                "decision_hash": personalization_decision.get("decision_hash"),
                "child_id": personalization_decision.get("child_id"),
                "event_id": personalization_decision.get("event_id"),
                "event_type": personalization_decision.get("event_type"),
                "event_text_hash": personalization_decision.get("event_text_hash"),
                "workflow": personalization_decision.get("workflow"),
                "proposal_hash": personalization_decision.get("proposal_hash"),
                "baseline_plan_id": personalization_decision.get("baseline_plan_id"),
                "baseline_plan_hash": personalization_decision.get("baseline_plan_hash"),
                "concept_ids": personalization_decision.get("concept_ids"),
                "concept_refs": personalization_decision.get("concept_refs"),
                "question_intent": personalization_decision.get("question_intent"),
                "policy_id": personalization_decision.get("policy_id"),
                "policy_version": personalization_decision.get("policy_version"),
                "definition_hash": personalization_decision.get("definition_hash"),
                "family_authorization_epoch": personalization_decision.get(
                    "family_authorization_epoch"
                ),
                "policy_authorization_epoch": personalization_decision.get(
                    "policy_authorization_epoch"
                ),
                "evaluation_id": personalization_decision.get("evaluation_id"),
                "promotion_review_id": personalization_decision.get("promotion_review_id"),
                "mode": "active",
                "baseline_strategy_id": (personalization_decision.get("baseline") or {}).get("strategy_id"),
                "interest_support": (personalization_decision.get("candidate") or {}).get("interest_support"),
                "challenge_assumption": (personalization_decision.get("candidate") or {}).get(
                    "challenge_assumption"
                ),
                "applied_fields": ["interest_support", "challenge_assumption"],
            }
            if binding.model_dump(mode="json") != expected_binding:
                raise ValueError("personalization plan and decision do not match")
            if any(
                (
                    approved.child_id != personalization_decision.get("child_id"),
                    approved.event_id != personalization_decision.get("event_id"),
                    approved.workflow != personalization_decision.get("workflow"),
                    approved.question_intent != personalization_decision.get("question_intent"),
                    approved.question_intent in {"direct_answer", "sensitive_support"},
                    [item.id for item in approved.concepts] != personalization_decision.get("concept_ids"),
                    [
                        {"id": item.id, "version": item.version}
                        for item in approved.concepts
                    ]
                    != personalization_decision.get("concept_refs"),
                    approved.primary_strategy.id
                    != (personalization_decision.get("baseline") or {}).get("strategy_id"),
                    approved.interest_support
                    != (personalization_decision.get("candidate") or {}).get("interest_support"),
                    approved.challenge_assumption
                    != (personalization_decision.get("candidate") or {}).get("challenge_assumption"),
                    sha256(jdump(proposal).encode()).hexdigest()
                    != personalization_decision.get("proposal_hash"),
                )
            ):
                raise ValueError("personalization decision cannot be rebound to this plan")
        elif personalization_decision and personalization_decision.get("mode") != "shadow":
            raise ValueError("active personalization decision requires an active plan binding")
        elif personalization_decision and any(
            (
                approved.child_id != personalization_decision.get("child_id"),
                approved.event_id != personalization_decision.get("event_id"),
                approved.workflow != personalization_decision.get("workflow"),
                approved.plan_id != personalization_decision.get("baseline_plan_id"),
                approved.plan_hash != personalization_decision.get("baseline_plan_hash"),
                sha256(jdump(proposal).encode()).hexdigest()
                != personalization_decision.get("proposal_hash"),
            )
        ):
            raise ValueError("shadow personalization decision cannot be rebound to this plan")
        try:
            PedagogyRegistry().validate_plan(approved)
        except ValueError as exc:
            raise IdempotencyConflict("invalid or conflicting pedagogy plan replay") from exc
        payload = approved.model_dump(mode="json")
        materialize_pedagogy_definitions(self.db_path, approved)
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            event = conn.execute("SELECT child_id FROM events WHERE id=?", (approved.event_id,)).fetchone()
            if not event or str(event["child_id"] or "") != approved.child_id:
                raise ValueError("pedagogy plan child/event binding does not match")
            if run_id is not None:
                run = conn.execute("SELECT event_id FROM runs WHERE id=?", (run_id,)).fetchone()
                if not run or str(run["event_id"] or "") != approved.event_id:
                    raise ValueError("pedagogy plan run/event binding does not match")
            if approved.routing_provenance == "artifact_derivation":
                parent = conn.execute(
                    "SELECT event_id,child_id FROM pedagogy_plans WHERE plan_id=?",
                    (approved.supersedes_plan_id,),
                ).fetchone()
                if (
                    approved.supersedes_plan_id is None
                    or not parent
                    or str(parent["event_id"]) != approved.event_id
                    or str(parent["child_id"]) != approved.child_id
                ):
                    raise ValueError("artifact pedagogy supersession must stay within its source event")
            elif approved.supersedes_plan_id is not None and approved.routing_provenance != "visual_revision":
                raise ValueError("only artifact or visual revisions may supersede a pedagogy plan")
            conn.execute(
                """INSERT INTO pedagogy_plans(
                     plan_id,plan_hash,event_id,child_id,initial_run_id,workflow,status,proposal_json,approved_json,
                     registry_version,registry_digest,contract_version,validator_version,routing_provenance,
                     supersedes_plan_id,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,'approved',?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(plan_id) DO NOTHING""",
                (
                    approved.plan_id,
                    approved.plan_hash,
                    approved.event_id,
                    approved.child_id,
                    run_id,
                    approved.workflow,
                    jdump(proposal),
                    jdump(payload),
                    approved.registry_version,
                    approved.registry_digest,
                    approved.contract_version,
                    approved.validator_version,
                    approved.routing_provenance,
                    approved.supersedes_plan_id,
                    approved.created_at.isoformat(),
                    now,
                ),
            )
            stored = conn.execute(
                "SELECT plan_hash,event_id,child_id,workflow,approved_json,status FROM pedagogy_plans WHERE plan_id=?",
                (approved.plan_id,),
            ).fetchone()
            stored_payload = jload(stored["approved_json"]) if stored else None
            stored_authority = dict(stored_payload or {})
            contender_authority = dict(payload)
            # Creation time is audit metadata, not pedagogical authority. Two
            # workers can compile the same deterministic plan milliseconds apart;
            # the winner's timestamp becomes canonical while every authority-
            # bearing field must still be identical.
            stored_authority.pop("created_at", None)
            contender_authority.pop("created_at", None)
            if not stored or any(
                (
                    str(stored["plan_hash"]) != approved.plan_hash,
                    str(stored["event_id"]) != approved.event_id,
                    str(stored["child_id"]) != approved.child_id,
                    str(stored["workflow"]) != approved.workflow,
                    jdump(stored_authority) != jdump(contender_authority),
                )
            ):
                raise IdempotencyConflict("conflicting pedagogy plan replay")
            if stored["status"] not in {"approved", "generated"}:
                raise IdempotencyConflict("pedagogy plan is no longer reusable")
            if run_id is not None:
                conn.execute(
                    "UPDATE runs SET pedagogy_plan_id=?,pedagogy_plan_hash=? WHERE id=?",
                    (approved.plan_id, approved.plan_hash, run_id),
                )
            if personalization_decision is not None:
                from .personalization import insert_decision_in_transaction

                insert_decision_in_transaction(
                    conn,
                    decision=personalization_decision,
                    plan=approved,
                )
        return stored_payload

    def bind_pedagogy_plan_to_run(self, *, plan: Any, run_id: int) -> None:
        from .pedagogy import ApprovedPedagogyPlan

        approved = plan if isinstance(plan, ApprovedPedagogyPlan) else ApprovedPedagogyPlan.model_validate(plan)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute("SELECT event_id FROM runs WHERE id=?", (run_id,)).fetchone()
            stored = conn.execute(
                "SELECT event_id,child_id,plan_hash,status FROM pedagogy_plans WHERE plan_id=?",
                (approved.plan_id,),
            ).fetchone()
            if (
                not run
                or not stored
                or str(run["event_id"] or "") != approved.event_id
                or str(stored["event_id"]) != approved.event_id
                or str(stored["child_id"]) != approved.child_id
                or str(stored["plan_hash"]) != approved.plan_hash
                or stored["status"] not in {"approved", "generated"}
            ):
                raise ValueError("cannot bind run to stale or mismatched pedagogy plan")
            conn.execute(
                "UPDATE runs SET pedagogy_plan_id=?,pedagogy_plan_hash=? WHERE id=?",
                (approved.plan_id, approved.plan_hash, run_id),
            )

    def record_initial_mutations(
        self,
        event_id: str,
        child_id: str,
        run_id: int,
        mutations: list[GraphMutation],
    ) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self._apply_mutations(conn, event_id, child_id, run_id, mutations)

    def complete_event(
        self,
        *,
        event_id: str,
        job_id: str,
        run_id: int,
        workflow: str,
        output: dict[str, Any],
        graph_updates: list[GraphMutation],
        actions: list[ActionRequest],
        child_id: str | None,
        pedagogy_plan: Any | None = None,
    ) -> RunResult:
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            if pedagogy_plan is not None:
                from .personalization import assert_plan_authorization_in_transaction

                assert_plan_authorization_in_transaction(conn, pedagogy_plan)
            effects = self._apply_mutations(conn, event_id, child_id, run_id, graph_updates, skip_invalid=True)
            stored_actions = self._store_actions(conn, event_id, run_id, actions, pedagogy_plan)
            plan_id = pedagogy_plan.plan_id if pedagogy_plan is not None else None
            plan_hash = pedagogy_plan.plan_hash if pedagogy_plan is not None else None
            conn.execute(
                """INSERT INTO responses(event_id,run_id,workflow,status,output_json,created_at,updated_at,
                   pedagogy_plan_id,pedagogy_plan_hash)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(event_id) DO UPDATE SET run_id=excluded.run_id,workflow=excluded.workflow,
                   status=excluded.status,output_json=excluded.output_json,updated_at=excluded.updated_at,
                   pedagogy_plan_id=excluded.pedagogy_plan_id,pedagogy_plan_hash=excluded.pedagogy_plan_hash""",
                (event_id, run_id, workflow, "completed", jdump(output), now, now, plan_id, plan_hash),
            )
            if pedagogy_plan is not None:
                conn.execute(
                    """UPDATE pedagogy_plans SET status='generated',updated_at=?
                       WHERE plan_id=? AND plan_hash=? AND event_id=? AND child_id=?""",
                    (now, plan_id, plan_hash, event_id, child_id),
                )
            result = RunResult(
                event_id=event_id,
                run_id=run_id,
                workflow=workflow,
                status="completed",
                output=output,
                graph_updates=effects,
                actions=stored_actions,
            )
            serialized = result.model_dump(mode="json")
            conn.execute(
                "UPDATE runs SET status='completed',completed_at=?,details_json=?,result_json=? WHERE id=?",
                (now, jdump({"workflow": workflow}), jdump(serialized), run_id),
            )
            conn.execute(
                "UPDATE events SET status='completed',processed_at=?,result_json=?,error=NULL WHERE id=?",
                (now, jdump(serialized), event_id),
            )
            conn.execute(
                "UPDATE jobs SET status='completed',lease_owner=NULL,leased_at=NULL,updated_at=?,last_error=NULL WHERE id=?",
                (now, job_id),
            )
            return result

    def reject_event(
        self,
        *,
        event_id: str,
        job_id: str,
        run_id: int,
        workflow: str,
        output: dict[str, Any],
        reason: str,
        pedagogy_plan: Any | None = None,
    ) -> RunResult:
        now = utcnow()
        result = RunResult(event_id=event_id, run_id=run_id, workflow=workflow, status="rejected", output=output)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            plan_id = pedagogy_plan.plan_id if pedagogy_plan is not None else None
            plan_hash = pedagogy_plan.plan_hash if pedagogy_plan is not None else None
            conn.execute(
                """INSERT INTO responses(event_id,run_id,workflow,status,output_json,created_at,updated_at,
                   pedagogy_plan_id,pedagogy_plan_hash)
                   VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(event_id) DO UPDATE SET status='rejected',
                   output_json=excluded.output_json,updated_at=excluded.updated_at,
                   pedagogy_plan_id=excluded.pedagogy_plan_id,pedagogy_plan_hash=excluded.pedagogy_plan_hash""",
                (event_id, run_id, workflow, "rejected", jdump(output), now, now, plan_id, plan_hash),
            )
            if pedagogy_plan is not None:
                conn.execute(
                    "UPDATE pedagogy_plans SET status='rejected',updated_at=? WHERE plan_id=? AND plan_hash=?",
                    (now, plan_id, plan_hash),
                )
            serialized = result.model_dump(mode="json")
            conn.execute(
                "UPDATE runs SET status='rejected',completed_at=?,details_json=?,result_json=? WHERE id=?",
                (now, jdump({"reason": reason}), jdump(serialized), run_id),
            )
            conn.execute(
                "UPDATE events SET status='rejected',processed_at=?,result_json=?,error=? WHERE id=?",
                (now, jdump(serialized), reason, event_id),
            )
            conn.execute(
                "UPDATE jobs SET status='completed',updated_at=?,last_error=? WHERE id=?", (now, reason, job_id)
            )
        return result

    def fail_job(
        self, job_id: str, event_id: str | None, run_id: int | None, error: Exception, max_attempts: int = 3
    ) -> None:
        now = datetime.now(UTC)
        with connect(self.db_path) as conn:
            row = conn.execute("SELECT attempts FROM jobs WHERE id=?", (job_id,)).fetchone()
            attempts = int(row["attempts"]) if row else max_attempts
            status = "failed" if attempts >= max_attempts else "queued"
            delay = min(60, 2 ** max(0, attempts - 1))
            available = (now + timedelta(seconds=delay)).isoformat()
            conn.execute(
                """UPDATE jobs SET status=?,available_at=?,lease_owner=NULL,leased_at=NULL,updated_at=?,last_error=? WHERE id=?""",
                (status, available, now.isoformat(), repr(error), job_id),
            )
            if run_id is not None:
                conn.execute(
                    "UPDATE runs SET status='failed',completed_at=?,details_json=? WHERE id=?",
                    (now.isoformat(), jdump({"error": repr(error)}), run_id),
                )
            if event_id and status == "failed":
                conn.execute("UPDATE events SET status='failed',error=? WHERE id=?", (repr(error), event_id))

    def get_response(self, event_id: str) -> RunResult | None:
        with connect(self.db_path) as conn:
            return self._response_from_conn(conn, event_id)

    def _response_from_conn(self, conn, event_id: str) -> RunResult | None:
        row = conn.execute("SELECT result_json FROM events WHERE id=?", (event_id,)).fetchone()
        if not row or not row["result_json"]:
            return None
        return RunResult.model_validate(jload(row["result_json"]))

    @staticmethod
    def _apply_mutations(
        conn,
        event_id: str,
        child_id: str | None,
        run_id: int,
        mutations: list[GraphMutation],
        *,
        skip_invalid: bool = False,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for mutation in mutations:
            bound_state = {**mutation.state, "source_event_id": event_id}
            if mutation.kind.value == "add_observation":
                bound_state["event_id"] = event_id
            mutation = mutation.model_copy(update={"state": bound_state})
            payload = jdump(mutation.model_dump(mode="json", exclude_none=True))
            existing = conn.execute(
                "SELECT status FROM graph_effects WHERE event_id=? AND mutation_json=?",
                (event_id, payload),
            ).fetchone()
            if existing and existing["status"] == "applied":
                results.append({"mutation": jload(payload), "status": "already_applied"})
                continue
            conn.execute(
                """INSERT INTO graph_effects(event_id,run_id,mutation_json,status)
                   VALUES(?,?,?,'pending') ON CONFLICT(event_id,mutation_json) DO UPDATE SET run_id=excluded.run_id""",
                (event_id, run_id, payload),
            )
            try:
                effect = apply_graph_mutation(conn, mutation, child_id)
            except ValueError as exc:
                conn.execute(
                    "UPDATE graph_effects SET status='failed',error=? WHERE event_id=? AND mutation_json=?",
                    (repr(exc), event_id, payload),
                )
                if not skip_invalid:
                    raise
                logger.warning(
                    "skipped invalid optional graph mutation event_id=%s run_id=%s kind=%s error=%s",
                    event_id,
                    run_id,
                    mutation.kind.value,
                    str(exc),
                )
                results.append(
                    {
                        "mutation": jload(payload),
                        "status": "skipped_invalid",
                        "error": str(exc),
                    }
                )
                continue
            conn.execute(
                "UPDATE graph_effects SET status='applied',applied_at=?,error=NULL WHERE event_id=? AND mutation_json=?",
                (utcnow(), event_id, payload),
            )
            results.append({"mutation": jload(payload), "status": "applied", "effect": effect})
        return results

    @staticmethod
    def _store_actions(
        conn,
        event_id: str,
        run_id: int,
        actions: list[ActionRequest],
        pedagogy_plan: Any | None = None,
    ) -> list[dict[str, Any]]:
        stored: list[dict[str, Any]] = []
        now = utcnow()
        for index, action in enumerate(actions):
            payload = action.model_dump(mode="json")
            idempotency = sha256(f"{event_id}:{index}:{jdump(payload)}".encode()).hexdigest()
            action_id = f"act_{uuid4().hex[:16]}"
            conn.execute(
                """INSERT INTO actions(id,event_id,run_id,action_type,payload_json,rationale,status,idempotency_key,
                   created_at,updated_at,pedagogy_plan_id,pedagogy_plan_hash)
                   VALUES(?,?,?,?,?,?,'proposed',?,?,?,?,?) ON CONFLICT(idempotency_key) DO NOTHING""",
                (
                    action_id,
                    event_id,
                    run_id,
                    action.type.value,
                    jdump(action.payload),
                    action.rationale,
                    idempotency,
                    now,
                    now,
                    pedagogy_plan.plan_id if pedagogy_plan is not None else None,
                    pedagogy_plan.plan_hash if pedagogy_plan is not None else None,
                ),
            )
            row = conn.execute("SELECT id,status FROM actions WHERE idempotency_key=?", (idempotency,)).fetchone()
            stored.append(
                {"id": row["id"], "type": action.type.value, "status": row["status"], "payload": action.payload}
            )
        return stored
