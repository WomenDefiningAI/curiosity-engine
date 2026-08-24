from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import ActionRequest, Event, GraphMutation, RunResult
from .db import connect, init_db, jdump, jload, utcnow
from .graph import apply_graph_mutation


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

    def submit_event(self, event: Event) -> Submission:
        job_id = f"job_{event.id}"
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload_hash FROM events WHERE id=?", (event.id,)).fetchone()
            if row:
                if row["payload_hash"] and row["payload_hash"] != event.payload_hash:
                    raise IdempotencyConflict(f"event id {event.id!r} was already used for a different payload")
                response = self._response_from_conn(conn, event.id)
                return Submission(event.id, job_id, True, response)
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
            conn.execute(
                """INSERT INTO evidence(child_id,event_id,evidence_type,content,source,confidence,metadata_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (event.child_id, event.id, "raw_event", event.text, event.source, 1.0, jdump(event.metadata), now),
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
        return Submission(event.id, job_id, False)

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
    ) -> RunResult:
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            effects = self._apply_mutations(conn, event_id, child_id, run_id, graph_updates)
            stored_actions = self._store_actions(conn, event_id, run_id, actions)
            conn.execute(
                """INSERT INTO responses(event_id,run_id,workflow,status,output_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(event_id) DO UPDATE SET run_id=excluded.run_id,workflow=excluded.workflow,
                   status=excluded.status,output_json=excluded.output_json,updated_at=excluded.updated_at""",
                (event_id, run_id, workflow, "completed", jdump(output), now, now),
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
    ) -> RunResult:
        now = utcnow()
        result = RunResult(event_id=event_id, run_id=run_id, workflow=workflow, status="rejected", output=output)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO responses(event_id,run_id,workflow,status,output_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?) ON CONFLICT(event_id) DO UPDATE SET status='rejected',
                   output_json=excluded.output_json,updated_at=excluded.updated_at""",
                (event_id, run_id, workflow, "rejected", jdump(output), now, now),
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
        conn, event_id: str, child_id: str | None, run_id: int, mutations: list[GraphMutation]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for mutation in mutations:
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
            except Exception as exc:
                conn.execute(
                    "UPDATE graph_effects SET status='failed',error=? WHERE event_id=? AND mutation_json=?",
                    (repr(exc), event_id, payload),
                )
                raise
            conn.execute(
                "UPDATE graph_effects SET status='applied',applied_at=?,error=NULL WHERE event_id=? AND mutation_json=?",
                (utcnow(), event_id, payload),
            )
            results.append({"mutation": jload(payload), "status": "applied", "effect": effect})
        return results

    @staticmethod
    def _store_actions(conn, event_id: str, run_id: int, actions: list[ActionRequest]) -> list[dict[str, Any]]:
        stored: list[dict[str, Any]] = []
        now = utcnow()
        for index, action in enumerate(actions):
            payload = action.model_dump(mode="json")
            idempotency = sha256(f"{event_id}:{index}:{jdump(payload)}".encode()).hexdigest()
            action_id = f"act_{uuid4().hex[:16]}"
            conn.execute(
                """INSERT INTO actions(id,event_id,run_id,action_type,payload_json,rationale,status,idempotency_key,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,'proposed',?,?,?) ON CONFLICT(idempotency_key) DO NOTHING""",
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
                ),
            )
            row = conn.execute("SELECT id,status FROM actions WHERE idempotency_key=?", (idempotency,)).fetchone()
            stored.append(
                {"id": row["id"], "type": action.type.value, "status": row["status"], "payload": action.payload}
            )
        return stored
