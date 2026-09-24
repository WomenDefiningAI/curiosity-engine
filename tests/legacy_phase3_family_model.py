"""Historical Phase 3 adversarial/fault matrix; excluded from the default beta gate."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from curiosity_engine.backups import BackupError, create_snapshot, restore_snapshot
from curiosity_engine.cli import build_parser
from curiosity_engine.collection_policy import CollectionPausedError
from curiosity_engine.contracts import Event, InteractionEvent, InteractionPlan
from curiosity_engine.db import SCHEMA_VERSION, connect, init_db, jdump, utcnow
from curiosity_engine.deletion_ledger import (
    database_deletion_conflicts,
    migration_backup_candidates,
    migration_backup_sidecars,
    purge_managed_backups,
    record_event_deletion,
)
from curiosity_engine.episodes import apply_episode_correction
from curiosity_engine.family_model import (
    child_reflection_card,
    correct_family_evidence,
    delete_family_event,
    delete_family_learning_source,
    delete_family_model,
    export_family_model,
    inspect_family_model,
    record_child_reflection,
    set_collection_enabled,
    set_learning_source_retention,
    write_family_model_export,
)
from curiosity_engine.graph import add_child, add_observation, add_school_signal, capture_question, upsert_node
from curiosity_engine.interaction import TransportConflict, begin_receipt, finish_receipt
from curiosity_engine.interactions import create_interaction, resolve_interaction
from curiosity_engine.learner_model import (
    EvidenceConflict,
    expire_learning_records,
    learner_snapshot,
    record_learning_opportunity,
    register_concept,
    register_learning_plan,
    register_learning_task,
    register_rubric,
)
from curiosity_engine.learning_artifacts import LearningArtifactService
from curiosity_engine.reasoning import ReasoningEngine, StubBackend
from curiosity_engine.repository import IdempotencyConflict, Repository
from curiosity_engine.runtime import CuriosityHarness
from curiosity_engine.service import CuriosityService
from curiosity_engine.sessions import SessionStore
from curiosity_engine.transports.contracts import InboundMessage
from curiosity_engine.web import create_app


def _setup(db: Path) -> None:
    init_db(db)
    add_child(db, "child-a", "Synthetic Child A", 2020, "1")
    add_child(db, "child-b", "Synthetic Child B", 2020, "1")
    register_concept(
        db,
        {
            "concept_id": "science.shadows",
            "label": "Shadows",
            "domain": "science",
            "grade_band": "K-2",
            "description": "Observable relationships among light, objects, and shadows.",
        },
    )
    register_rubric(
        db,
        {
            "rubric_id": "rubric.shadows",
            "label": "Shadow observation",
            "criteria": [
                {
                    "criterion_id": "criterion.shadows",
                    "concept_id": "science.shadows",
                    "description": "Describes one observed relationship without a global ability judgment.",
                }
            ],
        },
    )
    register_learning_task(
        db,
        {
            "task_id": "task.shadows",
            "title": "Notice a shadow",
            "learning_goal": "Describe how a shadow changes when an object moves.",
            "strategy": "predict-test-explain",
            "difficulty": "familiar",
            "difficulty_provenance": "reviewed_task",
            "modalities": ["spoken", "gesture_or_manipulative"],
            "context_key": "flashlight-and-object",
            "representation_key": "physical-observation",
            "concepts": [{"concept_id": "science.shadows", "role": "target"}],
            "rubric_id": "rubric.shadows",
            "rubric_version": 1,
        },
    )


def _record(
    db: Path,
    *,
    child_id: str = "child-a",
    event_id: str = "evt-shadow-a",
    opportunity_id: str = "opp-shadow-a",
    text: str = "Synthetic child noticed that a shadow moved.",
    retention_class: str = "learning_record",
) -> dict[str, object]:
    Repository(db).submit_event(
        Event(id=event_id, type="child_question", child_id=child_id, text=text, source="parent")
    )
    plan_id = f"plan-{opportunity_id}"
    register_learning_plan(
        db,
        {
            "plan_id": plan_id,
            "child_id": child_id,
            "source_event_id": event_id,
            "task_id": "task.shadows",
        },
    )
    return record_learning_opportunity(
        db,
        {
            "opportunity_id": opportunity_id,
            "child_id": child_id,
            "event_id": event_id,
            "plan_id": plan_id,
            "actor": "child",
            "initiation": "child",
            "response_mode": "spoken",
            "assistance_level": "none",
            "criterion_results": [{"criterion_id": "criterion.shadows", "outcome": "correct"}],
            "scorer": "reviewed-parent-observation",
            "scorer_version": "1",
            "source_type": "direct_observation",
            "retention_class": retention_class,
        },
    )


def _knowledge_evidence_id(db: Path, child_id: str = "child-a") -> str:
    with connect(db) as conn:
        return str(
            conn.execute(
                """SELECT evidence_id FROM evidence_records
                   WHERE child_id=? AND construct='knowledge_performance'""",
                (child_id,),
            ).fetchone()[0]
        )


def _add_binding(db: Path, *, parent_id: str = "parent-a", binding_id: str = "binding-a") -> None:
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO parent_principals VALUES(?,?,?,?,?,?)",
            (parent_id, "Synthetic Parent", "owner", "active", now, now),
        )
        conn.execute(
            "INSERT INTO transport_bindings VALUES(?,?,?,?,?,?,?,?,?)",
            (binding_id, "slack", f"team-{binding_id}", f"user-{binding_id}", "channel", parent_id, "active", now, now),
        )


def _record_file_event(db: Path, output_root: Path, *, event_id: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    _record(db, event_id=event_id, opportunity_id=f"opp-{event_id}")
    path = output_root / f"{event_id}.txt"
    path.write_text(f"private bytes for {event_id}", encoding="utf-8")
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO experiences(
               id,child_id,experience_type,title,spec_json,status,created_at,source_event_id
               ) VALUES(?,?, 'activity','Synthetic file','{}','generated',?,?)""",
            (f"experience-{event_id}", "child-a", now, event_id),
        )
        conn.execute(
            """INSERT INTO artifacts(
               id,experience_id,child_id,artifact_type,path,spec_json,created_at
               ) VALUES(?,?,?,'reference',?,'{}',?)""",
            (f"artifact-{event_id}", f"experience-{event_id}", "child-a", str(path), now),
        )
    return path


def _backup_environment(tmp_path: Path, name: str) -> tuple[Path, Path, Path, Path, Path]:
    repo = tmp_path / name / "repo"
    private = repo / "private"
    output = private / "output"
    data = private / "data"
    backup_root = tmp_path / name / "managed-backups"
    output.mkdir(parents=True)
    data.mkdir(parents=True)
    return repo, private, output, data / "curiosity.db", backup_root


def _write_legacy_v1_snapshot(
    *,
    db: Path,
    private: Path,
    output: Path,
    backup_root: Path,
    snapshot_id: str = "curiosity-backup-20250101T000000Z-legacy",
) -> Path:
    """Build the exact pre-identity v1 shape for compatibility tests."""

    backup_root.mkdir(mode=0o700)
    (backup_root / ".curiosity-backup-root.json").write_text(
        json.dumps({"format": "curiosity-family-backup", "format_version": 1}),
        encoding="utf-8",
    )
    snapshot = backup_root / snapshot_id
    database = snapshot / "data" / "curiosity.db"
    database.parent.mkdir(parents=True, mode=0o700)
    with sqlite3.connect(db) as source, sqlite3.connect(database) as target:
        source.backup(target)
    with sqlite3.connect(database) as legacy:
        legacy.execute("DROP TABLE family_database_identity")
        legacy.execute("PRAGMA journal_mode=DELETE")
        schema_version = int(legacy.execute("PRAGMA user_version").fetchone()[0])
    database.chmod(0o600)
    manifest = {
        "format": "curiosity-family-backup",
        "format_version": 1,
        "app_version": "0.1.0",
        "created_at": "2025-01-01T00:00:00+00:00",
        "snapshot_id": snapshot_id,
        "database_schema_version": schema_version,
        "source_roots": {"private": str(private), "output": str(output)},
        "files": [
            {
                "path": "data/curiosity.db",
                "bytes": database.stat().st_size,
                "sha256": sha256(database.read_bytes()).hexdigest(),
            }
        ],
        "totals": {"bytes": database.stat().st_size, "files": 1},
        "credentials_included": False,
    }
    (snapshot / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in (backup_root / ".curiosity-backup-root.json", snapshot / "manifest.json"):
        path.chmod(0o600)
    snapshot.chmod(0o700)
    return snapshot


def test_v21_through_v23_migrations_are_additive_repairable_and_ledgered(tmp_path: Path) -> None:
    db = tmp_path / "legacy-v20.sqlite"
    _setup(db)
    raw = sqlite3.connect(db)
    try:
        for table in (
            "child_reflection_responses",
            "family_model_audit_log",
            "learner_retention_events",
            "learner_model_controls",
            "deleted_event_tombstones",
        ):
            raw.execute(f"DROP TABLE {table}")
        raw.execute("DELETE FROM schema_migrations WHERE version=21")
        raw.execute("PRAGMA user_version=20")
        raw.commit()
    finally:
        raw.close()

    backup = init_db(db)
    assert backup is not None and backup.is_file()
    with connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 26
        assert conn.execute("SELECT name FROM children WHERE id='child-a'").fetchone()[0] == "Synthetic Child A"
        assert conn.execute("SELECT name FROM schema_migrations WHERE version=21").fetchone()[0] == (
            "open family learner model controls"
        )
        assert conn.execute("SELECT name FROM schema_migrations WHERE version=22").fetchone()[0] == (
            "global event replay tombstones"
        )
        assert conn.execute("SELECT name FROM schema_migrations WHERE version=23").fetchone()[0] == (
            "complete deletion lineage and recovery"
        )
        for table, column in (
            ("agent_turns", "event_id"),
            ("tool_calls", "event_id"),
            ("capability_runs", "produced_event_id"),
            ("release_units", "event_id"),
        ):
            assert column in {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "learner_model_controls",
        "learner_retention_events",
        "family_model_audit_log",
        "child_reflection_responses",
        "deleted_event_tombstones",
        "deleted_transport_event_tombstones",
        "family_file_deletion_journal",
        "family_event_deletion_locks",
        "agent_message_event_lineage",
        "agent_turn_event_lineage",
        "tool_call_event_lineage",
        "capability_run_event_lineage",
        "release_unit_event_lineage",
        "interaction_instance_event_lineage",
    } <= tables
    assert init_db(db) is None


def test_v22_repairs_child_scoped_event_tombstones_without_losing_replay_history(tmp_path: Path) -> None:
    db = tmp_path / "legacy-v21.sqlite"
    _setup(db)
    raw = sqlite3.connect(db)
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        raw.execute("DROP INDEX idx_deleted_event_child_hash")
        raw.execute("DROP TABLE deleted_event_tombstones")
        raw.execute(
            """CREATE TABLE deleted_event_tombstones (
               tombstone_id TEXT PRIMARY KEY,
               child_id TEXT NOT NULL,
               event_id_hash TEXT NOT NULL,
               reason TEXT NOT NULL,
               actor_type TEXT NOT NULL,
               deleted_at TEXT NOT NULL,
               UNIQUE(child_id,event_id_hash),
               FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
             )"""
        )
        raw.execute(
            "INSERT INTO deleted_event_tombstones VALUES(?,?,?,?,?,?)",
            (
                "legacy-tomb",
                "child-a",
                sha256(b"evt-old").hexdigest(),
                "parent_deleted",
                "parent",
                datetime.now(UTC).isoformat(),
            ),
        )
        raw.execute("DELETE FROM schema_migrations WHERE version=22")
        raw.execute("PRAGMA user_version=21")
        raw.commit()
    finally:
        raw.close()

    backup = init_db(db)
    assert backup is not None and backup.is_file()
    with connect(db) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(deleted_event_tombstones)")}
        assert "child_id" not in columns
        assert "original_child_id_hash" in columns
        row = conn.execute("SELECT * FROM deleted_event_tombstones").fetchone()
        assert row["event_id_hash"] == sha256(b"evt-old").hexdigest()
        assert row["original_child_id_hash"] == sha256(b"child-a").hexdigest()
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_v23_adds_and_backfills_produced_event_lineage_without_guessing(tmp_path: Path) -> None:
    db = tmp_path / "legacy-v22.sqlite"
    _setup(db)
    _record(db)
    _record(db, child_id="child-b", event_id="evt-v22-child-b", opportunity_id="opp-v22-child-b")
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO agent_sessions(
               id,origin,conversation_ref,thread_ref,child_id,created_at,updated_at
               ) VALUES('session-v22','cli','conversation-v22','thread-v22','child-a',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO agent_turns(
               id,session_id,ordinal,status,request_hash,response_json,policy_hash,started_at,completed_at
               ) VALUES('turn-v22','session-v22',1,'completed','request',?,'policy',?,?)""",
            (jdump({"event_id": "evt-shadow-a", "message": "Synthetic result"}), now, now),
        )
        conn.execute(
            """INSERT INTO tool_calls(
               id,session_id,turn_id,ordinal,tool_name,tool_version,argument_hash,arguments_json,risk,
               decision,status,result_json,created_at,updated_at
               ) VALUES('tool-v22','session-v22','turn-v22',1,'continue_learning_thread','1','argument',
                 '{}','low','allowed','succeeded',?,?,?)""",
            (jdump({"event_id": "evt-shadow-a"}), now, now),
        )
        conn.execute(
            """INSERT INTO capability_runs(
               id,session_id,capability_id,capability_version,status,result_json,created_at,updated_at
               ) VALUES('cap-v22','session-v22','parent_chat','1','completed',?,?,?)""",
            (jdump({"event_id": "evt-shadow-a"}), now, now),
        )
        conn.execute(
            """INSERT INTO release_units(
               id,capability_run_id,unit_type,ordinal,status,content_hash,payload_json,created_at,updated_at
               ) VALUES('release-v22','cap-v22','answer',1,'reviewed','content',?,?,?)""",
            (jdump({"event_id": "evt-shadow-a"}), now, now),
        )
        conn.execute(
            """INSERT INTO agent_turns(
               id,session_id,ordinal,status,request_hash,response_json,policy_hash,started_at,completed_at
               ) VALUES('turn-v22-cross','session-v22',2,'completed','request-cross',?,'policy',?,?)""",
            (jdump({"event_id": "evt-v22-child-b"}), now, now),
        )
        conn.execute(
            """INSERT INTO tool_calls(
               id,session_id,turn_id,ordinal,tool_name,tool_version,argument_hash,arguments_json,risk,
               decision,status,result_json,created_at,updated_at
               ) VALUES('tool-v22-cross','session-v22','turn-v22-cross',1,'continue_learning_thread','1',
                 'argument-cross','{}','low','allowed','succeeded',?,?,?)""",
            (jdump({"event_id": "evt-v22-child-b"}), now, now),
        )
        conn.execute(
            """INSERT INTO capability_runs(
               id,session_id,capability_id,capability_version,status,result_json,created_at,updated_at
               ) VALUES('cap-v22-cross','session-v22','parent_chat','1','completed',?,?,?)""",
            (jdump({"event_id": "evt-v22-child-b"}), now, now),
        )
        conn.execute(
            """INSERT INTO release_units(
               id,capability_run_id,unit_type,ordinal,status,content_hash,payload_json,created_at,updated_at
               ) VALUES('release-v22-cross','cap-v22-cross','answer',1,'reviewed','content-cross',?,?,?)""",
            (jdump({"event_id": "evt-v22-child-b"}), now, now),
        )

    raw = sqlite3.connect(db)
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        phase3_triggers = [
            str(row[0])
            for row in raw.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
            if str(row[0]).startswith(("guard_", "block_paused_profile_", "block_deleting_event_"))
        ]
        for trigger in phase3_triggers:
            raw.execute(f'DROP TRIGGER "{trigger}"')
        for index in (
            "idx_agent_turns_event",
            "idx_tool_calls_event",
            "idx_capability_runs_produced_event",
            "idx_release_units_event",
        ):
            raw.execute(f'DROP INDEX "{index}"')
        for table, column in (
            ("agent_turns", "event_id"),
            ("tool_calls", "event_id"),
            ("capability_runs", "produced_event_id"),
            ("release_units", "event_id"),
        ):
            raw.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}"')
        for table in (
            "agent_message_event_lineage",
            "agent_turn_event_lineage",
            "tool_call_event_lineage",
            "capability_run_event_lineage",
            "release_unit_event_lineage",
            "interaction_instance_event_lineage",
        ):
            raw.execute(f'DROP TABLE "{table}"')
        raw.execute("DROP TABLE deleted_transport_event_tombstones")
        raw.execute("DROP TABLE family_file_deletion_journal")
        raw.execute("DROP TABLE family_event_deletion_locks")
        raw.execute("DELETE FROM schema_migrations WHERE version=23")
        raw.execute("PRAGMA user_version=22")
        raw.commit()
    finally:
        raw.close()

    backup = init_db(db)
    assert backup is not None and backup.is_file()
    with connect(db) as conn:
        assert conn.execute("SELECT event_id FROM agent_turns WHERE id='turn-v22'").fetchone()[0] == "evt-shadow-a"
        assert conn.execute("SELECT event_id FROM tool_calls WHERE id='tool-v22'").fetchone()[0] == "evt-shadow-a"
        assert (
            conn.execute("SELECT produced_event_id FROM capability_runs WHERE id='cap-v22'").fetchone()[0]
            == "evt-shadow-a"
        )
        assert conn.execute("SELECT event_id FROM release_units WHERE id='release-v22'").fetchone()[0] == "evt-shadow-a"
        assert conn.execute("SELECT COUNT(*) FROM agent_turn_event_lineage").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM tool_call_event_lineage").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM capability_run_event_lineage").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM release_unit_event_lineage").fetchone()[0] == 1
        assert conn.execute("SELECT event_id FROM agent_turns WHERE id='turn-v22-cross'").fetchone()[0] is None
        assert conn.execute("SELECT event_id FROM tool_calls WHERE id='tool-v22-cross'").fetchone()[0] is None
        assert (
            conn.execute("SELECT produced_event_id FROM capability_runs WHERE id='cap-v22-cross'").fetchone()[0]
            is None
        )
        assert conn.execute("SELECT event_id FROM release_units WHERE id='release-v22-cross'").fetchone()[0] is None
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_agent_lineage_rejects_cross_child_links_at_api_and_database_boundaries(tmp_path: Path) -> None:
    db = tmp_path / "cross-child-lineage.sqlite"
    _setup(db)
    _record(db, child_id="child-b", event_id="evt-child-b", opportunity_id="opp-child-b")
    store = SessionStore(db)
    session = store.get_or_create(
        origin="cli",
        conversation_ref="cross-child",
        thread_ref="cross-child",
        child_id="child-a",
    )
    with pytest.raises(ValueError, match="does not belong"):
        store.append_message(
            str(session["id"]),
            role="user",
            content="Synthetic cross-child source",
            event_id="evt-child-b",
        )
    message_id = store.append_message(str(session["id"]), role="user", content="Synthetic unbound source")
    with pytest.raises(ValueError, match="does not belong"):
        store.attach_event(message_id, "evt-child-b")
    turn = store.start_turn(str(session["id"]), {"message": "synthetic"}, policy_hash="policy")
    with pytest.raises(ValueError, match="does not belong"):
        store.record_tool_call(
            session_id=str(session["id"]),
            turn_id=str(turn["id"]),
            ordinal=1,
            tool_name="continue_learning_thread",
            tool_version="1",
            arguments={},
            risk="low",
            decision="allowed",
            status="succeeded",
            event_id="evt-child-b",
        )
    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError, match="cross-child"):
        conn.execute(
            "UPDATE agent_turns SET event_id='evt-child-b' WHERE id=?",
            (turn["id"],),
        )


def test_parent_inspector_explains_inferences_evidence_and_provenance_without_cross_child_leakage(
    tmp_path: Path,
) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _record(db, text="Synthetic A private observation.")
    _record(
        db,
        child_id="child-b",
        event_id="evt-shadow-b",
        opportunity_id="opp-shadow-b",
        text="Synthetic B private observation.",
    )

    view = inspect_family_model(db, "child-a")
    encoded = json.dumps(view)
    assert view["view_version"] == "phase3-v1"
    assert view["controls"]["collection_enabled"] is True
    assert view["controls"]["recommendation_use_enabled"] is False
    assert view["labels_are_not_grades_or_diagnoses"] is True
    assert view["inferences"]["knowledge"][0]["explanation"]
    assert "eligible observation" in view["inferences"]["knowledge"][0]["why"]
    assert view["evidence"][0]["why_counted"]
    assert view["evidence"][0]["rule_id"]
    assert "Synthetic A private observation" in encoded
    assert "Synthetic B private observation" not in encoded
    assert "opp-shadow-b" not in encoded


def test_parent_correction_recomputes_and_is_audited_and_child_scoped(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _record(db)
    _record(db, child_id="child-b", event_id="evt-b", opportunity_id="opp-b", text="Synthetic B.")
    evidence_id = _knowledge_evidence_id(db)
    before = learner_snapshot(db, "child-a")["snapshot_digest"]

    result = correct_family_evidence(
        db,
        child_id="child-a",
        evidence_id=evidence_id,
        action="exclude",
        reason="Parent says this was not the child's response.",
    )
    after = learner_snapshot(db, "child-a")
    assert result["audit_id"].startswith("fma_")
    assert after["snapshot_digest"] != before
    item = next(row for row in inspect_family_model(db, "child-a")["evidence"] if row["evidence_id"] == evidence_id)
    assert item["active"] is False
    assert item["counted"] is False
    assert item["corrections"][-1]["action"] == "exclude"
    assert inspect_family_model(db, "child-a")["audit"][0]["action"] == "evidence_corrected"

    with pytest.raises(ValueError, match="does not belong"):
        correct_family_evidence(
            db,
            child_id="child-b",
            evidence_id=evidence_id,
            action="restore",
            reason="Cross-child attempt.",
        )
    correct_family_evidence(
        db,
        child_id="child-a",
        evidence_id=evidence_id,
        action="restore",
        reason="Parent restored the observation.",
    )
    assert next(row for row in inspect_family_model(db, "child-a")["evidence"] if row["evidence_id"] == evidence_id)[
        "active"
    ]
    correct_family_evidence(
        db,
        child_id="child-a",
        evidence_id=evidence_id,
        action="correct",
        reason="The response was not actually observed.",
        replacement={"result": "not_observed"},
    )
    corrected = next(
        row for row in inspect_family_model(db, "child-a")["evidence"] if row["evidence_id"] == evidence_id
    )
    assert corrected["effective_result"] == "not_observed"
    assert corrected["effective_stance"] == "context_only"
    assert corrected["counted"] is False


def test_collection_pause_blocks_new_evidence_but_allows_idempotent_replay_and_resume(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    original = _record(db)
    paused = set_collection_enabled(db, child_id="child-a", enabled=False, reason="Family is taking a privacy pause.")
    assert paused["changed"] is True
    assert paused["collection_enabled"] is False
    assert _record(db)["duplicate"] is True

    Repository(db).submit_event(
        Event(id="evt-paused", type="child_question", child_id="child-a", text="New synthetic observation.")
    )
    payload = {
        "opportunity_id": "opp-paused",
        "child_id": "child-a",
        "event_id": "evt-paused",
        "plan_id": "plan-paused",
        "actor": "child",
        "initiation": "child",
        "response_mode": "spoken",
        "assistance_level": "none",
        "criterion_results": [{"criterion_id": "criterion.shadows", "outcome": "correct"}],
        "scorer": "reviewed-parent-observation",
        "scorer_version": "1",
        "source_type": "direct_observation",
    }
    with pytest.raises(ValueError, match="collection is paused"):
        record_learning_opportunity(db, payload)
    set_collection_enabled(db, child_id="child-a", enabled=True, reason="Family resumed collection.")
    Repository(db).submit_event(
        Event(id="evt-resumed", type="child_question", child_id="child-a", text="New resumed observation.")
    )
    register_learning_plan(
        db,
        {
            "plan_id": "plan-resumed",
            "child_id": "child-a",
            "source_event_id": "evt-resumed",
            "task_id": "task.shadows",
        },
    )
    payload.update({"opportunity_id": "opp-resumed", "event_id": "evt-resumed", "plan_id": "plan-resumed"})
    recorded = record_learning_opportunity(db, payload)
    assert recorded["duplicate"] is False
    assert original["opportunity_id"] != recorded["opportunity_id"]


def test_paused_child_still_gets_answer_without_new_profile_or_audit_records(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    set_collection_enabled(db, child_id="child-a", enabled=False, reason="Synthetic privacy pause.")
    harness = CuriosityHarness(db, reasoning_engine=ReasoningEngine(StubBackend()))

    result = harness.dispatch(
        Event(
            id="evt-paused-answer",
            type="child_question",
            child_id="child-a",
            text="Why does the Moon seem to move with us?",
        )
    )

    assert result.status == "completed"
    assert result.output["observation_candidates"]
    assert result.output["_reasoning"]["observation_annotation_ids"] == []
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id='evt-paused-answer'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM responses WHERE event_id='evt-paused-answer'").fetchone()[0] == 1
        for table in (
            "episode_memberships",
            "episodes",
            "evidence",
            "observations",
            "nodes",
            "graph_effects",
            "evidence_annotations",
            "learning_opportunities",
            "evidence_records",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


@pytest.mark.parametrize(
    "writer",
    (
        lambda db: upsert_node(db, "child-a", "interest", "Synthetic paused interest"),
        lambda db: add_observation(db, "child-a", "curiosity", "Synthetic paused observation"),
        lambda db: add_school_signal(db, "child-a", "teacher_language", "Synthetic paused signal"),
        lambda db: capture_question(db, "child-a", "Synthetic paused question?", ["shadows"]),
    ),
)
def test_collection_pause_is_a_hard_floor_for_every_public_legacy_writer(
    tmp_path: Path,
    writer: object,
) -> None:
    db = tmp_path / "paused-writers.sqlite"
    _setup(db)
    set_collection_enabled(db, child_id="child-a", enabled=False, reason="Synthetic hard-floor test.")
    with pytest.raises(CollectionPausedError, match="collection is paused"):
        writer(db)  # type: ignore[operator]
    with connect(db) as conn:
        for table in ("nodes", "edges", "observations", "school_signals"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE child_id='child-a'").fetchone()[0] == 0


def test_delete_all_leaves_collection_hard_floor_and_preserves_explicit_support_preferences(tmp_path: Path) -> None:
    db = tmp_path / "delete-all-hard-floor.sqlite"
    _setup(db)
    _record(db)
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO preference_support_records(
               record_id,child_id,kind,value_json,source_event_id,source_type,status,sensitivity,created_at,updated_at
               ) VALUES('explicit-support','child-a','explicit_preference',?,'evt-shadow-a','parent_statement',
                 'active','ordinary',?,?)""",
            (jdump({"response_mode": "spoken_or_drawing"}), now, now),
        )
    delete_family_model(db, child_id="child-a", confirmation="child-a", reason="Synthetic delete-all test.")
    with connect(db) as conn:
        preserved = conn.execute(
            "SELECT kind,value_json FROM preference_support_records WHERE record_id='explicit-support'"
        ).fetchone()
        assert preserved and preserved["kind"] == "explicit_preference"
    with pytest.raises(CollectionPausedError, match="collection is paused"):
        add_observation(db, "child-a", "curiosity", "Must remain paused after delete-all.")


def test_parent_retention_deadline_hides_then_physically_expires_source(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _record(db)
    now = datetime.now(UTC)
    deadline = now + timedelta(days=2)
    retention = set_learning_source_retention(
        db,
        child_id="child-a",
        opportunity_id="opp-shadow-a",
        retain_until=deadline,
        reason="Family chose a shorter retention window.",
    )
    assert retention["retain_until"] == deadline.isoformat()
    assert inspect_family_model(db, "child-a")["evidence"][0]["effective_retain_until"] == deadline.isoformat()
    with pytest.raises(ValueError, match="only shorten"):
        set_learning_source_retention(
            db,
            child_id="child-a",
            opportunity_id="opp-shadow-a",
            retain_until=now + timedelta(days=30),
            reason="A later date must not resurrect the source.",
        )
    assert learner_snapshot(db, "child-a", as_of=deadline + timedelta(seconds=1))["knowledge"] == []

    with pytest.raises(ValueError, match="does not belong"):
        set_learning_source_retention(
            db,
            child_id="child-b",
            opportunity_id="opp-shadow-a",
            retain_until=deadline,
            reason="Cross-child attempt.",
        )
    expired = expire_learning_records(db, as_of=deadline + timedelta(seconds=1))
    assert expired["expired_parent_deadlines"] == 1
    assert expired["expired_opportunities"] == 1
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM learning_opportunities WHERE child_id='child-a'").fetchone()[0] == 0


def test_parent_retention_cannot_extend_a_built_in_transient_window(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _record(db, retention_class="transient")
    with pytest.raises(ValueError, match="only shorten"):
        set_learning_source_retention(
            db,
            child_id="child-a",
            opportunity_id="opp-shadow-a",
            retain_until=datetime.now(UTC) + timedelta(days=45),
            reason="Synthetic attempt to extend transient retention.",
        )


def test_source_event_and_full_model_deletion_are_confirmed_scoped_and_replay_safe(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _record(db)
    _record(db, child_id="child-b", event_id="evt-b", opportunity_id="opp-b", text="Synthetic B.")
    with pytest.raises(ValueError, match="exact event ID"):
        delete_family_event(
            db,
            child_id="child-a",
            event_id="evt-shadow-a",
            confirmation="wrong",
            reason="Parent requested raw event deletion.",
        )
    deleted_event = delete_family_event(
        db,
        child_id="child-a",
        event_id="evt-shadow-a",
        confirmation="evt-shadow-a",
        reason="Parent requested raw event deletion.",
    )
    assert deleted_event["learning_sources_deleted"] == 1
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id='evt-shadow-a'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM learning_opportunities WHERE child_id='child-a'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM learning_opportunities WHERE child_id='child-b'").fetchone()[0] == 1
        event_tombstone = dict(conn.execute("SELECT * FROM deleted_event_tombstones").fetchone())
    assert event_tombstone["reason"] == "parent_deleted"
    assert "evt-shadow-a" not in json.dumps(event_tombstone)
    with pytest.raises(IdempotencyConflict, match="deleted event"):
        Repository(db).submit_event(
            Event(
                id="evt-shadow-a",
                type="child_question",
                child_id="child-a",
                text="Synthetic child noticed that a shadow moved.",
                source="parent",
            )
        )
    with pytest.raises(IdempotencyConflict, match="deleted event"):
        Repository(db).submit_event(
            Event(
                id="evt-shadow-a",
                type="child_question",
                child_id="child-b",
                text="Cross-child stale replay.",
                source="parent",
            )
        )
    with pytest.raises(IdempotencyConflict, match="deleted event"):
        Repository(db).submit_event(
            Event(id="evt-shadow-a", type="system_note", text="Unbound stale replay.", source="system")
        )

    _record(db, event_id="evt-new-a", opportunity_id="opp-new-a")
    now = utcnow()
    with connect(db) as conn:
        legacy_evidence_id = conn.execute(
            """INSERT INTO evidence(
               child_id,event_id,evidence_type,content,source,created_at
               ) VALUES('child-a','evt-new-a','legacy_note','Synthetic legacy inference','model',?)""",
            (now,),
        ).lastrowid
        conn.execute(
            """INSERT INTO observations(
               child_id,kind,text,source,occurred_at,event_id
               ) VALUES('child-a','legacy','Synthetic legacy observation','model',?,'evt-new-a')""",
            (now,),
        )
        conn.execute(
            """INSERT INTO nodes(
               child_id,kind,label,canonical_key,state_json,first_seen,last_seen
               ) VALUES('child-a','interest','Synthetic legacy node','legacy-node','{}',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO claims(
               child_id,subject,predicate,object,status,confidence,supporting_evidence_json,created_at,updated_at
               ) VALUES('child-a','child-a','likes','Synthetic legacy claim','hypothesis',0.5,?,?,?)""",
            (jdump([legacy_evidence_id]), now, now),
        )
        conn.execute(
            """INSERT INTO school_signals(
               child_id,category,value,source_ref,observed_at
               ) VALUES('child-a','synthetic','Synthetic legacy signal','evt-new-a',?)""",
            (now,),
        )
        conn.execute(
            """INSERT INTO nodes(
               child_id,kind,label,canonical_key,state_json,first_seen,last_seen
               ) VALUES('child-b','interest','Keep B node','keep-b-node','{}',?,?)""",
            (now, now),
        )
    with pytest.raises(ValueError, match="exact child ID"):
        delete_family_model(db, child_id="child-a", confirmation="child-b", reason="Parent requested model deletion.")
    deleted_model = delete_family_model(
        db, child_id="child-a", confirmation="child-a", reason="Parent requested model deletion."
    )
    assert deleted_model["collection_enabled"] is False
    assert deleted_model["raw_events_retained"] is True
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id='evt-new-a'").fetchone()[0] == 1
        for table in (
            "nodes",
            "claims",
            "observations",
            "school_signals",
            "evidence",
            "episodes",
            "learning_episode_plans",
            "learning_opportunities",
            "projection_runs",
        ):
            assert conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE child_id=?', ("child-a",)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM nodes WHERE child_id='child-b'").fetchone()[0] == 1
    assert inspect_family_model(db, "child-a")["counts"]["evidence_records"] == 0
    assert inspect_family_model(db, "child-b")["counts"]["evidence_records"] > 0


def test_event_deletion_erases_linked_durable_rows_private_files_and_reviewed_output(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    output_root = tmp_path / "private-output"
    output_root.mkdir()
    _setup(db)
    _record(db, child_id="child-b", event_id="evt-keep-b", opportunity_id="opp-keep-b", text="Keep B.")
    marker = "DELETE-MARKER-917"
    event_id = "evt-delete-complete"
    result = CuriosityHarness(db, reasoning_engine=ReasoningEngine(StubBackend())).dispatch(
        Event(
            id=event_id,
            type="child_question",
            child_id="child-a",
            text=f"Why does the Moon seem to move with us? {marker}",
        )
    )
    assert result.status == "completed" and result.run_id is not None
    register_learning_plan(
        db,
        {
            "plan_id": "plan-delete-complete",
            "child_id": "child-a",
            "source_event_id": event_id,
            "task_id": "task.shadows",
        },
    )
    record_learning_opportunity(
        db,
        {
            "opportunity_id": "opp-delete-complete",
            "child_id": "child-a",
            "event_id": event_id,
            "plan_id": "plan-delete-complete",
            "actor": "child",
            "initiation": "child",
            "response_mode": "spoken",
            "assistance_level": "none",
            "criterion_results": [{"criterion_id": "criterion.shadows", "outcome": "correct"}],
            "scorer": "reviewed-parent-observation",
            "scorer_version": "1",
            "source_type": "direct_observation",
        },
    )

    artifact_pdf = output_root / "linked-artifact.pdf"
    artifact_preview = output_root / "linked-artifact.png"
    visual_path = output_root / "linked-visual.png"
    inbound_path = output_root / "linked-inbound.png"
    for path in (artifact_pdf, artifact_preview, visual_path, inbound_path):
        path.write_bytes(f"synthetic {marker}".encode())
    now = utcnow()
    with connect(db) as conn:
        plan = conn.execute("SELECT plan_id,plan_hash FROM pedagogy_plans WHERE event_id=?", (event_id,)).fetchone()
        assert plan is not None
        plan_id, plan_hash = str(plan["plan_id"]), str(plan["plan_hash"])
        legacy_evidence_id = int(
            conn.execute("SELECT id FROM evidence WHERE event_id=? ORDER BY id LIMIT 1", (event_id,)).fetchone()[0]
        )
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """INSERT INTO claims(
               child_id,subject,predicate,object,status,confidence,supporting_evidence_json,
               contradicting_evidence_json,created_at,updated_at
               ) VALUES('child-a','child-a','noticed',?,'hypothesis',0.5,?,'[]',?,?)""",
            (marker, jdump([legacy_evidence_id]), now, now),
        )
        conn.execute(
            "INSERT INTO parent_principals VALUES(?,?,?,?,?,?)",
            ("parent-delete", "Synthetic Parent", "owner", "active", now, now),
        )
        conn.execute(
            "INSERT INTO transport_bindings VALUES(?,?,?,?,?,?,?,?,?)",
            ("binding-delete", "slack", "team", "user", "channel", "parent-delete", "active", now, now),
        )
        conn.execute(
            "INSERT INTO parent_principals VALUES(?,?,?,?,?,?)",
            ("parent-replay", "Synthetic Replay Parent", "parent", "active", now, now),
        )
        conn.execute(
            "INSERT INTO transport_bindings VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "binding-replay",
                "slack",
                "team-replay",
                "user-replay",
                "channel-replay",
                "parent-replay",
                "active",
                now,
                now,
            ),
        )
        conn.execute(
            """INSERT INTO transport_receipts(
               transport,external_event_id,payload_hash,binding_id,status,event_id,received_at,processed_at
               ) VALUES('slack','external-delete','payload','binding-delete','completed',?,?,?)""",
            (event_id, now, now),
        )
        conn.execute(
            """INSERT INTO capture_inbox(
               id,parent_id,transport,external_event_id,text,status,child_id,created_at,resolved_at
               ) VALUES('inbox-delete','parent-delete','slack','external-delete',?,'assigned','child-a',?,?)""",
            (f"captured {marker}", now, now),
        )
        conn.execute(
            """INSERT INTO delivery_outbox(
               id,transport,binding_id,payload_json,idempotency_key,status,attempts,available_at,
               created_at,updated_at
               ) VALUES('delivery-delete','slack','binding-delete',?,'slack:external-delete:answer',
                 'sent',0,?,?,?)""",
            (jdump({"text": marker}), now, now, now),
        )
        conn.execute(
            """INSERT INTO agent_sessions(
               id,origin,transport,binding_id,conversation_ref,thread_ref,child_id,created_at,updated_at
               ) VALUES('session-delete','slack','slack','binding-delete','conversation','thread','child-a',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO agent_messages(
               id,session_id,ordinal,role,kind,content,event_id,created_at
               ) VALUES('message-delete','session-delete',1,'user','parent_chat',?,?,?)""",
            (f"message {marker}", event_id, now),
        )
        conn.execute(
            """INSERT INTO agent_session_preference_events(
               id,session_id,ordinal,category,operation,value,source_message_id,created_at
               ) VALUES('preference-delete','session-delete',1,'answer_style','set',?,
                 'message-delete',?)""",
            (marker, now),
        )
        conn.execute(
            """UPDATE agent_sessions SET summary_json=? WHERE id='session-delete'""",
            (
                jdump(
                    {
                        "parent_agent": {
                            "version": 1,
                            "preferences": [
                                {
                                    "category": "answer_style",
                                    "value": marker,
                                    "source_message_id": "message-delete",
                                }
                            ],
                        }
                    }
                ),
            ),
        )
        conn.execute(
            """INSERT INTO inbound_assets(
               id,binding_id,external_event_id,external_ref_hash,inbox_id,session_id,status,media_type,
               byte_count,sha256,path,observation_json,created_at,updated_at
               ) VALUES('inbound-delete','binding-delete','external-delete','ref','inbox-delete',
                 'session-delete','ready','image/png',1,'hash',?,?,?,?)""",
            (str(inbound_path), jdump({"summary": marker}), now, now),
        )
        conn.execute(
            """INSERT INTO capability_runs(
               id,session_id,capability_id,capability_version,skill_versions_json,status,
               source_event_id,result_json,created_at,updated_at
               ) VALUES('cap-delete','session-delete','parent_chat','1','{}','completed',?,?,?,?)""",
            (event_id, jdump({"message": marker}), now, now),
        )
        conn.execute(
            """INSERT INTO release_units(
               id,capability_run_id,unit_type,ordinal,status,content_hash,payload_json,created_at,updated_at
               ) VALUES('release-delete','cap-delete','answer',1,'reviewed','hash',?,?,?)""",
            (jdump({"text": marker}), now, now),
        )
        conn.execute(
            """INSERT INTO experiences(
               id,child_id,experience_type,title,spec_json,status,created_at,source_event_id,
               pedagogy_plan_id,pedagogy_plan_hash
               ) VALUES('experience-delete','child-a','worksheet',?,?,'generated',?,?,?,?)""",
            (marker, jdump({"private": marker}), now, event_id, plan_id, plan_hash),
        )
        conn.execute(
            """INSERT INTO artifacts(
               id,experience_id,child_id,artifact_type,path,spec_json,created_at,validated_at,sha256,
               validation_json,approval_status,pedagogy_plan_id,pedagogy_plan_hash
               ) VALUES('artifact-delete','experience-delete','child-a','worksheet',?,?,?,?,'hash',
                 '{}','approved',?,?)""",
            (str(artifact_pdf), jdump({"private": marker}), now, now, plan_id, plan_hash),
        )
        conn.execute(
            "INSERT INTO approvals VALUES('approval-delete','artifact-delete','hash','approved','parent',?,?)",
            (now, marker),
        )
        conn.execute(
            """INSERT INTO print_attempts(
               id,artifact_id,artifact_sha256,approval_id,printer,status,command_json,result_json,created_at
               ) VALUES('print-delete','artifact-delete','hash','approval-delete','printer','completed',?,?,?)""",
            (jdump({"marker": marker}), jdump({"marker": marker}), now),
        )
        conn.execute(
            """INSERT INTO artifact_file_outbox(
               id,artifact_id,binding_id,depends_on_delivery_id,channel_id,path,filename,mime_type,
               byte_count,sha256,title,comment,idempotency_key,status,attempts,available_at,created_at,updated_at
               ) VALUES('artifact-outbox-delete','artifact-delete','binding-delete','delivery-delete','channel',
                 ?,'linked-artifact.pdf','application/pdf',1,'hash',?,?,'artifact-key','sent',0,?,?,?)""",
            (str(artifact_pdf), marker, marker, now, now, now),
        )
        conn.execute(
            """INSERT INTO visual_jobs(
               id,event_id,intent_json,method,strategy,status,attempts,available_at,created_at,updated_at,
               pedagogy_plan_id,pedagogy_plan_hash
               ) VALUES('visual-job-delete',?,?,'deterministic','deterministic','completed',0,?,?,?, ?,?)""",
            (event_id, jdump({"marker": marker}), now, now, now, plan_id, plan_hash),
        )
        conn.execute(
            """INSERT INTO visual_assets(
               id,job_id,event_id,method,strategy,trust_tier,renderer_version,path,filename,mime_type,
               width,height,byte_count,sha256,title,caption,alt_text,provenance_json,validation_json,created_at,
               pedagogy_plan_id,pedagogy_plan_hash
               ) VALUES('visual-asset-delete','visual-job-delete',?,'deterministic','deterministic','B','test',
                 ?,'linked-visual.png','image/png',10,10,1,'hash',?,?,?,?,?,?,?,?)""",
            (
                event_id,
                str(visual_path),
                marker,
                marker,
                marker,
                jdump({"marker": marker}),
                "{}",
                now,
                plan_id,
                plan_hash,
            ),
        )
        conn.execute(
            """INSERT INTO slack_file_outbox(
               id,visual_job_id,visual_asset_id,binding_id,depends_on_delivery_id,channel_id,
               idempotency_key,status,attempts,available_at,created_at,updated_at
               ) VALUES('visual-outbox-delete','visual-job-delete','visual-asset-delete','binding-delete',
                 'delivery-delete','channel','visual-key','sent',0,?,?,?)""",
            (now, now, now),
        )
        conn.execute(
            """INSERT INTO actions(
               id,event_id,run_id,action_type,payload_json,rationale,status,idempotency_key,
               created_at,updated_at,result_json,pedagogy_plan_id,pedagogy_plan_hash
               ) VALUES('action-delete',?,?,'propose_artifact',?,?,'completed','action-key',?,?,?,?,?)""",
            (
                event_id,
                result.run_id,
                jdump({"marker": marker}),
                marker,
                now,
                now,
                jdump({"marker": marker}),
                plan_id,
                plan_hash,
            ),
        )
        conn.execute(
            """INSERT INTO feedback(
               child_id,event_id,experience_id,artifact_id,outcome,note,source,created_at,
               pedagogy_plan_id,pedagogy_plan_hash
               ) VALUES('child-a',?,'experience-delete','artifact-delete','used',?,'parent',?,?,?)""",
            (event_id, marker, now, plan_id, plan_hash),
        )
        conn.execute(
            """INSERT INTO output_evaluations(
               event_id,visual_asset_id,artifact_id,response_rating,visual_rating,note,created_at
               ) VALUES(?,'visual-asset-delete','artifact-delete','useful','useful',?,?)""",
            (event_id, marker, now),
        )
        conn.execute(
            """INSERT INTO onboarding_reviews(
               event_id,brain_config_hash,response_hash,workflow,factuality,grade_fit,curiosity_value,
               parent_effort,note,decision,created_at
               ) VALUES(?,'brain','response','pull_thread','pass','pass','pass','pass',?,'pass',?)""",
            (event_id, marker, now),
        )

    deleted = delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Parent requested complete local erasure.",
        output_root=output_root,
    )
    assert deleted["local_files_deleted"] == 4
    assert all(not path.exists() for path in (artifact_pdf, artifact_preview, visual_path, inbound_path))
    replay = InboundMessage(
        external_event_id="external-delete",
        team_id="team-replay",
        user_id="user-replay",
        channel_id="channel-replay",
        text="Replayed after a binding change.",
    )
    with pytest.raises(TransportConflict, match="deleted and cannot be replayed"):
        begin_receipt(db, replay, "binding-replay")
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM onboarding_reviews WHERE event_id=?", (event_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id='evt-keep-b'").fetchone()[0] == 1
        transport_tombstone = dict(conn.execute("SELECT * FROM deleted_transport_event_tombstones").fetchone())
        assert "external-delete" not in json.dumps(transport_tombstone)
        assert "binding-delete" not in json.dumps(transport_tombstone)
        tables = (
            "events",
            "episodes",
            "evidence",
            "claims",
            "nodes",
            "observations",
            "jobs",
            "runs",
            "responses",
            "graph_effects",
            "actions",
            "feedback",
            "output_evaluations",
            "onboarding_reviews",
            "transport_receipts",
            "capture_inbox",
            "delivery_outbox",
            "agent_sessions",
            "agent_messages",
            "agent_session_preference_events",
            "inbound_assets",
            "capability_runs",
            "release_units",
            "experiences",
            "artifacts",
            "approvals",
            "print_attempts",
            "artifact_file_outbox",
            "visual_jobs",
            "visual_assets",
            "slack_file_outbox",
            "pedagogy_plans",
            "learning_episode_plans",
            "learning_opportunities",
            "evidence_records",
            "evidence_annotations",
        )
        for table in tables:
            text_columns = [
                str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")') if "TEXT" in str(row[2]).upper()
            ]
            for column in text_columns:
                matches = conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE instr(CAST("{column}" AS TEXT),?)>0',
                    (marker,),
                ).fetchone()[0]
                assert matches == 0, f"private marker survived in {table}.{column}"


def test_real_parent_agent_turn_binds_and_erases_every_current_event_copy(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    output_root = tmp_path / "output"
    _setup(db)
    _add_binding(db)
    service = CuriosityService(db, output_root)
    marker = "REAL-PARENT-LINEAGE-443"

    result = service.chat(
        binding_id="binding-a",
        team_id="team-binding-a",
        channel_id="channel",
        thread_id="thread-lineage",
        text=f"Why does a shadow move when the flashlight moves? {marker}",
        child_id="child-a",
    )
    assert result["status"] == "completed"
    event_id = str(result["event_id"])
    session_id = str(result["session_id"])
    Repository(db).submit_event(
        Event(
            id="evt-lineage-child-b",
            type="child_question",
            child_id="child-b",
            text="Synthetic other-child event.",
        )
    )
    presented = create_interaction(
        db,
        binding_id="binding-a",
        session_id=session_id,
        plan=InteractionPlan.model_validate(result["interaction"]),
    )
    interaction_event = InteractionEvent(
        interaction_id=str(presented["interaction_id"]),
        option_token=str(presented["options"][0]["token"]),
        team_id="team-binding-a",
        user_id="user-binding-a",
        channel_id="channel",
        external_event_id="synthetic-interaction-click",
    )
    with connect(db) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM agent_messages WHERE session_id=? AND event_id=?",
                (session_id, event_id),
            ).fetchone()[0]
            == 2
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM agent_turns WHERE session_id=? AND event_id=?",
                (session_id, event_id),
            ).fetchone()[0]
            == 1
        )
        for table, record_column in (
            ("agent_message_event_lineage", "message_id"),
            ("agent_turn_event_lineage", "turn_id"),
            ("tool_call_event_lineage", "tool_call_id"),
            ("capability_run_event_lineage", "capability_run_id"),
            ("release_unit_event_lineage", "release_unit_id"),
            ("interaction_instance_event_lineage", "interaction_id"),
        ):
            record = conn.execute(
                f'SELECT "{record_column}" FROM "{table}" WHERE event_id=? LIMIT 1',
                (event_id,),
            ).fetchone()
            assert record, table
            with pytest.raises(sqlite3.IntegrityError, match="cross-child agent event lineage"):
                conn.execute(
                    f'''UPDATE "{table}" SET event_id='evt-lineage-child-b'
                        WHERE "{record_column}"=? AND event_id=?''',
                    (record[record_column], event_id),
                )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE session_id=? AND event_id=?",
                (session_id, event_id),
            ).fetchone()[0]
            >= 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM capability_runs WHERE session_id=? AND produced_event_id=?",
                (session_id, event_id),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                """SELECT COUNT(*) FROM release_units ru JOIN capability_runs cr
                 ON cr.id=ru.capability_run_id WHERE cr.session_id=? AND ru.event_id=?""",
                (session_id, event_id),
            ).fetchone()[0]
            >= 3
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM interaction_instance_event_lineage WHERE event_id=?",
                (event_id,),
            ).fetchone()[0]
            == 1
        )
        now = utcnow()
        conn.execute(
            """INSERT INTO agent_sessions(
               id,origin,conversation_ref,thread_ref,child_id,capability_id,state,
               summary_json,created_at,updated_at
               ) VALUES('ses-lineage-child-b','cli','other-conversation','other-thread','child-b',
               'parent_chat','active','{}',?,?)""",
            (now, now),
        )
        conn.execute(
            """INSERT INTO agent_turns(
               id,session_id,ordinal,status,request_hash,policy_hash,started_at
               ) VALUES('turn-lineage-child-b','ses-lineage-child-b',1,'running','request','policy',?)""",
            (now,),
        )
        conn.execute(
            """INSERT INTO capability_runs(
               id,session_id,capability_id,capability_version,status,created_at,updated_at
               ) VALUES('cap-lineage-child-b','ses-lineage-child-b','parent_chat','1','running',?,?)""",
            (now, now),
        )
        ownership_attacks = [
            ("UPDATE agent_sessions SET child_id='child-b' WHERE id=?", (session_id,)),
            ("UPDATE events SET child_id='child-b' WHERE id=?", (event_id,)),
            (
                "UPDATE agent_messages SET session_id='ses-lineage-child-b' WHERE id=(SELECT message_id FROM agent_message_event_lineage WHERE event_id=? LIMIT 1)",
                (event_id,),
            ),
            (
                "UPDATE agent_turns SET session_id='ses-lineage-child-b' WHERE id=(SELECT turn_id FROM agent_turn_event_lineage WHERE event_id=? LIMIT 1)",
                (event_id,),
            ),
            (
                "UPDATE tool_calls SET session_id='ses-lineage-child-b' WHERE id=(SELECT tool_call_id FROM tool_call_event_lineage WHERE event_id=? LIMIT 1)",
                (event_id,),
            ),
            (
                "UPDATE tool_calls SET turn_id='turn-lineage-child-b' WHERE id=(SELECT tool_call_id FROM tool_call_event_lineage WHERE event_id=? LIMIT 1)",
                (event_id,),
            ),
            (
                "UPDATE capability_runs SET session_id='ses-lineage-child-b' WHERE id=(SELECT capability_run_id FROM capability_run_event_lineage WHERE event_id=? LIMIT 1)",
                (event_id,),
            ),
            (
                "UPDATE release_units SET capability_run_id='cap-lineage-child-b' WHERE id=(SELECT release_unit_id FROM release_unit_event_lineage WHERE event_id=? LIMIT 1)",
                (event_id,),
            ),
            (
                "UPDATE interaction_instances SET session_id='ses-lineage-child-b' WHERE id=(SELECT interaction_id FROM interaction_instance_event_lineage WHERE event_id=? LIMIT 1)",
                (event_id,),
            ),
        ]
        for statement, parameters in ownership_attacks:
            with pytest.raises(
                sqlite3.IntegrityError,
                match="(?:(?:child ownership|lineage ownership).*immutable|turn belongs)",
            ):
                conn.execute(statement, parameters)

    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Synthetic real-path erasure test.",
        output_root=output_root,
    )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 0
        for table in ("agent_messages", "agent_turns", "tool_calls", "capability_runs"):
            assert (
                conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE session_id=?',
                    (session_id,),
                ).fetchone()[0]
                == 0
            )
        assert conn.execute("SELECT COUNT(*) FROM release_units").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM interaction_instances").fetchone()[0] == 0
        summary = conn.execute("SELECT summary_json FROM agent_sessions WHERE id=?", (session_id,)).fetchone()[0]
        assert marker not in str(summary)
        for table, column in (
            ("agent_messages", "content"),
            ("agent_turns", "response_json"),
            ("tool_calls", "arguments_json"),
            ("tool_calls", "result_json"),
            ("capability_runs", "result_json"),
            ("release_units", "payload_json"),
        ):
            assert (
                conn.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE instr(CAST("{column}" AS TEXT),?)>0',
                    (marker,),
                ).fetchone()[0]
                == 0
            )
    with pytest.raises(ValueError, match="not found"):
        resolve_interaction(db, interaction_event, binding_id="binding-a", parent_id="parent-a")


def test_parent_agent_event_and_source_lineage_survive_process_kill_before_post_bind(tmp_path: Path) -> None:
    db = tmp_path / "atomic-agent-lineage.sqlite"
    output_root = tmp_path / "atomic-agent-output"
    _setup(db)
    _add_binding(db)
    marker = "ATOMIC-LINEAGE-KILL-219"
    script = """
import os
import signal
import sys
from curiosity_engine.service import CuriosityService
from curiosity_engine.sessions import SessionStore

def crash_before_post_bind(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)

SessionStore.bind_event_lineage = crash_before_post_bind
CuriosityService(sys.argv[1], sys.argv[2]).chat(
    binding_id="binding-a",
    team_id="team-binding-a",
    channel_id="channel",
    thread_id="atomic-lineage-thread",
    text=sys.argv[3],
    child_id="child-a",
)
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(db), str(output_root), f"Why do shadows move? {marker}"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        timeout=30,
    )
    assert completed.returncode == -signal.SIGKILL
    with connect(db) as conn:
        event = conn.execute("SELECT id FROM events WHERE instr(text,?)>0", (marker,)).fetchone()
        assert event
        event_id = str(event["id"])
        assert conn.execute(
            """SELECT COUNT(*) FROM agent_messages m JOIN agent_message_event_lineage l ON l.message_id=m.id
               WHERE m.role='user' AND instr(m.content,?)>0 AND l.event_id=?""",
            (marker, event_id),
        ).fetchone()[0] == 1
        for table in (
            "agent_turn_event_lineage",
            "tool_call_event_lineage",
            "capability_run_event_lineage",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE event_id=?", (event_id,)).fetchone()[0] >= 1

    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Synthetic crash-window deletion.",
        output_root=output_root,
    )
    with connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM agent_messages WHERE instr(content,?)>0",
            (marker,),
        ).fetchone()[0] == 0


def test_existing_event_prebind_does_not_depend_on_post_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "existing-event-lineage.sqlite"
    output_root = tmp_path / "existing-event-output"
    _setup(db)
    _add_binding(db)
    service = CuriosityService(db, output_root)
    conversation_ref = sha256(b"team-binding-a:channel").hexdigest()[:20]
    thread_ref = sha256(b"team-binding-a:channel:existing-event-thread").hexdigest()[:20]
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id="binding-a",
        conversation_ref=conversation_ref,
        thread_ref=thread_ref,
        child_id="child-a",
    )
    event_id = "evt-existing-context"
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(
               id,type,child_id,text,source,metadata_json,created_at,status,processed_at,result_json
               ) VALUES(?,'context_input','child-a',?,'test',? ,?,'completed',?,'{}')""",
            (
                event_id,
                "We made paper shadows today.",
                jdump({"learning_scope": "context_only", "evidence_eligible": False}),
                utcnow(),
                utcnow(),
            ),
        )
    SessionStore(db).append_message(
        str(session["id"]),
        role="user",
        content="We made paper shadows today.",
        kind="context_input",
        event_id=event_id,
    )
    marker = "EXISTING-EVENT-PREBIND-721"

    class ExistingEventBackend:
        name = "existing-event-test"
        model = "existing-event-test-v1"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, **_kwargs: object) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                return {
                    "tool_calls": [
                        {
                            "name": "continue_learning_thread",
                            "arguments": {"message": "one easy idea", "mode": "quick"},
                            "rationale": "Use the existing context without creating a new event.",
                        }
                    ],
                    "done": False,
                }
            return {"message": "Done.", "done": True}

    monkeypatch.setattr(
        "curiosity_engine.service.configured_backend",
        lambda *_args, **_kwargs: ExistingEventBackend(),
    )

    def fail_post_bind(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic post-bind failure")

    monkeypatch.setattr(SessionStore, "bind_event_lineage", fail_post_bind)
    with pytest.raises(RuntimeError, match="synthetic post-bind failure"):
        service.chat(
            binding_id="binding-a",
            team_id="team-binding-a",
            channel_id="channel",
            thread_id="existing-event-thread",
            text=f"Please give us one easy idea from that context. {marker}",
            child_id="child-a",
        )
    with connect(db) as conn:
        assert conn.execute(
            """SELECT COUNT(*) FROM agent_messages m JOIN agent_message_event_lineage l
               ON l.message_id=m.id WHERE instr(m.content,?)>0 AND l.event_id=?""",
            (marker, event_id),
        ).fetchone()[0] == 1
        assert conn.execute(
            """SELECT COUNT(*) FROM tool_calls t JOIN tool_call_event_lineage l
               ON l.tool_call_id=t.id WHERE instr(t.arguments_json,?)>0 AND l.event_id=?""",
            (marker, event_id),
        ).fetchone()[0] == 1

    monkeypatch.undo()
    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Erase the reused context and every prebound copy.",
        output_root=output_root,
    )
    with connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM agent_messages WHERE instr(content,?)>0", (marker,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE instr(arguments_json,?)>0", (marker,)
        ).fetchone()[0] == 0


def test_event_deletion_keeps_a_private_file_still_referenced_by_another_child(tmp_path: Path) -> None:
    db = tmp_path / "shared-file.sqlite"
    output_root = tmp_path / "shared-output"
    output_root.mkdir()
    _setup(db)
    _record(db, event_id="evt-shared-a", opportunity_id="opp-shared-a")
    _record(db, child_id="child-b", event_id="evt-shared-b", opportunity_id="opp-shared-b")
    shared = output_root / "shared.pdf"
    shared.write_bytes(b"shared private bytes")
    now = utcnow()
    with connect(db) as conn:
        for suffix, child_id, event_id in (
            ("a", "child-a", "evt-shared-a"),
            ("b", "child-b", "evt-shared-b"),
        ):
            conn.execute(
                """INSERT INTO experiences(
                   id,child_id,experience_type,title,spec_json,status,created_at,source_event_id
                   ) VALUES(?,?, 'activity','Shared path','{}','generated',?,?)""",
                (f"experience-shared-{suffix}", child_id, now, event_id),
            )
            conn.execute(
                """INSERT INTO artifacts(id,experience_id,child_id,artifact_type,path,spec_json,created_at)
                   VALUES(?,?,?,'reference',?,'{}',?)""",
                (f"artifact-shared-{suffix}", f"experience-shared-{suffix}", child_id, str(shared), now),
            )
    result = delete_family_event(
        db,
        child_id="child-a",
        event_id="evt-shared-a",
        confirmation="evt-shared-a",
        reason="Synthetic shared ownership test.",
        output_root=output_root,
    )
    assert result["local_files_deleted"] == 0
    assert shared.is_file()
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM artifacts WHERE id='artifact-shared-a'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM artifacts WHERE id='artifact-shared-b'").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("crash_point", "event_survives"),
    (("after_files_staged", True), ("after_db_commit", False)),
)
def test_private_file_erasure_recovers_after_process_kill(
    tmp_path: Path,
    crash_point: str,
    event_survives: bool,
) -> None:
    db = tmp_path / f"{crash_point}.sqlite"
    output_root = tmp_path / crash_point
    _setup(db)
    event_id = f"evt-{crash_point}"
    private_file = _record_file_event(db, output_root, event_id=event_id)
    script = """
import os
import signal
import sys
from curiosity_engine.family_model import delete_family_event

def crash(point):
    if point == sys.argv[4]:
        os.kill(os.getpid(), signal.SIGKILL)

delete_family_event(
    sys.argv[1],
    child_id="child-a",
    event_id=sys.argv[3],
    confirmation=sys.argv[3],
    reason="Synthetic crash recovery.",
    output_root=sys.argv[2],
    _test_crash_hook=crash,
)
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(db), str(output_root), event_id, crash_point],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        timeout=30,
    )
    assert completed.returncode == -signal.SIGKILL
    assert not private_file.exists()
    with connect(db) as conn:
        assert bool(conn.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone()) is event_survives
        assert conn.execute("SELECT COUNT(*) FROM family_file_deletion_journal").fetchone()[0] == 1

    init_db(db)
    assert private_file.exists() is event_survives
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM family_file_deletion_journal").fetchone()[0] == 0
        assert bool(conn.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone()) is event_survives


def test_snapshot_recovers_committed_file_quarantine_before_copying(tmp_path: Path) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "quarantine-backup")
    _setup(db)
    event_id = "evt-quarantine-backup"
    private_file = _record_file_event(db, output, event_id=event_id)
    private_bytes = private_file.read_bytes()
    script = """
import os
import signal
import sys
from curiosity_engine.family_model import delete_family_event

def crash(point):
    if point == "after_db_commit":
        os.kill(os.getpid(), signal.SIGKILL)

delete_family_event(
    sys.argv[1],
    child_id="child-a",
    event_id=sys.argv[3],
    confirmation=sys.argv[3],
    reason="Crash before quarantine cleanup.",
    output_root=sys.argv[2],
    _test_crash_hook=crash,
)
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(db), str(output), event_id],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        timeout=30,
    )
    assert completed.returncode == -signal.SIGKILL
    assert list((output / ".family-delete-journal").rglob("*.bin"))

    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    snapshot_path = backup_root / str(snapshot["snapshot"])
    manifest = json.loads((snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    assert not any(".family-delete-journal" in str(item["path"]) for item in manifest["files"])
    assert not list((output / ".family-delete-journal").rglob("*"))
    assert all(path.read_bytes() != private_bytes for path in snapshot_path.rglob("*") if path.is_file())

    recovered = tmp_path / "quarantine-backup" / "recovered"
    restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=recovered,
        backup_root=backup_root,
        repo_root=repo,
    )
    assert not (recovered / "output" / ".family-delete-journal").exists()
    with sqlite3.connect(recovered / "data" / "curiosity.db") as restored_db:
        assert restored_db.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 0


def test_private_file_erasure_restores_on_transaction_failure(tmp_path: Path) -> None:
    db = tmp_path / "rollback.sqlite"
    output_root = tmp_path / "rollback-output"
    _setup(db)
    event_id = "evt-rollback"
    private_file = _record_file_event(db, output_root, event_id=event_id)

    def fail_after_staging(point: str) -> None:
        if point == "after_files_staged":
            raise RuntimeError("synthetic transaction stop")

    with pytest.raises(RuntimeError, match="synthetic transaction stop"):
        delete_family_event(
            db,
            child_id="child-a",
            event_id=event_id,
            confirmation=event_id,
            reason="Synthetic rollback recovery.",
            output_root=output_root,
            _test_crash_hook=fail_after_staging,
        )
    assert private_file.is_file()
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM family_file_deletion_journal").fetchone()[0] == 0


def test_event_deletion_blocks_real_artifact_writer_and_removes_rejected_render(tmp_path: Path) -> None:
    db = tmp_path / "writer-race.sqlite"
    output_root = tmp_path / "writer-race-output"
    _setup(db)
    event_id = "evt-writer-race"
    result = CuriosityHarness(db, reasoning_engine=ReasoningEngine(StubBackend())).dispatch(
        Event(
            id=event_id,
            type="child_question",
            child_id="child-a",
            text="Why does ice float?",
        )
    )
    assert result.status == "completed"
    writer_ran = False

    def run_late_writer(point: str) -> None:
        nonlocal writer_ran
        if point != "after_files_staged":
            return
        writer_ran = True
        with pytest.raises(sqlite3.IntegrityError, match="event deletion is pending"):
            LearningArtifactService(db, output_root, backend=StubBackend()).create_from_event(
                event_id=event_id,
                artifact_type="reference",
                revision="Make one simple reviewed reference page about floating ice.",
            )

    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Exercise a real artifact-writer race.",
        output_root=output_root,
        _test_crash_hook=run_late_writer,
    )
    assert writer_ran is True
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM experiences WHERE source_event_id=?", (event_id,)).fetchone()[0] == 0
    artifact_output = output_root / "artifacts"
    assert not list(artifact_output.glob("*"))


def test_private_file_erasure_retries_after_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "unlink.sqlite"
    output_root = tmp_path / "unlink-output"
    _setup(db)
    event_id = "evt-unlink"
    private_file = _record_file_event(db, output_root, event_id=event_id)
    original_unlink = Path.unlink

    def fail_quarantine_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if ".family-delete-journal" in path.parts and path.suffix == ".bin":
            raise OSError("synthetic unlink failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_quarantine_unlink)
    with pytest.raises(OSError, match="synthetic unlink failure"):
        delete_family_event(
            db,
            child_id="child-a",
            event_id=event_id,
            confirmation=event_id,
            reason="Synthetic unlink recovery.",
            output_root=output_root,
        )
    assert not private_file.exists()
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM family_file_deletion_journal").fetchone()[0] == 1

    monkeypatch.setattr(Path, "unlink", original_unlink)
    init_db(db)
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM family_file_deletion_journal").fetchone()[0] == 0
    assert not private_file.exists()


def test_event_deletion_purges_managed_backups_and_blocks_stale_restore(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    private = repo / "private"
    output = private / "output"
    data = private / "data"
    backup_root = tmp_path / "managed-backups"
    escrow_root = tmp_path / "old-snapshot-copy"
    output.mkdir(parents=True)
    data.mkdir(parents=True)
    db = data / "curiosity.db"
    _setup(db)
    _record(db, event_id="evt-backup-delete", opportunity_id="opp-backup-delete")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    snapshot_path = backup_root / str(snapshot["snapshot"])
    shutil.copytree(snapshot_path, escrow_root)
    with sqlite3.connect(db) as raw:
        raw.execute("PRAGMA user_version=22")
    migration_copy = init_db(db)
    assert migration_copy is not None
    assert migration_copy.name.startswith(f"{db.name}.backup-v22-")

    deleted = delete_family_event(
        db,
        child_id="child-a",
        event_id="evt-backup-delete",
        confirmation="evt-backup-delete",
        reason="Synthetic managed-backup deletion.",
        output_root=output,
    )
    assert deleted["managed_snapshots_purged"] == 1
    assert deleted["migration_backups_purged"] >= 1
    assert not snapshot_path.exists()
    assert not migration_copy.exists()
    # Even if stale bytes later reappear under the managed root, its monotonic,
    # hash-only ledger refuses to resurrect the deleted event.
    shutil.copytree(escrow_root, snapshot_path)
    with pytest.raises(BackupError, match="deleted after the snapshot"):
        restore_snapshot(
            snapshot_path.name,
            target_private=tmp_path / "refused-restore",
            backup_root=backup_root,
            repo_root=repo,
        )


@pytest.mark.parametrize(
    "crash_point",
    ("before_rebase", "after_target_reservation", "after_publish"),
)
def test_restore_staging_and_publication_survive_process_kill(
    tmp_path: Path,
    crash_point: str,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, crash_point)
    _setup(db)
    event_id = f"evt-restore-kill-{crash_point}"
    _record(db, event_id=event_id, opportunity_id=f"opp-{crash_point}")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    target = tmp_path / crash_point / "managed-recovery"
    script = """
import os
import signal
import sys
from curiosity_engine.backups import restore_snapshot

def crash(point):
    if point == sys.argv[5]:
        os.kill(os.getpid(), signal.SIGKILL)

restore_snapshot(
    sys.argv[1],
    target_private=sys.argv[2],
    backup_root=sys.argv[3],
    repo_root=sys.argv[4],
    _test_hook=crash,
)
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(snapshot["snapshot"]),
            str(target),
            str(backup_root),
            str(repo),
            crash_point,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        timeout=30,
    )
    assert completed.returncode == -signal.SIGKILL
    if crash_point == "after_publish":
        assert target.is_dir()
        with sqlite3.connect(target / "data" / "curiosity.db") as recovered:
            assert recovered.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
    elif crash_point == "after_target_reservation":
        assert target.is_dir() and not any(target.iterdir())
        retried = restore_snapshot(
            str(snapshot["snapshot"]),
            target_private=target,
            backup_root=backup_root,
            repo_root=repo,
        )
        assert retried["status"] == "restored"
    else:
        assert not target.exists()
        assert list(target.parent.glob(f".{target.name}.partial-*"))

    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Purge every crash-interrupted managed restore copy.",
        output_root=output,
    )
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.partial-*"))


@pytest.mark.parametrize("operation", ("event", "source", "retention", "model"))
def test_active_disaster_restore_inherits_backup_root_and_future_deletions(
    tmp_path: Path,
    operation: str,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, f"active-{operation}")
    _setup(db)
    event_id = f"evt-active-{operation}"
    opportunity_id = f"opp-active-{operation}"
    _record(
        db,
        event_id=event_id,
        opportunity_id=opportunity_id,
        retention_class="transient" if operation == "retention" else "learning_record",
    )
    if operation == "model":
        upsert_node(db, "child-a", "interest", "Synthetic restored legacy model state")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    snapshot_path = backup_root / str(snapshot["snapshot"])
    registered_recovery = tmp_path / f"active-{operation}" / "registered-recovery"
    restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=registered_recovery,
        backup_root=backup_root,
        repo_root=repo,
    )

    shutil.rmtree(private)
    restored = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=private,
        backup_root=backup_root,
        repo_root=repo,
    )
    restored_db = Path(str(restored["database"]))
    assert restored["active"] is True
    with sqlite3.connect(restored_db) as raw:
        raw.execute("PRAGMA user_version=22")
    migration_copy = init_db(restored_db)
    assert migration_copy is not None and migration_copy.is_file()

    if operation == "event":
        delete_family_event(
            restored_db,
            child_id="child-a",
            event_id=event_id,
            confirmation=event_id,
            reason="Delete after active disaster recovery.",
            output_root=private / "output",
        )
    elif operation == "source":
        delete_family_learning_source(
            restored_db,
            child_id="child-a",
            opportunity_id=opportunity_id,
            reason="Delete a source after active disaster recovery.",
        )
    elif operation == "retention":
        expire_learning_records(restored_db, as_of=datetime.now(UTC) + timedelta(days=31))
    else:
        delete_family_model(
            restored_db,
            child_id="child-a",
            confirmation="child-a",
            reason="Delete the model after active disaster recovery.",
        )

    assert restored_db.is_file()
    assert not snapshot_path.exists()
    assert not registered_recovery.exists()
    assert not migration_copy.exists()

    replacement = create_snapshot(
        db_path=restored_db,
        private_dir=private,
        output_dir=private / "output",
        backup_root=backup_root,
        repo_root=repo,
    )
    post_delete_restore = tmp_path / f"active-{operation}" / "post-delete-recovery"
    result = restore_snapshot(
        str(replacement["snapshot"]),
        target_private=post_delete_restore,
        backup_root=backup_root,
        repo_root=repo,
    )
    assert result["status"] == "restored"


def test_rejected_cross_family_backup_registration_grants_no_purge_authority(
    tmp_path: Path,
) -> None:
    repo_a, private_a, output_a, db_a, backup_root = _backup_environment(tmp_path, "family-a")
    repo_b, private_b, output_b, db_b, _unused_root = _backup_environment(tmp_path, "family-b")
    _setup(db_a)
    _setup(db_b)
    _record(db_a, event_id="evt-family-a", opportunity_id="opp-family-a")
    upsert_node(db_b, "child-a", "interest", "Family B private state")
    snapshot = create_snapshot(
        db_path=db_a,
        private_dir=private_a,
        output_dir=output_a,
        backup_root=backup_root,
        repo_root=repo_a,
    )
    snapshot_path = backup_root / str(snapshot["snapshot"])

    with pytest.raises(RuntimeError, match="different family database"):
        create_snapshot(
            db_path=db_b,
            private_dir=private_b,
            output_dir=output_b,
            backup_root=backup_root,
            repo_root=repo_b,
        )
    local_registry = db_b.parent / "family-backup-roots.json"
    assert not local_registry.exists()
    assert snapshot_path.is_dir()

    local_registry.write_text(
        json.dumps(
            {
                "format": "curiosity-backup-root-registry",
                "version": 1,
                "roots": [str(backup_root.resolve())],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not reciprocally authorized"):
        delete_family_model(
            db_b,
            child_id="child-a",
            confirmation="child-a",
            reason="A forged one-sided registry must fail closed.",
        )
    assert snapshot_path.is_dir()
    with connect(db_b) as conn:
        assert conn.execute("SELECT COUNT(*) FROM nodes WHERE child_id='child-a'").fetchone()[0] > 0


def test_restore_rejects_a_foreign_database_substituted_at_registered_source(
    tmp_path: Path,
) -> None:
    repo_a, private_a, output_a, db_a, backup_root = _backup_environment(tmp_path, "source-family-a")
    _repo_b, _private_b, output_b, db_b, _unused_root = _backup_environment(
        tmp_path,
        "source-family-b",
    )
    _setup(db_a)
    _setup(db_b)
    event_id = "evt-registered-source-family-collision"
    _record(db_a, event_id=event_id, opportunity_id="opp-source-family-a")
    snapshot = create_snapshot(
        db_path=db_a,
        private_dir=private_a,
        output_dir=output_a,
        backup_root=backup_root,
        repo_root=repo_a,
    )
    _record(db_b, event_id=event_id, opportunity_id="opp-source-family-b")
    delete_family_event(
        db_b,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Create foreign deletion authority with a colliding event ID.",
        output_root=output_b,
    )
    root_ledger = backup_root / "family-deletions.json"
    ledger_before = root_ledger.read_bytes()
    for suffix in ("-wal", "-shm"):
        db_a.with_name(db_a.name + suffix).unlink(missing_ok=True)
    db_a.unlink()
    shutil.copyfile(db_b, db_a)

    target = repo_a.parent / "foreign-source-restore"
    with pytest.raises(BackupError, match="deletion ledger is unreadable"):
        restore_snapshot(
            str(snapshot["snapshot"]),
            target_private=target,
            backup_root=backup_root,
            repo_root=repo_a,
        )
    assert not target.exists()
    assert root_ledger.read_bytes() == ledger_before


@pytest.mark.parametrize("damage", ("missing", "corrupt", "mismatch"))
def test_registered_backup_root_marker_damage_blocks_deletion_before_commit(
    tmp_path: Path,
    damage: str,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, f"root-{damage}")
    _setup(db)
    event_id = f"evt-root-{damage}"
    _record(db, event_id=event_id, opportunity_id=f"opp-root-{damage}")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    marker = backup_root / ".curiosity-backup-root.json"
    if damage == "missing":
        marker.unlink()
    elif damage == "corrupt":
        marker.write_text("{", encoding="utf-8")
    else:
        marker.write_text(
            json.dumps({"format": "another-backup-format", "format_version": 1}),
            encoding="utf-8",
        )

    with pytest.raises(RuntimeError):
        delete_family_event(
            db,
            child_id="child-a",
            event_id=event_id,
            confirmation=event_id,
            reason="Damaged root metadata must stop deletion.",
            output_root=output,
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
    assert (backup_root / str(snapshot["snapshot"])).is_dir()


@pytest.mark.parametrize("damage", ("missing", "corrupt", "mismatch"))
def test_registered_restore_marker_damage_blocks_deletion_before_commit(
    tmp_path: Path,
    damage: str,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, f"restore-{damage}")
    _setup(db)
    event_id = f"evt-restore-{damage}"
    _record(db, event_id=event_id, opportunity_id=f"opp-restore-{damage}")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    target = tmp_path / f"restore-{damage}" / "registered-recovery"
    restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=target,
        backup_root=backup_root,
        repo_root=repo,
    )
    marker = target / ".curiosity-managed-restore.json"
    if damage == "missing":
        marker.unlink()
    elif damage == "corrupt":
        marker.write_text("{", encoding="utf-8")
    else:
        value = json.loads(marker.read_text(encoding="utf-8"))
        value["snapshot"] = "curiosity-backup-mismatched"
        marker.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RuntimeError):
        delete_family_event(
            db,
            child_id="child-a",
            event_id=event_id,
            confirmation=event_id,
            reason="Damaged restore metadata must stop deletion.",
            output_root=output,
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
    with sqlite3.connect(target / "data" / "curiosity.db") as restored_db:
        assert restored_db.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1


def test_managed_deletion_check_fails_closed_for_unreadable_database(tmp_path: Path) -> None:
    database = tmp_path / "corrupt.db"
    database.write_bytes(b"not a sqlite database")
    manifest = {
        "events": {sha256(b"evt-deleted").hexdigest(): {}},
        "learning_sources": {},
        "learner_models": {},
    }

    with pytest.raises(RuntimeError, match="managed database image is unreadable"):
        database_deletion_conflicts(database, manifest)


def test_pre_identity_migration_backup_and_sqlite_sidecars_do_not_block_current_database(
    tmp_path: Path,
) -> None:
    db = tmp_path / "curiosity.db"
    _setup(db)
    legacy = db.with_name(f"{db.name}.backup-v17-legacy")
    with sqlite3.connect(db) as source, sqlite3.connect(legacy) as target:
        source.backup(target)
    with sqlite3.connect(legacy) as connection:
        connection.execute("DROP TABLE family_database_identity")
        connection.execute("PRAGMA user_version=17")

    Path(f"{legacy}-wal").touch()
    Path(f"{legacy}-shm").touch()
    Path(f"{legacy}-journal").touch()

    assert tuple(migration_backup_candidates(db)) == (legacy,)
    assert set(migration_backup_sidecars(legacy)) == {
        Path(f"{legacy}-wal"),
        Path(f"{legacy}-shm"),
        Path(f"{legacy}-journal"),
    }
    assert purge_managed_backups(db)["migration_backups_purged"] == 0


def test_current_migration_backup_without_database_identity_is_rejected(tmp_path: Path) -> None:
    db = tmp_path / "curiosity.db"
    _setup(db)
    backup = db.with_name(f"{db.name}.backup-v{SCHEMA_VERSION}-missing-identity")
    with sqlite3.connect(db) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    with sqlite3.connect(backup) as connection:
        connection.execute("DROP TABLE family_database_identity")

    with pytest.raises(RuntimeError, match="not a Curiosity database"):
        purge_managed_backups(db)


def test_wal_backed_migration_backup_is_scanned_and_purged_as_one_image(tmp_path: Path) -> None:
    db = tmp_path / "curiosity.db"
    _setup(db)
    backup = db.with_name(f"{db.name}.backup-v{SCHEMA_VERSION}-wal-backed")
    with sqlite3.connect(db) as source, sqlite3.connect(backup) as target:
        source.backup(target)

    event_id = "evt-wal-only-deleted"
    writer = sqlite3.connect(backup)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES(?, 'question', 'child-a', 'synthetic', 'test', '{}', ?, 'received')""",
            (event_id, utcnow()),
        )
        writer.commit()
        wal = Path(f"{backup}-wal")
        assert wal.stat().st_size > 0
        record_event_deletion(
            db,
            event_hash=sha256(event_id.encode()).hexdigest(),
            child_hash=sha256(b"child-a").hexdigest(),
            deleted_at=utcnow(),
            actor_type="parent",
        )

        assert purge_managed_backups(db)["migration_backups_purged"] == 1
        assert not backup.exists()
        assert not any(migration_backup_sidecars(backup))
    finally:
        writer.close()


def test_migration_backup_sidecar_unlink_failure_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "curiosity.db"
    _setup(db)
    event_id = "evt-sidecar-retry"
    backup = db.with_name(f"{db.name}.backup-v{SCHEMA_VERSION}-sidecar-retry")
    with sqlite3.connect(db) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    writer = sqlite3.connect(backup)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        writer.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES(?, 'question', 'child-a', 'synthetic', 'test', '{}', ?, 'received')""",
            (event_id, utcnow()),
        )
        writer.commit()
        wal = Path(f"{backup}-wal")
        assert wal.stat().st_size > 0
        record_event_deletion(
            db,
            event_hash=sha256(event_id.encode()).hexdigest(),
            child_hash=sha256(b"child-a").hexdigest(),
            deleted_at=utcnow(),
            actor_type="parent",
        )

        original_unlink = Path.unlink
        failed = False

        def fail_wal_once(path: Path, *args: object, **kwargs: object) -> None:
            nonlocal failed
            if path == wal and not failed:
                failed = True
                raise OSError("synthetic sidecar unlink failure")
            original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_wal_once)
        with pytest.raises(OSError, match="synthetic sidecar unlink failure"):
            purge_managed_backups(db)
        assert backup.exists()
        assert wal.exists()

        monkeypatch.setattr(Path, "unlink", original_unlink)
        assert purge_managed_backups(db)["migration_backups_purged"] == 1
        assert not backup.exists()
        assert not any(migration_backup_sidecars(backup))
    finally:
        writer.close()


def _damage_managed_database_image(
    *,
    repo: Path,
    private: Path,
    output: Path,
    db: Path,
    backup_root: Path,
    managed_kind: str,
    damage: str,
) -> Path:
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    snapshot_path = backup_root / str(snapshot["snapshot"])
    if managed_kind == "migration":
        image = db.with_name(f"{db.name}.backup-v23-preflight")
        shutil.copy2(db, image)
    elif managed_kind == "snapshot":
        image = snapshot_path / "data" / "curiosity.db"
    elif managed_kind == "partial":
        partial = backup_root / ".partial-preflight"
        shutil.copytree(snapshot_path, partial)
        image = partial / "data" / "curiosity.db"
    elif managed_kind == "restore":
        target = repo.parent / "registered-preflight-restore"
        restore_snapshot(
            str(snapshot["snapshot"]),
            target_private=target,
            backup_root=backup_root,
            repo_root=repo,
        )
        image = target / "data" / "curiosity.db"
    else:  # pragma: no cover - parametrization owns this boundary
        raise AssertionError(managed_kind)
    if damage == "missing":
        image.unlink()
        # Migration copies are discovered by canonical filename rather than a
        # separate registry. Preserve that discoverable path as a dangling,
        # unsafe image; a wholly absent copy contains no remaining family data.
        if managed_kind == "migration":
            image.symlink_to(image.with_name("absent-migration-image"))
    else:
        image.write_bytes(b"not a sqlite database")
    return image


@pytest.mark.parametrize("managed_kind", ("migration", "snapshot", "partial", "restore"))
@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_every_managed_database_image_is_checked_before_event_deletion_commit(
    tmp_path: Path,
    managed_kind: str,
    damage: str,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(
        tmp_path,
        f"preflight-event-{managed_kind}-{damage}",
    )
    _setup(db)
    event_id = f"evt-preflight-{managed_kind}-{damage}"
    opportunity_id = f"opp-preflight-{managed_kind}-{damage}"
    _record(db, event_id=event_id, opportunity_id=opportunity_id)
    _damage_managed_database_image(
        repo=repo,
        private=private,
        output=output,
        db=db,
        backup_root=backup_root,
        managed_kind=managed_kind,
        damage=damage,
    )

    with pytest.raises(RuntimeError, match="managed database image"):
        delete_family_event(
            db,
            child_id="child-a",
            event_id=event_id,
            confirmation=event_id,
            reason="A malformed managed copy must stop deletion before commit.",
            output_root=output,
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM learning_opportunities WHERE opportunity_id=?",
                (opportunity_id,),
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize("operation", ("event", "source", "model", "retention"))
@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_malformed_snapshot_leaves_each_destructive_family_operation_uncommitted(
    tmp_path: Path,
    operation: str,
    damage: str,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(
        tmp_path,
        f"preflight-{operation}-{damage}",
    )
    _setup(db)
    event_id = f"evt-preflight-{operation}-{damage}"
    opportunity_id = f"opp-preflight-{operation}-{damage}"
    _record(
        db,
        event_id=event_id,
        opportunity_id=opportunity_id,
        retention_class="transient" if operation == "retention" else "learning_record",
    )
    _damage_managed_database_image(
        repo=repo,
        private=private,
        output=output,
        db=db,
        backup_root=backup_root,
        managed_kind="snapshot",
        damage=damage,
    )

    with pytest.raises(RuntimeError, match="managed database image"):
        if operation == "event":
            delete_family_event(
                db,
                child_id="child-a",
                event_id=event_id,
                confirmation=event_id,
                reason="Preflight must precede event deletion.",
                output_root=output,
            )
        elif operation == "source":
            delete_family_learning_source(
                db,
                child_id="child-a",
                opportunity_id=opportunity_id,
                reason="Preflight must precede source deletion.",
            )
        elif operation == "model":
            delete_family_model(
                db,
                child_id="child-a",
                confirmation="child-a",
                reason="Preflight must precede model deletion.",
            )
        else:
            expire_learning_records(db, as_of=datetime.now(UTC) + timedelta(days=31))

    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM learning_opportunities WHERE opportunity_id=?",
                (opportunity_id,),
            ).fetchone()[0]
            == 1
        )


def test_missing_managed_restore_registry_cannot_forget_a_published_restore(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "missing-restore-registry")
    _setup(db)
    event_id = "evt-missing-restore-registry"
    _record(db, event_id=event_id, opportunity_id="opp-missing-restore-registry")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    target = repo.parent / "published-restore"
    restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=target,
        backup_root=backup_root,
        repo_root=repo,
    )
    (backup_root / "family-managed-restores.json").unlink()

    with pytest.raises(RuntimeError, match="managed-restore registry is missing"):
        delete_family_event(
            db,
            child_id="child-a",
            event_id=event_id,
            confirmation=event_id,
            reason="Lost restore authority must fail before deletion.",
            output_root=output,
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
    with sqlite3.connect(target / "data" / "curiosity.db") as restored:
        assert restored.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1


@pytest.mark.parametrize("managed_kind", ("snapshot", "restore"))
def test_valid_unrelated_sqlite_image_cannot_hide_backed_up_output(
    tmp_path: Path,
    managed_kind: str,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(
        tmp_path,
        f"wrong-database-{managed_kind}",
    )
    _setup(db)
    event_id = f"evt-wrong-database-{managed_kind}"
    output_marker = _record_file_event(db, output, event_id=event_id)
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    snapshot_path = backup_root / str(snapshot["snapshot"])
    if managed_kind == "snapshot":
        managed_database = snapshot_path / "data" / "curiosity.db"
        retained_output = snapshot_path / "output" / output_marker.name
    else:
        target = repo.parent / "wrong-database-restore"
        restore_snapshot(
            str(snapshot["snapshot"]),
            target_private=target,
            backup_root=backup_root,
            repo_root=repo,
        )
        managed_database = target / "data" / "curiosity.db"
        retained_output = target / "output" / output_marker.name
    unrelated = repo.parent / "unrelated" / "curiosity.db"
    unrelated.parent.mkdir()
    _setup(unrelated)
    shutil.copyfile(unrelated, managed_database)
    assert retained_output.read_text(encoding="utf-8") == f"private bytes for {event_id}"

    with pytest.raises(RuntimeError, match="different family database|does not match its manifest"):
        delete_family_event(
            db,
            child_id="child-a",
            event_id=event_id,
            confirmation=event_id,
            reason="A substituted database must not hide retained output.",
            output_root=output,
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
    assert retained_output.read_text(encoding="utf-8") == f"private bytes for {event_id}"


def test_wal_aware_scan_never_deletes_the_active_disaster_restore(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "active-wal")
    _setup(db)
    event_id = "evt-active-wal"
    _record(db, event_id=event_id, opportunity_id="opp-active-wal")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    shutil.rmtree(private)
    restored = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=private,
        backup_root=backup_root,
        repo_root=repo,
    )
    restored_db = Path(str(restored["database"]))
    reader = sqlite3.connect(restored_db)
    try:
        reader.execute("PRAGMA journal_mode=WAL")
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
        delete_family_event(
            restored_db,
            child_id="child-a",
            event_id=event_id,
            confirmation=event_id,
            reason="WAL state must be read without deleting the active installation.",
            output_root=private / "output",
        )
        assert private.is_dir() and restored_db.is_file()
        immutable = sqlite3.connect(
            f"{restored_db.resolve().as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        try:
            assert immutable.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 1
        finally:
            immutable.close()
        with connect(restored_db) as current:
            assert current.execute("SELECT COUNT(*) FROM events WHERE id=?", (event_id,)).fetchone()[0] == 0
    finally:
        reader.close()


def test_wal_only_model_state_in_registered_restore_is_detected_and_purged(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "restore-wal-model")
    _setup(db)
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    target = repo.parent / "wal-model-restore"
    restored = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=target,
        backup_root=backup_root,
        repo_root=repo,
    )
    restored_db = Path(str(restored["database"]))
    upsert_node(db, "child-a", "interest", "Active model state to delete")
    reader = sqlite3.connect(restored_db)
    try:
        reader.execute("PRAGMA journal_mode=WAL")
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
        writer = sqlite3.connect(restored_db)
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            now = utcnow()
            writer.execute(
                """INSERT INTO nodes(
                   child_id,kind,label,canonical_key,state_json,first_seen,last_seen
                   ) VALUES('child-a','interest','WAL-only model state','wal-only-model','{}',?,?)""",
                (now, now),
            )
            writer.commit()
        finally:
            writer.close()
        immutable = sqlite3.connect(
            f"{restored_db.resolve().as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        try:
            assert immutable.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
        finally:
            immutable.close()
        wal_aware = sqlite3.connect(f"{restored_db.resolve().as_uri()}?mode=ro", uri=True)
        try:
            assert wal_aware.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 1
        finally:
            wal_aware.close()

        delete_family_model(
            db,
            child_id="child-a",
            confirmation="child-a",
            reason="WAL-only restore state must be discovered.",
        )
        assert not target.exists()
    finally:
        reader.close()


def test_committed_event_tombstone_blocks_restore_after_post_commit_crash(tmp_path: Path) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "event-crash")
    _setup(db)
    _record(db, event_id="evt-crash-restore", opportunity_id="opp-crash-restore")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )

    def crash_after_commit(point: str) -> None:
        if point == "after_db_commit":
            raise SystemExit("synthetic post-commit crash")

    with pytest.raises(SystemExit, match="synthetic post-commit crash"):
        delete_family_event(
            db,
            child_id="child-a",
            event_id="evt-crash-restore",
            confirmation="evt-crash-restore",
            reason="Exercise committed tombstone recovery.",
            output_root=output,
            _test_crash_hook=crash_after_commit,
        )
    with pytest.raises(BackupError, match="deleted after the snapshot"):
        restore_snapshot(
            str(snapshot["snapshot"]),
            target_private=tmp_path / "crash-restore-target",
            backup_root=backup_root,
            repo_root=repo,
        )


def test_source_retention_and_model_deletions_block_stale_managed_restores(tmp_path: Path) -> None:
    operations = ("source", "retention", "model")
    for operation in operations:
        repo, private, output, db, backup_root = _backup_environment(tmp_path, operation)
        _setup(db)
        if operation == "model":
            upsert_node(db, "child-a", "interest", "Synthetic legacy-only model state")
        else:
            _record(
                db,
                event_id=f"evt-{operation}-restore",
                opportunity_id=f"opp-{operation}-restore",
                retention_class="transient" if operation == "retention" else "learning_record",
            )
        snapshot = create_snapshot(
            db_path=db,
            private_dir=private,
            output_dir=output,
            backup_root=backup_root,
            repo_root=repo,
        )
        snapshot_path = backup_root / str(snapshot["snapshot"])
        escrow = tmp_path / operation / "stale-snapshot"
        shutil.copytree(snapshot_path, escrow)
        if operation == "source":
            delete_family_learning_source(
                db,
                child_id="child-a",
                opportunity_id="opp-source-restore",
                reason="Synthetic source deletion.",
            )
        elif operation == "retention":
            expire_learning_records(db, as_of=datetime.now(UTC) + timedelta(days=31))
        else:
            delete_family_model(
                db,
                child_id="child-a",
                confirmation="child-a",
                reason="Synthetic full-model deletion.",
            )
        assert not snapshot_path.exists(), operation
        shutil.copytree(escrow, snapshot_path)
        with pytest.raises(BackupError, match="deleted after the snapshot"):
            restore_snapshot(
                str(snapshot["snapshot"]),
                target_private=tmp_path / operation / "refused-target",
                backup_root=backup_root,
                repo_root=repo,
            )


def test_snapshot_creation_serializes_with_event_deletion(tmp_path: Path) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "create-race")
    _setup(db)
    _record(db, event_id="evt-create-race", opportunity_id="opp-create-race")
    database_copied = threading.Event()
    release_creation = threading.Event()
    delete_finished = threading.Event()
    results: dict[str, dict[str, object]] = {}
    failures: list[BaseException] = []

    def creation_hook(point: str) -> None:
        if point == "after_database_copy":
            database_copied.set()
            assert release_creation.wait(10)

    def create_backup() -> None:
        try:
            results["snapshot"] = create_snapshot(
                db_path=db,
                private_dir=private,
                output_dir=output,
                backup_root=backup_root,
                repo_root=repo,
                _test_hook=creation_hook,
            )
        except BaseException as exc:
            failures.append(exc)

    def delete_event() -> None:
        try:
            delete_family_event(
                db,
                child_id="child-a",
                event_id="evt-create-race",
                confirmation="evt-create-race",
                reason="Serialize deletion with snapshot publication.",
                output_root=output,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            delete_finished.set()

    creator = threading.Thread(target=create_backup)
    creator.start()
    assert database_copied.wait(10)
    deleter = threading.Thread(target=delete_event)
    deleter.start()
    assert delete_finished.wait(0.2) is False
    release_creation.set()
    creator.join(15)
    deleter.join(15)
    assert not creator.is_alive() and not deleter.is_alive()
    assert failures == []
    snapshot_id = str(results["snapshot"]["snapshot"])
    assert not (backup_root / snapshot_id).exists()


def test_restore_serializes_with_deletion_and_registered_recovery_is_purged(tmp_path: Path) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "restore-race")
    _setup(db)
    _record(db, event_id="evt-restore-race", opportunity_id="opp-restore-race")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    target = tmp_path / "restore-race" / "managed-recovery"
    restore_paused = threading.Event()
    release_restore = threading.Event()
    delete_finished = threading.Event()
    failures: list[BaseException] = []

    def restore_hook(point: str) -> None:
        if point == "before_rebase":
            restore_paused.set()
            assert release_restore.wait(10)

    def restore_backup() -> None:
        try:
            restore_snapshot(
                str(snapshot["snapshot"]),
                target_private=target,
                backup_root=backup_root,
                repo_root=repo,
                _test_hook=restore_hook,
            )
        except BaseException as exc:
            failures.append(exc)

    def delete_event() -> None:
        try:
            delete_family_event(
                db,
                child_id="child-a",
                event_id="evt-restore-race",
                confirmation="evt-restore-race",
                reason="Serialize deletion with restore publication.",
                output_root=output,
            )
        except BaseException as exc:
            failures.append(exc)
        finally:
            delete_finished.set()

    restorer = threading.Thread(target=restore_backup)
    restorer.start()
    assert restore_paused.wait(10)
    deleter = threading.Thread(target=delete_event)
    deleter.start()
    assert delete_finished.wait(0.2) is False
    release_restore.set()
    restorer.join(15)
    deleter.join(15)
    assert not restorer.is_alive() and not deleter.is_alive()
    assert failures == []
    assert not target.exists()
    assert not (backup_root / str(snapshot["snapshot"])).exists()


def test_legacy_v1_root_is_authenticated_adopted_restorable_and_deletion_safe(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "legacy-adoption")
    _setup(db)
    event_id = "evt-legacy-adoption"
    _record(db, event_id=event_id, opportunity_id="opp-legacy-adoption")
    legacy_snapshot = _write_legacy_v1_snapshot(
        db=db,
        private=private,
        output=output,
        backup_root=backup_root,
    )

    current = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    assert json.loads((backup_root / ".curiosity-backup-root.json").read_text())["format_version"] == 2
    assert (backup_root / "family-backup-sources.json").is_file()
    assert (backup_root / "family-managed-restores.json").is_file()
    assert (backup_root / "family-backup-identity.json").is_file()

    restored_target = repo.parent / "legacy-restored"
    restored = restore_snapshot(
        legacy_snapshot.name,
        target_private=restored_target,
        backup_root=backup_root,
        repo_root=repo,
    )
    with sqlite3.connect(restored["database"]) as restored_db:
        assert restored_db.execute(
            "SELECT COUNT(*) FROM events WHERE id=?", (event_id,)
        ).fetchone()[0] == 1
        assert restored_db.execute("SELECT COUNT(*) FROM family_database_identity").fetchone()[0] == 1

    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Legacy managed copies must obey current deletion authority.",
        output_root=output,
    )
    assert not legacy_snapshot.exists()
    assert not (backup_root / str(current["snapshot"])).exists()
    assert not restored_target.exists()


def test_legacy_v1_root_with_different_source_roots_is_not_adopted(tmp_path: Path) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "legacy-wrong-family")
    _setup(db)
    _write_legacy_v1_snapshot(
        db=db,
        private=private.parent / "another-family",
        output=output,
        backup_root=backup_root,
    )

    with pytest.raises(BackupError, match="sources do not match"):
        create_snapshot(
            db_path=db,
            private_dir=private,
            output_dir=output,
            backup_root=backup_root,
            repo_root=repo,
        )
    marker = json.loads((backup_root / ".curiosity-backup-root.json").read_text())
    assert marker["format_version"] == 1
    assert not (backup_root / "family-backup-sources.json").exists()


def test_restore_imports_external_event_and_transport_replay_tombstones(tmp_path: Path) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "restore-replay-ledger")
    _setup(db)
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    _add_binding(db, binding_id="binding-before-delete")
    message = InboundMessage(
        external_event_id="external-after-snapshot",
        team_id="team-binding-before-delete",
        user_id="user-binding-before-delete",
        channel_id="channel",
        text="Synthetic transport event after the snapshot.",
    )
    assert begin_receipt(db, message, "binding-before-delete") is True
    event_id = "evt-after-snapshot"
    Repository(db).submit_event(
        Event(
            id=event_id,
            type="child_question",
            child_id="child-a",
            text=message.text,
            source="slack",
        )
    )
    finish_receipt(db, message, status="completed", event_id=event_id)
    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Replay must remain blocked after restoring an older image.",
        output_root=output,
    )

    shutil.rmtree(private)
    restored = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=private,
        backup_root=backup_root,
        repo_root=repo,
    )
    restored_db = Path(str(restored["database"]))
    with pytest.raises(IdempotencyConflict, match="deleted event"):
        Repository(restored_db).submit_event(
            Event(
                id=event_id,
                type="child_question",
                child_id="child-b",
                text="Changed child and payload must still be rejected.",
                source="parent",
            )
        )
    with pytest.raises(TransportConflict, match="deleted and cannot be replayed"):
        begin_receipt(restored_db, message, "binding-before-delete")
    changed_binding_message = message.model_copy(
        update={
            "team_id": "another-team",
            "user_id": "another-user",
            "text": "Changed binding and payload.",
        }
    )
    with pytest.raises(TransportConflict, match="deleted and cannot be replayed"):
        begin_receipt(restored_db, changed_binding_message, "binding-after-delete")


def test_model_deletion_purges_control_and_audit_only_historical_images(tmp_path: Path) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "control-only-model")
    _setup(db)
    set_collection_enabled(
        db,
        child_id="child-a",
        enabled=False,
        reason="Create control and audit-only historical state.",
    )
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    snapshot_path = backup_root / str(snapshot["snapshot"])
    with sqlite3.connect(snapshot_path / "data" / "curiosity.db") as image:
        assert image.execute(
            "SELECT COUNT(*) FROM learner_model_controls WHERE child_id='child-a'"
        ).fetchone()[0] == 1
        assert image.execute(
            "SELECT COUNT(*) FROM family_model_audit_log WHERE child_id='child-a'"
        ).fetchone()[0] >= 1
        assert all(
            image.execute(f"SELECT COUNT(*) FROM {table} WHERE child_id='child-a'").fetchone()[0]
            == 0
            for table in ("nodes", "claims", "observations", "learning_opportunities")
        )

    delete_family_model(
        db,
        child_id="child-a",
        confirmation="child-a",
        reason="Control and audit-only copies are still part of the child model.",
    )
    assert not snapshot_path.exists()


def test_legacy_adoption_immediately_purges_deletions_that_predate_registration(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "legacy-prior-delete")
    _setup(db)
    event_id = "evt-deleted-before-adoption"
    _record(db, event_id=event_id, opportunity_id="opp-deleted-before-adoption")
    legacy_snapshot = _write_legacy_v1_snapshot(
        db=db,
        private=private,
        output=output,
        backup_root=backup_root,
    )
    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Deletion happened while the legacy root was undiscoverable.",
        output_root=output,
    )
    assert legacy_snapshot.is_dir()

    current = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    assert not legacy_snapshot.exists()
    assert (backup_root / str(current["snapshot"])).is_dir()


def test_missing_required_root_ledger_blocks_restore_when_active_database_is_lost(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "missing-root-ledger")
    _setup(db)
    _add_binding(db, binding_id="binding-root-ledger")
    message = InboundMessage(
        external_event_id="external-root-ledger",
        team_id="team-binding-root-ledger",
        user_id="user-binding-root-ledger",
        channel_id="channel",
        text="Synthetic event retained only by the stale snapshot.",
    )
    assert begin_receipt(db, message, "binding-root-ledger") is True
    event_id = "evt-root-ledger"
    Repository(db).submit_event(
        Event(id=event_id, type="child_question", child_id="child-a", text=message.text, source="slack")
    )
    finish_receipt(db, message, status="completed", event_id=event_id)
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    snapshot_path = backup_root / str(snapshot["snapshot"])
    escrow = repo.parent / "stale-escrow"
    shutil.copytree(snapshot_path, escrow)
    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Root ledger must remain authoritative without the active database.",
        output_root=output,
    )
    assert not snapshot_path.exists()
    shutil.copytree(escrow, snapshot_path)
    shutil.rmtree(private)
    (backup_root / "family-deletions.json").unlink()

    with pytest.raises(BackupError, match="deletion ledger is unreadable"):
        restore_snapshot(
            str(snapshot["snapshot"]),
            target_private=private,
            backup_root=backup_root,
            repo_root=repo,
        )
    assert not private.exists()


def test_future_deletion_propagates_replay_guards_into_retained_restore(tmp_path: Path) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "future-restore-delete")
    _setup(db)
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    retained_target = repo.parent / "retained-before-event"
    restored = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=retained_target,
        backup_root=backup_root,
        repo_root=repo,
    )
    _add_binding(db, binding_id="binding-future-delete")
    message = InboundMessage(
        external_event_id="external-future-delete",
        team_id="team-binding-future-delete",
        user_id="user-binding-future-delete",
        channel_id="channel",
        text="Synthetic event created after the recovery copy.",
    )
    assert begin_receipt(db, message, "binding-future-delete") is True
    event_id = "evt-future-delete"
    Repository(db).submit_event(
        Event(id=event_id, type="child_question", child_id="child-a", text=message.text, source="slack")
    )
    finish_receipt(db, message, status="completed", event_id=event_id)
    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Future deletion must update retained replay authority.",
        output_root=output,
    )
    assert retained_target.is_dir()
    retained_db = Path(str(restored["database"]))
    with pytest.raises(IdempotencyConflict, match="deleted event"):
        Repository(retained_db).submit_event(
            Event(id=event_id, type="system_note", text="Changed replay payload.", source="system")
        )
    changed = message.model_copy(
        update={"team_id": "changed-team", "user_id": "changed-user", "text": "Changed replay."}
    )
    with pytest.raises(TransportConflict, match="deleted and cannot be replayed"):
        begin_receipt(retained_db, changed, "changed-binding")


@pytest.mark.parametrize("deletion_kind", ("parent", "retention"))
def test_pre_opportunity_restore_imports_source_replay_tombstone(
    tmp_path: Path,
    deletion_kind: str,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(
        tmp_path,
        f"source-replay-{deletion_kind}",
    )
    _setup(db)
    event_id = f"evt-source-replay-{deletion_kind}"
    plan_id = f"plan-source-replay-{deletion_kind}"
    opportunity_id = f"opp-source-replay-{deletion_kind}"
    Repository(db).submit_event(
        Event(
            id=event_id,
            type="child_question",
            child_id="child-a",
            text="Synthetic plan exists before its learning observation.",
            source="parent",
        )
    )
    register_learning_plan(
        db,
        {
            "plan_id": plan_id,
            "child_id": "child-a",
            "source_event_id": event_id,
            "task_id": "task.shadows",
        },
    )
    child_b_event_id = f"evt-source-replay-child-b-{deletion_kind}"
    child_b_plan_id = f"plan-source-replay-child-b-{deletion_kind}"
    Repository(db).submit_event(
        Event(
            id=child_b_event_id,
            type="child_question",
            child_id="child-b",
            text="Synthetic alternate-child replay path.",
            source="parent",
        )
    )
    register_learning_plan(
        db,
        {
            "plan_id": child_b_plan_id,
            "child_id": "child-b",
            "source_event_id": child_b_event_id,
            "task_id": "task.shadows",
        },
    )
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    opportunity = {
        "opportunity_id": opportunity_id,
        "child_id": "child-a",
        "event_id": event_id,
        "plan_id": plan_id,
        "actor": "child",
        "initiation": "child",
        "response_mode": "spoken",
        "assistance_level": "none",
        "criterion_results": [{"criterion_id": "criterion.shadows", "outcome": "correct"}],
        "scorer": "reviewed-parent-observation",
        "scorer_version": "1",
        "source_type": "direct_observation",
        "retention_class": "transient" if deletion_kind == "retention" else "learning_record",
    }
    record_learning_opportunity(db, opportunity)
    if deletion_kind == "parent":
        delete_family_learning_source(
            db,
            child_id="child-a",
            opportunity_id=opportunity_id,
            reason="Parent removed the synthetic source.",
        )
    else:
        expire_learning_records(db, as_of=datetime.now(UTC) + timedelta(days=31))

    shutil.rmtree(private)
    restored = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=private,
        backup_root=backup_root,
        repo_root=repo,
    )
    with pytest.raises(EvidenceConflict, match="deleted learning opportunity"):
        record_learning_opportunity(Path(str(restored["database"])), opportunity)
    changed_child_opportunity = {
        **opportunity,
        "child_id": "child-b",
        "event_id": child_b_event_id,
        "plan_id": child_b_plan_id,
    }
    with pytest.raises(EvidenceConflict, match="deleted learning opportunity"):
        record_learning_opportunity(
            Path(str(restored["database"])),
            changed_child_opportunity,
        )


def test_post_model_deletion_collection_restart_can_be_snapshotted_and_restored(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "post-delete-restart")
    _setup(db)
    old = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    delete_family_model(
        db,
        child_id="child-a",
        confirmation="child-a",
        reason="Reset the model before a new parent-approved learning epoch.",
    )
    assert not (backup_root / str(old["snapshot"])).exists()
    set_collection_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Parent explicitly restarted collection.",
    )
    _record(db, event_id="evt-new-epoch", opportunity_id="opp-new-epoch")
    current = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    restored = restore_snapshot(
        str(current["snapshot"]),
        target_private=repo.parent / "new-epoch-restore",
        backup_root=backup_root,
        repo_root=repo,
    )
    with sqlite3.connect(restored["database"]) as restored_db:
        assert restored_db.execute(
            "SELECT COUNT(*) FROM learning_opportunities WHERE opportunity_id='opp-new-epoch'"
        ).fetchone()[0] == 1
        assert restored_db.execute(
            "SELECT collection_enabled FROM learner_model_controls WHERE child_id='child-a'"
        ).fetchone()[0] == 1


def test_deleted_learning_opportunity_id_is_global_across_children(tmp_path: Path) -> None:
    db = tmp_path / "global-source-replay.sqlite"
    _setup(db)
    opportunity_id = "opp-global-replay"
    _record(db, event_id="evt-global-replay-a", opportunity_id=opportunity_id)
    delete_family_learning_source(
        db,
        child_id="child-a",
        opportunity_id=opportunity_id,
        reason="The identifier is globally retired after deletion.",
    )
    Repository(db).submit_event(
        Event(
            id="evt-global-replay-b",
            type="child_question",
            child_id="child-b",
            text="Synthetic cross-child replay.",
            source="parent",
        )
    )
    register_learning_plan(
        db,
        {
            "plan_id": "plan-global-replay-b",
            "child_id": "child-b",
            "source_event_id": "evt-global-replay-b",
            "task_id": "task.shadows",
        },
    )
    with pytest.raises(EvidenceConflict, match="deleted learning opportunity"):
        record_learning_opportunity(
            db,
            {
                "opportunity_id": opportunity_id,
                "child_id": "child-b",
                "event_id": "evt-global-replay-b",
                "plan_id": "plan-global-replay-b",
                "actor": "child",
                "initiation": "child",
                "response_mode": "spoken",
                "assistance_level": "none",
                "criterion_results": [
                    {"criterion_id": "criterion.shadows", "outcome": "correct"}
                ],
                "scorer": "reviewed-parent-observation",
                "scorer_version": "1",
                "source_type": "direct_observation",
            },
        )


def test_database_rejects_a_replayed_opportunity_with_a_forged_hash(tmp_path: Path) -> None:
    db = tmp_path / "forged-source-hash.sqlite"
    _setup(db)
    deleted_id = "opp-forged-hash-deleted"
    template_id = "opp-forged-hash-template"
    _record(db, event_id="evt-forged-hash-deleted", opportunity_id=deleted_id)
    _record(db, event_id="evt-forged-hash-template", opportunity_id=template_id)
    delete_family_learning_source(
        db,
        child_id="child-a",
        opportunity_id=deleted_id,
        reason="The direct database replay must remain blocked.",
    )
    with connect(db) as conn:
        columns = [str(row[1]) for row in conn.execute("PRAGMA table_info(learning_opportunities)")]
        select_values: list[str] = []
        parameters: list[str] = []
        for column in columns:
            if column == "opportunity_id":
                select_values.append("?")
                parameters.append(deleted_id)
            elif column == "opportunity_id_hash":
                select_values.append("?")
                parameters.append("0" * 64)
            else:
                select_values.append(f'"{column}"')
        parameters.append(template_id)
        with pytest.raises(sqlite3.IntegrityError, match="deleted learning opportunity"):
            conn.execute(
                f"""INSERT INTO learning_opportunities({','.join(columns)})
                    SELECT {','.join(select_values)} FROM learning_opportunities
                    WHERE opportunity_id=?""",
                parameters,
            )


def test_event_deletion_closes_cross_event_plan_opportunity_lineage(tmp_path: Path) -> None:
    db = tmp_path / "cross-event-plan.sqlite"
    _setup(db)
    source_event_id = "evt-plan-source-delete"
    observation_event_id = "evt-plan-observation-keep"
    Repository(db).submit_event(
        Event(
            id=source_event_id,
            type="child_question",
            child_id="child-a",
            text="Synthetic source question.",
            source="parent",
        )
    )
    Repository(db).submit_event(
        Event(
            id=observation_event_id,
            type="child_question",
            child_id="child-a",
            text="Synthetic follow-up observation.",
            source="parent",
        )
    )
    apply_episode_correction(
        db,
        child_id="child-a",
        event_id=observation_event_id,
        action="retry",
        related_event_id=source_event_id,
    )
    register_learning_plan(
        db,
        {
            "plan_id": "plan-cross-event-delete",
            "child_id": "child-a",
            "source_event_id": source_event_id,
            "task_id": "task.shadows",
        },
    )
    record_learning_opportunity(
        db,
        {
            "opportunity_id": "opp-cross-event-delete",
            "child_id": "child-a",
            "event_id": observation_event_id,
            "plan_id": "plan-cross-event-delete",
            "actor": "child",
            "initiation": "child",
            "response_mode": "spoken",
            "assistance_level": "none",
            "criterion_results": [{"criterion_id": "criterion.shadows", "outcome": "correct"}],
            "scorer": "reviewed-parent-observation",
            "scorer_version": "1",
            "source_type": "direct_observation",
        },
    )

    result = delete_family_event(
        db,
        child_id="child-a",
        event_id=source_event_id,
        confirmation=source_event_id,
        reason="Delete the complete plan-derived learning closure.",
    )
    assert result["learning_sources_deleted"] == 1
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events WHERE id=?", (source_event_id,)).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE id=?", (observation_event_id,)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM learning_opportunities WHERE opportunity_id='opp-cross-event-delete'"
        ).fetchone()[0] == 0
        assert conn.execute(
            """SELECT COUNT(*) FROM evidence_source_tombstones
               WHERE target_id_hash=?""",
            (sha256(b"opp-cross-event-delete").hexdigest(),),
        ).fetchone()[0] == 1


def test_event_deletion_recursively_closes_transfer_plan_lineage(tmp_path: Path) -> None:
    db = tmp_path / "recursive-transfer-delete.sqlite"
    _setup(db)
    root_event = "evt-transfer-root-delete"
    _record(db, event_id=root_event, opportunity_id="opp-transfer-root-delete")

    previous_plan = "plan-opp-transfer-root-delete"
    descendant_events: list[str] = []
    for ordinal in (1, 2):
        event_id = f"evt-transfer-descendant-{ordinal}"
        plan_id = f"plan-transfer-descendant-{ordinal}"
        opportunity_id = f"opp-transfer-descendant-{ordinal}"
        descendant_events.append(event_id)
        Repository(db).submit_event(
            Event(
                id=event_id,
                type="child_question",
                child_id="child-a",
                text=f"Synthetic transfer observation {ordinal}.",
                source="parent",
            )
        )
        register_learning_plan(
            db,
            {
                "plan_id": plan_id,
                "child_id": "child-a",
                "source_event_id": event_id,
                "task_id": "task.shadows",
                "context_key": f"transfer-context-{ordinal}",
                "representation_key": f"transfer-representation-{ordinal}",
                "transfer_from_plan_id": previous_plan,
            },
        )
        record_learning_opportunity(
            db,
            {
                "opportunity_id": opportunity_id,
                "child_id": "child-a",
                "event_id": event_id,
                "plan_id": plan_id,
                "actor": "child",
                "initiation": "child",
                "response_mode": "spoken",
                "assistance_level": "none",
                "criterion_results": [
                    {"criterion_id": "criterion.shadows", "outcome": "correct"}
                ],
                "scorer": "reviewed-parent-observation",
                "scorer_version": "1",
                "source_type": "direct_observation",
            },
        )
        previous_plan = plan_id

    result = delete_family_event(
        db,
        child_id="child-a",
        event_id=root_event,
        confirmation=root_event,
        reason="Delete the recursive transfer-plan learning closure.",
    )
    assert result["learning_sources_deleted"] == 3
    with connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM learning_opportunities WHERE opportunity_id LIKE 'opp-transfer-%'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM learning_episode_plans WHERE plan_id LIKE 'plan-transfer-%'"
        ).fetchone()[0] == 0
        assert conn.execute(
            f"SELECT COUNT(*) FROM events WHERE id IN ({','.join('?' for _ in descendant_events)})",
            descendant_events,
        ).fetchone()[0] == len(descendant_events)


def test_pausing_after_model_restart_keeps_active_restore_epoch_valid(tmp_path: Path) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "paused-new-epoch")
    _setup(db)
    old = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    delete_family_model(
        db,
        child_id="child-a",
        confirmation="child-a",
        reason="Start a clean learner-model epoch.",
    )
    assert not (backup_root / str(old["snapshot"])).exists()
    set_collection_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Parent approved a new collection epoch.",
    )
    _record(db, event_id="evt-paused-new-epoch", opportunity_id="opp-paused-new-epoch")
    current = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    shutil.rmtree(private)
    restored = restore_snapshot(
        str(current["snapshot"]),
        target_private=private,
        backup_root=backup_root,
        repo_root=repo,
    )
    restored_db = Path(str(restored["database"]))
    set_collection_enabled(
        restored_db,
        child_id="child-a",
        enabled=False,
        reason="Pause without invalidating the parent-approved epoch.",
    )
    init_db(restored_db)
    paused_snapshot = create_snapshot(
        db_path=restored_db,
        private_dir=private,
        output_dir=private / "output",
        backup_root=backup_root,
        repo_root=repo,
    )
    assert (backup_root / str(paused_snapshot["snapshot"])).is_dir()


def test_killed_local_first_backup_registration_recovers_before_deletion(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "killed-registration")
    _setup(db)
    event_id = "evt-killed-registration"
    _record(db, event_id=event_id, opportunity_id="opp-killed-registration")
    legacy_snapshot = _write_legacy_v1_snapshot(
        db=db,
        private=private,
        output=output,
        backup_root=backup_root,
    )
    escrow = repo.parent / "killed-registration-escrow"
    shutil.copytree(legacy_snapshot, escrow)
    script = """
import os
import signal
import sys
from curiosity_engine.backups import create_snapshot

def crash(point):
    if point == 'after_local_backup_registration':
        os.kill(os.getpid(), signal.SIGKILL)

create_snapshot(
    db_path=sys.argv[1],
    private_dir=sys.argv[2],
    output_dir=sys.argv[3],
    backup_root=sys.argv[4],
    repo_root=sys.argv[5],
    _test_hook=crash,
)
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(db),
            str(private),
            str(output),
            str(backup_root),
            str(repo),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        timeout=30,
    )
    assert completed.returncode == -signal.SIGKILL
    local_registry = json.loads((db.parent / "family-backup-roots.json").read_text())
    root_registry = json.loads((backup_root / "family-backup-sources.json").read_text())
    assert str(backup_root.resolve()) in local_registry["roots"]
    assert root_registry["databases"] == []
    assert (backup_root / "family-backup-registration-pending.json").is_file()
    assert not (backup_root / "family-backup-identity.json").exists()

    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Pending local-first registration must recover before deletion.",
        output_root=output,
    )
    assert not legacy_snapshot.exists()
    assert not (backup_root / "family-backup-registration-pending.json").exists()
    shutil.copytree(escrow, legacy_snapshot)
    shutil.rmtree(private)
    with pytest.raises(BackupError, match="deleted after the snapshot"):
        restore_snapshot(
            legacy_snapshot.name,
            target_private=private,
            backup_root=backup_root,
            repo_root=repo,
        )
    assert not private.exists()


def test_preadoption_deletion_cannot_restore_after_registration_kill_and_database_loss(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "killed-adoption")
    _setup(db)
    event_id = "evt-killed-adoption"
    _record(db, event_id=event_id, opportunity_id="opp-killed-adoption")
    legacy_snapshot = _write_legacy_v1_snapshot(
        db=db,
        private=private,
        output=output,
        backup_root=backup_root,
    )
    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Delete before the legacy backup root is adopted.",
        output_root=output,
    )
    assert legacy_snapshot.exists()

    script = """
import os
import signal
import sys
from curiosity_engine.backups import create_snapshot

def crash(point):
    if point == 'after_backup_registration_pending':
        os.kill(os.getpid(), signal.SIGKILL)

create_snapshot(
    db_path=sys.argv[1],
    private_dir=sys.argv[2],
    output_dir=sys.argv[3],
    backup_root=sys.argv[4],
    repo_root=sys.argv[5],
    _test_hook=crash,
)
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(db),
            str(private),
            str(output),
            str(backup_root),
            str(repo),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        timeout=30,
    )
    assert completed.returncode == -signal.SIGKILL
    assert (backup_root / "family-backup-registration-pending.json").is_file()
    assert not (backup_root / "family-backup-identity.json").exists()
    local_registry_path = db.parent / "family-backup-roots.json"
    if local_registry_path.exists():
        assert str(backup_root.resolve()) not in json.loads(local_registry_path.read_text())["roots"]

    shutil.rmtree(private)
    with pytest.raises(BackupError, match="registration is incomplete"):
        restore_snapshot(
            legacy_snapshot.name,
            target_private=private,
            backup_root=backup_root,
            repo_root=repo,
        )
    assert not private.exists()


def test_missing_pending_barrier_after_identity_kill_still_cannot_restore(
    tmp_path: Path,
) -> None:
    repo, private, output, db, backup_root = _backup_environment(tmp_path, "missing-pending")
    _setup(db)
    event_id = "evt-missing-pending"
    _record(db, event_id=event_id, opportunity_id="opp-missing-pending")
    legacy_snapshot = _write_legacy_v1_snapshot(
        db=db,
        private=private,
        output=output,
        backup_root=backup_root,
    )
    delete_family_event(
        db,
        child_id="child-a",
        event_id=event_id,
        confirmation=event_id,
        reason="Delete before testing loss of incomplete registration metadata.",
        output_root=output,
    )

    script = """
import os
import signal
import sys
from curiosity_engine.backups import create_snapshot

def crash(point):
    if point == 'after_backup_identity_binding':
        os.kill(os.getpid(), signal.SIGKILL)

create_snapshot(
    db_path=sys.argv[1],
    private_dir=sys.argv[2],
    output_dir=sys.argv[3],
    backup_root=sys.argv[4],
    repo_root=sys.argv[5],
    _test_hook=crash,
)
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(db),
            str(private),
            str(output),
            str(backup_root),
            str(repo),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        timeout=30,
    )
    assert completed.returncode == -signal.SIGKILL
    pending = backup_root / "family-backup-registration-pending.json"
    assert pending.is_file()
    assert (backup_root / "family-backup-identity.json").is_file()
    assert json.loads((backup_root / "family-backup-sources.json").read_text())["databases"] == []

    pending.unlink()
    shutil.rmtree(private)
    with pytest.raises(BackupError, match="registration is incomplete"):
        restore_snapshot(
            legacy_snapshot.name,
            target_private=private,
            backup_root=backup_root,
            repo_root=repo,
        )
    assert not private.exists()


def test_single_source_delete_is_idempotent_and_audited(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _record(db)
    result = delete_family_learning_source(
        db,
        child_id="child-a",
        opportunity_id="opp-shadow-a",
        reason="Parent removed one learning source.",
    )
    assert result["duplicate"] is False
    assert result["audit_id"].startswith("fma_")
    assert inspect_family_model(db, "child-a")["counts"]["evidence_records"] == 0


def test_export_is_owner_only_child_scoped_and_refuses_unrequested_overwrite(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _record(db, text="Synthetic A export text.")
    _record(
        db,
        child_id="child-b",
        event_id="evt-b",
        opportunity_id="opp-b",
        text="Synthetic B must not export.",
    )
    now = utcnow()
    with connect(db) as conn:
        run_id = conn.execute(
            """INSERT INTO runs(job,status,started_at,completed_at,details_json,event_id)
               VALUES('synthetic-export','completed',?,?,'{}','evt-shadow-a')""",
            (now, now),
        ).lastrowid
        conn.execute(
            """INSERT INTO graph_effects(event_id,run_id,mutation_json,status,applied_at)
               VALUES('evt-shadow-a',?,'{"kind":"synthetic-export"}','applied',?)""",
            (run_id, now),
        )
        conn.execute(
            """INSERT INTO context_corrections(
               child_id,event_id,action,previous_json,note,created_at
               ) VALUES('child-a','evt-shadow-a','deepening','{}','Synthetic export correction',?)""",
            (now,),
        )
    retention = set_learning_source_retention(
        db,
        child_id="child-a",
        opportunity_id="opp-shadow-a",
        retain_until=datetime.now(UTC) + timedelta(days=5),
        reason="Synthetic export provenance.",
    )
    assert retention["changed"] is True
    for index in range(54):
        set_collection_enabled(
            db,
            child_id="child-a",
            enabled=bool(index % 2),
            reason=f"Synthetic audit event {index}.",
        )
    output = tmp_path / "exports" / "child-a.json"
    report = write_family_model_export(db, "child-a", output)
    assert report["mode"] == "0o600"
    payload = output.read_text(encoding="utf-8")
    assert "Synthetic A export text" in payload
    assert "Synthetic B must not export" not in payload
    assert "child-b" not in payload
    decoded = json.loads(payload)
    assert decoded["format_version"] == 2
    assert len(decoded["records"]["family_model_audit_log"]) > 50
    assert len(decoded["inspector"]["audit"]) > 50
    assert decoded["retention_events"][0]["opportunity_id"] == "opp-shadow-a"
    assert decoded["records"]["learning_episode_plans"][0]["plan_id"] == "plan-opp-shadow-a"
    assert decoded["records"]["source_events"][0]["id"] == "evt-shadow-a"
    assert decoded["records"]["context_corrections"][0]["note"] == "Synthetic export correction"
    assert decoded["records"]["graph_effects"][0]["event_id"] == "evt-shadow-a"
    assert decoded["records"]["runs"][0]["event_id"] == "evt-shadow-a"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_family_model_export(db, "child-a", output)


def test_export_has_no_hidden_evidence_record_cap(tmp_path: Path) -> None:
    db = tmp_path / "complete-export.sqlite"
    _setup(db)
    _record(db)
    with connect(db) as conn:
        base_evidence_id = str(
            conn.execute(
                """SELECT evidence_id FROM evidence_records
                   WHERE child_id='child-a' AND construct='knowledge_performance' LIMIT 1"""
            ).fetchone()[0]
        )
        conn.execute("BEGIN IMMEDIATE")
        for index in range(501):
            event_id = f"evt-export-{index:04d}"
            plan_id = f"plan-export-{index:04d}"
            opportunity_id = f"opp-export-{index:04d}"
            attempt_group_id = f"attempt-export-{index:04d}"
            conn.execute(
                """INSERT INTO events(
                   id,type,child_id,text,source,metadata_json,created_at,status
                   ) VALUES(?,'child_question','child-a',?,'synthetic','{}',?,'completed')""",
                (event_id, f"Synthetic export source {index}", utcnow()),
            )
            conn.execute(
                """INSERT INTO learning_episode_plans(
                   plan_id,child_id,episode_id,task_id,task_version,source_event_id,attempt_group_id,
                   context_key,representation_key,transfer_from_plan_id,status,plan_hash,created_at,updated_at
                   ) SELECT ?,child_id,episode_id,task_id,task_version,?,?,context_key,representation_key,
                     NULL,status,?,created_at,updated_at
                   FROM learning_episode_plans WHERE plan_id='plan-opp-shadow-a'""",
                (plan_id, event_id, attempt_group_id, sha256(plan_id.encode()).hexdigest()),
            )
            conn.execute(
                """INSERT INTO learning_opportunities(
                   opportunity_id,child_id,event_id,episode_id,task_id,task_version,plan_id,rubric_id,
                   rubric_version,evidence_rule_id,evidence_rule_version,attempt_group_id,attempt_number,
                   actor,initiation,response_mode,assistance_level,result,score,criterion_results_json,
                   scorer,scorer_version,source_type,independence_status,context_key,representation_key,
                   is_transfer,is_trivial,child_continued,child_new_question,retention_class,
                   opportunity_hash,opportunity_id_hash,occurred_at,created_at
                   ) SELECT ?,child_id,?,episode_id,task_id,task_version,?,rubric_id,rubric_version,
                     evidence_rule_id,evidence_rule_version,?,1,actor,initiation,response_mode,
                     assistance_level,result,score,criterion_results_json,scorer,scorer_version,source_type,
                     independence_status,context_key,representation_key,is_transfer,is_trivial,
                     child_continued,child_new_question,retention_class,?,?,occurred_at,created_at
                   FROM learning_opportunities WHERE opportunity_id='opp-shadow-a'""",
                (
                    opportunity_id,
                    event_id,
                    plan_id,
                    attempt_group_id,
                    sha256(opportunity_id.encode()).hexdigest(),
                    sha256(opportunity_id.encode()).hexdigest(),
                ),
            )
            conn.execute(
                """INSERT INTO evidence_records(
                   evidence_id,child_id,opportunity_id,concept_id,concept_version,construct,
                   observation_kind,stance,result,score,rubric_id,rubric_version,scored_criteria_json,
                   source_type,source_reliability,evidence_strength,projection_eligible,episode_id,
                   context_key,representation_key,assistance_level,attempt_number,is_transfer,is_trivial,
                   rule_id,rule_version,retention_class,expires_at,created_at
                   ) SELECT ?,child_id,?,concept_id,concept_version,construct,observation_kind,stance,result,
                     score,rubric_id,rubric_version,scored_criteria_json,source_type,source_reliability,
                     evidence_strength,projection_eligible,episode_id,context_key,representation_key,
                     assistance_level,attempt_number,is_transfer,is_trivial,rule_id,rule_version,
                     retention_class,expires_at,created_at
                   FROM evidence_records WHERE evidence_id=?""",
                (f"evidence-export-{index:04d}", opportunity_id, base_evidence_id),
            )

    payload = export_family_model(db, "child-a")
    assert len(payload["inspector"]["evidence"]) > 500
    assert len(payload["records"]["evidence_records"]) > 500


def test_child_reflection_is_gentle_optional_and_never_becomes_evidence(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _record(db)
    before = learner_snapshot(db, "child-a")
    card = child_reflection_card(db, "child-a")
    visible_copy = json.dumps(
        {"title": card["title"], "intro": card["intro"], "cards": card["cards"]},
        ensure_ascii=False,
    ).casefold()
    assert card["parent_present_required"] is True
    assert card["responses_become_evidence"] is False
    assert "grade" in visible_copy
    assert "mastered" not in visible_copy
    assert "diagnosis" not in visible_copy
    first = card["cards"][0]
    result = record_child_reflection(
        db,
        child_id="child-a",
        concept_id=first["concept_id"],
        concept_version=first["concept_version"],
        response="not_yet",
        note="Synthetic co-review note.",
    )
    after = learner_snapshot(db, "child-a")
    assert result["became_evidence"] is False
    assert result["projection_changed"] is False
    assert after["snapshot_digest"] == before["snapshot_digest"]
    with connect(db) as conn:
        assert conn.execute("SELECT authority FROM child_reflection_responses").fetchone()[0] == "review_only"


def test_local_web_inspector_mutations_export_and_child_card_are_csrf_protected(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _record(db, text="Synthetic local inspector evidence.")
    app = create_app(db, tmp_path / "output")
    client = TestClient(app)

    home = client.get("/?child=child-a")
    assert home.status_code == 200
    assert "OPEN FAMILY MODEL" in home.text
    assert "Synthetic local inspector evidence" in home.text
    evidence_id = _knowledge_evidence_id(db)
    denied = client.post(
        f"/family-model/evidence/{evidence_id}/correct",
        data={"child_id": "child-a", "action": "exclude", "reason": "Synthetic correction.", "csrf": "wrong"},
    )
    assert denied.status_code == 403

    corrected = client.post(
        f"/family-model/evidence/{evidence_id}/correct",
        data={
            "child_id": "child-a",
            "action": "exclude",
            "reason": "Synthetic correction.",
            "csrf": app.state.csrf,
        },
        follow_redirects=False,
    )
    assert corrected.status_code == 303
    exported = client.post(
        "/family-model/export",
        data={"child_id": "child-a", "csrf": app.state.csrf},
    )
    assert exported.status_code == 200
    assert exported.headers["content-disposition"].startswith("attachment;")
    assert exported.json()["inspector"]["child"]["id"] == "child-a"
    reflection = client.get("/family-model/child-reflection?child=child-a")
    assert reflection.status_code == 200
    assert "you are the expert on your own experience" in reflection.text


def test_cli_exposes_explicit_family_model_controls() -> None:
    parser = build_parser()
    assert parser.parse_args(["learner-model", "inspect", "--child", "child-a"]).learner_model_cmd == "inspect"
    assert (
        parser.parse_args(
            [
                "learner-model",
                "delete-all",
                "--child",
                "child-a",
                "--confirm-child",
                "child-a",
                "--reason",
                "Synthetic deletion.",
            ]
        ).learner_model_cmd
        == "delete-all"
    )
