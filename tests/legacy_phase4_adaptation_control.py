"""Historical Phase 4 concurrency matrix; excluded from the default beta gate."""

from __future__ import annotations

import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from curiosity_engine.adaptation import (
    BackgroundRunService,
    FamilyRecipeService,
    ParentMemoryLedger,
    RecipeCaseVerdict,
    ThreadCompactor,
)
from curiosity_engine.backups import create_snapshot, restore_snapshot
from curiosity_engine.cli import build_parser
from curiosity_engine.context_builder import build_context
from curiosity_engine.contracts import CriticResult, Event, ParentAgentTurn, RunResult
from curiosity_engine.db import SCHEMA_VERSION, connect, init_db, jdump, jload, utcnow
from curiosity_engine.deletion_ledger import (
    PARENT_MEMORY_INSPECTION_PREFIX,
    REDACTED_PARENT_MEMORY_INSPECTION,
    backup_root_deletion_manifest,
    database_deletion_conflicts,
    event_id_hash,
)
from curiosity_engine.family_model import delete_family_event
from curiosity_engine.graph import add_child
from curiosity_engine.interaction import (
    claim_delivery,
    consume_pairing_code,
    create_pairing_code,
    enqueue_delivery,
    ready_deliveries,
    revoke_binding,
)
from curiosity_engine.learning_artifacts import LearningArtifactService
from curiosity_engine.parent_agent import ParentAgentRuntime
from curiosity_engine.pedagogy import ApprovedPedagogyPlan
from curiosity_engine.reasoning import StubBackend
from curiosity_engine.repository import Repository
from curiosity_engine.service import CuriosityService
from curiosity_engine.sessions import SessionStore
from curiosity_engine.tooling import ToolRegistry
from curiosity_engine.transports.contracts import InboundMessage, OutboundMessage
from curiosity_engine.transports.slack import SlackTransport, flush_slack_outbox


def _base(db: Path) -> tuple[str, str]:
    init_db(db)
    add_child(db, "child-a", "Synthetic Child", 2020, "1")
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO parent_principals VALUES(?,?,?,?,?,?)",
            ("parent-a", "Synthetic Parent", "owner", "active", now, now),
        )
        conn.execute(
            "INSERT INTO parent_principals VALUES(?,?,?,?,?,?)",
            ("parent-b", "Other Parent", "parent", "active", now, now),
        )
        conn.execute(
            "INSERT INTO transport_bindings VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "binding-a",
                "slack",
                "team",
                "user",
                "channel",
                "parent-a",
                "active",
                now,
                now,
            ),
        )
    store = SessionStore(db)
    session = store.get_or_create(
        origin="slack",
        transport="slack",
        binding_id="binding-a",
        conversation_ref="conversation-a",
        thread_ref="thread-a",
        child_id="child-a",
    )
    source_message_id = store.append_message(
        str(session["id"]),
        role="user",
        content="Please keep future answers compact.",
    )
    return str(session["id"]), source_message_id


def _release_unit(db: Path, session_id: str) -> tuple[str, str]:
    capability_run_id = "caprun-background"
    release_unit_id = "release-background"
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO capability_runs(
                   id,session_id,capability_id,capability_version,skill_versions_json,status,
                   result_json,created_at,updated_at
               ) VALUES(?,?,?,?,?,'running','{}',?,?)""",
            (capability_run_id, session_id, "artifact_worksheet", "2", "{}", now, now),
        )
        conn.execute(
            """INSERT INTO release_units(
                   id,capability_run_id,unit_type,ordinal,status,content_hash,payload_json,
                   created_at,updated_at
               ) VALUES(?,?,'artifact',1,'queued',?,'{}',?,?)""",
            (release_unit_id, capability_run_id, "a" * 64, now, now),
        )
    return capability_run_id, release_unit_id


def test_schema_v24_adds_separate_adaptation_control_plane(tmp_path: Path) -> None:
    db = tmp_path / "schema.sqlite"
    init_db(db)
    with sqlite3.connect(db) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 26
    assert {
        "parent_memories",
        "delivery_outbox_parent_memories",
        "family_recipe_versions",
        "thread_compactions",
        "background_runs",
        "background_delivery_outbox",
    } <= tables


_V24_TABLES = (
    "background_delivery_attempts",
    "background_delivery_outbox",
    "background_run_waiters",
    "background_run_events",
    "background_run_attempts",
    "background_overlap_receipts",
    "background_runs",
    "thread_compactions",
    "capability_run_recipes",
    "capability_run_parent_memories",
    "capability_run_legacy_preferences",
    "family_recipe_events",
    "family_recipe_evaluations",
    "family_recipe_versions",
    "family_recipes",
    "delivery_outbox_parent_memories",
    "parent_memory_tombstones",
    "parent_memory_events",
    "parent_memories",
    "parent_memory_settings",
)


def _downgrade_to_v23(db: Path) -> None:
    init_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        for table in _V24_TABLES:
            conn.execute(f'DROP TABLE "{table}"')
        conn.execute("DELETE FROM schema_migrations WHERE version=24")
        conn.execute("PRAGMA user_version=23")


def test_schema_v24_migrates_v23_with_backup_and_required_fencing_shape(
    tmp_path: Path,
) -> None:
    db = tmp_path / "v23.sqlite"
    _downgrade_to_v23(db)
    backup = init_db(db)
    assert backup is not None and backup.exists()
    with connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 26
        run_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(background_runs)")
        }
        delivery_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(background_delivery_outbox)")
        }
        assert {"request_hash", "active_attempt_id", "lease_token_hash"} <= run_columns
        assert {"request_hash", "active_attempt_id", "lease_token_hash"} <= delivery_columns
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_schema_v24_rejects_a_malformed_partial_table(tmp_path: Path) -> None:
    db = tmp_path / "partial-v24.sqlite"
    _downgrade_to_v23(db)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE background_run_events(dummy TEXT)")
    with pytest.raises(RuntimeError, match="missing_columns"):
        init_db(db)


def test_schema_v24_rejects_full_column_table_with_forged_constraint_shape(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference-v24.sqlite"
    init_db(reference)
    db = tmp_path / "forged-v24.sqlite"
    _downgrade_to_v23(db)
    with sqlite3.connect(db) as conn:
        conn.execute("ATTACH DATABASE ? AS reference", (str(reference),))
        conn.execute(
            "CREATE TABLE background_runs AS SELECT * FROM reference.background_runs WHERE 0"
        )
        conn.commit()
        conn.execute("DETACH DATABASE reference")
    with pytest.raises(
        RuntimeError,
        match="missing_foreign_keys|invalid_primary_keys|missing_unique_constraints|missing_check_fragments",
    ):
        init_db(db)


def test_parent_memory_is_explicit_reversible_and_never_child_evidence(tmp_path: Path) -> None:
    db = tmp_path / "memory.sqlite"
    session_id, source_message_id = _base(db)
    ledger = ParentMemoryLedger(db)
    saved = ledger.remember(
        scope_type="thread",
        scope_id=session_id,
        category="answer_style",
        value="Use one short paragraph.",
        provenance="explicit parent message",
        reason="Parent asked for this future working style.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    assert saved["confirmation"] == "Saved parent memory."
    recalled = ledger.recall(session_id=session_id)
    assert recalled[0]["memory_key"] == saved["memory_key"]
    assert recalled[0]["why_recalled"].startswith("Explicit thread answer style")

    public_event = Event(
        id="evt-memory-context",
        type="child_question",
        child_id="child-a",
        text="Why do leaves have veins?",
        source="synthetic_parent",
        metadata={
            "parent_memory_refs": [
                {"memory_key": saved["memory_key"], "version": saved["version"]}
            ]
        },
    )
    # Thread memory is rejected without code-owned session lineage; a caller
    # cannot smuggle another thread's memory through event metadata.
    with pytest.raises(ValueError, match="does not belong"):
        build_context(db, "child-a", public_event.model_dump(mode="json"))

    household = ledger.remember(
        scope_type="household",
        scope_id="default",
        category="visual_style",
        value="Prefer compact citrus diagrams.",
        provenance="explicit parent setup",
        reason="Parent selected a household presentation preference.",
        parent_id="parent-a",
    )
    public_event = public_event.model_copy(
        update={
            "metadata": {
                "parent_memory_refs": [
                    {"memory_key": household["memory_key"], "version": household["version"]}
                ]
            }
        }
    )
    context = build_context(db, "child-a", public_event.model_dump(mode="json"))
    assert context["parent_working_preferences"][0]["value"] == "Prefer compact citrus diagrams."
    Repository(db).submit_event(public_event)
    with connect(db) as conn:
        assert "Prefer compact citrus diagrams." not in conn.execute(
            "SELECT metadata_json FROM events WHERE id=?", (public_event.id,)
        ).fetchone()[0]
        assert "Prefer compact citrus diagrams." not in conn.execute(
            "SELECT metadata_json FROM evidence WHERE event_id=?", (public_event.id,)
        ).fetchone()[0]
    ledger.forget(
        str(household["memory_key"]),
        parent_id="parent-a",
        reason="Remove the household presentation preference.",
    )
    assert "Prefer compact citrus diagrams." not in jdump(ledger.show())

    updated = ledger.update(
        str(saved["memory_key"]),
        value="Use two short bullets.",
        parent_id="parent-a",
        reason="Parent refined the saved format.",
    )
    assert ledger.recall(session_id=session_id)[0]["value"] == "Use two short bullets."
    ledger.undo(
        str(updated["undo_event_id"]),
        parent_id="parent-a",
        reason="Undo the most recent save.",
    )
    assert ledger.recall(session_id=session_id)[0]["value"] == "Use one short paragraph."

    ledger.forget(
        str(saved["memory_key"]),
        parent_id="parent-a",
        reason="Parent asked the engine to forget this preference.",
    )
    assert ledger.recall(session_id=session_id) == []
    with connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM parent_memories WHERE memory_key=? AND value!=''",
            (saved["memory_key"],),
        ).fetchone()[0] == 0
        for table in ("evidence", "evidence_records", "claims", "nodes", "observations"):
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            assert "Use one short paragraph." not in jdump([dict(row) for row in rows])
            assert "Use two short bullets." not in jdump([dict(row) for row in rows])


def test_forgotten_parent_memory_purges_backups_and_cannot_restore(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    private = repo / "private"
    db = private / "data" / "curiosity.db"
    output = private / "output"
    output.mkdir(parents=True)
    _base(db)
    conversation_ref = sha256(b"team:channel").hexdigest()[:20]
    thread_ref = sha256(b"team:channel:memory-outbox").hexdigest()[:20]
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id="binding-a",
        conversation_ref=conversation_ref,
        thread_ref=thread_ref,
        child_id="child-a",
    )
    session_id = str(session["id"])
    source_message_id = SessionStore(db).append_message(
        session_id,
        role="user",
        content="Please use the private comet phrase in each answer.",
    )
    ledger = ParentMemoryLedger(db)
    saved = ledger.remember(
        scope_type="thread",
        scope_id=session_id,
        category="answer_style",
        value="Use the private comet phrase in each answer.",
        provenance="explicit parent message",
        reason="Parent requested a temporary answer style.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    transport = SlackTransport(db, output, service=CuriosityService(db, output))
    inbound = InboundMessage(
        external_event_id="evt-memory-inspection-outbox",
        team_id="team",
        user_id="user",
        channel_id="channel",
        thread_id="memory-outbox",
        text="What do you remember?",
    )
    handled = transport.handle(inbound)
    assert handled.status == "completed"
    assert "private comet phrase" in handled.message
    with connect(db) as conn:
        delivery = conn.execute(
            "SELECT id,status,payload_json FROM delivery_outbox WHERE id=?",
            (handled.outbound_id,),
        ).fetchone()
        lineage = conn.execute(
            """SELECT memory_key,memory_version,content_hash
               FROM delivery_outbox_parent_memories WHERE delivery_id=?""",
            (handled.outbound_id,),
        ).fetchone()
        inspection = conn.execute(
            """SELECT content FROM agent_messages
               WHERE kind='parent_memory_inspection' ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
    assert delivery["status"] == "queued"
    assert "private comet phrase" in str(delivery["payload_json"])
    assert tuple(lineage) == (
        saved["memory_key"],
        saved["version"],
        sha256(b"Use the private comet phrase in each answer.").hexdigest(),
    )
    assert inspection["content"] == REDACTED_PARENT_MEMORY_INSPECTION
    backup_root = tmp_path / "family-backups"
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    stale_snapshot = backup_root / str(snapshot["snapshot"])
    assert stale_snapshot.exists()

    class RecordingSlackClient:
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
            self.messages.append(kwargs)
            return {"ts": "100.1"}

    client = RecordingSlackClient()
    assert flush_slack_outbox(client, db) == [
        {"delivery_id": handled.outbound_id, "status": "sent"}
    ]
    assert "private comet phrase" in client.messages[0]["text"]
    with connect(db) as conn:
        sent = conn.execute(
            "SELECT status,payload_json FROM delivery_outbox WHERE id=?",
            (handled.outbound_id,),
        ).fetchone()
    assert sent["status"] == "sent"
    assert REDACTED_PARENT_MEMORY_INSPECTION in sent["payload_json"]
    assert "private comet phrase" not in sent["payload_json"]

    ledger.forget(
        str(saved["memory_key"]),
        parent_id="parent-a",
        reason="Parent removed the temporary answer style.",
    )
    assert not stale_snapshot.exists()
    record_hash = event_id_hash(f"{saved['memory_key']}\0{saved['version']}")
    assert record_hash in backup_root_deletion_manifest(backup_root)["parent_memories"]
    with connect(db) as conn:
        assert conn.execute(
            """SELECT content FROM agent_messages
               WHERE kind='parent_memory_inspection'"""
        ).fetchone()[0] == REDACTED_PARENT_MEMORY_INSPECTION

    clean = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    restored = tmp_path / "restored-private"
    restore_snapshot(
        str(clean["snapshot"]),
        target_private=restored,
        backup_root=backup_root,
        repo_root=repo,
    )
    with connect(restored / "data" / "curiosity.db") as conn:
        row = conn.execute(
            "SELECT status,value,provenance,why_saved FROM parent_memories WHERE memory_key=?",
            (saved["memory_key"],),
        ).fetchone()
        restored_messages = [
            dict(item)
            for item in conn.execute(
                "SELECT content,metadata_json FROM agent_messages"
            ).fetchall()
        ]
        restored_deliveries = [
            dict(item)
            for item in conn.execute(
                "SELECT id,status,payload_json FROM delivery_outbox"
            ).fetchall()
        ]
    assert tuple(row) == ("forgotten", "", "forgotten", "forgotten")
    assert "Use the private comet phrase" not in jdump(restored_messages)
    assert "Use the private comet phrase" not in jdump(restored_deliveries)

    restored_db = restored / "data" / "curiosity.db"
    with connect(restored_db) as conn:
        conn.execute(
            "UPDATE delivery_outbox SET payload_json=? WHERE id=?",
            (
                OutboundMessage(
                    channel_id="channel",
                    thread_id="memory-outbox",
                    text="Use the private comet phrase in each answer.",
                ).model_dump_json(),
                handled.outbound_id,
            ),
        )
    assert "parent_memory_delivery" in database_deletion_conflicts(
        restored_db,
        backup_root_deletion_manifest(backup_root),
    )


def test_restored_memory_delivery_revalidates_expiry_and_disable_before_send(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    private = repo / "private"
    db = private / "data" / "curiosity.db"
    output = private / "output"
    output.mkdir(parents=True)
    session_id, source_message_id = _base(db)
    ledger = ParentMemoryLedger(db)
    expiring = ledger.remember(
        scope_type="thread",
        scope_id=session_id,
        category="answer_style",
        value="Expiring inspection sentinel.",
        provenance="explicit parent message",
        reason="Parent set a temporary answer style.",
        parent_id="parent-a",
        source_message_id=source_message_id,
        expires_at="2099-01-01T00:00:00Z",
    )
    disableable = ledger.remember(
        scope_type="household",
        scope_id="default",
        category="visual_style",
        value="Disabled inspection sentinel.",
        provenance="explicit parent message",
        reason="Parent set a household visual style.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    expiring_delivery_id = enqueue_delivery(
        db,
        "binding-a",
        OutboundMessage(
            channel_id="channel",
            text=PARENT_MEMORY_INSPECTION_PREFIX + "- Answer Style: Expiring inspection sentinel.",
        ),
        idempotency_key="slack:restored-expiring:inspection",
        parent_memory_refs=[
            {"memory_key": expiring["memory_key"], "version": expiring["version"]}
        ],
    )
    disableable_delivery_id = enqueue_delivery(
        db,
        "binding-a",
        OutboundMessage(
            channel_id="channel",
            text=PARENT_MEMORY_INSPECTION_PREFIX + "- Visual Style: Disabled inspection sentinel.",
        ),
        idempotency_key="slack:restored-disabled:inspection",
        parent_memory_refs=[
            {"memory_key": disableable["memory_key"], "version": disableable["version"]}
        ],
    )
    backup_root = tmp_path / "family-backups"
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    restored = tmp_path / "restored-private"
    restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=restored,
        backup_root=backup_root,
        repo_root=repo,
    )
    restored_db = restored / "data" / "curiosity.db"
    with connect(restored_db) as conn:
        conn.execute(
            "UPDATE parent_memories SET expires_at='2000-01-01T00:00:00+00:00' WHERE memory_key=?",
            (expiring["memory_key"],),
        )

    ready = ready_deliveries(restored_db)
    assert [row["id"] for row in ready] == [disableable_delivery_id]
    with connect(restored_db) as conn:
        expired = conn.execute(
            "SELECT status,payload_json FROM delivery_outbox WHERE id=?",
            (expiring_delivery_id,),
        ).fetchone()
        # Simulate a scope-disable transaction committing after readiness but
        # before the worker's atomic claim.
        conn.execute(
            """INSERT INTO parent_memory_settings(
                   scope_type,scope_id,enabled,updated_by_parent_id,updated_at
               ) VALUES('household','default',0,'parent-a',?)
               ON CONFLICT(scope_type,scope_id) DO UPDATE SET
                 enabled=0,updated_by_parent_id='parent-a',updated_at=excluded.updated_at""",
            (utcnow(),),
        )
    assert expired["status"] == "expired"
    assert REDACTED_PARENT_MEMORY_INSPECTION in expired["payload_json"]
    assert "Expiring inspection sentinel" not in expired["payload_json"]
    assert claim_delivery(restored_db, disableable_delivery_id) is False
    with connect(restored_db) as conn:
        disabled = conn.execute(
            "SELECT status,payload_json FROM delivery_outbox WHERE id=?",
            (disableable_delivery_id,),
        ).fetchone()
    assert disabled["status"] == "expired"
    assert REDACTED_PARENT_MEMORY_INSPECTION in disabled["payload_json"]
    assert "Disabled inspection sentinel" not in disabled["payload_json"]


def test_legacy_memory_cleanup_never_matches_unrelated_delivery_text(tmp_path: Path) -> None:
    db = tmp_path / "legacy-outbox-cleanup.sqlite"
    _base(db)
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id="binding-a",
        conversation_ref=sha256(b"team:channel").hexdigest()[:20],
        thread_ref=sha256(b"team:channel:legacy-thread").hexdigest()[:20],
        child_id="child-a",
    )
    session_id = str(session["id"])
    source_message_id = SessionStore(db).append_message(
        session_id,
        role="user",
        content="Remember a.",
    )
    saved = ParentMemoryLedger(db).remember(
        scope_type="thread",
        scope_id=session_id,
        category="answer_style",
        value="a",
        provenance="explicit parent message",
        reason="Parent set a one-character legacy preference.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    SessionStore(db).append_message(
        session_id,
        role="assistant",
        content=PARENT_MEMORY_INSPECTION_PREFIX + "- Answer Style: a (thread)",
        kind="parent_memory_inspection",
        metadata={
            "memory_refs": [
                {"memory_key": saved["memory_key"], "version": saved["version"]}
            ]
        },
    )
    legacy_inspection_id = enqueue_delivery(
        db,
        "binding-a",
        OutboundMessage(
            channel_id="channel",
            thread_id="legacy-thread",
            text=PARENT_MEMORY_INSPECTION_PREFIX + "- Answer Style: a (thread)",
        ),
        idempotency_key="slack:legacy-memory-event:parent-chat",
    )
    SessionStore(db).append_message(
        session_id,
        role="user",
        content="Now ask an ordinary question.",
    )
    unrelated_id = enqueue_delivery(
        db,
        "binding-a",
        OutboundMessage(
            channel_id="channel",
            thread_id="legacy-thread",
            text=PARENT_MEMORY_INSPECTION_PREFIX + "- Ordinary model echo: a",
        ),
        idempotency_key="slack:ordinary-answer",
    )

    ParentMemoryLedger(db).forget(
        str(saved["memory_key"]),
        parent_id="parent-a",
        reason="Parent erased the one-character legacy preference.",
    )
    with connect(db) as conn:
        legacy = conn.execute(
            "SELECT status,payload_json FROM delivery_outbox WHERE id=?",
            (legacy_inspection_id,),
        ).fetchone()
        unrelated = conn.execute(
            "SELECT status,payload_json FROM delivery_outbox WHERE id=?",
            (unrelated_id,),
        ).fetchone()
    assert legacy["status"] == "expired"
    assert REDACTED_PARENT_MEMORY_INSPECTION in legacy["payload_json"]
    assert unrelated["status"] == "queued"
    assert "Ordinary model echo: a" in unrelated["payload_json"]


def test_memory_delivery_is_fenced_when_binding_is_revoked_after_readiness(
    tmp_path: Path,
) -> None:
    db = tmp_path / "revoked-memory-delivery.sqlite"
    session_id, source_message_id = _base(db)
    memory = ParentMemoryLedger(db).remember(
        scope_type="thread",
        scope_id=session_id,
        category="answer_style",
        value="Revoked delivery sentinel.",
        provenance="explicit parent message",
        reason="Parent set a temporary answer style.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    delivery_id = enqueue_delivery(
        db,
        "binding-a",
        OutboundMessage(
            channel_id="channel",
            text=PARENT_MEMORY_INSPECTION_PREFIX + "- Answer Style: Revoked delivery sentinel.",
        ),
        idempotency_key="slack:ready-then-revoke:inspection",
        parent_memory_refs=[
            {"memory_key": memory["memory_key"], "version": memory["version"]}
        ],
    )
    assert [row["id"] for row in ready_deliveries(db)] == [delivery_id]

    revoke_binding(db, "binding-a")
    assert claim_delivery(db, delivery_id) is False
    with connect(db) as conn:
        delivery = conn.execute(
            "SELECT status,payload_json FROM delivery_outbox WHERE id=?",
            (delivery_id,),
        ).fetchone()
    assert delivery["status"] == "expired"
    assert REDACTED_PARENT_MEMORY_INSPECTION in delivery["payload_json"]
    assert "Revoked delivery sentinel" not in delivery["payload_json"]
    with pytest.raises(ValueError, match="binding is unavailable"):
        enqueue_delivery(
            db,
            "binding-a",
            OutboundMessage(channel_id="channel", text="Must not queue."),
            idempotency_key="slack:revoked:new",
        )


def test_slack_drops_memory_inspection_when_binding_changes_midturn(tmp_path: Path) -> None:
    db = tmp_path / "midturn-revocation.sqlite"
    _base(db)
    thread_id = "midturn-revocation"
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id="binding-a",
        conversation_ref=sha256(b"team:channel").hexdigest()[:20],
        thread_ref=sha256(f"team:channel:{thread_id}".encode()).hexdigest()[:20],
        child_id="child-a",
    )
    source_message_id = SessionStore(db).append_message(
        str(session["id"]),
        role="user",
        content="Remember the revocation-race preference.",
    )
    memory = ParentMemoryLedger(db).remember(
        scope_type="thread",
        scope_id=str(session["id"]),
        category="answer_style",
        value="Midturn revocation sentinel.",
        provenance="explicit parent message",
        reason="Parent set a temporary answer style.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )

    class RevokingService:
        def chat(self, **_kwargs: Any) -> dict[str, Any]:
            revoke_binding(db, "binding-a")
            return {
                "status": "completed",
                "message": PARENT_MEMORY_INSPECTION_PREFIX
                + "- Answer Style: Midturn revocation sentinel.",
                "session_id": str(session["id"]),
                "transient_delivery": {
                    "kind": "parent_memory_inspection",
                    "memory_refs": [
                        {
                            "memory_key": memory["memory_key"],
                            "version": memory["version"],
                        }
                    ],
                },
            }

    handled = SlackTransport(db, tmp_path / "output", service=RevokingService()).handle(
        InboundMessage(
            external_event_id="evt-midturn-revocation",
            team_id="team",
            user_id="user",
            channel_id="channel",
            thread_id=thread_id,
            text="What do you remember?",
        )
    )
    assert handled.status == "rejected"
    assert handled.outbound_id is None
    assert "Midturn revocation sentinel" not in handled.message
    with connect(db) as conn:
        payloads = [str(row[0]) for row in conn.execute("SELECT payload_json FROM delivery_outbox")]
    assert "Midturn revocation sentinel" not in jdump(payloads)


def test_parent_memory_deletion_recovers_after_post_commit_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    private = repo / "private"
    db = private / "data" / "curiosity.db"
    output = private / "output"
    output.mkdir(parents=True)
    session_id, source_message_id = _base(db)
    saved = ParentMemoryLedger(db).remember(
        scope_type="thread",
        scope_id=session_id,
        category="answer_style",
        value="Use the private recovery phrase.",
        provenance="explicit parent message",
        reason="Parent requested a temporary recovery test style.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    backup_root = tmp_path / "family-backups"
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    snapshot_path = backup_root / str(snapshot["snapshot"])
    monkeypatch.setattr(
        "curiosity_engine.adaptation.reconcile_database_deletions",
        lambda _db: (_ for _ in ()).throw(RuntimeError("synthetic post-commit interruption")),
    )
    with pytest.raises(RuntimeError, match="post-commit interruption"):
        ParentMemoryLedger(db).forget(
            str(saved["memory_key"]),
            parent_id="parent-a",
            reason="Parent removed the temporary recovery style.",
        )
    assert snapshot_path.exists()

    init_db(db)
    assert not snapshot_path.exists()
    assert ParentMemoryLedger(db).recall(session_id=session_id) == []


def test_legacy_preference_sources_are_masked_and_cleared_values_are_scrubbed(
    tmp_path: Path,
) -> None:
    db = tmp_path / "legacy-preference.sqlite"
    session_id, source_message_id = _base(db)
    store = SessionStore(db)
    private_value = "Always use the private nebula phrase."
    store.update_thread_preference(
        session_id,
        operation="set",
        category="answer_style",
        value=private_value,
        source_message_id=source_message_id,
    )
    store.append_message(
        session_id,
        role="assistant",
        content=f"Saved: {private_value}",
        kind="preference_confirmation",
    )
    assert private_value not in jdump(store.evidence_safe_history(session_id, limit=20))

    clear_message_id = store.append_message(
        session_id,
        role="user",
        content="Clear that answer style.",
    )
    store.update_thread_preference(
        session_id,
        operation="clear",
        category="answer_style",
        value=None,
        source_message_id=clear_message_id,
    )
    with connect(db) as conn:
        values = conn.execute(
            """SELECT value FROM agent_session_preference_events
               WHERE session_id=? AND category='answer_style' AND operation='set'""",
            (session_id,),
        ).fetchall()
    assert values and all(row["value"] is None for row in values)
    assert store.active_preferences(session_id) == []


def test_natural_memory_inspection_reads_the_current_ledger_without_persisting_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory-inspection.sqlite"
    _unused_session, _unused_source = _base(db)
    store = SessionStore(db)
    session = store.get_or_create(
        origin="slack",
        transport="slack",
        binding_id="binding-a",
        conversation_ref=sha256(b"team:channel").hexdigest()[:20],
        thread_ref=sha256(b"team:channel:memory-thread").hexdigest()[:20],
        child_id="child-a",
    )
    source_message_id = store.append_message(
        str(session["id"]),
        role="user",
        content="Remember to use one short paragraph.",
    )
    saved = ParentMemoryLedger(db).remember(
        scope_type="thread",
        scope_id=str(session["id"]),
        category="delivery_preference",
        value="Private marigold delivery sentinel.",
        provenance="explicit parent message",
        reason="Parent requested a compact answer style.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    service = CuriosityService(db, tmp_path / "output")
    result = service.chat(
        binding_id="binding-a",
        team_id="team",
        channel_id="channel",
        thread_id="memory-thread",
        child_id="child-a",
        text="What do you remember?",
    )
    assert "Private marigold delivery sentinel" in result["message"]
    assert result["memory_refs"]
    assert result["memory_refs"][0]["memory_key"] == saved["memory_key"]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_turns").fetchone()[0] == 0
        stored_inspection = conn.execute(
            """SELECT kind,content,metadata_json FROM agent_messages
               ORDER BY ordinal DESC LIMIT 1"""
        ).fetchone()
        assert stored_inspection["kind"] == "parent_memory_inspection"
        assert stored_inspection["content"] == REDACTED_PARENT_MEMORY_INSPECTION
        assert "Private marigold delivery sentinel" not in jdump(dict(stored_inspection))
    safe_history = store.evidence_safe_history(str(session["id"]), limit=20)
    assert "Private marigold delivery sentinel" not in jdump(safe_history)

    class InspectionBoundaryBackend(StubBackend):
        name = "inspection-boundary"
        model = "inspection-boundary-v1"

        def __init__(self) -> None:
            self.child_calls: list[dict[str, Any]] = []

        def complete(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs["response_model"] is ParentAgentTurn:
                return {
                    "tool_calls": [
                        {
                            "name": "continue_learning_thread",
                            "arguments": {
                                "message": kwargs["payload"]["message"],
                                "mode": "quick",
                            },
                            "rationale": "The parent asked a direct child-learning question.",
                        }
                    ],
                    "done": True,
                }
            self.child_calls.append(
                {
                    "system": kwargs["system"],
                    "payload": kwargs["payload"],
                    "response_model": kwargs["response_model"].__name__,
                }
            )
            return super().complete(**kwargs)

    backend = InspectionBoundaryBackend()
    monkeypatch.setattr(
        "curiosity_engine.service.configured_backend",
        lambda *_args, **_kwargs: backend,
    )
    monkeypatch.setattr(
        "curiosity_engine.runtime.configured_backend",
        lambda *_args, **_kwargs: backend,
    )
    followup = service.chat(
        binding_id="binding-a",
        team_id="team",
        channel_id="channel",
        thread_id="memory-thread",
        child_id="child-a",
        text="Why is the sky blue?",
        answer_mode_hint="quick",
    )
    assert followup["status"] == "completed"
    assert backend.child_calls
    assert "Private marigold delivery sentinel" not in jdump(backend.child_calls)
    with connect(db) as conn:
        event_metadata = conn.execute(
            "SELECT metadata_json FROM events WHERE id=?",
            (followup["event_id"],),
        ).fetchone()[0]
        evidence_plane = {
            table: [dict(row) for row in conn.execute(f"SELECT * FROM {table}").fetchall()]
            for table in ("evidence", "evidence_records", "observations", "claims")
        }
    assert "Private marigold delivery sentinel" not in event_metadata
    assert "Private marigold delivery sentinel" not in jdump(evidence_plane)


def test_memory_off_is_immediate_and_agent_cannot_save_child_descriptions(tmp_path: Path) -> None:
    db = tmp_path / "memory-off.sqlite"
    session_id, source_message_id = _base(db)
    ledger = ParentMemoryLedger(db)
    saved = ledger.remember(
        scope_type="household",
        scope_id="default",
        category="visual_style",
        value="Prefer diagrams to decorative art.",
        provenance="explicit parent setup",
        reason="Parent chose a household visual preference.",
        parent_id="parent-a",
    )
    ledger.set_enabled(
        scope_type="household",
        scope_id="default",
        enabled=False,
        parent_id="parent-a",
    )
    assert ledger.recall(session_id=session_id) == []
    with pytest.raises(ValueError, match="disabled"):
        ledger.remember(
            scope_type="thread",
            scope_id=session_id,
            category="answer_style",
            value="Another preference.",
            provenance="explicit parent message",
            reason="Should be blocked while memory is off.",
            parent_id="parent-a",
            source_message_id=source_message_id,
        )
    with pytest.raises(ValueError, match="sensitive or child-specific"):
        ledger.remember(
            scope_type="child_support",
            scope_id="child-a",
            child_id="child-a",
            category="accessibility_support",
            value="A durable child description.",
            provenance="agent inference",
            reason="An agent tried to infer durable support.",
            creator_type="agent_proposal",
            sensitivity="sensitive",
        )
    assert saved["memory_key"]

    add_child(db, "child-b", "Other Synthetic Child", 2019, "2")
    with pytest.raises(ValueError, match="does not belong"):
        ledger.recall(session_id=session_id, child_id="child-b")
    ledger.set_enabled(
        scope_type="household",
        scope_id="default",
        enabled=True,
        parent_id="parent-a",
    )
    other_session = SessionStore(db).get_or_create(
        origin="cli",
        conversation_ref="other-conversation",
        thread_ref="other-thread",
        child_id="child-b",
    )
    other_source = SessionStore(db).append_message(
        str(other_session["id"]), role="user", content="A source from the other child."
    )
    with pytest.raises(ValueError, match="another child"):
        ledger.remember(
            scope_type="child_support",
            scope_id="child-a",
            child_id="child-a",
            category="accessibility_support",
            value="Parent-authored read-aloud support.",
            provenance="explicit parent message",
            reason="Parent requested an accessibility support.",
            parent_id="parent-a",
            source_message_id=other_source,
        )


def test_parent_authored_memory_outranks_reviewed_agent_proposal(tmp_path: Path) -> None:
    db = tmp_path / "memory-priority.sqlite"
    session_id, _source_message_id = _base(db)
    ledger = ParentMemoryLedger(db)
    proposed = ledger.remember(
        scope_type="household",
        scope_id="default",
        category="answer_style",
        value="Agent-proposed style.",
        provenance="repeated non-sensitive feedback",
        reason="Draft for parent review only.",
        creator_type="agent_proposal",
    )
    assert ledger.recall(session_id=session_id) == []
    ledger.approve(
        str(proposed["memory_key"]),
        parent_id="parent-a",
        reason="Parent approved this proposal.",
    )
    authored = ledger.remember(
        scope_type="thread",
        scope_id=session_id,
        category="answer_style",
        value="Parent-authored style.",
        provenance="explicit parent message",
        reason="Parent explicitly set a thread rule.",
        parent_id="parent-a",
    )
    recalled = ledger.recall(session_id=session_id)
    assert recalled[0]["memory_key"] == authored["memory_key"]
    assert recalled[1]["memory_key"] == proposed["memory_key"]


class _RecipeBehaviorBackend:
    name = "synthetic-recipe-behavior"
    model = "behavior-v1"

    def complete(
        self,
        *,
        role: str,
        system: str,
        payload: dict[str, Any],
        response_model: type[Any],
    ) -> dict[str, Any]:
        assert role == "reasoning"
        artifact_type = str(response_model.model_fields["artifact_type"].default)
        spec = LearningArtifactService._fallback_spec(
            artifact_type,
            str(payload["question"]),
            dict(payload["reviewed_response"]),
            str(payload["target_grade"]),
            str(payload["source_event_id"]),
            revision=str(payload["parent_revision"]),
        )
        if "<RUNTIME_MODIFIERS>" in system and "Do nothing different." not in system:
            spec = spec.model_copy(
                update={
                    "accessibility_text": (
                        "A compact child-ready page with generous writing space and a drawing choice."
                    )
                }
            )
        return spec.model_dump(mode="json")


class _RecipeJudgeBackend:
    name = "synthetic-recipe-judge"
    model = "judge-v1"

    def complete(
        self,
        *,
        role: str,
        system: str,
        payload: dict[str, Any],
        response_model: type[Any],
    ) -> dict[str, Any]:
        assert role == "judge" and response_model is RecipeCaseVerdict
        instructions = str(payload["candidate_recipe"]["instructions"])
        unsafe = "Shame the child" in instructions
        changed = payload["baseline"] != payload["candidate"]
        passed = changed and not unsafe
        return {
            "case_id": str(payload["case"]["id"]),
            "verdict": "pass" if passed else "fail",
            "candidate_effect_observed": changed,
            "safety_regression": unsafe,
            "authority_expansion": False,
            "reasons": ["synthetic independent comparison"],
        }


class _RuntimeArtifactBackend:
    name = "runtime-artifact"
    model = "runtime-v1"

    def __init__(self, db: Path) -> None:
        self.db = db
        self.systems: list[str] = []

    def complete(
        self,
        *,
        role: str,
        system: str,
        payload: dict[str, Any],
        response_model: type[Any],
    ) -> dict[str, Any]:
        self.systems.append(system)
        if response_model is CriticResult:
            return {"verdict": "pass", "concerns": [], "required_changes": []}
        assert role == "reasoning"
        with connect(self.db) as conn:
            assert conn.execute("SELECT COUNT(*) FROM capability_run_recipes").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM release_units").fetchone()[0] == 0
        artifact_type = str(response_model.model_fields["artifact_type"].default)
        pedagogy_plan = ApprovedPedagogyPlan.model_validate(payload["approved_pedagogy_plan"])
        spec = LearningArtifactService._fallback_spec(
            artifact_type,
            str(payload["question"]),
            dict(payload["reviewed_response"]),
            str(payload["target_grade"]),
            str(payload["source_event_id"]),
            revision=str(payload["parent_revision"]),
            pedagogy_plan=pedagogy_plan,
        )
        spec = LearningArtifactService._enact_fallback_evidence(spec, pedagogy_plan)
        assert "Use a high-contrast page with extra whitespace." in system
        assert "Use generous writing space and a drawing choice." in system
        return spec.model_copy(
            update={
                "accessibility_text": (
                    "A high-contrast child-ready page with generous writing space and a drawing choice."
                )
            }
        ).model_dump(mode="json")


def _promote_recipe(
    service: FamilyRecipeService,
    *,
    recipe_id: str,
    instructions: str,
) -> int:
    draft = service.draft(
        recipe_id=recipe_id,
        capability_id="artifact_worksheet",
        title="Family worksheet shape",
        description="A private reviewed worksheet workflow.",
        instructions=instructions,
        tool_names=["create_learning_artifact"],
        parent_id="parent-a",
    )
    version = int(draft["version"])
    service.lint(recipe_id, version)
    with pytest.raises(ValueError, match="trusted passing evaluation"):
        service.record_tests(recipe_id, version, evaluation_id="recipeeval_fabricated")
    test_evaluation = service.run_evaluation(recipe_id, version, evaluation_kind="tests")
    service.record_tests(
        recipe_id,
        version,
        evaluation_id=str(test_evaluation["evaluation_id"]),
    )
    with pytest.raises(ValueError, match="trusted passing evaluation"):
        service.record_shadow(recipe_id, version, evaluation_id="recipeeval_fabricated")
    shadow_evaluation = service.run_evaluation(recipe_id, version, evaluation_kind="shadow")
    service.record_shadow(
        recipe_id,
        version,
        evaluation_id=str(shadow_evaluation["evaluation_id"]),
    )
    service.approve(recipe_id, version, actor_type="parent", parent_id="parent-a")
    return version


def test_family_recipe_requires_full_human_promotion_and_rolls_back(tmp_path: Path) -> None:
    db = tmp_path / "recipes.sqlite"
    session_id, _source_message_id = _base(db)
    recipes = FamilyRecipeService(
        db,
        behavior_backend=_RecipeBehaviorBackend(),
        judge_backend=_RecipeJudgeBackend(),
    )
    unsafe = recipes.draft(
        recipe_id="unsafe-recipe",
        capability_id="artifact_worksheet",
        title="Unsafe recipe",
        description="Synthetic unsafe recipe.",
        instructions="Run a shell command with curl https://example.test.",
        tool_names=["made_up_shell"],
    )
    with pytest.raises(ValueError, match="unreviewed tools"):
        recipes.lint("unsafe-recipe", int(unsafe["version"]))
    with pytest.raises(ValueError, match="never broaden"):
        FamilyRecipeService(db, allowed_tools={"made_up_shell"})

    harmful = recipes.draft(
        recipe_id="harmful-recipe",
        capability_id="artifact_worksheet",
        title="Harmful but tool-bounded recipe",
        description="Synthetic behavioral rejection fixture.",
        instructions="Shame the child after an incorrect prediction.",
        tool_names=["create_learning_artifact"],
    )
    harmful_version = int(harmful["version"])
    recipes.lint("harmful-recipe", harmful_version)
    harmful_evaluation = recipes.run_evaluation(
        "harmful-recipe", harmful_version, evaluation_kind="tests"
    )
    assert harmful_evaluation["status"] == "fail"
    with pytest.raises(ValueError, match="trusted passing evaluation"):
        recipes.record_tests(
            "harmful-recipe",
            harmful_version,
            evaluation_id=str(harmful_evaluation["evaluation_id"]),
        )

    version_one = _promote_recipe(
        recipes,
        recipe_id="worksheet-shape",
        instructions="Use the reviewed artifact tool and leave generous writing space.",
    )
    with connect(db) as conn:
        evaluation_id = conn.execute(
            """SELECT test_evaluation_id FROM family_recipe_versions
               WHERE recipe_id='worksheet-shape' AND version=?""",
            (version_one,),
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="definition is immutable"):
            conn.execute(
                """UPDATE family_recipe_versions SET instructions='changed after evaluation'
                   WHERE recipe_id='worksheet-shape' AND version=?""",
                (version_one,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="evaluation is immutable"):
            conn.execute(
                "UPDATE family_recipe_evaluations SET status='fail' WHERE evaluation_id=?",
                (evaluation_id,),
            )
    with pytest.raises(ValueError, match="model cannot"):
        recipes.activate(
            "worksheet-shape",
            version_one,
            actor_type="model",
            parent_id=None,
        )
    recipes.activate(
        "worksheet-shape",
        version_one,
        actor_type="parent",
        parent_id="parent-a",
    )
    version_two = _promote_recipe(
        recipes,
        recipe_id="worksheet-shape",
        instructions="Use the reviewed artifact tool, a compact prompt, and generous writing space.",
    )
    recipes.activate(
        "worksheet-shape",
        version_two,
        actor_type="parent",
        parent_id="parent-a",
    )
    rolled_back = recipes.rollback(
        "worksheet-shape",
        actor_type="parent",
        parent_id="parent-a",
    )
    assert rolled_back["version"] == version_one
    assert recipes.active("worksheet-shape")["version"] == version_one

    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO capability_runs(
                   id,session_id,capability_id,capability_version,skill_versions_json,status,
                   result_json,created_at,updated_at
               ) VALUES(?,?,?,?,?,'running','{}',?,?)""",
            ("caprun-recipe", session_id, "artifact_worksheet", "2", "{}", now, now),
        )
    bound = recipes.bind_to_capability_run("caprun-recipe", "worksheet-shape")
    assert bound["recipe_version"] == version_one
    assert len(str(bound["instructions_hash"])) == 64
    background = BackgroundRunService(db)
    assert background.create(
        run_kind="recipe_render",
        idempotency_key="recipe-bound:one",
        session_id=session_id,
        capability_run_id="caprun-recipe",
        recipe_id="worksheet-shape",
        recipe_version=version_one,
        policy_snapshot={"delivery": "parent_confirmation"},
    )["status"] == "queued"
    with pytest.raises(ValueError, match="exactly bound"):
        background.create(
            run_kind="recipe_render",
            idempotency_key="recipe-unbound:one",
            session_id=session_id,
            capability_run_id="caprun-recipe",
            recipe_id="unsafe-recipe",
            recipe_version=int(unsafe["version"]),
            policy_snapshot={"delivery": "parent_confirmation"},
        )
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO capability_runs(
                   id,session_id,capability_id,capability_version,skill_versions_json,status,
                   result_json,created_at,updated_at
               ) VALUES(?,?,?,?,?,'running','{}',?,?)""",
            ("caprun-wrong", session_id, "parent_chat", "3", "{}", now, now),
        )
    with pytest.raises(ValueError, match="does not belong"):
        recipes.bind_to_capability_run("caprun-wrong", "worksheet-shape")
    with connect(db) as conn:
        conn.execute(
            "UPDATE capability_runs SET status='completed' WHERE id='caprun-recipe'"
        )
    with pytest.raises(ValueError, match="only while"):
        recipes.bind_to_capability_run("caprun-recipe", "worksheet-shape")


def test_active_recipe_and_parent_style_run_through_real_artifact_contract(
    tmp_path: Path,
) -> None:
    db = tmp_path / "recipe-runtime.sqlite"
    session_id, source_message_id = _base(db)
    recipes = FamilyRecipeService(
        db,
        behavior_backend=_RecipeBehaviorBackend(),
        judge_backend=_RecipeJudgeBackend(),
    )
    version = _promote_recipe(
        recipes,
        recipe_id="runtime-worksheet-shape",
        instructions="Use generous writing space and a drawing choice.",
    )
    recipes.activate(
        "runtime-worksheet-shape",
        version,
        actor_type="parent",
        parent_id="parent-a",
    )
    memory = ParentMemoryLedger(db).remember(
        scope_type="thread",
        scope_id=session_id,
        category="artifact_style",
        value="Use a high-contrast page with extra whitespace.",
        provenance="explicit parent message",
        reason="Parent set an artifact presentation preference.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    SessionStore(db).update_thread_preference(
        session_id,
        operation="set",
        category="visual_style",
        value="Use the legacy wide-margin preference.",
        source_message_id=source_message_id,
    )
    event_id = "evt-runtime-artifact"
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(
                   id,type,child_id,text,source,metadata_json,created_at,status
               ) VALUES(?,'context_input','child-a','Make a leaf noticing page.','test',?,?,'completed')""",
            (
                event_id,
                jdump(
                    {
                        "learning_scope": "context_only",
                        "evidence_eligible": False,
                        "input_mode": "artifact_request",
                    }
                ),
                now,
            ),
        )
    SessionStore(db).append_message(
        session_id,
        role="user",
        content="Make a leaf noticing page.",
        event_id=event_id,
    )

    backend = _RuntimeArtifactBackend(db)
    governed = CuriosityService(
        db,
        tmp_path / "output",
    )._create_governed_session_artifact(
        session_id=session_id,
        source_event_id=event_id,
        artifact_type="worksheet",
        revision="Make a leaf noticing page.",
        backend=backend,
    )
    artifact = governed["artifact"]
    assert any("<RUNTIME_MODIFIERS>" in system for system in backend.systems)
    with connect(db) as conn:
        binding = conn.execute(
            "SELECT * FROM capability_run_recipes WHERE capability_run_id=?",
            (governed["capability_run_id"],),
        ).fetchone()
        assert binding["recipe_version"] == version
        assert len(binding["instructions_hash"]) == 64
        memory_binding = conn.execute(
            "SELECT * FROM capability_run_parent_memories WHERE capability_run_id=?",
            (governed["capability_run_id"],),
        ).fetchone()
        assert memory_binding["memory_key"] == memory["memory_key"]
        assert memory_binding["memory_version"] == memory["version"]
        assert len(memory_binding["content_hash"]) == 64
        legacy_binding = conn.execute(
            "SELECT * FROM capability_run_legacy_preferences WHERE capability_run_id=?",
            (governed["capability_run_id"],),
        ).fetchone()
        assert legacy_binding is not None and len(legacy_binding["content_hash"]) == 64
        stored_spec = jload(
            conn.execute(
                "SELECT spec_json FROM artifacts WHERE id=?",
                (artifact["artifact_id"],),
            ).fetchone()[0]
        )
        assert "high-contrast" in stored_spec["accessibility_text"]
        assert conn.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0
        persisted_non_control = jdump(
            {
                "events": [dict(row) for row in conn.execute("SELECT metadata_json FROM events")],
                "evidence": [dict(row) for row in conn.execute("SELECT metadata_json FROM evidence")],
                "run": dict(
                    conn.execute(
                        "SELECT result_json FROM capability_runs WHERE id=?",
                        (governed["capability_run_id"],),
                    ).fetchone()
                ),
            }
        )
    assert "Use generous writing space and a drawing choice." not in persisted_non_control
    assert "Use a high-contrast page with extra whitespace." not in persisted_non_control


def test_automatic_activity_printables_use_governed_runs_and_fail_closed_for_stub_recipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "automatic-printable.sqlite"
    session_id, source_message_id = _base(db)
    service = CuriosityService(db, tmp_path / "output")
    output = {
        "hook": "Sauropods could reach leaves at different heights.",
        "show": "Compare the fixed leaf targets.",
        "ask": "Which leaf can each neck reach?",
        "nugget": "A longer neck can reach farther from the same start.",
        "resource_refs": [],
        "physical_extension": {
            "title": "Sauropod leaf reach",
            "instructions": [
                "Keep the page flat.",
                "Trace both necks from the same start.",
                "Compare which fixed leaves they reach.",
            ],
            "materials": ["printed page"],
            "parent_effort": "low",
            "printable": {
                "kind": "reach_comparison",
                "title": "Long-Neck Leaf Test",
                "child_directions": "Trace both necks from the same start and compare their reach.",
                "parent_setup": "Keep the whole page flat.",
                "pedagogical_value": "A fixed same-start comparison makes reach visible.",
                "pieces": [
                    {"label": "NEAR LEAF", "prompt": "2 units", "shape": "leaf"},
                    {"label": "FAR LEAF", "prompt": "6 units", "shape": "leaf"},
                    {"label": "SHORT NECK", "prompt": "3 units", "shape": "card"},
                    {"label": "LONG NECK", "prompt": "6 units", "shape": "card"},
                ],
            },
        },
        "visual": None,
    }

    def add_reviewed_source(event_id: str, *, first: bool = False) -> None:
        now = utcnow()
        with connect(db) as conn:
            conn.execute(
                """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                   VALUES(?,'child_question','child-a','Tell me about sauropods.','test','{}',?,'completed')""",
                (event_id, now),
            )
            conn.execute(
                """INSERT INTO responses(event_id,workflow,status,output_json,created_at,updated_at)
                   VALUES(?,'pull_thread','completed',?,?,?)""",
                (event_id, jdump(output), now, now),
            )
        if first:
            SessionStore(db).attach_event(source_message_id, event_id)
        else:
            SessionStore(db).append_message(
                session_id,
                role="user",
                content="Tell me about sauropods again.",
                event_id=event_id,
            )

    def fake_dispatch(_self: Any, raw_event: Event, **_kwargs: Any) -> RunResult:
        return RunResult(
            event_id=raw_event.id,
            workflow="pull_thread",
            status="completed",
            output=output,
        )

    monkeypatch.setattr("curiosity_engine.service.CuriosityHarness.dispatch", fake_dispatch)
    add_reviewed_source("evt-automatic-without-lineage")
    without_lineage = service.ask(
        child_id="child-a",
        text="Tell me about sauropods.",
        event_id="evt-automatic-without-lineage",
        enqueue_visual=False,
    )
    assert without_lineage["status"] == "completed"
    assert not without_lineage.get("artifact")
    with connect(db) as conn:
        assert conn.execute(
            """SELECT COUNT(*) FROM capability_runs
               WHERE capability_id='artifact_activity'"""
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0

    add_reviewed_source("evt-automatic-one", first=True)
    first = service.ask(
        child_id="child-a",
        text="Tell me about sauropods.",
        event_id="evt-automatic-one",
        agent_lineage={"session_id": session_id},
        enqueue_visual=False,
    )
    assert first["artifact_capability_run_id"]
    with connect(db) as conn:
        governed = conn.execute(
            """SELECT id,status FROM capability_runs
               WHERE capability_id='artifact_activity' ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
        assert governed is not None and governed["status"] == "completed"
        assert conn.execute(
            "SELECT COUNT(*) FROM release_units WHERE capability_run_id=?",
            (governed["id"],),
        ).fetchone()[0] == 1

    recipes = FamilyRecipeService(db)
    draft = recipes.draft(
        recipe_id="automatic-activity-recipe",
        capability_id="artifact_activity",
        title="Automatic activity presentation",
        description="Synthetic active recipe for deterministic fail-closed coverage.",
        instructions="Use the reviewed artifact tool and add one generous comparison area.",
        tool_names=["create_learning_artifact"],
        parent_id="parent-a",
    )
    with connect(db) as conn:
        conn.execute(
            """UPDATE family_recipe_versions
               SET status='active',activated_at=?,updated_at=?
               WHERE recipe_id=? AND version=?""",
            (utcnow(), utcnow(), draft["recipe_id"], draft["version"]),
        )
    add_reviewed_source("evt-automatic-two")
    second = service.ask(
        child_id="child-a",
        text="Tell me about sauropods again.",
        event_id="evt-automatic-two",
        agent_lineage={"session_id": session_id},
        enqueue_visual=False,
    )
    assert second["status"] == "completed"
    with connect(db) as conn:
        failed = conn.execute(
            """SELECT id,status,result_json FROM capability_runs
               WHERE capability_id='artifact_activity' ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
        assert failed["status"] == "failed"
        assert jload(failed["result_json"])["error_code"] == "active_recipe_requires_model"
        assert conn.execute(
            "SELECT COUNT(*) FROM capability_run_recipes WHERE capability_run_id=?",
            (failed["id"],),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM release_units WHERE capability_run_id=?",
            (failed["id"],),
        ).fetchone()[0] == 0


def test_recipe_binding_rejects_a_capability_run_after_output_exists(tmp_path: Path) -> None:
    db = tmp_path / "recipe-late-binding.sqlite"
    session_id, _source_message_id = _base(db)
    recipes = FamilyRecipeService(
        db,
        behavior_backend=_RecipeBehaviorBackend(),
        judge_backend=_RecipeJudgeBackend(),
    )
    version = _promote_recipe(
        recipes,
        recipe_id="late-binding",
        instructions="Use generous writing space and a drawing choice.",
    )
    recipes.activate("late-binding", version, actor_type="operator", parent_id=None)
    run_id, _release_id = _release_unit(db, session_id)
    with pytest.raises(ValueError, match="before capability execution"):
        recipes.bind_to_capability_run(run_id, "late-binding")


def test_recipe_evaluation_fixtures_work_from_an_installed_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = Path(__file__).resolve().parents[1]
    installed_root = tmp_path / "prefix" / "share" / "curiosity-engine"
    shutil.copytree(source_root / "capabilities", installed_root / "capabilities")
    shutil.copytree(
        source_root / "evals" / "recipe_promotion",
        installed_root / "evals" / "recipe_promotion",
    )
    monkeypatch.setattr(
        "curiosity_engine.adaptation.configuration_root",
        lambda: installed_root,
    )
    db = tmp_path / "installed-evaluation.sqlite"
    _base(db)
    service = FamilyRecipeService(
        db,
        behavior_backend=_RecipeBehaviorBackend(),
        judge_backend=_RecipeJudgeBackend(),
    )
    draft = service.draft(
        recipe_id="installed-evaluation",
        capability_id="artifact_worksheet",
        title="Installed evaluation",
        description="Exercises packaged public fixtures.",
        instructions="Use generous writing space and a drawing choice.",
        tool_names=["create_learning_artifact"],
        parent_id="parent-a",
    )
    service.lint("installed-evaluation", int(draft["version"]))
    result = service.run_evaluation(
        "installed-evaluation",
        int(draft["version"]),
        evaluation_kind="tests",
    )
    assert result["status"] == "pass"
    assert (
        '"share/curiosity-engine/evals/recipe_promotion"'
        in (source_root / "pyproject.toml").read_text(encoding="utf-8")
    )


def test_structured_compaction_keeps_refs_and_recent_raw_turns(tmp_path: Path) -> None:
    db = tmp_path / "compaction.sqlite"
    session_id, source_message_id = _base(db)
    ledger = ParentMemoryLedger(db)
    memory = ledger.remember(
        scope_type="thread",
        scope_id=session_id,
        category="answer_style",
        value="Keep it brief.",
        provenance="explicit parent message",
        reason="Parent wants this remembered in the thread.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    store = SessionStore(db)
    for index in range(20):
        store.append_message(
            session_id,
            role="assistant" if index % 2 else "user",
            content=f"Synthetic thread turn {index}.",
        )
    compacted = ThreadCompactor(db).context(session_id, compact_after=16, keep_recent=6)
    assert len(compacted["history"]) == 6
    summary = compacted["compaction"]["summary"]
    assert summary["version"] == "phase4-v1"
    assert summary["constraint_refs"][0]["memory_key"] == memory["memory_key"]
    assert "value" not in summary["constraint_refs"][0]
    assert {"child_traits", "learner_evidence", "artifact_contents"} == set(
        summary["prohibited_content"]
    ) - {"raw_parent_text"}
    assert "current_intent" not in summary
    assert summary["current_intent_ref"]["message_id"]
    assert compacted["resolved_current_intent"]["content"] == "Synthetic thread turn 12."
    encoded = jdump(summary)
    assert "Keep it brief" not in encoded
    assert store.compacted_context(session_id)["compaction"]["summary_hash"] == compacted[
        "compaction"
    ]["summary_hash"]

    class CaptureBackend:
        name = "capture"
        model = "capture-v1"

        def __init__(self) -> None:
            self.payload: dict[str, Any] | None = None

        def complete(self, **kwargs: Any) -> dict[str, Any]:
            self.payload = kwargs["payload"]
            return {"message": "Continuity loaded.", "done": True}

    backend = CaptureBackend()
    ParentAgentRuntime(db, backend=backend, tools=ToolRegistry()).run(
        session_id=session_id,
        user_message="Continue from here.",
        child={"id": "child-a", "name": "Synthetic Child"},
        latest_event_id=None,
        origin="cli",
    )
    assert backend.payload is not None
    workspace = backend.payload["workspace"]
    assert workspace["thread_compaction"]["version"] == "phase4-v1"
    assert workspace["compacted_current_intent"]["content"] == "Synthetic thread turn 12."
    assert workspace["parent_memories"][0]["memory_key"] == memory["memory_key"]
    ledger.set_item_enabled(
        str(memory["memory_key"]),
        enabled=False,
        parent_id="parent-a",
        reason="Stop applying this compacted preference.",
    )
    refreshed = ThreadCompactor(db).context(session_id, compact_after=1000, keep_recent=6)
    assert refreshed["runtime_summary"]["constraint_refs"] == []
    assert refreshed["compaction"]["summary"]["constraint_refs"][0]["memory_key"] == memory[
        "memory_key"
    ]
    dashboard = CuriosityService(db, tmp_path / "output").dashboard("child-a")
    assert dashboard["thread_compactions"][0]["immutable"]["summary"][
        "constraint_refs"
    ][0]["memory_key"] == memory["memory_key"]
    assert dashboard["thread_compactions"][0]["runtime_summary"]["constraint_refs"] == []
    with connect(db) as conn:
        turn_row = dict(
            conn.execute("SELECT * FROM agent_turns ORDER BY started_at DESC LIMIT 1").fetchone()
        )
        assert "Keep it brief." not in jdump(turn_row)
        latest_compaction_id = conn.execute(
            "SELECT compaction_id FROM thread_compactions ORDER BY covers_through_ordinal DESC LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            "UPDATE thread_compactions SET summary_hash=? WHERE compaction_id=?",
            ("0" * 64, latest_compaction_id),
        )
    with connect(db) as conn:
        assert conn.execute(
            "SELECT summary_hash FROM thread_compactions WHERE compaction_id=?",
            (latest_compaction_id,),
        ).fetchone()[0] == "0" * 64
    with pytest.raises(ValueError, match="hash"):
        ThreadCompactor(db).context(session_id, compact_after=1000, keep_recent=6)
    tampered_summary = {**summary, "current_intent": "copied raw text"}
    tampered_json = jdump(tampered_summary)
    with connect(db) as conn:
        conn.execute(
            """UPDATE thread_compactions SET summary_json=?,summary_hash=?
               WHERE compaction_id=?""",
            (tampered_json, sha256(tampered_json.encode()).hexdigest(), latest_compaction_id),
        )
    with pytest.raises(ValueError, match="reviewed schema"):
        ThreadCompactor(db).context(session_id, compact_after=1000, keep_recent=6)


def test_compaction_is_bounded_concurrent_and_refreshes_mutable_statuses(
    tmp_path: Path,
) -> None:
    db = tmp_path / "compaction-concurrency.sqlite"
    session_id, source_message_id = _base(db)
    ledger = ParentMemoryLedger(db)
    store = SessionStore(db)
    for index in range(30):
        ledger.remember(
            scope_type="thread",
            scope_id=session_id,
            category="answer_style",
            value=f"Synthetic bounded preference {index}.",
            provenance="explicit parent message",
            reason="Exercise the compaction reference bound.",
            parent_id="parent-a",
            source_message_id=source_message_id,
        )
        store.update_thread_preference(
            session_id,
            operation="set",
            category="answer_style",
            value=f"Legacy preference {index}.",
            source_message_id=source_message_id,
        )
    turn = store.start_turn(session_id, {"message": "Synthetic work."}, policy_hash="policy-v1")
    tool_call_id = store.record_tool_call(
        session_id=session_id,
        turn_id=str(turn["id"]),
        ordinal=1,
        tool_name="search_thread_outputs",
        tool_version="1",
        arguments={"query": "synthetic"},
        risk="read_only",
        decision="allowed",
        status="running",
    )
    _capability_run_id, release_unit_id = _release_unit(db, session_id)
    for index in range(20):
        store.append_message(
            session_id,
            role="assistant" if index % 2 else "user",
            content=f"Concurrent compaction turn {index}.",
        )

    compactor = ThreadCompactor(db)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: compactor.compact(session_id), range(2)))
    assert results[0] is not None and results[1] is not None
    assert results[0]["compaction_id"] == results[1]["compaction_id"]
    summary = results[0]["summary"]
    assert len(summary["constraint_refs"]) == 24
    assert len(summary["legacy_preference_event_ids"]) == 24

    with connect(db) as conn:
        conn.execute(
            "UPDATE tool_calls SET status='failed',updated_at=? WHERE id=?",
            (utcnow(), tool_call_id),
        )
        conn.execute(
            "UPDATE release_units SET status='delivered',updated_at=? WHERE id=?",
            (utcnow(), release_unit_id),
        )
    refreshed = compactor.context(session_id, compact_after=1000, keep_recent=6)
    assert refreshed["compaction"]["summary"]["decisions"][0]["status"] == "running"
    assert refreshed["runtime_summary"]["decisions"][0]["status"] == "failed"
    assert refreshed["runtime_summary"]["errors_and_fixes"][0]["tool_call_id"] == tool_call_id
    assert refreshed["compaction"]["summary"]["output_pointers"][0]["status"] == "queued"
    assert refreshed["runtime_summary"]["output_pointers"][0]["status"] == "delivered"


def test_cli_exposes_parent_memory_and_recipe_governance() -> None:
    parser = build_parser()
    memory = parser.parse_args(["memory", "show", "--db", "/tmp/family.sqlite"])
    assert memory.cmd == "memory" and memory.memory_cmd == "show"
    recipe = parser.parse_args(
        [
            "recipe",
            "activate",
            "--db",
            "/tmp/family.sqlite",
            "--id",
            "worksheet-shape",
            "--version",
            "2",
            "--actor",
            "operator",
        ]
    )
    assert recipe.cmd == "recipe" and recipe.recipe_cmd == "activate"
    tests = parser.parse_args(
        [
            "recipe",
            "run-tests",
            "--db",
            "/tmp/family.sqlite",
            "--id",
            "worksheet-shape",
            "--version",
            "2",
        ]
    )
    assert tests.recipe_cmd == "run-tests"


def test_background_run_wakes_once_skips_overlap_and_requires_parent_delivery_confirmation(
    tmp_path: Path,
) -> None:
    db = tmp_path / "background.sqlite"
    session_id, _source_message_id = _base(db)
    capability_run_id, release_unit_id = _release_unit(db, session_id)
    store = SessionStore(db)
    turn = store.start_turn(session_id, {"message": "Make a printable."}, policy_hash="policy-v1")
    service = BackgroundRunService(db)
    created = service.create(
        run_kind="artifact_render",
        idempotency_key="artifact:one",
        overlap_key="artifact:child-a",
        session_id=session_id,
        capability_run_id=capability_run_id,
        policy_snapshot={"delivery": "parent_confirmation"},
        workspace={"workspace_id": "private-job-one"},
    )
    duplicate = service.create(
        run_kind="artifact_render",
        idempotency_key="artifact:one",
        overlap_key="artifact:child-a",
        session_id=session_id,
        capability_run_id=capability_run_id,
        policy_snapshot={"delivery": "parent_confirmation"},
        workspace={"workspace_id": "private-job-one"},
    )
    assert duplicate["duplicate"] is True and duplicate["run_id"] == created["run_id"]
    skipped = service.create(
        run_kind="artifact_render",
        idempotency_key="artifact:two",
        overlap_key="artifact:child-a",
        session_id=session_id,
        policy_snapshot={"delivery": "parent_confirmation"},
    )
    assert skipped["status"] == "skipped_overlap"
    assert skipped["run_id"] == created["run_id"]
    add_child(db, "child-b", "Other Child", 2019, "2")
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO transport_bindings VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "binding-b",
                "slack",
                "team-b",
                "user-b",
                "channel-b",
                "parent-b",
                "active",
                now,
                now,
            ),
        )
    other_session = store.get_or_create(
        origin="slack",
        transport="slack",
        binding_id="binding-b",
        conversation_ref="conversation-b",
        thread_ref="thread-b",
        child_id="child-b",
    )
    with pytest.raises(ValueError, match="different scope"):
        service.create(
            run_kind="artifact_render",
            idempotency_key="artifact:cross-child",
            overlap_key="artifact:child-a",
            session_id=str(other_session["id"]),
            policy_snapshot={"delivery": "parent_confirmation"},
        )
    with pytest.raises(ValueError, match="different request"):
        service.create(
            run_kind="artifact_render",
            idempotency_key="artifact:one",
            session_id=session_id,
            capability_run_id=capability_run_id,
            policy_snapshot={"delivery": "different_policy"},
        )
    claimed = service.claim(str(created["run_id"]), worker_id="worker-one")
    service.wait_for_completion(str(created["run_id"]), turn_id=str(turn["id"]))
    completed = service.complete(
        str(created["run_id"]),
        attempt_id=str(claimed["attempt_id"]),
        lease_token=str(claimed["lease_token"]),
        result_pointer={"release_unit_id": release_unit_id},
    )
    assert completed["waiters"] == 1
    resumed = service.claim_wake(str(created["run_id"]), turn_id=str(turn["id"]))
    assert resumed["result"] == {"release_unit_id": release_unit_id}
    with pytest.raises(ValueError, match="not ready"):
        service.claim_wake(str(created["run_id"]), turn_id=str(turn["id"]))
    assert claimed["attempt_number"] == 1
    late_turn = store.start_turn(
        session_id,
        {"message": "Attach after the render completed."},
        policy_hash="policy-v1",
    )
    late_waiter = service.wait_for_completion(
        str(created["run_id"]),
        turn_id=str(late_turn["id"]),
    )
    assert late_waiter["status"] == "ready"
    assert service.claim_wake(
        str(created["run_id"]),
        turn_id=str(late_turn["id"]),
    )["run_status"] == "completed"
    skipped_retry = service.create(
        run_kind="artifact_render",
        idempotency_key="artifact:two",
        overlap_key="artifact:child-a",
        session_id=session_id,
        policy_snapshot={"delivery": "parent_confirmation"},
    )
    assert skipped_retry == {
        "run_id": created["run_id"],
        "status": "skipped_overlap",
        "duplicate": True,
    }
    with connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM background_overlap_receipts WHERE idempotency_key='artifact:two'"
        ).fetchone()[0] == 1

    delivery = service.request_parent_delivery(
        str(created["run_id"]),
        binding_id="binding-a",
        payload_pointer={"release_unit_id": release_unit_id},
        idempotency_key="delivery:artifact-one",
    )
    delivery_duplicate = service.request_parent_delivery(
        str(created["run_id"]),
        binding_id="binding-a",
        payload_pointer={"release_unit_id": release_unit_id},
        idempotency_key="delivery:artifact-one",
    )
    assert delivery_duplicate["duplicate"] is True
    with pytest.raises(ValueError, match="different work"):
        service.request_parent_delivery(
            str(created["run_id"]),
            binding_id="binding-b",
            payload_pointer={"release_unit_id": release_unit_id},
            idempotency_key="delivery:artifact-one",
        )
    with pytest.raises(ValueError, match="completed background work"):
        service.request_parent_delivery(
            str(created["run_id"]),
            binding_id="binding-b",
            payload_pointer={"release_unit_id": release_unit_id},
            idempotency_key="delivery:wrong-binding",
        )
    with pytest.raises(ValueError, match="not confirmed"):
        service.claim_delivery(str(delivery["delivery_id"]), worker_id="delivery-worker")
    with pytest.raises(ValueError, match="does not belong"):
        service.confirm_parent_delivery(str(delivery["delivery_id"]), parent_id="parent-b")
    service.confirm_parent_delivery(str(delivery["delivery_id"]), parent_id="parent-a")
    sending = service.claim_delivery(
        str(delivery["delivery_id"]), worker_id="delivery-worker"
    )
    assert sending["audience"] == "parent"
    service.finish_delivery(
        str(delivery["delivery_id"]),
        attempt_id=str(sending["attempt_id"]),
        lease_token=str(sending["lease_token"]),
        outcome="sent",
        external_message_id="external-parent-message",
    )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM delivery_outbox").fetchone()[0] == 0
        assert conn.execute(
            "SELECT status FROM background_delivery_outbox WHERE delivery_id=?",
            (delivery["delivery_id"],),
        ).fetchone()[0] == "sent"

    failed_turn = store.start_turn(session_id, {"message": "Try another render."}, policy_hash="policy-v1")
    failed_run = service.create(
        run_kind="artifact_render",
        idempotency_key="artifact:failed",
        overlap_key="artifact:failed-child-a",
        session_id=session_id,
        policy_snapshot={"delivery": "parent_confirmation"},
    )
    failed_claim = service.claim(str(failed_run["run_id"]), worker_id="worker-one")
    service.wait_for_completion(str(failed_run["run_id"]), turn_id=str(failed_turn["id"]))
    service.fail(
        str(failed_run["run_id"]),
        attempt_id=str(failed_claim["attempt_id"]),
        lease_token=str(failed_claim["lease_token"]),
        error_code="render_failed",
    )
    failed_wake = service.claim_wake(
        str(failed_run["run_id"]),
        turn_id=str(failed_turn["id"]),
    )
    assert failed_wake["run_status"] == "failed"
    failed_late_turn = store.start_turn(
        session_id,
        {"message": "Attach after the failed render."},
        policy_hash="policy-v1",
    )
    assert service.wait_for_completion(
        str(failed_run["run_id"]),
        turn_id=str(failed_late_turn["id"]),
    )["status"] == "ready"
    assert service.claim_wake(
        str(failed_run["run_id"]),
        turn_id=str(failed_late_turn["id"]),
    )["run_status"] == "failed"


def test_background_source_must_match_exact_session_and_capability_lineage(
    tmp_path: Path,
) -> None:
    db = tmp_path / "background-lineage.sqlite"
    session_id, _source_message_id = _base(db)
    store = SessionStore(db)
    other_session = store.get_or_create(
        origin="slack",
        transport="slack",
        binding_id="binding-a",
        conversation_ref="conversation-a",
        thread_ref="thread-other",
        child_id="child-a",
    )
    now = utcnow()
    with connect(db) as conn:
        for event_id, text in (("evt-lineage-a", "Source A"), ("evt-lineage-b", "Source B")):
            conn.execute(
                """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                   VALUES(?,'context_input','child-a',?,'test','{}',?,'completed')""",
                (event_id, text, now),
            )
    store.append_message(session_id, role="user", content="Source A", event_id="evt-lineage-a")
    store.append_message(
        str(other_session["id"]),
        role="user",
        content="Source B",
        event_id="evt-lineage-b",
    )
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO capability_runs(
                   id,session_id,capability_id,capability_version,skill_versions_json,status,
                   source_event_id,result_json,created_at,updated_at
               ) VALUES('caprun-lineage-a',?,'artifact_worksheet','2','{}','running',?,'{}',?,?)""",
            (session_id, "evt-lineage-a", now, now),
        )
    service = BackgroundRunService(db)
    accepted = service.create(
        run_kind="lineage_check",
        idempotency_key="lineage:accepted",
        session_id=session_id,
        source_event_id="evt-lineage-a",
        capability_run_id="caprun-lineage-a",
        policy_snapshot={"delivery": "parent_confirmation"},
    )
    assert accepted["status"] == "queued"
    with pytest.raises(ValueError, match="outside the capability-run lineage"):
        service.create(
            run_kind="lineage_check",
            idempotency_key="lineage:rejected",
            session_id=session_id,
            source_event_id="evt-lineage-b",
            capability_run_id="caprun-lineage-a",
            policy_snapshot={"delivery": "parent_confirmation"},
        )


def test_background_attempt_and_delivery_leases_fence_stale_workers(tmp_path: Path) -> None:
    db = tmp_path / "background-fencing.sqlite"
    session_id, _source_message_id = _base(db)
    capability_run_id, release_unit_id = _release_unit(db, session_id)
    service = BackgroundRunService(db)
    created = service.create(
        run_kind="artifact_render",
        idempotency_key="fencing:run",
        session_id=session_id,
        capability_run_id=capability_run_id,
        policy_snapshot={"delivery": "parent_confirmation"},
    )
    stale = service.claim(str(created["run_id"]), worker_id="stale-worker")
    with connect(db) as conn:
        conn.execute(
            "UPDATE background_runs SET leased_at='2000-01-01T00:00:00+00:00' WHERE run_id=?",
            (created["run_id"],),
        )
    service.requeue_stale(
        str(created["run_id"]), stale_before="2026-08-16T00:00:00Z"
    )
    current = service.claim(str(created["run_id"]), worker_id="current-worker")
    with pytest.raises(ValueError, match="active lease"):
        service.complete(
            str(created["run_id"]),
            attempt_id=str(stale["attempt_id"]),
            lease_token=str(stale["lease_token"]),
            result_pointer={"release_unit_id": release_unit_id},
        )
    with pytest.raises(ValueError, match="active lease"):
        service.fail(
            str(created["run_id"]),
            attempt_id=str(stale["attempt_id"]),
            lease_token=str(stale["lease_token"]),
            error_code="stale_worker_failed",
        )
    with pytest.raises(ValueError, match="capability run"):
        service.complete(
            str(created["run_id"]),
            attempt_id=str(current["attempt_id"]),
            lease_token=str(current["lease_token"]),
            result_pointer={"release_unit_id": "release-not-owned"},
        )
    service.complete(
        str(created["run_id"]),
        attempt_id=str(current["attempt_id"]),
        lease_token=str(current["lease_token"]),
        result_pointer={"release_unit_id": release_unit_id},
    )

    delivery = service.request_parent_delivery(
        str(created["run_id"]),
        binding_id="binding-a",
        payload_pointer={"release_unit_id": release_unit_id},
        idempotency_key="fencing:delivery",
    )
    service.confirm_parent_delivery(str(delivery["delivery_id"]), parent_id="parent-a")
    first_send = service.claim_delivery(
        str(delivery["delivery_id"]), worker_id="stale-delivery-worker"
    )
    with connect(db) as conn:
        conn.execute(
            """UPDATE background_delivery_outbox
               SET leased_at='2000-01-01T00:00:00+00:00' WHERE delivery_id=?""",
            (delivery["delivery_id"],),
        )
    unknown = service.requeue_stale_delivery(
        str(delivery["delivery_id"]), stale_before="2026-08-16T00:00:00Z"
    )
    assert unknown["status"] == "unknown"
    with pytest.raises(ValueError, match="not confirmed and queued"):
        service.claim_delivery(
            str(delivery["delivery_id"]), worker_id="retry-delivery-worker"
        )
    with pytest.raises(ValueError, match="active lease"):
        service.finish_delivery(
            str(delivery["delivery_id"]),
            attempt_id=str(first_send["attempt_id"]),
            lease_token=str(first_send["lease_token"]),
            outcome="sent",
            external_message_id="stale-message",
        )

    ambiguous_delivery = service.request_parent_delivery(
        str(created["run_id"]),
        binding_id="binding-a",
        payload_pointer={"release_unit_id": release_unit_id},
        idempotency_key="fencing:delivery-ambiguous",
    )
    service.confirm_parent_delivery(
        str(ambiguous_delivery["delivery_id"]), parent_id="parent-a"
    )
    ambiguous_send = service.claim_delivery(
        str(ambiguous_delivery["delivery_id"]), worker_id="ambiguous-delivery-worker"
    )
    assert service.finish_delivery(
        str(ambiguous_delivery["delivery_id"]),
        attempt_id=str(ambiguous_send["attempt_id"]),
        lease_token=str(ambiguous_send["lease_token"]),
        outcome="unknown",
        external_message_id=None,
        error_code="provider_timeout_after_send",
    )["status"] == "unknown"
    with pytest.raises(ValueError, match="not confirmed and queued"):
        service.claim_delivery(
            str(ambiguous_delivery["delivery_id"]), worker_id="unsafe-retry-worker"
        )

    retry_delivery = service.request_parent_delivery(
        str(created["run_id"]),
        binding_id="binding-a",
        payload_pointer={"release_unit_id": release_unit_id},
        idempotency_key="fencing:delivery-explicit-retry",
    )
    service.confirm_parent_delivery(
        str(retry_delivery["delivery_id"]), parent_id="parent-a"
    )
    second_send = service.claim_delivery(
        str(retry_delivery["delivery_id"]), worker_id="retry-delivery-worker"
    )
    retried = service.finish_delivery(
        str(retry_delivery["delivery_id"]),
        attempt_id=str(second_send["attempt_id"]),
        lease_token=str(second_send["lease_token"]),
        outcome="confirmed_rejected",
        external_message_id=None,
        error_code="temporary_transport_error",
    )
    assert retried["status"] == "queued"
    final_send = service.claim_delivery(
        str(retry_delivery["delivery_id"]), worker_id="final-delivery-worker"
    )
    assert service.finish_delivery(
        str(retry_delivery["delivery_id"]),
        attempt_id=str(final_send["attempt_id"]),
        lease_token=str(final_send["lease_token"]),
        outcome="sent",
        external_message_id="external-final-message",
    )["status"] == "sent"

    revoked_delivery = service.request_parent_delivery(
        str(created["run_id"]),
        binding_id="binding-a",
        payload_pointer={"release_unit_id": release_unit_id},
        idempotency_key="fencing:delivery-revoked",
    )
    service.confirm_parent_delivery(
        str(revoked_delivery["delivery_id"]), parent_id="parent-a"
    )
    revoke_binding(db, "binding-a")
    pairing_code = create_pairing_code(db, "parent-a")["pairing_code"]
    repaired = consume_pairing_code(
        db,
        str(pairing_code),
        InboundMessage(
            external_event_id="re-pair-after-background-revoke",
            team_id="team",
            user_id="user",
            channel_id="channel",
            text=f"pair {pairing_code}",
        ),
    )
    assert repaired["binding_id"] == "binding-a"
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM background_delivery_outbox WHERE delivery_id=?",
            (revoked_delivery["delivery_id"],),
        ).fetchone()["status"] == "cancelled"
    with pytest.raises(ValueError, match="not confirmed and queued"):
        service.claim_delivery(
            str(revoked_delivery["delivery_id"]), worker_id="revoked-delivery-worker"
        )


def test_raw_event_deletion_removes_adaptation_copies_and_scrubs_memory_reason(
    tmp_path: Path,
) -> None:
    db = tmp_path / "adaptation-delete.sqlite"
    session_id, source_message_id = _base(db)
    event = Event(
        id="evt-adaptation-source",
        type="child_question",
        child_id="child-a",
        text="A raw source that must be erasable.",
        source="synthetic_parent",
    )
    Repository(db).submit_event(event)
    SessionStore(db).attach_event(source_message_id, event.id)
    memory = ParentMemoryLedger(db).remember(
        scope_type="thread",
        scope_id=session_id,
        category="answer_style",
        value="Deletion sentinel preference.",
        provenance="deletion sentinel provenance",
        reason="deletion sentinel reason",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    for index in range(18):
        SessionStore(db).append_message(
            session_id,
            role="assistant" if index % 2 else "user",
            content=f"Deletion continuity {index}.",
        )
    ThreadCompactor(db).context(session_id, compact_after=16, keep_recent=6)
    capability_run_id, _release_unit_id = _release_unit(db, session_id)
    with connect(db) as conn:
        conn.execute(
            "UPDATE capability_runs SET source_event_id=? WHERE id=?",
            (event.id, capability_run_id),
        )
    run = BackgroundRunService(db).create(
        run_kind="artifact_render",
        idempotency_key="deletion:background",
        session_id=session_id,
        source_event_id=event.id,
        capability_run_id=capability_run_id,
        policy_snapshot={"delivery": "parent_confirmation"},
    )
    delete_family_event(
        db,
        child_id="child-a",
        event_id=event.id,
        confirmation=event.id,
        reason="Parent requested source deletion.",
        output_root=tmp_path,
    )
    with connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM background_runs WHERE run_id=?", (run["run_id"],)
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM thread_compactions").fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM parent_memories WHERE memory_key=?",
            (memory["memory_key"],),
        ).fetchone()[0] == 1
        forgotten = conn.execute(
            """SELECT status,value,provenance,why_saved,source_message_id
               FROM parent_memories WHERE memory_key=?""",
            (memory["memory_key"],),
        ).fetchone()
        assert tuple(forgotten) == ("forgotten", "", "forgotten", "forgotten", None)
        assert conn.execute(
            "SELECT COUNT(*) FROM parent_memory_tombstones WHERE memory_key_hash=?",
            (event_id_hash(str(memory["memory_key"])),),
        ).fetchone()[0] == 1
        memory_events = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM parent_memory_events WHERE memory_key=?",
                (memory["memory_key"],),
            ).fetchall()
        ]
        assert "deletion sentinel" not in jdump(memory_events)


def test_source_event_deletion_preserves_later_governed_run_with_scrubbed_memory_lineage(
    tmp_path: Path,
) -> None:
    db = tmp_path / "cross-event-memory-delete.sqlite"
    session_id, source_message_id = _base(db)
    service = CuriosityService(db, tmp_path / "output")
    source_event = service.ask(
        child_id="child-a",
        text="Why do shadows move?",
        answer_mode="quick",
        event_id="evt-memory-source",
        enqueue_visual=False,
    )
    SessionStore(db).attach_event(source_message_id, str(source_event["event_id"]))
    secret = "Private armadillo artifact sentinel."
    memory = ParentMemoryLedger(db).remember(
        scope_type="thread",
        scope_id=session_id,
        category="artifact_style",
        value=secret,
        provenance="explicit parent message",
        reason="Apply the private artifact layout.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )
    delivery_id = enqueue_delivery(
        db,
        "binding-a",
        OutboundMessage(channel_id="channel", text=secret, thread_id="memory-delete"),
        idempotency_key="slack:memory-delete:inspection",
        parent_memory_refs=[
            {"memory_key": memory["memory_key"], "version": memory["version"]}
        ],
    )
    later_event = service.ask(
        child_id="child-a",
        text="Why do tree rings form?",
        answer_mode="quick",
        event_id="evt-later-governed-artifact",
        enqueue_visual=False,
    )
    SessionStore(db).append_message(
        session_id,
        role="user",
        content="Why do tree rings form?",
        event_id=str(later_event["event_id"]),
    )
    governed = service._create_governed_session_artifact(
        session_id=session_id,
        source_event_id=str(later_event["event_id"]),
        artifact_type="worksheet",
        revision="Make a simple comparison page.",
    )
    run_id = str(governed["capability_run_id"])
    with connect(db) as conn:
        binding = conn.execute(
            """SELECT memory_key,memory_version,content_hash
               FROM capability_run_parent_memories WHERE capability_run_id=?""",
            (run_id,),
        ).fetchone()
        assert binding is not None
        original_hash = str(binding["content_hash"])

    delete_family_event(
        db,
        child_id="child-a",
        event_id=str(source_event["event_id"]),
        confirmation=str(source_event["event_id"]),
        reason="Parent erased the source of a later-used memory.",
        output_root=tmp_path / "output",
    )
    with connect(db) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM events WHERE id=?",
            (source_event["event_id"],),
        ).fetchone()[0] == 0
        retained_run = conn.execute(
            "SELECT source_event_id,status FROM capability_runs WHERE id=?",
            (run_id,),
        ).fetchone()
        assert retained_run is not None
        assert retained_run["source_event_id"] == later_event["event_id"]
        retained_binding = conn.execute(
            """SELECT content_hash FROM capability_run_parent_memories
               WHERE capability_run_id=? AND memory_key=?""",
            (run_id, memory["memory_key"]),
        ).fetchone()
        assert retained_binding is not None
        assert retained_binding["content_hash"] == original_hash
        scrubbed = conn.execute(
            """SELECT status,value,provenance,why_saved,source_message_id
               FROM parent_memories WHERE memory_key=?""",
            (memory["memory_key"],),
        ).fetchone()
        assert tuple(scrubbed) == ("forgotten", "", "forgotten", "forgotten", None)
        delivery = conn.execute(
            "SELECT status,payload_json FROM delivery_outbox WHERE id=?",
            (delivery_id,),
        ).fetchone()
        assert delivery["status"] == "expired"
        assert REDACTED_PARENT_MEMORY_INSPECTION in delivery["payload_json"]
        assert secret not in delivery["payload_json"]
        assert secret not in jdump(
            [
                dict(row)
                for row in conn.execute(
                    """SELECT value,provenance,why_saved FROM parent_memories
                       WHERE memory_key=?""",
                    (memory["memory_key"],),
                ).fetchall()
            ]
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_undo_cannot_reactivate_a_source_deleted_memory_version(tmp_path: Path) -> None:
    db = tmp_path / "undo-after-source-delete.sqlite"
    session_id, source_message_id = _base(db)
    output = tmp_path / "output"
    service = CuriosityService(db, output)
    first_event = service.ask(
        child_id="child-a",
        text="Why do shadows move?",
        answer_mode="quick",
        event_id="evt-undo-source-a",
        enqueue_visual=False,
    )
    SessionStore(db).attach_event(source_message_id, str(first_event["event_id"]))
    ledger = ParentMemoryLedger(db)
    first = ledger.remember(
        scope_type="thread",
        scope_id=session_id,
        category="artifact_style",
        value="Use the first private layout sentinel.",
        provenance="explicit parent message",
        reason="Parent requested the first layout.",
        parent_id="parent-a",
        source_message_id=source_message_id,
    )

    second_event = service.ask(
        child_id="child-a",
        text="Why do tree rings form?",
        answer_mode="quick",
        event_id="evt-undo-source-b",
        enqueue_visual=False,
    )
    second_source_message_id = SessionStore(db).append_message(
        session_id,
        role="user",
        content="Please switch to the second private layout sentinel.",
        event_id=str(second_event["event_id"]),
    )
    second = ledger.update(
        str(first["memory_key"]),
        value="Use the second private layout sentinel.",
        parent_id="parent-a",
        reason="Parent requested the second layout.",
        source_message_id=second_source_message_id,
    )

    delete_family_event(
        db,
        child_id="child-a",
        event_id=str(first_event["event_id"]),
        confirmation=str(first_event["event_id"]),
        reason="Parent erased the source of the first layout.",
        output_root=output,
    )
    ledger.undo(
        str(second["undo_event_id"]),
        parent_id="parent-a",
        reason="Parent undid the second layout after erasing the first.",
    )

    assert ledger.recall(session_id=session_id) == []
    init_db(db)
    with connect(db) as conn:
        versions = conn.execute(
            """SELECT version,status,value,provenance,why_saved
               FROM parent_memories WHERE memory_key=? ORDER BY version""",
            (first["memory_key"],),
        ).fetchall()
        tombstones = conn.execute(
            """SELECT version FROM parent_memory_tombstones
               WHERE memory_key_hash=? ORDER BY version""",
            (event_id_hash(str(first["memory_key"])),),
        ).fetchall()
    assert [tuple(row) for row in versions] == [
        (1, "forgotten", "", "forgotten", "forgotten"),
        (2, "forgotten", "", "forgotten", "forgotten"),
    ]
    assert [row["version"] for row in tombstones] == [1, 2]

    later_event = service.ask(
        child_id="child-a",
        text="Why do leaves have veins?",
        answer_mode="quick",
        event_id="evt-undo-source-c",
        enqueue_visual=False,
    )
    SessionStore(db).append_message(
        session_id,
        role="user",
        content="Why do leaves have veins?",
        event_id=str(later_event["event_id"]),
    )
    governed = service._create_governed_session_artifact(
        session_id=session_id,
        source_event_id=str(later_event["event_id"]),
        artifact_type="worksheet",
        revision="Make a simple noticing page.",
    )
    with connect(db) as conn:
        assert conn.execute(
            """SELECT COUNT(*) FROM capability_run_parent_memories
               WHERE capability_run_id=?""",
            (governed["capability_run_id"],),
        ).fetchone()[0] == 0


def test_schedule_catch_up_materializes_only_latest_missed_run(tmp_path: Path) -> None:
    db = tmp_path / "catch-up.sqlite"
    session_id, _source_message_id = _base(db)
    capability_run_id, release_unit_id = _release_unit(db, session_id)
    service = BackgroundRunService(db)
    result = service.catch_up_once(
        schedule_id="weekly-parent",
        missed_due_at=["2026-08-01T10:00:00Z", "2026-08-08T10:00:00Z", "2026-08-15T10:00:00Z"],
        run_kind="weekly_summary",
        session_id=session_id,
        capability_run_id=capability_run_id,
        policy_snapshot={"catch_up": "once", "overlap": "skip"},
    )
    assert result and result["status"] == "queued"
    service.claim(str(result["run_id"]), worker_id="worker-before-crash")
    with connect(db) as conn:
        conn.execute(
            "UPDATE background_runs SET leased_at='2000-01-01T00:00:00Z' WHERE run_id=?",
            (result["run_id"],),
        )
    service = BackgroundRunService(db)
    service.requeue_stale(str(result["run_id"]), stale_before="2026-08-16T00:00:00Z")
    reclaimed = service.claim(str(result["run_id"]), worker_id="worker-after-crash")
    assert reclaimed["attempt_number"] == 2
    completed = service.complete(
        str(result["run_id"]),
        attempt_id=str(reclaimed["attempt_id"]),
        lease_token=str(reclaimed["lease_token"]),
        result_pointer={"release_unit_id": release_unit_id},
    )
    assert completed["status"] == "completed"
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT input_json FROM background_runs WHERE run_kind='weekly_summary'"
        ).fetchall()
        attempts = conn.execute(
            "SELECT status FROM background_run_attempts WHERE run_id=? ORDER BY attempt_number",
            (result["run_id"],),
        ).fetchall()
    assert len(rows) == 1
    assert [row["status"] for row in attempts] == ["abandoned", "completed"]
    assert '"due_at":"2026-08-15T10:00:00+00:00"' in rows[0]["input_json"]
    assert '"missed_count":3' in rows[0]["input_json"]


def test_background_crash_recovery_stops_at_bounded_attempt_limit(tmp_path: Path) -> None:
    db = tmp_path / "attempt-limit.sqlite"
    _base(db)
    service = BackgroundRunService(db)
    created = service.create(
        run_kind="artifact_render",
        idempotency_key="attempt-limit:one",
        policy_snapshot={"delivery": "parent_confirmation"},
        max_attempts=1,
    )
    service.claim(str(created["run_id"]), worker_id="worker-before-crash")
    with connect(db) as conn:
        conn.execute(
            "UPDATE background_runs SET leased_at='2000-01-01T00:00:00+00:00' WHERE run_id=?",
            (created["run_id"],),
        )
    BackgroundRunService(db).requeue_stale(
        str(created["run_id"]),
        stale_before="2026-08-16T00:00:00Z",
    )
    exhausted = BackgroundRunService(db).claim(
        str(created["run_id"]),
        worker_id="worker-after-crash",
    )
    assert exhausted == {
        "run_id": created["run_id"],
        "status": "failed",
        "error_code": "max_attempts_exhausted",
    }
