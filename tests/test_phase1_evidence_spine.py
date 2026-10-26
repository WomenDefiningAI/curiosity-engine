from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import curiosity_engine.db as db_module
import curiosity_engine.learner_model as learner_model_module
from curiosity_engine.context_builder import MAX_CONTEXT_CHARS, build_context, prepare_model_inputs
from curiosity_engine.contracts import Event, ObservationCandidate
from curiosity_engine.db import SCHEMA, SCHEMA_VERSION, connect, init_db
from curiosity_engine.director import AutonomousDirector
from curiosity_engine.episodes import apply_episode_correction, child_episode_context, episode_for_event
from curiosity_engine.graph import add_child
from curiosity_engine.learner_model import (
    EvidenceConflict,
    correct_evidence,
    delete_learning_source,
    expire_learning_records,
    learner_snapshot,
    recompute_child_projections,
    record_learning_opportunity,
    record_model_annotations,
    register_concept,
    register_concept_relation,
    register_learning_plan,
    register_learning_task,
    register_rubric,
)
from curiosity_engine.reasoning import ReasoningEngine, StubBackend
from curiosity_engine.repository import Repository
from curiosity_engine.runtime import CuriosityHarness

START = datetime.now(UTC).replace(microsecond=0) - timedelta(days=10)


def _setup(db: Path) -> None:
    init_db(db)
    add_child(db, "child-a", "Synthetic Child A", 2020, "1")
    add_child(db, "child-b", "Synthetic Child B", 2020, "1")
    register_concept(
        db,
        {
            "concept_id": "science.phase-change",
            "version": 1,
            "label": "Phase change",
            "domain": "science",
            "grade_band": "K-2",
            "description": "Matter changing between solid and liquid states.",
            "aliases": ["melting", "freezing"],
        },
    )
    register_concept(
        db,
        {
            "concept_id": "science.matter",
            "version": 1,
            "label": "Matter",
            "domain": "science",
        },
    )
    register_concept_relation(
        db,
        {
            "relation_id": "relation.phase-change-part-of-matter",
            "source": {"concept_id": "science.phase-change", "version": 1},
            "relation": "part_of",
            "target": {"concept_id": "science.matter", "version": 1},
            "provenance": {"source": "synthetic-reviewed-fixture"},
            "relation_version": "1",
        },
    )
    register_rubric(
        db,
        {
            "rubric_id": "rubric.phase-change",
            "version": 1,
            "label": "Phase-change explanation",
            "criteria": [
                {
                    "criterion_id": "criterion.phase-change",
                    "concept_id": "science.phase-change",
                    "concept_version": 1,
                    "description": "Names the observed change without a global ability judgment.",
                    "required": True,
                    "weight": 1,
                }
            ],
        },
    )
    register_learning_task(db, _task())


def _task(*, version: int = 1, difficulty: str = "familiar", provenance: str = "reviewed_task") -> dict[str, Any]:
    return {
        "task_id": "task.phase-change",
        "version": version,
        "title": f"Observe melting v{version}",
        "learning_goal": "Explain an observable phase change.",
        "strategy": "predict-test-explain",
        "difficulty": difficulty,
        "difficulty_provenance": provenance,
        "modalities": ["spoken", "drawn"],
        "context_key": "ice-in-kitchen",
        "representation_key": "physical-observation",
        "concepts": [{"concept_id": "science.phase-change", "version": 1, "role": "target"}],
        "rubric_id": "rubric.phase-change",
        "rubric_version": 1,
    }


def _event(
    db: Path,
    event_id: str,
    *,
    child_id: str = "child-a",
    day: int = 0,
    text: str | None = None,
    force_new_episode: bool = False,
) -> str:
    Repository(db).submit_event(
        Event(
            id=event_id,
            type="child_question",
            child_id=child_id,
            text=text or f"Synthetic phase-change observation {event_id}",
            created_at=START + timedelta(days=day),
        )
    )
    if force_new_episode:
        apply_episode_correction(db, child_id=child_id, event_id=event_id, action="new_episode")
    episode = episode_for_event(db, event_id)
    assert episode is not None
    return str(episode["episode_id"])


def _record(db: Path, event_id: str, opportunity_id: str, **overrides: Any) -> dict[str, Any]:
    task_id = str(overrides.pop("task_id", "task.phase-change"))
    task_version = int(overrides.pop("task_version", 1))
    plan_id = str(overrides.pop("plan_id", f"plan-{opportunity_id}"))
    plan_source_event_id = str(overrides.pop("plan_source_event_id", event_id))
    context_key = overrides.pop("context_key", None)
    representation_key = overrides.pop("representation_key", None)
    transfer_from_plan_id = overrides.pop("transfer_from_plan_id", None)
    result = str(overrides.pop("result", "correct"))
    criterion_results = overrides.pop(
        "criterion_results",
        [{"criterion_id": "criterion.phase-change", "outcome": result}],
    )
    child_id = str(overrides.get("child_id", "child-a"))
    register_learning_plan(
        db,
        {
            "plan_id": plan_id,
            "child_id": child_id,
            "source_event_id": plan_source_event_id,
            "task_id": task_id,
            "task_version": task_version,
            "context_key": context_key,
            "representation_key": representation_key,
            "transfer_from_plan_id": transfer_from_plan_id,
        },
    )
    payload: dict[str, Any] = {
        "opportunity_id": opportunity_id,
        "child_id": child_id,
        "event_id": event_id,
        "plan_id": plan_id,
        "actor": "child",
        "initiation": "child",
        "response_mode": "spoken",
        "assistance_level": "none",
        "criterion_results": criterion_results,
        "scorer": "reviewed-parent-observation",
        "scorer_version": "1",
        "source_type": "direct_observation",
        "retention_class": "learning_record",
    }
    payload.update(overrides)
    return record_learning_opportunity(db, payload)


def _knowledge(db: Path) -> dict[str, Any] | None:
    items = learner_snapshot(db, "child-a")["knowledge"]
    return items[0] if items else None


def _interest(db: Path) -> dict[str, Any] | None:
    items = learner_snapshot(db, "child-a")["interests"]
    return items[0] if items else None


def test_v18_migration_is_additive_transactional_and_ledgered(tmp_path: Path) -> None:
    db = tmp_path / "legacy-v17.sqlite"
    connection = sqlite3.connect(db)
    connection.executescript(SCHEMA)
    connection.execute(
        """INSERT INTO children(id,name,birth_year,grade,created_at,updated_at)
           VALUES('legacy-child','Synthetic Legacy',2020,'1','now','now')"""
    )
    connection.execute("PRAGMA user_version = 17")
    connection.commit()
    connection.close()

    backup = init_db(db)

    assert backup is not None and backup.is_file()
    with connect(db) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 26
        assert dict(connection.execute("SELECT version,name FROM schema_migrations").fetchone()) == {
            "version": 18,
            "name": "versioned evidence spine",
        }
        assert (
            connection.execute("SELECT name FROM children WHERE id='legacy-child'").fetchone()[0] == "Synthetic Legacy"
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "schema_migrations",
        "concepts",
        "concept_aliases",
        "concept_relations",
        "rubrics",
        "learning_tasks",
        "learning_task_concepts",
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
        "evidence_source_tombstones",
    } <= tables
    assert init_db(db) is None
    with connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM schema_migrations WHERE version=18").fetchone()[0] == 1


def test_v18_migration_repairs_partial_state_rolls_back_failures_and_serializes_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    concurrent_db = tmp_path / "concurrent.sqlite"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(init_db, [concurrent_db] * 4))
    assert results == [None, None, None, None]
    with connect(concurrent_db) as connection:
        connection.execute("DROP TABLE concept_aliases")
    # Simulate a fresh process after deliberately corrupting the current schema.
    # Hot request paths intentionally trust an already initialized local database.
    db_module._INITIALIZED_DATABASES.clear()
    assert init_db(concurrent_db) is None
    with connect(concurrent_db) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='concept_aliases'"
        ).fetchone()

    failing_db = tmp_path / "failing-v17.sqlite"
    connection = sqlite3.connect(failing_db)
    connection.executescript(SCHEMA)
    connection.execute("PRAGMA user_version = 17")
    connection.commit()
    connection.close()
    monkeypatch.setattr(db_module, "MIGRATION_18_SQL", db_module.MIGRATION_18_SQL + "\nINVALID SQL;")
    with pytest.raises(sqlite3.OperationalError):
        init_db(failing_db)
    connection = sqlite3.connect(failing_db)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 17
        assert (
            connection.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='concepts'").fetchone()[
                0
            ]
            == 0
        )
    finally:
        connection.close()


def test_immutable_registrations_are_idempotent_and_conflict_on_redefinition(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)

    assert (
        register_concept(
            db,
            {
                "concept_id": "science.phase-change",
                "label": "Phase change",
                "domain": "science",
                "grade_band": "K-2",
                "description": "Matter changing between solid and liquid states.",
                "aliases": ["melting", "freezing"],
            },
        )["duplicate"]
        is True
    )
    assert register_learning_task(db, _task())["duplicate"] is True
    changed = _task()
    changed["learning_goal"] = "A conflicting replacement goal."
    with pytest.raises(EvidenceConflict, match="different immutable definition"):
        register_learning_task(db, changed)


def test_learning_plans_are_immutable_and_own_retry_and_transfer_lineage(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_plan_source")
    first = _record(db, "evt_plan_source", "opp_plan_source")
    assert first["plan_id"] == "plan-opp_plan_source"
    assert (
        register_learning_plan(
            db,
            {
                "plan_id": first["plan_id"],
                "child_id": "child-a",
                "source_event_id": "evt_plan_source",
                "task_id": "task.phase-change",
            },
        )["duplicate"]
        is True
    )
    with pytest.raises(EvidenceConflict, match="different payload"):
        register_learning_plan(
            db,
            {
                "plan_id": first["plan_id"],
                "child_id": "child-a",
                "source_event_id": "evt_plan_source",
                "task_id": "task.phase-change",
                "context_key": "conflicting-context",
            },
        )

    with pytest.raises(EvidenceConflict, match="attempt already has"):
        _record(
            db,
            "evt_plan_source",
            "opp_alternate_attempt_one",
            plan_id=first["plan_id"],
            plan_source_event_id="evt_plan_source",
        )
    with pytest.raises(ValueError, match="is_transfer"):
        record_learning_opportunity(
            db,
            {
                "opportunity_id": "opp_forged_transfer_flag",
                "child_id": "child-a",
                "event_id": "evt_plan_source",
                "plan_id": first["plan_id"],
                "actor": "child",
                "response_mode": "spoken",
                "assistance_level": "none",
                "criterion_results": [{"criterion_id": "criterion.phase-change", "outcome": "correct"}],
                "scorer": "reviewed-parent-observation",
                "scorer_version": "1",
                "source_type": "direct_observation",
                "is_transfer": True,
            },
        )
    with pytest.raises(ValueError, match="follow the prior attempt"):
        _record(
            db,
            "evt_plan_source",
            "opp_attempt_gap",
            plan_id=first["plan_id"],
            plan_source_event_id="evt_plan_source",
            attempt_number=3,
        )

    _event(db, "evt_same_episode_transfer", text="Why does this same ice melt?")
    apply_episode_correction(
        db,
        child_id="child-a",
        event_id="evt_same_episode_transfer",
        action="retry",
        related_event_id="evt_plan_source",
    )
    with pytest.raises(ValueError, match="distinct learning episodes"):
        _record(
            db,
            "evt_same_episode_transfer",
            "opp_same_episode_transfer",
            transfer_from_plan_id=first["plan_id"],
            context_key="different-context",
        )


def test_child_ownership_is_checked_by_api_and_composite_foreign_keys(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_owner")
    with pytest.raises(ValueError, match="does not match"):
        _record(db, "evt_owner", "opp_wrong-child", child_id="child-b")

    _record(db, "evt_owner", "opp_owner")
    child_b_episode = _event(db, "evt_owner_b", child_id="child-b")
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as connection:
            connection.execute("UPDATE learning_opportunities SET child_id='child-b' WHERE opportunity_id='opp_owner'")
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as connection:
            connection.execute(
                "UPDATE evidence_records SET episode_id=? WHERE opportunity_id='opp_owner'",
                (child_b_episode,),
            )


def test_parent_report_actor_source_and_response_mode_are_all_or_nothing(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_mismatched_parent_mode")
    with pytest.raises(ValueError, match="must occur together"):
        _record(
            db,
            "evt_mismatched_parent_mode",
            "opp_mismatched_parent_mode",
            actor="child",
            source_type="direct_observation",
            response_mode="parent_report",
        )
    assert _knowledge(db) is None
    with connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM learning_opportunities").fetchone()[0] == 0


def test_model_annotations_and_generated_outputs_never_compile_their_own_evidence(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_model")
    with connect(db) as connection:
        connection.execute(
            """INSERT INTO runs(id,job,status,started_at,details_json,event_id)
               VALUES(100,'synthetic','completed','now','{}','evt_model')"""
        )
    annotation_ids = record_model_annotations(
        db,
        child_id="child-a",
        event_id="evt_model",
        run_id=100,
        candidates=[ObservationCandidate(kind="concept_mention", label="phase change")],
    )
    assert len(annotation_ids) == 1
    _record(
        db,
        "evt_model",
        "opp_generated",
        actor="system",
        initiation="engine",
        response_mode="selection",
        source_type="system_generated",
    )

    assert _knowledge(db) is None
    assert _interest(db) is None
    with connect(db) as connection:
        assert connection.execute("SELECT authority FROM evidence_annotations").fetchone()[0] == "audit_only"
        assert connection.execute("SELECT COUNT(*) FROM evidence_records").fetchone()[0] == 2
        assert connection.execute("SELECT SUM(projection_eligible) FROM evidence_records").fetchone()[0] == 0


def test_annotations_are_event_bound_audit_only_and_expire(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_annotation_a")
    _event(db, "evt_annotation_b", day=1)
    with connect(db) as connection:
        connection.execute(
            """INSERT INTO runs(id,job,status,started_at,details_json,event_id)
               VALUES(101,'synthetic','completed','now','{}','evt_annotation_a')"""
        )
    annotation_id = record_model_annotations(
        db,
        child_id="child-a",
        event_id="evt_annotation_a",
        run_id=101,
        candidates=[ObservationCandidate(kind="concept_mention", label="phase change")],
    )[0]
    with pytest.raises(ValueError, match="run does not belong"):
        record_model_annotations(
            db,
            child_id="child-a",
            event_id="evt_annotation_b",
            run_id=101,
            candidates=[ObservationCandidate(kind="concept_mention", label="matter")],
        )
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as connection:
            connection.execute(
                "UPDATE evidence_annotations SET event_id='evt_annotation_b' WHERE annotation_id=?",
                (annotation_id,),
            )
    expired = expire_learning_records(db, as_of=datetime.now(UTC) + timedelta(days=31))
    assert expired["expired_annotations"] == 1
    with connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM evidence_annotations").fetchone()[0] == 0


def test_helped_trivial_unknown_and_retry_evidence_cannot_over_promote(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_helped", text="Why does this ice melt?")
    _record(db, "evt_helped", "opp_helped", assistance_level="hint")
    assert _knowledge(db)["state"] == "practicing_with_support"

    _event(db, "evt_repeat", day=1, text="Why does this ice melt?")
    assert episode_for_event(db, "evt_repeat")["independence_status"] != "eligible"
    _record(
        db,
        "evt_repeat",
        "opp_repeat",
        plan_id="plan-opp_helped",
        plan_source_event_id="evt_helped",
        attempt_number=2,
    )
    assert _knowledge(db)["state"] == "practicing_with_support"

    register_learning_task(db, _task(version=2, difficulty="trivial"))
    _event(db, "evt_trivial", day=3, force_new_episode=True)
    _record(db, "evt_trivial", "opp_trivial", task_version=2)
    assert _knowledge(db)["state"] == "practicing_with_support"

    _event(db, "evt_unknown-help", day=6, force_new_episode=True)
    _record(db, "evt_unknown-help", "opp_unknown-help", assistance_level="unknown")
    assert _knowledge(db)["state"] == "practicing_with_support"

    register_learning_task(db, _task(version=3, provenance="estimated"))
    _event(db, "evt_estimated", day=9, force_new_episode=True)
    _record(db, "evt_estimated", "opp_estimated", task_version=3)
    assert _knowledge(db)["state"] == "practicing_with_support"


def test_knowledge_promotes_only_through_independent_and_real_transfer_evidence(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_success-one")
    first = _record(db, "evt_success-one", "opp_success-one")
    assert first["duplicate"] is False
    assert _knowledge(db)["state"] == "demonstrated_in_context"
    assert _record(db, "evt_success-one", "opp_success-one")["duplicate"] is True

    _event(db, "evt_success-two", day=3, force_new_episode=True)
    _record(db, "evt_success-two", "opp_success-two", response_mode="drawn")
    assert _knowledge(db)["state"] == "independently_demonstrated"

    _event(db, "evt_fake-transfer", day=6, force_new_episode=True)
    with pytest.raises(ValueError, match="vary context or representation"):
        _record(
            db,
            "evt_fake-transfer",
            "opp_fake-transfer",
            transfer_from_plan_id="plan-opp_success-two",
        )
    assert _knowledge(db)["state"] == "independently_demonstrated"

    _event(db, "evt_real-transfer", day=9, force_new_episode=True)
    _record(
        db,
        "evt_real-transfer",
        "opp_real-transfer",
        transfer_from_plan_id="plan-opp_success-two",
        context_key="sunny-window-puddle",
        representation_key="child-drawn-particle-model",
    )
    knowledge = _knowledge(db)
    assert knowledge["state"] == "transfer_observed"
    assert knowledge["eligible_episode_count"] == 3
    assert knowledge["representation_diversity"] == 2
    with connect(db) as connection:
        connection.execute("DELETE FROM children WHERE id='child-a'")
        assert (
            connection.execute("SELECT COUNT(*) FROM learning_episode_plans WHERE child_id='child-a'").fetchone()[0]
            == 0
        )


def test_transfer_projection_requires_success_from_the_exact_linked_source_plan(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_linked_success_one")
    _record(db, "evt_linked_success_one", "opp_linked_success_one")
    _event(db, "evt_linked_success_two", day=3, force_new_episode=True)
    _record(db, "evt_linked_success_two", "opp_linked_success_two")
    assert _knowledge(db)["state"] == "independently_demonstrated"

    _event(db, "evt_noneligible_source", day=6, force_new_episode=True)
    _record(
        db,
        "evt_noneligible_source",
        "opp_noneligible_source",
        actor="system",
        initiation="engine",
        response_mode="selection",
        source_type="system_generated",
        context_key="system-source-context",
    )
    _event(db, "evt_linked_target", day=9, force_new_episode=True)
    _record(
        db,
        "evt_linked_target",
        "opp_linked_target",
        transfer_from_plan_id="plan-opp_noneligible_source",
        context_key="different-target-context",
    )
    assert _knowledge(db)["state"] == "independently_demonstrated"


def test_required_per_concept_rubric_prevents_blanket_multi_concept_promotion(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    register_rubric(
        db,
        {
            "rubric_id": "rubric.phase-and-matter",
            "version": 1,
            "label": "Phase and matter explanation",
            "criteria": [
                {
                    "criterion_id": "criterion.phase",
                    "concept_id": "science.phase-change",
                    "description": "Identifies the phase change.",
                },
                {
                    "criterion_id": "criterion.matter",
                    "concept_id": "science.matter",
                    "description": "Explains that the material remains matter.",
                },
            ],
        },
    )
    register_rubric(
        db,
        {
            "rubric_id": "rubric.optional-only",
            "version": 1,
            "label": "Invalid optional-only target rubric",
            "criteria": [
                {
                    "criterion_id": "criterion.optional-matter",
                    "concept_id": "science.matter",
                    "description": "An optional note cannot establish target performance.",
                    "required": False,
                }
            ],
        },
    )
    optional_task = {
        **_task(),
        "task_id": "task.optional-only",
        "concepts": [{"concept_id": "science.matter", "role": "target"}],
        "rubric_id": "rubric.optional-only",
    }
    with pytest.raises(ValueError, match="require a score for every target concept"):
        register_learning_task(db, optional_task)
    register_learning_task(
        db,
        {
            "task_id": "task.phase-and-matter",
            "version": 1,
            "title": "Explain melting and matter",
            "learning_goal": "Distinguish phase change from disappearance of matter.",
            "strategy": "predict-test-explain",
            "difficulty": "familiar",
            "difficulty_provenance": "reviewed_task",
            "modalities": ["spoken"],
            "context_key": "ice-in-kitchen",
            "representation_key": "physical-observation",
            "concepts": [
                {"concept_id": "science.phase-change", "role": "target"},
                {"concept_id": "science.matter", "role": "target"},
            ],
            "rubric_id": "rubric.phase-and-matter",
            "rubric_version": 1,
        },
    )
    _event(db, "evt_multi_concept")
    _record(
        db,
        "evt_multi_concept",
        "opp_multi_concept",
        task_id="task.phase-and-matter",
        criterion_results=[
            {"criterion_id": "criterion.phase", "outcome": "correct", "score": 1},
            {"criterion_id": "criterion.matter", "outcome": "incorrect", "score": 0},
        ],
    )
    states = {
        item["concept_id"]: (item["state"], item["contradicting_count"])
        for item in learner_snapshot(db, "child-a")["knowledge"]
    }
    assert states == {
        "science.matter": ("exposure_recorded", 1),
        "science.phase-change": ("demonstrated_in_context", 0),
    }
    with connect(db) as connection:
        lineage = {
            row["concept_id"]: json.loads(row["scored_criteria_json"])[0]["criterion_id"]
            for row in connection.execute(
                """SELECT concept_id,scored_criteria_json FROM evidence_records
                   WHERE opportunity_id='opp_multi_concept' AND construct='knowledge_performance'"""
            )
        }
    assert lineage == {
        "science.matter": "criterion.matter",
        "science.phase-change": "criterion.phase",
    }

    _event(db, "evt_missing_criterion", day=1)
    with pytest.raises(ValueError, match="exactly match rubric"):
        _record(
            db,
            "evt_missing_criterion",
            "opp_missing_criterion",
            task_id="task.phase-and-matter",
            criterion_results=[{"criterion_id": "criterion.phase", "outcome": "correct"}],
        )


def test_uncalibrated_advanced_incorrect_result_cannot_demote_existing_state(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_baseline")
    _record(db, "evt_baseline", "opp_baseline")
    assert _knowledge(db)["state"] == "demonstrated_in_context"
    register_learning_task(db, _task(version=4, difficulty="advanced"))
    _event(db, "evt_advanced", day=3, force_new_episode=True)
    _record(db, "evt_advanced", "opp_advanced", task_version=4, result="incorrect")
    knowledge = _knowledge(db)
    assert knowledge["state"] == "demonstrated_in_context"
    assert knowledge["contradicting_count"] == 0
    with connect(db) as connection:
        row = connection.execute(
            """SELECT stance,projection_eligible FROM evidence_records
               WHERE opportunity_id='opp_advanced' AND construct='knowledge_performance'"""
        ).fetchone()
    assert dict(row) == {"stance": "context_only", "projection_eligible": 0}


def test_interest_is_separate_situational_recurrent_and_parent_confirmed(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_interest-one")
    _record(db, "evt_interest-one", "opp_interest-one", result="not_observed")
    assert _interest(db)["phase"] == "triggered_situational"

    _record(
        db,
        "evt_interest-one",
        "opp_interest-continued",
        result="not_observed",
        child_continued=True,
    )
    assert _interest(db)["phase"] == "maintained_situational"

    _event(db, "evt_interest-two", day=3, force_new_episode=True)
    _record(
        db,
        "evt_interest-two",
        "opp_interest-two",
        result="not_observed",
        context_key="outdoor-puddle",
        child_new_question=True,
    )
    assert _interest(db)["phase"] == "emerging_recurrent"

    _event(db, "evt_parent-confirms", day=6, force_new_episode=True)
    _record(
        db,
        "evt_parent-confirms",
        "opp_parent-confirms",
        actor="parent_report",
        response_mode="parent_report",
        source_type="parent_report",
        result="not_observed",
        retention_class="parent_confirmed",
    )
    assert _interest(db)["phase"] == "sustained_parent_confirmed"


def test_retention_expiry_is_deterministic_and_parent_confirmation_persists(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_situational")
    _record(db, "evt_situational", "opp_situational", result="not_observed")
    assert _interest(db)["phase"] == "triggered_situational"
    future = START + timedelta(days=100)
    assert learner_snapshot(db, "child-a", as_of=future)["interests"] == []

    _event(db, "evt_durable_interest", day=1)
    _record(
        db,
        "evt_durable_interest",
        "opp_durable_interest",
        actor="parent_report",
        response_mode="parent_report",
        source_type="parent_report",
        result="not_observed",
        retention_class="parent_confirmed",
    )
    durable = learner_snapshot(db, "child-a", as_of=START + timedelta(days=500))["interests"]
    assert durable[0]["phase"] == "sustained_parent_confirmed"

    _event(db, "evt_transient", day=2, force_new_episode=True)
    transient = _record(
        db,
        "evt_transient",
        "opp_transient",
        result="not_observed",
        retention_class="transient",
    )
    expired = expire_learning_records(db, as_of=START + timedelta(days=33))
    assert expired["expired_opportunities"] == 1
    assert transient["plan_id"]
    with connect(db) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM learning_opportunities WHERE opportunity_id='opp_transient'"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT status FROM learning_episode_plans WHERE plan_id=?", (transient["plan_id"],)
            ).fetchone()[0]
            == "cancelled"
        )


def test_as_of_excludes_future_observations_and_expiry_job_rebuilds_stored_interest(
    tmp_path: Path,
) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_temporal", day=3)
    _record(db, "evt_temporal", "opp_temporal", result="not_observed")
    before_observation = learner_snapshot(db, "child-a", as_of=START + timedelta(days=2))
    assert before_observation["knowledge"] == []
    assert before_observation["interests"] == []
    assert (
        learner_snapshot(db, "child-a", as_of=START + timedelta(days=4))["interests"][0]["phase"]
        == "triggered_situational"
    )

    expired = expire_learning_records(db, as_of=START + timedelta(days=94))
    assert expired["affected_children"] == ["child-a"]
    assert expired["expired_evidence_records"] == 1
    assert expired["expired_opportunities"] == 0
    with connect(db) as connection:
        assert (
            connection.execute(
                """SELECT phase FROM interest_projections
               WHERE child_id='child-a' AND concept_id='science.phase-change'"""
            ).fetchone()[0]
            == "not_observed"
        )
        assert (
            connection.execute(
                """SELECT COUNT(*) FROM derived_claims_v2
               WHERE child_id='child-a' AND construct='interest'"""
            ).fetchone()[0]
            == 0
        )
    repeated = expire_learning_records(db, as_of=START + timedelta(days=94))
    assert repeated["affected_children"] == []
    assert repeated["expired_evidence_records"] == 0

    _event(db, "evt_future_dated", day=30, force_new_episode=True)
    with pytest.raises(ValueError, match="future-dated"):
        _record(db, "evt_future_dated", "opp_future_dated")


def test_correction_restore_and_full_recompute_are_deterministic(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_correct")
    result = _record(db, "evt_correct", "opp_correct")
    before = learner_snapshot(db, "child-a")
    with connect(db) as connection:
        evidence_id = connection.execute(
            """SELECT evidence_id FROM evidence_records
               WHERE opportunity_id='opp_correct' AND construct='knowledge_performance'"""
        ).fetchone()[0]

    excluded = correct_evidence(
        db,
        child_id="child-a",
        evidence_id=evidence_id,
        action="exclude",
        reason="Synthetic parent exclusion.",
        correction_id="correction-exclude",
    )
    assert excluded["projection_digest"] != result["projection_digest"]
    assert _knowledge(db)["state"] == "exposure_recorded"
    correct_evidence(
        db,
        child_id="child-a",
        evidence_id=evidence_id,
        action="restore",
        reason="Synthetic parent restoration.",
        correction_id="correction-restore",
    )
    assert learner_snapshot(db, "child-a")["snapshot_digest"] == before["snapshot_digest"]
    first_rebuild = recompute_child_projections(db, "child-a")
    second_rebuild = recompute_child_projections(db, "child-a")
    assert first_rebuild == second_rebuild
    with pytest.raises(EvidenceConflict, match="different payload"):
        correct_evidence(
            db,
            child_id="child-a",
            evidence_id=evidence_id,
            action="exclude",
            reason="Different use of the same correction ID.",
            correction_id="correction-restore",
        )


def test_projector_digest_changes_when_rule_version_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_versioned_projection")
    _record(db, "evt_versioned_projection", "opp_versioned_projection")
    with connect(db) as connection:
        before = dict(
            connection.execute(
                """SELECT input_digest,rule_version FROM learner_state_projections
                   WHERE child_id='child-a' AND concept_id='science.phase-change'"""
            ).fetchone()
        )
    monkeypatch.setattr(learner_model_module, "KNOWLEDGE_RULE_VERSION", "1.0.1-test")
    recompute_child_projections(db, "child-a")
    with connect(db) as connection:
        after = dict(
            connection.execute(
                """SELECT input_digest,rule_version FROM learner_state_projections
                   WHERE child_id='child-a' AND concept_id='science.phase-change'"""
            ).fetchone()
        )
    assert after["input_digest"] != before["input_digest"]
    assert after["rule_version"] == "1.0.1-test"


def test_unresolved_contradiction_caps_cross_episode_promotion(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_contradict-one")
    _record(db, "evt_contradict-one", "opp_contradict-one")
    _event(db, "evt_contradict-two", day=3, force_new_episode=True)
    _record(db, "evt_contradict-two", "opp_contradict-two")
    assert _knowledge(db)["state"] == "independently_demonstrated"

    _event(db, "evt_contradiction", day=6, force_new_episode=True)
    _record(db, "evt_contradiction", "opp_contradiction", result="incorrect")
    knowledge = _knowledge(db)
    assert knowledge["state"] == "demonstrated_in_context"
    assert knowledge["contradicting_count"] == 1


def test_episode_correction_rebinds_lineage_and_recomputes_in_same_transaction(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    first_episode = _event(db, "evt_episode-one")
    _record(db, "evt_episode-one", "opp_episode-one")
    _event(db, "evt_episode-two", day=3, force_new_episode=True)
    _record(db, "evt_episode-two", "opp_episode-two")
    assert _knowledge(db)["state"] == "independently_demonstrated"

    correction = apply_episode_correction(
        db,
        child_id="child-a",
        event_id="evt_episode-two",
        action="retry",
        related_event_id="evt_episode-one",
    )
    assert correction["phase1_projection_digest"]
    assert _knowledge(db)["state"] == "demonstrated_in_context"
    with connect(db) as connection:
        opportunity = dict(
            connection.execute(
                """SELECT episode_id,independence_status FROM learning_opportunities
                   WHERE opportunity_id='opp_episode-two'"""
            ).fetchone()
        )
        evidence_episode = connection.execute(
            """SELECT DISTINCT episode_id FROM evidence_records
               WHERE opportunity_id='opp_episode-two'"""
        ).fetchone()[0]
    assert opportunity == {"episode_id": first_episode, "independence_status": "same_episode"}
    assert evidence_episode == first_episode
    assert _record(db, "evt_episode-two", "opp_episode-two")["duplicate"] is True


def test_deletion_cascades_source_lineage_leaves_no_raw_identifier_and_rebuilds(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_delete-one")
    _record(db, "evt_delete-one", "opp_delete-one")
    _event(db, "evt_delete-two", day=3, force_new_episode=True)
    _record(db, "evt_delete-two", "opp_delete-two")
    assert _knowledge(db)["state"] == "independently_demonstrated"
    with connect(db) as connection:
        corrected_evidence_id = connection.execute(
            """SELECT evidence_id FROM evidence_records
               WHERE opportunity_id='opp_delete-two' AND construct='knowledge_performance'"""
        ).fetchone()[0]
    correct_evidence(
        db,
        child_id="child-a",
        evidence_id=corrected_evidence_id,
        action="exclude",
        reason="Synthetic correction retained after source deletion.",
        correction_id="correction-before-delete",
    )

    deleted = delete_learning_source(db, child_id="child-a", opportunity_id="opp_delete-two")
    assert deleted["duplicate"] is False
    assert _knowledge(db)["state"] == "demonstrated_in_context"
    assert delete_learning_source(db, child_id="child-a", opportunity_id="opp_delete-two")["duplicate"] is True
    with connect(db) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM evidence_records WHERE opportunity_id='opp_delete-two'"
            ).fetchone()[0]
            == 0
        )
        tombstone = dict(connection.execute("SELECT * FROM evidence_source_tombstones").fetchone())
        correction = dict(
            connection.execute(
                "SELECT * FROM evidence_corrections WHERE correction_id='correction-before-delete'"
            ).fetchone()
        )
    assert "opp_delete-two" not in json.dumps(tombstone)
    assert "opp_delete-two" not in json.dumps(correction)
    assert len(tombstone["target_id_hash"]) == 64
    assert len(correction["target_evidence_hash"]) == 64
    with pytest.raises(EvidenceConflict, match="cannot accept|cannot be recreated"):
        _record(db, "evt_delete-two", "opp_delete-two")
    with pytest.raises(EvidenceConflict, match="cannot accept"):
        _record(
            db,
            "evt_delete-two",
            "opp_alternate_replay",
            plan_id="plan-opp_delete-two",
            plan_source_event_id="evt_delete-two",
        )


class _CapturingBackend(StubBackend):
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def complete(self, *, role, system, payload, response_model):
        self.payloads.append(payload)
        return super().complete(role=role, system=system, payload=payload, response_model=response_model)


def test_malformed_cross_child_episode_membership_cannot_leak_context(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    child_a_episode = _event(db, "evt_context_child_a", text="Synthetic child A context")
    _event(
        db,
        "evt_context_child_b",
        child_id="child-b",
        text="SENTINEL-CHILD-B-PRIVATE-CONTEXT",
    )
    with connect(db) as connection:
        connection.execute(
            "UPDATE episode_memberships SET episode_id=? WHERE event_id='evt_context_child_b'",
            (child_a_episode,),
        )
    context = build_context(
        str(db),
        "child-a",
        {"type": "child_question", "text": "What should we explore?", "metadata": {}},
        depth=1,
    )
    assert "SENTINEL-CHILD-B-PRIVATE-CONTEXT" not in json.dumps(context)
    assert "SENTINEL-CHILD-B-PRIVATE-CONTEXT" not in json.dumps(child_episode_context(db, "child-a"))


def test_context_has_hard_budget_and_shadow_snapshot_is_physically_removed_before_model(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_context-source")
    _record(db, "evt_context-source", "opp_context-source")
    huge_event = {
        "type": "child_question",
        "text": "Short question with oversized metadata",
        "metadata": {f"field-{index}": "y" * 2_000 for index in range(150)},
    }
    context = build_context(str(db), "child-a", huge_event, depth=4)
    serialized = json.dumps(context, ensure_ascii=False, sort_keys=True)
    assert len(serialized) <= MAX_CONTEXT_CHARS
    assert context["context_budget"]["truncated"] is True
    assert context["shadow_learner_snapshot"]["raw_evidence_included"] is False
    assert context["claims"] == []
    model_context, model_event, diagnostic = prepare_model_inputs(context, huge_event)
    assert (
        len(
            json.dumps(
                {"context": model_context, "event": model_event},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        <= MAX_CONTEXT_CHARS
    )
    assert model_event["metadata"] == {}
    assert diagnostic["passed_to_model"] is False

    backend = _CapturingBackend()
    result = CuriosityHarness(db, reasoning_engine=ReasoningEngine(backend)).dispatch(
        Event(
            id="evt_context-model",
            type="child_question",
            child_id="child-a",
            text="What happens to an ice cube?",
            metadata={f"field-{index}": "z" * 2_000 for index in range(150)},
            created_at=START + timedelta(days=10),
        )
    )
    contexts = [payload["context"] for payload in backend.payloads if "context" in payload]
    assert contexts
    assert all("shadow_learner_snapshot" not in item for item in contexts)
    assert all("profile_projection" not in item for item in contexts)
    assert all(
        len(
            json.dumps(
                {"context": payload["context"], "event": payload["event"]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        <= MAX_CONTEXT_CHARS
        for payload in backend.payloads
        if "context" in payload and "event" in payload
    )
    assert result.output["_reasoning"]["shadow_learner_snapshot"]["passed_to_model"] is False
    assert result.output["_reasoning"]["shadow_learner_snapshot"]["knowledge_count"] == 1


def test_director_uses_the_same_sanitized_model_input_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "db.sqlite"
    _setup(db)
    _event(db, "evt_director_source")
    _record(db, "evt_director_source", "opp_director_source")
    backend = _CapturingBackend()
    director = AutonomousDirector(db, reasoning=ReasoningEngine(backend))
    monkeypatch.setattr(director, "_context_proactivity_available", lambda: True)
    director.reflect_for_child("child-a", now=START + timedelta(days=10))
    payloads = [payload for payload in backend.payloads if "context" in payload and "event" in payload]
    assert payloads
    for payload in payloads:
        assert "shadow_learner_snapshot" not in payload["context"]
        assert "profile_projection" not in payload["context"]
        assert (
            len(
                json.dumps(
                    {"context": payload["context"], "event": payload["event"]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            <= MAX_CONTEXT_CHARS
        )
