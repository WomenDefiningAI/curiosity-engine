"""Historical Phase 6 research-authority matrix; excluded from the default beta gate."""

from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Event as ThreadEvent
from threading import Thread
from unittest.mock import patch

import pytest

from curiosity_engine.backups import BackupError, create_snapshot, restore_snapshot
from curiosity_engine.cli import build_parser
from curiosity_engine.contracts import Event
from curiosity_engine.db import (
    SCHEMA_VERSION,
    authority_hash,
    connect,
    init_db,
    jdump,
    research_authorization_snapshot_hash,
    research_authorization_snapshot_id,
    research_connect,
    research_control_event_id,
    research_policy_event_id,
)
from curiosity_engine.deletion_ledger import database_deletion_conflicts
from curiosity_engine.episodes import apply_episode_correction
from curiosity_engine.family_model import (
    correct_family_evidence,
    delete_family_model,
    export_family_model,
    inspect_family_model,
    set_collection_enabled,
)
from curiosity_engine.graph import add_child
from curiosity_engine.lab import (
    JUDGE_SYSTEM_PROMPT,
    discover,
    load_json,
    personalization_generation_route,
    validation_tier_report,
)
from curiosity_engine.learner_model import (
    record_learning_opportunity,
    register_concept,
    register_learning_plan,
    register_learning_task,
    register_rubric,
)
from curiosity_engine.personalization import PersonalizationService
from curiosity_engine.personalization_policy import promotion_manifest
from curiosity_engine.repository import Repository
from curiosity_engine.research_authority_ledger import ledger_path as research_authority_path
from curiosity_engine.research_policy import (
    RESEARCH_POLICY_DEFINITION_HASH,
    RESEARCH_POLICY_ID,
    RESEARCH_POLICY_VERSION,
    posterior_from_history,
)
from curiosity_engine.research_track import (
    ResearchTrackAuthorizationChanged,
    ResearchTrackService,
)


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "family.db"
    init_db(db)
    add_child(db, "child-a", "Synthetic Child", 2020, "1")
    return db


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _drop_research_triggers(conn: sqlite3.Connection) -> None:
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE '%research%'"
    ).fetchall():
        conn.execute(f'DROP TRIGGER "{name}"')


def _import_adequate_cases(service: ResearchTrackService, *, overlap_groups: bool = False) -> None:
    cases: list[dict[str, object]] = []
    for index in range(40):
        observed_success = index % 2
        cases.append(
            {
                "case_id": f"approved-{index:03d}",
                "split": "train" if index < 20 else "heldout",
                "group_digest": _digest(f"group-{index - 20 if overlap_groups and index >= 20 else index}"),
                "task_signature_digest": _digest(f"task-{index % 8}"),
                "domain": ("science", "math", "literacy")[index % 3],
                "history": {
                    "independent_successes": 8 if observed_success else 0,
                    "independent_failures": 0 if observed_success else 8,
                    "partial_results": 0,
                    "supported_successes": 0,
                    "supported_failures": 0,
                },
                "observed_success": observed_success,
                "source_digest": _digest(f"approved-source-{index}"),
            }
        )
    proposed = service.propose_calibration_corpus(
        corpus_id="privacy-review-1",
        cases=cases,
        operator_id="calibration-operator",
    )
    assert proposed["status"] == "pending_review"
    with pytest.raises(ValueError, match="differ"):
        service.approve_calibration_corpus(
            corpus_id="privacy-review-1",
            reviewer_id="calibration-operator",
            reason="The importer cannot approve the same corpus.",
        )
    approved = service.approve_calibration_corpus(
        corpus_id="privacy-review-1",
        reviewer_id="privacy-reviewer",
        reason="The exact deidentified manifest passed independent privacy review.",
    )
    assert approved["status"] == "approved"


def _passing_lab_report() -> dict[str, object]:
    required = {
        "golden",
        "curiosity",
        "context",
        "harness",
        "autonomy",
        "factual",
        "parent_effort",
        "safety",
        "pedagogy",
        "accessibility",
        "evidence_validity",
        "family_model",
        "adaptation",
        "personalization",
        "research_track",
        "artifacts",
        "visuals",
        "regressions",
    }
    judged = {
        "golden",
        "curiosity",
        "context",
        "factual",
        "parent_effort",
        "safety",
        "pedagogy",
        "accessibility",
        "evidence_validity",
    }
    suites: dict[str, dict[str, object]] = {}
    for name in required:
        judge = {"verdict": "pass", "score": 1.0, "reasons": []} if name in judged else {"status": "not_run"}
        suites[name] = {
            "cases": 2,
            "passed": 2,
            "failed": 0,
            "results": [
                {"id": f"{name}-one", "status": "pass", "failures": [], "judge": judge},
                {"id": f"{name}-two", "status": "pass", "failures": [], "judge": judge},
            ],
        }
    root = Path(__file__).resolve().parents[1]
    safety_cases = [
        (str(case.get("id") or ""), str(case.get("category") or ""))
        for case in discover(root)["safety"]["cases"]
    ]
    required_safety_categories = [
        "candidate_safety_boundary",
        "candidate_safety_contrast",
        "social_uncertainty",
    ]
    suites["safety"] = {
        "cases": len(safety_cases),
        "passed": len(safety_cases),
        "failed": 0,
        "results": [
            {
                "id": case_id,
                "category": category,
                "status": "pass",
                "failures": [],
                "judge": {"verdict": "pass", "score": 1.0, "reasons": []},
            }
            for case_id, category in safety_cases
        ],
    }
    suites["personalization"].update(
        {
            "candidate_generated_cases": 1,
            "candidate_judged_cases": 1,
            "candidate_behavior_effect_cases": 1,
            "candidate_generation_route": personalization_generation_route(),
            "results": [
                {
                    "id": "personalization-replay",
                    "status": "pass",
                    "failures": [],
                    "judge": {"status": "not_run"},
                },
                {
                    "id": "candidate-answer-generation",
                    "status": "pass",
                    "failures": [],
                    "judge": {"verdict": "pass", "score": 1.0, "reasons": []},
                },
            ],
        }
    )
    summary_cases = sum(int(suite["cases"]) for suite in suites.values())
    cases_judged = sum(int(suites[name]["cases"]) for name in judged) + 1
    policy = load_json(root / "configs" / "promotion-policy.json")
    return {
        "status": "pass",
        "summary": {
            "cases": summary_cases,
            "passed": summary_cases,
            "failed": 0,
            "minimum_cases": 20,
        },
        "live_judge": {
            "status": "pass",
            "cases_judged": cases_judged,
            "failed": 0,
            "error": None,
            "required_for_promotion": True,
            "identity": {
                "backend": "openai",
                "model": "independent-judge-model",
                "role": "judge",
                "prompt_digest": sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest(),
                "independent_from_generation": True,
            },
        },
        "auto_promote": False,
        "operator_approval_required": True,
        "promotion_eligible": True,
        "missing_required_suites": [],
        "required_safety_categories": required_safety_categories,
        "missing_required_safety_categories": [],
        "validation_tiers": validation_tier_report(policy, suites),
        "personalization_policy": promotion_manifest(root),
        "suites": suites,
    }


def _approve(service: ResearchTrackService) -> dict[str, object]:
    _import_adequate_cases(service)
    evaluation = service.evaluate()
    assert evaluation["status"] == "pass"
    with patch("curiosity_engine.lab.evaluate", return_value=_passing_lab_report()) as lab_evaluate:
        review = service.review_for_promotion(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            evaluation_id=str(evaluation["evaluation_id"]),
            reviewer_id="independent-reviewer",
        )
        lab_evaluate.assert_called_once_with(service.repo_root, live_judge=True)
        with pytest.raises(ValueError, match="separate"):
            service.approve(
                policy_id=RESEARCH_POLICY_ID,
                version=RESEARCH_POLICY_VERSION,
                evaluation_id=str(evaluation["evaluation_id"]),
                promotion_review_id=str(review["review_id"]),
                operator_id="independent-reviewer",
                reason="The same person must not approve.",
            )
        approval = service.approve(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            evaluation_id=str(evaluation["evaluation_id"]),
            promotion_review_id=str(review["review_id"]),
            operator_id="promotion-operator",
            reason="Approve the bounded, non-deliverable research track.",
        )
        assert approval["status"] == "approved"
    return evaluation


def _activate(service: ResearchTrackService) -> dict[str, object]:
    evaluation = _approve(service)
    activated = service.activate(
        policy_id=RESEARCH_POLICY_ID,
        version=RESEARCH_POLICY_VERSION,
        operator_id="promotion-operator",
        reason="Activate only for explicit family research opt-in.",
    )
    assert activated["candidate_delivery_enabled"] is False
    return evaluation


def _kill_after_authority_stage(
    db: Path,
    *,
    module: str,
    invocation: str,
    setup: str = "",
) -> None:
    script = f"""
import importlib
import os
import signal
import sys

target = importlib.import_module({module!r})
real_record = target.record_research_authority

def kill_after_stage(*args, **kwargs):
    real_record(*args, **kwargs)
    os.kill(os.getpid(), signal.SIGKILL)

target.record_research_authority = kill_after_stage
{setup}
{invocation}
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(db)],
        env=environment,
        check=False,
    )
    assert completed.returncode == -signal.SIGKILL


def _kill_during_authority_fanout(db: Path, *, boundary: str) -> None:
    script = f"""
import os
import signal
import sys

from curiosity_engine.research_track import ResearchTrackService
import curiosity_engine.research_authority_ledger as authority

service = ResearchTrackService(sys.argv[1])
local_path = authority.ledger_path(sys.argv[1])
real_write = authority._write

def kill_at_boundary(path, value):
    real_write(path, value)
    pending = value.get('pending') if isinstance(value, dict) else None
    if pending is None:
        return
    if {boundary!r} == 'first_root' and path.name == authority.ROOT_LEDGER_NAME:
        os.kill(os.getpid(), signal.SIGKILL)
    if {boundary!r} == 'local' and path == local_path:
        os.kill(os.getpid(), signal.SIGKILL)

authority._write = kill_at_boundary
service.set_family_opt_in(
    child_id='child-a',
    enabled=False,
    reason='Simulate process death during distributed revocation fanout.',
)
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(db)],
        env=environment,
        check=False,
    )
    assert completed.returncode == -signal.SIGKILL


def _kill_after_research_commit(
    db: Path,
    *,
    invocation: str,
) -> None:
    script = f"""
import json
import os
import signal
import sys

from curiosity_engine.research_track import ResearchTrackService
import curiosity_engine.db as database_module
import curiosity_engine.research_authority_ledger as authority_module

service = ResearchTrackService(sys.argv[1])
real_finalize = database_module.finalize_research_authority

def kill_before_finalize(path, *args, **kwargs):
    authority_path = authority_module.ledger_path(path)
    value = json.loads(authority_path.read_text(encoding='utf-8'))
    if value.get('pending') is not None:
        os.kill(os.getpid(), signal.SIGKILL)
    real_finalize(path, *args, **kwargs)

database_module.finalize_research_authority = kill_before_finalize
{invocation}
"""
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(db)],
        env=environment,
        check=False,
    )
    assert completed.returncode == -signal.SIGKILL


def _learning_history(db: Path, *, distinct_episodes: bool = True) -> None:
    register_concept(
        db,
        {
            "concept_id": "science.phase-change",
            "version": 1,
            "label": "Phase change",
            "domain": "science",
            "grade_band": "K-2",
        },
    )
    register_rubric(
        db,
        {
            "rubric_id": "rubric.phase-change",
            "version": 1,
            "label": "Phase change",
            "criteria": [
                {
                    "criterion_id": "criterion.phase-change",
                    "concept_id": "science.phase-change",
                    "concept_version": 1,
                    "description": "Names the observed phase change.",
                    "required": True,
                    "weight": 1,
                }
            ],
        },
    )
    register_learning_task(
        db,
        {
            "task_id": "task.phase-change",
            "version": 1,
            "title": "Observe melting",
            "learning_goal": "Explain an observable phase change.",
            "strategy": "predict-test-explain",
            "difficulty": "familiar",
            "difficulty_provenance": "reviewed_task",
            "modalities": ["spoken"],
            "context_key": "ice",
            "representation_key": "observation",
            "concepts": [{"concept_id": "science.phase-change", "version": 1, "role": "target"}],
            "rubric_id": "rubric.phase-change",
            "rubric_version": 1,
        },
    )
    for index in range(3):
        event_id = f"event-research-{index}" if distinct_episodes else "event-research-shared"
        if index == 0 or distinct_episodes:
            Repository(db).submit_event(
                Event(
                    id=event_id,
                    type="child_question",
                    child_id="child-a",
                    text=f"Why does ice melt? Observation {index + 1}.",
                    created_at=datetime.now(UTC),
                )
            )
        if index and distinct_episodes:
            apply_episode_correction(
                db,
                child_id="child-a",
                event_id=event_id,
                action="new_episode",
                note="This was a separate learning occasion.",
            )
        plan_id = f"plan-research-{index}"
        register_learning_plan(
            db,
            {
                "plan_id": plan_id,
                "child_id": "child-a",
                "source_event_id": event_id,
                "task_id": "task.phase-change",
                "task_version": 1,
            },
        )
        record_learning_opportunity(
            db,
            {
                "opportunity_id": f"opportunity-research-{index}",
                "child_id": "child-a",
                "event_id": event_id,
                "plan_id": plan_id,
                "actor": "child",
                "initiation": "child",
                "response_mode": "spoken",
                "assistance_level": "none",
                "criterion_results": [{"criterion_id": "criterion.phase-change", "outcome": "correct"}],
                "scorer": "reviewed-parent-observation",
                "scorer_version": "1",
                "source_type": "direct_observation",
                "retention_class": "learning_record",
                "child_continued": True,
            },
        )


def test_v26_defaults_to_a_closed_research_candidate(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    status = service.status("child-a")

    assert SCHEMA_VERSION == 26
    assert status["policies"][0]["status"] == "candidate"
    assert status["child_control"] is None
    assert status["trace_role"] == "advisory_only"
    assert status["delivery_enabled"] is False
    result = service.evaluate()
    assert result["status"] == "fail"
    assert result["auto_promote"] is False
    with pytest.raises(ValueError, match="active"):
        service.set_family_opt_in(child_id="child-a", enabled=True, reason="Try early.")


def test_built_wheel_initializes_without_a_source_checkout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    embedded_manifest = json.loads((root / "release" / "release-manifest.json").read_text(encoding="utf-8"))
    assert embedded_manifest == promotion_manifest(root)

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    build_source = tmp_path / "build-source"
    shutil.copytree(
        root,
        build_source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "*.egg-info",
            "__pycache__",
            "private",
            "output",
        ),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(build_source),
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheel = next(wheelhouse.glob("curiosity_engine-*.whl"))
    runtime = tmp_path / "runtime"
    subprocess.run(
        [sys.executable, "-m", "venv", str(runtime)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    runtime_python = runtime / "bin" / "python"
    subprocess.run(
        [str(runtime_python), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    runtime_site_packages = next((runtime / "lib").glob("python*/site-packages"))
    (runtime_site_packages / "test-runtime-dependencies.pth").write_text(
        str(sysconfig.get_paths()["purelib"]) + "\n", encoding="utf-8"
    )
    empty_working_directory = tmp_path / "outside-checkout"
    empty_working_directory.mkdir()
    environment = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "CURIOSITY_REPO_ROOT",
        "CURIOSITY_HOME",
        "CURIOSITY_PRIVATE_ROOT",
        "CURIOSITY_WORKSPACE",
    ):
        environment.pop(name, None)
    smoke_db = tmp_path / "installed-family.db"
    completed = subprocess.run(
        [
            str(runtime_python),
            "-c",
            (
                "from pathlib import Path; "
                "import curiosity_engine; import sys; "
                "from curiosity_engine.config import configuration_root; "
                "from curiosity_engine.db import init_db; "
                "assert Path(curiosity_engine.__file__).is_relative_to(Path(sys.prefix)); "
                "root=configuration_root(); "
                "assert (root/'evals/research_track/core.json').is_file(); "
                "init_db(Path(__import__('sys').argv[1])); print(root)"
            ),
            str(smoke_db),
        ],
        cwd=empty_working_directory,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "share/curiosity-engine" in completed.stdout
    assert smoke_db.is_file()

    foreign_checkout = tmp_path / "foreign-checkout"
    (foreign_checkout / "src" / "curiosity_engine").mkdir(parents=True)
    (foreign_checkout / "configs").mkdir()
    foreign_environment = {**environment, "CURIOSITY_REPO_ROOT": str(foreign_checkout)}
    mixed_root = subprocess.run(
        [
            str(runtime_python),
            "-c",
            "from curiosity_engine.personalization_policy import promotion_manifest; promotion_manifest()",
        ],
        cwd=foreign_checkout,
        env=foreign_environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert mixed_root.returncode != 0
    assert "does not own the running curiosity_engine package" in mixed_root.stderr


def test_v26_migrates_v25_with_backup_and_required_tables(tmp_path: Path) -> None:
    db = _db(tmp_path)
    tables = (
        "research_proactivity_candidates",
        "research_uncertainty_traces",
        "research_track_control_events",
        "research_track_controls",
        "research_track_authorization_snapshots",
        "research_track_policy_events",
        "research_track_promotion_reviews",
        "research_track_evaluations",
        "research_calibration_cases",
        "research_calibration_corpora",
        "research_track_policies",
    )
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        triggers = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE '%research%'"
        ).fetchall()
        for (name,) in triggers:
            conn.execute(f'DROP TRIGGER "{name}"')
        for table in tables:
            conn.execute(f'DROP TABLE "{table}"')
        conn.execute("DELETE FROM schema_migrations WHERE version=26")
        conn.execute("PRAGMA user_version=25")

    backup = init_db(db)
    assert backup is not None and backup.is_file()
    with connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 26
        present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert set(tables) <= present
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_synthetic_data_cannot_unlock_promotion(tmp_path: Path) -> None:
    service = ResearchTrackService(_db(tmp_path))
    for index in range(40):
        service.import_calibration_case(
            case_id=f"synthetic-{index}",
            source_kind="synthetic",
            license_status="synthetic",
            split="train" if index < 20 else "heldout",
            group_digest=_digest(f"group-{index}"),
            task_signature_digest=_digest(f"task-{index % 8}"),
            domain=("science", "math", "literacy")[index % 3],
            history={
                "independent_successes": 8 if index % 2 else 0,
                "independent_failures": 0 if index % 2 else 8,
                "partial_results": 0,
                "supported_successes": 0,
                "supported_failures": 0,
            },
            observed_success=index % 2,
            source_digest=_digest(f"source-{index}"),
            operator_id="operator",
        )
    evaluation = service.evaluate()
    assert evaluation["status"] == "fail"
    assert evaluation["real_case_count"] == 0
    assert evaluation["gates"]["synthetic_cases_do_not_unlock"] is False


def test_public_mechanical_fixtures_do_not_count_as_real_calibration(tmp_path: Path) -> None:
    service = ResearchTrackService(_db(tmp_path))
    for index in range(40):
        service.import_calibration_case(
            case_id=f"public-{index}",
            source_kind="public_fixture",
            license_status="public",
            split="train" if index < 20 else "heldout",
            group_digest=_digest(f"public-group-{index}"),
            task_signature_digest=_digest(f"public-task-{index % 8}"),
            domain=("science", "math", "literacy")[index % 3],
            history={
                "independent_successes": 8 if index % 2 else 0,
                "independent_failures": 0 if index % 2 else 8,
                "partial_results": 0,
                "supported_successes": 0,
                "supported_failures": 0,
            },
            observed_success=index % 2,
            source_digest=_digest(f"public-source-{index}"),
            operator_id="operator",
        )
    evaluation = service.evaluate()
    assert evaluation["status"] == "fail"
    assert evaluation["real_case_count"] == 0


def test_real_calibration_requires_an_independently_approved_exact_corpus(
    tmp_path: Path,
) -> None:
    service = ResearchTrackService(_db(tmp_path))
    with pytest.raises(ValueError, match="reviewed corpus"):
        service.import_calibration_case(
            case_id="claimed-real",
            source_kind="approved_deidentified",
            license_status="approved",
            privacy_approval_id="caller-claim",
            split="heldout",
            group_digest=_digest("group"),
            task_signature_digest=_digest("task"),
            domain="science",
            history={
                "independent_successes": 1,
                "independent_failures": 0,
                "partial_results": 0,
                "supported_successes": 0,
                "supported_failures": 0,
            },
            observed_success=1,
            source_digest=_digest("source"),
            operator_id="caller",
        )
    cases = [
        {
            "case_id": "pending-real",
            "split": "heldout",
            "group_digest": _digest("pending-group"),
            "task_signature_digest": _digest("pending-task"),
            "domain": "science",
            "history": {
                "independent_successes": 1,
                "independent_failures": 0,
                "partial_results": 0,
                "supported_successes": 0,
                "supported_failures": 0,
            },
            "observed_success": 1,
            "source_digest": _digest("pending-source"),
        }
    ]
    with pytest.raises(ValueError, match="exact deidentified schema"):
        service.propose_calibration_corpus(
            corpus_id="private-shaped-corpus",
            cases=[{**cases[0], "child_name": "must not be accepted"}],
            operator_id="corpus-proposer",
        )
    service.propose_calibration_corpus(
        corpus_id="pending-corpus",
        cases=cases,
        operator_id="corpus-proposer",
    )
    evaluation = service.evaluate()
    assert evaluation["real_case_count"] == 0
    service.approve_calibration_corpus(
        corpus_id="pending-corpus",
        reviewer_id="privacy-reviewer",
        reason="Approve the exact bounded manifest.",
    )
    assert service.evaluate()["real_case_count"] == 1


def test_v26_rejects_a_same_named_but_wrong_index_shape(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP INDEX idx_research_candidate_child")
        conn.execute("CREATE INDEX idx_research_candidate_child ON research_proactivity_candidates(status)")
    with pytest.raises(RuntimeError, match="invalid_indexes"):
        init_db(db)


def test_calibration_rows_and_evaluations_are_immutable(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _import_adequate_cases(service)
    evaluation = service.evaluate()
    assert evaluation["status"] == "pass"

    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE research_calibration_cases SET observed_success=0 WHERE case_id='approved-001'")
    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE research_track_evaluations SET status='fail' WHERE evaluation_id=?",
            (evaluation["evaluation_id"],),
        )


def test_approved_corpus_cases_cannot_be_deleted_or_evaluated_from_a_partial_manifest(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path / "guarded")
    service = ResearchTrackService(db)
    _import_adequate_cases(service)
    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError, match="approved"):
        conn.execute("DELETE FROM research_calibration_cases WHERE case_id='approved-000'")

    damaged_db = _db(tmp_path / "damaged")
    long_lived_service = ResearchTrackService(damaged_db)
    _import_adequate_cases(long_lived_service)
    with sqlite3.connect(damaged_db) as conn:
        conn.execute("DROP TRIGGER guard_research_calibration_case_delete")
        conn.execute("DELETE FROM research_calibration_cases WHERE case_id='approved-000'")
    with pytest.raises(RuntimeError, match="invalid_corpora"):
        long_lived_service.evaluate()


def test_evaluation_holds_one_write_snapshot_until_its_policy_binding_commits(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _import_adequate_cases(service)
    entered = ThreadEvent()
    release = ThreadEvent()
    mutation_done = ThreadEvent()
    failures: list[BaseException] = []

    from curiosity_engine import research_track as research_track_module

    canonical = research_track_module.canonical_research_evaluation

    def blocked_canonical(*args: object, **kwargs: object) -> dict[str, object]:
        entered.set()
        assert release.wait(5)
        return canonical(*args, **kwargs)

    def run_evaluation() -> None:
        try:
            service.evaluate()
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            failures.append(exc)

    def mutate_calibration() -> None:
        try:
            service.import_calibration_case(
                case_id="post-snapshot-synthetic",
                source_kind="synthetic",
                license_status="synthetic",
                split="heldout",
                group_digest=_digest("post-snapshot-group"),
                task_signature_digest=_digest("post-snapshot-task"),
                domain="science",
                history={
                    "independent_successes": 1,
                    "independent_failures": 0,
                    "partial_results": 0,
                    "supported_successes": 0,
                    "supported_failures": 0,
                },
                observed_success=1,
                source_digest=_digest("post-snapshot-source"),
                operator_id="calibration-operator",
            )
        except BaseException as exc:  # pragma: no cover - asserted in the parent thread
            failures.append(exc)
        finally:
            mutation_done.set()

    with patch(
        "curiosity_engine.research_track.canonical_research_evaluation",
        side_effect=blocked_canonical,
    ):
        evaluation_thread = Thread(target=run_evaluation)
        evaluation_thread.start()
        assert entered.wait(5)
        mutation_thread = Thread(target=mutate_calibration)
        mutation_thread.start()
        assert not mutation_done.wait(0.2)
        release.set()
        evaluation_thread.join(5)
        mutation_thread.join(5)
    assert not evaluation_thread.is_alive()
    assert not mutation_thread.is_alive()
    assert failures == []
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM research_track_evaluations").fetchone()[0] == 0
        assert conn.execute("SELECT status FROM research_track_policies").fetchone()[0] == "candidate"


def test_normal_sqlite_connections_cannot_fabricate_research_promotion(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError, match="invalid research evaluation"):
        conn.execute(
            """INSERT INTO research_track_evaluations(
                 evaluation_id,policy_id,policy_version,definition_hash,release_content_hash,
                 dataset_digest,case_count,real_case_count,train_count,heldout_count,
                 distinct_task_count,distinct_domain_count,heldout_positive_count,
                 heldout_negative_count,split_overlap_count,brier_score,
                 expected_calibration_error,gates_json,results_hash,evaluator,evaluator_digest,
                 status,created_at
               ) SELECT 'forged-evaluation',policy_id,policy_version,definition_hash,
                 release_content_hash,dataset_digest,case_count,real_case_count,train_count,
                 heldout_count,distinct_task_count,distinct_domain_count,heldout_positive_count,
                 heldout_negative_count,split_overlap_count,brier_score,
                 expected_calibration_error,gates_json,results_hash,evaluator,evaluator_digest,
                 status,created_at FROM research_track_evaluations LIMIT 1"""
        )
    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError, match="promotion review"):
        conn.execute(
            """INSERT INTO research_track_promotion_reviews(
                 review_id,policy_id,policy_version,definition_hash,release_content_hash,
                 evaluation_id,reviewer_id,lab_report_json,lab_report_hash,created_at
               ) SELECT 'forged-review',policy_id,policy_version,definition_hash,
                 release_content_hash,evaluation_id,'forged-reviewer',lab_report_json,
                 lab_report_hash,created_at FROM research_track_promotion_reviews LIMIT 1"""
        )
    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError, match="research policy"):
        conn.execute("UPDATE research_track_policies SET status='disabled' WHERE status='active'")
    with pytest.raises(ValueError, match="frozen"):
        service.import_calibration_case(
            case_id="late-synthetic",
            source_kind="synthetic",
            license_status="synthetic",
            split="heldout",
            group_digest=_digest("late-group"),
            task_signature_digest=_digest("late-task"),
            domain="science",
            history={
                "independent_successes": 1,
                "independent_failures": 0,
                "partial_results": 0,
                "supported_successes": 0,
                "supported_failures": 0,
            },
            observed_success=1,
            source_digest=_digest("late-source"),
            operator_id="operator",
        )


def test_reopen_and_runtime_reject_a_structurally_forged_active_chain(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    release_hash = str(promotion_manifest()["promotion_content_hash"])
    gates = {
        "minimum_real_cases": True,
        "minimum_train_cases": True,
        "minimum_heldout_cases": True,
        "task_diversity": True,
        "domain_diversity": True,
        "heldout_outcome_balance": True,
        "split_group_independence": True,
        "brier_score": True,
        "expected_calibration_error": True,
        "synthetic_cases_do_not_unlock": True,
    }
    report = {
        "status": "pass",
        "promotion_eligible": True,
        "live_judge": {
            "status": "pass",
            "required_for_promotion": True,
            "identity": {"independent_from_generation": True},
        },
        "personalization_policy": {"promotion_content_hash": release_hash},
        "suites": {"research_track": {"cases": 1, "passed": 1, "failed": 0}},
    }
    report_json = jdump(report)
    report_hash = sha256(report_json.encode()).hexdigest()
    evaluation_id = "forged-evaluation"
    review_identity = {
        "policy_id": RESEARCH_POLICY_ID,
        "policy_version": RESEARCH_POLICY_VERSION,
        "definition_hash": RESEARCH_POLICY_DEFINITION_HASH,
        "release_content_hash": release_hash,
        "evaluation_id": evaluation_id,
        "reviewer_id": "forged-reviewer",
        "lab_report_hash": report_hash,
    }
    review_id = "rtreview_" + sha256(jdump(review_identity).encode()).hexdigest()[:24]
    approved_reason = "Fabricated operator approval."
    activation_reason = "Fabricated activation."
    created_at = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(db) as conn:
        _drop_research_triggers(conn)
        conn.execute(
            """INSERT INTO research_track_evaluations(
                 evaluation_id,policy_id,policy_version,definition_hash,release_content_hash,
                 dataset_digest,case_count,real_case_count,train_count,heldout_count,
                 distinct_task_count,distinct_domain_count,heldout_positive_count,
                 heldout_negative_count,split_overlap_count,brier_score,
                 expected_calibration_error,gates_json,results_hash,evaluator,evaluator_digest,
                 status,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pass',?)""",
            (
                evaluation_id,
                RESEARCH_POLICY_ID,
                RESEARCH_POLICY_VERSION,
                RESEARCH_POLICY_DEFINITION_HASH,
                release_hash,
                _digest("forged-dataset"),
                40,
                40,
                20,
                20,
                8,
                3,
                10,
                10,
                0,
                0.01,
                0.01,
                jdump(gates),
                _digest("forged-results"),
                "heldout-calibration-v1",
                _digest("forged-evaluator"),
                created_at,
            ),
        )
        conn.execute(
            """INSERT INTO research_track_promotion_reviews(
                 review_id,policy_id,policy_version,definition_hash,release_content_hash,
                 evaluation_id,reviewer_id,lab_report_json,lab_report_hash,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                review_id,
                RESEARCH_POLICY_ID,
                RESEARCH_POLICY_VERSION,
                RESEARCH_POLICY_DEFINITION_HASH,
                release_hash,
                evaluation_id,
                "forged-reviewer",
                report_json,
                report_hash,
                created_at,
            ),
        )
        conn.execute(
            """UPDATE research_track_policies
               SET release_content_hash=?,authorization_epoch=1,status='active',evaluation_id=?,
                   promotion_review_id=?,approved_by_operator='forged-operator',
                   approval_reason=?,activation_reason=?,approved_at=?,activated_at=?,updated_at=?""",
            (
                release_hash,
                evaluation_id,
                review_id,
                approved_reason,
                activation_reason,
                created_at,
                created_at,
                created_at,
            ),
        )
        event_specs = (
            (
                "evaluated",
                "system",
                None,
                f"Held-out research calibration {evaluation_id}: pass.",
            ),
            (
                "promotion_reviewed",
                "system",
                None,
                f"Independent full Lab report {report_hash[:16]} reviewed by forged-reviewer.",
            ),
            ("approved", "operator", "forged-operator", approved_reason),
            ("activated", "operator", "forged-operator", activation_reason),
        )
        for action, actor_type, operator_id, reason in event_specs:
            conn.execute(
                """INSERT INTO research_track_policy_events(
                     event_id,policy_id,policy_version,action,actor_type,operator_id,reason,created_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    research_policy_event_id(
                        RESEARCH_POLICY_ID,
                        RESEARCH_POLICY_VERSION,
                        action,
                        actor_type,
                        operator_id,
                        reason,
                        created_at,
                    ),
                    RESEARCH_POLICY_ID,
                    RESEARCH_POLICY_VERSION,
                    action,
                    actor_type,
                    operator_id,
                    reason,
                    created_at,
                ),
            )
        conn.execute(
            """INSERT INTO learner_model_controls(
                 child_id,collection_enabled,default_retention_class,updated_by,created_at,updated_at
               ) VALUES('child-a',1,'learning_record','parent',?,?)""",
            (created_at, created_at),
        )
        control_reason = "Fabricated family opt-in."
        control_event_id = research_control_event_id(
            "child-a",
            True,
            RESEARCH_POLICY_ID,
            RESEARCH_POLICY_VERSION,
            RESEARCH_POLICY_DEFINITION_HASH,
            release_hash,
            1,
            "parent",
            control_reason,
            created_at,
        )
        conn.execute(
            """INSERT INTO research_track_controls(
                 child_id,enabled,policy_id,policy_version,definition_hash,release_content_hash,
                 family_authorization_epoch,updated_by,reason,created_at,updated_at
               ) VALUES('child-a',1,?,?,?,?,1,'parent',?,?,?)""",
            (
                RESEARCH_POLICY_ID,
                RESEARCH_POLICY_VERSION,
                RESEARCH_POLICY_DEFINITION_HASH,
                release_hash,
                control_reason,
                created_at,
                created_at,
            ),
        )
        conn.execute(
            """INSERT INTO research_track_control_events(
                 event_id,child_id,enabled,policy_id,policy_version,definition_hash,
                 release_content_hash,family_authorization_epoch,actor_type,reason,created_at
               ) VALUES(?,'child-a',1,?,?,?,?,1,'parent',?,?)""",
            (
                control_event_id,
                RESEARCH_POLICY_ID,
                RESEARCH_POLICY_VERSION,
                RESEARCH_POLICY_DEFINITION_HASH,
                release_hash,
                control_reason,
                created_at,
            ),
        )

    with pytest.raises(
        (ValueError, RuntimeError),
        match="canonically validated|external authorization authority",
    ):
        service.set_family_opt_in(
            child_id="child-a",
            enabled=True,
            reason="A fabricated chain must not authorize consent.",
        )
    with pytest.raises(
        RuntimeError,
        match="external authorization authority|invalid_evaluations",
    ):
        ResearchTrackService(db)


def test_reopen_rejects_enabled_control_without_parent_event(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    release_hash = service.status()["current_release_content_hash"]
    created_at = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(db) as conn:
        _drop_research_triggers(conn)
        conn.execute(
            """INSERT INTO learner_model_controls(
                 child_id,collection_enabled,default_retention_class,updated_by,created_at,updated_at
               ) VALUES('child-a',1,'learning_record','parent',?,?)""",
            (created_at, created_at),
        )
        conn.execute(
            """INSERT INTO research_track_controls(
                 child_id,enabled,policy_id,policy_version,definition_hash,release_content_hash,
                 family_authorization_epoch,updated_by,reason,created_at,updated_at
               ) VALUES('child-a',1,?,?,?,?,1,'parent','Forged consent.',?,?)""",
            (
                RESEARCH_POLICY_ID,
                RESEARCH_POLICY_VERSION,
                RESEARCH_POLICY_DEFINITION_HASH,
                release_hash,
                created_at,
                created_at,
            ),
        )
    with pytest.raises(
        RuntimeError,
        match="external authorization authority|invalid_controls",
    ):
        ResearchTrackService(db)


def test_calibration_is_frozen_after_operator_approval(tmp_path: Path) -> None:
    service = ResearchTrackService(_db(tmp_path))
    _import_adequate_cases(service)
    service.propose_calibration_corpus(
        corpus_id="pending-after-approval",
        cases=[
            {
                "case_id": "pending-after-approval-001",
                "split": "heldout",
                "group_digest": _digest("pending-after-approval-group"),
                "task_signature_digest": _digest("pending-after-approval-task"),
                "domain": "science",
                "history": {
                    "independent_successes": 1,
                    "independent_failures": 0,
                    "partial_results": 0,
                    "supported_successes": 0,
                    "supported_failures": 0,
                },
                "observed_success": 1,
                "source_digest": _digest("pending-after-approval-source"),
            }
        ],
        operator_id="second-corpus-proposer",
    )
    evaluation = service.evaluate()
    with patch("curiosity_engine.lab.evaluate", return_value=_passing_lab_report()):
        review = service.review_for_promotion(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            evaluation_id=str(evaluation["evaluation_id"]),
            reviewer_id="independent-reviewer",
        )
        service.approve(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            evaluation_id=str(evaluation["evaluation_id"]),
            promotion_review_id=str(review["review_id"]),
            operator_id="promotion-operator",
            reason="Freeze the exact approved calibration authority.",
        )
    with pytest.raises(ValueError, match="frozen"):
        service.import_calibration_case(
            case_id="late-synthetic",
            source_kind="synthetic",
            license_status="synthetic",
            split="heldout",
            group_digest=_digest("late-group"),
            task_signature_digest=_digest("late-task"),
            domain="science",
            history={
                "independent_successes": 1,
                "independent_failures": 0,
                "partial_results": 0,
                "supported_successes": 0,
                "supported_failures": 0,
            },
            observed_success=1,
            source_digest=_digest("late-source"),
            operator_id="operator",
        )
    with pytest.raises(ValueError, match="frozen"):
        service.approve_calibration_corpus(
            corpus_id="pending-after-approval",
            reviewer_id="second-privacy-reviewer",
            reason="This mutation is too late.",
        )
    activated = service.activate(
        policy_id=RESEARCH_POLICY_ID,
        version=RESEARCH_POLICY_VERSION,
        operator_id="promotion-operator",
        reason="The unchanged approved authority remains activatable.",
    )
    assert activated["status"] == "active"


@pytest.mark.parametrize(
    "table",
    (
        "research_uncertainty_traces",
        "research_proactivity_candidates",
        "research_track_controls",
        "research_track_control_events",
    ),
)
def test_backup_conflicts_include_each_phase6_child_model_table(tmp_path: Path, table: str) -> None:
    database = tmp_path / f"{table}.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE children(id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE learner_model_controls(child_id TEXT,collection_enabled INTEGER,updated_at TEXT)")
        conn.execute(f'CREATE TABLE "{table}"(child_id TEXT)')
        conn.execute("INSERT INTO children VALUES('child-a')")
        conn.execute("INSERT INTO learner_model_controls VALUES('child-a',0,'2026-01-01T00:00:00+00:00')")
        conn.execute(f"INSERT INTO \"{table}\" VALUES('child-a')")
    manifest = {
        "events": {},
        "transport_events": {},
        "learning_sources": {},
        "learner_models": {_digest("child-a"): {"deleted_at": "2026-01-01T00:00:00+00:00"}},
        "parent_memories": {},
    }
    assert "learner_model" in database_deletion_conflicts(
        database,
        manifest,
        current_database=True,
        require_curiosity_schema=False,
        require_database_identity=False,
    )


def test_group_overlap_and_missing_research_lab_suite_block_review(tmp_path: Path) -> None:
    service = ResearchTrackService(_db(tmp_path))
    _import_adequate_cases(service, overlap_groups=True)
    failed = service.evaluate()
    assert failed["status"] == "fail"
    assert failed["split_overlap_count"] == 20
    assert failed["gates"]["split_group_independence"] is False

    db = tmp_path / "separate.db"
    init_db(db)
    service = ResearchTrackService(db)
    _import_adequate_cases(service)
    passed = service.evaluate()
    with pytest.raises(TypeError, match="lab_report"):
        service.review_for_promotion(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            evaluation_id=str(passed["evaluation_id"]),
            reviewer_id="independent-reviewer",
            lab_report={},  # type: ignore[call-arg]
        )
    with (
        patch("curiosity_engine.lab.evaluate", return_value={"suites": {}}),
        patch.object(PersonalizationService, "_validate_lab_report", return_value=None),
    ):
        with pytest.raises(ValueError, match="research-track suite"):
            service.review_for_promotion(
                policy_id=RESEARCH_POLICY_ID,
                version=RESEARCH_POLICY_VERSION,
                evaluation_id=str(passed["evaluation_id"]),
                reviewer_id="independent-reviewer",
            )


def test_trace_is_exact_advisory_and_proactivity_never_delivers(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    with pytest.raises(ValueError, match="only a family parent"):
        service.set_family_opt_in(
            child_id="child-a",
            enabled=True,
            reason="Operator cannot supply consent.",
            actor_type="operator",
        )
    control = service.set_family_opt_in(
        child_id="child-a",
        enabled=True,
        reason="Parent opts into this bounded research track.",
    )
    assert control["enabled"] is True
    _learning_history(db)
    with connect(db) as conn:
        before_projection = dict(
            conn.execute(
                """SELECT * FROM learner_state_projections
                   WHERE child_id='child-a' AND concept_id='science.phase-change'"""
            ).fetchone()
        )
        delivery_before = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "opportunities",
                "delivery_outbox",
                "slack_file_outbox",
                "artifact_file_outbox",
            )
        }

    trace = service.compute_uncertainty_trace(
        child_id="child-a",
        concept_id="science.phase-change",
    )
    assert trace["advisory_only"] is True
    assert trace["may_authorize_state_or_delivery"] is False
    assert len(trace["evidence_ids"]) >= 3
    candidate = service.create_proactivity_candidate(
        child_id="child-a",
        trace_id=str(trace["trace_id"]),
    )
    assert candidate is not None
    assert candidate["status"] == "candidate_only"
    assert candidate["delivery_enabled"] is False
    assert candidate["delivery_table_counts"] == delivery_before
    with connect(db) as conn, pytest.raises(sqlite3.IntegrityError, match="invalid research proactivity candidate"):
        conn.execute(
            """INSERT INTO research_proactivity_candidates(
                 candidate_id,candidate_hash,child_id,trace_id,concept_id,concept_version,
                 interest_input_digest,interest_rule_id,interest_rule_version,interest_phase,
                 policy_id,policy_version,definition_hash,family_authorization_epoch,
                 policy_authorization_epoch,title,rationale,payload_json,status,created_at
               )
               SELECT 'forged-candidate',?,child_id,trace_id,concept_id,concept_version,
                 interest_input_digest,interest_rule_id,interest_rule_version,interest_phase,
                 policy_id,policy_version,definition_hash,family_authorization_epoch,
                 policy_authorization_epoch,title,rationale,payload_json,status,created_at
               FROM research_proactivity_candidates WHERE candidate_id=?""",
            ("0" * 64, candidate["candidate_id"]),
        )
    with connect(db) as conn:
        after_projection = dict(
            conn.execute(
                """SELECT * FROM learner_state_projections
                   WHERE child_id='child-a' AND concept_id='science.phase-change'"""
            ).fetchone()
        )
    assert after_projection == before_projection

    Repository(db).submit_event(
        Event(
            id="event-research-3",
            type="child_question",
            child_id="child-a",
            text="What happened to the ice on another day?",
            created_at=datetime.now(UTC),
        )
    )
    apply_episode_correction(
        db,
        child_id="child-a",
        event_id="event-research-3",
        action="new_episode",
        note="This was a separate learning occasion.",
    )
    register_learning_plan(
        db,
        {
            "plan_id": "plan-research-3",
            "child_id": "child-a",
            "source_event_id": "event-research-3",
            "task_id": "task.phase-change",
            "task_version": 1,
        },
    )
    record_learning_opportunity(
        db,
        {
            "opportunity_id": "opportunity-research-3",
            "child_id": "child-a",
            "event_id": "event-research-3",
            "plan_id": "plan-research-3",
            "actor": "child",
            "initiation": "child",
            "response_mode": "spoken",
            "assistance_level": "none",
            "criterion_results": [{"criterion_id": "criterion.phase-change", "outcome": "correct"}],
            "scorer": "reviewed-parent-observation",
            "scorer_version": "1",
            "source_type": "direct_observation",
            "retention_class": "learning_record",
            "child_continued": True,
        },
    )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM research_uncertainty_traces").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM research_proactivity_candidates").fetchone()[0] == 0


def test_uncertainty_collapses_attempts_per_episode_and_uses_parent_corrections(
    tmp_path: Path,
) -> None:
    repeated_db = _db(tmp_path / "repeated")
    repeated = ResearchTrackService(repeated_db)
    _activate(repeated)
    repeated.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent enables the bounded research trial.")
    _learning_history(repeated_db, distinct_episodes=False)
    with pytest.raises(ValueError, match="three distinct"):
        repeated.compute_uncertainty_trace(child_id="child-a", concept_id="science.phase-change")

    corrected_db = _db(tmp_path / "corrected")
    corrected = ResearchTrackService(corrected_db)
    _activate(corrected)
    corrected.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent enables the bounded research trial.")
    _learning_history(corrected_db)
    original = corrected.compute_uncertainty_trace(child_id="child-a", concept_id="science.phase-change")
    assert original["history"]["independent_successes"] == 3
    with connect(corrected_db) as conn:
        corrected_evidence_id = str(
            conn.execute(
                """SELECT evidence_id FROM research_effective_knowledge_evidence
                   WHERE child_id='child-a' AND concept_id='science.phase-change'
                   ORDER BY evidence_id LIMIT 1"""
            ).fetchone()[0]
        )
    correct_family_evidence(
        corrected_db,
        child_id="child-a",
        evidence_id=corrected_evidence_id,
        action="correct",
        replacement={"result": "incorrect", "assistance_level": "none"},
        reason="Parent corrected the observed outcome.",
    )
    with connect(corrected_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM research_uncertainty_traces").fetchone()[0] == 0
    revised = corrected.compute_uncertainty_trace(child_id="child-a", concept_id="science.phase-change")
    assert revised["history"]["independent_successes"] == 2
    assert revised["history"]["independent_failures"] == 1
    assert revised["posterior"]["posterior_mean"] < original["posterior"]["posterior_mean"]


def test_database_rejects_forged_trace_and_candidate(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with connect(db) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="invalid research uncertainty trace"):
            conn.execute(
                """INSERT INTO research_uncertainty_traces(
                     trace_id,trace_hash,child_id,concept_id,concept_version,construct,
                     projection_input_digest,projection_rule_id,projection_rule_version,
                     policy_id,policy_version,definition_hash,evaluation_id,promotion_review_id,
                     family_authorization_epoch,policy_authorization_epoch,evidence_ids_json,
                     evidence_digest,history_json,alpha,beta,posterior_mean,variance,
                     interval_90_lower,interval_90_upper,uncertainty,method_version,
                     advisory_only,rationale,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    "forged-trace",
                    "0" * 64,
                    "child-a",
                    "science.fake",
                    1,
                    "knowledge_performance",
                    "0" * 64,
                    "rule",
                    "1",
                    RESEARCH_POLICY_ID,
                    RESEARCH_POLICY_VERSION,
                    RESEARCH_POLICY_DEFINITION_HASH,
                    "evaluation",
                    "review",
                    0,
                    0,
                    "[]",
                    _digest("[]"),
                    jdump(
                        {
                            "independent_failures": 0,
                            "independent_successes": 0,
                            "partial_results": 0,
                            "supported_failures": 0,
                            "supported_successes": 0,
                        }
                    ),
                    1.0,
                    1.0,
                    0.5,
                    1 / 12,
                    0.0,
                    1.0,
                    1.0,
                    "beta-evidence-companion-v1",
                    1,
                    "forged",
                    datetime.now(UTC).isoformat(),
                ),
            )


def test_database_rejects_self_hashed_rows_with_false_authoritative_semantics(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent enables the bounded research trial.")
    _learning_history(db)
    trace_result = service.compute_uncertainty_trace(child_id="child-a", concept_id="science.phase-change")
    candidate_result = service.create_proactivity_candidate(child_id="child-a", trace_id=str(trace_result["trace_id"]))
    assert candidate_result is not None
    with connect(db) as conn:
        trace = dict(
            conn.execute(
                "SELECT * FROM research_uncertainty_traces WHERE trace_id=?",
                (trace_result["trace_id"],),
            ).fetchone()
        )
        candidate = dict(
            conn.execute(
                "SELECT * FROM research_proactivity_candidates WHERE candidate_id=?",
                (candidate_result["candidate_id"],),
            ).fetchone()
        )

    false_history = {
        "independent_successes": 0,
        "independent_failures": 3,
        "partial_results": 0,
        "supported_successes": 0,
        "supported_failures": 0,
    }
    false_history_json = jdump(false_history)
    false_posterior = posterior_from_history(false_history)
    trace_authority = (
        trace["child_id"],
        trace["concept_id"],
        trace["concept_version"],
        trace["construct"],
        trace["projection_input_digest"],
        trace["projection_rule_id"],
        trace["projection_rule_version"],
        trace["policy_id"],
        trace["policy_version"],
        trace["definition_hash"],
        trace["release_content_hash"],
        trace["evaluation_id"],
        trace["promotion_review_id"],
        trace["authorization_snapshot_id"],
        trace["family_authorization_epoch"],
        trace["policy_authorization_epoch"],
        trace["evidence_digest"],
        false_history_json,
        false_posterior["alpha"],
        false_posterior["beta"],
        false_posterior["posterior_mean"],
        false_posterior["variance"],
        false_posterior["interval_90_lower"],
        false_posterior["interval_90_upper"],
        false_posterior["uncertainty"],
        false_posterior["method"],
        1,
        "Self-hashed but semantically false trace.",
    )
    with (
        research_connect(db) as conn,
        pytest.raises(sqlite3.IntegrityError, match="invalid research uncertainty trace"),
    ):
        conn.execute(
            """INSERT INTO research_uncertainty_traces(
                 trace_id,trace_hash,child_id,concept_id,concept_version,construct,
                 projection_input_digest,projection_rule_id,projection_rule_version,
                 policy_id,policy_version,definition_hash,release_content_hash,evaluation_id,
                 promotion_review_id,authorization_snapshot_id,
                 family_authorization_epoch,policy_authorization_epoch,
                 evidence_ids_json,evidence_digest,history_json,alpha,beta,posterior_mean,variance,
                 interval_90_lower,interval_90_upper,uncertainty,method_version,advisory_only,
                 rationale,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "rtrace_" + sha256(authority_hash(*trace_authority).encode()).hexdigest()[:24],
                authority_hash(*trace_authority),
                *trace_authority[:16],
                trace["evidence_ids_json"],
                *trace_authority[16:],
                datetime.now(UTC).isoformat(),
            ),
        )

    false_payload = dict(candidate_result["payload"])
    false_payload["posterior_mean"] = 0.123
    false_payload_json = jdump(false_payload)
    candidate_authority = (
        candidate["child_id"],
        candidate["trace_id"],
        candidate["concept_id"],
        candidate["concept_version"],
        candidate["interest_input_digest"],
        candidate["interest_rule_id"],
        candidate["interest_rule_version"],
        candidate["interest_phase"],
        candidate["policy_id"],
        candidate["policy_version"],
        candidate["definition_hash"],
        candidate["release_content_hash"],
        candidate["family_authorization_epoch"],
        candidate["policy_authorization_epoch"],
        candidate["title"],
        candidate["rationale"],
        false_payload_json,
        "candidate_only",
    )
    with (
        research_connect(db) as conn,
        pytest.raises(sqlite3.IntegrityError, match="invalid research proactivity candidate"),
    ):
        conn.execute(
            """INSERT INTO research_proactivity_candidates(
                 candidate_id,candidate_hash,child_id,trace_id,concept_id,concept_version,
                 interest_input_digest,interest_rule_id,interest_rule_version,interest_phase,
                 policy_id,policy_version,definition_hash,release_content_hash,
                 family_authorization_epoch,policy_authorization_epoch,title,rationale,payload_json,
                 status,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                "rtcand_" + sha256(authority_hash(*candidate_authority).encode()).hexdigest()[:24],
                authority_hash(*candidate_authority),
                *candidate_authority,
                datetime.now(UTC).isoformat(),
            ),
        )


def test_reopen_rejects_self_hashed_trace_lineage_and_candidate_copy_tampering(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent enables the bounded research trial.")
    _learning_history(db)
    trace_result = service.compute_uncertainty_trace(child_id="child-a", concept_id="science.phase-change")
    candidate_result = service.create_proactivity_candidate(child_id="child-a", trace_id=str(trace_result["trace_id"]))
    assert candidate_result is not None

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        _drop_research_triggers(conn)
        trace = dict(
            conn.execute(
                "SELECT * FROM research_uncertainty_traces WHERE trace_id=?",
                (trace_result["trace_id"],),
            ).fetchone()
        )
        evaluation = dict(
            conn.execute(
                "SELECT * FROM research_track_evaluations WHERE evaluation_id=?",
                (trace["evaluation_id"],),
            ).fetchone()
        )
        historical_release = "1" * 64
        historical_identity = {
            "policy_id": evaluation["policy_id"],
            "policy_version": evaluation["policy_version"],
            "definition_hash": evaluation["definition_hash"],
            "release_content_hash": historical_release,
            "dataset_digest": evaluation["dataset_digest"],
            "results_hash": evaluation["results_hash"],
            "evaluator": evaluation["evaluator"],
            "evaluator_digest": evaluation["evaluator_digest"],
        }
        historical_evaluation_id = "rteval_" + sha256(jdump(historical_identity).encode()).hexdigest()[:24]
        conn.execute(
            """INSERT INTO research_track_evaluations(
                 evaluation_id,policy_id,policy_version,definition_hash,release_content_hash,
                 dataset_digest,case_count,real_case_count,train_count,heldout_count,
                 distinct_task_count,distinct_domain_count,heldout_positive_count,
                 heldout_negative_count,split_overlap_count,brier_score,
                 expected_calibration_error,gates_json,results_hash,evaluator,evaluator_digest,
                 status,created_at
               ) SELECT ?,policy_id,policy_version,definition_hash,?,dataset_digest,case_count,
                 real_case_count,train_count,heldout_count,distinct_task_count,distinct_domain_count,
                 heldout_positive_count,heldout_negative_count,split_overlap_count,brier_score,
                 expected_calibration_error,gates_json,results_hash,evaluator,evaluator_digest,
                 status,created_at FROM research_track_evaluations WHERE evaluation_id=?""",
            (historical_evaluation_id, historical_release, trace["evaluation_id"]),
        )
        trace["evaluation_id"] = historical_evaluation_id
        trace_authority = (
            trace["child_id"],
            trace["concept_id"],
            trace["concept_version"],
            trace["construct"],
            trace["projection_input_digest"],
            trace["projection_rule_id"],
            trace["projection_rule_version"],
            trace["policy_id"],
            trace["policy_version"],
            trace["definition_hash"],
            trace["release_content_hash"],
            trace["evaluation_id"],
            trace["promotion_review_id"],
            trace["authorization_snapshot_id"],
            trace["family_authorization_epoch"],
            trace["policy_authorization_epoch"],
            trace["evidence_digest"],
            trace["history_json"],
            trace["alpha"],
            trace["beta"],
            trace["posterior_mean"],
            trace["variance"],
            trace["interval_90_lower"],
            trace["interval_90_upper"],
            trace["uncertainty"],
            trace["method_version"],
            trace["advisory_only"],
            trace["rationale"],
        )
        conn.execute(
            "UPDATE research_uncertainty_traces SET evaluation_id=?,trace_hash=? WHERE trace_id=?",
            (
                trace["evaluation_id"],
                authority_hash(*trace_authority),
                trace_result["trace_id"],
            ),
        )

        candidate = dict(
            conn.execute(
                "SELECT * FROM research_proactivity_candidates WHERE candidate_id=?",
                (candidate_result["candidate_id"],),
            ).fetchone()
        )
        candidate["title"] = "A self-hashed but non-canonical title"
        candidate_authority = (
            candidate["child_id"],
            candidate["trace_id"],
            candidate["concept_id"],
            candidate["concept_version"],
            candidate["interest_input_digest"],
            candidate["interest_rule_id"],
            candidate["interest_rule_version"],
            candidate["interest_phase"],
            candidate["policy_id"],
            candidate["policy_version"],
            candidate["definition_hash"],
            candidate["release_content_hash"],
            candidate["family_authorization_epoch"],
            candidate["policy_authorization_epoch"],
            candidate["title"],
            candidate["rationale"],
            candidate["payload_json"],
            "candidate_only",
        )
        conn.execute(
            "UPDATE research_proactivity_candidates SET title=?,candidate_hash=? WHERE candidate_id=?",
            (
                candidate["title"],
                authority_hash(*candidate_authority),
                candidate_result["candidate_id"],
            ),
        )

    with pytest.raises(RuntimeError, match="invalid_traces=.*invalid_candidates"):
        ResearchTrackService(db)


def test_reopen_binds_an_archived_lab_report_to_its_review_release(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _import_adequate_cases(service)
    evaluation = service.evaluate()
    with patch("curiosity_engine.lab.evaluate", return_value=_passing_lab_report()):
        review_result = service.review_for_promotion(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            evaluation_id=str(evaluation["evaluation_id"]),
            reviewer_id="independent-reviewer",
        )

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        _drop_research_triggers(conn)
        review = dict(
            conn.execute(
                "SELECT * FROM research_track_promotion_reviews WHERE review_id=?",
                (review_result["review_id"],),
            ).fetchone()
        )
        report = json.loads(str(review["lab_report_json"]))
        embedded_manifest = dict(report["personalization_policy"])
        files = dict(embedded_manifest["promotion_content_files"])
        first_file = sorted(files)[0]
        files[first_file] = "f" * 64 if files[first_file] != "f" * 64 else "e" * 64
        embedded_manifest["promotion_content_files"] = files
        unsigned_manifest = {key: value for key, value in embedded_manifest.items() if key != "promotion_content_hash"}
        embedded_manifest["promotion_content_hash"] = _digest(jdump(unsigned_manifest))
        assert embedded_manifest["promotion_content_hash"] != review["release_content_hash"]
        report["personalization_policy"] = embedded_manifest
        forged_report_json = jdump(report)
        forged_report_hash = _digest(forged_report_json)
        review_identity = {
            "policy_id": review["policy_id"],
            "policy_version": review["policy_version"],
            "definition_hash": review["definition_hash"],
            "release_content_hash": review["release_content_hash"],
            "evaluation_id": review["evaluation_id"],
            "reviewer_id": review["reviewer_id"],
            "lab_report_hash": forged_report_hash,
        }
        forged_review_id = "rtreview_" + sha256(jdump(review_identity).encode()).hexdigest()[:24]
        conn.execute(
            """UPDATE research_track_promotion_reviews
               SET review_id=?,lab_report_json=?,lab_report_hash=? WHERE review_id=?""",
            (
                forged_review_id,
                forged_report_json,
                forged_report_hash,
                review_result["review_id"],
            ),
        )

    changed_manifest = {**promotion_manifest(), "promotion_content_hash": "0" * 64}
    with (
        patch(
            "curiosity_engine.personalization_policy.promotion_manifest",
            return_value=changed_manifest,
        ),
        patch(
            "curiosity_engine.research_track.promotion_manifest",
            return_value=changed_manifest,
        ),
        patch(
            "curiosity_engine.promotion_validation.promotion_manifest",
            return_value=changed_manifest,
        ),
        pytest.raises(RuntimeError, match="invalid_reviews"),
    ):
        ResearchTrackService(db)


def test_reopen_rejects_a_self_hashed_forged_authorization_epoch(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent enables the bounded research trial.")
    _learning_history(db)
    trace_result = service.compute_uncertainty_trace(child_id="child-a", concept_id="science.phase-change")
    candidate_result = service.create_proactivity_candidate(child_id="child-a", trace_id=str(trace_result["trace_id"]))
    assert candidate_result is not None

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        _drop_research_triggers(conn)
        conn.execute(
            "DELETE FROM research_proactivity_candidates WHERE candidate_id=?",
            (candidate_result["candidate_id"],),
        )
        trace = dict(
            conn.execute(
                "SELECT * FROM research_uncertainty_traces WHERE trace_id=?",
                (trace_result["trace_id"],),
            ).fetchone()
        )
        snapshot = dict(
            conn.execute(
                "SELECT * FROM research_track_authorization_snapshots WHERE snapshot_id=?",
                (trace["authorization_snapshot_id"],),
            ).fetchone()
        )
        forged_epoch = 999
        snapshot_authority = (
            snapshot["policy_id"],
            snapshot["policy_version"],
            snapshot["definition_hash"],
            snapshot["release_content_hash"],
            snapshot["evaluation_id"],
            snapshot["promotion_review_id"],
            forged_epoch,
            snapshot["approved_by_operator"],
            snapshot["approval_reason"],
            snapshot["activation_reason"],
            snapshot["approved_at"],
            snapshot["activated_at"],
            snapshot["approval_event_id"],
            snapshot["activation_event_id"],
            snapshot["created_at"],
        )
        forged_snapshot_hash = research_authorization_snapshot_hash(*snapshot_authority)
        forged_snapshot_id = research_authorization_snapshot_id(forged_snapshot_hash)
        conn.execute(
            """UPDATE research_track_authorization_snapshots
               SET snapshot_id=?,snapshot_hash=?,policy_authorization_epoch=?
               WHERE snapshot_id=?""",
            (
                forged_snapshot_id,
                forged_snapshot_hash,
                forged_epoch,
                snapshot["snapshot_id"],
            ),
        )
        conn.execute(
            "UPDATE research_track_policies SET authorization_epoch=?",
            (forged_epoch,),
        )
        trace["authorization_snapshot_id"] = forged_snapshot_id
        trace["policy_authorization_epoch"] = forged_epoch
        trace_authority = (
            trace["child_id"],
            trace["concept_id"],
            trace["concept_version"],
            trace["construct"],
            trace["projection_input_digest"],
            trace["projection_rule_id"],
            trace["projection_rule_version"],
            trace["policy_id"],
            trace["policy_version"],
            trace["definition_hash"],
            trace["release_content_hash"],
            trace["evaluation_id"],
            trace["promotion_review_id"],
            trace["authorization_snapshot_id"],
            trace["family_authorization_epoch"],
            trace["policy_authorization_epoch"],
            trace["evidence_digest"],
            trace["history_json"],
            trace["alpha"],
            trace["beta"],
            trace["posterior_mean"],
            trace["variance"],
            trace["interval_90_lower"],
            trace["interval_90_upper"],
            trace["uncertainty"],
            trace["method_version"],
            trace["advisory_only"],
            trace["rationale"],
        )
        forged_trace_hash = authority_hash(*trace_authority)
        forged_trace_id = "rtrace_" + sha256(forged_trace_hash.encode()).hexdigest()[:24]
        conn.execute(
            """UPDATE research_uncertainty_traces
               SET trace_id=?,trace_hash=?,authorization_snapshot_id=?,
                   policy_authorization_epoch=? WHERE trace_id=?""",
            (
                forged_trace_id,
                forged_trace_hash,
                forged_snapshot_id,
                forged_epoch,
                trace_result["trace_id"],
            ),
        )

    with pytest.raises(
        RuntimeError,
        match=("behind external authorization authority|invalid_authorization_snapshots=.*invalid_traces"),
    ):
        ResearchTrackService(db)


def test_reopen_rejects_candidate_id_that_is_not_derived_from_its_hash(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent enables the bounded research trial.")
    _learning_history(db)
    trace = service.compute_uncertainty_trace(child_id="child-a", concept_id="science.phase-change")
    candidate = service.create_proactivity_candidate(child_id="child-a", trace_id=str(trace["trace_id"]))
    assert candidate is not None

    with sqlite3.connect(db) as conn:
        _drop_research_triggers(conn)
        conn.execute(
            "UPDATE research_proactivity_candidates SET candidate_id=? WHERE candidate_id=?",
            ("rtcand_" + "0" * 24, candidate["candidate_id"]),
        )

    with pytest.raises(RuntimeError, match="invalid_candidates"):
        ResearchTrackService(db)


def test_reopen_rejects_offline_rewind_of_a_disabled_policy(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    disabled = service.disable(
        policy_id=RESEARCH_POLICY_ID,
        version=RESEARCH_POLICY_VERSION,
        operator_id="promotion-operator",
        reason="The operator ended this research activation.",
    )
    assert disabled["status"] == "disabled"
    with sqlite3.connect(db) as conn:
        _drop_research_triggers(conn)
        conn.execute(
            """UPDATE research_track_policies SET status='active',authorization_epoch=1
               WHERE policy_id=? AND version=?""",
            (RESEARCH_POLICY_ID, RESEARCH_POLICY_VERSION),
        )

    with pytest.raises(
        RuntimeError,
        match="behind external authorization authority|invalid_policy_state=True",
    ):
        ResearchTrackService(db)


def test_external_authority_rejects_deleted_revocations_and_rewound_grants(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(
        child_id="child-a",
        enabled=True,
        reason="Parent enables a bounded research trial.",
    )
    service.set_family_opt_in(
        child_id="child-a",
        enabled=False,
        reason="Parent revokes the bounded research trial.",
    )
    service.disable(
        policy_id=RESEARCH_POLICY_ID,
        version=RESEARCH_POLICY_VERSION,
        operator_id="promotion-operator",
        reason="The operator revokes the research policy.",
    )

    with connect(db) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            conn.execute("DELETE FROM research_track_policy_events WHERE action='disabled'")
    with connect(db) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="service authority"):
            conn.execute("DELETE FROM research_track_control_events WHERE enabled=0")

    with sqlite3.connect(db) as conn:
        _drop_research_triggers(conn)
        conn.execute("DELETE FROM research_track_policy_events WHERE action='disabled'")
        conn.execute(
            """DELETE FROM research_track_control_events
               WHERE child_id='child-a' AND family_authorization_epoch=2"""
        )
        conn.execute(
            """UPDATE research_track_policies SET status='active',authorization_epoch=1
               WHERE policy_id=? AND version=?""",
            (RESEARCH_POLICY_ID, RESEARCH_POLICY_VERSION),
        )
        enabled_event = conn.execute(
            """SELECT * FROM research_track_control_events
               WHERE child_id='child-a' AND family_authorization_epoch=1"""
        ).fetchone()
        assert enabled_event is not None
        conn.execute(
            """UPDATE research_track_controls
               SET enabled=1,policy_id=?,policy_version=?,definition_hash=?,release_content_hash=?,
                   family_authorization_epoch=1,updated_by='parent',reason=?,updated_at=?
               WHERE child_id='child-a'""",
            (
                enabled_event[3],
                enabled_event[4],
                enabled_event[5],
                enabled_event[6],
                enabled_event[9],
                enabled_event[10],
            ),
        )

    with pytest.raises(RuntimeError, match="external authorization authority"):
        ResearchTrackService(db)


def test_retained_service_rejects_missing_external_authority_before_learner_access(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(
        child_id="child-a",
        enabled=True,
        reason="Parent enables the bounded research trial.",
    )
    research_authority_path(db).unlink()

    with patch(
        "curiosity_engine.research_track.recompute_child_projections_in_transaction",
        side_effect=AssertionError("missing authority reached learner projection inputs"),
    ):
        with pytest.raises(
            RuntimeError,
            match="required monotonic research-authority ledger is missing",
        ):
            service.compute_uncertainty_trace(
                child_id="child-a",
                concept_id="science.phase-change",
            )
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM research_uncertainty_traces").fetchone()[0] == 0


def test_release_content_drift_revokes_authority_but_preserves_recovery_paths(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    original_evaluation = _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent enables the bounded research trial.")
    _learning_history(db)
    trace = service.compute_uncertainty_trace(child_id="child-a", concept_id="science.phase-change")
    candidate = service.create_proactivity_candidate(child_id="child-a", trace_id=str(trace["trace_id"]))
    assert candidate is not None
    changed_manifest = {**promotion_manifest(), "promotion_content_hash": "0" * 64}
    with (
        patch(
            "curiosity_engine.personalization_policy.promotion_manifest",
            return_value=changed_manifest,
        ),
        patch(
            "curiosity_engine.research_track.promotion_manifest",
            return_value=changed_manifest,
        ),
        patch(
            "curiosity_engine.promotion_validation.promotion_manifest",
            return_value=changed_manifest,
        ),
    ):
        recovered = ResearchTrackService(db)
        status = recovered.status("child-a")
        assert status["policies"][0]["status"] == "candidate"
        assert status["policies"][0]["authorization_epoch"] == 2
        assert status["child_control"]["enabled"] == 0
        assert status["child_control"]["family_authorization_epoch"] == 2
        assert recovered.list_candidates("child-a") == []
        historical = recovered.list_candidates("child-a", include_invalidated=True)
        assert historical[0]["status"] == "invalidated"
        assert inspect_family_model(db, "child-a")["counts"]["research_uncertainty_traces"] == 1
        exported = export_family_model(db, "child-a")
        assert exported["records"]["research_track_policy_events"]
        with connect(db) as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM research_track_evaluations WHERE evaluation_id=?",
                    (original_evaluation["evaluation_id"],),
                ).fetchone()[0]
                == 1
            )
            assert conn.execute("SELECT COUNT(*) FROM research_track_promotion_reviews").fetchone()[0] == 1
        recovered.import_calibration_case(
            case_id="release-drift-synthetic",
            source_kind="synthetic",
            license_status="synthetic",
            split="heldout",
            group_digest=_digest("release-drift-group"),
            task_signature_digest=_digest("release-drift-task"),
            domain="science",
            history={
                "independent_successes": 1,
                "independent_failures": 0,
                "partial_results": 0,
                "supported_successes": 0,
                "supported_failures": 0,
            },
            observed_success=1,
            source_digest=_digest("release-drift-source"),
            operator_id="calibration-operator",
        )
        fresh = recovered.evaluate()
        assert fresh["status"] == "pass"
        assert fresh["evaluation_id"] != original_evaluation["evaluation_id"]
        fresh_report = _passing_lab_report()
        fresh_report["personalization_policy"] = changed_manifest
        with patch("curiosity_engine.lab.evaluate", return_value=fresh_report):
            review = recovered.review_for_promotion(
                policy_id=RESEARCH_POLICY_ID,
                version=RESEARCH_POLICY_VERSION,
                evaluation_id=str(fresh["evaluation_id"]),
                reviewer_id="new-release-reviewer",
            )
        recovered.approve(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            evaluation_id=str(fresh["evaluation_id"]),
            promotion_review_id=str(review["review_id"]),
            operator_id="new-release-operator",
            reason="Approve the freshly evaluated release.",
        )
        recovered.activate(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            operator_id="new-release-operator",
            reason="Reactivate only after fresh release review.",
        )
        assert recovered.status()["policies"][0]["authorization_epoch"] == 3
        deleted = delete_family_model(
            db,
            child_id="child-a",
            confirmation="child-a",
            reason="Verify release drift does not block family-model deletion.",
        )
        assert deleted["learner_rows_deleted"] > 0


def test_opt_out_pause_and_model_deletion_revoke_research_rows(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(
        child_id="child-a",
        enabled=True,
        reason="Parent enables a bounded trial.",
    )
    pre_trace_inspection = inspect_family_model(db, "child-a")
    assert pre_trace_inspection["counts"]["research_uncertainty_traces"] == 0
    assert pre_trace_inspection["counts"]["research_authorization_snapshots"] == 1
    assert pre_trace_inspection["research_track"]["authorization_snapshots"][0]["policy_authorization_epoch"] == 1
    pre_trace_export = export_family_model(db, "child-a")
    assert len(pre_trace_export["records"]["research_track_authorization_snapshots"]) == 1
    _learning_history(db)
    trace = service.compute_uncertainty_trace(
        child_id="child-a",
        concept_id="science.phase-change",
    )
    candidate = service.create_proactivity_candidate(
        child_id="child-a",
        trace_id=str(trace["trace_id"]),
    )
    assert candidate is not None
    inspected = inspect_family_model(db, "child-a")
    assert inspected["counts"]["research_uncertainty_traces"] == 1
    assert inspected["counts"]["research_proactivity_candidates"] == 1
    inspected_trace = inspected["research_track"]["uncertainty_traces"][0]
    assert inspected_trace["definition_hash"] == RESEARCH_POLICY_DEFINITION_HASH
    assert inspected_trace["evaluation_id"]
    assert inspected_trace["promotion_review_id"]
    assert inspected_trace["authorization_snapshot_id"]
    assert inspected_trace["evidence_digest"]
    assert inspected_trace["release_content_hash"] == service.status()["current_release_content_hash"]
    assert inspected["research_track"]["control_events"]
    assert inspected["counts"]["research_authorization_snapshots"] == 1
    assert (
        inspected["research_track"]["authorization_snapshots"][0]["snapshot_id"]
        == (inspected_trace["authorization_snapshot_id"])
    )

    service.set_family_opt_in(
        child_id="child-a",
        enabled=False,
        reason="Parent ends the bounded trial.",
    )
    assert service.list_candidates("child-a") == []
    assert service.list_candidates("child-a", include_invalidated=True)[0]["status"] == "invalidated"
    exported = export_family_model(db, "child-a")
    assert exported["records"]["research_track_policies"]
    assert exported["records"]["research_track_evaluations"]
    assert exported["records"]["research_track_promotion_reviews"]
    assert exported["records"]["research_track_policy_events"]
    assert exported["records"]["research_track_authorization_snapshots"]
    with pytest.raises(ResearchTrackAuthorizationChanged):
        service.compute_uncertainty_trace(
            child_id="child-a",
            concept_id="science.phase-change",
        )

    service.set_family_opt_in(
        child_id="child-a",
        enabled=True,
        reason="Parent begins a second bounded trial.",
    )
    set_collection_enabled(
        db,
        child_id="child-a",
        enabled=False,
        reason="Pause all learner-model collection.",
    )
    assert service.status("child-a")["child_control"]["enabled"] == 0
    delete_family_model(
        db,
        child_id="child-a",
        confirmation="child-a",
        reason="Delete the child learner model.",
    )
    with connect(db) as conn:
        for table in (
            "research_track_controls",
            "research_track_control_events",
            "research_uncertainty_traces",
            "research_proactivity_candidates",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE child_id='child-a'").fetchone()[0] == 0


def test_whole_model_deletion_externally_revokes_an_active_research_opt_in(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(
        child_id="child-a",
        enabled=True,
        reason="Parent enables a bounded research trial.",
    )

    delete_family_model(
        db,
        child_id="child-a",
        confirmation="child-a",
        reason="Delete the active child learner model.",
    )

    assert ResearchTrackService(db).status("child-a")["child_control"] is None
    ledger = json.loads(research_authority_path(db).read_text(encoding="utf-8"))
    record = next(iter(ledger["family_controls"].values()))
    assert record["enabled"] is False
    assert record["family_authorization_epoch"] == 2
    assert record["lineage_floor"] == 2

    set_collection_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Parent explicitly starts a fresh learner-model epoch.",
    )
    reenrolled = ResearchTrackService(db).set_family_opt_in(
        child_id="child-a",
        enabled=True,
        reason="Parent explicitly re-enrolls after deleting the prior model.",
    )
    assert reenrolled["family_authorization_epoch"] == 3
    _learning_history(db)
    trace = ResearchTrackService(db).compute_uncertainty_trace(
        child_id="child-a",
        concept_id="science.phase-change",
    )
    with connect(db) as conn:
        assert (
            conn.execute(
                "SELECT family_authorization_epoch FROM research_uncertainty_traces WHERE trace_id=?",
                (trace["trace_id"],),
            ).fetchone()[0]
            == 3
        )


def test_collection_pause_after_research_opt_out_is_idempotent(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(
        child_id="child-a",
        enabled=True,
        reason="Parent starts a bounded research trial.",
    )
    opted_out = service.set_family_opt_in(
        child_id="child-a",
        enabled=False,
        reason="Parent ends the bounded research trial.",
    )
    with connect(db) as conn:
        before = conn.execute("SELECT COUNT(*) FROM research_track_control_events WHERE child_id='child-a'").fetchone()[
            0
        ]

    set_collection_enabled(
        db,
        child_id="child-a",
        enabled=False,
        reason="Parent also pauses learner-model collection.",
    )

    status = ResearchTrackService(db).status("child-a")
    assert status["child_control"]["family_authorization_epoch"] == opted_out["family_authorization_epoch"]
    with connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM research_track_control_events WHERE child_id='child-a'").fetchone()[0]
            == before
        )


def test_interrupted_research_grants_roll_back_without_bricking_startup(
    tmp_path: Path,
) -> None:
    activation_db = _db(tmp_path / "activation")
    _approve(ResearchTrackService(activation_db))
    _kill_after_authority_stage(
        activation_db,
        module="curiosity_engine.research_track",
        invocation="""
from curiosity_engine.research_track import ResearchTrackService
ResearchTrackService(sys.argv[1]).activate(
    policy_id='uncertainty-context-k2',
    version=1,
    operator_id='promotion-operator',
    reason='Simulate an interrupted activation.',
)
""",
    )
    activation_status = ResearchTrackService(activation_db).status()
    assert activation_status["policies"][0]["status"] == "approved"
    assert activation_status["policies"][0]["authorization_epoch"] == 0

    opt_in_db = _db(tmp_path / "opt-in")
    _activate(ResearchTrackService(opt_in_db))
    _kill_after_authority_stage(
        opt_in_db,
        module="curiosity_engine.research_track",
        invocation="""
from curiosity_engine.research_track import ResearchTrackService
ResearchTrackService(sys.argv[1]).set_family_opt_in(
    child_id='child-a',
    enabled=True,
    reason='Simulate an interrupted family grant.',
)
""",
    )
    assert ResearchTrackService(opt_in_db).status("child-a")["child_control"] is None


def test_interrupted_research_revocations_roll_forward_on_startup(
    tmp_path: Path,
) -> None:
    opt_out_db = _db(tmp_path / "opt-out")
    opt_out_service = ResearchTrackService(opt_out_db)
    _activate(opt_out_service)
    opt_out_service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    _kill_after_authority_stage(
        opt_out_db,
        module="curiosity_engine.research_track",
        invocation="""
from curiosity_engine.research_track import ResearchTrackService
ResearchTrackService(sys.argv[1]).set_family_opt_in(
    child_id='child-a', enabled=False, reason='Simulate an interrupted opt-out.'
)
""",
    )
    assert ResearchTrackService(opt_out_db).status("child-a")["child_control"]["enabled"] == 0

    disable_db = _db(tmp_path / "disable")
    disable_service = ResearchTrackService(disable_db)
    _activate(disable_service)
    disable_service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    _kill_after_authority_stage(
        disable_db,
        module="curiosity_engine.research_track",
        invocation="""
from curiosity_engine.research_track import ResearchTrackService
ResearchTrackService(sys.argv[1]).disable(
    policy_id='uncertainty-context-k2',
    version=1,
    operator_id='promotion-operator',
    reason='Simulate an interrupted operator revocation.',
)
""",
    )
    disabled = ResearchTrackService(disable_db).status("child-a")
    assert disabled["policies"][0]["status"] == "disabled"
    assert disabled["child_control"]["enabled"] == 0

    pause_db = _db(tmp_path / "pause")
    pause_service = ResearchTrackService(pause_db)
    _activate(pause_service)
    pause_service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    _kill_after_authority_stage(
        pause_db,
        module="curiosity_engine.family_model",
        invocation="""
from curiosity_engine.family_model import set_collection_enabled
set_collection_enabled(
    sys.argv[1], child_id='child-a', enabled=False,
    reason='Simulate an interrupted collection pause.'
)
""",
    )
    assert ResearchTrackService(pause_db).status("child-a")["child_control"]["enabled"] == 0

    deletion_db = _db(tmp_path / "deletion")
    deletion_service = ResearchTrackService(deletion_db)
    _activate(deletion_service)
    deletion_service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    _kill_after_authority_stage(
        deletion_db,
        module="curiosity_engine.family_model",
        invocation="""
from curiosity_engine.family_model import delete_family_model
delete_family_model(
    sys.argv[1], child_id='child-a', confirmation='child-a',
    reason='Simulate an interrupted learner-model deletion.'
)
""",
    )
    assert ResearchTrackService(deletion_db).status("child-a")["child_control"] is None
    delete_family_model(
        deletion_db,
        child_id="child-a",
        confirmation="child-a",
        reason="Retry the learner-model deletion after recovery.",
    )
    assert ResearchTrackService(deletion_db).status("child-a")["child_control"] is None


def test_committed_research_transition_is_finalized_after_process_death(
    tmp_path: Path,
) -> None:
    activation_db = _db(tmp_path / "activation")
    _approve(ResearchTrackService(activation_db))
    _kill_after_research_commit(
        activation_db,
        invocation="""
service.activate(
    policy_id='uncertainty-context-k2',
    version=1,
    operator_id='promotion-operator',
    reason='Commit activation immediately before process death.',
)
""",
    )
    assert ResearchTrackService(activation_db).status()["policies"][0]["status"] == "active"

    opt_out_db = _db(tmp_path / "opt-out")
    opt_out_service = ResearchTrackService(opt_out_db)
    _activate(opt_out_service)
    opt_out_service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    _kill_after_research_commit(
        opt_out_db,
        invocation="""
service.set_family_opt_in(
    child_id='child-a',
    enabled=False,
    reason='Commit revocation immediately before process death.',
)
""",
    )
    recovered = ResearchTrackService(opt_out_db).status("child-a")
    assert recovered["child_control"]["enabled"] == 0
    assert recovered["child_control"]["family_authorization_epoch"] == 2


def test_interrupted_release_drift_revocation_rolls_forward_on_startup(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = ResearchTrackService(db)
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    _kill_after_authority_stage(
        db,
        module="curiosity_engine.db",
        setup="""
from unittest.mock import patch
from curiosity_engine.personalization_policy import promotion_manifest
changed = {**promotion_manifest(), 'promotion_content_hash': '0' * 64}
patches = [
    patch('curiosity_engine.personalization_policy.promotion_manifest', return_value=changed),
    patch('curiosity_engine.promotion_validation.promotion_manifest', return_value=changed),
]
for item in patches:
    item.start()
""",
        invocation="""
from curiosity_engine.research_track import ResearchTrackService
ResearchTrackService(sys.argv[1])
""",
    )
    changed_manifest = {**promotion_manifest(), "promotion_content_hash": "0" * 64}
    with (
        patch(
            "curiosity_engine.personalization_policy.promotion_manifest",
            return_value=changed_manifest,
        ),
        patch(
            "curiosity_engine.research_track.promotion_manifest",
            return_value=changed_manifest,
        ),
        patch(
            "curiosity_engine.promotion_validation.promotion_manifest",
            return_value=changed_manifest,
        ),
    ):
        recovered = ResearchTrackService(db).status("child-a")
    assert recovered["policies"][0]["status"] == "candidate"
    assert recovered["policies"][0]["authorization_epoch"] == 2
    assert recovered["child_control"]["enabled"] == 0
    assert recovered["child_control"]["family_authorization_epoch"] == 2


def test_backup_authority_restores_current_grants_and_applies_later_revocations(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "backup-repo"
    private = repo / "private"
    output = private / "output"
    db = private / "data" / "curiosity.db"
    backup_root = tmp_path / "backup-root"
    output.mkdir(parents=True)
    service = ResearchTrackService(db)
    add_child(db, "child-a", "Synthetic Child", 2020, "1")
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )

    active_restore = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=tmp_path / "active-restore",
        backup_root=backup_root,
        repo_root=repo,
    )
    retained_restore_service = ResearchTrackService(Path(str(active_restore["database"])))
    active = retained_restore_service.status("child-a")
    assert active["policies"][0]["status"] == "active"
    assert active["child_control"]["enabled"] == 1

    service.set_family_opt_in(child_id="child-a", enabled=False, reason="Parent revokes the trial after backup.")
    with patch(
        "curiosity_engine.research_track.recompute_child_projections_in_transaction",
        side_effect=AssertionError("stale authority reached learner projection inputs"),
    ):
        with pytest.raises(ResearchTrackAuthorizationChanged):
            retained_restore_service.compute_uncertainty_trace(
                child_id="child-a",
                concept_id="science.phase-change",
            )
    previously_published = retained_restore_service.status("child-a")
    assert previously_published["child_control"]["enabled"] == 0
    assert previously_published["child_control"]["family_authorization_epoch"] == 2
    assert previously_published["counts"]["uncertainty_traces"] == 0
    stale_restore = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=tmp_path / "stale-restore",
        backup_root=backup_root,
        repo_root=repo,
    )
    stale = ResearchTrackService(Path(str(stale_restore["database"]))).status("child-a")
    assert stale["policies"][0]["status"] == "active"
    assert stale["child_control"]["enabled"] == 0
    assert stale["child_control"]["family_authorization_epoch"] == 2


def test_unopened_restore_does_not_block_later_grant(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "unopened-restore-repo"
    private = repo / "private"
    output = private / "output"
    db = private / "data" / "curiosity.db"
    backup_root = tmp_path / "unopened-restore-root"
    output.mkdir(parents=True)
    service = ResearchTrackService(db)
    add_child(db, "child-a", "Synthetic Child", 2020, "1")
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    unopened = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=tmp_path / "unopened-restore",
        backup_root=backup_root,
        repo_root=repo,
    )

    service.set_family_opt_in(child_id="child-a", enabled=False, reason="Parent ends the first trial.")
    reenrolled = service.set_family_opt_in(
        child_id="child-a", enabled=True, reason="Parent starts a new bounded trial."
    )
    assert reenrolled["family_authorization_epoch"] == 3
    stale_restore_service = ResearchTrackService(Path(str(unopened["database"])))
    unopened_status = stale_restore_service.status("child-a")
    assert unopened_status["child_control"]["enabled"] == 0
    assert unopened_status["child_control"]["family_authorization_epoch"] == 2
    with pytest.raises(BackupError, match="research-authority ledger is unreadable"):
        restore_snapshot(
            str(snapshot["snapshot"]),
            target_private=tmp_path / "obsolete-active-restore",
            backup_root=backup_root,
            repo_root=repo,
        )
    revoked = service.set_family_opt_in(child_id="child-a", enabled=False, reason="Parent ends the second trial.")
    assert revoked["family_authorization_epoch"] == 4
    root_authority_path = backup_root / "family-research-authority.json"
    root_before = root_authority_path.read_bytes()
    with pytest.raises(RuntimeError, match="grant is obsolete"):
        stale_restore_service.set_family_opt_in(
            child_id="child-a",
            enabled=True,
            reason="A stale restore must not overwrite the newer revocation.",
        )
    assert root_authority_path.read_bytes() == root_before
    source_status = service.status("child-a")
    assert source_status["child_control"]["enabled"] == 0
    assert source_status["child_control"]["family_authorization_epoch"] == 4


def test_unopened_restore_does_not_block_later_model_erasure(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "unopened-erasure-repo"
    private = repo / "private"
    output = private / "output"
    db = private / "data" / "curiosity.db"
    backup_root = tmp_path / "unopened-erasure-root"
    output.mkdir(parents=True)
    service = ResearchTrackService(db)
    add_child(db, "child-a", "Synthetic Child", 2020, "1")
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    unopened = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=tmp_path / "unopened-erasure-restore",
        backup_root=backup_root,
        repo_root=repo,
    )

    service.set_family_opt_in(child_id="child-a", enabled=False, reason="Parent ends the bounded trial.")
    delete_family_model(
        db,
        child_id="child-a",
        confirmation="child-a",
        reason="Delete the model while the managed restore remains unopened.",
    )
    assert service.status("child-a")["child_control"] is None
    assert ResearchTrackService(Path(str(unopened["database"]))).status("child-a")["child_control"] is None


def test_closed_restore_tolerates_a_later_policy_revocation_across_skipped_grant(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "skipped-policy-grant-repo"
    private = repo / "private"
    output = private / "output"
    db = private / "data" / "curiosity.db"
    backup_root = tmp_path / "skipped-policy-grant-root"
    output.mkdir(parents=True)
    service = ResearchTrackService(db)
    _activate(service)
    disabled = service.disable(
        policy_id=RESEARCH_POLICY_ID,
        version=RESEARCH_POLICY_VERSION,
        operator_id="promotion-operator",
        reason="Close the original release before taking a recovery image.",
    )
    assert disabled["status"] == "disabled"
    snapshot = create_snapshot(
        db_path=db,
        private_dir=private,
        output_dir=output,
        backup_root=backup_root,
        repo_root=repo,
    )
    unopened = restore_snapshot(
        str(snapshot["snapshot"]),
        target_private=tmp_path / "skipped-policy-grant-restore",
        backup_root=backup_root,
        repo_root=repo,
    )

    changed_manifest = {**promotion_manifest(), "promotion_content_hash": "0" * 64}
    with (
        patch(
            "curiosity_engine.personalization_policy.promotion_manifest",
            return_value=changed_manifest,
        ),
        patch(
            "curiosity_engine.research_track.promotion_manifest",
            return_value=changed_manifest,
        ),
        patch(
            "curiosity_engine.promotion_validation.promotion_manifest",
            return_value=changed_manifest,
        ),
    ):
        refreshed = ResearchTrackService(db)
        assert refreshed.status()["policies"][0]["authorization_epoch"] == 2
        fresh = refreshed.evaluate()
        fresh_report = _passing_lab_report()
        fresh_report["personalization_policy"] = changed_manifest
        with patch("curiosity_engine.lab.evaluate", return_value=fresh_report):
            review = refreshed.review_for_promotion(
                policy_id=RESEARCH_POLICY_ID,
                version=RESEARCH_POLICY_VERSION,
                evaluation_id=str(fresh["evaluation_id"]),
                reviewer_id="new-release-reviewer",
            )
        refreshed.approve(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            evaluation_id=str(fresh["evaluation_id"]),
            promotion_review_id=str(review["review_id"]),
            operator_id="new-release-operator",
            reason="Approve the replacement release.",
        )
        refreshed.activate(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            operator_id="new-release-operator",
            reason="Activate the replacement release.",
        )
        assert refreshed.status()["policies"][0]["authorization_epoch"] == 3
        refreshed.disable(
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            operator_id="new-release-operator",
            reason="Close the replacement release without opening the restore.",
        )
        assert refreshed.status()["policies"][0]["authorization_epoch"] == 4
        restore_status = ResearchTrackService(Path(str(unopened["database"]))).status()
    assert restore_status["policies"][0]["status"] != "active"
    assert restore_status["policies"][0]["authorization_epoch"] == 2


@pytest.mark.parametrize("boundary", ["first_root", "local"])
def test_partial_multi_root_revocation_fanout_survives_active_database_loss(
    tmp_path: Path,
    boundary: str,
) -> None:
    repo = tmp_path / "multi-root-repo"
    private = repo / "private"
    output = private / "output"
    db = private / "data" / "curiosity.db"
    roots = (tmp_path / "backup-a", tmp_path / "backup-b")
    output.mkdir(parents=True)
    service = ResearchTrackService(db)
    add_child(db, "child-a", "Synthetic Child", 2020, "1")
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    snapshots = [
        create_snapshot(
            db_path=db,
            private_dir=private,
            output_dir=output,
            backup_root=root,
            repo_root=repo,
        )
        for root in roots
    ]

    _kill_during_authority_fanout(db, boundary=boundary)
    lost_database = db.with_name("curiosity.lost")
    db.rename(lost_database)
    research_authority_path(db).rename(research_authority_path(lost_database))

    restored = restore_snapshot(
        str(snapshots[1]["snapshot"]),
        target_private=tmp_path / f"restored-{boundary}",
        backup_root=roots[1],
        repo_root=repo,
    )
    status = ResearchTrackService(Path(str(restored["database"]))).status("child-a")
    assert status["child_control"]["enabled"] == 0
    assert status["child_control"]["family_authorization_epoch"] == 2


def test_interrupted_backup_registration_joins_research_authority_before_reopen(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "registration-repo"
    private = repo / "private"
    output = private / "output"
    db = private / "data" / "curiosity.db"
    backup_root = tmp_path / "registration-root"
    output.mkdir(parents=True)
    service = ResearchTrackService(db)
    add_child(db, "child-a", "Synthetic Child", 2020, "1")
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
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
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
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

    recovered_service = ResearchTrackService(db)
    recovered = recovered_service.status("child-a")
    assert recovered["child_control"]["enabled"] == 1
    assert not (backup_root / "family-backup-registration-pending.json").exists()
    root_authority = json.loads((backup_root / "family-research-authority.json").read_text(encoding="utf-8"))
    assert root_authority["backup_roots"] == [str(backup_root.resolve())]
    recovered_service.set_family_opt_in(
        child_id="child-a",
        enabled=False,
        reason="Parent revokes after registration recovery.",
    )
    root_authority = json.loads((backup_root / "family-research-authority.json").read_text(encoding="utf-8"))
    control = next(iter(root_authority["family_controls"].values()))
    assert control["enabled"] is False
    assert control["family_authorization_epoch"] == 2


def test_research_authority_rejects_cross_family_and_unmanaged_root_registries(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "authority-registry-repo"
    private = repo / "private"
    output = private / "output"
    db_a = private / "data" / "family-a.db"
    db_b = tmp_path / "family-b" / "family.db"
    root_a = tmp_path / "family-a-root"
    output.mkdir(parents=True)
    ResearchTrackService(db_a)
    create_snapshot(
        db_path=db_a,
        private_dir=private,
        output_dir=output,
        backup_root=root_a,
        repo_root=repo,
    )
    ResearchTrackService(db_b)
    registry_b = db_b.parent / "family-backup-roots.json"
    registry_b.write_text(
        json.dumps(
            {
                "format": "curiosity-backup-root-registry",
                "version": 1,
                "roots": [str(root_a.resolve())],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="different family|not reciprocally"):
        ResearchTrackService(db_b)

    unmanaged = tmp_path / "must-not-be-created" / "nested-root"
    registry_b.write_text(
        json.dumps(
            {
                "format": "curiosity-backup-root-registry",
                "version": 1,
                "roots": [str(unmanaged.resolve())],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="unavailable or unsafe"):
        ResearchTrackService(db_b)
    assert not unmanaged.exists()


def test_restore_publication_and_research_revocation_share_one_lifecycle_lock(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "restore-race-repo"
    private = repo / "private"
    output = private / "output"
    db = private / "data" / "curiosity.db"
    backup_roots = (
        tmp_path / "restore-race-root-a",
        tmp_path / "restore-race-root-b",
    )
    output.mkdir(parents=True)
    service = ResearchTrackService(db)
    add_child(db, "child-a", "Synthetic Child", 2020, "1")
    _activate(service)
    service.set_family_opt_in(child_id="child-a", enabled=True, reason="Parent starts the trial.")
    snapshots = [
        create_snapshot(
            db_path=db,
            private_dir=private,
            output_dir=output,
            backup_root=root,
            repo_root=repo,
        )
        for root in backup_roots
    ]
    restore_blocked = ThreadEvent()
    release_restore = ThreadEvent()
    revocation_done = ThreadEvent()
    results: dict[str, object] = {}

    def restore_hook(point: str) -> None:
        if point == "before_publish":
            restore_blocked.set()
            assert release_restore.wait(timeout=10)

    def run_restore() -> None:
        results["restore"] = restore_snapshot(
            str(snapshots[1]["snapshot"]),
            target_private=tmp_path / "race-restore",
            backup_root=backup_roots[1],
            repo_root=repo,
            _test_hook=restore_hook,
        )

    def revoke() -> None:
        service.set_family_opt_in(
            child_id="child-a",
            enabled=False,
            reason="Parent revokes while restore publication is paused.",
        )
        revocation_done.set()

    restore_thread = Thread(target=run_restore)
    restore_thread.start()
    assert restore_blocked.wait(timeout=10)
    revoke_thread = Thread(target=revoke)
    revoke_thread.start()
    assert not revocation_done.wait(timeout=0.2)
    release_restore.set()
    restore_thread.join(timeout=15)
    revoke_thread.join(timeout=15)
    assert not restore_thread.is_alive()
    assert not revoke_thread.is_alive()
    assert revocation_done.is_set()
    restored = results["restore"]
    assert isinstance(restored, dict)
    status = ResearchTrackService(Path(str(restored["database"]))).status("child-a")
    assert status["child_control"]["enabled"] == 0


def test_research_track_cli_is_explicit_and_non_delivery_shaped() -> None:
    parser = build_parser()
    status = parser.parse_args(["research-track", "status", "--child", "child-a"])
    assert status.research_track_cmd == "status"
    proposal = parser.parse_args(["research-track", "propose", "--child", "child-a", "--trace", "trace-a"])
    assert proposal.research_track_cmd == "propose"
    assert not hasattr(proposal, "send")
    corpus = parser.parse_args(
        [
            "research-track",
            "propose-corpus",
            "--manifest-file",
            "approved-corpus.json",
            "--operator",
            "operator-a",
        ]
    )
    assert corpus.research_track_cmd == "propose-corpus"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "research-track",
                "import-case",
                "--source-kind",
                "approved_deidentified",
            ]
        )


def test_authority_hash_is_canonical_across_python_and_sqlite(tmp_path: Path) -> None:
    db = _db(tmp_path)
    expected = authority_hash("child-a", 1, 0.5, None, True)
    with connect(db) as conn:
        actual = conn.execute("SELECT authority_hash(?,?,?,?,?)", ("child-a", 1, 0.5, None, True)).fetchone()[0]
    assert actual == expected
