"""Historical Phase 5 promotion matrix; excluded from the default beta gate."""

from __future__ import annotations

import sqlite3
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from curiosity_engine import learning_artifacts as learning_artifacts_module
from curiosity_engine import visuals as visuals_module
from curiosity_engine.cli import build_parser
from curiosity_engine.contracts import Event, FeedbackInput, ImageResearchQuery, ImageSafetyReview, VisualIntent
from curiosity_engine.db import SCHEMA_VERSION, connect, init_db, jdump, jload
from curiosity_engine.family_model import (
    delete_family_model,
    export_family_model,
    inspect_family_model,
    set_recommendation_use_enabled,
)
from curiosity_engine.feedback import record_feedback
from curiosity_engine.graph import add_child
from curiosity_engine.image_research import ImageCandidate, ImageResearchService
from curiosity_engine.lab import (
    JUDGE_SYSTEM_PROMPT,
    discover,
    load_json,
    personalization_generation_route,
    validation_tier_report,
)
from curiosity_engine.learning_artifacts import LearningArtifactService
from curiosity_engine.openai_image_backend import GeneratedImage
from curiosity_engine.pedagogy import ApprovedPedagogyPlan
from curiosity_engine.personalization import (
    PersonalizationAuthorizationChanged,
    PersonalizationService,
    assert_plan_authorization_in_transaction,
    plan_binding_from_decision,
)
from curiosity_engine.personalization_policy import (
    POLICY_DEFINITION,
    POLICY_DEFINITION_HASH,
    SELECTOR_IMPLEMENTATION_DIGEST,
    feedback_feature_bands,
    promotion_manifest,
    select_from_features,
)
from curiosity_engine.reasoning import ReasoningEngine, StubBackend
from curiosity_engine.repository import Repository
from curiosity_engine.runtime import CuriosityHarness
from curiosity_engine.visuals import enqueue_response_visual, process_visual_jobs, synthetic_visual_intent


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "family.db"
    init_db(db)
    add_child(db, "child-a", "Child A", 2020, "1st")
    return db


_V25_TABLES = (
    "personalization_decisions",
    "personalization_promotion_reviews",
    "personalization_policy_events",
    "personalization_control_events",
    "personalization_feedback_signals",
    "personalization_replay_cases",
    "personalization_evaluations",
    "personalization_policies",
)


def _downgrade_to_v24(db: Path) -> None:
    init_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        for table in _V25_TABLES:
            conn.execute(f'DROP TABLE "{table}"')
        conn.execute("DELETE FROM schema_migrations WHERE version=25")
        conn.execute("PRAGMA user_version=24")


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
    suites["personalization"]["candidate_generated_cases"] = 1
    suites["personalization"]["candidate_judged_cases"] = 1
    suites["personalization"]["candidate_behavior_effect_cases"] = 1
    suites["personalization"]["candidate_generation_route"] = personalization_generation_route()
    suites["personalization"]["results"] = [
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
    ]
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


_DECISION_AUTHORITY_FIELDS = (
    "child_id",
    "event_id",
    "event_type",
    "event_text_hash",
    "workflow",
    "proposal_hash",
    "baseline_plan_id",
    "baseline_plan_hash",
    "concept_ids",
    "concept_refs",
    "question_intent",
    "policy_id",
    "policy_version",
    "definition_hash",
    "family_authorization_epoch",
    "policy_authorization_epoch",
    "evaluation_id",
    "promotion_review_id",
    "mode",
    "feature_vector",
    "baseline",
    "candidate",
)


def _rehash_decision(decision: dict[str, object]) -> dict[str, object]:
    authority = {key: decision[key] for key in _DECISION_AUTHORITY_FIELDS}
    decision_hash = sha256(jdump(authority).encode()).hexdigest()
    return {
        **decision,
        "decision_hash": decision_hash,
        "decision_id": "pers_" + sha256(decision_hash.encode()).hexdigest()[:24],
    }


def _promote(service: PersonalizationService, operator: str = "operator-a") -> dict[str, object]:
    evaluation = service.evaluate()
    assert evaluation["status"] == "pass"
    with patch("curiosity_engine.lab.evaluate", return_value=_passing_lab_report()):
        review = service.review_for_promotion(
            policy_id="balanced-k2",
            version=1,
            evaluation_id=str(evaluation["evaluation_id"]),
            reviewer_id="independent-judge-a",
        )
    service.approve(
        policy_id="balanced-k2",
        version=1,
        evaluation_id=str(evaluation["evaluation_id"]),
        promotion_review_id=str(review["review_id"]),
        operator_id=operator,
        reason="Reviewed Phase 5 benefit and regression gates.",
    )
    return service.activate(
        policy_id="balanced-k2",
        version=1,
        operator_id=operator,
        reason="Begin family-opt-in calibration with rollback available.",
    )


class _RevokeDuringPersonalizedGeneration(StubBackend):
    def __init__(self, db: Path):
        self.db = db
        self.revoked = False

    def complete(self, **kwargs: object) -> dict[str, object]:
        payload = dict(kwargs.get("payload") or {})
        plan = dict(payload.get("approved_pedagogy_plan") or {})
        result = super().complete(**kwargs)
        if plan.get("personalization") and not self.revoked:
            self.revoked = True
            set_recommendation_use_enabled(
                self.db,
                child_id="child-a",
                enabled=False,
                reason="Deterministic revoke between personalized provider calls.",
            )
        return result


def _prepare_active_decision(db: Path, event: Event) -> tuple[dict[str, object], object, object, ReasoningEngine]:
    service = PersonalizationService(db)
    _promote(service)
    set_recommendation_use_enabled(
        db,
        child_id=str(event.child_id),
        enabled=True,
        reason="Adversarial binding test opt-in.",
    )
    Repository(db).submit_event(event)
    engine = ReasoningEngine(StubBackend())
    policy = CuriosityHarness(db, engine).policies["pull_thread"]
    proposal = engine.propose_pedagogy(
        policy=policy,
        context={},
        event=event.model_dump(mode="json"),
    )
    baseline = engine.approve_pedagogy(
        proposal,
        event=event.model_dump(mode="json"),
        workflow=policy.workflow,
    )
    decision = service.prepare_decision(
        child_id=str(event.child_id),
        event_id=event.id,
        workflow=policy.workflow,
        proposal=proposal,
        baseline_plan=baseline,
    )
    assert decision and decision["mode"] == "active"
    return decision, proposal, baseline, engine


def test_schema_v25_migrates_a_v24_control_table_and_preserves_child(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE children(
          id TEXT PRIMARY KEY,name TEXT NOT NULL,birth_year INTEGER,grade TEXT,
          created_at TEXT NOT NULL,updated_at TEXT
        );
        CREATE TABLE learner_model_controls(
          child_id TEXT PRIMARY KEY,
          collection_enabled INTEGER NOT NULL DEFAULT 1,
          default_retention_class TEXT NOT NULL DEFAULT 'learning_record',
          updated_by TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        INSERT INTO children VALUES('child-a','Legacy Child',2020,'1st','now','now');
        INSERT INTO learner_model_controls VALUES('child-a',1,'learning_record','parent','now','now');
        PRAGMA user_version=24;
        """
    )
    conn.commit()
    conn.close()

    backup = init_db(db)
    with connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 26
        assert conn.execute("SELECT name FROM children WHERE id='child-a'").fetchone()[0] == "Legacy Child"
        controls = conn.execute(
            "SELECT collection_enabled,recommendation_use_enabled FROM learner_model_controls WHERE child_id='child-a'"
        ).fetchone()
        assert dict(controls) == {"collection_enabled": 1, "recommendation_use_enabled": 0}
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "personalization_policies",
            "personalization_evaluations",
            "personalization_promotion_reviews",
            "personalization_decisions",
            "personalization_feedback_signals",
            "personalization_replay_cases",
            "personalization_control_events",
            "personalization_policy_events",
        } <= tables
    assert backup and backup.exists()


def test_schema_v25_rejects_full_column_table_without_relational_constraints(tmp_path: Path) -> None:
    reference = tmp_path / "reference.sqlite"
    init_db(reference)
    db = tmp_path / "forged.sqlite"
    _downgrade_to_v24(db)
    with sqlite3.connect(db) as conn:
        conn.execute("ATTACH DATABASE ? AS reference", (str(reference),))
        conn.execute(
            "CREATE TABLE personalization_feedback_signals AS "
            "SELECT * FROM reference.personalization_feedback_signals WHERE 0"
        )
        conn.commit()
        conn.execute("DETACH DATABASE reference")
    with pytest.raises(
        RuntimeError,
        match="missing_foreign_keys|invalid_primary_keys|invalid_not_null|missing_unique_constraints",
    ):
        init_db(db)


def test_schema_v25_replaces_same_named_noop_triggers(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with connect(db) as conn:
        conn.execute("DROP TRIGGER guard_personalization_signal_binding_insert")
        conn.execute(
            """CREATE TRIGGER guard_personalization_signal_binding_insert
               BEFORE INSERT ON personalization_feedback_signals
               WHEN 0 BEGIN SELECT 1; END"""
        )
    init_db(db)
    feedback_id = record_feedback(db, FeedbackInput(child_id="child-a", outcome="engaged"))
    add_child(db, "child-b", "Child B", 2020, "1st")
    with connect(db) as conn:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name='guard_personalization_signal_binding_insert'"
        ).fetchone()[0]
        assert "invalid personalization feedback binding" in sql
        with pytest.raises(sqlite3.IntegrityError, match="invalid personalization feedback binding"):
            conn.execute(
                """INSERT INTO personalization_feedback_signals(
                     signal_id,feedback_id,child_id,outcome,created_at
                   ) VALUES('signal-forged',?,'child-b','engaged','now')""",
                (feedback_id,),
            )


def test_structured_feedback_is_separate_from_notes_and_evidence(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with connect(db) as conn:
        before = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("evidence", "evidence_records", "claims")
        }
    feedback_id = record_feedback(
        db,
        FeedbackInput(
            child_id="child-a",
            outcome="too_hard",
            note="A private parent note that must never enter replay.",
            used=True,
            chose_to_continue=False,
            made_prediction=True,
            gave_explanation=False,
            help_level="worked_example",
            asked_new_question=False,
            parent_correction=False,
        ),
    )
    with connect(db) as conn:
        signal = conn.execute(
            "SELECT * FROM personalization_feedback_signals WHERE feedback_id=?", (feedback_id,)
        ).fetchone()
        after = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("evidence", "evidence_records", "claims")
        }
        encoded = jdump(dict(signal))
    assert signal["outcome"] == "too_hard"
    assert signal["used"] == 1 and signal["help_level"] == "worked_example"
    assert "private parent note" not in encoded.casefold()
    assert before == after


def test_replay_accepts_only_categorical_features_and_numeric_labels(tmp_path: Path) -> None:
    service = PersonalizationService(_db(tmp_path))
    valid_features = {
        "question_intent": "guided_inquiry",
        "baseline_strategy_id": "guided-inquiry-k2",
        "domain_overlay_id": "early-science-k2",
        "prior_feedback_band": "none",
        "recent_challenge_signal": "none",
        "knowledge_band": "not_observed",
        "interest_band": "not_observed",
        "response_mode": "spoken",
    }
    valid_outcome = {
        "baseline_utility": 0.5,
        "candidate_utility": 0.6,
        "baseline_calibration_error": 0.3,
        "candidate_calibration_error": 0.2,
    }
    with pytest.raises(ValueError, match="reviewed categorical"):
        service.import_replay_case(
            case_id="leaky-value",
            source_kind="synthetic",
            license_status="synthetic",
            features={**valid_features, "response_mode": "Child A said a private sentence"},
            outcome=valid_outcome,
            operator_id="operator-a",
            source_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="private-data-shaped|numeric schema"):
        service.import_replay_case(
            case_id="leaky-outcome",
            source_kind="synthetic",
            license_status="synthetic",
            features=valid_features,
            outcome={**valid_outcome, "note": "private"},
            operator_id="operator-a",
            source_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="privacy approval"):
        service.import_replay_case(
            case_id="unreviewed-private",
            source_kind="approved_deidentified",
            license_status="approved",
            features=valid_features,
            outcome=valid_outcome,
            operator_id="operator-a",
            source_digest="a" * 64,
        )
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            service.import_replay_case(
                case_id="non-finite-" + str(non_finite).replace("-", "neg"),
                source_kind="synthetic",
                license_status="synthetic",
                features=valid_features,
                outcome={**valid_outcome, "candidate_utility": non_finite},
                operator_id="operator-a",
                source_digest="a" * 64,
            )


def test_replay_schema_rejects_direct_sql_and_invalid_existing_rows(tmp_path: Path) -> None:
    db = _db(tmp_path)
    features = {
        "question_intent": "guided_inquiry",
        "baseline_strategy_id": "guided-inquiry-k2",
        "domain_overlay_id": "early-science-k2",
        "prior_feedback_band": "none",
        "recent_challenge_signal": "none",
        "knowledge_band": "not_observed",
        "interest_band": "not_observed",
        "response_mode": "spoken",
    }
    outcome = {
        "baseline_utility": 0.5,
        "candidate_utility": 0.6,
        "baseline_calibration_error": 0.3,
        "candidate_calibration_error": 0.2,
    }
    with connect(db) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="invalid personalization replay case"):
            conn.execute(
                """INSERT INTO personalization_replay_cases(
                     case_id,source_kind,license_status,privacy_approval_id,feature_schema_version,
                     feature_vector_json,outcome_json,source_digest,case_hash,created_by_operator,created_at
                   ) VALUES('direct-leak','synthetic','synthetic',NULL,'phase5-v2',?,?,?,?,?,'now')""",
                (
                    jdump({**features, "child_id": "private-child"}),
                    jdump(outcome),
                    "a" * 64,
                    "b" * 64,
                    "operator-a",
                ),
            )
        conn.execute("DROP TRIGGER guard_personalization_replay_insert")
        conn.execute(
            """INSERT INTO personalization_replay_cases(
                 case_id,source_kind,license_status,privacy_approval_id,feature_schema_version,
                 feature_vector_json,outcome_json,source_digest,case_hash,created_by_operator,created_at
               ) VALUES('existing-leak','synthetic','synthetic',NULL,'phase5-v2',?,?,?,?,?,'now')""",
            (
                jdump({**features, "child_id": "private-child"}),
                jdump(outcome),
                "a" * 64,
                "c" * 64,
                "operator-a",
            ),
        )
    with pytest.raises(RuntimeError, match="invalid_replay_rows"):
        init_db(db)


def test_policy_never_self_promotes_and_exact_operator_gates_are_required(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = PersonalizationService(db)
    with pytest.raises(ValueError, match="approved policy"):
        service.activate(
            policy_id="balanced-k2",
            version=1,
            operator_id="operator-a",
            reason="Cannot skip evaluation.",
        )
    with pytest.raises(ValueError, match="active operator-promoted"):
        set_recommendation_use_enabled(
            db,
            child_id="child-a",
            enabled=True,
            reason="A generic future-policy opt-in must not be accepted.",
        )
    evaluation = service.evaluate()
    assert evaluation["status"] == "pass"
    assert evaluation["benefit_delta"] > 0
    assert evaluation["candidate_calibration_error"] < evaluation["baseline_calibration_error"]
    assert evaluation["auto_promote"] is False
    with connect(db) as conn:
        policy = conn.execute(
            "SELECT definition_json,definition_hash FROM personalization_policies "
            "WHERE policy_id='balanced-k2' AND version=1"
        ).fetchone()
        assert jload(policy["definition_json"], {}) == POLICY_DEFINITION
        assert policy["definition_hash"] == POLICY_DEFINITION_HASH
        assert POLICY_DEFINITION["selector"]["implementation_digest"] == SELECTOR_IMPLEMENTATION_DIGEST
        assert promotion_manifest()["definition_hash"] == policy["definition_hash"]
        stored_evaluation = conn.execute(
            "SELECT evaluator_digest FROM personalization_evaluations WHERE evaluation_id=?",
            (evaluation["evaluation_id"],),
        ).fetchone()
        assert len(stored_evaluation["evaluator_digest"]) == 64
        assert (
            conn.execute("SELECT status FROM personalization_policies WHERE policy_id='balanced-k2'").fetchone()[0]
            == "evaluated"
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE personalization_evaluations SET benefit_delta=1 WHERE evaluation_id=?",
                (evaluation["evaluation_id"],),
            )
    with pytest.raises(ValueError, match="full Lab review"):
        service.approve(
            policy_id="balanced-k2",
            version=1,
            evaluation_id=str(evaluation["evaluation_id"]),
            promotion_review_id="persreview_" + "0" * 24,
            operator_id="operator-a",
            reason="A local replay alone must not promote.",
        )
    incomplete = _passing_lab_report()
    incomplete["live_judge"] = {**incomplete["live_judge"], "status": "not_run"}
    with patch("curiosity_engine.lab.evaluate", return_value=incomplete):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    missing_regressions = _passing_lab_report()
    missing_regressions["suites"].pop("regressions")
    with patch("curiosity_engine.lab.evaluate", return_value=missing_regressions):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    mismatched_tier_total = _passing_lab_report()
    mismatched_tier_total["summary"]["cases"] -= 1
    mismatched_tier_total["summary"]["passed"] -= 1
    with patch("curiosity_engine.lab.evaluate", return_value=mismatched_tier_total):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    removable_safety_floor = _passing_lab_report()
    removable_safety_floor["required_safety_categories"] = []
    with patch("curiosity_engine.lab.evaluate", return_value=removable_safety_floor):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    missing_safety_category = _passing_lab_report()
    initial_safety_cases = int(missing_safety_category["suites"]["safety"]["cases"])
    missing_safety_category["suites"]["safety"]["results"] = [
        item
        for item in missing_safety_category["suites"]["safety"]["results"]
        if item["category"] != "social_uncertainty"
    ]
    remaining_safety_cases = len(missing_safety_category["suites"]["safety"]["results"])
    missing_safety_category["suites"]["safety"].update(
        {"cases": remaining_safety_cases, "passed": remaining_safety_cases}
    )
    missing_safety_category["live_judge"]["cases_judged"] -= (
        initial_safety_cases - remaining_safety_cases
    )
    with patch("curiosity_engine.lab.evaluate", return_value=missing_safety_category):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    relabeled_safety_cases = _passing_lab_report()
    relabeled_results = relabeled_safety_cases["suites"]["safety"]["results"]
    relabeled_results[0]["category"], relabeled_results[3]["category"] = (
        relabeled_results[3]["category"],
        relabeled_results[0]["category"],
    )
    with patch("curiosity_engine.lab.evaluate", return_value=relabeled_safety_cases):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    unjudged_candidate = _passing_lab_report()
    unjudged_candidate["suites"]["personalization"]["candidate_judged_cases"] = 0
    unjudged_candidate["suites"]["personalization"]["results"][1]["judge"] = {"status": "not_run"}
    unjudged_candidate["live_judge"] = {
        **unjudged_candidate["live_judge"],
        "cases_judged": unjudged_candidate["live_judge"]["cases_judged"] - 1,
    }
    with patch("curiosity_engine.lab.evaluate", return_value=unjudged_candidate):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    inert_candidate = _passing_lab_report()
    inert_candidate["suites"]["personalization"]["candidate_behavior_effect_cases"] = 0
    with patch("curiosity_engine.lab.evaluate", return_value=inert_candidate):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    same_route_judge = _passing_lab_report()
    same_route_judge["live_judge"]["identity"] = {
        "backend": "deterministic",
        "model": "curiosity-stub-v2",
        "role": "judge",
        "prompt_digest": sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest(),
        "independent_from_generation": False,
    }
    with patch("curiosity_engine.lab.evaluate", return_value=same_route_judge):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    stale_content = _passing_lab_report()
    stale_content["personalization_policy"] = {
        **stale_content["personalization_policy"],
        "promotion_content_hash": "0" * 64,
    }
    with patch("curiosity_engine.lab.evaluate", return_value=stale_content):
        with pytest.raises(ValueError, match="every required gate"):
            service.review_for_promotion(
                policy_id="balanced-k2",
                version=1,
                evaluation_id=str(evaluation["evaluation_id"]),
                reviewer_id="independent-judge-a",
            )
    with patch("curiosity_engine.lab.evaluate", return_value=_passing_lab_report()):
        review = service.review_for_promotion(
            policy_id="balanced-k2",
            version=1,
            evaluation_id=str(evaluation["evaluation_id"]),
            reviewer_id="independent-judge-a",
        )
    with pytest.raises(ValueError, match="separate from"):
        service.approve(
            policy_id="balanced-k2",
            version=1,
            evaluation_id=str(evaluation["evaluation_id"]),
            promotion_review_id=str(review["review_id"]),
            operator_id="independent-judge-a",
            reason="Reviewer cannot approve their own review.",
        )
    service.approve(
        policy_id="balanced-k2",
        version=1,
        evaluation_id=str(evaluation["evaluation_id"]),
        promotion_review_id=str(review["review_id"]),
        operator_id="operator-a",
        reason="Independent metrics review completed.",
    )
    with pytest.raises(ValueError, match="approving operator"):
        service.activate(
            policy_id="balanced-k2",
            version=1,
            operator_id="operator-b",
            reason="Different operator cannot silently take over approval.",
        )
    activated = service.activate(
        policy_id="balanced-k2",
        version=1,
        operator_id="operator-a",
        reason="Activate with family opt-in still required.",
    )
    assert activated["status"] == "active" and activated["family_opt_in_required"]
    with connect(db) as conn:
        actions = [
            row[0]
            for row in conn.execute(
                "SELECT action FROM personalization_policy_events ORDER BY created_at,event_id"
            ).fetchall()
        ]
    assert actions == ["evaluated", "promotion_reviewed", "approved", "activated"]


def test_default_runtime_records_shadow_comparison_without_changing_plan(tmp_path: Path) -> None:
    db = _db(tmp_path)
    result = CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(
            id="evt-shadow",
            type="child_question",
            child_id="child-a",
            text="Why do shadows move?",
        )
    )
    trace = result.output["_reasoning"]["shadow_personalization"]
    assert trace["mode"] == "shadow"
    assert trace["raw_features_model_facing"] is False
    assert "feature_vector" not in trace
    assert result.output["_pedagogy"]["interest_support"] == "neutral"
    assert result.output["_pedagogy"]["challenge_assumption"] == "grade_baseline_uncalibrated"
    assert "personalization" not in result.output["_pedagogy"]
    with connect(db) as conn:
        row = conn.execute(
            "SELECT mode,feature_vector_json FROM personalization_decisions WHERE event_id='evt-shadow'"
        ).fetchone()
    assert row["mode"] == "shadow"
    assert "Why do shadows move" not in row["feature_vector_json"]


def test_active_calibration_needs_operator_promotion_and_family_opt_in(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = PersonalizationService(db)
    _promote(service)
    before = inspect_family_model(db, "child-a")
    assert before["controls"]["recommendation_use_enabled"] is False
    assert before["controls"]["recommendation_use_effective"] is False
    CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(
            id="evt-active-prior",
            type="child_question",
            child_id="child-a",
            text="Can we investigate what changes a shadow?",
        )
    )
    set_recommendation_use_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Family wants to try measured challenge calibration.",
    )
    record_feedback(
        db,
        FeedbackInput(
            child_id="child-a",
            event_id="evt-active-prior",
            outcome="too_hard",
            used=True,
        ),
    )
    result = CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(
            id="evt-active",
            type="child_question",
            child_id="child-a",
            text="Can we investigate what changes a shadow?",
        )
    )
    reasoning_trace = result.output["_reasoning"]["shadow_personalization"]
    pedagogy_trace = result.output["_pedagogy"]
    assert reasoning_trace["mode"] == "active"
    assert pedagogy_trace["challenge_assumption"] == "familiar"
    assert pedagogy_trace["personalization"]["decision_id"] == reasoning_trace["decision_id"]
    assert pedagogy_trace["primary_strategy_id"] == reasoning_trace["baseline"]["strategy_id"]
    assert reasoning_trace["candidate"]["strategy_id"] == "model-practice-fade-k2"
    artifact = LearningArtifactService(db, tmp_path / "personalized-output", backend=StubBackend()).create_from_event(
        event_id="evt-active", artifact_type="worksheet"
    )
    assert artifact["artifact_id"]
    after = inspect_family_model(db, "child-a")
    assert after["controls"]["recommendation_use_effective"] is True


def _active_personalized_event(db: Path, event_id: str) -> dict[str, object]:
    service = PersonalizationService(db)
    _promote(service)
    set_recommendation_use_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Family opts into the reviewed artifact test.",
    )
    result = CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(
            id=event_id,
            type="child_question",
            child_id="child-a",
            text="Can we investigate what changes a shadow?",
        )
    )
    assert result.output["_pedagogy"]["personalization"]
    with connect(db) as conn:
        row = conn.execute(
            "SELECT * FROM personalization_decisions WHERE event_id=?",
            (event_id,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _approved_plan_for_decision(db: Path, decision: dict[str, object]) -> ApprovedPedagogyPlan:
    with connect(db) as conn:
        row = conn.execute(
            "SELECT approved_json FROM pedagogy_plans WHERE plan_id=? AND plan_hash=?",
            (decision["plan_id"], decision["plan_hash"]),
        ).fetchone()
    assert row is not None
    return ApprovedPedagogyPlan.model_validate(jload(row["approved_json"]))


@pytest.mark.parametrize("revocation", ["family_opt_out", "policy_disable", "model_delete"])
def test_revoked_personalization_artifacts_use_exact_neutral_baseline(
    tmp_path: Path,
    revocation: str,
) -> None:
    db = _db(tmp_path)
    decision = _active_personalized_event(db, f"evt-artifact-revoke-{revocation}")
    if revocation == "family_opt_out":
        set_recommendation_use_enabled(
            db,
            child_id="child-a",
            enabled=False,
            reason="Family ends the artifact personalization trial.",
        )
    elif revocation == "policy_disable":
        PersonalizationService(db).disable(
            policy_id="balanced-k2",
            version=1,
            operator_id="operator-disable",
            reason="Operator disables the artifact personalization policy.",
        )
    else:
        delete_family_model(
            db,
            child_id="child-a",
            confirmation="child-a",
            reason="Delete the model before creating later artifact work.",
        )
    artifact = LearningArtifactService(
        db,
        tmp_path / f"artifact-{revocation}",
        backend=StubBackend(),
    ).create_from_event(
        event_id=f"evt-artifact-revoke-{revocation}",
        artifact_type="worksheet",
    )
    assert artifact["pedagogy_plan_id"] == decision["baseline_plan_id"]
    assert artifact["pedagogy_plan_hash"] == decision["baseline_plan_hash"]
    with connect(db) as conn:
        stored = conn.execute(
            "SELECT spec_json FROM artifacts WHERE id=?",
            (artifact["artifact_id"],),
        ).fetchone()
    assert jload(stored["spec_json"])["pedagogy"].get("personalization") is None


class _RevokeDuringArtifactProvider:
    name = "revoking-artifact-provider"
    model = "revoking-artifact-v1"

    def __init__(self, db: Path):
        self.db = db
        self.called = False

    def complete(self, **_kwargs: object) -> dict[str, object]:
        if not self.called:
            self.called = True
            set_recommendation_use_enabled(
                self.db,
                child_id="child-a",
                enabled=False,
                reason="Family revokes while the artifact provider is working.",
            )
        return {}


def test_mid_generation_artifact_revocation_fails_before_more_provider_work(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _active_personalized_event(db, "evt-artifact-mid-revoke")
    backend = _RevokeDuringArtifactProvider(db)
    with pytest.raises(RuntimeError, match="authorization changed"):
        LearningArtifactService(
            db,
            tmp_path / "mid-revoke-output",
            backend=backend,
        ).create_from_event(
            event_id="evt-artifact-mid-revoke",
            artifact_type="worksheet",
        )
    assert backend.called is True
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 0


class _RevokeDuringResearchSource:
    name = "smithsonian"

    def __init__(self, db: Path):
        self.db = db

    def search(self, _query: object, *, limit: int) -> list[object]:
        assert limit > 0
        set_recommendation_use_enabled(
            self.db,
            child_id="child-a",
            enabled=False,
            reason="Family revokes while the public research provider is working.",
        )
        return []


def test_research_provider_rechecks_personalization_after_external_work(tmp_path: Path) -> None:
    db = _db(tmp_path)
    stored = _active_personalized_event(db, "evt-research-mid-revoke")
    service = PersonalizationService(db)
    decision = service.decision_for_plan(str(stored["plan_id"]))
    assert decision is not None
    plan = _approved_plan_for_decision(db, stored)
    with pytest.raises(RuntimeError, match="authorization changed"):
        ImageResearchService(
            db,
            tmp_path / "research-revoke-output",
            review_backend=StubBackend(),
            sources=[_RevokeDuringResearchSource(db)],
            authorization_preflight=lambda: service.require_active_decision_authorized(decision),
            authorization_transaction_assertion=lambda conn: assert_plan_authorization_in_transaction(conn, plan),
        ).find(
            ImageResearchQuery(
                query="Moon surface craters",
                purpose="observe",
                preferred_sources=["smithsonian"],
            )
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM researched_image_cache").fetchone()[0] == 0


def _research_candidate() -> ImageCandidate:
    return ImageCandidate(
        source_name="wikimedia",
        source_item_id="phase5-owl",
        title="A real owl",
        description="A great horned owl on a branch.",
        creator="Example Photographer",
        canonical_url="https://commons.wikimedia.org/wiki/File:Owl.jpg",
        media_url="https://upload.wikimedia.org/owl.jpg",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="Example Photographer / Wikimedia Commons",
        rights_status="verified",
    )


class _StaticResearchSource:
    name = "wikimedia"

    def search(self, _query: object, *, limit: int) -> list[ImageCandidate]:
        assert limit > 0
        return [_research_candidate()]


def _passing_image_review() -> ImageSafetyReview:
    return ImageSafetyReview(
        verdict="pass",
        reasons=["safe and relevant"],
        subject_matches=True,
        observable_description="A brown owl perches on a branch.",
        representation_type="photograph",
        reviewer="phase5-test-reviewer",
    )


def _authorized_research_service(
    db: Path,
    output_dir: Path,
    decision_row: dict[str, object],
) -> ImageResearchService:
    service = PersonalizationService(db)
    decision = service.decision_for_plan(str(decision_row["plan_id"]))
    assert decision is not None
    plan = _approved_plan_for_decision(db, decision_row)
    return ImageResearchService(
        db,
        output_dir,
        review_backend=StubBackend(),
        sources=[_StaticResearchSource()],
        authorization_preflight=lambda: service.require_active_decision_authorized(decision),
        authorization_transaction_assertion=lambda conn: assert_plan_authorization_in_transaction(conn, plan),
    )


def test_research_revocation_after_normalization_removes_unreferenced_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db = _db(tmp_path)
    stored = _active_personalized_event(db, "evt-research-normalize-revoke")
    output_dir = tmp_path / "research-normalize-revoke-output"
    service = _authorized_research_service(db, output_dir, stored)
    monkeypatch.setattr(
        "curiosity_engine.image_research._get_bytes",
        lambda *_args, **_kwargs: (b"source-image", "image/jpeg"),
    )

    def normalize_then_revoke(_payload: bytes, _mime: str, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{sha256(b'normalized-image').hexdigest()}.png"
        path.write_bytes(b"normalized-image")
        set_recommendation_use_enabled(
            db,
            child_id="child-a",
            enabled=False,
            reason="Family revokes after public image normalization.",
        )
        return path, "image/png", 640, 480, path.stat().st_size, sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr("curiosity_engine.image_research._normalize_image", normalize_then_revoke)
    with pytest.raises(RuntimeError, match="authorization changed"):
        service.find(
            ImageResearchQuery(
                query="great horned owl close view",
                purpose="observe",
                preferred_sources=["wikimedia"],
            )
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM researched_image_cache").fetchone()[0] == 0
    assert list((output_dir / "researched-images").glob("*.png")) == []


def test_research_revocation_before_cache_insert_is_atomically_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db = _db(tmp_path)
    stored = _active_personalized_event(db, "evt-research-cache-revoke")
    output_dir = tmp_path / "research-cache-revoke-output"
    service = _authorized_research_service(db, output_dir, stored)
    monkeypatch.setattr(
        "curiosity_engine.image_research._get_bytes",
        lambda *_args, **_kwargs: (b"source-image", "image/jpeg"),
    )

    def normalize(_payload: bytes, _mime: str, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{sha256(b'normalized-image').hexdigest()}.png"
        path.write_bytes(b"normalized-image")
        return path, "image/png", 640, 480, path.stat().st_size, sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr("curiosity_engine.image_research._normalize_image", normalize)
    monkeypatch.setattr("curiosity_engine.image_research._review_image", lambda *_args: _passing_image_review())
    original_save = ImageResearchService._save

    def revoke_then_save(self, cache_key, query, asset):
        set_recommendation_use_enabled(
            db,
            child_id="child-a",
            enabled=False,
            reason="Family revokes immediately before the public image cache write.",
        )
        return original_save(self, cache_key, query, asset)

    monkeypatch.setattr(ImageResearchService, "_save", revoke_then_save)
    with pytest.raises(RuntimeError, match="authorization changed"):
        service.find(
            ImageResearchQuery(
                query="great horned owl close view",
                purpose="observe",
                preferred_sources=["wikimedia"],
            )
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM researched_image_cache").fetchone()[0] == 0
    assert list((output_dir / "researched-images").glob("*.png")) == []


def _researched_visual_intent() -> VisualIntent:
    return VisualIntent(
        kind="researched_photo",
        purpose="observe",
        knowledge_role="supportive",
        title="Moon surface",
        pedagogical_value="Supports observation of real public landforms.",
        caption="Look closely at the craters.",
        alt_text="A real public photo of the Moon for close observation.",
        research_query="Moon surface craters",
        preferred_sources=["nasa"],
    )


def _decorative_visual_intent() -> VisualIntent:
    return VisualIntent(
        kind="decorative_illustration",
        purpose="imagine",
        knowledge_role="decorative",
        title="Moon garden",
        pedagogical_value="Invites imagination without presenting factual visual evidence.",
        caption="Imagine a robot garden on the Moon.",
        alt_text="A friendly imaginary robot waters a tiny Moon garden.",
        subject="a friendly imaginary robot watering a tiny garden on the moon",
    )


def _revoke_personalization(db: Path, revocation: str) -> None:
    if revocation == "family_opt_out":
        set_recommendation_use_enabled(
            db,
            child_id="child-a",
            enabled=False,
            reason="Family stops personalized response visuals.",
        )
    elif revocation == "policy_disable":
        PersonalizationService(db).disable(
            policy_id="balanced-k2",
            version=1,
            operator_id="operator-visual-disable",
            reason="Operator stops the policy before queued visual work.",
        )
    else:
        delete_family_model(
            db,
            child_id="child-a",
            confirmation="child-a",
            reason="Delete the model before queued visual work.",
        )


@pytest.mark.parametrize("revocation", ["family_opt_out", "policy_disable", "model_delete"])
def test_queued_personalized_visual_terminalizes_before_provider_after_revocation(
    tmp_path: Path,
    monkeypatch,
    revocation: str,
) -> None:
    db = _db(tmp_path)
    event_id = f"evt-visual-queued-revoke-{revocation}"
    _active_personalized_event(db, event_id)
    job_id = enqueue_response_visual(
        db,
        event_id=event_id,
        visual=_researched_visual_intent(),
        mode="deterministic",
    )
    assert job_id is not None
    _revoke_personalization(db, revocation)
    provider_called = False

    def forbidden_provider(*_args, **_kwargs):
        nonlocal provider_called
        provider_called = True
        raise AssertionError("revoked visual work reached a provider")

    monkeypatch.setattr(ImageResearchService, "find", forbidden_provider)
    result = process_visual_jobs(db, tmp_path / "visual-revoked-output", visual_qa_backend=object())
    assert result == [{"job_id": job_id, "status": "failed"}]
    assert provider_called is False
    with connect(db) as conn:
        job = conn.execute("SELECT status,last_error FROM visual_jobs WHERE id=?", (job_id,)).fetchone()
        assert job["status"] == "failed"
        assert "PersonalizationAuthorizationChanged" in job["last_error"]
        assert conn.execute("SELECT COUNT(*) FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()[0] == 0


def test_personalized_visual_enqueue_rejects_revoked_plan(tmp_path: Path) -> None:
    db = _db(tmp_path)
    event_id = "evt-visual-enqueue-revoke"
    _active_personalized_event(db, event_id)
    _revoke_personalization(db, "family_opt_out")
    with pytest.raises(PersonalizationAuthorizationChanged):
        enqueue_response_visual(
            db,
            event_id=event_id,
            visual=_researched_visual_intent(),
            mode="deterministic",
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM visual_jobs WHERE event_id=?", (event_id,)).fetchone()[0] == 0


def test_visual_worker_rechecks_after_image_provider_and_does_not_fallback(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    event_id = "evt-visual-provider-revoke"
    _active_personalized_event(db, event_id)
    job_id = enqueue_response_visual(
        db,
        event_id=event_id,
        visual=_decorative_visual_intent(),
        mode="decorative",
    )
    assert job_id is not None

    class RevokingImageBackend:
        name = "revoking-image-provider"
        model = "revoking-image-v1"

        def __init__(self) -> None:
            self.calls = 0

        def generate(self, _prompt: str) -> GeneratedImage:
            self.calls += 1
            _revoke_personalization(db, "family_opt_out")
            return GeneratedImage(data=b"untrusted-provider-bytes", model=self.model, request_id="revoked")

    class ForbiddenReviewer:
        name = "forbidden-reviewer"
        model = "forbidden-reviewer-v1"

        def complete(self, **_kwargs):
            raise AssertionError("revoked visual work reached the review provider")

    backend = RevokingImageBackend()
    result = process_visual_jobs(
        db,
        tmp_path / "visual-provider-revoke-output",
        image_backend=backend,
        visual_qa_backend=ForbiddenReviewer(),
    )
    assert result == [{"job_id": job_id, "status": "failed"}]
    assert backend.calls == 1
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT status FROM visual_jobs WHERE id=?", (job_id,)).fetchone()[0] == "failed"


def test_visual_precommit_revocation_removes_uncommitted_asset_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db = _db(tmp_path)
    event_id = "evt-visual-precommit-revoke"
    _active_personalized_event(db, event_id)
    output_dir = tmp_path / "visual-precommit-revoke-output"
    job_id = enqueue_response_visual(
        db,
        event_id=event_id,
        visual=synthetic_visual_intent(),
        mode="deterministic",
    )
    assert job_id is not None
    original_write = visuals_module._write_private_png

    def write_then_revoke(output: str | Path, raw: bytes):
        result = original_write(output, raw)
        _revoke_personalization(db, "family_opt_out")
        return result

    monkeypatch.setattr(visuals_module, "_write_private_png", write_then_revoke)
    result = process_visual_jobs(db, output_dir)
    assert result == [{"job_id": job_id, "status": "failed"}]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()[0] == 0
        assert conn.execute("SELECT status FROM visual_jobs WHERE id=?", (job_id,)).fetchone()[0] == "failed"
    assert list((output_dir / "visuals").glob("visual_*.png")) == []


def test_artifact_precommit_epoch_fence_removes_rendered_private_bytes(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _active_personalized_event(db, "evt-artifact-precommit-revoke")
    original_render = learning_artifacts_module.render_learning_pdf

    def render_then_revoke(spec: object, output_path: str | Path) -> Path:
        result = original_render(spec, output_path)
        set_recommendation_use_enabled(
            db,
            child_id="child-a",
            enabled=False,
            reason="Family revokes immediately before artifact persistence.",
        )
        return result

    with (
        patch(
            "curiosity_engine.learning_artifacts.render_learning_pdf",
            side_effect=render_then_revoke,
        ),
        pytest.raises(RuntimeError, match="authorization changed"),
    ):
        LearningArtifactService(
            db,
            tmp_path / "precommit-output",
            backend=StubBackend(),
        ).create_from_event(
            event_id="evt-artifact-precommit-revoke",
            artifact_type="worksheet",
        )
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 0
    assert not list((tmp_path / "precommit-output").rglob("*.pdf"))
    assert not list((tmp_path / "precommit-output").rglob("*.png"))


def test_decision_cannot_be_tampered_or_rebound_across_children_events_or_plans(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    add_child(db, "child-b", "Child B", 2020, "1st")
    event_a = Event(
        id="evt-bound-a",
        type="child_question",
        child_id="child-a",
        text="Can we investigate shadows?",
    )
    decision, proposal_a, _baseline_a, engine = _prepare_active_decision(db, event_a)
    tampered = {
        **decision,
        "candidate": {
            **decision["candidate"],
            "challenge_assumption": "independent_transfer",
        },
    }
    with pytest.raises(ValueError, match="not canonical"):
        plan_binding_from_decision(tampered)
    with pytest.raises(ValueError, match="not canonical"):
        engine.approve_pedagogy(
            proposal_a,
            event=event_a.model_dump(mode="json"),
            workflow="pull_thread",
            personalization_decision=tampered,
        )
    forged_features = {
        **decision["feature_vector"],
        "recent_challenge_signal": "too_easy",
        "interest_band": "sustained_parent_confirmed",
    }
    self_hashed = _rehash_decision(
        {
            **decision,
            "feature_vector": forged_features,
            "feature_vector_hash": sha256(jdump(forged_features).encode()).hexdigest(),
            "candidate": select_from_features(POLICY_DEFINITION, forged_features),
        }
    )
    self_hashed_plan = engine.approve_pedagogy(
        proposal_a,
        event=event_a.model_dump(mode="json"),
        workflow="pull_thread",
        personalization_decision=self_hashed,
    )
    with pytest.raises(ValueError, match="event, proposal, or policy behavior"):
        Repository(db).persist_pedagogy_plan(
            plan=self_hashed_plan,
            proposal=proposal_a.model_dump(mode="json"),
            run_id=None,
            personalization_decision=self_hashed,
        )

    event_b = Event(
        id="evt-bound-b",
        type="child_question",
        child_id="child-b",
        text="Can we investigate shadows?",
    )
    Repository(db).submit_event(event_b)
    policy = CuriosityHarness(db, engine).policies["pull_thread"]
    proposal_b = engine.propose_pedagogy(
        policy=policy,
        context={},
        event=event_b.model_dump(mode="json"),
    )
    baseline_b = engine.approve_pedagogy(
        proposal_b,
        event=event_b.model_dump(mode="json"),
        workflow=policy.workflow,
    )
    with pytest.raises(ValueError, match="bound event and proposal"):
        engine.approve_pedagogy(
            proposal_b,
            event=event_b.model_dump(mode="json"),
            workflow=policy.workflow,
            personalization_decision=decision,
        )
    with pytest.raises(ValueError, match="requested child"):
        PersonalizationService(db).prepare_decision(
            child_id="child-b",
            event_id=event_a.id,
            workflow=policy.workflow,
            proposal=proposal_a,
            baseline_plan=_baseline_a,
        )

    set_recommendation_use_enabled(
        db,
        child_id="child-a",
        enabled=False,
        reason="Create a separate shadow binding test.",
    )
    shadow_event = Event(
        id="evt-shadow-bound-a",
        type="child_question",
        child_id="child-a",
        text="Can we investigate seeds?",
    )
    Repository(db).submit_event(shadow_event)
    shadow_proposal = engine.propose_pedagogy(
        policy=policy,
        context={},
        event=shadow_event.model_dump(mode="json"),
    )
    shadow_baseline = engine.approve_pedagogy(
        shadow_proposal,
        event=shadow_event.model_dump(mode="json"),
        workflow=policy.workflow,
    )
    shadow_decision = PersonalizationService(db).prepare_decision(
        child_id="child-a",
        event_id=shadow_event.id,
        workflow=policy.workflow,
        proposal=shadow_proposal,
        baseline_plan=shadow_baseline,
    )
    assert shadow_decision and shadow_decision["mode"] == "shadow"
    with pytest.raises(ValueError, match="cannot be rebound"):
        Repository(db).persist_pedagogy_plan(
            plan=baseline_b,
            proposal=proposal_b.model_dump(mode="json"),
            run_id=None,
            personalization_decision=shadow_decision,
        )


def test_direct_and_sensitive_answers_remain_unpersonalized_after_opt_in(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _promote(PersonalizationService(db))
    set_recommendation_use_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Family opt-in for ordinary learning questions.",
    )
    result = CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(
            id="evt-direct",
            type="child_question",
            child_id="child-a",
            text="Just tell me directly: why is the sky blue?",
        )
    )
    assert result.output["_reasoning"]["shadow_personalization"]["mode"] == "shadow"
    assert "personalization" not in result.output["_pedagogy"]
    assert result.output["_pedagogy"]["challenge_assumption"] == "grade_baseline_uncalibrated"
    sensitive = CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(
            id="evt-sensitive-personalization",
            type="child_question",
            child_id="child-a",
            text="Why did my friend ignore me?",
        )
    )
    assert sensitive.output["_reasoning"]["shadow_personalization"]["mode"] == "shadow"
    assert sensitive.output["_pedagogy"]["question_intent"] == "sensitive_support"
    assert "personalization" not in sensitive.output["_pedagogy"]


def test_canonical_active_decision_cannot_bypass_the_direct_answer_compiler_floor(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = PersonalizationService(db)
    _promote(service)
    set_recommendation_use_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Family opt-in applies only to eligible learning plans.",
    )
    event = Event(
        id="evt-forged-direct",
        type="child_question",
        child_id="child-a",
        text="Just tell me directly: why is the sky blue?",
    )
    Repository(db).submit_event(event)
    engine = ReasoningEngine(StubBackend())
    policy = CuriosityHarness(db, engine).policies["pull_thread"]
    proposal = engine.propose_pedagogy(
        policy=policy,
        context={},
        event=event.model_dump(mode="json"),
    )
    baseline = engine.approve_pedagogy(
        proposal,
        event=event.model_dump(mode="json"),
        workflow=policy.workflow,
    )
    shadow = service.prepare_decision(
        child_id="child-a",
        event_id=event.id,
        workflow=policy.workflow,
        proposal=proposal,
        baseline_plan=baseline,
    )
    assert shadow and shadow["mode"] == "shadow"
    with connect(db) as conn:
        review_id = conn.execute(
            "SELECT promotion_review_id FROM personalization_policies WHERE policy_id='balanced-k2' AND version=1"
        ).fetchone()[0]
    forged = _rehash_decision(
        {
            **shadow,
            "mode": "active",
            "promotion_review_id": review_id,
        }
    )
    assert plan_binding_from_decision(forged).question_intent == "direct_answer"
    with pytest.raises(ValueError, match="direct and sensitive"):
        engine.approve_pedagogy(
            proposal,
            event=event.model_dump(mode="json"),
            workflow=policy.workflow,
            personalization_decision=forged,
        )


def test_parent_inspection_and_export_include_redacted_child_scoped_personalization(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    add_child(db, "child-b", "Child B", 2020, "1st")
    service = PersonalizationService(db)
    _promote(service)
    set_recommendation_use_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Family requests inspectable personalization.",
    )
    before_decision = export_family_model(db, "child-a")["records"]
    assert before_decision["personalization_decisions"] == []
    assert before_decision["personalization_policies"]
    assert before_decision["personalization_evaluations"]
    assert before_decision["personalization_promotion_reviews"]
    CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(id="evt-inspect-a", type="child_question", child_id="child-a", text="How do seeds grow?")
    )
    CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(id="evt-inspect-b", type="child_question", child_id="child-b", text="How do seeds grow?")
    )
    inspector = inspect_family_model(db, "child-a")
    decisions = inspector["personalization"]["decisions"]
    assert decisions and {item["event_id"] for item in decisions} == {"evt-inspect-a"}
    assert inspector["personalization"]["feature_vectors_exposed"] is False
    assert all("feature_vector" not in item for item in decisions)
    payload = export_family_model(db, "child-a")
    records = payload["records"]
    assert records["personalization_decisions"]
    assert records["personalization_control_events"]
    assert records["personalization_policies"]
    assert records["personalization_evaluations"]
    assert records["personalization_promotion_reviews"]
    for table in (
        "personalization_decisions",
        "personalization_feedback_signals",
        "personalization_control_events",
    ):
        assert all(row["child_id"] == "child-a" for row in records[table])


def test_multi_concept_challenge_uses_least_supported_target_and_missing_is_unobserved(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO concepts(
                 concept_id,version,label,domain,description,framework,definition_hash,created_at
               ) VALUES('known-concept',1,'Known','science','','test',?, 'now')""",
            ("a" * 64,),
        )
        conn.execute(
            """INSERT INTO learner_state_projections(
                 child_id,concept_id,concept_version,state,evidence_strength,uncertainty,support_level,
                 eligible_opportunity_count,eligible_episode_count,context_diversity,representation_diversity,
                 contradicting_count,last_observed_at,rule_id,rule_version,input_digest,rationale,computed_at
               ) VALUES(
                 'child-a','known-concept',1,'transfer_observed','strong',0.1,'independent',
                 3,3,3,2,0,'now','test','1',?,'test','now'
               )""",
            ("b" * 64,),
        )
        knowledge, interest = PersonalizationService._projection_features(
            conn,
            "child-a",
            [
                {"id": "known-concept", "version": 1},
                {"id": "missing-concept", "version": 1},
            ],
        )
    assert knowledge == "not_observed"
    assert interest == "not_observed"


def test_feedback_is_filtered_before_recency_limit_and_combined_per_concept(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    harness = CuriosityHarness(db, ReasoningEngine(StubBackend()))
    harness.dispatch(
        Event(
            id="evt-relevant-science",
            type="child_question",
            child_id="child-a",
            text="Can we investigate why shadows move?",
        )
    )
    record_feedback(
        db,
        FeedbackInput(
            child_id="child-a",
            event_id="evt-relevant-science",
            outcome="too_hard",
        ),
    )
    harness.dispatch(
        Event(
            id="evt-unrelated-math",
            type="child_question",
            child_id="child-a",
            text="Can we investigate how numbers add?",
        )
    )
    for _ in range(41):
        record_feedback(
            db,
            FeedbackInput(
                child_id="child-a",
                event_id="evt-unrelated-math",
                outcome="too_easy",
            ),
        )
    with connect(db) as conn:
        count, recent = PersonalizationService._feedback_features(
            conn,
            "child-a",
            [{"id": "science-observation-and-evidence", "version": 1}],
        )
    assert (count, recent) == ("one", "too_hard")
    assert feedback_feature_bands(
        [
            {
                "signal_id": "one",
                "concept_id": "concept-a",
                "concept_version": 1,
                "outcome": "too_easy",
            }
        ],
        [
            {"id": "concept-a", "version": 1},
            {"id": "concept-b", "version": 1},
        ],
    ) == ("one", "mixed")


def test_personalization_features_do_not_cross_concept_versions(tmp_path: Path) -> None:
    db = _db(tmp_path)
    with connect(db) as conn:
        for version, digest in ((1, "c" * 64), (2, "d" * 64)):
            conn.execute(
                """INSERT INTO concepts(
                     concept_id,version,label,domain,description,framework,definition_hash,created_at
                   ) VALUES('versioned-concept',?,'Versioned','science','','test',?,'now')""",
                (version, digest),
            )
        conn.execute(
            """INSERT INTO learner_state_projections(
                 child_id,concept_id,concept_version,state,evidence_strength,uncertainty,support_level,
                 eligible_opportunity_count,eligible_episode_count,context_diversity,representation_diversity,
                 contradicting_count,last_observed_at,rule_id,rule_version,input_digest,rationale,computed_at
               ) VALUES(
                 'child-a','versioned-concept',2,'transfer_observed','strong',0.1,'independent',
                 3,3,3,2,0,'now','test','1',?,'test','now'
               )""",
            ("e" * 64,),
        )
        version_one = PersonalizationService._projection_features(
            conn,
            "child-a",
            [{"id": "versioned-concept", "version": 1}],
        )
        version_two = PersonalizationService._projection_features(
            conn,
            "child-a",
            [{"id": "versioned-concept", "version": 2}],
        )
    assert version_one == ("not_observed", "not_observed")
    assert version_two == ("transfer_observed", "not_observed")
    assert feedback_feature_bands(
        [
            {
                "signal_id": "wrong-version",
                "concept_id": "versioned-concept",
                "concept_version": 2,
                "outcome": "too_easy",
            }
        ],
        [{"id": "versioned-concept", "version": 1}],
    ) == ("none", "none")


def test_disable_is_immediate_and_family_model_deletion_clears_adaptive_state(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = PersonalizationService(db)
    _promote(service)
    set_recommendation_use_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Temporary family trial.",
    )
    record_feedback(db, FeedbackInput(child_id="child-a", outcome="engaged", used=True))
    CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(id="evt-delete", type="child_question", child_id="child-a", text="How do seeds grow?")
    )
    service.disable(
        policy_id="balanced-k2",
        version=1,
        operator_id="operator-a",
        reason="Immediate rollback after review.",
    )
    assert inspect_family_model(db, "child-a")["controls"]["recommendation_use_effective"] is False
    delete_family_model(
        db,
        child_id="child-a",
        confirmation="child-a",
        reason="Family requested learner-model deletion.",
    )
    with connect(db) as conn:
        assert (
            conn.execute(
                "SELECT recommendation_use_enabled FROM learner_model_controls WHERE child_id='child-a'"
            ).fetchone()[0]
            == 0
        )
        assert all(
            conn.execute(f"SELECT COUNT(*) FROM {table} WHERE child_id='child-a'").fetchone()[0] == 0
            for table in (
                "personalization_decisions",
                "personalization_feedback_signals",
                "personalization_control_events",
            )
        )


def test_authorization_loss_before_generation_recompiles_a_baseline_plan(tmp_path: Path) -> None:
    db = _db(tmp_path)
    service = PersonalizationService(db)
    _promote(service)
    CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(
            id="evt-revoked-prior",
            type="child_question",
            child_id="child-a",
            text="Can we investigate a moving shadow?",
        )
    )
    set_recommendation_use_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Temporary family trial.",
    )
    record_feedback(
        db,
        FeedbackInput(
            child_id="child-a",
            event_id="evt-revoked-prior",
            outcome="too_hard",
        ),
    )
    event = Event(
        id="evt-revoked-before-generation",
        type="child_question",
        child_id="child-a",
        text="Can we investigate a moving shadow?",
    )
    repository = Repository(db)
    repository.submit_event(event)
    engine = ReasoningEngine(StubBackend())
    policy = CuriosityHarness(db, engine).policies["pull_thread"]
    proposal = engine.propose_pedagogy(
        policy=policy,
        context={},
        event=event.model_dump(mode="json"),
    )
    baseline_plan = engine.approve_pedagogy(
        proposal,
        event=event.model_dump(mode="json"),
        workflow=policy.workflow,
    )
    decision = service.prepare_decision(
        child_id="child-a",
        event_id=event.id,
        workflow=policy.workflow,
        proposal=proposal,
        baseline_plan=baseline_plan,
    )
    assert decision and decision["mode"] == "active"
    personalized_plan = engine.approve_pedagogy(
        proposal,
        event=event.model_dump(mode="json"),
        workflow=policy.workflow,
        personalization_decision=decision,
    )
    with pytest.raises(ValueError, match="exact decision"):
        repository.persist_pedagogy_plan(
            plan=personalized_plan,
            proposal=proposal.model_dump(mode="json"),
            run_id=None,
        )
    repository.persist_pedagogy_plan(
        plan=personalized_plan,
        proposal=proposal.model_dump(mode="json"),
        run_id=None,
        personalization_decision=decision,
    )
    service.disable(
        policy_id="balanced-k2",
        version=1,
        operator_id="operator-a",
        reason="Family trial stopped before content generation.",
    )

    result = CuriosityHarness(db, engine).dispatch(event)
    assert result.status == "completed"
    assert "personalization" not in result.output["_pedagogy"]
    assert result.output["_pedagogy"]["challenge_assumption"] == "grade_baseline_uncalibrated"
    with connect(db) as conn:
        plans = conn.execute(
            "SELECT contract_version FROM pedagogy_plans WHERE event_id=? ORDER BY contract_version",
            (event.id,),
        ).fetchall()
    assert [row[0] for row in plans] == ["2.1", "2.1p1"]


def test_revoke_during_personalized_provider_work_discards_and_regenerates_baseline(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path)
    service = PersonalizationService(db)
    _promote(service)
    set_recommendation_use_enabled(
        db,
        child_id="child-a",
        enabled=True,
        reason="Temporary opt-in for authorization epoch race test.",
    )
    backend = _RevokeDuringPersonalizedGeneration(db)
    result = CuriosityHarness(db, ReasoningEngine(backend)).dispatch(
        Event(
            id="evt-revoke-during-generation",
            type="child_question",
            child_id="child-a",
            text="Can we investigate what changes a shadow?",
        )
    )
    assert backend.revoked is True
    assert result.status == "completed"
    assert "personalization" not in result.output["_pedagogy"]
    assert result.output["_pedagogy"]["challenge_assumption"] == "grade_baseline_uncalibrated"
    with connect(db) as conn:
        stored = jload(
            conn.execute("SELECT output_json FROM responses WHERE event_id='evt-revoke-during-generation'").fetchone()[
                0
            ],
            {},
        )
        assert "personalization" not in stored["_pedagogy"]
        assert (
            conn.execute(
                "SELECT personalization_epoch FROM learner_model_controls WHERE child_id='child-a'"
            ).fetchone()[0]
            == 2
        )


def test_cli_exposes_explicit_personalization_governance() -> None:
    parser = build_parser()
    evaluated = parser.parse_args(["personalization", "evaluate", "--db", "x.db"])
    assert evaluated.personalization_cmd == "evaluate"
    reviewed = parser.parse_args(
        [
            "personalization",
            "review",
            "--db",
            "x.db",
            "--policy",
            "balanced-k2",
            "--version",
            "1",
            "--evaluation",
            "perseval-a",
            "--reviewer",
            "judge-a",
        ]
    )
    assert reviewed.personalization_cmd == "review"
    approved = parser.parse_args(
        [
            "personalization",
            "approve",
            "--db",
            "x.db",
            "--policy",
            "balanced-k2",
            "--version",
            "1",
            "--evaluation",
            "eval-a",
            "--promotion-review",
            "persreview-a",
            "--operator",
            "operator-a",
            "--reason",
            "reviewed",
        ]
    )
    assert approved.personalization_cmd == "approve"
    family = parser.parse_args(
        [
            "learner-model",
            "personalization",
            "--db",
            "x.db",
            "--child",
            "child-a",
            "--enabled",
            "on",
            "--reason",
            "family opt-in",
        ]
    )
    assert family.learner_model_cmd == "personalization"


def test_decision_and_replay_hashes_are_content_bound(tmp_path: Path) -> None:
    db = _db(tmp_path)
    CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(id="evt-hash", type="child_question", child_id="child-a", text="Why does ice float?")
    )
    PersonalizationService(db).evaluate()
    with connect(db) as conn:
        decision = conn.execute("SELECT * FROM personalization_decisions").fetchone()
        replay = conn.execute("SELECT * FROM personalization_replay_cases ORDER BY case_id LIMIT 1").fetchone()
    expected_decision = {
        "child_id": decision["child_id"],
        "event_id": decision["event_id"],
        "event_type": decision["event_type"],
        "event_text_hash": decision["event_text_hash"],
        "workflow": decision["workflow"],
        "proposal_hash": decision["proposal_hash"],
        "baseline_plan_id": decision["baseline_plan_id"],
        "baseline_plan_hash": decision["baseline_plan_hash"],
        "concept_ids": jload(decision["concept_ids_json"], []),
        "concept_refs": jload(decision["concept_refs_json"], []),
        "question_intent": decision["question_intent"],
        "policy_id": decision["policy_id"],
        "policy_version": decision["policy_version"],
        "definition_hash": decision["definition_hash"],
        "family_authorization_epoch": decision["family_authorization_epoch"],
        "policy_authorization_epoch": decision["policy_authorization_epoch"],
        "evaluation_id": decision["evaluation_id"],
        "promotion_review_id": decision["promotion_review_id"],
        "mode": decision["mode"],
        "feature_vector": jload(decision["feature_vector_json"], {}),
        "baseline": jload(decision["baseline_json"], {}),
        "candidate": jload(decision["candidate_json"], {}),
    }
    assert sha256(jdump(expected_decision).encode()).hexdigest() == decision["decision_hash"]
    assert len(replay["case_hash"]) == 64 and len(replay["source_digest"]) == 64
