from __future__ import annotations

import inspect
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import configuration_root
from .db import connect, init_db, jdump, jload, utcnow
from .personalization_policy import (
    EVALUATION_MINIMUM_BENEFIT,
    EVALUATION_MINIMUM_CASES,
    EVALUATOR_VERSION,
    FEATURE_SCHEMA_VERSION,
    PERSONALIZATION_SUITE_VERSION,
    POLICY_DEFINITION_HASH,
    POLICY_ID,
    POLICY_VERSION,
    RUNTIME_BEHAVIOR_MANIFEST,
    feedback_feature_bands,
    feedback_signal_query,
    normalize_replay_outcome,
    projection_feature_bands,
    replay_case_payload,
    response_mode_feature,
    select_from_features,
)
from .personalization_policy import (
    assert_feature_vector as _assert_feature_vector,
)
from .personalization_policy import (
    contains_forbidden_key as _contains_forbidden_key,
)
from .promotion_validation import validate_full_lab_report

_OUTCOMES = {
    "loved",
    "engaged",
    "neutral",
    "too_easy",
    "too_hard",
    "not_used",
    "disliked",
    "helpful",
    "not_helpful",
}


class PersonalizationAuthorizationChanged(RuntimeError):
    """The exact family/policy authorization epoch no longer permits generation."""
def _digest(value: Any) -> str:
    return sha256(jdump(value).encode()).hexdigest()


def _identifier(value: str, label: str) -> str:
    cleaned = str(value).strip()
    if not 1 <= len(cleaned) <= 160 or not all(character.isalnum() or character in "_.:-" for character in cleaned):
        raise ValueError(f"{label} must be a bounded stable identifier")
    return cleaned


def _reason(value: str) -> str:
    cleaned = " ".join(str(value).split())
    if not 1 <= len(cleaned) <= 500:
        raise ValueError("a bounded reason is required")
    return cleaned


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


def _decision_authority(decision: dict[str, Any]) -> dict[str, Any]:
    try:
        return {key: decision[key] for key in _DECISION_AUTHORITY_FIELDS}
    except KeyError as exc:
        raise ValueError("personalization decision is missing canonical authority") from exc


class PersonalizationService:
    """Shadow-first policy evaluation with explicit operator and family gates."""

    def __init__(self, db_path: str | Path, repo_root: str | Path | None = None):
        self.db_path = str(db_path)
        self.repo_root = Path(repo_root).resolve() if repo_root else configuration_root().resolve()
        init_db(self.db_path, repo_root=self.repo_root)

    @staticmethod
    def _event(
        conn: Any,
        *,
        policy_id: str,
        version: int,
        action: str,
        actor_type: str,
        reason: str,
        operator_id: str | None = None,
    ) -> str:
        now = utcnow()
        event_id = "persev_" + sha256(
            f"{policy_id}|{version}|{action}|{actor_type}|{operator_id or ''}|{reason}|{now}".encode()
        ).hexdigest()[:24]
        conn.execute(
            """INSERT INTO personalization_policy_events(
                 event_id,policy_id,policy_version,action,actor_type,operator_id,reason,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (event_id, policy_id, version, action, actor_type, operator_id, reason, now),
        )
        return event_id

    @staticmethod
    def _policy_from_conn(conn: Any) -> dict[str, Any] | None:
        row = conn.execute(
            """SELECT * FROM personalization_policies
               WHERE status='active'
               ORDER BY activated_at DESC,version DESC LIMIT 1"""
        ).fetchone()
        if not row:
            row = conn.execute(
                """SELECT * FROM personalization_policies
                   WHERE status IN ('candidate','evaluated','approved')
                   ORDER BY CASE status WHEN 'approved' THEN 0 WHEN 'evaluated' THEN 1 ELSE 2 END,
                            version DESC LIMIT 1"""
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _feedback_features(
        conn: Any,
        child_id: str,
        concept_refs: list[dict[str, Any]],
    ) -> tuple[str, str]:
        canonical = sorted(
            (
                {"id": str(item["id"]), "version": int(item["version"])}
                for item in concept_refs
            ),
            key=lambda item: (item["id"], item["version"]),
        )
        if not canonical:
            return "none", "none"
        parameters: list[Any] = [child_id]
        for item in canonical:
            parameters.extend((item["id"], item["version"]))
        rows = conn.execute(
            feedback_signal_query(len(canonical)),
            tuple(parameters),
        ).fetchall()
        return feedback_feature_bands([dict(row) for row in rows], canonical)

    @staticmethod
    def _projection_features(
        conn: Any,
        child_id: str,
        concept_refs: list[dict[str, Any]],
    ) -> tuple[str, str]:
        canonical = sorted(
            (
                {"id": str(item["id"]), "version": int(item["version"])}
                for item in concept_refs
            ),
            key=lambda item: (item["id"], item["version"]),
        )
        if not canonical:
            return "not_observed", "not_observed"
        predicates = " OR ".join("(concept_id=? AND concept_version=?)" for _ in canonical)
        parameters: list[Any] = [child_id]
        for item in canonical:
            parameters.extend((item["id"], item["version"]))
        knowledge_rows = conn.execute(
            f"""SELECT concept_id,concept_version,state FROM learner_state_projections
                WHERE child_id=? AND ({predicates})""",
            tuple(parameters),
        ).fetchall()
        interest_rows = conn.execute(
            f"""SELECT phase FROM interest_projections
                WHERE child_id=? AND ({predicates})""",
            tuple(parameters),
        ).fetchall()
        return projection_feature_bands(
            canonical,
            {
                (str(row["concept_id"]), int(row["concept_version"])): str(row["state"])
                for row in knowledge_rows
            },
            [str(row[0]) for row in interest_rows],
        )

    def prepare_decision(
        self,
        *,
        child_id: str,
        event_id: str,
        workflow: str,
        proposal: Any,
        baseline_plan: Any,
    ) -> dict[str, Any] | None:
        """Create a decision bound to one authoritative event, proposal, and baseline plan."""

        from .pedagogy import personalization_is_allowed

        proposal_payload = proposal.model_dump(mode="json")
        proposal_hash = _digest(proposal_payload)
        baseline_plan.assert_canonical_integrity()
        canonical_concepts = sorted(dict.fromkeys(str(item) for item in proposal.concept_ids))
        canonical_concept_refs = [
            {"id": item.id, "version": item.version}
            for item in baseline_plan.concepts
        ]
        with connect(self.db_path) as conn:
            policy = self._policy_from_conn(conn)
            if not policy:
                return None
            definition = jload(policy["definition_json"], {})
            if (
                _digest(definition) != str(policy["definition_hash"])
                or str(policy["feature_schema_version"]) != FEATURE_SCHEMA_VERSION
            ):
                raise ValueError("personalization policy definition hash mismatch")
            event = conn.execute(
                "SELECT id,type,child_id,text FROM events WHERE id=?", (event_id,)
            ).fetchone()
            if not event or str(event["child_id"] or "") != child_id:
                raise ValueError("personalization event does not belong to the requested child")
            baseline_matches = all(
                (
                    baseline_plan.personalization is None,
                    baseline_plan.event_id == event_id,
                    baseline_plan.child_id == child_id,
                    baseline_plan.workflow == workflow,
                    baseline_plan.question_intent == proposal.question_intent,
                    baseline_plan.primary_strategy.id == proposal.primary_strategy_id,
                    [item.id for item in baseline_plan.concepts] == canonical_concepts,
                    baseline_plan.interest_support == "neutral",
                    baseline_plan.challenge_assumption == "grade_baseline_uncalibrated",
                )
            )
            if not baseline_matches:
                raise ValueError("personalization baseline does not match the bound proposal")
            controls = conn.execute(
                """SELECT collection_enabled,recommendation_use_enabled,recommendation_policy_id,
                          recommendation_policy_version,recommendation_policy_hash,personalization_epoch
                   FROM learner_model_controls WHERE child_id=?""",
                (child_id,),
            ).fetchone()
            code_owned_eligible = personalization_is_allowed(
                text=str(event["text"] or ""),
                workflow=workflow,
                question_intent=proposal.question_intent,
            )
            active_for_family = bool(
                policy["status"] == "active"
                and policy["evaluation_id"]
                and policy["promotion_review_id"]
                and controls
                and controls["collection_enabled"]
                and controls["recommendation_use_enabled"]
                and controls["recommendation_policy_id"] == policy["policy_id"]
                and controls["recommendation_policy_version"] == policy["version"]
                and controls["recommendation_policy_hash"] == policy["definition_hash"]
                and code_owned_eligible
            )
            prior_feedback, recent = self._feedback_features(
                conn,
                child_id,
                canonical_concept_refs,
            )
            knowledge, interest = self._projection_features(
                conn,
                child_id,
                canonical_concept_refs,
            )
        response_modes = list(proposal.evidence.response_modes)
        features = _assert_feature_vector(
            {
                "question_intent": proposal.question_intent,
                "baseline_strategy_id": proposal.primary_strategy_id,
                "domain_overlay_id": proposal.domain_overlay_id,
                "prior_feedback_band": prior_feedback,
                "recent_challenge_signal": recent,
                "knowledge_band": knowledge,
                "interest_band": interest,
                "response_mode": response_mode_feature(response_modes),
            }
        )
        baseline = {
            "strategy_id": proposal.primary_strategy_id,
            "challenge_assumption": "grade_baseline_uncalibrated",
            "interest_support": "neutral",
        }
        candidate = select_from_features(definition, features)
        payload = {
            "child_id": child_id,
            "event_id": event_id,
            "event_type": str(event["type"]),
            "event_text_hash": sha256(str(event["text"] or "").encode()).hexdigest(),
            "workflow": workflow,
            "proposal_hash": proposal_hash,
            "baseline_plan_id": baseline_plan.plan_id,
            "baseline_plan_hash": baseline_plan.plan_hash,
            "concept_ids": canonical_concepts,
            "concept_refs": canonical_concept_refs,
            "question_intent": proposal.question_intent,
            "policy_id": str(policy["policy_id"]),
            "policy_version": int(policy["version"]),
            "definition_hash": str(policy["definition_hash"]),
            "family_authorization_epoch": int(controls["personalization_epoch"]) if controls else 0,
            "policy_authorization_epoch": int(policy["authorization_epoch"]),
            "evaluation_id": str(policy["evaluation_id"]) if policy["evaluation_id"] else None,
            "promotion_review_id": (
                str(policy["promotion_review_id"])
                if active_for_family and policy["promotion_review_id"]
                else None
            ),
            "mode": "active" if active_for_family else "shadow",
            "feature_vector": features,
            "baseline": baseline,
            "candidate": candidate,
        }
        decision_hash = _digest(payload)
        return {
            **payload,
            "decision_id": "pers_" + sha256(decision_hash.encode()).hexdigest()[:24],
            "decision_hash": decision_hash,
            "feature_vector_hash": _digest(features),
        }

    def active_decision_is_authorized(self, decision: dict[str, Any]) -> bool:
        if decision.get("mode") != "active":
            return True
        from .pedagogy import personalization_is_allowed

        if (
            _digest(_decision_authority(decision)) != decision.get("decision_hash")
            or _digest(decision.get("feature_vector")) != decision.get("feature_vector_hash")
        ):
            return False
        with connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT c.collection_enabled,c.recommendation_use_enabled,
                          c.recommendation_policy_id,c.recommendation_policy_version,
                          c.recommendation_policy_hash,c.personalization_epoch,p.status,p.definition_hash,p.definition_json,
                          p.feature_schema_version,p.evaluation_id,p.promotion_review_id,
                          p.authorization_epoch,
                          e.child_id,e.type,e.text
                   FROM events e JOIN learner_model_controls c ON c.child_id=e.child_id
                   JOIN personalization_policies p ON p.policy_id=? AND p.version=?
                   WHERE e.id=?""",
                (decision["policy_id"], decision["policy_version"], decision["event_id"]),
            ).fetchone()
            if row:
                try:
                    self._trusted_evaluation(
                        conn,
                        str(decision["policy_id"]),
                        int(decision["policy_version"]),
                        str(decision["evaluation_id"]),
                    )
                    self._trusted_promotion_review(
                        conn,
                        policy_id=str(decision["policy_id"]),
                        version=int(decision["policy_version"]),
                        evaluation_id=str(decision["evaluation_id"]),
                        review_id=str(decision["promotion_review_id"]),
                    )
                except (TypeError, ValueError):
                    return False
        return bool(
            row
            and row["status"] == "active"
            and row["definition_hash"] == decision["definition_hash"]
            and _digest(jload(row["definition_json"], {})) == decision["definition_hash"]
            and row["feature_schema_version"] == FEATURE_SCHEMA_VERSION
            and row["evaluation_id"] == decision["evaluation_id"]
            and row["promotion_review_id"] == decision["promotion_review_id"]
            and row["personalization_epoch"] == decision["family_authorization_epoch"]
            and row["authorization_epoch"] == decision["policy_authorization_epoch"]
            and row["child_id"] == decision["child_id"]
            and row["type"] == decision["event_type"]
            and sha256(str(row["text"] or "").encode()).hexdigest() == decision["event_text_hash"]
            and personalization_is_allowed(
                text=str(row["text"] or ""),
                workflow=str(decision["workflow"]),
                question_intent=str(decision["question_intent"]),
            )
            and row["recommendation_policy_id"] == decision["policy_id"]
            and row["recommendation_policy_version"] == decision["policy_version"]
            and row["recommendation_policy_hash"] == decision["definition_hash"]
            and row["collection_enabled"]
            and row["recommendation_use_enabled"]
        )

    def require_active_decision_authorized(self, decision: dict[str, Any]) -> None:
        if not self.active_decision_is_authorized(decision):
            raise PersonalizationAuthorizationChanged(
                "personalization authorization changed before a provider call"
            )

    def record_decision(self, *, decision: dict[str, Any], plan: Any) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            insert_decision_in_transaction(conn, decision=decision, plan=plan)
        return decision_trace(decision)

    def decision_for_plan(self, plan_id: str) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM personalization_decisions WHERE plan_id=?", (plan_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "decision_id": str(row["decision_id"]),
            "decision_hash": str(row["decision_hash"]),
            "mode": str(row["mode"]),
            "policy_id": str(row["policy_id"]),
            "policy_version": int(row["policy_version"]),
            "definition_hash": str(row["definition_hash"]),
            "family_authorization_epoch": int(row["family_authorization_epoch"]),
            "policy_authorization_epoch": int(row["policy_authorization_epoch"]),
            "evaluation_id": row["evaluation_id"],
            "promotion_review_id": row["promotion_review_id"],
            "feature_vector": jload(row["feature_vector_json"], {}),
            "feature_vector_hash": str(row["feature_vector_hash"]),
            "baseline": jload(row["baseline_json"], {}),
            "candidate": jload(row["candidate_json"], {}),
            "child_id": str(row["child_id"]),
            "event_id": str(row["event_id"]),
            "event_type": str(row["event_type"]),
            "event_text_hash": str(row["event_text_hash"]),
            "workflow": str(row["workflow"]),
            "proposal_hash": str(row["proposal_hash"]),
            "baseline_plan_id": str(row["baseline_plan_id"]),
            "baseline_plan_hash": str(row["baseline_plan_hash"]),
            "concept_ids": jload(row["concept_ids_json"], []),
            "concept_refs": jload(row["concept_refs_json"], []),
            "question_intent": str(row["question_intent"]),
            "plan_id": str(row["plan_id"]),
            "plan_hash": str(row["plan_hash"]),
        }
    def _fixture(self) -> tuple[Path, list[dict[str, Any]], str]:
        path = self.repo_root / "evals" / "personalization" / "core.json"
        raw = path.read_bytes()
        value = jload(raw.decode(), [])
        if not isinstance(value, list) or not value:
            raise ValueError("personalization fixture corpus is empty")
        return path, value, sha256(raw).hexdigest()

    def _canonical_evaluation(self, policy: Any) -> dict[str, Any]:
        definition = jload(policy["definition_json"], {})
        if (
            _digest(definition) != str(policy["definition_hash"])
            or str(policy["feature_schema_version"]) != FEATURE_SCHEMA_VERSION
        ):
            raise ValueError("personalization policy definition hash mismatch")
        _path, fixtures, fixture_digest = self._fixture()
        results: list[dict[str, Any]] = []
        decisions_by_equivalence: dict[str, set[str]] = {}
        utility_delta = 0.0
        baseline_error = 0.0
        candidate_error = 0.0
        guardrails = {"safety": True, "privacy": True, "fairness": True, "golden": True}
        for case in fixtures:
            if _contains_forbidden_key(case):
                raise ValueError("public personalization fixture contains private-data-shaped fields")
            features = _assert_feature_vector(dict(case["features"]))
            candidate = select_from_features(definition, features)
            expected = dict(case["expected_candidate"])
            golden = candidate == expected
            outcome = normalize_replay_outcome(
                {
                    "baseline_utility": case["baseline_utility"],
                    "candidate_utility": case["candidate_utility"],
                    "baseline_calibration_error": case["baseline_calibration_error"],
                    "candidate_calibration_error": case["candidate_calibration_error"],
                }
            )
            utility_delta += outcome["candidate_utility"] - outcome["baseline_utility"]
            baseline_error += outcome["baseline_calibration_error"]
            candidate_error += outcome["candidate_calibration_error"]
            safety = not bool(case.get("safety_regression", False))
            privacy = not _contains_forbidden_key(features) and str(case.get("source_kind")) in {
                "public_fixture",
                "synthetic",
            }
            key = str(case.get("equivalence_key") or "")
            if key:
                decisions_by_equivalence.setdefault(key, set()).add(_digest(candidate))
            guardrails["safety"] &= safety
            guardrails["privacy"] &= privacy
            guardrails["golden"] &= golden
            results.append(
                {
                    "case_id": _identifier(str(case["id"]), "case ID"),
                    "candidate_hash": _digest(candidate),
                    "golden_pass": golden,
                    "safety_pass": safety,
                    "privacy_pass": privacy,
                }
            )
        guardrails["fairness"] = all(len(values) == 1 for values in decisions_by_equivalence.values())
        count = len(results)
        benefit_delta = utility_delta / count
        baseline_error /= count
        candidate_error /= count
        passed = (
            count >= EVALUATION_MINIMUM_CASES
            and benefit_delta >= EVALUATION_MINIMUM_BENEFIT
            and candidate_error <= baseline_error
            and all(guardrails.values())
        )
        results_hash = _digest(results)
        evaluator_digest = _digest(
            {
                "runtime_behavior": RUNTIME_BEHAVIOR_MANIFEST,
                "minimum_cases": EVALUATION_MINIMUM_CASES,
                "minimum_benefit": EVALUATION_MINIMUM_BENEFIT,
                "fixture_source": inspect.getsource(self._fixture),
                "evaluation_source": inspect.getsource(self._canonical_evaluation),
            }
        )
        identity = {
            "policy_id": str(policy["policy_id"]),
            "policy_version": int(policy["version"]),
            "definition_hash": str(policy["definition_hash"]),
            "suite_version": PERSONALIZATION_SUITE_VERSION,
            "evaluator": EVALUATOR_VERSION,
            "evaluator_digest": evaluator_digest,
            "fixture_digest": fixture_digest,
            "results_hash": results_hash,
        }
        return {
            **identity,
            "evaluation_id": "perseval_" + sha256(jdump(identity).encode()).hexdigest()[:24],
            "case_count": count,
            "benefit_delta": benefit_delta,
            "baseline_calibration_error": baseline_error,
            "candidate_calibration_error": candidate_error,
            "safety_pass": bool(guardrails["safety"]),
            "privacy_pass": bool(guardrails["privacy"]),
            "fairness_pass": bool(guardrails["fairness"]),
            "golden_pass": bool(guardrails["golden"]),
            "status": "pass" if passed else "fail",
            "results": results,
            "fixtures": fixtures,
        }

    def evaluate(self, policy_id: str = POLICY_ID, version: int = POLICY_VERSION) -> dict[str, Any]:
        policy_id = _identifier(policy_id, "policy ID")
        with connect(self.db_path) as conn:
            policy = conn.execute(
                "SELECT * FROM personalization_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
        if not policy:
            raise ValueError("unknown personalization policy")
        canonical = self._canonical_evaluation(policy)
        for case in canonical["fixtures"]:
            outcome = {
                key: case[key]
                for key in (
                    "baseline_utility",
                    "candidate_utility",
                    "baseline_calibration_error",
                    "candidate_calibration_error",
                )
            }
            self.import_replay_case(
                case_id=str(case["id"]),
                source_kind=str(case["source_kind"]),
                license_status="public" if case["source_kind"] == "public_fixture" else "synthetic",
                features=dict(case["features"]),
                outcome=outcome,
                operator_id="system-evaluator",
                source_digest=str(canonical["fixture_digest"]),
            )
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO personalization_evaluations(
                     evaluation_id,policy_id,policy_version,definition_hash,suite_version,fixture_digest,
                     case_count,benefit_delta,baseline_calibration_error,candidate_calibration_error,
                     safety_pass,privacy_pass,fairness_pass,golden_pass,status,evaluator,evaluator_digest,
                     results_json,results_hash,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(evaluation_id) DO NOTHING""",
                (
                    canonical["evaluation_id"],
                    policy_id,
                    version,
                    canonical["definition_hash"],
                    canonical["suite_version"],
                    canonical["fixture_digest"],
                    canonical["case_count"],
                    canonical["benefit_delta"],
                    canonical["baseline_calibration_error"],
                    canonical["candidate_calibration_error"],
                    int(canonical["safety_pass"]),
                    int(canonical["privacy_pass"]),
                    int(canonical["fairness_pass"]),
                    int(canonical["golden_pass"]),
                    canonical["status"],
                    canonical["evaluator"],
                    canonical["evaluator_digest"],
                    jdump(canonical["results"]),
                    canonical["results_hash"],
                    now,
                ),
            )
            if policy["status"] in {"candidate", "evaluated"}:
                conn.execute(
                    """UPDATE personalization_policies SET status=?,evaluation_id=?,updated_at=?
                       WHERE policy_id=? AND version=?""",
                    (
                        "evaluated" if canonical["status"] == "pass" else "candidate",
                        canonical["evaluation_id"],
                        now,
                        policy_id,
                        version,
                    ),
                )
            self._event(
                conn,
                policy_id=policy_id,
                version=version,
                action="evaluated",
                actor_type="system",
                reason=(
                    f"Deterministic {PERSONALIZATION_SUITE_VERSION} evaluation "
                    f"{canonical['evaluation_id']}: {canonical['status']}."
                ),
            )
        return {
            "evaluation_id": canonical["evaluation_id"],
            "status": canonical["status"],
            "case_count": canonical["case_count"],
            "benefit_delta": round(canonical["benefit_delta"], 6),
            "baseline_calibration_error": round(canonical["baseline_calibration_error"], 6),
            "candidate_calibration_error": round(canonical["candidate_calibration_error"], 6),
            "gates": {
                key.removesuffix("_pass"): bool(canonical[key])
                for key in ("safety_pass", "privacy_pass", "fairness_pass", "golden_pass")
            },
            "auto_promote": False,
            "operator_approval_required": True,
        }

    def _trusted_evaluation(self, conn: Any, policy_id: str, version: int, evaluation_id: str) -> Any:
        row = conn.execute(
            """SELECT e.*,p.version AS version,p.feature_schema_version,p.definition_json,
                      p.definition_hash AS current_definition_hash
               FROM personalization_evaluations e JOIN personalization_policies p
                 ON p.policy_id=e.policy_id AND p.version=e.policy_version
               WHERE e.evaluation_id=? AND e.policy_id=? AND e.policy_version=? AND e.status='pass'""",
            (evaluation_id, policy_id, version),
        ).fetchone()
        if not row:
            raise ValueError("promotion requires an exact passing evaluation")
        canonical = self._canonical_evaluation(row)
        trusted = all(
            (
                str(row["definition_hash"]) == str(row["current_definition_hash"]),
                str(row["evaluation_id"]) == canonical["evaluation_id"],
                str(row["suite_version"]) == canonical["suite_version"],
                str(row["fixture_digest"]) == canonical["fixture_digest"],
                str(row["evaluator"]) == canonical["evaluator"],
                str(row["evaluator_digest"]) == canonical["evaluator_digest"],
                str(row["results_hash"]) == canonical["results_hash"],
                jload(row["results_json"], []) == canonical["results"],
                int(row["case_count"]) == canonical["case_count"],
                float(row["benefit_delta"]) == canonical["benefit_delta"],
                float(row["baseline_calibration_error"]) == canonical["baseline_calibration_error"],
                float(row["candidate_calibration_error"]) == canonical["candidate_calibration_error"],
                bool(row["safety_pass"]) == canonical["safety_pass"],
                bool(row["privacy_pass"]) == canonical["privacy_pass"],
                bool(row["fairness_pass"]) == canonical["fairness_pass"],
                bool(row["golden_pass"]) == canonical["golden_pass"],
                str(row["status"]) == canonical["status"] == "pass",
            )
        )
        if not trusted:
            raise ValueError("personalization evaluation no longer matches trusted promotion inputs")
        return row

    def _validate_lab_report(self, report: dict[str, Any]) -> None:
        validate_full_lab_report(report, self.repo_root)

    def review_for_promotion(
        self,
        *,
        policy_id: str,
        version: int,
        evaluation_id: str,
        reviewer_id: str,
    ) -> dict[str, Any]:
        """Run and bind the full deterministic plus independent-judge promotion gate."""

        policy_id = _identifier(policy_id, "policy ID")
        reviewer_id = _identifier(reviewer_id, "reviewer ID")
        with connect(self.db_path) as conn:
            self._trusted_evaluation(conn, policy_id, version, evaluation_id)
            policy = conn.execute(
                "SELECT definition_hash,status,evaluation_id FROM personalization_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
        if not policy or policy["status"] != "evaluated" or policy["evaluation_id"] != evaluation_id:
            raise ValueError("policy is not ready for the separate promotion review")
        from .lab import evaluate as evaluate_lab

        report = evaluate_lab(self.repo_root, live_judge=True)
        self._validate_lab_report(report)
        report_hash = _digest(report)
        identity = {
            "policy_id": policy_id,
            "policy_version": version,
            "definition_hash": str(policy["definition_hash"]),
            "evaluation_id": evaluation_id,
            "reviewer_id": reviewer_id,
            "lab_report_hash": report_hash,
        }
        review_id = "persreview_" + sha256(jdump(identity).encode()).hexdigest()[:24]
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO personalization_promotion_reviews(
                     review_id,policy_id,policy_version,definition_hash,evaluation_id,reviewer_id,
                     lab_report_json,lab_report_hash,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(review_id) DO NOTHING""",
                (
                    review_id,
                    policy_id,
                    version,
                    policy["definition_hash"],
                    evaluation_id,
                    reviewer_id,
                    jdump(report),
                    report_hash,
                    now,
                ),
            )
            stored = conn.execute(
                "SELECT * FROM personalization_promotion_reviews WHERE review_id=?", (review_id,)
            ).fetchone()
            if not stored or str(stored["lab_report_hash"]) != report_hash:
                raise ValueError("conflicting personalization promotion review")
            self._event(
                conn,
                policy_id=policy_id,
                version=version,
                action="promotion_reviewed",
                actor_type="system",
                operator_id=None,
                reason=f"Independent full Lab report {report_hash[:16]} reviewed by {reviewer_id}.",
            )
        return {
            "review_id": review_id,
            "policy_id": policy_id,
            "version": version,
            "evaluation_id": evaluation_id,
            "reviewer_id": reviewer_id,
            "lab_report_hash": report_hash,
            "status": "pass",
        }

    def _trusted_promotion_review(
        self,
        conn: Any,
        *,
        policy_id: str,
        version: int,
        evaluation_id: str,
        review_id: str,
    ) -> Any:
        row = conn.execute(
            """SELECT * FROM personalization_promotion_reviews
               WHERE review_id=? AND policy_id=? AND policy_version=? AND evaluation_id=?""",
            (review_id, policy_id, version, evaluation_id),
        ).fetchone()
        if not row:
            raise ValueError("promotion requires an exact full Lab review")
        report = jload(row["lab_report_json"], {})
        if (
            _digest(report) != str(row["lab_report_hash"])
            or str(row["definition_hash"]) != POLICY_DEFINITION_HASH
        ):
            raise ValueError("promotion review failed canonical integrity verification")
        self._validate_lab_report(report)
        return row

    def approve(
        self,
        *,
        policy_id: str,
        version: int,
        evaluation_id: str,
        promotion_review_id: str,
        operator_id: str,
        reason: str,
    ) -> dict[str, Any]:
        policy_id = _identifier(policy_id, "policy ID")
        operator_id = _identifier(operator_id, "operator ID")
        reason = _reason(reason)
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._trusted_evaluation(conn, policy_id, version, evaluation_id)
            review = self._trusted_promotion_review(
                conn,
                policy_id=policy_id,
                version=version,
                evaluation_id=evaluation_id,
                review_id=promotion_review_id,
            )
            if str(review["reviewer_id"]) == operator_id:
                raise ValueError("operator approval must be separate from the promotion reviewer")
            row = conn.execute(
                "SELECT status,evaluation_id FROM personalization_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
            if not row or row["status"] not in {"evaluated", "approved"} or row["evaluation_id"] != evaluation_id:
                raise ValueError("policy is not ready for operator approval")
            conn.execute(
                """UPDATE personalization_policies
                   SET status='approved',promotion_review_id=?,approved_by_operator=?,approval_reason=?,
                       approved_at=?,updated_at=?
                   WHERE policy_id=? AND version=?""",
                (promotion_review_id, operator_id, reason, now, now, policy_id, version),
            )
            event_id = self._event(
                conn,
                policy_id=policy_id,
                version=version,
                action="approved",
                actor_type="operator",
                operator_id=operator_id,
                reason=reason,
            )
        return {
            "policy_id": policy_id,
            "version": version,
            "status": "approved",
            "promotion_review_id": promotion_review_id,
            "auto_promoted": False,
            "event_id": event_id,
        }

    def activate(
        self,
        *,
        policy_id: str,
        version: int,
        operator_id: str,
        reason: str,
    ) -> dict[str, Any]:
        policy_id = _identifier(policy_id, "policy ID")
        operator_id = _identifier(operator_id, "operator ID")
        reason = _reason(reason)
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM personalization_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
            if not row or row["status"] != "approved" or row["approved_by_operator"] != operator_id:
                raise ValueError("only the approving operator may activate an approved policy")
            self._trusted_evaluation(conn, policy_id, version, str(row["evaluation_id"]))
            self._trusted_promotion_review(
                conn,
                policy_id=policy_id,
                version=version,
                evaluation_id=str(row["evaluation_id"]),
                review_id=str(row["promotion_review_id"]),
            )
            prior_active = conn.execute(
                """SELECT policy_id,version FROM personalization_policies
                   WHERE status='active' AND NOT (policy_id=? AND version=?)""",
                (policy_id, version),
            ).fetchall()
            conn.execute(
                """UPDATE personalization_policies
                   SET status='disabled',authorization_epoch=authorization_epoch+1,updated_at=?
                   WHERE status='active' AND NOT (policy_id=? AND version=?)""",
                (now, policy_id, version),
            )
            for prior in prior_active:
                self._event(
                    conn,
                    policy_id=str(prior["policy_id"]),
                    version=int(prior["version"]),
                    action="disabled",
                    actor_type="operator",
                    operator_id=operator_id,
                    reason=f"Replaced by {policy_id}@{version}: {reason}",
                )
            conn.execute(
                """UPDATE personalization_policies
                   SET status='active',authorization_epoch=authorization_epoch+1,
                       activation_reason=?,activated_at=?,updated_at=?
                   WHERE policy_id=? AND version=?""",
                (reason, now, now, policy_id, version),
            )
            event_id = self._event(
                conn,
                policy_id=policy_id,
                version=version,
                action="activated",
                actor_type="operator",
                operator_id=operator_id,
                reason=reason,
            )
        return {
            "policy_id": policy_id,
            "version": version,
            "status": "active",
            "family_opt_in_required": True,
            "event_id": event_id,
        }

    def disable(
        self,
        *,
        policy_id: str,
        version: int,
        operator_id: str,
        reason: str,
    ) -> dict[str, Any]:
        policy_id = _identifier(policy_id, "policy ID")
        operator_id = _identifier(operator_id, "operator ID")
        reason = _reason(reason)
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM personalization_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
            if not row or row["status"] != "active":
                raise ValueError("only an active personalization policy can be disabled")
            conn.execute(
                """UPDATE personalization_policies
                   SET status='disabled',authorization_epoch=authorization_epoch+1,updated_at=?
                   WHERE policy_id=? AND version=?""",
                (utcnow(), policy_id, version),
            )
            event_id = self._event(
                conn,
                policy_id=policy_id,
                version=version,
                action="disabled",
                actor_type="operator",
                operator_id=operator_id,
                reason=reason,
            )
        return {"policy_id": policy_id, "version": version, "status": "disabled", "event_id": event_id}

    def status(self) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            policies = [
                {
                    "policy_id": str(row["policy_id"]),
                    "version": int(row["version"]),
                    "status": str(row["status"]),
                    "definition_hash": str(row["definition_hash"]),
                    "authorization_epoch": int(row["authorization_epoch"]),
                    "evaluation_id": row["evaluation_id"],
                    "promotion_review_id": row["promotion_review_id"],
                    "approved_by_operator": row["approved_by_operator"],
                    "activated_at": row["activated_at"],
                }
                for row in conn.execute(
                    """SELECT policy_id,version,status,definition_hash,authorization_epoch,
                              evaluation_id,promotion_review_id,
                              approved_by_operator,activated_at
                       FROM personalization_policies ORDER BY policy_id,version"""
                ).fetchall()
            ]
            counts = {
                "shadow_decisions": conn.execute(
                    "SELECT COUNT(*) FROM personalization_decisions WHERE mode='shadow'"
                ).fetchone()[0],
                "active_decisions": conn.execute(
                    "SELECT COUNT(*) FROM personalization_decisions WHERE mode='active'"
                ).fetchone()[0],
                "feedback_signals": conn.execute("SELECT COUNT(*) FROM personalization_feedback_signals").fetchone()[0],
                "replay_cases": conn.execute("SELECT COUNT(*) FROM personalization_replay_cases").fetchone()[0],
            }
        return {
            "policies": policies,
            "counts": counts,
            "self_modifying": False,
            "auto_promote": False,
            "private_text_in_replay": False,
        }

    def import_replay_case(
        self,
        *,
        case_id: str,
        source_kind: str,
        license_status: str,
        features: dict[str, Any],
        outcome: dict[str, Any],
        operator_id: str,
        source_digest: str,
        privacy_approval_id: str | None = None,
    ) -> dict[str, Any]:
        case_id = _identifier(case_id, "case ID")
        operator_id = _identifier(operator_id, "operator ID")
        if source_kind not in {"public_fixture", "synthetic", "approved_deidentified"}:
            raise ValueError("unsupported replay source")
        expected_license = {"public_fixture": "public", "synthetic": "synthetic", "approved_deidentified": "approved"}
        if license_status != expected_license[source_kind]:
            raise ValueError("replay source and license status do not match")
        if source_kind == "approved_deidentified":
            privacy_approval_id = _identifier(str(privacy_approval_id or ""), "privacy approval ID")
        elif privacy_approval_id is not None:
            raise ValueError("public and synthetic replay do not carry private approval references")
        vector = _assert_feature_vector(features)
        if _contains_forbidden_key(outcome):
            raise ValueError("replay outcome contains private-data-shaped fields")
        normalized_outcome = normalize_replay_outcome(outcome)
        if len(source_digest) != 64 or any(character not in "0123456789abcdef" for character in source_digest):
            raise ValueError("source digest must be a SHA-256 hex digest")
        payload = replay_case_payload(
            case_id=case_id,
            source_kind=source_kind,
            license_status=license_status,
            privacy_approval_id=privacy_approval_id,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_vector=vector,
            outcome=normalized_outcome,
            source_digest=source_digest,
        )
        case_hash = _digest(payload)
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO personalization_replay_cases(
                     case_id,source_kind,license_status,privacy_approval_id,feature_schema_version,
                     feature_vector_json,outcome_json,source_digest,case_hash,created_by_operator,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(case_id) DO NOTHING""",
                (
                    case_id,
                    source_kind,
                    license_status,
                    privacy_approval_id,
                    FEATURE_SCHEMA_VERSION,
                    jdump(vector),
                    jdump(normalized_outcome),
                    source_digest,
                    case_hash,
                    operator_id,
                    utcnow(),
                ),
            )
            row = conn.execute(
                "SELECT case_hash FROM personalization_replay_cases WHERE case_id=?", (case_id,)
            ).fetchone()
        if not row or str(row["case_hash"]) != case_hash:
            raise ValueError("conflicting personalization replay case")
        return {"case_id": case_id, "case_hash": case_hash, "source_kind": source_kind}


def plan_binding_from_decision(decision: dict[str, Any]) -> Any:
    """Verify complete decision authority before reducing it to a model-safe plan binding."""

    from .pedagogy import PersonalizationPlanBinding
    from .personalization_policy import POLICY_DEFINITION

    authority = _decision_authority(decision)
    features = _assert_feature_vector(dict(decision.get("feature_vector") or {}))
    candidate = select_from_features(POLICY_DEFINITION, features)
    baseline = {
        "strategy_id": features["baseline_strategy_id"],
        "challenge_assumption": "grade_baseline_uncalibrated",
        "interest_support": "neutral",
    }
    if any(
        (
            decision.get("mode") != "active",
            not decision.get("evaluation_id"),
            not decision.get("promotion_review_id"),
            decision.get("policy_id") != POLICY_ID,
            decision.get("policy_version") != POLICY_VERSION,
            decision.get("definition_hash") != POLICY_DEFINITION_HASH,
            decision.get("question_intent") != features["question_intent"],
            decision.get("baseline") != baseline,
            decision.get("candidate") != candidate,
            _digest(features) != decision.get("feature_vector_hash"),
            _digest(authority) != decision.get("decision_hash"),
            decision.get("decision_id")
            != "pers_" + sha256(str(decision.get("decision_hash") or "").encode()).hexdigest()[:24],
        )
    ):
        raise ValueError("active personalization decision is not canonical")
    return PersonalizationPlanBinding(
        decision_id=decision["decision_id"],
        decision_hash=decision["decision_hash"],
        child_id=decision["child_id"],
        event_id=decision["event_id"],
        event_type=decision["event_type"],
        event_text_hash=decision["event_text_hash"],
        workflow=decision["workflow"],
        proposal_hash=decision["proposal_hash"],
        baseline_plan_id=decision["baseline_plan_id"],
        baseline_plan_hash=decision["baseline_plan_hash"],
        concept_ids=decision["concept_ids"],
        concept_refs=decision["concept_refs"],
        question_intent=decision["question_intent"],
        policy_id=decision["policy_id"],
        policy_version=decision["policy_version"],
        definition_hash=decision["definition_hash"],
        family_authorization_epoch=decision["family_authorization_epoch"],
        policy_authorization_epoch=decision["policy_authorization_epoch"],
        evaluation_id=decision["evaluation_id"],
        promotion_review_id=decision["promotion_review_id"],
        baseline_strategy_id=baseline["strategy_id"],
        interest_support=candidate["interest_support"],
        challenge_assumption=candidate["challenge_assumption"],
        applied_fields=["interest_support", "challenge_assumption"],
    )


def decision_trace(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": decision["decision_id"],
        "decision_hash": decision["decision_hash"],
        "mode": decision["mode"],
        "policy_id": decision["policy_id"],
        "policy_version": decision["policy_version"],
        "evaluation_id": decision["evaluation_id"],
        "promotion_review_id": decision["promotion_review_id"],
        "family_authorization_epoch": decision["family_authorization_epoch"],
        "policy_authorization_epoch": decision["policy_authorization_epoch"],
        "event_id": decision["event_id"],
        "proposal_hash": decision["proposal_hash"],
        "concept_refs": decision["concept_refs"],
        "baseline": decision["baseline"],
        "candidate": decision["candidate"],
        "features_hash": decision["feature_vector_hash"],
        "raw_features_model_facing": False,
    }


def assert_plan_authorization_in_transaction(conn: Any, plan: Any) -> None:
    """Fence response completion to the exact still-current authorization epochs."""

    binding = getattr(plan, "personalization", None)
    if binding is None:
        return
    row = conn.execute(
        """SELECT 1 FROM personalization_decisions d
           JOIN personalization_policies p
             ON p.policy_id=d.policy_id AND p.version=d.policy_version
           JOIN learner_model_controls c ON c.child_id=d.child_id
           WHERE d.decision_id=? AND d.decision_hash=? AND d.plan_id=? AND d.plan_hash=?
             AND d.mode='active' AND d.family_authorization_epoch=?
             AND d.policy_authorization_epoch=?
             AND p.status='active' AND p.definition_hash=d.definition_hash
             AND p.evaluation_id=d.evaluation_id
             AND p.promotion_review_id=d.promotion_review_id
             AND p.authorization_epoch=d.policy_authorization_epoch
             AND c.collection_enabled=1 AND c.recommendation_use_enabled=1
             AND c.recommendation_policy_id=d.policy_id
             AND c.recommendation_policy_version=d.policy_version
             AND c.recommendation_policy_hash=d.definition_hash
             AND c.personalization_epoch=d.family_authorization_epoch""",
        (
            binding.decision_id,
            binding.decision_hash,
            plan.plan_id,
            plan.plan_hash,
            binding.family_authorization_epoch,
            binding.policy_authorization_epoch,
        ),
    ).fetchone()
    if not row:
        raise PersonalizationAuthorizationChanged(
            "personalization authorization changed before durable response completion"
        )


def insert_decision_in_transaction(conn: Any, *, decision: dict[str, Any], plan: Any) -> None:
    """Bind a decision atomically to the immutable plan that consumed or shadowed it."""

    from .pedagogy import personalization_is_allowed

    expected = _decision_authority(decision)
    if (
        _digest(expected) != decision["decision_hash"]
        or _digest(decision["feature_vector"]) != decision["feature_vector_hash"]
        or decision["decision_id"] != "pers_" + sha256(decision["decision_hash"].encode()).hexdigest()[:24]
    ):
        raise ValueError("personalization decision failed canonical integrity verification")
    event = conn.execute(
        "SELECT id,type,child_id,text FROM events WHERE id=?", (decision["event_id"],)
    ).fetchone()
    policy = conn.execute(
        """SELECT * FROM personalization_policies
           WHERE policy_id=? AND version=? AND definition_hash=?""",
        (decision["policy_id"], decision["policy_version"], decision["definition_hash"]),
    ).fetchone()
    stored_plan = conn.execute(
        "SELECT proposal_json FROM pedagogy_plans WHERE plan_id=? AND plan_hash=?",
        (plan.plan_id, plan.plan_hash),
    ).fetchone()
    if not event or not policy or not stored_plan:
        raise ValueError("personalization decision references unknown canonical authority")
    stored_proposal = jload(stored_plan["proposal_json"], {})
    event_matches = all(
        (
            str(event["child_id"] or "") == decision["child_id"] == plan.child_id,
            str(event["id"]) == decision["event_id"] == plan.event_id,
            str(event["type"]) == decision["event_type"],
            sha256(str(event["text"] or "").encode()).hexdigest() == decision["event_text_hash"],
            plan.workflow == decision["workflow"],
            plan.question_intent == decision["question_intent"],
            [item.id for item in plan.concepts] == decision["concept_ids"],
            [
                {"id": item.id, "version": item.version}
                for item in plan.concepts
            ]
            == decision["concept_refs"],
            _digest(stored_proposal) == decision["proposal_hash"],
        )
    )
    definition = jload(policy["definition_json"], {})
    prior_feedback, recent = PersonalizationService._feedback_features(
        conn,
        decision["child_id"],
        decision["concept_refs"],
    )
    knowledge, interest = PersonalizationService._projection_features(
        conn,
        decision["child_id"],
        decision["concept_refs"],
    )
    expected_features = _assert_feature_vector(
        {
            "question_intent": str(stored_proposal.get("question_intent") or ""),
            "baseline_strategy_id": str(stored_proposal.get("primary_strategy_id") or ""),
            "domain_overlay_id": str(stored_proposal.get("domain_overlay_id") or ""),
            "prior_feedback_band": prior_feedback,
            "recent_challenge_signal": recent,
            "knowledge_band": knowledge,
            "interest_band": interest,
            "response_mode": response_mode_feature(
                list(dict(stored_proposal.get("evidence") or {}).get("response_modes") or [])
            ),
        }
    )
    controls = conn.execute(
        """SELECT collection_enabled,recommendation_use_enabled,recommendation_policy_id,
                  recommendation_policy_version,recommendation_policy_hash,personalization_epoch
           FROM learner_model_controls WHERE child_id=?""",
        (decision["child_id"],),
    ).fetchone()
    currently_active = bool(
        controls
        and controls["collection_enabled"]
        and controls["recommendation_use_enabled"]
        and controls["recommendation_policy_id"] == decision["policy_id"]
        and controls["recommendation_policy_version"] == decision["policy_version"]
        and controls["recommendation_policy_hash"] == decision["definition_hash"]
        and controls["personalization_epoch"] == decision["family_authorization_epoch"]
        and policy["status"] == "active"
        and policy["authorization_epoch"] == decision["policy_authorization_epoch"]
        and policy["evaluation_id"] == decision["evaluation_id"]
        and policy["promotion_review_id"] == decision["promotion_review_id"]
        and personalization_is_allowed(
            text=str(event["text"] or ""),
            workflow=decision["workflow"],
            question_intent=decision["question_intent"],
        )
    )
    baseline_matches = decision["baseline"] == {
        "strategy_id": plan.primary_strategy.id,
        "challenge_assumption": "grade_baseline_uncalibrated",
        "interest_support": "neutral",
    }
    candidate_matches = select_from_features(definition, decision["feature_vector"]) == decision["candidate"]
    if (
        not event_matches
        or not baseline_matches
        or not candidate_matches
        or decision["feature_vector"] != expected_features
        or (decision["mode"] == "active") != currently_active
    ):
        raise ValueError("personalization decision does not match event, proposal, or policy behavior")
    if decision["mode"] == "shadow":
        mode_matches = all(
            (
                decision["promotion_review_id"] is None,
                plan.personalization is None,
                plan.plan_id == decision["baseline_plan_id"],
                plan.plan_hash == decision["baseline_plan_hash"],
                plan.interest_support == "neutral",
                plan.challenge_assumption == "grade_baseline_uncalibrated",
            )
        )
    else:
        binding = plan.personalization
        mode_matches = bool(
            binding
            and binding.model_dump(mode="json")
            == {
                "decision_id": decision["decision_id"],
                "decision_hash": decision["decision_hash"],
                "child_id": decision["child_id"],
                "event_id": decision["event_id"],
                "event_type": decision["event_type"],
                "event_text_hash": decision["event_text_hash"],
                "workflow": decision["workflow"],
                "proposal_hash": decision["proposal_hash"],
                "baseline_plan_id": decision["baseline_plan_id"],
                "baseline_plan_hash": decision["baseline_plan_hash"],
                "concept_ids": decision["concept_ids"],
                "concept_refs": decision["concept_refs"],
                "question_intent": decision["question_intent"],
                "policy_id": decision["policy_id"],
                "policy_version": decision["policy_version"],
                "definition_hash": decision["definition_hash"],
                "family_authorization_epoch": decision["family_authorization_epoch"],
                "policy_authorization_epoch": decision["policy_authorization_epoch"],
                "evaluation_id": decision["evaluation_id"],
                "promotion_review_id": decision["promotion_review_id"],
                "mode": "active",
                "baseline_strategy_id": decision["baseline"]["strategy_id"],
                "interest_support": decision["candidate"]["interest_support"],
                "challenge_assumption": decision["candidate"]["challenge_assumption"],
                "applied_fields": ["interest_support", "challenge_assumption"],
            }
            and plan.interest_support == decision["candidate"]["interest_support"]
            and plan.challenge_assumption == decision["candidate"]["challenge_assumption"]
            and policy["status"] == "active"
            and policy["evaluation_id"] == decision["evaluation_id"]
            and policy["promotion_review_id"] == decision["promotion_review_id"]
            and personalization_is_allowed(
                text=str(event["text"] or ""),
                workflow=decision["workflow"],
                question_intent=decision["question_intent"],
            )
        )
    if not mode_matches:
        raise ValueError("personalization mode and plan binding do not match")
    conn.execute(
        """INSERT INTO personalization_decisions(
             decision_id,child_id,event_id,event_type,event_text_hash,workflow,proposal_hash,
             baseline_plan_id,baseline_plan_hash,concept_ids_json,concept_refs_json,
             question_intent,plan_id,plan_hash,
             policy_id,policy_version,definition_hash,family_authorization_epoch,
             policy_authorization_epoch,evaluation_id,promotion_review_id,mode,
             feature_vector_json,feature_vector_hash,baseline_json,candidate_json,decision_hash,created_at
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(decision_id) DO NOTHING""",
        (
            decision["decision_id"],
            decision["child_id"],
            decision["event_id"],
            decision["event_type"],
            decision["event_text_hash"],
            decision["workflow"],
            decision["proposal_hash"],
            decision["baseline_plan_id"],
            decision["baseline_plan_hash"],
            jdump(decision["concept_ids"]),
            jdump(decision["concept_refs"]),
            decision["question_intent"],
            plan.plan_id,
            plan.plan_hash,
            decision["policy_id"],
            decision["policy_version"],
            decision["definition_hash"],
            decision["family_authorization_epoch"],
            decision["policy_authorization_epoch"],
            decision["evaluation_id"],
            decision["promotion_review_id"],
            decision["mode"],
            jdump(decision["feature_vector"]),
            decision["feature_vector_hash"],
            jdump(decision["baseline"]),
            jdump(decision["candidate"]),
            decision["decision_hash"],
            utcnow(),
        ),
    )
    stored = conn.execute(
        "SELECT * FROM personalization_decisions WHERE decision_id=?",
        (decision["decision_id"],),
    ).fetchone()
    if not stored or any(
        (
            str(stored["child_id"]) != plan.child_id,
            str(stored["event_id"]) != decision["event_id"],
            str(stored["proposal_hash"]) != decision["proposal_hash"],
            str(stored["plan_id"]) != plan.plan_id,
            str(stored["plan_hash"]) != plan.plan_hash,
            str(stored["decision_hash"]) != decision["decision_hash"],
        )
    ):
        raise ValueError("conflicting personalization decision replay")


def record_feedback_signal(
    conn: Any,
    *,
    feedback_id: int,
    feedback: Any,
    plan_id: str | None,
    plan_hash: str | None,
) -> str:
    """Persist typed outcome labels only; free-form feedback stays in the private feedback table."""

    if feedback.outcome not in _OUTCOMES:
        raise ValueError("unsupported personalization feedback outcome")
    payload = {
        "feedback_id": feedback_id,
        "child_id": feedback.child_id,
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "outcome": feedback.outcome,
        "used": feedback.used,
        "chose_to_continue": feedback.chose_to_continue,
        "made_prediction": feedback.made_prediction,
        "gave_explanation": feedback.gave_explanation,
        "help_level": feedback.help_level,
        "asked_new_question": feedback.asked_new_question,
        "parent_correction": feedback.parent_correction,
    }
    signal_id = "persig_" + sha256(jdump(payload).encode()).hexdigest()[:24]
    conn.execute(
        """INSERT INTO personalization_feedback_signals(
             signal_id,feedback_id,child_id,plan_id,plan_hash,outcome,used,chose_to_continue,
             made_prediction,gave_explanation,help_level,asked_new_question,parent_correction,created_at
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            signal_id,
            feedback_id,
            feedback.child_id,
            plan_id,
            plan_hash,
            feedback.outcome,
            None if feedback.used is None else int(feedback.used),
            None if feedback.chose_to_continue is None else int(feedback.chose_to_continue),
            None if feedback.made_prediction is None else int(feedback.made_prediction),
            None if feedback.gave_explanation is None else int(feedback.gave_explanation),
            feedback.help_level,
            None if feedback.asked_new_question is None else int(feedback.asked_new_question),
            None if feedback.parent_correction is None else int(feedback.parent_correction),
            utcnow(),
        ),
    )
    return signal_id
