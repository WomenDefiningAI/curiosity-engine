from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import configuration_root
from .db import (
    authority_hash,
    init_db,
    jdump,
    jload,
    research_authorization_snapshot_hash,
    research_authorization_snapshot_id,
    research_connect,
    research_control_event_id,
    research_policy_event_id,
    utcnow,
)
from .learner_model import recompute_child_projections_in_transaction
from .personalization_policy import promotion_manifest
from .promotion_validation import validate_full_lab_report
from .research_authority_ledger import (
    family_authorization_high_water,
    record_research_authority,
)
from .research_policy import (
    DOMAINS,
    RESEARCH_FEATURE_SCHEMA_VERSION,
    RESEARCH_POLICY_DEFINITION_HASH,
    RESEARCH_POLICY_ID,
    RESEARCH_POLICY_VERSION,
    TRACE_METHOD_VERSION,
    calibration_case_hash,
    calibration_case_payload,
    calibration_corpus_manifest,
    calibration_corpus_manifest_hash,
    canonical_calibration_case,
    canonical_research_evaluation,
    history_from_evidence_rows,
    normalize_history,
    posterior_from_history,
    research_evaluation_matches,
)


class ResearchTrackAuthorizationChanged(RuntimeError):
    """The exact family or research-policy authorization is no longer current."""


def _digest(value: Any) -> str:
    return sha256(jdump(value).encode()).hexdigest()


def _identifier(value: str, label: str) -> str:
    cleaned = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", cleaned):
        raise ValueError(f"{label} must be a bounded stable identifier")
    return cleaned


def _sha256(value: str, label: str) -> str:
    cleaned = str(value).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", cleaned):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return cleaned


def _reason(value: str) -> str:
    cleaned = " ".join(str(value).split())
    if not 1 <= len(cleaned) <= 500:
        raise ValueError("a bounded reason is required")
    return cleaned


class ResearchTrackService:
    """Govern calibration, advisory traces, and non-deliverable proactive candidates."""

    def __init__(self, db_path: str | Path, repo_root: str | Path | None = None):
        self.db_path = str(db_path)
        self.repo_root = Path(repo_root).resolve() if repo_root else configuration_root().resolve()
        init_db(self.db_path, repo_root=self.repo_root)

    def _current_release_hash(self) -> str:
        manifest = promotion_manifest(self.repo_root)
        return _sha256(str(manifest.get("promotion_content_hash") or ""), "release content hash")

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
        event_id = research_policy_event_id(
            policy_id, version, action, actor_type, operator_id, reason, now
        )
        conn.execute(
            """INSERT INTO research_track_policy_events(
                 event_id,policy_id,policy_version,action,actor_type,operator_id,reason,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (event_id, policy_id, version, action, actor_type, operator_id, reason, now),
        )
        return event_id

    def _prepare_calibration_mutation(self, conn: Any) -> None:
        """Freeze approved authority and clear pre-approval evaluation snapshots."""

        protected = conn.execute(
            """SELECT status FROM research_track_policies
               WHERE status IN ('approved','active','disabled','retired') LIMIT 1"""
        ).fetchone()
        if protected:
            raise ValueError(
                "research calibration data is frozen once promotion approval is bound"
            )
        has_evaluation = bool(
            conn.execute("SELECT 1 FROM research_track_evaluations LIMIT 1").fetchone()
        )
        has_review = bool(
            conn.execute("SELECT 1 FROM research_track_promotion_reviews LIMIT 1").fetchone()
        )
        if not (has_evaluation or has_review):
            return
        now = utcnow()
        conn.execute(
            """UPDATE research_track_policies
               SET status='candidate',evaluation_id=NULL,updated_at=?
               WHERE status IN ('candidate','evaluated')""",
            (now,),
        )
        conn.execute(
            """DELETE FROM research_track_promotion_reviews
               WHERE NOT EXISTS(
                 SELECT 1 FROM research_uncertainty_traces trace
                 WHERE trace.promotion_review_id=research_track_promotion_reviews.review_id
               ) AND NOT EXISTS(
                 SELECT 1 FROM research_track_authorization_snapshots snapshot
                 WHERE snapshot.promotion_review_id=research_track_promotion_reviews.review_id
               )"""
        )
        conn.execute(
            """DELETE FROM research_track_evaluations
               WHERE NOT EXISTS(
                 SELECT 1 FROM research_uncertainty_traces trace
                 WHERE trace.evaluation_id=research_track_evaluations.evaluation_id
               ) AND NOT EXISTS(
                 SELECT 1 FROM research_track_promotion_reviews review
                 WHERE review.evaluation_id=research_track_evaluations.evaluation_id
               ) AND NOT EXISTS(
                 SELECT 1 FROM research_track_authorization_snapshots snapshot
                 WHERE snapshot.evaluation_id=research_track_evaluations.evaluation_id
               )"""
        )
        self._event(
            conn,
            policy_id=RESEARCH_POLICY_ID,
            version=RESEARCH_POLICY_VERSION,
            action="rolled_back",
            actor_type="system",
            reason="Calibration inputs changed before approval; prior evaluation authority was cleared.",
        )

    def import_calibration_case(
        self,
        *,
        case_id: str,
        source_kind: str,
        license_status: str,
        split: str,
        group_digest: str,
        task_signature_digest: str,
        domain: str,
        history: dict[str, Any],
        observed_success: int,
        source_digest: str,
        operator_id: str,
        privacy_approval_id: str | None = None,
    ) -> dict[str, Any]:
        case_id = _identifier(case_id, "case ID")
        operator_id = _identifier(operator_id, "operator ID")
        group_digest = _sha256(group_digest, "group digest")
        task_signature_digest = _sha256(task_signature_digest, "task signature digest")
        source_digest = _sha256(source_digest, "source digest")
        if source_kind not in {"public_fixture", "synthetic"}:
            raise ValueError("approved real cases must arrive through a reviewed corpus manifest")
        expected_license = {
            "public_fixture": "public",
            "synthetic": "synthetic",
        }[source_kind]
        if license_status != expected_license:
            raise ValueError("research source and license status do not match")
        if privacy_approval_id is not None:
            raise ValueError("public and synthetic cases cannot carry private approval references")
        if split not in {"train", "heldout"}:
            raise ValueError("research calibration split must be train or heldout")
        if domain not in DOMAINS:
            raise ValueError("unsupported deidentified research domain")
        if isinstance(observed_success, bool) or observed_success not in {0, 1}:
            raise ValueError("research calibration outcome must be the integer zero or one")
        normalized_history = normalize_history(history)
        payload = calibration_case_payload(
            case_id=case_id,
            source_kind=source_kind,
            license_status=license_status,
            privacy_approval_id=privacy_approval_id,
            corpus_id=None,
            split=split,
            group_digest=group_digest,
            task_signature_digest=task_signature_digest,
            domain=domain,
            history=normalized_history,
            observed_success=observed_success,
            source_digest=source_digest,
        )
        case_hash = calibration_case_hash(
            case_id=case_id,
            source_kind=source_kind,
            license_status=license_status,
            privacy_approval_id=privacy_approval_id,
            corpus_id=None,
            split=split,
            group_digest=group_digest,
            task_signature_digest=task_signature_digest,
            domain=domain,
            history=normalized_history,
            observed_success=observed_success,
            source_digest=source_digest,
        )
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            stored = conn.execute(
                "SELECT * FROM research_calibration_cases WHERE case_id=?", (case_id,)
            ).fetchone()
            if stored is None:
                self._prepare_calibration_mutation(conn)
                conn.execute(
                    """INSERT INTO research_calibration_cases(
                     case_id,source_kind,license_status,privacy_approval_id,corpus_id,feature_schema_version,
                     split,group_digest,task_signature_digest,domain,history_json,observed_success,
                     source_digest,case_hash,created_by_operator,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(case_id) DO NOTHING""",
                    (
                        case_id,
                        source_kind,
                        license_status,
                        privacy_approval_id,
                        None,
                        RESEARCH_FEATURE_SCHEMA_VERSION,
                        split,
                        group_digest,
                        task_signature_digest,
                        domain,
                        jdump(normalized_history),
                        observed_success,
                        source_digest,
                        case_hash,
                        operator_id,
                        utcnow(),
                    ),
                )
                stored = conn.execute(
                    "SELECT * FROM research_calibration_cases WHERE case_id=?", (case_id,)
                ).fetchone()
        if not stored or str(stored["case_hash"]) != case_hash:
            raise ValueError("conflicting research calibration case")
        return {**payload, "case_hash": case_hash}

    def propose_calibration_corpus(
        self,
        *,
        corpus_id: str,
        cases: list[dict[str, Any]],
        operator_id: str,
    ) -> dict[str, Any]:
        corpus_id = _identifier(corpus_id, "corpus ID")
        operator_id = _identifier(operator_id, "operator ID")
        if not 1 <= len(cases) <= 10_000:
            raise ValueError("a calibration corpus must contain 1..10000 cases")
        canonical: list[dict[str, Any]] = []
        seen: set[str] = set()
        allowed_case_fields = {
            "case_id",
            "split",
            "group_digest",
            "task_signature_digest",
            "domain",
            "history",
            "observed_success",
            "source_digest",
        }
        for raw in cases:
            if not isinstance(raw, dict) or set(raw) != allowed_case_fields:
                raise ValueError("calibration corpus cases must use the exact deidentified schema")
            case_id = _identifier(str(raw.get("case_id") or ""), "case ID")
            if case_id in seen:
                raise ValueError("calibration corpus case IDs must be unique")
            seen.add(case_id)
            split = str(raw.get("split") or "")
            domain = str(raw.get("domain") or "")
            if split not in {"train", "heldout"} or domain not in DOMAINS:
                raise ValueError("calibration corpus case has an invalid split or domain")
            observed_success = raw.get("observed_success")
            if isinstance(observed_success, bool) or observed_success not in {0, 1}:
                raise ValueError("calibration corpus outcomes must be integer zero or one")
            group_digest = _sha256(str(raw.get("group_digest") or ""), "group digest")
            task_digest = _sha256(
                str(raw.get("task_signature_digest") or ""), "task signature digest"
            )
            source_digest = _sha256(str(raw.get("source_digest") or ""), "source digest")
            history = normalize_history(dict(raw.get("history") or {}))
            payload = calibration_case_payload(
                case_id=case_id,
                source_kind="approved_deidentified",
                license_status="approved",
                privacy_approval_id=corpus_id,
                corpus_id=corpus_id,
                split=split,
                group_digest=group_digest,
                task_signature_digest=task_digest,
                domain=domain,
                history=history,
                observed_success=int(observed_success),
                source_digest=source_digest,
            )
            canonical.append({**payload, "case_hash": calibration_case_hash(**payload)})
        manifest = calibration_corpus_manifest(
            corpus_id, [str(item["case_hash"]) for item in canonical]
        )
        manifest_hash = calibration_corpus_manifest_hash(
            corpus_id, [str(item["case_hash"]) for item in canonical]
        )
        now = utcnow()
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing_corpus = conn.execute(
                "SELECT manifest_hash,status FROM research_calibration_corpora WHERE corpus_id=?",
                (corpus_id,),
            ).fetchone()
            if existing_corpus is None:
                self._prepare_calibration_mutation(conn)
            conn.execute(
                """INSERT INTO research_calibration_corpora(
                     corpus_id,feature_schema_version,manifest_json,manifest_hash,case_count,status,
                     proposed_by_operator,created_at
                   ) VALUES(?,?,?,?,?,'pending_review',?,?) ON CONFLICT(corpus_id) DO NOTHING""",
                (
                    corpus_id,
                    RESEARCH_FEATURE_SCHEMA_VERSION,
                    jdump(manifest),
                    manifest_hash,
                    len(canonical),
                    operator_id,
                    now,
                ),
            )
            stored = conn.execute(
                "SELECT manifest_hash,status FROM research_calibration_corpora WHERE corpus_id=?",
                (corpus_id,),
            ).fetchone()
            if not stored or stored["manifest_hash"] != manifest_hash or stored["status"] != "pending_review":
                raise ValueError("conflicting or already reviewed calibration corpus")
            for item in canonical:
                conn.execute(
                    """INSERT INTO research_calibration_cases(
                         case_id,source_kind,license_status,privacy_approval_id,corpus_id,
                         feature_schema_version,split,group_digest,task_signature_digest,domain,
                         history_json,observed_success,source_digest,case_hash,created_by_operator,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(case_id) DO NOTHING""",
                    (
                        item["case_id"],
                        item["source_kind"],
                        item["license_status"],
                        item["privacy_approval_id"],
                        item["corpus_id"],
                        item["feature_schema_version"],
                        item["split"],
                        item["group_digest"],
                        item["task_signature_digest"],
                        item["domain"],
                        jdump(item["history"]),
                        item["observed_success"],
                        item["source_digest"],
                        item["case_hash"],
                        operator_id,
                        now,
                    ),
                )
            count = conn.execute(
                "SELECT COUNT(*) FROM research_calibration_cases WHERE corpus_id=?", (corpus_id,)
            ).fetchone()[0]
            if count != len(canonical):
                raise ValueError("calibration corpus case IDs conflict with existing data")
        return {
            "corpus_id": corpus_id,
            "manifest_hash": manifest_hash,
            "case_count": len(canonical),
            "status": "pending_review",
            "privacy_review_required": True,
        }

    def approve_calibration_corpus(
        self,
        *,
        corpus_id: str,
        reviewer_id: str,
        reason: str,
    ) -> dict[str, Any]:
        corpus_id = _identifier(corpus_id, "corpus ID")
        reviewer_id = _identifier(reviewer_id, "reviewer ID")
        reason = _reason(reason)
        now = utcnow()
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            corpus = conn.execute(
                "SELECT * FROM research_calibration_corpora WHERE corpus_id=?", (corpus_id,)
            ).fetchone()
            if not corpus or corpus["status"] != "pending_review":
                raise ValueError("calibration corpus is not pending independent review")
            if corpus["proposed_by_operator"] == reviewer_id:
                raise ValueError("calibration corpus reviewer must differ from its proposing operator")
            rows = conn.execute(
                """SELECT *,'pending_review' AS corpus_status
                   FROM research_calibration_cases WHERE corpus_id=? ORDER BY case_id""",
                (corpus_id,),
            ).fetchall()
            cases = [self._canonical_case(row) for row in rows]
            expected = calibration_corpus_manifest(
                corpus_id, [str(item["case_hash"]) for item in cases]
            )
            if (
                len(cases) != int(corpus["case_count"])
                or jdump(expected) != str(corpus["manifest_json"])
                or _digest(expected) != str(corpus["manifest_hash"])
            ):
                raise ValueError("calibration corpus manifest failed canonical integrity")
            self._prepare_calibration_mutation(conn)
            conn.execute(
                """UPDATE research_calibration_corpora
                   SET status='approved',approved_by_reviewer=?,approval_reason=?,approved_at=?
                   WHERE corpus_id=?""",
                (reviewer_id, reason, now, corpus_id),
            )
        return {
            "corpus_id": corpus_id,
            "manifest_hash": str(corpus["manifest_hash"]),
            "status": "approved",
            "reviewer_id": reviewer_id,
        }

    @staticmethod
    def _canonical_case(row: Any) -> dict[str, Any]:
        return canonical_calibration_case(row)

    def _canonical_evaluation(
        self,
        policy: Any,
        *,
        conn: Any | None = None,
        release_content_hash: str | None = None,
    ) -> dict[str, Any]:
        release_content_hash = release_content_hash or self._current_release_hash()
        query = """SELECT c.*,corpus.status AS corpus_status
                   FROM research_calibration_cases c
                   LEFT JOIN research_calibration_corpora corpus ON corpus.corpus_id=c.corpus_id
                   WHERE c.source_kind!='approved_deidentified' OR corpus.status='approved'
                   ORDER BY c.case_id"""
        corpus_query = """SELECT * FROM research_calibration_corpora
                           WHERE status='approved' ORDER BY corpus_id"""
        if conn is None:
            with research_connect(self.db_path) as read_conn:
                rows = read_conn.execute(query).fetchall()
                corpus_rows = read_conn.execute(corpus_query).fetchall()
        else:
            rows = conn.execute(query).fetchall()
            corpus_rows = conn.execute(corpus_query).fetchall()
        return canonical_research_evaluation(
            policy,
            list(rows),
            list(corpus_rows),
            release_content_hash,
        )

    def evaluate(
        self,
        policy_id: str = RESEARCH_POLICY_ID,
        version: int = RESEARCH_POLICY_VERSION,
    ) -> dict[str, Any]:
        policy_id = _identifier(policy_id, "policy ID")
        init_db(self.db_path, repo_root=self.repo_root)
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            policy = conn.execute(
                "SELECT * FROM research_track_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
            if not policy:
                raise ValueError("unknown research-track policy")
            if str(policy["status"]) not in {"candidate", "evaluated"}:
                raise ValueError("only a candidate research policy can be evaluated")
            release_content_hash = self._current_release_hash()
            canonical = self._canonical_evaluation(
                policy,
                conn=conn,
                release_content_hash=release_content_hash,
            )
            now = utcnow()
            conn.execute(
                """INSERT INTO research_track_evaluations(
                     evaluation_id,policy_id,policy_version,definition_hash,release_content_hash,
                     dataset_digest,case_count,
                     real_case_count,train_count,heldout_count,distinct_task_count,distinct_domain_count,
                     heldout_positive_count,heldout_negative_count,split_overlap_count,brier_score,
                     expected_calibration_error,gates_json,results_hash,evaluator,evaluator_digest,status,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(evaluation_id) DO NOTHING""",
                (
                    canonical["evaluation_id"],
                    policy_id,
                    version,
                    canonical["definition_hash"],
                    canonical["release_content_hash"],
                    canonical["dataset_digest"],
                    canonical["case_count"],
                    canonical["real_case_count"],
                    canonical["train_count"],
                    canonical["heldout_count"],
                    canonical["distinct_task_count"],
                    canonical["distinct_domain_count"],
                    canonical["heldout_positive_count"],
                    canonical["heldout_negative_count"],
                    canonical["split_overlap_count"],
                    canonical["brier_score"],
                    canonical["expected_calibration_error"],
                    jdump(canonical["gates"]),
                    canonical["results_hash"],
                    canonical["evaluator"],
                    canonical["evaluator_digest"],
                    canonical["status"],
                    now,
                ),
            )
            if self._current_release_hash() != release_content_hash:
                raise ResearchTrackAuthorizationChanged(
                    "research release changed while the evaluation was being stored"
                )
            conn.execute(
                """UPDATE research_track_policies SET status=?,evaluation_id=?,updated_at=?
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
                reason=f"Held-out research calibration {canonical['evaluation_id']}: {canonical['status']}.",
            )
        return {
            key: canonical[key]
            for key in (
                "evaluation_id",
                "status",
                "case_count",
                "real_case_count",
                "train_count",
                "heldout_count",
                "distinct_task_count",
                "distinct_domain_count",
                "heldout_positive_count",
                "heldout_negative_count",
                "split_overlap_count",
                "brier_score",
                "expected_calibration_error",
                "gates",
            )
        } | {"auto_promote": False, "operator_approval_required": True}

    def _trusted_evaluation(self, conn: Any, policy_id: str, version: int, evaluation_id: str) -> Any:
        row = conn.execute(
            """SELECT e.*,p.version AS version,p.feature_schema_version,p.definition_json,
                      p.definition_hash AS current_definition_hash
               FROM research_track_evaluations e JOIN research_track_policies p
                 ON p.policy_id=e.policy_id AND p.version=e.policy_version
               WHERE e.evaluation_id=? AND e.policy_id=? AND e.policy_version=? AND e.status='pass'""",
            (evaluation_id, policy_id, version),
        ).fetchone()
        if not row:
            raise ValueError("research promotion requires an exact passing held-out evaluation")
        current_release_hash = self._current_release_hash()
        canonical = self._canonical_evaluation(
            row, conn=conn, release_content_hash=current_release_hash
        )
        if (
            str(row["current_definition_hash"]) != canonical["definition_hash"]
            or not research_evaluation_matches(row, canonical)
        ):
            raise ValueError("research evaluation no longer matches current trusted inputs")
        return row

    def review_for_promotion(
        self,
        *,
        policy_id: str,
        version: int,
        evaluation_id: str,
        reviewer_id: str,
    ) -> dict[str, Any]:
        policy_id = _identifier(policy_id, "policy ID")
        reviewer_id = _identifier(reviewer_id, "reviewer ID")
        with research_connect(self.db_path) as conn:
            self._trusted_evaluation(conn, policy_id, version, evaluation_id)
            policy = conn.execute(
                "SELECT * FROM research_track_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
        if not policy or policy["status"] != "evaluated" or policy["evaluation_id"] != evaluation_id:
            raise ValueError("research policy is not ready for separate promotion review")
        from .lab import evaluate as evaluate_lab

        lab_report = evaluate_lab(self.repo_root, live_judge=True)
        from .personalization import PersonalizationService

        PersonalizationService(self.db_path, self.repo_root)._validate_lab_report(lab_report)
        research_suite = dict((lab_report.get("suites") or {}).get("research_track") or {})
        if (
            int(research_suite.get("cases", 0)) <= 0
            or int(research_suite.get("failed", -1)) != 0
            or int(research_suite.get("passed", 0)) != int(research_suite.get("cases", 0))
        ):
            raise ValueError("full Lab report is missing the passing research-track suite")
        lab_report_hash = _digest(lab_report)
        release_content_hash = self._current_release_hash()
        report_release_hash = str(
            (lab_report.get("personalization_policy") or {}).get("promotion_content_hash") or ""
        )
        if report_release_hash != release_content_hash:
            raise ResearchTrackAuthorizationChanged(
                "research release changed while the full Lab review was running"
            )
        identity = {
            "policy_id": policy_id,
            "policy_version": version,
            "definition_hash": str(policy["definition_hash"]),
            "release_content_hash": release_content_hash,
            "evaluation_id": evaluation_id,
            "reviewer_id": reviewer_id,
            "lab_report_hash": lab_report_hash,
        }
        review_id = "rtreview_" + sha256(jdump(identity).encode()).hexdigest()[:24]
        now = utcnow()
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._trusted_evaluation(conn, policy_id, version, evaluation_id)
            conn.execute(
                """INSERT INTO research_track_promotion_reviews(
                     review_id,policy_id,policy_version,definition_hash,release_content_hash,
                     evaluation_id,reviewer_id,
                     lab_report_json,lab_report_hash,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(review_id) DO NOTHING""",
                (
                    review_id,
                    policy_id,
                    version,
                    policy["definition_hash"],
                    release_content_hash,
                    evaluation_id,
                    reviewer_id,
                    jdump(lab_report),
                    lab_report_hash,
                    now,
                ),
            )
            stored = conn.execute(
                "SELECT lab_report_hash FROM research_track_promotion_reviews WHERE review_id=?",
                (review_id,),
            ).fetchone()
            if not stored or str(stored["lab_report_hash"]) != lab_report_hash:
                raise ValueError("conflicting research promotion review")
            self._event(
                conn,
                policy_id=policy_id,
                version=version,
                action="promotion_reviewed",
                actor_type="system",
                reason=f"Independent full Lab report {lab_report_hash[:16]} reviewed by {reviewer_id}.",
            )
        return {
            "review_id": review_id,
            "policy_id": policy_id,
            "version": version,
            "evaluation_id": evaluation_id,
            "reviewer_id": reviewer_id,
            "lab_report_hash": lab_report_hash,
            "release_content_hash": release_content_hash,
            "status": "pass",
        }

    def _trusted_review(
        self,
        conn: Any,
        *,
        policy_id: str,
        version: int,
        evaluation_id: str,
        review_id: str,
        validate_lab: bool = True,
    ) -> Any:
        row = conn.execute(
            """SELECT * FROM research_track_promotion_reviews
               WHERE review_id=? AND policy_id=? AND policy_version=? AND evaluation_id=?""",
            (review_id, policy_id, version, evaluation_id),
        ).fetchone()
        if not row:
            raise ValueError("research promotion requires an exact full Lab review")
        report = jload(row["lab_report_json"], {})
        lab_report_hash = _digest(report)
        identity = {
            "policy_id": policy_id,
            "policy_version": version,
            "definition_hash": str(row["definition_hash"]),
            "release_content_hash": str(row["release_content_hash"]),
            "evaluation_id": evaluation_id,
            "reviewer_id": str(row["reviewer_id"]),
            "lab_report_hash": lab_report_hash,
        }
        if (
            lab_report_hash != str(row["lab_report_hash"])
            or str(row["review_id"])
            != "rtreview_" + sha256(jdump(identity).encode()).hexdigest()[:24]
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", str(row["reviewer_id"]))
        ):
            raise ValueError("research promotion review failed canonical integrity")
        current_release_hash = self._current_release_hash()
        if (
            str(row["release_content_hash"]) != current_release_hash
            or str((report.get("personalization_policy") or {}).get("promotion_content_hash") or "")
            != current_release_hash
        ):
            raise ResearchTrackAuthorizationChanged(
                "research promotion review belongs to a different software release"
            )
        if validate_lab:
            validate_full_lab_report(report, self.repo_root)
        suite = dict((report.get("suites") or {}).get("research_track") or {})
        if (
            int(suite.get("failed", -1)) != 0
            or int(suite.get("cases", 0)) <= 0
            or int(suite.get("passed", 0)) != int(suite.get("cases", 0))
        ):
            raise ValueError("research promotion review no longer passes its suite")
        return row

    @staticmethod
    def _policy_lifecycle_is_canonical(conn: Any, policy: Any) -> bool:
        policy_id = str(policy["policy_id"])
        version = int(policy["version"])
        rows = conn.execute(
            """SELECT * FROM research_track_policy_events
               WHERE policy_id=? AND policy_version=? ORDER BY created_at,event_id""",
            (policy_id, version),
        ).fetchall()
        if not all(
            (
                str(row["event_id"])
                == research_policy_event_id(
                    row["policy_id"],
                    row["policy_version"],
                    row["action"],
                    row["actor_type"],
                    row["operator_id"],
                    row["reason"],
                    row["created_at"],
                )
                and (
                    (row["actor_type"] == "system" and row["operator_id"] is None)
                    or (row["actor_type"] == "operator" and bool(row["operator_id"]))
                )
            )
            for row in rows
        ):
            return False
        snapshots = conn.execute(
            """SELECT * FROM research_track_authorization_snapshots
               WHERE policy_id=? AND policy_version=?
               ORDER BY policy_authorization_epoch,snapshot_id""",
            (policy_id, version),
        ).fetchall()
        expected_snapshot_epochs = list(range(1, len(snapshots) * 2, 2))
        if [int(row["policy_authorization_epoch"]) for row in snapshots] != expected_snapshot_epochs:
            return False
        snapshot_by_activation = {
            str(row["activation_event_id"]): row for row in snapshots
        }
        activation_ids = [
            str(row["event_id"]) for row in rows if row["action"] == "activated"
        ]
        if activation_ids != [str(row["activation_event_id"]) for row in snapshots]:
            return False

        active = False
        derived_epoch = 0
        latest_governing_action: str | None = None
        for row in rows:
            action = str(row["action"])
            if action == "activated":
                if active:
                    return False
                derived_epoch += 1
                snapshot = snapshot_by_activation.get(str(row["event_id"]))
                if (
                    snapshot is None
                    or int(snapshot["policy_authorization_epoch"]) != derived_epoch
                ):
                    return False
                active = True
                latest_governing_action = action
            elif action == "disabled":
                if not active:
                    return False
                derived_epoch += 1
                active = False
                latest_governing_action = action
            elif action == "rolled_back":
                if active:
                    derived_epoch += 1
                active = False
                latest_governing_action = action

        status = str(policy["status"])
        if derived_epoch != int(policy["authorization_epoch"]):
            return False
        if (status == "active") != active:
            return False
        if status == "active" and latest_governing_action != "activated":
            return False
        if status in {"disabled", "retired"} and latest_governing_action != "disabled":
            return False
        if status in {"candidate", "evaluated", "approved"} and latest_governing_action not in {
            None,
            "rolled_back",
        }:
            return False
        return True

    @staticmethod
    def _required_policy_event(
        conn: Any,
        *,
        policy_id: str,
        version: int,
        action: str,
        actor_type: str,
        reason: str,
        operator_id: str | None,
    ) -> bool:
        return bool(
            conn.execute(
                """SELECT 1 FROM research_track_policy_events
                   WHERE policy_id=? AND policy_version=? AND action=? AND actor_type=?
                     AND operator_id IS ? AND reason=? LIMIT 1""",
                (policy_id, version, action, actor_type, operator_id, reason),
            ).fetchone()
        )

    def _trusted_promoted_policy(
        self,
        conn: Any,
        policy: Any,
        *,
        allowed_statuses: set[str],
    ) -> Any:
        if not policy or str(policy["status"]) not in allowed_statuses:
            raise ResearchTrackAuthorizationChanged("research policy is not in the required state")
        policy_id = str(policy["policy_id"])
        version = int(policy["version"])
        current_release_hash = self._current_release_hash()
        if (
            str(policy["definition_hash"]) != RESEARCH_POLICY_DEFINITION_HASH
            or str(policy["release_content_hash"]) != current_release_hash
            or not policy["evaluation_id"]
            or not policy["promotion_review_id"]
            or not policy["approved_by_operator"]
            or not policy["approval_reason"]
            or not policy["approved_at"]
        ):
            raise ResearchTrackAuthorizationChanged(
                "research policy promotion authority is incomplete or stale"
            )
        evaluation = self._trusted_evaluation(
            conn, policy_id, version, str(policy["evaluation_id"])
        )
        review = self._trusted_review(
            conn,
            policy_id=policy_id,
            version=version,
            evaluation_id=str(policy["evaluation_id"]),
            review_id=str(policy["promotion_review_id"]),
        )
        if (
            str(evaluation["definition_hash"]) != str(policy["definition_hash"])
            or str(review["definition_hash"]) != str(policy["definition_hash"])
            or str(review["release_content_hash"]) != current_release_hash
            or str(review["reviewer_id"]) == str(policy["approved_by_operator"])
            or not self._policy_lifecycle_is_canonical(conn, policy)
            or not self._required_policy_event(
                conn,
                policy_id=policy_id,
                version=version,
                action="evaluated",
                actor_type="system",
                operator_id=None,
                reason=f"Held-out research calibration {policy['evaluation_id']}: pass.",
            )
            or not self._required_policy_event(
                conn,
                policy_id=policy_id,
                version=version,
                action="promotion_reviewed",
                actor_type="system",
                operator_id=None,
                reason=(
                    f"Independent full Lab report {str(review['lab_report_hash'])[:16]} "
                    f"reviewed by {review['reviewer_id']}."
                ),
            )
            or not self._required_policy_event(
                conn,
                policy_id=policy_id,
                version=version,
                action="approved",
                actor_type="operator",
                operator_id=str(policy["approved_by_operator"]),
                reason=str(policy["approval_reason"]),
            )
        ):
            raise ResearchTrackAuthorizationChanged(
                "research policy promotion lineage failed canonical validation"
            )
        status = str(policy["status"])
        if status == "approved" and (
            int(policy["authorization_epoch"]) < 0
            or int(policy["authorization_epoch"]) % 2 != 0
            or policy["activation_reason"] is not None
            or policy["activated_at"] is not None
        ):
            raise ResearchTrackAuthorizationChanged(
                "approved research policy has an invalid authorization epoch"
            )
        if status in {"active", "disabled", "retired"} and (
            int(policy["authorization_epoch"]) < (1 if status == "active" else 2)
            or not policy["activation_reason"]
            or not policy["activated_at"]
            or not self._required_policy_event(
                conn,
                policy_id=policy_id,
                version=version,
                action="activated",
                actor_type="operator",
                operator_id=str(policy["approved_by_operator"]),
                reason=str(policy["activation_reason"]),
            )
        ):
            raise ResearchTrackAuthorizationChanged(
                "active research policy lacks its exact activation lineage"
            )
        if status == "active" and (
            int(policy["authorization_epoch"]) <= 0
            or int(policy["authorization_epoch"]) % 2 != 1
            or not conn.execute(
                """SELECT 1 FROM research_track_authorization_snapshots
                   WHERE policy_id=? AND policy_version=? AND definition_hash=?
                     AND release_content_hash=? AND evaluation_id=?
                     AND promotion_review_id=? AND policy_authorization_epoch=?
                     AND approved_by_operator=? AND approval_reason=?
                     AND activation_reason=? AND approved_at=? AND activated_at=?""",
                (
                    policy_id,
                    version,
                    policy["definition_hash"],
                    policy["release_content_hash"],
                    policy["evaluation_id"],
                    policy["promotion_review_id"],
                    policy["authorization_epoch"],
                    policy["approved_by_operator"],
                    policy["approval_reason"],
                    policy["activation_reason"],
                    policy["approved_at"],
                    policy["activated_at"],
                ),
            ).fetchone()
        ):
            raise ResearchTrackAuthorizationChanged(
                "active research policy has an invalid authorization epoch"
            )
        if status in {"disabled", "retired"} and (
            int(policy["authorization_epoch"]) <= 0
            or int(policy["authorization_epoch"]) % 2 != 0
            or not conn.execute(
                """SELECT 1 FROM research_track_policy_events
                   WHERE policy_id=? AND policy_version=? AND action='disabled'
                     AND actor_type IN ('operator','system') LIMIT 1""",
                (policy_id, version),
            ).fetchone()
        ):
            raise ResearchTrackAuthorizationChanged(
                "stopped research policy lacks its disable lineage"
            )
        return policy

    @staticmethod
    def _control_lineage_is_canonical(conn: Any, control: Any) -> bool:
        rows = conn.execute(
            """SELECT * FROM research_track_control_events
               WHERE child_id=? ORDER BY family_authorization_epoch,event_id""",
            (control["child_id"],),
        ).fetchall()
        epoch = int(control["family_authorization_epoch"])
        epochs = [int(row["family_authorization_epoch"]) for row in rows]
        if not epochs or epochs[-1] != epoch or epochs != list(range(epochs[0], epoch + 1)):
            return False
        for row in rows:
            enabled = bool(row["enabled"])
            binding = (
                row["policy_id"],
                row["policy_version"],
                row["definition_hash"],
                row["release_content_hash"],
            )
            if enabled:
                if any(value is None for value in binding) or row["actor_type"] != "parent":
                    return False
            elif any(value is not None for value in binding):
                return False
            if (
                row["actor_type"] == "system"
                and row["reason"] == "Learner-model collection was paused."
            ):
                from .db import research_collection_pause_event_id

                valid_id = str(row["event_id"]) == research_collection_pause_event_id(
                    row["child_id"], row["created_at"]
                )
            else:
                valid_id = str(row["event_id"]) == research_control_event_id(
                    row["child_id"],
                    row["enabled"],
                    row["policy_id"],
                    row["policy_version"],
                    row["definition_hash"],
                    row["release_content_hash"],
                    row["family_authorization_epoch"],
                    row["actor_type"],
                    row["reason"],
                    row["created_at"],
                )
            if not valid_id:
                return False
        latest = rows[-1] if rows else None
        return bool(
            latest
            and int(latest["enabled"]) == int(control["enabled"])
            and latest["policy_id"] == control["policy_id"]
            and latest["policy_version"] == control["policy_version"]
            and latest["definition_hash"] == control["definition_hash"]
            and latest["release_content_hash"] == control["release_content_hash"]
            and str(latest["reason"]) == str(control["reason"])
            and str(latest["created_at"]) == str(control["updated_at"])
        )

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
        # Lab validation initializes and reads the database. Do it before taking
        # the promotion write lock; the review row is immutable, and the exact
        # row is rechecked within the transaction below.
        with research_connect(self.db_path) as conn:
            self._trusted_review(
                conn,
                policy_id=policy_id,
                version=version,
                evaluation_id=evaluation_id,
                review_id=promotion_review_id,
            )
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._trusted_evaluation(conn, policy_id, version, evaluation_id)
            review = self._trusted_review(
                conn,
                policy_id=policy_id,
                version=version,
                evaluation_id=evaluation_id,
                review_id=promotion_review_id,
                validate_lab=False,
            )
            if str(review["reviewer_id"]) == operator_id:
                raise ValueError("research operator must be separate from the promotion reviewer")
            row = conn.execute(
                "SELECT status,evaluation_id FROM research_track_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
            if not row or row["status"] not in {"evaluated", "approved"} or row["evaluation_id"] != evaluation_id:
                raise ValueError("research policy is not ready for operator approval")
            conn.execute(
                """UPDATE research_track_policies
                   SET status='approved',promotion_review_id=?,approved_by_operator=?,approval_reason=?,
                       release_content_hash=?,approved_at=?,updated_at=?
                   WHERE policy_id=? AND version=?""",
                (
                    promotion_review_id,
                    operator_id,
                    reason,
                    str(review["release_content_hash"]),
                    now,
                    now,
                    policy_id,
                    version,
                ),
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
            "release_content_hash": str(review["release_content_hash"]),
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
        release_content_hash = self._current_release_hash()
        with research_connect(self.db_path) as conn:
            candidate = conn.execute(
                "SELECT evaluation_id,promotion_review_id FROM research_track_policies "
                "WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
            if candidate and candidate["evaluation_id"] and candidate["promotion_review_id"]:
                self._trusted_review(
                    conn,
                    policy_id=policy_id,
                    version=version,
                    evaluation_id=str(candidate["evaluation_id"]),
                    review_id=str(candidate["promotion_review_id"]),
                )
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM research_track_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
            if not row or row["status"] != "approved" or row["approved_by_operator"] != operator_id:
                raise ValueError("only the approving operator may activate an approved research policy")
            if str(row["release_content_hash"]) != release_content_hash:
                raise ResearchTrackAuthorizationChanged(
                    "approved research policy belongs to a different software release"
                )
            self._trusted_promoted_policy(
                conn,
                row,
                allowed_statuses={"approved"},
            )
            conn.execute(
                """UPDATE research_track_policies
                   SET status='disabled',authorization_epoch=authorization_epoch+1,updated_at=?
                   WHERE status='active' AND NOT (policy_id=? AND version=?)""",
                (now, policy_id, version),
            )
            conn.execute(
                """UPDATE research_track_controls
                   SET enabled=0,policy_id=NULL,policy_version=NULL,definition_hash=NULL,
                       release_content_hash=NULL,
                       family_authorization_epoch=family_authorization_epoch+1,
                       updated_by='operator',reason=?,updated_at=?
                   WHERE enabled=1 AND NOT (policy_id=? AND policy_version=?)""",
                (f"Research policy replaced: {reason}", now, policy_id, version),
            )
            conn.execute(
                """UPDATE research_track_policies
                   SET status='active',authorization_epoch=authorization_epoch+1,
                       activation_reason=?,activated_at=?,updated_at=?
                   WHERE policy_id=? AND version=?""",
                (reason, now, now, policy_id, version),
            )
            if self._current_release_hash() != release_content_hash:
                raise ResearchTrackAuthorizationChanged(
                    "research release changed while the policy was being activated"
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
            activated = conn.execute(
                "SELECT * FROM research_track_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
            approval_event = conn.execute(
                """SELECT event_id FROM research_track_policy_events
                   WHERE policy_id=? AND policy_version=? AND action='approved'
                     AND actor_type='operator' AND operator_id=? AND reason=?
                   ORDER BY created_at DESC,event_id DESC LIMIT 1""",
                (
                    policy_id,
                    version,
                    activated["approved_by_operator"],
                    activated["approval_reason"],
                ),
            ).fetchone()
            if approval_event is None:
                raise ResearchTrackAuthorizationChanged(
                    "research activation lacks its exact approval event"
                )
            snapshot_authority = (
                policy_id,
                version,
                str(activated["definition_hash"]),
                str(activated["release_content_hash"]),
                str(activated["evaluation_id"]),
                str(activated["promotion_review_id"]),
                int(activated["authorization_epoch"]),
                str(activated["approved_by_operator"]),
                str(activated["approval_reason"]),
                str(activated["activation_reason"]),
                str(activated["approved_at"]),
                str(activated["activated_at"]),
                str(approval_event["event_id"]),
                event_id,
                str(activated["activated_at"]),
            )
            snapshot_hash = research_authorization_snapshot_hash(*snapshot_authority)
            snapshot_id = research_authorization_snapshot_id(snapshot_hash)
            conn.execute(
                """INSERT INTO research_track_authorization_snapshots(
                     snapshot_id,snapshot_hash,policy_id,policy_version,definition_hash,
                     release_content_hash,evaluation_id,promotion_review_id,
                     policy_authorization_epoch,approved_by_operator,approval_reason,
                     activation_reason,approved_at,activated_at,approval_event_id,
                     activation_event_id,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (snapshot_id, snapshot_hash, *snapshot_authority),
            )
            record_research_authority(
                self.db_path, conn, policy_event_id=event_id
            )
        return {
            "policy_id": policy_id,
            "version": version,
            "status": "active",
            "family_opt_in_required": True,
            "candidate_delivery_enabled": False,
            "release_content_hash": release_content_hash,
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
        now = utcnow()
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM research_track_policies WHERE policy_id=? AND version=?",
                (policy_id, version),
            ).fetchone()
            if not row or row["status"] != "active":
                raise ValueError("only an active research policy can be disabled")
            enabled_controls = conn.execute(
                """SELECT child_id,family_authorization_epoch FROM research_track_controls
                   WHERE policy_id=? AND policy_version=? AND enabled=1""",
                (policy_id, version),
            ).fetchall()
            conn.execute(
                """UPDATE research_track_policies
                   SET status='disabled',authorization_epoch=authorization_epoch+1,updated_at=?
                   WHERE policy_id=? AND version=?""",
                (now, policy_id, version),
            )
            control_reason = f"Research policy disabled: {reason}"
            conn.execute(
                """UPDATE research_track_controls
                   SET enabled=0,policy_id=NULL,policy_version=NULL,definition_hash=NULL,
                       release_content_hash=NULL,
                       family_authorization_epoch=family_authorization_epoch+1,
                       updated_by='operator',reason=?,updated_at=?
                   WHERE policy_id=? AND policy_version=? AND enabled=1""",
                (control_reason, now, policy_id, version),
            )
            for control in enabled_controls:
                epoch = int(control["family_authorization_epoch"]) + 1
                control_event_id = research_control_event_id(
                    control["child_id"],
                    False,
                    None,
                    None,
                    None,
                    None,
                    epoch,
                    "operator",
                    control_reason,
                    now,
                )
                conn.execute(
                    """INSERT INTO research_track_control_events(
                         event_id,child_id,enabled,policy_id,policy_version,definition_hash,
                         release_content_hash,family_authorization_epoch,actor_type,reason,created_at
                       ) VALUES(?,?,0,NULL,NULL,NULL,NULL,?,'operator',?,?)""",
                    (control_event_id, control["child_id"], epoch, control_reason, now),
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
            record_research_authority(
                self.db_path,
                conn,
                policy_event_id=event_id,
                control_event_ids=tuple(
                    research_control_event_id(
                        control["child_id"],
                        False,
                        None,
                        None,
                        None,
                        None,
                        int(control["family_authorization_epoch"]) + 1,
                        "operator",
                        control_reason,
                        now,
                    )
                    for control in enabled_controls
                ),
            )
        return {"policy_id": policy_id, "version": version, "status": "disabled", "event_id": event_id}

    def _authorization_from_conn(
        self, conn: Any, child_id: str, release_content_hash: str
    ) -> dict[str, Any]:
        row = conn.execute(
            """SELECT c.*,p.authorization_epoch AS policy_authorization_epoch,
                      p.evaluation_id,p.promotion_review_id,
                      s.snapshot_id AS authorization_snapshot_id
               FROM research_track_controls c JOIN research_track_policies p
                 ON p.policy_id=c.policy_id AND p.version=c.policy_version
               JOIN research_track_authorization_snapshots s
                 ON s.policy_id=p.policy_id AND s.policy_version=p.version
                AND s.definition_hash=p.definition_hash
                AND s.release_content_hash=p.release_content_hash
                AND s.evaluation_id=p.evaluation_id
                AND s.promotion_review_id=p.promotion_review_id
                AND s.policy_authorization_epoch=p.authorization_epoch
               JOIN learner_model_controls l ON l.child_id=c.child_id
               WHERE c.child_id=? AND c.enabled=1 AND l.collection_enabled=1
                 AND p.status='active' AND p.definition_hash=c.definition_hash
                 AND p.release_content_hash=c.release_content_hash
                 AND p.release_content_hash=?
                 AND p.evaluation_id IS NOT NULL AND p.promotion_review_id IS NOT NULL""",
            (child_id, release_content_hash),
        ).fetchone()
        if not row:
            raise ResearchTrackAuthorizationChanged(
                "research track requires current family opt-in and an active reviewed policy"
            )
        policy = conn.execute(
            "SELECT * FROM research_track_policies WHERE policy_id=? AND version=?",
            (row["policy_id"], row["policy_version"]),
        ).fetchone()
        try:
            self._trusted_promoted_policy(conn, policy, allowed_statuses={"active"})
        except (TypeError, ValueError) as exc:
            raise ResearchTrackAuthorizationChanged(
                "active research policy failed canonical promotion validation"
            ) from exc
        if not self._control_lineage_is_canonical(conn, row):
            raise ResearchTrackAuthorizationChanged(
                "family research opt-in lacks an exact parent control lineage"
            )
        return dict(row)

    def set_family_opt_in(
        self,
        *,
        child_id: str,
        enabled: bool,
        reason: str,
        actor_type: str = "parent",
    ) -> dict[str, Any]:
        child_id = _identifier(child_id, "child ID")
        reason = _reason(reason)
        if actor_type not in {"parent", "operator"}:
            raise ValueError("research control actor must be parent or operator")
        if enabled and actor_type != "parent":
            raise ValueError("only a family parent action can opt a child into the research track")
        now = utcnow()
        release_content_hash = self._current_release_hash() if enabled else None
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            child = conn.execute("SELECT 1 FROM children WHERE id=?", (child_id,)).fetchone()
            if not child:
                raise ValueError("unknown child")
            # The family control row is lazily materialized elsewhere too. Bind
            # the research opt-in to the same explicit, current collection state.
            conn.execute(
                """INSERT OR IGNORE INTO learner_model_controls(
                     child_id,collection_enabled,default_retention_class,updated_by,created_at,updated_at
                   ) VALUES(?,1,'learning_record',?,?,?)""",
                (child_id, actor_type, now, now),
            )
            existing = conn.execute(
                "SELECT * FROM research_track_controls WHERE child_id=?", (child_id,)
            ).fetchone()
            policy = conn.execute(
                "SELECT * FROM research_track_policies WHERE status='active'"
                + (" AND release_content_hash=?" if enabled else ""),
                (release_content_hash,) if enabled else (),
            ).fetchone()
            if enabled and not policy:
                raise ValueError("family opt-in requires an active separately reviewed research policy")
            if enabled:
                try:
                    self._trusted_promoted_policy(conn, policy, allowed_statuses={"active"})
                except (ResearchTrackAuthorizationChanged, TypeError, ValueError) as exc:
                    raise ValueError(
                        "family opt-in requires a canonically validated active research policy"
                    ) from exc
            if enabled and existing and bool(existing["enabled"]) and (
                existing["policy_id"],
                existing["policy_version"],
                existing["definition_hash"],
                existing["release_content_hash"],
            ) == (
                policy["policy_id"],
                policy["version"],
                policy["definition_hash"],
                policy["release_content_hash"],
            ):
                if not self._control_lineage_is_canonical(conn, existing):
                    raise ValueError("existing family research opt-in failed lineage validation")
                return dict(existing)
            if not enabled and (not existing or not bool(existing["enabled"])):
                return {
                    "child_id": child_id,
                    "enabled": False,
                    "family_authorization_epoch": int(existing["family_authorization_epoch"]) if existing else 0,
                }
            external_epoch, external_floor = family_authorization_high_water(
                self.db_path, child_id
            )
            if existing is None and external_epoch and external_floor != external_epoch:
                raise ValueError("family research control is missing without an erasure boundary")
            epoch = max(
                int(existing["family_authorization_epoch"]) if existing else 0,
                external_epoch,
            ) + 1
            binding = (
                (
                    str(policy["policy_id"]),
                    int(policy["version"]),
                    str(policy["definition_hash"]),
                    str(policy["release_content_hash"]),
                )
                if enabled
                else (None, None, None, None)
            )
            conn.execute(
                """INSERT INTO research_track_controls(
                     child_id,enabled,policy_id,policy_version,definition_hash,release_content_hash,
                     family_authorization_epoch,
                     updated_by,reason,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(child_id) DO UPDATE SET enabled=excluded.enabled,policy_id=excluded.policy_id,
                     policy_version=excluded.policy_version,definition_hash=excluded.definition_hash,
                     release_content_hash=excluded.release_content_hash,
                     family_authorization_epoch=excluded.family_authorization_epoch,
                     updated_by=excluded.updated_by,reason=excluded.reason,updated_at=excluded.updated_at""",
                (
                    child_id,
                    int(enabled),
                    binding[0],
                    binding[1],
                    binding[2],
                    binding[3],
                    epoch,
                    actor_type,
                    reason,
                    now,
                    now,
                ),
            )
            control_event_id = research_control_event_id(
                child_id,
                enabled,
                binding[0],
                binding[1],
                binding[2],
                binding[3],
                epoch,
                actor_type,
                reason,
                now,
            )
            conn.execute(
                """INSERT INTO research_track_control_events(
                     event_id,child_id,enabled,policy_id,policy_version,definition_hash,release_content_hash,
                     family_authorization_epoch,actor_type,reason,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    control_event_id,
                    child_id,
                    int(enabled),
                    binding[0],
                    binding[1],
                    binding[2],
                    binding[3],
                    epoch,
                    actor_type,
                    reason,
                    now,
                ),
            )
            record_research_authority(
                self.db_path, conn, control_event_ids=(control_event_id,)
            )
        return {
            "child_id": child_id,
            "enabled": enabled,
            "policy_id": binding[0],
            "policy_version": binding[1],
            "definition_hash": binding[2],
            "release_content_hash": binding[3],
            "family_authorization_epoch": epoch,
            "event_id": control_event_id,
        }

    def compute_uncertainty_trace(
        self,
        *,
        child_id: str,
        concept_id: str,
        concept_version: int = 1,
    ) -> dict[str, Any]:
        child_id = _identifier(child_id, "child ID")
        concept_id = _identifier(concept_id, "concept ID")
        now = utcnow()
        release_content_hash = self._current_release_hash()
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            authorization = self._authorization_from_conn(
                conn, child_id, release_content_hash
            )
            recompute_child_projections_in_transaction(conn, child_id)
            authorization = self._authorization_from_conn(
                conn, child_id, release_content_hash
            )
            projection = conn.execute(
                """SELECT * FROM learner_state_projections
                   WHERE child_id=? AND concept_id=? AND concept_version=?""",
                (child_id, concept_id, concept_version),
            ).fetchone()
            if not projection:
                raise ValueError("uncertainty tracing requires an existing deterministic learner projection")
            evidence = conn.execute(
                """SELECT pe.evidence_id FROM learner_projection_evidence pe
                   WHERE pe.child_id=? AND pe.concept_id=? AND pe.concept_version=?
                   ORDER BY pe.evidence_id""",
                (child_id, concept_id, concept_version),
            ).fetchall()
            effective = conn.execute(
                """SELECT episode_id,result,assistance_level
                   FROM research_effective_knowledge_evidence
                   WHERE child_id=? AND concept_id=? AND concept_version=?
                   ORDER BY evidence_id""",
                (child_id, concept_id, concept_version),
            ).fetchall()
            episode_count = len({str(row["episode_id"]) for row in effective})
            if episode_count < 3:
                raise ValueError(
                    "uncertainty tracing requires at least three distinct eligible learning episodes"
                )
            history = history_from_evidence_rows([dict(row) for row in effective])
            posterior = posterior_from_history(history)
            evidence_ids = [str(row["evidence_id"]) for row in evidence]
            evidence_ids_json = jdump(evidence_ids)
            evidence_digest = sha256(evidence_ids_json.encode()).hexdigest()
            rationale = (
                f"Advisory estimate from {episode_count} distinct eligible episodes linked to deterministic "
                f"projection {projection['rule_id']}@{projection['rule_version']}; it cannot change state or deliver."
            )
            authority = [
                child_id,
                concept_id,
                concept_version,
                "knowledge_performance",
                str(projection["input_digest"]),
                str(projection["rule_id"]),
                str(projection["rule_version"]),
                str(authorization["policy_id"]),
                int(authorization["policy_version"]),
                str(authorization["definition_hash"]),
                release_content_hash,
                str(authorization["evaluation_id"]),
                str(authorization["promotion_review_id"]),
                str(authorization["authorization_snapshot_id"]),
                int(authorization["family_authorization_epoch"]),
                int(authorization["policy_authorization_epoch"]),
                evidence_digest,
                jdump(history),
                posterior["alpha"],
                posterior["beta"],
                posterior["posterior_mean"],
                posterior["variance"],
                posterior["interval_90_lower"],
                posterior["interval_90_upper"],
                posterior["uncertainty"],
                TRACE_METHOD_VERSION,
                1,
                rationale,
            ]
            trace_hash = authority_hash(*authority)
            trace_id = "rtrace_" + sha256(trace_hash.encode()).hexdigest()[:24]
            conn.execute(
                """INSERT INTO research_uncertainty_traces(
                     trace_id,trace_hash,child_id,concept_id,concept_version,construct,
                     projection_input_digest,projection_rule_id,projection_rule_version,
                     policy_id,policy_version,definition_hash,release_content_hash,
                     evaluation_id,promotion_review_id,authorization_snapshot_id,
                     family_authorization_epoch,policy_authorization_epoch,evidence_ids_json,evidence_digest,
                     history_json,alpha,beta,posterior_mean,variance,interval_90_lower,interval_90_upper,
                     uncertainty,method_version,advisory_only,rationale,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(trace_id) DO NOTHING""",
                (
                    trace_id,
                    trace_hash,
                    *authority[:16],
                    evidence_ids_json,
                    *authority[16:],
                    now,
                ),
            )
            stored = conn.execute(
                "SELECT * FROM research_uncertainty_traces WHERE trace_id=?", (trace_id,)
            ).fetchone()
            if not stored or str(stored["trace_hash"]) != trace_hash:
                raise ValueError("conflicting research uncertainty trace")
            if self._current_release_hash() != release_content_hash:
                raise ResearchTrackAuthorizationChanged(
                    "research release changed while the uncertainty trace was being stored"
                )
        return {
            "trace_id": trace_id,
            "trace_hash": trace_hash,
            "child_id": child_id,
            "concept": {"id": concept_id, "version": concept_version},
            "deterministic_projection": {
                "state": str(projection["state"]),
                "input_digest": str(projection["input_digest"]),
                "rule_id": str(projection["rule_id"]),
                "rule_version": str(projection["rule_version"]),
            },
            "evidence_ids": evidence_ids,
            "history": history,
            "posterior": posterior,
            "advisory_only": True,
            "release_content_hash": release_content_hash,
            "may_authorize_state_or_delivery": False,
            "rationale": rationale,
        }

    def create_proactivity_candidate(
        self,
        *,
        child_id: str,
        trace_id: str,
    ) -> dict[str, Any] | None:
        child_id = _identifier(child_id, "child ID")
        trace_id = _identifier(trace_id, "trace ID")
        now = utcnow()
        release_content_hash = self._current_release_hash()
        with research_connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            authorization = self._authorization_from_conn(
                conn, child_id, release_content_hash
            )
            trace = conn.execute(
                """SELECT t.*,p.state,c.label,i.phase,i.input_digest AS interest_input_digest,
                          i.rule_id AS interest_rule_id,i.rule_version AS interest_rule_version
                   FROM research_uncertainty_traces t
                   JOIN learner_state_projections p
                     ON p.child_id=t.child_id AND p.concept_id=t.concept_id
                    AND p.concept_version=t.concept_version
                    AND p.input_digest=t.projection_input_digest
                   JOIN concepts c ON c.concept_id=t.concept_id AND c.version=t.concept_version
                   LEFT JOIN interest_projections i
                     ON i.child_id=t.child_id AND i.concept_id=t.concept_id
                    AND i.concept_version=t.concept_version
                   WHERE t.trace_id=? AND t.child_id=?""",
                (trace_id, child_id),
            ).fetchone()
            if not trace:
                raise ValueError("unknown or stale research uncertainty trace")
            if any(
                (
                    str(trace["policy_id"]) != str(authorization["policy_id"]),
                    int(trace["policy_version"]) != int(authorization["policy_version"]),
                    str(trace["definition_hash"]) != str(authorization["definition_hash"]),
                    str(trace["release_content_hash"]) != release_content_hash,
                    str(trace["evaluation_id"]) != str(authorization["evaluation_id"]),
                    str(trace["promotion_review_id"])
                    != str(authorization["promotion_review_id"]),
                    int(trace["family_authorization_epoch"])
                    != int(authorization["family_authorization_epoch"]),
                    int(trace["policy_authorization_epoch"])
                    != int(authorization["policy_authorization_epoch"]),
                )
            ):
                raise ResearchTrackAuthorizationChanged("research trace belongs to an earlier authorization epoch")
            if str(trace["phase"] or "not_observed") not in {
                "maintained_situational",
                "emerging_recurrent",
                "sustained_parent_confirmed",
            }:
                return None
            title = f"Possible revisit: {str(trace['label'])[:120]}"
            rationale = (
                f"Candidate only: the current interest projection is {trace['phase']} and the advisory uncertainty "
                f"interval is {float(trace['uncertainty']):.2f} wide. A parent must inspect any future delivery design."
            )
            payload = {
                "concept": {"id": str(trace["concept_id"]), "version": int(trace["concept_version"])},
                "trace_id": trace_id,
                "evidence_ids": jload(trace["evidence_ids_json"], []),
                "deterministic_state": str(trace["state"]),
                "posterior_mean": float(trace["posterior_mean"]),
                "uncertainty": float(trace["uncertainty"]),
                "interest_projection": {
                    "input_digest": str(trace["interest_input_digest"]),
                    "rule_id": str(trace["interest_rule_id"]),
                    "rule_version": str(trace["interest_rule_version"]),
                    "phase": str(trace["phase"]),
                },
                "delivery": "disabled",
            }
            payload_json = jdump(payload)
            candidate_authority = [
                child_id,
                trace_id,
                str(trace["concept_id"]),
                int(trace["concept_version"]),
                str(trace["interest_input_digest"]),
                str(trace["interest_rule_id"]),
                str(trace["interest_rule_version"]),
                str(trace["phase"]),
                str(authorization["policy_id"]),
                int(authorization["policy_version"]),
                str(authorization["definition_hash"]),
                release_content_hash,
                int(authorization["family_authorization_epoch"]),
                int(authorization["policy_authorization_epoch"]),
                title,
                rationale,
                payload_json,
                "candidate_only",
            ]
            candidate_hash = authority_hash(*candidate_authority)
            candidate_id = "rtcand_" + sha256(candidate_hash.encode()).hexdigest()[:24]
            conn.execute(
                """INSERT INTO research_proactivity_candidates(
                     candidate_id,candidate_hash,child_id,trace_id,concept_id,concept_version,
                     interest_input_digest,interest_rule_id,interest_rule_version,interest_phase,
                     policy_id,policy_version,definition_hash,release_content_hash,
                     family_authorization_epoch,
                     policy_authorization_epoch,title,rationale,payload_json,status,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(candidate_id) DO NOTHING""",
                (candidate_id, candidate_hash, *candidate_authority, now),
            )
            stored = conn.execute(
                "SELECT candidate_hash FROM research_proactivity_candidates WHERE candidate_id=?",
                (candidate_id,),
            ).fetchone()
            if not stored or str(stored["candidate_hash"]) != candidate_hash:
                raise ValueError("conflicting research proactivity candidate")
            if self._current_release_hash() != release_content_hash:
                raise ResearchTrackAuthorizationChanged(
                    "research release changed while the candidate was being stored"
                )
            delivery_counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("opportunities", "delivery_outbox", "slack_file_outbox", "artifact_file_outbox")
            }
        return {
            "candidate_id": candidate_id,
            "candidate_hash": candidate_hash,
            "child_id": child_id,
            "title": title,
            "rationale": rationale,
            "payload": payload,
            "status": "candidate_only",
            "delivery_enabled": False,
            "release_content_hash": release_content_hash,
            "delivery_table_counts": delivery_counts,
        }

    def list_candidates(self, child_id: str, *, include_invalidated: bool = False) -> list[dict[str, Any]]:
        child_id = _identifier(child_id, "child ID")
        clause = "" if include_invalidated else " AND status='candidate_only'"
        with research_connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM research_proactivity_candidates WHERE child_id=?" + clause + " ORDER BY created_at DESC",
                (child_id,),
            ).fetchall()
            if not include_invalidated:
                try:
                    release_content_hash = self._current_release_hash()
                    authorization = self._authorization_from_conn(
                        conn, child_id, release_content_hash
                    )
                except ResearchTrackAuthorizationChanged:
                    return []
                rows = [
                    row
                    for row in rows
                    if int(row["family_authorization_epoch"])
                    == int(authorization["family_authorization_epoch"])
                    and int(row["policy_authorization_epoch"])
                    == int(authorization["policy_authorization_epoch"])
                    and str(row["release_content_hash"]) == release_content_hash
                ]
        return [
            {
                **dict(row),
                "payload": jload(row["payload_json"], {}),
                "delivery_enabled": False,
            }
            for row in rows
        ]

    def status(self, child_id: str | None = None) -> dict[str, Any]:
        release_content_hash = self._current_release_hash()
        with research_connect(self.db_path) as conn:
            policies = [
                dict(row)
                for row in conn.execute(
                    """SELECT policy_id,version,status,definition_hash,release_content_hash,
                              authorization_epoch,
                              evaluation_id,promotion_review_id,approved_by_operator,activated_at
                       FROM research_track_policies ORDER BY policy_id,version"""
                ).fetchall()
            ]
            counts = {
                "calibration_cases": conn.execute("SELECT COUNT(*) FROM research_calibration_cases").fetchone()[0],
                "real_calibration_cases": conn.execute(
                    """SELECT COUNT(*) FROM research_calibration_cases c
                       JOIN research_calibration_corpora corpus ON corpus.corpus_id=c.corpus_id
                       WHERE c.source_kind='approved_deidentified' AND corpus.status='approved'"""
                ).fetchone()[0],
                "approved_calibration_corpora": conn.execute(
                    "SELECT COUNT(*) FROM research_calibration_corpora WHERE status='approved'"
                ).fetchone()[0],
                "uncertainty_traces": conn.execute(
                    "SELECT COUNT(*) FROM research_uncertainty_traces"
                ).fetchone()[0],
                "candidate_only": conn.execute(
                    "SELECT COUNT(*) FROM research_proactivity_candidates WHERE status='candidate_only'"
                ).fetchone()[0],
            }
            control = None
            if child_id is not None:
                child_id = _identifier(child_id, "child ID")
                row = conn.execute(
                    "SELECT * FROM research_track_controls WHERE child_id=?", (child_id,)
                ).fetchone()
                control = dict(row) if row else None
        return {
            "policies": policies,
            "counts": counts,
            "child_control": control,
            "trace_role": "advisory_only",
            "proactivity_output": "inspectable_candidate_only",
            "delivery_enabled": False,
            "auto_promote": False,
            "current_release_content_hash": release_content_hash,
            "active_release_current": all(
                row["status"] != "active"
                or str(row["release_content_hash"]) == release_content_hash
                for row in policies
            ),
            "private_text_in_calibration": False,
        }
