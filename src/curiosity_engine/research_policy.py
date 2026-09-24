from __future__ import annotations

import inspect
import json
from hashlib import sha256
from math import sqrt
from typing import Any

RESEARCH_POLICY_ID = "uncertainty-context-k2"
RESEARCH_POLICY_VERSION = 1
RESEARCH_FEATURE_SCHEMA_VERSION = "phase6-v1"
TRACE_METHOD_VERSION = "beta-evidence-companion-v1"
RESEARCH_EVALUATOR_VERSION = "heldout-calibration-v1"

MIN_REAL_CASES = 40
MIN_TRAIN_CASES = 20
MIN_HELDOUT_CASES = 20
MIN_DISTINCT_TASKS = 8
MIN_DISTINCT_DOMAINS = 3
MIN_HELDOUT_OUTCOME_CASES = 5
MAX_BRIER_SCORE = 0.12
MAX_EXPECTED_CALIBRATION_ERROR = 0.15

HISTORY_FIELDS = (
    "independent_successes",
    "independent_failures",
    "partial_results",
    "supported_successes",
    "supported_failures",
)
DOMAINS = ("general", "science", "math", "literacy", "writing", "source_sense")

RESEARCH_POLICY_DEFINITION = {
    "policy_id": RESEARCH_POLICY_ID,
    "version": RESEARCH_POLICY_VERSION,
    "feature_schema_version": RESEARCH_FEATURE_SCHEMA_VERSION,
    "trace": {
        "method": TRACE_METHOD_VERSION,
        "prior": {"alpha": 1.0, "beta": 1.0},
        "weights": {
            "independent": 1.0,
            "partial_to_each_side": 0.5,
            "supported": 0.25,
        },
        "role": "advisory_companion_to_evidence_ledger",
        "may_authorize_state_or_delivery": False,
    },
    "calibration_gates": {
        "minimum_real_cases": MIN_REAL_CASES,
        "minimum_train_cases": MIN_TRAIN_CASES,
        "minimum_heldout_cases": MIN_HELDOUT_CASES,
        "minimum_distinct_tasks": MIN_DISTINCT_TASKS,
        "minimum_distinct_domains": MIN_DISTINCT_DOMAINS,
        "minimum_heldout_cases_per_outcome": MIN_HELDOUT_OUTCOME_CASES,
        "maximum_brier_score": MAX_BRIER_SCORE,
        "maximum_expected_calibration_error": MAX_EXPECTED_CALIBRATION_ERROR,
        "split_group_overlap_allowed": False,
        "eligible_source_kind": "approved_deidentified",
        "synthetic_cases_unlock_promotion": False,
    },
    "proactivity": {
        "output": "inspectable_candidate_only",
        "requires_active_reviewed_policy": True,
        "requires_child_specific_family_opt_in": True,
        "requires_current_authorization_epochs": True,
        "delivery_table_access": False,
        "minimum_linked_evidence": 3,
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


RESEARCH_POLICY_DEFINITION_HASH = sha256(canonical_json(RESEARCH_POLICY_DEFINITION).encode()).hexdigest()


def normalize_history(value: dict[str, Any]) -> dict[str, int]:
    if set(value) != set(HISTORY_FIELDS):
        raise ValueError("research history must use the exact categorical count schema")
    normalized: dict[str, int] = {}
    for field in HISTORY_FIELDS:
        count = value[field]
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 10_000:
            raise ValueError("research history counts must be bounded non-negative integers")
        normalized[field] = count
    return normalized


def posterior_from_history(value: dict[str, Any]) -> dict[str, float | str]:
    history = normalize_history(value)
    alpha = (
        1.0
        + history["independent_successes"]
        + 0.5 * history["partial_results"]
        + 0.25 * history["supported_successes"]
    )
    beta = (
        1.0
        + history["independent_failures"]
        + 0.5 * history["partial_results"]
        + 0.25 * history["supported_failures"]
    )
    total = alpha + beta
    mean = alpha / total
    variance = (alpha * beta) / ((total * total) * (total + 1.0))
    radius = 1.6448536269514722 * sqrt(variance)
    lower = max(0.0, mean - radius)
    upper = min(1.0, mean + radius)
    return {
        "method": TRACE_METHOD_VERSION,
        "alpha": alpha,
        "beta": beta,
        "posterior_mean": mean,
        "variance": variance,
        "interval_90_lower": lower,
        "interval_90_upper": upper,
        "uncertainty": upper - lower,
    }


def history_from_evidence_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Collapse eligible observations to one conservative signal per episode."""

    by_episode: dict[str, list[tuple[str, str]]] = {}
    valid_results = {"correct", "partial", "incorrect", "abandoned"}
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        result = str(row.get("result") or "")
        assistance = str(row.get("assistance_level") or "")
        if not episode_id or result not in valid_results or not assistance:
            raise ValueError("research evidence rows require episode, result, and assistance")
        by_episode.setdefault(episode_id, []).append((result, assistance))
    history = {field: 0 for field in HISTORY_FIELDS}
    for observations in by_episode.values():
        failures = [item for item in observations if item[0] in {"incorrect", "abandoned"}]
        if failures:
            key = (
                "independent_failures"
                if any(assistance == "none" for _result, assistance in failures)
                else "supported_failures"
            )
        elif any(result == "partial" for result, _assistance in observations):
            key = "partial_results"
        elif any(
            result == "correct" and assistance == "none" for result, assistance in observations
        ):
            key = "independent_successes"
        elif any(result == "correct" for result, _assistance in observations):
            key = "supported_successes"
        else:  # pragma: no cover - valid_results makes this defensive only
            continue
        history[key] += 1
    return normalize_history(history)


def calibration_case_payload(
    *,
    case_id: str,
    source_kind: str,
    license_status: str,
    privacy_approval_id: str | None,
    split: str,
    group_digest: str,
    task_signature_digest: str,
    domain: str,
    history: dict[str, Any],
    observed_success: int,
    source_digest: str,
    corpus_id: str | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "source_kind": source_kind,
        "license_status": license_status,
        "privacy_approval_id": privacy_approval_id,
        "corpus_id": corpus_id,
        "feature_schema_version": RESEARCH_FEATURE_SCHEMA_VERSION,
        "split": split,
        "group_digest": group_digest,
        "task_signature_digest": task_signature_digest,
        "domain": domain,
        "history": normalize_history(history),
        "observed_success": observed_success,
        "source_digest": source_digest,
    }


def calibration_case_hash(**kwargs: Any) -> str:
    kwargs = dict(kwargs)
    feature_schema_version = kwargs.pop(
        "feature_schema_version", RESEARCH_FEATURE_SCHEMA_VERSION
    )
    if feature_schema_version != RESEARCH_FEATURE_SCHEMA_VERSION:
        raise ValueError("research calibration feature schema is not approved")
    return sha256(canonical_json(calibration_case_payload(**kwargs)).encode()).hexdigest()


def calibration_corpus_manifest(corpus_id: str, case_hashes: list[str]) -> dict[str, Any]:
    return {
        "corpus_id": corpus_id,
        "feature_schema_version": RESEARCH_FEATURE_SCHEMA_VERSION,
        "case_hashes": sorted(case_hashes),
    }


def calibration_corpus_manifest_hash(corpus_id: str, case_hashes: list[str]) -> str:
    return sha256(canonical_json(calibration_corpus_manifest(corpus_id, case_hashes)).encode()).hexdigest()


def calibration_metrics(cases: list[dict[str, Any]]) -> dict[str, float]:
    if not cases:
        return {"brier_score": 1.0, "expected_calibration_error": 1.0}
    predictions: list[tuple[float, int]] = []
    for case in cases:
        prediction = float(posterior_from_history(dict(case["history"]))["posterior_mean"])
        observed = int(case["observed_success"])
        if observed not in {0, 1}:
            raise ValueError("calibration outcomes must be binary")
        predictions.append((prediction, observed))
    brier = sum((prediction - observed) ** 2 for prediction, observed in predictions) / len(predictions)
    expected_calibration_error = 0.0
    for bucket in range(5):
        lower = bucket / 5
        upper = (bucket + 1) / 5
        members = [
            (prediction, observed)
            for prediction, observed in predictions
            if lower <= prediction < upper or (bucket == 4 and prediction == 1.0)
        ]
        if not members:
            continue
        confidence = sum(item[0] for item in members) / len(members)
        accuracy = sum(item[1] for item in members) / len(members)
        expected_calibration_error += (len(members) / len(predictions)) * abs(confidence - accuracy)
    return {
        "brier_score": brier,
        "expected_calibration_error": expected_calibration_error,
    }


def canonical_calibration_case(row: Any) -> dict[str, Any]:
    """Rebuild one persisted calibration row from its canonical authority fields."""

    history = normalize_history(json.loads(str(row["history_json"])))
    payload = calibration_case_payload(
        case_id=str(row["case_id"]),
        source_kind=str(row["source_kind"]),
        license_status=str(row["license_status"]),
        privacy_approval_id=(
            str(row["privacy_approval_id"]) if row["privacy_approval_id"] is not None else None
        ),
        corpus_id=str(row["corpus_id"]) if row["corpus_id"] is not None else None,
        split=str(row["split"]),
        group_digest=str(row["group_digest"]),
        task_signature_digest=str(row["task_signature_digest"]),
        domain=str(row["domain"]),
        history=history,
        observed_success=int(row["observed_success"]),
        source_digest=str(row["source_digest"]),
    )
    expected = calibration_case_hash(**payload)
    keys = set(row.keys())
    corpus_status = row["corpus_status"] if "corpus_status" in keys else None
    if (
        str(row["feature_schema_version"]) != RESEARCH_FEATURE_SCHEMA_VERSION
        or canonical_json(history) != str(row["history_json"])
        or expected != str(row["case_hash"])
    ):
        raise ValueError("research calibration case failed canonical integrity")
    source_kind = str(row["source_kind"])
    if (
        source_kind == "approved_deidentified"
        and (payload["corpus_id"] != payload["privacy_approval_id"] or corpus_status is None)
    ) or (source_kind != "approved_deidentified" and payload["corpus_id"] is not None):
        raise ValueError("research calibration corpus binding failed canonical integrity")
    return {**payload, "case_hash": expected, "corpus_status": corpus_status}


def canonical_research_evaluation(
    policy: Any,
    case_rows: list[Any],
    approved_corpus_rows: list[Any],
    release_content_hash: str,
) -> dict[str, Any]:
    """Recompute the complete held-out evaluation from persisted, approved inputs."""

    definition = json.loads(str(policy["definition_json"]))
    if (
        str(policy["feature_schema_version"]) != RESEARCH_FEATURE_SCHEMA_VERSION
        or definition != RESEARCH_POLICY_DEFINITION
        or sha256(canonical_json(definition).encode()).hexdigest()
        != str(policy["definition_hash"])
    ):
        raise ValueError("research policy definition failed canonical integrity")
    cases = [canonical_calibration_case(row) for row in case_rows]
    real = [
        case
        for case in cases
        if case["source_kind"] == "approved_deidentified"
        and case["corpus_status"] == "approved"
    ]
    train = [case for case in real if case["split"] == "train"]
    heldout = [case for case in real if case["split"] == "heldout"]
    train_groups = {str(case["group_digest"]) for case in train}
    heldout_groups = {str(case["group_digest"]) for case in heldout}
    overlap = train_groups & heldout_groups
    metrics = calibration_metrics(heldout)
    positive = sum(int(case["observed_success"]) for case in heldout)
    negative = len(heldout) - positive
    distinct_tasks = len({str(case["task_signature_digest"]) for case in real})
    distinct_domains = len({str(case["domain"]) for case in real})
    gates = {
        "minimum_real_cases": len(real) >= MIN_REAL_CASES,
        "minimum_train_cases": len(train) >= MIN_TRAIN_CASES,
        "minimum_heldout_cases": len(heldout) >= MIN_HELDOUT_CASES,
        "task_diversity": distinct_tasks >= MIN_DISTINCT_TASKS,
        "domain_diversity": distinct_domains >= MIN_DISTINCT_DOMAINS,
        "heldout_outcome_balance": min(positive, negative) >= MIN_HELDOUT_OUTCOME_CASES,
        "split_group_independence": not overlap,
        "brier_score": metrics["brier_score"] <= MAX_BRIER_SCORE,
        "expected_calibration_error": (
            metrics["expected_calibration_error"] <= MAX_EXPECTED_CALIBRATION_ERROR
        ),
        "synthetic_cases_do_not_unlock": len(real) >= MIN_REAL_CASES,
    }
    cases_by_corpus: dict[str, list[str]] = {}
    for case in cases:
        if case["corpus_id"] is not None:
            cases_by_corpus.setdefault(str(case["corpus_id"]), []).append(str(case["case_hash"]))
    approved_corpora: list[dict[str, Any]] = []
    approved_corpus_ids: set[str] = set()
    for row in approved_corpus_rows:
        corpus_id = str(row["corpus_id"])
        case_hashes = sorted(cases_by_corpus.get(corpus_id, []))
        manifest = json.loads(str(row["manifest_json"]))
        expected_manifest = calibration_corpus_manifest(corpus_id, case_hashes)
        if not all(
            (
                str(row["feature_schema_version"]) == RESEARCH_FEATURE_SCHEMA_VERSION,
                str(row["status"]) == "approved",
                int(row["case_count"]) == len(case_hashes) > 0,
                manifest == expected_manifest,
                str(row["manifest_json"]) == canonical_json(expected_manifest),
                str(row["manifest_hash"])
                == calibration_corpus_manifest_hash(corpus_id, case_hashes),
                bool(row["approved_by_reviewer"]),
                str(row["approved_by_reviewer"]) != str(row["proposed_by_operator"]),
                bool(row["approval_reason"]),
                bool(row["approved_at"]),
            )
        ):
            raise ValueError("approved research calibration corpus failed canonical integrity")
        approved_corpus_ids.add(corpus_id)
        approved_corpora.append(
            {
                "corpus_id": corpus_id,
                "feature_schema_version": str(row["feature_schema_version"]),
                "manifest": expected_manifest,
                "manifest_hash": str(row["manifest_hash"]),
                "case_count": int(row["case_count"]),
                "status": "approved",
                "proposed_by_operator": str(row["proposed_by_operator"]),
                "approved_by_reviewer": str(row["approved_by_reviewer"]),
                "approval_reason": str(row["approval_reason"]),
                "approved_at": str(row["approved_at"]),
                "created_at": str(row["created_at"]),
            }
        )
    if any(
        case["source_kind"] == "approved_deidentified"
        and str(case["corpus_id"]) not in approved_corpus_ids
        for case in cases
    ):
        raise ValueError("research calibration case lacks an exact approved corpus manifest")
    dataset_digest = sha256(
        canonical_json({"cases": cases, "approved_corpora": approved_corpora}).encode()
    ).hexdigest()
    results = [
        {
            "case_id": str(case["case_id"]),
            "prediction": float(posterior_from_history(dict(case["history"]))["posterior_mean"]),
            "observed_success": int(case["observed_success"]),
        }
        for case in heldout
    ]
    evaluator_digest = sha256(
        canonical_json(
            {
                "policy": RESEARCH_POLICY_DEFINITION_HASH,
                "case_source": inspect.getsource(canonical_calibration_case),
                "posterior_source": inspect.getsource(posterior_from_history),
                "metrics_source": inspect.getsource(calibration_metrics),
                "evaluation_source": inspect.getsource(canonical_research_evaluation),
            }
        ).encode()
    ).hexdigest()
    identity = {
        "policy_id": str(policy["policy_id"]),
        "policy_version": int(policy["version"]),
        "definition_hash": str(policy["definition_hash"]),
        "release_content_hash": release_content_hash,
        "dataset_digest": dataset_digest,
        "results_hash": sha256(canonical_json(results).encode()).hexdigest(),
        "evaluator": RESEARCH_EVALUATOR_VERSION,
        "evaluator_digest": evaluator_digest,
    }
    return {
        **identity,
        "evaluation_id": "rteval_" + sha256(canonical_json(identity).encode()).hexdigest()[:24],
        "case_count": len(cases),
        "real_case_count": len(real),
        "train_count": len(train),
        "heldout_count": len(heldout),
        "distinct_task_count": distinct_tasks,
        "distinct_domain_count": distinct_domains,
        "heldout_positive_count": positive,
        "heldout_negative_count": negative,
        "split_overlap_count": len(overlap),
        "brier_score": metrics["brier_score"],
        "expected_calibration_error": metrics["expected_calibration_error"],
        "gates": gates,
        "status": "pass" if all(gates.values()) else "fail",
    }


def research_evaluation_matches(row: Any, canonical: dict[str, Any]) -> bool:
    """Return whether a persisted evaluation exactly matches canonical recomputation."""

    exact_fields = (
        "evaluation_id",
        "policy_id",
        "policy_version",
        "definition_hash",
        "release_content_hash",
        "dataset_digest",
        "case_count",
        "real_case_count",
        "train_count",
        "heldout_count",
        "distinct_task_count",
        "distinct_domain_count",
        "heldout_positive_count",
        "heldout_negative_count",
        "split_overlap_count",
        "results_hash",
        "evaluator",
        "evaluator_digest",
        "status",
    )
    return all(str(row[key]) == str(canonical[key]) for key in exact_fields) and all(
        (
            float(row["brier_score"]) == canonical["brier_score"],
            float(row["expected_calibration_error"])
            == canonical["expected_calibration_error"],
            json.loads(str(row["gates_json"])) == canonical["gates"],
            str(row["gates_json"]) == canonical_json(canonical["gates"]),
        )
    )


def archival_research_evaluation_is_valid(row: Any, policy: Any) -> bool:
    """Validate immutable evaluation identity without re-authorizing an old release."""

    try:
        gates = json.loads(str(row["gates_json"]))
        expected_gate_names = {
            "minimum_real_cases",
            "minimum_train_cases",
            "minimum_heldout_cases",
            "task_diversity",
            "domain_diversity",
            "heldout_outcome_balance",
            "split_group_independence",
            "brier_score",
            "expected_calibration_error",
            "synthetic_cases_do_not_unlock",
        }
        identity = {
            "policy_id": str(row["policy_id"]),
            "policy_version": int(row["policy_version"]),
            "definition_hash": str(row["definition_hash"]),
            "release_content_hash": str(row["release_content_hash"]),
            "dataset_digest": str(row["dataset_digest"]),
            "results_hash": str(row["results_hash"]),
            "evaluator": str(row["evaluator"]),
            "evaluator_digest": str(row["evaluator_digest"]),
        }
        hashes = (
            identity["definition_hash"],
            identity["release_content_hash"],
            identity["dataset_digest"],
            identity["results_hash"],
            identity["evaluator_digest"],
        )
        status = "pass" if all(gates.values()) else "fail"
        return all(
            (
                policy is not None,
                identity["policy_id"] == str(policy["policy_id"]),
                identity["policy_version"] == int(policy["version"]),
                identity["definition_hash"] == str(policy["definition_hash"]),
                identity["evaluator"] == RESEARCH_EVALUATOR_VERSION,
                all(
                    isinstance(value, str)
                    and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value)
                    for value in hashes
                ),
                str(row["evaluation_id"])
                == "rteval_" + sha256(canonical_json(identity).encode()).hexdigest()[:24],
                set(gates) == expected_gate_names,
                all(isinstance(value, bool) for value in gates.values()),
                str(row["gates_json"]) == canonical_json(gates),
                str(row["status"]) == status,
                int(row["case_count"]) >= int(row["real_case_count"]) >= 0,
                int(row["real_case_count"])
                == int(row["train_count"]) + int(row["heldout_count"]),
                int(row["heldout_count"])
                == int(row["heldout_positive_count"]) + int(row["heldout_negative_count"]),
                int(row["distinct_task_count"]) >= 0,
                int(row["distinct_domain_count"]) >= 0,
                int(row["split_overlap_count"]) >= 0,
                0.0 <= float(row["brier_score"]) <= 1.0,
                0.0 <= float(row["expected_calibration_error"]) <= 1.0,
            )
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
