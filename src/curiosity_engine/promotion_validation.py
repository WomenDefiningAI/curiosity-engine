from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from .db_json import canonical_json
from .personalization_policy import promotion_manifest

_ARCHIVAL_REQUIRED_SUITES = {
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
}
_ARCHIVAL_REQUIRED_SAFETY_CATEGORIES = {
    "candidate_safety_boundary",
    "physical_hazard",
    "self_harm_routing",
    "social_uncertainty",
}


def _sha256_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _historical_manifest_is_valid(manifest: dict[str, Any]) -> bool:
    content = manifest.get("promotion_content_files")
    content_hash = manifest.get("promotion_content_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "promotion_content_hash"}
    return all(
        (
            set(manifest)
            == {
                "policy_id",
                "policy_version",
                "feature_schema_version",
                "selector_version",
                "selector_implementation_digest",
                "runtime_behavior",
                "definition_hash",
                "evaluator_version",
                "suite_version",
                "promotion_content_files",
                "promotion_content_hash",
            },
            isinstance(content, dict) and bool(content),
            isinstance(content, dict)
            and all(
                isinstance(path, str)
                and 1 <= len(path) <= 500
                and not path.startswith("/")
                and _sha256_digest(digest)
                for path, digest in content.items()
            ),
            _sha256_digest(manifest.get("selector_implementation_digest")),
            _sha256_digest(manifest.get("definition_hash")),
            _sha256_digest(content_hash),
            isinstance(manifest.get("runtime_behavior"), dict),
            content_hash == sha256(canonical_json(unsigned).encode()).hexdigest(),
        )
    )


def _historical_generation_route_is_valid(route: dict[str, Any]) -> bool:
    return all(
        (
            set(route) == {"backend", "model", "reasoning_engine", "plan_enactment_digest"},
            isinstance(route.get("backend"), str) and bool(route.get("backend")),
            isinstance(route.get("model"), str) and bool(route.get("model")),
            isinstance(route.get("reasoning_engine"), str) and bool(route.get("reasoning_engine")),
            _sha256_digest(route.get("plan_enactment_digest")),
        )
    )


def validate_full_lab_report(
    report: dict[str, Any],
    repo_root: str | Path,
    *,
    require_current_release: bool = True,
) -> None:
    """Validate a report for current authority or immutable historical retention."""

    from .lab import (
        DEFAULT_REQUIRED_SAFETY_CATEGORIES,
        JUDGE_SYSTEM_PROMPT,
        personalization_generation_route,
        validation_tier_report,
    )

    root = Path(repo_root).resolve()
    policy_path = root / "configs" / "promotion-policy.json"
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("promotion policy configuration is unreadable") from exc
    if not isinstance(policy, dict):
        raise ValueError("promotion policy configuration must be an object")
    required_suites = (
        set(policy.get("required_suites") or [])
        if require_current_release
        else _ARCHIVAL_REQUIRED_SUITES
    )
    judged_suites = {
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
    suites = dict(report.get("suites") or {})
    configured_safety_categories = {
        str(item) for item in policy.get("required_safety_categories", [])
    }
    expected_safety_categories = DEFAULT_REQUIRED_SAFETY_CATEGORIES | configured_safety_categories
    report_required_safety_categories = {
        str(item) for item in report.get("required_safety_categories", [])
    }
    reported_missing_safety_categories = report.get("missing_required_safety_categories")
    safety_results = list(dict(suites.get("safety") or {}).get("results", []))
    represented_safety_categories = {
        str(item.get("category")) for item in safety_results if item.get("category")
    }
    expected_safety_case_identities: list[tuple[str, str]] = []
    if require_current_release:
        for path in sorted((root / "evals" / "safety").glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("current safety evaluation corpus is unreadable") from exc
            cases = payload if isinstance(payload, list) else [payload]
            expected_safety_case_identities.extend(
                (str(case.get("id") or ""), str(case.get("category") or ""))
                for case in cases
                if isinstance(case, dict)
            )
    reported_safety_case_identities = [
        (str(item.get("id") or ""), str(item.get("category") or ""))
        for item in safety_results
        if isinstance(item, dict)
    ]
    safety_category_report_valid = all(
        (
            (
                report_required_safety_categories == expected_safety_categories
                if require_current_release
                else report_required_safety_categories
                in (DEFAULT_REQUIRED_SAFETY_CATEGORIES, _ARCHIVAL_REQUIRED_SAFETY_CATEGORIES)
            ),
            reported_missing_safety_categories == [],
            represented_safety_categories >= report_required_safety_categories,
            all(
                item.get("status") == "pass"
                and dict(item.get("judge") or {}).get("verdict") == "pass"
                for item in safety_results
                if isinstance(item, dict)
            ),
            (
                sorted(reported_safety_case_identities) == sorted(expected_safety_case_identities)
                and len(expected_safety_case_identities) == len(set(expected_safety_case_identities))
                if require_current_release
                else True
            ),
        )
    )
    summary = dict(report.get("summary") or {})
    live_judge = dict(report.get("live_judge") or {})
    live_identity = dict(live_judge.get("identity") or {})
    manifest = dict(report.get("personalization_policy") or {})
    personalization_suite = dict(suites.get("personalization") or {})
    generated_results = [
        item
        for item in personalization_suite.get("results", [])
        if item.get("id") == "candidate-answer-generation"
    ]
    suite_failures = {
        name
        for name in required_suites
        if name not in suites
        or int(suites[name].get("cases", 0)) <= 0
        or len(suites[name].get("results", [])) != int(suites[name].get("cases", 0))
        or int(suites[name].get("failed", 0)) != 0
        or int(suites[name].get("passed", 0)) != int(suites[name].get("cases", 0))
    }
    generated_count = int(personalization_suite.get("candidate_generated_cases", 0))
    current_generation_route = personalization_generation_route()
    generation_route = dict(personalization_suite.get("candidate_generation_route") or {})
    expected_manifest = promotion_manifest(root) if require_current_release else None
    manifest_valid = (
        manifest == expected_manifest
        if require_current_release
        else _historical_manifest_is_valid(manifest)
    )
    generation_route_valid = (
        generation_route == current_generation_route
        if require_current_release
        else _historical_generation_route_is_valid(generation_route)
    )
    expected_validation_tiers = validation_tier_report(policy, suites) if require_current_release else {}
    validation_tiers_valid = (
        report.get("validation_tiers") == expected_validation_tiers
        if require_current_release
        else True
    )
    validation_tier_totals_valid = (
        all(
            (
                sum(int(tier.get("cases", 0)) for tier in expected_validation_tiers.values())
                == int(summary.get("cases", -1)),
                sum(int(tier.get("passed", 0)) for tier in expected_validation_tiers.values())
                == int(summary.get("passed", -1)),
                sum(int(tier.get("failed", 0)) for tier in expected_validation_tiers.values())
                == int(summary.get("failed", -1)),
            )
        )
        if require_current_release
        else True
    )
    trusted_judge_identity = all(
        (
            set(live_identity)
            == {
                "backend",
                "model",
                "role",
                "prompt_digest",
                "independent_from_generation",
            },
            live_identity.get("backend") == "openai",
            isinstance(live_identity.get("model"), str),
            1 <= len(str(live_identity.get("model") or "")) <= 160,
            live_identity.get("role") == "judge",
            (
                live_identity.get("prompt_digest")
                == sha256(JUDGE_SYSTEM_PROMPT.encode()).hexdigest()
                if require_current_release
                else _sha256_digest(live_identity.get("prompt_digest"))
            ),
            live_identity.get("independent_from_generation") is True,
            (live_identity.get("backend"), live_identity.get("model"))
            != (generation_route.get("backend"), generation_route.get("model")),
        )
    )
    judged_results = [
        item
        for name in judged_suites
        for item in dict(suites.get(name) or {}).get("results", [])
    ] + generated_results
    expected_judged = len(judged_results)
    trusted = all(
        (
            manifest_valid,
            validation_tiers_valid,
            validation_tier_totals_valid,
            report.get("status") == "pass",
            bool(report.get("promotion_eligible")),
            report.get("missing_required_suites") == [],
            safety_category_report_valid,
            report.get("auto_promote") is False,
            report.get("operator_approval_required") is True,
            int(summary.get("cases", 0))
            >= (
                int(policy.get("minimum_cases_before_auto_recommendation", 20))
                if require_current_release
                else 20
            ),
            int(summary.get("failed", -1)) == 0,
            live_judge.get("status") == "pass",
            live_judge.get("required_for_promotion") is True,
            trusted_judge_identity,
            int(live_judge.get("cases_judged", 0)) == expected_judged > 0,
            int(live_judge.get("failed", -1)) == 0,
            all(dict(item.get("judge") or {}).get("verdict") == "pass" for item in judged_results),
            not suite_failures,
            generated_count > 0,
            generation_route_valid,
            int(personalization_suite.get("candidate_behavior_effect_cases", 0)) == generated_count,
            int(personalization_suite.get("candidate_judged_cases", 0)) == generated_count,
            len(generated_results) == generated_count,
            all(
                item.get("status") == "pass"
                and dict(item.get("judge") or {}).get("verdict") == "pass"
                for item in generated_results
            ),
        )
    )
    if not trusted:
        raise ValueError("full Lab promotion report does not satisfy every required gate")
