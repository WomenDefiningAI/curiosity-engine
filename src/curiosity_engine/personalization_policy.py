from __future__ import annotations

import inspect
import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any

from .db_json import canonical_json

POLICY_ID = "balanced-k2"
POLICY_VERSION = 1
FEATURE_SCHEMA_VERSION = "phase5-v2"
SELECTOR_VERSION = "declarative-balanced-k2-v2"
EVALUATOR_VERSION = "deterministic-policy-replay-v2"
PERSONALIZATION_SUITE_VERSION = "phase5-v2"
EVALUATION_MINIMUM_CASES = 6
EVALUATION_MINIMUM_BENEFIT = 0.05

ALLOWED_FEATURES = {
    "question_intent",
    "baseline_strategy_id",
    "domain_overlay_id",
    "prior_feedback_band",
    "recent_challenge_signal",
    "knowledge_band",
    "interest_band",
    "response_mode",
}
FORBIDDEN_REPLAY_KEYS = {
    "child_id",
    "child_name",
    "family_id",
    "parent_id",
    "event_id",
    "message_id",
    "raw_text",
    "text",
    "note",
    "birth_year",
    "grade",
}
CHALLENGE_VALUES = {
    "grade_baseline_uncalibrated",
    "familiar",
    "stretch_with_help",
    "independent_transfer",
}
INTEREST_VALUES = {"neutral", "situational_trigger", "maintenance", "recurring_depth"}
FEATURE_VALUES = {
    "question_intent": {
        "direct_answer",
        "answer_and_wonder",
        "guided_inquiry",
        "compare_connect",
        "practice_with_scaffolds",
        "dialogic_reading",
        "teach_back",
        "retrieval_revisit",
        "make_design_iterate",
        "sensitive_support",
    },
    "baseline_strategy_id": {
        "answer-and-wonder-k2",
        "guided-inquiry-k2",
        "compare-connect-k2",
        "model-practice-fade-k2",
        "dialogic-reading-k2",
        "teach-back-k2",
        "retrieval-revisit-k2",
        "make-design-iterate-k2",
    },
    "domain_overlay_id": {
        "general-learning-k2",
        "early-science-k2",
        "early-math-k2",
        "early-literacy-k2",
        "early-writing-k2",
        "source-sense-k2",
        "sensitive-support-k2",
    },
    "prior_feedback_band": {"none", "one", "multiple"},
    "recent_challenge_signal": {"none", "positive", "too_easy", "too_hard", "mixed"},
    "knowledge_band": {
        "not_observed",
        "exposure_recorded",
        "practicing_with_support",
        "demonstrated_in_context",
        "independently_demonstrated",
        "transfer_observed",
    },
    "interest_band": {
        "not_observed",
        "triggered_situational",
        "maintained_situational",
        "emerging_recurrent",
        "sustained_parent_confirmed",
    },
    "response_mode": {
        "none",
        "multiple",
        "spoken",
        "written",
        "drawn",
        "gesture_or_manipulative",
        "selection",
        "demonstration",
        "symbolic",
    },
}


def _digest(value: Any) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


def assert_feature_vector(features: dict[str, Any]) -> dict[str, str]:
    if set(features) != ALLOWED_FEATURES:
        raise ValueError("personalization features must match the reviewed feature schema")
    normalized = {key: str(value) for key, value in features.items()}
    if any(value not in FEATURE_VALUES[key] for key, value in normalized.items()):
        raise ValueError("personalization features must use reviewed categorical values")
    return normalized


def contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_REPLAY_KEYS & {str(key).casefold() for key in value}) or any(
            contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(contains_forbidden_key(item) for item in value)
    return False


def normalize_replay_outcome(outcome: dict[str, Any]) -> dict[str, float]:
    required = {
        "baseline_utility",
        "candidate_utility",
        "baseline_calibration_error",
        "candidate_calibration_error",
    }
    if set(outcome) != required:
        raise ValueError("replay outcome must match the reviewed numeric schema")
    try:
        normalized = {key: float(value) for key, value in outcome.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("replay outcomes must be numeric") from exc
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in normalized.values()):
        raise ValueError("replay outcome values must be finite and between zero and one")
    return normalized


def replay_case_payload(
    *,
    case_id: str,
    source_kind: str,
    license_status: str,
    privacy_approval_id: str | None,
    feature_schema_version: str,
    feature_vector: dict[str, Any],
    outcome: dict[str, Any],
    source_digest: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "source_kind": source_kind,
        "license_status": license_status,
        "privacy_approval_id": privacy_approval_id,
        "feature_schema_version": feature_schema_version,
        "feature_vector": assert_feature_vector(feature_vector),
        "outcome": normalize_replay_outcome(outcome),
        "source_digest": source_digest,
    }


def feedback_signal_query(concept_count: int) -> str:
    """Return the reviewed concept-first query; recency is bounded per target concept."""

    if not 1 <= concept_count <= 3:
        raise ValueError("personalization requires one to three canonical concepts")
    placeholders = ",".join("(?,?)" for _ in range(concept_count))
    return f"""WITH matched AS (
      SELECT s.signal_id,s.outcome,s.created_at,
             json_extract(concept.value,'$.id') AS concept_id,
             CAST(json_extract(concept.value,'$.version') AS INTEGER) AS concept_version,
             ROW_NUMBER() OVER (
               PARTITION BY json_extract(concept.value,'$.id'),
                            CAST(json_extract(concept.value,'$.version') AS INTEGER)
               ORDER BY s.created_at DESC,s.signal_id DESC
             ) AS concept_rank
      FROM personalization_feedback_signals s
      JOIN pedagogy_plans p ON p.plan_id=s.plan_id AND p.plan_hash=s.plan_hash
      JOIN json_each(p.approved_json,'$.concepts') AS concept
      WHERE s.child_id=?
        AND (
          json_extract(concept.value,'$.id'),
          CAST(json_extract(concept.value,'$.version') AS INTEGER)
        ) IN ({placeholders})
    )
    SELECT signal_id,outcome,concept_id,concept_version FROM matched
    WHERE concept_rank<=8
    ORDER BY created_at DESC,signal_id DESC,concept_id,concept_version"""


def feedback_feature_bands(
    records: list[dict[str, Any]],
    concept_refs: list[dict[str, Any]],
) -> tuple[str, str]:
    """Conservatively combine concept-attributed feedback for a multi-concept plan."""

    targets = list(
        dict.fromkeys(
            (str(item.get("id") or ""), int(item.get("version") or 0))
            for item in concept_refs
        )
    )
    outcomes_by_concept: dict[tuple[str, int], list[str]] = {
        concept: [] for concept in targets
    }
    signal_ids: set[str] = set()
    for record in records:
        concept = (
            str(record.get("concept_id") or ""),
            int(record.get("concept_version") or 0),
        )
        outcome = str(record.get("outcome") or "")
        if concept not in outcomes_by_concept or len(outcomes_by_concept[concept]) >= 8:
            continue
        outcomes_by_concept[concept].append(outcome)
        signal_ids.add(str(record.get("signal_id") or ""))
    count = "none" if not signal_ids else "one" if len(signal_ids) == 1 else "multiple"
    concept_bands: list[str] = []
    for concept in targets:
        outcomes = outcomes_by_concept[concept]
        challenge = [value for value in outcomes if value in {"too_easy", "too_hard"}]
        if len(set(challenge)) > 1:
            concept_bands.append("mixed")
        elif challenge:
            concept_bands.append(challenge[0])
        elif any(value in {"loved", "engaged", "helpful"} for value in outcomes):
            concept_bands.append("positive")
        elif outcomes:
            concept_bands.append("mixed")
        else:
            concept_bands.append("none")
    if "too_hard" in concept_bands:
        recent = "too_hard"
    elif "mixed" in concept_bands:
        recent = "mixed"
    elif concept_bands and all(value == "too_easy" for value in concept_bands):
        recent = "too_easy"
    elif "positive" in concept_bands:
        recent = "positive"
    elif "too_easy" in concept_bands:
        recent = "mixed"
    else:
        recent = "none"
    return count, recent


def projection_feature_bands(
    concept_refs: list[dict[str, Any]],
    knowledge_by_concept: dict[tuple[str, int], str],
    interest_phases: list[str],
) -> tuple[str, str]:
    """Use the least-supported target for challenge and strongest supported interest."""

    knowledge_order = {
        "not_observed": 0,
        "exposure_recorded": 1,
        "practicing_with_support": 2,
        "demonstrated_in_context": 3,
        "independently_demonstrated": 4,
        "transfer_observed": 5,
    }
    interest_order = {
        "not_observed": 0,
        "triggered_situational": 1,
        "maintained_situational": 2,
        "emerging_recurrent": 3,
        "sustained_parent_confirmed": 4,
    }
    targets = list(
        dict.fromkeys(
            (str(item.get("id") or ""), int(item.get("version") or 0))
            for item in concept_refs
        )
    )
    if not targets:
        return "not_observed", "not_observed"
    normalized = {
        concept: (
            "not_observed"
            if str(state) == "not_observed_in_harness"
            else str(state)
        )
        for concept, state in knowledge_by_concept.items()
    }
    knowledge = min(
        (normalized.get(concept, "not_observed") for concept in targets),
        key=lambda item: knowledge_order.get(item, -1),
    )
    interest = max(
        (str(value) for value in interest_phases),
        key=lambda item: interest_order.get(item, -1),
        default="not_observed",
    )
    return knowledge, interest


def response_mode_feature(response_modes: list[str]) -> str:
    return response_modes[0] if len(response_modes) == 1 else "multiple" if response_modes else "none"


def select_from_features(definition: dict[str, Any], features: dict[str, Any]) -> dict[str, str]:
    """Interpret only the complete, hash-bound declarative policy definition."""

    vector = assert_feature_vector(features)
    if str(definition.get("feature_schema_version")) != FEATURE_SCHEMA_VERSION:
        raise ValueError("personalization feature schema is not the reviewed version")
    if set(definition.get("allowed_features") or []) != ALLOWED_FEATURES:
        raise ValueError("personalization policy allowed-feature set is incomplete")
    selector = dict(definition.get("selector") or {})
    if selector.get("version") != SELECTOR_VERSION:
        raise ValueError("personalization selector version is not approved")
    if selector.get("implementation_digest") != SELECTOR_IMPLEMENTATION_DIGEST:
        raise ValueError("personalization selector implementation changed without a new policy")
    if definition.get("runtime_behavior") != RUNTIME_BEHAVIOR_MANIFEST:
        raise ValueError("personalization runtime behavior changed without a new policy")

    recent = vector["recent_challenge_signal"]
    knowledge = vector["knowledge_band"]
    interest = vector["interest_band"]
    challenge_rules = dict(definition.get("challenge_rules") or {})
    fallback_rules = dict(definition.get("knowledge_fallback_rules") or {})
    fallback_signals = set(definition.get("knowledge_fallback_signals") or [])
    interest_rules = dict(definition.get("interest_rules") or {})
    strategy_rules = dict(definition.get("strategy_suggestions") or {})
    recurrent_bands = set(definition.get("recurrent_interest_bands") or [])

    challenge = str(challenge_rules.get(recent, "grade_baseline_uncalibrated"))
    if recent in fallback_signals:
        challenge = str(fallback_rules.get(knowledge, challenge))
    interest_support = str(interest_rules.get(interest, "neutral"))
    strategy_key = "recurrent" if interest in recurrent_bands else recent
    candidate_strategy = str(strategy_rules.get(strategy_key, strategy_rules.get("default", "baseline")))
    if candidate_strategy == "baseline":
        candidate_strategy = vector["baseline_strategy_id"]
    if challenge not in CHALLENGE_VALUES or interest_support not in INTEREST_VALUES:
        raise ValueError("personalization policy emitted an unsupported plan value")
    return {
        "strategy_id": candidate_strategy,
        "challenge_assumption": challenge,
        "interest_support": interest_support,
    }


SELECTOR_IMPLEMENTATION_DIGEST = sha256(
    "\n".join(
        (
            inspect.getsource(assert_feature_vector),
            inspect.getsource(select_from_features),
        )
    ).encode()
).hexdigest()
FEATURE_EXTRACTION_IMPLEMENTATION_DIGEST = sha256(
    "\n".join(
        inspect.getsource(item)
        for item in (
            feedback_signal_query,
            feedback_feature_bands,
            projection_feature_bands,
            response_mode_feature,
        )
    ).encode()
).hexdigest()
REPLAY_VALIDATION_IMPLEMENTATION_DIGEST = sha256(
    "\n".join(
        inspect.getsource(item)
        for item in (
            contains_forbidden_key,
            normalize_replay_outcome,
            replay_case_payload,
        )
    ).encode()
).hexdigest()
RUNTIME_BEHAVIOR_MANIFEST = {
    "categorical_schema_digest": _digest(
        {
            "allowed_features": sorted(ALLOWED_FEATURES),
            "feature_values": {key: sorted(values) for key, values in sorted(FEATURE_VALUES.items())},
            "forbidden_replay_keys": sorted(FORBIDDEN_REPLAY_KEYS),
            "challenge_values": sorted(CHALLENGE_VALUES),
            "interest_values": sorted(INTEREST_VALUES),
        }
    ),
    "selector_implementation_digest": SELECTOR_IMPLEMENTATION_DIGEST,
    "feature_extraction_implementation_digest": FEATURE_EXTRACTION_IMPLEMENTATION_DIGEST,
    "replay_validation_implementation_digest": REPLAY_VALIDATION_IMPLEMENTATION_DIGEST,
}

POLICY_DEFINITION = {
    "feature_schema_version": FEATURE_SCHEMA_VERSION,
    "allowed_features": sorted(ALLOWED_FEATURES),
    "selector": {
        "version": SELECTOR_VERSION,
        "implementation_digest": SELECTOR_IMPLEMENTATION_DIGEST,
    },
    "runtime_behavior": RUNTIME_BEHAVIOR_MANIFEST,
    "challenge_rules": {
        "too_hard": "familiar",
        "too_easy": "independent_transfer",
        "positive": "stretch_with_help",
        "mixed": "familiar",
        "none": "grade_baseline_uncalibrated",
    },
    "knowledge_fallback_signals": ["none"],
    "knowledge_fallback_rules": {
        "transfer_observed": "independent_transfer",
        "independently_demonstrated": "independent_transfer",
        "demonstrated_in_context": "stretch_with_help",
        "practicing_with_support": "familiar",
        "exposure_recorded": "familiar",
        "not_observed": "grade_baseline_uncalibrated",
    },
    "interest_rules": {
        "sustained_parent_confirmed": "recurring_depth",
        "emerging_recurrent": "maintenance",
        "maintained_situational": "maintenance",
        "triggered_situational": "situational_trigger",
        "not_observed": "neutral",
    },
    "recurrent_interest_bands": ["emerging_recurrent", "sustained_parent_confirmed"],
    "strategy_suggestions": {
        "too_hard": "model-practice-fade-k2",
        "too_easy": "teach-back-k2",
        "recurrent": "retrieval-revisit-k2",
        "default": "baseline",
    },
    "application_scope": ["challenge_assumption", "interest_support"],
}
POLICY_DEFINITION_HASH = _digest(POLICY_DEFINITION)

RELEASE_MANIFEST_NAME = "release-manifest.json"
RELEASE_MANIFEST_KEYS = {
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
}


def _promotion_files(repo_root: Path) -> list[Path]:
    explicit = [repo_root / "pyproject.toml"]
    discovered: list[Path] = []
    for pattern in (
        "src/curiosity_engine/**/*.py",
        "src/curiosity_engine/templates/**/*",
        "src/curiosity_engine/static/**/*",
        "configs/*.json",
        # Only active, one-level suites bind release authority. Numbered
        # historical review rounds remain reproducible but are not release gates.
        "evals/*/*.json",
        "prompts/**/*",
        "schemas/**/*",
        "capabilities/**/*",
    ):
        discovered.extend(path for path in repo_root.glob(pattern) if path.is_file())
    files = sorted(set([*explicit, *discovered]))
    if not files or any(not path.is_file() for path in files):
        raise ValueError("personalization promotion content manifest is incomplete")
    return files


def _manifest_for_source_root(root: Path) -> dict[str, Any]:
    content = {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in _promotion_files(root)
    }
    manifest = {
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "selector_implementation_digest": SELECTOR_IMPLEMENTATION_DIGEST,
        "runtime_behavior": RUNTIME_BEHAVIOR_MANIFEST,
        "definition_hash": POLICY_DEFINITION_HASH,
        "evaluator_version": EVALUATOR_VERSION,
        "suite_version": PERSONALIZATION_SUITE_VERSION,
        "promotion_content_files": content,
    }
    return {**manifest, "promotion_content_hash": _digest(manifest)}


def _installed_content_path(root: Path, relative: str) -> Path:
    package_prefix = "src/curiosity_engine/"
    if relative.startswith(package_prefix):
        return Path(__file__).resolve().parent / relative.removeprefix(package_prefix)
    return root / relative


def _manifest_for_installed_root(root: Path) -> dict[str, Any]:
    path = root / RELEASE_MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"installed release manifest is unavailable or invalid: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != RELEASE_MANIFEST_KEYS:
        raise ValueError("installed release manifest has an invalid schema")
    content = manifest.get("promotion_content_files")
    if not isinstance(content, dict) or not content:
        raise ValueError("installed release manifest has no promotion content")
    if any(
        not isinstance(relative, str)
        or not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or not isinstance(digest, str)
        or len(digest) != 64
        for relative, digest in content.items()
    ):
        raise ValueError("installed release manifest content map is invalid")
    expected_identity = {
        "policy_id": POLICY_ID,
        "policy_version": POLICY_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selector_version": SELECTOR_VERSION,
        "selector_implementation_digest": SELECTOR_IMPLEMENTATION_DIGEST,
        "runtime_behavior": RUNTIME_BEHAVIOR_MANIFEST,
        "definition_hash": POLICY_DEFINITION_HASH,
        "evaluator_version": EVALUATOR_VERSION,
        "suite_version": PERSONALIZATION_SUITE_VERSION,
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("installed release manifest does not match the running policy code")
    unsigned = {key: value for key, value in manifest.items() if key != "promotion_content_hash"}
    if manifest.get("promotion_content_hash") != _digest(unsigned):
        raise ValueError("installed release manifest self-hash is invalid")
    for relative, expected_digest in content.items():
        installed_path = _installed_content_path(root, relative)
        if not installed_path.is_file() or sha256(installed_path.read_bytes()).hexdigest() != expected_digest:
            raise ValueError(f"installed promotion content is missing or changed: {relative}")
    return manifest


def promotion_manifest(repo_root: str | Path | None = None) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent
    if repo_root is None:
        source = Path(__file__).resolve().parents[2]
        if (source / "src" / "curiosity_engine").is_dir() and (source / "configs").is_dir():
            return _manifest_for_source_root(source)
        from .config import configuration_root

        root = configuration_root().resolve()
    else:
        root = Path(repo_root).resolve()
    if (root / "src" / "curiosity_engine").is_dir():
        if (root / "src" / "curiosity_engine").resolve() != package_root:
            raise ValueError("source release root does not own the running curiosity_engine package")
        return _manifest_for_source_root(root)
    return _manifest_for_installed_root(root)
