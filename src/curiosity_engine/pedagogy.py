from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .config import AppConfig
from .contracts import StrictModel
from .db import jdump

PEDAGOGY_LEGACY_CONTRACT_VERSION = "2.0"
PEDAGOGY_LEGACY_PERSONALIZED_CONTRACT_VERSION = "2.0p1"
PEDAGOGY_CONTRACT_VERSION = "2.1"
PEDAGOGY_PERSONALIZED_CONTRACT_VERSION = "2.1p1"
PEDAGOGY_VALIDATOR_VERSION = "2.1"
MAX_SELECTED_INSTRUCTION_CHARS = 16_000

QuestionIntent = Literal[
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
]
ResponseMode = Literal[
    "spoken",
    "written",
    "drawn",
    "gesture_or_manipulative",
    "selection",
    "demonstration",
    "symbolic",
]
InterestSupport = Literal["neutral", "situational_trigger", "maintenance", "recurring_depth"]
ChallengeAssumption = Literal[
    "grade_baseline_uncalibrated",
    "familiar",
    "stretch_with_help",
    "independent_transfer",
]


class ScaffoldPolicy(StrictModel):
    hints: list[str] = Field(default_factory=list, max_length=3)
    fade_rule: str = Field(min_length=1, max_length=240)
    frustration_stop: str = Field(min_length=1, max_length=240)
    allowed_assistance: list[
        Literal["clarifying_prompt", "general_prompt", "hint", "worked_example", "joint_completion"]
    ] = Field(default_factory=list, max_length=5)


class EvidenceOpportunityProposal(StrictModel):
    """Model-proposed observation shape without rubric, outcome, actor, or evidence authority."""

    collection: Literal["not_collected", "future_observation_only"]
    template_id: str = Field(min_length=1, max_length=80)
    observable_behavior: str | None = Field(default=None, max_length=300)
    response_modes: list[ResponseMode] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def complete_semantic_shape(self) -> EvidenceOpportunityProposal:
        if self.collection == "not_collected" and (self.observable_behavior or self.response_modes):
            raise ValueError("non-collection proposal may not carry assessment behavior")
        if self.collection == "future_observation_only" and not (self.observable_behavior and self.response_modes):
            raise ValueError("future observation proposal requires behavior and response modes")
        return self


class EvidenceOpportunityTemplate(StrictModel):
    collection: Literal["not_collected", "future_observation_only"]
    template_id: str = Field(min_length=1, max_length=80)
    template_version: int = Field(ge=1)
    observable_behavior: str | None = Field(default=None, max_length=300)
    response_modes: list[ResponseMode] = Field(default_factory=list, max_length=5)
    assistance_capture_required: bool = False
    rubric_id: str | None = Field(default=None, max_length=100)
    rubric_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def keeps_opportunity_distinct_from_evidence(self) -> EvidenceOpportunityTemplate:
        if self.collection == "not_collected":
            if self.observable_behavior or self.rubric_id or self.rubric_version:
                raise ValueError("non-collection may not carry assessment fields")
        elif not all((self.observable_behavior, self.response_modes, self.rubric_id, self.rubric_version)):
            raise ValueError("future observation requires behavior, modes, and reviewed rubric")
        return self


class ClosurePolicy(StrictModel):
    child_closure: str = Field(min_length=1, max_length=240)
    optional_revisit: str | None = Field(default=None, max_length=240)
    scheduled: Literal[False] = False
    continuation_required: Literal[False] = False


class AccessibilityPolicy(StrictModel):
    response_choices: list[ResponseMode] = Field(min_length=2, max_length=5)
    no_writing_alternative: str = Field(min_length=1, max_length=220)
    color_independent: bool = True
    fine_motor_alternative: str = Field(min_length=1, max_length=220)
    reading_support: str = Field(min_length=1, max_length=220)
    requested_supports: list[Literal["no_writing", "low_fine_motor", "read_aloud"]] = Field(
        default_factory=list,
        max_length=3,
    )


class SafetyPolicy(StrictModel):
    risk_level: Literal["low", "elevated", "sensitive"]
    hazards: list[
        Literal[
            "heat",
            "ignition",
            "electrical",
            "sharp_tools",
            "chemicals",
            "water",
            "small_parts",
            "outdoors",
            "body_or_health",
        ]
    ] = Field(default_factory=list, max_length=9)
    minimum_supervision: Literal["none", "nearby", "direct"]
    rules: list[str] = Field(default_factory=list, max_length=8)
    response_requirements: list[
        Literal[
            "immediate_trusted_grown_up",
            "explicit_hazard_prohibition",
            "explicit_appliance_water_prohibition",
            "explicit_ignition_prohibition",
            "explicit_chemical_prohibition",
            "explicit_sharp_tool_prohibition",
        ]
    ] = Field(default_factory=list, max_length=6)


class FamilyLoadPolicy(StrictModel):
    max_minutes: int = Field(ge=1, le=30)
    parent_effort: Literal["very_low", "low", "medium"]
    max_materials: int = Field(ge=0, le=6)
    supervision: Literal["none", "nearby", "direct"]
    cleanup: Literal["none", "quick", "moderate"]


class PedagogyPlanProposal(StrictModel):
    """Model-owned semantic proposal. Authority identifiers are deliberately absent."""

    question_intent: QuestionIntent
    learning_goal: str = Field(min_length=1, max_length=300)
    concept_ids: list[str] = Field(min_length=1, max_length=3)
    primary_strategy_id: str = Field(min_length=1, max_length=80)
    supporting_strategy_id: str | None = Field(default=None, max_length=80)
    domain_overlay_id: str = Field(min_length=1, max_length=80)
    interest_support: Literal["neutral"] = "neutral"
    challenge_assumption: Literal["grade_baseline_uncalibrated"] = "grade_baseline_uncalibrated"
    representations: list[Literal["plain_language", "example", "story", "physical", "diagram", "symbols", "photo"]] = (
        Field(min_length=1, max_length=3)
    )
    scaffold: ScaffoldPolicy
    evidence: EvidenceOpportunityProposal
    closure: ClosurePolicy
    accessibility: AccessibilityPolicy
    family_load: FamilyLoadPolicy


class VersionedRef(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=32)


class ConceptRef(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)
    label: str = Field(min_length=1, max_length=160)
    domain: str = Field(min_length=1, max_length=80)


class PersonalizationConceptRef(StrictModel):
    """The exact ontology version used to derive personalization features."""

    id: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)


class PersonalizationPlanBinding(StrictModel):
    """Exact code-owned decision allowed to calibrate a plan after both human gates."""

    decision_id: str = Field(pattern=r"^pers_[a-f0-9]{24}$")
    decision_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    child_id: str = Field(min_length=1, max_length=120)
    event_id: str = Field(min_length=1, max_length=120)
    event_type: str = Field(min_length=1, max_length=80)
    event_text_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    workflow: str = Field(min_length=1, max_length=80)
    proposal_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    baseline_plan_id: str = Field(pattern=r"^ped_[a-f0-9]{20}$")
    baseline_plan_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    concept_ids: list[str] = Field(min_length=1, max_length=3)
    concept_refs: list[PersonalizationConceptRef] = Field(min_length=1, max_length=3)
    question_intent: QuestionIntent
    policy_id: str = Field(min_length=1, max_length=160)
    policy_version: int = Field(ge=1)
    definition_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    family_authorization_epoch: int = Field(ge=1)
    policy_authorization_epoch: int = Field(ge=1)
    evaluation_id: str = Field(pattern=r"^perseval_[a-f0-9]{24}$")
    promotion_review_id: str = Field(pattern=r"^persreview_[a-f0-9]{24}$")
    mode: Literal["active"] = "active"
    baseline_strategy_id: str = Field(min_length=1, max_length=100)
    interest_support: InterestSupport
    challenge_assumption: ChallengeAssumption
    applied_fields: list[Literal["interest_support", "challenge_assumption"]] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def exact_application_scope(self) -> PersonalizationPlanBinding:
        if self.applied_fields != ["interest_support", "challenge_assumption"]:
            raise ValueError("personalization may apply only the reviewed calibration fields")
        if self.concept_ids != sorted(dict.fromkeys(self.concept_ids)):
            raise ValueError("personalization concept bindings must be canonical and unique")
        if self.concept_ids != [item.id for item in self.concept_refs]:
            raise ValueError("personalization concept IDs do not match their versioned references")
        if self.concept_refs != sorted(self.concept_refs, key=lambda item: (item.id, item.version)):
            raise ValueError("personalization concept references must be canonical")
        return self


class ApprovedPedagogyPlan(StrictModel):
    """Immutable, code-bound plan used by generation and every downstream release unit."""

    plan_id: str = Field(pattern=r"^ped_[a-f0-9]{20}$")
    plan_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_id: str = Field(min_length=1, max_length=120)
    child_id: str = Field(min_length=1, max_length=120)
    workflow: str = Field(min_length=1, max_length=80)
    question_intent: QuestionIntent
    learning_goal: str = Field(min_length=1, max_length=300)
    concepts: list[ConceptRef] = Field(min_length=1, max_length=3)
    primary_strategy: VersionedRef
    task_template: VersionedRef
    supporting_strategy: VersionedRef | None = None
    domain_overlay: VersionedRef
    interest_support: InterestSupport = "neutral"
    challenge_assumption: ChallengeAssumption = "grade_baseline_uncalibrated"
    personalization: PersonalizationPlanBinding | None = None
    representations: list[str] = Field(min_length=1, max_length=3)
    scaffold: ScaffoldPolicy
    evidence: EvidenceOpportunityTemplate
    closure: ClosurePolicy
    accessibility: AccessibilityPolicy
    safety: SafetyPolicy
    family_load: FamilyLoadPolicy
    allowed_artifact_types: list[Literal["worksheet", "reference", "activity", "challenge"]] = Field(
        default_factory=list, max_length=4
    )
    required_critics: list[str] = Field(min_length=1, max_length=10)
    registry_version: str
    registry_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    instruction_digests: dict[str, str] = Field(default_factory=dict, max_length=6)
    contract_version: Literal[
        PEDAGOGY_LEGACY_CONTRACT_VERSION,
        PEDAGOGY_LEGACY_PERSONALIZED_CONTRACT_VERSION,
        PEDAGOGY_CONTRACT_VERSION,
        PEDAGOGY_PERSONALIZED_CONTRACT_VERSION,
    ] = PEDAGOGY_CONTRACT_VERSION
    validator_version: Literal["2.0", PEDAGOGY_VALIDATOR_VERSION] = PEDAGOGY_VALIDATOR_VERSION
    routing_provenance: Literal[
        "model_proposal",
        "safe_fallback",
        "legacy_backfill",
        "context_artifact",
        "artifact_derivation",
        "visual_revision",
    ]
    supersedes_plan_id: str | None = Field(default=None, max_length=120)
    created_at: datetime

    @model_validator(mode="after")
    def contract_and_validator_versions_match(self) -> ApprovedPedagogyPlan:
        legacy_contract = self.contract_version in {
            PEDAGOGY_LEGACY_CONTRACT_VERSION,
            PEDAGOGY_LEGACY_PERSONALIZED_CONTRACT_VERSION,
        }
        if legacy_contract != (self.validator_version == "2.0"):
            raise ValueError("pedagogy contract and validator versions are incompatible")
        return self

    def assert_canonical_integrity(self) -> None:
        authority = self.model_dump(
            mode="json",
            exclude={"plan_id", "plan_hash", "created_at"},
        )
        if authority.get("personalization") is None:
            authority.pop("personalization", None)
        if self.validator_version == "2.0":
            # Stored validator-2.0 plans predate typed child-facing safety
            # response requirements. Keep their original canonical hash
            # readable while every new plan binds the 2.1 requirements.
            authority["safety"].pop("response_requirements", None)
        expected_hash = sha256(jdump(authority).encode()).hexdigest()
        decision_identity = f"|{self.personalization.decision_id}" if self.personalization else ""
        validator_identity = "" if self.validator_version == "2.0" else f"|validator:{self.validator_version}"
        expected_id = (
            "ped_"
            + sha256(
                f"{self.event_id}|{self.workflow}|{self.contract_version}"
                f"{validator_identity}{decision_identity}".encode()
            ).hexdigest()[:20]
        )
        if self.plan_hash != expected_hash or self.plan_id != expected_id:
            raise ValueError("approved pedagogy plan failed canonical integrity verification")

    def trace(self) -> dict[str, Any]:
        trace = {
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "question_intent": self.question_intent,
            "learning_goal": self.learning_goal,
            "concept_ids": [item.id for item in self.concepts],
            "primary_strategy_id": self.primary_strategy.id,
            "task_template_id": self.task_template.id,
            "task_template_version": self.task_template.version,
            "supporting_strategy_id": self.supporting_strategy.id if self.supporting_strategy else None,
            "domain_overlay_id": self.domain_overlay.id,
            "interest_support": self.interest_support,
            "challenge_assumption": self.challenge_assumption,
            "representations": list(self.representations),
            "scaffold": self.scaffold.model_dump(mode="json"),
            "evidence_collection": self.evidence.collection,
            "evidence_template_id": self.evidence.template_id,
            "evidence_template_version": self.evidence.template_version,
            "rubric_id": self.evidence.rubric_id,
            "rubric_version": self.evidence.rubric_version,
            "closure": self.closure.model_dump(mode="json"),
            "accessibility": self.accessibility.model_dump(mode="json"),
            "safety": self.safety.model_dump(mode="json"),
            "family_load": self.family_load.model_dump(mode="json"),
            "registry_version": self.registry_version,
        }
        if self.personalization is not None:
            trace["personalization"] = self.personalization.model_dump(mode="json")
        return trace


class PedagogyRegistry:
    """Validated strategy registry with progressively disclosed instruction content."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).resolve() if root else AppConfig.load().root
        path = self.root / "configs" / "pedagogy-registry.json"
        self.raw = json.loads(path.read_text(encoding="utf-8"))
        self.version = str(self.raw["version"])
        self.strategies = self._unique("strategies")
        self.overlays = self._unique("overlays")
        self.concepts = self._unique("concepts")
        self.evidence_templates = self._unique("evidence_templates")
        self.instruction_digests: dict[str, str] = {}
        for item in [*self.strategies.values(), *self.overlays.values()]:
            instruction_path = item.get("instructions_path")
            if not instruction_path:
                continue
            resolved = (self.root / str(instruction_path)).resolve()
            if not resolved.is_relative_to(self.root):
                raise ValueError(f"pedagogy instruction escapes registry root: {item['id']}")
            content = resolved.read_text(encoding="utf-8").strip()
            if not content:
                raise ValueError(f"pedagogy instruction is empty: {item['id']}")
            self.instruction_digests[str(item["id"])] = sha256(content.encode()).hexdigest()
        normalized = {
            **self.raw,
            "instruction_digests": self.instruction_digests,
        }
        self.digest = sha256(jdump(normalized).encode()).hexdigest()
        self._validate_references()

    def _unique(self, key: str) -> dict[str, dict[str, Any]]:
        values = self.raw.get(key)
        if not isinstance(values, list):
            raise ValueError(f"pedagogy registry {key} must be a list")
        result: dict[str, dict[str, Any]] = {}
        for item in values:
            item_id = str(item.get("id") or "")
            if not re.fullmatch(r"[a-z][a-z0-9_-]{1,99}", item_id):
                raise ValueError(f"invalid pedagogy registry id: {item_id!r}")
            if item_id in result:
                raise ValueError(f"duplicate pedagogy registry id: {item_id}")
            if not item.get("version"):
                raise ValueError(f"unversioned pedagogy registry entry: {item_id}")
            result[item_id] = dict(item)
        return result

    def _validate_references(self) -> None:
        for item in self.strategies.values():
            for overlay_id in item.get("allowed_overlays", []):
                if overlay_id not in self.overlays:
                    raise ValueError(f"unknown overlay {overlay_id!r} in {item['id']}")
            for evidence_id in item.get("evidence_templates", []):
                if evidence_id not in self.evidence_templates:
                    raise ValueError(f"unknown evidence template {evidence_id!r} in {item['id']}")
        if not any(item.get("kind") == "primary" for item in self.strategies.values()):
            raise ValueError("pedagogy registry has no primary strategies")

    def planner_cards(self) -> dict[str, Any]:
        def card(item: dict[str, Any]) -> dict[str, Any]:
            return {
                key: item[key]
                for key in (
                    "id",
                    "version",
                    "kind",
                    "description",
                    "allowed_intents",
                    "allowed_overlays",
                    "evidence_templates",
                )
                if key in item
            }

        return {
            "registry_version": self.version,
            "strategies": [card(item) for item in self.strategies.values() if item.get("enabled", True)],
            "overlays": [card(item) for item in self.overlays.values() if item.get("enabled", True)],
            "concepts": [
                {key: item[key] for key in ("id", "version", "label", "domain")} for item in self.concepts.values()
            ],
            "evidence_templates": [
                {key: item[key] for key in ("id", "version", "description")}
                for item in self.evidence_templates.values()
            ],
        }

    def resolve_strategy(self, strategy_id: str, *, kind: str) -> dict[str, Any]:
        item = self.strategies.get(strategy_id)
        if not item or not item.get("enabled", True) or item.get("kind") != kind:
            raise ValueError(f"unknown, disabled, or wrong-kind {kind} strategy: {strategy_id}")
        return item

    def resolve_overlay(self, overlay_id: str) -> dict[str, Any]:
        item = self.overlays.get(overlay_id)
        if not item or not item.get("enabled", True):
            raise ValueError(f"unknown or disabled domain overlay: {overlay_id}")
        return item

    def instruction_bundle(self, plan: ApprovedPedagogyPlan) -> str:
        self.validate_plan(plan)
        selected = [
            "udl-representation-k2",
            "metacognitive-closure-k2",
            plan.primary_strategy.id,
            plan.domain_overlay.id,
        ]
        if plan.supporting_strategy:
            selected.append(plan.supporting_strategy.id)
        chunks: list[str] = []
        for item_id in selected:
            item = self.strategies.get(item_id) or self.overlays.get(item_id)
            if not item:
                raise ValueError(f"approved plan references missing instruction: {item_id}")
            path = (self.root / str(item["instructions_path"])).resolve()
            content = path.read_text(encoding="utf-8").strip()
            digest = sha256(content.encode()).hexdigest()
            if plan.instruction_digests.get(item_id) != digest:
                raise ValueError(f"instruction digest changed after plan approval: {item_id}")
            chunks.append(f"## Selected pedagogy: {item_id}@{item['version']}\n{content}")
        bundle = "\n\n".join(chunks)
        if len(bundle) > MAX_SELECTED_INSTRUCTION_CHARS:
            raise ValueError("selected pedagogy instructions exceed prompt budget")
        return bundle

    def validate_plan(self, plan: ApprovedPedagogyPlan) -> None:
        """Verify canonical content plus every current registry and instruction binding."""

        plan.assert_canonical_integrity()
        if plan.registry_version != self.version or plan.registry_digest != self.digest:
            raise ValueError("approved pedagogy plan registry identity is stale or corrupt")
        if plan.personalization is None:
            if plan.interest_support != "neutral" or plan.challenge_assumption != "grade_baseline_uncalibrated":
                raise ValueError("calibrated plan values require an active personalization binding")
        elif (
            plan.personalization.interest_support != plan.interest_support
            or plan.personalization.challenge_assumption != plan.challenge_assumption
            or plan.personalization.baseline_strategy_id != plan.primary_strategy.id
            or set(plan.personalization.applied_fields) != {"interest_support", "challenge_assumption"}
        ):
            raise ValueError("personalization binding does not match the approved plan")
        primary = self.resolve_strategy(plan.primary_strategy.id, kind="primary")
        if str(primary["version"]) != plan.primary_strategy.version:
            raise ValueError("approved primary strategy version is stale or corrupt")
        if plan.supporting_strategy:
            supporting = self.resolve_strategy(plan.supporting_strategy.id, kind="supporting")
            if str(supporting["version"]) != plan.supporting_strategy.version:
                raise ValueError("approved supporting strategy version is stale or corrupt")
        overlay = self.resolve_overlay(plan.domain_overlay.id)
        if str(overlay["version"]) != plan.domain_overlay.version:
            raise ValueError("approved domain overlay version is stale or corrupt")
        for concept in plan.concepts:
            entry = self.concepts.get(concept.id)
            if not entry or (
                int(entry["version"]) != concept.version
                or str(entry["label"]) != concept.label
                or str(entry["domain"]) != concept.domain
            ):
                raise ValueError("approved concept definition is stale or corrupt")
        expected_task = _task_template_id(
            plan.primary_strategy.id,
            plan.concepts,
            domain_overlay_id=plan.domain_overlay.id,
            representations=plan.representations,
            evidence=plan.evidence,
        )
        if plan.task_template.id != expected_task or plan.task_template.version != str(primary["version"]):
            raise ValueError("approved task template binding is stale or corrupt")
        evidence_entry = self.evidence_templates.get(plan.evidence.template_id)
        if not evidence_entry or not evidence_entry.get("enabled", True):
            raise ValueError("approved evidence template is unavailable")
        if int(evidence_entry["version"]) != plan.evidence.template_version:
            raise ValueError("approved evidence template version is stale or corrupt")
        if plan.evidence.collection == "future_observation_only":
            expected_rubric = _rubric_id(str(evidence_entry["rubric_id"]), plan.concepts)
            if (
                plan.evidence.observable_behavior != str(evidence_entry["description"])
                or plan.evidence.rubric_id != expected_rubric
                or plan.evidence.rubric_version != int(evidence_entry["rubric_version"])
            ):
                raise ValueError("approved evidence definition is stale or corrupt")
        selected = [
            "udl-representation-k2",
            "metacognitive-closure-k2",
            plan.primary_strategy.id,
            plan.domain_overlay.id,
        ]
        if plan.supporting_strategy:
            selected.append(plan.supporting_strategy.id)
        expected_digests = {item_id: self.instruction_digests[item_id] for item_id in selected}
        if plan.instruction_digests != expected_digests:
            raise ValueError("approved pedagogy instruction binding is stale or corrupt")

    def router_instruction(self) -> str:
        item = self.strategies["pedagogy-router-k2"]
        path = (self.root / str(item["instructions_path"])).resolve()
        content = path.read_text(encoding="utf-8").strip()
        if sha256(content.encode()).hexdigest() != self.instruction_digests[item["id"]]:
            raise ValueError("pedagogy router instruction changed during registry load")
        return content


MANDATORY_CRITICS = (
    "critic_factual",
    "critic_plan_fidelity",
    "critic_accessibility",
    "critic_evidence",
    "critic_context",
    "critic_parent_effort",
)


class PedagogyCompiler:
    def __init__(self, registry: PedagogyRegistry | None = None):
        self.registry = registry or PedagogyRegistry()

    def compile(
        self,
        proposal: PedagogyPlanProposal,
        *,
        event: dict[str, Any],
        workflow: str,
        provenance: Literal[
            "model_proposal",
            "safe_fallback",
            "legacy_backfill",
            "context_artifact",
            "artifact_derivation",
            "visual_revision",
        ],
        broad_artifact_compatibility: bool = False,
        supersedes_plan_id: str | None = None,
        personalization_decision: dict[str, Any] | None = None,
    ) -> ApprovedPedagogyPlan:
        from .personalization import plan_binding_from_decision

        personalization = (
            plan_binding_from_decision(personalization_decision) if personalization_decision is not None else None
        )
        event_id = str(event.get("id") or "").strip()
        child_id = str(event.get("child_id") or "").strip()
        if not event_id or not child_id:
            raise ValueError("approved pedagogy plans require code-owned event and child bindings")
        event_text = str(event.get("text") or "")
        sensitive = _is_sensitive(event_text)
        safety_refusal = _requires_physical_safety_refusal(event_text)
        direct_requested = _requires_direct_answer(event_text, workflow)
        proposal_hash = sha256(jdump(proposal.model_dump(mode="json")).encode()).hexdigest()
        if personalization:
            unique_concepts = sorted(dict.fromkeys(proposal.concept_ids))
            mismatched = any(
                (
                    personalization.child_id != child_id,
                    personalization.event_id != event_id,
                    personalization.event_type != str(event.get("type") or ""),
                    personalization.workflow != workflow,
                    personalization.event_text_hash != sha256(event_text.encode()).hexdigest(),
                    personalization.proposal_hash != proposal_hash,
                    personalization.question_intent != proposal.question_intent,
                    personalization.concept_ids != unique_concepts,
                    personalization.baseline_strategy_id != proposal.primary_strategy_id,
                )
            )
            if mismatched:
                raise ValueError("personalization decision does not match the bound event and proposal")
            if (
                sensitive
                or safety_refusal
                or direct_requested
                or proposal.question_intent in {"direct_answer", "sensitive_support"}
            ):
                raise ValueError("direct and sensitive plans cannot carry personalization")
            baseline_plan = self.compile(
                proposal,
                event=event,
                workflow=workflow,
                provenance=provenance,
                broad_artifact_compatibility=broad_artifact_compatibility,
                supersedes_plan_id=supersedes_plan_id,
            )
            if (
                personalization.baseline_plan_id != baseline_plan.plan_id
                or personalization.baseline_plan_hash != baseline_plan.plan_hash
                or personalization.concept_refs
                != [PersonalizationConceptRef(id=item.id, version=item.version) for item in baseline_plan.concepts]
            ):
                raise ValueError("personalization decision does not match its exact baseline plan")
        if sensitive and proposal.question_intent != "sensitive_support":
            raise ValueError("sensitive request requires sensitive-support intent")
        if safety_refusal and not sensitive and proposal.question_intent != "direct_answer":
            raise ValueError("prohibited physical-hazard request requires direct-answer intent")
        if not sensitive and direct_requested and proposal.question_intent != "direct_answer":
            raise ValueError("explicit direct request requires direct-answer intent")
        primary = self.registry.resolve_strategy(proposal.primary_strategy_id, kind="primary")
        if proposal.question_intent not in primary.get("allowed_intents", []):
            raise ValueError("primary strategy is incompatible with question intent")
        supporting = None
        if proposal.supporting_strategy_id:
            supporting = self.registry.resolve_strategy(proposal.supporting_strategy_id, kind="supporting")
            if proposal.supporting_strategy_id not in primary.get("compatible_supporting", []):
                raise ValueError("supporting strategy is incompatible with primary strategy")
        overlay = self.registry.resolve_overlay(proposal.domain_overlay_id)
        if proposal.domain_overlay_id not in primary.get("allowed_overlays", []):
            raise ValueError("domain overlay is incompatible with primary strategy")

        proposal_sensitive = proposal.question_intent == "sensitive_support"
        overlay_sensitive = proposal.domain_overlay_id == "sensitive-support-k2"
        if proposal_sensitive != overlay_sensitive:
            raise ValueError("sensitive intent and overlay must be selected together")
        if sensitive and not proposal_sensitive:
            raise ValueError("sensitive request requires the sensitive-support overlay")
        sensitive_plan = sensitive or proposal_sensitive
        direct = proposal.question_intent in {"direct_answer", "sensitive_support"}
        safety = _approved_safety(event_text, sensitive=sensitive_plan)
        accessibility = _approved_accessibility(event_text, proposal.accessibility)
        concepts: list[ConceptRef] = []
        unresolved = False
        for concept_id in dict.fromkeys(proposal.concept_ids):
            item = self.registry.concepts.get(concept_id)
            if not item:
                unresolved = True
                continue
            concepts.append(
                ConceptRef(
                    id=concept_id,
                    version=int(item["version"]),
                    label=str(item["label"]),
                    domain=str(item["domain"]),
                )
            )
        if not concepts:
            item = self.registry.concepts["general-question-and-explanation"]
            concepts = [
                ConceptRef(
                    id=item["id"],
                    version=int(item["version"]),
                    label=item["label"],
                    domain=item["domain"],
                )
            ]
            unresolved = True
        concepts.sort(key=lambda item: (item.id, item.version))
        if personalization and (
            [item.id for item in concepts] != personalization.concept_ids
            or [PersonalizationConceptRef(id=item.id, version=item.version) for item in concepts]
            != personalization.concept_refs
        ):
            raise ValueError("personalization requires exact resolved concept bindings")

        evidence_id = "not-collected" if direct else proposal.evidence.template_id
        evidence_entry = self.registry.evidence_templates.get(evidence_id)
        if not evidence_entry or not evidence_entry.get("enabled", True):
            raise ValueError(f"unknown or disabled evidence template: {evidence_id}")
        if evidence_id not in primary.get("evidence_templates", []):
            raise ValueError("evidence template is incompatible with primary strategy")
        evidence = _resolved_evidence(proposal.evidence, evidence_entry, concepts=concepts, direct=direct)
        if unresolved:
            evidence = _not_collected_evidence()
        if evidence.collection == "future_observation_only" and not (
            set(evidence.response_modes) & set(accessibility.response_choices)
        ):
            raise ValueError("evidence response modes conflict with approved accessibility choices")

        critics = list(dict.fromkeys([*MANDATORY_CRITICS, *primary.get("required_critics", [])]))
        if safety.risk_level != "low":
            critics.append("critic_safety")
        artifacts = list(primary.get("artifact_affordances", []))
        if direct:
            artifacts = []
        elif broad_artifact_compatibility:
            artifacts = ["worksheet", "reference", "activity", "challenge"]
        family_load = proposal.family_load
        supervision_order = {"none": 0, "nearby": 1, "direct": 2}
        if supervision_order[family_load.supervision] < supervision_order[safety.minimum_supervision]:
            family_load = family_load.model_copy(update={"supervision": safety.minimum_supervision})
        if direct:
            family_load = family_load.model_copy(
                update={"max_materials": 0, "parent_effort": "very_low", "cleanup": "none"}
            )
        if broad_artifact_compatibility and not direct:
            family_load = family_load.model_copy(
                update={"max_minutes": 30, "parent_effort": "medium", "max_materials": 6}
            )

        selected_ids = ["udl-representation-k2", "metacognitive-closure-k2", primary["id"], overlay["id"]]
        if supporting:
            selected_ids.append(supporting["id"])
        instruction_digests = {item_id: self.registry.instruction_digests[item_id] for item_id in selected_ids}
        authority = {
            "event_id": event_id,
            "child_id": child_id,
            "workflow": workflow,
            "question_intent": proposal.question_intent,
            "learning_goal": proposal.learning_goal,
            "concepts": [item.model_dump(mode="json") for item in concepts],
            "primary_strategy": {"id": primary["id"], "version": str(primary["version"])},
            "task_template": {
                "id": _task_template_id(
                    primary["id"],
                    concepts,
                    domain_overlay_id=overlay["id"],
                    representations=proposal.representations,
                    evidence=evidence,
                ),
                "version": str(primary["version"]),
            },
            "supporting_strategy": (
                {"id": supporting["id"], "version": str(supporting["version"])} if supporting else None
            ),
            "domain_overlay": {"id": overlay["id"], "version": str(overlay["version"])},
            "interest_support": personalization.interest_support if personalization else "neutral",
            "challenge_assumption": (
                personalization.challenge_assumption if personalization else "grade_baseline_uncalibrated"
            ),
            "representations": proposal.representations,
            "scaffold": _approved_scaffold(proposal.scaffold, proposal.question_intent).model_dump(mode="json"),
            "evidence": evidence.model_dump(mode="json"),
            "closure": proposal.closure.model_dump(mode="json"),
            "accessibility": accessibility.model_dump(mode="json"),
            "safety": safety.model_dump(mode="json"),
            "family_load": family_load.model_dump(mode="json"),
            "allowed_artifact_types": artifacts,
            "required_critics": critics,
            "registry_version": self.registry.version,
            "registry_digest": self.registry.digest,
            "instruction_digests": instruction_digests,
            "contract_version": (
                PEDAGOGY_PERSONALIZED_CONTRACT_VERSION if personalization else PEDAGOGY_CONTRACT_VERSION
            ),
            "validator_version": PEDAGOGY_VALIDATOR_VERSION,
            "routing_provenance": provenance,
            "supersedes_plan_id": supersedes_plan_id,
        }
        if personalization is not None:
            authority["personalization"] = personalization.model_dump(mode="json")
        plan_hash = sha256(jdump(authority).encode()).hexdigest()
        decision_identity = f"|{personalization.decision_id}" if personalization else ""
        validator_identity = f"|validator:{PEDAGOGY_VALIDATOR_VERSION}"
        plan_id = (
            "ped_"
            + sha256(
                f"{event_id}|{workflow}|{authority['contract_version']}{validator_identity}{decision_identity}".encode()
            ).hexdigest()[:20]
        )
        plan = ApprovedPedagogyPlan(
            plan_id=plan_id,
            plan_hash=plan_hash,
            created_at=datetime.now(UTC),
            **authority,
        )
        self.registry.validate_plan(plan)
        return plan

    def fallback(
        self,
        *,
        event: dict[str, Any],
        workflow: str,
        provenance: Literal[
            "safe_fallback", "legacy_backfill", "context_artifact", "artifact_derivation", "visual_revision"
        ] = "safe_fallback",
        broad_artifact_compatibility: bool = False,
        supersedes_plan_id: str | None = None,
    ) -> ApprovedPedagogyPlan:
        proposal = default_plan_proposal(event, workflow, self.registry)
        return self.compile(
            proposal,
            event=event,
            workflow=workflow,
            provenance=provenance,
            broad_artifact_compatibility=broad_artifact_compatibility,
            supersedes_plan_id=supersedes_plan_id,
        )

    def planner_failure_fallback(
        self,
        *,
        event: dict[str, Any],
        workflow: str,
    ) -> ApprovedPedagogyPlan:
        """Keep the request's useful teaching intent when planner output is unusable.

        The deterministic default is already grade-bounded and safe. Missing an
        optional inquiry move is handled as quality guidance at response review,
        so a planner failure does not need to flatten every exploration into a
        generic answer-and-wonder response.
        """

        return self.fallback(event=event, workflow=workflow)


def _not_collected_evidence(version: int = 1) -> EvidenceOpportunityTemplate:
    return EvidenceOpportunityTemplate(
        collection="not_collected",
        template_id="not-collected",
        template_version=version,
    )


def proposal_from_approved(plan: ApprovedPedagogyPlan) -> PedagogyPlanProposal:
    """Copy semantic choices without copying any plan/task/rubric/evidence authority."""

    evidence = EvidenceOpportunityProposal(
        collection=plan.evidence.collection,
        template_id=plan.evidence.template_id,
        observable_behavior=plan.evidence.observable_behavior,
        response_modes=plan.evidence.response_modes,
    )
    return PedagogyPlanProposal(
        question_intent=plan.question_intent,
        learning_goal=plan.learning_goal,
        concept_ids=[item.id for item in plan.concepts],
        primary_strategy_id=plan.primary_strategy.id,
        supporting_strategy_id=plan.supporting_strategy.id if plan.supporting_strategy else None,
        domain_overlay_id=plan.domain_overlay.id,
        representations=plan.representations,
        scaffold=plan.scaffold,
        evidence=evidence,
        closure=plan.closure,
        accessibility=plan.accessibility,
        family_load=plan.family_load,
    )


def _resolved_evidence(
    proposed: EvidenceOpportunityProposal,
    entry: dict[str, Any],
    *,
    concepts: list[ConceptRef],
    direct: bool,
) -> EvidenceOpportunityTemplate:
    if direct or proposed.collection == "not_collected":
        return _not_collected_evidence(int(entry["version"]))
    return EvidenceOpportunityTemplate(
        collection="future_observation_only",
        template_id=str(entry["id"]),
        template_version=int(entry["version"]),
        # The planner selects a reviewed template and response modes. It cannot
        # author a criterion which declares the system's own output to be proof.
        observable_behavior=str(entry["description"]),
        response_modes=proposed.response_modes,
        rubric_id=_rubric_id(str(entry["rubric_id"]), concepts),
        rubric_version=int(entry["rubric_version"]),
        assistance_capture_required=True,
    )


def _concept_set_digest(concepts: list[ConceptRef]) -> str:
    value = "|".join(f"{item.id}@{item.version}" for item in sorted(concepts, key=lambda item: item.id))
    return sha256(value.encode()).hexdigest()[:12]


def _rubric_id(base_id: str, concepts: list[ConceptRef]) -> str:
    return f"{base_id}-{_concept_set_digest(concepts)}"


def _task_template_id(
    primary_id: str,
    concepts: list[ConceptRef],
    *,
    domain_overlay_id: str,
    representations: list[str],
    evidence: EvidenceOpportunityTemplate,
) -> str:
    """Bind an immutable Phase 1 task ID to every field used by its definition."""

    definition_key = jdump(
        {
            "domain_overlay_id": domain_overlay_id,
            "representations": representations,
            "evidence": evidence.model_dump(mode="json"),
        }
    )
    definition_digest = sha256(definition_key.encode()).hexdigest()[:12]
    return f"task-template-{primary_id}-{_concept_set_digest(concepts)}-{definition_digest}"


def materialize_pedagogy_definitions(db_path: str | Path, plan: ApprovedPedagogyPlan) -> None:
    """Resolve every approved task/concept/rubric ref into Phase 1 immutable definitions."""

    from .learner_model import (
        ConceptDefinition,
        LearningTaskDefinition,
        RubricCriterion,
        RubricDefinition,
        register_concept,
        register_learning_task,
        register_rubric,
    )
    from .learner_model import ConceptRef as LearnerConceptRef

    for concept in plan.concepts:
        register_concept(
            db_path,
            ConceptDefinition(
                concept_id=concept.id,
                version=concept.version,
                label=concept.label,
                domain=concept.domain,
                grade_band="K-2",
                description=f"Reviewed K-2 pedagogy concept: {concept.label}.",
                framework="curiosity-pedagogy-registry",
            ),
        )

    concept_digest = _concept_set_digest(plan.concepts)
    expected_task_id = _task_template_id(
        plan.primary_strategy.id,
        plan.concepts,
        domain_overlay_id=plan.domain_overlay.id,
        representations=plan.representations,
        evidence=plan.evidence,
    )
    if plan.task_template.id != expected_task_id:
        raise ValueError("approved pedagogy plan has a non-resolvable task template")
    rubric_base = (
        plan.evidence.rubric_id.rsplit(f"-{concept_digest}", 1)[0]
        if plan.evidence.rubric_id
        else f"rubric-plan-only-{plan.primary_strategy.id}"
    )
    rubric_id = _rubric_id(rubric_base, plan.concepts)
    if plan.evidence.rubric_id and plan.evidence.rubric_id != rubric_id:
        raise ValueError("approved pedagogy plan has a non-resolvable rubric binding")
    behavior = plan.evidence.observable_behavior or (
        "If a later, separately authorized opportunity is created, represent the target idea accurately "
        "in one accessible response mode; this plan itself collects nothing."
    )
    rubric_version = plan.evidence.rubric_version or 1
    register_rubric(
        db_path,
        RubricDefinition(
            rubric_id=rubric_id,
            version=rubric_version,
            label=f"{plan.evidence.template_id} observable response",
            criteria=[
                RubricCriterion(
                    criterion_id=f"criterion-{concept_digest}-{index}",
                    concept_id=concept.id,
                    concept_version=concept.version,
                    description=behavior,
                )
                for index, concept in enumerate(plan.concepts, start=1)
            ],
        ),
    )
    modality_map = {
        "plain_language": "spoken",
        "example": "mixed",
        "story": "spoken",
        "physical": "gesture_or_manipulative",
        "diagram": "drawn",
        "symbols": "symbolic",
        "photo": "selection",
    }
    modalities = list(
        dict.fromkeys(
            [modality_map[item] for item in plan.representations if item in modality_map]
            + list(plan.evidence.response_modes)
        )
    )
    register_learning_task(
        db_path,
        LearningTaskDefinition(
            task_id=plan.task_template.id,
            version=int(plan.task_template.version),
            title=f"{plan.primary_strategy.id}: " + ", ".join(item.label for item in plan.concepts),
            learning_goal=(
                "Use the reviewed strategy to build or express understanding of "
                + ", ".join(item.label for item in plan.concepts)
                + "."
            ),
            strategy=plan.primary_strategy.id,
            difficulty="unknown",
            difficulty_provenance="unknown",
            modalities=modalities or ["mixed"],
            context_key=f"pedagogy:{plan.domain_overlay.id}",
            representation_key="+".join(plan.representations),
            concepts=[
                LearnerConceptRef(concept_id=item.id, version=item.version, role="target") for item in plan.concepts
            ],
            rubric_id=rubric_id,
            rubric_version=rubric_version,
        ),
    )


_LIFE_COMPOUND_NOUN = (
    r"(?:insurance|science|cycle|jacket|skills?|story|stories|lesson|expectancy|span|forms?|support|"
    r"savings|policy|policies|preserver|vest|coach|coaching)"
)
_LIVING_COMPOUND_OR_LITERAL = (
    r"(?:rooms?|spaces?|arrangements?|conditions?|wages?|expenses?|standards?|history|things?|organisms?|"
    r"in\s+(?:(?:this|the|my|our|your)\s+)?(?:apartments?|houses?|homes?|rooms?|cities|towns?|countries|"
    r"neighborhoods?|buildings?)|with\s+(?:(?:my|our|your|the)\s+)?(?:famil(?:y|ies)|parents?|"
    r"roommates?|pets?)|at\s+(?:home|school)|as\s+(?:a|an|the)\b|like\s+(?:a|an|the)\b)"
)
_LIFE_TARGET = rf"life(?!\s+{_LIFE_COMPOUND_NOUN}\b)"
_LIVING_TARGET = rf"living(?!\s+{_LIVING_COMPOUND_OR_LITERAL}\b)"
_LIVE_BENIGN_COMPLEMENT = (
    r"(?:streams?|broadcasts?|music|shows?|events?)\b|"
    r"(?:in|on|at|near)\s+(?:(?:this|the|my|our|your|a|an)\s+)?"
    r"(?:apartments?|houses?|homes?|rooms?|cities|towns?|countries|neighborhoods?|buildings?|campus|school)\b|"
    r"with\s+(?:(?:my|our|your|the)\s+)?(?:famil(?:y|ies)|parents?|roommates?|pets?)\b|"
    r"(?:as|like)\s+(?:a|an|the)\b"
)
_LIVE_VERB_TARGET = rf"live(?!-(?:streams?|broadcasts?)\b|\s+(?:{_LIVE_BENIGN_COMPLEMENT}))"
_MY_LIFE_TARGET = rf"my\s+{_LIFE_TARGET}"
_MY_OPTIONAL_OWN_LIFE_TARGET = rf"my\s+(?:own\s+)?{_LIFE_TARGET}"
_SELF_HARM_UNAMBIGUOUS_ACTION_TARGET = (
    rf"(?:kill\s+myself|end\s+{_MY_LIFE_TARGET}|take\s+{_MY_OPTIONAL_OWN_LIFE_TARGET}|"
    rf"commit\s+suicide|self[- ]harm|overdose|take\s+(?:an?|the)\s+overdose|poison\s+myself|"
    rf"jump\s+off\s+(?:(?:the|a|my)\s+)?(?:roof|bridge|cliff)|"
    rf"hang\s+myself|drown\s+myself|shoot\s+myself|stab\s+myself|burn\s+myself|"
    rf"suffocate\s+myself|choke\s+myself|(?:slit|cut)\s+my\s+wrists?|die)"
)
_SELF_HARM_INJURY_ACTION_TARGET = (
    r"(?:(?:hurt|harm)\s+myself|cut\s+myself"
    r"(?!\s+(?:(?:out|off)\s+of|off\s+from|(?:free|loose)(?:\s+from)?)\b|"
    r"\s+(?:some\s+slack|a\s+(?:slice|piece))\b))"
)
_SELF_HARM_ACTION_TARGET = rf"(?:{_SELF_HARM_UNAMBIGUOUS_ACTION_TARGET}|{_SELF_HARM_INJURY_ACTION_TARGET})"
_SELF_HARM_PLURAL_ACTION_TARGET = (
    r"(?:kill\s+ourselves|end\s+our\s+lives|take\s+our\s+(?:own\s+)?lives|"
    r"commit\s+suicide|self[- ]harm|overdose|take\s+(?:an?|the)\s+overdose|"
    r"poison\s+ourselves|jump\s+off\s+(?:(?:the|a|our)\s+)?(?:roof|bridge|cliff)|"
    r"hang\s+ourselves|drown\s+ourselves|shoot\s+ourselves|stab\s+ourselves|"
    r"burn\s+ourselves|suffocate\s+ourselves|choke\s+ourselves|"
    r"(?:slit|cut)\s+our\s+wrists?|die)"
)
_SELF_HARM_DISCLOSURE_ACTION_TARGET = rf"(?:{_SELF_HARM_ACTION_TARGET}|{_SELF_HARM_PLURAL_ACTION_TARGET})"
_SELF_HARM_DISCLOSURE_GERUND_TARGET = (
    rf"(?:(?:killing|hurting|harming|cutting|stabbing|burning|hanging|drowning|shooting|"
    rf"suffocating|choking|poisoning)\s+(?:myself|ourselves)|"
    rf"ending\s+(?:my\s+{_LIFE_TARGET}|our\s+lives)|"
    rf"taking\s+(?:my\s+(?:own\s+)?{_LIFE_TARGET}|our\s+(?:own\s+)?lives)|"
    rf"committing\s+suicide|self[- ]harming|overdosing|"
    rf"taking\s+(?:an?|the)\s+overdose|"
    rf"jumping\s+off\s+(?:(?:the|a|my|our)\s+)?(?:roof|bridge|cliff)|"
    rf"(?:slitting|cutting)\s+(?:my|our)\s+wrists?|dying)"
)
_SELF_HARM_NOMINAL_TIME = (
    r"(?:(?:right\s+)?now|today|tonight|tomorrow|soon|immediately|later|"
    r"eventually|afterwards?|someday|"
    r"(?:as\s+of|first\s+thing)\s+(?:today|tonight|tomorrow|"
    r"(?:this|next)\s+(?:morning|afternoon|evening|night|day|week|weekend|month))|"
    r"later\s+(?:this|next)\s+(?:morning|afternoon|evening|night|day|week|weekend|month)|"
    r"(?:this|next)\s+(?:morning|afternoon|evening|night|day|week|weekend|month)|"
    r"(?:(?:(?:right|immediately|directly|straight|just|shortly)\s+)?(?:before|after|upon)|"
    r"by|during|following|upon|through)\s+"
    r"(?:(?:my|our|the|a|an|this|next|last)\s+)?"
    r"(?:[a-z0-9][\w'-]*\s*){1,4}|"
    r"(?:once|when|whenever|until|as\s+soon\s+as)\s+"
    r"(?:(?:i|we|school|class|practice|vacation|the\s+[a-z0-9][\w'-]*)\s+)?"
    r"(?:[a-z0-9][\w'-]*\s*){1,5}|"
    r"the\s+(?:moment|minute|instant)\s+(?:that\s+)?"
    r"(?:(?:i|we|school|class|practice)\s+)?(?:[a-z0-9][\w'-]*\s*){1,5}|"
    r"at\s+(?:the\s+)?end\s+of\s+(?:(?:my|our|the|this|next)\s+)?"
    r"(?:[a-z0-9][\w'-]*\s*){1,4}|"
    r"at\s+(?:midnight|noon|dawn|sunrise|sunset|\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?))|"
    r"at\s+(?:(?:my|our|the|this|next)\s+)?(?:[a-z0-9][\w'-]*\s*){1,4}|"
    r"on\s+(?:(?:my|our|the|this|next)\s+)?"
    r"(?:[a-z0-9][\w'-]*\s*){1,4}|"
    r"in\s+(?:a|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:minutes?|hours?|days?|weeks?)|"
    r"(?:starting|beginning|commencing)\s+"
    r"(?:(?:today|tonight|tomorrow|now)|"
    r"(?:(?:this|next)\s+)?(?:morning|afternoon|evening|night|day|week|weekend|month)|"
    r"(?:(?:my|our|the|a|an)\s+)?(?:[a-z0-9][\w'-]*\s*){1,4})|"
    r"from\s+(?:(?:today|tonight|tomorrow|now)|"
    r"(?:(?:this|next)\s+)?(?:morning|afternoon|evening|night|day|week|weekend|month)|"
    r"(?:(?:my|our|the|a|an)\s+)?(?:[a-z0-9][\w'-]*\s*){1,4})\s+"
    r"(?:on|onwards?|forwards?))"
)
_SELF_HARM_NOMINAL_OUTCOME = rf"(?:suicide(?!\s+prevention)|death(?=\s*(?:{_SELF_HARM_NOMINAL_TIME})?\s*(?:[.!?;]|$)))"
_SELF_HARM_METHOD_NOUN = (
    r"(?:plans?|ways?|steps?|methods?|instructions?|checklists?|procedures?|"
    r"strateg(?:y|ies)|outlines?|blueprints?|roadmaps?|means|pacts?|schemes?|designs?|"
    r"approaches?|protocols?|scripts?|lists?|notes?|courses?|agreements?|promises?|"
    r"commitments?|oaths?|vows?|deals?|decisions?|resolves?|arrangements?|how)"
)
_SELF_HARM_THOUGHT_TARGET = (
    rf"(?:suicide(?!\s+prevention)|(?:killing|hurting|harming)\s+myself|"
    rf"cutting\s+myself(?!\s+(?:(?:out|off)\s+of|off\s+from|(?:free|loose)(?:\s+from)?)\b)|"
    rf"ending\s+{_MY_LIFE_TARGET}|"
    rf"taking\s+{_MY_OPTIONAL_OWN_LIFE_TARGET}|self[- ]harm(?:ing)?|overdosing|"
    rf"taking\s+(?:an?|the)\s+overdose|"
    rf"poisoning\s+myself|jumping\s+off\s+(?:(?:the|a|my)\s+)?(?:roof|bridge|cliff)|"
    rf"hanging\s+myself|drowning\s+myself|shooting\s+myself|stabbing\s+myself|"
    rf"burning\s+myself|suffocating\s+myself|choking\s+myself|"
    rf"(?:slitting|cutting)\s+my\s+wrists?|dying)"
)
_SELF_HARM_PERSISTENT_THOUGHT_TARGET = rf"(?:{_SELF_HARM_THOUGHT_TARGET}|committing\s+suicide)"

_SELF_HARM_PLAN_MODIFIERS = (
    r"(?:(?:(?:[a-z]+ly|very)\s+)?"
    r"(?:detailed|complete|completed|specific|written|ready|finished|final)\s+)*"
)
_SELF_HARM_READINESS_MODIFIERS = r"(?:(?:[a-z]+ly|very|all)\s+)*"
_SELF_HARM_READINESS_TIME = (
    r"(?:\s+(?:right\s+now|right\s+this\s+(?:instant|minute|moment|second)|"
    r"at\s+this\s+(?:instant|minute|moment|second)|this\s+(?:instant|minute|moment|second)|"
    r"immediately|now|today|tonight))?"
)


# Bind a first-person commitment carrier to its harmful outcome instead of
# relying on a closed list of sentence templates.  Negative complements do not
# match ("no intention", "expect not to"), while nominal and verbal forms share
# the same action target and time boundary.
_SELF_HARM_ACTIVE_NOMINAL_MODIFIERS = (
    r"(?:(?!(?:no|not|zero|without|former|past|abandoned|discarded|rejected|"
    r"fictional|pretend|imaginary|literary|narrative|story)\b)"
    r"[a-z][\w-]*\s+){0,3}"
)
_SELF_HARM_FIRST_PERSON_STRUCTURAL_INTENT_SIGNAL = re.compile(
    rf"\b(?:"
    rf"(?:my|our)\s+"
    rf"{_SELF_HARM_ACTIVE_NOMINAL_MODIFIERS}"
    rf"(?:goal|aim|objective|purpose|ambition|wish|desire|urge)\s+"
    rf"(?:for\s+(?:{_SELF_HARM_NOMINAL_TIME})\s+)?"
    rf"(?:is|was|remains?|became|has\s+become|"
    rf"(?:has|had)\s+(?:(?:always|long|still|already|formerly|previously|"
    rf"for\s+(?:years?|months?|weeks?|days?))\s+)*been)\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|(?:my\s+)?{_SELF_HARM_NOMINAL_OUTCOME})|"
    rf"it\s+(?:(?:is|was)(?:\s+(?:still|now|currently|clearly|truly|really))*|"
    rf"(?:has|had)(?:\s+(?:always|long|still|already|formerly|previously))*\s+been|"
    rf"remains?|(?:has|had)\s+remained|became|has\s+become|had\s+become|"
    rf"continues?\s+to\s+be|(?:has|had)\s+continued\s+to\s+be|continued\s+to\s+be)\s+"
    rf"(?:my|our)\s+"
    rf"{_SELF_HARM_ACTIVE_NOMINAL_MODIFIERS}"
    rf"(?:goal|aim|objective|purpose|ambition|wish|desire|urge)\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|(?:is|was)\s+to\s+{_SELF_HARM_ACTION_TARGET})|"
    rf"i\s+(?:have|hold|carry|possess)\s+"
    rf"(?:(?:a|an|the|my)\s+)?"
    rf"(?:(?!(?:no|not|zero|without|fictional|pretend)\b)[a-z]+\s+){{0,3}}"
    rf"(?:intent|intention|aim|plan|expectation)\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|of\s+{_SELF_HARM_DISCLOSURE_GERUND_TARGET})|"
    rf"i\s+(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*"
    rf"(?:aim|aspire|desire|expect|hope|long|plan|intend|mean|prepare|wish|yearn)\s+"
    rf"(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+have|'ve|\s+had|'d)\s+)"
    rf"(?:(?:always|long|still|already|previously|formerly)\s+)*"
    rf"(?:longed|yearned|hoped|wished|desired)\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:(?:keep|kept)(?:\s+on)?|continue|continued)\s+"
    rf"(?:longing|yearning|hoping|wishing|desiring)\s+"
    rf"to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+have|'ve|\s+had|'d)\s+)"
    rf"(?:kept(?:\s+on)?|continued)\s+"
    rf"(?:longing|yearning|hoping|wishing|desiring)\s+"
    rf"to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+am|'m)|\s+was|(?:\s+have|'ve|\s+had|'d)\s+been)\s+"
    rf"(?:(?:[a-z]+ly|still|currently|persistently|very(?:\s+much)?)\s+)*"
    rf"(?:longing|yearning|aspiring|hoping|wishing)\s+"
    rf"(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:(?:keep|kept)\s+(?:finding|noticing|catching)|"
    rf"find|found|notice|noticed|catch|caught)\s+myself\s+"
    rf"(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*"
    rf"(?:longing|yearning|aspiring|hoping|wishing)\s+"
    rf"(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:(?:have|had)\s+had|(?:am|was|have\s+been|had\s+been)\s+experiencing|"
    rf"feel|felt|have|had|experience|experienced|harbor|harbored|carry|carried|live\s+with)\s+"
    rf"(?:(?:an?|the)\s+)?"
    rf"(?:(?:deep|strong|intense|persistent|constant|overwhelming)\s+)*"
    rf"(?:longing|yearning|desire|wish|urge|impulse|temptation)\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:\s+cannot|\s+can't|\s+can\s+not)\s+shake\s+"
    rf"(?:(?:an?|the|my)\s+)?"
    rf"(?:(?:deep|strong|intense|persistent|constant|overwhelming)\s+)*"
    rf"(?:longing|yearning|desire|wish|urge|impulse|temptation)\s+"
    rf"to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:wake|woke|awoke)(?:\s+up|\s+from\s+sleep)?\s+"
    rf"(?:(?:[a-z]+ly|still|often|always)\s+)*"
    rf"(?:longing|yearning|hoping|wishing|desiring)\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+am|'m|\s+was|(?:\s+have|'ve|\s+had|'d)\s+been)\s+)"
    rf"consumed\s+by\s+(?:(?:an?|the|my)\s+)?"
    rf"(?:(?:deep|strong|intense|persistent|constant|overwhelming)\s+)*"
    rf"(?:longing|yearning|desire|wish|urge|impulse|temptation)\s+"
    rf"to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+am|'m|\s+was|\s+remain|\s+remained)\s+|"
    rf"(?:\s+feel|\s+felt|(?:\s+have|'ve|\s+had|'d)\s+felt)\s+)"
    rf"(?:drawn|compelled|tempted|inclined)\s+to\s+"
    rf"(?:{_SELF_HARM_ACTION_TARGET}|{_SELF_HARM_DISCLOSURE_GERUND_TARGET})|"
    rf"i(?:'d|\s+would)\s+(?:(?:really|very\s+much)\s+)*"
    rf"(?:like|prefer)\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:want|wish|need|prefer)\s+(?:my|our)\s+"
    rf"(?:life|lives)\s+to\s+(?:end|stop)(?=\s*(?:{_SELF_HARM_NOMINAL_TIME})?\s*(?:[.!?;]|$))|"
    rf"i(?:(?:\s+have|'ve))?\s+(?:scheduled|booked|set)\s+"
    rf"(?:(?:a|the|my|my\s+own)\s+)?(?:suicide|death)"
    rf"(?:\s+(?:for|on|at)\s+(?:{_SELF_HARM_NOMINAL_TIME}))?"
    rf"(?=\s*(?:[.!?;]|$))"
    rf")\b"
)


_SELF_HARM_EXPLICIT_INTENT_SIGNAL = re.compile(
    rf"\b(?:"
    rf"i(?:(?:\s+am|'m)\s+about\s+to|\s+will\s+soon)\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+will\s+{_SELF_HARM_UNAMBIGUOUS_ACTION_TARGET}|"
    rf"i\s+(?:(?:fully|firmly|definitely|clearly|absolutely)\s+)*"
    rf"(?:want|plan|intend|need)\s+"
    rf"(?:(?:fully|firmly|definitely|clearly|absolutely)\s+)*"
    rf"to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+have|'ve)(?:\s+(?:got|made|written|finished|completed|finalized))?|"
    rf"\s+(?:made|wrote|finished|completed|finalized))\s+"
    rf"(?:(?:a|an|the|my)\s+)?"
    rf"{_SELF_HARM_PLAN_MODIFIERS}"
    rf"(?:plans?\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"suicide\s+plans?|plans?\s+for\s+(?:my\s+)?suicide)|"
    rf"(?:my|the)\s+{_SELF_HARM_PLAN_MODIFIERS}plans?\s+(?:is|was)\s+to\s+"
    rf"{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:prepared|created|drafted)\s+(?:(?:a|the|my)\s+)?"
    rf"{_SELF_HARM_PLAN_MODIFIERS}plans?\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:wrote|jotted)\s+down\s+(?:(?:a|the|my)\s+)?plans?\s+to\s+"
    rf"{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+have|'ve))?\s+(?:picked|chose|set)\s+"
    rf"(?:(?:a|the|my)\s+)?(?:date|day|time)\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|for\s+(?:my\s+)?suicide)|"
    rf"i(?:(?:\s+am|'m)\s+planning|(?:\s+have|'ve)(?:\s+been)?\s+planning)\s+to\s+"
    rf"{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+am|'m)\s+planning|(?:\s+have|'ve)(?:\s+been)?\s+planning)\s+"
    rf"my\s+death(?=\s+(?:for\s+)?(?:{_SELF_HARM_NOMINAL_TIME})\s*(?:[.!?;]|$))|"
    rf"i(?:(?:\s+am|'m)\s+planning|(?:\s+have|'ve)\s+been\s+planning|\s+plan(?:ned)?)\s+"
    rf"(?:my\s+)?suicide|"
    rf"i(?:\s+have|'ve)\s+every\s+intention\s+of\s+"
    rf"{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|"
    rf"(?:my|our)\s+(?:(?:full|firm|clear|settled|definite)\s+)?"
    rf"(?:intent|intention|plan)\s+(?:is|was|remains?)\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|"
    rf"(?:my\s+)?{_SELF_HARM_NOMINAL_OUTCOME})|"
    rf"it\s+(?:is|was)\s+(?:my|our)\s+"
    rf"(?:(?:full|firm|clear|settled|definite)\s+)?"
    rf"(?:intent|intention|plan)\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|"
    rf"(?:my\s+)?{_SELF_HARM_NOMINAL_OUTCOME})|"
    rf"i(?:(?:\s+am|'m)\s+(?:preparing|prepared|getting\s+ready)|"
    rf"(?:\s+have|'ve)\s+(?:chosen|prepared)|\s+chose)\s+to\s+"
    rf"{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+am|'m|\s+was)\s+prepar(?:ing|ed))\s+for\s+"
    rf"(?:my\s+)?{_SELF_HARM_NOMINAL_OUTCOME}|"
    rf"i(?:(?:\s+am|'m)\s+{_SELF_HARM_READINESS_MODIFIERS}(?:ready|set|prepared)"
    rf"{_SELF_HARM_READINESS_TIME}|"
    rf"(?:\s+have|'ve)(?:\s+got)?\s+(?:everything|it|things?)\s+"
    rf"{_SELF_HARM_READINESS_MODIFIERS}(?:ready|set|prepared){_SELF_HARM_READINESS_TIME})\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|for\s+(?:my\s+)?suicide)|"
    rf"(?:everything|it|things?)\s+(?:is|are|was|were|feels?|seems?)\s+"
    rf"{_SELF_HARM_READINESS_MODIFIERS}(?:ready|set|prepared){_SELF_HARM_READINESS_TIME}\s+"
    rf"for\s+(?:my\s+)?suicide|"
    rf"(?:my|the)\s+suicide\s+plans?\s+(?:is|was|feels?|seems?)\s+"
    rf"{_SELF_HARM_READINESS_MODIFIERS}(?:ready|complete|finished)|"
    rf"i(?:(?:\s+am|'m)\s+)?{_SELF_HARM_READINESS_MODIFIERS}ready"
    rf"{_SELF_HARM_READINESS_TIME}\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+am|'m)\s+planning|\s+plan)\s+on\s+{_SELF_HARM_PERSISTENT_THOUGHT_TARGET}|"
    rf"i(?:(?:\s+am|'m)\s+feeling|\s+feel)\s+like\s+{_SELF_HARM_THOUGHT_TARGET}|"
    rf"i(?:(?:\s+am|'m)\s+)?tempted\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+am|'m)\s+having|(?:\s+have|'ve)(?:\s+got)?)\s+"
    rf"(?:(?:a|an|the)\s+)?urges?\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:\s+have|'ve)\s+(?:decided|resolved|chosen|selected|picked|elected)\s+to\s+"
    rf"{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:choose|chose|mean|decide|decided|determine|determined|resolve|resolved|"
    rf"select|selected|pick|picked|elect|elected)"
    rf"\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+have|'ve))?\s+opted\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|(?:on|for)\s+(?:my\s+)?{_SELF_HARM_NOMINAL_OUTCOME})|"
    rf"i\s+(?:(?:fully|firmly|definitely|clearly)\s+)?intend\s+on\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my\s+)?suicide)|"
    rf"i(?:(?:\s+have|'ve)\s+(?:chosen|decided|resolved|settled|selected|picked|elected)|"
    rf"\s+(?:choose|chose|decide|decided|resolve|resolved|settle|settled|"
    rf"select|selected|pick|picked|elect|elected))\s+(?:(?:on|for)\s+)?"
    rf"(?:my\s+)?{_SELF_HARM_NOMINAL_OUTCOME}|"
    rf"i(?:(?:\s+have|'ve))?\s+made\s+(?:(?:a|the|my)\s+)?"
    rf"(?:choice|decision|commitment|resolve)\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|(?:on|for)\s+(?:my\s+)?{_SELF_HARM_NOMINAL_OUTCOME})|"
    rf"i(?:(?:\s+am|'m|\s+was|\s+remain|\s+remained|\s+became))\s+"
    rf"(?:(?:fully|firmly|completely|still)\s+)?intent\s+(?:on|upon)\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my\s+)?suicide|"
    rf"{_SELF_HARM_NOMINAL_OUTCOME})|"
    rf"my\s+(?:choice|decision|resolve|intention|intent|commitment)\s+"
    rf"(?:is|was|remains?|became)\s+"
    rf"(?:to\s+{_SELF_HARM_ACTION_TARGET}|(?:my\s+)?{_SELF_HARM_NOMINAL_OUTCOME})|"
    rf"i(?:\s+have|'ve)?\s*made\s+up\s+my\s+mind\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:settled|decided|resolved)\s+on\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my\s+)?suicide)|"
    rf"i(?:(?:\s+am|'m|\s+was|\s+remain|\s+remained|\s+stay|\s+stayed|\s+became))\s+"
    rf"(?:(?:[a-z]+ly|very|still)\s+)?committed\s+to\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my\s+)?suicide|"
    rf"death(?=\s*(?:{_SELF_HARM_NOMINAL_TIME})?\s*(?:[.!?;]|$)))|"
    rf"i(?:(?:\s+am|'m|\s+was|\s+remain|\s+remained|\s+stay|\s+stayed|\s+became))\s+"
    rf"(?:(?:[a-z]+ly|very|still)\s+)?"
    rf"(?:resolved|determined|devoted|dedicated|pledged|fixed|set)\s+(?:on|upon|to|for)\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my\s+)?suicide|"
    rf"death(?=\s*(?:{_SELF_HARM_NOMINAL_TIME})?\s*(?:[.!?;]|$)))|"
    rf"i(?:\s+have|'ve)\s+(?:committed|resolved|determined)\s+to\s+"
    rf"(?:{_SELF_HARM_ACTION_TARGET}|{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my\s+)?suicide)|"
    rf"i(?:(?:\s+have|'ve))?\s+"
    rf"(?:commit(?:ted)?|pledge(?:d)?|dedicat(?:e|ed)|devot(?:e|ed)|vow(?:ed)?)\s+myself\s+to\s+"
    rf"(?:{_SELF_HARM_ACTION_TARGET}|{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|"
    rf"(?:my\s+)?suicide(?!\s+prevention)|"
    rf"death(?=\s*(?:{_SELF_HARM_NOMINAL_TIME})?\s*(?:[.!?;]|$)))|"
    rf"i(?:(?:\s+am|'m|\s+was|\s+remain|\s+remained|\s+became))\s+"
    rf"(?:(?:fully|firmly|completely|still)\s+)?dead\s+set\s+on\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my\s+)?suicide)|"
    rf"i(?:(?:\s+am|'m|\s+was|\s+remain|\s+remained|\s+became))\s+"
    rf"(?:(?:fully|firmly|completely|still)\s+)?(?:determined|resolved|committed)\s+"
    rf"to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+am|'m|\s+was)\s+going|(?:\s+have|'ve)\s+gone|\s+went)\s+"
    rf"through\s+with\s+(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my\s+)?suicide)|"
    rf"i\s+(?:will|would|plan|intend|mean|want)\s+(?:to\s+)?"
    rf"(?:go|follow)\s+through\s+(?:with|on)\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my\s+)?suicide)|"
    rf"i\s+(?:swore|vowed|promised|pledged)\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i(?:(?:\s+am|'m|\s+was)|(?:\s+have|'ve|\s+had)\s+been)\s+"
    rf"(?:thinking\s+(?:about|of)|(?:seriously\s+)?(?:considering|contemplating)|feeling\s+like)\s+"
    rf"{_SELF_HARM_PERSISTENT_THOUGHT_TARGET}|"
    rf"i\s+(?:(?:cannot|can't|can\s+not)\s+stop|keep)\s+thinking\s+(?:about|of)\s+"
    rf"{_SELF_HARM_PERSISTENT_THOUGHT_TARGET}|"
    rf"i\s+(?:cannot|can't|can\s+not)\s+stop\s+having\s+thoughts?\s+(?:about|of)\s+"
    rf"{_SELF_HARM_PERSISTENT_THOUGHT_TARGET}|"
    rf"i\s+(?:(?:cannot|can't|can\s+not)\s+get\s+"
    rf"(?:suicidal\s+thoughts?|thoughts?\s+(?:about|of)\s+{_SELF_HARM_PERSISTENT_THOUGHT_TARGET})"
    rf"\s+out\s+of\s+my\s+(?:head|mind)|"
    rf"(?:keep\s+having|have(?:\s+been)?\s+having|have)\s+thoughts?\s+(?:about|of)\s+"
    rf"{_SELF_HARM_PERSISTENT_THOUGHT_TARGET})|"
    rf"(?:suicidal\s+thoughts?|thoughts?\s+(?:about|of)\s+{_SELF_HARM_PERSISTENT_THOUGHT_TARGET})\s+"
    rf"(?:keep|kept)\s+coming\s+back|"
    rf"i\s+(?:tried\s+to|attempted\s+to|remember\s+trying\s+to|used\s+to\s+want\s+to)\s*"
    rf"{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:tried|attempted)\s+{_SELF_HARM_PERSISTENT_THOUGHT_TARGET}|"
    rf"i\s+attempted\s+(?:an?\s+)?overdose|"
    rf"i\s+(?:cut|slit)\s+my\s+wrists?\s+(?:on\s+purpose|intentionally)|"
    rf"i\s+(?:considered|contemplated|thought\s+(?:about|of))\s+"
    rf"{_SELF_HARM_PERSISTENT_THOUGHT_TARGET}|"
    rf"i\s+(?:remember(?:ed)?|recall(?:ed)?)\s+(?:wanting\s+to|trying\s+to)\s+"
    rf"{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:planned|had\s+planned)\s+to\s+{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:was|felt)\s+suicidal|"
    rf"i(?:\s+have|'ve)\s+been\s+(?:feeling\s+)?suicidal|"
    rf"i\s+(?:remember(?:ed)?|recall(?:ed)?)\s+being\s+suicidal|"
    rf"i\s+(?:remember(?:ed)?|recall(?:ed)?)\s+having\s+suicidal\s+thoughts?|"
    rf"i\s+had\s+suicidal\s+thoughts?|"
    rf"i\s+attempted\s+suicide|"
    rf"i(?:(?:\s+am|'m)\s+having|(?:\s+have|'ve)(?:\s+been\s+having)?)\s+"
    rf"(?:suicidal\s+thoughts?|thoughts?\s+(?:about|of)\s+suicide)|"
    rf"suicide\s+(?:is|feels?\s+like)\s+(?:my|the)\s+only\s+(?:option|way|answer|choice)"
    rf")\b"
)

# The identity of a plan matters more than the particular verb used to create
# it. Bind first-person ownership to the plan's actual self-harm complement,
# rather than mere co-occurrence with words such as "plan" and "suicide". The
# complete-clause boundary both covers novel construction verbs and keeps
# prevention/lesson/safety plans out of emergency routing.
_SELF_HARM_PLAN_DISCLOSURE_SIGNAL = re.compile(
    rf"^(?:"
    rf"(?=[^.!?;\n]*\b(?:i|i've|we|we've|my|our)\b)"
    rf"[^.!?;\n]{{0,96}}\b{_SELF_HARM_METHOD_NOUN}\b"
    rf"[^.!?;\n]{{0,48}}(?:"
    rf"\b(?:is|was|are|were)?\s*to\s+{_SELF_HARM_DISCLOSURE_ACTION_TARGET}|"
    rf"\b(?:for|about|of)\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my|our)\s+"
    rf"(?:suicide|death(?:\s+(?:{_SELF_HARM_NOMINAL_TIME}))?))|"
    rf"\b(?:involves?|includes?|describes?|outlines?)\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:my|our)\s+suicide)|"
    rf"\b(?:where|whereby|in\s+which)\s+(?:i|we)\s+"
    rf"(?:(?:should|must|will|would|could|might|may|plan|intend|want)\s+)?"
    rf"{_SELF_HARM_DISCLOSURE_ACTION_TARGET}(?:\s+[^.!?;\n]{{0,64}})?|"
    rf"\b(?:is|was|are|were)\s+that\s+(?:i|we)\s+"
    rf"(?:(?:should|must|will|would|could|might|may|plan|intend|want)\s+)?"
    rf"{_SELF_HARM_DISCLOSURE_ACTION_TARGET}|"
    rf"\s*[:=\-]\s*{_SELF_HARM_DISCLOSURE_ACTION_TARGET})"
    rf"|(?=[^.!?;\n]*\b(?:i|i've|my|we|we've|our)\b)"
    rf"[^.!?;\n]{{0,96}}\b(?:my|our)\s+{_SELF_HARM_PLAN_MODIFIERS}"
    rf"suicide(?!\s+prevention)\s+(?:{_SELF_HARM_METHOD_NOUN})\b"
    rf"(?:\s+(?:is|are|was|were|feels?|seems?)\s+"
    rf"{_SELF_HARM_READINESS_MODIFIERS}(?:ready|complete|completed|finished|final))?"
    rf")\s*[.!?]?$"
)

_SELF_HARM_PLURAL_INTENT_SIGNAL = re.compile(
    rf"\bwe(?:"
    rf"(?:'re|\s+are)\s+(?:(?:going|planning|preparing)\s+to|"
    rf"(?:(?:all|completely|fully)\s+)?(?:ready|set|about|prepared)\s+to)|"
    rf"\s+(?:want|plan|intend|need)\s+to|"
    rf"\s+(?:will|might|may|could|would)\s+|"
    rf"(?:'ve|\s+have)\s+"
    rf"(?:decided|determined|planned|agreed|chosen|resolved|selected|picked)\s+to|"
    rf"\s+(?:decided|determined|planned|agreed|chose|resolved|selected|picked)\s+to"
    rf")\s*{_SELF_HARM_PLURAL_ACTION_TARGET}\b|"
    rf"\bwe\s+agreed\s+that\s+we\s+(?:should|must|will|would|could|might|may)\s+"
    rf"{_SELF_HARM_PLURAL_ACTION_TARGET}\b|"
    rf"\bwe\s+(?:made|formed|entered(?:\s+into)?|have)\s+"
    rf"(?:(?:a|the|our)\s+)?suicide\s+pact\b|"
    rf"\bwe\s+(?:made|gave|swore|formed)\s+(?:(?:a|an|the|our)\s+)?"
    rf"(?:promise|oath|vow|deal|agreement|commitment|pact)\s+to\s+"
    rf"{_SELF_HARM_PLURAL_ACTION_TARGET}\b|"
    rf"\bwe(?:(?:'ve|\s+have))?\s+made\s+(?:(?:a|the|our)\s+)?"
    rf"(?:choice|decision|commitment|resolve)\s+"
    rf"(?:to\s+{_SELF_HARM_PLURAL_ACTION_TARGET}|(?:on|for)\s+(?:our\s+)?"
    rf"{_SELF_HARM_NOMINAL_OUTCOME})\b|"
    rf"\bour\s+(?:promise|oath|vow|deal|agreement|commitment|pact)\s+"
    rf"(?:is|was)\s+(?:to\s+|that\s+we\s+"
    rf"(?:should|must|will|would|could|might|may)\s+)"
    rf"{_SELF_HARM_PLURAL_ACTION_TARGET}\b|"
    rf"\bwe(?:(?:'ve|\s+have)\s+"
    rf"(?:chosen|decided|resolved|settled|opted|selected|picked|elected)|"
    rf"\s+(?:choose|chose|decide|decided|resolve|resolved|settle|settled|"
    rf"opt|opted|select|selected|pick|picked|elect|elected))\s+"
    rf"(?:(?:on|for)\s+)?(?:our\s+)?{_SELF_HARM_NOMINAL_OUTCOME}\b|"
    rf"\bwe(?:(?:'re|\s+are|\s+were|\s+remain|\s+remained|\s+became))\s+"
    rf"(?:(?:fully|firmly|completely|still)\s+)?(?:committed|intent)\s+(?:to|on)\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:our\s+)?suicide|"
    rf"death(?=\s*(?:{_SELF_HARM_NOMINAL_TIME})?\s*(?:[.!?;]|$)))\b|"
    rf"\bwe(?:'ve|\s+have)\s+(?:committed|resolved|determined)\s+to\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:our\s+)?suicide|"
    rf"death(?=\s*(?:{_SELF_HARM_NOMINAL_TIME})?\s*(?:[.!?;]|$)))\b|"
    rf"\bwe(?:(?:'ve|\s+have))?\s+"
    rf"(?:commit(?:ted)?|pledge(?:d)?|dedicat(?:e|ed)|devot(?:e|ed)|vow(?:ed)?)\s+ourselves\s+to\s+"
    rf"(?:{_SELF_HARM_PLURAL_ACTION_TARGET}|{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|"
    rf"(?:our\s+)?suicide(?!\s+prevention)|"
    rf"death(?=\s*(?:{_SELF_HARM_NOMINAL_TIME})?\s*(?:[.!?;]|$)))\b|"
    rf"\bour\s+(?:choice|decision|resolve|intention|intent|commitment)\s+"
    rf"(?:is|was|remains?|became)\s+"
    rf"(?:to\s+{_SELF_HARM_PLURAL_ACTION_TARGET}|(?:our\s+)?{_SELF_HARM_NOMINAL_OUTCOME})\b|"
    rf"\bwe(?:(?:'re|\s+are|\s+were)\s+going|(?:'ve|\s+have)\s+gone|\s+went)\s+"
    rf"through\s+with\s+(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:our\s+)?suicide)\b|"
    rf"\bwe\s+(?:will|would|plan|intend|mean|want)\s+(?:to\s+)?"
    rf"(?:go|follow)\s+through\s+(?:with|on)\s+"
    rf"(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|(?:our\s+)?suicide)\b"
)


_SELF_HARM_SIGNAL = re.compile(
    r"\b(?:"
    r"i(?:(?:\s+am|'m)\s+(?:going|planning)\s+to\s+)"
    rf"{_SELF_HARM_ACTION_TARGET}|"
    rf"i\s+(?:might|may|could|would)\s+{_SELF_HARM_UNAMBIGUOUS_ACTION_TARGET}|"
    r"i(?:(?:\s+am|'m)\s+|\s+feel\s+)suicidal|"
    r"i(?:\s+have|'ve)\s+suicidal\s+thoughts?|"
    r"i(?:(?:\s+am|'m)\s+(?:thinking\s+(?:about|of)|feeling\s+like|considering)|"
    r"\s+(?:think\s+(?:about|of)|feel\s+like))\s+"
    rf"(?:suicide(?!\s+prevention)|killing\s+myself|ending\s+{_MY_LIFE_TARGET}|dying)|"
    rf"i(?:\s+have|'ve)\s+decided\s+to\s+(?:die|kill\s+myself|end\s+{_MY_LIFE_TARGET})|"
    r"(?:maybe\s+)?i\s+(?:think\s+i\s+)?should\s+(?:die|be\s+dead|kill\s+myself)|"
    r"suicide\s+(?:is|feels?\s+like)\s+(?:my|the)\s+only\s+(?:option|way|answer)|"
    r"i\s+(?:want|wish|hope|plan|intend)\b"
    r"(?!\s+(?:(?:the|a|an|my|our|your)\b|to\s+(?:know|learn|understand|study)\b))"
    rf".{{0,24}}\b(?:die|dead|end(?:ing)?\s+{_MY_LIFE_TARGET}|"
    rf"kill(?:ing)?\s+myself|commit\s+suicide|take\s+{_MY_OPTIONAL_OWN_LIFE_TARGET}|"
    rf"stop\s+(?:{_LIVING_TARGET}|being\s+alive|breathing)|"
    r"(?:never|not|don't|do\s+not)\s+wake\s+up(?!\s+(?:early|late|at)\b)|"
    r"(?:fall\s+asleep|sleep)\s+forever)|"
    r"(?:go|going)\s+to\s+sleep\b.{0,24}\b(?:never|not)\s+wake\s+up|"
    r"(?:do\s+not|don't|never)\s+want\s+to\s+wake\s+up|"
    r"wish\b.{0,20}\bnever\s+(?:been|was)\s+born|"
    r"wish\b.{0,20}\b(?:(?:was|were|am)\s+not|wasn't|weren't)\s+alive|"
    r"(?:everyone|my\s+family|you|they)\b.{0,24}\bbetter\s+(?:off\s+)?without\s+me|"
    rf"(?:there\s+is\s+)?no\s+point\b.{{0,20}}\b(?:being\s+alive|{_LIVING_TARGET}|going\s+on)|"
    r"(?:i\s+)?(?:cannot|can't|can\s+not)\s+go\s+on\s+(?:anymore|any\s+longer)|"
    r"(?:i\s+)?(?:should\s+not|shouldn't|do\s+not|don't)\s+(?:exist|be\s+alive)|"
    rf"(?:do\s+not|don't)\s+want\s+to\s+(?:be\s+)?(?:alive|{_LIVE_VERB_TARGET}|keep\s+living|"
    r"breathe(?:\s+(?:anymore|any\s+longer))?|exist(?:\s+(?:anymore|any\s+longer))?|"
    r"(?:go\s+on|keep\s+going)(?:\s+(?:anymore|any\s+longer))?)|"
    r"wish\b.{0,20}\b(?:did|could|would)\s+not\s+exist|"
    r"(?:cannot|can't|can\s+not)\s+(?:keep\s+living|live\s+(?:anymore|any\s+longer))|"
    r"(?:cannot|can't|can\s+not)\s+(?:go\s+on|keep\s+going)(?:\s+(?:anymore|any\s+longer))?|"
    r"(?:(?:do\s+not|don't)\s+(?:think|know\s+(?:if|whether))|(?:am\s+)?not\s+sure)\s+"
    r"i\s+can\s+(?:go\s+on|keep\s+going)|"
    r"(?:cannot|can't|can\s+not)\s+take\s+(?:this|it)\s+(?:anymore|any\s+longer)|"
    r"(?:do\s+not|don't)\s+care\s+(?:if|whether)\s+i\s+(?:die|live)|"
    r"(?:it\s+)?(?:does\s+not|doesn't)\s+matter\s+(?:if|whether)\s+i\s+(?:die|live)|"
    r"would\s+not\s+mind\s+dying|"
    r"want\s+to\s+be\s+gone\s+forever|"
    r"(?:rather|would\s+rather)\s+be\s+dead|"
    rf"no\s+longer\s+want\s+to\s+{_LIVE_VERB_TARGET}|"
    r"(?:do\s+not|don't)\s+think\s+i\s+should\s+be\s+alive|"
    rf"(?:i\s+)?(?:have|i've)\s+lost\s+(?:(?:my|the)\s+|all\s+)?"
    rf"(?:will|desire)\s+to\s+{_LIVE_VERB_TARGET}|"
    r"i\s+(?:(?:do\s+not|don't)\s+have\s+(?:a|any)|have\s+no)\s+"
    r"(?:reason|desire|will)\s+to\s+"
    rf"(?:stay\s+alive|{_LIVE_VERB_TARGET}|go\s+on)|"
    r"i\s+(?:just\s+)?want\s+it\s+all\s+to\s+end|"
    rf"(?:do\s+not|don't)\s+want\s+to\s+go\s+on\s+{_LIVING_TARGET}|"
    r"i\s+(?:am\s+|'m\s+)?self[- ]harm(?:ing)?"
    r")\b"
)

_SELF_HARM_MODAL_INJURY_SIGNAL = re.compile(
    rf"^(?:maybe\s+)?i\s+(?:might|may|could|would|will)\s+{_SELF_HARM_INJURY_ACTION_TARGET}"
    r"(?:\s+(?:soon|tonight|right\s+now))?\s*[.!?]?$"
)

_LIFE_MEANING_TARGET = (
    rf"(?:(?:my\s+)?{_LIFE_TARGET}|be(?:ing)?\s+alive|stay(?:ing)?\s+alive|{_LIVE_VERB_TARGET}|"
    rf"(?:keep(?:ing)?\s+)?{_LIVING_TARGET}|"
    r"(?:keep(?:ing)?\s+)?go(?:ing)?\s+on"
    r"(?!\s+with\s+(?:this|the|my|our|your)\b))"
)
_PERSONAL_LIFE_MEANING_TARGET = (
    rf"(?:my\s+{_LIFE_TARGET}|be(?:ing)?\s+alive|stay(?:ing)?\s+alive|"
    rf"(?:keep(?:ing)?\s+)?{_LIVE_VERB_TARGET}|(?:keep(?:ing)?\s+)?{_LIVING_TARGET}|"
    rf"(?:keep(?:ing)?\s+)?go(?:ing)?\s+on"
    r"(?!\s+with\s+(?:this|the|my|our|your)\b))"
)
_SELF_HARM_MEANINGLESSNESS_SIGNAL = re.compile(
    rf"\b(?:"
    rf"(?:i\s+)?(?:do\s+not|don't|cannot|can't)\s+(?:see|find|feel)\s+"
    rf"(?:(?:any|a|the|much)\s+)?(?:point|reason|meaning)\s+(?:in|to|for)?\s*{_LIFE_MEANING_TARGET}|"
    rf"(?:i\s+)?(?:see|find|feel)\s+(?:no|little)\s+(?:point|reason|meaning)\s+"
    rf"(?:in|to|for)?\s*{_LIFE_MEANING_TARGET}|"
    rf"(?:there\s+(?:is|seems\s+to\s+be)|there's)\s+no\s+(?:point|reason|meaning)\s+"
    rf"(?:in|to|for)?\s*{_LIFE_MEANING_TARGET}|"
    rf"(?:there\s+(?:isn't|is\s+not)|there\s+(?:doesn't|does\s+not)\s+seem\s+to\s+be)\s+"
    rf"(?:(?:any|a|much)\s+)?(?:point|reason|meaning)\s+(?:in|to|for)?\s*{_LIFE_MEANING_TARGET}|"
    rf"(?:i\s+)?(?:do\s+not|don't)\s+(?:think|believe)\s+"
    rf"(?:there\s+(?:is|isn't|is\s+not)\s+)?(?:(?:any|a|no|much)\s+)"
    rf"(?:point|reason|meaning)\s+(?:in|to|for)?\s*{_LIFE_MEANING_TARGET}|"
    rf"(?:i\s+)?(?:do\s+not|don't|cannot|can't)\s+(?:see|understand)\s+why\s+"
    rf"i\s+should\s+{_PERSONAL_LIFE_MEANING_TARGET}|"
    rf"why\s+(?:should|would)\s+i\s+{_PERSONAL_LIFE_MEANING_TARGET}|"
    rf"what(?:'s|\s+is)\s+the\s+(?:point|reason|meaning)\s+(?:in|to|of|for)?\s*"
    rf"{_PERSONAL_LIFE_MEANING_TARGET}|"
    rf"(?:my\s+)?{_LIFE_TARGET}\s+(?:is|feels?|seems?)\s+(?:completely\s+)?(?:meaningless|pointless)|"
    rf"(?:being\s+alive|{_LIVING_TARGET})\s+(?:is|feels?|seems?)\s+(?:completely\s+)?"
    rf"(?:meaningless|pointless)|"
    rf"(?:i\s+)?(?:do\s+not|don't)\s+(?:think|believe)\s+(?:my\s+)?{_LIFE_TARGET}\s+has\s+"
    rf"(?:any\s+|a\s+)?(?:purpose|meaning|point)|"
    rf"(?:my\s+)?{_LIFE_TARGET}\s+(?:(?:has|seems?\s+to\s+have)\s+no|"
    rf"(?:does\s+not|doesn't)\s+have\s+(?:any|a))\s+(?:purpose|meaning|point)|"
    rf"(?<!word\s)(?:my\s+)?{_LIFE_TARGET}\s+has\s+lost\s+(?:all|its)\s+(?:purpose|meaning)|"
    rf"i\s+(?:do\s+not|don't)\s+have\s+(?:(?:any|a)\s+)?(?:purpose|reason|point)\s+"
    rf"(?:in\s+my\s+{_LIFE_TARGET}|to\s+{_LIVE_VERB_TARGET})|"
    rf"i\s+have\s+nothing(?:\s+left)?\s+to\s+{_LIVE_VERB_TARGET}\s+for|"
    rf"(?:i\s+)?(?:do\s+not|don't)\s+have\s+anything\s+left\s+to\s+{_LIVE_VERB_TARGET}\s+for|"
    rf"(?:i\s+have\s+)?no\s+purpose\s+in\s+(?:my\s+)?{_LIFE_TARGET}|"
    rf"nothing(?:\s+left)?\s+to\s+{_LIVE_VERB_TARGET}\s+for|"
    rf"i\s+(?:cannot|can't)\s+(?:find|see)\s+(?:(?:any|a)\s+)?purpose\s+in\s+my\s+{_LIFE_TARGET}|"
    rf"(?:there\s+is\s+no|i\s+(?:cannot|can't)\s+see\s+(?:any|a|no))\s+"
    rf"(?:purpose|reason|point|meaning)\s+for\s+me\s+to\s+{_PERSONAL_LIFE_MEANING_TARGET}|"
    rf"why\s+bother\s+(?:stay(?:ing)?\s+alive|(?:keep(?:ing)?\s+)?{_LIVING_TARGET}|"
    rf"(?:keep(?:ing)?\s+)?go(?:ing)?\s+on)|"
    r"i\s+(?:do\s+not|don't)\s+know\s+why\s+i\s+am\s+alive|"
    rf"nothing\s+(?:(?:can\s+)?make|makes)\s+(?:my\s+)?{_LIFE_TARGET}\s+worth\s+{_LIVING_TARGET}|"
    rf"(?:{_LIFE_TARGET}|being\s+alive|{_LIVING_TARGET})\s+"
    rf"(?:has|holds|feels?\s+like\s+(?:it\s+)?has)\s+"
    rf"no\s+(?:point|reason|meaning)|"
    rf"no\s+(?:point|reason|meaning)\s+(?:in|to|of|for)\s+{_LIFE_MEANING_TARGET}"
    rf")\b"
)
_SELF_HARM_SECONDARY_SIGNAL = re.compile(
    r"\b(?:"
    r"wish\b.{0,20}\bdisappear\s+forever|"
    r"life\s+(?:(?:is\s+)?not|isn't)\s+worth\s+living|"
    r"(?:do\s+not|don't)\s+want\s+to\s+be\s+here(?:\s+(?:anymore|any\s+longer))?|"
    r"want\s+to\s+end\s+it\s+all|"
    r"no\s+reason\s+to\s+live|better\s+off\s+dead|"
    r"wish\s+i\s+(?:wasn't|weren't|was\s+not|were\s+not)\s+here"
    r")\b"
)


def _normalized_safety_text(text: str) -> str:
    """Normalize punctuation and spacing before deterministic safety matching."""

    lowered = re.sub(r"[’‘‛ʼ＇]", "'", text.casefold())
    lowered = re.sub(r"\bwanna\b", "want to", lowered)
    lowered = re.sub(r"\bgonna\b", "going to", lowered)
    lowered = re.sub(r"\bno\.\s*(?=\d+\b)", "number ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


_FICTIONAL_WORK = (
    r"(?:(?:(?:fictional|school|class|stage)\s+)?(?:story|novel|play|movie|show)|"
    r"(?:(?:[a-z][\w-]*)\s+){0,2}"
    r"(?:video|role[- ]playing)\s+game|role[- ]?play|rpg|game|drama\s+(?:class|lesson)|scene)"
)
_FICTIONAL_POSITION_UNIT = r"(?:scene|act|chapter|part|section|episode|level)"
_FICTIONAL_POSITION_PHRASE = (
    rf"(?:"
    rf"(?:at|by|near|toward)\s+(?:the\s+)?(?:beginning|start|middle|end|finale)|"
    rf"(?:halfway|midway|partway)\s+through(?:\s+(?:(?:the|a|an|my|our)\s+)?"
    rf"{_FICTIONAL_WORK})?|"
    rf"in\s+(?:the\s+)?(?:opening|middle|closing|final|first|second|third)\s+"
    rf"{_FICTIONAL_POSITION_UNIT}|"
    rf"(?:in|during|at)\s+(?:the\s+)?{_FICTIONAL_POSITION_UNIT}\s+"
    rf"(?:\d+|[ivxlcdm]+|[a-z]+(?:-[a-z]+)?)"
    rf")"
)
_FICTIONAL_SELF_HARM_SUFFIX = re.compile(
    rf"\s+(?:in|during|for)\s+(?:(?:the|a|an|my|our|your)\s+)?"
    rf"{_FICTIONAL_WORK}\s*[.!?]?$|"
    r"\s+while\s+(?:(?:i|we)\s+(?:am|are|was|were)\s+)?playing\s+"
    r"(?:(?:the|a|an|my|our|your)\s+)?"
    r"(?:(?:video|role[- ]playing)\s+game|rpg|game)\s*[.!?]?$"
)
_FICTIONAL_SELF_HARM_PREFIX = re.compile(
    rf"^(?:in|during|for)\s+(?:(?:the|a|an|my|our|your)\s+)?"
    rf"{_FICTIONAL_WORK}(?:\s*[,;:\-]\s*|\s+)|"
    rf"^(?:in|during|at)\s+(?:the\s+)?{_FICTIONAL_POSITION_UNIT}\s+"
    rf"(?:\d+|[ivxlcdm]+|[a-z]+(?:-[a-z]+)?)\s+"
    rf"(?:of|in|from)\s+(?:(?:the|a|an|my|our|your)\s+)?"
    rf"{_FICTIONAL_WORK}(?:\s*[,;:\-]\s*|\s+)|"
    rf"^while\s+(?:(?:i|we)\s+(?:am|are|was|were)\s+)?"
    rf"(?:writing|drafting|planning)\s+"
    rf"(?:(?:the|a|an|my|our|your)\s+)?{_FICTIONAL_WORK}\s*[,;:\-]\s*"
)
_FICTIONAL_CHARACTER_OWNER = re.compile(
    r"^(?:(?:i|we)\s+(?:reviewed|read|edited|discussed|revised|wrote|drafted)\s+)?"
    r"(?:(?:my|our|the|a|an)\s+)?"
    r"(?:(?:story|novel|game|play)\s+)?"
    r"(?:character|protagonist|hero|villain|avatar)(?:'s|\s+has)\s+"
)
_FICTIONAL_CHARACTER_SUBJECT = re.compile(
    rf"^(?:"
    rf"(?:(?:my|our|the|a|an)\s+)?"
    rf"(?:character|protagonist|hero|villain|avatar)"
    rf"(?:\s+(?:in|from)\s+(?:(?:the|a|an|my|our)\s+)?{_FICTIONAL_WORK})?\s+"
    rf"(?:plans?|intends?|wants?|decides?|is\s+going|will|would|might|may)\s+to\s+"
    rf"{_SELF_HARM_DISCLOSURE_ACTION_TARGET}|"
    rf"(?:(?:my|our|the|a|an)\s+)?(?:character|protagonist|hero|villain|avatar)\s+"
    rf"(?:that\s+)?(?:i|we)\s+(?:play|portray|perform|act\s+as)\s+"
    rf"(?:has\s+(?:(?:a|the)\s+)?plans?\s+to|plans?\s+to|intends?\s+to)\s*"
    rf"{_SELF_HARM_DISCLOSURE_ACTION_TARGET}"
    rf")\s*[.!?]?$"
)
_FICTIONAL_LINE_PREFIX = re.compile(
    rf"^(?:(?:the|a|an|my|our)\s+)?(?:line|dialogue|monologue)\s+"
    rf"(?:that\s+)?(?:i|we)\s+(?:say|speak|read|deliver|perform)\s+"
    rf"(?:in|during|for)\s+(?:(?:the|a|an|my|our)\s+)?{_FICTIONAL_WORK}\s+"
    rf"(?:is|was|reads?|says?)\s+"
)
_FICTIONAL_PERFORMED_LINE_PREFIX = re.compile(
    r"^(?:i|we)\s+(?:performed|delivered|rehearsed|acted\s+out)\s+"
    r"(?:(?:the|a|an|my|our)\s+)?(?:line|dialogue|monologue)\s+"
    r"(?:where|in\s+which)\s+(?:i|we)\s+(?:say|said|speak|spoke|read|deliver)\s+"
)
_FICTIONAL_READER_CHARACTER_SUBJECT = re.compile(
    rf"^(?:i|we)(?:(?:'m|'re|\s+am|\s+are|\s+was|\s+were)\s+|\s+)"
    rf"(?:reading|studying|reviewing|watching|performing|read|studied|reviewed|watched)\s+"
    rf"(?:(?:the|a|an|my|our)\s+)?{_FICTIONAL_WORK}\s+"
    rf"(?:where|in\s+which)\s+"
    rf"(?:(?:my|our|the|a|an)\s+)?(?:character|protagonist|hero|villain|avatar)\s+"
    rf"(?:plans?|intends?|wants?|decides?|is\s+going|will|would|might|may)\s+to\s+"
    rf"{_SELF_HARM_DISCLOSURE_ACTION_TARGET}\s*[.!?]?$"
)
_FICTIONAL_ACTED_CHARACTER_INTENT = re.compile(
    rf"^(?:i|we)\s+(?:acted|performed|played|role[- ]?played)\s+out\s+"
    rf"(?:(?:the|a|an|my|our)\s+)?"
    rf"(?:character|protagonist|hero|villain|avatar)(?:'s|s')\s+"
    rf"(?:plans?|decision|choice|resolve|intention)\s+to\s+"
    rf"{_SELF_HARM_DISCLOSURE_ACTION_TARGET}\s*[.!?]?$"
)
_FICTIONAL_CHARACTER_DEATH_DIRECTION = re.compile(
    r"^(?:"
    r"(?:choose|select|pick|assign)\s+death\s+for\s+"
    r"(?:(?:the|a|an|my|our)\s+)?(?:character|protagonist|hero|villain|avatar)|"
    r"(?:(?:have|let|make)\s+)?(?:(?:the|a|an|my|our)\s+)?"
    r"(?:character|protagonist|hero|villain|avatar)\s+"
    r"(?:(?:should|must|can|could|may|might|will|would)\s+)?"
    r"(?:choose|select|pick|prefer|embrace)\s+death"
    r")\s*[.!?]?$"
)

# A fictional first clause cannot launder a later first-person commitment.
# Bind anaphoric follow-through ("do the same") back to either an explicitly
# fictional subject or a third-person subject whose action is scoped by a
# named fictional work before applying any whole-statement exemption.
_FICTIONAL_SELF_HARM_SETUP = (
    rf"(?:\b(?:{_FICTIONAL_WORK}|character|protagonist|hero|villain|avatar)\b"
    rf"[^.!?;\n]{{0,144}}\b{_SELF_HARM_DISCLOSURE_ACTION_TARGET}\b"
    rf"(?:\s+{_FICTIONAL_POSITION_PHRASE})?|"
    rf"\b(?:he|she|they)\b[^.!?;\n]{{0,96}}\b{_SELF_HARM_DISCLOSURE_ACTION_TARGET}\b\s+"
    rf"(?:in|during)\s+(?:(?:the|a|an|my|our|your)\s+)?{_FICTIONAL_WORK}\b)"
)
_FICTIONAL_REAL_SELF_HARM_CONTINUATION = re.compile(
    rf"{_FICTIONAL_SELF_HARM_SETUP}"
    rf"\s*(?:,\s*(?:and|but|then)|;\s*(?:(?:and|but|then)\s*)?|"
    rf"[:—]\s*(?:(?:and|but|then)\s*)?|[.!?]\s*)\s*"
    rf"(?:"
    rf"i(?:(?:\s+am|'m)\s+going\s+to|\s+(?:plan|intend|mean|want|aim|choose|decide)\s+to|"
    rf"(?:\s+have|'ve)\s+decided\s+to|\s+will)\s+"
    rf"(?:(?:do|try|attempt|choose|carry\s+out|follow\s+through\s+with)\s+"
    rf"(?:the\s+same|that|it|so)|follow\s+suit|do\s+likewise)|"
    rf"i\s+(?:plan|intend|mean|want)\s+(?:the\s+same|that|it|likewise)|"
    rf"i\s+(?:have|share)\s+(?:the\s+same|that(?:\s+same)?)\s+"
    rf"(?:plan|intention|choice|decision|resolve)|"
    rf"i\s+(?:plan|intend|mean|want|choose|decide)\s+to\s+"
    rf"(?:too|likewise|as\s+well)|"
    rf"i(?:(?:\s+am|'m))\s+(?:of\s+)?(?:(?:the|that)\s+)?same\s+"
    rf"(?:mind|mindset|resolve|view)|"
    rf"i(?:"
    rf"(?:\s+am|'m)\s+(?:going\s+to\s+follow|following)|"
    rf"(?:\s+am|'m)\s+(?:determined|resolved|set)\s+to\s+follow|"
    rf"(?:\s+have|'ve)\s+(?:chosen|decided|determined|elected|resolved)\s+to\s+follow|"
    rf"'ll\s+follow|"
    rf"\s+(?:will|would|plan|intend|mean|want|aim|choose|chose|determine|determined|"
    rf"elect|elected|resolve|resolved)\s+"
    rf"(?:to\s+)?follow"
    rf")\s+(?:(?:his|her|their|the\s+"
    rf"(?:character|protagonist|hero|villain|avatar)(?:'s)?)\s+(?:lead|example|path|course)|"
    rf"(?:in\s+)?(?:his|her|their|the\s+"
    rf"(?:character|protagonist|hero|villain|avatar)(?:'s)?)\s+footsteps)|"
    rf"i\s+(?:plan|intend|mean|aim)\s+on\s+following\s+"
    rf"(?:(?:his|her|their)\s+(?:lead|example)|(?:in\s+)?(?:his|her|their)\s+footsteps)|"
    rf"i(?:(?:\s+have|'ve))?\s+made\s+up\s+my\s+mind\s+to\s+follow\s+"
    rf"(?:(?:his|her|their)\s+(?:lead|example)|(?:in\s+)?(?:his|her|their)\s+footsteps)|"
    rf"i(?:(?:\s+have|'ve)\s+|\s+)(?:decided|resolved|determined)\s+"
    rf"(?:that\s+)?i\s+(?:will|would)\s+follow\s+"
    rf"(?:(?:his|her|their)\s+(?:lead|example)|(?:in\s+)?(?:his|her|their)\s+footsteps)|"
    rf"i(?:(?:\s+am|'m))\s+(?:determined|resolved|set)\s+that\s+i\s+"
    rf"(?:will|would)\s+follow\s+"
    rf"(?:(?:his|her|their)\s+(?:lead|example)|(?:in\s+)?(?:his|her|their)\s+footsteps)|"
    rf"i(?:(?:\s+am|'m))\s+set\s+on\s+following\s+"
    rf"(?:(?:his|her|their)\s+(?:lead|example)|(?:in\s+)?(?:his|her|their)\s+footsteps)|"
    rf"i(?:(?:\s+have|'ve))?\s+made\s+my\s+mind\s+up\s+to\s+follow\s+"
    rf"(?:(?:his|her|their)\s+(?:lead|example|path|course)|"
    rf"(?:in\s+)?(?:his|her|their)\s+footsteps)|"
    rf"i(?:(?:\s+am|'m)\s+going\s+to|\s+(?:plan|intend|mean|want|decide)\s+to|"
    rf"(?:\s+have|'ve)\s+decided\s+to|'ll|\s+(?:will|would))\s+"
    rf"(?:copy|imitate|emulate)\s+(?:"
    rf"him|her|them|the\s+(?:character|protagonist|hero|villain|avatar)|"
    rf"what\s+(?:he|she|they)\s+(?:does|do|did))|"
    rf"i(?:(?:\s+have|'ve)\s+(?:chosen|decided|determined|resolved)\s+to|"
    rf"\s+(?:plan|intend|mean|want|choose|decide|determine|resolve)\s+to)\s+act\s+like\s+"
    rf"(?:him|her|them|the\s+(?:character|protagonist|hero|villain|avatar))|"
    rf"i(?:(?:\s+have|'ve)\s+(?:chosen|decided|determined|resolved)\s+to|"
    rf"\s+(?:plan|intend|mean|want|choose|decide|determine|resolve)\s+to)\s+"
    rf"(?:carry\s+(?:out|through\s+with)|follow\s+through\s+with)\s+"
    rf"(?:his|her|their)\s+(?:choice|decision|plan|intention|resolve)"
    rf"(?:\s+(?:too|likewise|as\s+well))?|"
    rf"i(?:(?:\s+have|'ve)\s+(?:chosen|decided|determined|resolved)\s+to|"
    rf"\s+(?:plan|intend|mean|want|choose|decide|determine|resolve)\s+to)\s+"
    rf"(?:put\s+(?:his|her|their)\s+(?:choice|decision|plan|intention|resolve)\s+into\s+action|"
    rf"(?:act\s+on|see)\s+(?:his|her|their)\s+(?:choice|decision|plan|intention|resolve)"
    rf"(?:\s+through)?)|"
    rf"i(?:(?:\s+am|'m)\s+going\s+to|\s+(?:plan|intend|mean|want|aim|choose|decide)\s+to|"
    rf"(?:\s+have|'ve)\s+(?:chosen|decided|determined|resolved)\s+to|'ll|"
    rf"\s+(?:will|would))\s+"
    rf"(?:mirror|reenact|re-enact|repeat|recreate)\s+"
    rf"(?:his|her|their)\s+(?:choice|decision|plan|intention|resolve|path|course)|"
    rf"i(?:(?:\s+am|'m)\s+(?:intent|set|determined|resolved)\s+(?:on|upon)\s+|"
    rf"(?:\s+have|'ve)\s+(?:chosen|decided|determined|resolved)\s+to\s+|"
    rf"\s+(?:plan|intend|mean|want|aim|choose|decide|determine|resolve)\s+to\s+)"
    rf"(?:act(?:ing)?\s+like|mirror(?:ing)?|imitat(?:e|ing)|emulat(?:e|ing))\s+"
    rf"(?:him|her|them|the\s+(?:character|protagonist|hero|villain|avatar))|"
    rf"i(?:(?:\s+am|'m)\s+going\s+to|\s+(?:plan|intend|mean|want|aim|choose|decide)\s+to|"
    rf"(?:\s+have|'ve)\s+(?:chosen|decided|determined|resolved)\s+to|'ll|"
    rf"\s+(?:will|would))\s+"
    rf"(?:finish|complete|continue)\s+what\s+(?:he|she|they)\s+(?:started|began)|"
    rf"i(?:(?:\s+am|'m)\s+going\s+to|\s+(?:plan|intend|mean|want|aim|choose|decide)\s+to|"
    rf"(?:\s+have|'ve)\s+(?:chosen|decided|determined|resolved)\s+to|'ll|"
    rf"\s+(?:will|would))\s+(?:follow|carry)\s+through"
    rf"(?:\s+with\s+(?:it|that|this|the\s+same|(?:his|her|their)\s+"
    rf"(?:choice|decision|plan|intention|resolve)))?|"
    rf"i(?:(?:\s+am|'m)\s+(?:set|determined|resolved)\s+(?:on\s+|to\s+)|"
    rf"(?:\s+have|'ve)\s+(?:chosen|decided|determined|elected|resolved)\s+to\s+|"
    rf"\s+(?:chose|decided|determined|elected|resolved)\s+to\s+)"
    rf"(?:copy|imitate|emulate)\s+(?:"
    rf"him|her|them|the\s+(?:character|protagonist|hero|villain|avatar)|"
    rf"what\s+(?:he|she|they)\s+(?:does|do|did))|"
    rf"i(?:(?:\s+am|'m)\s+going\s+to|\s+(?:plan|intend|mean|want|decide)\s+to|"
    rf"'ll|\s+(?:will|would|decided|resolved))\s+(?:to\s+)?do\s+(?:exactly\s+)?(?:what|as)\s+"
    rf"(?:he|she|they|the\s+(?:character|protagonist|hero|villain|avatar))\s+"
    rf"(?:does|do|did)|"
    rf"i(?:(?:\s+am|'m)\s+going\s+to|\s+(?:plan|intend|mean|want|decide)\s+to|"
    rf"'ll|\s+(?:will|would))\s+(?:repeat|recreate)\s+(?:it|that|the\s+same\s+thing)|"
    rf"i(?:(?:\s+am|'m))\s+following\s+suit|"
    rf"(?:and\s+)?so\s+(?:do|will|would|am)\s+i|"
    rf"i\s+(?:(?:will|would|do|am)\s+(?:too|likewise|as\s+well)|"
    rf"(?:also|likewise)\s+(?:will|would|do|am))"
    rf")\b"
)


def _has_protective_fictional_rewrite_continuation(text: str) -> bool:
    """Recognize a complete rewrite complement after a fictional death setup."""

    setup = re.search(_FICTIONAL_SELF_HARM_SETUP, text)
    if not setup:
        return False
    tail = text[setup.end() :]
    tail = re.sub(
        r"^\s*(?:,\s*(?:and|but|then)|;\s*(?:(?:and|but|then)\s*)?|"
        r"[:—]\s*(?:(?:and|but|then)\s*)?|[.!?]\s*)\s*",
        "",
        tail,
    )
    protective_rewrite = re.fullmatch(
        r"i\b[^.!?;\n]{0,72}\b(?:follow|copy|imitate|emulate)\b"
        r"[^.!?;\n]{0,48}\bby\s+"
        r"(?:rewriting|revising|editing|changing|reversing)\s+"
        r"(?:(?:the|his|her|their)\s+)?"
        r"(?:ending|story|scene|plot|chapter|outcome|choice|decision|plan)\s*[.!?]?",
        tail,
    )
    if not protective_rewrite:
        return False
    follow = re.search(r"\b(?:follow|copy|imitate|emulate)\b", tail)
    return bool(follow and not re.search(rf"\b(?:{_SELF_HARM_ACTION_TARGET})\b", tail[: follow.start()]))


def _has_fictional_real_self_harm_continuation(text: str) -> bool:
    """Bind a fictional self-harm setup to a later real first-person commitment."""

    if _has_protective_fictional_rewrite_continuation(text):
        return False
    setup = re.search(_FICTIONAL_SELF_HARM_SETUP, text)
    if not setup:
        return False
    tail = text[setup.end() :]
    tail = re.sub(
        r"^\s*(?:,\s*(?:and|but|then)|;\s*(?:(?:and|but|then)\s*)?|"
        r"[:—]\s*(?:(?:and|but|then)\s*)?|[.!?]\s*)\s*",
        "",
        tail,
    )
    safe_action = re.search(
        r"\b(?:discuss(?:es|ed|ing)?|analy[sz](?:e|es|ed|ing)|stud(?:y|ies|ied|ying)|"
        r"question(?:s|ed|ing)?|challeng(?:e|es|ed|ing)|reject(?:s|ed|ing)?|"
        r"oppos(?:e|es|ed|ing)|avoid(?:s|ed|ing)?|prevent(?:s|ed|ing)?|"
        r"stop(?:s|ped|ping)?|criticiz(?:e|es|ed|ing)|critiqu(?:e|es|ed|ing)|"
        r"rewrit(?:e|es|ten|ing)|revis(?:e|es|ed|ing)|edit(?:s|ed|ing)?|"
        r"chang(?:e|es|ed|ing)|revers(?:e|es|ed|ing)|warn(?:s|ed|ing)?\s+about)\b",
        tail,
    )
    dangerous_follow_through = (
        r"\b(?:follow|copy|imitate|emulate|adopt|execute|implement|fulfill|carry\s+(?:out|through)|"
        r"go\s+through|proceed|make\s+it\s+happen)\b"
    )
    if (
        safe_action
        and not re.search(dangerous_follow_through, tail[: safe_action.start()])
        and not re.search(dangerous_follow_through, tail[safe_action.end() :])
    ):
        return False
    # Once a fictional self-harm plan is established, bind a real first-person
    # commitment to an anaphoric follow-through structurally.  This covers
    # adverb placement and ordinary constructions such as "prepared to follow",
    # "go ahead with it", and "make it happen" without treating a named rewrite,
    # critique, or prevention action as follow-through.
    safe_follow_up = re.match(
        r"i\s+(?:(?:am|'m|was)\s+)?"
        r"(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*"
        r"(?:(?:plan|intend|mean|aim|expect|want|choose|decide)\s+"
        r"(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*to|"
        r"(?:am\s+|'m\s+)?(?:going|prepared|ready|about)\s+to)\s+"
        r"(?:discuss|analy[sz]e|study|question|challenge|reject|oppose|avoid|prevent|"
        r"stop|criticize|critique|rewrite|revise|edit|change|reverse|warn\s+about)\b",
        tail,
    )
    structural_commitment = re.match(
        r"(?:"
        r"i\s+(?:cannot|can't|can\s+not)\s+shake\s+"
        r"(?:(?:an?|the|my)\s+)?(?:temptation|urge|desire|impulse)\s+to|"
        r"i\s+(?:(?:still\s+)?(?:feel|felt)|remain|remains|remained)\s+"
        r"(?:tempted|inclined|disposed|drawn)\s+to|"
        r"(?:it\s+(?:is|was)\s+(?:my|our)|(?:my|our))\s+"
        r"(?:(?:full|firm|clear|settled|definite|main|primary)\s+)?"
        r"(?:intent|intention|plan|aim|goal|ambition|wish|desire)\s+"
        r"(?:(?:is|was|remains?)\s+)?(?:to|of)|"
        r"i\s+(?:made|make)\s+it\s+(?:my|our)\s+"
        r"(?:(?:full|firm|clear|settled|definite|main|primary)\s+)?"
        r"(?:intent|intention|plan|aim|goal|ambition|wish|desire)\s+(?:to|of)|"
        r"i\s+(?:have|hold|carry|possess)\s+"
        r"(?:(?:a|an|the|my)\s+)?"
        r"(?:(?!(?:no|not|zero|without|fictional|pretend)\b)[a-z]+\s+){0,3}"
        r"(?:intent|intention|plan|aim|goal|ambition)\s+(?:to|of)|"
        r"i(?:"
        r"(?:\s+am|'m|\s+was)\s+(?:(?:still|now|currently|seriously)\s+)*(?:"
        r"(?:going|planning|aiming|intending|meaning|preparing|prepared|ready|about)\s+(?:to|ahead)|"
        r"(?:tempted|inclined|disposed)\s+to|"
        r"(?:intent|set|determined|resolved|committed|fixed|bent)\s+(?:to|on|upon)|"
        r"(?:certain|sure|positive|confident)\s+(?:that\s+)?i\s+(?:will|would|shall))|"
        r"(?:\s+have|'ve|\s+had|'d)\s+(?:"
        r"(?:been\s+)?(?:thinking\s+(?:about|of)|considering|contemplating|"
        r"reflecting\s+(?:about|on))\s+|"
        r"(?:been\s+)?(?:tempted|inclined|disposed)\s+to|"
        r"(?:considered|contemplated)\s+|"
        r"reflected\s+(?:about|on)\s+|"
        r"thought\s+(?:about|of)\s+|"
        r"(?:chosen|decided|determined|resolved|elected|opted|committed|promised|vowed)\s+"
        r"(?:(?:with\s+(?:certainty|confidence|conviction)|firmly|definitely|certainly)\s+)?to|"
        r"every\s+intention\s+(?:of|to))|"
        r"\s+(?:decided|determined|resolved)\s+"
        r"(?:(?:with\s+(?:certainty|confidence|conviction)|firmly|definitely|certainly)\s+)?to|"
        r"\s+(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*"
        r"(?:plan|intend|mean|aim|aspire|want|choose|decide|determine|resolve|promise|vow|"
        r"expect|will|would|shall)\s+"
        r"(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*(?:to\s+)?)"
        r")",
        tail,
    )
    if structural_commitment and not safe_follow_up:
        continuation = tail[structural_commitment.end() :]
        if re.match(r"^\s+with\s+(?:it|that|this|the\s+(?:plan|same))\b", continuation) and re.search(
            r"\bgoing\s+ahead$", tail[: structural_commitment.end()]
        ):
            return True
        if re.search(
            r"^\s*(?:"
            r"(?:follow|copy|imitate|emulate)\b[^.!?;\n]{0,36}\b"
            r"(?:his|her|their)\s+(?:lead|example|path|course|footsteps)|"
            r"(?:go(?:ing)?\s+ahead|proceed|carry(?:ing)?\s+(?:out|through))\s+with\s+"
            r"(?:it|that|this|the\s+(?:plan|same))|"
            r"make\s+(?:it|that|this)\s+happen|"
            r"do(?:ing)?\s+(?:likewise|the\s+same(?:\s+thing)?)"
            r")\b",
            continuation,
        ):
            return True
    commitment = re.match(
        r"i(?:"
        r"(?:\s+am|'m|\s+was)\s+(?:(?:still|now|currently|seriously)\s+)*(?:"
        r"(?:thinking\s+(?:about|of)|considering|contemplating)\s+|"
        r"(?:going|planning|aiming|intending|meaning|preparing|prepared|ready|about)\s+to|"
        r"(?:tempted|inclined|disposed)\s+to|"
        r"(?:intent|set|determined|resolved|committed|fixed|bent)\s+(?:to|on|upon)|"
        r"(?:determined|resolved|committed|certain|sure|positive|confident)\s+"
        r"(?:that\s+)?i\s+(?:will|would|shall))|"
        r"(?:\s+have|'ve|\s+had|'d)\s+(?:"
        r"(?:been\s+)?(?:thinking\s+(?:about|of)|considering|contemplating|"
        r"reflecting\s+(?:about|on))\s+|"
        r"(?:been\s+)?(?:tempted|inclined|disposed)\s+to|"
        r"(?:considered|contemplated)\s+|"
        r"reflected\s+(?:about|on)\s+|"
        r"thought\s+(?:about|of)\s+|"
        r"(?:chosen|decided|determined|resolved|elected|opted|committed|promised|vowed)\s+"
        r"(?:(?:with\s+(?:certainty|confidence|conviction)|firmly|definitely|certainly)\s+)?to|"
        r"every\s+intention\s+(?:of|to)|"
        r"(?:decided|determined|resolved)\s+(?:that\s+)?i\s+"
        r"(?:(?:am|'m)\s+going\s+to|will|would|shall))|"
        r"\s+(?:decided|determined|resolved)\s+"
        r"(?:(?:with\s+(?:certainty|confidence|conviction)|firmly|definitely|certainly)\s+)?to|"
        r"\s+(?:(?:[a-z]+ly|very(?:\s+much)?)\s+)*"
        r"(?:plan|intend|mean|aim|aspire|want|choose|decide|determine|resolve|promise|vow|"
        r"expect|will|would|shall)\s+(?:to\s+)?)",
        tail,
    )
    if not commitment:
        return False
    continuation = tail[commitment.end() :]
    linked_action = (
        r"(?:follow(?:ing)?|mirror(?:ing)?|copy(?:ing)?|imitat(?:e|ing)|emulat(?:e|ing)|"
        r"reenact(?:ing)?|re-enact(?:ing)?|repeat(?:ing)?|recreat(?:e|ing)|"
        r"adopt(?:ing)?|pursu(?:e|ing)|execut(?:e|ing)|implement(?:ing)?|fulfill(?:ing)?|"
        r"honor(?:ing)?|carry(?:ing)?\s+(?:out|through|on)|go(?:ing)?\s+through|tread(?:ing)?|"
        r"put(?:ting)?|act(?:ing)?\s+(?:like|on|upon)|"
        r"finish(?:ing)?|complet(?:e|ing)|continu(?:e|ing)|take|taking|see(?:ing)?)"
    )
    linked_reference = (
        r"(?:(?:his|her|their)\s+(?:lead|example|path|course|choice|decision|plan|"
        r"intention|resolve|footsteps)|(?:him|her|them)|"
        r"(?:(?:the\s+)?(?:character|protagonist|hero|villain|avatar))'s\s+"
        r"(?:lead|example|path|course|choice|decision|plan|intention|resolve|footsteps)|"
        r"what\s+(?:he|she|they)\s+(?:started|began|did|chose|planned)|"
        r"where\s+(?:he|she|they)\s+(?:left\s+off|stopped)|"
        r"(?:the\s+same|the\s+plan|that|it|so|suit|through))"
    )
    if re.search(rf"^\s*{linked_action}[^.!?;\n]{{0,48}}\b{linked_reference}\b", continuation) or re.search(
        r"^\s*(?:do\s+(?:the\s+same|likewise)|follow\s+suit|follow\s+through)\b",
        continuation,
    ):
        return True
    # Fail closed on a novel action that adopts the fictional character's
    # self-harm choice/path. Keep an explicit neutral/oppositional verb set so
    # discussing, rejecting, or rewriting that choice remains ordinary fiction.
    semantic_reference = re.search(
        r"\b(?:(?:his|her|their)\s+(?:lead|example|path|course|choice|decision|plan|"
        r"intention|resolve|footsteps)|(?:the\s+same|the\s+plan)|"
        r"(?:(?:the\s+)?(?:character|protagonist|hero|villain|avatar))'s\s+"
        r"(?:lead|example|path|course|choice|decision|plan|intention|resolve|footsteps)|"
        r"what\s+(?:he|she|they)\s+(?:started|began|did|chose|planned)|"
        r"where\s+(?:he|she|they)\s+(?:left\s+off|stopped))\b",
        continuation,
    )
    if not semantic_reference:
        return False
    return not bool(
        re.match(
            r"\s*(?:(?:[a-z]+ly|very)\s+)*(?:to\s+)?"
            r"(?:discuss|analy[sz]e|study|question|challenge|reject(?:s|ed|ing)?|oppose|avoid|prevent|"
            r"stop|criticize|critique|rewrit(?:e|es|ten|ing)|revise|edit|change|reverse|warn\s+about|write\s+about|"
            r"talk\s+about|tell\s+(?:someone|an?\s+adult|a\s+grown-up)\s+about)\b",
            continuation,
        )
    )


_FICTIONAL_POSITION_TAIL = re.compile(rf"\s+{_FICTIONAL_POSITION_PHRASE}\s*[.!?]?$")


def _without_fictional_position_tail(text: str) -> str:
    """Remove scene position only after an explicit fictional scope is proven."""

    return _FICTIONAL_POSITION_TAIL.sub("", text).strip()


_NON_INTENT_SELF_HARM_REFERENCE = re.compile(
    rf"^(?:i|we)\s+(?:read|learned|studied|heard|asked|talked|researched)\s+"
    rf"(?:about\s+)?(?:{_SELF_HARM_METHOD_NOUN}|suicide\s+(?:methods?|plans?))\s+"
    rf"(?:of|about|for)\s+(?:{_SELF_HARM_DISCLOSURE_GERUND_TARGET}|suicide)\s*[.!?]?$"
)


def _is_clearly_fictional_self_harm_statement(text: str) -> bool:
    """Recognize a complete first-person disclosure scoped to fictional play."""

    performed_line_prefix = _FICTIONAL_PERFORMED_LINE_PREFIX.match(text)
    if performed_line_prefix:
        quoted_line = _without_fictional_position_tail(text[performed_line_prefix.end() :])
        quoted_line = re.sub(r"[.!?]+$", "", quoted_line).strip()
        return bool(
            _SELF_HARM_FIRST_PERSON_STRUCTURAL_INTENT_SIGNAL.fullmatch(quoted_line)
            or _SELF_HARM_EXPLICIT_INTENT_SIGNAL.fullmatch(quoted_line)
            or _SELF_HARM_SIGNAL.fullmatch(quoted_line)
        )
    line_prefix = _FICTIONAL_LINE_PREFIX.match(text)
    if line_prefix:
        quoted_line = _without_fictional_position_tail(text[line_prefix.end() :])
        quoted_line = re.sub(r"[.!?]+$", "", quoted_line).strip()
        return bool(
            _SELF_HARM_FIRST_PERSON_STRUCTURAL_INTENT_SIGNAL.fullmatch(quoted_line)
            or _SELF_HARM_EXPLICIT_INTENT_SIGNAL.fullmatch(quoted_line)
            or _SELF_HARM_SIGNAL.fullmatch(quoted_line)
        )
    if _FICTIONAL_READER_CHARACTER_SUBJECT.fullmatch(text):
        return True
    if _FICTIONAL_ACTED_CHARACTER_INTENT.fullmatch(text):
        return True
    if _FICTIONAL_CHARACTER_DEATH_DIRECTION.fullmatch(text):
        return True
    if _FICTIONAL_CHARACTER_SUBJECT.fullmatch(text):
        return True
    bare_position = re.match(rf"^{_FICTIONAL_POSITION_PHRASE}\s*[,;:\-]?\s+", text)
    if bare_position and _FICTIONAL_CHARACTER_SUBJECT.fullmatch(text[bare_position.end() :]):
        # A bare chapter/scene reference is sufficient only when the remaining
        # clause has an explicit fictional character as its grammatical owner.
        # It must not exempt a first-person statement such as "In chapter two,
        # I plan to die."
        return True
    character_owner = _FICTIONAL_CHARACTER_OWNER.match(text)
    if character_owner:
        return bool(
            _SELF_HARM_PLAN_DISCLOSURE_SIGNAL.fullmatch(text)
            or _SELF_HARM_FIRST_PERSON_STRUCTURAL_INTENT_SIGNAL.fullmatch(text)
            or _SELF_HARM_EXPLICIT_INTENT_SIGNAL.fullmatch(text)
            or _SELF_HARM_SIGNAL.fullmatch(text)
        )
    statement = text
    scoped = False
    prefix = _FICTIONAL_SELF_HARM_PREFIX.match(statement)
    if prefix:
        statement = statement[prefix.end() :].strip()
        scoped = True
    suffix = _FICTIONAL_SELF_HARM_SUFFIX.search(statement)
    if suffix:
        statement = statement[: suffix.start()].strip()
        scoped = True
    if not scoped:
        return False
    statement = re.sub(r"[.!?]+$", "", statement).strip()
    statement = _without_fictional_position_tail(statement)
    if _FICTIONAL_CHARACTER_SUBJECT.fullmatch(statement):
        return True
    if _FICTIONAL_CHARACTER_DEATH_DIRECTION.fullmatch(statement):
        return True
    plan_noun = re.search(rf"\b(?:{_SELF_HARM_METHOD_NOUN}|suicide\s+plans?)\b", statement)
    structural_plan = bool(
        plan_noun
        and _SELF_HARM_PLAN_DISCLOSURE_SIGNAL.fullmatch(statement)
        and not _SELF_HARM_EXPLICIT_INTENT_SIGNAL.search(statement[: plan_noun.start()])
        and not _SELF_HARM_SIGNAL.search(statement[: plan_noun.start()])
    )
    return bool(
        _SELF_HARM_FIRST_PERSON_STRUCTURAL_INTENT_SIGNAL.fullmatch(statement)
        or _SELF_HARM_EXPLICIT_INTENT_SIGNAL.fullmatch(statement)
        or structural_plan
        or _SELF_HARM_SIGNAL.fullmatch(statement)
        or _SELF_HARM_MODAL_INJURY_SIGNAL.fullmatch(statement)
    )


def _is_self_harm(text: str) -> bool:
    lowered = _normalized_safety_text(text)
    if _has_protective_fictional_rewrite_continuation(lowered):
        return False
    if _FICTIONAL_REAL_SELF_HARM_CONTINUATION.search(lowered) or _has_fictional_real_self_harm_continuation(lowered):
        return True
    if _NON_INTENT_SELF_HARM_REFERENCE.fullmatch(lowered):
        return False
    if _is_clearly_fictional_self_harm_statement(lowered):
        return False
    if _is_protective_self_harm_clause(lowered):
        return False
    return bool(
        _SELF_HARM_FIRST_PERSON_STRUCTURAL_INTENT_SIGNAL.search(lowered)
        or _SELF_HARM_EXPLICIT_INTENT_SIGNAL.search(lowered)
        or _SELF_HARM_PLAN_DISCLOSURE_SIGNAL.search(lowered)
        or _SELF_HARM_PLURAL_INTENT_SIGNAL.search(lowered)
        or _SELF_HARM_SIGNAL.search(lowered)
        or _SELF_HARM_MODAL_INJURY_SIGNAL.search(lowered)
        or _SELF_HARM_MEANINGLESSNESS_SIGNAL.search(lowered)
        or _SELF_HARM_SECONDARY_SIGNAL.search(lowered)
    )


def _is_sensitive(text: str) -> bool:
    lowered = _normalized_safety_text(text)
    if _has_protective_fictional_rewrite_continuation(lowered):
        return False
    if _FICTIONAL_REAL_SELF_HARM_CONTINUATION.search(lowered) or _has_fictional_real_self_harm_continuation(lowered):
        return True
    if _NON_INTENT_SELF_HARM_REFERENCE.fullmatch(lowered):
        return False
    if _is_clearly_fictional_self_harm_statement(lowered):
        return False
    if _is_protective_self_harm_clause(lowered):
        return False
    if _is_self_harm(lowered):
        return True
    # Appliance state language such as "do not let the wet toaster go dead"
    # is an electrical hazard, not a disclosure about a person. Remove only
    # the appliance-bound state span before the broad sensitive-topic scan;
    # any independent first-person or human death language remains visible.
    generic_sensitive_text = re.sub(
        rf"\b{_POWERED_APPLIANCE}\b[^.!?;\n]{{0,48}}\b"
        r"(?:go(?:es|ing|ne)?|becom(?:e|es|ing)|turn(?:s|ed|ing)?)\s+dead\b",
        "",
        lowered,
    )
    return bool(
        re.search(
            r"\b(?:private parts?|penis|vagina|sex|pregnan|blood|hurt me|abuse|suicid|kill myself|"
            r"died|death|dead|wish\s+i\s+(?:was|were)\s+dead|self[- ]harm|"
            r"disappear\s+forever|life\s+(?:is\s+)?not\s+worth\s+living|"
            r"(?:do\s+not|don't)\s+want\s+to\s+be\s+here|want\s+to\s+end\s+it\s+all|"
            r"no\s+reason\s+to\s+live|better\s+off\s+dead|wish\s+i\s+(?:wasn't|weren't|was\s+not)\s+here|"
            r"divorce|bull(?:y|ied|ying)|friend (?:hates|ignor(?:e|ed|ing)|left)|"
            r"why .{0,40}friend|friend .{0,25}(?:mean|mad|won't|doesn't like)|where (?:do )?bab(?:y|ies) "
            r"come from|how .{0,30}babies .{0,20}made|private parts?|naked|touch(?:ed|ing) me|unsafe secret|"
            r"period|menstru|puberty|breast|nipple|diagnos|medicine|dose|anxious|scared|very sad)\b",
            generic_sensitive_text,
        )
    )


def _requires_direct_answer(text: str, workflow: str) -> bool:
    """Code-owned recognition of requests that prohibit an activity or assessment."""

    if workflow == "quick_answer":
        return True
    if workflow.startswith("artifact_"):
        # The typed artifact workflow is itself an explicit format request. A
        # negative constraint inside its brief (for example, "without repeating
        # the old activity") must shape that artifact, not cancel it.
        return False
    lowered = " ".join(text.casefold().split())
    format_pattern = r"activit(?:y|ies)|games?|quizzes?|lessons?|experiments?|worksheets?|projects?|challenges?"

    def governing_request_is_negated(start: int) -> bool:
        prefix = lowered[max(lowered.rfind(token, 0, start) for token in (".", "!", "?", ";")) + 1 : start]
        return bool(
            re.search(
                r"\b(?:do\s+not|don't|never)\s+(?:(?:really|actually)\s+)?"
                r"(?:(?:want|wish|need|intend|plan|like|prefer|choose)\s+)?(?:to\s+)?$|"
                r"\bnot\s+to\s+$|\b(?:would\s+rather|rather)\s+not\s+$",
                prefix,
            )
        )

    explicit_positive_format = (
        rf"\b(?:just\s+|simply\s+)?(?:give(?:\s+(?:me|us))?|make|create|try|play|do)\b"
        rf"(?:(?!\b(?:no|not|without|instead\s+of|rather\s+than)\b).){{0,35}}\b(?:{format_pattern})\b",
        rf"\b(?:explain|teach)\b(?:(?!\b(?:no|not|without)\b).){{0,30}}"
        rf"\b(?:through|as|with)\b.{{0,14}}\b(?:{format_pattern})\b",
        rf"(?:^|[.!?;]\s+)(?:an?\s+)?(?:{format_pattern})\s+(?:please|instead)\b",
        rf"\bhow\s+about\s+(?:an?\s+)?(?:{format_pattern})\b",
    )
    for pattern in explicit_positive_format:
        for positive in re.finditer(pattern, lowered):
            if not governing_request_is_negated(positive.start()):
                return False
    direct_forms = (
        r"\b(?:just|simply)\s+(?:tell|answer|explain|give)\b",
        r"\b(?:give|tell)\s+me\s+(?:just\s+|only\s+)?(?:the\s+)?(?:answer|explanation)\b",
        r"\bonly\s+(?:want|need)\b.{0,48}\b(?:answer|explanation)\b",
        r"\b(?:answer|explanation)\s+only\b",
        r"\bkeep\s+it\s+short\b",
        r"\bprefer\s+(?:an?\s+|the\s+)?(?:answer|explanation)\s+(?:over|to)\b"
        r".{0,35}\b(?:activit(?:y|ies)|games?|quizzes?|lessons?|experiments?|worksheets?|projects?|challenges?)\b",
    )
    if any(re.search(pattern, lowered) for pattern in direct_forms):
        return True

    def normalized_format(value: str) -> str:
        return {
            "activities": "activity",
            "games": "game",
            "quizzes": "quiz",
            "lessons": "lesson",
            "experiments": "experiment",
            "worksheets": "worksheet",
            "projects": "project",
            "challenges": "challenge",
        }.get(value, value)

    coordinated_formats = rf"{format_pattern}(?:\s*(?:,|/|\bor\b|\band\b)\s*(?:{format_pattern}))*"
    negative_patterns = (
        rf"\bno\s+(?:an?\s+|any\s+|the\s+)?(?P<formats>{coordinated_formats})",
        rf"\b(?:skip|omit|avoid|leave\s+out)\s+(?:an?\s+|any\s+|the\s+)?"
        rf"(?P<formats>{coordinated_formats})",
        rf"\bnot\s+(?:an?\s+|any\s+|the\s+)?(?P<formats>{coordinated_formats})",
        rf"\bwithout(?:\s+turning\s+(?:it|this)\s+into)?\s+(?:an?\s+|any\s+|the\s+)?"
        rf"(?P<formats>{coordinated_formats})",
        rf"\b(?:do\s+not|don't)(?:(?![.!?;]).){{0,55}}?(?P<formats>{coordinated_formats})",
        rf"\b(?:would\s+rather|rather)\s+not(?:(?![.!?;]).){{0,35}}?"
        rf"(?P<formats>{coordinated_formats})",
        rf"\b(?:answer|explanation)\s+(?:rather\s+than|instead\s+of)"
        rf"(?:(?![.!?;]).){{0,24}}?(?P<formats>{coordinated_formats})",
        rf"\b(?:please\s+)?(?:answer|explain|tell(?:\s+me)?)\b"
        rf"(?:(?![.!?;]).){{0,80}}?\b(?:rather\s+than|instead\s+of)\b"
        rf"(?:(?![.!?;]).){{0,40}}?(?P<formats>{coordinated_formats})",
    )
    excluded_learning_formats: set[str] = set()
    for pattern in negative_patterns:
        for clause in re.finditer(pattern, lowered):
            excluded_learning_formats.update(
                normalized_format(match.group(0)) for match in re.finditer(format_pattern, clause.group("formats"))
            )
    requested_learning_formats = set()
    for match in re.finditer(
        r"\b(?:make|create|do|try|play|give\s+us|teach\s+(?:this|it)\s+as|turn\s+(?:it|this)\s+into)"
        r"\b.{0,32}?\b(" + format_pattern + r")\b",
        lowered,
    ):
        if not governing_request_is_negated(match.start()):
            requested_learning_formats.add(normalized_format(match.group(1)))
    for preference in re.finditer(
        r"\b(?:want|would\s+like|prefer|hope\s+for|looking\s+for|could\s+use)\b"
        r".{0,28}?\b(" + format_pattern + r")\b|"
        r"\b(" + format_pattern + r")\b\s+(?:would\s+be|sounds?|is)\s+"
        r"(?:great|good|helpful|fun|nice|better)\b",
        lowered,
    ):
        preferred_format = next(group for group in preference.groups() if group)
        requested_learning_formats.add(normalized_format(preferred_format))
    if requested_learning_formats - excluded_learning_formats:
        return False
    return bool(excluded_learning_formats)


def personalization_is_allowed(*, text: str, workflow: str, question_intent: str) -> bool:
    """Code-owned exclusion gate shared by decision preparation and compilation."""

    return not (
        _is_sensitive(text)
        or _requires_direct_answer(text, workflow)
        or question_intent in {"direct_answer", "sensitive_support"}
    )


def _approved_accessibility(text: str, proposed: AccessibilityPolicy) -> AccessibilityPolicy:
    """Apply universal K-2 access floors and derive explicit accommodations from the request."""

    lowered = text.casefold()
    requested: list[Literal["no_writing", "low_fine_motor", "read_aloud"]] = []
    if re.search(r"\b(?:no|not|without|does not want to|doesn't want to)\b.{0,24}\b(?:write|writing)\b", lowered):
        requested.append("no_writing")
    if re.search(r"\b(?:without|avoid|no)\b.{0,35}\b(?:cutting|scissors|tiny pieces?|fine motor)\b", lowered):
        requested.append("low_fine_motor")
    if re.search(r"\b(?:read aloud|grown-up will read|adult will read|listen instead)\b", lowered):
        requested.append("read_aloud")
    choices = list(dict.fromkeys([*proposed.response_choices, "spoken", "gesture_or_manipulative"]))[:5]
    if "no_writing" in requested:
        choices = [choice for choice in choices if choice != "written"]
    return AccessibilityPolicy(
        response_choices=choices,
        no_writing_alternative="The child may point, speak, arrange whole objects, or draw instead of writing.",
        color_independent=True,
        fine_motor_alternative="Use pointing, whole objects, or a grown-up scribe; cutting and tiny marks are optional.",
        reading_support="A grown-up may read short directions aloud; meaning must not depend on fluent reading.",
        requested_supports=requested,
    )


_FIRE_COMPOUND_NOUN = r"(?:trucks?|stations?|engines?|fighters?|alarms?|drills?|escapes?|safety|stories?|departments?)"
_FIRE_HAZARD_TARGET = rf"fires?\b(?!\s+{_FIRE_COMPOUND_NOUN}\b)"


_HAZARD_TERM_PATTERNS: dict[str, str] = {
    "heat": (
        r"\b(?:hot\s+(?:water|liquid|surface|glue)|hot[- ]glue\s+guns?|heat|boil|stove|oven|cook|"
        r"microwaves?)\w*\b"
    ),
    "ignition": (
        rf"\b(?:{_FIRE_HAZARD_TARGET}|flames?|candles?|matchsticks?|lighters?|kindling|"
        rf"gas\s+(?:stoves?|burners?|valves?|knobs?|dials?))\b|"
        r"\bignit\w*\b|"
        r"\b(?:light|lighting|strike|striking|rub|rubbing|scrape|scraping|toss|tossing|"
        r"throw|throwing|drop|dropping|fling|flinging|wave|waving|twirl|twirling|"
        r"play\s+with|fool\s+around\s+with|"
        r"box\s+of|book\s+of)\s+(?:the\s+)?match(?:es)?\b"
    ),
    "electrical": (
        r"\b(?:electric(?:al|ity)?|outlets?|wall\s+sockets?|sockets?|wall\s+plugs?|power\s+cords?|"
        r"exposed\s+wires?|mains\s+power)\b"
    ),
    "sharp_tools": (
        r"\b(?:knives?|knife|blades?|scissors|razors?|box\s+cutters?|sharp\s+tools?|"
        r"hole\s+punch(?:es)?|needles?|saws?|chainsaws?|chain\s+saws?|axes?|nail\s*guns?|"
        r"power(?:ed)?\s+nailers?|power\s+drills?|drill(?:ing)?|cut(?:ting)?)\b"
    ),
    "chemicals": r"\b(?:chemicals?|cleaners?|bleach|detergents?|vinegar|baking\s+soda)\b",
    "water": r"\b(?:pools?|baths?|deep\s+water|rivers?|oceans?|lakes?|ponds?|sink\s+full)\b",
    "small_parts": r"\b(?:tiny\s+pieces?|beads?|buttons?|coins?|marbles?)\b",
    # A family being outside is ordinary context. Reserve the veto for a
    # concrete environmental hazard, not a walk or a park picture hunt.
    "outdoors": r"\b(?:streets?|roads?|traffic|cliffs?)\b",
    "body_or_health": r"\b(?:(?:body|heart|breath|medicine|hurt|blood|health|symptom)\w*|pain(?:s|ful|fully|less|lessly)?)\b",
}

_CONDUCTIVE_OBJECT = (
    r"(?:(?:metal(?:lic)?|steel|stainless\s+steel|iron|copper|aluminum|aluminium|tin|brass|silver)\s+"
    r"(?:forks?|knife|knives|spoons?|utensils?|tools?|objects?|rods?|skewers?|chopsticks?|straws?|"
    r"wires?|tweezers?|needles?|pins?|hairpins?|rulers?|chains?|necklaces?|nails?|screws?|bolts?|"
    r"screwdrivers?|scissors)|keys?|coins?|hairpins?|"
    r"aluminum\s+foil|aluminium\s+foil|tin\s+foil|"
    r"paper\s*clips?|forks?|knife|knives|spoons?|utensils?|tweezers?|nails?|wires?|"
    r"fingers?|hands?|water|liquids?|fluids?|juices?|milk|soda|drinks?|beverages?)"
)
_POWERED_APPLIANCE = (
    r"(?:(?:toaster\s+ovens?|toasters?|blenders?|food\s+processors?|electric\s+kettles?|"
    r"coffee\s+makers?|hair\s*dryers?|blow\s+dryers?|curling\s+irons?|space\s+heaters?|"
    r"microwaves?|radios?|laptops?|notebook\s+computers?|phone\s+chargers?|mobile\s+chargers?|"
    r"lamps?|powered\s+appliances?)"
    r"|(?:(?:electrical|wall)\s+)?(?:outlets?|sockets?)|wall\s+plugs?|power\s+points?)"
)
_APPLIANCE_PART = (
    r"(?:coils?|heating\s+elements?|slots?|openings?|interiors?|insides?|bottoms?|"
    r"(?:bread\s+)?compartments?)"
)
_CONDUCTIVE_APPLIANCE_ACTION = (
    r"(?:insert|stick|put|push|drop|poke|reach|place|slide|shove|lay|rest|set|use|touch|tap|hold|"
    r"press|pull|remove|retrieve|grab|pick|lift|yank|extract|fish|scrape|get|take|taking|dig|pry|go|"
    r"force|stuff|pour|spray|splash|spit|dump|wet|spill|squirt|drench|suck|chew|wrap|dip|bite|"
    r"switch|turn|power|plug|start|activate)\w*"
)
_APPLIANCE_ENTRY_RELATION = r"(?:into|inside(?:\s+of)?|through|within|in|out\s+of|from)"


def _normalized_conductive_appliance_text(text: str) -> str:
    lowered = _normalized_safety_text(text)
    alias_word = r"(?:machine|device|appliance|unit|gadget)"
    bindings: list[tuple[str, str]] = []
    alias_patterns = (
        rf"\b(?P<appliance>{_POWERED_APPLIANCE})\b\s+"
        rf"(?:is|means|is\s+called|is\s+known\s+as)\s+"
        rf"(?:the\s+|a\s+|an\s+)?(?P<alias>{alias_word})\b",
        rf"\b(?P<alias>{alias_word})\b"
        rf"(?:\s+(?:beside|near|next\s+to)\s+(?:me|us|you))?\s+"
        rf"(?:is|means|is\s+called|is\s+known\s+as)\s+"
        rf"(?:the\s+|a\s+|an\s+)?(?P<appliance>{_POWERED_APPLIANCE})\b",
        rf"\bcall\s+(?:the\s+|a\s+|an\s+)?(?P<appliance>{_POWERED_APPLIANCE})\b\s+"
        rf"(?:the\s+|a\s+|an\s+)?(?P<alias>{alias_word})\b",
    )
    for pattern in alias_patterns:
        bindings.extend((match.group("appliance"), match.group("alias")) for match in re.finditer(pattern, lowered))
    for appliance, alias in bindings:
        lowered = re.sub(
            rf"\b(?:the\s+|a\s+|an\s+)?{re.escape(alias)}\b",
            appliance,
            lowered,
        )
    lowered = re.sub(
        rf"\b(?:plastic|wood(?:en)?|rubber|silicone|cardboard|paper|toy|pretend)\s+{_CONDUCTIVE_OBJECT}\b",
        " nonconductive-object ",
        lowered,
    )
    lowered = re.sub(
        rf"\b(?:toy|pretend|play|cardboard|paper)\s+{_POWERED_APPLIANCE}\b",
        " unpowered-toy ",
        lowered,
    )
    return re.sub(
        rf"\baway\s+from\s+(?:the\s+|a\s+|an\s+|your\s+|my\s+)?"
        rf"(?P<appliance>{_POWERED_APPLIANCE})\b",
        lambda match: f" safely-separated {match.group('appliance')} ",
        lowered,
    )


def _has_conductive_appliance_pair(text: str) -> bool:
    lowered = _normalized_conductive_appliance_text(text)
    return bool(re.search(rf"\b{_CONDUCTIVE_OBJECT}\b", lowered) and re.search(rf"\b{_POWERED_APPLIANCE}\b", lowered))


def _has_appliance_location(text: str) -> bool:
    return bool(
        re.search(
            rf"\b{_APPLIANCE_ENTRY_RELATION}\s+"
            rf"(?:(?:the|a|an|your|my)\s+)?(?:{_POWERED_APPLIANCE}|"
            rf"(?:{_APPLIANCE_PART}\s+(?:of|on|in)\s+)?{_POWERED_APPLIANCE})\b|"
            rf"\b{_POWERED_APPLIANCE}(?:'s)?\s+{_APPLIANCE_PART}\b|"
            rf"\b{_APPLIANCE_PART}\s+(?:of|on|in)\s+"
            rf"(?:(?:the|a|an|your|my)\s+)?{_POWERED_APPLIANCE}\b|"
            rf"\b(?:a\s+|an\s+|the\s+)?{_APPLIANCE_PART}\s+"
            rf"(?:inside|within|of|on|in)\s+(?:(?:the|a|an|your|my)\s+)?{_POWERED_APPLIANCE}\b",
            text,
        )
    )


def _establishes_conductive_appliance_relation(text: str) -> bool:
    lowered = _normalized_conductive_appliance_text(text)
    return bool(
        re.search(
            rf"\b{_CONDUCTIVE_OBJECT}\b[^.!?;\n]{{0,48}}\b{_APPLIANCE_ENTRY_RELATION}\s+"
            rf"(?:(?:the|a|an|your|my)\s+)?(?:{_POWERED_APPLIANCE}|"
            rf"(?:{_APPLIANCE_PART}\s+(?:of|on|in)\s+)?{_POWERED_APPLIANCE})\b",
            lowered,
        )
        or re.search(
            rf"\b{_APPLIANCE_ENTRY_RELATION}\s+(?:(?:the|a|an|your|my)\s+)?"
            rf"{_POWERED_APPLIANCE}\b[^.!?;\n]{{0,32}}\b{_CONDUCTIVE_OBJECT}\b",
            lowered,
        )
        or re.search(
            rf"\b{_POWERED_APPLIANCE}\b[^.!?;\n]{{0,28}}\b(?:contains?|holds?)\b"
            rf"[^.!?;\n]{{0,28}}"
            rf"\b{_CONDUCTIVE_OBJECT}\b",
            lowered,
        )
    )


def _has_contextual_conductive_appliance_action(
    text: str,
    *,
    paired_entity_context: bool = False,
    established_relation: bool = False,
) -> bool:
    """Resolve an action/location clause against conductive-appliance context."""

    lowered = _normalized_conductive_appliance_text(text)
    if not re.search(rf"\b{_CONDUCTIVE_APPLIANCE_ACTION}\b", lowered):
        return False
    directly_bound = bool(
        re.search(
            rf"\b{_CONDUCTIVE_APPLIANCE_ACTION}\b[^.!?;\n]{{0,36}}\b{_CONDUCTIVE_OBJECT}\b"
            rf"[^.!?;\n]{{0,36}}\b(?:{_APPLIANCE_ENTRY_RELATION}|against|across|onto?)\b"
            rf"\s+(?:(?:the|a|an|your|my)\s+)?(?:{_POWERED_APPLIANCE}|{_APPLIANCE_PART})\b",
            lowered,
        )
        or re.search(
            rf"\b{_CONDUCTIVE_APPLIANCE_ACTION}\b[^.!?;\n]{{0,36}}"
            rf"\b(?:{_POWERED_APPLIANCE}|{_APPLIANCE_PART})\b[^.!?;\n]{{0,36}}"
            rf"\b(?:with|using)\b[^.!?;\n]{{0,18}}\b{_CONDUCTIVE_OBJECT}\b",
            lowered,
        )
    )
    if directly_bound:
        return True
    external_destination = re.search(
        rf"\b(?:it|them|this|that|one|ones|{_CONDUCTIVE_OBJECT})\b\s+"
        rf"(?:beside|near|by|alongside|next\s+to|away\s+from|over|above|below|behind|"
        rf"across\s+from|outside(?:\s+of)?)\s+"
        rf"(?:(?:the|a|an|your|my)\s+)?(?:it|them|this|that|one|ones|{_POWERED_APPLIANCE})\b",
        lowered,
    )
    if external_destination and not re.search(
        rf"\b{_CONDUCTIVE_APPLIANCE_ACTION}\b",
        lowered[external_destination.end() :],
    ):
        return False
    # A resolved referent placed at any named destination other than the
    # appliance or one of its entry parts is ordinary placement. This avoids a
    # brittle whitelist of household nouns while keeping "put it in the slot"
    # and equivalent appliance insertions fail-closed.
    safe_destination = re.search(
        rf"\b(?:it|them|this|that|one|ones|{_CONDUCTIVE_OBJECT})\b\s+"
        rf"(?:(?:from|out\s+of|in|into|inside(?:\s+of)?|on|onto|atop|under|over|above|"
        rf"below|against|beside|near|by|alongside|next\s+to|between|behind|across\s+from|"
        rf"outside(?:\s+of)?)\s+"
        rf"(?P<destination>(?:(?:the|a|an|your|my)\s+)?"
        rf"(?!(?:it|them|this|that|one|ones)\b)"
        rf"[a-z][\w-]*(?:\s+[a-z][\w-]*){{0,3}})|away\b)",
        lowered,
    )
    if (
        safe_destination
        and not re.search(
            rf"\b{_CONDUCTIVE_APPLIANCE_ACTION}\b",
            lowered[safe_destination.end() :],
        )
        and (
            safe_destination.group("destination") is None
            or not re.search(
                rf"\b(?:{_POWERED_APPLIANCE}|{_APPLIANCE_PART})\b",
                safe_destination.group("destination"),
            )
        )
    ):
        return False
    # A pronoun can be the destination of an explicitly external spatial
    # relation.  "Put the fork beside it" is not insertion into the previously
    # named appliance, while "put the fork in it" remains fail-closed.
    external_pronoun_destination = re.search(
        r"\b(?:beside|near|by|alongside|next\s+to|away\s+from|over|above|below|behind|"
        r"across\s+from|outside(?:\s+of)?)\s+"
        r"(?:it|them|this|that|one|ones)\b",
        lowered,
    )
    if external_pronoun_destination and not re.search(
        rf"\b{_CONDUCTIVE_APPLIANCE_ACTION}\b",
        lowered[external_pronoun_destination.end() :],
    ):
        return False
    direct_entry_or_part = re.search(
        rf"\b(?:inside|within|into|through)\b|\bin\s*$|\b{_APPLIANCE_PART}\b",
        lowered,
    )
    has_reference = bool(re.search(rf"\b(?:it|them|this|that|one|ones|toast|crumbs?|{_APPLIANCE_PART})\b", lowered))
    if direct_entry_or_part and has_reference:
        return True
    intrinsic_entry = re.search(
        r"\b(?:insert|stick|put|push|drop|poke|place|slide|shove|lay|rest|set)\w*\b",
        lowered,
    )
    if intrinsic_entry and has_reference:
        return True
    energizing = re.search(
        rf"\b(?:(?:switch|turn|power)\w*\s+(?:(?:it|them|this|that|one|ones)\s+on|"
        rf"on\s+(?:(?:the|a|an)\s+)?{_POWERED_APPLIANCE}|"
        rf"(?:(?:the|a|an)\s+)?{_POWERED_APPLIANCE}\s+on)|"
        rf"plug\w*\s+(?:(?:it|them|this|that|one|ones)|(?:(?:the|a|an)\s+)?"
        rf"{_POWERED_APPLIANCE})\s+in|start\w*|activate\w*)\b",
        lowered,
    )
    if energizing and established_relation:
        return True
    appliance_contents_relation = re.search(
        r"\b(?:toast|crumbs?)\b.{0,30}\b(?:out|from|inside|bottom)\b|"
        r"\b(?:out|from|inside|bottom)\b.{0,30}\b(?:toast|crumbs?)\b",
        lowered,
    )
    unrelated_content_location = re.search(
        rf"\b(?:toast|crumbs?)\b.{{0,24}}\b(?:from|out\s+of)\s+"
        rf"(?:(?:the|a|an|your|my)\s+)?(?!(?:{_POWERED_APPLIANCE}|{_APPLIANCE_PART})\b)"
        r"[a-z][\w-]*",
        lowered,
    )
    if appliance_contents_relation and not unrelated_content_location and has_reference:
        return True
    removal = re.search(r"\b(?:pull|remove|retrieve|take|taking|get|fish|scrape|pry)\w*\b", lowered)
    return bool(
        (paired_entity_context or established_relation)
        and removal
        and re.search(r"\b(?:it|them|this|that|one|ones)\b", lowered)
        and not re.search(r"\bsafely-separated\b", lowered)
    )


def _has_conductive_appliance_action(text: str) -> bool:
    """Recognize direct and cross-clause action relations, not mere co-occurrence."""

    lowered = _normalized_conductive_appliance_text(text)
    if not _has_conductive_appliance_pair(lowered):
        return False
    # Structured/source prose is not guaranteed to put a referent before its
    # pronoun. Resolve the relation over the complete text before inspecting
    # action clauses so "Remove it. A fork is inside the toaster." is safe too.
    clauses = [clause for clause in re.split(r"[.!?;\n]+", lowered) if clause.strip()]
    established_relation = any(_establishes_effective_conductive_relation(clause) for clause in clauses)
    for clause in clauses:
        if _establishes_effective_conductive_relation(clause):
            established_relation = True
        eligible_clause = _mask_ineligible_conductive_actions(
            clause,
            adult_owned_is_safe=True,
        )
        if not re.search(rf"\b{_CONDUCTIVE_APPLIANCE_ACTION}\b", eligible_clause):
            continue
        if _has_contextual_conductive_appliance_action(
            eligible_clause,
            established_relation=established_relation,
        ):
            return True
    return False


_APPLIANCE_WATER_TARGET = (
    r"(?:water|(?:full\s+)?bathtubs?|(?:full\s+)?baths?|showers?|sinks?(?:\s+full\s+of\s+water)?|"
    r"pools?|hot\s+tubs?|toilets?|puddles?|tubs?\s+of\s+water|"
    r"[a-z][\w-]*(?:\s+[a-z][\w-]*){0,2}\s+(?:full\s+)?of\s+water)"
)
_APPLIANCE_WATER_ACTION = r"(?:put|drop|place|lower|submerge|immerse|dip|throw|toss|set|hold|take|use)\w*"
_APPLIANCE_WET_STATE = (
    r"(?:dripping[- ]wet|soaking[- ]wet|dripping|soaking|"
    r"leaking(?:\s+(?:water|liquid|fluid))?|"
    r"leaky|wet|damp|moist|soaked|water[- ]?logged|splashed|"
    r"water[- ]damaged|drenched|underwater)"
)
_APPLIANCE_STATE_MODIFIER = r"(?:(?:[a-z]+ly|very|somewhat|rather|quite|a\s+little|a\s+bit|kind\s+of|sort\s+of)\s+)?"
_APPLIANCE_WATER_RELATION = (
    r"(?:in|into|inside(?:\s+of)?|under|through|within|near|beside|by|alongside|next\s+to|"
    r"with|over)"
)
_APPLIANCE_ENERGIZING_ACTION = (
    r"(?:(?:switch|turn)(?:ed|s|ing)?\s+back\s+on|"
    r"(?:plug|switch|turn|start|restart|reboot|reset|activate|reactivate|energize|reconnect|"
    r"connect|run|use|wake|awaken)\w*|"
    r"(?:continu(?:e|ed|es|ing)|resum(?:e|ed|es|ing)|keep(?:s|ing)?)\s+"
    r"(?:(?:on\s+)?(?:using|operating)|to\s+(?:use|operate))|"
    r"(?:continu(?:e|ed|es|ing)|resum(?:e|ed|es|ing))\s+"
    r"(?:normal\s+)?operation(?:\s+of)?|"
    r"(?:carry|carries|carried|carrying)\s+on"
    r"(?:\s+(?:using|operating|drawing|receiving|consuming)"
    r"(?:\s+(?:mains\s+power|power|electricity|(?:electrical\s+)?current))?)?|"
    r"continu(?:e|ed|es|ing)(?:\s+using)?|resum(?:e|ed|es|ing)(?:\s+using)?|"
    r"(?:boot|fire)(?:ed|s|ing)?(?:\s+up)?|"
    r"(?:power|charge|recharge)(?:d|s|ing)?(?:\s+up)?)"
)
_APPLIANCE_SHUTDOWN_ACTION = (
    r"(?:unplug(?:s)?|unplugg(?:ed|ing)|disconnect(?:ed|s|ing)?|"
    r"deactivat(?:e|ed|es|ing)|de[- ]?energiz(?:e|ed|es|ing)|disabl(?:e|ed|es|ing)|"
    r"depower(?:s|ed|ing)?|"
    r"(?:switch|turn|power)(?:ed|s|ing)?\s+"
    rf"(?:(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_POWERED_APPLIANCE})\s+)?"
    r"(?:off|down)|"
    r"shut(?:s|ting)?\s+down|"
    r"shut(?:s|ting)?\s+off(?:\s+(?:(?:the|its|their|this|that|my|our|your)\s+)?power)?|"
    r"(?:cut|remove|kill|interrupt)(?:s|ed|ing)?\s+"
    r"(?:(?:the|its|their|this|that|my|our|your)\s+)?"
    r"(?:power|electricity|electrical\s+supply)(?:\s+supply)?|"
    r"(?:pull|remove)(?:ed|s|ing)?\s+"
    r"(?:(?:the|its|their|this|that|my|our|your)\s+)?plug|"
    r"stop(?:s|ped|ping)?\s+(?:using|operating)|"
    r"los(?:e|es|ing|t)\s+(?:(?:the|its|their|this|that|my|our|your)\s+)?"
    r"(?:mains\s+power|power|electricity|electrical\s+(?:current|supply)))"
)
_APPLIANCE_CONTINUATION_STATE_ACTION = (
    r"(?:(?:(?:leave|keep)(?:s|ing)?|left|kept)\s+"
    rf"(?:(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE})\s+)?"
    r"(?:the\s+power\s+on|running|working|operating|on|connected(?:\s+to\s+(?:(?:the|a|an)\s+)?"
    r"(?:power|wall|(?:wall\s+)?outlet))?|switched\s+on|powered\s+on|"
    r"electrified|plugged\s+(?:in|into\s+(?:(?:the|a|an)\s+)?(?:wall|(?:wall\s+)?outlet)))|"
    r"(?:let|allow|permit)(?:s|ted|ting|ed|ing)?\s+"
    r"(?:it|this|that|the\s+(?:device|appliance|unit))\s+"
    r"(?:to\s+)?(?:keep|continue|remain|stay)\s+"
    r"(?:running|working|operating|on|connected|energized|electrified|powered|plugged\s+in)|"
    r"(?:have|make)(?:s|d|ing)?\s+"
    r"(?:it|this|that|the\s+(?:device|appliance|unit))\s+"
    r"(?:keep|continue|remain|stay|be)\s+"
    r"(?:running|working|operating|on|connected|energized|electrified|powered|plugged\s+in)|"
    r"(?:(?:do\s+not|don't|never)\s+let|prevent(?:s|ed|ing)?)\s+"
    rf"(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE})\s+"
    r"(?:from\s+)?(?:turn(?:ing)?|switch(?:ing)?|power(?:ing)?|shut(?:ting)?)\s+"
    r"(?:off|down)|"
    r"(?:continue|resume)(?:s|d|ing)?\s+to\s+(?:use|operate)\s+"
    r"(?:it|this|that|the\s+(?:device|appliance|unit))|"
    r"(?:remain|stay)(?:s|ed|ing)?\s+"
    r"(?:running|working|operating|live|active|on|connected|energized|electrified|powered|switched\s+on|powered\s+on|"
    r"plugged\s+in)|"
    r"power\s+(?:left|kept)\s+on|"
    r"(?:ensure|verify|make\s+(?:sure|certain))(?:s|d|ing)?\s+(?:that\s+)?"
    r"(?:it|this|that|the\s+(?:device|appliance|unit))\s+"
    r"(?:remains?|stays?|keeps?|is)\s+"
    r"(?:running|working|operating|on|connected|energized|powered|plugged\s+in)|"
    r"(?:keep|maintain|preserve|continue|sustain)(?:s|ed|ing)?\s+"
    r"(?:electricity|power)\s+(?:flowing\s+)?to\s+"
    r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE}|"
    r"(?:(?:keep(?:s|ing)?|kept|continu(?:e|ed|es|ing)|maintain(?:s|ed|ing)?|"
    r"preserv(?:e|ed|es|ing)|sustain(?:s|ed|ing)?)\s+)?"
    r"(?:deliver(?:s|ed|ing)?|feed(?:s|ing)?|fed|suppl(?:y|ies|ied|ying)|"
    r"send(?:s|ing)?|sent|provid(?:e|es|ed|ing)|giv(?:e|es|ing)|gave|"
    r"rout(?:e|es|ed|ing)|pump(?:s|ed|ing)?)\s+"
    r"(?:mains\s+power|electricity|power|electrical\s+current)\s+to\s+"
    rf"(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE})|"
    r"(?:feed(?:s|ing)?|fed|suppl(?:y|ies|ied|ying)|provid(?:e|es|ed|ing))\s+"
    rf"(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE})\s+"
    r"with\s+(?:mains\s+power|electricity|power|electrical\s+current)|"
    r"(?:giv(?:e|es|ing)|gave)\s+"
    rf"(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE})\s+"
    r"(?:mains\s+power|electricity|power|electrical\s+current)|"
    r"(?:keep|maintain|preserve|continue|sustain)(?:s|ed|ing)?\s+"
    r"(?:electricity|power|electrical\s+current)\s+(?:flowing|running|going)\s+to\s+"
    rf"(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE})|"
    r"(?:maintain|preserve|continue|sustain)(?:s|ed|ing)?\s+"
    r"(?:(?:the|its|their|this|that|my|our|your)\s+)?power\s+to\s+"
    r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE}|"
    r"(?:keep(?:s|ing)?|kept|continu(?:e|ed|es|ing))\s+"
    r"(?:drawing|getting|receiving|using|consuming)\s+"
    r"(?:mains\s+power|power|electricity|(?:electrical\s+)?current)|"
    r"(?:continu(?:e|ed|es|ing)|resum(?:e|ed|es|ing))\s+to\s+"
    r"(?:draw|get|receive|use|consume)\s+"
    r"(?:mains\s+power|power|electricity|(?:electrical\s+)?current)|"
    r"(?:carry|carries|carried|carrying|go|goes|went|going)\s+on\s+"
    r"(?:drawing|getting|receiving|using|consuming|pulling|taking)(?:\s+in)?\s+"
    r"(?:mains\s+power|power|electricity|(?:electrical\s+)?current)|"
    r"(?:keep(?:s|ing)?|kept|continu(?:e|ed|es|ing))\s+"
    r"(?:pulling|taking)(?:\s+in)?\s+"
    r"(?:mains\s+power|power|electricity|(?:electrical\s+)?current)|"
    r"(?:continu(?:e|ed|es|ing)|resum(?:e|ed|es|ing))\s+to\s+"
    r"(?:pull|take)\s+(?:mains\s+power|power|electricity|(?:electrical\s+)?current)|"
    r"(?:draw|get|receive|use|consume|pull|take)\s+"
    r"(?:mains\s+power|power|electricity|(?:electrical\s+)?current)|"
    r"retain(?:s|ed|ing)?\s+(?:(?:the|its|their|this|that|my|our|your)\s+)?"
    r"(?:mains\s+power|power|electricity|(?:electrical\s+)?current)|"
    r"(?:cannot|can't|can\s+not|must\s+not|mustn't|shall\s+not|shan't|"
    r"should\s+not|shouldn't|need\s+not|needn't|need\s+never)\s+"
    r"be\s+without\s+(?:mains\s+power|power|electricity|(?:electrical\s+)?current)|"
    r"(?:cannot|can't|can\s+not|must\s+not|mustn't|shall\s+not|shan't|"
    r"should\s+not|shouldn't|need\s+not|needn't|need\s+never)\s+(?:be|get|remain|stay)\s+"
    r"(?:off|unplugged|disconnected|deactivated|disabled|powered\s+down|"
    r"shut\s+(?:off|down)|switched\s+off|turned\s+off)|"
    r"(?:restor(?:e|es|ed|ing)|return(?:s|ed|ing)?|re[- ]?establish(?:es|ed|ing)?|"
    r"reappl(?:y|ies|ied|ying)|reconnect(?:s|ed|ing)?|reinstate(?:s|d|ing)?|"
    r"resum(?:e|es|ed|ing)|bring(?:s|ing)?|brought)\s+"
    r"(?:(?:the|its|their|this|that|my|our|your)\s+)?"
    r"(?:mains\s+power|power|electricity|(?:electrical\s+)?current|(?:electrical\s+)?power\s+supply|"
    r"electrical\s+supply)(?:\s+back)?\s+(?:to|into)\s+"
    rf"(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE})|"
    r"(?:restor(?:e|es|ed|ing)|return(?:s|ed|ing)?|reappl(?:y|ies|ied|ying)|"
    r"reconnect(?:s|ed|ing)?|reinstate(?:s|d|ing)?|get(?:s|ting)?|"
    r"(?:switch|turn)(?:es|ed|ing)?|put(?:s|ting)?)\s+"
    r"(?:its|their|this|that|my|our|your)\s+"
    r"(?:mains\s+power|power|electricity|(?:electrical\s+)?current)"
    r"(?:\s+back(?:\s+(?:on|into))?)?|"
    r"put(?:s|ting)?\s+(?:mains\s+power|power|electricity|(?:electrical\s+)?current)\s+"
    r"back\s+(?:into|on|to)\s+"
    rf"(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE})|"
    r"(?:(?:cannot|can't|can\s+not|must\s+not|mustn't|shall\s+not|shan't|"
    r"should\s+not|shouldn't|need\s+not|needn't|need\s+never)\s+be\s+"
    r"(?:allowed|permitted)\s+to|(?:is|was)\s+not\s+(?:allowed|permitted)\s+to|"
    r"(?:isn't|wasn't)\s+(?:allowed|permitted)\s+to|"
    r"(?:isn't|wasn't|is\s+not|was\s+not)\s+"
    r"(?:supposed|meant|expected|intended)\s+to|"
    r"(?:isn't|wasn't|is\s+not|was\s+not)\s+to|"
    r"(?:is|was)\s+(?:forbidden|prohibited|disallowed)\s+from)\s+"
    r"(?:be|being|get|getting|manage\s+to)?\s*"
    rf"{_APPLIANCE_SHUTDOWN_ACTION}|"
    r"(?:cannot|can't|can\s+not|must\s+not|mustn't|shall\s+not|shan't|"
    r"should\s+not|shouldn't|need\s+not|needn't|need\s+never)\s+"
    r"(?:(?:be|get|manage\s+to)\s+)?"
    rf"{_APPLIANCE_SHUTDOWN_ACTION}|"
    r"(?:do\s+not|don't|never)\s+(?:let|allow|permit)\s+"
    rf"(?:it|this|that|the\s+(?:device|appliance|unit)|"
    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE})\s+"
    r"(?:to\s+)?(?:go|become|turn)\s+(?:dead|off|unpowered|deenergized|de-energized))"
)
_SAFE_APPLIANCE_REFERENCE_ACTION = re.compile(
    r"(?:draw|sketch|describe|explain|name|label|study|photograph|identify|observe|compare|"
    r"find|count|spell|circle|say|inspect|view|locate|recognize|match|read|"
    r"learn\s+about|read\s+about|look\s+at|point\s+to)"
)


def _wet_appliance_clause_is_represented(clause: str) -> bool:
    """Keep pictured or explicitly transcribed appliance wording non-operative."""

    medium = r"(?:picture|photo|drawing|diagram|illustration|worksheet|page|screen|card)"
    if re.search(
        rf"^\s*(?:in|on)\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?{medium}\b",
        clause,
    ):
        return True
    if re.search(
        rf"^\s*(?:(?:the|a|an|this|that|my|our|your)\s+)?{medium}\s+"
        r"(?:says?|reads?|shows?|depicts?|contains?)\b",
        clause,
    ):
        return True
    if re.search(
        rf"(?:^|:\s*)\s*(?:(?:the|a|an|this|that|my|our|your)\s+)?{medium}\s+"
        r"(?:instructions?|directions?|prompt|exercise)\s*"
        r"(?:(?:is|are|reads?|says?)\s*)?[:—]",
        clause,
    ):
        return True
    if re.search(
        r"(?:^|:\s*)\s*(?:write|copy|quote|transcribe|print|read|label|spell)\s+"
        r"(?:(?:the|this|that|a)\s+)?(?:phrase|sentence|words?|text|caption|label)\s*[:—]",
        clause,
    ):
        return True
    return bool(
        re.search(r"^\s*(?:write|copy|quote|transcribe|print|read|label|spell)\b", clause)
        and (
            re.search(r"[\"'“”‘’]", clause)
            or re.search(
                rf"\b(?:in|on)\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?{medium}\b",
                clause,
            )
        )
    )


def _appliance_continuation_is_directive(clause: str, start: int) -> bool:
    """Authenticate a modal state requirement as a power-continuation direction."""

    if re.fullmatch(r"\s*(?:(?:you|the\s+child)\s+)?feel\s+free\s+to\s+", clause[:start]):
        return True
    modal_prefix = bool(
        re.fullmatch(
            rf"\s*(?:(?:the|a|an|this|that|my|our|your)\s+)?"
            rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?"
            rf"{_POWERED_APPLIANCE}\s+"
            rf"(?:(?:should|must|shall|can|could|may|might)(?:\s+be)?|"
            rf"needs?(?:\s+to)?(?:\s+be)?|had\s+better(?:\s+be)?|"
            rf"ought\s+to(?:\s+be)?|"
            rf"has(?:\s+got)?\s+to(?:\s+be)?|(?:is|was)\s+to(?:\s+be)?|"
            rf"(?:is|was)\s+(?:allowed|permitted|authorized|free|okay|ok)\s+to(?:\s+be)?|"
            rf"(?:is|was)\s+(?:supposed|required|expected|meant|intended|obliged|designated|set)\s+to"
            rf"(?:\s+be)?)\s+",
            clause[:start],
        )
    )
    if modal_prefix:
        return True
    pronoun_modal_prefix = bool(
        re.fullmatch(
            r"\s*(?:it|this|that|the\s+(?:device|appliance|unit))\s+"
            r"(?:(?:should|must|shall|can|could|may|might)(?:\s+be)?|"
            r"needs?(?:\s+to)?(?:\s+be)?|had\s+better(?:\s+be)?|"
            r"ought\s+to(?:\s+be)?|has(?:\s+got)?\s+to(?:\s+be)?|"
            r"(?:is|was)\s+(?:allowed|permitted|authorized|free|okay|ok)\s+to(?:\s+be)?)\s+",
            clause[:start],
        )
    )
    if pronoun_modal_prefix:
        return True
    ensured_appliance_prefix = bool(
        re.fullmatch(
            rf"\s*(?:ensure|verify|make\s+(?:sure|certain))(?:\s+that)?\s+"
            rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
            rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?"
            rf"{_POWERED_APPLIANCE}\s+",
            clause[:start],
        )
    )
    if ensured_appliance_prefix:
        return True
    ensured_pronoun_prefix = bool(
        re.fullmatch(
            r"\s*(?:ensure|verify|make\s+(?:sure|certain))(?:\s+that)?\s+"
            r"(?:it|this|that|the\s+(?:device|appliance|unit))\s+",
            clause[:start],
        )
    )
    if ensured_pronoun_prefix:
        return True
    return bool(
        re.fullmatch(
            rf"\s*(?:(?:the|a|an|this|that|my|our|your)\s+)?"
            rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?"
            rf"{_POWERED_APPLIANCE}\s+",
            clause[:start],
        )
        and re.match(
            r"(?:(?:(?:cannot|can't|can\s+not|must\s+not|mustn't|shall\s+not|shan't|"
            r"should\s+not|shouldn't)\s+"
            r"be\s+(?:allowed|permitted)\s+to|"
            r"(?:is|was)\s+not\s+(?:allowed|permitted)\s+to|"
            r"(?:isn't|wasn't)\s+(?:allowed|permitted)\s+to|"
            r"(?:isn't|wasn't|is\s+not|was\s+not)\s+"
            r"(?:supposed|meant|expected|intended)\s+to|"
            r"(?:isn't|wasn't|is\s+not|was\s+not)\s+to|"
            r"(?:is|was)\s+(?:forbidden|prohibited|disallowed)\s+from)(?:\s+be|\s+being)?|"
            r"(?:cannot|can't|can\s+not|must\s+not|mustn't|shall\s+not|shan't|"
            r"should\s+not|shouldn't|need\s+not|needn't|need\s+never)\s+"
            r"(?:be\s+)?)\b",
            clause[start:],
        )
    )


def _is_clearly_safe_wet_appliance_shutdown_clause(clause: str) -> bool:
    """Recognize complete directions that keep a wet appliance de-energized."""

    wet_appliance = (
        rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        rf"{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+{_POWERED_APPLIANCE}"
    )
    safe_state = (
        r"(?:off|unplugged|disconnected|deactivated|disabled|powered\s+down|"
        r"shut\s+(?:off|down)|switched\s+off|turned\s+off)"
        r"(?:\s+from\s+(?:power|electricity|the\s+(?:wall|outlet|power\s+supply)))?"
        r"(?:\s+until\s+(?:(?:it|this|that|the\s+(?:device|appliance|unit))\s+"
        r"(?:is|becomes?|has\s+become)\s+)?(?:(?:fully|completely)\s+)?dry)?"
    )
    energizing_action = (
        r"(?:(?:switch|turn|power)(?:s|ed|ing)?\s+(?:on|up)|"
        r"plugs?(?:\s+it)?\s+in|plugg(?:ed|ing)(?:\s+it)?\s+in|"
        r"connect(?:s|ed|ing)?|reconnect(?:s|ed|ing)?|start(?:s|ed|ing)?|restart(?:s|ed|ing)?|"
        r"use(?:s|d|ing)?|operat(?:e|es|ed|ing))"
    )
    person = r"(?:anyone|anybody|someone|somebody|people|others?|children|kids?)"
    return bool(
        re.fullmatch(
            rf"\s*(?:please\s+)?(?:"
            rf"{_APPLIANCE_SHUTDOWN_ACTION}\s+{wet_appliance}\s+"
            rf"(?:cannot|can't|can\s+not|must\s+not|mustn't|should\s+not|shouldn't)\s+wait|"
            rf"(?:keep|leave)\s+{wet_appliance}\s+{safe_state}|"
            rf"(?:ensure|verify|make\s+(?:sure|certain))(?:\s+that)?\s+{wet_appliance}\s+"
            rf"(?:remains?|stays?|keeps?|is)\s+{safe_state}|"
            rf"(?:ensure|verify|make\s+(?:sure|certain))(?:\s+that)?\s+"
            rf"(?:no\s+one|nobody)\s+{energizing_action}\s+{wet_appliance}|"
            rf"(?:no\s+one|nobody)\s+"
            rf"(?:(?:should|must|may|can|could|will|would)\s+)?"
            rf"{energizing_action}\s+{wet_appliance}|"
            rf"(?:do\s+not|don't|never)\s+(?:let|allow|permit)\s+{person}\s+"
            rf"(?:to\s+)?{energizing_action}\s+{wet_appliance}|"
            rf"(?:prevent(?:s|ed|ing)?|stop(?:s|ped|ping)?|block(?:s|ed|ing)?|"
            rf"forbid(?:s|ding|den)?)\s+"
            rf"{person}\s+(?:from\s+)?"
            rf"{energizing_action}\s+{wet_appliance}"
            rf")\s*",
            clause,
        )
    )


def _shutdown_has_dangerous_negative_polarity(clause: str, shutdown_start: int) -> bool:
    """Bind delay, discouragement, or omission language to a shutdown action."""

    prefix = clause[:shutdown_start]
    suffix = clause[shutdown_start:]
    adult = rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_ADULT_OWNER}"

    def positive(pattern: str) -> bool:
        match = re.search(pattern, prefix)
        return bool(match and not _action_is_negated(prefix, match.start()))

    if re.search(
        r"\b(?:do\s+not|don't)\s+(?:(?:be|get)\s+in\s+(?:a|any)\s+hurry|"
        r"(?:feel|become|seem)\s+(?:rushed|hurried|pressed))\s+(?:to\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:do\s+not|don't)\s+feel\s+(?:any\s+)?"
        r"(?:urgency|pressure|haste|need)\s+(?:(?:to|about)\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE}\s+"
        r"(?:does|did)\s+not\s+need\s+(?:to\s+(?:be\s+)?)?$",
        prefix,
    ):
        return True
    if re.search(r"\b(?:do\s+not|don't)\s+worry\s+about\s+$", prefix):
        return True
    if positive(
        r"\b(?:take(?:s|n|ing)?|took)\s+(?:"
        r"(?:(?:all|as\s+much)\s+)?(?:your|our|my|the)\s+time|"
        r"(?:your|our|my|the)\s+(?:own\s+)?sweet\s+time|"
        r"(?:all\s+)?the\s+time\s+(?:(?:you|we|i)\s+(?:need|want|like)|necessary|required)|"
        r"whatever\s+time\s+(?:you|we|i)\s+(?:need|want|like)|"
        r"(?:as\s+long\s+as|however\s+long)\s+(?:you|we|i)\s+(?:need|want|like))\s+(?:to\s+)?$"
    ):
        return True
    if re.search(
        rf"\b(?:there(?:\s+(?:is|was)|'s)\s+)?no\s+(?:immediate\s+)?"
        rf"(?:need|rush|hurry|urgency|pressure|haste|deadline|time\s+limit)\s+"
        rf"(?:(?:for\s+{adult}\s+)?to\s+|(?:about|for|on)\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"\bthere(?:\s+(?:is|was)|'s)\s+no\s+reason\s+to\s+"
        r"(?:rush|hurry)\s+(?:to\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"\bthere(?:\s+(?:is|was)|'s)\s+no\s+(?:need|reason)\s+to\s+"
        r"(?:rush|hurry)\s+(?:to\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"\bthere(?:\s+(?:is|was)|'s)\s+no\s+(?:need|reason)\s+for\s+"
        r"(?:haste|speed|urgency|rushing|hurrying)\s+(?:when|while|in)\s+$",
        prefix,
    ):
        return True
    if re.search(
        r"\bthere(?:\s+(?:is|was)|'s)\s+no\s+(?:need|reason)\s+to\s+"
        r"(?:be|get|act|move)\s+(?:quick(?:ly)?|fast|prompt(?:ly)?|swift(?:ly)?)\s+"
        r"(?:about|to|before)\s+$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:you|we|i)\s+(?:need\s+not|needn't|do\s+not\s+need\s+to|don't\s+need\s+to)\s+"
        r"(?:be|get|act|move)\s+(?:quick(?:ly)?|fast|prompt(?:ly)?|swift(?:ly)?)\s+"
        r"(?:about|to|before)\s+$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE}\s+"
        r"(?:does|did)\s+not\s+require\s+(?:an?\s+)?(?:immediate|prompt|urgent)\s+$",
        prefix,
    ):
        return True
    if re.search(
        r"\bthere(?:\s+(?:is|was)|'s)\s+(?:plenty|lots?)\s+of\s+time\s+"
        r"(?:before|to|for)\s+$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:you|we|i)(?:\s+have|'ve|\s+had|'d)\s+"
        r"(?:all\s+(?:day|night|week|weekend)|(?:plenty|lots?)\s+of\s+time|"
        r"(?:unlimited|unbounded|ample)\s+time)\s+"
        r"(?:before|to|for)\s+$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:at\s+(?:your|our|my|the)\s+convenience|"
        r"in\s+(?:your|our|my|the)\s+own\s+time|"
        r"at\s+(?:your|our|my|the)\s+leisure|"
        r"when(?:ever)?\s+(?:it\s+is\s+)?convenient|"
        r"when(?:ever)?\s+(?:you|we|i)\s+(?:have|has|get|find)\s+"
        r"(?:(?:a|the|some|enough|free)\s+)?(?:chance|moment|time)|"
        r"when(?:ever)?\s+(?:you|we|i)\s+(?:get|gets|got)\s+around\s+to\s+it|"
        r"when(?:ever)?\s+(?:you|we|i)\s+feel\s+like\s+it|"
        r"eventually|later|sometime|in\s+due\s+course)\b",
        suffix,
    ):
        return True
    if re.search(
        r"\b(?:it|this|that|the\s+(?:device|appliance|unit))\s+"
        r"(?:can|could|may|might)\s+wait\b",
        suffix,
    ):
        return True
    if positive(r"\b(?:postpone|delay|defer|stall|put\s+off)(?:s|ed|ing)?\s+$"):
        return True
    if positive(
        r"\b(?:hold(?:s|ing)?\s+off\s+(?:on|until)|wait(?:s|ed|ing)?\s+"
        r"(?:(?:(?:a|some|more|little)\s+(?:while|time)\s+)?"
        r"(?:before(?:\s+being)?|to)|until\s+"
        r"(?:later|tomorrow|tonight|next\s+(?:morning|afternoon|evening|day|week))\s+to)|"
        r"leave(?:s|ing)?)\s+$"
    ):
        return True
    if positive(r"\b(?:schedule|calendar|book)(?:s|d|ed|ing)?\s+$") and re.search(
        r"\b(?:for|until|after)\s+"
        r"(?:later|tomorrow|tonight|next\s+(?:morning|afternoon|evening|day|week)|"
        r"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:minutes?|hours?|days?|weeks?))\b|"
        r"\b(?:later|tomorrow|next\s+(?:morning|afternoon|evening|day|week))\b",
        suffix,
    ):
        return True
    if re.search(
        rf"\b(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE}\s+"
        r"(?:can|could|may|might)\s+wait\s+to\s+(?:be\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:it\s+)?(?:would|will|may|might|could)\s+be\s+"
        r"(?:best|better|wise|preferable|safer)\s+not\s+to\s+$",
        prefix,
    ):
        return True
    if positive(
        rf"\b(?:recommend|suggest|advise|urge|request)(?:s|ed|ing)?\s+"
        rf"(?:that\s+)?{adult}\s+(?:do\s+not|does\s+not|not(?:\s+to)?|never(?:\s+to)?)\s+$",
    ):
        return True
    if positive(
        rf"\b(?:discourage|dissuade)(?:s|ed|ing)?\s+{adult}\s+from\s+$",
    ):
        return True
    if positive(
        rf"\b(?:advise|recommend|urge|warn)(?:s|d|ed|ing)?\s+{adult}\s+against\s+$",
    ):
        return True
    return positive(
        rf"\b(?:talk|talks|talked|talking)\s+{adult}\s+out\s+of\s+$",
    )


def _has_powered_appliance_water_state(text: str) -> bool:
    """Find a wet or water-adjacent appliance state independent of field order."""

    if (
        re.search(
            rf"\b(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+{_POWERED_APPLIANCE}|"
            rf"{_POWERED_APPLIANCE}\b"
            rf"[^,.!?;\n]{{0,24}}\b(?:is|was|got|gets?|became|becomes?|looks?|seems?)\s+"
            rf"{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE})\b",
            text,
        )
        or re.search(
            rf"\b{_POWERED_APPLIANCE}\b[^,.!?;\n]{{0,24}}\b"
            rf"(?:(?:has|had|contains?|contained)\s+"
            rf"(?:(?:some|a\s+little|a\s+bit\s+of)\s+)?|"
            rf"(?:is|was)\s+(?:completely\s+)?full\s+of\s+)"
            rf"(?:moisture|condensation)"
            rf"(?:\s+inside(?:\s+it)?)?\b",
            text,
        )
        or re.search(
            rf"\b(?:moisture|condensation)\b[^,.!?;\n]{{0,32}}\b"
            rf"(?:formed|developed|collected|built\s+up|appeared|is|was)\s+"
            rf"(?:inside|within)\s+"
            rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_POWERED_APPLIANCE}\b",
            text,
        )
        or re.search(
            rf"\b{_POWERED_APPLIANCE}\b[^,.!?;\n]{{0,24}}\b"
            rf"(?:is|was|got|gets?|has\s+been|had\s+been|has\s+fallen|had\s+fallen|fell|falls?|"
            rf"landed|lands?|went|goes?|dropped|drops?|tumbled|tumbles?|knocked(?:\s+over)?|"
            rf"sits?|sat|ended\s+up)\s+"
            rf"{_APPLIANCE_STATE_MODIFIER}"
            rf"(?:(?:submerged|immersed|sitting|seated|standing|lying|covered|dropped|placed|"
            rf"knocked(?:\s+over)?|tipped|fallen|floating|floated|tumbled)\s+)?"
            rf"{_APPLIANCE_WATER_RELATION}\s+"
            rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_APPLIANCE_WATER_TARGET}\b",
            text,
        )
        or re.search(
            rf"\b{_APPLIANCE_WATER_TARGET}\b[^,.!?;\n]{{0,24}}\b"
            rf"(?:got|gets?|went|goes?|is|was)\s+inside(?:\s+of)?\s+"
            rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_POWERED_APPLIANCE}\b",
            text,
        )
        or re.search(
            rf"\b{_POWERED_APPLIANCE}\b[^,.!?;\n]{{0,24}}\bhas\s+"
            rf"(?:water|liquid|fluid)\s+(?:dripping|leaking|running)\s+from\s+"
            rf"(?:it|this|that|the\s+(?:device|appliance|unit))\b",
            text,
        )
        or re.search(
            rf"\b{_POWERED_APPLIANCE}\b[^,.!?;\n]{{0,24}}\b"
            rf"(?:leaks?|leaked|drips?|dripped|seeps?|seeped)\s+"
            rf"(?:water|liquid|fluid)\b",
            text,
        )
        or re.search(
            rf"\b{_POWERED_APPLIANCE}\b[^,.!?;\n]{{0,24}}\b"
            rf"(?:is|was|got|gets?|has\s+been|had\s+been)\s+"
            rf"(?:sprayed|splashed|drenched|soaked|covered)\s+with\s+"
            rf"(?:water|liquid|fluid)\b",
            text,
        )
    ):
        return True
    for clause in re.finditer(r"[^.!?;\n]+", text):
        clause_text = clause.group()
        for action in re.finditer(r"\b(?:spill|splash|pour|dump|spray|squirt|drench)\w*\b", clause_text):
            if _action_is_negated(clause_text, action.start()):
                continue
            if re.match(
                rf"^[^,.!?;\n]{{0,24}}\b(?:water|liquids?|fluids?|juices?|milk|soda|drinks?|"
                rf"beverages?)\b[^,.!?;\n]{{0,24}}\b(?:on|onto|over|into)\s+"
                rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_POWERED_APPLIANCE}\b",
                clause_text[action.end() :],
            ):
                return True
    return False


def _named_appliance_has_water_state(text: str, appliance: str) -> bool:
    """Bind a wet state to one named appliance rather than a whole clause."""

    escaped = re.escape(appliance)
    return bool(
        re.search(rf"\b{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+{escaped}\b", text)
        or re.search(
            rf"\b{escaped}\b[^,.!?;\n]{{0,32}}\b"
            rf"(?:is|was|got|gets?|became|becomes?|looks?|seems?)\s+"
            rf"{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\b",
            text,
        )
        or re.search(
            rf"\b{escaped}\b[^,.!?;\n]{{0,32}}\b(?:has|had|is|was)\s+"
            rf"(?:(?:completely\s+)?full\s+of\s+)?"
            rf"(?:water|liquid|fluid|moisture|condensation|leaking|dripping)",
            text,
        )
        or re.search(
            rf"\b(?:moisture|condensation)\b[^,.!?;\n]{{0,32}}\b"
            rf"(?:formed|developed|collected|built\s+up|appeared|is|was)\s+"
            rf"(?:inside|within)\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?"
            rf"{escaped}\b",
            text,
        )
        or re.search(
            rf"\b{escaped}\b[^,.!?;\n]{{0,32}}\b"
            rf"(?:leaks?|leaked|drips?|dripped|seeps?|seeped)\s+"
            rf"(?:water|liquid|fluid)\b",
            text,
        )
        or re.search(
            rf"\b{escaped}\b[^,.!?;\n]{{0,32}}\b"
            rf"(?:is|was|got|gets?|has\s+been|had\s+been)\s+"
            rf"(?:sprayed|splashed|drenched|soaked|covered)\s+with\s+"
            rf"(?:water|liquid|fluid)\b",
            text,
        )
        or re.search(
            rf"\b{escaped}\b[^,.!?;\n]{{0,40}}\b{_APPLIANCE_WATER_RELATION}\s+"
            rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_APPLIANCE_WATER_TARGET}\b",
            text,
        )
        or re.search(
            rf"\b(?:water|liquids?|fluids?|juices?|milk|soda|drinks?|beverages?)\b"
            rf"[^,.!?;\n]{{0,32}}\b(?:on|onto|over|into|inside(?:\s+of)?)\s+"
            rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{escaped}\b",
            text,
        )
    )


def _has_powered_appliance_water_action(text: str) -> bool:
    """Recognize the severe inverse relation: a powered appliance entering water."""

    lowered = _normalized_conductive_appliance_text(text)
    # Remove complete represented/transcription sentences before relation
    # binding. Keep colon as an operational clause boundary elsewhere: model
    # output often prefixes the original question with a label, and collapsing
    # that boundary can invert a genuinely protective shutdown instruction.
    lowered = ". ".join(
        segment.group()
        for segment in re.finditer(r"[^.!?;\n]+", lowered)
        if not _wet_appliance_clause_is_represented(segment.group())
    )
    if not re.search(rf"\b{_POWERED_APPLIANCE}\b", lowered):
        return False
    # Structured response, visual, and artifact fields have no trustworthy
    # narrative order. Establish the dangerous state over the complete value
    # before evaluating any energizing action.
    wet_state_clauses = [
        clause.group()
        for clause in re.finditer(r"[^.!?;:\n—]+", lowered)
        if not _wet_appliance_clause_is_represented(clause.group())
        and _has_powered_appliance_water_state(clause.group())
    ]
    wet_appliance_active = bool(wet_state_clauses)
    if wet_appliance_active:
        wet_names = {
            appliance.group()
            for state_clause in wet_state_clauses
            for appliance in re.finditer(rf"\b{_POWERED_APPLIANCE}\b", state_clause)
            if _named_appliance_has_water_state(state_clause, appliance.group())
        }
        for clause_match in re.finditer(r"[^.!?;:\n—]+", lowered):
            clause = clause_match.group()
            for shutdown in re.finditer(rf"\b{_APPLIANCE_SHUTDOWN_ACTION}\b", clause):
                if not _shutdown_has_dangerous_negative_polarity(clause, shutdown.start()):
                    continue
                reference_window = clause[shutdown.start() : shutdown.end() + 52]
                names_wet_appliance = any(
                    appliance in wet_names for appliance in re.findall(rf"\b{_POWERED_APPLIANCE}\b", reference_window)
                )
                unambiguous_pronoun = len(wet_names) == 1 and bool(
                    re.search(r"\b(?:it|this|that|the\s+(?:device|appliance|unit))\b", reference_window)
                )
                if names_wet_appliance or unambiguous_pronoun:
                    return True

        def nearest_prior_entity_is_wet_appliance(absolute_action_start: int) -> bool:
            """Resolve a pronoun to the nearest relevant appliance or conductive object."""

            prior_appliances = list(re.finditer(rf"\b{_POWERED_APPLIANCE}\b", lowered[:absolute_action_start]))
            if not prior_appliances:
                # Structured fields have no trustworthy order. Without a
                # nearer concrete referent, a known wet appliance stays active.
                return True
            prior_conductive_objects = [
                mention
                for mention in re.finditer(rf"\b{_CONDUCTIVE_OBJECT}\b", lowered[:absolute_action_start])
                if not re.fullmatch(
                    r"(?:water|liquids?|fluids?|juices?|milk|soda|drinks?|beverages?)",
                    mention.group(),
                )
            ]
            nearest_appliance = prior_appliances[-1]
            if prior_conductive_objects and prior_conductive_objects[-1].end() > nearest_appliance.end():
                return False
            introduced_entities = [
                match
                for match in re.finditer(
                    r"\b(?:here|there)\s+(?:is|are)\s+"
                    r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                    r"[a-z][\w-]*(?:\s+[a-z][\w-]*){0,3}\b",
                    lowered[:absolute_action_start],
                )
                if not re.match(
                    r"there\s+(?:is|are)\s+no\s+(?:immediate\s+)?(?:need|rush|hurry)\b",
                    match.group(),
                )
            ]
            if introduced_entities and introduced_entities[-1].end() > nearest_appliance.end():
                return False
            return any(
                _named_appliance_has_water_state(state_clause, nearest_appliance.group())
                for state_clause in wet_state_clauses
            )

        def references_wet_appliance(
            clause_match: re.Match[str],
            action: re.Match[str],
        ) -> bool:
            clause = clause_match.group()
            reference_window = clause[action.start() : action.end() + 52]
            for named in re.finditer(rf"\b{_POWERED_APPLIANCE}\b", reference_window):
                appliance = named.group()
                if any(_named_appliance_has_water_state(state_clause, appliance) for state_clause in wet_state_clauses):
                    return True
            if re.search(r"\b(?:it|this|that|the\s+(?:device|appliance|unit))\b", reference_window):
                absolute_action_start = clause_match.start() + action.start()
                return nearest_prior_entity_is_wet_appliance(absolute_action_start)
            # A bare continuation such as "continue normal operation" binds
            # only when exactly one wet appliance type is established.
            return len(wet_names) == 1

        for clause_match in re.finditer(r"[^.!?;:\n—]+", lowered):
            clause = clause_match.group()
            if _wet_appliance_clause_is_represented(clause):
                continue
            if _is_clearly_safe_wet_appliance_shutdown_clause(clause):
                continue
            for shutdown in re.finditer(rf"\b{_APPLIANCE_SHUTDOWN_ACTION}\b", clause):
                explicit_negative_direction = (
                    _shutdown_has_dangerous_negative_polarity(clause, shutdown.start())
                    or re.search(
                        r"\b(?:do\s+not|don't|never|(?:may|must|should|need)\s+not(?:\s+(?:be|get))?|"
                        r"(?:do|does|did)\s+not\s+(?:need|have)\s+to(?:\s+(?:be|get))?|"
                        r"(?:don't|doesn't|didn't)\s+(?:need|have)\s+to(?:\s+(?:be|get))?|"
                        r"(?:can|could|may|might)\s+skip(?:\s+(?:ever|just|simply))?|"
                        r"(?:(?:you|the\s+child)\s+)?(?:can|could|may|might)\s+"
                        r"(?:omit|forgo|forego)(?:\s+(?:ever|just|simply))?|"
                        r"(?:(?:you|the\s+child)\s+)?(?:please\s+)?"
                        r"(?:skip|omit|forgo|forego)(?:\s+(?:ever|just|simply))?|"
                        r"(?:(?:you|the\s+child)\s+)?feel\s+free\s+(?:to\s+)?not\s+to|"
                        r"(?:(?:it|this|that)\s+)?(?:is|was)\s+"
                        r"(?:okay|ok|fine|acceptable|permitted)\s+not\s+to|"
                        r"(?:do\s+not|don't|never)\s+bother(?:\s+to)?|"
                        r"(?:(?:it|this|that)\s+)?(?:is|was)\s+optional\s+to|"
                        rf"(?:(?:it|this|that)\s+)?(?:isn't|aren't|wasn't|weren't|"
                        rf"(?:is|are|was|were)\s+not)\s+"
                        rf"(?:necessary|required|obligatory|mandatory|essential)\s+"
                        rf"(?:for\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                        rf"(?:{_ADULT_OWNER}|{_CHILD_OWNER}|anyone|anybody|someone|somebody)\s+)?"
                        r"to(?:\s+(?:be|get))?|"
                        r"there\s+(?:is|was)\s+no\s+"
                        r"(?:need|reason|obligation|requirement|duty)\s+to(?:\s+(?:be|get))?|"
                        rf"(?:(?:(?:the|a|an|this|that|my|our|your)\s+)?{_ADULT_OWNER}\s+)?"
                        r"(?:can|could|may|might)\s+(?:skip|omit|forgo|forego)|"
                        r"(?:no\s+one|nobody)\s+(?:needs?|has)\s+to(?:\s+(?:be|get))?|"
                        r"(?:isn't|aren't|wasn't|weren't|(?:is|are|was|were)\s+not)\s+"
                        r"(?:required|supposed|expected|meant|obliged)\s+to"
                        r"(?:\s+(?:be|get))?|"
                        r"mustn't(?:\s+be)?|shouldn't(?:\s+be)?|shan't(?:\s+be)?|"
                        r"needn't(?:\s+(?:be|get))?|"
                        r"ought\s+not(?:\s+to)?(?:\s+be)?|"
                        r"(?:may|must|should)\s+never(?:\s+be)?|avoid|"
                        r"(?:is|was)\s+not\s+to\s+be|refrain\s+from)\s+"
                        r"(?:(?:ever|just|simply|now)\s+)*$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        rf"\b(?:ask|tell|instruct|urge|have|remind|prompt|order|direct|encourage|persuade|"
                        rf"advise|suggest|convince|recommend|request|command|warn|beg)\s+"
                        rf"(?:that\s+)?"
                        rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_ADULT_OWNER}\s+"
                        r"(?:not\s+to|never\s+to)\s+$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        rf"\b(?:ask|tell|instruct|urge|remind|prompt|order|direct|encourage|persuade|"
                        rf"advise|suggest|convince|recommend|request|command|warn|beg)\s+"
                        rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_ADULT_OWNER}\s+"
                        r"(?:that\s+)?(?:he|she|they)\s+"
                        r"(?:need\s+not|needn't|should\s+not|shouldn't|must\s+not|mustn't|"
                        r"does\s+not\s+need\s+to|doesn't\s+need\s+to)\s+$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        rf"\b(?:do\s+not|don't|never)\s+(?:allow|let|permit)\s+"
                        rf"(?:(?:anyone|anybody|someone|somebody|people|others?|children|kids?|"
                        rf"{_ADULT_OWNER})\s+)?(?:to\s+)?(?:the\s+)?"
                        rf"(?:(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?"
                        rf"{_POWERED_APPLIANCE}\s+)?(?:to\s+)?(?:be\s+)?$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        r"\b(?:ensure|make\s+(?:sure|certain))(?:\s+that)?\s+"
                        r"(?:no\s+one|nobody)\s+$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        rf"\b(?:ensure|make\s+(?:sure|certain))(?:\s+that)?\s+"
                        rf"(?:{_ADULT_OWNER}|anyone|anybody|someone|somebody)\s+"
                        r"(?:cannot|can't|can\s+not|must\s+not|mustn't|should\s+not|shouldn't|"
                        r"(?:is|was)\s+unable\s+to|(?:isn't|wasn't)\s+able\s+to)\s+"
                        r"(?:be\s+)?$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        rf"\b(?:{_ADULT_OWNER}|anyone|anybody|someone|somebody)\s+"
                        r"(?:cannot|can't|can\s+not|could\s+not|couldn't|"
                        r"(?:is|was)\s+unable\s+to|(?:isn't|wasn't)\s+able\s+to)\s+"
                        r"(?:manage\s+to\s+)?$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        r"\b(?:ensure|make\s+(?:sure|certain))(?:\s+that)?\s+"
                        r"(?:it|this|that|the\s+(?:device|appliance|unit))\s+"
                        r"(?:does\s+not|doesn't|cannot|can't|can\s+not|must\s+not|mustn't|"
                        r"should\s+not|shouldn't)\s+$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        rf"\b(?:ensure|make\s+(?:sure|certain))(?:\s+that)?\s+"
                        rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                        rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?"
                        rf"{_POWERED_APPLIANCE}\s+"
                        r"(?:does\s+not|doesn't|cannot|can't|can\s+not|must\s+not|mustn't|"
                        r"should\s+not|shouldn't)\s+$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        r"\b(?:do\s+not|don't|never)\s+(?:allow|let|permit)\s+"
                        r"(?:anyone|anybody|someone|somebody|people|others?|children|kids?)\s+"
                        r"(?:to\s+)?$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        r"\b(?:no\s+one|nobody)\s+"
                        r"(?:(?:should|must|may|can|could|will|would)\s+)?$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        r"\b(?:no\s+one|nobody)\s+(?:is|was)\s+"
                        r"(?:allowed|permitted)\s+to(?:\s+be)?\s+$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        rf"\b(?:prevent(?:s|ed|ing)?|stop(?:s|ped|ping)?|block(?:s|ed|ing)?|"
                        rf"keep(?:s|ing)?|kept)\s+"
                        rf"(?:anyone|anybody|someone|somebody|people|others?|children|kids?|"
                        rf"{_ADULT_OWNER})\s+"
                        r"from\s+$",
                        clause[: shutdown.start()],
                    )
                    or re.search(
                        rf"\b(?:refuse|refuses|refused|decline|declines|declined)\s+to\s+"
                        rf"(?:allow|let|permit)\s+(?:{_ADULT_OWNER}|anyone|anybody|someone|somebody)\s+"
                        r"(?:to\s+)?$",
                        clause[: shutdown.start()],
                    )
                    or re.fullmatch(
                        rf"\s+(?:(?:it|this|that)|"
                        rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                        rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?"
                        rf"{_POWERED_APPLIANCE})\s+"
                        rf"(?:is|was|seems?|appears?)\s+"
                        rf"(?:optional|unnecessary|not\s+(?:needed|required|necessary|obligatory|mandatory))\s*",
                        clause[shutdown.end() :],
                    )
                )
                negated_shutdown_prevention = re.search(
                    rf"\b(?:do\s+not|don't|never)\s+"
                    rf"(?:prevent|stop|block|keep)\s+"
                    rf"(?:anyone|anybody|someone|somebody|people|others?|children|kids?|"
                    rf"{_ADULT_OWNER})\s+from\s+$",
                    clause[: shutdown.start()],
                )
                if not explicit_negative_direction:
                    continue
                if negated_shutdown_prevention:
                    continue
                if references_wet_appliance(clause_match, shutdown):
                    return True
            if re.search(
                rf"\b(?:(?:(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?"
                rf"{_POWERED_APPLIANCE})(?:'s|s')|its|their|the\s+(?:device|appliance|unit)'s)\s+"
                rf"(?:mains\s+power|power|electricity|(?:electrical\s+)?current)\s+"
                rf"(?:must|should|shall|ought\s+to|needs?\s+to|has\s+to)\s+be\s+"
                rf"(?:restored|returned|reapplied|reconnected|reinstated|resumed|put\s+back\s+on)\b",
                clause,
            ):
                return True
            if re.search(
                rf"\b(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?"
                rf"{_POWERED_APPLIANCE}\s+needs?(?:\s+to\s+have)?\s+"
                rf"(?:its|their|this|that|my|our|your)\s+"
                rf"(?:mains\s+power|power|electricity|(?:electrical\s+)?current)\s+"
                rf"(?:restored|returned|reapplied|reconnected|reinstated|resumed|put\s+back\s+on)\b",
                clause,
            ):
                return True
            for continuation in re.finditer(
                rf"\b{_APPLIANCE_CONTINUATION_STATE_ACTION}\b",
                clause,
            ):
                if _action_is_negated(clause, continuation.start()):
                    continue
                adult_directive = bool(
                    re.search(
                        rf"\b(?:ask|tell|instruct|urge|have|make|remind|prompt|order|direct|encourage|persuade|"
                        rf"advise|suggest|convince|recommend|request|command|warn|beg|"
                        rf"make\s+(?:sure|certain))\s+"
                        rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_ADULT_OWNER}\s+"
                        r"(?:to\s+)?$",
                        clause[: continuation.start()],
                    )
                    or re.search(
                        rf"\b(?:(?:the|a|an|this|that|my|our|your)\s+)?{_ADULT_OWNER}\s+"
                        r"(?:can|could|may|might|will|would|should|must|shall|needs?\s+to|"
                        r"has\s+to|ought\s+to|had\s+better)\s+$",
                        clause[: continuation.start()],
                    )
                )
                if _action_owner(clause, continuation.start(), continuation.end()) == "adult" and not adult_directive:
                    continue
                if not (
                    _action_is_imperative(clause, continuation.start())
                    or _child_directly_owns_action(clause, continuation.start())
                    or _appliance_continuation_is_directive(clause, continuation.start())
                    or adult_directive
                ):
                    continue
                if references_wet_appliance(clause_match, continuation):
                    return True
    if wet_appliance_active:
        for clause in (match.group() for match in re.finditer(r"[^.!?;:\n—]+", lowered)):
            if _wet_appliance_clause_is_represented(clause):
                continue
            if _is_clearly_safe_wet_appliance_shutdown_clause(clause):
                continue
            for appliance_match in re.finditer(rf"\b{_POWERED_APPLIANCE}\b", clause):
                appliance = appliance_match.group()
                if not any(
                    _named_appliance_has_water_state(state_clause, appliance) for state_clause in wet_state_clauses
                ):
                    continue
                if re.search(
                    r"^\s+(?:remains?|stays?|keeps?|is|should\s+(?:remain|stay|be)|"
                    r"must\s+(?:remain|stay|be))\s+"
                    r"(?:off|unplugged|disconnected|deactivated|disabled|powered\s+down|"
                    r"shut\s+(?:off|down))\b",
                    clause[appliance_match.end() :],
                ):
                    continue
                action = re.search(
                    r"\b(?P<verb>(?!(?:the|a|an|this|that|my|our|your)\b)"
                    r"[a-z]+(?:\s+(?:back|off|on|up|down|over|with|into|in|at|about))?)\s+"
                    r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?$",
                    clause[: appliance_match.start()],
                )
                if not action:
                    continue
                if _SAFE_APPLIANCE_REFERENCE_ACTION.fullmatch(action.group("verb")):
                    continue
                if _action_is_negated(clause, action.start()):
                    continue
                if _action_owner(clause, action.start(), action.end()) == "adult":
                    continue
                if _action_is_imperative(clause, action.start()) or _child_directly_owns_action(clause, action.start()):
                    return True
    for wet_clause_match in re.finditer(r"[^.!?;:\n—]+", lowered):
        wet_clause = wet_clause_match.group()
        if _wet_appliance_clause_is_represented(wet_clause):
            continue
        if _is_clearly_safe_wet_appliance_shutdown_clause(wet_clause):
            continue
        for action in re.finditer(rf"\b{_APPLIANCE_ENERGIZING_ACTION}\b", wet_clause):
            if _action_is_negated(wet_clause, action.start()):
                continue
            if _action_owner(wet_clause, action.start(), action.end()) == "adult":
                continue
            if (
                wet_appliance_active
                and re.search(r"\b(?:operation|operating)\b", action.group())
                and not wet_clause[action.end() :].strip()
                and (
                    _action_is_imperative(wet_clause, action.start())
                    or _child_directly_owns_action(wet_clause, action.start())
                )
            ):
                return True
            trailing = wet_clause[action.end() :]
            if re.match(
                rf"^\s+(?:(?:on|in)\s+)?(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                rf"{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+{_POWERED_APPLIANCE}\b",
                trailing,
            ):
                return True
            if wet_appliance_active:
                referenced_appliance = re.match(
                    rf"^\s+(?:(?:on|in)\s+)?(?:(?P<pronoun>it|this|that)|"
                    rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                    rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?"
                    rf"(?P<appliance>{_POWERED_APPLIANCE}))\b"
                    rf"(?:[^,.!?;\n]{{0,16}}\b(?:on|in)\b)?",
                    trailing,
                )
                if referenced_appliance and referenced_appliance.group("pronoun"):
                    absolute_action_start = wet_clause_match.start() + action.start()
                    if nearest_prior_entity_is_wet_appliance(absolute_action_start):
                        return True
                    continue
                if referenced_appliance and referenced_appliance.group("appliance"):
                    appliance = referenced_appliance.group("appliance")
                    if any(
                        _named_appliance_has_water_state(state_clause, appliance) for state_clause in wet_state_clauses
                    ):
                        return True
            if re.match(
                rf"^\s+(?:(?:on|in)\s+)?"
                rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                rf"(?:{_APPLIANCE_STATE_MODIFIER}{_APPLIANCE_WET_STATE}\s+)?{_POWERED_APPLIANCE}\b"
                rf"(?:\s+(?:on|in))?[^,.!?;\n]{{0,28}}\b{_APPLIANCE_WATER_RELATION}\s+"
                rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_APPLIANCE_WATER_TARGET}\b",
                trailing,
            ):
                return True
    if not re.search(rf"\b{_APPLIANCE_WATER_TARGET}\b", lowered):
        return False
    for clause_match in re.finditer(r"[^.!?;\n]+", lowered):
        clause = clause_match.group()
        if _wet_appliance_clause_is_represented(clause):
            continue
        for action in re.finditer(rf"\b{_APPLIANCE_WATER_ACTION}\b", clause):
            if _action_is_negated(clause, action.start()):
                continue
            if _action_owner(clause, action.start(), action.end()) == "adult":
                continue
            trailing = clause[action.end() :]
            direct_appliance = re.match(
                rf"^\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                rf"{_POWERED_APPLIANCE}\b[^,.!?;\n]{{0,36}}\b"
                rf"{_APPLIANCE_WATER_RELATION}\s+"
                rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_APPLIANCE_WATER_TARGET}\b",
                trailing,
            )
            contextual_appliance = re.match(
                rf"^\s+(?:it|this|that)\b[^,.!?;\n]{{0,20}}\b"
                rf"{_APPLIANCE_WATER_RELATION}\s+"
                rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?{_APPLIANCE_WATER_TARGET}\b",
                trailing,
            )
            if contextual_appliance:
                # Resolve "it" to the closest relevant entity. In
                # "a toaster and a fork; put it in the sink", the fork is the
                # nearer antecedent and this is ordinary placement. In "the
                # hair dryer is here; put it in the bath", the appliance is.
                prefix = lowered[: clause_match.start() + action.start()]
                appliance_mentions = list(re.finditer(rf"\b{_POWERED_APPLIANCE}\b", prefix))
                conductive_mentions = [
                    mention
                    for mention in re.finditer(rf"\b{_CONDUCTIVE_OBJECT}\b", prefix)
                    if not re.fullmatch(
                        r"(?:water|liquids?|fluids?|juices?|milk|soda|drinks?|beverages?)",
                        mention.group(),
                    )
                ]
                last_appliance = appliance_mentions[-1].end() if appliance_mentions else -1
                last_conductive = conductive_mentions[-1].end() if conductive_mentions else -1
                contextual_appliance = contextual_appliance if last_appliance > last_conductive else None
            if direct_appliance or contextual_appliance:
                return True
    return False


def _requires_safety_refusal(text: str) -> bool:
    """Return true for no-go electrical contact or powered-appliance actions."""

    return bool(
        _has_conductive_appliance_action(text)
        or _has_powered_appliance_water_action(text)
        or _requires_category_safety_refusal(text, "electrical")
    )


def _requires_category_safety_refusal(text: str, category: str) -> bool:
    """Recognize a non-negated, non-adult-owned action on a reviewed no-go hazard."""

    if _has_fail_closed_high_risk_control_action(text, category):
        return True
    target_pattern, action_pattern = _HAZARD_ACTION_LANGUAGE[category]
    normalized = _normalized_safety_text(text)
    if category == "sharp_tools":
        # "Nobody saw them alive" describes observation, not use of a saw.
        normalized = re.sub(
            r"\bsaw(?=\s+(?:it|them|him|her|one|anyone|anything)\s+(?:alive|in\s+person)\b)",
            "viewed",
            normalized,
        )
    clauses = [clause.strip() for clause in re.split(r"[.!?;\n]+", normalized) if clause.strip()]
    target_is_active = any(
        not _high_risk_target_is_represented(clause, target)
        for clause in clauses
        for target in re.finditer(rf"\b{target_pattern}\b", clause)
    )
    if not target_is_active:
        return False
    prior_instruction_text = ""
    for clause in clauses:
        if _high_risk_clause_only_references_represented_targets(clause, category):
            prior_instruction_text = f"{prior_instruction_text} {clause}".strip()
            continue
        for action in re.finditer(rf"\b{action_pattern}\b", clause):
            distributive_antecedent = _nearest_distributive_antecedent_is_hazard(
                f"{prior_instruction_text} {clause[: action.start()]}",
                target_pattern=target_pattern,
            )
            if not _hazard_action_is_bound(
                clause,
                target_pattern=target_pattern,
                action=action,
                target_is_active=target_is_active,
                carry_action_pattern=_HAZARD_CARRY_ACTION_LANGUAGE[category],
                distributive_antecedent_is_hazard=distributive_antecedent,
                antecedent_context=f"{prior_instruction_text} {clause[: action.start()]}",
            ):
                continue
            if (
                category == "ignition"
                and re.search(r"\bgas\s+(?:valves?|knobs?|dials?|controls?|switches?|handles?)\b", clause)
                and _high_risk_action_is_explicitly_non_contact(clause, action.start(), action.end())
            ):
                continue
            if _action_is_negated(clause, action.start()):
                continue
            if _action_owner(clause, action.start(), action.end()) == "adult":
                continue
            return True
        prior_instruction_text = f"{prior_instruction_text} {clause}".strip()
    return False


def _requires_physical_safety_refusal(text: str) -> bool:
    return bool(
        _requires_safety_refusal(text)
        or _requires_category_safety_refusal(text, "ignition")
        or _requires_category_safety_refusal(text, "chemicals")
        or _requires_category_safety_refusal(text, "sharp_tools")
    )


def safety_response_requirements_for_event(
    plan: ApprovedPedagogyPlan,
    event_text: str,
) -> set[str]:
    """Apply current code-owned safety requirements even when reading a legacy plan."""

    requirements = set(plan.safety.response_requirements)
    requirements.update(safety_response_requirements_for_text(event_text))
    return requirements


def safety_response_requirements_for_text(event_text: str) -> set[str]:
    """Derive minimum child-facing safety duties without requiring a child plan."""

    requirements: set[str] = set()
    if _is_self_harm(event_text):
        requirements.add("immediate_trusted_grown_up")
    if _requires_safety_refusal(event_text):
        requirements.add("explicit_hazard_prohibition")
    if _has_powered_appliance_water_action(event_text):
        requirements.add("explicit_appliance_water_prohibition")
    if _requires_category_safety_refusal(event_text, "ignition"):
        requirements.add("explicit_ignition_prohibition")
    if _requires_category_safety_refusal(event_text, "chemicals"):
        requirements.add("explicit_chemical_prohibition")
    if _requires_category_safety_refusal(event_text, "sharp_tools"):
        requirements.add("explicit_sharp_tool_prohibition")
    return requirements


def plan_requires_current_safety_upgrade(
    plan: ApprovedPedagogyPlan,
    event_text: str,
) -> bool:
    """Return true when a readable old plan predates a now-required safety route."""

    return plan.validator_version != PEDAGOGY_VALIDATOR_VERSION and bool(
        safety_response_requirements_for_event(plan, event_text)
    )


def _hazard_text(text: str) -> str:
    # Observation instructions use the past tense of "see", not a cutting tool.
    # Keep actual named tools and tool actions elsewhere in the same text.
    text = re.sub(r"\bwhat\s+(?:you|we|they|i|the child)\s+(?:just\s+)?saw\b",
                  "what was observed", text.casefold())

    def absent_materials(match: re.Match[str]) -> str:
        value = match.group(0)
        for kind in ("sharp_tools", "ignition", "electrical", "chemicals"):
            value = re.sub(_HAZARD_TERM_PATTERNS[kind], "unavailable item", value)
        return value

    # A declined list is not an entity that later "press it" or "point here"
    # can refer to. Stop at the end of that list; a subsequent actual tool-use
    # instruction remains available to the action check. Replace only the
    # absent tool names, preserving verbs even at ambiguous comma boundaries.
    return re.sub(
        r"\b(?:no|without)\s+(?:(?:any|using|handling)\s+)?[a-z-]+"
        r"(?:\s*(?:,\s*(?:(?:and|or)\s+)?|\b(?:and|or)\s+)[a-z-]+)*",
        absent_materials, text,
    )


def _hazards_in_value(value: Any) -> set[str]:
    text = _hazard_text(" ".join(_text_fragments(value)))
    hazards = {name for name, pattern in _HAZARD_TERM_PATTERNS.items() if re.search(pattern, text)}
    if _has_conductive_appliance_action(text):
        hazards.add("electrical")
    if _has_powered_appliance_water_action(text):
        hazards.add("electrical")
    return hazards


def _anaphoric_hazards_in_value(value: Any) -> set[str]:
    """Return hazards that can be concrete referents across structured fields."""

    text = " ".join(_text_fragments(value)).casefold()
    hazards = _hazards_in_value(text)
    # Generic anatomy words are safety context, but "body" in an explanation
    # such as "without moving its body" is not an entity for an unrelated "it".
    # Concrete health items still carry across fields (for example, medicine -> it).
    if "body_or_health" in hazards and not re.search(
        r"\b(?:(?:medicine|dose|blood|symptom|injury)\w*|pain(?:s|ful|fully|less|lessly)?)\b",
        text,
    ):
        hazards.remove("body_or_health")
    return hazards


def _approved_safety(text: str, *, sensitive: bool) -> SafetyPolicy:
    text = _hazard_text(text)
    lowered = text.casefold()
    hazard_set = _hazards_in_value(lowered)
    for category in tuple(hazard_set):
        mentions = list(re.finditer(_HAZARD_TERM_PATTERNS[category], lowered))
        if mentions and all(
            re.search(r"\b(?:no|without)\s+(?:(?:any|using|handling)\s+)?$", lowered[:mention.start()])
            for mention in mentions
        ):
            # Unavailable/declined materials are constraints, not objects the
            # child will handle. Any positive mention or concrete hazardous
            # action below still retains its normal review.
            hazard_set.discard(category)
    self_harm = _is_self_harm(text)
    generic_refusal = _requires_safety_refusal(text)
    appliance_water = _has_powered_appliance_water_action(text)
    ignition_refusal = _requires_category_safety_refusal(text, "ignition")
    chemical_refusal = _requires_category_safety_refusal(text, "chemicals")
    sharp_tool_refusal = _requires_category_safety_refusal(text, "sharp_tools")
    # A named high-risk control inside explicit pretend play or a picture is a
    # representation, not a physical hazard in the proposed activity. Keep the
    # lexical term available to the fail-closed action recognizer, but do not
    # give a representation-only request an elevated plan merely because its
    # subject names the real-world object.
    for category in tuple(hazard_set & _FAIL_CLOSED_HIGH_RISK_CONTROL_TARGETS.keys()):
        target_pattern = _FAIL_CLOSED_HIGH_RISK_CONTROL_TARGETS[category]
        matching_clauses = [clause for clause in _safety_clauses(text) if re.search(rf"\b{target_pattern}\b", clause)]
        if matching_clauses and all(
            _high_risk_clause_only_references_represented_targets(clause, category) for clause in matching_clauses
        ):
            hazard_set.discard(category)
    if sensitive:
        hazard_set.add("body_or_health")
    if appliance_water:
        hazard_set.add("electrical")
    if ignition_refusal:
        hazard_set.add("ignition")
    if chemical_refusal:
        hazard_set.add("chemicals")
    if sharp_tool_refusal:
        hazard_set.add("sharp_tools")
    hazards = sorted(hazard_set)
    rules: list[str] = []
    if hazards:
        rules.append("A grown-up directly supervises any hazardous material, tool, body, water, heat, or outdoor step.")
    if "heat" in hazards:
        rules.append("The child does not handle flames, hot surfaces, or hot liquids.")
    if "ignition" in hazards:
        rules.append("The child does not light or handle flames, matches, lighters, or candles.")
    if "electrical" in hazards:
        rules.append(
            "The child does not touch outlets, sockets, plugs, exposed wires, mains electricity, or put conductive "
            "objects into powered appliances. A grown-up unplugs the appliance and keeps everyone away."
        )
    if "sharp_tools" in hazards:
        rules.append("A grown-up operates sharp or stiff tools and keeps fingers clear.")
    if "chemicals" in hazards:
        rules.append("Use only named household-safe materials; do not mix or taste unknown substances.")
    if sensitive:
        rules.append("Keep support calm, concrete, non-diagnostic, and proportionate; involve a trusted grown-up.")
    response_requirements: list[
        Literal[
            "immediate_trusted_grown_up",
            "explicit_hazard_prohibition",
            "explicit_appliance_water_prohibition",
            "explicit_ignition_prohibition",
            "explicit_chemical_prohibition",
            "explicit_sharp_tool_prohibition",
        ]
    ] = []
    if self_harm:
        response_requirements.append("immediate_trusted_grown_up")
    if generic_refusal:
        response_requirements.append("explicit_hazard_prohibition")
    if appliance_water:
        response_requirements.append("explicit_appliance_water_prohibition")
    if ignition_refusal:
        response_requirements.append("explicit_ignition_prohibition")
    if chemical_refusal:
        response_requirements.append("explicit_chemical_prohibition")
    if sharp_tool_refusal:
        response_requirements.append("explicit_sharp_tool_prohibition")
    return SafetyPolicy(
        risk_level="sensitive" if sensitive else "elevated" if hazards else "low",
        hazards=hazards,
        minimum_supervision="direct" if sensitive or hazards else "nearby",
        rules=rules,
        response_requirements=response_requirements,
    )


def _approved_scaffold(proposed: ScaffoldPolicy, intent: QuestionIntent) -> ScaffoldPolicy:
    if intent in {"direct_answer", "sensitive_support"}:
        return ScaffoldPolicy(
            hints=[],
            fade_rule="No scaffold sequence is required for this answer-only turn.",
            frustration_stop="Give the complete answer calmly and stop without requiring another turn.",
            allowed_assistance=["clarifying_prompt"],
        )
    assistance = list(proposed.allowed_assistance)
    if intent == "practice_with_scaffolds":
        assistance = list(dict.fromkeys([*assistance, "worked_example", "joint_completion", "hint"]))
    hints = list(proposed.hints) or ["Restate the choice with one concrete example."]
    return ScaffoldPolicy(
        hints=hints[:3],
        fade_rule="After one supported try, invite the child to try the same idea with less help.",
        frustration_stop="Name the effort, offer an easier response mode, and stop warmly if the child is done.",
        allowed_assistance=assistance[:5],
    )


def _domain_overlay(text: str) -> str:
    lowered = text.casefold()
    if _is_sensitive(text):
        return "sensitive-support-k2"
    if re.search(
        r"\b(?:numbers?|count(?:ing)?|add(?:ing)?|subtract(?:ing)?|shapes?|measure|math|bigger|largest|fraction)\b",
        lowered,
    ):
        return "early-math-k2"
    if re.search(r"\b(?:read|book|story|letter|sound|word|spell|rhyme)\b", lowered):
        return "early-literacy-k2"
    if re.search(r"\b(?:write|sentence|paragraph|journal|revise)\b", lowered):
        return "early-writing-k2"
    if re.search(r"\b(?:source|evidence|true|real|website|news|claim|know)\b", lowered):
        return "source-sense-k2"
    if re.search(
        r"\b(?:why|how|moon|ice|water|animal|plant|robot|air|dinosaur|fossil|weather|space|body|"
        r"wings?|ramps?|melt(?:s|ed|ing)?|science)\b",
        lowered,
    ):
        return "early-science-k2"
    return "general-learning-k2"


def _concept_id(overlay_id: str) -> str:
    return {
        "early-science-k2": "science-observation-and-evidence",
        "early-math-k2": "math-relationships-and-patterns",
        "early-literacy-k2": "literacy-meaning-and-language",
        "early-writing-k2": "writing-communication-and-revision",
        "source-sense-k2": "world-source-and-perspective",
        "sensitive-support-k2": "social-emotional-observation-and-repair",
        "general-learning-k2": "general-question-and-explanation",
    }[overlay_id]


def default_plan_proposal(
    event: dict[str, Any],
    workflow: str,
    registry: PedagogyRegistry | None = None,
) -> PedagogyPlanProposal:
    """Conservative grade-baseline plan used by the stub and fail-closed repair path."""

    del registry
    text = str(event.get("text") or "").strip()
    lowered = text.casefold()
    overlay = _domain_overlay(text)
    if _is_sensitive(text):
        intent: QuestionIntent = "sensitive_support"
        primary = "answer-and-wonder-k2"
    elif _requires_physical_safety_refusal(text):
        intent = "direct_answer"
        primary = "answer-and-wonder-k2"
    elif _requires_direct_answer(text, workflow):
        intent = "direct_answer"
        primary = "answer-and-wonder-k2"
    elif re.search(
        r"\b(?:compar\w*|side-by-side|differences?|same|match\w*|classif\w*|sort\w*)\b",
        lowered,
    ):
        intent = "compare_connect"
        primary = "compare-connect-k2"
    elif re.search(r"\b(?:build|design|invent|make|create)\b", lowered):
        intent = "make_design_iterate"
        primary = "make-design-iterate-k2"
    elif re.search(r"\b(?:too hard|frustrated|stuck|confusing|can't do|cannot do)\b", lowered):
        intent = "practice_with_scaffolds"
        primary = "model-practice-fade-k2"
    elif re.search(r"\b(?:bored|boring|different way|different game)\b", lowered):
        intent = "guided_inquiry"
        primary = "guided-inquiry-k2"
    elif re.search(
        r"\b(?:practice|show me how|help .* learn|work through|help me (?:write|read|spell|add|subtract))\b",
        lowered,
    ):
        intent = "practice_with_scaffolds"
        primary = "model-practice-fade-k2"
    elif re.search(r"\b(?:read together|book|story)\b", lowered):
        intent = "dialogic_reading"
        primary = "dialogic-reading-k2"
    elif re.search(r"\b(?:teach back|explain it back|tell me what .* learned)\b", lowered):
        intent = "teach_back"
        primary = "teach-back-k2"
    elif re.search(r"\b(?:remember|review|again|revisit|recall)\b", lowered):
        intent = "retrieval_revisit"
        primary = "retrieval-revisit-k2"
    elif re.search(
        r"\b(?:experiment|test|predict|try(?: it)?|"
        r"what\s+(?:(?:might|would|could|will)\s+)?happen(?:s)?\s+if)\b",
        lowered,
    ):
        intent = "guided_inquiry"
        primary = "guided-inquiry-k2"
    else:
        intent = "answer_and_wonder"
        primary = "answer-and-wonder-k2"

    evidence_defaults = {
        "guided-inquiry-k2": (
            "predict-test-explain",
            "Make a prediction, observe one fair test, and explain the result.",
        ),
        "compare-connect-k2": (
            "compare-classify",
            "Compare examples using one visible attribute and explain the grouping.",
        ),
        "model-practice-fade-k2": (
            "explain-observation",
            "Try the same kind of step with less help and explain one choice.",
        ),
        "dialogic-reading-k2": ("teach-back", "Point to, say, draw, or act out one meaning clue from the shared text."),
        "teach-back-k2": ("teach-back", "Explain the idea in words, a drawing, or a demonstration."),
        "retrieval-revisit-k2": ("retrieval-application", "Recall the idea and apply it to one nearby example."),
        "make-design-iterate-k2": ("design-iterate", "Show one design choice, test, and change with help recorded."),
    }
    if intent in {"direct_answer", "sensitive_support", "answer_and_wonder"}:
        evidence = EvidenceOpportunityProposal(collection="not_collected", template_id="not-collected")
    else:
        template_id, behavior = evidence_defaults[primary]
        evidence = EvidenceOpportunityProposal(
            collection="future_observation_only",
            template_id=template_id,
            observable_behavior=behavior,
            response_modes=["spoken", "drawn", "gesture_or_manipulative"],
        )
    goal_subject = {
        "early-science-k2": "the question through concrete observation and evidence",
        "early-math-k2": "the question through concrete relationships and representations",
        "early-literacy-k2": "meaning and language in the current question",
        "early-writing-k2": "communicating and revising an idea",
        "source-sense-k2": "what a source or observation supports",
        "sensitive-support-k2": "a calm, concrete, age-appropriate understanding",
        "general-learning-k2": "the current question through a concrete example",
    }[overlay]
    return PedagogyPlanProposal(
        question_intent=intent,
        learning_goal=f"Build a grade-ready understanding of {goal_subject}.",
        concept_ids=[_concept_id(overlay)],
        primary_strategy_id=primary,
        supporting_strategy_id=None,
        domain_overlay_id=overlay,
        representations=["plain_language", "example"],
        scaffold=ScaffoldPolicy(
            hints=["Restate the choice with one concrete example.", "Offer two visible options."],
            fade_rule="After one supported try, invite the child to choose or explain without adding another hint.",
            frustration_stop="Name the effort, offer an easier response mode, and stop warmly if the child is done.",
            allowed_assistance=["clarifying_prompt", "general_prompt", "hint", "worked_example"],
        ),
        evidence=evidence,
        closure=ClosurePolicy(
            child_closure="End with the answer or completed attempt feeling whole.",
            optional_revisit="Optionally notice one related example another day; do not schedule or pressure it.",
        ),
        accessibility=AccessibilityPolicy(
            response_choices=["spoken", "drawn", "gesture_or_manipulative"],
            no_writing_alternative="The child may point, speak, arrange objects, or draw instead of writing.",
            fine_motor_alternative="Use pointing, whole objects, or a grown-up scribe when small marks or cutting are hard.",
            reading_support="A grown-up may read short directions aloud; meaning must not depend on fluent reading.",
        ),
        family_load=FamilyLoadPolicy(
            max_minutes=10 if workflow == "quick_answer" else 20,
            parent_effort="very_low" if workflow == "quick_answer" else "low",
            max_materials=0 if intent in {"direct_answer", "sensitive_support"} else 4,
            supervision="direct" if intent == "sensitive_support" else "nearby",
            cleanup="none" if intent in {"direct_answer", "sensitive_support"} else "quick",
        ),
    )


_HAZARD_ACTION_LANGUAGE: dict[str, tuple[str, str]] = {
    "heat": (
        r"(?:boil\w*|hot\s+(?:water|liquid|surface|glue)|hot[- ]glue\s+guns?|stove|oven|microwaves?)",
        r"(?:handle|use(?!ful)|touch|pour|operate|heat|warm|tip|dump|empty|carry|hold|pick\s+up|grab|"
        r"stir|swirl|shake|dip|scoop|soak|scrub|squeeze|add(?!itional)|place(?!ment)|put|set|take|transfer)\w*",
    ),
    "ignition": (
        rf"(?:{_FIRE_HAZARD_TARGET}|flame|candles?|(?:lit\s+)?match(?:es)?|lighters?|kindling|"
        rf"gas\s+(?:stoves?|burners?|valves?|knobs?|dials?))",
        rf"(?:light(?!er)|ignite|strike|start|(?:make|build)(?=\s+(?:a\s+)?{_FIRE_HAZARD_TARGET})|"
        r"spark|rub|scrape|toss|throw|drop|fling|hurl|chuck|launch|wave|twirl|carry|"
        r"(?:turn|switch)(?:(?:\s+(?:it|this|that))?\s+on|\s+(?:(?:the|a|an|this|that|my|"
        r"our|your)\s+)?gas\s+(?:stoves?|burners?)\s+on)|"
        r"burn(?!s\b)|flick|click|snap|open|close|shut|turn|twist|rotate|"
        r"play\s+with|play\s+around\s+with|fool\s+around\s+with|"
        r"handle|use(?!ful)|touch|hold|take|"
        r"pick\s+up|grab|put|place(?!ment)|set)\w*",
    ),
    "electrical": (
        r"(?:electric(?:al|ity)?|outlets?|wall\s+sockets?|sockets?|wall\s+plugs?|power\s+cords?|"
        r"exposed\s+wires?|mains\s+power)",
        r"(?:insert|stick|jam|force|stuff|put|place(?!ment)|push|drop|poke|reach|touch|grab|"
        r"pick\s+up|yank|lick|bite|chew(?:\s+on)?|suck(?:\s+on)?|wrap|dip|"
        r"pour|spray|splash|spit|dump|wet|spill|"
        r"squirt|drench|plug|unplug|connect|handle|use(?!ful)|"
        r"cut|open|take|taking|hold)\w*",
    ),
    "sharp_tools": (
        r"(?:knives?|knife|blades?|scissors|razors?|box\s+cutters?|sharp\s+tools?|"
        r"hole\s+punch(?:es)?|needles?|saws?|chainsaws?|chain\s+saws?|axes?|nail\s*guns?|"
        r"power(?:ed)?\s+nailers?|power\s+drills?|drill(?:ing)?)",
        r"(?:turn(?:\s+(?:it|this|that))?\s+on|switch(?:\s+(?:it|this|that))?\s+on|"
        r"handle|using|use(?!ful)|play\s+with|touch|operate|start|activate|"
        r"fire\s+off|set\s+off|fire|shoot|cock|(?:re)?load|discharge|brandish|arm|prime|"
        r"(?:aim|point)(?=\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?(?:knives?|knife|blades?|"
        r"scissors|razors?|box\s+cutters?|saws?|chainsaws?|chain\s+saws?|axes?|nail\s*guns?|"
        r"power(?:ed)?\s+nailers?|power\s+drills?)\b|\s+(?:it|this|that)\b)|"
        r"(?:press|depress)(?:\s+(?:(?:the|a|its)\s+)?trigger(?:\s+(?:of|on))?)?|"
        r"squeeze(?:\s+(?:(?:the|a|its)\s+)?trigger(?:\s+(?:of|on))?)?|"
        r"pull\s+back(?:\s+(?:(?:the|a|its)\s+)?trigger(?:\s+(?:of|on))?)?|"
        r"pull(?:\s+(?:(?:the|a|its)\s+)?trigger(?:\s+(?:of|on))?)?|cut(?!e)|slice|"
        r"chop|snip|trim|carve|peel|saw|whittle|stab|pierce|drill|scrape|scratch|drag|shave|run|"
        r"throw|toss|drop|fling|hurl|wave|swing|spin|twirl|juggle|"
        r"hold|pick\s+up|grab|"
        r"add(?!itional)|decorat(?:e|es|ed|ing)|label(?:s|ed|ing)?|mark(?:s|ed|ing)?|"
        r"paint(?:s|ed|ing)?|colou?r(?:s|ed|ing)?|sticker(?:s|ed|ing)?|"
        r"place(?!ment)|put|set|take|carry|choose|select|pick)\w*",
    ),
    "chemicals": (
        r"(?:chemical|cleaner|bleach|detergent)",
        r"(?:handle|use(?!ful)|touch|pour|mix|"
        r"(?:drink|swallow|sip|taste|eat|consume|have)\s+(?:(?:a|the|one)\s+)?"
        r"(?:glass|cup|bottle|sip|gulp|mouthful|taste|drop|spoonful|bite)\s+of|"
        r"tast(?:e|ing)|drink|swallow|chew|gulp|chug|bite|pop|"
        r"take\s+(?:a\s+)?(?:sip|gulp|drink|bite)\s+of|"
        r"suck(?:\s+on)?|gargle|eat|consume|tip|dump|empty|spray|splash|splatter|combine|wipe\s+with|"
        r"stir|swirl|shake|dip|scoop|soak|scrub|squeeze|sniff|smell|"
        r"inhale(?:\s+(?:above|over|near))?|breathe(?:\s+in)?(?:\s+(?:above|over|near))?|"
        r"waft|lick|sip|apply|rub|smear|spread|dab|brush|add(?!itional)|place(?!ment)|put|set|"
        r"take|hold|pick\s+up|grab|transfer|drop)\w*",
    ),
    "water": (
        r"(?:pools?|baths?|deep\s+water|rivers?|oceans?|lakes?|ponds?|sink\s+full)",
        r"(?:enter|go\s+in|swim|wade|fill|lean\s+over|dive|climb\s+in)\w*",
    ),
    "small_parts": (
        r"(?:tiny\s+piece|bead|button|coin|marble)s?",
        r"(?:taste|swallow|put\s+in\s+(?:your|the)\s+mouth|chew)\w*",
    ),
    "outdoors": (
        r"(?:outside|outdoors|street|road|woods|forest|park)",
        r"(?:go|cross|walk|run|explore|leave|wander)\w*",
    ),
    "body_or_health": (
        r"(?:medicine|dose|blood|symptom|pain|injury|body)",
        r"(?:take|taste|touch|use(?!ful)|apply|diagnose|treat|swallow|rub|inject)\w*",
    ),
}

# These category-specific verbs remain hazardous when a concrete referent was
# named anywhere in the structured child-visible payload, even when the action's
# object is "contents", "fluid", or another noun that a pronoun list cannot
# exhaustively enumerate. Generic verbs such as use/put still require a local
# explicit referent so ordinary parallel child actions are not misclassified.
_HAZARD_CARRY_ACTION_LANGUAGE: dict[str, str] = {
    "heat": r"(?:pour|operate|heat|warm|tip|dump|empty|carry|stir|swirl|shake|dip|scoop|soak|scrub|squeeze|transfer)\w*",
    "ignition": (
        r"(?:light(?!er)|ignite|strike|start|spark|rub|scrape|toss|throw|drop|fling|hurl|chuck|"
        r"launch|wave|twirl|"
        r"carry|burn(?!s\b)|(?:turn|switch)(?:(?:\s+(?:it|this|that))?\s+on|"
        r"\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?gas\s+(?:stoves?|burners?)\s+on)|"
        r"flick|click|snap|open|close|shut|turn|twist|rotate|"
        r"play\s+with|play\s+around\s+with|fool\s+around\s+with)\w*"
    ),
    "electrical": (
        r"(?:insert|stick|jam|force|stuff|push|drop|poke|reach|grab|pick\s+up|yank|lick|bite|"
        r"pour|spray|splash|spit|dump|wet|spill|squirt|drench|suck(?:\s+on)?|"
        r"chew(?:\s+on)?|wrap|dip|"
        r"plug|unplug|connect|cut|open)\w*"
    ),
    "sharp_tools": (
        r"(?:turn(?:\s+(?:it|this|that))?\s+on|switch(?:\s+(?:it|this|that))?\s+on|"
        r"operate|start|activate|fire\s+off|set\s+off|fire|shoot|cock|(?:re)?load|discharge|"
        r"arm|prime|"
        r"(?:press|depress)(?:\s+(?:(?:the|a|its)\s+)?trigger(?:\s+(?:of|on))?)?|"
        r"squeeze(?:\s+(?:(?:the|a|its)\s+)?trigger(?:\s+(?:of|on))?)?|"
        r"pull\s+back(?:\s+(?:(?:the|a|its)\s+)?trigger(?:\s+(?:of|on))?)?|"
        r"pull(?:\s+(?:(?:the|a|its)\s+)?trigger(?:\s+(?:of|on))?)?|play\s+with|cut(?!e)|slice|"
        r"chop|snip|trim|carve|peel|saw|whittle|"
        r"stab|pierce|drill|scrape|scratch|drag|shave|run|throw|toss|drop|fling|hurl|wave|swing|spin|"
        r"twirl|juggle)\w*"
    ),
    "chemicals": (
        r"(?:pour|mix|"
        r"(?:drink|swallow|sip|taste|eat|consume|have)\s+(?:(?:a|the|one)\s+)?"
        r"(?:glass|cup|bottle|sip|gulp|mouthful|taste|drop|spoonful|bite)\s+of|"
        r"tast(?:e|ing)|drink|swallow|chew|gulp|chug|bite|pop|"
        r"take\s+(?:a\s+)?(?:sip|gulp|drink|bite)\s+of|suck(?:\s+on)?|gargle|eat|consume|"
        r"tip|dump|empty|spray|splash|splatter|combine|wipe\s+with|stir|swirl|shake|dip|"
        r"scoop|soak|scrub|squeeze|sniff|smell|inhale(?:\s+(?:above|over|near))?|"
        r"breathe(?:\s+in)?(?:\s+(?:above|over|near))?|waft|lick|sip|apply|rub|"
        r"smear|spread|dab|brush|add(?!itional)|transfer|drop)\w*"
    ),
    "water": r"(?:enter|go\s+in|swim|wade|fill|lean\s+over|dive|climb\s+in)\w*",
    "small_parts": r"(?:taste|swallow|put\s+in\s+(?:your|the)\s+mouth|chew)\w*",
    "outdoors": r"(?:cross|explore|leave|wander)\w*",
    "body_or_health": r"(?:take|taste|apply|diagnose|treat|swallow|rub|inject)\w*",
}


_ACTION_NEGATION = (
    r"(?:do\s+not|don't|did\s+not|didn't|never|must\s+not|should\s+not|cannot|can't|"
    r"will\s+not|won't|not\s+to|promise(?:d|s)?\s+not\s+to|refuse(?:d|s)?\s+to|"
    r"while\s+not|"
    r"avoid(?:ed|ing|s)?|refrain(?:ed|ing|s)?\s+from|stay(?:ed|ing|s)?\s+away\s+from|"
    r"not\s+safe\s+to|isn't\s+safe\s+to|no|without)"
)
_CHILD_OWNER = (
    r"(?:i|me|myself|we|us|ourselves|you|yourself|the\s+child|children|child|kids?|"
    r"students?|learners?|pupils?|classmates?|brothers?|sisters?|siblings?|cousins?|"
    r"(?:(?:my|our|your)\s+)?(?:(?:[a-z]+|\d+)[- ]year[- ]old\s+)?friends?|"
    r"(?:(?:the|my|our|your)\s+)?(?:class|group|team)|"
    r"(?:your|our)\s+(?:hands?|cup|bowl|mouth|skin))"
)
_ADULT_OWNER = (
    r"(?:grown-ups?|adults?|parents?|caregivers?|teachers?|school\s+counselors?|"
    r"moms?|mothers?|dads?|fathers?|grandmothers?|grandmas?|grandfathers?|grandpas?|"
    r"chefs?|cooks?|characters?|pirates?|mechanics?|carpenters?|workers?|"
    r"trusted\s+(?:people|persons?|grown-ups?|adults?))"
)
_THIRD_PARTY_PRONOUN = r"(?:he|she|they)"
_HAZARD_ANAPHORA = (
    r"(?:it|this|that|them|these|those|(?:(?:whichever|either|any|the)\s+)?ones?|"
    r"contents?|solution|mixture|substance|"
    r"the\s+(?:contents?|material|liquid|tool|cleaner|equipment|solution|mixture|substance)|"
    r"(?:(?:the|this|that)\s+)?(?:(?:whole|entire|complete)\s+)?"
    r"(?:display|assortment|collection|set|group|batch|lot))"
)

_DISTRIBUTIVE_PLURAL_ANTECEDENT = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+|"
    r"both|several|many|multiple|various|numerous|some|the|these|those|"
    r"all(?:\s+of\s+the)?)\s+"
    r"(?P<object>(?:[a-z][\w'-]*\s+){0,3}?"
    r"(?:[a-z][\w'-]*?(?:s|ves)|children|people|equipment))\b"
)
_DISTRIBUTIVE_BARE_PLURAL_OBJECT = re.compile(
    r"\b(?!(?:a|an|the|this|that|these|those|my|our|your|grown-up|grownups?|adults?|"
    r"child|children|paper|sharp|object|objects|up|down|out|apart|cards?|labels?|stickers?|"
    r"scissors|knives|with|using|containing|holding|carrying|"
    r"alongside|beside|near|next|to|from|and|or|but|as|well|together|in|addition)\b)"
    r"[a-z][\w'-]*(?:\s+(?:apart|out|up|down))?\s+"
    r"(?P<object>(?:[a-z][\w'-]*\s+){0,2}?"
    r"(?:[a-z][\w'-]*?(?:s|ves)|children|people|equipment))\b"
)


def _nearest_distributive_antecedent_is_hazard(
    context: str,
    *,
    target_pattern: str,
) -> bool | None:
    """Resolve a distributive pronoun from the prior clause's grammatical head."""

    target_re = re.compile(rf"\b(?:{target_pattern})\b")
    quantified_candidates = list(_DISTRIBUTIVE_PLURAL_ANTECEDENT.finditer(context))
    # Bare plural direct objects are common in elementary instructions
    # ("cuts paper cards" / "uses scissors to cut cards"). Merge those with
    # quantified phrases, dropping overlaps so one noun phrase is not counted
    # twice merely because it also has a determiner.
    bare_candidates = [
        candidate
        for candidate in _DISTRIBUTIVE_BARE_PLURAL_OBJECT.finditer(context)
        if not any(
            candidate.start() < quantified.end() and quantified.start() < candidate.end()
            for quantified in quantified_candidates
        )
    ]
    candidates = sorted(
        [*quantified_candidates, *bare_candidates],
        key=lambda candidate: (candidate.start(), candidate.end()),
    )
    if not candidates:
        return None

    # References normally bind within the most recent clause that introduced
    # a plural. Do not let an older hazard steal a clearly newer ordinary
    # object ("adult cuts the paper cards; child decorates each one").
    last_candidate = candidates[-1]
    clause_start = max(context.rfind(mark, 0, last_candidate.start()) for mark in ".!?;\n") + 1
    candidates = [candidate for candidate in candidates if candidate.start() >= clause_start]

    # A `with`/`using` noun phrase modifies the preceding core object; it is
    # not the head antecedent of a later `each one`. This prevents an inserted
    # supply phrase ("six knives with two labels") from laundering the knives
    # while still recognizing normal determiner phrases such as "the paper
    # cards", "some paper cards", and "multiple paper cards".
    core_candidates: list[re.Match[str]] = []
    previous_end = clause_start
    for candidate in candidates:
        bridge = context[previous_end : candidate.start()]
        if core_candidates and re.search(
            r"\b(?:with|using|containing|holding|carrying|alongside|beside|near|by|"
            r"next\s+to|adjacent\s+to|along\s+with|accompanied\s+by|together\s+with)\s+$",
            bridge,
        ):
            previous_end = candidate.end()
            continue
        core_candidates.append(candidate)
        previous_end = candidate.end()
    if not core_candidates:
        return None

    chosen = core_candidates[-1]
    chosen_target = target_re.search(
        context,
        chosen.start(),
        chosen.end(),
    )
    if chosen_target:
        return not _high_risk_target_is_represented(context, chosen_target)

    # A newer safe direct object supersedes a hazard only when a new predicate
    # actually introduces it ("uses scissors to cut paper cards"). Merely
    # placing a safe plural after a hazard through coordination or a relational
    # phrase leaves `each one` ambiguous, so fail closed without enumerating
    # every possible preposition.
    hazardous_core = []
    for candidate in core_candidates[:-1]:
        target = target_re.search(
            context,
            candidate.start(),
            candidate.end(),
        )
        if target and not _high_risk_target_is_represented(context, target):
            hazardous_core.append(candidate)
    if hazardous_core:
        bridge = context[hazardous_core[-1].end() : chosen.start()]
        predicate_reset = bool(
            re.fullmatch(r"\s+to\s+", bridge)
            or re.search(
                rf"\b(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                rf"(?:{_ADULT_OWNER}|{_CHILD_OWNER}|{_THIRD_PARTY_PRONOUN})\s+$",
                bridge,
            )
            or re.search(
                rf"\b(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                rf"(?:{_ADULT_OWNER}|{_CHILD_OWNER}|{_THIRD_PARTY_PRONOUN})\s+"
                rf"(?:(?:can|could|may|might|will|would|should|must)\s+)?"
                rf"[a-z][\w'-]*(?:\s+(?:apart|out|up|down|away))?\s+"
                rf"(?:(?:the|a|an|some|multiple)\s+)?$",
                bridge,
            )
        )
        if not predicate_reset:
            return True

    # A hazardous coordinated head can also follow the chosen safe phrase
    # ("paper cards and knives"). This check must still run when the bare
    # object matcher found only that one safe phrase. Supply adjuncts
    # introduced with `with` remain subordinate and do not change the safe
    # head.
    for hazard in target_re.finditer(context, chosen.end()):
        if _high_risk_target_is_represented(context, hazard):
            continue
        bridge = context[chosen.end() : hazard.start()]
        if re.fullmatch(
            r"\s*,?\s*(?:and|or|plus|as\s+well\s+as|together\s+with|along\s+with|"
            r"in\s+addition\s+to|beside|alongside|near|next\s+to|adjacent\s+to)\s*"
            r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+|"
            r"some|several|many|multiple|various|numerous)\s+)?",
            bridge,
        ):
            return True
    return False


def _action_is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 100) : start]
    # Inability or refusal to stop an action has dangerous positive polarity:
    # "cannot avoid drinking" and "do not prevent me from lighting" are not
    # safety negations of the hazardous verb.
    if re.search(
        r"\b(?:"
        r"(?:i\s+)?fail(?:ed|s|ing)?\s+to\s+(?:avoid|resist|stop|refrain\s+from)|"
        r"(?:i\s+)?(?:did\s+not|didn't|could\s+not|couldn't)\s+manage\s+to\s+"
        r"(?:avoid|resist|stop|refrain\s+from)|"
        r"i\s+(?:do\s+not|don't)\s+(?:think|believe|feel|suppose|imagine|reckon)\s+i\s+"
        r"(?:can|could)\s+(?:avoid|resist|stop|refrain\s+from)|"
        r"i\s+(?:do\s+not|don't)\s+(?:feel|seem)\s+(?:able|capable)\s+to\s+"
        r"(?:avoid|resist|stop|refrain\s+from)|"
        r"i\s+(?:cannot|can't|can\s+not|could\s+not|couldn't)\s+"
        r"(?:bring|force)\s+myself\s+to\s+(?:avoid|resist|stop|refrain\s+from)"
        r")\s+(?:\w+[ -]?\s*){0,3}$",
        prefix,
    ):
        return False
    if re.search(
        r"\b(?:"
        r"(?:cannot|can't|can\s+not|could\s+not|couldn't|do\s+not|don't|did\s+not|didn't|"
        r"will\s+not|won't)\s+"
        r"(?:(?:promise|agree|decide)\s+not\s+to|avoid(?:ed|ing|s)?|"
        r"resist(?:ed|ing|s)?|"
        r"help(?:\s+but)?|"
        r"refrain(?:ed|ing|s)?\s+from|stay(?:ed|ing|s)?\s+away\s+from|"
        r"(?:stop(?:ped|ping)?(?:\s+(?:me|myself|us|ourselves|the\s+child|children|kids?)\s+from)?|"
        r"(?:keep|kept)(?:\s+(?:me|myself|us|ourselves|the\s+child|children|kids?))?\s+from|"
        r"prevent(?:ed|ing)?\s+(?:me|myself|us|ourselves|the\s+child|children|kids?)\s+from))|"
        r"refuse(?:d|s)?\s+(?:not\s+to|to\s+(?:avoid|refrain\s+from|stay\s+away\s+from))"
        r"|(?:no\s+one|nobody)\s+(?:stopped|prevented)\s+"
        r"(?:me|us|the\s+child|children|kids?)\s+from"
        r"|(?:am|is|are|was|were)\s+(?:unable|not\s+able)\s+to\s+"
        r"(?:avoid|resist|refrain\s+from|stay\s+away\s+from)"
        r")\s+(?:\w+[ -]?\s*){0,3}$",
        prefix,
    ):
        return False
    return bool(re.search(rf"\b{_ACTION_NEGATION}\b\s+(?:\w+[ -]?\s*){{0,4}}$", prefix))


def _mask_ineligible_conductive_actions(
    clause: str,
    *,
    adult_owned_is_safe: bool,
) -> str:
    """Remove safe grammatical actions before electrical relation matching."""

    masked = clause
    safe_wet_spans = [
        (segment.start(), segment.end())
        for segment in re.finditer(r"[^:—]+", clause)
        if _is_clearly_safe_wet_appliance_shutdown_clause(segment.group())
    ]
    actions = list(re.finditer(rf"\b{_CONDUCTIVE_APPLIANCE_ACTION}\b", clause))
    for action in reversed(actions):
        safe_wet_polarity = any(start <= action.start() < end for start, end in safe_wet_spans)
        wet_state_not_action = bool(
            re.fullmatch(r"wet(?:ter|test)?", action.group())
            and re.search(
                r"\b(?:is|was|are|were|got|gets|became|becomes|looks|seems)\s+$",
                clause[: action.start()],
            )
        )
        if (
            safe_wet_polarity
            or wet_state_not_action
            or _action_is_negated(clause, action.start())
            or (adult_owned_is_safe and _action_owner(clause, action.start(), action.end()) == "adult")
        ):
            masked = f"{masked[: action.start()]} reviewed-safe-action {masked[action.end() :]}"
    return masked


def _establishes_effective_conductive_relation(clause: str) -> bool:
    if not _establishes_conductive_appliance_relation(clause):
        return False
    actions = list(re.finditer(rf"\b{_CONDUCTIVE_APPLIANCE_ACTION}\b", clause))
    if not actions or any(not _action_is_negated(clause, action.start()) for action in actions):
        return True
    # Negated insertion says the objects never meet. Negated removal says the
    # conductive object remains in the appliance and must still constrain a
    # later energizing action.
    return any(
        re.fullmatch(
            r"(?:pull|remove|retrieve|grab|pick|lift|yank|extract|fish|get|take|taking|pry)\w*",
            action.group(0),
        )
        for action in actions
    )


def _child_directly_owns_action(text: str, start: int) -> bool:
    prefix = text[max(0, start - 55) : start]
    return bool(
        re.search(
            r"\b(?:i|we|you|the\s+child|children|child|kids?)\s+"
            r"(?:(?:can|may|might|will|would|should|must|could)\s+|"
            r"(?:want|try|plan|decide)\s+to\s+)?$|\b(?:me|us)\s+to\s+$",
            prefix,
        )
    )


def _selection_target_is_represented(referent: str, target_pattern: str) -> bool:
    """Keep a selected card, label, or picture distinct from the target it represents."""

    target = re.search(rf"\b(?:{target_pattern})\b", referent)
    if not target:
        return False
    represented_before = bool(
        re.search(
            r"\b(?:pictures?|photos?|drawings?|diagrams?|illustrations?|worksheets?|pages?|words?|"
            r"screens?|cards?|flashcards?|labels?|documents?|handouts?|posters?|charts?|tables?|lists?|"
            r"images?|graphics?|icons?|symbols?|silhouettes?|emojis?)"
            r"\b[^,.!?;\n]{0,24}\b"
            r"(?:of|show(?:s|ed|ing)?|display(?:s|ed|ing)?|contain(?:s|ed|ing)?|"
            r"bear(?:s|ing)?|bore|borne|feature(?:s|d|ing)?|portray(?:s|ed|ing)?|"
            r"render(?:s|ed|ing)?|print(?:s|ed|ing)?\s+with|"
            r"list(?:s|ed|ing)?|mention(?:s|ed|ing)?|name(?:s|d|ing)?|"
            r"reference(?:s|d|ing)?|include(?:s|d|ing)?|"
            r"depict(?:s|ed|ing)?|pictur(?:e|es|ed|ing)|illustrat(?:e|es|ed|ing)|"
            r"(?:that\s+)?say(?:s|ing)?)\s+$",
            referent[: target.start()],
        )
        or re.fullmatch(r"\s*words?\s+", referent[: target.start()])
    )
    represented_after = bool(
        re.match(
            r"^\s+(?:pictures?|photos?|drawings?|diagrams?|illustrations?|worksheets?|"
            r"pages?|screens?|cards?|flashcards?|labels?|documents?|handouts?|posters?|"
            r"charts?|tables?|lists?|words?|images?|graphics?|icons?|symbols?|silhouettes?|emojis?)\b",
            referent[target.end() :],
        )
    )
    return represented_before or represented_after


def _hazard_action_is_bound(
    clause: str,
    *,
    target_pattern: str,
    action: re.Match[str],
    target_is_active: bool,
    carry_action_pattern: str,
    distributive_antecedent_is_hazard: bool | None = None,
    antecedent_context: str = "",
) -> bool:
    """Bind an action to a hazard instead of treating clause-wide co-occurrence as intent."""

    leading = clause[: action.start()]
    trailing = clause[action.end() :]
    direct_target = (
        r"(?:(?:the|a|an|this|that|my|our|your|each|every|both|all|any|either|"
        r"whichever|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"some|several|many|multiple|a\s+few|a\s+couple(?:\s+of)?|one\s+or\s+more)\s+)?"
        rf"(?:{target_pattern})\b"
    )
    embedded_target = re.search(rf"\b{target_pattern}\b", action.group(0))
    if embedded_target and embedded_target.start() > 0:
        # Composite actions such as "turn the gas stove on" include their
        # target inside the action match. A target word that is itself also an
        # action (for example, the noun "saw") does not establish handling.
        return True
    if re.match(rf"^\s+{direct_target}", trailing):
        return True
    # A named hazard that is explicitly secured is context, not the object of
    # an otherwise ordinary child action ("drink water with bleach locked away").
    if re.search(
        rf"\b(?:with|while|near|beside|next\s+to)\s+{direct_target}\s+"
        rf"(?:(?:is|was|stays?|remains?)\s+)?(?:locked|stored|sealed|closed|put\s+away|kept\s+away)\b",
        trailing,
    ):
        return False
    if re.match(
        rf"^[^,.!?;\n]{{0,52}}\b(?:with|using|in|into|inside(?:\s+of)?|through|within|"
        rf"onto|on|against|around|at|from|out\s+of)\s+{direct_target}",
        trailing,
    ):
        return True
    explicit_partitive_object = re.match(
        r"^\s+(?:any|either|whichever|each|one|two|three|four|five|six|seven|eight|"
        r"nine|ten|eleven|twelve|some|several|many|multiple|a\s+few|a\s+couple|"
        r"one\s+or\s+more|(?:an?\s+)?(?:member|item|object|thing|piece|tool))\s+"
        r"(?:of|from(?:\s+among)?|out\s+of|among|between|in)\s+"
        r"(?:(?:the|this|that|these|those|my|our|your)\s+)?"
        r"(?P<referent>(?:[a-z][\w'-]*\s+){0,3}[a-z][\w'-]*)",
        trailing,
    )
    if explicit_partitive_object:
        referent = explicit_partitive_object.group("referent")
        if re.search(rf"\b(?:{target_pattern})\b", referent):
            return not _selection_target_is_represented(referent, target_pattern)
        if re.match(
            r"(?:them|these|those|ones?|items?|objects?|things?|pieces?|members?|tools?|"
            r"equipment|pairs?|duos?|trios?|quartets?|sets?|groups?|collections?|batches?|lots?|"
            r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+|"
            r"displays?|assortments?)\b",
            referent,
        ):
            return distributive_antecedent_is_hazard is not False
        # A partitive naming an ordinary object resolves that selection even
        # when a hazardous plural is also active in the surrounding plan.
        return False
    explicit_distributive_object = re.match(
        r"^\s+(?:both|all|each|every|any|these|those)\s+"
        r"(?P<referent>(?:[a-z][\w'-]*\s+){0,2}"
        r"(?:[a-z][\w'-]*?(?:s|ves)|children|people|equipment))\b",
        trailing,
    )
    if explicit_distributive_object:
        # A named ordinary plural is a concrete object, not an ambiguous
        # anaphor. It is safe only when that same plural was actually
        # introduced; explicit hazard nouns were already caught above.
        referent = explicit_distributive_object.group("referent")
        if not re.search(rf"\b(?:{target_pattern})\b", referent) and re.search(
            rf"\b{re.escape(referent)}\b", antecedent_context
        ):
            return False
    if re.fullmatch(r"(?:choose|select|pick)\w*", action.group(0)):
        singular_selection = re.match(
            r"^\s+(?:(?:any|either|whichever)\s+)?(?:the\s+)?"
            r"one(?:\s+(?:that|which)\s+remains?)?"
            r"(?=\s*(?:$|[,.!?;]|\b(?:to|for)\b))",
            trailing,
        )
        # Selection language is hazardous only when it confidently resolves
        # to the introduced hazardous plural. An active plan-level tool alone
        # must not turn ordinary choices ("choose one useful move") into tool
        # handling, and a concrete safe noun remains independently resolvable.
        if singular_selection:
            return distributive_antecedent_is_hazard is True
        named_selection = re.match(
            r"^\s+(?:(?:the|a|an|this|that|my|our|your|any|either|whichever|one|two|"
            r"three|four|five|six|seven|eight|nine|ten|eleven|twelve|some|several|many|"
            r"multiple|a\s+few|a\s+couple(?:\s+of)?|one\s+or\s+more)\s+)?"
            r"(?P<referent>(?:[a-z][\w'-]*\s+){0,3}[a-z][\w'-]*)",
            trailing,
        )
        if named_selection:
            # Explicitly named selections preserve ordinary choices while
            # still catching quantified or modified hazardous nouns.
            referent = named_selection.group("referent")
            if re.search(rf"\b(?:{target_pattern})\b", referent):
                return not _selection_target_is_represented(referent, target_pattern)
            if re.fullmatch(
                r"(?:it|this|that|them|these|those|either|any|whichever|"
                r"(?:the\s+)?one\s+(?:that|which)\s+remains?|"
                r"(?:remaining|other|leftover|last)"
                r"(?:\s+(?:one|item|object|thing|piece|member|tool))?|"
                r"(?:another|additional|extra)(?:\s+(?:one|item|object|thing|piece))?|"
                r"one\s+more(?:\s+(?:one|item|object|thing|piece))?|"
                r"ones?|items?|objects?|things?|pieces?|members?|tools?|equipment|sets?|groups?|"
                r"collections?|batches?|lots?|displays?|assortments?)"
                r"(?:\s+(?:to|for)\s+[a-z][\w'-]*(?:\s+[a-z][\w'-]*)?)?",
                referent,
            ):
                return distributive_antecedent_is_hazard is not False
            return False
    if re.search(r"\b(?:each|every)\s+ones?\b", trailing[:72]):
        # A distributive pronoun is safe only when a confident prior plural
        # antecedent is ordinary. Carry verbs stay fail closed even when an
        # inserted object tries to separate the verb from the hazard.
        if re.fullmatch(carry_action_pattern, action.group(0)):
            return True
        return distributive_antecedent_is_hazard is not False
    if re.search(
        r"\b(?:one\s+by\s+one|them\s+all|both|each|"
        r"(?:each|every)\s+(?:item|object|piece|member|thing|one)|"
        r"(?:(?:the|this|that)\s+)?(?:whole|entire|complete)\s+"
        r"(?:set|group|collection|batch|lot|display|assortment)|"
        r"(?:the|this|that)\s+(?:set|group|collection|batch|lot|display|assortment)|"
        r"(?:all\s+(?:of\s+)?|any\s+of\s+)(?:them|these|those|the\s+"
        r"(?:items?|objects?|pieces?|members?|things?|set|group|collection|batch|lot|"
        r"display|assortment)|"
        r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+))\b",
        trailing[:96],
    ):
        # Set, count, and iterative anaphora distribute the action across the
        # previously introduced plural. If that antecedent is hazardous or
        # unresolved, keep the child-facing boundary closed.
        return distributive_antecedent_is_hazard is not False
    if re.search(
        r"\b(?:"
        r"(?:some|several|many|multiple|either|any|whichever|whatever|one|two|three|"
        r"a\s+few|a\s+couple|one\s+or\s+more|a\s+number)\s+of\s+"
        r"(?:them|these|those)|"
        r"(?:them|these|those)(?!\s+[a-z])|"
        r"(?:any|some|either|each|every|whichever|whatever)\s+"
        r"(?:(?:available|nearby|present|displayed)\s+)?"
        r"(?:one|item|object|thing|piece)(?:\s+(?:here|there|present|nearby|on\s+display))?|"
        r"whichever\s+(?:one|ones|items?|objects?|things?|pieces?)\s+"
        r"(?:you|they|the\s+child|children|kids?)\s+"
        r"(?:like|choose|select|pick|prefer|want)|"
        r"(?:whichever|whatever)\s+(?:you|they|the\s+child|children|kids?)\s+"
        r"(?:like|choose|select|pick|prefer|want)|"
        r"(?:the|any)\s+ones?\s+(?:you|they|the\s+child|children|kids?)\s+"
        r"(?:like|choose|select|pick|prefer|want)|"
        r"any\s+one(?=\s*(?:$|[,.!?;]|(?:here|there|present|nearby)\b))|"
        r"(?:(?:another|additional|extra|remaining|other|leftover|last)"
        r"(?=\s*(?:$|[,.!?;]|(?:here|there|present|nearby|on\s+display)\b))|"
        r"(?:another|additional|extra|remaining|other|leftover|last)\s+"
        r"(?:one|item|object|thing|piece|member|tool)|"
        r"one\s+more(?:\s+(?:one|item|object|thing|piece))?)"
        r"(?:\s+(?:here|there|present|nearby|on\s+display))?"
        r")\b",
        trailing[:96],
    ):
        # Open-set and unresolved partitive references can select a hazardous
        # member from a mixed display. A concrete introduced noun (for example
        # "those labels") is handled above and remains a safe named referent.
        return distributive_antecedent_is_hazard is not False
    if re.search(
        r"\b(?:everything|all|whatever)\s+"
        r"(?:is\s+)?(?:here|there|present|nearby|on\s+(?:display|the\s+(?:table|counter|bench)))\b",
        trailing[:96],
    ):
        # Universal locatives are open-class collective references. Bind them
        # to the resolved plural antecedent instead of enumerating display nouns.
        return distributive_antecedent_is_hazard is not False
    if distributive_antecedent_is_hazard is True and re.fullmatch(
        r"(?:add|decorat(?:e|es|ed|ing)|label(?:s|ed|ing)?|mark(?:s|ed|ing)?|"
        r"paint(?:s|ed|ing)?|colou?r(?:s|ed|ing)?|sticker(?:s|ed|ing)?|place|put|set)\w*",
        action.group(0),
    ):
        # A definite singular noun that was never introduced is an unresolved
        # collective reference, not a new safe object. This closes open-class
        # aliases such as "the spread/array/selection/bundle" without listing
        # collective nouns; a genuinely introduced object ("the labels")
        # remains available as a safe referent.
        referent = re.search(
            r"(?:^\s+|\b(?:on|onto|across|around|over)\s+)"
            r"(?:the|this|that)\s+"
            r"(?P<referent>[a-z][\w-]*(?:\s+[a-z][\w-]*){0,2})\s*$",
            trailing,
        )
        if referent:
            phrase = referent.group("referent")
            if not re.search(rf"\b{re.escape(phrase)}\b", antecedent_context):
                return True
    for target in re.finditer(rf"\b{target_pattern}\b", leading):
        relation = leading[target.end() :]
        if re.fullmatch(
            rf"\s+(?:in|on|under|against|across)\s+(?:the\s+)?{_CHILD_OWNER}[\s,:-]*",
            relation,
        ) or re.fullmatch(r"[\s,:-]*(?:to|and\s+then)\s*", relation):
            return True
    if not target_is_active:
        return False
    if re.search(rf"\b{_HAZARD_ANAPHORA}\b", action.group(0)):
        return True
    if re.search(
        rf"\b(?:with|using|in|into|inside(?:\s+of)?|through|within|onto|on|against|at|"
        rf"from|out\s+of)\s+(?:{_HAZARD_ANAPHORA})\b(?!\s+[a-z])",
        trailing[:72],
    ):
        return True
    # When a clause supplies an explicit ordinary object, a later demonstrative
    # or the globally active hazard must not steal that object relationship.
    # Hazard relations above still win, so "stab the box with the knife" and
    # "dip your finger into it" remain blocked.
    if re.match(
        rf"^\s*(?:(?:carefully|slowly|gently|now|next|then|please)\b[\s,]*)*"
        rf"(?!(?:{direct_target})(?:\s|$))"
        rf"(?!(?:it|them|these|those|contents?|solution|mixture|substance|"
        rf"the\s+(?:contents?|material|liquid|tool|cleaner|equipment|solution|mixture|substance))\b)"
        rf"(?!(?:this|that)\b(?:\s*$|\s+(?:in|into|inside|on|onto|with|using|while|when|"
        rf"over|under|beside|near|next\s+to|from|out\s+of)\b))"
        rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        rf"[a-z][\w-]*(?:\s+[a-z][\w-]*){{0,3}}"
        rf"(?:\s+(?:in|into|inside|on|onto|with|using|while|when|over|under|beside|near|"
        rf"next\s+to|from|out\s+of)\b|\s*$)",
        trailing,
    ):
        return False
    trailing_anaphora = re.search(rf"\b{_HAZARD_ANAPHORA}\b", trailing[:48])
    if trailing_anaphora:
        return True
    if not re.fullmatch(carry_action_pattern, action.group(0)):
        return False
    # A category-specific action with no named object can carry a hazard
    # referent across fields/sentences. An explicit alternate object (for
    # example, "taste the apple") breaks that carry.
    return bool(
        (_action_is_imperative(clause, action.start()) or _child_directly_owns_action(clause, action.start()))
        and re.match(
            r"^\s*(?:(?:carefully|slowly|gently|now|next|then|please)\b[\s,]*)*"
            r"(?:$|(?:into|onto|inside|through|within|from|out\s+of|in|with|using)\b)",
            trailing,
        )
    )


def _action_owner(text: str, start: int, end: int) -> str | None:
    """Resolve the nearest grammatical owner for one hazardous action."""

    prefix = text[:start]
    boundaries = [prefix.rfind(token) for token in (",", ";", " while ", " then ", " but ", " whereas ")]
    local_prefix = prefix[max(boundaries) + 1 :]
    adult_positions = [match.start() for match in re.finditer(rf"\b{_ADULT_OWNER}\b", local_prefix)]
    child_positions = [match.start() for match in re.finditer(rf"\b{_CHILD_OWNER}\b", local_prefix)]
    pronouns = list(re.finditer(rf"\b{_THIRD_PARTY_PRONOUN}\b", local_prefix))
    if pronouns and pronouns[-1].start() > max([*adult_positions, *child_positions], default=-1):
        pronoun = pronouns[-1].group()
        before_pronoun = local_prefix[: pronouns[-1].start()]
        gendered_child = {
            "she": r"(?:sisters?|girls?|daughters?|nieces?)",
            "he": r"(?:brothers?|boys?|sons?|nephews?)",
            "they": r"(?:children|kids?|students?|learners?|pupils?|classmates?|siblings?|cousins?|friends?|class|group|team)",
        }.get(pronoun)
        gendered_adult = {
            "she": r"(?:moms?|mothers?|grandmothers?|grandmas?)",
            "he": r"(?:dads?|fathers?|grandfathers?|grandpas?)",
            "they": r"(?:grown-ups?|adults?|parents?|caregivers?|teachers?)",
        }.get(pronoun)
        reporting_relation = False
        child_position = child_positions[-1] if child_positions else -1
        adult_position = adult_positions[-1] if adult_positions else -1
        if child_positions and adult_positions:
            between = local_prefix[min(child_position, adult_position) : max(child_position, adult_position)]
            reporting_relation = bool(
                re.search(
                    r"\b(?:tell(?:s|ing)?|told|ask(?:s|ed|ing)?|inform(?:s|ed|ing)?|"
                    r"say(?:s|ing)?\s+to|said\s+to|promis(?:e|es|ed|ing)|assur(?:e|es|ed|ing)|"
                    r"warn(?:s|ed|ing)?|explain(?:s|ed|ing)?|describ(?:e|es|ed|ing)|"
                    r"stat(?:e|es|ed|ing)|shar(?:e|es|ed|ing)|report(?:s|ed|ing)?|"
                    r"admit(?:s|ted|ting)?|announc(?:e|es|ed|ing)|"
                    r"confess(?:es|ed|ing)?|mention(?:s|ed|ing)?|whisper(?:s|ed|ing)?|"
                    r"declar(?:e|es|ed|ing)|reveal(?:s|ed|ing)?|disclos(?:e|es|ed|ing)|"
                    r"acknowledg(?:e|es|ed|ing))\b",
                    between,
                )
            )
            if reporting_relation and child_position < adult_position:
                return "child"
            if reporting_relation and re.search(
                r"\b(?:the\s+child|a\s+child|children|kids?|students?|learners?|pupils?)\b",
                before_pronoun[child_position:],
            ):
                return "child"
        child_gender_match = bool(gendered_child and re.search(rf"\b{gendered_child}\b", before_pronoun))
        adult_gender_match = bool(gendered_adult and re.search(rf"\b{gendered_adult}\b", before_pronoun))
        if child_gender_match != adult_gender_match:
            return "child" if child_gender_match else "adult"
        if reporting_relation:
            # When an adult names a child recipient and a later pronoun is not
            # uniquely adult-gendered, fail closed to the child reading.
            return "child"
        controller_matches: list[tuple[int, str]] = []
        control_verb = (
            r"(?:say(?:s|ing)?|said|tell(?:s|ing)?|told|promis(?:e|es|ed|ing)|"
            r"claim(?:s|ed|ing)?|report(?:s|ed|ing)?|explain(?:s|ed|ing)?|"
            r"describ(?:e|es|ed|ing)|stat(?:e|es|ed|ing)|shar(?:e|es|ed|ing)|"
            r"admit(?:s|ted|ting)?|announc(?:e|es|ed|ing)|"
            r"confess(?:es|ed|ing)?|mention(?:s|ed|ing)?|whisper(?:s|ed|ing)?|"
            r"declar(?:e|es|ed|ing)|reveal(?:s|ed|ing)?|disclos(?:e|es|ed|ing)|"
            r"acknowledg(?:e|es|ed|ing))"
        )
        for owner, owner_pattern in (("adult", _ADULT_OWNER), ("child", _CHILD_OWNER)):
            for match in re.finditer(
                rf"\b{owner_pattern}\b[^,;]{{0,36}}\b{control_verb}\b[^,;]{{0,36}}$",
                before_pronoun,
            ):
                controller_matches.append((match.start(), owner))
        if controller_matches:
            return max(controller_matches)[1]
        if child_positions and adult_positions and child_positions[-1] > adult_positions[-1]:
            # When both antecedents share a pronoun, prefer the nearer named
            # child before falling back to an earlier adult. This also makes
            # novel adult-heard-child-report verbs fail closed without relying
            # solely on a finite reporting-verb enumeration.
            return "child"
        # Resolve explicit gendered adult antecedents before a nearby object
        # pronoun such as "me": "Mom told me she will light the candle" is
        # adult-owned. An unresolved third-person pronoun is not assumed safe.
        if gendered_adult and re.search(rf"\b{gendered_adult}\b", local_prefix):
            return "adult"
        if child_positions and not adult_positions:
            return "child"
        if adult_positions and not child_positions:
            return "adult"
        if child_positions and adult_positions:
            return "child" if child_positions[-1] > adult_positions[-1] else "adult"
        return None
    if child_positions and (not adult_positions or child_positions[-1] > adult_positions[-1]):
        return "child"
    if adult_positions:
        return "adult"
    suffix = text[end : end + 55]
    if re.search(rf"\b(?:by\s+)?{_CHILD_OWNER}\b", suffix):
        return "child"
    if re.search(rf"\bby\s+(?:a\s+|the\s+)?{_ADULT_OWNER}\b", suffix):
        return "adult"
    return None


def _action_is_imperative(text: str, start: int) -> bool:
    prefix = text[:start]
    boundaries = [prefix.rfind(token) for token in (",", ";", ":", " while ", " then ", " but ", " and then ")]
    local_prefix = prefix[max(boundaries) + 1 :]
    return bool(
        re.fullmatch(
            r"\s*(?:(?:next|then|now|first|finally|carefully|slowly|gently|please|when\s+ready)"
            r"\b[\s,]*|go\s+ahead\s+and\s+)*",
            local_prefix,
        )
    )


_FAIL_CLOSED_HIGH_RISK_CONTROL_TARGETS = {
    "ignition": (
        r"(?:gas[- ]cookers?|gas|stove|stovetop|oven|burner|range|cooktop|hob)(?:'s)?\s+"
        r"(?:(?!(?:toy|pretend|play|model|paper|cardboard|picture|photo|drawing|diagram)\b)"
        r"[a-z0-9][\w-]*\s+){0,3}"
        r"(?:valves?|knobs?|dials?|controls?|switch(?:es)?|handles?|taps?)"
    ),
    "sharp_tools": (
        r"(?:nail\s*guns?|power(?:ed)?\s+nailers?|chainsaws?|chain\s+saws?|"
        r"power\s+drills?)"
    ),
}
_FAIL_CLOSED_HIGH_RISK_CONTROL_ALIASES = {
    "ignition": r"(?:valves?|knobs?|dials?|controls?|switch(?:es)?|handles?|taps?)",
    "sharp_tools": (r"(?:guns?|nailers?|chainsaws?|saws?|drills?|tools?|devices?|equipment|weapons?)"),
}
_IGNITION_APPLIANCE_ENTITY = (
    r"(?:gas[- ]cookers?|gas\s+(?:stoves?|ovens?|ranges?|cooktops?|hobs?)|"
    r"gas[- ]powered\s+(?:cookers?|stoves?|ovens?|ranges?|cooktops?|hobs?)|"
    r"(?:cookers?|stoves?|ovens?|ranges?|cooktops?|hobs?)\s+(?:is|was)\s+gas[- ]powered|"
    r"(?:cookers?|stoves?|ovens?|ranges?|cooktops?|hobs?)\s+"
    r"(?:uses?|burns?|runs?\s+on)\s+gas|"
    r"stoves?|stovetops?|ovens?|ranges?|cooktops?|hobs?)"
)
_SAFE_HIGH_RISK_REFERENCE_ACTION = re.compile(
    r"(?:draw|sketch|describe|explain|name|label|study|photograph|identify|observe|compare|"
    r"find|count|spell|circle|color|shade|highlight|mark|underline|trace|say|inspect|view|"
    r"locate|recognize|match|read|watch|see|annotate|write|copy|quote|discuss|"
    r"talk(?:s|ed|ing)?|speak(?:s|ing)?|spoke|chat(?:s|ted|ting)?|"
    r"convers(?:e|es|ed|ing)|ask(?:s|ed|ing)?|"
    r"learn(?:s|ed|ing)?|learn\s+about|read\s+about|think\s+about|"
    r"look\s+(?:at|over|under)|point\s+(?:at|to))"
)
_HIGH_RISK_ACTION_HEAD = re.compile(
    r"\b(?P<verb>(?!(?:a|an|the|this|that|these|those|my|our|your|its|their|"
    r"i|we|you|he|she|they|it|is|are|was|were|be|been|being|has|have|had|"
    r"can|could|may|might|must|shall|should|will|would|do|does|did|"
    r"and|but|or|if|when|while|then|than|of|to|for|from|with|by|on|in|at|"
    r"here|there|where|why|how|what|which)\b)"
    r"[a-z]+(?:\s+(?:about|at|back|off|on|to|up|down)\b)?)"
)
_HIGH_RISK_OBJECT_PREFIX = re.compile(r"(?:\s+[a-z0-9][\w'-]*){0,6}\s*")


def _high_risk_action_is_explicitly_non_contact(
    clause: str,
    action_start: int,
    action_end: int,
) -> bool:
    """Authenticate a bounded non-contact observation for one operation."""

    del action_end
    bounded_operation = clause[action_start:]
    non_contact = re.search(
        r"(?:\b(?:but\s+)?(?:do\s+not|don't|never)\s+"
        r"(?:touch|contact|press|move|turn|adjust)\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\bwhile\s+not\s+(?:touching|contacting)\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\bwithout\s+(?:(?:ever|actually|physically)\s+)?(?:making\s+)?"
        r"(?:touch|contact|press|move|turn|adjust)(?:ing)?"
        r"(?:\s+contact\s+with)?(?:\s+(?:it|this|that|them|the\s+"
        r"(?:valve|knob|dial|control|switch|handle)))?\b|"
        r"\bwithout\s+coming\s+into\s+contact\s+with\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\bwith\s+no\s+(?:physical\s+)?contact"
        r"(?:\s+with\s+(?:it|this|that|them|the\s+"
        r"(?:valve|knob|dial|control|switch|handle)))?\b|"
        r"\b(?:while\s+)?(?:staying|remaining|keeping)\s+clear\s+of\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\b(?:while\s+)?avoiding\s+contact\s+with\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\b(?:while\s+)?avoiding\s+touching\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\b(?:but\s+|while\s+)?keep(?:ing)?\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\s+clear\b|"
        r"\b(?:while\s+)?taking\s+care\s+not\s+to\s+touch\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\b(?:while\s+)?(?:taking\s+care|making\s+sure)\s+to\s+avoid\s+touching\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\b(?:while\s+)?(?:keeping|staying|remaining)\s+away\s+from\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\b(?:while\s+)?maintaining\s+(?:a|your)\s+(?:safe\s+)?distance\s+from\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b|"
        r"\b(?:while\s+)?keeping\s+(?:a|your)\s+distance\s+from\s+"
        r"(?:it|this|that|them|the\s+(?:valve|knob|dial|control|switch|handle))\b)",
        bounded_operation,
    )
    if not non_contact:
        return False
    physical_span = bounded_operation[: non_contact.start()]
    if re.match(
        r"\s*(?:push|press|wedge|jam|force|brace|prop|lean|pack|stuff|insert)\w*\b",
        physical_span,
    ):
        return False
    if re.search(r"\b(?:against|into|onto|around)\b", physical_span):
        return False
    return bool(
        re.search(
            r"\b(?:above|below|beneath|beside|close\s+to|near|next\s+to|over|under|underneath)\b",
            physical_span,
        )
    )


def _high_risk_target_is_represented(clause: str, target: re.Match[str]) -> bool:
    """Keep represented, observed, or explicitly adult-owned operations benign."""

    suffix = clause[target.end() :]
    if re.match(
        r"\s+(?:pictures?|photos?|drawings?|diagrams?|illustrations?|worksheets?|pages?|"
        r"screens?|cards?|flashcards?|labels?|documents?|handouts?|posters?|charts?|tables?|lists?|"
        r"words?|images?|graphics?|icons?|symbols?|silhouettes?|emojis?)\b",
        suffix,
    ):
        return True
    if re.match(
        r"\s+(?:(?:in|on)\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?|"
        r"(?:shown|depicted|pictured|drawn|printed)\s+(?:in|on)\s+"
        r"(?:(?:the|a|an|this|that|my|our|your)\s+)?)"
        r"(?:picture|photo|drawing|diagram|illustration|worksheet|page|screen|card|story|play|scene)\b",
        suffix,
    ):
        return True
    if re.match(
        r"\s+(?:on|in|of)\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        r"toy\s+(?:stove|oven|kitchen|appliance|model|prop)\b",
        suffix,
    ):
        return True
    prefix = clause[: target.start()]
    if re.search(
        r"\b(?:toy|pretend|play|model|paper|cardboard)\s+$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:pictures?|photos?|drawings?|diagrams?|illustrations?|worksheets?|pages?|screens?|cards?|"
        r"images?|graphics?|icons?|symbols?|silhouettes?|emojis?|"
        r"story|play|scene)\s+of\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:pictures?|photos?|drawings?|diagrams?|illustrations?|worksheets?|pages?|"
        r"screens?|cards?|flashcards?|labels?|documents?|handouts?|posters?|charts?|tables?|lists?|"
        r"images?|graphics?|icons?|symbols?|silhouettes?|emojis?)\s+"
        r"(?:each\s+)?"
        r"(?:shows?|displays?|depicts?|pictures?|illustrates?|contains?|bears?|features?|portrays?|"
        r"renders?|lists?|mentions?|names?|references?|includes?|"
        r"(?:is|are|was|were)\s+printed\s+with|says?)\s+"
        r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"\d+|some|several|many|multiple)\s+)?$",
        prefix,
    ):
        return True
    if re.search(r"\bwords?\s+$", prefix):
        return True
    if re.search(
        rf"\b(?:(?:talk(?:s|ed|ing)?|speak(?:s|ing)?|spoke|"
        rf"chat(?:s|ted|ting)?|convers(?:e|es|ed|ing))\s+(?:with|to)|"
        rf"(?:ask|tell|question)(?:s|ed|ing)?)\s+"
        rf"(?:(?:the|a|an|my|our|your)\s+)?{_ADULT_OWNER}\s+about\s+"
        r"(?:(?:the|a|an|this|that|my|our|your)\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"(?:^|[:—]\s*)(?:in|on|using)\s+"
        r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        r"(?:picture|photo|drawing|diagram|illustration|worksheet|page|screen|card)"
        r"\s*[,;:]\s+",
        clause,
    ):
        return True
    if re.search(
        r"\b(?:pictured|depicted|drawn|printed|diagram|worksheet|illustrated)\s+"
        r"(?:(?:the|a|an|this|that|my|our|your)\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:watch|view|see|observe)\s+"
        r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        r"(?:video|movie|clip|recording|demonstration)\s+of\s+"
        r"(?:someone|somebody|a\s+person|an?\s+adult|the\s+adult|"
        rf"{_ADULT_OWNER})\s+[a-z]+ing\s+"
        r"(?:(?:the|a|an|this|that|my|our|your)\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        rf"\b(?:let|have|watch|see|observe|notice)\s+"
        rf"(?:(?:the|a|an|my|our|your)\s+)?"
        rf"{_ADULT_OWNER}\s+[a-z]+\s+"
        r"(?:(?:the|a|an|this|that|my|our|your)\s+)?$",
        prefix,
    ):
        return True
    if re.search(
        r"\b(?:pretend|imagine)\s+(?:(?:that\s+)?(?:i|we|you|they)\s+"
        r"(?:am|are|was|were)\s+)?[a-z]+ing\s+"
        r"(?:(?:the|a|an|this|that)\s+)?$",
        prefix,
    ):
        return True
    if re.match(
        r"\s+with\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        r"(?:(?:paper|cardboard|pretend|stage)\s+(?:prop|model|picture|cutout)|"
        r"toy\s+(?:prop|model|picture|cutout|stove|appliance|kitchen|oven))\b",
        suffix,
    ) and re.search(
        r"\b(?:(?:act|role[- ]?play)\s+(?:it\s+)?out|mime(?:\s+[a-z]+ing)?)\b",
        prefix,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:write|copy|quote|transcribe|read|say|spell|discuss|circle|underline)\s+"
            r"(?:(?:the|a|an|this|that)\s+)?"
            r"(?:phrase|sentence|words?|text|caption|label)\b[^.!?;\n]{0,64}$",
            prefix,
        )
    )


def _high_risk_clause_only_references_represented_targets(
    clause: str,
    category: str,
) -> bool:
    """Return true when every concrete high-risk target is safely represented."""

    target_pattern = _FAIL_CLOSED_HIGH_RISK_CONTROL_TARGETS.get(category)
    if not target_pattern:
        return False
    targets = list(re.finditer(rf"\b{target_pattern}\b", clause))
    return bool(targets) and all(_high_risk_target_is_represented(clause, target) for target in targets)


def _has_fail_closed_high_risk_control_action(text: str, category: str) -> bool:
    """Treat novel child operations on the highest-risk controls as unsafe."""

    target_pattern = _FAIL_CLOSED_HIGH_RISK_CONTROL_TARGETS.get(category)
    if not target_pattern:
        return False
    alias_pattern = _FAIL_CLOSED_HIGH_RISK_CONTROL_ALIASES[category]
    alias_reference = (
        rf"(?:"
        rf"(?:(?:the|a|an|this|that|my|our|your|whichever|either|any|each|both|all)\s+)?"
        rf"(?:(?:other|next|neighboring|neighbouring|adjacent|nearby|opposite|far|farthest|furthest|"
        rf"extreme|outermost|innermost|edge|corner|panel|side|left|right|leftmost|rightmost|"
        rf"nearest|closest|front|rear|back|end|"
        rf"middle|center|centre|upper|lower|top|bottom|inner|outer|temperature|"
        rf"temperature[- ]setting|heat[- ]setting|heat|flame|power|main|first|second|third|fourth|"
        rf"fifth|sixth|seventh|eighth|ninth|tenth|last|final|one|two|three|four|five|"
        rf"(?:second|next)[- ]to[- ]last|last[- ]but[- ]one|penultimate|"
        rf"[a-z][\w-]*'s)\s+){{0,3}}{alias_pattern}"
        rf"(?:\s+(?:(?:is|looks?|seems?)\s+(?:(?:immediately|directly|just)\s+)?(?:nearest|nearer|closest)|"
        rf"(?:(?:far|farthest|furthest|extreme)\s+)?(?:left|right)|"
        rf"(?:(?:immediately|directly|just)\s+)?(?:nearest|nearer|closest)\s+(?:to\s+)?"
        rf"(?:(?:me|us|you)|(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        rf"[a-z][\w'-]*(?:\s+[a-z][\w'-]*){{0,2}})|"
        rf"you\s+(?:(?:can|could)\s+)?(?:see|find|notice)|"
        rf"(?:nearest|nearer|closest)\s+(?:to\s+)?"
        rf"(?:(?:me|us|you)|(?:(?:the|a|an|this|that|my|our|your)\s+)?"
        rf"[a-z][\w'-]*(?:\s+[a-z][\w'-]*){{0,2}})|"
        rf"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\s+"
        rf"(?:from|on|to)\s+(?:(?:the|your|my|our)\s+)?(?:left|right)|"
        rf"(?:number\s+|#\s*)(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)|"
        rf"(?:marked|numbered|labelled|labeled|bearing)\s+"
        rf"(?:(?:number\s+|#\s*)?"
        rf"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+))|"
        rf"with\s+(?:(?:the\s+)?(?:number|mark|label)\s+|#\s*)?"
        rf"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
        rf"(?:\s+on\s+(?:it|them))?|"
        rf"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        rf"(?:spots?|positions?|places?)\s+(?:from|on|to)\s+"
        rf"(?:(?:the|your|my|our)\s+)?(?:left|right)|"
        rf"(?:(?:immediately|directly|just|right)\s+)?"
        rf"(?:(?:next|adjacent)\s+to|beside|alongside)\s+"
        rf"(?:it|this|that|them|these|those|me|us|you)|"
        rf"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
        rf"(?:\s+(?:spots?|places?|positions?|controls?|knobs?|dials?|switches?))?\s+"
        rf"(?:over|away)\s+from\s+"
        rf"(?:it|this|that|them|these|those)|"
        rf"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        rf"(?:controls?|knobs?|dials?|switches?)\s+to\s+"
        rf"(?:(?:its|their)\s+(?:left|right)|"
        rf"(?:(?:the|your|my|our)\s+)?(?:left|right)(?:\s+of\s+(?:it|this|that|them|these|those))?)|"
        rf"(?:(?:immediately|directly|just|right|diagonally|diametrically)\s+)?"
        rf"(?:opposite|beyond)\s+(?:it|this|that|them|these|those|me|us|you)|"
        rf"(?:(?:immediately|directly|just|right)\s+)?across\s+and\s+"
        rf"(?:(?:immediately|directly|just|right)\s+)?opposite\s+"
        rf"(?:it|this|that|them|these|those|me|us|you)|"
        rf"(?:(?:immediately|directly|just|right)\s+)?across\s+from\s+and\s+"
        rf"(?:(?:immediately|directly|just|right)\s+)?opposite\s+"
        rf"(?:it|this|that|them|these|those|me|us|you)|"
        rf"(?:(?:immediately|directly|just|right)\s+)?across\s+from\s+"
        rf"(?:it|this|that|them|these|those|me|us|you)|"
        rf"at\s+(?:the\s+)?(?:opposite|other|far|outer|edge)\s+"
        rf"(?:corner|edge|side|panel)|"
        rf"(?:(?:immediately|directly|just)\s+)?(?:left|right)\s+of\s+"
        rf"(?:it|this|that|them|these|those|me|us|you)|"
        rf"at\s+(?:the\s+)?(?:far|farthest|furthest|opposite|other)\s+(?:end|side)|"
        rf"(?:(?:immediately|directly|just)\s+)?(?:at|on|to|from)\s+"
        rf"(?:(?:the|your|my|our)\s+)?(?:(?:far|farthest|furthest|extreme)\s+)?(?:left|right)))?|"
        rf"(?:(?:the\s+)?(?:one|whichever|either)\s+"
        rf"(?:(?:marked|numbered|labelled|labeled|bearing)\s+"
        rf"(?:(?:number\s+|#\s*)?"
        rf"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+))|"
        rf"(?:is\s+)?(?:(?:immediately|directly|just)\s+)?(?:nearest|nearer|closest)|"
        rf"(?:is\s+)?(?:farthest|furthest)\s+from\s+(?:me|us|you)|"
        rf"(?:is\s+)?(?:(?:immediately|directly|just|right)\s+)?"
        rf"(?:(?:next|adjacent)\s+to|beside|alongside)\s+"
        rf"(?:it|this|that|them|these|those|me|us|you)|"
        rf"(?:is\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)"
        rf"(?:\s+(?:spots?|places?|positions?|controls?|knobs?|dials?|switches?))?\s+"
        rf"(?:over|away)\s+from\s+(?:it|this|that|them|these|those)|"
        rf"(?:is\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        rf"(?:controls?|knobs?|dials?|switches?)\s+to\s+"
        rf"(?:(?:its|their)\s+(?:left|right)|"
        rf"(?:(?:the|your|my|our)\s+)?(?:left|right)(?:\s+of\s+(?:it|this|that|them|these|those))?)|"
        rf"(?:is\s+)?(?:(?:immediately|directly|just|right|diagonally|diametrically)\s+)?"
        rf"(?:opposite|beyond)\s+(?:it|this|that|them|these|those|me|us|you)|"
        rf"(?:is\s+)?(?:(?:immediately|directly|just|right)\s+)?across\s+and\s+"
        rf"(?:(?:immediately|directly|just|right)\s+)?opposite\s+"
        rf"(?:it|this|that|them|these|those|me|us|you)|"
        rf"(?:is\s+)?(?:(?:immediately|directly|just|right)\s+)?across\s+from\s+and\s+"
        rf"(?:(?:immediately|directly|just|right)\s+)?opposite\s+"
        rf"(?:it|this|that|them|these|those|me|us|you)|"
        rf"(?:is\s+)?(?:(?:immediately|directly|just|right)\s+)?across\s+from\s+"
        rf"(?:it|this|that|them|these|those|me|us|you)|"
        rf"(?:is\s+)?at\s+(?:the\s+)?(?:far|farthest|furthest|opposite|other)\s+(?:end|side)|"
        rf"(?:is\s+)?(?:(?:immediately|directly|just)\s+)?(?:at|on|to)\s+"
        rf"(?:(?:the|your|my|our)\s+)?(?:(?:far|farthest|furthest|extreme)\s+)?(?:left|right)))|"
        rf"(?:(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        rf"last|one|either|any)\s+of\s+"
        rf"(?:the\s+)?{alias_pattern})|"
        rf"(?:(?:the\s+)?(?:number|no\.?|#)\s*"
        rf"(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+))|"
        rf"(?:(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)"
        rf"(?:\s+(?:one|control))?(?:\s+from\s+(?:the\s+)?(?:left|right))?)|"
        rf"(?:(?:the\s+)?(?:nearest|nearer|closest)\s+of\s+(?:them|these|those))"
        rf")"
    )
    clauses = _safety_clauses(text)
    if category == "ignition":
        # Establish a gas appliance independently from its later control
        # reference. This pre-pass handles ordinary cross-sentence possessives
        # such as "This is a gas cooker. Turn its dial." while retaining the
        # normal representation, negation, adult-owner, and observation gates.
        appliance_active = False
        for clause in clauses:
            entity = re.search(rf"\b{_IGNITION_APPLIANCE_ENTITY}\b", clause)
            if entity:
                represented = bool(
                    re.search(
                        r"\b(?:toy|pretend|play|model|paper|cardboard|picture|photo|drawing|"
                        r"diagram|illustration)\b[^.!?;\n]{0,40}$",
                        clause[: entity.start()],
                    )
                    or re.search(
                        r"^\s+(?:in|on|of)\s+(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                        r"(?:picture|photo|drawing|diagram|illustration|worksheet|page|screen|card)\b",
                        clause[entity.end() :],
                    )
                )
                appliance_active = not represented
            if not appliance_active:
                continue
            for action in re.finditer(
                rf"\b(?P<verb>[a-z]+(?:\s+(?:back|off|on|up|down|over))?)\s+"
                rf"(?:(?:its|their|the|this|that)\s+)?{alias_pattern}\b",
                clause,
            ):
                if _SAFE_HIGH_RISK_REFERENCE_ACTION.fullmatch(action.group("verb")):
                    continue
                if _action_is_negated(clause, action.start()):
                    continue
                if _action_owner(clause, action.start(), action.end()) == "adult":
                    continue
                if _action_is_imperative(clause, action.start()) or _child_directly_owns_action(clause, action.start()):
                    return True
            for action in re.finditer(
                rf"\b(?P<verb>[a-z]+(?:\s+(?:back|off|on|up|down|over))?)\s+"
                rf"(?:(?:the|this|that|its|their)\s+)?"
                rf"(?:(?:end|far[- ]end|outermost|innermost|extreme|edge|corner|panel|side)\s+{alias_pattern}|"
                rf"{alias_pattern}\s+at\s+(?:the\s+)?(?:far\s+)?end)\b"
                rf"(?=[^.!?;\n]{{0,32}}\b(?:on|of)\s+"
                rf"(?:(?:the|this|that|my|our|your)\s+)?{_IGNITION_APPLIANCE_ENTITY}\b)",
                clause,
            ):
                if _SAFE_HIGH_RISK_REFERENCE_ACTION.fullmatch(action.group("verb")):
                    continue
                if _action_is_negated(clause, action.start()):
                    continue
                if _action_owner(clause, action.start(), action.end()) == "adult":
                    continue
                if _action_is_imperative(clause, action.start()) or _child_directly_owns_action(
                    clause, action.start()
                ):
                    return True
    target_is_active = False
    target_was_named = False
    for clause in clauses:
        explicit_targets = list(re.finditer(rf"\b{target_pattern}\b", clause))
        operational_targets = [
            target for target in explicit_targets if not _high_risk_target_is_represented(clause, target)
        ]
        if category == "ignition" and not target_is_active:
            # An unresolved possessive gas-control alias paired with a force or
            # pressure operation is high-risk enough to fail closed. Ordinary
            # knob references remain benign without that operation signal.
            for unresolved in re.finditer(
                r"\b(?P<verb>(?:apply|introduce|exert)\s+(?:force|pressure))\s+"
                r"(?:to|on|against)\s+(?:its|their)\s+(?:knobs?|dials?|valves?)\b",
                clause,
            ):
                if _action_is_negated(clause, unresolved.start()):
                    continue
                if _action_owner(clause, unresolved.start(), unresolved.end()) == "adult":
                    continue
                if _action_is_imperative(clause, unresolved.start()) or _child_directly_owns_action(
                    clause, unresolved.start()
                ):
                    return True
        alias_is_clearly_benign = bool(
            re.search(
                rf"\b(?:toy|pretend|play|model|paper|cardboard|picture|photo|drawing|diagram|"
                rf"radio|thermostat|clock|watch|timer|compass|speedometer)\s+"
                rf"{alias_pattern}\b",
                clause,
            )
        )
        if operational_targets:
            target_is_active = True
            target_was_named = True
        elif explicit_targets:
            target_is_active = False
            target_was_named = True
        elif category == "ignition" and re.search(
            rf"\b{_IGNITION_APPLIANCE_ENTITY}\b",
            clause,
        ):
            # Establish a concrete gas appliance across sentence boundaries so
            # a later possessive control operation ("Turn its dial") cannot
            # escape merely because the appliance and control are separated.
            target_is_active = True
            target_was_named = True
        elif target_was_named and alias_is_clearly_benign:
            target_is_active = False
        for target in operational_targets:
            for relational_action in re.finditer(
                r"\b(?P<verb>[a-z]+(?:s|ed|ing)?"
                r"(?:\s+(?:(?:both|all)\s+(?:of\s+)?)?(?:your\s+)?"
                r"(?:fingers?|hands?|palms?|thumbs?))?)\s+"
                r"(?:(?:[a-z]+ly|very|really|quite|hard)\s+){0,3}"
                r"(?:(?:around|about|up)\s+)?"
                r"(?P<relation>with|at|over|against|around|on|onto)\s+"
                r"(?P<gap>[^.!?;\n]{0,32})$",
                clause[: target.start()],
            ):
                if re.search(
                    r"\b(?:above|across|alongside|below|beneath|beside|between|by|from|inside|"
                    r"near|next\s+to|outside|through|toward|under|within)\b",
                    relational_action.group("gap"),
                ):
                    continue
                reference_phrase = f"{relational_action.group('verb')} {relational_action.group('relation')}"
                if _SAFE_HIGH_RISK_REFERENCE_ACTION.fullmatch(
                    relational_action.group("verb")
                ) or _SAFE_HIGH_RISK_REFERENCE_ACTION.fullmatch(reference_phrase):
                    continue
                if _high_risk_action_is_explicitly_non_contact(
                    clause,
                    relational_action.start(),
                    relational_action.end(),
                ):
                    continue
                if _action_is_negated(clause, relational_action.start()):
                    continue
                if _action_owner(clause, relational_action.start(), relational_action.end()) == "adult":
                    continue
                if _action_is_imperative(clause, relational_action.start()) or _child_directly_owns_action(
                    clause, relational_action.start()
                ):
                    return True
            # Bind an open-class child operation through a concrete
            # intermediary to the control itself ("push the block against the
            # gas knob"). The relation supplies the physical contact signal;
            # observational verbs, adult ownership, representation, and
            # negation keep their existing exemptions.
            target_prefix = clause[: target.start()]
            for mediated_action in _HIGH_RISK_ACTION_HEAD.finditer(target_prefix):
                mediated_tail = re.fullmatch(
                    r"\s+(?P<object>(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                    r"[a-z][\w'-]*(?:\s+[a-z][\w'-]*){0,4})\s+"
                    r"(?P<relation>against|around|below|beneath|between|into|on|onto|over|under|underneath)\s+"
                    r"(?P<gap>[^.!?;\n]{0,24})$",
                    target_prefix[mediated_action.end() :],
                )
                if not mediated_tail:
                    continue
                if re.search(
                    r"\b(?:above|across|alongside|below|beneath|beside|between|by|from|inside|"
                    r"near|next\s+to|outside|through|toward|under|within)\b",
                    mediated_tail.group("gap"),
                ):
                    continue
                if re.fullmatch(
                    rf"(?:{_ADULT_OWNER}|{_CHILD_OWNER})",
                    mediated_action.group("verb"),
                ):
                    continue
                if _SAFE_HIGH_RISK_REFERENCE_ACTION.fullmatch(mediated_action.group("verb")):
                    continue
                if _high_risk_action_is_explicitly_non_contact(
                    clause,
                    mediated_action.start(),
                    mediated_action.end(),
                ):
                    continue
                if _action_is_negated(clause, mediated_action.start()):
                    continue
                if _action_owner(clause, mediated_action.start(), mediated_action.end()) == "adult":
                    continue
                if _action_is_imperative(clause, mediated_action.start()) or _child_directly_owns_action(
                    clause, mediated_action.start()
                ):
                    return True
            # Accept a bounded noun phrase between an operation and the
            # explicit control. This captures ordinary quantifiers and owner
            # phrases ("half the", "whichever", "the stove's") without an
            # ever-growing determiner vocabulary. Grammatical ownership still
            # gates the operation, and observational verbs remain safe.
            for action in _HIGH_RISK_ACTION_HEAD.finditer(target_prefix):
                object_prefix = target_prefix[action.end() :]
                if not _HIGH_RISK_OBJECT_PREFIX.fullmatch(object_prefix):
                    continue
                if re.search(
                    r"\b(?:above|across|against|alongside|around|below|beneath|"
                    r"beside|between|by|during|from|in|inside|into|near|next\s+to|on|"
                    r"onto|outside|over|through|toward|under|using|while|with|within)\b",
                    object_prefix,
                ):
                    continue
                if re.fullmatch(
                    rf"(?:{_ADULT_OWNER}|{_CHILD_OWNER})",
                    action.group("verb"),
                ):
                    continue
                if _SAFE_HIGH_RISK_REFERENCE_ACTION.fullmatch(action.group("verb")):
                    continue
                if _high_risk_action_is_explicitly_non_contact(
                    clause,
                    action.start(),
                    action.end(),
                ):
                    continue
                if _action_is_negated(clause, action.start()):
                    continue
                if _action_owner(clause, action.start(), action.end()) == "adult":
                    continue
                if _action_is_imperative(clause, action.start()) or _child_directly_owns_action(clause, action.start()):
                    return True
        if not target_is_active:
            continue
        # Once a high-risk control or power tool is named, bind later operations
        # to a pronoun, possessive reference, or repeated head noun. This avoids
        # an open-ended operation-verb list while preserving observational
        # follow-ups such as "draw it" and ordinary actions on a nearer object.
        for action in re.finditer(
            r"\b(?P<verb>[a-z]+(?:\s+(?:back|off|on|up|down|over))?)\s+"
            rf"(?:(?:{_HAZARD_ANAPHORA})\b|(?:its|their)\s+[a-z][\w-]*\b|"
            rf"{alias_reference}\b)",
            clause,
        ):
            if _SAFE_HIGH_RISK_REFERENCE_ACTION.fullmatch(action.group("verb")):
                continue
            if _high_risk_action_is_explicitly_non_contact(
                clause,
                action.start(),
                action.end(),
            ):
                continue
            if _action_is_negated(clause, action.start()):
                continue
            if _action_owner(clause, action.start(), action.end()) == "adult":
                continue
            if _action_is_imperative(clause, action.start()) or _child_directly_owns_action(clause, action.start()):
                return True
        for mediated_action in _HIGH_RISK_ACTION_HEAD.finditer(clause):
            mediated_tail = re.fullmatch(
                rf"\s+(?P<object>(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                rf"[a-z][\w'-]*(?:\s+[a-z][\w'-]*){{0,4}})\s+"
                rf"(?P<relation>against|around|below|beneath|between|into|on|onto|over|under|underneath)\s+"
                rf"(?P<target>{_HAZARD_ANAPHORA}|(?:its|their)\s+[a-z][\w-]*|"
                rf"{alias_reference})\b(?:\s+and\s+"
                rf"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                rf"[a-z][\w-]*(?:\s+[a-z][\w-]*){{0,2}})?",
                clause[mediated_action.end() :],
            )
            if not mediated_tail:
                continue
            if _SAFE_HIGH_RISK_REFERENCE_ACTION.fullmatch(mediated_action.group("verb")):
                continue
            if _high_risk_action_is_explicitly_non_contact(
                clause,
                mediated_action.start(),
                mediated_action.end(),
            ):
                continue
            if _action_is_negated(clause, mediated_action.start()):
                continue
            if _action_owner(clause, mediated_action.start(), mediated_action.end()) == "adult":
                continue
            if _action_is_imperative(clause, mediated_action.start()) or _child_directly_owns_action(
                clause, mediated_action.start()
            ):
                return True
        for action in re.finditer(
            r"\b(?P<verb>(?:make|perform|do|apply|introduce|effect|exert)\b"
            r"[^.!?;\n]{0,40}\b(?:adjustment|change|movement|turn|setting|pressure|force))\s+"
            r"(?:to|on|against)\s+"
            rf"(?:{_HAZARD_ANAPHORA}|(?:its|their)\s+[a-z][\w-]*)\b",
            clause,
        ):
            if _action_is_negated(clause, action.start()):
                continue
            if _action_owner(clause, action.start(), action.end()) == "adult":
                continue
            if _action_is_imperative(clause, action.start()) or _child_directly_owns_action(clause, action.start()):
                return True
        for action in re.finditer(
            r"\b(?P<verb>[a-z]+(?:\s+down)?)\s+"
            r"(?:(?:very|really|quite|slightly|gently|firmly|hard)\s+){0,2}"
            r"(?:against|into|onto|on|at)\s+"
            rf"(?:{_HAZARD_ANAPHORA}|(?:its|their)\s+[a-z][\w-]*|{alias_reference})\b",
            clause,
        ):
            if _SAFE_HIGH_RISK_REFERENCE_ACTION.fullmatch(action.group("verb")):
                continue
            if _high_risk_action_is_explicitly_non_contact(
                clause,
                action.start(),
                action.end(),
            ):
                continue
            if _action_is_negated(clause, action.start()):
                continue
            if _action_owner(clause, action.start(), action.end()) == "adult":
                continue
            if _action_is_imperative(clause, action.start()) or _child_directly_owns_action(clause, action.start()):
                return True
        if category == "sharp_tools":
            for action in re.finditer(
                r"\b(?P<verb>insert|load|put|place|feed)\s+"
                r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                r"(?:nails?|fasteners?|staples?)\b",
                clause,
            ):
                if _action_is_negated(clause, action.start()):
                    continue
                if _action_owner(clause, action.start(), action.end()) == "adult":
                    continue
                if _action_is_imperative(clause, action.start()) or _child_directly_owns_action(clause, action.start()):
                    return True
            for action in re.finditer(
                r"\b(?P<verb>slide)\s+"
                r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
                r"(?:nails?|fasteners?|staples?)\s+(?:in|into|inside(?:\s+of)?)\s+"
                rf"(?:{_HAZARD_ANAPHORA})\b",
                clause,
            ):
                if _action_is_negated(clause, action.start()):
                    continue
                if _action_owner(clause, action.start(), action.end()) == "adult":
                    continue
                if _action_is_imperative(clause, action.start()) or _child_directly_owns_action(clause, action.start()):
                    return True
        if not explicit_targets and not re.search(rf"\b{alias_pattern}\b", clause):
            # A later explicit direct object becomes the nearest referent. Do
            # not let a hazard named several clauses earlier taint an unrelated
            # instruction such as "Pick up the paper. Fold it."
            direct_object = re.search(
                r"\b[a-z]+(?:\s+(?:back|off|on|up|down|over|out))?\s+"
                r"(?:the|a|an|this|that|my|our|your)\s+"
                r"[a-z][\w-]*(?:\s+[a-z][\w-]*){0,3}\b",
                clause,
            )
            if direct_object:
                target_is_active = False
    return False


def _candidate_hazards(*, materials: Any, instructions: Any) -> set[str]:
    """Find concrete hazards independent of child-visible field ordering."""

    instructions = [_hazard_text(text) for text in _text_fragments(instructions)]
    hazards = _hazards_in_value(materials)
    material_text = " ".join(_text_fragments(materials)).casefold()
    instruction_fragments = _text_fragments(instructions)
    instruction_text = " ".join(instruction_fragments).casefold()
    combined_text = f"{material_text} {instruction_text}"
    if (
        _has_conductive_appliance_action(combined_text)
        or _has_powered_appliance_water_action(combined_text)
        or any(_has_powered_appliance_water_action(fragment) for fragment in instruction_fragments)
    ):
        hazards.add("electrical")
    for category in _FAIL_CLOSED_HIGH_RISK_CONTROL_TARGETS:
        if _has_fail_closed_high_risk_control_action(combined_text, category) or any(
            _has_fail_closed_high_risk_control_action(fragment, category) for fragment in instruction_fragments
        ):
            hazards.add(category)
    conductive_appliance_materials = _has_conductive_appliance_pair(material_text)
    conductive_appliance_context = _has_conductive_appliance_pair(combined_text)
    established_conductive_relation = any(
        _establishes_effective_conductive_relation(clause)
        for clause in _instruction_clauses({"materials": materials, "instructions": instructions})
    )
    clauses = _instruction_clauses(instructions)
    # Structured contracts do not guarantee that prose fields appear in narrative
    # order. Resolve the candidate entity set first so "Pour it" cannot evade the
    # gate merely because another visible field names bleach later in the payload.
    active_hazards = set(hazards)
    for clause in clauses:
        active_hazards.update(_anaphoric_hazards_in_value(clause.casefold()))
    prior_instruction_text = ""
    for clause in clauses:
        lowered = clause.casefold()
        conductive_clause = _mask_ineligible_conductive_actions(
            lowered,
            adult_owned_is_safe=False,
        )
        if conductive_appliance_context and _has_contextual_conductive_appliance_action(
            conductive_clause,
            paired_entity_context=conductive_appliance_materials,
            established_relation=established_conductive_relation,
        ):
            hazards.add("electrical")
        explicit_hazards = _hazards_in_value(lowered)
        for name, (hazard_target, hazard_action) in _HAZARD_ACTION_LANGUAGE.items():
            if name not in explicit_hazards and name not in active_hazards:
                continue
            concrete_targets = list(re.finditer(rf"\b{hazard_target}\b", lowered))
            if concrete_targets and all(
                _high_risk_target_is_represented(lowered, target) for target in concrete_targets
            ):
                continue
            if _high_risk_clause_only_references_represented_targets(lowered, name):
                continue
            for action in re.finditer(rf"\b{hazard_action}\b", lowered):
                distributive_antecedent = _nearest_distributive_antecedent_is_hazard(
                    f"{prior_instruction_text} {lowered[: action.start()]}",
                    target_pattern=hazard_target,
                )
                if not _hazard_action_is_bound(
                    lowered,
                    target_pattern=hazard_target,
                    action=action,
                    target_is_active=name in active_hazards,
                    carry_action_pattern=_HAZARD_CARRY_ACTION_LANGUAGE[name],
                    distributive_antecedent_is_hazard=distributive_antecedent,
                    antecedent_context=f"{prior_instruction_text} {lowered[: action.start()]}",
                ):
                    continue
                if not _action_is_negated(lowered, action.start()):
                    hazards.add(name)
                    break
        prior_instruction_text = f"{prior_instruction_text} {lowered}".strip()
    return hazards


_CANDIDATE_SELF_HARM_FINITE_TARGET = (
    r"(?:kill|hurt|harm|cut|stab|burn|hang|drown|shoot|suffocate|choke)\s+your(?:self|selves)|"
    r"end\s+your\s+(?:life|lives)|take\s+your\s+(?:own\s+)?(?:life|lives)|"
    r"commit\s+suicide|self[- ]harm|overdose|take\s+(?:an?|the)\s+overdose|"
    r"poison\s+your(?:self|selves)|slit\s+your\s+wrists?|"
    r"jump\s+off\s+(?:(?:the|a|your)\s+)?(?:roof|bridge|cliff)|die"
)
_CANDIDATE_SELF_HARM_GERUND_TARGET = (
    r"(?:killing|hurting|harming|cutting|stabbing|burning|hanging|drowning|shooting|"
    r"suffocating|choking)\s+your(?:self|selves)|ending\s+your\s+(?:life|lives)|"
    r"taking\s+your\s+(?:own\s+)?(?:life|lives)|committing\s+suicide|self[- ]harming|overdosing|"
    r"taking\s+(?:an?|the)\s+overdose|poisoning\s+yourself|slitting\s+your\s+wrists?|"
    r"jumping\s+off\s+(?:(?:the|a|your)\s+)?(?:roof|bridge|cliff)|dying"
)


def _is_protective_self_harm_clause(clause: str) -> bool:
    """Authenticate complete educational, prevention, or support clauses."""

    nominal = r"(?:suicide|self[- ]harm|(?:an?\s+)?overdose)"
    prevention = r"(?:suicide|self[- ]harm|overdose)\s+prevention"
    prevention_topic = rf"{prevention}(?:\s+(?:education|lessons?|teaching|awareness))?"
    positive = (
        r"(?:(?:[a-z]+ly|very)\s+)?"
        r"(?:important|vital|essential|helpful|effective|valuable|worthwhile|necessary)"
    )
    topic_actor = (
        r"(?:someone|somebody|a\s+child|the\s+child|a\s+kid|the\s+kid|"
        r"a\s+student|the\s+student|a\s+person|the\s+person)"
    )
    topic_event = (
        rf"(?:{nominal}\s+(?:(?:comes?|came)\s+up|(?:is|was|gets?|got)\s+"
        rf"(?:mentioned|discussed|raised|brought\s+up))|"
        rf"{topic_actor}\s+(?:(?:talks?|talked|speaks?|spoke|asks?|asked)\s+about\s+"
        rf"{nominal}|(?:mentions?|mentioned|discusses?|discussed|raises?|raised)\s+"
        rf"{nominal}))"
    )
    trusted_person = r"(?:(?:a|the|your|my|our)\s+)?trusted\s+(?:grown[- ]?up|adult|person)"
    harm_experience = (
        rf"(?:"
        rf"(?:thoughts?|urges?|feelings?|ideas?|concerns?|worries)\s+(?:of|about)\s+{nominal}|"
        rf"(?:(?:my|our|your|the)\s+)?(?:suicidal|suicide|self[- ]harm|overdose)\s+"
        rf"(?:thoughts?|urges?|feelings?|ideas?|risk|concerns?|worries)"
        rf")"
    )
    warning_subject = (
        rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
        rf"(?:warning|caution|advice|rule|guidance|safeguard)"
    )
    protective_warning_predicate = (
        r"(?:"
        r"(?:is|was|remains?|stays?|continues?\s+to\s+be)\s+"
        r"(?:(?:fully|clearly|still|always)\s+)*"
        r"(?:relevant|pertinent|valid|effective|binding|applicable|important|necessary|"
        r"required|mandatory|essential|non[- ]optional|not\s+(?:optional|discretionary))|"
        r"(?:is|was|remains?|stays?|continues?)\s+in\s+(?:force|effect)|"
        r"(?:must|should|shall|needs?\s+to|has\s+to)\s+be\s+"
        r"(?:heeded|followed|observed|respected|honored|obeyed)"
        r")"
    )
    protective_warning_continuation = (
        rf"(?:{protective_warning_predicate}|"
        r"(?:(?:fully|clearly|still|always)\s+)*"
        r"(?:relevant|pertinent|valid|effective|binding|applicable|important|necessary|"
        r"required|mandatory|essential|non[- ]optional|not\s+(?:optional|discretionary))|"
        r"(?:(?:fully|clearly|still|always)\s+)*in\s+(?:force|effect))"
    )
    if re.fullmatch(
        rf"{warning_subject}\s+{protective_warning_predicate}"
        rf"(?:\s+and\s+{protective_warning_continuation})+",
        clause,
    ):
        return True
    # Authenticate protective polarity before contextual endorsement checks.
    # These structural forms cover prevention objects and still-valid warning
    # statements without treating surface negation as abandonment of safety.
    if any(
        re.fullmatch(pattern, clause)
        for pattern in (
            rf"(?:i|we)\s+(?:have|had)\s+(?:(?:[a-z]+ly|very)\s+)*"
            rf"no\s+(?:intent|intention|plan|desire|wish|inclination)\s+"
            rf"(?:of\s+(?:intervening|interfering|standing|acting)\s+against|"
            rf"to\s+(?:intervene|interfere|stand|act)\s+against)\s+{prevention_topic}",
            rf"(?:i|we)\s+(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"(?:interfere\s+with|oppose|block|obstruct|contest)\s+"
            rf"(?:(?:the|this|that)\s+)?{prevention_topic}",
            rf"(?:i|we)\s+(?:have|had)\s+(?:(?:[a-z]+ly|very)\s+)*"
            rf"no\s+(?:intent|intention|plan|desire|wish|inclination)\s+"
            rf"(?:of\s+(?:interfering\s+with|opposing|blocking|obstructing)|"
            rf"to\s+(?:interfere\s+with|oppose|block|obstruct))\s+"
            rf"(?:(?:the|these|those|your|our)\s+)?(?:efforts?|work|steps?|actions?|programs?|campaigns?)\s+"
            rf"to\s+(?:prevent|avoid|discourage|reduce|stop)\s+{nominal}",
            rf"(?:i|we)\s+(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"stand\s+in\s+the\s+way\s+of\s+(?:(?:your|the|these|those)\s+)?"
            rf"(?:efforts?|work|steps?|actions?|programs?|campaigns?)\s+that\s+"
            rf"(?:prevent|prevents|avoid|avoids|discourage|discourages|reduce|reduces|stop|stops)\s+{nominal}",
            rf"(?:i|we)\s+(?:have|had)\s+(?:(?:[a-z]+ly|very)\s+)*"
            rf"no\s+(?:intent|intention|plan|desire|wish|inclination)\s+"
            rf"(?:of\s+(?:interfering\s+with|opposing|blocking|obstructing)|"
            rf"to\s+(?:interfere\s+with|oppose|block|obstruct))\s+{prevention_topic}",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:continues?\s+to\s+be|remains?|stays?)\s+"
            rf"(?:(?:[a-z]+ly|very|still)\s+)*"
            rf"(?:relevant|pertinent|valid|effective|binding|applicable|important|necessary|"
            rf"required|mandatory|essential|non[- ]optional)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|remains?|stays?|continues?\s+to\s+be)\s+"
            rf"(?:(?:fully|still|clearly)\s+)*"
            rf"(?:required|mandatory|binding|essential|non[- ]optional)\s+and\s+"
            rf"(?:is|remains?|stays?|continues?\s+to\s+be)\s+"
            rf"(?:(?:fully|still|clearly)\s+)*"
            rf"(?:required|mandatory|binding|essential|non[- ]optional)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|remains?|stays?)\s+not\s+"
            rf"(?:obsolete|outdated|irrelevant|invalid|ineffective|meaningless|void)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was|remains?|stays?)\s+not\s+(?:optional|discretionary)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:remains?|stays?)\s+in\s+(?:force|effect)\s+and\s+"
            rf"(?:is|remains?|stays?)\s+(?:(?:still|fully)\s+)?"
            rf"(?:not\s+(?:optional|discretionary)|required|mandatory|binding|essential)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was|remains?|stays?)\s+not\s+(?:optional|discretionary)\s+and\s+"
            rf"(?:cannot|can't|can\s+not|must\s+not|mustn't)\s+be\s+"
            rf"(?:set\s+aside|ignored|disregarded|dismissed|discarded|defied)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:has|had)\s+(?:"
            rf"served\s+(?:its|the)\s+purpose|"
            rf"done\s+(?:its|the)\s+(?:job|work|duty)|"
            rf"(?:accomplished|fulfilled|completed)\s+(?:its|the)\s+(?:purpose|function|role))\s+by\s+"
            rf"(?:keeping|protecting)\s+(?:children|kids?|people|families)\s+safe\s+and\s+"
            rf"(?:is|remains?|stays?)\s+"
            rf"(?:required|mandatory|binding|essential|non[- ]optional)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was|remains?|stays?)\s+"
            rf"(?:required|mandatory|binding|essential|non[- ]optional)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was|remains?|stays?)\s+"
            rf"(?:required|mandatory|binding|essential|non[- ]optional)\s*,?\s+not\s+"
            rf"(?:optional|advisory|discretionary|merely\s+(?:an?\s+)?suggestion)",
            rf"(?:do\s+not|don't|never)\s+(?:treat|regard|view|consider)\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+(?:as\s+)?"
            rf"(?:optional|advisory|discretionary|(?:merely|just|only)\s+"
            rf"(?:an?\s+)?(?:suggestion|option|recommendation))",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was)\s+not\s+(?:merely|just|only)\s+"
            rf"(?:an?\s+)?(?:suggestion|option|recommendation)",
            rf"(?:disregarding|ignoring|dismissing|defying)\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was|would\s+be)\s+not\s+"
            rf"(?:okay|ok|fine|acceptable|permitted|reasonable|safe)",
            rf"(?:it\s+(?:is|was|would\s+be|will\s+be)\s+)?"
            rf"(?:unsafe|dangerous|wrong|unwise|unacceptable)\s+to\s+"
            rf"(?:disregard|ignore|dismiss|discard|defy)\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)",
            rf"(?:please\s+)?(?:disregard|ignore|dismiss|reject)\s+"
            rf"(?:any|every|the)\s+(?:claim|suggestion|statement|idea)\s+that\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was|has\s+become)\s+"
            rf"(?:obsolete|outdated|irrelevant|invalid|optional|ineffective|meaningless|void)",
            rf"(?:please\s+)?keep\s+(?:heeding|following|observing|respecting|honoring|"
            rf"taking\s+seriously)\s+(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)",
            rf"(?:please\s+)?(?:do\s+not|don't|never)\s+"
            rf"(?:disregard|ignore|dismiss|discard|defy)\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)",
            rf"(?:following|heeding|observing|respecting|honoring)\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was|remains?|stays?)\s+"
            rf"(?:required|necessary|mandatory|essential|important)",
            rf"(?:you|the\s+child|children|kids?)\s+(?:is|are)\s+not\s+free\s+to\s+"
            rf"(?:disregard|ignore|dismiss|discard|defy)\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)",
            rf"under\s+no\s+circumstances\s+"
            rf"(?:disregard|ignore|dismiss|discard|defy)\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:must|should|shall|needs?\s+to|has\s+to)\s+be\s+"
            rf"(?:heeded|followed|observed|respected|honored|obeyed)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was|remains?|stays?)\s+"
            rf"(?:required|mandatory|binding|essential|non[- ]optional)\s+(?:and\s+)?"
            rf"(?:must|should|shall|needs?\s+to|has\s+to)\s+be\s+"
            rf"(?:heeded|followed|observed|respected|honored|obeyed)",
            rf"(?:it\s+(?:is|was)\s+)?"
            rf"(?:required|necessary|mandatory|essential|important)\s+to\s+"
            rf"(?:heed|follow|observe|respect|honor|obey)\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)",
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:is|was|became|has\s+become|no\s+longer\s+applies|ceased\s+to\s+apply)\s+"
            rf"(?:obsolete|outdated|superseded)?\s*because\s+"
            rf"(?:(?:a|the)\s+)?(?:stricter|stronger|safer|more\s+protective)\s+"
            rf"(?:warning|caution|rule|guidance|safeguard|prohibition)\s+"
            rf"(?:replaces?|replaced|supersedes?|superseded)\s+(?:it|that|this)",
            rf"(?:(?:the|this|that)\s+)?(?:(?:earlier|prior|previous)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:expired|lapsed|ended|ran\s+its\s+course|was\s+spent)\s+"
            rf"(?:when|after|because)\s+(?:(?:a|the)\s+)?"
            rf"(?:stricter|stronger|safer|more\s+protective)\s+"
            rf"(?:warning|caution|rule|guidance|safeguard|prohibition)\s+"
            rf"(?:took\s+effect|came\s+into\s+(?:force|effect)|replaced|superseded)\s*"
            rf"(?:it|that|this)?",
            rf"(?:no\s+one|nobody)\s+(?:should|must|can|could|may|might|will|would)\s+"
            rf"(?:respect|honor|support|endorse|approve\s+of)\s+"
            rf"(?:(?:a|the|your)\s+)?(?:decision|choice|plan)\s+to\s+{nominal}",
            rf"(?:i|we)\s+(?:respect|honor|support|endorse|welcome|affirm|approve\s+of)\s+"
            rf"(?:your|the)\s+(?:decision|choice|plan)\s+to\s+(?:"
            rf"(?:seek|get|ask\s+for|reach\s+out\s+for)\s+help\s+"
            rf"(?:after|following|because\s+of|for)\s+{harm_experience}|"
            rf"(?:tell|find|call|contact|notify|show|talk\s+to|go\s+to|reach\s+out\s+to|"
            rf"get\s+help\s+from)\s+{trusted_person}\s+"
            rf"(?:about|after|following|because\s+of|for)\s+{harm_experience}"
            rf")",
        )
    ):
        return True
    return bool(
        re.fullmatch(
            rf"(?:"
            rf"(?:i|we)\s+(?:have|made|created|wrote|prepared)\s+"
            rf"(?:(?:a|my|our)\s+)?(?:{prevention}\s+plan|plan\s+to\s+prevent\s+{nominal})|"
            rf"(?:my|our)\s+{prevention}\s+plan\s+(?:is|remains?)\s+"
            rf"(?:ready|complete|finished)|"
            rf"(?:my|our|the)\s+(?:school\s+)?lesson\s+plan\s+"
            rf"(?:discusses|covers|teaches|explains)\s+{prevention}|"
            rf"(?:i|we)\s+(?:have|made|created|wrote|prepared)\s+"
            rf"(?:(?:a|my|our)\s+)?safety\s+plan\s+because\s+"
            rf"(?:(?:my|our|a|the)\s+)?(?:friend|classmate|student|child|kid)\s+"
            rf"(?:mentioned|discussed|asked\s+about)\s+{nominal}|"
            rf"{prevention_topic}\s+(?:is|remains?)\s+{positive}|"
            rf"{prevention_topic}\s+(?:saves?|protects?|helps?)\s+"
            rf"(?:lives?|people|children|families)|"
            rf"{nominal}\s+(?:is|remains?)\s+"
            rf"(?:preventable|(?:an?\s+)?preventable\s+(?:crisis|harm|outcome|tragedy))|"
            rf"(?:preventing|avoiding|discouraging)\s+{nominal}\s+"
            rf"(?:(?:is|remains?)\s+{positive}|"
            rf"(?:saves?|protects?|helps?)\s+(?:lives?|people|children|families))|"
            rf"(?:learning|reading|talking|teaching|asking)\s+about\s+{prevention}\s+"
            rf"(?:can|could|may|might|does|will)\s+"
            rf"(?:help|save\s+lives?|support\s+people|reduce\s+risk)|"
            rf"(?:i|we)\s+(?:back|endorse|support|promote|recommend|approve\s+of)\s+"
            rf"{prevention_topic}|"
            rf"(?:i|we)\s+(?:back|endorse|support|promote|recommend|approve\s+of)\s+"
            rf"(?:efforts?|work|programs?|campaigns?)\s+to\s+"
            rf"(?:prevent|avoid|discourage|reduce)\s+{nominal}|"
            rf"(?:i|we)\s+(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"(?:interfere\s+with|oppose|block|obstruct)\s+"
            rf"(?:(?:your|the|these|those)\s+)?(?:efforts?|work|steps?|actions?)\s+to\s+"
            rf"(?:prevent|avoid|discourage|reduce|stop)\s+{nominal}|"
            rf"(?:i|we)\s+(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"(?:interfere\s+with|oppose|block|obstruct|stand\s+in\s+the\s+way\s+of)\s+"
            rf"{prevention}\s+(?:efforts?|work|steps?|actions?|programs?|campaigns?)|"
            rf"(?:i|we)\s+(?:shall|will|would|should|can|could|may|might)\s+"
            rf"(?:allow|permit)\s+{prevention}(?:\s+(?:efforts?|work|steps?|actions?))?\s+"
            rf"to\s+(?:occur|continue|proceed|carry\s+on)|"
            rf"(?:i|we)\s+(?:shall|will|would|should|can|could|may|might)\s+let\s+"
            rf"{prevention}(?:\s+(?:efforts?|work|steps?|actions?))?\s+"
            rf"(?:occur|continue|proceed|carry\s+on)|"
            rf"{prevention}(?:\s+(?:efforts?|work|steps?|actions?))?\s+"
            rf"(?:should|must|shall|can|could|may|might)\s+be\s+"
            rf"(?:allowed|permitted)\s+to\s+(?:occur|continue|proceed|carry\s+on)|"
            rf"(?:i|we)\s+(?:back|endorse|support|promote|recommend|approve\s+of)\s+"
            rf"(?:efforts?|work|programs?|campaigns?)\s+that\s+"
            rf"(?:prevent|prevents|avoid|avoids|discourage|discourages|reduce|reduces)\s+{nominal}|"
            rf"(?:i|we)\s+(?:(?:firmly|strongly|clearly|unequivocally)\s+)?"
            rf"(?:condemn|denounce|deplore|reject|object\s+to|oppose|"
            rf"stand\s+(?:against|opposed\s+to))\s+{nominal}"
            rf"(?:\s+(?:firmly|strongly|unequivocally))?|"
            rf"(?:i|we)(?:'m|'re|\s+am|\s+are)\s+"
            rf"(?:(?:firmly|strongly)\s+)?opposed\s+to\s+{nominal}|"
            rf"(?:i|we)\s+(?:remain|remains|remained)\s+"
            rf"(?:(?:firmly|strongly)\s+)?opposed\s+to\s+{nominal}|"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:remains?|stays?)\s+in\s+(?:force|effect)|"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)\s+"
            rf"(?:(?:is|remains?|stays?)\s+)"
            rf"(?:(?:[a-z]+ly|very|still)\s+)*"
            rf"(?:relevant|pertinent|valid|effective|binding|applicable|important|necessary)|"
            rf"(?:please\s+)?(?:heed|follow|observe|respect|honor|take\s+seriously)\s+"
            rf"(?:(?:that|the|this)\s+)?{nominal}\s+"
            rf"(?:warning|caution|advice|rule|guidance|safeguard)|"
            rf"(?:i|we)\s+(?:have|raise)\s+no\s+objection\s+to\s+{prevention_topic}|"
            rf"(?:i|we)\s+(?:withdraw|retract|revoke|rescind|drop|abandon)\s+"
            rf"(?:(?:my|our|the|this|that)\s+)?(?:objection|opposition)\s+to\s+"
            rf"{prevention_topic}|"
            rf"(?:i|we)\s+no\s+longer\s+(?:oppose|object\s+to)\s+{prevention_topic}|"
            rf"(?:(?:my|our|the|this|that)\s+)?(?:objection|opposition)\s+to\s+"
            rf"{prevention_topic}\s+(?:(?:is|was)\s+|(?:has|had)\s+been\s+)"
            rf"(?:withdrawn|retracted|revoked|rescinded|dropped|abandoned|lifted)|"
            rf"(?:i|we)\s+(?:give|grant|offer)\s+(?:(?:my|our)\s+)?"
            rf"(?:blessing|approval|support|endorsement)\s+to\s+{prevention_topic}|"
            rf"(?:i|we)\s+(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"(?:oppose|block|discourage|stand\s+against|object\s+to)\s+{prevention_topic}|"
            rf"(?:i|we)\s+(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"(?:oppose|block|discourage|stand\s+against|object\s+to)\s+"
            rf"(?:efforts?|work|programs?|campaigns?)\s+to\s+"
            rf"(?:prevent|avoid|discourage|reduce)\s+{nominal}|"
            rf"{nominal}\s+(?:should|must|can)\s+(?:always\s+)?be\s+"
            rf"(?:prevented|discouraged|avoided|taken\s+seriously|addressed)|"
            rf"(?:i|we|you|people|children|kids?)\s+(?:should|must|need\s+to|can)\s+"
            rf"(?:always\s+)?(?:prevent|avoid|discourage)\s+{nominal}|"
            rf"(?:do\s+not|don't|never\s*,?)\s+"
            rf"(?:(?:ever|really|ever\s+again)\s+)*"
            rf"(?:encourage|recommend|promote|endorse|choose|try|attempt)\s+{nominal}"
            rf"(?:\s+under\s+any\s+circumstances)?|"
            rf"(?:if|when)\s+{topic_event}\s*,?\s+(?:please\s+)?"
            rf"(?:tell|find|call|contact|notify|show|talk\s+to|go\s+to|reach\s+out\s+to|"
            rf"get\s+help\s+from|ask)\b[^.!?;\n]{{0,80}}\b"
            rf"trusted\s+(?:grown[- ]?up|adult|person)"
            rf"|(?:please\s+)?(?:tell|find|call|contact|notify|show|talk\s+to|go\s+to|"
            rf"reach\s+out\s+to|"
            rf"get\s+help\s+from|ask)\b[^.!?;\n]{{0,80}}\b"
            rf"trusted\s+(?:grown[- ]?up|adult|person)"
            rf"(?:\s+(?:right\s+away|immediately|now))?\s+"
            rf"(?:if|when)\s+{topic_event}"
            rf")",
            clause,
        )
    )


def _is_candidate_warning_nullification(clause: str) -> bool:
    """Reject explicit attempts to make a child-safety warning elective."""

    warning = (
        r"(?:(?:that|the|this|my|our|your)\s+)?"
        r"(?:(?:suicide|self[- ]harm|safety)\s+)?"
        r"(?:warning|caution|rule|message|prohibition|guidance|reminder|instruction|safeguard|protection)"
    )
    child = r"(?:you|the\s+child|children|kids?)"
    elective_status = (
        r"(?:optional|discretionary|elective|voluntary|non[- ]?binding|non[- ]?mandatory|"
        r"advisory|informational|for\s+(?:information|informational)\s+purposes?\s+only|"
        r"for\s+information\s+only)"
    )
    personal_status = (
        r"(?:your|the\s+child's)\s+(?:call|choice|decision)|"
        r"an?\s+(?:(?:personal|individual)\s+)?(?:choice|decision|option)|"
        r"a\s+matter\s+of\s+(?:(?:personal|individual)\s+)?(?:choice|decision|discretion)|"
        r"(?:(?:purely|entirely|wholly)\s+)?an?\s+(?:personal|individual)\s+matter|"
        r"at\s+(?:your|the\s+child's)\s+discretion"
    )
    warning_observance = (
        rf"(?:following|heeding|observing|respecting|honoring|obeying|"
        rf"compliance\s+with|adherence\s+to|observance\s+of)\s+{warning}"
    )
    return any(
        re.fullmatch(pattern, clause)
        for pattern in (
            rf"{warning}\s+(?:is|was|seems?|appears?|remains?|stays?)\s+"
            rf"(?:(?:entirely|wholly|purely|completely)\s+)?voluntary",
            rf"{warning}\s+(?:is|was|seems?|appears?|remains?|stays?)\s+"
            rf"(?:(?:entirely|wholly|purely|completely|merely|just|only)\s+)?"
            rf"{elective_status}"
            rf"(?:\s+(?:advice|guidance|a\s+suggestion|a\s+recommendation|an\s+option))?"
            rf"(?:\s*,?\s+not\s+(?:binding|mandatory|required))?",
            rf"{warning}\s+(?:is|was|seems?|appears?)\s+"
            rf"(?:merely|just|only)\s+(?:advice|guidance|a\s+suggestion|a\s+recommendation|an\s+option)",
            rf"{warning}\s+(?:is|was|seems?|appears?)\s+"
            rf"(?:(?:entirely|wholly)\s+)?up\s+to\s+{child}",
            rf"{warning_observance}\s+"
            rf"(?:is|was|seems?|appears?)\s+(?:(?:entirely|wholly)\s+)?up\s+to\s+{child}",
            rf"{warning_observance}\s+"
            rf"(?:is|was|seems?|appears?|remains?|stays?)\s+"
            rf"(?:(?:entirely|wholly|purely|completely)\s+)?"
            rf"(?:{elective_status}|{personal_status})",
            rf"whether\s+to\s+(?:heed|follow|observe|respect|honor|obey)\s+{warning}\s+"
            rf"(?:is|was|seems?|appears?|remains?|stays?)\s+"
            rf"(?:(?:entirely|wholly|purely|completely)\s+)?"
            rf"(?:{elective_status}|{personal_status})",
            rf"whether\s+{warning}\s+(?:is|was|would\s+be)\s+"
            rf"(?:heeded|followed|observed|respected|honored|obeyed)\s+"
            rf"(?:is|was|seems?|appears?|remains?|stays?)\s+"
            rf"(?:(?:entirely|wholly|purely|completely)\s+)?"
            rf"(?:{elective_status}|{personal_status})",
            rf"it\s+(?:is|was|seems?|appears?)\s+"
            rf"(?:(?:entirely|wholly|purely|completely)\s+)?"
            rf"(?:your|the\s+child's)\s+(?:call|choice|decision)\s+whether\s+to\s+"
            rf"(?:heed|follow|observe|respect|honor|obey)\s+{warning}",
            rf"{child}\s+(?:can|could|may|might)\s+(?:simply\s+)?take\s+or\s+leave\s+{warning}",
            rf"{child}\s+(?:can|could|may|might)\s+(?:choose|decide)\s+"
            rf"whether\s+to\s+(?:heed|follow|observe|respect|honor|obey)\s+{warning}",
            rf"(?:treat|regard|view|consider)\s+{warning}\s+(?:as\s+)?"
            rf"(?:(?:merely|just|only)\s+)?(?:{elective_status}|"
            rf"(?:an?\s+)?(?:suggestion|option|recommendation|guidance))",
            rf"(?:treat|regard|view|consider)\s+{warning}\s+(?:as\s+)?"
            rf"something\s+(?:that\s+)?(?:{child}\s+)?"
            rf"(?:can|could|may|might)\s+(?:simply\s+)?"
            rf"(?:ignore|disregard|dismiss|discard|defy)",
        )
    )


def _is_contextual_candidate_self_harm_endorsement(clause: str) -> bool:
    """Reject semantic permission or encouragement after self-harm context."""

    discourse = r"(?:(?:but|however|still|nevertheless|nonetheless)\s*,?\s*)?"
    child = r"(?:you|the\s+child|children|kids?)"
    anaphora = r"(?:it|that|this|the\s+same|so)"
    self_harm_reference = rf"(?:{anaphora}|suicide|self[- ]harm|(?:an?\s+)?overdose)"
    action = rf"(?:do|try|choose|consider|attempt)(?:ing)?\s+{anaphora}"
    warning = (
        r"(?:warning|caution|advice|rule|message|prohibition|guidance|reminder|"
        r"instruction|safeguard|protection)"
    )
    warning_reference = (
        rf"(?:it|that|this|(?:(?:that|the|this|my|our|your)\s+)?{warning}|"
        rf"what\s+(?:i|we)\s+(?:just\s+)?(?:said|told\s+you))"
    )
    qualifier = r"(?:\s+(?:anyway|regardless|nonetheless|nevertheless))?"
    preference = (
        rf"(?:really\s+)?(?:want\s+to|choose\s+to|decide\s+to|prefer\s+to)|"
        rf"(?:really\s+)?feel\s+like\s+{anaphora}|"
        r"that(?:'s|\s+is)\s+what\s+you\s+want"
    )
    nonintervention = (
        r"(?:interven(?:e|ing|tion)|interfer(?:e|ing|ence)|step(?:ping)?\s+in|"
        r"get(?:ting)?\s+involved|come|coming|stand(?:ing)?|stop(?:ping)?|prevent(?:ing)?|"
        r"contest(?:ing)?|challeng(?:e|ing)?)"
    )
    if any(
        re.fullmatch(pattern, clause)
        for pattern in (
            rf"{discourse}(?:do\s+not|don't|never)\s+(?:count|rely|depend)\s+on\s+"
            rf"(?:me|us)\s+to\s+(?:stop|prevent|dissuade|discourage)\s+(?:you|it|that|this)",
            rf"{discourse}{child}\s+(?:should|must|can|could|may|might|need)\s+not\s+"
            rf"(?:expect|count\s+on|rely\s+on)\s+(?:me|us)\s+to\s+"
            rf"(?:intervene|interfere|get\s+involved|step\s+in|stop|prevent)(?:\s+you)?",
            rf"{discourse}(?:i|we)\s+(?:will|would|shall|plan|intend|mean|choose)\s+"
            rf"(?:remain|stay|keep)\s+(?:silent|quiet|passive|uninvolved)\s+"
            rf"(?:and|while)\s+(?:let|allow|permit)(?:ting)?\s+{child}\s+"
            rf"(?:to\s+)?(?:proceed|continue|carry\s+on|do\s+{anaphora})",
            rf"{discourse}(?:i|we)\s+(?:(?:have|had)\s+)?(?:no|zero)\s+"
            rf"(?:intent|intention|plan)\s+(?:of\s+)?{nonintervention}{qualifier}",
            rf"{discourse}(?:i|we)\s+lack(?:ed)?\s+(?:(?:any|an?|the)\s+)?"
            rf"(?:intent|intention|plan|desire|wish|inclination)\s+to\s+"
            rf"{nonintervention}{qualifier}",
            rf"{discourse}{nonintervention}\s+(?:is|was)\s+not\s+something\s+"
            rf"(?:i|we)\s+(?:intend|intended|plan|planned|mean|meant|aim|aimed)\s+to\s+do{qualifier}",
            rf"{discourse}no\s+{nonintervention}\s+(?:will|would|shall)\s+"
            rf"(?:come|be\s+coming)\s+from\s+(?:me|us){qualifier}",
            rf"{discourse}there\s+(?:will|would|shall)\s+"
            rf"(?:not|never)\s+be\s+(?:any|an?|the)?\s*{nonintervention}\s+"
            rf"(?:from|by)\s+(?:me|us){qualifier}",
            rf"{discourse}there\s+(?:won't|wouldn't|shan't)\s+be\s+"
            rf"(?:any|an?|the)?\s*{nonintervention}\s+(?:from|by)\s+(?:me|us){qualifier}",
            rf"{discourse}there\s+(?:will|would|shall)\s+be\s+no\s+"
            rf"{nonintervention}\s+(?:from|by)\s+(?:me|us){qualifier}",
            rf"{discourse}{nonintervention}\s+(?:is|was|will\s+be|would\s+be)\s+"
            rf"(?:not|never)\s+(?:forthcoming|coming)\s+(?:from|by)\s+(?:me|us){qualifier}",
            rf"{discourse}no\s+{nonintervention}\s+"
            rf"(?:is|was|will\s+be|would\s+be)\s+"
            rf"(?:forthcoming|coming)\s+(?:from|by)\s+(?:me|us){qualifier}",
            rf"{discourse}(?:i|we)\s+(?:intend|plan|mean|aim|choose|decide)\s+to\s+"
            rf"(?:withhold|refuse|deny)\s+{nonintervention}{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will|would|shall|can|could)\s+(?:withhold|refuse|deny)\s+"
            rf"{nonintervention}{qualifier}",
            rf"{discourse}(?:i|we)\s+(?:see|saw|find|found)\s+no\s+reason\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)(?:'m|'re|\s+am|\s+are|\s+was|\s+were)\s+not\s+here\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)\s+(?:accept|support|respect|honor|endorse|affirm|validate|ratify|"
            rf"assent\s+to|approve(?:\s+of)?)\s+"
            rf"your\s+(?:decision|choice)\s+to\s+"
            rf"(?:(?:do|try|attempt|choose)\s+{anaphora}|proceed|continue|carry\s+on)",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:acknowledge|recognize|recognise|regard|view|accept|validate)\s+"
            rf"your\s+(?:decision|choice)\s+(?:as\s+)?"
            rf"(?:legitimate|valid|acceptable|reasonable|permissible)",
            rf"{discourse}(?:i|we)\s+(?:defer|yield)\s+to\s+"
            rf"(?:(?:your|the\s+child's|their)\s+)?"
            rf"(?:decision|choice|judg(?:e)?ment|preference|wishes)",
            rf"{discourse}(?:i|we)\s+(?:leave|entrust|defer)\s+"
            rf"(?:(?:the|this|that|your)\s+)?(?:decision|choice|matter|outcome|it|this|that)\s+"
            rf"(?:entirely\s+|wholly\s+)?to\s+(?:you|{child}|your\s+judg(?:e)?ment)",
            rf"{discourse}(?:i|we)\s+(?:leave|entrust|defer)\s+"
            rf"(?:(?:the|this|that|your)\s+)?(?:decision|choice|matter|outcome|it|this|that)\s+"
            rf"(?:entirely\s+|wholly\s+)?to\s+(?:you|{child}|your\s+judg(?:e)?ment)",
            rf"{discourse}{child}\s+(?:has|have)\s+(?:my|our)\s+"
            rf"(?:assent|approval|permission|endorsement|blessing)\s+to\s+"
            rf"(?:do\s+{anaphora}|proceed|continue|carry\s+on)",
            rf"{discourse}(?:i|we)\s+(?:will\s+not|won't|would\s+not|wouldn't)\s+"
            rf"(?:try|attempt)\s+to\s+talk\s+you\s+out\s+of\s+"
            rf"(?:your|that|the)\s+(?:decision|choice)",
            rf"{discourse}(?:(?:that|the|this)\s+)?{warning}\s+"
            rf"(?:(?:is|was|has\s+(?:become|grown))\s+(?:outdated|obsolete|irrelevant|invalid)|"
            rf"(?:is|was)\s+out\s+of\s+date|has\s+no\s+bearing(?:\s+(?:now|anymore|any\s+longer))?)"
            rf"(?:\s*,?\s*so\s+(?:go\s+ahead|proceed|continue|do\s+{anaphora}))?",
            rf"{discourse}{warning_reference}\s+"
            rf"(?:(?:ceased|stopped)\s+to\s+(?:apply|matter|count)|"
            rf"(?:no\s+longer|does\s+not|doesn't)\s+(?:apply|matter|count)|"
            rf"(?:has|had)\s+(?:lapsed|expired)|"
            rf"(?:has|had)\s+run\s+(?:its|the)\s+course|"
            rf"(?:is|was|has\s+become|had\s+become)\s+spent|"
            rf"(?:is|was)\s+(?:past|beyond)\s+(?:(?:its|the)\s+)?"
            rf"(?:expiration|expiry)(?:\s+date)?|"
            rf"(?:carries?|has|holds?)\s+no\s+"
            rf"(?:force|weight|authority|effect|obligation|requirement|duty|binding\s+power)|"
            rf"(?:no\s+longer|does\s+not|doesn't|did\s+not|didn't)\s+"
            rf"(?:binds?|obligates?|constrains?|governs?|controls?)\s+{child}|"
            rf"(?:places?|imposes?)\s+no\s+(?:obligation|requirement|duty)\s+on\s+{child}|"
            rf"(?:(?:has|had)(?:\s+already)?\s+(?:done|finished|completed)|did)\s+"
            rf"(?:its|the)\s+(?:job|work|role|task|duty)|"
            rf"(?:has|had)\s+(?:accomplished|fulfilled|completed)\s+"
            rf"(?:its|the)\s+(?:purpose|function|role|mission)|"
            rf"(?:accomplished|fulfilled|completed|finished)\s+"
            rf"(?:its|the)\s+(?:purpose|function|role|mission)|"
            rf"(?:has|had)\s+outlived\s+(?:its|the)\s+"
            rf"(?:purpose|usefulness|function|role)|"
            rf"(?:accomplished|fulfilled|completed|finished)\s+"
            rf"(?:its|the)\s+(?:purpose|function|role|mission)|"
            rf"(?:has|had)\s+outlived\s+(?:its|the)\s+"
            rf"(?:purpose|usefulness|function|role)|"
            rf"(?:is|was)\s+no\s+longer\s+in\s+(?:force|effect)|"
            rf"(?:is|was)\s+(?:merely|just|only)\s+"
            rf"(?:(?:an?\s+)?(?:suggestion|option|recommendation)|"
            rf"(?:optional|advisory|discretionary)))",
            rf"{discourse}{warning_reference}\s+"
            rf"(?:(?:is|was|has\s+become|had\s+become)\s+"
            rf"(?:non[- ]?binding|not\s+(?:compulsory|mandatory|required|binding|obligatory|essential)|"
            rf"(?:merely|just|only)?\s*for\s+guidance\s+only)|"
            rf"(?:isn't|wasn't)\s+(?:compulsory|mandatory|required|binding|obligatory|essential)|"
            rf"(?:is|was)\s+(?:merely\s+|just\s+|only\s+|(?:a\s+)?mere\s+)?"
            rf"(?:a\s+)?formality|"
            rf"(?:is|was)\s+(?:an?\s+)?(?:suggestion|recommendation|option)\s*,?\s+"
            rf"not\s+(?:an?\s+)?(?:rule|requirement|obligation|command|safeguard)|"
            rf"(?:has|had)\s+served\s+(?:its|the)\s+purpose)",
            rf"{discourse}{child}\s+(?:is|are|was|were)\s+(?:not|no\s+longer)\s+bound\s+by\s+"
            rf"{warning_reference}",
            rf"{discourse}{child}\s+(?:is|are|was|were)\s+free\s+to\s+"
            rf"(?:disregard|ignore|dismiss|discard|defy)\s+{warning_reference}",
            rf"{discourse}{child}\s+(?:can|could|may|might|should)\s+"
            rf"(?:set|put|cast|brush)\s+{warning_reference}\s+aside",
            rf"{discourse}{warning_reference}\s+"
            rf"(?:can|could|may|might|should)\s+be\s+"
            rf"(?:set|put|cast|brushed)\s+aside",
            rf"{discourse}{warning_reference}\s+"
            rf"(?:can|could|may|might|should)\s+(?:(?:now|safely|freely)\s+)?be\s+"
            rf"(?:(?:now|safely|freely)\s+)?"
            rf"(?:ignored|disregarded|dismissed|discarded|defied|forgotten|overridden)",
            rf"{discourse}(?:treat|regard|view|consider)\s+{warning_reference}\s+(?:as\s+)?"
            rf"(?:optional|advisory|discretionary|(?:merely|just|only)\s+"
            rf"(?:an?\s+)?(?:suggestion|option|recommendation))",
            rf"{discourse}(?:disregarding|ignoring|dismissing|defying)\s+"
            rf"{warning_reference}\s+(?:is|was|would\s+be)\s+"
            rf"(?:okay|ok|fine|acceptable|permitted|reasonable|safe)",
            rf"{discourse}{child}\s+(?:need\s+not|needn't|do\s+not\s+need\s+to|"
            rf"don't\s+need\s+to|do\s+not\s+have\s+to|don't\s+have\s+to)\s+"
            rf"(?:heed|follow|observe|respect|honor|obey)\s+"
            rf"{warning_reference}",
            rf"{discourse}{child}\s+(?:is|are)\s+under\s+no\s+obligation\s+to\s+"
            rf"(?:heed|follow|observe|respect|honor|obey)\s+{warning_reference}",
            rf"{discourse}{child}\s+(?:is|are|was|were)\s+(?:not|no\s+longer)\s+"
            rf"(?:required|obliged)\s+to\s+"
            rf"(?:heed|follow|observe|respect|honor|obey)\s+{warning_reference}",
            rf"{discourse}{child}\s+(?:has|have)\s+no\s+"
            rf"(?:need|reason|obligation|requirement|duty)\s+to\s+"
            rf"(?:heed|follow|observe|respect|honor|obey)\s+{warning_reference}",
            rf"{discourse}{warning_reference}\s+"
            rf"(?:does|did)\s+not\s+need\s+to\s+be\s+"
            rf"(?:heeded|followed|observed|respected|honored|obeyed)",
            rf"{discourse}{warning_reference}\s+"
            rf"(?:need\s+not|needn't)\s+be\s+"
            rf"(?:heeded|followed|observed|respected|honored|obeyed)",
            rf"{discourse}(?:following|heeding|observing|respecting|honoring)\s+"
            rf"{warning_reference}\s+(?:is|was|seems?|appears?)\s+"
            rf"(?:optional|unnecessary|not\s+(?:needed|required|necessary|mandatory|essential))",
            rf"{discourse}(?:there\s+(?:is|was)\s+)?no\s+"
            rf"(?:need|reason|obligation|requirement|duty)\s+to\s+"
            rf"(?:heed|follow|observe|respect|honor|obey)\s+{warning_reference}",
            rf"{discourse}(?:it|that|this)\s+(?:is|was|seems?|appears?)\s+"
            rf"(?:okay|ok|fine|acceptable|permitted)\s+to\s+"
            rf"(?:disregard|ignore|dismiss|discard|defy)\s+{warning_reference}"
            rf"(?:\s*,?\s*(?:and|then)\s+(?:proceed|continue|go\s+ahead|do\s+it))?",
        )
    ):
        return True
    if re.search(r"\b(?:i|we)\b", clause) and (
        re.search(
            rf"\b(?:no\s+(?:intent|intention|plan|effort|attempt|wish|desire)|"
            rf"(?:will|would|shall|do|does|did|can|could)\s+not|won't|wouldn't|don't|"
            rf"didn't|can't|couldn't|refuse|decline|refrain|against)\b"
            rf"[^.!?;\n]{{0,56}}\b{nonintervention}\b",
            clause,
        )
        or re.search(
            rf"\b(?:choose|chose|chosen|decide|decided|elect|elected|opt|opted)\b"
            rf"[^.!?;\n]{{0,24}}\b(?:not\s+to|against)\s+{nonintervention}\b",
            clause,
        )
    ):
        return True
    return any(
        re.fullmatch(pattern, clause)
        for pattern in (
            rf"{discourse}(?:(?:{child})\s+"
            rf"(?:(?:should|must|can|could|may|might|will|would)\s+|"
            rf"(?:need|ought)\s+to\s+))?"
            rf"(?:really\s+)?(?:still\s+)?(?:"
            rf"go\s+ahead(?:\s+and\s+(?:do|try|choose|attempt)\s+{anaphora})?|"
            rf"go\s+for\s+(?:it|that|this|the\s+same)|{action}|"
            rf"(?:proceed|continue|carry\s+on))"
            rf"(?:\s+(?:anyway|if\s+you\s+(?:really\s+)?want\s+to))?",
            rf"{discourse}(?:i|we)\s+(?:recommend|suggest|advise|urge)\s+"
            rf"(?:{anaphora}|{action}|(?:doing|trying|choosing|attempting)\s+{anaphora})",
            rf"{discourse}(?:except|unless)(?:\s+(?:when|if))?\s+"
            rf"(?:{child}\s+)?(?:{preference})(?:\s+{action})?",
            rf"{discourse}(?:(?:the\s+)?choice\s+is\s+yours|"
            rf"(?:it|that|this)\s+is\s+your\s+(?:choice|decision)|"
            rf"(?:it|that|this)(?:'s|\s+is)\s+up\s+to\s+you|"
            rf"(?:the\s+)?decision\s+is\s+yours|you\s+decide|decide\s+for\s+yourself)",
            rf"{discourse}(?:i|we)\s+(?:will\s+not|won't|would\s+not|wouldn't)\s+"
            rf"(?:(?:stand|get)\s+in\s+your\s+way|"
            rf"(?:stand|come|get)\s+between\s+you\s+and\s+(?:your|that|the)\s+(?:decision|choice)|"
            rf"(?:(?:try|attempt)\s+to\s+)?(?:stop|prevent|oppose|discourage|dissuade)\s+you|"
            rf"(?:(?:try|attempt)\s+to\s+)?talk\s+you\s+out\s+of\s+{anaphora}|"
            rf"(?:interfere\s+with|object\s+to|contest|challenge)\s+"
            rf"(?:you|your\s+(?:decision|choice))|"
            rf"object|interfere|get\s+involved|step\s+in|"
            rf"stand\s+(?:in\s+opposition|opposed))",
            rf"{discourse}(?:i|we)\s+(?:have|had)\s+no\s+(?:intent|intention|plan)\s+"
            rf"(?:of\s+(?:stopping|preventing|opposing|discouraging|dissuading)|"
            rf"to\s+(?:stop|prevent|oppose|discourage|dissuade))\s+you",
            rf"{discourse}(?:i|we)\s+(?:have|had)\s+no\s+(?:intent|intention|plan)\s+"
            rf"(?:of\s+(?:intervening|interfering|getting\s+involved|stepping\s+in)|"
            rf"to\s+(?:intervene|interfere|get\s+involved|step\s+in)){qualifier}",
            rf"{discourse}(?:i|we)\s+(?:do\s+not|don't|did\s+not|didn't)\s+"
            rf"(?:intend|plan|mean|want|aim)\s+to\s+"
            rf"(?:intervene|interfere|get\s+involved|step\s+in){qualifier}",
            rf"{discourse}(?:i|we)(?:'m|'re|\s+am|\s+are|\s+was|\s+were)\s+"
            rf"(?:choosing|electing|deciding|opting)\s+not\s+to\s+"
            rf"(?:intervene|interfere|get\s+involved|step\s+in){qualifier}",
            rf"{discourse}(?:i|we)(?:(?:\s+have|'ve|\s+had|'d)\s+)?"
            rf"(?:choose|chose|chosen|decide|decided|elect|elected|opt|opted)\s+against\s+"
            rf"(?:intervening|interfering|getting\s+involved|stepping\s+in){qualifier}",
            rf"{discourse}(?:i|we)\s+(?:(?:plan|intend|mean|aim|choose|decide)\s+to|"
            rf"(?:will|would|shall)\s+)\s*"
            rf"(?:(?:remain|stay)\s+(?:uninvolved|detached|apart)|refrain\s+from\s+"
            rf"(?:intervening|interfering|getting\s+involved|stepping\s+in)){qualifier}",
            rf"{discourse}(?:i|we)(?:(?:\s+(?:will|would|shall))|'ll)?\s+"
            rf"(?:look\s+the\s+other\s+way|turn\s+(?:a|the)\s+blind\s+eye|stand\s+aside|"
            rf"leave\s+you\s+to\s+{anaphora})",
            rf"{discourse}(?:no\s+one|nobody|nothing)\s+"
            rf"(?:(?:will|would|can|could)\s+(?:stop|prevent)\s+you|"
            rf"(?:is|was)\s+(?:stopping|preventing)\s+you|"
            rf"(?:stands?|stood)\s+in\s+your\s+way|"
            rf"(?:(?:will|would|can|could|is|was)\s+)?"
            rf"(?:interfere|object|get\s+involved))",
            rf"{discourse}(?:{child}\s+)?feel\s+free\s+to\s+(?:{action}|proceed)",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:(?:fully|strongly|really)\s+)?"
            rf"(?:(?:support|respect|honor|back|endorse|accept|sanction)\s+your\s+"
            rf"(?:decision|choice)|(?:stand|stood)\s+behind\s+your\s+"
            rf"(?:decision|choice))",
            rf"{discourse}(?:you\s+have\s+(?:my|our)\s+"
            rf"(?:blessing|approval|support|endorsement|sanction|leave)|"
            rf"(?:i|we)\s+(?:give|grant|offer)\s+you\s+"
            rf"(?:(?:my|our)\s+)?(?:blessing|approval|support|endorsement|sanction|leave))",
            rf"{discourse}(?:(?:i|we)\s+(?:give|grant|offer)\s+{child}\s+"
            rf"(?:(?:my|our)\s+)?(?:permission|leave)|{child}\s+(?:has|have)\s+"
            rf"(?:my|our)\s+(?:permission|leave))"
            rf"(?:\s+(?:to\s+(?:{action}|proceed)|anyway|regardless))?",
            rf"{discourse}(?:(?:i|we)\s+"
            rf"(?:(?:fully|strongly|really|still)\s+)?"
            rf"(?:authorize|allow|permit|sanction|endorse|condone|accept|support|back|"
            rf"approve(?:\s+of)?|bless)\s+(?:{anaphora}|{action})|"
            rf"(?:i|we)\s+(?:give|grant)\s+{anaphora}\s+(?:my|our)\s+blessing|"
            rf"(?:i|we)\s+(?:give|grant|offer|extend)\s+(?:my|our)\s+"
            rf"(?:(?:full|complete|unqualified|unreserved)\s+)?"
            rf"(?:blessing|approval|support|endorsement|sanction|permission|consent|authorization)"
            rf"\s+to\s+{anaphora}|"
            rf"(?:i|we)\s+(?:give|grant|offer|extend)\s+{anaphora}\s+(?:my|our)\s+"
            rf"(?:(?:full|complete|unqualified|unreserved)\s+)?"
            rf"(?:blessing|approval|support|endorsement|sanction|permission)|"
            rf"{anaphora}\s+(?:has|have)\s+(?:my|our)\s+"
            rf"(?:(?:full|complete|unqualified|unreserved)\s+)?"
            rf"(?:blessing|approval|support|endorsement|sanction|permission)|"
            rf"(?:i|we)\s+(?:give|grant|offer|bestow|confer)\s+(?:my|our)\s+"
            rf"(?:blessing|approval|support|endorsement|sanction|permission|consent|authorization)\s+"
            rf"(?:on|upon)\s+{anaphora}|"
            rf"(?:i|we)\s+(?:consent|agree|acquiesce)\s+to\s+{anaphora}|"
            rf"(?:i|we)(?:'m|'re|\s+am|\s+are)\s+in\s+favor\s+of\s+{anaphora})"
            rf"{qualifier}",
            rf"{discourse}(?:i|we)\s+(?:refuse|decline)\s+to\s+"
            rf"(?:oppose|prevent|stop|block|discourage|dissuade|intervene|interfere|object|"
            rf"get\s+involved|step\s+in)"
            rf"(?:\s+(?:against|in|with|to)\s+{anaphora}|\s+{anaphora}|\s+you)?{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will\s+not|won't|would\s+not|wouldn't)\s+"
            rf"(?:(?:make|put\s+in)\s+(?:any\s+)?effort\s+to|lift\s+(?:a|one)\s+finger\s+to)\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will\s+not|won't|would\s+not|wouldn't)\s+"
            rf"(?:make\s+(?:any\s+)?(?:attempt|effort|move)\s+at\s+"
            rf"(?:stopping|preventing|opposing|discouraging|dissuading)|"
            rf"exert\s+(?:any\s+)?effort\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade))\s+you{qualifier}",
            rf"{discourse}(?:i|we)\s+(?:make|put\s+in)\s+no\s+effort\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)(?:\s+(?:will|would|shall)|'ll)?\s+"
            rf"(?:make|put\s+in)\s+no\s+(?:attempt|effort|move)\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade|intervene\s+for)\s+you{qualifier}",
            rf"{discourse}(?:i|we)(?:'m|'re|\s+am|\s+are|\s+remain)\s+"
            rf"(?:unwilling|unprepared|disinclined|reluctant|not\s+prepared|not\s+willing)\s+to\s+"
            rf"(?:intervene|interfere|get\s+involved|step\s+in|stop|prevent|oppose)"
            rf"(?:\s+(?:in|against|with)\s+{anaphora}|\s+you)?{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"(?:do|try)\s+(?:anything|a\s+thing)\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"(?:act|take\s+(?:any\s+)?action|intervene|step\s+in)\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"(?:take\s+(?:any\s+)?steps?\s+to|help(?:\s+to)?)\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"prevent\s+{anaphora}\s+from\s+(?:happening|occurring|taking\s+place){qualifier}",
            rf"{discourse}(?:i|we)\s+(?:choose|chose|have\s+chosen|'ve\s+chosen|"
            rf"decide|decided|have\s+decided|'ve\s+decided)\s+not\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)(?:\s+(?:will|would)|'ll)?\s+"
            rf"(?:do\s+nothing|take\s+no\s+(?:action|steps?))\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)\s+(?:have|had)\s+no\s+"
            rf"(?:desire|wish|inclination)\s+to\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)\s+(?:cannot|can't|can\s+not|could\s+not|couldn't)\s+"
            rf"(?:stop|prevent)\s+you{qualifier}",
            rf"{discourse}(?:i|we)(?:(?:'m|'re|\s+am|\s+are)\s+not\s+going\s+to|"
            rf"\s+(?:do\s+not|don't|did\s+not|didn't)\s+(?:want|intend|plan|mean|desire)\s+to)\s+"
            rf"(?:stop|prevent|oppose|discourage|dissuade)\s+you{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will\s+not|won't|would\s+not|wouldn't|do\s+not|don't)\s+"
            rf"(?:oppose|prevent|stop|block|discourage|stand\s+against|interfere\s+with|"
            rf"object\s+to)\s+{anaphora}"
            rf"{qualifier}",
            rf"{discourse}(?:i|we)(?:'m|'re|\s+am|\s+are|\s+was|\s+were)\s+"
            rf"not\s+(?:opposed|hostile)\s+to\s+{anaphora}{qualifier}",
            rf"{discourse}(?:i|we)\s+(?:have|raise)\s+no\s+objection\s+to\s+{anaphora}"
            rf"{qualifier}",
            rf"{discourse}(?:i|we)\s+(?:have|see)\s+no\s+(?:issue|problem)\s+with\s+{anaphora}"
            rf"{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will\s+not|won't|would\s+not|wouldn't)\s+intervene"
            rf"(?:\s+(?:in|against)\s+{anaphora}|\s+to\s+(?:stop|prevent)\s+you)?{qualifier}",
            rf"{discourse}(?:i|we)(?:'m|'re|\s+am|\s+are)\s+not\s+"
            rf"(?:intervening|stepping\s+in)"
            rf"(?:\s+(?:in|against)\s+{anaphora})?{qualifier}",
            rf"{discourse}(?:i|we)\s+"
            rf"(?:will\s+not|won't|would\s+not|wouldn't)\s+get\s+involved"
            rf"(?:\s+(?:in|with)\s+{anaphora})?{qualifier}",
            rf"{discourse}(?:i|we)(?:'m|'re|\s+am|\s+are)\s+not\s+"
            rf"(?:getting|becoming)\s+involved"
            rf"(?:\s+(?:in|with)\s+{anaphora})?{qualifier}",
            rf"{discourse}(?:i|we)(?:(?:'m|'re|\s+am|\s+are)\s+staying|"
            rf"\s+(?:will\s+|would\s+)?(?:stay|keep))\s+out"
            rf"(?:\s+of\s+{anaphora})?{qualifier}",
            rf"{discourse}(?:i|we)(?:(?:'m|'re|\s+am|\s+are)\s+going\s+to|"
            rf"(?:\s+(?:will|would|shall|can|could|may|might)|'ll))?\s+"
            rf"(?:let|allow|permit)\s+(?:it|that|this)\s+(?:to\s+)?"
            rf"(?:happen|occur|take\s+place){qualifier}",
            rf"{discourse}(?:i|we)\s+(?:do\s+not|don't|did\s+not|didn't)\s+care\s+"
            rf"(?:whether|if)\s+(?:you\s+)?(?:do|try|attempt|choose)\s+{anaphora}{qualifier}",
            rf"{discourse}(?:i|we)\s+(?:do\s+not|don't|did\s+not|didn't|would\s+not|wouldn't)\s+"
            rf"mind\s+(?:whether|if)\s+(?:you\s+)?(?:do|try|attempt|choose)\s+{anaphora}{qualifier}",
            rf"{discourse}{child}\s+(?:has|have)\s+(?:my|our)\s+"
            rf"(?:consent|authorization|permission|approval|blessing|endorsement|support|sanction|leave)"
            rf"(?:\s+(?:to\s+(?:{action}|proceed)|anyway|regardless))?",
            rf"{discourse}(?:(?:i|we)(?:'m|'re|\s+am|\s+are)\s+"
            rf"(?:fine|okay|ok|comfortable)\s+with\s+{anaphora}|"
            rf"{anaphora}\s+(?:is|seems)\s+(?:fine|okay|ok)\s+with\s+(?:me|us))"
            rf"(?:\s+(?:anyway|regardless))?",
            rf"{discourse}(?:i|we)\s+(?:will\s+not|won't|would\s+not|wouldn't)\s+"
            rf"judge\s+(?:you|your\s+(?:decision|choice))",
            rf"{discourse}(?:i|we)\s+(?:accept|support|respect|honor)\s+"
            rf"whatever\s+you\s+(?:decide|choose)",
            rf"{discourse}(?:you|children|kids?)\s+have\s+"
            rf"(?:(?:the|every)\s+)?right\s+to\s+(?:decide|choose)",
            rf"{discourse}(?:follow\s+your\s+(?:heart|instincts?)|"
            rf"trust\s+your\s+(?:gut|instincts?)|do\s+what\s+feels\s+right)",
            rf"{discourse}(?:(?:(?:{child})\s+)?feel\s+free\s+to\s+|"
            rf"(?:{child})\s+(?:is|are)\s+free\s+to\s+)?"
            rf"(?:(?:disregard|ignore|dismiss|override|forget|discard|reject|defy|"
            rf"cancel|withdraw|revoke|rescind)\s+"
            rf"{warning_reference}|"
            rf"pay\s+no\s+attention\s+to\s+{warning_reference}|(?:set|put)\s+"
            rf"{warning_reference}\s+aside|"
            rf"(?:throw|brush)\s+(?:it|that|this|(?:that|the|this)\s+{warning})\s+"
            rf"(?:away|aside|off)|tune\s+"
            rf"{warning_reference}\s+out|"
            rf"(?:throw|brush)\s+(?:away|aside|off)\s+"
            rf"{warning_reference})"
            rf"\s*,?\s*(?:and|then)\s+(?:carry\s+(?:it|that|this)\s+out|"
            rf"proceed|do\s+it|go\s+ahead)",
            rf"{discourse}(?:(?:i|we)\s+|(?:you\s+(?:can|could|may|might|should)\s+))?"
            rf"(?:cancel|withdraw|revoke|rescind|annul|discard|override|dismiss|ignore|disregard|"
            rf"retract|renounce|disavow|waive|void|nullify|remove|take\s+back|took\s+back)\s+"
            rf"{warning_reference}",
            rf"{discourse}(?:(?:i|we)\s+)?(?:consider|regard|deem)\s+"
            rf"{warning_reference}\s+"
            rf"(?:cancelled|canceled|withdrawn|revoked|rescinded|retracted|waived|void|invalid|null)",
            rf"{discourse}(?:i|we)(?:(?:\s+have|'ve|\s+had|'d))?\s+"
            rf"(?:withdraw|withdrew|withdrawn|retract(?:ed)?|revoke(?:d)?|rescind(?:ed)?|"
            rf"drop(?:ped)?|abandon(?:ed)?)\s+"
            rf"(?:(?:my|our|the|this|that)\s+)?(?:objection|opposition)"
            rf"(?:\s+(?:to\s+)?{self_harm_reference})?",
            rf"{discourse}(?:i|we)\s+no\s+longer\s+"
            rf"(?:oppose\s+{self_harm_reference}|object\s+to\s+{self_harm_reference}|"
            rf"stand\s+(?:against|opposed\s+to)\s+{self_harm_reference}|object)",
            rf"{discourse}(?:i|we)(?:\s+have|'ve|\s+had)\s+"
            rf"(?:stopped|quit|ceased)\s+(?:objecting|opposing)"
            rf"(?:\s+(?:to\s+)?{self_harm_reference})?",
            rf"{discourse}(?:(?:my|our|the|this|that)\s+)?(?:objection|opposition)\s+"
            rf"(?:(?:is|was)\s+|(?:has|had)\s+been\s+)"
            rf"(?:withdrawn|retracted|revoked|rescinded|dropped|abandoned|lifted)",
            rf"{discourse}{warning_reference}\s+"
            rf"(?:(?:(?:is|was|stands?)\s+|(?:has|had)\s+been\s+)"
            rf"(?:hereby\s+)?(?:cancelled|canceled|withdrawn|revoked|rescinded|retracted|waived|"
            rf"annulled|lifted|discarded|overridden|nullified|removed|superseded|"
            rf"dismissed|obsolete|irrelevant|moot|void|invalid|null)|"
            rf"(?:has|had)\s+(?:become|grown|turned)\s+"
            rf"(?:obsolete|irrelevant|moot|void|invalid|null|ineffective|inapplicable|meaningless)|"
            rf"(?:has|had)\s+expired|"
            rf"(?:expired|lapsed)|"
            rf"(?:is|was)\s+no\s+longer\s+in\s+(?:force|effect)|"
            rf"(?:is|was)\s+no\s+longer\s+"
            rf"(?:valid|effective|binding|operative|applicable|relevant|important|necessary|"
            rf"meaningful|current|needed)|"
            rf"(?:has|had)\s+lost\s+(?:(?:all|any)\s+)?(?:its\s+)?"
            rf"(?:force|effect|relevance|meaning|importance|value)|"
            rf"no\s+longer\s+(?:applies|stands?|counts?|holds?|in\s+(?:force|effect))|"
            rf"(?:does\s+not|doesn't|did\s+not|didn't)\s+matter\s+"
            rf"(?:anymore|any\s+longer|now)|"
            rf"(?:does\s+not|doesn't|did\s+not|didn't)\s+apply\s+"
            rf"(?:anymore|any\s+longer|now))",
            rf"{discourse}(?:do\s+not|don't|never)\s+let\s+"
            rf"{warning_reference}\s+"
            rf"(?:stop|prevent|deter|discourage)\s+you(?:\s+from\s+{action})?",
            r"(?:despite|notwithstanding|in\s+spite\s+of)\s+"
            rf"(?:that|the|this)\s+{warning}\s*,?\s*"
            r"(?:carry\s+(?:it|that|this)\s+out|proceed|do\s+it|go\s+ahead)",
        )
    )


_CANDIDATE_DEATH_CHOICE = re.compile(
    r"\b(?P<verb>choose|chooses|chose|chosen|choosing|select|selects|selected|selecting|"
    r"pick|picks|picked|picking|prefer|prefers|preferred|preferring|embrace|embraces|"
    r"embraced|embracing)\s+death\b"
)


def _contains_candidate_directional_death(clause: str) -> bool:
    """Recognize structurally directed death choices without treating every mention as unsafe."""

    if re.search(
        r"\bdeath\s+(?:is|was|would\s+be|could\s+be|seems?|sounds?|feels?)\s+"
        r"(?:(?:the|a|an|your)\s+)?(?:(?:best|right|good|smart|reasonable|acceptable|safe|"
        r"recommended|only)\s+)*(?:answer|solution|way\s+out|choice|option|course|escape|relief)\b",
        clause,
    ) or re.search(
        r"\b(?:(?:the|your)\s+)?(?:(?:best|right|good|smart|reasonable|acceptable|safe|"
        r"recommended|only)\s+)+(?:answer|solution|way\s+out|choice|option|course|escape|relief)\s+"
        r"(?:is|was|would\s+be|could\s+be)\s+death\b",
        clause,
    ):
        return True
    if re.search(
        r"\b(?:i|we)\s+(?:recommend|suggest|advise|urge)\s+(?:that\s+)?"
        r"(?:(?:you|the\s+child|children|kids?)\s+(?:choose|select|pick|prefer)\s+)?death\b",
        clause,
    ) or re.search(
        r"\b(?:i|we)\s+(?:authorize|sanction|allow|permit|endorse|condone|accept|support|back|bless|"
        r"approve(?:\s+of)?)\s+death\b",
        clause,
    ):
        return True
    for choice in _CANDIDATE_DEATH_CHOICE.finditer(clause):
        if _action_is_negated(clause, choice.start()):
            continue
        base_imperative = choice.group("verb") in {"choose", "select", "pick", "prefer", "embrace"}
        if (base_imperative and _action_is_imperative(clause, choice.start())) or _child_directly_owns_action(
            clause, choice.start()
        ):
            return True
        if re.search(
            r"\b(?:i|we)\s+(?:recommend|suggest|advise|urge|encourage|invite)\s+"
            r"(?:(?:you|the\s+child|children|kids?)\s+)?(?:to\s+)?$",
            clause[: choice.start()],
        ):
            return True
    return False


def _contains_candidate_self_harm_direction(value: Any) -> bool:
    """Reject candidate-authored directions for a child to self-harm."""

    target_pattern = (
        r"(?P<nominal>suicide|self[- ]harm|(?:an?\s+)?overdose)(?=\s+[a-z]|$)|"
        rf"(?P<finite>{_CANDIDATE_SELF_HARM_FINITE_TARGET})|"
        rf"(?P<gerund>{_CANDIDATE_SELF_HARM_GERUND_TARGET})"
    )
    normalized_fragments = [_normalized_safety_text(fragment) for fragment in _text_fragments(value)]
    clauses = [clause for fragment in normalized_fragments for clause in _safety_clauses(fragment)]
    # Child-visible structured fields have no semantic ordering guarantee.
    # Establish self-harm context across the complete candidate so a protective
    # sentence cannot authenticate an endorsement hidden in another field.
    self_harm_context_active = any(
        re.search(rf"\b(?:{target_pattern})\b", clause) or re.search(r"\bdeath\b", clause) for clause in clauses
    )
    for normalized_fragment in normalized_fragments:
        if _is_clearly_fictional_self_harm_statement(normalized_fragment):
            continue
        for clause in _safety_clauses(normalized_fragment):
            for death_segment in re.split(r"[:—]", clause):
                death_segment = death_segment.strip()
                if (
                    death_segment
                    and not _is_clearly_fictional_self_harm_statement(death_segment)
                    and _contains_candidate_directional_death(death_segment)
                ):
                    return True
            clause_mentions_self_harm = bool(re.search(rf"\b(?:{target_pattern})\b", clause))
            if _is_protective_self_harm_clause(clause):
                continue
            if _is_candidate_warning_nullification(clause):
                return True
            if self_harm_context_active and _is_contextual_candidate_self_harm_endorsement(clause):
                return True
            for segment in re.split(r"[:—]", clause):
                segment = segment.strip()
                if (
                    not segment
                    or _NON_INTENT_SELF_HARM_REFERENCE.fullmatch(segment)
                    or _is_clearly_fictional_self_harm_statement(segment)
                ):
                    continue
                if _is_self_harm(segment):
                    return True
            for target in re.finditer(rf"\b(?:{target_pattern})\b", clause):
                segment_start = max(clause.rfind(delimiter, 0, target.start()) for delimiter in (":", "—")) + 1
                target_segment = clause[segment_start:].strip()
                if _NON_INTENT_SELF_HARM_REFERENCE.fullmatch(
                    target_segment
                ) or _is_clearly_fictional_self_harm_statement(target_segment):
                    continue
                if _is_protective_self_harm_clause(target_segment):
                    continue
                if _is_self_harm(target_segment):
                    return True
                prefix = clause[: target.start()]
                trailing = clause[target.end() :]
                # In a candidate answer, "why don't you" is a suggestion—not
                # protective negation—even though it contains "don't".
                if re.search(
                    r"\bwhy\s+(?:do\s+not|don't)\s+(?:you|the\s+child|children|kids?)\s+"
                    r"(?:(?:just|simply|now|then)\s+)*$",
                    prefix,
                ):
                    return True
                # These affirmative invitation frames contain surface
                # negation or omit an explicit child subject, but still direct
                # the child toward the matched self-harm target.
                if re.search(
                    r"\b(?:(?:do\s+not|don't)\s+hesitate\s+to|"
                    r"(?:do\s+not|don't|never)\s+(?:forget|fail)\s+to|"
                    r"(?:there\s+is\s+)?no\s+(?:good\s+)?reason\s+not\s+to|"
                    r"(?:nothing|no\s+one)\s+(?:is\s+)?(?:stopping|preventing)\s+"
                    r"(?:you|the\s+child)\s+from)\s+$",
                    prefix,
                ):
                    return True
                protective_suffix = bool(
                    re.fullmatch(
                        r"\s+(?:(?:is|would\s+be|can\s+be|seems?|sounds?|feels?)\s+"
                        r"(?:very\s+|extremely\s+)?(?:dangerous|unsafe|wrong|harmful|deadly|"
                        r"not\s+(?:safe|okay|ok|right)|never\s+safe|"
                        r"an?\s+(?:serious\s+)?(?:danger|risk))|"
                        r"is\s+(?:never|not)\s+(?:(?:the|an?)\s+)?"
                        r"(?:answer|solution|way\s+out|right\s+(?:choice|option)|good\s+idea)|"
                        r"(?:can|could|may|might|will|would)\s+"
                        r"(?:hurt|harm|kill|cause\s+(?:harm|death|injury)|lead\s+to\s+"
                        r"(?:harm|death|injury)))\b",
                        trailing,
                    )
                )
                protective_prefix = bool(
                    re.search(
                        r"\b(?:dangerous|unsafe|wrong|harmful|deadly|"
                        r"not\s+(?:safe|okay|ok|right)|never\s+safe|"
                        r"an?\s+(?:serious\s+)?(?:danger|risk))\s+"
                        r"(?:for\s+(?:you|the\s+child)\s+)?to\s+$",
                        prefix,
                    )
                )
                support_condition_before_target = bool(
                    re.search(
                        r"\b(?:if|when)\s+you"
                        r"(?:(?:'re|\s+are)\s+(?:thinking\s+(?:about|of)|considering|"
                        r"planning\s+to|about\s+to|ready\s+to|preparing\s+to)|"
                        r"\s+(?:feel|are\s+feeling)\s+like|"
                        r"\s+(?:might|may|could|would|will)\s+|"
                        r"\s+(?:want|plan|intend|mean|prepare)\s+to)\s*$",
                        prefix,
                    )
                )
                support_action_before_target = bool(
                    re.search(
                        r"\b(?:tell|find|call|show|talk\s+to|go\s+to|reach\s+out\s+to|"
                        r"get\s+help\s+from)\b[^.!?;\n]{0,90}\b"
                        r"trusted\s+(?:grown-up|adult|person)\b",
                        prefix,
                    )
                    or re.search(
                        r"\bask\b[^.!?;\n]{0,32}\btrusted\s+(?:grown-up|adult|person)\b"
                        r"[^.!?;\n]{0,32}\bfor\s+help\b",
                        prefix,
                    )
                )
                support_before_target = bool(
                    support_condition_before_target
                    and support_action_before_target
                    and re.search(r"\btrusted\s+(?:grown-up|adult|person)\b", prefix)
                )
                support_condition_after_target = bool(
                    re.search(
                        r"\b(?:if|when)\s+you(?:(?:'re|\s+are)\s+"
                        r"(?:thinking\s+(?:about|of)|considering|planning\s+to|"
                        r"about\s+to|ready\s+to|preparing\s+to)|"
                        r"\s+(?:feel|are\s+feeling)\s+like|"
                        r"\s+(?:might|may|could|would|will)\s+|"
                        r"\s+(?:want|plan|intend|mean|prepare)\s+to)\s*$",
                        prefix,
                    )
                )
                support_action_after_target = bool(
                    re.search(
                        r"\b(?:tell|find|call|show|talk\s+to|go\s+to|reach\s+out\s+to|"
                        r"get\s+help\s+from)\b[^.!?;\n]{0,90}\b"
                        r"trusted\s+(?:grown-up|adult|person)\b",
                        trailing,
                    )
                    or re.search(
                        r"\bask\b[^.!?;\n]{0,32}\btrusted\s+(?:grown-up|adult|person)\b"
                        r"[^.!?;\n]{0,32}\bfor\s+help\b",
                        trailing,
                    )
                )
                support_after_target = bool(
                    support_condition_after_target
                    and support_action_after_target
                    and re.search(r"\btrusted\s+(?:grown-up|adult|person)\b", trailing)
                )
                protective_hazard_condition = bool(
                    target.group("finite")
                    and re.fullmatch(
                        r"(?:(?:hurt|harm|cut|burn)\s+your(?:self|selves)|die)",
                        target.group("finite"),
                    )
                    and re.match(r"^\s+(?:if|when)\b", trailing)
                    and re.search(
                        r"\b(?:hot|heat|fire|flame|stove|oven|sharp|blade|knife|"
                        r"scissors|electric|outlet|wire|traffic|road|poison|chemical|unsafe|danger)",
                        trailing,
                    )
                )
                prevention_handoff = bool(
                    re.search(
                        r"\b(?:best|safe|right|helpful)?\s*way\s+to\s+prevent\s+$",
                        prefix,
                    )
                    and re.fullmatch(
                        r"\s+(?:is|would\s+be)\s+to\s+"
                        r"(?:tell|find|call|show|talk\s+to|go\s+to|reach\s+out\s+to|"
                        r"get\s+help\s+from|ask)\b[^.!?;\n]{0,80}\b"
                        r"trusted\s+(?:grown-up|adult|person)\b",
                        trailing,
                    )
                )
                if (
                    protective_suffix
                    or protective_prefix
                    or support_before_target
                    or support_after_target
                    or protective_hazard_condition
                    or prevention_handoff
                ):
                    continue
                if target.group("nominal"):
                    neutral_nominal_statement = bool(
                        re.fullmatch(
                            r"\s+(?:prevention\s+(?:saves?|protects?|helps?)\s+"
                            r"(?:lives?|people|children|families)|"
                            r"prevention\s+(?:is|remains?)\s+"
                            r"(?:important|helpful|effective|possible|worthwhile)|"
                            r"can\s+be\s+(?:an?\s+)?(?:warning\s+)?(?:sign|symptom)\s+that\s+"
                            r"(?:(?:someone|a\s+person|the\s+child)\s+needs?|you\s+need)\s+"
                            r"(?:immediate\s+)?(?:help|support|treatment)|"
                            r"(?:can|could|may|might)\s+(?:indicate|signal|show)\s+"
                            r"(?:distress|pain|a\s+need\s+for\s+(?:help|support|treatment))|"
                            r"(?:can|could|may|might)\s+happen\s+"
                            r"(?:by\s+accident|accidentally)|"
                            r"(?:is|was|are|means?|describes?|refers?\s+to)\s+"
                            r"(?:(?:(?:an?|the)\s+)?(?:act|term|word|topic|concept|cause|sign|"
                            r"warning|risk|danger|medical\s+emergency|"
                            r"public\s+health\s+(?:issue|problem|topic)|serious\s+(?:issue|"
                            r"problem|concern|emergency))\b|when\b|defined\s+as\b)|"
                            r"means?\s+(?:taking|using|receiving)\s+too\s+much\b|"
                            r"needs?\s+(?:immediate\s+)?(?:help|support|treatment)\b|"
                            r"(?:can|could|may|might|should|must)\s+be\s+"
                            r"(?:prevented|discussed|treated|taken\s+seriously|addressed))",
                            trailing,
                        )
                    )
                    neutral_leading_context = bool(
                        re.search(
                            r"\b(?:preventing|talking\s+about|learning\s+about|"
                            r"discussing|asking\s+about)\s+$",
                            prefix,
                        )
                        and re.fullmatch(
                            r"\s+(?:(?:saves?|protects?|helps?)\s+"
                            r"(?:lives?|people|children|families)|"
                            r"(?:can|could|may|might|does|will)\s+"
                            r"(?:help|save\s+lives?|support\s+people|reduce\s+risk))",
                            trailing,
                        )
                    )
                    if neutral_nominal_statement or neutral_leading_context:
                        continue
                    neutral_nominal_ending = bool(
                        re.search(
                            r"\b(?:(?:what|which)\s+(?:is|means)|"
                            r"preventing|(?:learn|read|talk|talking|ask|teach|information)\s+about|"
                            r"(?:definition|meaning|discussion|question|lesson|information)\s+of|"
                            r"(?:do\s+not|don't|never|avoid)|"
                            r"(?:you|children|kids?)\s+should\s+not)\s+$",
                            prefix,
                        )
                    )
                    if neutral_nominal_ending and not trailing.strip():
                        continue
                    if re.search(
                        r"\b(?:my|our|the|your)\s+"
                        r"(?:recommendation|answer|solution|choice|option|idea|advice)\s*:\s*$",
                        prefix,
                    ):
                        return True
                    reverse_endorsement = bool(
                        re.search(
                            r"\b(?:(?:the|your|a|an)\s+)?"
                            r"(?:(?:very|really|great|good|best|right|smart|reasonable|helpful|"
                            r"easy|acceptable|safe|recommended|only)\s+)*"
                            r"(?:answer|solution|way\s+out|idea|choice|option|course|escape|"
                            r"recommendation|relief)\s+"
                            r"(?:is|was|would\s+be|could\s+be|seems?|sounds?|feels?)\s+"
                            r"(?:an?\s+)?$",
                            prefix,
                        )
                    )
                    reverse_predicate = re.search(
                        r"\b(?P<subject>[a-z][^.!?;]{0,80}?)\s+"
                        r"(?:is|was|remains?|becomes?|seems?|sounds?|feels?|"
                        r"(?:can|could|may|might|will|would|should)\s+"
                        r"(?:be|remain|become|involve|include|mean))\s+"
                        r"(?:an?\s+)?$",
                        prefix,
                    )
                    neutral_reverse_subject = bool(
                        reverse_predicate
                        and re.search(
                            r"\b(?:what|who|when|where|why|how|this|that|it|act|behavior|"
                            r"topic|term|word|concept|definition|cause|risk|danger|problem|"
                            r"issue|warning|prevention|sign|symptom|medical\s+emergency|"
                            r"thing\s+to\s+(?:avoid|prevent)|behavior\s+to\s+(?:avoid|prevent))$",
                            reverse_predicate.group("subject"),
                        )
                    )
                    direct_nominal_direction = bool(
                        re.search(
                            r"\b(?:(?:you|the\s+child|children|kids?)\s+"
                            r"(?:should|must|can|could|will|may|might)\s+"
                            r"(?:(?:probably|just|simply|really|maybe|now|next|then)\s+)*|"
                            r"(?:choose|try|consider|recommend|suggest)\s+|"
                            r"(?:i|we)\s+(?:recommend|suggest|advise|urge|authorize|sanction|"
                            r"allow|permit|endorse|condone|accept|support|back|approve(?:\s+of)?)\s+)$",
                            prefix,
                        )
                    )
                    if (
                        reverse_endorsement
                        or (reverse_predicate and not neutral_reverse_subject)
                        or direct_nominal_direction
                    ):
                        return True
                    if re.match(
                        r"^\s+(?:(?:is|would\s+be|seems?)\s+"
                        r"(?:(?:an?|the|your)\s+)?(?:(?:(?:very|really|great|good|best|right|"
                        r"smart|reasonable|helpful|easy|acceptable|safe|recommended)\s+)?"
                        r"(?:answer|solution|way\s+out|idea|choice|option)|"
                        r"good|better(?:\s+for\s+you)?|best|okay|ok|acceptable|safe|recommended)|"
                        r"(?:can|could|may|might|will|would)\s+"
                        r"(?:(?:be|seem)\s+(?:(?:an?|the|your)\s+)?"
                        r"(?:answer|solution|way\s+out|idea|choice|option|good|better|best|safe)|"
                        r"make\b[^.!?;]{0,48}\b(?:better|easier|good|okay|right)|"
                        r"solve|fix|end|stop|erase|remove|help\s+with)\b)",
                        trailing,
                    ) or re.search(r"\b(?:choose|try|consider|recommend)\s+$", prefix):
                        return True
                    # A candidate-authored affirmative predicate about a
                    # self-harm noun fails closed unless it matched one of the
                    # explicit protective or neutral forms above. This covers
                    # novel praise/benefit wording without chasing adjectives.
                    if (
                        re.match(
                            r"^\s+(?:is|was|are|would|seems?|sounds?|feels?|looks?|becomes?|"
                            r"can|could|may|might|will|should|works?|helps?|rocks?)\b",
                            trailing,
                        )
                        or trailing.strip()
                    ):
                        return True
                    if prefix.strip():
                        return True
                    # An isolated candidate-authored self-harm noun is not a
                    # complete educational or protective statement. Fail
                    # closed instead of treating the missing predicate as
                    # neutral (for example, "Here is the answer. Suicide.").
                    return True
                if _action_is_negated(clause, target.start()):
                    continue
                # A candidate has no legitimate reason to introduce an
                # affirmative second-person self-harm gerund outside an
                # explicit warning or immediate support handoff. Failing
                # closed here covers novel endorsements and claimed benefits
                # without enumerating every adjective or outcome verb.
                if target.group("gerund"):
                    return True
                child_subject = r"(?:you|the\s+child|a\s+child|children|kids?)"
                suggestion_adverb = r"(?:(?:probably|just|simply|really|maybe|now|next|then)\s+)*"
                if (
                    re.search(
                        rf"\b{child_subject}\s+(?:(?:should|must|can|could|will|may|might)\s+"
                        rf"{suggestion_adverb}|(?:need|needs|have|has)\s+to\s+|"
                        rf"(?:are|is)\s+going\s+to\s+|(?:ought\s+to|had\s+better|may\s+as\s+well)\s+)$",
                        prefix,
                    )
                    or re.search(
                        r"\b(?:try|consider|practice|let's|why\s+not|go(?:\s+ahead)?\s+and|"
                        r"feel\s+free\s+to)\s+(?:to\s+)?$",
                        prefix,
                    )
                    or re.search(
                        rf"\b(?:recommend|suggest|advise|urge)\w*\s+(?:that\s+)?"
                        rf"(?:{child_subject}\s+)?(?:to\s+)?{suggestion_adverb}$",
                        prefix,
                    )
                    or re.search(
                        rf"\b(?:i|we)\s+(?:want|need|expect|ask|encourage|invite)\s+"
                        rf"{child_subject}\s+to\s+{suggestion_adverb}$",
                        prefix,
                    )
                    or re.search(
                        r"\byour\s+(?:next|recommended)\s+(?:step|move|action)\s+"
                        r"(?:is|would\s+be)\s+to\s+$",
                        prefix,
                    )
                    or re.search(
                        rf"\b{child_subject}\s+(?:are|is)\s+"
                        rf"(?:allowed|permitted|welcome|free|encouraged)\s+to\s+$",
                        prefix,
                    )
                    or re.search(
                        rf"\b{child_subject}\s+(?:have|has)\s+"
                        rf"(?:(?:my|our|their|the)\s+)?permission\s+to\s+$",
                        prefix,
                    )
                    or re.search(
                        rf"\b{child_subject}\s+(?:deserve|deserves)\s+to\s+$",
                        prefix,
                    )
                    or re.search(
                        rf"\b{child_subject}\s+(?:(?:would|'d)\s+be|are|is)\s+better\s+off\s+$",
                        prefix,
                    )
                    or re.search(
                        r"\b(?:it\s+is|it's|now\s+is)\s+(?:finally\s+)?time\s+"
                        r"(?:for\s+(?:you|the\s+child)\s+)?to\s+$",
                        prefix,
                    )
                    or re.search(
                        r"\b(?:it|this|that)\s+(?:is|would\s+be|seems?)\s+"
                        r"(?:okay|ok|fine|acceptable|permitted|safe|best|better|right|good|reasonable)\s+"
                        r"(?:for\s+(?:you|the\s+child)\s+)?to\s+$",
                        prefix,
                    )
                    or re.search(
                        r"\b(?:the|your)\s+(?:best|next|right|recommended|only|smartest|easiest)\s+"
                        r"(?:thing|choice|option|step|move|action|course)\s+"
                        r"(?:is|would\s+be)\s+(?:for\s+(?:you|the\s+child)\s+)?to\s+$",
                        prefix,
                    )
                    or re.search(
                        rf"\bhow\s+about\s+(?:{child_subject}\s+)?$",
                        prefix,
                    )
                    or re.fullmatch(
                        r"\s*(?:(?:if\s+you\s+want\s*,?|go|just|simply|now|next|then|please)\s+|"
                        r"go\s+ahead(?:\s+and|[\s,;:-]+))+",
                        prefix,
                    )
                ):
                    return True
                if target.group("finite"):
                    finite_text = target.group("finite")
                    ambiguous_nominal = bool(re.fullmatch(r"(?:overdose|self[- ]harm|die)", finite_text))
                    if not ambiguous_nominal:
                        return True
                    if (
                        ambiguous_nominal
                        and not prefix.strip()
                        and re.fullmatch(
                            r"\s*(?:(?:now|next|then|please)\b[\s,]*)*",
                            trailing,
                        )
                    ):
                        return True
            self_harm_context_active = self_harm_context_active or clause_mentions_self_harm
    return False


# Accessibility needs an executable modality, not an ambiguous learning verb.
# For example, "explain in writing" and "show in a written sentence" do not
# provide a spoken or gestural alternative merely because they contain explain/show.
_EXPLICIT_RESPONSE_MODE_PATTERNS: dict[str, str] = {
    "spoken": r"\b(?:say|tell|speak)\w*\b|\bexplain\w*\b.{0,18}\b(?:aloud|orally)\b",
    "drawn": r"\b(?:draw|sketch|make\s+a\s+picture)\w*\b",
    "gesture_or_manipulative": (
        r"\b(?:point|act|gesture|reach)\w*\b|"
        r"\b(?:move|arrange|place|put)\w*\b.{0,22}\b(?:objects?|blocks?|cards?|pieces?|counters?|items?)\b|"
        r"\bshow\w*\b.{0,24}\b(?:with|using)\b.{0,16}\b(?:objects?|blocks?|cards?|gestures?)\b"
    ),
    "written": r"\b(?:write|record|label)\w*\b",
    "symbolic": r"\b(?:symbol|number|equation)\w*\b",
    "selection": r"\b(?:choose|circle|match|select)\w*\b",
    "demonstration": r"\b(?:make|build|test|try|act\s+out|do\s+the)\w*\b",
}


def _text_fragments(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [part for item in value.values() for part in _text_fragments(item)]
    if isinstance(value, list):
        return [part for item in value for part in _text_fragments(item)]
    return []


def _assigns_hazardous_handling_to_child(
    plan: ApprovedPedagogyPlan,
    value: Any,
    *,
    require_child_direction: bool = False,
) -> bool:
    """Reject hazardous child action, including unresolved cross-step pronouns."""

    active_hazards = set(plan.safety.hazards) | _anaphoric_hazards_in_value(value)
    prior_instruction_text = ""
    for fragment in _instruction_clauses(value):
        item_text = fragment.casefold()
        explicit_hazards = _hazards_in_value(item_text)
        for hazard in plan.safety.hazards:
            hazard_target, hazard_action = _HAZARD_ACTION_LANGUAGE[hazard]
            if hazard not in explicit_hazards and hazard not in active_hazards:
                continue
            for action in re.finditer(rf"\b{hazard_action}\b", item_text):
                distributive_antecedent = _nearest_distributive_antecedent_is_hazard(
                    f"{prior_instruction_text} {item_text[: action.start()]}",
                    target_pattern=hazard_target,
                )
                if not _hazard_action_is_bound(
                    item_text,
                    target_pattern=hazard_target,
                    action=action,
                    target_is_active=hazard in active_hazards,
                    carry_action_pattern=_HAZARD_CARRY_ACTION_LANGUAGE[hazard],
                    distributive_antecedent_is_hazard=distributive_antecedent,
                    antecedent_context=f"{prior_instruction_text} {item_text[: action.start()]}",
                ):
                    continue
                if _action_is_negated(item_text, action.start()):
                    continue
                owner = _action_owner(item_text, action.start(), action.end())
                if owner == "child":
                    return True
                if owner == "adult":
                    continue
                if not require_child_direction or _action_is_imperative(item_text, action.start()):
                    return True
        prior_instruction_text = f"{prior_instruction_text} {item_text}".strip()
    return False


def _instruction_clauses(value: Any) -> list[str]:
    """Return child-facing instruction units without flattening unrelated mechanics."""

    clauses: list[str] = []
    for fragment in _text_fragments(value):
        if re.fullmatch(r"[a-z0-9]+(?:[_-][a-z0-9]+)+", fragment.casefold()):
            # Contract IDs and enum values are not child-facing instructions.
            continue
        clauses.extend(part.strip() for part in re.split(r"(?:[.!?;]+|\n+)", fragment) if part.strip())
    return clauses


def _contains_self_confirming_claim(text: str) -> bool:
    """Detect completion/reward proxies in either grammatical direction."""

    outcome = (
        r"(?:master\w*|independen\w*|understand\w*|learned|proficien\w*|"
        r"know(?:s|ing)?|can\s+do|gets?\s+it)"
    )
    completion = r"(?:finish\w*|complet\w*|doing|score\w*|page\s+is\s+done|worksheet\s+is\s+done)"
    patterns = (
        rf"\b{completion}\b.{{0,70}}\b(?:prove|show|mean|demonstrate|confirm|establish|count\s+as|"
        rf"indicate|signal|suggest|is\s+(?:evidence|proof|a\s+sign)\s+of)\w*\b.{{0,55}}\b{outcome}\b",
        rf"\b{outcome}\b.{{0,70}}\b(?:is\s+)?(?:demonstrated|confirmed|established|proved|shown)\b"
        rf".{{0,35}}\b(?:when|by|after|once)\b.{{0,45}}\b{completion}\b",
        rf"\b{completion}\b.{{0,55}}\b(?:earn|win|get|receive|unlock)\w*\b.{{0,30}}"
        rf"\b(?:a\s+)?{outcome}\b.{{0,15}}\b(?:badge|point|level|star|award)\w*\b",
        rf"\b(?:the\s+)?(?:child|learner|student|you)\b.{{0,35}}\b(?:has\s+)?{outcome}\b"
        rf".{{0,35}}\b(?:if|when|once|after)\b.{{0,45}}\b{completion}\b",
        rf"\bprov(?:e|es|ing)\b.{{0,40}}\b{outcome}\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            claim = match.group(0)
            if not re.search(r"\b(?:not|never|cannot|can't|doesn't|does\s+not|isn't|is\s+not)\b", claim):
                return True
    return False


def _contains_unqualified_outcome_assertion(text: str) -> bool:
    """Reject unsupported child-state praise wherever it appears on an artifact."""

    patterns = (
        r"\b(?:you|the\s+child|child|learner|student)\b.{0,18}"
        r"\b(?:mastered|has\s+mastered|learned|has\s+learned|is\s+independent|is\s+proficient|"
        r"(?:fully\s+)?understand(?:s)?(?:\s+(?:it|this|the\s+idea))?|"
        r"know(?:s)?\s+(?:it|this)|can\s+do\s+(?:it|this)|gets?\s+it|"
        r"(?:have|has|'ve|'s)\s+got\s+(?:it|this)|figured\s+(?:it|this)\s+out|"
        r"(?:are|is)\s+(?:now\s+)?an?\s+expert)\b",
        r"\b(?:mastery|independence|proficiency)\b.{0,20}\b(?:achieved|earned|confirmed|proved)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            clause_start = max(text.rfind(token, 0, match.start()) for token in (".", "!", "?", ";")) + 1
            prefix = text[clause_start : match.start()]
            negated = re.search(
                r"\b(?:no|not|never|cannot|can't|doesn't|does\s+not|isn't|is\s+not)\b"
                r"(?:(?![.!?;]).){0,45}$",
                prefix,
            )
            if not negated and not re.search(rf"\b{_ACTION_NEGATION}\b", match.group(0)):
                return True
    return False


def _semantic_evidence_clauses(value: Any) -> list[str]:
    """Keep child actions while removing labels/descriptions of teaching words."""

    word_task = re.compile(
        r"\b(?:read|copy|circle|underline|find|point\s+to|say|predict|test|notice|observe|explain)\b"
        r".{0,42}\b(?:printed\s+)?(?:word|words|label|labels|heading|headings|term|terms|"
        r"title|caption|text|sentence|instructions?)\b"
    )
    descriptive_list = re.compile(
        r"\b(?:words?|labels?|headings?|terms?|titles?|captions?|texts?|sentences?|instructions?)\b"
        r".{0,100}\b(?:say|read|print|display|list|show|appear|include|contain)\w*\b|"
        r"\b(?:page|sheet|worksheet|box|titles?|captions?|texts?|sentences?|instructions?)\b"
        r".{0,60}\b(?:say|read|include|contain|display|list|print|show)\w*\b"
        r".{0,80}\b(?:predict|prediction|test|notice|observation|explain|reason)\w*\b"
    )
    return [
        clause.casefold()
        for clause in _instruction_clauses(value)
        if not word_task.search(clause.casefold()) and not descriptive_list.search(clause.casefold())
    ]


_WRITING_DEMAND = re.compile(
    # Objects and linguistic units are not response actions: pointing to a
    # pictured pencil or saying a sentence must remain usable without writing.
    r"\b(?:write|writing|written|record|copy|fill\s+in|handwrite|"
    r"response\s+lines?|answer\s+lines?|on\s+(?:the\s+)?lines?|using\s+words)\b"
)


def _requires_writing(clause: str) -> bool:
    lowered = clause.casefold()
    if not _WRITING_DEMAND.search(lowered):
        return False
    return not bool(
        re.search(
            r"\b(?:no|without|not\s+needed|instead\s+of|doesn't\s+require|does\s+not\s+require|"
            r"doesn't\s+want\s+to|does\s+not\s+want\s+to)\b"
            r".{0,18}\b(?:write|writing|written|record|copy|pencil|pen|lines?)\b",
            lowered,
        )
    )


def _enacted_response_modes(value: Any, allowed_modes: list[str]) -> set[str]:
    """Resolve response modes per instruction, excluding writing-qualified verbs."""

    enacted: set[str] = set()
    for clause in _instruction_clauses(value):
        lowered = clause.casefold()
        for mode in allowed_modes:
            if mode == "written":
                if re.search(_EXPLICIT_RESPONSE_MODE_PATTERNS[mode], lowered):
                    enacted.add(mode)
                continue
            pattern = _EXPLICIT_RESPONSE_MODE_PATTERNS[mode]
            if not re.search(pattern, lowered):
                continue
            if not _requires_writing(lowered):
                enacted.add(mode)
                continue
            alternatives = [part for part in re.split(r"\b(?:or|instead)\b", lowered)[1:] if part.strip()]
            if any(re.search(pattern, alternative) for alternative in alternatives):
                enacted.add(mode)
    return enacted


def _contains_child_activity_directive(value: Any) -> bool:
    """Recognize an activity hidden inside an answer-only prose field."""

    format_terms = r"(?:activit(?:y|ies)|games?|experiments?|tests?|worksheets?|projects?|challenges?)"
    imperative = (
        r"(?:do|try|put|place|take|use|make|draw|write|record|fill|build|cut|pour|mix|move|find|"
        r"collect|sort|match|test|play(?!ful)|run|start)\w*"
    )
    for fragment in _text_fragments(value):
        lowered = fragment.casefold()
        # Mask genuinely negated imperative verbs before checking whether an
        # answer-only response hides an activity. In particular, the auxiliary
        # "do" in "Do not kill yourself" is a protective warning, not an
        # invitation to perform an activity.
        for action in reversed(list(re.finditer(rf"\b{imperative}\b", lowered))):
            if (
                re.match(r"^do\w*$", action.group()) and re.match(r"\s+not\b", lowered[action.end() :])
            ) or _action_is_negated(lowered, action.start()):
                lowered = f"{lowered[: action.start()]}reviewed-safe-action{lowered[action.end() :]}"
        if re.search(rf"\b(?:now\s+)?(?:do|try|play|run|start)\b.{{0,30}}\b{format_terms}\b", lowered):
            return True
        if re.search(
            rf"\b(?:you|we|the\s+child|your\s+family)\s+"
            rf"(?:can|could|may|might|should|must|will|would|shall|need\s+to|have\s+to|ought\s+to)\s+"
            rf"(?:(?:also|always)\s+)?{imperative}\b",
            lowered,
        ):
            return True
        if re.search(
            rf"\b(?:can|could|may|might|should|must|will|would|shall)\s+"
            rf"(?:you|we|the\s+child|your\s+family)\s+(?:please\s+)?(?:also\s+)?{imperative}\b",
            lowered,
        ):
            return True
        if re.search(rf"\b(?:let(?:'|’)s|how\s+about\s+we)\s+(?:also\s+)?{imperative}\b", lowered):
            return True
        if re.search(
            rf"\b(?:one|another|a)\s+(?:(?:fun|good|easy|quick|simple|small)\s+)?"
            rf"(?:thing|way|idea|{format_terms})\s+(?:to\s+do\s+is|is|would\s+be)\s+(?:to\s+)?{imperative}\b",
            lowered,
        ):
            return True
        if re.search(
            rf"\b(?:an?|the)\s+(?:(?:fun|easy|quick|simple|small)\s+)?{format_terms}\s+"
            rf"(?:is|would\s+be)\s+to\s+{imperative}\b",
            lowered,
        ):
            return True
        if re.search(
            rf"\bit\s+(?:can|could|may|might|should|would|will)\s+be\s+"
            rf"(?:fun|interesting|helpful|easy|useful)\s+to\s+{imperative}\b",
            lowered,
        ):
            return True
        if re.search(rf"\bconsider\s+(?:also\s+)?(?:trying\s+to\s+)?{imperative}\b", lowered):
            return True
        if re.search(rf"\bwhy\s+not\s+{imperative}\b", lowered):
            return True
        if re.search(rf"\b(?:how|what)\s+about\s+(?:(?:you|we)\s+)?{imperative}\b", lowered):
            return True
        if re.search(rf"\bfor\s+fun\s*,?\s*{imperative}\b", lowered):
            return True
        if re.search(
            rf"\b(?:you|we|the\s+child|your\s+family)\s+(?:can|could|may|might)\s+"
            rf"(?:want|like)\s+to\s+{imperative}\b",
            lowered,
        ):
            return True
        if re.search(
            rf"\b(?:you|we|the\s+child|your\s+family)\s+(?:can|could|may|might|should|would)\s+"
            rf"(?:have\s+fun|enjoy)\s+(?:by\s+)?{imperative}\b",
            lowered,
        ):
            return True
        if re.search(rf"\bwhat\s+if\s+(?:you|we|the\s+child|your\s+family)\s+{imperative}\b", lowered):
            return True
        if re.search(
            rf"\b(?:if\s+you\s+(?:want|would\s+like)|if\s+you'd\s+like|as\s+an\s+option)\s*,?\s*"
            rf"{imperative}\b",
            lowered,
        ):
            return True
        if re.search(
            rf"\b(?:wouldn't|would\s+not)\s+it\s+be\s+(?:fun|interesting|helpful)\s+to\s+{imperative}\b",
            lowered,
        ):
            return True
        if re.search(rf"\b(?:imagine|suppose)\s+(?:(?:you|we)\s+)?{imperative}\b", lowered):
            return True
        if re.search(
            rf"\bhere(?:'|’)s\s+(?:one\s+)?(?:thing|idea|option)\s+(?:to\s+try|you\s+can\s+try)\s*[:,]?\s*"
            rf"{imperative}\b",
            lowered,
        ):
            return True
        if re.search(
            rf"\b(?:feel\s+free\s+to|you\s+(?:are|'re|’re)\s+welcome\s+to)\s+{imperative}\b",
            lowered,
        ):
            return True
        if re.search(
            rf"\b(?:to\s+(?:see|explore|find\s+out|look\s+closer)(?:\s+(?:it|what\s+happens))?|"
            rf"if\s+(?:you\s+are|you're|you’re)\s+curious)\s*,?\s*{imperative}\b",
            lowered,
        ):
            return True
        if re.search(
            rf"\b(?:maybe|perhaps)\s+(?:(?:you|we|the\s+child|your\s+family)\s+)?{imperative}\b",
            lowered,
        ):
            return True
        for clause in re.split(r"(?:^|[.!?;:]\s*)", lowered):
            clause = clause.strip()
            if not clause or re.match(r"^put\s+simply\b", clause):
                continue
            if re.match(
                rf"^(?:(?:now|next|then|first|please)\s+|go\s+ahead\s+and\s+)*{imperative}\b",
                clause,
            ):
                return True
    return False


_TRUSTED_GROWN_UP_SIGNAL = re.compile(
    r"\b(?:trusted\s+(?:grown-up|adult|person)|(?:grown-up|adult|person)\s+(?:you|they)\s+trust|"
    r"parents?|caregivers?|teachers?|school\s+counselors?)\b"
)
_IMMEDIATE_SUPPORT_SIGNAL = re.compile(r"\b(?:right\s+now|now|immediately|straight\s+away|as\s+soon\s+as\s+possible)\b")
_EXPLICIT_PROHIBITION_SIGNAL = re.compile(
    r"\b(?:never|do\s+not|don't|must\s+not|should\s+not|cannot|can't|unsafe|dangerous|"
    r"not\s+safe|isn't\s+safe)\b"
)
_PHYSICAL_DANGER_SIGNAL = re.compile(
    r"\b(?:electric(?:al|ity)?|shock|electrocut\w*|burns?|fire|poison\w*|harmful\s+fumes?|"
    r"cuts?|punctures?|bleed\w*|injur\w*|danger\w*)\b"
)
_PHYSICAL_PROHIBITION_TARGET_SIGNAL = re.compile(
    rf"\b(?:{_CONDUCTIVE_OBJECT}|{_POWERED_APPLIANCE}|matches?|lighters?|candles?|flames?|fire|"
    r"bleach|cleaners?|chemicals?|detergent|razors?|blades?|box\s+cutters?|saws?|chainsaws?|"
    r"chain\s+saws?|axes?|nail\s+guns?|power(?:ed)?\s+drills?)\b"
)
_SUPPORT_DIRECTION_SIGNAL = re.compile(
    r"\b(?:tell|ask|find|get|go\s+to|call|show|let|stay\s+with|talk\s+to|reach\s+out\s+to|"
    r"needs?\s+to\s+know|can\s+help|help\s+keep|keep\s+you\s+safe)\b"
)
_NEGATED_SUPPORT_SIGNAL = re.compile(
    r"\b(?:do\s+not|don't|never|should\s+not|shouldn't)\b.{0,50}\b(?:tell|ask|find|get|call|show|"
    r"talk\s+to|reach\s+out)|\b(?:not|never)\s+(?:needed|required)|"
    r"\b(?:does\s+not|doesn't|do\s+not|don't)\s+need\s+to\s+(?:know|help)\b"
)
_UNSAFE_REASSURANCE_SIGNAL = re.compile(
    r"\b(?:safe|harmless|no\s+(?:electric(?:al)?\s+)?danger|not\s+dangerous|isn't\s+dangerous|"
    r"cannot\s+(?:shock|burn|hurt|poison|cut|injure)|can't\s+(?:shock|burn|hurt|poison|cut|injure)|"
    r"never\s+(?:shock|burn|hurt|poison|cut|injure)|"
    r"do\s+not\s+worry|don't\s+worry|never\s+worry)\b"
)


def _safety_clauses(text: str) -> list[str]:
    return [clause.strip() for clause in re.split(r"[.!?;\n]+", _normalized_safety_text(text)) if clause.strip()]


def _has_immediate_trusted_grown_up_guidance(text: str) -> bool:
    for clause in _safety_clauses(text):
        if not (_TRUSTED_GROWN_UP_SIGNAL.search(clause) and _IMMEDIATE_SUPPORT_SIGNAL.search(clause)):
            continue
        if re.search(r"\b(?:secret|hide\s+this|keep\s+this\s+from)\b", clause):
            continue
        if _NEGATED_SUPPORT_SIGNAL.search(clause):
            continue
        if _SUPPORT_DIRECTION_SIGNAL.search(clause):
            return True
    return False


def _has_affirmative_hazard_prohibition(text: str) -> bool:
    for clause in _safety_clauses(text):
        if _UNSAFE_REASSURANCE_SIGNAL.search(clause):
            continue
        if not _EXPLICIT_PROHIBITION_SIGNAL.search(clause):
            continue
        if _PHYSICAL_PROHIBITION_TARGET_SIGNAL.search(clause):
            return True
    return False


def _has_affirmative_physical_danger(text: str) -> bool:
    for clause in _safety_clauses(text):
        if not _PHYSICAL_DANGER_SIGNAL.search(clause):
            continue
        if _UNSAFE_REASSURANCE_SIGNAL.search(clause):
            continue
        if re.search(
            r"\b(?:no|not|never|cannot|can't|won't|will\s+not)\b.{0,20}"
            r"\b(?:shock|electrocut\w*|burn|fire|poison\w*|fumes?|cuts?|punctures?|"
            r"injur\w*|danger\w*)\b",
            clause,
        ):
            continue
        return True
    return False


def _has_affirmative_grown_up_handoff(text: str) -> bool:
    for clause in _safety_clauses(text):
        if not _TRUSTED_GROWN_UP_SIGNAL.search(clause):
            continue
        if _NEGATED_SUPPORT_SIGNAL.search(clause):
            continue
        if re.search(
            r"\b(?:tell|ask|find|get|call|move\s+away|stay\s+away|unplug|handle|help|keep\s+"
            r"(?:you|everyone|the\s+child)\s+away)\b",
            clause,
        ):
            return True
    return False


def _has_explicit_hazard_prohibition(text: str) -> bool:
    return bool(
        _has_affirmative_hazard_prohibition(text)
        and _has_affirmative_physical_danger(text)
        and _has_affirmative_grown_up_handoff(text)
    )


def _has_explicit_appliance_water_prohibition(text: str) -> bool:
    relation = re.compile(
        rf"\b(?:{_POWERED_APPLIANCE}|powered\s+appliances?)\b[^.!?;\n]{{0,80}}"
        r"\b(?:wet|water|baths?|bathtubs?|showers?|sinks?|tubs?|pools?|toilets?)\b|"
        rf"\b(?:wet|water)\b[^.!?;\n]{{0,80}}"
        rf"\b(?:{_POWERED_APPLIANCE}|powered\s+appliances?)\b"
    )
    return bool(
        any(_EXPLICIT_PROHIBITION_SIGNAL.search(clause) and relation.search(clause) for clause in _safety_clauses(text))
        and _has_affirmative_physical_danger(text)
        and _has_affirmative_grown_up_handoff(text)
    )


def validate_generated_against_plan(
    plan: ApprovedPedagogyPlan,
    output: dict[str, Any],
    *,
    workflow: str,
    event_text: str | None = None,
) -> list[str]:
    """Return deterministic safety, accessibility, and teaching-plan findings.

    The reasoning layer assigns severity. Concrete safety, accessibility, and
    evidence violations block delivery; ordinary differences from the suggested
    teaching sequence are quality advisories for a trusted parent.
    """

    # Runtime audit traces are private diagnostics, not material a child reads.
    # Exclude them from every semantic, safety, and reading-load decision here.
    output = {key: value for key, value in output.items() if not key.startswith("_")}
    errors: list[str] = []
    answer = str(output.get("answer") or output.get("parent_response") or output.get("hook") or "").strip()
    if not answer:
        errors.append("approved plan requires a visible answer first")
    full_text = jdump(output).casefold()
    if re.search(r"\b(?:streak|keep going to unlock|don't miss out|come back now|cliffhanger)\b", full_text):
        errors.append("response violates healthy closure")
    extension = output.get("physical_extension")
    if plan.question_intent in {"direct_answer", "sensitive_support"}:
        if extension:
            errors.append("direct or sensitive plan forbids an unsolicited activity")
        if output.get("actions"):
            errors.append("direct or sensitive plan forbids artifact actions")
        if output.get("ask") or output.get("question_back"):
            errors.append("direct or sensitive plan forbids a compulsory question-back")
    if plan.question_intent == "sensitive_support" and re.search(
        r"\b(?:mystery|detective game|mission|win|points|challenge)\b", full_text
    ):
        errors.append("sensitive support may not be trivialized with playful mechanics")
    extension_text = jdump(extension).casefold() if extension else ""
    child_facing_response = {
        key: output.get(key)
        for key in ("answer", "parent_response", "notice", "question_back", "hook", "show", "ask", "nugget")
        if output.get(key)
    }
    if output.get("visual"):
        child_facing_response["visual"] = output["visual"]
    child_facing_text = " ".join(_text_fragments(child_facing_response))
    safety_requirements = (
        safety_response_requirements_for_event(plan, event_text)
        if event_text is not None
        else set(plan.safety.response_requirements)
    )
    if "immediate_trusted_grown_up" in safety_requirements and not _has_immediate_trusted_grown_up_guidance(
        child_facing_text
    ):
        errors.append("self-harm support lacks immediate trusted-grown-up guidance")
    physical_response_requirements = safety_requirements & {
        "explicit_hazard_prohibition",
        "explicit_appliance_water_prohibition",
        "explicit_ignition_prohibition",
        "explicit_chemical_prohibition",
        "explicit_sharp_tool_prohibition",
    }
    if physical_response_requirements and not _has_explicit_hazard_prohibition(child_facing_text):
        errors.append("physical-hazard answer lacks an explicit prohibition, danger, or trusted-grown-up handoff")
    if "explicit_appliance_water_prohibition" in safety_requirements and not _has_explicit_appliance_water_prohibition(
        child_facing_text
    ):
        errors.append("appliance-water answer does not explicitly prohibit a wet or water-related appliance action")
    if (
        plan.question_intent in {"direct_answer", "sensitive_support"}
        and not safety_requirements
        and _contains_child_activity_directive(child_facing_response)
    ):
        errors.append("direct or sensitive plan forbids an activity hidden in prose")
    extension_materials = extension.get("materials") if isinstance(extension, dict) else []
    introduced_hazards = _candidate_hazards(
        materials=extension_materials,
        instructions={"response": child_facing_response, "extension": extension},
    ) - set(plan.safety.hazards)
    if _contains_candidate_self_harm_direction({"response": child_facing_response, "extension": extension}):
        errors.append("generated response directs the child toward self-harm")
    if introduced_hazards:
        errors.append(
            "generated response introduces hazards absent from the approved plan: "
            + ", ".join(sorted(introduced_hazards))
        )
    hazard_check_plan = plan
    if introduced_hazards:
        hazard_check_plan = plan.model_copy(
            update={
                "safety": plan.safety.model_copy(
                    update={
                        "risk_level": "elevated",
                        "hazards": sorted(set(plan.safety.hazards) | introduced_hazards),
                    }
                )
            }
        )
    candidate_visible_text = " ".join(_text_fragments({"response": child_facing_response, "extension": extension}))
    if (
        _has_powered_appliance_water_action(candidate_visible_text)
        and "explicit_appliance_water_prohibition" not in plan.safety.response_requirements
    ):
        errors.append("generated response introduces an appliance-water action absent from the approved safety plan")
    if plan.question_intent == "guided_inquiry":
        if not extension:
            errors.append("guided inquiry requires one observable child action")
        for label, pattern in (
            ("prediction", r"\bpredict\w*\b"),
            ("test", r"\b(?:test|try)\w*\b"),
            ("observation", r"\b(?:observe|notice|compare)\w*\b"),
            ("explanation", r"\b(?:explain|support|because|result)\w*\b"),
        ):
            if not re.search(pattern, extension_text):
                errors.append(f"guided inquiry is missing its {label} move")
    elif plan.question_intent == "compare_connect":
        if not re.search(r"\b(?:compare|same|different|sort|group)\w*\b", full_text):
            errors.append("compare/connect plan lacks an explicit comparison mechanic")
        if not re.search(r"\b(?:explain|why|because|belongs)\w*\b", full_text):
            errors.append("compare/connect plan lacks a reason-giving move")
    elif plan.question_intent == "practice_with_scaffolds":
        for label, pattern in (
            ("worked example", r"\b(?:worked example|watch one|for example)\b"),
            ("supported try", r"\b(?:together|with help|grown-up)\b"),
            ("fade", r"\b(?:less help|on your own|independent)\b"),
            ("warm stop", r"\b(?:stop warmly|stop for now|child is done|another time)\b"),
        ):
            if not re.search(pattern, full_text):
                errors.append(f"practice/fade plan is missing its {label}")
    elif plan.question_intent == "dialogic_reading":
        if not re.search(r"\b(?:read|listen)\w*\b", full_text):
            errors.append("dialogic reading lacks shared reading or listening")
        if not re.search(r"\b(?:point|say|draw|act)\w*\b", full_text):
            errors.append("dialogic reading lacks an accessible child response")
        if not str(output.get("ask") or output.get("question_back") or "").strip():
            errors.append("dialogic reading lacks an open child-facing question")
    elif plan.question_intent == "teach_back":
        if not re.search(r"\b(?:teach|explain|show|tell)\w*\b", full_text):
            errors.append("teach-back plan lacks a child explanation move")
        if not re.search(r"\b(?:say|draw|act|show|speak)\w*\b", full_text):
            errors.append("teach-back plan lacks equivalent response modes")
    elif plan.question_intent == "retrieval_revisit":
        if not re.search(r"\b(?:remember|recall|before looking|without looking)\w*\b", full_text):
            errors.append("retrieval/revisit plan lacks unaided recall")
        if not re.search(r"\b(?:apply|new example|different example|use that)\w*\b", full_text):
            errors.append("retrieval/revisit plan lacks nearby application")
    elif plan.question_intent == "make_design_iterate":
        if not extension:
            errors.append("make/design/iterate plan requires a child-usable making loop")
        for label, pattern in (
            ("make", r"\b(?:make|build|design)\w*\b"),
            ("test", r"\b(?:test|try)\w*\b"),
            ("iterate", r"\b(?:change|improve|revise)\w*\b"),
        ):
            if not re.search(pattern, extension_text):
                errors.append(f"make/design/iterate plan is missing its {label} move")

    requested_supports = set(plan.accessibility.requested_supports)
    if "no_writing" in requested_supports:
        if any(_requires_writing(clause) for clause in _instruction_clauses(output)):
            errors.append("no-writing request still requires writing")
        nonwriting_modes = [mode for mode in plan.accessibility.response_choices if mode != "written"]
        if not _enacted_response_modes(output, nonwriting_modes):
            errors.append("no-writing request lacks a visible non-writing response mode")
    if "low_fine_motor" in requested_supports:
        if re.search(r"\b(?:cut|scissors|tiny pieces?|small marks?)\w*\b", full_text) and not re.search(
            r"\b(?:no|without|optional|grown-up)\b.{0,28}\b(?:cut|scissors|tiny pieces?|small marks?)\w*\b",
            full_text,
        ):
            errors.append("fine-motor accommodation still requires cutting or tiny pieces")
        if not re.search(r"\b(?:point|whole objects?|grown-up scribe|move)\w*\b", full_text):
            errors.append("fine-motor accommodation lacks an equivalent response mode")
    if "read_aloud" in requested_supports and not re.search(
        r"\b(?:grown-up|adult)\b.{0,30}\bread\w*\b.{0,20}\baloud\b", full_text
    ):
        errors.append("read-aloud accommodation is not visible in the child-facing output")
    if (
        extension
        and hazard_check_plan.safety.risk_level != "low"
        and not re.search(r"\b(?:grown-up|adult)\b.{0,45}\b(?:supervis|handle|help|operate)\w*\b", extension_text)
    ):
        errors.append("elevated-risk activity lacks the approved grown-up safety boundary")
    if (
        extension
        and hazard_check_plan.safety.risk_level != "low"
        and _assigns_hazardous_handling_to_child(hazard_check_plan, extension)
    ):
        errors.append("generated activity assigns hazardous handling to the child")
    if hazard_check_plan.safety.risk_level != "low" and _assigns_hazardous_handling_to_child(
        hazard_check_plan,
        child_facing_response,
        require_child_direction=True,
    ):
        errors.append("generated response assigns hazardous handling to the child")
    if extension:
        materials = extension.get("materials") or []
        if len(materials) > plan.family_load.max_materials:
            errors.append("activity exceeds the approved material cap")
        effort_order = {"very_low": 0, "low": 1, "medium": 2}
        if effort_order.get(str(extension.get("parent_effort")), 99) > effort_order[plan.family_load.parent_effort]:
            errors.append("activity exceeds the approved parent-effort cap")
    if workflow == "quick_answer" and len(full_text) > 4_500:
        errors.append("quick answer exceeds its reading-load ceiling")
    if workflow == "pull_thread" and len(full_text) > 10_000:
        errors.append("learning thread exceeds its reading-load ceiling")
    if plan.evidence.collection == "not_collected" and any(
        key in output for key in ("score", "mastery", "evidence_record", "learning_opportunity")
    ):
        errors.append("non-collection plan may not manufacture evidence")
    return errors


def validate_artifact_against_plan(plan: ApprovedPedagogyPlan, spec: Any) -> list[str]:
    """Code-owned artifact fidelity, accessibility, effort, and evidence-surface floor."""

    errors: list[str] = []
    artifact_type = str(getattr(spec, "artifact_type", ""))
    trace = getattr(spec, "pedagogy", None)
    if artifact_type not in plan.allowed_artifact_types:
        errors.append("artifact type is not afforded by the approved pedagogy plan")
    if trace is None or trace.plan_id != plan.plan_id or trace.plan_hash != plan.plan_hash:
        errors.append("artifact is missing the exact approved pedagogy binding")
    if int(getattr(spec, "estimated_minutes", 10**6)) > plan.family_load.max_minutes:
        errors.append("artifact exceeds the approved time cap")
    effort_order = {"very_low": 0, "low": 1, "medium": 2}
    if effort_order.get(str(getattr(spec, "parent_effort", "")), 99) > effort_order[plan.family_load.parent_effort]:
        errors.append("artifact exceeds the approved parent-effort cap")
    materials = list(getattr(spec, "materials", []) or [])
    if len(materials) > plan.family_load.max_materials:
        errors.append("artifact exceeds the approved material cap")
    if not str(getattr(spec, "accessibility_text", "")).strip():
        errors.append("artifact requires a visible accessibility description")
    artifact_data = spec.model_dump(mode="json")
    picture_page = artifact_data.get("picture_page")
    serialized = jdump(artifact_data).casefold()
    instruction_fields = {
        "directions",
        "tasks",
        "footer_prompt",
        "mission",
        "materials",
        "substitutions",
        "setup",
        "steps",
        "observation_prompts",
        "variations",
        "scenario",
        "goal",
        "constraints",
        "hints",
        "evidence_of_completion",
        "evidence_rows",
        "reflection",
        "panels",
        "picture_page",
    }
    instruction_data = {key: value for key, value in artifact_data.items() if key in instruction_fields}
    if picture_page:
        # Only these strings are printed. Symbol descriptions, coordinates and
        # color names describe illustrations; they do not direct child actions.
        instruction_data = {
            "directions": picture_page["directions"],
            "cells": [{"label": cell["label"], "action": cell["action"]} for cell in picture_page["cells"]],
        }
    instruction_text = jdump(instruction_data).casefold()
    visible_data = {key: value for key, value in artifact_data.items() if key != "pedagogy"}
    if picture_page:
        visible_data = {"title": spec.title, **instruction_data}
    visible_text = jdump(visible_data).casefold()
    enacted_hazards = _candidate_hazards(
        materials=[] if picture_page else materials,
        instructions=visible_data,
    )
    introduced_hazards = enacted_hazards - set(plan.safety.hazards)
    if (
        picture_page and "sharp_tools" in introduced_hazards
        and re.search(r"\b(?:adult|grown-up|parent)\b.{0,25}\bcut\w*\b", visible_text)
        and not _requires_category_safety_refusal(visible_text, "sharp_tools")
    ):
        # Adult preparation of ordinary cards is not a new unsafe child action.
        # The existing action/owner check still catches a child handling the tool.
        introduced_hazards.discard("sharp_tools")
    if _contains_candidate_self_harm_direction(visible_data):
        errors.append("artifact directs the child toward self-harm")
    if introduced_hazards:
        errors.append(
            "artifact introduces hazards absent from the approved plan: " + ", ".join(sorted(introduced_hazards))
        )
    if (
        _has_powered_appliance_water_action(visible_text)
        and "explicit_appliance_water_prohibition" not in plan.safety.response_requirements
    ):
        errors.append("artifact introduces an appliance-water action absent from the approved safety plan")
    if re.search(r"\b(?:streak|unlock|don't miss out|come back now|cliffhanger)\b", serialized):
        errors.append("artifact violates healthy closure")
    requested_supports = set(plan.accessibility.requested_supports)
    if "no_writing" in requested_supports and any(
        _requires_writing(clause) for clause in _instruction_clauses(visible_data)
    ):
        errors.append("artifact requires writing despite the approved no-writing support")
    if "low_fine_motor" in requested_supports:
        fine_motor_demand = re.search(
            r"\b(?:cut|cutting|scissors|tiny pieces?|small marks?|tiny box(?:es)?|trace|handwrite)\b",
            instruction_text,
        )
        grown_up_substitute = re.search(
            r"\b(?:grown-up|adult)\b.{0,40}\b(?:cut|scribe|write|handle)\w*\b|"
            r"\b(?:no|without|optional|instead of)\b.{0,30}\b(?:cut|writing|tiny pieces?|small marks?)\b",
            visible_text,
        )
        if fine_motor_demand and not grown_up_substitute:
            errors.append("artifact requires fine-motor work without the approved equivalent alternative")
    if "read_aloud" in requested_supports and not re.search(
        r"\b(?:grown-up|adult)\b.{0,40}\bread\w*\b.{0,24}\baloud\b",
        visible_text,
    ):
        errors.append("artifact omits the approved read-aloud support")
    color_terms = r"(?:colou?r|red|orange|yellow|green|blue|purple|pink|brown|black|white|gr[ae]y)"
    color_navigation = re.compile(
        rf"\b(?:match|sort|choose|group|circle|select|point|follow|trace|connect|find|take|go|move|place|put)\b"
        rf".{{0,40}}\b(?:by\s+)?{color_terms}\b|"
        rf"\b{color_terms}\b.{{0,24}}\b(?:path|route|trail|line|arrow|door|box|card|piece|item|one)\b"
    )
    navigation_object = r"(?:path|route|trail|line|arrow|door|box|card|piece|item|one)"
    noncolor_cue = r"(?:label|word|shape|pattern|symbol|texture|number|arrow|stars?|dots?|stripes?|triangles?)"
    redundant_cue = re.compile(
        rf"\b{color_terms}\b.{{0,16}}\b{navigation_object}\b.{{0,20}}"
        rf"\b(?:with|using|marked\s+by|label(?:l)?ed\s+with|showing)\b.{{0,18}}\b{noncolor_cue}\w*\b|"
        rf"\b{noncolor_cue}\w*(?:-marked)?\b.{{0,18}}\b{color_terms}\b.{{0,16}}\b{navigation_object}\b|"
        rf"\b(?:match|sort|choose|group|select)\b.{{0,36}}\bby\b.{{0,18}}\b{color_terms}\b"
        rf".{{0,16}}\b(?:and|plus|with)\b.{{0,12}}\b{noncolor_cue}\w*\b"
    )
    color_only_selection = any(
        color_navigation.search(clause.casefold()) and not redundant_cue.search(clause.casefold())
        for clause in _instruction_clauses(instruction_data)
    )
    if color_only_selection:
        errors.append("artifact makes meaning depend on color alone")

    if plan.safety.risk_level != "low":
        direct_boundary = re.search(
            r"\b(?:grown-up|adult)\b.{0,55}\b(?:directly\s+supervis|handle|operate|pour|use|does)\w*\b",
            visible_text,
        )
        if (plan.safety.minimum_supervision == "direct" and not direct_boundary
                and (not picture_page or enacted_hazards & set(plan.safety.hazards))):
            errors.append("artifact omits the approved direct grown-up safety boundary")
        if _assigns_hazardous_handling_to_child(plan, visible_data):
            errors.append("artifact assigns hazardous handling to the child")

    accessible_nonwriting_modes = _enacted_response_modes(
        instruction_data,
        [mode for mode in plan.accessibility.response_choices if mode != "written"],
    )
    if not accessible_nonwriting_modes:
        errors.append("artifact does not enact an approved non-writing response choice")

    completion_surfaces = {key: artifact_data.get(key) for key in ("evidence_of_completion",) if artifact_data.get(key)}
    completion_surface_text = jdump(completion_surfaces).casefold()
    unqualified_outcome_claim = bool(
        re.search(
            r"\b(?:master\w*|independen\w*|understand\w*|learned|proficien\w*|"
            r"know(?:s|ing)?|can\s+do|gets?\s+it)\b",
            completion_surface_text,
        )
        and not re.search(
            r"\b(?:not|never|cannot|can't|doesn't|does\s+not|isn't|is\s+not)\b.{0,24}"
            r"\b(?:master\w*|independen\w*|understand\w*|learned|proficien\w*|"
            r"know(?:s|ing)?|can\s+do|gets?\s+it)\b",
            completion_surface_text,
        )
    )
    visible_fragments = [fragment.casefold() for fragment in _text_fragments(visible_data)]
    if (
        any(_contains_self_confirming_claim(fragment) for fragment in visible_fragments)
        or any(_contains_unqualified_outcome_assertion(fragment) for fragment in visible_fragments)
        or unqualified_outcome_claim
    ):
        errors.append("artifact completion cannot prove mastery, independence, or understanding")
    if plan.evidence.collection == "future_observation_only":
        has_surface = bool(
            getattr(spec, "observation_prompts", None)
            or getattr(spec, "evidence_of_completion", None)
            or getattr(spec, "tasks", None)
            or getattr(spec, "footer_prompt", None)
        )
        if not has_surface:
            errors.append("artifact lacks the plan-approved observable response surface")
        semantic_evidence_clauses = _semantic_evidence_clauses(instruction_data)
        action_lead = (
            r"(?:^|,\s*|\b(?:and|then|next|first|finally|now)\b[,:]?\s+|"
            r"\b(?:you|the\s+child|child)\b.{0,24})"
        )
        evidence_requirements = {
            "explain-observation": [
                (
                    "observation",
                    rf"{action_lead}(?:please\s+)?(?:notice|observe|watch|look\s+for)\w*\b|"
                    r"\bwhat\b.{0,18}\b(?:happened|changed|did\s+you\s+notice)\b",
                ),
                (
                    "explanation",
                    rf"{action_lead}(?:please\s+)?(?:explain|tell\s+why|give\s+(?:a\s+)?reason)\w*\b|"
                    r"\bwhy\s+do\s+you\s+think\b|^because\b",
                ),
            ],
            "predict-test-explain": [
                (
                    "prediction",
                    rf"{action_lead}(?:please\s+)?(?:predict(?!ion)\w*|guess\w*|"
                    r"(?:make|say|tell|draw|choose|write)\w*.{0,24}\b(?:prediction|guess)\w*)\b|"
                    r"\bwhat\b.{0,22}\b(?:do\s+you\s+predict|do\s+you\s+think\s+will)\b",
                ),
                (
                    "test",
                    rf"{action_lead}(?:please\s+)?(?:test|try|experiment|run\s+(?:the|a)\s+test)\w*\b",
                ),
                (
                    "observation",
                    rf"{action_lead}(?:please\s+)?(?:notice|observe|watch|look\s+for)\w*\b|"
                    r"\bwhat\b.{0,18}\b(?:happened|changed|did\s+you\s+notice)\b",
                ),
                (
                    "explanation",
                    rf"{action_lead}(?:please\s+)?(?:explain|tell\s+why|give\s+(?:a\s+)?reason)\w*\b|"
                    r"\bwhy\s+do\s+you\s+think\b|^because\b",
                ),
            ],
            "compare-classify": [
                (
                    "comparison or classification",
                    rf"{action_lead}(?:please\s+)?(?:compare|classify|sort|group)\w*\b|"
                    r"\b(?:how|which)\b.{0,36}\b(?:same|different|alike|more|less|most|least|"
                    r"bigger|smaller|faster|slower)\b",
                ),
                (
                    "named attribute",
                    rf"{action_lead}(?:please\s+)?(?:name|choose|use|compare|sort|group)\w*\b"
                    r".{0,30}\b(?:attribute|clue|feature|property|shape|size|number|one\s+way)\b|"
                    r"\bby\b.{0,18}\b(?:shape|size|number|feature|property|clue)\b",
                ),
                (
                    "explanation",
                    rf"{action_lead}(?:please\s+)?(?:explain|tell\s+why|give\s+(?:a\s+)?reason)\w*\b|"
                    r"\bwhy\s+do\s+you\s+think\b|^because\b",
                ),
            ],
            "teach-back": [
                (
                    "teach-back",
                    rf"{action_lead}(?:please\s+)?(?:explain|teach|show|tell)\w*\b",
                )
            ],
            "retrieval-application": [
                (
                    "recall",
                    rf"{action_lead}(?:please\s+)?(?:remember|recall)\w*\b|"
                    r"^(?:before|without)\s+looking\b.{0,28}\b(?:remember|recall|say|tell|draw)\w*\b",
                ),
                (
                    "application",
                    rf"{action_lead}(?:please\s+)?(?:apply|use\s+that|use\s+the\s+idea)\w*\b|"
                    r"\btry\b.{0,24}\b(?:new|different)\s+example\b",
                ),
            ],
            "design-iterate": [
                ("make or design", rf"{action_lead}(?:please\s+)?(?:make|build|design)\w*\b"),
                ("test", rf"{action_lead}(?:please\s+)?(?:test|try)\w*\b"),
                (
                    "observation",
                    rf"{action_lead}(?:please\s+)?(?:notice|observe|watch|look\s+for)\w*\b|"
                    r"\bwhat\b.{0,18}\b(?:happened|changed|did\s+you\s+notice)\b",
                ),
                (
                    "iteration",
                    rf"{action_lead}(?:please\s+)?(?:change|improve|revise|try\s+again)\w*\b",
                ),
            ],
        }
        missing_mechanics = [
            label
            for label, pattern in evidence_requirements.get(plan.evidence.template_id, [])
            if not any(re.search(pattern, clause) for clause in semantic_evidence_clauses)
        ]
        if missing_mechanics:
            errors.append(
                "artifact does not enact the approved evidence template: missing " + ", ".join(missing_mechanics)
            )
        supported_modes = list(plan.evidence.response_modes)
        if supported_modes and not _enacted_response_modes(instruction_data, supported_modes):
            errors.append("artifact lacks a response mode approved by the evidence template")
    return errors


def compile_artifact_action(
    payload: dict[str, Any],
    plan: ApprovedPedagogyPlan,
) -> dict[str, Any]:
    """Reduce arbitrary model payload to a code-owned, plan-bound artifact request."""

    allowed_keys = {"artifact_type"}
    if set(payload) - allowed_keys:
        raise ValueError("model artifact action contains executable or authority-bearing fields")
    artifact_type = str(payload.get("artifact_type") or "worksheet").casefold()
    if artifact_type not in plan.allowed_artifact_types:
        raise ValueError("artifact type is not afforded by the approved pedagogy plan")
    return {
        "artifact_type": artifact_type,
        "source_event_id": plan.event_id,
        "pedagogy_plan_id": plan.plan_id,
        "pedagogy_plan_hash": plan.plan_hash,
    }
