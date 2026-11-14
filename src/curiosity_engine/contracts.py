from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from .construction_patterns import PATTERNS, ConstructionPattern

FORBIDDEN_3D_PRINT_TERMS = (
    "3d printer",
    "3-d printer",
    "3d-print",
    "3-d-print",
    "slicer software",
    "slicer file",
    "filament spool",
    ".gcode",
    ".stl",
    ".3mf",
)


def _reject_3d_printing(values: list[str]) -> None:
    combined = "\n".join(values).casefold()
    if any(term in combined for term in FORBIDDEN_3D_PRINT_TERMS):
        raise ValueError("3D-printing materials and files are outside the MVP")


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Event(StrictModel):
    type: str = Field(min_length=1, max_length=80)
    child_id: str | None = Field(default=None, max_length=120)
    text: str = Field(min_length=1, max_length=20_000)
    source: str = Field(default="parent", min_length=1, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:16]}")
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def payload_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at"})
        import json

        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class GraphMutationKind(StrEnum):
    ADD_OBSERVATION = "add_observation"
    UPSERT_NODE = "upsert_node"
    ADD_EDGE = "add_edge"


class GraphMutation(StrictModel):
    kind: GraphMutationKind
    child_id: str | None = None
    node_kind: str | None = None
    label: str | None = None
    confidence: float = Field(default=0.6, ge=0, le=1)
    state: dict[str, Any] = Field(default_factory=dict)
    observation_kind: str | None = None
    text: str | None = None
    source: str = "reasoning"
    source_node_id: int | None = None
    relation: str | None = None
    target_node_id: int | None = None
    evidence_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def has_fields_required_by_kind(self) -> GraphMutation:
        if self.kind == GraphMutationKind.ADD_OBSERVATION:
            if not self.observation_kind or not self.text:
                raise ValueError("add_observation requires observation_kind and text")
        elif self.kind == GraphMutationKind.UPSERT_NODE:
            if not self.node_kind or not self.label:
                raise ValueError("upsert_node requires node_kind and label")
        elif self.kind == GraphMutationKind.ADD_EDGE:
            if self.source_node_id is None or self.target_node_id is None or not self.relation:
                raise ValueError("add_edge requires source_node_id, target_node_id, and relation")
        return self


class ObservationCandidate(StrictModel):
    """Non-authoritative semantic annotation proposed by a model.

    Candidates deliberately contain no child, event, source, node, edge, evidence,
    confidence, learner-state, or arbitrary metadata fields. Phase 0 retains them
    only in the reviewed response audit; Phase 1 may compile selected candidates
    through versioned evidence rules.
    """

    kind: Literal["concept_mention", "learning_process"]
    label: str | None = Field(default=None, min_length=1, max_length=160)
    text: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def has_kind_payload(self) -> ObservationCandidate:
        if self.kind == "concept_mention" and not self.label:
            raise ValueError("concept_mention requires label")
        if self.kind == "learning_process" and not self.text:
            raise ValueError("learning_process requires text")
        return self


class ActionType(StrEnum):
    PROPOSE_ARTIFACT = "propose_artifact"


class ActionRequest(StrictModel):
    type: ActionType
    payload: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1, max_length=1_000)


class PrintablePiece(StrictModel):
    """One large child-usable part of a deterministic activity page."""

    label: str = Field(min_length=1, max_length=32)
    prompt: str = Field(default="", max_length=90)
    shape: Literal["leaf", "target", "card", "circle", "arrow"] = "card"


class PrintablePlan(StrictModel):
    """A semantic plan for a useful printout; code owns its layout and drawing."""

    kind: Literal["target_set", "play_cards", "recording_sheet", "reach_comparison"]
    title: str = Field(min_length=1, max_length=90)
    child_directions: str = Field(min_length=1, max_length=220)
    parent_setup: str = Field(default="", max_length=220)
    pedagogical_value: str = Field(min_length=1, max_length=300)
    pieces: list[PrintablePiece] = Field(min_length=2, max_length=6)


class PhysicalExtension(StrictModel):
    title: str = Field(min_length=1, max_length=160)
    construction_pattern: ConstructionPattern | None = None
    instructions: list[str] = Field(default_factory=list, max_length=6)
    prompts: list[str] = Field(default_factory=list, max_length=6)
    materials: list[str] = Field(default_factory=list, max_length=8)
    parent_effort: Literal["very_low", "low", "medium"] = "low"
    printable: PrintablePlan | None = None

    @model_validator(mode="after")
    def uses_ordinary_materials(self) -> PhysicalExtension:
        if self.construction_pattern:
            pattern = PATTERNS[self.construction_pattern]
            self.instructions = list(pattern["steps"])
            self.materials = list(pattern["materials"])
            self.printable = None
        elif not self.instructions:
            raise ValueError("an activity needs instructions or a construction pattern")
        printable_text = []
        if self.printable:
            printable_text = [
                self.printable.title,
                self.printable.child_directions,
                self.printable.parent_setup,
                self.printable.pedagogical_value,
                *[text for piece in self.printable.pieces for text in (piece.label, piece.prompt)],
            ]
        _reject_3d_printing([self.title, *self.instructions, *self.prompts, *self.materials, *printable_text])
        return self


class VisualPanel(StrictModel):
    label: str = Field(min_length=1, max_length=48)
    detail: str = Field(min_length=1, max_length=140)
    icon: Literal[
        "height",
        "weight",
        "strength",
        "look",
        "predict",
        "try",
        "go",
        "stop",
        "turn",
        "robot",
        "water",
        "ice",
        "wing_wide",
        "wing_narrow",
        "wing_swept",
        "wing_straight",
        "question",
        "idea",
    ]


class VisualIntent(StrictModel):
    """A semantic request for one child-facing visual; code still owns whether and how it renders."""

    kind: Literal["comparison_cards", "activity_sequence", "decorative_illustration", "researched_photo"]
    purpose: Literal["compare", "sequence", "notice", "imagine", "observe"]
    knowledge_role: Literal["decorative", "supportive", "instructional"]
    title: str = Field(min_length=1, max_length=120)
    pedagogical_value: str = Field(min_length=1, max_length=300)
    caption: str = Field(min_length=1, max_length=280)
    alt_text: str = Field(min_length=12, max_length=1_000)
    subject: str | None = Field(default=None, max_length=160)
    research_query: str | None = Field(default=None, max_length=180)
    preferred_sources: list[Literal["smithsonian", "nasa", "noaa", "usgs", "wikimedia"]] = Field(
        default_factory=list,
        max_length=5,
    )
    panels: list[VisualPanel] = Field(default_factory=list, max_length=4)
    source_refs: list[str] = Field(default_factory=list, max_length=4)
    not_to_scale: bool = False

    @model_validator(mode="after")
    def valid_shape(self) -> VisualIntent:
        if self.kind in {"comparison_cards", "activity_sequence"} and len(self.panels) < 2:
            raise ValueError("deterministic visual cards require at least two panels")
        if self.kind == "decorative_illustration":
            if not self.subject:
                raise ValueError("decorative illustrations require a short subject")
            if self.panels:
                raise ValueError("decorative illustrations may not contain instructional panels")
        if self.kind == "researched_photo":
            if not self.research_query:
                raise ValueError("researched photos require a short public research query")
            if self.knowledge_role != "supportive":
                raise ValueError("researched response photos must support a reviewed claim")
            if self.panels:
                raise ValueError("researched photos may not contain model-authored instructional panels")
            if self.subject:
                raise ValueError("researched photos use research_query rather than a generative subject")
        if self.kind == "comparison_cards" and not self.not_to_scale:
            raise ValueError("MVP comparison cards must be explicitly marked not to scale")
        return self


class ImageResearchQuery(StrictModel):
    """A public, minimized query that is safe to send to an allowlisted image collection."""

    query: str = Field(min_length=2, max_length=180)
    purpose: Literal["spark", "observe", "compare", "evidence"]
    desired_view: str | None = Field(default=None, max_length=120)
    age_band: Literal["k2", "elementary"] = "k2"
    preferred_sources: list[Literal["smithsonian", "nasa", "noaa", "usgs", "wikimedia"]] = Field(
        default_factory=list,
        max_length=5,
    )
    exclude_people: bool = True
    exclusions: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("query", "desired_view")
    @classmethod
    def public_terms_only(cls, value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.casefold()
        forbidden = (
            "my child",
            "my son",
            "my daughter",
            "our family",
            "school email",
            "private resource",
            "purchased resource",
            "slack",
            "school",
            "elementary",
            "academy",
            "classroom",
            "teacher",
            "student",
            "address",
            "street",
            "avenue",
            "boulevard",
            "apartment",
            "neighborhood",
            "near me",
            "doctor",
            "diagnosis",
            "therapy",
            "medication",
            "hospital",
            "clinic",
            "health",
            "medical",
            "condition",
            "birthday",
            "zip code",
            "home address",
            "our home",
            "our house",
            "my house",
            "my mom",
            "my dad",
            "my brother",
            "my sister",
            "my friend",
        )
        if any(marker in lowered for marker in forbidden):
            raise ValueError("image research queries may not contain family or private-resource context")
        if any(marker in value for marker in ("<@", "/Users/", "private/", "http://", "https://", "@", '"', "'")):
            raise ValueError("image research query contains an identifier, path, or URL")
        normalized = " ".join(lowered.split())
        if len(normalized.split()) > 12 or not re.fullmatch(r"[a-z][a-z -]*", normalized):
            raise ValueError("image research queries must be short public-topic words without identifiers")
        return normalized


class ImageRights(StrictModel):
    source_name: Literal["smithsonian", "nasa", "noaa", "usgs", "wikimedia"]
    source_item_id: str = Field(min_length=1, max_length=240)
    canonical_url: str = Field(min_length=12, max_length=2_000)
    creator: str = Field(default="", max_length=500)
    license_name: str = Field(min_length=1, max_length=160)
    license_url: str = Field(min_length=12, max_length=2_000)
    attribution: str = Field(min_length=1, max_length=1_000)
    rights_status: Literal["verified", "policy_based"]
    modifications_allowed: bool | None
    commercial_use_allowed: bool | None
    share_alike: bool = False
    rights_notes: list[str] = Field(default_factory=list, max_length=12)
    retrieved_at: datetime = Field(default_factory=utc_now)
    metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def has_public_canonical_urls(self) -> ImageRights:
        if not self.canonical_url.startswith("https://") or not self.license_url.startswith("https://"):
            raise ValueError("image rights URLs must use HTTPS")
        if self.rights_status == "verified" and (
            self.modifications_allowed is not True or self.commercial_use_allowed is not True
        ):
            raise ValueError("verified reusable images require affirmative modification and commercial-use rights")
        return self


class ImageSafetyReview(StrictModel):
    verdict: Literal["pass", "fail"]
    reasons: list[str] = Field(default_factory=list, max_length=20)
    people_present: bool = False
    frightening: bool = False
    graphic_or_gory: bool = False
    personal_information_visible: bool = False
    watermark_or_advertising: bool = False
    subject_matches: bool = True
    observable_description: str = Field(default="", max_length=700)
    representation_type: Literal["photograph", "processed_science_image", "illustration", "diagram", "unknown"] = (
        "unknown"
    )
    context_note: str = Field(default="", max_length=240)
    reviewed_at: datetime = Field(default_factory=utc_now)
    reviewer: str = Field(default="deterministic", max_length=160)

    @model_validator(mode="after")
    def internally_consistent_release(self) -> ImageSafetyReview:
        disqualifying = (
            self.people_present,
            self.frightening,
            self.graphic_or_gory,
            self.personal_information_visible,
            self.watermark_or_advertising,
            not self.subject_matches,
        )
        if self.verdict == "pass" and any(disqualifying):
            raise ValueError("a passing image review cannot contain a disqualifying flag")
        return self


class ResearchedImageAsset(StrictModel):
    title: str = Field(min_length=1, max_length=300)
    caption: str = Field(min_length=1, max_length=500)
    alt_text: str = Field(min_length=12, max_length=1_000)
    content_role: Literal["spark", "observe", "compare", "evidence"]
    local_path: str = Field(min_length=1, max_length=2_000)
    mime_type: Literal["image/png", "image/jpeg"]
    width: int = Field(ge=300, le=8_192)
    height: int = Field(ge=300, le=8_192)
    byte_count: int = Field(ge=1, le=12 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rights: ImageRights
    safety: ImageSafetyReview
    transformations: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def passed_release_gates(self) -> ResearchedImageAsset:
        if self.safety.verdict != "pass":
            raise ValueError("researched image asset did not pass child-safety review")
        if self.safety.representation_type not in {"photograph", "processed_science_image"}:
            raise ValueError("researched image asset is not verified observational imagery")
        if self.safety.representation_type == "processed_science_image" and not self.safety.context_note:
            raise ValueError("processed science imagery requires a visible context note")
        return self


class GeneratedIllustrationAsset(StrictModel):
    """Code-bound generated artwork composed into a learning page."""

    local_path: str = Field(min_length=1, max_length=2_000)
    mime_type: Literal["image/png", "image/jpeg"]
    width: int = Field(ge=512, le=8_192)
    height: int = Field(ge=512, le=8_192)
    byte_count: int = Field(ge=1, le=20 * 1024 * 1024)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: Literal["openai"]
    model: str = Field(min_length=1, max_length=240)
    request_id: str | None = Field(default=None, max_length=240)
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: Literal["challenge-art-v1", "picture-art-v1", "picture-sheet-v1"] = "challenge-art-v1"
    alt_text: str = Field(min_length=12, max_length=500)


class ChallengeVisualBrief(StrictModel):
    """Code-owned division of labor between illustration and instruction."""

    illustration_role: Literal["engagement"] = "engagement"
    subject_key: Literal["ice_lab", "balloon_and_open_cup", "paper_airplane_flight"]
    teaching_diagram: Literal["ice_mold_mystery", "air_escape_test", "paper_airplane_wing_tips"]
    answer_timing: Literal["withhold_until_observation", "explain_before_activity"]
    art_slot: Literal["hero_left"] = "hero_left"
    response_layout: Literal["paired_predictions", "predict_observe", "trial_table"]


class ChallengePageReview(StrictModel):
    """A compact rendered-page rubric that favors family value over polish theater."""

    usefulness: int = Field(ge=1, le=5)
    engagement: int = Field(ge=1, le=5)
    factuality: int = Field(ge=1, le=5)
    learning_value: int = Field(ge=1, le=5)
    kid_usability: int = Field(ge=1, le=5)
    visual_integrity: int = Field(ge=1, le=5)
    art_verdict: Literal["pass", "fallback"]
    critical_defects: list[str] = Field(default_factory=list, max_length=5)
    improvements: list[str] = Field(default_factory=list, max_length=5)


class VisualPlanUnit(StrictModel):
    kind: Literal["wonder_photo", "learning_photo", "learning_graphic", "printable", "story_art"]
    purpose: Literal["spark", "observe", "compare", "explain", "do"]
    required: bool = False
    visual_intent: VisualIntent | None = None
    artifact_type: Literal["worksheet", "reference", "activity", "challenge"] | None = None

    @model_validator(mode="after")
    def has_payload(self) -> VisualPlanUnit:
        if self.kind == "printable" and not self.artifact_type:
            raise ValueError("printable visual units require an artifact type")
        if self.kind != "printable" and self.visual_intent is None:
            raise ValueError("non-printable visual units require a visual intent")
        return self


class VisualPlan(StrictModel):
    role: Literal["wonder", "observe", "compare", "explain", "do", "none"]
    rationale: str = Field(min_length=1, max_length=500)
    units: list[VisualPlanUnit] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def respects_visual_budget(self) -> VisualPlan:
        if self.role == "none" and self.units:
            raise ValueError("no-visual plans may not contain release units")
        if sum(unit.kind in {"wonder_photo", "learning_photo", "learning_graphic"} for unit in self.units) > 1:
            raise ValueError("a response may contain only one primary visual")
        return self


class PullThreadOutput(StrictModel):
    # Natural parent chat can be a short game, several ideas or an explanation.
    # The legacy lesson fields remain readable for existing stored responses.
    parent_response: str | None = Field(default=None, min_length=1, max_length=3_500)
    hook: str = Field(default="", max_length=500)
    # A proportionate parent answer may not need a separate demonstration.
    show: str = Field(default="", max_length=1_000)
    # A parent revision may deliberately remove the question/game portion of an
    # otherwise useful learning thread.  Keep the field structurally stable,
    # but do not force a fake question into a concise reassurance or repair.
    ask: str = Field(default="", max_length=500)
    nugget: str = Field(default="", max_length=500)
    next_possible_concepts: list[str] = Field(default_factory=list, max_length=6)
    physical_extension: PhysicalExtension | None = None
    visual: VisualIntent | None = None
    observation_candidates: list[ObservationCandidate] = Field(default_factory=list, max_length=12)
    actions: list[ActionRequest] = Field(default_factory=list, max_length=4)
    resource_refs: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def has_parent_answer(self) -> PullThreadOutput:
        if self.parent_response:
            self.hook = self.show = self.ask = self.nugget = ""
        elif not self.hook.strip() or not self.nugget.strip():
            raise ValueError("provide parent_response or a complete legacy answer")
        return self

    @field_validator("ask")
    @classmethod
    def ask_is_a_question(cls, value: str) -> str:
        if value and "?" not in value:
            raise ValueError("ask must be phrased as a question")
        return value


class AnswerMode(StrEnum):
    """Requested depth for one reviewed child-facing answer."""

    AUTO = "auto"
    QUICK = "quick"
    EXPLORE = "explore"


class QuickAnswerOutput(StrictModel):
    """A proportionate child answer that preserves curiosity without forcing a lesson template."""

    answer: str = Field(min_length=1, max_length=1_200)
    notice: str | None = Field(default=None, max_length=500)
    question_back: str | None = Field(default=None, max_length=400)
    next_possible_concepts: list[str] = Field(default_factory=list, max_length=6)
    visual: VisualIntent | None = None
    observation_candidates: list[ObservationCandidate] = Field(default_factory=list, max_length=12)
    actions: list[ActionRequest] = Field(default_factory=list, max_length=4)
    resource_refs: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("question_back")
    @classmethod
    def question_back_is_a_question(cls, value: str | None) -> str | None:
        if value is not None and "?" not in value:
            raise ValueError("question_back must be phrased as a question")
        return value


class InterestSignalOutput(StrictModel):
    summary: str = Field(min_length=1, max_length=500)
    observation_candidates: list[ObservationCandidate] = Field(default_factory=list, max_length=12)
    actions: list[ActionRequest] = Field(default_factory=list, max_length=2)


class FeedbackOutput(StrictModel):
    acknowledged: str = Field(min_length=1, max_length=500)
    observation_candidates: list[ObservationCandidate] = Field(default_factory=list, max_length=8)
    actions: list[ActionRequest] = Field(default_factory=list, max_length=2)


class GenericOutput(StrictModel):
    summary: str = Field(min_length=1, max_length=1_000)
    observation_candidates: list[ObservationCandidate] = Field(default_factory=list, max_length=8)
    actions: list[ActionRequest] = Field(default_factory=list, max_length=2)


class CriticVerdict(StrEnum):
    PASS = "pass"
    REVISE = "revise"
    REJECT = "reject"


class CriticResult(StrictModel):
    verdict: CriticVerdict
    concerns: list[str] = Field(default_factory=list, max_length=12)
    required_changes: list[str] = Field(default_factory=list, max_length=12)


class ReasoningEnvelope(StrictModel):
    workflow: str
    output: dict[str, Any]
    critiques: list[CriticResult] = Field(default_factory=list)
    critique_rounds: list[list[CriticResult]] = Field(default_factory=list, max_length=6)
    revision_rounds: int = Field(default=0, ge=0, le=10)
    recovery_strategy: Literal["rebuild_from_scratch", "remove_unrelated_sources"] | None = None
    backend: str
    model: str | None = None


class RunResult(StrictModel):
    event_id: str
    workflow: str
    status: Literal["queued", "completed", "rejected", "failed"]
    output: dict[str, Any] = Field(default_factory=dict)
    graph_updates: list[dict[str, Any]] = Field(default_factory=list)
    actions: list[dict[str, Any]] = Field(default_factory=list)
    run_id: int | None = None
    duplicate: bool = False


class DirectorCandidate(StrictModel):
    kind: Literal["pull_thread", "artifact", "conversation", "do_nothing"]
    title: str = Field(min_length=1, max_length=160)
    rationale: str = Field(min_length=1, max_length=1_000)
    parent_effort: Literal["very_low", "low", "medium"] = "low"
    priority: float = Field(default=0.5, ge=0, le=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class DirectorOutput(StrictModel):
    choice: DirectorCandidate
    considered: list[DirectorCandidate] = Field(default_factory=list, max_length=8)


class FeedbackInput(StrictModel):
    event_id: str | None = None
    experience_id: str | None = None
    artifact_id: str | None = None
    child_id: str
    outcome: Literal[
        "loved",
        "engaged",
        "neutral",
        "too_easy",
        "too_hard",
        "not_used",
        "disliked",
        "helpful",
        "not_helpful",
    ]
    note: str | None = Field(default=None, max_length=2_000)
    source: str = "parent"
    used: bool | None = None
    chose_to_continue: bool | None = None
    made_prediction: bool | None = None
    gave_explanation: bool | None = None
    help_level: Literal[
        "none",
        "clarifying_prompt",
        "general_prompt",
        "hint",
        "worked_example",
        "joint_completion",
        "direct_answer",
        "unknown",
    ] | None = None
    asked_new_question: bool | None = None
    parent_correction: bool | None = None


class VisualQAResult(StrictModel):
    verdict: Literal["pass", "fail"]
    reasons: list[str] = Field(default_factory=list, max_length=20)
    inspected_pages: int = Field(ge=1)


class ArtifactAsset(StrictModel):
    kind: str
    method: Literal["trusted_source", "deterministic", "generative", "none"]
    knowledge_bearing: bool = False
    contains_text: bool = False
    exact_count: int | None = None
    source: str | None = None
    local_path: str | None = None


class ArtifactSpec(StrictModel):
    artifact_type: Literal[
        "wonder_page",
        "reference_page",
        "mini_poster",
        "challenge_card",
        "case_file",
        "field_guide",
        "cut_and_build",
        "mini_book",
        "follow_the_thread",
        "worksheet",
    ]
    title: str = Field(min_length=1, max_length=180)
    trust_tier: Literal["A", "B", "C"]
    target_age: int | None = Field(default=None, ge=2, le=18)
    target_grade: str | None = Field(default=None, max_length=40)
    kicker: str = Field(default="CURIOUS?", max_length=100)
    prompt: str = Field(min_length=1, max_length=500)
    body: list[str] = Field(default_factory=list, max_length=8)
    footer: str = Field(default="Predict first. Then investigate.", max_length=300)
    assets: list[ArtifactAsset] = Field(default_factory=list, max_length=8)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    fact_model: dict[str, Any] | None = None
    source_event_id: str | None = None

    @model_validator(mode="after")
    def has_audience(self) -> ArtifactSpec:
        if self.target_age is None and not self.target_grade:
            raise ValueError("target_age or target_grade is required")
        _reject_3d_printing([self.title, self.kicker, self.prompt, *self.body, self.footer])
        for asset in self.assets:
            if asset.local_path:
                _reject_3d_printing([asset.local_path])
        return self


class CapabilityCard(StrictModel):
    """Progressively disclosed description of a reviewed domain workflow."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,79}$")
    version: str = Field(min_length=1, max_length=32)
    description: str = Field(min_length=1, max_length=500)
    output_kinds: list[str] = Field(default_factory=list, max_length=8)
    skill_ids: list[str] = Field(default_factory=list, max_length=12)
    tool_names: list[str] = Field(default_factory=list, max_length=20)
    approval: Literal["none", "household_opt_in", "per_call"] = "none"
    expected_latency: Literal["instant", "short", "background"] = "short"


class SkillCard(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,79}$")
    version: str = Field(min_length=1, max_length=32)
    description: str = Field(min_length=1, max_length=500)
    instructions_path: str = Field(min_length=1, max_length=240)
    required_tools: list[str] = Field(default_factory=list, max_length=20)
    family_private: bool = False


class ToolSpec(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")
    version: str = Field(min_length=1, max_length=32)
    description: str = Field(min_length=1, max_length=500)
    risk: Literal["read_only", "low", "sensitive", "high"]
    side_effect: Literal["none", "local_write", "external_read", "external_write"]
    approval: Literal["none", "household_opt_in", "per_call"] = "none"
    data_classes: list[str] = Field(default_factory=list, max_length=12)
    origins: list[Literal["slack", "schedule", "cli"]] = Field(default_factory=list, max_length=3)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_retries: int = Field(default=0, ge=0, le=3)


class InteractionOption(StrictModel):
    label: str = Field(min_length=1, max_length=75)
    intent: str = Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")
    payload: dict[str, Any] = Field(default_factory=dict)
    style: Literal["default", "primary", "danger"] = "default"


class InteractionPlan(StrictModel):
    """Transport-neutral parent interaction; Slack JSON never enters model context."""

    kind: Literal[
        "choose_child",
        "choose_one",
        "confirm_action",
        "request_revision",
        "continue_thread",
        "ask_for_hint",
        "rate_output",
        "artifact_preview",
        "job_progress",
        "manage_children",
    ]
    title: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=500)
    options: list[InteractionOption] = Field(default_factory=list, max_length=10)
    allow_free_text: bool = True
    expires_in_minutes: int = Field(default=120, ge=1, le=10_080)

    @model_validator(mode="after")
    def has_useful_options(self) -> InteractionPlan:
        if self.kind not in {"job_progress"} and not self.options:
            raise ValueError("interactive plans require at least one option")
        return self


class InteractionEvent(StrictModel):
    interaction_id: str = Field(min_length=1, max_length=120)
    option_token: str = Field(min_length=16, max_length=240)
    transport: Literal["slack"] = "slack"
    team_id: str = Field(min_length=1, max_length=120)
    user_id: str = Field(min_length=1, max_length=120)
    channel_id: str = Field(min_length=1, max_length=120)
    thread_id: str | None = Field(default=None, max_length=120)
    external_event_id: str = Field(min_length=1, max_length=240)


class ParentAgentToolCall(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{1,79}$")
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1, max_length=500)


class ThreadOutputRef(StrictModel):
    """A code-resolved reference to one output in the current parent thread."""

    ref_id: str = Field(pattern=r"^(?:msg|rel|art)_[a-zA-Z0-9_-]{1,120}$")
    kind: Literal["answer", "visual", "artifact", "interaction"]
    title: str | None = Field(default=None, max_length=180)
    snippet: str = Field(min_length=1, max_length=500)
    event_id: str | None = Field(default=None, max_length=160)
    artifact_id: str | None = Field(default=None, max_length=160)


class ThreadPreference(StrictModel):
    """Explicit, reversible presentation guidance scoped to one thread."""

    category: Literal[
        "answer_style",
        "visual_style",
        "artifact_style",
        "activity_style",
        "interaction_style",
    ]
    value: str = Field(min_length=1, max_length=400)
    source_message_id: str = Field(min_length=1, max_length=160)


class ParentSessionState(StrictModel):
    version: int = Field(default=1, ge=1, le=10)
    preferences: list[ThreadPreference] = Field(default_factory=list, max_length=10)
    active_output_ref: str | None = Field(default=None, max_length=160)


class ParentAgentTurn(StrictModel):
    requested_output: Literal["conversation", "material"] = "conversation"
    message: str | None = Field(default=None, max_length=2_000)
    tool_calls: list[ParentAgentToolCall] = Field(default_factory=list, max_length=3)
    # Interactive controls are code-compiled after a reviewed result. A model may
    # write explanatory text or select typed tools, but it cannot author a label
    # whose hidden intent/payload carries durable authority.
    interaction: None = None
    done: bool = True

    @model_validator(mode="after")
    def contains_an_action(self) -> ParentAgentTurn:
        if not self.message and not self.tool_calls and not self.interaction:
            raise ValueError("parent agent turn must communicate or act")
        return self


class ImageContextOutput(StrictModel):
    """Bounded evidence extracted from one parent-shared image."""

    summary: str = Field(min_length=1, max_length=700)
    learning_concepts: list[str] = Field(default_factory=list, max_length=8)
    visible_tasks: list[str] = Field(default_factory=list, max_length=6)
    visible_details: list[str] = Field(default_factory=list, max_length=10)
    possible_play_threads: list[str] = Field(default_factory=list, max_length=5)
    uncertainties: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("summary", mode="before")
    @classmethod
    def bound_summary(cls, value: Any) -> Any:
        # Extra detail is not a reason to discard an otherwise readable photo.
        return value[:700] if isinstance(value, str) else value

    @field_validator("learning_concepts", "visible_tasks", "visible_details", "possible_play_threads", "uncertainties",
                     mode="before")
    @classmethod
    def bound_details(cls, value: Any, info: Any) -> Any:
        limits = {"learning_concepts": 8, "visible_tasks": 6, "visible_details": 10,
                  "possible_play_threads": 5, "uncertainties": 6}
        return value[:limits[info.field_name]] if isinstance(value, list) else value


class WorksheetTask(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,39}$")
    kind: Literal[
        "match", "sort", "sequence", "circle", "label", "draw", "predict", "measure", "record", "short_response"
    ]
    instruction: str = Field(min_length=1, max_length=220)
    choices: list[str] = Field(default_factory=list, max_length=8)
    answer: str | list[str] | None = None
    response_lines: int = Field(default=1, ge=0, le=10)

    @model_validator(mode="after")
    def matches_mechanic(self) -> WorksheetTask:
        if self.kind in {"match", "sort", "sequence", "circle"} and len(self.choices) < 2:
            raise ValueError(f"{self.kind} tasks require at least two choices")
        if self.kind in {"draw", "label"} and self.response_lines:
            self.response_lines = 0
        return self


WorksheetArchetype = Literal[
    "look_notice_wonder",
    "compare_circle_explain",
    "match_two_columns",
    "cut_sort_categories",
    "sequence_story_cards",
    "label_parts",
    "observation_checklist",
    "predict_test_record",
    "measure_tally_graph",
    "build_test_improve",
    "mystery_clue_case",
    "mini_field_guide",
    "play_pieces_targets",
    "legacy_mixed",
]

WorksheetTheme = Literal["storybook", "space", "nature", "ocean", "workshop", "mystery"]
WorksheetDiagramKind = Literal["auto", "plant", "insect", "boat", "generic"]
WorksheetPieceShape = Literal["auto", "leaf", "target", "card", "circle", "arrow"]


class WorksheetBrief(StrictModel):
    """Compact model-owned content; code owns the selected archetype and all page geometry."""

    title: str = Field(min_length=1, max_length=90)
    target_grade: str = Field(min_length=1, max_length=40)
    learning_objective: str = Field(min_length=1, max_length=300)
    estimated_minutes: int = Field(ge=3, le=30)
    parent_effort: Literal["very_low", "low", "medium"] = "low"
    story_theme: str = Field(default="curiosity mission", max_length=120)
    accessibility_text: str = Field(min_length=1, max_length=600)
    archetype: WorksheetArchetype
    theme: WorksheetTheme = "storybook"
    subject_labels: list[str] = Field(default_factory=list, max_length=3)
    category_labels: list[str] = Field(default_factory=list, max_length=3)
    diagram_kind: WorksheetDiagramKind = "auto"
    piece_shape: WorksheetPieceShape = "auto"
    directions: str = Field(min_length=1, max_length=260)
    tasks: list[WorksheetTask] = Field(min_length=2, max_length=8)
    celebration: str = Field(default="Mission complete!", max_length=100)
    suggested_research_query: str | None = Field(default=None, max_length=180)

    @field_validator("suggested_research_query")
    @classmethod
    def safe_public_research_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ImageResearchQuery(query=value, purpose="observe").query


class PedagogyPersonalizationConceptRef(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    version: int = Field(ge=1)


class PedagogyPersonalizationTrace(StrictModel):
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
    concept_refs: list[PedagogyPersonalizationConceptRef] = Field(min_length=1, max_length=3)
    question_intent: str = Field(min_length=1, max_length=80)
    policy_id: str = Field(min_length=1, max_length=160)
    policy_version: int = Field(ge=1)
    definition_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    family_authorization_epoch: int = Field(ge=1)
    policy_authorization_epoch: int = Field(ge=1)
    evaluation_id: str = Field(pattern=r"^perseval_[a-f0-9]{24}$")
    promotion_review_id: str = Field(pattern=r"^persreview_[a-f0-9]{24}$")
    mode: Literal["active"]
    baseline_strategy_id: str = Field(min_length=1, max_length=100)
    interest_support: Literal["neutral", "situational_trigger", "maintenance", "recurring_depth"]
    challenge_assumption: Literal[
        "grade_baseline_uncalibrated", "familiar", "stretch_with_help", "independent_transfer"
    ]
    applied_fields: list[Literal["interest_support", "challenge_assumption"]] = Field(
        min_length=2, max_length=2
    )

    @model_validator(mode="after")
    def exact_application_scope(self) -> PedagogyPersonalizationTrace:
        if self.applied_fields != ["interest_support", "challenge_assumption"]:
            raise ValueError("personalization trace contains an unapproved application scope")
        if self.concept_ids != [item.id for item in self.concept_refs]:
            raise ValueError("personalization trace concept versions do not match its IDs")
        if self.concept_refs != sorted(self.concept_refs, key=lambda item: (item.id, item.version)):
            raise ValueError("personalization trace concept references must be canonical")
        return self


class PedagogyTrace(StrictModel):
    """Compact code-owned lineage copied into child-facing release specifications."""

    plan_id: str = Field(pattern=r"^ped_[a-f0-9]{20}$")
    plan_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    question_intent: str = Field(min_length=1, max_length=80)
    learning_goal: str = Field(min_length=1, max_length=300)
    concept_ids: list[str] = Field(min_length=1, max_length=3)
    primary_strategy_id: str = Field(min_length=1, max_length=100)
    task_template_id: str = Field(min_length=1, max_length=120)
    task_template_version: str = Field(min_length=1, max_length=32)
    supporting_strategy_id: str | None = Field(default=None, max_length=100)
    domain_overlay_id: str = Field(min_length=1, max_length=100)
    interest_support: Literal["neutral", "situational_trigger", "maintenance", "recurring_depth"]
    challenge_assumption: Literal[
        "grade_baseline_uncalibrated", "familiar", "stretch_with_help", "independent_transfer"
    ]
    personalization: PedagogyPersonalizationTrace | None = None
    representations: list[str] = Field(min_length=1, max_length=3)
    scaffold: dict[str, Any]
    evidence_collection: Literal["not_collected", "future_observation_only"]
    evidence_template_id: str = Field(min_length=1, max_length=100)
    evidence_template_version: int = Field(ge=1)
    rubric_id: str | None = Field(default=None, max_length=120)
    rubric_version: int | None = Field(default=None, ge=1)
    closure: dict[str, Any]
    accessibility: dict[str, Any]
    safety: dict[str, Any]
    family_load: dict[str, Any]
    registry_version: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def personalization_matches_calibration(self) -> PedagogyTrace:
        if self.personalization is None:
            if self.interest_support != "neutral" or self.challenge_assumption != "grade_baseline_uncalibrated":
                raise ValueError("calibrated pedagogy trace requires exact personalization lineage")
        elif (
            self.interest_support != self.personalization.interest_support
            or self.challenge_assumption != self.personalization.challenge_assumption
            or self.primary_strategy_id != self.personalization.baseline_strategy_id
            or self.question_intent != self.personalization.question_intent
            or self.concept_ids != self.personalization.concept_ids
        ):
            raise ValueError("pedagogy trace does not match its personalization lineage")
        return self


class PictureStroke(StrictModel):
    """A small vector drawing, in a 100 by 100 square. No executable markup."""

    kind: Literal["ellipse", "rect", "polygon", "line"]
    points: list[tuple[Annotated[float, Field(ge=0, le=100)], Annotated[float, Field(ge=0, le=100)]]] = Field(min_length=2, max_length=32)
    fill: str = Field(default="#FFFFFF", pattern=r"^(?:#[0-9a-fA-F]{6}|none)$")
    stroke: str = Field(default="#25364A", pattern=r"^#[0-9a-fA-F]{6}$")


class PictureAnnotation(StrictModel):
    text: str = Field(min_length=1, max_length=30)
    position: tuple[Annotated[float, Field(ge=0, le=100)], Annotated[float, Field(ge=0, le=100)]]
    target: tuple[Annotated[float, Field(ge=0, le=100)], Annotated[float, Field(ge=0, le=100)]]


class PictureSymbol(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    description: str = Field(min_length=1, max_length=500)
    strokes: list[PictureStroke] = Field(default_factory=list, max_length=16)
    illustration: GeneratedIllustrationAsset | None = None
    annotations: list[PictureAnnotation] = Field(default_factory=list, max_length=4)


class PictureLabelPlacement(StrictModel):
    symbol_id: str
    annotations: list[PictureAnnotation] = Field(min_length=1, max_length=4)


class PictureLabelLayout(StrictModel):
    placements: list[PictureLabelPlacement] = Field(min_length=1, max_length=10)


class PictureItem(StrictModel):
    symbol_id: str
    count: int = Field(default=1, ge=1, le=20)


class PictureCell(StrictModel):
    column_span: int = Field(default=1, ge=1, le=3)
    label: str = Field(default="", max_length=120)
    action: str = Field(default="", max_length=180)
    items: list[PictureItem] = Field(default_factory=list, max_length=4)
    response_space: Literal["none", "draw", "mark"] = "none"


class PicturePage(StrictModel):
    """Topic-independent pictures, counted objects, cards, and response spaces."""

    directions: str = Field(min_length=1, max_length=320)
    display: Literal["paper", "screen"] = "paper"
    layout_locked: bool = False
    columns: int = Field(default=2, ge=1, le=3)
    picture_scale: float = Field(default=1, ge=0.5, le=1)
    show_borders: bool = True
    symbols: list[PictureSymbol] = Field(default_factory=list, max_length=10)
    cells: list[PictureCell] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def usable_references(self) -> PicturePage:
        ids = [symbol.id for symbol in self.symbols]
        if len(ids) != len(set(ids)):
            raise ValueError("picture symbols must have unique IDs")
        for cell in self.cells:
            if any(item.symbol_id not in ids for item in cell.items):
                raise ValueError("picture cell refers to an undefined symbol")
            if sum(item.count for item in cell.items) > 20:
                raise ValueError("picture cell has too many objects for a child to see")
        return self


class LearningArtifactBase(StrictModel):
    title: str = Field(min_length=1, max_length=140)
    target_grade: str = Field(min_length=1, max_length=40)
    learning_objective: str = Field(min_length=1, max_length=300)
    estimated_minutes: int = Field(ge=3, le=60)
    parent_effort: Literal["very_low", "low", "medium"] = "low"
    trust_tier: Literal["A", "B"] = "B"
    story_theme: str = Field(default="curiosity mission", max_length=120)
    accessibility_text: str = Field(min_length=1, max_length=600)
    source_event_id: str | None = Field(default=None, max_length=120)
    source_refs: list[str] = Field(default_factory=list, max_length=8)
    pedagogy: PedagogyTrace | None = None
    picture_page: PicturePage | None = None


class WorksheetSpec(LearningArtifactBase):
    artifact_type: Literal["worksheet"] = "worksheet"
    archetype: WorksheetArchetype = "legacy_mixed"
    theme: WorksheetTheme = "storybook"
    print_mode: Literal["color", "ink_saver"] = "color"
    subject_labels: list[str] = Field(default_factory=list, max_length=3)
    category_labels: list[str] = Field(default_factory=list, max_length=3)
    diagram_kind: WorksheetDiagramKind = "auto"
    piece_shape: WorksheetPieceShape = "auto"
    directions: str = Field(min_length=1, max_length=300)
    tasks: list[WorksheetTask] = Field(min_length=1, max_length=8)
    celebration: str = Field(default="Mission complete!", max_length=100)
    suggested_research_query: str | None = Field(default=None, max_length=180)
    researched_image: ResearchedImageAsset | None = None

    @model_validator(mode="after")
    def legacy_body_has_tasks(self) -> WorksheetSpec:
        if self.picture_page is None and len(self.tasks) < 2:
            raise ValueError("a legacy worksheet body requires at least two tasks")
        return self

    @field_validator("suggested_research_query")
    @classmethod
    def safe_public_research_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ImageResearchQuery(query=value, purpose="observe").query


class ReferencePanel(StrictModel):
    heading: str = Field(min_length=1, max_length=50)
    cue: Literal["circle", "oval", "spiral", "segments", "arrow", "box", "none"] = "box"
    segment_labels: list[str] = Field(default_factory=list, max_length=4)
    facts: list[str] = Field(min_length=1, max_length=4)


class ReferenceSpec(LearningArtifactBase):
    artifact_type: Literal["reference"] = "reference"
    layout: Literal["comparison", "sequence", "parts", "observation"]
    directions: str = Field(min_length=1, max_length=220)
    panels: list[ReferencePanel] = Field(min_length=2, max_length=5)
    footer_prompt: str = Field(default="What do you notice?", min_length=1, max_length=160)


class ActivitySpec(LearningArtifactBase):
    artifact_type: Literal["activity"] = "activity"
    mission: str = Field(min_length=1, max_length=300)
    materials: list[str] = Field(min_length=1, max_length=10)
    substitutions: list[str] = Field(default_factory=list, max_length=6)
    setup: list[str] = Field(default_factory=list, max_length=4)
    steps: list[str] = Field(min_length=3, max_length=8)
    observation_prompts: list[str] = Field(min_length=1, max_length=4)
    variations: list[str] = Field(default_factory=list, max_length=3)
    safety: str = Field(default="Use ordinary care and grown-up help when needed.", max_length=300)
    cleanup: str = Field(min_length=1, max_length=220)
    printable: PrintablePlan | None = None

    @model_validator(mode="after")
    def ordinary_materials_only(self) -> ActivitySpec:
        _reject_3d_printing([self.title, self.mission, *self.materials, *self.steps, *self.variations])
        return self


class ChallengeSpec(LearningArtifactBase):
    artifact_type: Literal["challenge"] = "challenge"
    scenario: str = Field(min_length=1, max_length=240)
    goal: str = Field(min_length=1, max_length=240)
    constraints: list[str] = Field(min_length=1, max_length=5)
    materials: list[str] = Field(default_factory=list, max_length=8)
    steps: list[str] = Field(min_length=1, max_length=7)
    hints: list[str] = Field(min_length=2, max_length=4)
    evidence_of_completion: str = Field(min_length=1, max_length=240)
    evidence_layout: Literal["trial_grid", "design_plan", "writing_lines"] = "trial_grid"
    evidence_rows: list[str] = Field(default_factory=list, max_length=4)
    reflection: str = Field(min_length=1, max_length=240)
    visual_brief: ChallengeVisualBrief | None = None
    illustration: GeneratedIllustrationAsset | None = None

    @model_validator(mode="after")
    def ordinary_challenge_only(self) -> ChallengeSpec:
        if self.picture_page is None and len(self.steps) < 3:
            raise ValueError("text challenges require at least three steps")
        _reject_3d_printing(
            [
                self.title,
                self.scenario,
                self.goal,
                *self.constraints,
                *self.materials,
                *self.steps,
                *self.evidence_rows,
            ]
        )
        return self


LearningArtifactSpec = Annotated[
    WorksheetSpec | ReferenceSpec | ActivitySpec | ChallengeSpec,
    Field(discriminator="artifact_type"),
]
LEARNING_ARTIFACT_ADAPTER = TypeAdapter(LearningArtifactSpec)


class ScheduleProposal(StrictModel):
    workflow: Literal["weekly_parent_checkin"] = "weekly_parent_checkin"
    child_id: str | None = Field(default=None, max_length=120)
    weekday: Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    local_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    timezone: str = Field(min_length=1, max_length=80)
    channel_id: str = Field(min_length=1, max_length=120)
    binding_id: str = Field(min_length=1, max_length=120)
    checkin_message: str = Field(
        default="Weekly curiosity check-in: what clicked, what got stuck, or what should we follow next?",
        min_length=1,
        max_length=600,
    )
    maximum_frequency_days: int = Field(default=7, ge=7, le=31)
    catch_up: Literal["one", "skip"] = "one"
