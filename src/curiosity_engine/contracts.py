from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    SET_KNOWLEDGE_STATE = "set_knowledge_state"


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
    node_id: int | None = None
    knowledge_state: Literal["unseen", "exposed", "emerging", "demonstrated", "unknown"] | None = None
    evidence_ids: list[int] = Field(default_factory=list)


class ActionType(StrEnum):
    PROPOSE_ARTIFACT = "propose_artifact"


class ActionRequest(StrictModel):
    type: ActionType
    payload: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1, max_length=1_000)


class PhysicalExtension(StrictModel):
    title: str = Field(min_length=1, max_length=160)
    instructions: list[str] = Field(min_length=1, max_length=6)
    materials: list[str] = Field(default_factory=list, max_length=8)
    parent_effort: Literal["very_low", "low", "medium"] = "low"

    @model_validator(mode="after")
    def uses_ordinary_materials(self) -> PhysicalExtension:
        _reject_3d_printing([self.title, *self.instructions, *self.materials])
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
        "question",
        "idea",
    ]


class VisualIntent(StrictModel):
    """A semantic request for one child-facing visual; code still owns whether and how it renders."""

    kind: Literal["comparison_cards", "activity_sequence", "decorative_illustration"]
    purpose: Literal["compare", "sequence", "notice", "imagine"]
    knowledge_role: Literal["decorative", "supportive", "instructional"]
    title: str = Field(min_length=1, max_length=120)
    pedagogical_value: str = Field(min_length=1, max_length=300)
    caption: str = Field(min_length=1, max_length=280)
    alt_text: str = Field(min_length=12, max_length=1_000)
    subject: str | None = Field(default=None, max_length=160)
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
        if self.kind == "comparison_cards" and not self.not_to_scale:
            raise ValueError("MVP comparison cards must be explicitly marked not to scale")
        return self


class PullThreadOutput(StrictModel):
    hook: str = Field(min_length=1, max_length=500)
    show: str = Field(min_length=1, max_length=1_000)
    ask: str = Field(min_length=1, max_length=500)
    nugget: str = Field(min_length=1, max_length=500)
    next_possible_concepts: list[str] = Field(default_factory=list, max_length=6)
    physical_extension: PhysicalExtension | None = None
    visual: VisualIntent | None = None
    graph_updates: list[GraphMutation] = Field(default_factory=list, max_length=12)
    actions: list[ActionRequest] = Field(default_factory=list, max_length=4)
    resource_refs: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("ask")
    @classmethod
    def ask_is_a_question(cls, value: str) -> str:
        if "?" not in value:
            raise ValueError("ask must be phrased as a question")
        return value


class InterestSignalOutput(StrictModel):
    summary: str = Field(min_length=1, max_length=500)
    graph_updates: list[GraphMutation] = Field(default_factory=list, max_length=12)
    actions: list[ActionRequest] = Field(default_factory=list, max_length=2)


class FeedbackOutput(StrictModel):
    acknowledged: str = Field(min_length=1, max_length=500)
    graph_updates: list[GraphMutation] = Field(default_factory=list, max_length=8)
    actions: list[ActionRequest] = Field(default_factory=list, max_length=2)


class GenericOutput(StrictModel):
    summary: str = Field(min_length=1, max_length=1_000)
    graph_updates: list[GraphMutation] = Field(default_factory=list, max_length=8)
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
    revision_rounds: int = Field(default=0, ge=0, le=5)
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
    experience_id: str | None = None
    artifact_id: str | None = None
    child_id: str
    outcome: Literal["loved", "engaged", "neutral", "too_easy", "too_hard", "not_used", "disliked"]
    note: str | None = Field(default=None, max_length=2_000)
    source: str = "parent"


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
