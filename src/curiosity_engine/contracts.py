from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

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
        elif self.kind == GraphMutationKind.SET_KNOWLEDGE_STATE:
            if self.node_id is None or self.knowledge_state is None:
                raise ValueError("set_knowledge_state requires node_id and knowledge_state")
        return self


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
    critique_rounds: list[list[CriticResult]] = Field(default_factory=list, max_length=6)
    revision_rounds: int = Field(default=0, ge=0, le=10)
    recovery_strategy: Literal["rebuild_from_scratch"] | None = None
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
    message: str | None = Field(default=None, max_length=2_000)
    tool_calls: list[ParentAgentToolCall] = Field(default_factory=list, max_length=3)
    interaction: InteractionPlan | None = None
    done: bool = True

    @model_validator(mode="after")
    def contains_an_action(self) -> ParentAgentTurn:
        if not self.message and not self.tool_calls and not self.interaction:
            raise ValueError("parent agent turn must communicate or act")
        return self


class ImageContextOutput(StrictModel):
    """Bounded evidence extracted from one parent-shared image."""

    summary: str = Field(min_length=1, max_length=700)
    visible_details: list[str] = Field(default_factory=list, max_length=10)
    possible_play_threads: list[str] = Field(default_factory=list, max_length=5)
    uncertainties: list[str] = Field(default_factory=list, max_length=6)


class WorksheetTask(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,39}$")
    kind: Literal["match", "sort", "sequence", "circle", "label", "draw", "predict", "measure", "record", "short_response"]
    instruction: str = Field(min_length=1, max_length=220)
    choices: list[str] = Field(default_factory=list, max_length=8)
    answer: str | list[str] | None = None
    response_lines: int = Field(default=1, ge=0, le=6)

    @model_validator(mode="after")
    def matches_mechanic(self) -> WorksheetTask:
        if self.kind in {"match", "sort", "sequence", "circle"} and len(self.choices) < 2:
            raise ValueError(f"{self.kind} tasks require at least two choices")
        if self.kind in {"draw", "label"} and self.response_lines:
            self.response_lines = 0
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


class WorksheetSpec(LearningArtifactBase):
    artifact_type: Literal["worksheet"] = "worksheet"
    directions: str = Field(min_length=1, max_length=300)
    tasks: list[WorksheetTask] = Field(min_length=2, max_length=8)
    celebration: str = Field(default="Mission complete!", max_length=100)


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
    steps: list[str] = Field(min_length=3, max_length=7)
    hints: list[str] = Field(min_length=2, max_length=4)
    evidence_of_completion: str = Field(min_length=1, max_length=240)
    evidence_rows: list[str] = Field(default_factory=list, max_length=4)
    reflection: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def ordinary_challenge_only(self) -> ChallengeSpec:
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


LearningArtifactSpec = Annotated[WorksheetSpec | ActivitySpec | ChallengeSpec, Field(discriminator="artifact_type")]
LEARNING_ARTIFACT_ADAPTER = TypeAdapter(LearningArtifactSpec)


class ScheduleProposal(StrictModel):
    workflow: Literal["weekly_parent_checkin"] = "weekly_parent_checkin"
    child_id: str | None = Field(default=None, max_length=120)
    weekday: Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    local_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    timezone: str = Field(min_length=1, max_length=80)
    channel_id: str = Field(min_length=1, max_length=120)
    binding_id: str = Field(min_length=1, max_length=120)
    maximum_frequency_days: int = Field(default=7, ge=7, le=31)
    catch_up: Literal["one", "skip"] = "one"
