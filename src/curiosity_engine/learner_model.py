from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import ObservationCandidate
from .db import connect, init_db, jdump, jload, utcnow
from .deletion_ledger import managed_data_lifecycle, reconcile_database_deletions

KNOWLEDGE_RULE_ID = "ce.knowledge.evidence"
KNOWLEDGE_RULE_VERSION = "1.0.0"
INTEREST_RULE_ID = "ce.interest.four-phase"
INTEREST_RULE_VERSION = "1.0.0"
PROJECTOR_VERSION = "phase1-v1"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
_CORRECTABLE_FIELDS = {"result", "assistance_level"}
_ASSISTANCE_ORDER = {
    "none": 0,
    "clarifying_prompt": 1,
    "general_prompt": 2,
    "hint": 3,
    "worked_example": 4,
    "joint_completion": 5,
    "direct_answer": 6,
    "unknown": 7,
}


class EvidenceConflict(ValueError):
    """An immutable identifier was reused with a different definition or payload."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ConceptRef(StrictModel):
    concept_id: str
    version: int = Field(default=1, gt=0)
    role: Literal["target", "prerequisite", "supporting"] = "target"

    @field_validator("concept_id")
    @classmethod
    def identifier_is_bounded(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("concept_id must be a bounded stable identifier")
        return value


class ConceptDefinition(StrictModel):
    concept_id: str
    version: int = Field(default=1, gt=0)
    label: str = Field(min_length=1, max_length=160)
    domain: str = Field(min_length=1, max_length=80)
    grade_band: str | None = Field(default=None, max_length=40)
    description: str = Field(default="", max_length=1_000)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    framework: str = Field(default="household", min_length=1, max_length=120)
    source_uri: str | None = Field(default=None, max_length=500)

    @field_validator("concept_id")
    @classmethod
    def identifier_is_bounded(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("concept_id must be a bounded stable identifier")
        return value

    @field_validator("aliases")
    @classmethod
    def aliases_are_useful(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if any(len(value) > 160 for value in cleaned):
            raise ValueError("concept aliases are limited to 160 characters")
        return list(dict.fromkeys(cleaned))


class ConceptRelationDefinition(StrictModel):
    relation_id: str
    source: ConceptRef
    relation: Literal["part_of", "prerequisite_of", "contrasts_with", "applies_in"]
    target: ConceptRef
    provenance: dict[str, Any] = Field(default_factory=dict)
    relation_version: str = Field(min_length=1, max_length=80)

    @field_validator("relation_id")
    @classmethod
    def identifier_is_bounded(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("relation_id must be a bounded stable identifier")
        return value


class RubricCriterion(StrictModel):
    criterion_id: str
    concept_id: str
    concept_version: int = Field(default=1, gt=0)
    description: str = Field(min_length=1, max_length=500)
    required: bool = True
    weight: float = Field(default=1.0, gt=0, le=10)

    @field_validator("criterion_id", "concept_id")
    @classmethod
    def identifiers_are_bounded(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("rubric criterion identifiers must be bounded and stable")
        return value


class RubricDefinition(StrictModel):
    rubric_id: str
    version: int = Field(default=1, gt=0)
    label: str = Field(min_length=1, max_length=160)
    criteria: list[RubricCriterion] = Field(min_length=1, max_length=40)
    evidence_rule_id: str = Field(default=KNOWLEDGE_RULE_ID, min_length=1, max_length=160)
    evidence_rule_version: str = Field(default=KNOWLEDGE_RULE_VERSION, min_length=1, max_length=80)

    @field_validator("rubric_id")
    @classmethod
    def identifier_is_bounded(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("rubric_id must be a bounded stable identifier")
        return value

    @model_validator(mode="after")
    def criteria_are_unique(self) -> RubricDefinition:
        identifiers = [criterion.criterion_id for criterion in self.criteria]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("rubric criterion IDs must be unique")
        return self


class LearningTaskDefinition(StrictModel):
    task_id: str
    version: int = Field(default=1, gt=0)
    title: str = Field(min_length=1, max_length=200)
    learning_goal: str = Field(min_length=1, max_length=1_000)
    strategy: str = Field(min_length=1, max_length=100)
    difficulty: Literal["trivial", "familiar", "stretch", "advanced", "unknown"]
    difficulty_provenance: Literal["reviewed_task", "parent_selected", "estimated", "unknown"]
    modalities: list[
        Literal[
            "spoken",
            "written",
            "drawn",
            "gesture_or_manipulative",
            "selection",
            "demonstration",
            "symbolic",
            "mixed",
        ]
    ] = Field(default_factory=list)
    context_key: str = Field(min_length=1, max_length=160)
    representation_key: str = Field(min_length=1, max_length=160)
    concepts: list[ConceptRef] = Field(min_length=1, max_length=20)
    rubric_id: str
    rubric_version: int = Field(gt=0)

    @field_validator("task_id", "rubric_id")
    @classmethod
    def identifier_is_bounded(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("task and rubric IDs must be bounded stable identifiers")
        return value

    @model_validator(mode="after")
    def has_target_and_complete_rubric(self) -> LearningTaskDefinition:
        if not any(item.role == "target" for item in self.concepts):
            raise ValueError("learning task must declare at least one target concept")
        keys = {(item.concept_id, item.version, item.role) for item in self.concepts}
        if len(keys) != len(self.concepts):
            raise ValueError("learning task concept links must be unique")
        return self


class LearningEpisodePlanInput(StrictModel):
    """A reviewed task placement whose lineage is fixed before evidence is recorded."""

    plan_id: str
    child_id: str
    source_event_id: str
    task_id: str
    task_version: int = Field(default=1, gt=0)
    context_key: str | None = Field(default=None, min_length=1, max_length=160)
    representation_key: str | None = Field(default=None, min_length=1, max_length=160)
    transfer_from_plan_id: str | None = None

    @field_validator(
        "plan_id",
        "child_id",
        "source_event_id",
        "task_id",
        "context_key",
        "representation_key",
        "transfer_from_plan_id",
    )
    @classmethod
    def identifiers_are_bounded(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTIFIER.fullmatch(value):
            raise ValueError("learning plan identifiers must be bounded and stable")
        return value

    @model_validator(mode="after")
    def transfer_is_not_recursive(self) -> LearningEpisodePlanInput:
        if self.transfer_from_plan_id == self.plan_id:
            raise ValueError("a learning plan cannot transfer from itself")
        return self


class AssistanceEventInput(StrictModel):
    level: Literal[
        "clarifying_prompt",
        "general_prompt",
        "hint",
        "worked_example",
        "joint_completion",
        "direct_answer",
    ]
    provider: Literal["parent", "engine", "other"]
    occurred_at: datetime | None = None


class CriterionResultInput(StrictModel):
    criterion_id: str
    outcome: Literal["not_observed", "correct", "partial", "incorrect"]
    score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("criterion_id")
    @classmethod
    def identifier_is_bounded(cls, value: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ValueError("criterion_id must be a bounded stable identifier")
        return value


class LearningOpportunityInput(StrictModel):
    opportunity_id: str
    child_id: str
    event_id: str
    plan_id: str
    attempt_number: int = Field(default=1, gt=0)
    actor: Literal["child", "parent_report", "system"]
    initiation: Literal["child", "parent", "engine", "unknown"] = "unknown"
    response_mode: Literal[
        "spoken",
        "written",
        "drawn",
        "gesture_or_manipulative",
        "selection",
        "demonstration",
        "symbolic",
        "mixed",
        "parent_report",
    ]
    assistance_level: Literal[
        "none",
        "clarifying_prompt",
        "general_prompt",
        "hint",
        "worked_example",
        "joint_completion",
        "direct_answer",
        "unknown",
    ]
    assistance_history: list[AssistanceEventInput] = Field(default_factory=list, max_length=20)
    criterion_results: list[CriterionResultInput] = Field(min_length=1, max_length=40)
    scorer: str = Field(min_length=1, max_length=120)
    scorer_version: str = Field(min_length=1, max_length=80)
    source_type: Literal[
        "direct_observation", "parent_report", "system_generated", "model_annotation", "output_feedback"
    ]
    child_continued: bool = False
    child_new_question: bool = False
    retention_class: Literal["transient", "learning_record", "parent_confirmed"] = "learning_record"

    @field_validator("opportunity_id", "child_id", "event_id", "plan_id")
    @classmethod
    def identifier_is_bounded(cls, value: str | None) -> str | None:
        if value is not None and not _IDENTIFIER.fullmatch(value):
            raise ValueError("opportunity identifiers must be bounded stable identifiers")
        return value

    @model_validator(mode="after")
    def source_and_actor_agree(self) -> LearningOpportunityInput:
        parent_report_fields = (
            self.actor == "parent_report",
            self.source_type == "parent_report",
            self.response_mode == "parent_report",
        )
        if any(parent_report_fields) and not all(parent_report_fields):
            raise ValueError("parent_report actor, source, and response mode must occur together")
        if self.actor == "system" and self.source_type not in {
            "system_generated",
            "model_annotation",
            "output_feedback",
        }:
            raise ValueError("system actor requires a non-observational system source")
        identifiers = [result.criterion_id for result in self.criterion_results]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("criterion results must use unique criterion IDs")
        return self


def _payload_hash(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return sha256(jdump(value).encode()).hexdigest()


def _normalized_alias(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _validate_existing(row: sqlite3.Row | None, expected_hash: str, kind: str, identifier: str) -> bool:
    if row is None:
        return False
    if row["definition_hash"] != expected_hash:
        raise EvidenceConflict(f"{kind} {identifier!r} already has a different immutable definition")
    return True


def _collection_enabled_from_conn(conn: sqlite3.Connection, child_id: str) -> bool:
    row = conn.execute(
        "SELECT collection_enabled FROM learner_model_controls WHERE child_id=?",
        (child_id,),
    ).fetchone()
    return row is None or bool(row["collection_enabled"])


def register_concept(db_path: str | Path, definition: ConceptDefinition | dict[str, Any]) -> dict[str, Any]:
    item = definition if isinstance(definition, ConceptDefinition) else ConceptDefinition.model_validate(definition)
    init_db(db_path)
    digest = _payload_hash(item)
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT definition_hash FROM concepts WHERE concept_id=? AND version=?",
            (item.concept_id, item.version),
        ).fetchone()
        if _validate_existing(row, digest, "concept", f"{item.concept_id}@{item.version}"):
            return {"concept_id": item.concept_id, "version": item.version, "duplicate": True}
        conn.execute(
            """INSERT INTO concepts(
               concept_id,version,label,domain,grade_band,description,framework,source_uri,definition_hash,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                item.concept_id,
                item.version,
                item.label,
                item.domain,
                item.grade_band,
                item.description,
                item.framework,
                item.source_uri,
                digest,
                now,
            ),
        )
        for alias in [item.label, *item.aliases]:
            normalized = _normalized_alias(alias)
            if normalized:
                conn.execute(
                    """INSERT OR IGNORE INTO concept_aliases(
                       concept_id,concept_version,alias,normalized_alias,source,created_at
                       ) VALUES(?,?,?,?,?,?)""",
                    (item.concept_id, item.version, alias, normalized, "registered", now),
                )
    return {"concept_id": item.concept_id, "version": item.version, "duplicate": False}


def register_concept_relation(
    db_path: str | Path, definition: ConceptRelationDefinition | dict[str, Any]
) -> dict[str, Any]:
    item = (
        definition
        if isinstance(definition, ConceptRelationDefinition)
        else ConceptRelationDefinition.model_validate(definition)
    )
    init_db(db_path)
    digest = _payload_hash(item)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT definition_hash FROM concept_relations WHERE relation_id=?", (item.relation_id,)
        ).fetchone()
        if _validate_existing(row, digest, "concept relation", item.relation_id):
            return {"relation_id": item.relation_id, "duplicate": True}
        conn.execute(
            """INSERT INTO concept_relations(
               relation_id,source_concept_id,source_concept_version,relation,target_concept_id,
               target_concept_version,provenance_json,relation_version,definition_hash,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                item.relation_id,
                item.source.concept_id,
                item.source.version,
                item.relation,
                item.target.concept_id,
                item.target.version,
                jdump(item.provenance),
                item.relation_version,
                digest,
                utcnow(),
            ),
        )
    return {"relation_id": item.relation_id, "duplicate": False}


def register_rubric(db_path: str | Path, definition: RubricDefinition | dict[str, Any]) -> dict[str, Any]:
    item = definition if isinstance(definition, RubricDefinition) else RubricDefinition.model_validate(definition)
    init_db(db_path)
    digest = _payload_hash(item)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT definition_hash FROM rubrics WHERE rubric_id=? AND version=?",
            (item.rubric_id, item.version),
        ).fetchone()
        if _validate_existing(row, digest, "rubric", f"{item.rubric_id}@{item.version}"):
            return {"rubric_id": item.rubric_id, "version": item.version, "duplicate": True}
        for criterion in item.criteria:
            concept = conn.execute(
                "SELECT 1 FROM concepts WHERE concept_id=? AND version=?",
                (criterion.concept_id, criterion.concept_version),
            ).fetchone()
            if not concept:
                raise ValueError(
                    f"rubric criterion references unknown concept {criterion.concept_id}@{criterion.concept_version}"
                )
        conn.execute(
            """INSERT INTO rubrics(
               rubric_id,version,label,criteria_json,evidence_rule_id,evidence_rule_version,definition_hash,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                item.rubric_id,
                item.version,
                item.label,
                jdump([criterion.model_dump(mode="json") for criterion in item.criteria]),
                item.evidence_rule_id,
                item.evidence_rule_version,
                digest,
                utcnow(),
            ),
        )
    return {"rubric_id": item.rubric_id, "version": item.version, "duplicate": False}


def register_learning_task(db_path: str | Path, definition: LearningTaskDefinition | dict[str, Any]) -> dict[str, Any]:
    item = (
        definition
        if isinstance(definition, LearningTaskDefinition)
        else LearningTaskDefinition.model_validate(definition)
    )
    init_db(db_path)
    digest = _payload_hash(item)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT definition_hash FROM learning_tasks WHERE task_id=? AND version=?",
            (item.task_id, item.version),
        ).fetchone()
        if _validate_existing(row, digest, "learning task", f"{item.task_id}@{item.version}"):
            return {"task_id": item.task_id, "version": item.version, "duplicate": True}
        rubric = conn.execute(
            """SELECT criteria_json,evidence_rule_id,evidence_rule_version FROM rubrics
               WHERE rubric_id=? AND version=?""",
            (item.rubric_id, item.rubric_version),
        ).fetchone()
        if not rubric:
            raise ValueError("learning task references an unknown rubric version")
        if (rubric["evidence_rule_id"], rubric["evidence_rule_version"]) != (
            KNOWLEDGE_RULE_ID,
            KNOWLEDGE_RULE_VERSION,
        ):
            raise ValueError("learning task rubric uses an unsupported evidence rule version")
        for concept in item.concepts:
            known = conn.execute(
                "SELECT 1 FROM concepts WHERE concept_id=? AND version=?",
                (concept.concept_id, concept.version),
            ).fetchone()
            if not known:
                raise ValueError(f"learning task references unknown concept {concept.concept_id}@{concept.version}")
        task_concepts = {(concept.concept_id, concept.version) for concept in item.concepts}
        target_concepts = {
            (concept.concept_id, concept.version) for concept in item.concepts if concept.role == "target"
        }
        rubric_criteria = jload(rubric["criteria_json"], [])
        criterion_concepts = {
            (str(criterion["concept_id"]), int(criterion["concept_version"])) for criterion in rubric_criteria
        }
        required_criterion_concepts = {
            (str(criterion["concept_id"]), int(criterion["concept_version"]))
            for criterion in rubric_criteria
            if bool(criterion["required"])
        }
        if not target_concepts <= required_criterion_concepts:
            raise ValueError("learning task rubric must require a score for every target concept")
        if not criterion_concepts <= task_concepts:
            raise ValueError("rubric criteria must reference concepts declared by the learning task")
        conn.execute(
            """INSERT INTO learning_tasks(
               task_id,version,title,learning_goal,strategy,difficulty,difficulty_provenance,modalities_json,
               context_key,representation_key,rubric_id,rubric_version,definition_hash,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item.task_id,
                item.version,
                item.title,
                item.learning_goal,
                item.strategy,
                item.difficulty,
                item.difficulty_provenance,
                jdump(item.modalities),
                item.context_key,
                item.representation_key,
                item.rubric_id,
                item.rubric_version,
                digest,
                utcnow(),
            ),
        )
        conn.executemany(
            """INSERT INTO learning_task_concepts(
               task_id,task_version,concept_id,concept_version,role
               ) VALUES(?,?,?,?,?)""",
            [
                (item.task_id, item.version, concept.concept_id, concept.version, concept.role)
                for concept in item.concepts
            ],
        )
    return {"task_id": item.task_id, "version": item.version, "duplicate": False}


def register_learning_plan(db_path: str | Path, plan: LearningEpisodePlanInput | dict[str, Any]) -> dict[str, Any]:
    """Register one immutable, code-reviewed placement of a task in an episode."""

    item = plan if isinstance(plan, LearningEpisodePlanInput) else LearningEpisodePlanInput.model_validate(plan)
    init_db(db_path)
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        event = conn.execute(
            """SELECT e.child_id,m.episode_id,m.learning_scope,m.independence_status
               FROM events e JOIN episode_memberships m ON m.event_id=e.id
               WHERE e.id=?""",
            (item.source_event_id,),
        ).fetchone()
        if not event:
            raise ValueError("learning plan requires a source event with code-resolved episode lineage")
        if event["child_id"] != item.child_id:
            raise ValueError("learning plan child does not match its source event")
        if event["learning_scope"] != "family_signal" or event["independence_status"] in {
            "diagnostic",
            "system",
            "excluded",
        }:
            raise ValueError("learning plan source event is not eligible family learning context")
        task = conn.execute(
            """SELECT context_key,representation_key FROM learning_tasks
               WHERE task_id=? AND version=?""",
            (item.task_id, item.task_version),
        ).fetchone()
        if not task:
            raise ValueError("learning plan references an unknown immutable task version")
        context_key = item.context_key or str(task["context_key"])
        representation_key = item.representation_key or str(task["representation_key"])
        attempt_group_id = "attempt_" + sha256(f"{item.child_id}|{item.plan_id}".encode()).hexdigest()[:24]
        normalized = {
            **item.model_dump(mode="json"),
            "context_key": context_key,
            "representation_key": representation_key,
            "attempt_group_id": attempt_group_id,
        }
        digest = _payload_hash(normalized)
        existing = conn.execute(
            """SELECT plan_hash,episode_id,status FROM learning_episode_plans
               WHERE plan_id=?""",
            (item.plan_id,),
        ).fetchone()
        if existing:
            if existing["plan_hash"] != digest:
                raise EvidenceConflict(f"learning plan {item.plan_id!r} was already used for a different payload")
            return {
                "plan_id": item.plan_id,
                "episode_id": str(existing["episode_id"]),
                "status": str(existing["status"]),
                "duplicate": True,
            }

        if item.transfer_from_plan_id:
            source = conn.execute(
                """SELECT child_id,episode_id,task_id,task_version,context_key,representation_key,status
                   FROM learning_episode_plans WHERE plan_id=?""",
                (item.transfer_from_plan_id,),
            ).fetchone()
            if not source or source["child_id"] != item.child_id:
                raise ValueError("transfer source plan does not belong to this child")
            if source["status"] != "completed":
                raise ValueError("transfer source plan must have completed evidence")
            if source["episode_id"] == event["episode_id"]:
                raise ValueError("transfer plans must belong to distinct learning episodes")
            if source["context_key"] == context_key and source["representation_key"] == representation_key:
                raise ValueError("transfer plans must vary context or representation")
            shared_target = conn.execute(
                """SELECT 1
                   FROM learning_task_concepts source_concept
                   JOIN learning_task_concepts target_concept
                     ON target_concept.concept_id=source_concept.concept_id
                    AND target_concept.concept_version=source_concept.concept_version
                    AND target_concept.role='target'
                   WHERE source_concept.task_id=? AND source_concept.task_version=?
                     AND source_concept.role='target'
                     AND target_concept.task_id=? AND target_concept.task_version=? LIMIT 1""",
                (
                    source["task_id"],
                    source["task_version"],
                    item.task_id,
                    item.task_version,
                ),
            ).fetchone()
            if not shared_target:
                raise ValueError("transfer plans must share at least one target concept")

        conn.execute(
            """INSERT INTO learning_episode_plans(
               plan_id,child_id,episode_id,task_id,task_version,source_event_id,attempt_group_id,
               context_key,representation_key,transfer_from_plan_id,status,plan_hash,created_at,updated_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,'planned',?,?,?)""",
            (
                item.plan_id,
                item.child_id,
                str(event["episode_id"]),
                item.task_id,
                item.task_version,
                item.source_event_id,
                attempt_group_id,
                context_key,
                representation_key,
                item.transfer_from_plan_id,
                digest,
                now,
                now,
            ),
        )
    return {
        "plan_id": item.plan_id,
        "episode_id": str(event["episode_id"]),
        "status": "planned",
        "duplicate": False,
    }


def _strongest_assistance(item: LearningOpportunityInput) -> str:
    levels = [item.assistance_level, *(entry.level for entry in item.assistance_history)]
    return max(levels, key=_ASSISTANCE_ORDER.__getitem__)


def _classification(row: dict[str, Any]) -> tuple[str, str, str, int]:
    """Return stance, reliability, strength, eligibility for one effective evidence row."""

    construct = str(row["construct"])
    result = str(row["result"])
    source = str(row["source_type"])
    independence = str(row["independence_status"])
    assistance = str(row["assistance_level"])
    direct = source == "direct_observation"
    reported = source == "parent_report"
    reliability = "direct" if direct else "reported" if reported else "generated" if source != "unknown" else "unknown"
    if construct == "exposure":
        eligible = (direct or reported) and independence not in {"diagnostic", "system", "excluded"}
        return "supports" if eligible else "context_only", reliability, "weak" if eligible else "none", int(eligible)
    if construct == "interest":
        parent_confirmed = (
            reported
            and str(row["retention_class"]) == "parent_confirmed"
            and str(row["initiation"]) == "child"
            and independence not in {"diagnostic", "system", "excluded"}
        )
        eligible = parent_confirmed or (
            (direct or reported)
            and str(row["actor"]) in {"child", "parent_report"}
            and str(row["initiation"]) == "child"
            and independence == "eligible"
            and int(row["attempt_number"]) == 1
        )
        return (
            "supports" if eligible else "context_only",
            reliability,
            "moderate" if eligible else "none",
            int(eligible),
        )

    substantive = (
        direct
        and str(row["actor"]) == "child"
        and independence == "eligible"
        and int(row["attempt_number"]) == 1
        and not bool(row["is_trivial"])
        and str(row["difficulty"]) != "trivial"
        and str(row["difficulty_provenance"]) in {"reviewed_task", "parent_selected"}
        and assistance != "unknown"
        and result in {"correct", "partial", "incorrect"}
    )
    if result == "incorrect" and str(row["difficulty"]) == "advanced":
        substantive = False
    if not substantive:
        return "context_only", reliability, "none", 0
    if result == "incorrect":
        return "contradicts", reliability, "moderate", 1
    if result == "partial":
        return "supports", reliability, "weak", 1
    strength = "strong" if assistance == "none" else "moderate"
    return "supports", reliability, strength, 1


def _score_criteria(
    rubric_criteria: list[dict[str, Any]],
    criterion_results: list[dict[str, Any]],
    *,
    concept_id: str | None = None,
    concept_version: int | None = None,
) -> tuple[str, float | None, list[dict[str, Any]]]:
    definitions = {
        str(criterion["criterion_id"]): criterion
        for criterion in rubric_criteria
        if concept_id is None
        or (criterion["concept_id"] == concept_id and int(criterion["concept_version"]) == concept_version)
    }
    scored = [
        {
            "criterion_id": str(result["criterion_id"]),
            "outcome": str(result["outcome"]),
            "score": result.get("score"),
            "weight": float(definitions[str(result["criterion_id"])]["weight"]),
            "required": bool(definitions[str(result["criterion_id"])]["required"]),
        }
        for result in criterion_results
        if str(result["criterion_id"]) in definitions
    ]
    if not scored:
        return "not_observed", None, []
    if any(item["required"] and item["outcome"] == "not_observed" for item in scored):
        return "not_observed", None, scored
    observed = [item for item in scored if item["outcome"] != "not_observed"]
    if not observed:
        return "not_observed", None, scored
    outcomes = {str(item["outcome"]) for item in observed}
    if outcomes == {"correct"}:
        result = "correct"
    elif outcomes == {"incorrect"}:
        result = "incorrect"
    else:
        result = "partial"
    default_scores = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    total_weight = sum(float(item["weight"]) for item in observed)
    score = (
        sum(
            float(item["weight"])
            * (float(item["score"]) if item["score"] is not None else default_scores[str(item["outcome"])])
            for item in observed
        )
        / total_weight
    )
    return result, round(score, 4), scored


def _evidence_expiry(opportunity: dict[str, Any], construct: str) -> str | None:
    occurred = datetime.fromisoformat(str(opportunity["occurred_at"]).replace("Z", "+00:00"))
    if opportunity["retention_class"] == "transient":
        return (occurred + timedelta(days=30)).isoformat()
    if construct == "interest":
        if opportunity["retention_class"] == "parent_confirmed":
            return None
        return (occurred + timedelta(days=90)).isoformat()
    return None


def _insert_evidence(
    conn: sqlite3.Connection,
    opportunity: dict[str, Any],
    concept: sqlite3.Row,
    construct: str,
    observation_kind: str,
) -> str:
    evidence_id = (
        "ev2_"
        + sha256(
            f"{opportunity['opportunity_id']}|{concept['concept_id']}|{concept['concept_version']}|"
            f"{construct}|{observation_kind}".encode()
        ).hexdigest()[:24]
    )
    row = {**opportunity, "construct": construct}
    stance, reliability, strength, eligible = _classification(row)
    rule_id = INTEREST_RULE_ID if construct == "interest" else opportunity["evidence_rule_id"]
    rule_version = INTEREST_RULE_VERSION if construct == "interest" else opportunity["evidence_rule_version"]
    conn.execute(
        """INSERT OR IGNORE INTO evidence_records(
           evidence_id,child_id,opportunity_id,concept_id,concept_version,construct,observation_kind,stance,
           result,score,rubric_id,rubric_version,scored_criteria_json,source_type,source_reliability,
           evidence_strength,projection_eligible,episode_id,context_key,representation_key,assistance_level,
           attempt_number,is_transfer,is_trivial,rule_id,rule_version,retention_class,expires_at,created_at
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            evidence_id,
            opportunity["child_id"],
            opportunity["opportunity_id"],
            concept["concept_id"],
            concept["concept_version"],
            construct,
            observation_kind,
            stance,
            opportunity["result"],
            opportunity["score"],
            opportunity["rubric_id"],
            opportunity["rubric_version"],
            jdump(opportunity["scored_criteria"]),
            opportunity["source_type"],
            reliability,
            strength,
            eligible,
            opportunity["episode_id"],
            opportunity["context_key"],
            opportunity["representation_key"],
            opportunity["assistance_level"],
            opportunity["attempt_number"],
            opportunity["is_transfer"],
            opportunity["is_trivial"],
            rule_id,
            rule_version,
            opportunity["retention_class"],
            _evidence_expiry(opportunity, construct),
            opportunity["occurred_at"],
        ),
    )
    return evidence_id


def _compile_evidence(conn: sqlite3.Connection, opportunity: dict[str, Any]) -> list[str]:
    concepts = conn.execute(
        """SELECT concept_id,concept_version FROM learning_task_concepts
           WHERE task_id=? AND task_version=? AND role='target'
           ORDER BY concept_id,concept_version""",
        (opportunity["task_id"], opportunity["task_version"]),
    ).fetchall()
    evidence_ids: list[str] = []
    for concept in concepts:
        result, score, scored = _score_criteria(
            opportunity["rubric_criteria"],
            opportunity["criterion_results"],
            concept_id=str(concept["concept_id"]),
            concept_version=int(concept["concept_version"]),
        )
        concept_opportunity = {
            **opportunity,
            "result": result,
            "score": score,
            "scored_criteria": scored,
        }
        evidence_ids.append(_insert_evidence(conn, concept_opportunity, concept, "exposure", "observed_opportunity"))
        if result != "not_observed":
            evidence_ids.append(
                _insert_evidence(conn, concept_opportunity, concept, "knowledge_performance", "rubric_scored_response")
            )
        if opportunity["initiation"] == "child" or opportunity["child_continued"] or opportunity["child_new_question"]:
            evidence_ids.append(
                _insert_evidence(conn, concept_opportunity, concept, "interest", "child_curiosity_signal")
            )
    return evidence_ids


def record_learning_opportunity(
    db_path: str | Path, observation: LearningOpportunityInput | dict[str, Any]
) -> dict[str, Any]:
    """Bind an observation to its reviewed plan and rebuild deterministic projections."""

    item = (
        observation
        if isinstance(observation, LearningOpportunityInput)
        else LearningOpportunityInput.model_validate(observation)
    )
    init_db(db_path)
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing_opportunity = conn.execute(
            "SELECT 1 FROM learning_opportunities WHERE opportunity_id=? AND child_id=?",
            (item.opportunity_id, item.child_id),
        ).fetchone()
        if not existing_opportunity and not _collection_enabled_from_conn(conn, item.child_id):
            raise ValueError("learner evidence collection is paused for this child")
        event = conn.execute(
            """SELECT e.id,e.child_id,e.created_at,m.episode_id,m.independence_status
               FROM events e JOIN episode_memberships m ON m.event_id=e.id
               WHERE e.id=?""",
            (item.event_id,),
        ).fetchone()
        if not event:
            raise ValueError("learning opportunity requires an event with code-resolved episode lineage")
        if event["child_id"] != item.child_id:
            raise ValueError("learning opportunity child does not match its source event")
        plan = conn.execute(
            """SELECT lp.child_id AS plan_child_id,lp.episode_id AS plan_episode_id,
                      lp.task_id,lp.task_version,lp.attempt_group_id,lp.context_key,
                      lp.representation_key,lp.transfer_from_plan_id,lp.status,
                      lt.difficulty,lt.difficulty_provenance,lt.rubric_id,lt.rubric_version,
                      r.criteria_json,r.evidence_rule_id,r.evidence_rule_version
               FROM learning_episode_plans lp
               JOIN learning_tasks lt ON lt.task_id=lp.task_id AND lt.version=lp.task_version
               JOIN rubrics r
                 ON r.rubric_id=lt.rubric_id AND r.version=lt.rubric_version
               WHERE lp.plan_id=?""",
            (item.plan_id,),
        ).fetchone()
        if not plan:
            raise ValueError("learning opportunity references an unknown immutable plan")
        if plan["plan_child_id"] != item.child_id:
            raise ValueError("learning opportunity child does not match its plan")
        if plan["status"] == "cancelled":
            raise EvidenceConflict("cancelled learning plan cannot accept new observations")
        if plan["plan_episode_id"] != event["episode_id"]:
            raise ValueError("learning opportunity event is outside its plan's bound episode")
        if (plan["evidence_rule_id"], plan["evidence_rule_version"]) != (
            KNOWLEDGE_RULE_ID,
            KNOWLEDGE_RULE_VERSION,
        ):
            raise ValueError("learning opportunity rubric uses an unsupported evidence rule version")
        target_hash = sha256(item.opportunity_id.encode()).hexdigest()
        tombstone = conn.execute(
            """SELECT tombstone_id FROM evidence_source_tombstones
               WHERE target_kind='learning_opportunity' AND target_id_hash=?""",
            (target_hash,),
        ).fetchone()
        if tombstone:
            raise EvidenceConflict("deleted learning opportunity cannot be recreated from a stale replay")
        rubric_criteria = jload(plan["criteria_json"], [])
        expected_criteria = {str(criterion["criterion_id"]) for criterion in rubric_criteria}
        criterion_results = [result.model_dump(mode="json") for result in item.criterion_results]
        observed_criteria = {str(result["criterion_id"]) for result in criterion_results}
        if observed_criteria != expected_criteria:
            missing = sorted(expected_criteria - observed_criteria)
            unknown = sorted(observed_criteria - expected_criteria)
            raise ValueError(f"criterion results must exactly match rubric; missing={missing!r}, unknown={unknown!r}")
        task_result, task_score, task_scored = _score_criteria(rubric_criteria, criterion_results)
        strongest_assistance = _strongest_assistance(item)
        occurred_at = _as_of_iso(str(event["created_at"]))
        if datetime.fromisoformat(occurred_at) > datetime.fromisoformat(now) + timedelta(minutes=5):
            raise ValueError("learning opportunity source event cannot be future-dated")
        opportunity = {
            **item.model_dump(mode="json"),
            "episode_id": str(event["episode_id"]),
            "task_id": str(plan["task_id"]),
            "task_version": int(plan["task_version"]),
            "independence_status": str(event["independence_status"]),
            "attempt_group_id": str(plan["attempt_group_id"]),
            "assistance_level": strongest_assistance,
            "context_key": str(plan["context_key"]),
            "representation_key": str(plan["representation_key"]),
            "rubric_id": str(plan["rubric_id"]),
            "rubric_version": int(plan["rubric_version"]),
            "evidence_rule_id": str(plan["evidence_rule_id"]),
            "evidence_rule_version": str(plan["evidence_rule_version"]),
            "rubric_criteria": rubric_criteria,
            "criterion_results": criterion_results,
            "scored_criteria": task_scored,
            "result": task_result,
            "score": task_score,
            "is_trivial": int(plan["difficulty"] == "trivial"),
            "is_transfer": int(plan["transfer_from_plan_id"] is not None),
            "child_continued": int(item.child_continued),
            "child_new_question": int(item.child_new_question),
            "difficulty": str(plan["difficulty"]),
            "difficulty_provenance": str(plan["difficulty_provenance"]),
            "occurred_at": occurred_at,
        }
        # Episode grouping and independence are correctable code-owned lineage,
        # so they are intentionally excluded from the immutable caller-payload hash.
        hash_payload = item.model_dump(mode="json")
        hash_payload.update({"assistance_level": strongest_assistance, "occurred_at": opportunity["occurred_at"]})
        digest = _payload_hash(hash_payload)
        existing = conn.execute(
            "SELECT opportunity_hash FROM learning_opportunities WHERE opportunity_id=?",
            (item.opportunity_id,),
        ).fetchone()
        if existing:
            if existing["opportunity_hash"] != digest:
                raise EvidenceConflict(
                    f"learning opportunity {item.opportunity_id!r} was already used for a different payload"
                )
            evidence_ids = [
                str(row[0])
                for row in conn.execute(
                    "SELECT evidence_id FROM evidence_records WHERE opportunity_id=? ORDER BY evidence_id",
                    (item.opportunity_id,),
                )
            ]
            return {
                "opportunity_id": item.opportunity_id,
                "episode_id": str(event["episode_id"]),
                "plan_id": item.plan_id,
                "evidence_ids": evidence_ids,
                "duplicate": True,
            }
        occupied_attempt = conn.execute(
            """SELECT opportunity_id FROM learning_opportunities
               WHERE plan_id=? AND attempt_number=?""",
            (item.plan_id, item.attempt_number),
        ).fetchone()
        if occupied_attempt:
            raise EvidenceConflict("learning plan attempt already has a different immutable opportunity identifier")
        if item.attempt_number > 1:
            prior_attempt = conn.execute(
                """SELECT MAX(attempt_number) FROM learning_opportunities
                   WHERE plan_id=?""",
                (item.plan_id,),
            ).fetchone()[0]
            if prior_attempt is None or int(prior_attempt) != item.attempt_number - 1:
                raise ValueError("retry observations must follow the prior attempt in their bound plan")
        conn.execute(
            """INSERT INTO learning_opportunities(
               opportunity_id,child_id,event_id,episode_id,task_id,task_version,plan_id,rubric_id,rubric_version,
               evidence_rule_id,evidence_rule_version,attempt_group_id,attempt_number,actor,initiation,
               response_mode,assistance_level,result,score,criterion_results_json,scorer,scorer_version,
               source_type,independence_status,context_key,representation_key,is_transfer,is_trivial,
               child_continued,child_new_question,retention_class,opportunity_hash,opportunity_id_hash,
               occurred_at,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item.opportunity_id,
                item.child_id,
                item.event_id,
                opportunity["episode_id"],
                opportunity["task_id"],
                opportunity["task_version"],
                item.plan_id,
                opportunity["rubric_id"],
                opportunity["rubric_version"],
                opportunity["evidence_rule_id"],
                opportunity["evidence_rule_version"],
                opportunity["attempt_group_id"],
                item.attempt_number,
                item.actor,
                item.initiation,
                item.response_mode,
                strongest_assistance,
                opportunity["result"],
                opportunity["score"],
                jdump(criterion_results),
                item.scorer,
                item.scorer_version,
                item.source_type,
                opportunity["independence_status"],
                opportunity["context_key"],
                opportunity["representation_key"],
                opportunity["is_transfer"],
                opportunity["is_trivial"],
                opportunity["child_continued"],
                opportunity["child_new_question"],
                item.retention_class,
                digest,
                target_hash,
                opportunity["occurred_at"],
                now,
            ),
        )
        for ordinal, assistance in enumerate(item.assistance_history, start=1):
            conn.execute(
                """INSERT INTO opportunity_assistance_events(
                   assistance_id,opportunity_id,child_id,ordinal,assistance_level,provider,occurred_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    f"assist_{sha256(f'{item.opportunity_id}|{ordinal}'.encode()).hexdigest()[:24]}",
                    item.opportunity_id,
                    item.child_id,
                    ordinal,
                    assistance.level,
                    assistance.provider,
                    assistance.occurred_at.isoformat() if assistance.occurred_at else opportunity["occurred_at"],
                ),
            )
        evidence_ids = _compile_evidence(conn, opportunity)
        conn.execute(
            """UPDATE learning_episode_plans SET status='completed',updated_at=?
               WHERE plan_id=? AND child_id=?""",
            (now, item.plan_id, item.child_id),
        )
        projection = recompute_child_projections_in_transaction(conn, item.child_id)
    return {
        "opportunity_id": item.opportunity_id,
        "plan_id": item.plan_id,
        "episode_id": opportunity["episode_id"],
        "evidence_ids": evidence_ids,
        "projection_digest": projection["output_digest"],
        "duplicate": False,
    }


def record_model_annotations(
    db_path: str | Path,
    *,
    child_id: str,
    event_id: str,
    run_id: int,
    candidates: list[ObservationCandidate],
) -> list[str]:
    """Persist bounded model semantics as audit-only rows with no evidence path."""

    init_db(db_path)
    created: list[str] = []
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        event = conn.execute("SELECT child_id FROM events WHERE id=?", (event_id,)).fetchone()
        if not event or event["child_id"] != child_id:
            raise ValueError("annotation child does not match its authoritative source event")
        run = conn.execute("SELECT 1 FROM runs WHERE id=? AND event_id=?", (run_id, event_id)).fetchone()
        if not run:
            raise ValueError("annotation run does not belong to its authoritative source event")
        created_at = datetime.now(UTC)
        expires_at = (created_at + timedelta(days=30)).isoformat()
        collection_enabled = _collection_enabled_from_conn(conn, child_id)
        for ordinal, candidate in enumerate(candidates):
            payload = candidate.model_dump(mode="json")
            digest = _payload_hash(payload)
            annotation_id = "ann_" + sha256(f"{event_id}|{run_id}|{ordinal}|{digest}".encode()).hexdigest()[:24]
            existing = conn.execute(
                """SELECT annotation_id,annotation_hash FROM evidence_annotations
                   WHERE event_id=? AND run_id=? AND ordinal=?""",
                (event_id, run_id, ordinal),
            ).fetchone()
            if existing:
                if existing["annotation_hash"] != digest:
                    raise EvidenceConflict("model annotation ordinal was reused with different content")
                created.append(str(existing["annotation_id"]))
                continue
            if not collection_enabled:
                # A privacy pause must never turn an otherwise valid answer into
                # a failed/retried job. Existing idempotent rows remain readable,
                # but no new model-authored audit record is stored.
                continue
            conn.execute(
                """INSERT INTO evidence_annotations(
                   annotation_id,child_id,event_id,run_id,ordinal,kind,label,text,author_type,authority,
                   purpose,retention_class,expires_at,annotation_hash,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    annotation_id,
                    child_id,
                    event_id,
                    run_id,
                    ordinal,
                    candidate.kind,
                    candidate.label,
                    candidate.text,
                    "model",
                    "audit_only",
                    "semantic_audit",
                    "transient",
                    expires_at,
                    digest,
                    created_at.isoformat(),
                ),
            )
            created.append(annotation_id)
    return created


def _as_of_iso(as_of: datetime | str | None) -> str:
    if as_of is None:
        return utcnow()
    if isinstance(as_of, str):
        parsed = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    else:
        parsed = as_of
    if parsed.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _effective_evidence(
    conn: sqlite3.Connection, child_id: str, *, as_of: datetime | str | None = None
) -> list[dict[str, Any]]:
    cutoff = _as_of_iso(as_of)
    rows = [
        dict(row)
        for row in conn.execute(
            """SELECT er.*,lo.plan_id,lp.transfer_from_plan_id,
                      lo.actor,lo.initiation,lo.child_continued,lo.child_new_question,
                      lo.response_mode,lo.independence_status,lo.occurred_at,
                      lt.difficulty,lt.difficulty_provenance
               FROM evidence_records er
               JOIN learning_opportunities lo
                 ON lo.opportunity_id=er.opportunity_id AND lo.child_id=er.child_id
               JOIN learning_episode_plans lp
                 ON lp.plan_id=lo.plan_id AND lp.child_id=lo.child_id
               JOIN learning_tasks lt ON lt.task_id=lo.task_id AND lt.version=lo.task_version
               WHERE er.child_id=? AND lo.occurred_at<=?
                 AND (er.expires_at IS NULL OR er.expires_at>?)
               ORDER BY er.evidence_id""",
            (child_id, cutoff, cutoff),
        )
    ]
    corrections: dict[str, list[sqlite3.Row]] = {}
    for correction in conn.execute(
        """SELECT evidence_id,action,replacement_json FROM evidence_corrections
           WHERE child_id=? ORDER BY created_at,correction_id""",
        (child_id,),
    ):
        corrections.setdefault(str(correction["evidence_id"]), []).append(correction)
    retention_deadlines: dict[str, str] = {}
    for retention in conn.execute(
        """SELECT opportunity_id,MIN(retain_until) AS retain_until FROM learner_retention_events
           WHERE child_id=? GROUP BY opportunity_id""",
        (child_id,),
    ):
        retention_deadlines[str(retention["opportunity_id"])] = _as_of_iso(str(retention["retain_until"]))
    effective: list[dict[str, Any]] = []
    for row in rows:
        if retention_deadlines.get(str(row["opportunity_id"]), "9999-12-31T23:59:59+00:00") <= cutoff:
            continue
        active = True
        for correction in corrections.get(str(row["evidence_id"]), []):
            if correction["action"] == "exclude":
                active = False
            elif correction["action"] == "restore":
                active = True
            else:
                row.update(jload(correction["replacement_json"], {}))
        if not active:
            continue
        stance, reliability, strength, eligible = _classification(row)
        row.update(
            {
                "stance": stance,
                "source_reliability": reliability,
                "evidence_strength": strength,
                "projection_eligible": eligible,
            }
        )
        effective.append(row)
    return effective


def _digest_rows(rows: list[dict[str, Any]]) -> str:
    stable = [
        {
            key: row.get(key)
            for key in (
                "evidence_id",
                "plan_id",
                "transfer_from_plan_id",
                "construct",
                "stance",
                "result",
                "projection_eligible",
                "episode_id",
                "context_key",
                "representation_key",
                "assistance_level",
                "attempt_number",
                "is_transfer",
                "is_trivial",
                "child_continued",
                "child_new_question",
                "retention_class",
                "rule_id",
                "rule_version",
            )
        }
        for row in rows
    ]
    return _payload_hash(
        {
            "projector_version": PROJECTOR_VERSION,
            "knowledge_rule_version": KNOWLEDGE_RULE_VERSION,
            "interest_rule_version": INTEREST_RULE_VERSION,
            "evidence": stable,
        }
    )


def _has_transfer(successes: list[dict[str, Any]]) -> bool:
    independent = [row for row in successes if row["assistance_level"] == "none"]
    for transfer in (row for row in independent if bool(row["is_transfer"])):
        source_plan_id = transfer.get("transfer_from_plan_id")
        if not source_plan_id:
            continue
        if any(
            other["evidence_id"] != transfer["evidence_id"]
            and other.get("plan_id") == source_plan_id
            and other["episode_id"] != transfer["episode_id"]
            and (
                other["context_key"] != transfer["context_key"]
                or other["representation_key"] != transfer["representation_key"]
            )
            for other in independent
        ):
            return True
    return False


def _knowledge_projection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exposure = [row for row in rows if row["construct"] == "exposure" and row["projection_eligible"]]
    performance = [row for row in rows if row["construct"] == "knowledge_performance" and row["projection_eligible"]]
    successes = [row for row in performance if row["result"] == "correct"]
    partial = [row for row in performance if row["result"] == "partial"]
    contradictions = [row for row in performance if row["stance"] == "contradicts"]
    independent_successes = [row for row in successes if row["assistance_level"] == "none"]
    successful_episodes = {row["episode_id"] for row in successes}
    if _has_transfer(successes) and not contradictions:
        state = "transfer_observed"
    elif len(successful_episodes) >= 2 and independent_successes and not contradictions:
        state = "independently_demonstrated"
    elif independent_successes:
        state = "demonstrated_in_context"
    elif successes or partial:
        state = "practicing_with_support"
    elif exposure:
        state = "exposure_recorded"
    else:
        state = "not_observed_in_harness"
    state_strength = {
        "not_observed_in_harness": "none",
        "exposure_recorded": "weak",
        "practicing_with_support": "weak",
        "demonstrated_in_context": "moderate",
        "independently_demonstrated": "strong",
        "transfer_observed": "strong",
    }[state]
    assistance = {row["assistance_level"] for row in performance}
    if not performance:
        support_level = "not_observed"
    elif "unknown" in assistance:
        support_level = "unknown"
    elif assistance & {"worked_example", "joint_completion", "direct_answer"}:
        support_level = "substantial_support"
    elif assistance - {"none"}:
        support_level = "some_support"
    else:
        support_level = "independent"
    eligible = exposure + performance
    sample = len(eligible)
    uncertainty = min(1.0, round((1 / (sample + 1)) + (0.12 * len(contradictions)), 4))
    return {
        "state": state,
        "evidence_strength": state_strength,
        "uncertainty": uncertainty,
        "support_level": support_level,
        "eligible_opportunity_count": len({row["opportunity_id"] for row in eligible}),
        "eligible_episode_count": len({row["episode_id"] for row in eligible}),
        "context_diversity": len({row["context_key"] for row in eligible}),
        "representation_diversity": len({row["representation_key"] for row in eligible}),
        "contradicting_count": len(contradictions),
        "last_observed_at": max((row["occurred_at"] for row in eligible), default=None),
        "rationale": (
            f"{state.replace('_', ' ')} from {len(successes)} successful, {len(partial)} partial, "
            f"and {len(contradictions)} contradicting eligible observations across "
            f"{len({row['episode_id'] for row in eligible})} episode(s)."
        ),
    }


def _interest_projection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = [row for row in rows if row["construct"] == "interest" and row["projection_eligible"]]
    initiated = len(evidence)
    continued = sum(bool(row["child_continued"]) or bool(row["child_new_question"]) for row in evidence)
    episodes = {row["episode_id"] for row in evidence}
    contexts = {row["context_key"] for row in evidence}
    parent_confirmed = any(
        row["source_type"] == "parent_report" and row["retention_class"] == "parent_confirmed" for row in evidence
    )
    if parent_confirmed:
        phase = "sustained_parent_confirmed"
    elif len(episodes) >= 2 and len(contexts) >= 2:
        phase = "emerging_recurrent"
    elif continued:
        phase = "maintained_situational"
    elif initiated:
        phase = "triggered_situational"
    else:
        phase = "not_observed"
    uncertainty = 1.0 if not evidence else round(1 / (len(evidence) + 1), 4)
    last = max((row["occurred_at"] for row in evidence), default=None)
    next_review = None
    if last:
        next_review = (datetime.fromisoformat(str(last).replace("Z", "+00:00")) + timedelta(days=90)).isoformat()
    return {
        "phase": phase,
        "child_initiated_count": initiated,
        "continued_or_question_count": continued,
        "eligible_episode_count": len(episodes),
        "context_diversity": len(contexts),
        "uncertainty": uncertainty,
        "next_review_at": next_review,
        "rationale": (
            f"{phase.replace('_', ' ')} from {initiated} child-attributed signal(s) across "
            f"{len(episodes)} episode(s) and {len(contexts)} context(s)."
        ),
    }


def _upsert_claim(
    conn: sqlite3.Connection,
    *,
    child_id: str,
    concept_id: str,
    concept_version: int,
    construct: str,
    value: str,
    status: str,
    rule_id: str,
    rule_version: str,
    input_digest: str,
    rationale: str,
    evidence: list[dict[str, Any]],
    now: str,
) -> None:
    claim_id = "claim2_" + sha256(f"{child_id}|{concept_id}|{concept_version}|{construct}".encode()).hexdigest()[:24]
    conn.execute(
        """INSERT INTO derived_claims_v2(
           claim_id,child_id,concept_id,concept_version,construct,value,status,rule_id,rule_version,
           input_digest,rationale,computed_at
           ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(child_id,concept_id,concept_version,construct) DO UPDATE SET
             value=excluded.value,status=excluded.status,rule_id=excluded.rule_id,
             rule_version=excluded.rule_version,input_digest=excluded.input_digest,rationale=excluded.rationale,
             computed_at=excluded.computed_at
           WHERE derived_claims_v2.input_digest!=excluded.input_digest
              OR derived_claims_v2.value!=excluded.value OR derived_claims_v2.status!=excluded.status""",
        (
            claim_id,
            child_id,
            concept_id,
            concept_version,
            construct,
            value,
            status,
            rule_id,
            rule_version,
            input_digest,
            rationale,
            now,
        ),
    )
    conn.execute("DELETE FROM derived_claim_evidence_v2 WHERE claim_id=?", (claim_id,))
    conn.executemany(
        """INSERT INTO derived_claim_evidence_v2(claim_id,child_id,evidence_id,stance)
           VALUES(?,?,?,?)""",
        [(claim_id, child_id, row["evidence_id"], row["stance"]) for row in evidence],
    )


def recompute_child_projections_in_transaction(
    conn: sqlite3.Connection,
    child_id: str,
    *,
    as_of: datetime | str | None = None,
) -> dict[str, Any]:
    """Full deterministic rebuild. Caller must hold the surrounding transaction."""

    child = conn.execute("SELECT 1 FROM children WHERE id=?", (child_id,)).fetchone()
    if not child:
        raise ValueError(f"unknown child {child_id!r}")
    evidence = _effective_evidence(conn, child_id, as_of=as_of)
    known = {
        (str(row["concept_id"]), int(row["concept_version"]))
        for row in conn.execute(
            """SELECT concept_id,concept_version FROM learner_state_projections WHERE child_id=?
               UNION SELECT concept_id,concept_version FROM interest_projections WHERE child_id=?
               UNION SELECT concept_id,concept_version FROM evidence_records WHERE child_id=?""",
            (child_id, child_id, child_id),
        )
    }
    grouped = {
        key: [row for row in evidence if (str(row["concept_id"]), int(row["concept_version"])) == key]
        for key in sorted(known)
    }
    global_input_digest = _digest_rows(evidence)
    now = _as_of_iso(as_of)
    output_rows: list[dict[str, Any]] = []
    for (concept_id, concept_version), rows in grouped.items():
        knowledge = _knowledge_projection(rows)
        knowledge_digest = _digest_rows(
            [row for row in rows if row["construct"] in {"exposure", "knowledge_performance"}]
        )
        conn.execute(
            """INSERT INTO learner_state_projections(
               child_id,concept_id,concept_version,state,evidence_strength,uncertainty,support_level,
               eligible_opportunity_count,eligible_episode_count,context_diversity,representation_diversity,
               contradicting_count,last_observed_at,rule_id,rule_version,input_digest,rationale,computed_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(child_id,concept_id,concept_version) DO UPDATE SET
                 state=excluded.state,evidence_strength=excluded.evidence_strength,
                 uncertainty=excluded.uncertainty,support_level=excluded.support_level,
                 eligible_opportunity_count=excluded.eligible_opportunity_count,
                 eligible_episode_count=excluded.eligible_episode_count,context_diversity=excluded.context_diversity,
                 representation_diversity=excluded.representation_diversity,
                 contradicting_count=excluded.contradicting_count,last_observed_at=excluded.last_observed_at,
                 rule_id=excluded.rule_id,rule_version=excluded.rule_version,input_digest=excluded.input_digest,
                 rationale=excluded.rationale,computed_at=excluded.computed_at
               WHERE learner_state_projections.input_digest!=excluded.input_digest""",
            (
                child_id,
                concept_id,
                concept_version,
                knowledge["state"],
                knowledge["evidence_strength"],
                knowledge["uncertainty"],
                knowledge["support_level"],
                knowledge["eligible_opportunity_count"],
                knowledge["eligible_episode_count"],
                knowledge["context_diversity"],
                knowledge["representation_diversity"],
                knowledge["contradicting_count"],
                knowledge["last_observed_at"],
                KNOWLEDGE_RULE_ID,
                KNOWLEDGE_RULE_VERSION,
                knowledge_digest,
                knowledge["rationale"],
                now,
            ),
        )
        conn.execute(
            """DELETE FROM learner_projection_evidence
               WHERE child_id=? AND concept_id=? AND concept_version=?""",
            (child_id, concept_id, concept_version),
        )
        knowledge_evidence = [row for row in rows if row["construct"] in {"exposure", "knowledge_performance"}]
        conn.executemany(
            """INSERT INTO learner_projection_evidence(
               child_id,concept_id,concept_version,evidence_id,stance
               ) VALUES(?,?,?,?,?)""",
            [(child_id, concept_id, concept_version, row["evidence_id"], row["stance"]) for row in knowledge_evidence],
        )

        interest = _interest_projection(rows)
        interest_evidence = [row for row in rows if row["construct"] == "interest"]
        interest_digest = _digest_rows(interest_evidence)
        conn.execute(
            """INSERT INTO interest_projections(
               child_id,concept_id,concept_version,phase,child_initiated_count,continued_or_question_count,
               eligible_episode_count,context_diversity,uncertainty,next_review_at,rule_id,rule_version,
               input_digest,rationale,computed_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(child_id,concept_id,concept_version) DO UPDATE SET
                 phase=excluded.phase,child_initiated_count=excluded.child_initiated_count,
                 continued_or_question_count=excluded.continued_or_question_count,
                 eligible_episode_count=excluded.eligible_episode_count,context_diversity=excluded.context_diversity,
                 uncertainty=excluded.uncertainty,next_review_at=excluded.next_review_at,rule_id=excluded.rule_id,
                 rule_version=excluded.rule_version,input_digest=excluded.input_digest,
                 rationale=excluded.rationale,computed_at=excluded.computed_at
               WHERE interest_projections.input_digest!=excluded.input_digest""",
            (
                child_id,
                concept_id,
                concept_version,
                interest["phase"],
                interest["child_initiated_count"],
                interest["continued_or_question_count"],
                interest["eligible_episode_count"],
                interest["context_diversity"],
                interest["uncertainty"],
                interest["next_review_at"],
                INTEREST_RULE_ID,
                INTEREST_RULE_VERSION,
                interest_digest,
                interest["rationale"],
                now,
            ),
        )
        conn.execute(
            """DELETE FROM interest_projection_evidence
               WHERE child_id=? AND concept_id=? AND concept_version=?""",
            (child_id, concept_id, concept_version),
        )
        conn.executemany(
            """INSERT INTO interest_projection_evidence(
               child_id,concept_id,concept_version,evidence_id,stance
               ) VALUES(?,?,?,?,?)""",
            [(child_id, concept_id, concept_version, row["evidence_id"], row["stance"]) for row in interest_evidence],
        )

        if knowledge["state"] == "not_observed_in_harness":
            claim_id = (
                "claim2_" + sha256(f"{child_id}|{concept_id}|{concept_version}|knowledge".encode()).hexdigest()[:24]
            )
            conn.execute("DELETE FROM derived_claims_v2 WHERE claim_id=?", (claim_id,))
        else:
            _upsert_claim(
                conn,
                child_id=child_id,
                concept_id=concept_id,
                concept_version=concept_version,
                construct="knowledge",
                value=knowledge["state"],
                status="shadow",
                rule_id=KNOWLEDGE_RULE_ID,
                rule_version=KNOWLEDGE_RULE_VERSION,
                input_digest=knowledge_digest,
                rationale=knowledge["rationale"],
                evidence=knowledge_evidence,
                now=now,
            )
        if interest["phase"] == "not_observed":
            claim_id = (
                "claim2_" + sha256(f"{child_id}|{concept_id}|{concept_version}|interest".encode()).hexdigest()[:24]
            )
            conn.execute("DELETE FROM derived_claims_v2 WHERE claim_id=?", (claim_id,))
        else:
            _upsert_claim(
                conn,
                child_id=child_id,
                concept_id=concept_id,
                concept_version=concept_version,
                construct="interest",
                value=interest["phase"],
                status=("parent_confirmed" if interest["phase"] == "sustained_parent_confirmed" else "shadow"),
                rule_id=INTEREST_RULE_ID,
                rule_version=INTEREST_RULE_VERSION,
                input_digest=interest_digest,
                rationale=interest["rationale"],
                evidence=interest_evidence,
                now=now,
            )
        output_rows.append(
            {
                "concept_id": concept_id,
                "concept_version": concept_version,
                "knowledge": knowledge["state"],
                "interest": interest["phase"],
                "knowledge_digest": knowledge_digest,
                "interest_digest": interest_digest,
            }
        )
    output_digest = _payload_hash(output_rows)
    run_id = (
        "prun_"
        + sha256(f"{child_id}|{PROJECTOR_VERSION}|{global_input_digest}|{output_digest}".encode()).hexdigest()[:24]
    )
    conn.execute(
        """INSERT OR IGNORE INTO projection_runs(
           run_id,child_id,projector_version,input_digest,output_digest,created_at
           ) VALUES(?,?,?,?,?,?)""",
        (run_id, child_id, PROJECTOR_VERSION, global_input_digest, output_digest, now),
    )
    return {
        "run_id": run_id,
        "input_digest": global_input_digest,
        "output_digest": output_digest,
        "concept_count": len(grouped),
    }


def recompute_child_projections(
    db_path: str | Path, child_id: str, *, as_of: datetime | str | None = None
) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        return recompute_child_projections_in_transaction(conn, child_id, as_of=as_of)


def correct_evidence(
    db_path: str | Path,
    *,
    child_id: str,
    evidence_id: str,
    action: Literal["exclude", "restore", "correct"],
    reason: str,
    replacement: dict[str, Any] | None = None,
    actor_type: Literal["parent", "operator", "system"] = "parent",
    correction_id: str | None = None,
    _transaction_hook: Callable[[sqlite3.Connection], None] | None = None,
) -> dict[str, Any]:
    """Append a correction and rebuild all dependent projections in one transaction."""

    if not reason.strip() or len(reason) > 500:
        raise ValueError("correction reason must be 1..500 characters")
    replacement = replacement or {}
    unknown = set(replacement) - _CORRECTABLE_FIELDS
    if unknown:
        raise ValueError(f"unsupported correction fields: {sorted(unknown)!r}")
    if action == "correct" and not replacement:
        raise ValueError("correct action requires a replacement")
    if action != "correct" and replacement:
        raise ValueError("replacement fields are only valid for correct action")
    valid_values = {
        "result": {"not_observed", "correct", "partial", "incorrect", "abandoned"},
        "assistance_level": set(_ASSISTANCE_ORDER),
    }
    for key, allowed in valid_values.items():
        if key in replacement and replacement[key] not in allowed:
            raise ValueError(f"invalid corrected {key}")
    init_db(db_path)
    correction_id = correction_id or f"ecorr_{uuid4().hex[:20]}"
    digest = _payload_hash(
        {
            "evidence_id": evidence_id,
            "child_id": child_id,
            "action": action,
            "replacement": replacement,
            "actor_type": actor_type,
            "reason": reason,
        }
    )
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        evidence = conn.execute(
            "SELECT 1 FROM evidence_records WHERE evidence_id=? AND child_id=?", (evidence_id, child_id)
        ).fetchone()
        if not evidence:
            raise ValueError("evidence does not belong to this child")
        existing = conn.execute(
            "SELECT correction_hash FROM evidence_corrections WHERE correction_id=?", (correction_id,)
        ).fetchone()
        if existing:
            if existing["correction_hash"] != digest:
                raise EvidenceConflict(f"correction id {correction_id!r} was reused with a different payload")
            return {"correction_id": correction_id, "duplicate": True}
        conn.execute(
            """INSERT INTO evidence_corrections(
               correction_id,evidence_id,target_evidence_hash,child_id,action,replacement_json,actor_type,reason,
               correction_hash,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                correction_id,
                evidence_id,
                sha256(evidence_id.encode()).hexdigest(),
                child_id,
                action,
                jdump(replacement),
                actor_type,
                reason,
                digest,
                utcnow(),
            ),
        )
        projection = recompute_child_projections_in_transaction(conn, child_id)
        if _transaction_hook:
            _transaction_hook(conn)
    return {"correction_id": correction_id, "projection_digest": projection["output_digest"], "duplicate": False}


def delete_learning_source(
    db_path: str | Path,
    *,
    child_id: str,
    opportunity_id: str,
    reason: Literal["parent_deleted", "retention_expired", "operator_cleanup"] = "parent_deleted",
    actor_type: Literal["parent", "operator"] = "parent",
    _transaction_hook: Callable[[sqlite3.Connection], None] | None = None,
) -> dict[str, Any]:
    with managed_data_lifecycle(db_path):
        result = _delete_learning_source_locked(
            db_path,
            child_id=child_id,
            opportunity_id=opportunity_id,
            reason=reason,
            actor_type=actor_type,
            _transaction_hook=_transaction_hook,
        )
        return {**result, **reconcile_database_deletions(db_path)}


def _delete_learning_source_locked(
    db_path: str | Path,
    *,
    child_id: str,
    opportunity_id: str,
    reason: Literal["parent_deleted", "retention_expired", "operator_cleanup"] = "parent_deleted",
    actor_type: Literal["parent", "operator"] = "parent",
    _transaction_hook: Callable[[sqlite3.Connection], None] | None = None,
) -> dict[str, Any]:
    """Physically remove one Phase 1 source, retain a content-free tombstone, and rebuild."""

    init_db(db_path)
    target_hash = sha256(opportunity_id.encode()).hexdigest()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT child_id,plan_id FROM learning_opportunities WHERE opportunity_id=?",
            (opportunity_id,),
        ).fetchone()
        if not row:
            prior = conn.execute(
                """SELECT tombstone_id FROM evidence_source_tombstones
                   WHERE target_kind='learning_opportunity' AND target_id_hash=?""",
                (target_hash,),
            ).fetchone()
            if prior:
                return {"tombstone_id": str(prior["tombstone_id"]), "duplicate": True}
            raise ValueError("learning opportunity does not exist for this child")
        if row["child_id"] != child_id:
            raise ValueError("learning opportunity does not belong to this child")
        tombstone_id = "tomb_" + sha256(f"{child_id}|{target_hash}".encode()).hexdigest()[:24]
        conn.execute(
            """INSERT INTO evidence_source_tombstones(
               tombstone_id,child_id,target_kind,target_id_hash,reason,actor_type,deleted_at
               ) VALUES(?,?,'learning_opportunity',?,?,?,?)""",
            (tombstone_id, child_id, target_hash, reason, actor_type, utcnow()),
        )
        conn.execute(
            "DELETE FROM learning_opportunities WHERE opportunity_id=? AND child_id=?",
            (opportunity_id, child_id),
        )
        conn.execute(
            """UPDATE learning_episode_plans SET status='cancelled',updated_at=?
               WHERE plan_id=? AND child_id=?""",
            (utcnow(), row["plan_id"], child_id),
        )
        projection = recompute_child_projections_in_transaction(conn, child_id)
        if _transaction_hook:
            _transaction_hook(conn)
    return {
        "tombstone_id": tombstone_id,
        "projection_digest": projection["output_digest"],
        "duplicate": False,
    }


def expire_learning_records(db_path: str | Path, *, as_of: datetime | str | None = None) -> dict[str, Any]:
    with managed_data_lifecycle(db_path):
        result = _expire_learning_records_locked(db_path, as_of=as_of)
        return {**result, **reconcile_database_deletions(db_path)}


def _expire_learning_records_locked(
    db_path: str | Path,
    *,
    as_of: datetime | str | None = None,
) -> dict[str, Any]:
    """Apply retention deadlines and rebuild affected children in one transaction."""

    init_db(db_path)
    cutoff = _as_of_iso(as_of)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        expired_annotations = int(
            conn.execute("SELECT COUNT(*) FROM evidence_annotations WHERE expires_at<=?", (cutoff,)).fetchone()[0]
        )
        conn.execute("DELETE FROM evidence_annotations WHERE expires_at<=?", (cutoff,))
        expired_evidence = conn.execute(
            """SELECT er.child_id,COUNT(*) AS expired_count FROM evidence_records er
               WHERE er.expires_at IS NOT NULL AND er.expires_at<=?
                 AND (EXISTS(
                   SELECT 1 FROM learner_projection_evidence lpe
                   WHERE lpe.evidence_id=er.evidence_id AND lpe.child_id=er.child_id
                 ) OR EXISTS(
                   SELECT 1 FROM interest_projection_evidence ipe
                   WHERE ipe.evidence_id=er.evidence_id AND ipe.child_id=er.child_id
                 ))
               GROUP BY er.child_id""",
            (cutoff,),
        ).fetchall()
        expired = [
            dict(row)
            for row in conn.execute(
                """SELECT lo.opportunity_id,lo.child_id,lo.plan_id
               FROM learning_opportunities lo
               WHERE lo.retention_class='transient' AND NOT EXISTS(
                 SELECT 1 FROM evidence_records er
                 WHERE er.opportunity_id=lo.opportunity_id
                   AND (er.expires_at IS NULL OR er.expires_at>?)
               ) ORDER BY lo.child_id,lo.opportunity_id""",
                (cutoff,),
            ).fetchall()
        ]
        parent_expired = [
            dict(row)
            for row in conn.execute(
                """SELECT lo.opportunity_id,lo.child_id,lo.plan_id
                   FROM learning_opportunities lo JOIN learner_retention_events re
                     ON re.opportunity_id=lo.opportunity_id AND re.child_id=lo.child_id
                   GROUP BY lo.opportunity_id,lo.child_id,lo.plan_id
                   HAVING MIN(re.retain_until)<=?
                   ORDER BY lo.child_id,lo.opportunity_id""",
                (cutoff,),
            ).fetchall()
        ]
        expired_by_key = {
            (str(row["child_id"]), str(row["opportunity_id"])): row for row in [*expired, *parent_expired]
        }
        expired = list(expired_by_key.values())
        children = sorted(
            {str(row["child_id"]) for row in expired} | {str(row["child_id"]) for row in expired_evidence}
        )
        for row in expired:
            opportunity_id = str(row["opportunity_id"])
            child_id = str(row["child_id"])
            target_hash = sha256(opportunity_id.encode()).hexdigest()
            tombstone_id = "tomb_" + sha256(f"{child_id}|{target_hash}".encode()).hexdigest()[:24]
            conn.execute(
                """INSERT OR IGNORE INTO evidence_source_tombstones(
                   tombstone_id,child_id,target_kind,target_id_hash,reason,actor_type,deleted_at
                   ) VALUES(?,?,'learning_opportunity',?,'retention_expired','operator',?)""",
                (tombstone_id, child_id, target_hash, cutoff),
            )
            conn.execute(
                "DELETE FROM learning_opportunities WHERE opportunity_id=? AND child_id=?",
                (opportunity_id, child_id),
            )
            conn.execute(
                """UPDATE learning_episode_plans SET status='cancelled',updated_at=?
                   WHERE plan_id=? AND child_id=?""",
                (cutoff, row["plan_id"], child_id),
            )
        projections = {
            child_id: recompute_child_projections_in_transaction(conn, child_id, as_of=cutoff)["output_digest"]
            for child_id in children
        }
    return {
        "as_of": cutoff,
        "expired_annotations": expired_annotations,
        "expired_evidence_records": sum(int(row["expired_count"]) for row in expired_evidence),
        "expired_opportunities": len(expired),
        "expired_parent_deadlines": len(parent_expired),
        "affected_children": children,
        "projection_digests": projections,
    }


def learner_snapshot_from_conn(
    conn: sqlite3.Connection,
    child_id: str,
    *,
    limit: int | None = 20,
    as_of: datetime | str | None = None,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("snapshot limit must be positive or None")
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    recompute_child_projections_in_transaction(conn, child_id, as_of=as_of)
    limit_clause = "" if limit is None else " LIMIT ?"
    parameters: tuple[Any, ...] = (child_id,) if limit is None else (child_id, limit)
    knowledge = [
        dict(row)
        for row in conn.execute(
            """SELECT p.concept_id,p.concept_version,c.label,c.domain,p.state,p.evidence_strength,
                      p.uncertainty,p.support_level,p.eligible_opportunity_count,p.eligible_episode_count,
                      p.context_diversity,p.representation_diversity,p.contradicting_count,p.last_observed_at,
                      p.rule_id,p.rule_version,p.input_digest,p.rationale
               FROM learner_state_projections p JOIN concepts c
                 ON c.concept_id=p.concept_id AND c.version=p.concept_version
               WHERE p.child_id=? AND p.state!='not_observed_in_harness'
               ORDER BY p.last_observed_at DESC,p.concept_id"""
            + limit_clause,
            parameters,
        )
    ]
    interest = [
        dict(row)
        for row in conn.execute(
            """SELECT p.concept_id,p.concept_version,c.label,c.domain,p.phase,p.child_initiated_count,
                      p.continued_or_question_count,p.eligible_episode_count,p.context_diversity,p.uncertainty,
                      p.next_review_at,p.rule_id,p.rule_version,p.input_digest,p.rationale
               FROM interest_projections p JOIN concepts c
                 ON c.concept_id=p.concept_id AND c.version=p.concept_version
               WHERE p.child_id=? AND p.phase!='not_observed'
               ORDER BY p.eligible_episode_count DESC,p.concept_id"""
            + limit_clause,
            parameters,
        )
    ]
    link_limit_clause = "" if limit is None else " LIMIT 20"
    for item in knowledge:
        links = [
            dict(row)
            for row in conn.execute(
                """SELECT evidence_id,stance FROM learner_projection_evidence
                   WHERE child_id=? AND concept_id=? AND concept_version=? ORDER BY evidence_id"""
                + link_limit_clause,
                (child_id, item["concept_id"], item["concept_version"]),
            )
        ]
        item["supporting_evidence_ids"] = [str(link["evidence_id"]) for link in links if link["stance"] == "supports"]
        item["contradicting_evidence_ids"] = [
            str(link["evidence_id"]) for link in links if link["stance"] == "contradicts"
        ]
        item["context_evidence_ids"] = [str(link["evidence_id"]) for link in links if link["stance"] == "context_only"]
    for item in interest:
        links = [
            dict(row)
            for row in conn.execute(
                """SELECT evidence_id,stance FROM interest_projection_evidence
                   WHERE child_id=? AND concept_id=? AND concept_version=? ORDER BY evidence_id"""
                + link_limit_clause,
                (child_id, item["concept_id"], item["concept_version"]),
            )
        ]
        item["supporting_evidence_ids"] = [str(link["evidence_id"]) for link in links if link["stance"] == "supports"]
        item["context_evidence_ids"] = [str(link["evidence_id"]) for link in links if link["stance"] == "context_only"]
    digest = _payload_hash({"knowledge": knowledge, "interest": interest})
    return {
        "status": "shadow_phase1",
        "may_affect_recommendations": False,
        "projector_version": PROJECTOR_VERSION,
        "knowledge_rule_version": KNOWLEDGE_RULE_VERSION,
        "interest_rule_version": INTEREST_RULE_VERSION,
        "knowledge": knowledge,
        "interests": interest,
        "snapshot_digest": digest,
        "raw_evidence_included": False,
        "legacy_graph_included": False,
    }


def learner_snapshot(
    db_path: str | Path,
    child_id: str,
    *,
    limit: int | None = 20,
    as_of: datetime | str | None = None,
) -> dict[str, Any]:
    init_db(db_path)
    with connect(db_path) as conn:
        child = conn.execute("SELECT 1 FROM children WHERE id=?", (child_id,)).fetchone()
        if not child:
            raise ValueError(f"unknown child {child_id!r}")
        return learner_snapshot_from_conn(conn, child_id, limit=limit, as_of=as_of)


__all__ = [
    "AssistanceEventInput",
    "ConceptDefinition",
    "ConceptRef",
    "ConceptRelationDefinition",
    "EvidenceConflict",
    "LearningEpisodePlanInput",
    "LearningOpportunityInput",
    "LearningTaskDefinition",
    "RubricDefinition",
    "correct_evidence",
    "delete_learning_source",
    "expire_learning_records",
    "learner_snapshot",
    "record_learning_opportunity",
    "record_model_annotations",
    "recompute_child_projections",
    "register_concept",
    "register_concept_relation",
    "register_learning_task",
    "register_learning_plan",
    "register_rubric",
]
