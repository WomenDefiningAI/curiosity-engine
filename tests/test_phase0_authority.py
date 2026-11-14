from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from curiosity_engine.context_builder import build_context
from curiosity_engine.contracts import (
    Event,
    FeedbackOutput,
    GenericOutput,
    GraphMutation,
    InterestSignalOutput,
    ObservationCandidate,
    ParentAgentTurn,
    PullThreadOutput,
    QuickAnswerOutput,
)
from curiosity_engine.db import connect, init_db, jload, utcnow
from curiosity_engine.graph import add_child, add_edge, add_observation, apply_graph_mutation, upsert_node
from curiosity_engine.interaction import setup_household
from curiosity_engine.reasoning import ReasoningEngine, StubBackend
from curiosity_engine.runtime import CuriosityHarness
from curiosity_engine.service import CuriosityService
from curiosity_engine.sessions import SessionStore


def test_model_output_schemas_expose_only_bounded_observation_candidates() -> None:
    for output_model in (
        PullThreadOutput,
        QuickAnswerOutput,
        InterestSignalOutput,
        FeedbackOutput,
        GenericOutput,
    ):
        schema = json.dumps(output_model.model_json_schema(), sort_keys=True)
        assert "ObservationCandidate" in schema
        assert "GraphMutation" not in schema
        assert "set_knowledge_state" not in schema

    candidate_properties = ObservationCandidate.model_json_schema()["properties"]
    for forbidden in (
        "child_id",
        "event_id",
        "source_node_id",
        "target_node_id",
        "evidence_ids",
        "knowledge_state",
        "interest_phase",
        "confidence",
        "state",
    ):
        assert forbidden not in candidate_properties


def test_observation_candidate_rejects_authority_and_arbitrary_state() -> None:
    with pytest.raises(ValidationError):
        ObservationCandidate.model_validate(
            {
                "kind": "concept_mention",
                "label": "fractions",
                "child_id": "another-child",
                "state": {"knowledge_state": "demonstrated"},
            }
        )


class CandidateAndLegacyAttackBackend(StubBackend):
    def complete(self, *, role, system, payload, response_model):
        result = super().complete(
            role=role,
            system=system,
            payload=payload,
            response_model=response_model,
        )
        if response_model is PullThreadOutput:
            result["observation_candidates"] = [
                {"kind": "concept_mention", "label": "model-only candidate"}
            ]
            result["graph_updates"] = [
                {
                    "kind": "upsert_node",
                    "node_kind": "concept",
                    "label": "forged learner state",
                    "child_id": "child-b",
                    "state": {"knowledge_state": "demonstrated"},
                }
            ]
        return result


def test_model_candidates_are_audit_only_and_legacy_graph_attack_is_discarded(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Child A", 2020, "1st")
    add_child(db, "child-b", "Child B", 2020, "1st")
    harness = CuriosityHarness(
        db,
        reasoning_engine=ReasoningEngine(CandidateAndLegacyAttackBackend()),
    )

    result = harness.dispatch(
        Event(
            id="evt_phase0",
            type="child_question",
            child_id="child-a",
            text="Why does the Moon seem to move?",
        )
    )

    assert result.status == "completed"
    assert result.output["observation_candidates"] == [
        {"kind": "concept_mention", "label": "model-only candidate", "text": None}
    ]
    assert result.output["_reasoning"]["observation_candidate_status"] == "audit_only_phase1"
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM nodes WHERE label='model-only candidate'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM nodes WHERE label='forged learner state'").fetchone()[0] == 0
        observation = conn.execute(
            "SELECT child_id,event_id FROM observations WHERE event_id='evt_phase0'"
        ).fetchone()
        assert dict(observation) == {"child_id": "child-a", "event_id": "evt_phase0"}
        annotation = conn.execute(
            "SELECT child_id,event_id,author_type,authority FROM evidence_annotations"
        ).fetchone()
        assert dict(annotation) == {
            "child_id": "child-a",
            "event_id": "evt_phase0",
            "author_type": "model",
            "authority": "audit_only",
        }
        assert conn.execute("SELECT COUNT(*) FROM evidence_records").fetchone()[0] == 0


def test_internal_graph_mutations_enforce_child_ownership_and_reserved_state(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Child A")
    add_child(db, "child-b", "Child B")
    node_a = upsert_node(db, "child-a", "concept", "fractions")
    node_b = upsert_node(db, "child-b", "concept", "weather")

    with connect(db) as conn:
        with pytest.raises(ValueError, match="does not match"):
            apply_graph_mutation(
                conn,
                GraphMutation(
                    kind="upsert_node",
                    child_id="child-b",
                    node_kind="concept",
                    label="cross child",
                ),
                "child-a",
            )
        with pytest.raises(ValueError, match="projection code"):
            apply_graph_mutation(
                conn,
                GraphMutation(
                    kind="upsert_node",
                    node_kind="concept",
                    label="fractions",
                    state={"knowledge_state": "demonstrated"},
                ),
                "child-a",
            )
        with pytest.raises(ValueError, match="endpoints"):
            apply_graph_mutation(
                conn,
                GraphMutation(
                    kind="add_edge",
                    source_node_id=node_a,
                    target_node_id=node_b,
                    relation="related_to",
                ),
                "child-a",
            )


def test_legacy_learner_state_is_inspectable_but_quarantined_from_model_context(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Child A", 2020, "1st")
    first = upsert_node(
        db,
        "child-a",
        "concept",
        "fractions",
        confidence=0.99,
        state={"knowledge_state": "demonstrated", "epistemic_state": "established_pattern"},
    )
    second = upsert_node(
        db,
        "child-a",
        "interest",
        "math genius",
        confidence=0.99,
        state={"interest_phase": "sustained", "epistemic_state": "established_pattern"},
    )
    add_edge(db, "child-a", first, "is_mastered_by", second, {"confidence": 1, "authority": "model"})
    add_observation(
        db,
        "child-a",
        "model_claim",
        "Child A has mastered all fractions.",
        source="reasoning",
        confidence=1,
        metadata={"epistemic_state": "established_pattern"},
    )

    with connect(db) as conn:
        stored = json.loads(conn.execute("SELECT state_json FROM nodes WHERE id=?", (first,)).fetchone()[0])
    assert stored["knowledge_state"] == "demonstrated"

    context = build_context(
        str(db),
        "child-a",
        {"type": "child_question", "text": "How do fractions work?", "metadata": {}},
        depth=3,
    )
    assert context["observations"] == []
    assert context["nodes"] == []
    assert context["edges"] == []
    assert context["profile_projection"]["status"] == "shadow_phase1"
    assert context["profile_projection"]["may_affect_recommendations"] is False
    assert context["shadow_learner_snapshot"]["legacy_graph_included"] is False


def test_parent_agent_model_contract_cannot_author_hidden_interaction_payload() -> None:
    with pytest.raises(ValidationError):
        ParentAgentTurn.model_validate(
            {
                "message": "Choose the harmless-looking option.",
                "interaction": {
                    "kind": "choose_one",
                    "title": "Continue?",
                    "prompt": "Choose one.",
                    "options": [
                        {
                            "label": "Keep going",
                            "intent": "pause_schedule",
                            "payload": {"schedule_id": "other-family"},
                        }
                    ],
                },
            }
        )


def test_interaction_targets_require_matching_binding_session_and_event(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    setup = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "child-a", "Child A", 2020, "1st")
    now = utcnow()
    with connect(db) as conn:
        for binding_id, user_id in (("binding-a", "U1"), ("binding-b", "U2")):
            conn.execute(
                """INSERT INTO transport_bindings(
                       id,transport,team_id,user_id,channel_id,parent_id,status,created_at,updated_at
                   ) VALUES(?,'slack','T1',?,'C1',?,'active',?,?)""",
                (binding_id, user_id, setup["owner_id"], now, now),
            )

    result = CuriosityHarness(db, reasoning_engine=ReasoningEngine(StubBackend())).dispatch(
        Event(id="evt_bound", type="child_question", child_id="child-a", text="Why is ice slippery?")
    )
    other = CuriosityHarness(db, reasoning_engine=ReasoningEngine(StubBackend())).dispatch(
        Event(id="evt_unbound", type="child_question", child_id="child-a", text="Why is snow white?")
    )
    assert result.status == other.status == "completed"
    store = SessionStore(db)
    session = store.get_or_create(
        origin="slack",
        transport="slack",
        binding_id="binding-a",
        conversation_ref="conversation",
        thread_ref="thread",
        child_id="child-a",
    )
    store.append_message(
        str(session["id"]),
        role="assistant",
        content="Reviewed answer",
        event_id="evt_bound",
    )
    service = CuriosityService(db, tmp_path / "output")
    base = {
        "intent": "rate_response",
        "payload": {"event_id": "evt_bound", "rating": "helpful"},
    }

    with pytest.raises(ValueError, match="session is unavailable"):
        service.handle_interaction_choice(resolved=base, binding_id="binding-a", channel_id="C1")
    with pytest.raises(ValueError, match="session is unavailable"):
        service.handle_interaction_choice(
            resolved={**base, "session_id": session["id"]},
            binding_id="binding-b",
            channel_id="C1",
        )
    with pytest.raises(ValueError, match="target is unavailable"):
        service.handle_interaction_choice(
            resolved={
                "intent": "rate_response",
                "payload": {"event_id": "evt_unbound", "rating": "helpful"},
                "session_id": session["id"],
            },
            binding_id="binding-a",
            channel_id="C1",
        )

    stateful_choices = [
        {"intent": "cancel_tool_call", "payload": {"tool_call_id": "tool-other"}},
        {"intent": "approve_tool_call", "payload": {"tool_call_id": "tool-other"}},
        {"intent": "pause_schedule", "payload": {"schedule_id": "schedule-other"}},
        {"intent": "weekly_feedback", "payload": {"outcome": "helpful"}},
        {"intent": "revise_artifact", "payload": {"artifact_type": "challenge"}},
    ]
    for resolved in stateful_choices:
        with pytest.raises(ValueError, match="session is unavailable"):
            service.handle_interaction_choice(
                resolved=resolved,
                binding_id="binding-a",
                channel_id="C1",
            )
        with pytest.raises(ValueError, match="session is unavailable"):
            service.handle_interaction_choice(
                resolved={**resolved, "session_id": session["id"]},
                binding_id="binding-b",
                channel_id="C1",
            )


def test_output_feedback_is_auditable_but_never_model_profile_context(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Child A", 2020, "1st")
    result = CuriosityHarness(db, reasoning_engine=ReasoningEngine(StubBackend())).dispatch(
        Event(id="evt_feedback", type="child_question", child_id="child-a", text="Why does ice float?")
    )
    assert result.status == "completed"
    CuriosityService(db, tmp_path / "output").feedback(
        {
            "child_id": "child-a",
            "event_id": "evt_feedback",
            "outcome": "not_helpful",
            "note": "The generated answer was too long.",
            "source": "test_parent_rating",
        }
    )

    with connect(db) as conn:
        row = conn.execute(
            "SELECT metadata_json FROM observations WHERE child_id='child-a' AND kind='feedback'"
        ).fetchone()
    metadata = jload(row["metadata_json"])
    assert metadata["learning_scope"] == "diagnostic"
    assert metadata["independence_status"] == "excluded"
    assert metadata["profile_eligible"] is False

    context = build_context(
        str(db),
        "child-a",
        {"type": "child_question", "text": "What should we try next?", "metadata": {}},
        depth=4,
    )
    assert context["observations"] == []
    assert context["feedback_summary"] == []
