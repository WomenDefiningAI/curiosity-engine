from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from curiosity_engine.contracts import ParentAgentToolCall, ParentAgentTurn, PullThreadOutput, QuickAnswerOutput
from curiosity_engine.db import connect, jload, utcnow
from curiosity_engine.graph import add_child
from curiosity_engine.interaction import setup_household
from curiosity_engine.parent_agent import ParentAgentRuntime, _explicit_artifact_request, _weekly_checkin_draft
from curiosity_engine.presentation import format_quick_answer, format_reviewed_response
from curiosity_engine.reasoning import ReasoningEngine, StubBackend, _release_gate_requirements, load_policies
from curiosity_engine.scheduler import checkin_message_from_request
from curiosity_engine.service import (
    CuriosityService,
    _context_detail_excerpt,
    _explicit_thread_preference_change,
    _explicit_visual_request,
    _name_subject_in_revision,
    _proportionate_thread_message,
)
from curiosity_engine.tooling import ToolRegistry
from curiosity_engine.transports.slack import (
    CHILD_ADD_MODAL_CALLBACK,
    CHILD_ADD_OPEN_ACTION,
    SlackTransport,
    _child_management_blocks,
    _interaction_progress_copy,
    _make_child_add_open_receiver,
    _make_child_add_submission_receiver,
)


def _family(tmp_path: Path, *, with_child: bool = True) -> tuple[Path, str]:
    db = tmp_path / "private" / "curiosity.db"
    setup = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    if with_child:
        add_child(db, "kid-a", "Kid A", grade="1st")
    now = utcnow()
    binding_id = "binding-owner"
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO transport_bindings(
                   id,transport,team_id,user_id,channel_id,parent_id,status,created_at,updated_at
               ) VALUES(?,'slack','T1','U1','C1',?,'active',?,?)""",
            (binding_id, setup["owner_id"], now, now),
        )
    return db, binding_id


def _offline_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[CuriosityService, Path, str]:
    db, binding_id = _family(tmp_path)
    monkeypatch.setattr("curiosity_engine.service.configured_backend", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("curiosity_engine.runtime.configured_backend", lambda *_args, **_kwargs: None)
    return CuriosityService(db, tmp_path / "output"), db, binding_id


def _fixed_reviewed_answer(monkeypatch, answer):
    """Exercise general generation/lineage without relying on a topic-specific runtime shortcut."""
    class AnswerBackend(StubBackend):
        def complete(self, **kwargs):
            if kwargs['response_model'] is QuickAnswerOutput:
                return {'answer': answer, 'visual': None}
            return super().complete(**kwargs)
    backend = AnswerBackend()
    monkeypatch.setattr('curiosity_engine.runtime.configured_backend', lambda *_args, **_kwargs: backend)


def test_slack_thread_roots_are_independent_parent_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _db, binding_id = _offline_service(tmp_path, monkeypatch)

    first = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="slack-root-one",
        child_id="kid-a",
        text="Kid A noticed a robin today. Just save it.",
    )
    continuation = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="slack-root-one",
        child_id="kid-a",
        text="Kid A drew the robin later. Just save it.",
    )
    second = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="slack-root-two",
        child_id="kid-a",
        text="Kid A noticed a beetle today. Just save it.",
    )

    assert continuation["session_id"] == first["session_id"]
    assert second["session_id"] != first["session_id"]


def test_quick_answer_is_reviewed_but_retry_can_offer_a_learning_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.ask(
        child_id="kid-a",
        text="Why does ice float?",
        answer_mode="quick",
        context_metadata={"learning_scope": "family_signal"},
        enqueue_visual=False,
    )

    assert result["status"] == "completed"
    assert result["workflow"] == "quick_answer"
    QuickAnswerOutput.model_validate(
        {key: value for key, value in result["output"].items() if key not in {"_reasoning", "_pedagogy"}}
    )
    rendered = format_reviewed_response(result)
    assert "*Start here*" not in rendered
    assert "*Try it" not in rendered
    assert "ice" in rendered.casefold()
    with connect(db) as conn:
        stored = conn.execute("SELECT workflow FROM responses WHERE event_id=?", (result["event_id"],)).fetchone()
    assert stored["workflow"] == "quick_answer"
    policies = load_policies()
    assert policies["quick_answer"].critic_roles == ("critic_family_usefulness",)
    assert policies["pull_thread"].critic_roles == (
        "critic_family_usefulness",
        "critic_pedagogy",
    )

    retried = service.retry_response(
        source_event_id=str(result["event_id"]),
        event_id="evt_quick_retry",
    )
    assert retried["workflow"] == "pull_thread"
    assert retried["status"] == "completed", retried
    assert retried["output"]["physical_extension"]
    assert "visual_job_id" not in retried

    session_id = service.record_thread_response(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="quick-answer-thread",
        child_id="kid-a",
        user_text="Why does ice float?",
        result=result,
    )
    artifact_result = service.handle_interaction_choice(
        resolved={
            "intent": "create_artifact",
            "payload": {"event_id": result["event_id"], "artifact_type": "challenge"},
            "session_id": session_id,
        },
        binding_id=binding_id,
        channel_id="C1",
    )
    assert artifact_result["artifact"]["artifact_type"] == "challenge"
    assert Path(artifact_result["artifact"]["pdf_path"]).is_file()


def test_quick_answer_does_not_repeat_a_question_already_in_the_notice():
    question = "What do you think will happen next?"
    rendered = format_quick_answer(
        {
            "workflow": "quick_answer",
            "output": {
                "answer": "The ice changes as it warms.",
                "notice": f"Watch one edge. {question}",
                "question_back": question,
            },
        }
    )
    assert rendered.count(question) == 1


def test_natural_child_wants_an_idea_is_answered_instead_of_saved_as_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="quick-dinner-idea",
        child_id="kid-a",
        text="Kid A wants a five-minute curiosity idea while I make dinner. Nothing to buy, please.",
    )

    assert "saved the moment as context" not in result["message"].casefold()
    with connect(db) as conn:
        event = conn.execute("SELECT type FROM events WHERE id=?", (result["event_id"],)).fetchone()
        assert event["type"] == "child_question"
        assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


def test_saved_context_recall_preserves_the_child_and_full_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    original = (
        "I can't send the picture, but please save this photo note about Kid A: "
        "they are crouched beside a cardboard castle, wearing a paper crown, with a big grin. Just save it."
    )
    saved = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="faithful-context-recall",
        child_id="kid-a",
        text=original,
    )
    recalled = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="faithful-context-recall",
        text="What did you save from that picture note?",
    )

    assert "Kid A" in saved["message"]
    assert "cardboard castle" in saved["message"]
    assert original in recalled["message"]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0


def test_context_excerpt_uses_the_note_after_the_save_command() -> None:
    text = "I can't attach it, but please save this note about Ava: she made a cardboard castle."
    assert _context_detail_excerpt(text).startswith("this note about Ava")


def test_parent_teacher_writing_request_does_not_create_child_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="teacher-note",
        child_id="kid-a",
        text="Please write a two-sentence friendly note to Kid A's teacher asking whether a smock is needed tomorrow.",
    )

    assert result.get("event_id") is None
    with connect(db) as conn:
        for table in ("events", "responses", "evidence", "nodes"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_checkin_message_extracts_the_child_facing_school_question() -> None:
    request = "a weekly Wednesday evening check-in with Ava about what she wants help with at school"
    assert checkin_message_from_request(request, "Ava") == "What would you like help with at school this week?"


def test_explicit_no_activity_removes_only_appended_imperative_instructions() -> None:
    parsed = ReasoningEngine._validate_and_normalize_candidate(
        {
            "answer": (
                "Moving air carries crumbs toward the vacuum, like wind pushing a leaf across a sidewalk. "
                "Put a tissue on the table and blow beside it to test this."
            ),
            "question_back": "Can you try it?",
            "actions": [],
            "observation_candidates": [],
            "next_possible_concepts": [],
            "resource_refs": [],
            "visual": None,
        },
        QuickAnswerOutput,
        {
            "text": "Why does a vacuum pick up crumbs?",
            "metadata": {
                    "response_revision": {"parent_request": "No activity. Give one everyday comparison."}
            },
        },
    )

    assert parsed.answer == "Moving air carries crumbs toward the vacuum, like wind pushing a leaf across a sidewalk."
    assert parsed.question_back is None


def test_different_idea_written_in_show_becomes_a_structured_activity() -> None:
    parsed = ReasoningEngine._validate_and_normalize_candidate(
        {
            "hook": "Let’s try a moving-surface comparison instead.",
            "show": (
                "Put the top on a tray. An adult spins it, then slowly slides the tray and stops. "
                "Next, try the same move with the top not spinning."
            ),
            "ask": "What changed when the top was spinning?",
            "nugget": "Fast spinning makes a top resist being tipped in a new direction.",
            "actions": [],
            "observation_candidates": [],
            "next_possible_concepts": [],
            "resource_refs": [],
            "physical_extension": None,
            "visual": None,
        },
        PullThreadOutput,
        {
            "text": "Why can a spinning top stand up?",
            "metadata": {
                "response_revision": {
                    "parent_request": "Give us a genuinely different idea, not another version of watching it spin.",
                    "previous_reviewed_response": {"physical_extension": {"title": "Watch the top"}},
                }
            },
        },
    )

    assert parsed.physical_extension is not None
    assert "Put the top on a tray" in parsed.physical_extension.instructions[0]


def test_context_can_source_an_artifact_without_becoming_graph_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    context = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="thread-1",
        child_id="kid-a",
        text="Kid A's teacher update says the class practiced comparing two story endings.",
    )
    assert "context" in context["message"].casefold()
    context_event_id = str(context["event_id"])

    with connect(db) as conn:
        event = conn.execute("SELECT type,metadata_json FROM events WHERE id=?", (context_event_id,)).fetchone()
        assert event["type"] == "context_input"
        assert jload(event["metadata_json"])["evidence_eligible"] is False
        assert conn.execute("SELECT COUNT(*) FROM responses WHERE event_id=?", (context_event_id,)).fetchone()[0] == 0
        for table in ("observations", "nodes", "edges", "episodes", "evidence"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

    artifact = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="thread-1",
        text="Make a playful challenge from that teacher update.",
    )
    assert artifact["status"] == "completed"
    assert artifact["artifact"]["artifact_type"] == "challenge"
    assert Path(artifact["artifact"]["pdf_path"]).is_file()
    with connect(db) as conn:
        source = conn.execute(
            "SELECT source_event_id FROM experiences WHERE id=?",
            (artifact["artifact"]["experience_id"],),
        ).fetchone()
        assert source["source_event_id"] == context_event_id
        for table in ("observations", "nodes", "edges", "episodes", "evidence"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_parent_observation_is_context_not_an_unsolicited_lesson(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="observation-thread",
        child_id="kid-a",
        text="Kid A has been listening to dinosaur stories all week.",
    )
    assert result["message"] == "Thanks for sharing that moment. I saved it as context for this thread."
    interaction = result["interaction"]
    assert interaction["kind"] == "continue_thread"
    assert [option["intent"] for option in interaction["options"]] == [
        "remember_context",
        "explain_context",
        "suggest_context_idea",
        "create_context_artifact",
    ]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


def test_ice_cup_context_gets_one_reviewed_idea_without_model_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    _fixed_reviewed_answer(monkeypatch, "Put one ice cup in sun and another in shade. Which melts faster?")
    context = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="ice-cup-idea",
        child_id="kid-a",
        text="Kid A keeps making giant ice cubes in plastic cups.",
    )
    followup = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="ice-cup-idea",
        text="give us one simple idea",
    )

    assert followup["status"] == "completed" and "Which melts faster?" in followup["message"]
    assert followup["event_id"] != context["event_id"]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1
        metadata = jload(conn.execute("SELECT metadata_json FROM events WHERE id=?", (followup["event_id"],)).fetchone()[0])
        assert metadata["learning_scope"] == "diagnostic"
        assert conn.execute("""SELECT COUNT(*) FROM evidence e JOIN episode_memberships m ON m.event_id=e.event_id
            WHERE m.learning_scope='family_signal'""").fetchone()[0] == 0


def test_context_shortcuts_preserve_the_non_evidence_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    context = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="ice-thread",
        child_id="kid-a",
        text="Kid A is still making giant ice cubes in cups for fun.",
    )
    assert context["message"] == "That sounds fun. I saved the moment as context for this thread."
    event_id = str(context["event_id"])
    session_id = str(context["session_id"])

    remembered = service.handle_interaction_choice(
        resolved={
            "intent": "remember_context",
            "payload": {"event_id": event_id},
            "session_id": session_id,
        },
        binding_id=binding_id,
        channel_id="C1",
    )
    assert remembered["message"] == "Done—I’ll keep it as context for this thread."

    explained = service.handle_interaction_choice(
        resolved={
            "intent": "explain_context",
            "payload": {"event_id": event_id},
            "session_id": session_id,
        },
        binding_id=binding_id,
        channel_id="C1",
    )
    assert explained["status"] == "completed"
    assert explained["event_id"] != event_id
    assert explained["visual_job_id"] is None

    artifact = service.handle_interaction_choice(
        resolved={
            "intent": "create_context_artifact",
            "payload": {"event_id": event_id, "artifact_type": "challenge"},
            "session_id": session_id,
        },
        binding_id=binding_id,
        channel_id="C1",
    )
    assert Path(artifact["artifact"]["pdf_path"]).is_file()
    with connect(db) as conn:
        source = conn.execute(
            "SELECT source_event_id FROM experiences WHERE id=?",
            (artifact["artifact"]["experience_id"],),
        ).fetchone()
        assert source["source_event_id"] == event_id
        response_event = conn.execute(
            "SELECT metadata_json FROM events WHERE id=?",
            (explained["event_id"],),
        ).fetchone()
        response_metadata = jload(response_event["metadata_json"])
        assert response_metadata["learning_scope"] == "diagnostic"
        assert response_metadata["context_source_event_id"] == event_id
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM visual_jobs WHERE event_id=?",
                (explained["event_id"],),
            ).fetchone()[0]
            == 0
        )
        for table in ("observations", "nodes", "edges"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_context_reference_preference_uses_bound_parent_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    context = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="context-preference-thread",
        child_id="kid-a",
        text="Kid A is still drawing picture answers instead of writing words for fun.",
    )
    assert "context" in context["message"].casefold()
    assert context.get("event_id")
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="context-preference-thread",
        text="Please use that later when you suggest activities.",
    )

    assert result["status"] == "completed"
    with connect(db) as conn:
        row = conn.execute(
            "SELECT value FROM parent_memories WHERE status='active' ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
    assert "drawing picture answers instead of writing words" in str(row["value"])
    assert str(row["value"]) != "Please use that later when you suggest activities."


def test_challenge_after_context_idea_uses_parent_visible_lineage_not_router_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    context = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="build-team-thread",
        child_id="kid-a",
        text="Kid A has been building LEGO creatures after school with a sibling and friend.",
    )
    idea = service.handle_interaction_choice(
        resolved={
            "intent": "suggest_context_idea",
            "payload": {"event_id": context["event_id"]},
            "session_id": context["session_id"],
        },
        binding_id=binding_id,
        channel_id="C1",
    )
    from curiosity_engine.sessions import SessionStore

    SessionStore(db).append_message(
        str(context["session_id"]),
        role="assistant",
        content=str(idea["message"]),
        kind="interaction_result",
        event_id=str(idea["event_id"]),
    )
    artifact = service.handle_interaction_choice(
        resolved={
            "intent": "create_artifact",
            "payload": {"event_id": idea["event_id"], "artifact_type": "challenge"},
            "session_id": context["session_id"],
        },
        binding_id=binding_id,
        channel_id="C1",
    )

    assert Path(artifact["artifact"]["pdf_path"]).is_file()
    with connect(db) as conn:
        stored = conn.execute(
            "SELECT title,spec_json FROM experiences WHERE id=?",
            (artifact["artifact"]["experience_id"],),
        ).fetchone()
    rendered = str(stored["spec_json"]).casefold()
    assert stored["title"] in {"Build-Team Idea Mission", "Design an Idea Mission"}
    assert "lego" in rendered
    assert "offer exactly one playful" not in rendered
    assert "parent-shared moment" not in rendered
    assert "water to ice" not in rendered
    assert "version a" not in rendered


def test_explicit_remember_only_confirmation_has_no_creation_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    context = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="remember-only-thread",
        child_id="kid-a",
        text="Kid A has been drawing tiny maps of the house.",
    )
    remembered = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="remember-only-thread",
        text="just remember this, don't make anything",
    )

    assert remembered["event_id"] == context["event_id"]
    assert remembered["message"] == "Done—I’ll keep it as context for this thread and won’t make anything from it."
    assert remembered.get("interaction") is None
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("intent", "working", "completed"),
    [
        ("explain_context", "Explaining that moment", "Explanation ready"),
        ("suggest_context_idea", "Thinking of one playful idea", "Idea ready"),
        ("create_context_artifact", "Drawing and building the challenge", "Visual challenge ready"),
    ],
)
def test_context_shortcuts_show_visible_progress(intent: str, working: str, completed: str):
    resolved = {"intent": intent, "payload": {}}
    assert working in str(_interaction_progress_copy(resolved))
    assert completed in str(_interaction_progress_copy(resolved, completed=True))


def test_vague_visual_followup_does_not_repeat_the_prior_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    first = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="visual-followup",
        child_id="kid-a",
        text="Why does water get bigger when it freezes?",
        answer_mode_hint="quick",
    )
    followup = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="visual-followup",
        text="better viz please",
    )

    assert "need to know what it should help" in followup["message"]
    assert followup["message"] != first["message"]
    assert [option["intent"] for option in followup["interaction"]["options"]] == [
        "create_artifact",
        "create_artifact",
    ]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


def test_vague_visual_followup_after_context_idea_keeps_the_idea_revisable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    _fixed_reviewed_answer(monkeypatch, "Put one ice cup in sun and another in shade. Which melts faster?")
    context = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="context-visual-followup",
        child_id="kid-a",
        text="Kid A is making giant ice in cups again.",
    )
    idea = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="context-visual-followup",
        text="give us one playful idea",
    )
    followup = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="context-visual-followup",
        text="better viz please",
    )

    assert idea["status"] == "completed" and "Which melts faster?" in idea["message"]
    assert followup["event_id"] == idea["event_id"]
    assert "need to know what it should help" in followup["message"]
    assert [option["intent"] for option in followup["interaction"]["options"]] == [
        "create_artifact",
        "create_artifact",
    ]

    activity = service.handle_interaction_choice(
        resolved={
            "intent": "create_artifact",
            "payload": {"event_id": context["event_id"], "artifact_type": "activity"},
            "session_id": context["session_id"],
        },
        binding_id=binding_id,
        channel_id="C1",
    )
    assert activity["artifact"]["artifact_type"] == "activity"
    assert Path(activity["artifact"]["pdf_path"]).is_file()
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1
        assert conn.execute("""SELECT COUNT(*) FROM evidence e JOIN episode_memberships m ON m.event_id=e.event_id
            WHERE m.learning_scope='family_signal'""").fetchone()[0] == 0


def test_explicit_step_visual_followup_creates_an_activity_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _db, binding_id = _offline_service(tmp_path, monkeypatch)
    service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="step-card",
        child_id="kid-a",
        text="How can we test ice melting?",
        answer_mode_hint="explore",
    )
    followup = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="step-card",
        text="show the activity steps",
    )

    assert followup["artifact"]["artifact_type"] == "activity"
    assert Path(followup["artifact"]["pdf_path"]).is_file()


def test_mystery_revision_changes_the_play_loop_not_only_the_theme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _db, binding_id = _offline_service(tmp_path, monkeypatch)
    service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="float-mystery",
        child_id="kid-a",
        text="Why do some things float and others sink?",
        answer_mode_hint="explore",
    )
    revision = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="float-mystery",
        text="make it feel like a mystery, not a worksheet",
    )

    assert revision["message"].startswith("*The Case of the Floating Suspects*")
    assert "inspect the clues" in revision["message"]
    assert "Predict *float* or *sink*" in revision["message"]
    assert "reveal the result" in revision["message"]
    assert "worksheet" not in revision["message"].casefold()


def test_negative_feedback_is_recorded_and_repaired_briefly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="shorter-feedback",
        child_id="kid-a",
        text="Why do leaves change color?",
        answer_mode_hint="quick",
    )
    followup = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="shorter-feedback",
        text="thumbs down. too much reading",
    )

    assert followup["message"].startswith("Got it—shorter:")
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1


def test_request_for_first_step_reuses_one_reviewed_activity_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    first = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="first-step-thread",
        child_id="kid-a",
        text="Give us a game for noticing how shadows change.",
        answer_mode_hint="explore",
    )
    followup = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="first-step-thread",
        text="too much. just give me the first thing to do",
    )

    assert followup["message"].startswith("Start with this:")
    assert not any(term in followup["message"].casefold() for term in ("print", "cut out", "draw the signs"))
    assert "place them" not in followup["message"].casefold()
    assert followup["event_id"] == first["event_id"]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1


def test_rejected_paper_wing_test_gets_a_different_reviewed_mechanic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    first = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="different-wing-thread",
        child_id="kid-a",
        text="How do airplane wings help an airplane stay up?",
        answer_mode_hint="explore",
    )
    followup = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="different-wing-thread",
        text="another idea, but do not repeat the paper wing test",
    )

    assert followup["event_id"] == first["event_id"]
    assert followup["message"].startswith("*Pilot’s Flap Mystery*")
    assert "paper wing" not in followup["message"].casefold()
    assert "more lift at slower speeds" in followup["message"].casefold()
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 1


def test_context_plus_question_gets_reviewed_help_without_graph_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="mixed-context-thread",
        child_id="kid-a",
        text=(
            "Kid A's teacher update says the class compared two story endings. "
            "How can we explore why the endings feel different?"
        ),
    )
    assert result["status"] == "completed"
    with connect(db) as conn:
        rows = conn.execute("SELECT type,metadata_json FROM events ORDER BY created_at").fetchall()
        assert [row["type"] for row in rows] == ["context_input", "child_question"]
        assert jload(rows[1]["metadata_json"])["learning_scope"] == "diagnostic"
        for table in ("observations", "nodes", "edges"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
        raw_evidence = conn.execute(
            "SELECT metadata_json FROM evidence WHERE event_id=?",
            (result["event_id"],),
        ).fetchone()
        assert jload(raw_evidence["metadata_json"])["learning_scope"] == "diagnostic"
        membership = conn.execute(
            "SELECT learning_scope,independence_status FROM episode_memberships WHERE event_id=?",
            (result["event_id"],),
        ).fetchone()
        assert dict(membership) == {"learning_scope": "diagnostic", "independence_status": "diagnostic"}


class _DirectReplyBackend:
    name = "test"
    model = "test-v1"

    def complete(self, **_kwargs: Any) -> dict[str, Any]:
        return ParentAgentTurn(message="A direct unreviewed answer.").model_dump(mode="json")


class _ParaphrasedRevisionBackend:
    name = "test"
    model = "test-v1"

    def complete(self, **_kwargs: Any) -> dict[str, Any]:
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="revise_learning_thread",
                    arguments={"revision": "Turn this into a generic worksheet."},
                    rationale="The parent requested a revision.",
                )
            ]
        ).model_dump(mode="json")


class _ArtifactMisrouteBackend:
    name = "test"
    model = "test-v1"

    def complete(self, **_kwargs: Any) -> dict[str, Any]:
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="create_learning_artifact",
                    arguments={"artifact_type": "activity", "revision": "Make an activity."},
                    rationale="The planner mistakenly assumed an existing source.",
                )
            ]
        ).model_dump(mode="json")


def test_parent_exact_revision_overrides_planner_paraphrase(tmp_path: Path):
    db, binding_id = _family(tmp_path)
    from curiosity_engine.sessions import SessionStore

    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref="conversation",
        thread_ref="thread",
        child_id="kid-a",
    )
    calls: list[dict[str, Any]] = []
    tools = ToolRegistry()
    tools.register(
        "revise_learning_thread",
        lambda arguments: calls.append(arguments) or {"status": "completed", "message": "Revised."},
    )
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_prior','child_question','kid-a','Why?','test','{}',?,'completed')""",
            (utcnow(),),
        )
    ParentAgentRuntime(db, backend=_ParaphrasedRevisionBackend(), tools=tools).run(
        session_id=session["id"],
        user_message="Please explain it as a detective story with one clue at a time.",
        child={"id": "kid-a", "grade": "1st"},
        latest_event_id="evt_prior",
    )

    assert calls == [{"revision": "Please explain it as a detective story with one clue at a time."}]


def test_new_learning_request_recovers_from_revision_misroute(tmp_path: Path):
    db, binding_id = _family(tmp_path)
    from curiosity_engine.sessions import SessionStore

    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref="conversation",
        thread_ref="new-game-thread",
        child_id="kid-a",
    )
    calls: list[dict[str, Any]] = []
    tools = ToolRegistry()
    tools.register(
        "continue_learning_thread",
        lambda arguments: calls.append(arguments) or {"status": "completed", "message": "Reviewed game."},
    )
    result = ParentAgentRuntime(db, backend=_ArtifactMisrouteBackend(), tools=tools).run(
        session_id=session["id"],
        user_message="Give us a game for noticing how shadows change.",
        child={"id": "kid-a", "grade": "1st"},
        latest_event_id=None,
    )

    assert result["message"] == "Reviewed game."
    assert calls == [{"message": "Give us a game for noticing how shadows change.", "mode": "explore"}]


def test_just_answer_reuses_reviewed_core_without_activity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _db, binding_id = _offline_service(tmp_path, monkeypatch)
    first = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="direct-answer-thread",
        child_id="kid-a",
        text="What are the biggest robots in the world?",
        answer_mode_hint="quick",
    )

    followup = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="direct-answer-thread",
        text="just answer him, no activity",
    )

    assert followup["status"] == "completed"
    assert followup["event_id"] == first["event_id"]
    assert "*Try it" not in followup["message"]
    assert "materials" not in followup["message"].casefold()


def test_mystery_not_worksheet_creates_topic_aligned_activity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service, _db, binding_id = _offline_service(tmp_path, monkeypatch)
    service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="mystery-thread",
        child_id="kid-a",
        text="Why do some things float and others sink?",
        answer_mode_hint="explore",
    )

    followup = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="mystery-thread",
        text="make it feel like a mystery, not a worksheet",
    )

    assert followup["status"] == "completed"
    assert followup.get("artifact") is None
    assert "printable" not in followup["message"].casefold()


def test_explicit_answer_hint_cannot_bypass_the_reviewed_answer_tool(tmp_path: Path):
    db, binding_id = _family(tmp_path)
    from curiosity_engine.sessions import SessionStore

    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref="conversation",
        thread_ref="thread",
        child_id="kid-a",
    )
    calls: list[dict[str, Any]] = []
    tools = ToolRegistry()
    tools.register(
        "continue_learning_thread",
        lambda arguments: calls.append(arguments) or {"status": "completed", "message": "A reviewed answer."},
    )
    result = ParentAgentRuntime(db, backend=_DirectReplyBackend(), tools=tools).run(
        session_id=session["id"],
        user_message="Why does ice float?",
        child={"id": "kid-a", "grade": "1st"},
        latest_event_id=None,
        answer_mode_hint="explore",
    )
    assert result["message"] == "A reviewed answer."
    assert calls == [{"message": "Why does ice float?", "mode": "explore"}]


def test_clear_natural_child_question_cannot_bypass_reviewed_generation(tmp_path: Path):
    db, binding_id = _family(tmp_path)
    from curiosity_engine.sessions import SessionStore

    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref="conversation",
        thread_ref="thread",
        child_id="kid-a",
    )
    calls: list[dict[str, Any]] = []
    tools = ToolRegistry()
    tools.register(
        "continue_learning_thread",
        lambda arguments: calls.append(arguments) or {"status": "completed", "message": "A reviewed quick answer."},
    )
    result = ParentAgentRuntime(db, backend=_DirectReplyBackend(), tools=tools).run(
        session_id=session["id"],
        user_message="Kid A asked: why does ice float?",
        child={"id": "kid-a", "grade": "1st"},
        latest_event_id=None,
    )
    assert result["message"] == "A reviewed quick answer."
    assert calls == [{"message": "Kid A asked: why does ice float?", "mode": "quick"}]


class _ModalClient:
    def __init__(self):
        self.views: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    def views_open(self, **kwargs: Any) -> None:
        self.views.append(kwargs)

    def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.messages.append(kwargs)
        return {"ts": "1.1"}


def test_owner_can_add_child_from_slack_form_without_model_or_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db, binding_id = _family(tmp_path, with_child=False)
    monkeypatch.setattr("curiosity_engine.service.configured_backend", lambda *_args, **_kwargs: None)
    service = CuriosityService(db, tmp_path / "output")
    transport = SlackTransport(db, tmp_path / "output", service=service)
    client = _ModalClient()
    blocks = _child_management_blocks([])
    assert blocks[1]["elements"][0]["action_id"] == CHILD_ADD_OPEN_ACTION

    opened: list[bool] = []
    _make_child_add_open_receiver(transport, db)(
        lambda: opened.append(True),
        {
            "trigger_id": "trigger-1",
            "team": {"id": "T1"},
            "user": {"id": "U1"},
            "channel": {"id": "C1"},
            "message": {"ts": "10.1"},
        },
        {"value": "open"},
        client,
    )
    assert opened == [True]
    view = client.views[0]["view"]
    assert view["callback_id"] == CHILD_ADD_MODAL_CALLBACK
    view.update(
        {
            "id": "view-1",
            "state": {
                "values": {
                    "child_name": {"value": {"value": "Demo Child"}},
                    "child_grade": {"value": {"value": "1st"}},
                    "child_birth_year": {"value": {"value": "2020"}},
                }
            },
        }
    )
    submissions: list[dict[str, Any]] = []

    def acknowledge(**kwargs: Any) -> None:
        submissions.append(kwargs)

    _make_child_add_submission_receiver(transport, db)(
        acknowledge,
        {"team": {"id": "T1"}, "user": {"id": "U1"}},
        view,
        client,
    )
    assert submissions == [{}]
    assert service.children()[0]["name"] == "Demo Child"
    assert "starts empty" in client.messages[-1]["text"]
    with connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM interaction_audit WHERE action='child_profile_created'").fetchone()[0]
            == 1
        )
        for table in ("observations", "nodes", "edges", "episodes", "evidence"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

    _make_child_add_submission_receiver(transport, db)(
        acknowledge,
        {"team": {"id": "T1"}, "user": {"id": "U1"}},
        view,
        client,
    )
    assert len(service.children()) == 1
    with connect(db) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM interaction_audit WHERE action='child_profile_created'").fetchone()[0]
            == 1
        )


def test_first_turn_remember_only_confirms_detail_without_creation_menu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="remember-specific-detail",
        child_id="kid-a",
        text="Kid A keeps humming a three-note tune. Please just remember that; I do not need ideas.",
    )

    assert "humming a three-note tune" in result["message"]
    assert "won’t make anything" in result["message"]
    assert result.get("interaction") is None
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0


@pytest.mark.parametrize(
    "message",
    [
        "Just explain it aloud—no chart or worksheet.",
        "Do not make a printable; give us a quiet game instead.",
        "The picture is not useful, but do not replace it with a PDF.",
    ],
)
def test_artifact_intent_requires_a_positive_tangible_request(message: str):
    assert not _explicit_artifact_request(message)
    turn = ParentAgentTurn(
        tool_calls=[
            ParentAgentToolCall(
                name="create_learning_artifact",
                arguments={"artifact_type": "worksheet", "revision": message},
                rationale="model routing slip",
            )
        ]
    )
    normalized = ParentAgentRuntime._normalize_artifact_intent(
        turn,
        {"message": message, "session": {"latest_event_id": "evt-prior"}},
    )
    assert [call.name for call in normalized.tool_calls] == ["revise_learning_thread"]


def test_recording_aid_is_a_tangible_worksheet_request():
    message = "Make a small recording aid with three boxes for what we hear."
    assert _explicit_artifact_request(message)
    turn = ParentAgentTurn(
        tool_calls=[
            ParentAgentToolCall(
                name="create_learning_artifact",
                arguments={"artifact_type": "challenge", "revision": "generic"},
                rationale="model chose the wrong artifact type",
            )
        ]
    )
    normalized = ParentAgentRuntime._normalize_artifact_intent(turn, {"message": message, "session": {}})
    assert normalized.tool_calls[0].arguments["artifact_type"] == "worksheet"
    assert normalized.tool_calls[0].arguments["revision"] == message


def test_specific_visual_request_can_build_from_saved_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    context = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="context-comparison-page",
        child_id="kid-a",
        text="Kid A noticed a smooth pebble beside a rough piece of bark.",
    )
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="context-comparison-page",
        text="Make a simple side-by-side diagram showing how they are alike and different.",
    )

    assert result["artifact"]["artifact_type"] == "reference"
    with connect(db) as conn:
        source = conn.execute(
            "SELECT source_event_id FROM experiences WHERE id=?",
            (result["artifact"]["experience_id"],),
        ).fetchone()
        assert source["source_event_id"] == context["event_id"]
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0


def test_self_contained_artifact_request_reuses_reviewed_answer_not_a_new_context_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    answer = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="reviewed-layer-comparison",
        child_id="kid-a",
        text="Kid A asked why layered clothing keeps us warm.",
    )
    artifact = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="reviewed-layer-comparison",
        text=(
            "Please make a side-by-side diagram comparing layered clothing with a blanket. "
            "Labels should be easy enough for Kid A to read."
        ),
    )

    assert artifact["artifact"]["artifact_type"] == "reference"
    with connect(db) as conn:
        source = conn.execute(
            "SELECT source_event_id FROM experiences WHERE id=?",
            (artifact["artifact"]["experience_id"],),
        ).fetchone()
        assert source["source_event_id"] == answer["event_id"]
        assert conn.execute("SELECT COUNT(*) FROM events WHERE type='context_input'").fetchone()[0] == 0


def test_schedule_clarification_stays_parent_chat_without_child_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    proposed = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="schedule-preview",
        child_id="kid-a",
        text="Could you check in every Thursday with one five-minute idea?",
    )
    assert proposed["interaction"]["kind"] == "confirm_action"
    clarified = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="schedule-preview",
        text="What would that weekly check-in look like?",
    )

    assert "still only a preview" in clarified["message"]
    assert clarified.get("event_id") is None
    with connect(db) as conn:
        for table in ("observations", "nodes", "edges", "evidence"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_temporary_preference_phrases_are_explicit_and_reversible():
    assert _explicit_thread_preference_change("For this chat, give one step at a time.", "set")
    assert _explicit_thread_preference_change("Next time, start with one sentence.", "set")
    assert _explicit_thread_preference_change("Normal short explanations are fine again.", "clear")
    assert _explicit_thread_preference_change(
        "Actually, switch back to regular paragraphs for the rest of this chat.", "clear"
    )
    assert _explicit_thread_preference_change("For this conversation, use shorter answers when I ask for help.", "set")


def test_narrowing_a_thread_preference_updates_instead_of_clearing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    first = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="conditional-preference",
        child_id="kid-a",
        text="For this conversation, use shorter answers when I ask for help.",
    )
    assert "saved" in first["message"].casefold()

    refined = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="conditional-preference",
        child_id="kid-a",
        text="Actually, only keep them short when I say quick—otherwise normal length is fine.",
    )

    assert "only keep them short when i say quick" in refined["message"].casefold()
    assert "cleared" not in refined["message"].casefold()
    with connect(db) as conn:
        latest = conn.execute(
            """SELECT status,value FROM parent_memories
               WHERE scope_type='thread' AND category='answer_style'
               ORDER BY version DESC LIMIT 1"""
        ).fetchone()
    assert latest["status"] == "active"
    assert "only keep them short when i say quick" in latest["value"].casefold()


def test_saved_preference_text_never_leaks_into_child_evidence_and_can_be_forgotten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    preference_text = "For this conversation, keep future answers to exactly three short sentences."
    saved = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="private-preference-lineage",
        child_id="kid-a",
        text=preference_text,
    )
    assert "saved" in saved["message"].casefold()

    answered = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="private-preference-lineage",
        child_id="kid-a",
        text="Why do some leaves turn red?",
    )
    assert answered["status"] == "completed"
    with connect(db) as conn:
        child_event = conn.execute(
            "SELECT metadata_json FROM events WHERE id=?",
            (answered["event_id"],),
        ).fetchone()
        evidence_rows = conn.execute(
            "SELECT metadata_json FROM evidence WHERE event_id=?",
            (answered["event_id"],),
        ).fetchall()
    assert preference_text not in child_event["metadata_json"]
    assert all(preference_text not in row["metadata_json"] for row in evidence_rows)

    cleared = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="private-preference-lineage",
        child_id="kid-a",
        text="Normal answers are fine again; forget that three-sentence preference.",
    )
    assert "cleared" in cleared["message"].casefold()
    with connect(db) as conn:
        current = conn.execute(
            """SELECT status FROM parent_memories
               WHERE scope_type='thread' AND category='answer_style'
               ORDER BY version DESC LIMIT 1"""
        ).fetchone()
    assert current["status"] == "forgotten"


def test_note_handling_question_stays_operational_and_never_becomes_child_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="note-handling",
        child_id="kid-a",
        text=(
            "Can you keep track of short notes within this chat and tell me whether something is just being saved "
            "or is a question you can answer?"
        ),
    )

    assert "Save this for later" in result["message"]
    assert "Ask a direct question" in result["message"]
    with connect(db) as conn:
        for table in ("events", "observations", "nodes", "edges", "evidence"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_reported_child_reaction_is_acknowledged_without_reteaching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="reaction-only",
        child_id="kid-a",
        text="Kid A listened, then said, ‘Not heat? Huh.’",
    )

    assert "without turning it into another lesson" in result["message"]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0


def test_schedule_approval_accepts_plain_approval_of_the_pending_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    proposed = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="approve-version",
        child_id="kid-a",
        text=(
            "I am considering a recurring Sunday evening check-in for Kid A, where you ask me for one small "
            "family highlight. Don't create it yet; show me the wording first."
        ),
    )
    assert proposed["status"] == "awaiting_approval"
    refined = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="approve-version",
        text="Make it a reminder to me only, and don't treat whatever I share as a report card.",
    )
    assert "for the parent" in refined["message"]
    assert "not a report card" in refined["message"]

    approved = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="approve-version",
        text="Yes, I approve that version.",
    )
    assert approved["status"] == "completed"
    assert approved["schedule"]["weekday"] == "sunday"
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM schedules WHERE enabled=1").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0


def test_unsupported_monthly_checkin_fails_honestly_after_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    proposed = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="monthly-limitation",
        child_id="kid-a",
        text="I'm considering a recurring check-in. Please don't start anything yet.",
    )
    assert proposed["status"] == "awaiting_approval"
    clarified = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="monthly-limitation",
        text="Monthly sounds good—maybe the first Monday. That is not approval yet.",
    )
    assert "currently support weekly" in clarified["message"]
    assert "nothing changed" in clarified["message"]
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="monthly-limitation",
        text="Yes, please start that monthly check-in.",
    )

    assert "isn’t available yet" in result["message"]
    assert "did not turn it on" in result["message"]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM schedules WHERE enabled=1").fetchone()[0] == 0


def test_explicit_artifact_request_creates_and_revises_the_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _db, binding_id = _offline_service(tmp_path, monkeypatch)
    created = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="cut-play-artifact",
        child_id="kid-a",
        text="Can you make a goofy printable for Kid A about what to do when a task feels frustrating?",
    )
    assert created["artifact"]["artifact_type"] == "worksheet"

    revised = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="cut-play-artifact",
        text="Make it a cut-and-play set, not a poster.",
    )
    assert revised["artifact"]["artifact_type"] == "activity"
    assert revised["artifact"]["experience_id"] != created["artifact"]["experience_id"]


def test_brief_spoken_answers_and_future_activity_framing_are_valid_preferences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    brief = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="future-framing",
        child_id="kid-a",
        text="Keep answers this brief when I ask for something to say out loud.",
    )
    assert "saved" in brief["message"].casefold()

    framing = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="future-framing",
        text="If we try clay someday, offer it as something to explore, not because you think Kid A knows it.",
    )
    assert "saved" in framing["message"].casefold()
    with connect(db) as conn:
        rows = conn.execute(
            """SELECT category,status,value FROM parent_memories
               WHERE scope_type='thread' AND status='active' ORDER BY created_at,memory_key"""
        ).fetchall()
        assert [(row["category"], row["status"]) for row in rows] == [
            ("answer_style", "active"),
            ("activity_style", "active"),
        ]
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0


def test_explicit_quick_spoken_answer_omits_optional_extras_and_balances_quote():
    rendered = format_quick_answer(
        {
            "workflow": "quick_answer",
            "output": {
                "answer": "“These marks help us find the same spot again.",
                "notice": "Try a scavenger hunt.",
                "question_back": "Can you find another?",
                "resource_refs": ["https://example.test/source"],
            },
            "visual_job_id": "vis_pending",
        },
        request="Quick answer please—just one sentence I can say aloud.",
    )

    assert rendered == "“These marks help us find the same spot again.”"


def test_unknown_visual_feedback_asks_for_the_missing_purpose_without_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="missing-visual-context",
        child_id="kid-a",
        text="That diagram was too cluttered. Can you make it clearer?",
    )

    assert "don’t have that earlier picture" in result["message"]
    assert "Upload it" in result["message"]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0


def test_initial_artifact_request_creates_a_non_evidence_source_and_reference_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="initial-reference",
        child_id="kid-a",
        text="Make a side-by-side diagram comparing a smooth shell and a jagged rock.",
    )

    assert result["artifact"]["artifact_type"] == "reference"
    refined = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="initial-reference",
        text="Please use short labels and arrows beside each surface.",
    )
    assert refined["artifact"]["artifact_type"] == "reference"
    with connect(db) as conn:
        event = conn.execute("SELECT type,metadata_json FROM events").fetchone()
        assert event["type"] == "context_input"
        assert jload(event["metadata_json"])["evidence_eligible"] is False
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_tiny_model_activity_keeps_the_explanatory_bridge():
    rendered = _proportionate_thread_message(
        {
            "status": "completed",
            "workflow": "pull_thread",
            "output": {
                "hook": "Use one orange as a model.",
                "show": "Point from several places toward the middle.",
                "ask": "What stays the same?",
                "nugget": (
                    "The pointing direction represents inward; the orange does not create the real force, so it is "
                    "only a direction model."
                ),
                "physical_extension": {
                    "title": "Point Inward",
                    "materials": ["one orange"],
                    "instructions": ["Touch three places.", "At each place, point toward the middle."],
                },
            },
        },
        "Can we try one small thing with an orange?",
    )

    assert "1. Touch three places." in rendered
    assert "*What this shows:*" in rendered
    assert "only a direction model" in rendered


def test_parent_requested_practice_is_not_child_learning_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    _fixed_reviewed_answer(monkeypatch, "How many fingers are on one hand? How many wheels on a bicycle? "
                           "How many corners on a triangle? How many corners on a square?")
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="parent-practice-request",
        child_id="kid-a",
        text="Make four counting questions for Kid A using numbers under ten.",
    )

    assert result["status"] == "completed"
    with connect(db) as conn:
        event = conn.execute("SELECT metadata_json FROM events").fetchone()
        assert jload(event["metadata_json"])["learning_scope"] == "diagnostic"
        assert (
            conn.execute(
                """SELECT COUNT(*) FROM evidence e JOIN episode_memberships m ON m.event_id=e.event_id
                   WHERE m.learning_scope='family_signal'"""
            ).fetchone()[0]
            == 0
        )


def test_pending_schedule_refinement_shows_the_question_without_new_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, _db, binding_id = _offline_service(tmp_path, monkeypatch)
    proposed = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="schedule-draft-refinement",
        child_id="kid-a",
        text="Set up a short check-in every Tuesday, but show me the draft before anything recurs.",
    )
    assert "Draft:" in proposed["message"]
    assert proposed["interaction"]["kind"] == "confirm_action"

    refined = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="schedule-draft-refinement",
        text="Make it about noticing colors outside, with one question.",
    )
    assert "What colors outside can you notice today?" in refined["message"]
    assert "Nothing recurring" in refined["message"]
    assert refined.get("interaction") is None


def test_picture_recall_question_is_not_misrouted_as_visual_creation():
    assert not _explicit_visual_request("What was the child watching in that picture?")
    assert _explicit_visual_request("Please make that picture clearer and less busy.")
    assert _explicit_visual_request("Create a side-by-side diagram of the two layers.")


def test_ambiguous_shared_fragment_is_saved_and_gets_one_clarifying_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="ambiguous-fragment",
        child_id="kid-a",
        text="Kid A said, “the shiny…” while looking outside. I could not tell what they meant.",
    )

    assert "unfinished thought" in result["message"]
    assert result["message"].endswith("?")
    with connect(db) as conn:
        event = conn.execute("SELECT type,metadata_json FROM events").fetchone()
        assert event["type"] == "context_input"
        assert jload(event["metadata_json"])["evidence_eligible"] is False


def test_child_is_wondering_routes_to_answer_not_context_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    result = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="wondering-question",
        child_id="kid-a",
        text="Kid A is wondering whether a submarine can stay underwater all day.",
    )

    assert result["status"] == "completed"
    with connect(db) as conn:
        event = conn.execute("SELECT type FROM events").fetchone()
        assert event["type"] == "child_question"


def test_checkin_draft_uses_the_parent_requested_question_and_never_bad_grammar():
    draft = _weekly_checkin_draft("It could ask me whether I want to plan the week.")
    assert "Do you want to plan the week?" in draft
    assert "What did your notice" not in draft


def test_release_gate_requirements_surface_qualified_category_causal_and_component_obligations():
    requirements = _release_gate_requirements(
        {
            "text": "And how does its color change?",
            "metadata": {
                "source_context": {"text": "Kid A said, “Volcanoes are mountains that explode.”"},
                "response_revision": {
                    "parent_request": "Make it genuinely different and include three customer prompts."
                },
            },
        }
    )

    joined = " ".join(requirements)
    assert "causal" in joined
    assert "reverse implication" in joined
    assert "child-usable components" in joined
    assert "core child action" in joined


def test_release_gate_preserves_connected_questions_and_child_led_noticing():
    requirements = _release_gate_requirements(
        {
            "text": "Why does it do that?",
            "metadata": {
                "response_revision": {
                    "parent_request": "Give me a short answer and then one gentle question that lets her do the thinking.",
                    "thread_history": [
                        {"role": "user", "content": "Why do people need sleep?"},
                        {"role": "user", "content": "Does my brain turn off when I sleep?"},
                    ],
                }
            },
        }
    )

    joined = " ".join(requirements)
    assert "each still-active question" in joined
    assert "what the child sees" in joined


def test_requested_game_prompts_render_as_ready_to_use_parent_lines():
    rendered = format_reviewed_response(
        {
            "workflow": "pull_thread",
            "output": {
                "hook": "Open a tiny pretend café.",
                "show": "The stuffed animal is the customer.",
                "ask": "What will the chef invent?",
                "nugget": "Pretend play uses memory and choices.",
                "physical_extension": {
                    "title": "Cushion Café",
                    "materials": ["two cushions", "one stuffed animal"],
                    "instructions": ["Stay seated and take one order at a time."],
                    "prompts": ["Please make purple soup.", "What crunchy breakfast do you recommend?"],
                },
            },
        }
    )

    assert "*Ready-to-use prompts*" in rendered
    assert "• Please make purple soup." in rendered
    assert "• What crunchy breakfast do you recommend?" in rendered

    followup = _proportionate_thread_message(
        {
            "workflow": "pull_thread",
            "output": {
                "physical_extension": {
                    "title": "Cushion Café",
                    "materials": ["two cushions"],
                    "instructions": ["Stay seated and take one order."],
                    "prompts": ["Please make purple soup."],
                }
            },
        },
        "Make it a seated restaurant game with customer prompts.",
    )
    assert "*Ready-to-use prompts*" in followup
    assert "• Please make purple soup." in followup


def test_negated_picture_noun_is_not_misread_as_visual_revision():
    assert not _explicit_visual_request("Make it shorter and explain what it is for, not every little picture.")


def test_pronoun_revision_names_the_active_object():
    message = _name_subject_in_revision(
        "Those little pictures tell people how to care for the shirt.",
        "Why do some shirts have a little picture tag inside?",
        "Make it shorter and explain what it is for.",
    )

    assert message == "The little picture tag tells people how to care for the shirt."


def test_hyphenated_every_other_schedule_request_stays_operational_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, db, binding_id = _offline_service(tmp_path, monkeypatch)
    first = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="schedule-every-other",
        child_id="kid-a",
        text="A Sunday check-in could ask me whether I want to plan the week. Show me the draft first.",
    )
    assert "Do you want to plan the week?" in first["message"]

    limitation = service.chat(
        binding_id=binding_id,
        team_id="T1",
        channel_id="C1",
        thread_id="schedule-every-other",
        text="Yes, I approve that wording and the every-other-Sunday plan. Please set it up.",
    )

    assert "isn’t available" in limitation["message"] or "not every-other-week" in limitation["message"]
    assert limitation.get("event_id") is None
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
