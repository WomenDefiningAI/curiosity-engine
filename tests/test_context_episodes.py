from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from curiosity_engine.claims import upsert_claim
from curiosity_engine.context_builder import build_context
from curiosity_engine.contracts import Event
from curiosity_engine.db import connect, init_db
from curiosity_engine.director import AutonomousDirector
from curiosity_engine.episodes import apply_episode_correction
from curiosity_engine.graph import add_child, child_context
from curiosity_engine.interaction import setup_household
from curiosity_engine.reasoning import ReasoningEngine, StubBackend
from curiosity_engine.runtime import CuriosityHarness
from curiosity_engine.service import CuriosityService


def _harness(db: Path) -> CuriosityHarness:
    return CuriosityHarness(db, ReasoningEngine(StubBackend()))


def _question(event_id: str, created_at: datetime, *, scope: str = "family_signal") -> Event:
    return Event(
        id=event_id,
        type="child_question",
        child_id="kid-a",
        text="Why do robots need sensors?",
        source="slack_parent_report" if scope == "family_signal" else "eval",
        created_at=created_at,
        metadata={
            "learning_scope": scope,
            "conversation_ref": "conversation-a",
            "thread_ref": "",
        },
    )


def test_exact_repeats_stay_one_episode_and_developed_return_is_independent(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "kid-a", "Demo Child", 2020, "1st")
    harness = _harness(db)
    started = datetime(2026, 8, 20, 10, tzinfo=UTC)
    harness.dispatch(_question("evt_first", started))
    harness.dispatch(_question("evt_retry", started + timedelta(minutes=5)))
    harness.dispatch(_question("evt_return", started + timedelta(days=2)))
    developed = _question("evt_developed", started + timedelta(days=4))
    developed.text = "Why do robots need different sensors?"
    harness.dispatch(developed)
    harness.dispatch(_question("evt_far_later_exact", started + timedelta(days=120)))

    with connect(db) as conn:
        memberships = {
            row["event_id"]: dict(row)
            for row in conn.execute(
                "SELECT event_id,episode_id,relation,independence_status FROM episode_memberships"
            )
        }
        evidence = {
            row["event_id"]: int(row["id"])
            for row in conn.execute("SELECT id,event_id FROM evidence WHERE event_id IS NOT NULL")
        }
    assert memberships["evt_first"]["independence_status"] == "eligible"
    assert memberships["evt_retry"]["episode_id"] == memberships["evt_first"]["episode_id"]
    assert memberships["evt_retry"]["relation"] == "exact_retry"
    assert memberships["evt_retry"]["independence_status"] == "same_episode"
    assert memberships["evt_return"]["episode_id"] == memberships["evt_first"]["episode_id"]
    assert memberships["evt_return"]["relation"] == "later_repeat_uncertain"
    assert memberships["evt_return"]["independence_status"] == "uncertain"
    assert memberships["evt_developed"]["episode_id"] != memberships["evt_first"]["episode_id"]
    assert memberships["evt_developed"]["relation"] == "related_return"
    assert memberships["evt_developed"]["independence_status"] == "eligible"
    assert memberships["evt_far_later_exact"]["episode_id"] == memberships["evt_first"]["episode_id"]
    assert memberships["evt_far_later_exact"]["independence_status"] == "uncertain"

    with pytest.raises(ValueError, match="independent.*episodes"):
        upsert_claim(
            db,
            child_id="kid-a",
            subject="child",
            predicate="returns_to",
            object_="robot sensors",
            supporting_evidence_ids=[evidence["evt_first"], evidence["evt_return"]],
            requested_status="established_pattern",
        )
    claim_id = upsert_claim(
        db,
        child_id="kid-a",
        subject="child",
        predicate="returns_to",
        object_="robot sensors",
        supporting_evidence_ids=[evidence["evt_first"], evidence["evt_developed"]],
        requested_status="established_pattern",
    )
    assert claim_id > 0

    context = child_context(db, "kid-a")
    assert context["graph_health"]["episode_count"] == 2
    assert context["graph_health"]["eligible_episode_count"] == 2
    assert context["graph_health"]["same_episode_turn_count"] == 1
    assert context["graph_health"]["uncertain_count"] == 2
    assert context["graph_health"]["durable_interest_inference_enabled"] is False


def test_failed_answer_followed_by_same_question_is_repair_not_interest(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "kid-a", "Demo Child", 2020, "1st")
    started = datetime(2026, 8, 20, 10, tzinfo=UTC)

    class FailingBackend:
        name = "failing"
        model = "none"

        def complete(self, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("synthetic provider failure")

    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        CuriosityHarness(db, ReasoningEngine(FailingBackend())).dispatch(_question("evt_failed", started))
    _harness(db).dispatch(_question("evt_repair", started + timedelta(minutes=2)))
    with connect(db) as conn:
        rows = {
            row["event_id"]: dict(row)
            for row in conn.execute(
                "SELECT event_id,episode_id,relation,independence_status FROM episode_memberships"
            )
        }
    assert rows["evt_repair"]["episode_id"] == rows["evt_failed"]["episode_id"]
    assert rows["evt_repair"]["relation"] == "answer_repair"
    assert rows["evt_repair"]["independence_status"] == "same_episode"


def test_parent_requested_response_retry_stays_in_same_episode(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    service = CuriosityService(db, tmp_path / "output")
    service.add_child("kid-a", "Demo Child", 2020, "1st")
    metadata = {
        "learning_scope": "family_signal",
        "conversation_ref": "conversation-a",
        "thread_ref": "thread-a",
    }
    first = service.ask(
        child_id="kid-a",
        text="Why do robots need sensors?",
        source="slack_parent_report",
        event_id="evt_first_response",
        context_metadata=metadata,
    )
    assert first["status"] == "completed"
    retry = service.retry_response(
        source_event_id="evt_first_response",
        event_id="evt_parent_retry",
        context_metadata=metadata,
    )
    assert retry["status"] == "completed"

    with connect(db) as conn:
        memberships = {
            row["event_id"]: dict(row)
            for row in conn.execute(
                "SELECT event_id,episode_id,relation,independence_status FROM episode_memberships"
            )
        }
        retry_metadata = conn.execute(
            "SELECT metadata_json FROM events WHERE id='evt_parent_retry'"
        ).fetchone()[0]
    assert memberships["evt_parent_retry"]["episode_id"] == memberships["evt_first_response"]["episode_id"]
    assert memberships["evt_parent_retry"]["relation"] == "parent_marked_retry"
    assert memberships["evt_parent_retry"]["independence_status"] == "same_episode"
    assert "different_approach" in retry_metadata


def test_diagnostic_events_are_visible_but_ineligible(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "kid-a", "Demo Child", 2020, "1st")
    _harness(db).dispatch(
        _question("evt_eval", datetime(2026, 8, 20, 10, tzinfo=UTC), scope="diagnostic")
    )
    context = child_context(db, "kid-a")
    assert context["episodes"][0]["events"][0]["independence_status"] == "diagnostic"
    assert context["graph_health"]["eligible_episode_count"] == 0
    assert context["graph_health"]["excluded_or_non_family_count"] == 1
    assert not context["nodes"]
    family_event = _question("evt_family", datetime(2026, 8, 22, 10, tzinfo=UTC))
    family_event.text = "How do airplanes fly?"
    _harness(db).dispatch(family_event)
    with connect(db) as conn:
        evidence = {
            row["event_id"]: int(row["id"])
            for row in conn.execute("SELECT id,event_id FROM evidence WHERE event_id IS NOT NULL")
        }
    with pytest.raises(ValueError, match="family-signal episodes"):
        upsert_claim(
            db,
            child_id="kid-a",
            subject="child",
            predicate="returns_to",
            object_="machines",
            supporting_evidence_ids=[evidence["evt_eval"], evidence["evt_family"]],
            requested_status="established_pattern",
        )


def test_parent_correction_splits_retry_without_erasing_history(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "kid-a", "Demo Child", 2020, "1st")
    harness = _harness(db)
    started = datetime(2026, 8, 20, 10, tzinfo=UTC)
    harness.dispatch(_question("evt_first", started))
    harness.dispatch(_question("evt_second", started + timedelta(minutes=5)))
    corrected = apply_episode_correction(
        db,
        child_id="kid-a",
        event_id="evt_second",
        action="new_episode",
        note="The child independently returned after the original message was recorded late.",
    )
    assert corrected["independence_status"] == "eligible"
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM context_corrections").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(DISTINCT episode_id) FROM evidence").fetchone()[0] == 2
    assert child_context(db, "kid-a")["graph_health"]["parent_correction_count"] == 1


def test_parent_exclusion_leaves_audit_history_but_removes_active_context(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "kid-a", "Demo Child", 2020, "1st")
    harness = _harness(db)
    started = datetime(2026, 8, 20, 10, tzinfo=UTC)
    harness.dispatch(_question("evt_first", started))
    developed = _question("evt_developed", started + timedelta(days=3))
    developed.text = "Why do robots need different sensors?"
    harness.dispatch(developed)
    with connect(db) as conn:
        evidence = {
            row["event_id"]: int(row["id"])
            for row in conn.execute("SELECT id,event_id FROM evidence WHERE event_id IS NOT NULL")
        }
    claim_id = upsert_claim(
        db,
        child_id="kid-a",
        subject="child",
        predicate="returns_to",
        object_="robot sensors",
        supporting_evidence_ids=[evidence["evt_first"], evidence["evt_developed"]],
        requested_status="established_pattern",
    )
    correction = apply_episode_correction(
        db,
        child_id="kid-a",
        event_id="evt_developed",
        action="exclude",
        note="Parent says this was an answer retry, not a child signal.",
    )
    assert correction["downgraded_claim_ids"] == [claim_id]
    inspector = child_context(db, "kid-a")
    excluded = next(
        event
        for episode in inspector["episodes"]
        for event in episode["events"]
        if event["event_id"] == "evt_developed"
    )
    assert excluded["independence_status"] == "excluded"
    model_context = build_context(
        str(db),
        "kid-a",
        {"type": "child_question", "text": "Tell me about sensor choices", "metadata": {}},
        depth=4,
    )
    assert all(item["text"] != developed.text for item in model_context["observations"])
    assert all(item["label"] != developed.text for item in model_context["nodes"])
    assert all(
        turn["text"] != developed.text
        for episode in model_context["episodes"]
        for turn in episode["turns"]
    )
    assert next(item for item in model_context["claims"] if item["id"] == claim_id)["status"] == "hypothesis"


def test_context_driven_scheduler_cannot_call_model_before_policy_enablement(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    setup_household(db, owner_name="Parent", timezone="Etc/UTC", proactive_enabled=True)
    add_child(db, "kid-a", "Demo Child", 2020, "1st")

    class NeverReason:
        def run(self, **_kwargs: Any) -> None:
            raise AssertionError("context-driven model must not run")

    director = AutonomousDirector(db, reasoning=NeverReason())
    director.ensure_weekly_schedule("kid-a", start_at=datetime(2026, 8, 20, tzinfo=UTC))
    assert director.run_due(now=datetime(2026, 8, 21, tzinfo=UTC)) == []
    reflection = director.reflect_for_child("kid-a")
    assert reflection["choice"]["kind"] == "do_nothing"
    assert reflection["choice"]["payload"]["policy"] == "context_proactivity_disabled_v1"
