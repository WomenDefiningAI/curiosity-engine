from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from curiosity_engine.contracts import ArtifactSpec, PhysicalExtension
from curiosity_engine.db import connect, jdump, utcnow
from curiosity_engine.director import AutonomousDirector
from curiosity_engine.graph import add_child
from curiosity_engine.interaction import (
    TransportConflict,
    active_binding,
    answer_stack_fingerprint,
    claim_delivery,
    configure_family_lens,
    create_pairing_code,
    list_inbox,
    onboarding_status,
    ready_deliveries,
    record_onboarding_review,
    reviewable_slack_events,
    revoke_binding,
    setup_household,
)
from curiosity_engine.onboarding import doctor
from curiosity_engine.transports.contracts import InboundMessage
from curiosity_engine.transports.slack import (
    INBOX_ASSIGN_ACTION,
    INBOX_DISMISS_ACTION,
    RESPONSE_HELPFUL_ACTION,
    RESPONSE_NOT_HELPFUL_ACTION,
    RESPONSE_RETRY_ACTION,
    SlackTransport,
    _make_slack_action_receiver,
    _make_slack_event_receiver,
    _make_slack_response_action_receiver,
    flush_slack_outbox,
)


class FakeService:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.calls: list[dict[str, Any]] = []
        self.feedback_calls: list[dict[str, Any]] = []

    def children(self) -> list[dict[str, Any]]:
        return [{"id": "kid-a", "name": "Kid A", "grade": "2"}]

    def ask(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["child_id"] != "kid-a":
            raise ValueError("child not found")
        self.calls.append(kwargs)
        stored_output = {"_reasoning": {"answer_stack_hash": answer_stack_fingerprint(self.db_path)}}
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                   VALUES(?,'child_question',?,?,?,'{}',?,'completed')""",
                (kwargs["event_id"], kwargs["child_id"], kwargs["text"], kwargs["source"], utcnow()),
            )
            conn.execute(
                """INSERT OR IGNORE INTO responses(
                   event_id,run_id,workflow,status,output_json,created_at,updated_at
                   ) VALUES(?,NULL,'pull_thread','completed',?,?,?)""",
                (kwargs["event_id"], jdump(stored_output), utcnow(), utcnow()),
            )
        return {
            "status": "completed",
            "output": {
                "hook": "Look closely.",
                "show": "Compare two things.",
                "ask": "What changed?",
                "nugget": "One small idea.",
                "physical_extension": {
                    "title": "Paper test",
                    "materials": ["paper", "pencil"],
                    "instructions": ["Draw what you notice."],
                },
            },
        }

    def feedback(self, payload: dict[str, Any]) -> int:
        self.feedback_calls.append(payload)
        return len(self.feedback_calls)

    def retry_response(self, **kwargs: Any) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            source = conn.execute(
                "SELECT child_id,text FROM events WHERE id=?", (kwargs["source_event_id"],)
            ).fetchone()
        assert source
        return self.ask(
            child_id=source["child_id"],
            text=source["text"],
            source="slack_parent_retry",
            include_private_excerpts=kwargs["include_private_excerpts"],
            event_id=kwargs["event_id"],
            context_metadata={
                **kwargs["context_metadata"],
                "episode_relation": "retry",
                "retry_of_event_id": kwargs["source_event_id"],
            },
        )


class FakeSlackClient:
    def __init__(self):
        self.messages: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.messages.append(kwargs)
        return {"ts": f"100.{len(self.messages)}"}

    def chat_update(self, **kwargs: Any) -> dict[str, str]:
        self.updates.append(kwargs)
        return {"ts": str(kwargs["ts"])}


def incoming(event_id: str, text: str, *, channel: str = "D_PARENT", user: str = "U_PARENT") -> InboundMessage:
    return InboundMessage(
        external_event_id=event_id,
        team_id="T_FAMILY",
        user_id=user,
        channel_id=channel,
        text=text,
    )


def paired_transport(
    tmp_path: Path, *, resource_context_mode: str = "metadata_only"
) -> tuple[Path, SlackTransport, FakeService, FakeSlackClient, str]:
    db = tmp_path / "private" / "curiosity.db"
    setup = setup_household(
        db,
        owner_name="Parent",
        timezone="Etc/UTC",
        resource_context_mode=resource_context_mode,
    )
    add_child(db, "kid-a", "Kid A")
    fake = FakeService(db)
    transport = SlackTransport(db, tmp_path / "output", service=fake)
    code = create_pairing_code(db, setup["owner_id"])["pairing_code"]
    result = transport.handle(incoming("EvPair", f"pair {code}"))
    assert result.status == "paired"
    client = FakeSlackClient()
    assert flush_slack_outbox(client, db)[0]["status"] == "sent"
    return db, transport, fake, client, code


def test_pairing_is_single_use_exact_and_idempotent(tmp_path: Path):
    db, transport, _fake, _client, paired_code = paired_transport(tmp_path)
    assert active_binding(db, incoming("lookup", "help")) is not None
    assert active_binding(db, incoming("lookup-two", "help", channel="D_OTHER")) is None

    duplicate = transport.handle(incoming("EvPair", f"pair {paired_code}"))
    assert duplicate.status == "duplicate"
    with pytest.raises(TransportConflict):
        transport.handle(incoming("EvPair", "pair ABCDEFGH"))

    code = create_pairing_code(db, onboarding_status(db)["parents"][0]["id"])["pairing_code"]
    first = transport.handle(incoming("EvPairOther", f"pair {code}", channel="D_OTHER"))
    assert first.status == "paired"
    replay = transport.handle(incoming("EvPairReplay", f"pair {code}", channel="D_THIRD"))
    assert replay.status == "rejected"

    binding_id = active_binding(db, incoming("lookup-three", "help", channel="D_OTHER"))["id"]
    revoke_binding(db, binding_id)
    assert active_binding(db, incoming("lookup-four", "help", channel="D_OTHER")) is None


def test_connection_is_fixed_non_model_path_and_records_confirmed_delivery(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path)
    result = transport.handle(incoming("EvConnection", "connection"))
    assert result.status == "completed"
    assert "did not contact an AI model" in result.message
    assert fake.calls == []
    state_before = onboarding_status(db)
    assert state_before["checkpoints"]["transport_verified"]["status"] == "pending"
    assert flush_slack_outbox(client, db)[0]["status"] == "sent"
    state_after = onboarding_status(db)
    assert state_after["checkpoints"]["transport_verified"]["status"] == "pass"

def test_bolt_listener_receives_injected_event_arguments(tmp_path: Path):
    db = tmp_path / "private" / "curiosity.db"
    setup = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    code = create_pairing_code(db, setup["owner_id"])["pairing_code"]
    transport = SlackTransport(db, tmp_path / "output", service=FakeService(db))
    receive = _make_slack_event_receiver(transport, db)
    client = FakeSlackClient()

    # Bolt only discovers positional argument names; keyword-only parameters are omitted.
    assert inspect.getfullargspec(receive).args == ["body", "event", "client"]
    receive(
        {"event_id": "EvMentionPair", "team_id": "T_FAMILY"},
        {
            "type": "app_mention",
            "user": "U_PARENT",
            "channel": "C_FAMILY",
            "text": f"<@U123ABC> pair {code}",
            "ts": "100.1",
            "event_ts": "100.1",
        },
        client,
    )

    assert active_binding(db, incoming("lookup", "help", channel="C_FAMILY")) is not None
    assert client.messages
    assert client.messages[0]["channel"] == "C_FAMILY"
    assert client.messages[0]["thread_ts"] == "100.1"
    assert client.messages[0]["text"].startswith("Paired.")


def test_expired_pairing_code_is_rejected(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    setup = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    code = create_pairing_code(db, setup["owner_id"])["pairing_code"]
    with connect(db) as conn:
        conn.execute("UPDATE transport_pairing_codes SET expires_at='2000-01-01T00:00:00+00:00'")
    result = SlackTransport(db, tmp_path / "output", service=FakeService(db)).handle(
        incoming("EvExpired", f"pair {code}")
    )
    assert result.status == "rejected"
    assert "expired" in result.message


def test_slack_never_guesses_child_and_assignment_answers(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path)
    saved = transport.handle(incoming("EvNote", "We found a feather on our walk"))
    assert saved.status == "unassigned"
    assert saved.inbox_id
    assert fake.calls == []
    assert list_inbox(db)[0]["child_id"] is None
    flush_slack_outbox(client, db)
    actions = client.messages[-1]["blocks"][1]
    assert actions["block_id"] == f"curiosity_inbox:{saved.inbox_id}"
    assert [element["action_id"] for element in actions["elements"]] == [
        INBOX_ASSIGN_ACTION,
        INBOX_DISMISS_ACTION,
    ]

    assigned = transport.handle(incoming("EvAssign", f"assign {saved.inbox_id} kid-a"))
    assert assigned.status == "completed"
    assert fake.calls[0]["text"] == "We found a feather on our walk"
    assert fake.calls[0]["include_private_excerpts"] is False
    assert list_inbox(db, status="assigned")[0]["child_id"] == "kid-a"

    immediate = transport.handle(incoming("EvAsk", "ask kid-a: Why are feathers light?"))
    assert immediate.status == "completed"
    assert fake.calls[-1]["source"] == "slack_parent_report"
    assert fake.calls[-1]["context_metadata"]["learning_scope"] == "family_signal"
    assert fake.calls[-1]["context_metadata"]["conversation_ref"]
    flush_slack_outbox(client, db)
    assert any("*Tiny explanation*" in message["text"] for message in client.messages)
    assert not any("purchased-resource excerpts" in message["text"] for message in client.messages)

    feedback = transport.handle(incoming("EvFeedback", "feedback kid-a engaged: Kept comparing feathers"))
    assert feedback.status == "completed"
    assert fake.feedback_calls == [
        {
            "child_id": "kid-a",
            "outcome": "engaged",
            "note": "Kept comparing feathers",
            "source": "slack_parent",
        }
    ]


def test_slack_accepts_backticked_commands_and_natural_health_check(tmp_path: Path):
    db, transport, fake, _client, _code = paired_transport(tmp_path)
    saved = transport.handle(incoming("EvBacktickNote", "A note to dismiss"))

    dismissed = transport.handle(incoming("EvBacktickDismiss", f"`dismiss {saved.inbox_id}`"))
    assert dismissed.status == "completed"
    assert list_inbox(db, status="dismissed")[0]["id"] == saved.inbox_id

    health = transport.handle(incoming("EvHealth", "still working?"))
    assert health.status == "completed"
    assert "Slack connection works" in health.message
    assert fake.calls == []
    assert len(list_inbox(db)) == 0


def test_slack_inbox_command_returns_controls_for_each_note(tmp_path: Path):
    db, transport, _fake, client, _code = paired_transport(tmp_path)
    first = transport.handle(incoming("EvInboxOne", "First saved note"))
    second = transport.handle(incoming("EvInboxTwo", "Second saved note"))
    flush_slack_outbox(client, db)

    result = transport.handle(incoming("EvInboxList", "inbox"))
    assert result.status == "completed"
    flush_slack_outbox(client, db)
    blocks = client.messages[-1]["blocks"]
    action_blocks = [block for block in blocks if block["type"] == "actions"]
    assert {block["block_id"] for block in action_blocks} == {
        f"curiosity_inbox:{first.inbox_id}",
        f"curiosity_inbox:{second.inbox_id}",
    }


def test_slack_blockkit_actions_remain_bound_and_resolve_inbox(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path)
    saved = transport.handle(incoming("EvActionNote", "We noticed a shiny rock"))
    flush_slack_outbox(client, db)
    actions = client.messages[-1]["blocks"][1]
    select = actions["elements"][0]
    dismissed_button = actions["elements"][1]
    body = {
        "trigger_id": "trigger-one",
        "team": {"id": "T_FAMILY"},
        "user": {"id": "U_PARENT"},
        "channel": {"id": "D_PARENT"},
        "container": {"channel_id": "D_PARENT", "message_ts": "100.2"},
        "message": {"ts": "100.2"},
    }
    acknowledgements: list[bool] = []
    assign = _make_slack_action_receiver(transport, db, action_id=INBOX_ASSIGN_ACTION)
    assign(
        lambda: acknowledgements.append(True),
        body,
        {
            "action_id": INBOX_ASSIGN_ACTION,
            "action_ts": "101.1",
            "block_id": actions["block_id"],
            "selected_option": select["options"][0],
        },
        client,
    )
    assert acknowledgements == [True]
    assert list_inbox(db, status="assigned")[0]["id"] == saved.inbox_id
    assert fake.calls[-1]["text"] == "We noticed a shiny rock"
    assert client.updates[-2]["text"] == "Making a learning thread for Kid A…"
    assert "hourglass_flowing_sand" in client.updates[-2]["blocks"][0]["text"]["text"]
    assert client.updates[-1]["text"].startswith("Assigned")

    second = transport.handle(incoming("EvDismissActionNote", "A second note"))
    flush_slack_outbox(client, db)
    second_actions = client.messages[-1]["blocks"][1]
    dismiss = _make_slack_action_receiver(transport, db, action_id=INBOX_DISMISS_ACTION)
    dismiss(
        lambda: acknowledgements.append(True),
        {**body, "trigger_id": "trigger-two", "message": {"ts": "100.4"}},
        {
            "action_id": INBOX_DISMISS_ACTION,
            "action_ts": "102.1",
            "block_id": second_actions["block_id"],
            "value": dismissed_button["value"].replace(str(saved.inbox_id), str(second.inbox_id)),
        },
        client,
    )
    assert acknowledgements == [True, True]
    assert list_inbox(db, status="dismissed")[0]["id"] == second.inbox_id
    assert client.updates[-1]["text"].startswith("Dismissed")


def test_slack_answer_controls_record_feedback_and_retry_in_a_thread(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path)
    answer = transport.handle(incoming("EvAnswerControls", "ask kid-a: Why are rocks shiny?"))
    assert answer.event_id
    flush_slack_outbox(client, db)
    answer_message = client.messages[-1]
    action_block = answer_message["blocks"][-1]
    assert [element["action_id"] for element in action_block["elements"]] == [
        RESPONSE_HELPFUL_ACTION,
        RESPONSE_NOT_HELPFUL_ACTION,
        RESPONSE_RETRY_ACTION,
    ]
    body = {
        "trigger_id": "answer-trigger-one",
        "team": {"id": "T_FAMILY"},
        "user": {"id": "U_PARENT"},
        "channel": {"id": "D_PARENT"},
        "container": {"channel_id": "D_PARENT", "message_ts": "200.1"},
        "message": {"ts": "200.1", "text": answer_message["text"], "blocks": answer_message["blocks"]},
    }
    acknowledgements: list[bool] = []
    helpful = _make_slack_response_action_receiver(
        transport, db, action_id=RESPONSE_HELPFUL_ACTION
    )
    helpful(
        lambda: acknowledgements.append(True),
        body,
        {
            "action_id": RESPONSE_HELPFUL_ACTION,
            "action_ts": "201.1",
            "block_id": action_block["block_id"],
            "value": answer.event_id,
        },
        client,
    )
    assert acknowledgements == [True]
    assert fake.feedback_calls[-1]["event_id"] == answer.event_id
    assert fake.feedback_calls[-1]["outcome"] == "helpful"
    assert client.updates[-1]["blocks"][-1]["type"] == "context"

    retry_answer = transport.handle(incoming("EvAnswerRetry", "ask kid-a: How do birds fly?"))
    assert retry_answer.event_id
    flush_slack_outbox(client, db)
    retry_message = client.messages[-1]
    retry_actions = retry_message["blocks"][-1]
    retry = _make_slack_response_action_receiver(transport, db, action_id=RESPONSE_RETRY_ACTION)
    retry(
        lambda: acknowledgements.append(True),
        {
            **body,
            "trigger_id": "answer-trigger-two",
            "message": {"ts": "200.2", "text": retry_message["text"], "blocks": retry_message["blocks"]},
        },
        {
            "action_id": RESPONSE_RETRY_ACTION,
            "action_ts": "202.1",
            "block_id": retry_actions["block_id"],
            "value": retry_answer.event_id,
        },
        client,
    )
    assert acknowledgements == [True, True]
    assert fake.feedback_calls[-1]["outcome"] == "not_helpful"
    assert fake.calls[-1]["context_metadata"]["episode_relation"] == "retry"
    assert client.updates[-2]["blocks"][-1]["type"] == "context"
    assert "Trying a different approach" in client.updates[-2]["blocks"][-1]["elements"][0]["text"]
    assert client.messages[-1]["thread_ts"] == "200.2"
    assert client.messages[-1]["blocks"][-1]["elements"][-1]["action_id"] == RESPONSE_RETRY_ACTION

    negative_answer = transport.handle(incoming("EvAnswerNegative", "ask kid-a: Why is snow cold?"))
    assert negative_answer.event_id
    flush_slack_outbox(client, db)
    negative_message = client.messages[-1]
    negative_actions = negative_message["blocks"][-1]
    not_helpful = _make_slack_response_action_receiver(
        transport, db, action_id=RESPONSE_NOT_HELPFUL_ACTION
    )
    not_helpful(
        lambda: acknowledgements.append(True),
        {
            **body,
            "trigger_id": "answer-trigger-three",
            "message": {
                "ts": "200.3",
                "text": negative_message["text"],
                "blocks": negative_message["blocks"],
            },
        },
        {
            "action_id": RESPONSE_NOT_HELPFUL_ACTION,
            "action_ts": "203.1",
            "block_id": negative_actions["block_id"],
            "value": negative_answer.event_id,
        },
        client,
    )
    assert acknowledgements == [True, True, True]
    assert fake.feedback_calls[-1]["outcome"] == "not_helpful"
    assert client.updates[-1]["blocks"][-1]["elements"][-1]["action_id"] == RESPONSE_RETRY_ACTION


def test_onboarding_review_requires_latest_confirmed_slack_answer(tmp_path: Path):
    db, transport, _fake, client, _code = paired_transport(tmp_path)
    answer = transport.handle(incoming("EvReview", "ask kid-a: Why do feathers float?"))
    assert answer.status == "completed"
    assert reviewable_slack_events(db) == []
    assert flush_slack_outbox(client, db)[0]["status"] == "sent"
    pending = reviewable_slack_events(db)
    assert len(pending) == 1
    assert pending[0]["event_id"] == answer.event_id
    assert pending[0]["workflow"] == "pull_thread"
    assert pending[0]["reviewed"] == 0
    assert pending[0]["current_answer_stack"] is True
    review = record_onboarding_review(
        db,
        event_id=str(answer.event_id),
        factuality="pass",
        grade_fit="pass",
        curiosity_value="pass",
        parent_effort="pass",
    )
    assert review["decision"] == "pass"
    assert onboarding_status(db)["quality_review_accepted"] is True
    configure_family_lens(
        db,
        {
            "pedagogy": ["show before explaining", "invite a prediction"],
            "themes": [],
            "activity_minutes": 8,
            "parent_effort": "very_low",
            "reading_load": "emerging",
            "materials": ["paper"],
            "content_boundaries": [],
        },
    )
    assert onboarding_status(db)["quality_review_accepted"] is False
    with pytest.raises(ValueError, match="current answer stack"):
        record_onboarding_review(
            db,
            event_id=str(answer.event_id),
            factuality="pass",
            grade_fit="pass",
            curiosity_value="pass",
            parent_effort="pass",
        )


def test_slack_uses_household_private_resource_opt_in(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path, resource_context_mode="selected_excerpts")
    result = transport.handle(incoming("EvPrivateAsk", "ask kid-a: Why does the Moon look close?"))
    assert result.status == "completed"
    assert fake.calls[-1]["include_private_excerpts"] is True

    privacy = transport.handle(incoming("EvPrivacy", "privacy"))
    assert privacy.status == "completed"
    flush_slack_outbox(client, db)
    assert any("opted in to selected excerpts" in message["text"] for message in client.messages)


def test_slack_hides_rejected_model_candidate_details(tmp_path: Path):
    db, _transport, _fake, client, _code = paired_transport(tmp_path)

    class RejectingService(FakeService):
        def ask(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            with connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                       VALUES(?,'child_question',?,?,?,'{}',?,'rejected')""",
                    (kwargs["event_id"], kwargs["child_id"], kwargs["text"], kwargs["source"], utcnow()),
                )
            return {
                "status": "rejected",
                "output": {"candidate": {"hook": "private draft"}, "critiques": ["validation error"]},
            }

    transport = SlackTransport(db, tmp_path / "output", service=RejectingService(db))
    result = transport.handle(incoming("EvRejectedAnswer", "ask kid-a: What is the biggest robot?"))
    assert result.status == "rejected"
    assert "reliable answer" in result.message
    assert "do not need to keep retyping" in result.message
    assert "validation error" not in result.message
    flush_slack_outbox(client, db)
    assert "private draft" not in client.messages[-1]["text"]


def test_slack_never_posts_raw_validation_error_details(tmp_path: Path):
    db, _transport, _fake, client, _code = paired_transport(tmp_path)

    class InvalidService(FakeService):
        def ask(self, **_kwargs: Any) -> dict[str, Any]:
            raise ValueError("schema path hook input_value='private child wording'")

    transport = SlackTransport(db, tmp_path / "output", service=InvalidService(db))
    result = transport.handle(incoming("EvInvalid", "ask kid-a: A private child question"))
    assert result.status == "rejected"
    assert "private child wording" not in result.message
    assert "schema path" not in result.message
    flush_slack_outbox(client, db)
    assert "private child wording" not in client.messages[-1]["text"]


def test_outbox_does_not_retry_ambiguous_delivery(tmp_path: Path):
    db, transport, _fake, _client, _code = paired_transport(tmp_path)
    transport.handle(incoming("EvHelp", "help"))

    class AmbiguousClient:
        def chat_postMessage(self, **_kwargs: Any) -> None:
            raise TimeoutError("connection broke")

    result = flush_slack_outbox(AmbiguousClient(), db)
    assert len(result) == 1
    assert result[0]["status"] == "unknown"
    with connect(db) as conn:
        assert conn.execute("SELECT status FROM delivery_outbox WHERE id=?", (result[0]["delivery_id"],)).fetchone()[0] == "unknown"


def test_outbox_delivery_claim_is_atomic(tmp_path: Path):
    db, transport, _fake, _client, _code = paired_transport(tmp_path)
    handled = transport.handle(incoming("EvAtomic", "help"))
    assert handled.outbound_id
    assert claim_delivery(db, handled.outbound_id) is True
    assert claim_delivery(db, handled.outbound_id) is False
    with connect(db) as conn:
        conn.execute(
            "UPDATE delivery_outbox SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (handled.outbound_id,),
        )
    assert ready_deliveries(db) == []
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM delivery_outbox WHERE id=?", (handled.outbound_id,)
        ).fetchone()[0] == "unknown"


def test_weekly_schedule_stays_disabled_until_episode_policy_is_ready(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    setup_household(db, owner_name="Parent", timezone="Etc/UTC", proactive_enabled=False)
    add_child(db, "kid-a", "Kid A")
    director = AutonomousDirector(db)
    director.ensure_weekly_schedule("kid-a")
    with connect(db) as conn:
        assert conn.execute("SELECT enabled FROM schedules WHERE id='weekly:kid-a'").fetchone()[0] == 0
    setup_household(db, owner_name="Parent", timezone="Etc/UTC", proactive_enabled=True)
    with connect(db) as conn:
        assert conn.execute("SELECT enabled FROM schedules WHERE id='weekly:kid-a'").fetchone()[0] == 0


def test_weekly_suggestion_limit_is_family_wide(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    setup_household(db, owner_name="Parent", timezone="Etc/UTC", proactive_enabled=True)
    add_child(db, "kid-a", "Kid A")
    add_child(db, "kid-b", "Kid B")
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO opportunities(id,child_id,kind,title,rationale,priority,parent_effort,payload_json,
               status,dedupe_key,created_at) VALUES('opp_existing','kid-a','pull_thread','One idea','Timely',
               0.8,'low','{}','suggested','family-cap-test',?)""",
            (now,),
        )
    director = AutonomousDirector(db)
    second = director.reflect_for_child("kid-b")
    assert second["choice"]["kind"] == "do_nothing"


def test_doctor_never_echoes_tokens_or_family_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "db.sqlite"
    setup_household(db, owner_name="Unique Private Parent", timezone="Etc/UTC")
    monkeypatch.setenv("SLACK_APP_TOKEN", "xapp-secret-value")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-secret-value")
    monkeypatch.setenv("CURIOSITY_BACKEND", "openai")
    fake_model_key = "sk-" + "private-model-key-value"
    monkeypatch.setenv("OPENAI_API_KEY", fake_model_key)
    rendered = json.dumps(doctor(db))
    assert "xapp-secret-value" not in rendered
    assert "xoxb-secret-value" not in rendered
    assert "Unique Private Parent" not in rendered
    assert fake_model_key not in rendered


def test_manifest_has_only_mvp_scopes():
    manifest = (Path(__file__).parents[1] / "integrations" / "slack" / "manifest.yaml").read_text()
    for scope in ("app_mentions:read", "chat:write", "files:write", "im:history"):
        assert f"- {scope}" in manifest
    for forbidden in ("channels:history", "groups:history", "users:read", "files:read"):
        assert forbidden not in manifest
    assert "interactivity:" in manifest
    assert "is_enabled: true" in manifest


def test_contract_rejects_specialized_fabrication_terms():
    with pytest.raises(ValidationError):
        PhysicalExtension(
            title="Make a model",
            instructions=["Export a .stl file"],
            materials=["paper"],
        )
    with pytest.raises(ValidationError):
        ArtifactSpec(
            artifact_type="wonder_page",
            title="Model",
            trust_tier="B",
            target_age=7,
            prompt="What do you notice?",
            assets=[{"kind": "model", "method": "deterministic", "local_path": "model.gcode"}],
        )


def test_public_tree_has_no_inherited_product_names():
    root = Path(__file__).parents[1]
    forbidden = ("Her" + "mes", "Family" + "OS", "Open" + "Claw")
    excluded = {".git", ".venv", ".Codex", "private", "build", "dist", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not any(name.casefold() in text.casefold() for name in forbidden), path


def test_importing_web_module_does_not_create_family_state(tmp_path: Path):
    clean_root = tmp_path / "clean-root"
    clean_root.mkdir()
    shutil.copytree(Path(__file__).parents[1] / "configs", clean_root / "configs")
    environment = {**os.environ, "CURIOSITY_REPO_ROOT": str(clean_root)}
    result = subprocess.run(
        [sys.executable, "-c", "import curiosity_engine.web"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert not (clean_root / "private").exists()
