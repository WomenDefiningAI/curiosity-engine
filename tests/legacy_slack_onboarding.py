from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from curiosity_engine.contracts import ArtifactSpec, InteractionOption, InteractionPlan, PhysicalExtension
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
from curiosity_engine.sessions import SessionStore
from curiosity_engine.transports.contracts import InboundMessage
from curiosity_engine.transports.slack import (
    INBOX_ASSIGN_ACTION,
    INBOX_DISMISS_ACTION,
    RESPONSE_CHALLENGE_ACTION,
    RESPONSE_HELPFUL_ACTION,
    RESPONSE_NOT_HELPFUL_ACTION,
    RESPONSE_RETRY_ACTION,
    SlackTransport,
    _help_text,
    _make_slack_action_receiver,
    _make_slack_event_receiver,
    _make_slack_response_action_receiver,
    _mrkdwn_sections,
    _response_blocks,
    _response_or_semantic_blocks,
    flush_slack_outbox,
)


def test_slack_response_blocks_convert_markdown_links_to_native_mrkdwn() -> None:
    source = "See [Smithsonian guide](https://example.test/guide?source=model) for the explanation."

    rendered = "\n".join(_mrkdwn_sections(source))

    assert rendered == "See <https://example.test/guide?source=model|Smithsonian guide> for the explanation."
    assert "[Smithsonian guide](" not in rendered


class FakeService:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.feedback_calls: list[dict[str, Any]] = []
        self.interaction_calls: list[dict[str, Any]] = []

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
            "event_id": kwargs["event_id"],
            "workflow": "pull_thread",
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

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.chat_calls.append(kwargs)
        lowered = str(kwargs["text"]).casefold()
        if kwargs.get("answer_mode_hint") == "explore" or "wondering why" in lowered:
            response = self.ask(
                child_id=kwargs["child_id"],
                text=kwargs["text"],
                source="slack_parent_followup",
                include_private_excerpts=kwargs["include_private_excerpts"],
                event_id=f"evt_fake_{len(self.chat_calls)}",
            )
            return {
                **response,
                "message": "A reviewed learning response.",
                "session_id": "session-test",
            }
        return {
            "status": "completed",
            "message": "Got it—I’ll keep that as context unless you ask me to make something from it.",
            "session_id": "session-test",
        }

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

    def handle_interaction_choice(self, **kwargs: Any) -> dict[str, Any]:
        self.interaction_calls.append(kwargs)
        return {
            "status": "completed",
            "message": "I made a challenge. The printable will follow here.",
        }


class FakeSlackClient:
    def __init__(self):
        self.messages: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.reactions_added: list[dict[str, Any]] = []
        self.reactions_removed: list[dict[str, Any]] = []
        self.calls: list[str] = []

    def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append("message")
        self.messages.append(kwargs)
        return {"ts": f"100.{len(self.messages)}"}

    def chat_update(self, **kwargs: Any) -> dict[str, str]:
        self.calls.append("update")
        self.updates.append(kwargs)
        return {"ts": str(kwargs["ts"])}

    def reactions_add(self, **kwargs: Any) -> dict[str, bool]:
        self.calls.append("reaction_add")
        self.reactions_added.append(kwargs)
        return {"ok": True}

    def reactions_remove(self, **kwargs: Any) -> dict[str, bool]:
        self.calls.append("reaction_remove")
        self.reactions_removed.append(kwargs)
        return {"ok": True}


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
    assert client.messages[-1]["unfurl_links"] is False
    assert client.messages[-1]["unfurl_media"] is False
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

    client.calls.clear()
    client.messages.clear()
    receive(
        {"event_id": "EvMentionWorking", "team_id": "T_FAMILY"},
        {
            "type": "app_mention",
            "user": "U_PARENT",
            "channel": "C_FAMILY",
            "text": "<@U123ABC> still working?",
            "ts": "100.2",
            "event_ts": "100.2",
        },
        client,
    )
    working = {"channel": "C_FAMILY", "timestamp": "100.2", "name": "eyes"}
    done = {"channel": "C_FAMILY", "timestamp": "100.2", "name": "white_check_mark"}
    assert client.reactions_added == [working, done]
    assert client.reactions_removed == []
    assert client.calls == ["reaction_add", "message", "reaction_add"]
    assert client.messages[0]["thread_ts"] == "100.2"


def test_help_explains_slack_native_session_boundaries():
    help_text = _help_text()

    assert "fresh session with a new top-level mention" in help_text
    assert "mention me in its thread to continue it" in help_text


def test_paired_message_keeps_acknowledgement_and_adds_done_after_reply(tmp_path: Path):
    db, transport, _fake, client, _code = paired_transport(tmp_path)
    client.calls.clear()
    client.messages.clear()
    receive = _make_slack_event_receiver(transport, db)

    receive(
        {"event_id": "EvWorking", "team_id": "T_FAMILY"},
        {
            "type": "message",
            "channel_type": "im",
            "user": "U_PARENT",
            "channel": "D_PARENT",
            "text": "still working?",
            "ts": "200.1",
            "event_ts": "200.1",
        },
        client,
    )

    working = {"channel": "D_PARENT", "timestamp": "200.1", "name": "eyes"}
    done = {"channel": "D_PARENT", "timestamp": "200.1", "name": "white_check_mark"}
    assert client.reactions_added == [working, done]
    assert client.reactions_removed == []
    assert client.calls == ["reaction_add", "message", "reaction_add"]
    assert client.messages[0]["text"].startswith("Slack connection works")


def test_missing_reaction_scope_never_blocks_reply(tmp_path: Path):
    db, transport, _fake, client, _code = paired_transport(tmp_path)
    client.messages.clear()

    def deny_reaction(**_kwargs: Any) -> None:
        raise RuntimeError("synthetic missing scope")

    client.reactions_add = deny_reaction  # type: ignore[method-assign]
    receive = _make_slack_event_receiver(transport, db)
    receive(
        {"event_id": "EvNoReactionScope", "team_id": "T_FAMILY"},
        {
            "type": "message",
            "channel_type": "im",
            "user": "U_PARENT",
            "channel": "D_PARENT",
            "text": "connection",
            "ts": "200.2",
            "event_ts": "200.2",
        },
        client,
    )

    assert client.reactions_removed == []
    assert client.messages[0]["text"].startswith("Slack connection works")


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


def test_slack_uses_only_child_and_assignment_handles_ambiguity(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path)
    inferred = transport.handle(incoming("EvOnlyChild", "We found a feather on our walk"))
    assert inferred.status == "completed"
    assert inferred.inbox_id is None
    assert fake.chat_calls[-1]["child_id"] == "kid-a"
    assert list_inbox(db) == []

    fake.chat_calls.clear()
    fake.children = lambda: [  # type: ignore[method-assign]
        {"id": "kid-a", "name": "Kid A", "grade": "2"},
        {"id": "kid-b", "name": "Kid B", "grade": "1"},
    ]
    saved = transport.handle(incoming("EvNote", "We found another feather on our walk"))
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

    fake.children = lambda: [{"id": "kid-a", "name": "Kid A", "grade": "2"}]  # type: ignore[method-assign]
    assigned = transport.handle(incoming("EvAssign", f"assign {saved.inbox_id} kid-a"))
    assert assigned.status == "completed"
    assert fake.chat_calls[0]["text"] == "We found another feather on our walk"
    assert fake.chat_calls[0]["include_private_excerpts"] is False
    assert list_inbox(db, status="assigned")[0]["child_id"] == "kid-a"

    immediate = transport.handle(incoming("EvAsk", "ask kid-a: Why are feathers light?"))
    assert immediate.status == "completed"
    assert fake.chat_calls[-1]["answer_mode_hint"] == "explore"
    assert fake.calls[-1]["source"] == "slack_parent_followup"
    flush_slack_outbox(client, db)
    assert any("reviewed learning response" in message["text"].casefold() for message in client.messages)
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


def test_slack_routes_one_strong_name_signal_without_an_inbox_prompt(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path)

    result = transport.handle(incoming("EvNamed", "Kid A keeps wondering why feathers float"))

    assert result.status == "completed"
    assert result.inbox_id is None
    assert fake.chat_calls[-1]["child_id"] == "kid-a"
    assert fake.chat_calls[-1]["text"] == "Kid A keeps wondering why feathers float"
    assert list_inbox(db) == []
    flush_slack_outbox(client, db)
    assert "Who was this for?" not in client.messages[-1]["text"]


def test_named_family_channel_context_uses_parent_router_not_lesson_template(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path)
    message = incoming("EvTeacherEmail", "Kid A's teacher email says they practiced vowel teams")
    message = message.model_copy(update={"thread_id": "200.1"})

    result = transport.handle(message)

    assert result.status == "completed"
    assert fake.calls == []
    assert fake.chat_calls[-1]["child_id"] == "kid-a"
    assert fake.chat_calls[-1]["text"] == message.text
    flush_slack_outbox(client, db)
    assert "*Start here*" not in client.messages[-1]["text"]
    assert "*Tiny explanation*" not in client.messages[-1]["text"]


def test_slack_keeps_multiple_or_weak_name_signals_unassigned(tmp_path: Path):
    db, transport, fake, _client, _code = paired_transport(tmp_path)
    fake.children = lambda: [  # type: ignore[method-assign]
        {"id": "kid-a", "name": "Kid A", "grade": "2"},
        {"id": "kid-b", "name": "Kid B", "grade": "1"},
    ]

    multiple = transport.handle(incoming("EvTwoNames", "Kid A and Kid B keep comparing feathers"))
    assert multiple.status == "unassigned"
    assert fake.calls == []

    fake.children = lambda: [  # type: ignore[method-assign]
        {"id": "kid-a", "name": "Will", "grade": "2"},
        {"id": "kid-b", "name": "Ava", "grade": "1"},
    ]
    weak = transport.handle(incoming("EvWeakName", "Will airplanes fly?"))
    assert weak.status == "unassigned"
    assert fake.calls == []


def test_slack_accepts_backticked_commands_and_natural_health_check(tmp_path: Path):
    db, transport, fake, _client, _code = paired_transport(tmp_path)
    fake.children = lambda: [  # type: ignore[method-assign]
        {"id": "kid-a", "name": "Kid A", "grade": "2"},
        {"id": "kid-b", "name": "Kid B", "grade": "1"},
    ]
    saved = transport.handle(incoming("EvBacktickNote", "A note to dismiss"))

    fake.children = lambda: [{"id": "kid-a", "name": "Kid A", "grade": "2"}]  # type: ignore[method-assign]
    dismissed = transport.handle(incoming("EvBacktickDismiss", f"`dismiss {saved.inbox_id}`"))
    assert dismissed.status == "completed"
    assert list_inbox(db, status="dismissed")[0]["id"] == saved.inbox_id

    health = transport.handle(incoming("EvHealth", "still working?"))
    assert health.status == "completed"
    assert "Slack connection works" in health.message
    assert fake.calls == []
    assert len(list_inbox(db)) == 0


def test_slack_inbox_command_returns_controls_for_each_note(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path)
    add_child(db, "kid-b", "Kid B")
    fake.children = lambda: [  # type: ignore[method-assign]
        {"id": "kid-a", "name": "Kid A", "grade": "2"},
        {"id": "kid-b", "name": "Kid B", "grade": "1"},
    ]
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
    add_child(db, "kid-b", "Kid B")
    fake.children = lambda: [  # type: ignore[method-assign]
        {"id": "kid-a", "name": "Kid A", "grade": "2"},
        {"id": "kid-b", "name": "Kid B", "grade": "1"},
    ]
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
    assert fake.chat_calls[-1]["text"] == "We noticed a shiny rock"
    assert client.updates[-2]["text"] == "Making a learning thread for Kid A…"
    assert "hourglass_flowing_sand" in client.updates[-2]["blocks"][0]["text"]["text"]
    assert client.updates[-1]["text"].startswith("Saved this context")

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
    legacy_blocks = _response_blocks(answer_message["text"], str(answer.event_id))
    action_block = legacy_blocks[-1]
    binding = active_binding(db, incoming("lookup-answer-controls", "help"))
    assert binding
    source_session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=str(binding["id"]),
        conversation_ref="answer-controls",
        thread_ref="answer-controls",
        child_id="kid-a",
    )
    SessionStore(db).append_message(
        str(source_session["id"]),
        role="assistant",
        content=answer_message["text"],
        event_id=str(answer.event_id),
    )
    assert [element["action_id"] for element in action_block["elements"]] == [
        RESPONSE_HELPFUL_ACTION,
        RESPONSE_NOT_HELPFUL_ACTION,
        RESPONSE_RETRY_ACTION,
        RESPONSE_CHALLENGE_ACTION,
    ]
    body = {
        "trigger_id": "answer-trigger-one",
        "team": {"id": "T_FAMILY"},
        "user": {"id": "U_PARENT"},
        "channel": {"id": "D_PARENT"},
        "container": {"channel_id": "D_PARENT", "message_ts": "200.1"},
        "message": {"ts": "200.1", "text": answer_message["text"], "blocks": legacy_blocks},
    }
    acknowledgements: list[bool] = []
    helpful = _make_slack_response_action_receiver(transport, db, action_id=RESPONSE_HELPFUL_ACTION)
    helpful_action = {
        "action_id": RESPONSE_HELPFUL_ACTION,
        "action_ts": "201.1",
        "block_id": action_block["block_id"],
        "value": answer.event_id,
    }
    helpful(
        lambda: acknowledgements.append(True),
        body,
        helpful_action,
        client,
    )
    assert acknowledgements == [True]
    assert fake.feedback_calls[-1]["event_id"] == answer.event_id
    assert fake.feedback_calls[-1]["outcome"] == "helpful"
    assert "Saving as helpful" in client.updates[-2]["blocks"][-1]["elements"][0]["text"]
    assert [element["action_id"] for element in client.updates[-1]["blocks"][-1]["elements"]] == [
        RESPONSE_RETRY_ACTION,
        RESPONSE_CHALLENGE_ACTION,
    ]

    update_count = len(client.updates)
    feedback_count = len(fake.feedback_calls)
    helpful(lambda: acknowledgements.append(True), body, helpful_action, client)
    assert acknowledgements == [True, True]
    assert len(client.updates) == update_count
    assert len(fake.feedback_calls) == feedback_count

    challenge_action = next(
        element for element in action_block["elements"] if element["action_id"] == RESPONSE_CHALLENGE_ACTION
    )
    challenge = _make_slack_response_action_receiver(
        transport,
        db,
        action_id=RESPONSE_CHALLENGE_ACTION,
    )
    challenge(
        lambda: acknowledgements.append(True),
        {
            **body,
            "trigger_id": "answer-trigger-after-rating",
            # Slack can deliver a callback carrying the original stale blocks.
            "message": {"ts": "200.1", "text": answer_message["text"], "blocks": legacy_blocks},
        },
        {
            **challenge_action,
            "action_ts": "201.2",
            "block_id": action_block["block_id"],
        },
        client,
    )
    assert acknowledgements == [True, True, True]
    assert fake.interaction_calls[-1]["resolved"]["intent"] == "create_artifact"
    assert "Drawing and building the challenge" in client.updates[-2]["blocks"][-1]["elements"][0]["text"]
    assert [element["action_id"] for element in client.updates[-1]["blocks"][-1]["elements"]] == [
        RESPONSE_RETRY_ACTION,
    ]

    retry_action = next(
        element for element in action_block["elements"] if element["action_id"] == RESPONSE_RETRY_ACTION
    )
    retry = _make_slack_response_action_receiver(transport, db, action_id=RESPONSE_RETRY_ACTION)
    retry(
        lambda: acknowledgements.append(True),
        {
            **body,
            "trigger_id": "answer-trigger-two",
            # This callback is stale too: rating and challenge are already complete.
            "message": {"ts": "200.1", "text": answer_message["text"], "blocks": legacy_blocks},
        },
        {
            **retry_action,
            "action_ts": "202.1",
            "block_id": action_block["block_id"],
        },
        client,
    )
    assert acknowledgements == [True, True, True, True]
    assert len(fake.feedback_calls) == feedback_count
    assert fake.feedback_calls[-1]["outcome"] == "helpful"
    assert fake.calls[-1]["context_metadata"]["episode_relation"] == "retry"
    assert client.updates[-2]["blocks"][-1]["type"] == "context"
    assert "Trying a different approach" in client.updates[-2]["blocks"][-1]["elements"][0]["text"]
    assert client.updates[-1]["blocks"][-1]["type"] == "context"
    assert client.messages[-1]["thread_ts"] == "200.1"
    assert client.messages[-1]["blocks"][-1]["elements"][-1]["action_id"] == RESPONSE_CHALLENGE_ACTION
    retry_event_id = str(fake.calls[-1]["event_id"])
    with connect(db) as conn:
        linked_retry = conn.execute(
            "SELECT 1 FROM agent_messages WHERE session_id=? AND event_id=?",
            (str(source_session["id"]), retry_event_id),
        ).fetchone()
    assert linked_retry

    retried_message = client.messages[-1]
    retried_blocks = retried_message["blocks"]
    retried_challenge_action = next(
        element for element in retried_blocks[-1]["elements"] if element["action_id"] == RESPONSE_CHALLENGE_ACTION
    )
    challenge(
        lambda: acknowledgements.append(True),
        {
            **body,
            "trigger_id": "answer-trigger-retried-challenge",
            "message": {
                "ts": "200.25",
                "thread_ts": "200.1",
                "text": retried_message["text"],
                "blocks": retried_blocks,
            },
        },
        {
            **retried_challenge_action,
            "action_ts": "202.2",
            "block_id": retried_blocks[-1]["block_id"],
        },
        client,
    )
    assert acknowledgements == [True, True, True, True, True]
    assert fake.interaction_calls[-1]["resolved"]["payload"]["event_id"] == retry_event_id

    negative_answer = transport.handle(incoming("EvAnswerNegative", "ask kid-a: Why is snow cold?"))
    assert negative_answer.event_id
    flush_slack_outbox(client, db)
    negative_message = client.messages[-1]
    negative_blocks = _response_blocks(negative_message["text"], str(negative_answer.event_id))
    negative_actions = negative_blocks[-1]
    not_helpful = _make_slack_response_action_receiver(transport, db, action_id=RESPONSE_NOT_HELPFUL_ACTION)
    not_helpful(
        lambda: acknowledgements.append(True),
        {
            **body,
            "trigger_id": "answer-trigger-three",
            "message": {
                "ts": "200.3",
                "text": negative_message["text"],
                "blocks": negative_blocks,
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
    assert acknowledgements == [True, True, True, True, True, True]
    assert fake.feedback_calls[-1]["outcome"] == "not_helpful"
    assert [element["action_id"] for element in client.updates[-1]["blocks"][-1]["elements"]] == [
        RESPONSE_RETRY_ACTION,
        RESPONSE_CHALLENGE_ACTION,
    ]


def test_rate_output_plan_uses_independent_response_actions(tmp_path: Path):
    db, _transport, _fake, _client, _code = paired_transport(tmp_path)
    binding = active_binding(db, incoming("lookup-independent-actions", "help"))
    assert binding
    event_id = "evt_independent_actions"
    blocks = _response_or_semantic_blocks(
        db,
        binding_id=str(binding["id"]),
        session_id="unused-for-direct-response-actions",
        event_id=event_id,
        text="A useful answer.",
        interaction=InteractionPlan(
            kind="rate_output",
            title="Was this useful?",
            prompt="Choose a next step.",
            options=[
                InteractionOption(
                    label="Helpful",
                    intent="rate_response",
                    payload={"event_id": event_id, "rating": "helpful"},
                )
            ],
        ),
    )

    assert [element["action_id"] for element in blocks[-1]["elements"]] == [
        RESPONSE_HELPFUL_ACTION,
        RESPONSE_NOT_HELPFUL_ACTION,
        RESPONSE_RETRY_ACTION,
        RESPONSE_CHALLENGE_ACTION,
    ]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM interaction_instances").fetchone()[0] == 0


def test_concurrent_response_redelivery_cannot_replace_success_with_progress(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path)
    answer = transport.handle(incoming("EvConcurrentAnswer", "ask kid-a: Why do pebbles sparkle?"))
    assert answer.event_id
    flush_slack_outbox(client, db)
    blocks = _response_blocks(client.messages[-1]["text"], str(answer.event_id))
    action_block = blocks[-1]
    action = {
        "action_id": RESPONSE_HELPFUL_ACTION,
        "action_ts": "301.1",
        "block_id": action_block["block_id"],
        "value": answer.event_id,
    }
    body = {
        "trigger_id": "concurrent-answer-trigger",
        "team": {"id": "T_FAMILY"},
        "user": {"id": "U_PARENT"},
        "channel": {"id": "D_PARENT"},
        "container": {"channel_id": "D_PARENT", "message_ts": "300.1"},
        "message": {"ts": "300.1", "text": client.messages[-1]["text"], "blocks": blocks},
    }
    entered_feedback = threading.Event()
    release_feedback = threading.Event()
    original_feedback = fake.feedback

    def blocking_feedback(payload: dict[str, Any]) -> int:
        entered_feedback.set()
        assert release_feedback.wait(timeout=5)
        return original_feedback(payload)

    fake.feedback = blocking_feedback  # type: ignore[method-assign]
    acknowledgements: list[bool] = []
    receive = _make_slack_response_action_receiver(transport, db, action_id=RESPONSE_HELPFUL_ACTION)
    first = threading.Thread(
        target=receive,
        args=(lambda: acknowledgements.append(True), body, action, client),
    )
    first.start()
    assert entered_feedback.wait(timeout=5)
    update_count = len(client.updates)
    receive(lambda: acknowledgements.append(True), body, action, client)
    assert len(client.updates) == update_count
    retry_action = next(
        element for element in action_block["elements"] if element["action_id"] == RESPONSE_RETRY_ACTION
    )
    retry = _make_slack_response_action_receiver(transport, db, action_id=RESPONSE_RETRY_ACTION)
    retry(
        lambda: acknowledgements.append(True),
        {**body, "trigger_id": "concurrent-retry-trigger"},
        {**retry_action, "action_ts": "301.2", "block_id": action_block["block_id"]},
        client,
    )
    assert fake.feedback_calls == []
    assert [element["action_id"] for element in client.updates[-1]["blocks"][-1]["elements"]] == [
        RESPONSE_CHALLENGE_ACTION,
    ]
    release_feedback.set()
    first.join(timeout=5)

    assert not first.is_alive()
    assert acknowledgements == [True, True, True]
    assert len(fake.feedback_calls) == 1
    assert "Marked helpful" in client.updates[-1]["blocks"][-2]["elements"][0]["text"]
    assert [element["action_id"] for element in client.updates[-1]["blocks"][-1]["elements"]] == [
        RESPONSE_CHALLENGE_ACTION,
    ]


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
    assert "original request" in result.message
    assert "same supplies" in result.message
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
        assert (
            conn.execute("SELECT status FROM delivery_outbox WHERE id=?", (result[0]["delivery_id"],)).fetchone()[0]
            == "unknown"
        )


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
        assert (
            conn.execute("SELECT status FROM delivery_outbox WHERE id=?", (handled.outbound_id,)).fetchone()[0]
            == "unknown"
        )


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
    for scope in (
        "app_mentions:read",
        "chat:write",
        "files:write",
        "files:read",
        "im:history",
        "reactions:write",
    ):
        assert f"- {scope}" in manifest
    for forbidden in ("channels:history", "groups:history", "users:read"):
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
