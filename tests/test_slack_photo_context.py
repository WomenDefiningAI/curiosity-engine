from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from curiosity_engine.attachments import link_assets_to_session
from curiosity_engine.db import connect
from curiosity_engine.graph import add_child
from curiosity_engine.interaction import create_pairing_code, setup_household
from curiosity_engine.parent_agent import ParentAgentRuntime
from curiosity_engine.service import _proportionate_thread_message
from curiosity_engine.sessions import SessionStore
from curiosity_engine.tooling import ToolRegistry
from curiosity_engine.transports.contracts import InboundMessage
from curiosity_engine.transports.slack import (
    SlackTransport,
    _download_slack_bytes,
    _make_slack_event_receiver,
    flush_slack_outbox,
)


def _png_bytes() -> bytes:
    data = BytesIO()
    Image.new("RGB", (40, 30), "orange").save(data, format="PNG")
    return data.getvalue()


class _PhotoService:
    def __init__(self, db: Path):
        self.db = db
        self.chat_calls: list[dict[str, Any]] = []

    def children(self) -> list[dict[str, Any]]:
        return [{"id": "kid-a", "name": "Kid A", "grade": "1"}]

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.chat_calls.append(kwargs)
        conversation = "conversation"
        thread = "thread"
        session = SessionStore(self.db).get_or_create(
            origin="slack",
            transport="slack",
            binding_id=kwargs["binding_id"],
            conversation_ref=conversation,
            thread_ref=thread,
            child_id=kwargs["child_id"],
        )
        link_assets_to_session(
            self.db,
            [str(item["id"]) for item in kwargs["attachments"]],
            str(session["id"]),
        )
        return {
            "status": "completed",
            "message": "Got it—I saved this as photo context. What would you like to explore?",
            "session_id": session["id"],
        }


class _SlackClient:
    token = "synthetic-test-token"

    def __init__(self):
        self.messages: list[dict[str, Any]] = []

    def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.messages.append(kwargs)
        return {"ts": f"100.{len(self.messages)}"}


def test_slack_photo_survives_inbox_assignment_and_stays_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = tmp_path / "private" / "data" / "curiosity.db"
    output = tmp_path / "private" / "output"
    household = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "kid-a", "Kid A", grade="1")
    service = _PhotoService(db)
    transport = SlackTransport(db, output, service=service)
    code = create_pairing_code(db, household["owner_id"])["pairing_code"]
    pair = InboundMessage(
        external_event_id="pair",
        team_id="T_FAMILY",
        user_id="U_PARENT",
        channel_id="C_FAMILY",
        text=f"pair {code}",
        thread_id="100.1",
    )
    transport.handle(pair)
    client = _SlackClient()
    flush_slack_outbox(client, db)
    monkeypatch.setattr(
        "curiosity_engine.transports.slack._download_slack_bytes",
        lambda _url, _token: _png_bytes(),
    )
    receive = _make_slack_event_receiver(transport, db)
    receive(
        {"event_id": "photo-event", "team_id": "T_FAMILY"},
        {
            "type": "app_mention",
            "user": "U_PARENT",
            "channel": "C_FAMILY",
            "text": "<@U_BOT> new play at home",
            "ts": "200.1",
            "event_ts": "200.1",
            "files": [
                {
                    "id": "F_PRIVATE",
                    "mimetype": "image/png",
                    "size": len(_png_bytes()),
                    "url_private_download": "https://files.slack.com/files-pri/private.png",
                }
            ],
        },
        client,
    )
    with connect(db) as conn:
        asset = conn.execute("SELECT * FROM inbound_assets").fetchone()
        inbox = conn.execute("SELECT id FROM capture_inbox WHERE status='unassigned'").fetchone()["id"]
    assert asset["status"] == "ready" and asset["inbox_id"] == inbox
    path = Path(asset["path"])
    assert path.is_relative_to(output / "inbound")
    assert path.stat().st_mode & 0o077 == 0

    assigned = transport.handle(
        InboundMessage(
            external_event_id="assign-photo",
            team_id="T_FAMILY",
            user_id="U_PARENT",
            channel_id="C_FAMILY",
            text=f"assign {inbox} kid-a",
            thread_id="200.1",
        )
    )
    assert assigned.status == "completed"
    assert "*Show*" not in assigned.message and "*Tiny explanation*" not in assigned.message
    assert len(service.chat_calls) == 1
    assert service.chat_calls[0]["attachments"][0]["sha256"] == asset["sha256"]
    with connect(db) as conn:
        assert conn.execute("SELECT session_id FROM inbound_assets").fetchone()[0]


def test_non_slack_attachment_host_is_rejected_before_network_access():
    with pytest.raises(ValueError, match="outside the Slack file host"):
        _download_slack_bytes("https://example.com/private.png", "synthetic-test-token")


class _WrongLessonBackend:
    name = "test"
    model = "test-v1"

    def complete(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "tool_calls": [
                {
                    "name": "continue_learning_thread",
                    "arguments": {"message": "new play at home"},
                    "rationale": "Create a new lesson.",
                }
            ]
        }


def test_plain_photo_context_cannot_be_normalized_into_an_unsolicited_lesson(tmp_path: Path):
    db = tmp_path / "private" / "data" / "curiosity.db"
    household = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "kid-a", "Kid A", grade="1")
    now_session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=None,
        conversation_ref="conversation",
        thread_ref="thread",
        child_id="kid-a",
    )
    del household
    tools = ToolRegistry()
    calls: list[dict[str, Any]] = []
    tools.register(
        "record_thread_context",
        lambda arguments: calls.append(arguments)
        or {"status": "completed", "message": "Saved as context only."},
    )
    tools.register("continue_learning_thread", lambda _arguments: pytest.fail("lesson tool ran"))
    tools.register("revise_learning_thread", lambda _arguments: {"status": "completed"})
    tools.register("create_learning_artifact", lambda _arguments: {"status": "completed"})
    tools.register("propose_weekly_checkin", lambda _arguments: {"status": "proposal"})
    tools.register("record_response_feedback", lambda _arguments: {"status": "completed"})
    result = ParentAgentRuntime(db, backend=_WrongLessonBackend(), tools=tools).run(
        session_id=str(now_session["id"]),
        user_message="new play at home",
        child={"id": "kid-a", "grade": "1"},
        latest_event_id=None,
        current_attachment_context=[{"status": "ready", "summary": "paper pieces on a table"}],
    )
    assert result["status"] == "completed"
    assert result["message"] == "Saved as context only."
    assert calls == [{"note": "new play at home"}]


def test_followup_renderer_does_not_force_full_lesson_headings():
    result = {
        "output": {
            "hook": "The wing bends because air pushes on it.",
            "show": "Compare a curved and flat wing.",
            "ask": "What changes?",
            "nugget": "A bend changes how the wing redirects the moving air.",
        }
    }
    message = _proportionate_thread_message(result, "Why does the wing bend?")
    assert message == (
        "The wing bends because air pushes on it. "
        "A bend changes how the wing redirects the moving air."
    )
    assert "Show" not in message and "Tiny explanation" not in message
