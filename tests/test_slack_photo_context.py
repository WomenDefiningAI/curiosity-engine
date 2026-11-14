from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from curiosity_engine.attachments import link_assets_to_session
from curiosity_engine.db import connect, jdump, utcnow
from curiosity_engine.graph import add_child
from curiosity_engine.interaction import create_pairing_code, setup_household
from curiosity_engine.parent_agent import ParentAgentRuntime
from curiosity_engine.service import CuriosityService, _proportionate_thread_message
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


def test_slack_photo_uses_the_only_child_without_an_assignment_menu_and_stays_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
                    "id": f"F_PRIVATE_{index}",
                    "mimetype": "image/png",
                    "size": len(_png_bytes()),
                    "url_private_download": "https://files.slack.com/files-pri/private.png",
                } for index in range(5)
            ],
        },
        client,
    )
    with connect(db) as conn:
        asset = conn.execute("SELECT * FROM inbound_assets").fetchone()
        assert conn.execute("SELECT COUNT(*) FROM capture_inbox").fetchone()[0] == 0
    assert asset["status"] == "ready" and asset["inbox_id"] is None
    path = Path(asset["path"])
    assert path.is_relative_to(output / "inbound")
    assert path.stat().st_mode & 0o077 == 0

    assert len(service.chat_calls) == 1
    assert service.chat_calls[0]["child_id"] == "kid-a"
    assert len(service.chat_calls[0]["attachments"]) == 5
    assert service.chat_calls[0]["attachments"][0]["sha256"] == asset["sha256"]
    with connect(db) as conn:
        assert conn.execute("SELECT session_id FROM inbound_assets").fetchone()[0]


def test_photo_concepts_survive_idea_button_challenge_button_and_open_followup(tmp_path, monkeypatch):
    from legacy_natural_routing_v2 import _offline_service

    from curiosity_engine.contracts import QuickAnswerOutput
    from curiosity_engine.interaction import delivered_slack_response
    from curiosity_engine.interactions import create_interaction
    from curiosity_engine.learning_artifacts import LearningArtifactService
    from curiosity_engine.reasoning import StubBackend
    from curiosity_engine.transports.slack import _make_semantic_interaction_receiver

    service, db, binding = _offline_service(tmp_path, monkeypatch)
    photos = [{"status": "ready", "summary": f"Practice page {index}: {concept}.",
               "learning_concepts": [concept], "visible_details": [f"Task {index}: compare examples."]}
              for index, concept in enumerate(("longer and shorter", "above and below", "empty and full",
                                               "repeating patterns", "equal groups"), 1)]
    context = service.chat(binding_id=binding, team_id="T1", channel_id="C1", thread_id="photo-thread",
                           child_id="kid-a", text="Here is today's schoolwork.", attachments=[
                               {"observation": photo} for photo in photos])
    assert all(photo["learning_concepts"][0] in context["message"] for photo in photos)

    captured = []

    class AnswerBackend(StubBackend):
        def complete(self, **kwargs):
            if kwargs["response_model"] is QuickAnswerOutput:
                observed = kwargs["payload"]["event"]["metadata"]["parent_shared_image_context"]
                captured.append(observed)
                assert not any("people agreed" in rule for rule in kwargs["payload"].get("release_gate_requirements", []))
                return {"answer": "Make a repeating movement pattern: clap, tap, clap, tap. "
                        "Pause and let the child choose which movement comes next. "
                        "No writing is needed; say the next sound or clap.", "visual": None}
            return super().complete(**kwargs)

    monkeypatch.setattr("curiosity_engine.runtime.configured_backend", lambda *_a, **_k: AnswerBackend())
    presented = create_interaction(db, binding_id=binding, session_id=context["session_id"],
                                   plan=context["interaction"])

    class Client(_SlackClient):
        def chat_update(self, **kwargs):
            return {"ts": kwargs["ts"]}

    client = Client()
    transport = SlackTransport(db, tmp_path / "output", service=service)
    receive = _make_semantic_interaction_receiver(transport, db)
    body = {"trigger_id": "synthetic-photo-idea", "team": {"id": "T1"}, "user": {"id": "U1"},
            "channel": {"id": "C1"}, "message": {"ts": "photo-thread", "blocks": []}}
    option = next(item for item in presented["options"] if item["label"] == "Give us an idea")
    action = {"block_id": f"curiosity_interaction:{presented['interaction_id']}",
              "value": option["token"], "action_ts": "1"}
    receive(lambda: None, body, action, client)
    receive(lambda: None, body, action, client)
    assert captured == [photos]
    assert len(client.messages) == 1
    with connect(db) as conn:
        idea_id = conn.execute("SELECT event_id FROM agent_messages WHERE kind='interaction_result'").fetchone()[0]
    assert delivered_slack_response(db, event_id=idea_id, binding_id=binding)
    briefs = []
    original = LearningArtifactService._design

    def design(self, *args, **kwargs):
        from curiosity_engine.attachments import source_photo_context
        briefs.append(source_photo_context(self.db_path, kwargs["source_event_id"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LearningArtifactService, "_design", design)
    result = transport.handle(InboundMessage(external_event_id="synthetic-challenge", team_id="T1", user_id="U1",
                                            channel_id="C1", thread_id="photo-thread",
                                            text=f"challenge_response {idea_id}"))
    assert result.status == "completed", result.message
    assert briefs == [photos]
    with connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 1
    result = service.chat(binding_id=binding, team_id="T1", channel_id="C1", thread_id="photo-thread",
                          text="Why does the fourth page repeat the same order?")
    assert result["status"] == "completed"
    assert captured[-1] == photos
    with connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM evidence_records").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM evidence WHERE "
                            "json_extract(metadata_json,'$.learning_scope')!='diagnostic'").fetchone()[0] == 0
    partial = service.chat(binding_id=binding, team_id="T1", channel_id="C1", thread_id="partial-photos",
                           child_id="kid-a", text="More schoolwork.",
                           attachments=[{"observation": photo} for photo in photos[:4]] + [{"status": "unavailable"}])
    assert partial["interaction"] and "couldn't inspect 1" in partial["message"]
    followup = service.ask(child_id="kid-a", text="Give us one idea we can play aloud. No writing.", answer_mode="quick",
                           context_metadata={"related_event_id": context["event_id"],
                                             "parent_shared_image_context": photos, "learning_scope": "diagnostic"})
    assert followup["status"] == "completed"

    class BriefCaptured(Exception):
        pass

    def inspect_brief(self, *args, **kwargs):
        assert "No writing." in kwargs["question"]
        assert "one idea we can play aloud" in kwargs["question"]
        raise BriefCaptured

    monkeypatch.setattr(LearningArtifactService, "_design", inspect_brief)
    with pytest.raises(BriefCaptured):
        LearningArtifactService(db, service.output_dir, backend=AnswerBackend()).create_from_event(
            event_id=followup["event_id"], artifact_type="challenge")


def test_recovered_photo_batch_is_available_without_rewriting_parent_event(tmp_path, monkeypatch):
    from legacy_natural_routing_v2 import _offline_service

    from curiosity_engine.attachments import (
        persist_inbound_attachments,
        save_attachment_observation,
        source_photo_context,
    )
    from curiosity_engine.interaction import begin_receipt, finish_receipt
    from curiosity_engine.repository import Repository
    from curiosity_engine.transports.slack import _receive_slack_images

    service, db, binding = _offline_service(tmp_path, monkeypatch)
    transport = SlackTransport(db, service.output_dir, service=service)
    files = [{"id": f"F_{index}", "mimetype": "image/png", "size": len(_png_bytes()),
              "url_private_download": "https://files.slack.com/synthetic"} for index in range(5)]
    attachments = _receive_slack_images(transport, {"files": files}, _SlackClient(),
                                        downloader=lambda *_: _png_bytes())
    message = InboundMessage(external_event_id="recover-photos", team_id="T1", user_id="U1", channel_id="C1",
                             thread_id="photo-recovery", text="Schoolwork for context.", attachments=attachments[:3])
    assets = persist_inbound_attachments(db, service.output_dir, binding_id=binding, message=message)
    photos = [{"status": "ready", "summary": f"Worksheet {i}", "learning_concepts": [f"Concept {i}"]}
              for i in range(5)]
    for asset, photo in zip(assets, photos, strict=False):
        save_attachment_observation(db, asset["id"], photo)
        asset["observation"] = photo
    context = service.chat(binding_id=binding, team_id="T1", channel_id="C1", thread_id=message.thread_id,
                           child_id="kid-a", text=message.text, attachments=assets)
    assert begin_receipt(db, message, binding)
    finish_receipt(db, message, status="completed", event_id=context["event_id"])
    original = Repository(db).event(context["event_id"])
    assert source_photo_context(db, context["event_id"]) == photos[:3]
    recovered = persist_inbound_attachments(db, service.output_dir, binding_id=binding,
                                            message=message.model_copy(update={"attachments": attachments}))
    class VisionBackend:
        calls = 0

        def complete(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"summary": ""}  # One malformed read should recover.
            return {"summary": f"Worksheet {self.calls + 1}. " * 100,
                    "learning_concepts": [f"Concept {self.calls + 1}"],
                    "visible_details": [f"Visible detail {i}" for i in range(12)]}

    service.visual_backend = VisionBackend()
    observed = service._attachment_context(recovered)
    assert service.visual_backend.calls == 3  # Cached pages were not reread.
    assert len(observed) == 5 and all(photo["status"] == "ready" for photo in observed)
    assert [photo["learning_concepts"] for photo in observed] == [photo["learning_concepts"] for photo in photos]
    assert len(observed[-1]["summary"]) == 700 and len(observed[-1]["visible_details"]) == 10
    assert source_photo_context(db, context["event_id"]) == observed
    assert Repository(db).event(context["event_id"]) == original


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
        lambda arguments: calls.append(arguments) or {"status": "completed", "message": "Saved as context only."},
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


def test_school_email_context_cannot_be_normalized_into_an_unsolicited_lesson(tmp_path: Path):
    db = tmp_path / "private" / "data" / "curiosity.db"
    setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "kid-a", "Kid A", grade="1")
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=None,
        conversation_ref="conversation",
        thread_ref="teacher-email",
        child_id="kid-a",
    )
    tools = ToolRegistry()
    calls: list[dict[str, Any]] = []
    tools.register(
        "record_thread_context",
        lambda arguments: calls.append(arguments) or {"status": "completed", "message": "Saved as context only."},
    )
    tools.register("continue_learning_thread", lambda _arguments: pytest.fail("lesson tool ran"))
    tools.register("revise_learning_thread", lambda _arguments: {"status": "completed"})
    tools.register("create_learning_artifact", lambda _arguments: {"status": "completed"})
    tools.register("propose_weekly_checkin", lambda _arguments: {"status": "proposal"})
    tools.register("record_response_feedback", lambda _arguments: {"status": "completed"})
    email = "Sharing Kid A's teacher email: We practiced vowel teams today. Can your child find the long-a words?"

    result = ParentAgentRuntime(db, backend=_WrongLessonBackend(), tools=tools).run(
        session_id=str(session["id"]),
        user_message=email,
        child={"id": "kid-a", "grade": "1"},
        latest_event_id=None,
    )

    assert result["message"] == "Saved as context only."
    assert calls == [{"note": email}]


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
    assert message == ("The wing bends because air pushes on it. A bend changes how the wing redirects the moving air.")
    assert "Show" not in message and "Tiny explanation" not in message


def test_followup_renderer_never_claims_a_rejected_revision_succeeded():
    message = _proportionate_thread_message({"status": "rejected", "output": {}}, "How about a diagram?")
    assert message == "I couldn't make a reliable change, so I kept the last reviewed answer unchanged."
    assert "followed that question" not in message


def test_latest_thread_event_skips_a_rejected_revision(tmp_path: Path):
    db = tmp_path / "private" / "data" / "curiosity.db"
    setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "kid-a", "Kid A", grade="1")
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        conversation_ref="conversation",
        thread_ref="thread",
        child_id="kid-a",
    )
    now = utcnow()
    output = jdump({"hook": "Reviewed answer"})
    with connect(db) as conn:
        for event_id, status in (("evt_reviewed", "completed"), ("evt_rejected", "rejected")):
            conn.execute(
                """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                   VALUES(?,'child_question','kid-a','question','test','{}',?,?)""",
                (event_id, now, status),
            )
            conn.execute(
                """INSERT INTO responses(event_id,workflow,status,output_json,created_at,updated_at)
                   VALUES(?,'pull_thread',?,?,?,?)""",
                (event_id, status, output, now, now),
            )
    store = SessionStore(db)
    store.append_message(
        str(session["id"]),
        role="assistant",
        content="Reviewed answer",
        event_id="evt_reviewed",
    )
    store.append_message(
        str(session["id"]),
        role="assistant",
        content="Rejected notice",
        event_id="evt_rejected",
    )

    assert store.latest_event_id(str(session["id"])) == "evt_reviewed"


def test_visual_only_revision_queues_diagram_without_regenerating_lesson(tmp_path: Path):
    db = tmp_path / "private" / "data" / "curiosity.db"
    output_dir = tmp_path / "private" / "output"
    setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "kid-a", "Kid A", grade="1")
    service = CuriosityService(db, output_dir)
    now = utcnow()
    reviewed = {
        "hook": "Ice can rise above the starting line.",
        "show": "Mark the water line before freezing.",
        "ask": "Where did the level move?",
        "nugget": "Water usually needs more room after it freezes.",
        "next_possible_concepts": [],
        "physical_extension": None,
        "visual": None,
        "observation_candidates": [],
        "actions": [],
        "resource_refs": [],
    }
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_reviewed','child_question','kid-a','Why does water expand when it freezes?',
                      'test','{}',?,'completed')""",
            (now,),
        )
        conn.execute(
            """INSERT INTO responses(event_id,workflow,status,output_json,created_at,updated_at)
               VALUES('evt_reviewed','pull_thread','completed',?,?,?)""",
            (jdump(reviewed), now, now),
        )

    result = service.revise_visual_response(
        source_event_id="evt_reviewed",
        revision="How about a diagram?",
        event_id="evt_visual_revision",
    )

    assert result is not None and result["status"] == "completed"
    assert result["visual_job_id"]
    with connect(db) as conn:
        stored = conn.execute("SELECT status,workflow FROM responses WHERE event_id='evt_visual_revision'").fetchone()
    assert dict(stored) == {"status": "completed", "workflow": "visual_revision"}
