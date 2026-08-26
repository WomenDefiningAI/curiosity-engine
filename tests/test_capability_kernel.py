from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from curiosity_engine.artifact_delivery import enqueue_artifact_delivery
from curiosity_engine.capabilities import CapabilityRegistry
from curiosity_engine.contracts import InteractionEvent, InteractionOption, InteractionPlan, ScheduleProposal
from curiosity_engine.db import SCHEMA_VERSION, connect, init_db, jdump, utcnow
from curiosity_engine.graph import add_child
from curiosity_engine.host import unit_definitions
from curiosity_engine.interaction import create_pairing_code, ready_deliveries, setup_household
from curiosity_engine.interactions import create_interaction, interaction_blocks, resolve_interaction
from curiosity_engine.learning_artifacts import LearningArtifactService
from curiosity_engine.parent_agent import ParentAgentRuntime
from curiosity_engine.reasoning import StubBackend
from curiosity_engine.scheduler import SchedulerService
from curiosity_engine.sessions import SessionStore
from curiosity_engine.setup_agent import prepare_agent_setup
from curiosity_engine.tooling import ToolRegistry
from curiosity_engine.transports.contracts import InboundMessage
from curiosity_engine.transports.slack import (
    SlackTransport,
    _make_semantic_interaction_receiver,
    flush_slack_artifact_outbox,
)


def _family_db(tmp_path: Path) -> tuple[Path, dict[str, Any], str]:
    db = tmp_path / "private" / "data" / "curiosity.db"
    setup = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "kid-a", "Kid A", grade="1")
    binding_id = "binding-test"
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO transport_bindings(
                   id,transport,team_id,user_id,channel_id,parent_id,status,created_at,updated_at
               ) VALUES(?,'slack','T_FAMILY','U_PARENT','C_FAMILY',?,'active',?,?)""",
            (binding_id, setup["owner_id"], now, now),
        )
    return db, setup, binding_id


def _stored_response(db: Path, event_id: str = "evt-source") -> None:
    now = utcnow()
    output = {
        "hook": "An airplane wing is an air detective.",
        "show": "Compare a flat paper strip with a gently curved one.",
        "ask": "Which shape changes the air more?",
        "nugget": "A moving wing redirects air downward, and the air pushes the wing upward.",
        "resource_refs": [],
    }
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES(?,'child_question','kid-a','How do airplanes fly?','parent','{}',?,'completed')""",
            (event_id, now),
        )
        conn.execute(
            """INSERT INTO responses(event_id,run_id,workflow,status,output_json,created_at,updated_at)
               VALUES(?,NULL,'pull_thread','completed',?,?,?)""",
            (event_id, jdump(output), now, now),
        )


def test_schema_v12_migrates_with_backup_and_preserves_data(tmp_path: Path):
    db = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE family_marker(value TEXT NOT NULL)")
        conn.execute("INSERT INTO family_marker VALUES('keep-me')")
        conn.execute("PRAGMA user_version = 11")
    backup = init_db(db)
    assert backup and backup.is_file()
    with connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert conn.execute("SELECT value FROM family_marker").fetchone()[0] == "keep-me"
        assert conn.execute("PRAGMA table_info(agent_sessions)").fetchall()


def test_capabilities_are_progressively_disclosed_and_reviewed():
    registry = CapabilityRegistry()
    cards = registry.capability_cards({"artifact_worksheet"})
    assert [card["id"] for card in cards] == ["artifact_worksheet"]
    assert "worksheet" in registry.instructions_for("artifact_worksheet").casefold()
    with pytest.raises(KeyError):
        registry.load_skill("made-up-skill")


def test_interaction_tokens_are_opaque_bound_and_single_use(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    presented = create_interaction(
        db,
        binding_id=binding_id,
        plan=InteractionPlan(
            kind="choose_one",
            title="Pick a path",
            prompt="Which sounds fun?",
            options=[InteractionOption(label="Mystery", intent="choose_mystery", payload={"level": 1})],
        ),
    )
    assert "choose_mystery" not in presented["options"][0]["token"]
    assert interaction_blocks(presented)[1]["elements"][0]["action_id"]
    event = InteractionEvent(
        interaction_id=presented["interaction_id"],
        option_token=presented["options"][0]["token"],
        team_id="T_FAMILY",
        user_id="U_PARENT",
        channel_id="C_FAMILY",
        external_event_id="action-one",
    )
    resolved = resolve_interaction(db, event, binding_id=binding_id, parent_id=None)
    assert resolved["intent"] == "choose_mystery"
    assert resolve_interaction(db, event, binding_id=binding_id, parent_id=None)["duplicate"] is True
    replay = event.model_copy(update={"external_event_id": "action-two"})
    with pytest.raises(ValueError, match="no longer active"):
        resolve_interaction(db, replay, binding_id=binding_id, parent_id=None)


def test_interaction_buttons_have_unique_slack_action_ids(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    presented = create_interaction(
        db,
        binding_id=binding_id,
        plan=InteractionPlan(
            kind="rate_output",
            title="What next?",
            prompt="Tune this response.",
            options=[
                InteractionOption(label=f"Choice {index}", intent=f"choice_{index}")
                for index in range(4)
            ],
        ),
    )
    elements = interaction_blocks(presented)[1]["elements"]
    action_ids = [element["action_id"] for element in elements]
    assert len(action_ids) == len(set(action_ids)) == 4
    assert all(action_id.startswith("curiosity_interaction_button_") for action_id in action_ids)


def test_duplicate_semantic_slack_action_does_not_execute_tool_twice(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    presented = create_interaction(
        db,
        binding_id=binding_id,
        plan=InteractionPlan(
            kind="confirm_action",
            title="Do it?",
            prompt="One reviewed action.",
            options=[InteractionOption(label="Yes", intent="test_action")],
        ),
    )

    class ChoiceService:
        def __init__(self):
            self.calls = 0

        def handle_interaction_choice(self, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            return {"status": "completed", "message": "Done once."}

    class ChoiceClient:
        def __init__(self):
            self.posts: list[dict[str, Any]] = []
            self.updates: list[dict[str, Any]] = []

        def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
            self.posts.append(kwargs)
            return {"ts": "reply-1"}

        def chat_update(self, **kwargs: Any) -> None:
            self.updates.append(kwargs)

    service = ChoiceService()
    transport = SlackTransport(db, tmp_path / "output", service=service)
    receive = _make_semantic_interaction_receiver(transport, db)
    client = ChoiceClient()
    body = {
        "trigger_id": "trigger-one",
        "team": {"id": "T_FAMILY"},
        "user": {"id": "U_PARENT"},
        "channel": {"id": "C_FAMILY"},
        "message": {"ts": "root-1", "blocks": []},
    }
    action = {
        "block_id": f"curiosity_interaction:{presented['interaction_id']}",
        "value": presented["options"][0]["token"],
        "action_ts": "action-one",
    }
    acknowledgements: list[bool] = []
    receive(lambda: acknowledgements.append(True), body, action, client)
    receive(lambda: acknowledgements.append(True), body, action, client)
    assert acknowledgements == [True, True]
    assert service.calls == 1
    assert len(client.posts) == len(client.updates) == 1


def test_parent_agent_persists_turn_tool_release_and_scoped_approval(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref="conversation",
        thread_ref="thread",
        child_id="kid-a",
    )
    tools = ToolRegistry()
    tools.register(
        "continue_learning_thread",
        lambda arguments: {"status": "completed", "message": f"Built on: {arguments['message']}"},
    )
    tools.register("revise_learning_thread", lambda arguments: {"status": "completed", "message": "Revised"})
    tools.register("create_learning_artifact", lambda arguments: {"status": "completed", "message": "Made it"})
    tools.register("propose_weekly_checkin", lambda arguments: {"status": "proposal"})
    tools.register("record_response_feedback", lambda arguments: {"status": "completed"})
    runtime = ParentAgentRuntime(db, backend=StubBackend(), tools=tools)
    result = runtime.run(
        session_id=session["id"],
        user_message="Why does the other wing bend?",
        child={"id": "kid-a", "grade": "1"},
        latest_event_id=None,
    )
    assert result["message"].startswith("Built on")
    approval = runtime.run(
        session_id=session["id"],
        user_message="Set up a weekly check-in Sunday at 6 pm",
        child={"id": "kid-a", "grade": "1"},
        latest_event_id=None,
    )
    option_payload = approval["interaction"]["options"][0]["payload"]
    assert set(option_payload) == {"tool_call_id"}
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_turns").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM release_units").fetchone()[0] >= 2
        row = conn.execute("SELECT decision,scope_hash FROM approval_requests").fetchone()
    assert row["decision"] == "pending" and len(row["scope_hash"]) == 64


class _MisroutingBackend:
    name = "test-model"
    model = "test-model-v1"

    def complete(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "tool_calls": [
                {
                    "name": "revise_learning_thread",
                    "arguments": {"revision": "generic"},
                    "rationale": "The parent asked for a revision.",
                }
            ]
        }


def test_printable_critique_is_routed_back_to_artifact_tool(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref="conversation",
        thread_ref="thread",
        child_id="kid-a",
    )
    tools = ToolRegistry()
    created: list[dict[str, Any]] = []
    tools.register("continue_learning_thread", lambda _arguments: {"status": "completed"})
    tools.register("revise_learning_thread", lambda _arguments: {"status": "completed"})
    tools.register(
        "create_learning_artifact",
        lambda arguments: created.append(arguments) or {"status": "completed", "message": "Rebuilt"},
    )
    tools.register("propose_weekly_checkin", lambda _arguments: {"status": "proposal"})
    tools.register("record_response_feedback", lambda _arguments: {"status": "completed"})
    runtime = ParentAgentRuntime(db, backend=_MisroutingBackend(), tools=tools)
    runtime.run(
        session_id=session["id"],
        user_message="The printable is too generic; include all three wing shapes.",
        child={"id": "kid-a", "grade": "1"},
        latest_event_id=None,
    )
    assert created == [
        {
            "artifact_type": "challenge",
            "revision": "The printable is too generic; include all three wing shapes.",
        }
    ]


def test_three_artifact_contracts_render_real_distinct_one_page_outputs(tmp_path: Path):
    db, _setup, _binding_id = _family_db(tmp_path)
    _stored_response(db)
    service = LearningArtifactService(db, tmp_path / "output", backend=StubBackend())
    artifacts = {
        kind: service.create_from_event(event_id="evt-source", artifact_type=kind)
        for kind in ("worksheet", "activity", "challenge")
    }
    assert {item["artifact_type"] for item in artifacts.values()} == {"worksheet", "activity", "challenge"}
    assert len({Path(item["pdf_path"]).read_bytes() for item in artifacts.values()}) == 3
    assert all(item["validation"]["structural"] == "pass" for item in artifacts.values())
    with connect(db) as conn:
        mechanics = {
            json.loads(row["spec_json"])["artifact_type"]
            for row in conn.execute("SELECT spec_json FROM artifacts")
        }
    assert mechanics == {"worksheet", "activity", "challenge"}


def test_challenge_fallback_preserves_named_comparison_and_play_sequence(tmp_path: Path):
    db, _setup, _binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source",
        artifact_type="challenge",
        revision="Compare folded-up, flat, and folded-down wing shapes without repeating the old activity.",
    )
    with connect(db) as conn:
        spec = json.loads(
            conn.execute("SELECT spec_json FROM artifacts WHERE id=?", (artifact["artifact_id"],)).fetchone()[0]
        )
    child_text = json.dumps(spec).casefold()
    assert "folded-up" in child_text and "folded-down" in child_text
    assert len(spec["steps"]) >= 3
    assert "same starting place" in child_text
    assert spec["evidence_rows"] == ["Folded-up wing", "Flat wing", "Folded-down wing"]
    assert LearningArtifactService._comparison_rows(
        "named rows for folded-up, flat, and folded-down wings, plus two trial boxes"
    ) == ["Folded-up wing", "Flat wing", "Folded-down wing"]


def test_model_artifact_surplus_lists_are_bounded_before_validation():
    candidate = {
        "constraints": [f"Rule {index}" for index in range(8)],
        "materials": [f"Item {index}" for index in range(10)],
        "steps": [f"Step {index}" for index in range(9)],
        "hints": [f"Hint {index}" for index in range(6)],
        "evidence_rows": [f"Version {index}" for index in range(6)],
    }
    bounded = LearningArtifactService._bound_candidate_lists(candidate, "challenge")
    assert {key: len(value) for key, value in bounded.items()} == {
        "constraints": 5,
        "materials": 8,
        "steps": 5,
        "hints": 3,
        "evidence_rows": 4,
    }
    assert LearningArtifactService._clip_at_sentence(
        "Keep this complete first idea; this second clause is much too long for the panel.",
        32,
    ) == "Keep this complete first idea."
    assert LearningArtifactService._compact_evidence_row(
        "DOWN — both back wing edges folded down"
    ) == "DOWN — folded down"


class _ArtifactSlackClient:
    def __init__(self):
        self.completed: list[dict[str, Any]] = []

    def files_getUploadURLExternal(self, **kwargs: Any) -> dict[str, str]:
        return {"upload_url": "https://upload.slack.example/test", "file_id": "F_TEST"}

    def files_completeUploadExternal(self, **kwargs: Any) -> dict[str, Any]:
        self.completed.append(kwargs)
        return {"files": [{"id": "F_TEST"}]}


def test_validated_artifact_upload_uses_hash_and_thread(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge"
    )
    enqueue_artifact_delivery(
        db,
        artifact=artifact,
        binding_id=binding_id,
        channel_id="C_FAMILY",
        thread_id="123.4",
        idempotency_key="artifact-one",
    )
    client = _ArtifactSlackClient()
    uploaded: list[bytes] = []
    results = flush_slack_artifact_outbox(
        client,
        db,
        tmp_path / "output",
        uploader=lambda _url, data: uploaded.append(data),
    )
    assert len(results) == 1 and results[0]["status"] == "sent"
    assert uploaded[0].startswith(b"%PDF-")
    assert client.completed[0]["thread_ts"] == "123.4"


def test_scheduler_materializes_once_and_does_not_create_child_evidence(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    scheduler = SchedulerService(db)
    scheduled = scheduler.create_weekly_checkin(
        ScheduleProposal(
            child_id="kid-a",
            weekday="sunday",
            local_time="18:00",
            timezone="Etc/UTC",
            channel_id="C_FAMILY",
            binding_id=binding_id,
        )
    )
    due = datetime.now(UTC) - timedelta(minutes=1)
    with connect(db) as conn:
        conn.execute("UPDATE schedules SET next_run_at=? WHERE id=?", (due.isoformat(), scheduled["schedule_id"]))
    assert len(scheduler.run_due(now=datetime.now(UTC))) == 1
    assert scheduler.run_due(now=datetime.now(UTC)) == []
    assert ready_deliveries(db)[0]["payload"]["blocks"]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0
    assert scheduler.pause(scheduled["schedule_id"])["status"] == "paused"


class _ConversationService:
    def __init__(self, db: Path):
        self.db = db
        self.chat_messages: list[str] = []

    def children(self) -> list[dict[str, Any]]:
        return [{"id": "kid-a", "name": "Kid A", "grade": "1"}]

    def ask(self, **kwargs: Any) -> dict[str, Any]:
        now = utcnow()
        output = {"hook": "Look up!", "show": "Try a paper wing.", "ask": "What changes?", "nugget": "Air pushes back."}
        with connect(self.db) as conn:
            conn.execute(
                """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                   VALUES(?,'child_question',?,?,?,'{}',?,'completed')""",
                (kwargs["event_id"], kwargs["child_id"], kwargs["text"], kwargs["source"], now),
            )
            conn.execute(
                """INSERT INTO responses(event_id,run_id,workflow,status,output_json,created_at,updated_at)
                   VALUES(?,NULL,'pull_thread','completed',?,?,?)""",
                (kwargs["event_id"], jdump(output), now, now),
            )
        return {
            "status": "completed",
            "event_id": kwargs["event_id"],
            "output": output,
        }

    def record_thread_response(self, **kwargs: Any) -> str:
        conversation_ref = sha256(f"{kwargs['team_id']}:{kwargs['channel_id']}".encode()).hexdigest()[:20]
        thread_ref = sha256(
            f"{kwargs['team_id']}:{kwargs['channel_id']}:{kwargs['thread_id']}".encode()
        ).hexdigest()[:20]
        return SessionStore(self.db).get_or_create(
            origin="slack",
            transport="slack",
            binding_id=kwargs["binding_id"],
            conversation_ref=conversation_ref,
            thread_ref=thread_ref,
            child_id=kwargs["child_id"],
        )["id"]

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.chat_messages.append(kwargs["text"])
        store = SessionStore(self.db)
        conversation_ref = sha256(f"{kwargs['team_id']}:{kwargs['channel_id']}".encode()).hexdigest()[:20]
        thread_ref = sha256(
            f"{kwargs['team_id']}:{kwargs['channel_id']}:{kwargs['thread_id']}".encode()
        ).hexdigest()[:20]
        session = store.get_or_create(
            origin="slack",
            transport="slack",
            binding_id=kwargs["binding_id"],
            conversation_ref=conversation_ref,
            thread_ref=thread_ref,
            child_id="kid-a",
        )
        return {"status": "completed", "message": "I changed the diagram without repeating the activity.", "session_id": session["id"]}


def test_public_slack_thread_routes_natural_parent_tuning_to_agent(tmp_path: Path):
    db = tmp_path / "private" / "data" / "curiosity.db"
    setup = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "kid-a", "Kid A", grade="1")
    service = _ConversationService(db)
    transport = SlackTransport(db, tmp_path / "output", service=service)
    code = create_pairing_code(db, setup["owner_id"])["pairing_code"]
    base = dict(team_id="T_FAMILY", user_id="U_PARENT", channel_id="C_FAMILY", thread_id="100.1")
    assert transport.handle(InboundMessage(external_event_id="pair", text=f"pair {code}", **base)).status == "paired"
    assert transport.handle(InboundMessage(external_event_id="ask", text="ask kid-a: How do airplanes fly?", **base)).status == "completed"
    follow = transport.handle(
        InboundMessage(
            external_event_id="follow",
            text="This image isn't helpful; show wing variations instead.",
            **base,
        )
    )
    assert follow.status == "completed"
    assert service.chat_messages == ["This image isn't helpful; show wing variations instead."]


def test_pre_upgrade_learning_thread_is_recovered_for_natural_chat(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    service = _ConversationService(db)
    transport = SlackTransport(db, tmp_path / "output", service=service)
    thread_id = "88.1"
    conversation_ref = sha256(b"T_FAMILY:C_FAMILY").hexdigest()[:20]
    thread_ref = sha256(f"T_FAMILY:C_FAMILY:{thread_id}".encode()).hexdigest()[:20]
    now = utcnow()
    output = {"hook": "Look up!", "show": "Compare two wings.", "ask": "What changed?", "nugget": "Air pushes back."}
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt-old','child_question','kid-a','How do planes fly?','parent',?,?,'completed')""",
            (jdump({"conversation_ref": conversation_ref, "thread_ref": thread_ref}), now),
        )
        conn.execute(
            """INSERT INTO responses(event_id,run_id,workflow,status,output_json,created_at,updated_at)
               VALUES('evt-old',NULL,'pull_thread','completed',?,?,?)""",
            (jdump(output), now, now),
        )
        conn.execute(
            """INSERT INTO transport_receipts(
                   transport,external_event_id,payload_hash,binding_id,status,event_id,received_at,processed_at
               ) VALUES('slack','old-slack-event','hash',?,'completed','evt-old',?,?)""",
            (binding_id, now, now),
        )
    result = transport.handle(
        InboundMessage(
            external_event_id="new-followup",
            team_id="T_FAMILY",
            user_id="U_PARENT",
            channel_id="C_FAMILY",
            thread_id=thread_id,
            text="Show me curved and flat wings instead.",
        )
    )
    assert result.status == "completed"
    assert service.chat_messages == ["Show me curved and flat wings instead."]
    with connect(db) as conn:
        assert conn.execute("SELECT child_id FROM agent_sessions").fetchone()[0] == "kid-a"


def test_no_clone_setup_creates_private_agent_handoff_and_safe_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "family-home"
    monkeypatch.setenv("CURIOSITY_HOME", str(home))
    monkeypatch.setattr("curiosity_engine.setup_agent.shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    state = prepare_agent_setup()
    assert state["agent"] == "codex"
    assert (home / "workspace" / "AGENTS.md").stat().st_mode & 0o077 == 0
    handoff = (home / "workspace" / "SETUP_HANDOFF.md").read_text(encoding="utf-8")
    assert "Slack" in handoff and "model" in handoff and "printable" in handoff
    units = unit_definitions("/opt/curiosity/bin/curiosity")
    combined = "\n".join(units.values())
    assert "worker\" \"--forever" in combined
    assert "xoxb-" not in combined and "OPENAI_API_KEY" not in combined
