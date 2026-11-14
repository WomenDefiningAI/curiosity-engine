from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

import curiosity_engine.transports.slack as slack_module
from curiosity_engine.artifact_delivery import (
    claim_artifact_delivery,
    enqueue_artifact_delivery,
    mark_artifact_delivery,
    ready_artifact_deliveries,
)
from curiosity_engine.capabilities import CapabilityRegistry
from curiosity_engine.contracts import InteractionEvent, InteractionOption, InteractionPlan, ScheduleProposal
from curiosity_engine.db import SCHEMA_VERSION, connect, init_db, jdump, utcnow
from curiosity_engine.graph import add_child
from curiosity_engine.host import unit_definitions
from curiosity_engine.interaction import (
    consume_pairing_code,
    create_pairing_code,
    ready_deliveries,
    revoke_binding,
    setup_household,
)
from curiosity_engine.interactions import create_interaction, interaction_blocks, resolve_interaction
from curiosity_engine.learning_artifacts import LearningArtifactService
from curiosity_engine.parent_agent import ParentAgentRuntime
from curiosity_engine.reasoning import StubBackend
from curiosity_engine.scheduler import SchedulerService
from curiosity_engine.service import CuriosityService, _explicit_thread_preference_change
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


def test_schema_v16_migrates_with_backup_and_preserves_data(tmp_path: Path):
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
        assert conn.execute("PRAGMA table_info(agent_session_preference_events)").fetchall()


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
            options=[InteractionOption(label=f"Choice {index}", intent=f"choice_{index}") for index in range(4)],
        ),
    )
    elements = interaction_blocks(presented)[1]["elements"]
    action_ids = [element["action_id"] for element in elements]
    assert len(action_ids) == len(set(action_ids)) == 4
    assert all(action_id.startswith("curiosity_interaction_button_") for action_id in action_ids)


def test_duplicate_semantic_slack_action_does_not_execute_tool_twice(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref="interaction-idempotency",
        thread_ref="interaction-idempotency",
        child_id="kid-a",
    )
    presented = create_interaction(
        db,
        binding_id=binding_id,
        session_id=str(session["id"]),
        plan=InteractionPlan(
            kind="confirm_action",
            title="Do it?",
            prompt="One reviewed action.",
            options=[
                InteractionOption(
                    label="Make it",
                    intent="create_artifact",
                    payload={"event_id": "evt-source", "artifact_type": "challenge"},
                )
            ],
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
    assert len(client.posts) == 1
    assert len(client.updates) == 2
    assert "Drawing and building the challenge" in client.updates[0]["blocks"][-1]["elements"][0]["text"]
    assert "Challenge ready; sending the picture and PDF" in client.updates[1]["blocks"][-1]["elements"][0]["text"]


def test_failed_artifact_choice_keeps_context_and_offers_a_fresh_retry(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    session = SessionStore(db).get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref="interaction-retry",
        thread_ref="interaction-retry",
        child_id="kid-a",
    )
    presented = create_interaction(
        db,
        binding_id=binding_id,
        session_id=str(session["id"]),
        plan=InteractionPlan(
            kind="confirm_action",
            title="Make it?",
            prompt="Create one reviewed printable.",
            options=[
                InteractionOption(
                    label="Make challenge",
                    intent="create_artifact",
                    payload={"event_id": "evt-source", "artifact_type": "challenge"},
                )
            ],
        ),
    )

    class FailingService:
        def handle_interaction_choice(self, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("artifact did not pass semantic validation")

    class ChoiceClient:
        def __init__(self):
            self.updates: list[dict[str, Any]] = []

        def chat_update(self, **kwargs: Any) -> None:
            self.updates.append(kwargs)

    transport = SlackTransport(db, tmp_path / "output", service=FailingService())
    receive = _make_semantic_interaction_receiver(transport, db)
    client = ChoiceClient()
    receive(
        lambda: None,
        {
            "trigger_id": "trigger-failure",
            "team": {"id": "T_FAMILY"},
            "user": {"id": "U_PARENT"},
            "channel": {"id": "C_FAMILY"},
            "message": {"ts": "root-failure", "blocks": []},
        },
        {
            "block_id": f"curiosity_interaction:{presented['interaction_id']}",
            "value": presented["options"][0]["token"],
            "action_ts": "action-failure",
        },
        client,
    )

    assert len(client.updates) == 2
    final_blocks = json.dumps(client.updates[-1]["blocks"])
    assert "did not send a template" in final_blocks
    assert "Try again" in final_blocks
    with connect(db) as conn:
        statuses = [
            row[0] for row in conn.execute("SELECT status FROM interaction_instances ORDER BY created_at").fetchall()
        ]
    assert statuses == ["completed", "active"]


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


class _RecallThenReviseBackend:
    name = "test-model"
    model = "test-model-v1"

    def __init__(self):
        self.payloads: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs["payload"]
        self.payloads.append(payload)
        if len(self.payloads) == 1:
            assert all(
                "air detective" not in item["snippet"].casefold() for item in payload["workspace"]["recent_outputs"]
            )
            return {
                "tool_calls": [
                    {
                        "name": "search_thread_outputs",
                        "arguments": {"query": "air detective wing"},
                        "rationale": "The parent referred to an older answer outside the recent window.",
                    }
                ],
                "done": False,
            }
        match = payload["tool_results"][0]["matches"][0]
        return {
            "tool_calls": [
                {
                    "name": "revise_learning_thread",
                    "arguments": {
                        "target_ref": match["ref_id"],
                        "revision": "Show three wing variations without repeating the activity.",
                    },
                    "rationale": "The bounded lookup resolved the requested earlier answer.",
                }
            ],
            "done": True,
        }


def test_parent_agent_can_recall_an_old_thread_output_then_act_once(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db, "evt-earlier")
    store = SessionStore(db)
    session = store.get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref="conversation",
        thread_ref="thread",
        child_id="kid-a",
    )
    store.append_message(
        session["id"], role="assistant", content="An airplane wing is an air detective.", event_id="evt-earlier"
    )
    for index in range(18):
        store.append_message(session["id"], role="assistant", content=f"Later idea number {index}.")
    backend = _RecallThenReviseBackend()
    revised: list[dict[str, Any]] = []
    tools = ToolRegistry()
    tools.register(
        "search_thread_outputs",
        lambda arguments: {
            "status": "completed",
            "matches": store.search_outputs(session["id"], str(arguments["query"])),
        },
    )
    tools.register(
        "revise_learning_thread",
        lambda arguments: (
            revised.append(arguments) or {"status": "completed", "message": "Revised the earlier wing visual."}
        ),
    )
    result = ParentAgentRuntime(db, backend=backend, tools=tools).run(
        session_id=session["id"],
        user_message="Change that earlier air-detective answer to show three wing variations.",
        child={"id": "kid-a", "grade": "1"},
        latest_event_id="evt-earlier",
    )
    assert result["message"] == "Revised the earlier wing visual."
    assert revised[0]["target_ref"].startswith("msg_")
    assert len(backend.payloads) == 2
    with connect(db) as conn:
        calls = conn.execute("SELECT ordinal,tool_name FROM tool_calls ORDER BY ordinal").fetchall()
        releases = conn.execute("SELECT unit_type FROM release_units").fetchall()
    assert [(row["ordinal"], row["tool_name"]) for row in calls] == [
        (1, "search_thread_outputs"),
        (2, "revise_learning_thread"),
    ]
    assert [row["unit_type"] for row in releases] == [
        "status",
        "status",
        "answer",
        "status",
        "status",
    ]


def test_service_validates_recalled_revision_target_inside_current_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db, "evt-earlier")
    backend = _RecallThenReviseBackend()
    monkeypatch.setattr("curiosity_engine.service.configured_backend", lambda *_args, **_kwargs: backend)
    service = CuriosityService(db, tmp_path / "output")
    session_id = service.record_thread_response(
        binding_id=binding_id,
        team_id="T_FAMILY",
        channel_id="C_FAMILY",
        thread_id="thread-root",
        child_id="kid-a",
        user_text="How do airplanes fly?",
        result={
            "status": "completed",
            "event_id": "evt-earlier",
            "output": {
                "hook": "An airplane wing is an air detective.",
                "show": "Compare two wings.",
                "ask": "What changes?",
                "nugget": "Air pushes back.",
                "resource_refs": [],
            },
        },
    )
    store = SessionStore(db)
    for index in range(18):
        store.append_message(session_id, role="assistant", content=f"Later idea number {index}.")
    revised: list[dict[str, Any]] = []

    def fake_revision(**kwargs: Any) -> dict[str, Any]:
        revised.append(kwargs)
        return {
            "status": "completed",
            "event_id": "evt-earlier",
            "output": {
                "hook": "Three wings, three air paths.",
                "nugget": "Changing a wing shape changes how it redirects air.",
                "show": "Compare flat, curved-up, and curved-down wings.",
            },
        }

    monkeypatch.setattr(service, "revise_response", fake_revision)
    result = service.chat(
        binding_id=binding_id,
        team_id="T_FAMILY",
        channel_id="C_FAMILY",
        thread_id="thread-root",
        text="Change that earlier air-detective answer to show three wing variations.",
    )
    assert result["status"] == "completed"
    assert revised[0]["source_event_id"] == "evt-earlier"


class _PreferenceBackend:
    name = "test-model"
    model = "test-model-v1"

    def __init__(self):
        self.payloads: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        payload = kwargs["payload"]
        self.payloads.append(payload)
        if len(self.payloads) == 1:
            return {
                "tool_calls": [
                    {
                        "name": "update_thread_preference",
                        "arguments": {
                            "operation": "set",
                            "category": "visual_style",
                            "value": "Use useful comparison diagrams instead of decorative art.",
                        },
                        "rationale": "The parent explicitly requested future behavior in this thread.",
                    }
                ],
                "done": False,
            }
        return {"message": "I’ll use the saved working preference.", "done": True}


def test_explicit_thread_preference_is_reversible_local_and_stops_after_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    backend = _PreferenceBackend()
    monkeypatch.setattr("curiosity_engine.service.configured_backend", lambda *_args, **_kwargs: backend)
    service = CuriosityService(db, tmp_path / "output")
    service.record_thread_response(
        binding_id=binding_id,
        team_id="T_FAMILY",
        channel_id="C_FAMILY",
        thread_id="thread-root",
        child_id="kid-a",
        user_text="How do airplanes fly?",
        result={
            "status": "completed",
            "event_id": "evt-source",
            "output": {
                "hook": "An airplane wing is an air detective.",
                "show": "Compare two wings.",
                "ask": "What changes?",
                "nugget": "Air pushes back.",
                "resource_refs": [],
            },
        },
    )
    saved = service.chat(
        binding_id=binding_id,
        team_id="T_FAMILY",
        channel_id="C_FAMILY",
        thread_id="thread-root",
        text="From now on in this thread, use useful comparison diagrams instead of decorative art.",
    )
    assert "saved for this thread" in saved["message"]
    assert len(backend.payloads) == 1
    session_id = saved["session_id"]
    assert SessionStore(db).active_preferences(session_id) == []
    assert SessionStore(db).active_parent_memories(session_id)[0]["category"] == "visual_style"
    follow = service.chat(
        binding_id=binding_id,
        team_id="T_FAMILY",
        channel_id="C_FAMILY",
        thread_id="thread-root",
        text="What working preference are you using?",
    )
    assert follow["message"] == "I’ll use the saved working preference."
    assert backend.payloads[1]["workspace"]["parent_memories"][0]["category"] == "visual_style"
    assert _explicit_thread_preference_change("This picture is not useful; change it.", "set") is False
    assert _explicit_thread_preference_change("Forget that visual preference.", "clear") is True

    cleared = service.chat(
        binding_id=binding_id,
        team_id="T_FAMILY",
        channel_id="C_FAMILY",
        thread_id="thread-root",
        text="Forget that visual preference.",
    )
    assert "cleared" in cleared["message"]
    assert SessionStore(db).active_preferences(session_id) == []
    assert SessionStore(db).active_parent_memories(session_id) == []
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_session_preference_events").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM parent_memories WHERE value!=''").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 0
        conn.execute("DELETE FROM agent_sessions WHERE id=?", (session_id,))
        assert conn.execute("SELECT COUNT(*) FROM agent_session_preference_events").fetchone()[0] == 0


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
            "artifact_type": "worksheet",
            "revision": "The printable is too generic; include all three wing shapes.",
        }
    ]


def test_four_artifact_contracts_render_real_distinct_one_page_outputs(tmp_path: Path):
    db, _setup, _binding_id = _family_db(tmp_path)
    _stored_response(db)
    service = LearningArtifactService(db, tmp_path / "output", backend=StubBackend())
    artifacts = {
        kind: service.create_from_event(event_id="evt-source", artifact_type=kind)
        for kind in ("worksheet", "reference", "activity", "challenge")
    }
    assert {item["artifact_type"] for item in artifacts.values()} == {
        "worksheet",
        "reference",
        "activity",
        "challenge",
    }
    assert len({Path(item["pdf_path"]).read_bytes() for item in artifacts.values()}) == 4
    assert all(item["validation"]["structural"] == "pass" for item in artifacts.values())
    with connect(db) as conn:
        mechanics = {
            json.loads(row["spec_json"])["artifact_type"] for row in conn.execute("SELECT spec_json FROM artifacts")
        }
    assert mechanics == {"worksheet", "reference", "activity", "challenge"}


def test_reviewed_activity_printable_becomes_child_usable_target_page(tmp_path: Path):
    db, _setup, _binding_id = _family_db(tmp_path)
    now = utcnow()
    output = {
        "hook": "Sauropods could reach leaves at different heights.",
        "show": "Compare three leaf targets.",
        "ask": "Which leaf was easiest to reach?",
        "nugget": "A long neck helped reach plants without moving the whole body.",
        "resource_refs": [],
        "physical_extension": {
            "title": "Sauropod leaf reach",
            "instructions": [
                "Keep the printed comparison whole and flat on a table.",
                "Trace the SHORT NECK and LONG NECK lines from SAME START.",
                "Point to the fixed leaves each line reaches, compare by length, and tell why.",
            ],
            "materials": ["printed Long-Neck Leaf Test page"],
            "parent_effort": "low",
            "printable": {
                "kind": "reach_comparison",
                "title": "Long-Neck Leaf Test",
                "child_directions": (
                    "Trace both printed necks from SAME START. Point to the fixed leaves each reaches, then compare."
                ),
                "parent_setup": "Keep this page whole and flat; the START, leaf distances, and neck lines are built in.",
                "pedagogical_value": "The fixed page compares how neck length changes reach.",
                "pieces": [
                    {"label": "NEAR LEAF", "prompt": "fixed at 2 units", "shape": "leaf"},
                    {"label": "MIDDLE LEAF", "prompt": "fixed at 4 units", "shape": "leaf"},
                    {"label": "FAR LEAF", "prompt": "fixed at 6 units", "shape": "leaf"},
                    {"label": "SHORT NECK", "prompt": "3 units from SAME START", "shape": "card"},
                    {"label": "LONG NECK", "prompt": "6 units from SAME START", "shape": "card"},
                    {"label": "HELP USED?", "prompt": "point: YES / NO", "shape": "card"},
                ],
            },
        },
    }
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt-leaves','child_question','kid-a','Tell me about sauropods','parent','{}',?,'completed')""",
            (now,),
        )
        conn.execute(
            """INSERT INTO responses(event_id,run_id,workflow,status,output_json,created_at,updated_at)
               VALUES('evt-leaves',NULL,'pull_thread','completed',?,?,?)""",
            (jdump(output), now, now),
        )
    artifact = LearningArtifactService(
        db,
        tmp_path / "output",
        backend=StubBackend(),
    ).create_from_extension(event_id="evt-leaves")
    assert artifact["artifact_type"] == "activity"
    assert Path(artifact["pdf_path"]).read_bytes().startswith(b"%PDF-")
    assert Path(artifact["preview_path"]).stat().st_size > 1_000
    with connect(db) as conn:
        spec = json.loads(
            conn.execute("SELECT spec_json FROM artifacts WHERE id=?", (artifact["artifact_id"],)).fetchone()[0]
        )
    assert spec["printable"]["kind"] == "reach_comparison"
    assert [piece["label"] for piece in spec["printable"]["pieces"]] == [
        "NEAR LEAF",
        "MIDDLE LEAF",
        "FAR LEAF",
        "SHORT NECK",
        "LONG NECK",
        "HELP USED?",
    ]
    child_facing = json.dumps(spec).casefold()
    assert "climbing" not in child_facing
    assert "stretch" not in child_facing


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
        "constraints": 3,
        "materials": 5,
        "steps": 4,
        "hints": 2,
        "evidence_rows": 4,
    }
    assert (
        LearningArtifactService._clip_at_sentence(
            "Keep this complete first idea; this second clause is much too long for the panel.",
            32,
        )
        == "Keep this complete first idea."
    )
    assert (
        LearningArtifactService._compact_evidence_row("DOWN — both back wing edges folded down") == "DOWN — folded down"
    )


class _ArtifactSlackClient:
    def __init__(self, *, ticket_error: Exception | None = None):
        self.tickets: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []
        self.ticket_error = ticket_error

    def files_getUploadURLExternal(self, **kwargs: Any) -> dict[str, str]:
        self.tickets.append(kwargs)
        if self.ticket_error:
            raise self.ticket_error
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


def test_artifact_claim_fails_after_ready_then_binding_revoke(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge"
    )
    delivery_id = enqueue_artifact_delivery(
        db,
        artifact=artifact,
        binding_id=binding_id,
        channel_id="C_FAMILY",
        thread_id="123.4",
        idempotency_key="artifact-revoke",
    )
    assert [row["id"] for row in ready_artifact_deliveries(db)] == [delivery_id]
    revoke_binding(db, binding_id)
    assert claim_artifact_delivery(db, delivery_id) is None
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM artifact_file_outbox WHERE id=?", (delivery_id,)
        ).fetchone()["status"] == "expired"


def test_artifact_claim_fails_when_parent_principal_is_revoked(tmp_path: Path):
    db, setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge"
    )
    delivery_id = enqueue_artifact_delivery(
        db,
        artifact=artifact,
        binding_id=binding_id,
        channel_id="C_FAMILY",
        thread_id=None,
        idempotency_key="artifact-principal-revoke",
    )
    assert [row["id"] for row in ready_artifact_deliveries(db)] == [delivery_id]
    with connect(db) as conn:
        conn.execute(
            "UPDATE parent_principals SET status='revoked' WHERE id=?",
            (setup["owner_id"],),
        )
    assert claim_artifact_delivery(db, delivery_id) is None
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM artifact_file_outbox WHERE id=?", (delivery_id,)
        ).fetchone()["status"] == "expired"


def test_artifact_worker_stops_after_claim_then_revoke_before_upload_ticket(
    tmp_path: Path,
    monkeypatch,
):
    db, setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge"
    )
    delivery_id = enqueue_artifact_delivery(
        db,
        artifact=artifact,
        binding_id=binding_id,
        channel_id="C_FAMILY",
        thread_id=None,
        idempotency_key="artifact-inflight-revoke",
    )
    original_claim = slack_module.claim_artifact_delivery
    stale_claim: dict[str, str] = {}

    def revoke_after_claim(db_path, candidate_id):
        claimed = original_claim(db_path, candidate_id)
        if claimed:
            stale_claim.update(claimed)
            revoke_binding(db_path, binding_id)
        return claimed

    monkeypatch.setattr(slack_module, "claim_artifact_delivery", revoke_after_claim)
    client = _ArtifactSlackClient()
    result = flush_slack_artifact_outbox(
        client,
        db,
        tmp_path / "output",
        uploader=lambda _url, _data: None,
    )
    assert result == [{"delivery_id": delivery_id, "status": "unknown"}]
    assert client.tickets == []
    assert (
        mark_artifact_delivery(
            db,
            delivery_id,
            status="failed",
            attempt_id=stale_claim["attempt_id"],
            lease_token=stale_claim["lease_token"],
            error="late worker",
        )
        is False
    )

    code = create_pairing_code(db, setup["owner_id"])["pairing_code"]
    repaired = consume_pairing_code(
        db,
        str(code),
        InboundMessage(
            external_event_id="artifact-repair",
            team_id="T_FAMILY",
            user_id="U_PARENT",
            channel_id="C_FAMILY",
            text=f"pair {code}",
        ),
    )
    assert repaired["binding_id"] == binding_id
    assert ready_artifact_deliveries(db) == []


def test_artifact_enqueue_rejects_cross_channel_target(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge"
    )
    with pytest.raises(ValueError, match="binding is unavailable"):
        enqueue_artifact_delivery(
            db,
            artifact=artifact,
            binding_id=binding_id,
            channel_id="C_OTHER",
            thread_id=None,
            idempotency_key="artifact-cross-channel",
        )


def test_artifact_cross_channel_tamper_is_fenced_at_claim(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge"
    )
    delivery_id = enqueue_artifact_delivery(
        db,
        artifact=artifact,
        binding_id=binding_id,
        channel_id="C_FAMILY",
        thread_id=None,
        idempotency_key="artifact-cross-channel-tamper",
    )
    assert [row["id"] for row in ready_artifact_deliveries(db)] == [delivery_id]
    with connect(db) as conn:
        conn.execute(
            "UPDATE artifact_file_outbox SET channel_id='C_OTHER' WHERE id=?",
            (delivery_id,),
        )
    assert claim_artifact_delivery(db, delivery_id) is None
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM artifact_file_outbox WHERE id=?", (delivery_id,)
        ).fetchone()["status"] == "expired"


def test_stale_artifact_worker_cannot_advance_a_recovered_later_claim(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge"
    )
    delivery_id = enqueue_artifact_delivery(
        db,
        artifact=artifact,
        binding_id=binding_id,
        channel_id="C_FAMILY",
        thread_id=None,
        idempotency_key="artifact-aba-recovery",
    )
    stale = claim_artifact_delivery(db, delivery_id)
    assert stale
    with connect(db) as conn:
        conn.execute(
            "UPDATE artifact_file_outbox SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (delivery_id,),
        )
    assert [row["id"] for row in ready_artifact_deliveries(db)] == [delivery_id]
    current = claim_artifact_delivery(db, delivery_id)
    assert current and current["attempt_id"] != stale["attempt_id"]
    assert (
        mark_artifact_delivery(
            db,
            delivery_id,
            status="ticket_acquiring",
            attempt_id=stale["attempt_id"],
            lease_token=stale["lease_token"],
        )
        is False
    )
    assert (
        mark_artifact_delivery(
            db,
            delivery_id,
            status="failed",
            attempt_id=stale["attempt_id"],
            lease_token=stale["lease_token"],
            error="stale worker",
        )
        is False
    )
    assert mark_artifact_delivery(
        db,
        delivery_id,
        status="ticket_acquiring",
        attempt_id=current["attempt_id"],
        lease_token=current["lease_token"],
    )


def test_artifact_worker_caps_pre_ticket_retries_with_backoff(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge"
    )
    delivery_id = enqueue_artifact_delivery(
        db,
        artifact=artifact,
        binding_id=binding_id,
        channel_id="C_FAMILY",
        thread_id=None,
        idempotency_key="artifact-bounded-retry",
    )
    client = _ArtifactSlackClient(ticket_error=TimeoutError("pre-ticket timeout"))
    for attempt in range(4):
        with connect(db) as conn:
            conn.execute(
                "UPDATE artifact_file_outbox SET available_at='2000-01-01T00:00:00+00:00' WHERE id=?",
                (delivery_id,),
            )
        result = flush_slack_artifact_outbox(
            client,
            db,
            tmp_path / "output",
            uploader=lambda _url, _data: None,
        )
        if attempt < 3:
            assert result == [{"delivery_id": delivery_id, "status": "failed"}]
            with connect(db) as conn:
                retry = conn.execute(
                    "SELECT available_at,updated_at FROM artifact_file_outbox WHERE id=?",
                    (delivery_id,),
                ).fetchone()
            assert retry["available_at"] > retry["updated_at"]
            assert claim_artifact_delivery(db, delivery_id) is None
        else:
            assert result == []
    assert len(client.tickets) == 3
    with connect(db) as conn:
        row = conn.execute(
            "SELECT status,attempts,available_at FROM artifact_file_outbox WHERE id=?",
            (delivery_id,),
        ).fetchone()
    assert row["status"] == "expired" and row["attempts"] == 3


def test_artifact_upload_timeout_after_bytes_is_terminal_unknown(tmp_path: Path):
    db, _setup, binding_id = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge"
    )
    delivery_id = enqueue_artifact_delivery(
        db,
        artifact=artifact,
        binding_id=binding_id,
        channel_id="C_FAMILY",
        thread_id=None,
        idempotency_key="artifact-upload-timeout",
    )
    client = _ArtifactSlackClient()
    uploads = 0

    def ambiguous_upload(_url, _data):
        nonlocal uploads
        uploads += 1
        raise TimeoutError("bytes may have reached Slack")

    assert flush_slack_artifact_outbox(
        client,
        db,
        tmp_path / "output",
        uploader=ambiguous_upload,
    ) == [{"delivery_id": delivery_id, "status": "unknown"}]
    assert flush_slack_artifact_outbox(
        client,
        db,
        tmp_path / "output",
        uploader=ambiguous_upload,
    ) == []
    assert uploads == 1 and len(client.tickets) == 1
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM artifact_file_outbox WHERE id=?", (delivery_id,)
        ).fetchone()["status"] == "unknown"


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
        thread_ref = sha256(f"{kwargs['team_id']}:{kwargs['channel_id']}:{kwargs['thread_id']}".encode()).hexdigest()[
            :20
        ]
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
        thread_ref = sha256(f"{kwargs['team_id']}:{kwargs['channel_id']}:{kwargs['thread_id']}".encode()).hexdigest()[
            :20
        ]
        session = store.get_or_create(
            origin="slack",
            transport="slack",
            binding_id=kwargs["binding_id"],
            conversation_ref=conversation_ref,
            thread_ref=thread_ref,
            child_id="kid-a",
        )
        return {
            "status": "completed",
            "message": "I changed the diagram without repeating the activity.",
            "session_id": session["id"],
        }


def test_public_slack_thread_routes_natural_parent_tuning_to_agent(tmp_path: Path):
    db = tmp_path / "private" / "data" / "curiosity.db"
    setup = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "kid-a", "Kid A", grade="1")
    service = _ConversationService(db)
    transport = SlackTransport(db, tmp_path / "output", service=service)
    code = create_pairing_code(db, setup["owner_id"])["pairing_code"]
    base = dict(team_id="T_FAMILY", user_id="U_PARENT", channel_id="C_FAMILY", thread_id="100.1")
    assert transport.handle(InboundMessage(external_event_id="pair", text=f"pair {code}", **base)).status == "paired"
    assert (
        transport.handle(
            InboundMessage(external_event_id="ask", text="ask kid-a: How do airplanes fly?", **base)
        ).status
        == "completed"
    )
    follow = transport.handle(
        InboundMessage(
            external_event_id="follow",
            text="This image isn't helpful; show wing variations instead.",
            **base,
        )
    )
    assert follow.status == "completed"
    assert service.chat_messages == [
        "How do airplanes fly?",
        "This image isn't helpful; show wing variations instead.",
    ]


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
    monkeypatch.setattr(
        "curiosity_engine.setup_agent.shutil.which", lambda name: "/usr/bin/codex" if name == "codex" else None
    )
    state = prepare_agent_setup()
    assert state["agent"] == "codex"
    assert (home / "workspace" / "AGENTS.md").stat().st_mode & 0o077 == 0
    handoff = (home / "workspace" / "SETUP_HANDOFF.md").read_text(encoding="utf-8")
    assert "Slack" in handoff and "model" in handoff and "printable" in handoff
    units = unit_definitions("/opt/curiosity/bin/curiosity")
    combined = "\n".join(units.values())
    assert 'worker" "--forever' in combined
    assert 'serve" "--db' in combined and '"--host" "127.0.0.1"' in combined
    assert "xoxb-" not in combined and "OPENAI_API_KEY" not in combined
