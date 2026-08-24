from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from curiosity_engine.artifacts import ArtifactService, render_pdf
from curiosity_engine.context_builder import build_context
from curiosity_engine.contracts import CriticResult, Event, GenericOutput, PullThreadOutput
from curiosity_engine.db import SCHEMA_VERSION, connect, init_db
from curiosity_engine.graph import add_child, add_school_signal, capture_question
from curiosity_engine.openai_backend import OpenAIBackend
from curiosity_engine.printer import approve_artifact, print_artifact
from curiosity_engine.providers import AnthropicBackend, OpenRouterBackend
from curiosity_engine.reasoning import ReasoningEngine, ReasoningPolicy, StubBackend
from curiosity_engine.repository import IdempotencyConflict
from curiosity_engine.resources import index_collection, resource_inventory, search_resources
from curiosity_engine.runtime import CuriosityHarness
from curiosity_engine.service import CuriosityService
from curiosity_engine.web import create_app


def artifact_spec(tier: str = "A") -> dict:
    spec = {
        "artifact_type": "wonder_page",
        "title": "Why does the Moon seem to follow us?",
        "trust_tier": tier,
        "target_age": 6,
        "prompt": "Which seems to move more: a nearby tree or the Moon?",
        "body": ["Predict first.", "Compare near and far.", "Describe what changed."],
        "assets": [],
    }
    if tier == "C":
        spec["fact_model"] = {
            "facts": [{"claim": "The Moon is distant.", "certainty": "established", "source": "test"}]
        }
    return spec


def test_profile_update_preserves_graph(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Demo Child", 2020, "1st")
    capture_question(db, "child-a", "Why?", ["moon"])
    add_child(db, "child-a", "Demo Child Updated", 2020, "2nd")
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 2
        assert conn.execute("SELECT name FROM children WHERE id='child-a'").fetchone()[0] == "Demo Child Updated"


def test_event_idempotency_and_payload_conflict(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Demo Child")
    harness = CuriosityHarness(db)
    event = Event(id="evt_repeat", type="child_question", child_id="child-a", text="Why does ice float?")
    first = harness.dispatch(event)
    second = harness.dispatch(event)
    assert first.run_id == second.run_id
    assert second.duplicate
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1
    with pytest.raises(IdempotencyConflict):
        harness.dispatch(Event(id="evt_repeat", type="child_question", child_id="child-a", text="Different payload"))


class RejectBackend(StubBackend):
    def complete(self, *, role, system, payload, response_model):
        if response_model is CriticResult:
            return {"verdict": "reject", "concerns": ["unsafe"], "required_changes": ["stop"]}
        return super().complete(role=role, system=system, payload=payload, response_model=response_model)


def test_critic_rejection_never_completes(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Demo Child")
    result = CuriosityHarness(db, ReasoningEngine(RejectBackend())).dispatch(
        Event(type="child_question", child_id="child-a", text="Why?")
    )
    assert result.status == "rejected"
    with connect(db) as conn:
        assert conn.execute("SELECT status FROM events").fetchone()[0] == "rejected"
        assert conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0] == 0


def test_expired_school_signals_do_not_enter_context(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Demo Child")
    add_school_signal(db, "child-a", "unit", "expired", expires_at="2000-01-01T00:00:00+00:00")
    context = build_context(db, "child-a", {"type": "test", "text": "expired", "metadata": {}}, depth=2)
    assert context["school_signals"] == []


def test_private_resource_index_and_excerpt_gate(tmp_path: Path):
    repo = tmp_path / "repo"
    collection = repo / "private" / "resources" / "provider" / "collection"
    unit = collection / "moon"
    unit.mkdir(parents=True)
    (repo / ".gitignore").write_text("private/\n", encoding="utf-8")
    (unit / "guide.pdf").write_bytes(b"%PDF-test-private")
    (unit / "guide.txt").write_text(
        "Moon perspective activity. Compare a near tree and the far Moon.\fSecond page.", encoding="utf-8"
    )
    catalog = {
        "id": "private-collection",
        "title": "Private Collection",
        "provider": "Provider",
        "source_url": "https://example.invalid",
        "access": {"scope": "family_private", "redistribution_allowed": False},
        "units": [
            {
                "id": "moon",
                "title": "Moon",
                "summary": "Moon and perspective.",
                "topic_tags": ["Moon", "perspective"],
                "documents": ["guide"],
            }
        ],
    }
    catalog_path = collection / "catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    db = repo / "private" / "data.sqlite"
    report = index_collection(db, catalog_path, repository_root=repo)
    assert (report.units, report.documents) == (1, 1)
    assert resource_inventory(db)["units"][0]["title"] == "Moon"
    metadata = search_resources(db, "Moon perspective")
    excerpts = search_resources(db, "Moon perspective", include_excerpts=True)
    assert metadata and "excerpt" not in metadata[0]
    assert excerpts and "near tree" in excerpts[0]["excerpt"]


def test_pdf_approval_is_bound_to_exact_bytes(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Demo Child")
    artifact = ArtifactService(db, tmp_path / "output").create(child_id="child-a", spec=artifact_spec())
    approval = approve_artifact(db, artifact["artifact_id"])
    dry_run = print_artifact(db, artifact["artifact_id"], approval["approval_id"])
    assert dry_run["status"] == "dry-run"
    Path(artifact["pdf_path"]).write_bytes(Path(artifact["pdf_path"]).read_bytes() + b"tamper")
    with pytest.raises(PermissionError):
        print_artifact(db, artifact["artifact_id"], approval["approval_id"])


def test_tier_c_generation_fails_closed(tmp_path: Path):
    with pytest.raises(ValueError, match="Tier C"):
        render_pdf(artifact_spec("C"), tmp_path / "tier-c.pdf")


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return type(
            "Response", (), {"output_text": json.dumps({"summary": "ok", "graph_updates": [], "actions": []})}
        )()


class FakeOpenAI:
    def __init__(self):
        self.responses = FakeResponses()


def test_openai_adapter_disables_storage_and_locally_validates():
    client = FakeOpenAI()
    backend = OpenAIBackend(client=client, model="test-model")
    result = backend.complete(
        role="reasoning",
        system="system",
        payload={"question": "test", "image_data_urls": ["data:image/png;base64,secret-image"]},
        response_model=GenericOutput,
    )
    request = client.responses.request
    assert result["summary"] == "ok"
    assert request["store"] is False
    assert request["text"]["format"]["type"] == "json_schema"
    assert "secret-image" not in request["input"][0]["content"][0]["text"]
    assert request["input"][0]["content"][1]["type"] == "input_image"


def test_openai_adapter_enables_bounded_web_search_and_keeps_sources():
    source_url = "https://example.org/robot-record"

    class SearchResponses:
        def __init__(self):
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            payload = {
                "hook": "Different robots win different size contests.",
                "show": "Compare their size with a person.",
                "ask": "Which measurement should decide the winner?",
                "nugget": "Height and weight measure different kinds of size.",
                "next_possible_concepts": ["measurement"],
                "physical_extension": None,
                "graph_updates": [],
                "actions": [],
                "resource_refs": [],
            }
            return type(
                "Response",
                (),
                {
                    "output_text": json.dumps(payload),
                    "output": [{"type": "web_search_call", "action": {"sources": [{"url": source_url}]}}],
                },
            )()

    client = type("Client", (), {"responses": SearchResponses()})()
    backend = OpenAIBackend(client=client, model="test-model")
    result = backend.complete(
        role="reasoning",
        system="system",
        payload={"policy": {"allowed_tools": ["web_search"]}},
        response_model=PullThreadOutput,
    )

    assert client.responses.request["tools"] == [{"type": "web_search"}]
    assert client.responses.request["max_tool_calls"] == 2
    assert result["resource_refs"] == [source_url]


def test_openai_candidate_with_overlong_hook_uses_reasoning_repair_round():
    valid = {
        "hook": "Different robots win different size contests.",
        "show": "Compare each robot with a person or a building.",
        "ask": "Should biggest mean tallest or heaviest?",
        "nugget": "Engineers must choose a measurement before comparing size.",
        "next_possible_concepts": ["measurement", "scale"],
        "physical_extension": None,
        "graph_updates": [],
        "actions": [],
        "resource_refs": [],
    }

    class RepairResponses:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            del kwargs
            self.calls += 1
            payload = {**valid, "hook": "x" * 501} if self.calls == 1 else valid
            return type("Response", (), {"output_text": json.dumps(payload), "output": []})()

    responses = RepairResponses()
    backend = OpenAIBackend(client=type("Client", (), {"responses": responses})(), model="test-model")
    result = ReasoningEngine(backend).run(
        policy=ReasoningPolicy("pull_thread", 2, critic_roles=(), max_revision_rounds=1),
        context={"child": {"grade": "1st"}},
        event={"type": "child_question", "text": "What are the biggest robots?"},
    )

    assert responses.calls == 2
    assert result.revision_rounds == 1
    assert result.output["hook"] == valid["hook"]


def test_contract_repair_does_not_consume_semantic_revision_budget():
    class ContractThenContentBackend(StubBackend):
        name = "repair-test"
        model = "repair-test-v1"

        def __init__(self):
            self.generator_calls = 0
            self.critic_calls = 0

        def complete(self, *, role, system, payload, response_model):
            del role, system
            if response_model is CriticResult:
                self.critic_calls += 1
                if payload["candidate"]["nugget"] == "Draft explanation.":
                    return {
                        "verdict": "revise",
                        "concerns": ["The explanation needs a robot/category distinction."],
                        "required_changes": ["Clarify the category."],
                    }
                return {"verdict": "pass", "concerns": [], "required_changes": []}

            self.generator_calls += 1
            if self.generator_calls == 1:
                hook = "x" * 501
                nugget = "Draft explanation."
            elif "validation_errors" in payload:
                hook = "Different robots win different size contests."
                nugget = "Draft explanation."
            else:
                hook = "Different robots win different size contests."
                nugget = "A robot senses and acts; a very large operated machine may not be a robot."
            return {
                "hook": hook,
                "show": "Compare each machine with a person.",
                "ask": "Should biggest mean tallest or heaviest?",
                "nugget": nugget,
                "next_possible_concepts": ["measurement"],
                "physical_extension": None,
                "graph_updates": [],
                "actions": [],
                "resource_refs": [],
            }

    backend = ContractThenContentBackend()
    result = ReasoningEngine(backend).run(
        policy=ReasoningPolicy(
            "pull_thread",
            2,
            critic_roles=("critic_factual",),
            max_revision_rounds=1,
        ),
        context={"child": {"grade": "1st"}},
        event={"type": "child_question", "text": "What are the biggest robots?"},
    )

    assert backend.generator_calls == 3
    assert backend.critic_calls == 2
    assert result.revision_rounds == 2
    assert "operated machine may not be a robot" in result.output["nugget"]


def test_anthropic_adapter_uses_native_messages_and_validates_schema():
    class Messages:
        def __init__(self):
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            block = type(
                "Block",
                (),
                {"model_dump": lambda self: {"type": "text", "text": json.dumps({"summary": "ok", "graph_updates": [], "actions": []})}},
            )()
            return type("Response", (), {"content": [block]})()

    client = type("Client", (), {"messages": Messages()})()
    backend = AnthropicBackend(client=client, model="test-model")
    result = backend.complete(
        role="reasoning",
        system="system",
        payload={"question": "test", "image_data_urls": ["data:image/png;base64,c2FmZQ=="]},
        response_model=GenericOutput,
    )
    request = client.messages.request
    assert result["summary"] == "ok"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["messages"][0]["content"][1]["type"] == "image"


def test_openrouter_adapter_uses_chat_completions_and_privacy_routing():
    class Completions:
        def __init__(self):
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            message = type("Message", (), {"content": json.dumps({"summary": "ok", "graph_updates": [], "actions": []})})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    completions = Completions()
    client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    backend = OpenRouterBackend(client=client, model="provider/model")
    result = backend.complete(
        role="reasoning",
        system="system",
        payload={"question": "test", "policy": {"allowed_tools": ["web_search"]}},
        response_model=GenericOutput,
    )
    request = completions.request
    provider = request["extra_body"]["provider"]
    assert result["summary"] == "ok"
    assert request["response_format"]["type"] == "json_schema"
    assert request["tools"] == [
        {"type": "openrouter:web_search", "parameters": {"max_results": 5}}
    ]
    assert provider == {
        "require_parameters": True,
        "allow_fallbacks": False,
        "data_collection": "deny",
        "zdr": True,
    }


def test_first_grade_biggest_robot_demo_stays_on_question(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Demo Child", 2020, "1st")
    result = CuriosityHarness(db).dispatch(
        Event(type="child_question", child_id="child-a", text="What are the biggest robots in the world?")
    )
    rendered = json.dumps(result.output).casefold()
    assert result.status == "completed"
    assert all(word in rendered for word in ("tallest", "heaviest", "strongest"))
    assert "scoreboard" in rendered
    assert "actuator" not in rendered


def test_web_onboarding_question_and_csrf(tmp_path: Path):
    app = create_app(tmp_path / "db.sqlite", tmp_path / "output")
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/", headers={"host": "rebinding.invalid"}).status_code == 403
    assert client.post("/children", data={"child_id": "child-a", "name": "Demo Child"}).status_code == 403
    token = app.state.csrf
    response = client.post(
        "/children",
        data={"csrf": token, "child_id": "child-a", "name": "Demo Child", "birth_year": "2020", "grade": "1st"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    response = client.post(
        "/ask",
        data={"csrf": token, "child_id": "child-a", "question": "Why does the Moon follow us?"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    page = client.get("/?child=child-a")
    assert "Nearby objects slide" in page.text
    assert page.headers["cache-control"] == "no-store"
    with connect(tmp_path / "db.sqlite") as conn:
        event_id = conn.execute("SELECT id FROM events").fetchone()[0]
    response = client.post(
        f"/responses/{event_id}/artifact",
        data={"csrf": token, "child_id": "child-a"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    client.post(
        f"/responses/{event_id}/artifact",
        data={"csrf": token, "child_id": "child-a"},
        follow_redirects=False,
    )
    with connect(tmp_path / "db.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1


def test_migration_backup_and_schema_version(tmp_path: Path):
    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE children(id TEXT PRIMARY KEY,name TEXT NOT NULL,birth_year INTEGER,grade TEXT,created_at TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO children VALUES('child-a','Demo Child',2020,'1st','now')")
    conn.commit()
    conn.close()
    backup = init_db(db)
    assert backup and backup.exists()
    with connect(db) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert conn.execute("SELECT name FROM children WHERE id='child-a'").fetchone()[0] == "Demo Child"


def test_context_driven_director_stays_disabled_until_episode_policy_is_ready(tmp_path: Path):
    service = CuriosityService(tmp_path / "db.sqlite", tmp_path / "output")
    service.add_child("child-a", "Demo Child", 2020, "1st")
    service.ask(child_id="child-a", text="Why do birds migrate?")
    reflection = service.reflect("child-a")
    assert reflection["choice"]["kind"] == "do_nothing"
    assert reflection["choice"]["payload"]["policy"] == "context_proactivity_disabled_v1"
    with connect(tmp_path / "db.sqlite") as conn:
        assert conn.execute("SELECT status FROM opportunities").fetchone()[0] == "no_action"


def test_feedback_is_recorded_as_observation_not_trait(tmp_path: Path):
    service = CuriosityService(tmp_path / "db.sqlite", tmp_path / "output")
    service.add_child("child-a", "Demo Child", 2020, "1st")
    artifact = service.create_artifact("child-a", artifact_spec())
    feedback_id = service.feedback(
        {
            "child_id": "child-a",
            "artifact_id": artifact["artifact_id"],
            "experience_id": artifact["experience_id"],
            "outcome": "engaged",
            "note": "Asked to try it again.",
        }
    )
    assert feedback_id > 0
    with connect(tmp_path / "db.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM observations WHERE kind='feedback'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0
