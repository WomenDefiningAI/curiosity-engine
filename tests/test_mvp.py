from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from curiosity_engine.artifacts import render_pdf
from curiosity_engine.context_builder import build_context
from curiosity_engine.contracts import CriticResult, Event, GenericOutput, GraphMutation, PullThreadOutput
from curiosity_engine.db import SCHEMA_VERSION, connect, init_db
from curiosity_engine.graph import add_child, add_school_signal, capture_question
from curiosity_engine.learning_artifacts import LearningArtifactService
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


def test_init_db_lock_order_does_not_deadlock_governed_artifact_and_visual_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The visual init path must wait outside `_INIT_LOCK` for artifact work."""

    from curiosity_engine import db as db_module

    database = tmp_path / "db.sqlite"
    init_db(database)
    db_module._INITIALIZED_DATABASES.clear()
    lifecycle = threading.RLock()
    artifact_has_lifecycle = threading.Event()
    visual_is_waiting = threading.Event()
    failures: list[BaseException] = []

    @contextmanager
    def controlled_lifecycle(*_args: Any, **_kwargs: Any):
        if threading.current_thread().name == "visual-init":
            visual_is_waiting.set()
        if not lifecycle.acquire(timeout=1):
            raise TimeoutError("lifecycle lock acquisition timed out")
        try:
            yield
        finally:
            lifecycle.release()

    monkeypatch.setattr(db_module, "managed_data_lifecycle", controlled_lifecycle)

    def artifact_path() -> None:
        try:
            with controlled_lifecycle(database):
                artifact_has_lifecycle.set()
                if not visual_is_waiting.wait(timeout=1):
                    raise TimeoutError("visual initializer did not reach the lifecycle boundary")
                db_module.init_db(database)
        except BaseException as exc:
            failures.append(exc)

    def visual_path() -> None:
        try:
            if not artifact_has_lifecycle.wait(timeout=1):
                raise TimeoutError("artifact did not reach the lifecycle boundary")
            db_module.init_db(database)
        except BaseException as exc:
            failures.append(exc)

    artifact_thread = threading.Thread(target=artifact_path, name="artifact-generation")
    visual_thread = threading.Thread(target=visual_path, name="visual-init")
    artifact_thread.start()
    visual_thread.start()
    artifact_thread.join(timeout=3)
    visual_thread.join(timeout=3)

    assert not artifact_thread.is_alive()
    assert not visual_thread.is_alive()
    assert failures == []


def test_current_database_initialization_is_cached_within_the_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hot request paths must not repeat lifecycle recovery and schema work."""

    from curiosity_engine import db as db_module

    database = tmp_path / "db.sqlite"
    init_db(database)
    lifecycle_entries = 0

    @contextmanager
    def counted_lifecycle(*_args: Any, **_kwargs: Any):
        nonlocal lifecycle_entries
        lifecycle_entries += 1
        yield

    monkeypatch.setattr(db_module, "managed_data_lifecycle", counted_lifecycle)

    for _ in range(10):
        assert db_module.init_db(database) is None

    assert lifecycle_entries == 0


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
    reasoning = result.output["_reasoning"]
    assert len(reasoning["critic_rounds"]) == 3
    assert reasoning["recovery_strategy"] == "rebuild_from_scratch"
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
    CuriosityHarness(db, ReasoningEngine(StubBackend())).dispatch(
        Event(
            id="evt-print-approval", type="child_question", child_id="child-a", text="Why does the Moon look far away?"
        )
    )
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-print-approval",
        artifact_type="worksheet",
    )
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
            "Response", (), {"output_text": json.dumps({"summary": "ok", "observation_candidates": [], "actions": []})}
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
                "observation_candidates": [],
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
        "observation_candidates": [],
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


def test_graph_mutation_requires_kind_specific_fields():
    with pytest.raises(ValidationError):
        GraphMutation.model_validate({"kind": "set_knowledge_state"})


class LegacyGraphUpdateBackend(StubBackend):
    def complete(self, *, role, system, payload, response_model):
        result = super().complete(
            role=role,
            system=system,
            payload=payload,
            response_model=response_model,
        )
        if response_model is PullThreadOutput:
            result["graph_updates"] = [{"kind": "set_knowledge_state"}]
        return result


def test_invalid_optional_graph_shape_does_not_drop_response():
    result = ReasoningEngine(LegacyGraphUpdateBackend()).run(
        policy=ReasoningPolicy("pull_thread", 2),
        context={"child": {"grade": "1st"}},
        event={"type": "child_question", "text": "How does electricity work?"},
    )

    assert result.output["hook"]
    assert "graph_updates" not in result.output
    assert result.output["observation_candidates"] == []


class HiddenLearnerStateBackend(StubBackend):
    def complete(self, *, role, system, payload, response_model):
        result = super().complete(
            role=role,
            system=system,
            payload=payload,
            response_model=response_model,
        )
        if response_model is PullThreadOutput:
            result["graph_updates"] = [
                {
                    "kind": "upsert_node",
                    "node_kind": "concept",
                    "label": "electricity",
                    "state": {"knowledge_state": "demonstrated"},
                }
            ]
        return result


def test_hidden_learner_state_does_not_create_graph_effect(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    init_db(db)
    add_child(db, "child-a", "Demo Child", 2020, "1st")
    harness = CuriosityHarness(
        db,
        reasoning_engine=ReasoningEngine(HiddenLearnerStateBackend()),
    )

    result = harness.dispatch(Event(type="child_question", child_id="child-a", text="How does electricity work?"))

    assert result.status == "completed"
    assert result.output["hook"]
    with connect(db) as conn:
        hidden = conn.execute(
            "SELECT COUNT(*) FROM graph_effects WHERE mutation_json LIKE '%knowledge_state%'"
        ).fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM graph_effects WHERE status='failed'").fetchone()[0]
        assert hidden == 0
        assert failed == 0


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
                "observation_candidates": [],
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


def test_final_resource_only_rejection_removes_links_instead_of_withholding_answer():
    class ResourceOnlyBackend(StubBackend):
        name = "resource-repair-test"
        model = "resource-repair-test-v1"

        def complete(self, *, role, system, payload, response_model):
            del role, system
            if response_model is CriticResult:
                if payload["candidate"].get("resource_refs"):
                    return {
                        "verdict": "reject",
                        "concerns": ["The listed resource links do not directly support the visible answer."],
                        "required_changes": ["Remove the unrelated resource references or omit sources."],
                    }
                return {"verdict": "pass", "concerns": [], "required_changes": []}
            return {
                "hook": "Nobody wrote down who said the very first version of this word.",
                "show": "Old writing gives us clues, not the name of the first speaker.",
                "ask": "What kind of old clue might survive?",
                "nugget": "Words can develop across groups of people over a long time.",
                "next_possible_concepts": ["language change"],
                "physical_extension": None,
                "observation_candidates": [],
                "actions": [],
                "resource_refs": ["https://example.test/unrelated"],
            }

    result = ReasoningEngine(ResourceOnlyBackend()).run(
        policy=ReasoningPolicy(
            "pull_thread",
            2,
            critic_roles=("critic_factual",),
            max_revision_rounds=1,
        ),
        context={"child": {"grade": "1st"}},
        event={"type": "child_question", "text": "Who decided the first word?"},
    )

    assert result.output["hook"].startswith("Nobody wrote down")
    assert result.output["resource_refs"] == []
    assert result.recovery_strategy == "remove_unrelated_sources"


def test_final_semantic_revision_rebuilds_without_anchoring_on_rejected_draft():
    class RebuildBackend(StubBackend):
        name = "rebuild-test"
        model = "rebuild-test-v1"

        def __init__(self):
            self.generator_calls = 0
            self.saw_required_changes = False

        def complete(self, *, role, system, payload, response_model):
            del role, system
            if response_model is CriticResult:
                extension = payload["candidate"].get("physical_extension")
                if extension:
                    return {
                        "verdict": "revise",
                        "concerns": ["The optional kit requires a purchase."],
                        "required_changes": ["Remove the purchased kit and use ordinary materials."],
                    }
                return {"verdict": "pass", "concerns": [], "required_changes": []}

            self.generator_calls += 1
            if payload.get("rebuild_from_scratch"):
                assert "candidate" not in payload
                self.saw_required_changes = payload["required_changes"] == [
                    "Remove the purchased kit and use ordinary materials."
                ]
                return {
                    **StubBackend._pull_thread("Can we build a robot?"),
                    "physical_extension": None,
                    "visual": None,
                }
            return {
                **StubBackend._pull_thread("Can we build a robot?"),
                "physical_extension": {
                    "title": "Use an optional robot kit",
                    "instructions": ["Ask an adult to assemble the kit."],
                    "materials": ["purchased robot kit"],
                    "parent_effort": "medium",
                },
            }

    backend = RebuildBackend()
    result = ReasoningEngine(backend).run(
        policy=ReasoningPolicy(
            "pull_thread",
            2,
            critic_roles=("critic_parent_effort",),
            max_revision_rounds=2,
            final_recovery="rebuild_from_scratch",
        ),
        context={"child": {"grade": "1st"}, "family_lens": {"materials": ["paper"]}},
        event={"type": "child_question", "text": "Can we build a robot?"},
    )

    assert backend.generator_calls == 3
    assert backend.saw_required_changes is True
    assert result.output["physical_extension"] is None
    assert result.recovery_strategy == "rebuild_from_scratch"
    assert len(result.critique_rounds) == 3


def test_final_recovery_accumulates_constraints_from_every_critic_round():
    class ChangingConcernBackend(StubBackend):
        name = "changing-concern-test"
        model = "changing-concern-v1"

        def __init__(self):
            self.generator_calls = 0
            self.final_constraints: list[str] = []

        def complete(self, *, role, system, payload, response_model):
            del role, system
            if response_model is CriticResult:
                title = str((payload["candidate"].get("physical_extension") or {}).get("title") or "")
                if "kit" in title.casefold():
                    return {
                        "verdict": "revise",
                        "concerns": ["The activity requires a purchase."],
                        "required_changes": ["Remove the purchased kit."],
                    }
                if "sharp" in title.casefold():
                    return {
                        "verdict": "revise",
                        "concerns": ["The activity uses a sharp tool."],
                        "required_changes": ["Remove the sharp tool."],
                    }
                return {"verdict": "pass", "concerns": [], "required_changes": []}

            self.generator_calls += 1
            if payload.get("rebuild_from_scratch"):
                self.final_constraints = payload["required_changes"]
                return {
                    **StubBackend._pull_thread("Can we build a robot?"),
                    "physical_extension": None,
                    "visual": None,
                }
            if "candidate" in payload:
                return {
                    **StubBackend._pull_thread("Can we build a robot?"),
                    "physical_extension": {
                        "title": "Use a sharp tool",
                        "instructions": ["Ask an adult to help."],
                        "materials": ["tool"],
                        "parent_effort": "medium",
                    },
                }
            return {
                **StubBackend._pull_thread("Can we build a robot?"),
                "physical_extension": {
                    "title": "Use a purchased kit",
                    "instructions": ["Open the kit."],
                    "materials": ["kit"],
                    "parent_effort": "medium",
                },
            }

    backend = ChangingConcernBackend()
    result = ReasoningEngine(backend).run(
        policy=ReasoningPolicy(
            "pull_thread",
            2,
            critic_roles=("critic_parent_effort",),
            max_revision_rounds=2,
            final_recovery="rebuild_from_scratch",
        ),
        context={"family_lens": {"materials": ["paper"]}},
        event={"type": "child_question", "text": "Can we build a robot?"},
    )

    assert backend.final_constraints == ["Remove the purchased kit.", "Remove the sharp tool."]
    assert result.output["physical_extension"] is None
    assert [round_items[0].verdict for round_items in result.critique_rounds] == [
        "revise",
        "revise",
        "pass",
    ]


def test_malformed_final_rebuild_gets_one_contract_repair_without_losing_constraints():
    class MalformedRebuildBackend(StubBackend):
        name = "malformed-rebuild-test"
        model = "malformed-rebuild-v1"

        def complete(self, *, role, system, payload, response_model):
            del role, system
            if response_model is CriticResult:
                if payload["candidate"].get("physical_extension"):
                    return {
                        "verdict": "revise",
                        "concerns": ["The activity requires a purchase."],
                        "required_changes": ["Remove the purchased kit."],
                    }
                return {"verdict": "pass", "concerns": [], "required_changes": []}
            if payload.get("rebuild_from_scratch") and "validation_errors" not in payload:
                return {"hook": "Malformed final rebuild"}
            if payload.get("rebuild_from_scratch") and "validation_errors" in payload:
                assert payload["required_changes"] == ["Remove the purchased kit."]
                return {
                    **StubBackend._pull_thread("Can we build a robot?"),
                    "physical_extension": None,
                    "visual": None,
                }
            return {
                **StubBackend._pull_thread("Can we build a robot?"),
                "physical_extension": {
                    "title": "Use a purchased kit",
                    "instructions": ["Open the kit."],
                    "materials": ["kit"],
                    "parent_effort": "medium",
                },
            }

    result = ReasoningEngine(MalformedRebuildBackend()).run(
        policy=ReasoningPolicy(
            "pull_thread",
            2,
            critic_roles=("critic_parent_effort",),
            max_revision_rounds=2,
            final_recovery="rebuild_from_scratch",
        ),
        context={"family_lens": {"materials": ["paper"]}},
        event={"type": "child_question", "text": "Can we build a robot?"},
    )

    assert result.output["physical_extension"] is None
    assert result.revision_rounds == 3


def test_other_workflows_do_not_rebuild_without_explicit_policy():
    class ScopedRecoveryBackend(StubBackend):
        def __init__(self):
            self.rebuild_flags: list[bool] = []

        def complete(self, *, role, system, payload, response_model):
            if response_model is CriticResult:
                return {
                    "verdict": "revise" if not self.rebuild_flags else "pass",
                    "concerns": ["revise once"] if not self.rebuild_flags else [],
                    "required_changes": ["revise once"] if not self.rebuild_flags else [],
                }
            if "critiques" in payload:
                self.rebuild_flags.append(bool(payload.get("rebuild_from_scratch")))
                return payload["candidate"]
            return super().complete(
                role=role,
                system=system,
                payload=payload,
                response_model=response_model,
            )

    backend = ScopedRecoveryBackend()
    ReasoningEngine(backend).run(
        policy=ReasoningPolicy(
            "pull_thread",
            2,
            critic_roles=("critic_context",),
            max_revision_rounds=1,
        ),
        context={},
        event={"type": "child_question", "text": "Why?"},
    )
    assert backend.rebuild_flags == [False]


def test_anthropic_adapter_uses_native_messages_and_validates_schema():
    class Messages:
        def __init__(self):
            self.request = None

        def create(self, **kwargs):
            self.request = kwargs
            block = type(
                "Block",
                (),
                {
                    "model_dump": lambda self: {
                        "type": "text",
                        "text": json.dumps({"summary": "ok", "observation_candidates": [], "actions": []}),
                    }
                },
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
            message = type(
                "Message", (), {"content": json.dumps({"summary": "ok", "observation_candidates": [], "actions": []})}
            )()
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
    assert request["tools"] == [{"type": "openrouter:web_search", "parameters": {"max_results": 5}}]
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
    assert response.headers["location"] == "/?child=child-a"
    response = client.post(
        "/ask",
        data={"csrf": token, "child_id": "child-a", "question": "Why does the Moon follow us?"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/?child=child-a#responses"
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
        artifact_id = conn.execute("SELECT id FROM artifacts").fetchone()[0]
    response = client.post(
        f"/artifacts/{artifact_id}/approve",
        data={"csrf": token, "child_id": "//outside.example"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/#artifacts"


def test_local_eval_lab_records_output_judgment_without_child_evidence(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    app = create_app(db, tmp_path / "output")
    client = TestClient(app)
    token = app.state.csrf
    client.post(
        "/children",
        data={"csrf": token, "child_id": "child-a", "name": "Demo Child", "grade": "1st"},
    )
    CuriosityService(db, tmp_path / "output").ask(
        child_id="child-a",
        text="How do long-neck dinosaurs reach leaves?",
        source="slack_parent_report",
    )
    with connect(db) as conn:
        event_id = conn.execute("SELECT id FROM events").fetchone()[0]
        observations_before = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]

    page = client.get("/evals")
    assert page.status_code == 200
    assert "How do long-neck dinosaurs reach leaves?" in page.text
    assert "No imagination art" in page.text
    assert "Activity proposed; no aid released" in page.text
    assert "Best response shape" in page.text
    assert "Best visual mix" in page.text
    saved = client.post(
        f"/evals/{event_id}",
        data={
            "csrf": token,
            "response_rating": "needs_work",
            "visual_rating": "missing_needed",
            "preferred_response_shape": "learning_thread",
            "preferred_visual_mix": "both",
            "note": "Make the leaves into usable targets.",
        },
        follow_redirects=False,
    )
    assert saved.status_code == 303
    assert saved.headers["location"] == "/evals"
    generated = client.post(
        "/evals/generate-activity-aids",
        data={"csrf": token},
    )
    assert generated.status_code == 200
    assert "Generated 1 activity aid(s)" in generated.text
    assert "Printable tool for doing the activity" in generated.text
    generated_again = client.post(
        "/evals/generate-activity-aids",
        data={"csrf": token},
    )
    assert "Generated 0 activity aid(s); 1 already existed" in generated_again.text
    with connect(db) as conn:
        evaluation = dict(conn.execute("SELECT * FROM output_evaluations").fetchone())
        artifact = dict(conn.execute("SELECT * FROM artifacts").fetchone())
        assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == observations_before
        assert conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0
    assert evaluation["event_id"] == event_id
    assert evaluation["response_rating"] == "needs_work"
    assert evaluation["visual_rating"] == "missing_needed"
    assert evaluation["preferred_response_shape"] == "learning_thread"
    assert evaluation["preferred_visual_mix"] == "both"
    assert evaluation["note"] == "Make the leaves into usable targets."
    artifact_spec = json.loads(artifact["spec_json"])
    assert artifact_spec["printable"]["kind"] == "reach_comparison"
    assert [piece["label"] for piece in artifact_spec["printable"]["pieces"]] == [
        "NEAR LEAF",
        "MIDDLE LEAF",
        "FAR LEAF",
        "SHORT NECK",
        "LONG NECK",
        "HELP USED?",
    ]
    assert "same start" in json.dumps(artifact_spec).casefold()
    assert "no climbing or stretching" in artifact_spec["safety"].casefold()


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
        stored = conn.execute(
            "SELECT pedagogy_plan_id,pedagogy_plan_hash FROM artifacts WHERE id=?",
            (artifact["artifact_id"],),
        ).fetchone()
        assert stored["pedagogy_plan_id"] and stored["pedagogy_plan_hash"]


def test_response_artifact_lookup_returns_the_latest_repair(tmp_path: Path):
    service = CuriosityService(tmp_path / "db.sqlite", tmp_path / "output")
    service.add_child("child-a", "Demo Child", 2020, "1st")
    response = service.ask(
        child_id="child-a",
        text="Can we make a creature build plan?",
        event_id="evt_artifact_repair",
    )
    assert response["status"] == "completed"
    first_spec = {**artifact_spec(), "title": "First Draft", "source_event_id": "evt_artifact_repair"}
    final_spec = {**artifact_spec(), "title": "Final Repair", "source_event_id": "evt_artifact_repair"}
    first = service.create_artifact("child-a", first_spec)
    final = service.create_artifact("child-a", final_spec)

    resolved = service.create_artifact_from_response("evt_artifact_repair")

    assert resolved["duplicate"] is True
    assert resolved["artifact_id"] == final["artifact_id"]
    assert resolved["artifact_id"] != first["artifact_id"]


def test_response_feedback_is_bound_to_the_completed_child_response(tmp_path: Path):
    service = CuriosityService(tmp_path / "db.sqlite", tmp_path / "output")
    service.add_child("child-a", "Demo Child", 2020, "1st")
    service.add_child("child-b", "Other Child", 2019, "2nd")
    response = service.ask(
        child_id="child-a",
        text="Why do birds have feathers?",
        event_id="evt_response_feedback",
    )
    assert response["status"] == "completed"
    feedback_id = service.feedback(
        {
            "child_id": "child-a",
            "event_id": "evt_response_feedback",
            "outcome": "helpful",
            "source": "slack_response:parent-test",
        }
    )
    assert feedback_id > 0
    with pytest.raises(ValueError, match="does not belong"):
        service.feedback(
            {
                "child_id": "child-b",
                "event_id": "evt_response_feedback",
                "outcome": "not_helpful",
                "source": "slack_response:parent-test",
            }
        )
    with connect(tmp_path / "db.sqlite") as conn:
        row = conn.execute("SELECT event_id,outcome FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        observation = conn.execute(
            "SELECT metadata_json FROM observations WHERE kind='feedback' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert tuple(row) == ("evt_response_feedback", "helpful")
    assert "evt_response_feedback" in observation["metadata_json"]
