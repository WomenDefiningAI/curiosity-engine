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
from curiosity_engine.db import connect, utcnow
from curiosity_engine.director import AutonomousDirector
from curiosity_engine.graph import add_child
from curiosity_engine.interaction import (
    TransportConflict,
    active_binding,
    create_pairing_code,
    list_inbox,
    onboarding_status,
    revoke_binding,
    setup_household,
)
from curiosity_engine.onboarding import doctor
from curiosity_engine.transports.contracts import InboundMessage
from curiosity_engine.transports.slack import (
    SlackTransport,
    _make_slack_event_receiver,
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
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                   VALUES(?,'child_question',?,?,?,'{}',?,'completed')""",
                (kwargs["event_id"], kwargs["child_id"], kwargs["text"], kwargs["source"], utcnow()),
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


class FakeSlackClient:
    def __init__(self):
        self.messages: list[dict[str, Any]] = []

    def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.messages.append(kwargs)
        return {"ts": f"100.{len(self.messages)}"}


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
            "event_ts": "100.1",
        },
        client,
    )

    assert active_binding(db, incoming("lookup", "help", channel="C_FAMILY")) is not None
    assert client.messages
    assert client.messages[0]["channel"] == "C_FAMILY"
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

    assigned = transport.handle(incoming("EvAssign", f"assign {saved.inbox_id} kid-a"))
    assert assigned.status == "completed"
    assert fake.calls[0]["text"] == "We found a feather on our walk"
    assert fake.calls[0]["include_private_excerpts"] is False
    assert list_inbox(db, status="assigned")[0]["child_id"] == "kid-a"

    immediate = transport.handle(incoming("EvAsk", "ask kid-a: Why are feathers light?"))
    assert immediate.status == "completed"
    assert fake.calls[-1]["source"] == "slack"
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


def test_slack_uses_household_private_resource_opt_in(tmp_path: Path):
    db, transport, fake, client, _code = paired_transport(tmp_path, resource_context_mode="selected_excerpts")
    result = transport.handle(incoming("EvPrivateAsk", "ask kid-a: Why does the Moon look close?"))
    assert result.status == "completed"
    assert fake.calls[-1]["include_private_excerpts"] is True

    privacy = transport.handle(incoming("EvPrivacy", "privacy"))
    assert privacy.status == "completed"
    flush_slack_outbox(client, db)
    assert any("opted in to selected excerpts" in message["text"] for message in client.messages)


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


def test_weekly_schedule_requires_parent_opt_in(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    setup_household(db, owner_name="Parent", timezone="Etc/UTC", proactive_enabled=False)
    add_child(db, "kid-a", "Kid A")
    director = AutonomousDirector(db)
    director.ensure_weekly_schedule("kid-a")
    with connect(db) as conn:
        assert conn.execute("SELECT enabled FROM schedules WHERE id='weekly:kid-a'").fetchone()[0] == 0
    setup_household(db, owner_name="Parent", timezone="Etc/UTC", proactive_enabled=True)
    with connect(db) as conn:
        assert conn.execute("SELECT enabled FROM schedules WHERE id='weekly:kid-a'").fetchone()[0] == 1


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
    for scope in ("app_mentions:read", "chat:write", "im:history"):
        assert f"- {scope}" in manifest
    for forbidden in ("channels:history", "groups:history", "users:read", "files:read"):
        assert forbidden not in manifest


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
