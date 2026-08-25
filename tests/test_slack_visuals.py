from __future__ import annotations

from pathlib import Path
from typing import Any

from curiosity_engine.db import connect
from curiosity_engine.graph import add_child
from curiosity_engine.interaction import (
    create_pairing_code,
    enqueue_delivery,
    enqueue_visual_delivery,
    onboarding_status,
    ready_visual_deliveries,
    setup_household,
)
from curiosity_engine.transports.contracts import InboundMessage, OutboundMessage
from curiosity_engine.transports.slack import (
    SlackTransport,
    flush_slack_file_outbox,
    flush_slack_outbox,
)
from curiosity_engine.visuals import (
    create_synthetic_visual_job,
    enqueue_response_visual,
    process_visual_jobs,
)


class SlackApiError(Exception):
    """Named like Slack SDK's exception so the transport takes its terminal branch."""


class FakeSlackClient:
    def __init__(
        self,
        *,
        ticket_error: Exception | None = None,
        completion_error: Exception | None = None,
    ):
        self.messages: list[dict[str, Any]] = []
        self.tickets: list[dict[str, Any]] = []
        self.completions: list[dict[str, Any]] = []
        self.ticket_error = ticket_error
        self.completion_error = completion_error

    def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.messages.append(kwargs)
        return {"ts": f"100.{len(self.messages)}"}

    def files_getUploadURLExternal(self, **kwargs: Any) -> dict[str, str]:
        self.tickets.append(kwargs)
        if self.ticket_error:
            raise self.ticket_error
        return {"upload_url": "https://files.slack.test/upload/one", "file_id": "F_TEST"}

    def files_completeUploadExternal(self, **kwargs: Any) -> dict[str, Any]:
        self.completions.append(kwargs)
        if self.completion_error:
            raise self.completion_error
        return {"files": [{"id": "F_TEST"}]}


def setup_visual_delivery(tmp_path: Path, *, purpose: str = "response_visual"):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    household = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    add_child(db, "kid-a", "Kid A", grade="1")
    code = create_pairing_code(db, household["owner_id"])["pairing_code"]
    transport = SlackTransport(db, output, service=None)
    pair = InboundMessage(
        external_event_id="EvPairVisual",
        team_id="T_TEST",
        user_id="U_TEST",
        channel_id="C_TEST",
        text=f"pair {code}",
    )
    transport.handle(pair)
    client = FakeSlackClient()
    flush_slack_outbox(client, db)
    with connect(db) as conn:
        binding = conn.execute("SELECT id FROM transport_bindings").fetchone()["id"]
    event_id = "evt_visual_connection_test"
    visual_job = create_synthetic_visual_job(db, event_id)
    text_delivery = enqueue_delivery(
        db,
        binding,
        OutboundMessage(channel_id="C_TEST", text="Accessible answer text."),
        idempotency_key="slack:visual-test:text",
    )
    visual_delivery = enqueue_visual_delivery(
        db,
        visual_job_id=visual_job,
        binding_id=binding,
        depends_on_delivery_id=text_delivery,
        channel_id="C_TEST",
        thread_id=None,
        idempotency_key="slack:visual-test:file",
        purpose=purpose,
    )
    process_visual_jobs(db, output)
    return db, output, client, text_delivery, visual_delivery


def test_visual_waits_for_text_then_uploads_with_alt_text_and_caption(tmp_path: Path):
    db, output, client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    assert ready_visual_deliveries(db) == []
    flush_slack_outbox(client, db)
    uploaded: list[tuple[str, bytes]] = []
    result = flush_slack_file_outbox(
        client,
        db,
        output,
        uploader=lambda url, data: uploaded.append((url, data)),
    )
    assert result == [{"delivery_id": visual_id, "status": "sent"}]
    assert client.tickets[0]["alt_txt"].startswith("Three colorful cards")
    assert client.completions[0]["channel_id"] == "C_TEST"
    assert "fixed test card" in client.completions[0]["initial_comment"].casefold()
    assert uploaded[0][0].startswith("https://") and uploaded[0][1].startswith(b"\x89PNG")


def test_synthetic_visual_connection_records_real_delivery_checkpoint(tmp_path: Path):
    db, output, client, _text_id, _visual_id = setup_visual_delivery(
        tmp_path, purpose="visual_connection"
    )
    flush_slack_outbox(client, db)
    flush_slack_file_outbox(client, db, output, uploader=lambda _url, _data: None)
    assert onboarding_status(db)["checkpoints"]["visual_delivery_verified"]["status"] == "pass"


def test_ambiguous_completion_is_never_retried(tmp_path: Path):
    db, output, _client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    client = FakeSlackClient(completion_error=TimeoutError("network broke after completion"))
    flush_slack_outbox(client, db)
    result = flush_slack_file_outbox(client, db, output, uploader=lambda _url, _data: None)
    assert result == [{"delivery_id": visual_id, "status": "unknown"}]
    assert ready_visual_deliveries(db) == []


def test_tampered_asset_fails_closed_without_upload(tmp_path: Path):
    db, output, client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    flush_slack_outbox(client, db)
    with connect(db) as conn:
        path = Path(conn.execute("SELECT path FROM visual_assets").fetchone()["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    result = flush_slack_file_outbox(client, db, output, uploader=lambda _url, _data: None)
    assert result == [{"delivery_id": visual_id, "status": "expired"}]
    assert client.tickets == []


def test_asset_path_outside_private_visual_root_fails_closed(tmp_path: Path):
    db, output, client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    flush_slack_outbox(client, db)
    outside = tmp_path / "outside.png"
    with connect(db) as conn:
        original = Path(conn.execute("SELECT path FROM visual_assets").fetchone()["path"])
        outside.write_bytes(original.read_bytes())
        conn.execute("UPDATE visual_assets SET path=?", (str(outside),))
    result = flush_slack_file_outbox(client, db, output, uploader=lambda _url, _data: None)
    assert result == [{"delivery_id": visual_id, "status": "expired"}]
    assert client.tickets == []


def test_slack_scope_rejection_is_terminal_and_not_retried(tmp_path: Path):
    db, output, _client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    client = FakeSlackClient(ticket_error=SlackApiError("missing_scope"))
    flush_slack_outbox(client, db)
    result = flush_slack_file_outbox(client, db, output, uploader=lambda _url, _data: None)
    assert result == [{"delivery_id": visual_id, "status": "expired"}]
    assert ready_visual_deliveries(db) == []


def test_visual_preparation_failure_never_blocks_text_delivery(tmp_path: Path):
    db, output, client, text_id, _visual_id = setup_visual_delivery(tmp_path)
    with connect(db) as conn:
        binding = conn.execute("SELECT id FROM transport_bindings").fetchone()["id"]
    event_id = "evt_decorative_failure"
    from curiosity_engine.db import utcnow

    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
               VALUES(?,'child_question','synthetic visual failure','test','{}',?,'completed')""",
            (event_id, utcnow()),
        )
    visual_job = enqueue_response_visual(
        db,
        event_id=event_id,
        mode="decorative",
        visual={
            "kind": "decorative_illustration",
            "purpose": "imagine",
            "knowledge_role": "decorative",
            "title": "Synthetic failure",
            "pedagogical_value": "Exercises failure isolation.",
            "caption": "A synthetic decorative test.",
            "alt_text": "A synthetic decorative image test.",
            "subject": "an imaginary robot tending a moon garden",
        },
    )
    assert visual_job
    enqueue_visual_delivery(
        db,
        visual_job_id=visual_job,
        binding_id=binding,
        depends_on_delivery_id=text_id,
        channel_id="C_TEST",
        thread_id=None,
        idempotency_key="slack:decorative-failure:file",
    )

    class FailingImageBackend:
        name = "fake"
        model = "image-test"

        def generate(self, _prompt: str):
            raise RuntimeError("synthetic generation failure")

    assert process_visual_jobs(db, output, image_backend=FailingImageBackend())[-1]["status"] == "failed"
    delivered = flush_slack_outbox(client, db)
    assert {item["delivery_id"] for item in delivered} >= {text_id}
    assert client.messages[-1]["text"] == "Accessible answer text."
    with connect(db) as conn:
        failed = conn.execute(
            "SELECT status FROM slack_file_outbox WHERE visual_job_id=?", (visual_job,)
        ).fetchone()
    assert failed["status"] == "expired"


def test_slack_visual_command_uses_no_model_or_family_question(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    household = setup_household(db, owner_name="Parent", timezone="Etc/UTC")
    code = create_pairing_code(db, household["owner_id"])["pairing_code"]
    transport = SlackTransport(db, output)
    pair = InboundMessage(
        external_event_id="EvPair",
        team_id="T_TEST",
        user_id="U_TEST",
        channel_id="C_TEST",
        text=f"pair {code}",
    )
    transport.handle(pair)
    result = transport.handle(
        InboundMessage(
            external_event_id="EvVisualConnection",
            team_id="T_TEST",
            user_id="U_TEST",
            channel_id="C_TEST",
            text="visual connection",
        )
    )
    assert result.status == "completed"
    with connect(db) as conn:
        job = conn.execute("SELECT intent_json,method FROM visual_jobs").fetchone()
    assert job["method"] == "deterministic"
    assert "child" not in job["intent_json"].casefold()
