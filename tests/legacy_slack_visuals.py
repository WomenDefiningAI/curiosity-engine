from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import curiosity_engine.transports.slack as slack_module
import curiosity_engine.visuals as visuals_module
from curiosity_engine.db import connect, jdump, jload
from curiosity_engine.graph import add_child
from curiosity_engine.interaction import (
    claim_visual_delivery,
    consume_pairing_code,
    create_pairing_code,
    enqueue_delivery,
    enqueue_visual_delivery,
    mark_delivery,
    mark_visual_delivery,
    onboarding_status,
    ready_deliveries,
    ready_visual_deliveries,
    revoke_binding,
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


def setup_visual_delivery(tmp_path: Path, *, purpose: str = "response_visual", process: bool = True):
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
    if process:
        process_visual_jobs(db, output)
    return db, output, client, text_delivery, visual_delivery


def test_transient_visual_retry_keeps_then_queues_real_slack_outbox(tmp_path: Path, monkeypatch):
    db, output, _client, _text_delivery, visual_delivery = setup_visual_delivery(tmp_path, process=False)
    with connect(db) as conn:
        visual_job = conn.execute(
            "SELECT visual_job_id FROM slack_file_outbox WHERE id=?", (visual_delivery,)
        ).fetchone()["visual_job_id"]
    original = visuals_module.render_deterministic_visual
    calls = 0

    def flaky(intent, path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary renderer outage")
        return original(intent, path)

    monkeypatch.setattr(visuals_module, "render_deterministic_visual", flaky)
    assert process_visual_jobs(db, output) == [{"job_id": visual_job, "status": "retrying"}]
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM slack_file_outbox WHERE id=?", (visual_delivery,)
        ).fetchone()["status"] == "waiting_asset"
        conn.execute("UPDATE visual_jobs SET available_at=? WHERE id=?", (visuals_module.utcnow(), visual_job))
    assert process_visual_jobs(db, output)[0]["status"] == "completed"
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM slack_file_outbox WHERE id=?", (visual_delivery,)
        ).fetchone()["status"] == "queued"


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


def test_visual_claim_fails_after_ready_then_binding_revoke(tmp_path: Path):
    db, _output, client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    flush_slack_outbox(client, db)
    assert [row["id"] for row in ready_visual_deliveries(db)] == [visual_id]
    with connect(db) as conn:
        binding_id = conn.execute(
            "SELECT binding_id FROM slack_file_outbox WHERE id=?", (visual_id,)
        ).fetchone()["binding_id"]
    revoke_binding(db, str(binding_id))
    assert claim_visual_delivery(db, visual_id) is None
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM slack_file_outbox WHERE id=?", (visual_id,)
        ).fetchone()["status"] == "expired"


def test_visual_claim_fails_when_parent_principal_is_revoked(tmp_path: Path):
    db, _output, client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    flush_slack_outbox(client, db)
    assert [row["id"] for row in ready_visual_deliveries(db)] == [visual_id]
    with connect(db) as conn:
        conn.execute("UPDATE parent_principals SET status='revoked'")
    assert claim_visual_delivery(db, visual_id) is None
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM slack_file_outbox WHERE id=?", (visual_id,)
        ).fetchone()["status"] == "expired"


def test_text_worker_stops_after_claim_then_revoke_and_repair_does_not_revive(
    tmp_path: Path,
    monkeypatch,
):
    db, _output, client, text_id, _visual_id = setup_visual_delivery(tmp_path)
    client.messages.clear()
    original_claim = slack_module.claim_delivery

    def revoke_after_claim(db_path, delivery_id):
        claimed = original_claim(db_path, delivery_id)
        if claimed:
            with connect(db_path) as conn:
                binding_id = conn.execute(
                    "SELECT binding_id FROM delivery_outbox WHERE id=?", (delivery_id,)
                ).fetchone()["binding_id"]
            revoke_binding(db_path, str(binding_id))
        return claimed

    monkeypatch.setattr(slack_module, "claim_delivery", revoke_after_claim)
    assert flush_slack_outbox(client, db) == [{"delivery_id": text_id, "status": "unknown"}]
    assert client.messages == []
    assert mark_delivery(db, text_id, status="failed", error="late worker") is False

    with connect(db) as conn:
        owner_id = conn.execute("SELECT parent_id FROM transport_bindings").fetchone()["parent_id"]
    code = create_pairing_code(db, str(owner_id))["pairing_code"]
    repaired = consume_pairing_code(
        db,
        str(code),
        InboundMessage(
            external_event_id="EvTextRepair",
            team_id="T_TEST",
            user_id="U_TEST",
            channel_id="C_TEST",
            text=f"pair {code}",
        ),
    )
    assert repaired["binding_id"]
    assert ready_deliveries(db) == []


def test_visual_worker_stops_after_claim_then_revoke_before_upload_ticket(
    tmp_path: Path,
    monkeypatch,
):
    db, output, client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    flush_slack_outbox(client, db)
    original_claim = slack_module.claim_visual_delivery
    stale_claim: dict[str, str] = {}

    def revoke_after_claim(db_path, delivery_id):
        claimed = original_claim(db_path, delivery_id)
        if claimed:
            stale_claim.update(claimed)
            with connect(db_path) as conn:
                binding_id = conn.execute(
                    "SELECT binding_id FROM slack_file_outbox WHERE id=?", (delivery_id,)
                ).fetchone()["binding_id"]
            revoke_binding(db_path, str(binding_id))
        return claimed

    monkeypatch.setattr(slack_module, "claim_visual_delivery", revoke_after_claim)
    result = flush_slack_file_outbox(client, db, output, uploader=lambda _url, _data: None)
    assert result == [{"delivery_id": visual_id, "status": "unknown"}]
    assert client.tickets == []
    assert (
        mark_visual_delivery(
            db,
            visual_id,
            status="failed",
            attempt_id=stale_claim["attempt_id"],
            lease_token=stale_claim["lease_token"],
            error="late worker",
        )
        is False
    )

    with connect(db) as conn:
        owner_id = conn.execute("SELECT parent_id FROM transport_bindings").fetchone()["parent_id"]
    code = create_pairing_code(db, str(owner_id))["pairing_code"]
    consume_pairing_code(
        db,
        str(code),
        InboundMessage(
            external_event_id="EvVisualRepair",
            team_id="T_TEST",
            user_id="U_TEST",
            channel_id="C_TEST",
            text=f"pair {code}",
        ),
    )
    assert ready_visual_deliveries(db) == []


def test_text_and_visual_enqueue_reject_cross_channel_targets(tmp_path: Path):
    db, _output, _client, text_id, visual_id = setup_visual_delivery(tmp_path)
    with connect(db) as conn:
        row = conn.execute(
            "SELECT binding_id,visual_job_id FROM slack_file_outbox WHERE id=?", (visual_id,)
        ).fetchone()
    with pytest.raises(ValueError, match="binding is unavailable"):
        enqueue_delivery(
            db,
            str(row["binding_id"]),
            OutboundMessage(channel_id="C_OTHER", text="Wrong target."),
            idempotency_key="slack:cross-channel:text",
        )
    with pytest.raises(ValueError, match="binding is unavailable"):
        enqueue_visual_delivery(
            db,
            visual_job_id=str(row["visual_job_id"]),
            binding_id=str(row["binding_id"]),
            depends_on_delivery_id=text_id,
            channel_id="C_OTHER",
            thread_id=None,
            idempotency_key="slack:cross-channel:visual",
        )


def test_cross_channel_tampering_is_fenced_at_readiness_and_claim(tmp_path: Path):
    text_db, _output, _client, text_id, _visual_id = setup_visual_delivery(tmp_path / "text")
    with connect(text_db) as conn:
        payload = jload(
            conn.execute(
                "SELECT payload_json FROM delivery_outbox WHERE id=?", (text_id,)
            ).fetchone()["payload_json"]
        )
        payload["channel_id"] = "C_OTHER"
        conn.execute(
            "UPDATE delivery_outbox SET payload_json=? WHERE id=?",
            (jdump(payload), text_id),
        )
    assert ready_deliveries(text_db) == []
    with connect(text_db) as conn:
        assert conn.execute(
            "SELECT status FROM delivery_outbox WHERE id=?", (text_id,)
        ).fetchone()["status"] == "expired"

    visual_db, _output, client, _text_id, visual_id = setup_visual_delivery(tmp_path / "visual")
    flush_slack_outbox(client, visual_db)
    assert [row["id"] for row in ready_visual_deliveries(visual_db)] == [visual_id]
    with connect(visual_db) as conn:
        conn.execute(
            "UPDATE slack_file_outbox SET channel_id='C_OTHER' WHERE id=?",
            (visual_id,),
        )
    assert claim_visual_delivery(visual_db, visual_id) is None
    with connect(visual_db) as conn:
        assert conn.execute(
            "SELECT status FROM slack_file_outbox WHERE id=?", (visual_id,)
        ).fetchone()["status"] == "expired"


def test_stale_visual_worker_cannot_advance_a_recovered_later_claim(tmp_path: Path):
    db, _output, client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    flush_slack_outbox(client, db)
    stale = claim_visual_delivery(db, visual_id)
    assert stale
    with connect(db) as conn:
        conn.execute(
            "UPDATE slack_file_outbox SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?",
            (visual_id,),
        )
    assert [row["id"] for row in ready_visual_deliveries(db)] == [visual_id]
    current = claim_visual_delivery(db, visual_id)
    assert current and current["attempt_id"] != stale["attempt_id"]
    assert (
        mark_visual_delivery(
            db,
            visual_id,
            status="ticket_acquiring",
            attempt_id=stale["attempt_id"],
            lease_token=stale["lease_token"],
        )
        is False
    )
    assert (
        mark_visual_delivery(
            db,
            visual_id,
            status="failed",
            attempt_id=stale["attempt_id"],
            lease_token=stale["lease_token"],
            error="stale worker",
        )
        is False
    )
    assert mark_visual_delivery(
        db,
        visual_id,
        status="ticket_acquiring",
        attempt_id=current["attempt_id"],
        lease_token=current["lease_token"],
    )


def test_visual_upload_timeout_after_bytes_is_terminal_unknown(tmp_path: Path):
    db, output, client, _text_id, visual_id = setup_visual_delivery(tmp_path)
    flush_slack_outbox(client, db)
    uploads = 0

    def ambiguous_upload(_url, _data):
        nonlocal uploads
        uploads += 1
        raise TimeoutError("bytes may have reached Slack")

    assert flush_slack_file_outbox(
        client,
        db,
        output,
        uploader=ambiguous_upload,
    ) == [{"delivery_id": visual_id, "status": "unknown"}]
    assert flush_slack_file_outbox(
        client,
        db,
        output,
        uploader=ambiguous_upload,
    ) == []
    assert uploads == 1 and len(client.tickets) == 1
    with connect(db) as conn:
        assert conn.execute(
            "SELECT status FROM slack_file_outbox WHERE id=?", (visual_id,)
        ).fetchone()["status"] == "unknown"


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
            "kind": "comparison_cards",
            "purpose": "compare",
            "knowledge_role": "supportive",
            "title": "Synthetic comparison failure",
            "pedagogical_value": "Exercises failure isolation.",
            "caption": "Compare the two imaginary robot gardens.",
            "alt_text": "Two labeled cards compare imaginary robot gardens with different kinds of plants.",
            "subject": "an imaginary robot tending a moon garden",
            "panels": [
                {"label": "FLOWERS", "detail": "Notice the colorful blooms.", "icon": "look"},
                {"label": "FERNS", "detail": "Notice the leafy shapes.", "icon": "look"},
            ],
            "not_to_scale": True,
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

    assert process_visual_jobs(db, output, image_backend=FailingImageBackend())[-1]["status"] == "retrying"
    delivered = flush_slack_outbox(client, db)
    assert {item["delivery_id"] for item in delivered} >= {text_id}
    assert client.messages[-1]["text"] == "Accessible answer text."
    for expected in ("retrying", "failed"):
        with connect(db) as conn:
            conn.execute("UPDATE visual_jobs SET available_at=? WHERE id=?", (visuals_module.utcnow(), visual_job))
        assert process_visual_jobs(db, output, image_backend=FailingImageBackend())[-1]["status"] == expected
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
