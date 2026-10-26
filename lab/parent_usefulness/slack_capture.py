"""Exercise Slack event and button handlers; replace only network I/O in local captures."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


class SlackCapture:
    def __init__(self, db, output, service):
        from curiosity_engine.transports.slack import SlackTransport

        self.db, self.output = db, output
        self.transport = SlackTransport(db, output, service=service)
        self.messages, self.tickets = [], []
        self.token = "synthetic-capture-token"
        self.thread = "1700000000.000001"
        self.team, self.user, self.channel = "EVAL_TEAM", "EVAL_PARENT", "EVAL_CHANNEL"

    def chat_postMessage(self, **kwargs):
        message = {**kwargs, "ts": f"1700000001.{len(self.messages) + 1:06d}"}
        self.messages.append(message)
        return {"ts": message["ts"]}

    def chat_update(self, **kwargs):
        for message in self.messages:
            if message["ts"] == kwargs["ts"]:
                message.update(kwargs)
        return {"ts": kwargs["ts"]}

    def reactions_add(self, **_kwargs):
        return {"ok": True}

    def files_getUploadURLExternal(self, **kwargs):
        self.tickets.append(kwargs)
        return {"file_id": f"F_CAPTURE_{len(self.tickets)}", "upload_url": "https://upload.invalid/capture"}

    def files_completeUploadExternal(self, **kwargs):
        assert kwargs["channel_id"] == self.channel and kwargs["thread_ts"] == self.thread
        return {"files": kwargs["files"]}

    def step(self, step, *, number, photos):
        from curiosity_engine.db import connect
        from curiosity_engine.interactions import SLACK_INTERACTION_BUTTON, SLACK_INTERACTION_SELECT
        from curiosity_engine.transports.slack import (
            _make_semantic_interaction_receiver,
            _make_slack_event_receiver,
            _make_slack_response_action_receiver,
            flush_slack_artifact_outbox,
            flush_slack_outbox,
        )

        with connect(self.db) as conn:
            old_receipts = {r[0] for r in conn.execute("SELECT external_event_id FROM transport_receipts")}
            old_files = {r[0] for r in conn.execute("SELECT id FROM artifact_file_outbox")}
        old_messages = len(self.messages)
        if step.get("button"):
            source = self.messages[-1]
            candidates = [(block, element) for block in source.get("blocks", [])
                          for element in block.get("elements", []) if element.get("type") == "button"]
            block, button = next((block, button) for block, button in candidates
                                 if step["button"].casefold() in button.get("text", {}).get("text", "").casefold())
            body = {"trigger_id": f"capture-button-{number}", "team": {"id": self.team},
                    "user": {"id": self.user}, "channel": {"id": self.channel}, "message": source.copy()}
            action = {**button, "block_id": block["block_id"], "action_ts": f"1700000002.{number:06d}"}
            receiver = (_make_semantic_interaction_receiver(self.transport, self.db)
                        if (button["action_id"].startswith(SLACK_INTERACTION_BUTTON + "_")
                            or button["action_id"] == SLACK_INTERACTION_SELECT)
                        else _make_slack_response_action_receiver(self.transport, self.db, action_id=button["action_id"]))
            receiver(lambda: None, body, action, self)
            delivered_once = len(self.messages)
            receiver(lambda: None, body, action, self)
            assert len(self.messages) == delivered_once, "Duplicate button posted another answer"
        else:
            downloads = {f"https://files.slack.com/capture/{i}": path.read_bytes() for i, path in enumerate(photos)}
            event = {"type": "app_mention", "user": self.user, "channel": self.channel,
                     "text": step["text"], "thread_ts": self.thread, "ts": f"1700000003.{number:06d}",
                     "files": [{"id": f"F_INPUT_{number}_{i}", "mimetype": "image/png" if path.suffix == ".png"
                                else "image/jpeg", "size": path.stat().st_size,
                                "url_private_download": f"https://files.slack.com/capture/{i}"}
                               for i, path in enumerate(photos)]}
            receiver = _make_slack_event_receiver(self.transport, self.db)
            body = {"event_id": f"capture-message-{number}", "team_id": self.team}
            with patch("curiosity_engine.transports.slack._download_slack_bytes", side_effect=lambda url, _token: downloads[url]):
                receiver(body, event, self)
                delivered_once = len(self.messages)
                receiver(body, event, self)
                assert len(self.messages) == delivered_once, "Duplicate event posted another answer"
        flush_slack_outbox(self, self.db)
        deliveries = flush_slack_artifact_outbox(self, self.db, self.output, uploader=lambda _url, _data: None)
        with connect(self.db) as conn:
            receipts = [dict(r) for r in conn.execute("SELECT * FROM transport_receipts")
                        if r["external_event_id"] not in old_receipts]
            files = [dict(r) for r in conn.execute("SELECT * FROM artifact_file_outbox") if r["id"] not in old_files]
        errors = [r["error"] for r in receipts if r["status"] in {"failed", "rejected"}]
        if not receipts and len(self.messages) == old_messages:
            raise RuntimeError("The selected Slack handler did not record a receipt")
        result = {"status": "rejected" if errors else "completed",
                  "message": "\n\n".join(m["text"] for m in self.messages[old_messages:]),
                  "event_id": next((r["event_id"] for r in reversed(receipts) if r["event_id"]), None),
                  "button_replay_verified": True, "delivery_errors": errors, "file_deliveries": deliveries}
        pdf = next((f for f in files if f["mime_type"] == "application/pdf"), None)
        if pdf:
            result["artifact"] = {"artifact_id": pdf["artifact_id"], "title": pdf["title"],
                                  "pdf_path": pdf["path"], "sha256": pdf["sha256"],
                                  "preview_path": str(Path(pdf["path"]).with_suffix(".png"))}
        return result
