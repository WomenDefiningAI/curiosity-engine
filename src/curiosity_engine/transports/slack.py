from __future__ import annotations

import logging
import re
import stat
import threading
import urllib.request
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from ..interaction import (
    TransportConflict,
    active_binding,
    begin_receipt,
    claim_delivery,
    claim_visual_delivery,
    consume_pairing_code,
    create_unassigned_capture,
    enqueue_delivery,
    enqueue_visual_delivery,
    finish_receipt,
    household_resource_context_mode,
    list_inbox,
    mark_delivery,
    mark_visual_delivery,
    ready_deliveries,
    ready_visual_deliveries,
    record_onboarding_checkpoint,
    resolve_inbox,
)
from ..service import CuriosityService
from ..visuals import create_synthetic_visual_job, process_visual_jobs
from .contracts import InboundMessage, OutboundMessage, TransportResult

PAIR_RE = re.compile(r"^pair\s+([A-Z2-9]{8})$", re.IGNORECASE)
ASK_RE = re.compile(r"^ask\s+([A-Za-z0-9_-]{1,120})\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
ASSIGN_RE = re.compile(
    r"^assign\s+(inbox_[A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]{1,120})$", re.IGNORECASE
)
DISMISS_RE = re.compile(r"^dismiss\s+(inbox_[A-Za-z0-9_-]+)$", re.IGNORECASE)
FEEDBACK_RE = re.compile(
    r"^feedback\s+([A-Za-z0-9_-]{1,120})\s+"
    r"(loved|engaged|neutral|too_easy|too_hard|not_used|disliked)(?:\s*:\s*(.*))?$",
    re.IGNORECASE | re.DOTALL,
)


def load_slack_tokens() -> dict[str, str]:
    """Read tokens from the process or the ignored owner-only setup file."""

    import os

    tokens = {
        "SLACK_APP_TOKEN": os.environ.get("SLACK_APP_TOKEN", ""),
        "SLACK_BOT_TOKEN": os.environ.get("SLACK_BOT_TOKEN", ""),
    }
    if all(tokens.values()):
        return tokens
    from ..config import repository_root

    token_file = repository_root() / "private" / "setup" / "slack.env"
    if not token_file.is_file():
        return tokens
    mode = stat.S_IMODE(token_file.stat().st_mode)
    if mode & 0o077:
        raise PermissionError("private/setup/slack.env must be owner-only; run: chmod 600 private/setup/slack.env")
    for raw_line in token_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in tokens and not tokens[key.strip()]:
            tokens[key.strip()] = value.strip().strip("'\"")
    return tokens


class CuriosityServiceLike(Protocol):
    def children(self) -> list[dict[str, Any]]: ...

    def ask(
        self,
        *,
        child_id: str,
        text: str,
        source: str = "parent",
        topics: list[str] | None = None,
        include_private_excerpts: bool = False,
        event_id: str | None = None,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def feedback(self, payload: dict[str, Any]) -> int: ...


def _engine_event_id(message: InboundMessage, suffix: str = "") -> str:
    material = f"{message.transport}:{message.team_id}:{message.external_event_id}:{suffix}"
    return f"evt_slack_{sha256(material.encode()).hexdigest()[:24]}"


def _episode_metadata(message: InboundMessage) -> dict[str, Any]:
    conversation = sha256(f"{message.team_id}:{message.channel_id}".encode()).hexdigest()[:20]
    thread = (
        sha256(f"{message.team_id}:{message.channel_id}:{message.thread_id}".encode()).hexdigest()[:20]
        if message.thread_id
        else ""
    )
    return {
        "learning_scope": "family_signal",
        "subject_role": "child",
        "reporter_role": "parent",
        "initiative": "unknown",
        "conversation_ref": conversation,
        "thread_ref": thread,
    }


def _help_text() -> str:
    return (
        "I help parents catch questions and turn them into small, hands-on learning threads.\n\n"
        "• `connection` — prove Slack transport works without contacting a model\n"
        "• `visual connection` — prove an accessible picture can reach this conversation\n"
        "• `children` — show child IDs\n"
        "• `ask CHILD_ID: Why does ice float?` — get a response now\n"
        "• Send any other note — save it unassigned so I never guess which child it belongs to\n"
        "• `assign INBOX_ID CHILD_ID` — attach a saved question and answer it\n"
        "• `dismiss INBOX_ID` — discard a saved note\n"
        "• `inbox` — list saved, unassigned notes\n"
        "• `feedback CHILD_ID engaged: optional note` — tell me how it went\n"
        "• `privacy` — review the message and storage boundary\n"
        "• `help` — show this again\n\n"
        "Use DMs or explicitly mention me in a paired family channel."
    )


def _format_thread(result: dict[str, Any]) -> str:
    output = result.get("output") or {}
    extension = output.get("physical_extension") or {}
    reasoning = output.get("_reasoning") or {}
    parts = [
        f"*Start here*\n{output.get('hook', 'Follow the question together.')}",
        f"*Show*\n{output.get('show', 'Notice what you can observe together.')}",
        f"*Ask*\n{output.get('ask', 'What do you notice?')}",
        f"*Tiny explanation*\n{output.get('nugget', 'Keep the explanation small and follow the next question.')}",
    ]
    if extension:
        materials = extension.get("materials") or []
        instructions = extension.get("instructions") or []
        physical = [f"*Try it: {extension.get('title', 'A quick investigation')}*"]
        if materials:
            physical.append("Materials: " + ", ".join(str(item) for item in materials))
        physical.extend(f"{index}. {step}" for index, step in enumerate(instructions, start=1))
        parts.append("\n".join(physical))
    public_sources = [str(item) for item in output.get("resource_refs") or [] if str(item).startswith(("https://", "http://"))]
    if public_sources:
        parts.append("*Sources*\n" + "\n".join(f"• <{url}>" for url in public_sources[:3]))
    if reasoning.get("backend") == "deterministic":
        parts.insert(
            0,
            "_Offline demo response — connect a reasoning provider for tailored answers to arbitrary questions._",
        )
    if result.get("visual_job_id"):
        parts.append("_A visual card is being prepared and will follow this answer._")
    return "\n\n".join(parts)[:4_000]


def _response_did_not_pass() -> str:
    return (
        "I could not produce a reliable answer, so I stopped instead of showing a flawed draft. The diagnostic is "
        "saved privately on the computer running Curiosity Engine; you do not need to keep retyping the question. "
        "Run `curiosity doctor` in that computer's terminal to see the redacted answer-quality status."
    )


def _parent_safe_error(exc: Exception) -> str:
    detail = str(exc).casefold()
    if "unassigned note" in detail or "inbox" in detail:
        return "I couldn't find that saved note for this parent. Run `inbox` and try the displayed ID."
    if "child" in detail and ("not found" in detail or "unknown" in detail):
        return "I couldn't find that child ID. Run `children` and try one of the displayed IDs."
    return "I couldn't complete that request. Check the command and IDs, then try again."


class SlackTransport:
    """Transport-neutral Slack message handling; Bolt is only the delivery shell."""

    def __init__(
        self,
        db_path: str | Path,
        output_dir: str | Path,
        *,
        service: CuriosityServiceLike | None = None,
    ):
        self.db_path = Path(db_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self._service = service

    @property
    def service(self) -> CuriosityServiceLike:
        if self._service is None:
            self._service = CuriosityService(self.db_path, self.output_dir)
        return self._service

    def _queue(
        self,
        message: InboundMessage,
        binding_id: str,
        text: str,
        *,
        purpose: str,
    ) -> str:
        outbound = OutboundMessage(channel_id=message.channel_id, text=text, thread_id=message.thread_id)
        return enqueue_delivery(
            self.db_path,
            binding_id,
            outbound,
            idempotency_key=f"slack:{message.external_event_id}:{purpose}",
        )

    def handle(self, message: InboundMessage) -> TransportResult:
        binding = active_binding(self.db_path, message)
        try:
            if not begin_receipt(self.db_path, message, binding["id"] if binding else None):
                return TransportResult(status="duplicate", message="This Slack event was already handled.")
        except TransportConflict:
            raise

        pair = PAIR_RE.fullmatch(message.text.strip())
        if not binding:
            if not pair:
                finish_receipt(self.db_path, message, status="rejected", error="unpaired Slack identity/channel")
                return TransportResult(
                    status="rejected",
                    message=(
                        "This conversation is not paired yet. On the computer running Curiosity Engine, create a "
                        "pairing code, then send `pair CODE` here."
                    ),
                )
            try:
                paired = consume_pairing_code(self.db_path, pair.group(1), message)
            except PermissionError as exc:
                finish_receipt(self.db_path, message, status="rejected", error=str(exc))
                return TransportResult(status="rejected", message=str(exc))
            outbound_id = self._queue(
                message,
                paired["binding_id"],
                "Paired. I will only act for this Slack identity in this conversation.\n\n" + _help_text(),
                purpose="paired",
            )
            finish_receipt(self.db_path, message, status="completed")
            return TransportResult(
                status="paired",
                message="Paired successfully.",
                binding_id=paired["binding_id"],
                outbound_id=outbound_id,
            )

        binding_id = str(binding["id"])
        text = message.text.strip()
        normalized = text.casefold()
        visual_job_id: str | None = None
        visual_purpose = "response_visual"
        try:
            if pair:
                reply = "This conversation is already paired. Use `help` to see what I can do."
                purpose = "already-paired"
                result_status = "completed"
                event_id = None
                inbox_id = None
            elif normalized in {"connection", "connection test", "ping"}:
                record_onboarding_checkpoint(
                    self.db_path,
                    "transport_verified",
                    status="pending",
                    evidence={"transport": "slack", "protocol": 1},
                )
                reply = (
                    "Slack connection works. This fixed response did not contact an AI model, load a child "
                    "profile, or read family resources. Delivery confirmation is being recorded locally."
                )
                purpose = "connection"
                result_status = "completed"
                event_id = None
                inbox_id = None
            elif normalized in {"visual connection", "visual connection test"}:
                record_onboarding_checkpoint(
                    self.db_path,
                    "visual_delivery_verified",
                    status="pending",
                    evidence={"transport": "slack", "protocol": 1, "family_data_sent": False},
                )
                event_id = _engine_event_id(message, "visual-connection")
                visual_job_id = create_synthetic_visual_job(self.db_path, event_id)
                visual_purpose = "visual_connection"
                reply = (
                    "The fixed visual connection card is being prepared. It does not contact an AI model or "
                    "use child, family, Slack-message, or private-resource content."
                )
                purpose = "visual-connection"
                result_status = "completed"
                inbox_id = None
            elif normalized in {"help", "commands"}:
                reply = _help_text()
                purpose = "help"
                result_status = "completed"
                event_id = None
                inbox_id = None
            elif normalized == "children":
                children = self.service.children()
                reply = (
                    "Child IDs:\n" + "\n".join(f"• `{child['id']}` — {child['name']}" for child in children)
                    if children
                    else "No child profiles exist yet. Add one in the local setup console or with `curiosity child add`."
                )
                purpose = "children"
                result_status = "completed"
                event_id = None
                inbox_id = None
            elif normalized == "inbox":
                rows = [row for row in list_inbox(self.db_path) if row["parent_id"] == binding["parent_id"]]
                reply = (
                    "Unassigned notes:\n"
                    + "\n".join(f"• `{row['id']}` — {row['text'][:180]}" for row in rows[:10])
                    + ("\n\nUse `assign INBOX_ID CHILD_ID` or `dismiss INBOX_ID`." if rows else "")
                    if rows
                    else "You have no unassigned notes."
                )
                purpose = "inbox"
                result_status = "completed"
                event_id = None
                inbox_id = None
            elif normalized == "privacy":
                resource_mode = household_resource_context_mode(self.db_path)
                include_private_excerpts = resource_mode == "selected_excerpts"
                resource_disclosure = (
                    "Your household has opted in to selected excerpts: small relevant passages may enter the "
                    "bounded hosted-model request, but source passages are not posted verbatim to Slack."
                    if include_private_excerpts
                    else "Purchased-resource excerpts stay local; only non-excerpt metadata may be considered."
                )
                reply = (
                    "Slack processes the messages and replies you send here. Curiosity Engine stores its durable "
                    "family record in the ignored `private/` directory on the computer running it. A configured "
                    "model provider processes only the bounded context selected for that request. Deterministic "
                    "visual cards are created locally; any card uploaded here is then stored by Slack under this "
                    "workspace's policies. Opt-in decorative generation sends a minimized generic scene prompt "
                    "derived from the broad topic, while code rejects known identities and private-context categories. "
                    + resource_disclosure
                )
                purpose = "privacy"
                result_status = "completed"
                event_id = None
                inbox_id = None
            elif match := FEEDBACK_RE.fullmatch(text):
                child_id, outcome, note = match.groups()
                if child_id not in {str(child["id"]) for child in self.service.children()}:
                    raise ValueError("child not found")
                self.service.feedback(
                    {
                        "child_id": child_id,
                        "outcome": outcome.casefold(),
                        "note": note.strip() if note and note.strip() else None,
                        "source": "slack_parent",
                    }
                )
                reply = f"Saved `{outcome.casefold()}` feedback for `{child_id}`."
                purpose = "feedback"
                result_status = "completed"
                event_id = None
                inbox_id = None
            elif match := ASK_RE.fullmatch(text):
                resource_mode = household_resource_context_mode(self.db_path)
                include_private_excerpts = resource_mode == "selected_excerpts"
                child_id, question = match.groups()
                event_id = _engine_event_id(message, child_id)
                response = self.service.ask(
                    child_id=child_id,
                    text=question.strip(),
                    source="slack_parent_report",
                    include_private_excerpts=include_private_excerpts,
                    event_id=event_id,
                    context_metadata=_episode_metadata(message),
                )
                if response.get("status") == "completed":
                    reply = _format_thread(response)
                    visual_job_id = response.get("visual_job_id")
                    purpose = "answer"
                    result_status = "completed"
                else:
                    reply = _response_did_not_pass()
                    purpose = "answer_rejected"
                    result_status = "rejected"
                inbox_id = None
            elif match := ASSIGN_RE.fullmatch(text):
                resource_mode = household_resource_context_mode(self.db_path)
                include_private_excerpts = resource_mode == "selected_excerpts"
                inbox_id, child_id = match.groups()
                row = next(
                    (item for item in list_inbox(self.db_path) if item["id"] == inbox_id),
                    None,
                )
                if not row or row["parent_id"] != binding["parent_id"]:
                    raise ValueError("unassigned note not found for this parent")
                event_id = _engine_event_id(message, f"assign:{inbox_id}:{child_id}")
                response = self.service.ask(
                    child_id=child_id,
                    text=str(row["text"]),
                    source="slack_parent_report",
                    include_private_excerpts=include_private_excerpts,
                    event_id=event_id,
                    context_metadata=_episode_metadata(message),
                )
                if response.get("status") == "completed":
                    resolve_inbox(self.db_path, inbox_id, child_id=child_id)
                    reply = f"Assigned `{inbox_id}` to `{child_id}`.\n\n" + _format_thread(response)
                    visual_job_id = response.get("visual_job_id")
                    purpose = "assign"
                    result_status = "completed"
                else:
                    reply = _response_did_not_pass() + f" The note `{inbox_id}` remains unassigned."
                    purpose = "assign_rejected"
                    result_status = "rejected"
            elif match := DISMISS_RE.fullmatch(text):
                inbox_id = match.group(1)
                row = next(
                    (item for item in list_inbox(self.db_path) if item["id"] == inbox_id),
                    None,
                )
                if not row or row["parent_id"] != binding["parent_id"]:
                    raise ValueError("unassigned note not found for this parent")
                resolve_inbox(self.db_path, inbox_id, child_id=None, dismiss=True)
                reply = f"Dismissed `{inbox_id}`."
                purpose = "dismiss"
                result_status = "completed"
                event_id = None
            else:
                capture = create_unassigned_capture(self.db_path, message, str(binding["parent_id"]))
                inbox_id = str(capture["inbox_id"])
                reply = (
                    f"Saved as `{inbox_id}` without choosing a child.\n\n"
                    f"Reply `assign {inbox_id} CHILD_ID` to attach it and get a learning thread, or "
                    f"`dismiss {inbox_id}`. Use `children` to see IDs."
                )
                purpose = "unassigned"
                result_status = "unassigned"
                event_id = None
            outbound_id = self._queue(message, binding_id, reply, purpose=purpose)
            if visual_job_id:
                enqueue_visual_delivery(
                    self.db_path,
                    visual_job_id=visual_job_id,
                    binding_id=binding_id,
                    depends_on_delivery_id=outbound_id,
                    channel_id=message.channel_id,
                    thread_id=message.thread_id,
                    idempotency_key=f"slack:{message.external_event_id}:{purpose}:visual",
                    purpose=visual_purpose,
                )
            finish_receipt(self.db_path, message, status="completed", event_id=event_id)
            return TransportResult(
                status=result_status,
                message=reply,
                event_id=event_id,
                inbox_id=inbox_id,
                binding_id=binding_id,
                outbound_id=outbound_id,
            )
        except (KeyError, ValueError) as exc:
            safe_error = _parent_safe_error(exc)
            outbound_id = self._queue(
                message,
                binding_id,
                f"I could not do that: {safe_error}\n\nUse `help` for the supported commands.",
                purpose="rejected",
            )
            finish_receipt(
                self.db_path,
                message,
                status="rejected",
                error=f"{exc.__class__.__name__}: {str(exc)[:500]}",
            )
            return TransportResult(
                status="rejected",
                message=safe_error,
                binding_id=binding_id,
                outbound_id=outbound_id,
            )
        except Exception as exc:
            outbound_id = self._queue(
                message,
                binding_id,
                "Something went wrong locally. Your family data stayed on this computer. Run `curiosity doctor` for details.",
                purpose="failed",
            )
            finish_receipt(
                self.db_path,
                message,
                status="failed",
                error=f"local processing failure: {exc.__class__.__name__}",
            )
            return TransportResult(
                status="failed",
                message="Local processing failed.",
                binding_id=binding_id,
                outbound_id=outbound_id,
            )


def flush_slack_outbox(client: Any, db_path: str | Path) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for row in ready_deliveries(db_path):
        if not claim_delivery(db_path, row["id"]):
            continue
        message = OutboundMessage.model_validate(row["payload"])
        kwargs: dict[str, Any] = {"channel": message.channel_id, "text": message.text}
        if message.thread_id:
            kwargs["thread_ts"] = message.thread_id
        if message.blocks:
            kwargs["blocks"] = message.blocks
        try:
            response = client.chat_postMessage(**kwargs)
        except Exception as exc:
            # SlackApiError confirms that Slack rejected the request. Other failures may be
            # ambiguous after a network break, so do not retry them automatically.
            if exc.__class__.__name__ == "SlackApiError":
                mark_delivery(db_path, row["id"], status="failed", error=str(exc))
                results.append({"delivery_id": row["id"], "status": "failed"})
            else:
                mark_delivery(db_path, row["id"], status="unknown", error=exc.__class__.__name__)
                results.append({"delivery_id": row["id"], "status": "unknown"})
            continue
        external_id = str(response.get("ts", ""))
        mark_delivery(db_path, row["id"], status="sent", external_message_id=external_id)
        if str(row.get("idempotency_key") or "").endswith(":connection"):
            record_onboarding_checkpoint(
                db_path,
                "transport_verified",
                status="pass",
                evidence={"transport": "slack", "protocol": 1, "delivery_confirmed": True},
            )
        results.append({"delivery_id": row["id"], "status": "sent"})
    return results


def _upload_bytes(upload_url: str, data: bytes) -> None:
    request = urllib.request.Request(
        upload_url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/octet-stream", "Content-Length": str(len(data))},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - Slack supplies the signed URL
        if int(response.status) < 200 or int(response.status) >= 300:
            raise RuntimeError(f"Slack byte upload failed with HTTP {response.status}")


def flush_slack_file_outbox(
    client: Any,
    db_path: str | Path,
    output_dir: str | Path,
    *,
    uploader: Callable[[str, bytes], None] | None = None,
) -> list[dict[str, str]]:
    """Deliver validated private assets with at-most-one automatic completion attempt."""

    results: list[dict[str, str]] = []
    allowed_root = (Path(output_dir).resolve() / "visuals").resolve()
    upload = uploader or _upload_bytes
    for row in ready_visual_deliveries(db_path):
        delivery_id = str(row["id"])
        if not claim_visual_delivery(db_path, delivery_id):
            continue
        stage = "ticket_acquiring"
        try:
            path = Path(str(row["path"])).resolve()
            if not path.is_relative_to(allowed_root) or not path.is_file():
                raise ValueError("visual asset path is outside the private output root")
            data = path.read_bytes()
            if len(data) != int(row["byte_count"]):
                raise ValueError("visual asset byte count changed after validation")
            if sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("visual asset hash changed after validation")
            if row["mime_type"] != "image/png" or not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("visual asset is not a validated PNG")

            ticket = client.files_getUploadURLExternal(
                filename=row["filename"],
                length=len(data),
                alt_txt=row["alt_text"],
            )
            upload_url = str(ticket.get("upload_url") or "")
            file_id = str(ticket.get("file_id") or "")
            if not upload_url.startswith("https://") or not file_id:
                raise RuntimeError("Slack did not return a valid upload ticket")
            stage = "ticket_acquired"
            mark_visual_delivery(db_path, delivery_id, status=stage, slack_file_id=file_id)
            stage = "uploading"
            mark_visual_delivery(db_path, delivery_id, status=stage, slack_file_id=file_id)
            upload(upload_url, data)
            stage = "bytes_uploaded"
            mark_visual_delivery(db_path, delivery_id, status=stage, slack_file_id=file_id)
            stage = "completing"
            mark_visual_delivery(db_path, delivery_id, status=stage, slack_file_id=file_id)
            kwargs: dict[str, Any] = {
                "files": [{"id": file_id, "title": row["title"]}],
                "channel_id": row["channel_id"],
                "initial_comment": row["caption"],
            }
            if row.get("thread_id"):
                kwargs["thread_ts"] = row["thread_id"]
            response = client.files_completeUploadExternal(**kwargs)
            external_id = file_id
            response_files = response.get("files") or []
            if response_files and isinstance(response_files[0], dict):
                external_id = str(response_files[0].get("id") or file_id)
            mark_visual_delivery(
                db_path,
                delivery_id,
                status="sent",
                slack_file_id=file_id,
                external_message_id=external_id,
            )
            if row.get("purpose") == "visual_connection":
                record_onboarding_checkpoint(
                    db_path,
                    "visual_delivery_verified",
                    status="pass",
                    evidence={
                        "transport": "slack",
                        "protocol": 1,
                        "delivery_confirmed": True,
                        "family_data_sent": False,
                    },
                )
            results.append({"delivery_id": delivery_id, "status": "sent"})
        except Exception as exc:
            slack_rejection = exc.__class__.__name__ == "SlackApiError"
            if stage == "completing" and not slack_rejection:
                status = "unknown"
            elif slack_rejection or isinstance(exc, ValueError):
                status = "expired"
            else:
                status = "failed"
            mark_visual_delivery(db_path, delivery_id, status=status, error=exc.__class__.__name__)
            results.append({"delivery_id": delivery_id, "status": status})
    return results


def _run_visual_worker(client: Any, db_path: str | Path, output_dir: str | Path, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            process_visual_jobs(db_path, output_dir)
            flush_slack_file_outbox(client, db_path, output_dir)
        except Exception:
            logging.getLogger(__name__).exception("visual worker iteration failed")
        stop.wait(1.0)


def _make_slack_event_receiver(transport: SlackTransport, db_path: str | Path) -> Any:
    # Slack Bolt discovers injectable arguments from positional parameter names.
    # These must not be keyword-only even though Bolt invokes the listener with kwargs.
    def receive(body: dict[str, Any], event: dict[str, Any], client: Any, **_: Any) -> None:
        if event.get("bot_id") or event.get("subtype"):
            return
        event_type = str(event.get("type", ""))
        if event_type == "message" and event.get("channel_type") != "im":
            return
        text = re.sub(r"<@[A-Z0-9]+>", "", str(event.get("text", ""))).strip()
        if not text:
            return
        external_event_id = str(body.get("event_id") or event.get("client_msg_id") or event.get("ts") or "")
        incoming = InboundMessage(
            external_event_id=external_event_id,
            team_id=str(body.get("team_id") or event.get("team") or ""),
            user_id=str(event.get("user") or ""),
            channel_id=str(event.get("channel") or ""),
            text=text,
            thread_id=str(event.get("thread_ts")) if event.get("thread_ts") else None,
            occurred_at=str(event.get("event_ts")) if event.get("event_ts") else None,
        )
        result = transport.handle(incoming)
        if result.status == "rejected" and not result.binding_id:
            direct: dict[str, Any] = {"channel": incoming.channel_id, "text": result.message}
            if incoming.thread_id:
                direct["thread_ts"] = incoming.thread_id
            client.chat_postMessage(**direct)
        flush_slack_outbox(client, db_path)

    return receive


def build_slack_app(db_path: str | Path, output_dir: str | Path) -> Any:
    try:
        from slack_bolt import App
    except ImportError as exc:  # pragma: no cover - exercised by doctor and install guide
        raise RuntimeError("Install Slack support with: python -m pip install -e '.[slack]'") from exc

    bot_token = load_slack_tokens()["SLACK_BOT_TOKEN"]
    if not bot_token.startswith("xoxb-"):
        raise RuntimeError("SLACK_BOT_TOKEN is missing or does not look like a bot token")
    app = App(token=bot_token)
    transport = SlackTransport(db_path, output_dir)
    receive = _make_slack_event_receiver(transport, db_path)

    app.event("message")(receive)
    app.event("app_mention")(receive)
    return app


def run_slack_connector(db_path: str | Path, output_dir: str | Path) -> None:
    try:
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError as exc:  # pragma: no cover - exercised by doctor and install guide
        raise RuntimeError("Install Slack support with: python -m pip install -e '.[slack]'") from exc

    app_token = load_slack_tokens()["SLACK_APP_TOKEN"]
    if not app_token.startswith("xapp-"):
        raise RuntimeError("SLACK_APP_TOKEN is missing or does not look like an app token")
    app = build_slack_app(db_path, output_dir)
    flush_slack_outbox(app.client, db_path)
    stop = threading.Event()
    worker = threading.Thread(
        target=_run_visual_worker,
        args=(app.client, db_path, output_dir, stop),
        name="curiosity-visual-worker",
        daemon=True,
    )
    worker.start()
    try:
        SocketModeHandler(app, app_token).start()
    finally:
        stop.set()
        worker.join(timeout=5)
