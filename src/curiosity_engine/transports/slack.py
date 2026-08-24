from __future__ import annotations

import re
import stat
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from ..interaction import (
    TransportConflict,
    active_binding,
    begin_receipt,
    consume_pairing_code,
    create_unassigned_capture,
    enqueue_delivery,
    finish_receipt,
    household_resource_context_mode,
    list_inbox,
    mark_delivery,
    ready_deliveries,
    resolve_inbox,
)
from ..service import CuriosityService
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
    ) -> dict[str, Any]: ...

    def feedback(self, payload: dict[str, Any]) -> int: ...


def _engine_event_id(message: InboundMessage, suffix: str = "") -> str:
    material = f"{message.transport}:{message.team_id}:{message.external_event_id}:{suffix}"
    return f"evt_slack_{sha256(material.encode()).hexdigest()[:24]}"


def _help_text() -> str:
    return (
        "I help parents catch questions and turn them into small, hands-on learning threads.\n\n"
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
    if reasoning.get("private_resource_matches", 0):
        parts.append("_Relevant context from your private family library was available for this response._")
    if reasoning.get("backend") == "deterministic":
        parts.insert(
            0,
            "_Offline demo response — connect a reasoning provider for tailored answers to arbitrary questions._",
        )
    return "\n\n".join(parts)[:4_000]


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
        self.service = service or CuriosityService(self.db_path, output_dir)

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
        resource_mode = household_resource_context_mode(self.db_path)
        include_private_excerpts = resource_mode == "selected_excerpts"
        text = message.text.strip()
        normalized = text.casefold()
        try:
            if pair:
                reply = "This conversation is already paired. Use `help` to see what I can do."
                purpose = "already-paired"
                result_status = "completed"
                event_id = None
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
                resource_disclosure = (
                    "Your household has opted in to selected excerpts: small relevant passages may enter the "
                    "bounded hosted-model request, but source passages are not posted verbatim to Slack."
                    if include_private_excerpts
                    else "Purchased-resource excerpts stay local; only non-excerpt metadata may be considered."
                )
                reply = (
                    "Slack processes the messages and replies you send here. Curiosity Engine stores its durable "
                    "family record in the ignored `private/` directory on the computer running it. A configured "
                    "model provider processes only the bounded context selected for that request. "
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
                child_id, question = match.groups()
                event_id = _engine_event_id(message, child_id)
                response = self.service.ask(
                    child_id=child_id,
                    text=question.strip(),
                    source="slack",
                    include_private_excerpts=include_private_excerpts,
                    event_id=event_id,
                )
                reply = _format_thread(response)
                purpose = "answer"
                result_status = "completed"
                inbox_id = None
            elif match := ASSIGN_RE.fullmatch(text):
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
                    source="slack_inbox",
                    include_private_excerpts=include_private_excerpts,
                    event_id=event_id,
                )
                resolve_inbox(self.db_path, inbox_id, child_id=child_id)
                reply = f"Assigned `{inbox_id}` to `{child_id}`.\n\n" + _format_thread(response)
                purpose = "assign"
                result_status = "completed"
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
            safe_error = str(exc)[:500]
            outbound_id = self._queue(
                message,
                binding_id,
                f"I could not do that: {safe_error}\n\nUse `help` for the supported commands.",
                purpose="rejected",
            )
            finish_receipt(self.db_path, message, status="rejected", error=safe_error)
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
        message = OutboundMessage.model_validate(row["payload"])
        mark_delivery(db_path, row["id"], status="sending")
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
        results.append({"delivery_id": row["id"], "status": "sent"})
    return results


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
    SocketModeHandler(app, app_token).start()
