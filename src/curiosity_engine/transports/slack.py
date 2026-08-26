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

from ..artifact_delivery import (
    claim_artifact_delivery,
    enqueue_artifact_delivery,
    mark_artifact_delivery,
    ready_artifact_deliveries,
)
from ..contracts import InteractionEvent, InteractionOption, InteractionPlan
from ..db import connect, jload
from ..interaction import (
    TransportConflict,
    active_binding,
    begin_receipt,
    claim_delivery,
    claim_visual_delivery,
    consume_pairing_code,
    create_unassigned_capture,
    delivered_slack_response,
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
from ..interactions import (
    SLACK_INTERACTION_BUTTON,
    SLACK_INTERACTION_SELECT,
    create_interaction,
    interaction_blocks,
    resolve_interaction,
)
from ..presentation import format_learning_thread, response_did_not_pass
from ..service import CuriosityService
from ..sessions import SessionStore
from ..visuals import create_synthetic_visual_job, process_visual_jobs
from .contracts import InboundMessage, OutboundMessage, TransportResult

PAIR_RE = re.compile(r"^pair\s+([A-Z2-9]{8})$", re.IGNORECASE)
ASK_RE = re.compile(r"^ask\s+([A-Za-z0-9_-]{1,120})\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
ASSIGN_RE = re.compile(
    r"^assign\s+(inbox_[A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]{1,120})$", re.IGNORECASE
)
DISMISS_RE = re.compile(r"^dismiss\s+(inbox_[A-Za-z0-9_-]+)$", re.IGNORECASE)
RATE_RESPONSE_RE = re.compile(
    r"^rate_response\s+([A-Za-z0-9_-]{1,120})\s+(helpful|not_helpful)$", re.IGNORECASE
)
RETRY_RESPONSE_RE = re.compile(r"^retry_response\s+([A-Za-z0-9_-]{1,120})$", re.IGNORECASE)
INBOX_BLOCK_RE = re.compile(r"^curiosity_inbox:(inbox_[A-Za-z0-9_-]+)$")
RESPONSE_BLOCK_RE = re.compile(r"^curiosity_response:([A-Za-z0-9_-]{1,120})$")
HEALTH_RE = re.compile(
    r"^(?:(?:are\s+you|is\s+(?:this|it))\s+)?(?:still\s+)?(?:working|online|connected)\??$",
    re.IGNORECASE,
)
FEEDBACK_RE = re.compile(
    r"^feedback\s+([A-Za-z0-9_-]{1,120})\s+"
    r"(loved|engaged|neutral|too_easy|too_hard|not_used|disliked)(?:\s*:\s*(.*))?$",
    re.IGNORECASE | re.DOTALL,
)

INBOX_ASSIGN_ACTION = "curiosity_inbox_assign"
INBOX_DISMISS_ACTION = "curiosity_inbox_dismiss"
RESPONSE_HELPFUL_ACTION = "curiosity_response_helpful"
RESPONSE_NOT_HELPFUL_ACTION = "curiosity_response_not_helpful"
RESPONSE_RETRY_ACTION = "curiosity_response_retry"


def _command_text(text: str) -> str:
    """Accept Slack inline-code formatting without changing saved note text."""

    value = text.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1].strip()
    return value


def _child_action_token(child_id: str) -> str:
    return sha256(child_id.encode()).hexdigest()[:20]


def _plain_text(value: Any, *, fallback: str) -> str:
    rendered = re.sub(r"[\x00-\x1f\x7f]", " ", str(value)).strip()
    return (rendered or fallback)[:75]


def _mrkdwn_text(value: Any, *, limit: int = 500) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")[:limit]


def _inbox_controls(inbox_id: str, children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    options = [
        {
            "text": {
                "type": "plain_text",
                "text": _plain_text(child.get("name"), fallback="Child"),
                "emoji": True,
            },
            "value": _child_action_token(str(child["id"])),
        }
        for child in children[:100]
        if child.get("id")
    ]
    if options:
        elements.append(
            {
                "type": "static_select",
                "action_id": INBOX_ASSIGN_ACTION,
                "placeholder": {"type": "plain_text", "text": "Choose a child", "emoji": True},
                "options": options,
            }
        )
    elements.append(
        {
            "type": "button",
            "action_id": INBOX_DISMISS_ACTION,
            "text": {"type": "plain_text", "text": "Dismiss", "emoji": True},
            "value": inbox_id,
        }
    )
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Who was this for?* Choose a child to make a learning thread, or dismiss it.",
            },
        },
        {
            "type": "actions",
            "block_id": f"curiosity_inbox:{inbox_id}",
            "elements": elements,
        },
    ]


def _inbox_list_blocks(
    rows: list[dict[str, Any]], children: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*Unassigned notes*"},
        }
    ]
    for row in rows[:10]:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": _mrkdwn_text(row.get("text", ""))},
            }
        )
        blocks.append(_inbox_controls(str(row["id"]), children)[1])
    return blocks


def _mrkdwn_sections(text: str, *, limit: int = 2_900) -> list[str]:
    sections: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            sections.append(current)
        while len(paragraph) > limit:
            sections.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        current = paragraph
    if current:
        sections.append(current)
    return sections


def _response_blocks(text: str, event_id: str, *, retry_only: bool = False) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    if not retry_only:
        elements.extend(
            [
                {
                    "type": "button",
                    "action_id": RESPONSE_HELPFUL_ACTION,
                    "text": {"type": "plain_text", "text": "👍 Helpful", "emoji": True},
                    "value": event_id,
                },
                {
                    "type": "button",
                    "action_id": RESPONSE_NOT_HELPFUL_ACTION,
                    "text": {"type": "plain_text", "text": "👎 Not for us", "emoji": True},
                    "value": event_id,
                },
            ]
        )
    elements.append(
        {
            "type": "button",
            "action_id": RESPONSE_RETRY_ACTION,
            "text": {"type": "plain_text", "text": "✨ Try another", "emoji": True},
            "value": event_id,
        }
    )
    return [
        *[
            {"type": "section", "text": {"type": "mrkdwn", "text": section}}
            for section in _mrkdwn_sections(text)
        ],
        {
            "type": "actions",
            "block_id": f"curiosity_response:{event_id}",
            "elements": elements,
        },
    ]


def _semantic_response_blocks(
    db_path: str | Path,
    *,
    binding_id: str,
    session_id: str,
    event_id: str,
    text: str,
) -> list[dict[str, Any]]:
    plan = InteractionPlan(
        kind="rate_output",
        title="What next?",
        prompt="Use a shortcut, or just tell me naturally what you want changed.",
        options=[
            InteractionOption(label="👍 Helpful", intent="rate_response", payload={"event_id": event_id, "rating": "helpful"}),
            InteractionOption(label="👎 Not for us", intent="rate_response", payload={"event_id": event_id, "rating": "not_helpful"}),
            InteractionOption(label="✨ Try another", intent="retry_response", payload={"event_id": event_id}),
            InteractionOption(label="🕵️ Make a challenge", intent="create_artifact", payload={"event_id": event_id, "artifact_type": "challenge"}),
        ],
    )
    presented = create_interaction(
        db_path,
        binding_id=binding_id,
        session_id=session_id,
        plan=plan,
    )
    return [
        *[
            {"type": "section", "text": {"type": "mrkdwn", "text": section}}
            for section in _mrkdwn_sections(text)
        ],
        *interaction_blocks(presented),
    ]


def _existing_thread_session(db_path: str | Path, message: InboundMessage, binding_id: str) -> dict[str, Any] | None:
    if not message.thread_id:
        return None
    conversation_ref = sha256(f"{message.team_id}:{message.channel_id}".encode()).hexdigest()[:20]
    thread_ref = sha256(f"{message.team_id}:{message.channel_id}:{message.thread_id}".encode()).hexdigest()[:20]
    store = SessionStore(db_path)
    existing = store.find(
        origin="slack",
        binding_id=binding_id,
        conversation_ref=conversation_ref,
        thread_ref=thread_ref,
    )
    if existing:
        return existing
    # Upgrade older delivered learning threads lazily. Exact binding plus the
    # hashed conversation/thread lineage prevents cross-channel attribution.
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT e.id AS event_id,e.child_id,e.text,r.output_json
               FROM events e JOIN responses r ON r.event_id=e.id
               JOIN transport_receipts t ON t.event_id=e.id AND t.binding_id=? AND t.status='completed'
               WHERE e.type='child_question' AND e.child_id IS NOT NULL AND r.status='completed'
                 AND json_extract(e.metadata_json,'$.conversation_ref')=?
                 AND json_extract(e.metadata_json,'$.thread_ref')=?
               ORDER BY e.created_at DESC LIMIT 1""",
            (binding_id, conversation_ref, thread_ref),
        ).fetchone()
    if not row:
        return None
    recovered = store.get_or_create(
        origin="slack",
        transport="slack",
        binding_id=binding_id,
        conversation_ref=conversation_ref,
        thread_ref=thread_ref,
        child_id=str(row["child_id"]),
    )
    output = jload(row["output_json"])
    store.append_message(
        str(recovered["id"]),
        role="user",
        content=str(row["text"]),
        kind="recovered_child_question_report",
        event_id=str(row["event_id"]),
    )
    store.append_message(
        str(recovered["id"]),
        role="assistant",
        content=format_learning_thread(
            {"status": "completed", "event_id": row["event_id"], "output": output}
        ),
        kind="recovered_learning_thread",
        event_id=str(row["event_id"]),
    )
    return recovered


def load_slack_tokens() -> dict[str, str]:
    """Read tokens from the process or the ignored owner-only setup file."""

    import os

    tokens = {
        "SLACK_APP_TOKEN": os.environ.get("SLACK_APP_TOKEN", ""),
        "SLACK_BOT_TOKEN": os.environ.get("SLACK_BOT_TOKEN", ""),
    }
    if all(tokens.values()):
        return tokens
    from ..config import private_root

    token_file = private_root() / "setup" / "slack.env"
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

    def retry_response(
        self,
        *,
        source_event_id: str,
        event_id: str,
        include_private_excerpts: bool = False,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def record_thread_response(
        self,
        *,
        binding_id: str,
        team_id: str,
        channel_id: str,
        thread_id: str,
        child_id: str,
        user_text: str,
        result: dict[str, Any],
    ) -> str: ...

    def chat(
        self,
        *,
        binding_id: str,
        team_id: str,
        channel_id: str,
        thread_id: str,
        text: str,
        include_private_excerpts: bool = False,
    ) -> dict[str, Any]: ...

    def handle_interaction_choice(
        self,
        *,
        resolved: dict[str, Any],
        binding_id: str,
        channel_id: str,
    ) -> dict[str, Any]: ...


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
        "• Learning threads include Helpful, Not for us, and Try another controls\n"
        "• Send any other note — choose a child or dismiss it with the buttons I show\n"
        "• `assign INBOX_ID CHILD_ID` — attach a saved question and answer it\n"
        "• `dismiss INBOX_ID` — discard a saved note\n"
        "• `inbox` — list saved, unassigned notes\n"
        "• `feedback CHILD_ID engaged: optional note` — tell me how it went\n"
        "• `privacy` — review the message and storage boundary\n"
        "• `help` — show this again\n\n"
        "Use DMs or explicitly mention me in a paired family channel."
    )


def _format_thread(result: dict[str, Any]) -> str:
    return format_learning_thread(result)


def _response_did_not_pass() -> str:
    return response_did_not_pass()


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
        blocks: list[dict[str, Any]] | None = None,
    ) -> str:
        outbound = OutboundMessage(
            channel_id=message.channel_id,
            text=text,
            thread_id=message.thread_id,
            blocks=blocks or [],
        )
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

        text = _command_text(message.text)
        pair = PAIR_RE.fullmatch(text)
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
        normalized = text.casefold()
        visual_job_id: str | None = None
        visual_purpose = "response_visual"
        reply_blocks: list[dict[str, Any]] = []
        artifact_to_deliver: dict[str, Any] | None = None
        session_id: str | None = None
        try:
            if pair:
                reply = "This conversation is already paired. Use `help` to see what I can do."
                purpose = "already-paired"
                result_status = "completed"
                event_id = None
                inbox_id = None
            elif normalized in {"connection", "connection test", "ping"} or HEALTH_RE.fullmatch(text):
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
                children = self.service.children() if rows else []
                reply = (
                    "Unassigned notes:\n"
                    + "\n".join(f"• `{row['id']}` — {row['text'][:180]}" for row in rows[:10])
                    + ("\n\nUse `assign INBOX_ID CHILD_ID` or `dismiss INBOX_ID`." if rows else "")
                    if rows
                    else "You have no unassigned notes."
                )
                reply_blocks = _inbox_list_blocks(rows, children) if rows else []
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
            elif match := RATE_RESPONSE_RE.fullmatch(text):
                source_event_id, rating = match.groups()
                delivered = delivered_slack_response(
                    self.db_path,
                    event_id=source_event_id,
                    binding_id=binding_id,
                )
                if not delivered or delivered["status"] != "completed" or not delivered["child_id"]:
                    raise ValueError("response not found for this parent")
                self.service.feedback(
                    {
                        "child_id": str(delivered["child_id"]),
                        "event_id": source_event_id,
                        "outcome": rating.casefold(),
                        "note": "Parent rated this Slack learning thread.",
                        "source": f"slack_response:{binding['parent_id']}",
                    }
                )
                reply = (
                    "Thanks — marked helpful."
                    if rating.casefold() == "helpful"
                    else "Thanks — marked not for us."
                )
                purpose = "response-feedback"
                result_status = "completed"
                event_id = None
                inbox_id = None
            elif match := RETRY_RESPONSE_RE.fullmatch(text):
                source_event_id = match.group(1)
                delivered = delivered_slack_response(
                    self.db_path,
                    event_id=source_event_id,
                    binding_id=binding_id,
                )
                if not delivered or not delivered["child_id"]:
                    raise ValueError("response not found for this parent")
                if delivered["status"] == "completed":
                    self.service.feedback(
                        {
                            "child_id": str(delivered["child_id"]),
                            "event_id": source_event_id,
                            "outcome": "not_helpful",
                            "note": "Parent requested a different response.",
                            "source": f"slack_response:{binding['parent_id']}",
                        }
                    )
                resource_mode = household_resource_context_mode(self.db_path)
                include_private_excerpts = resource_mode == "selected_excerpts"
                event_id = _engine_event_id(message, f"retry:{source_event_id}")
                response = self.service.retry_response(
                    source_event_id=source_event_id,
                    event_id=event_id,
                    include_private_excerpts=include_private_excerpts,
                    context_metadata=_episode_metadata(message),
                )
                if response.get("status") == "completed":
                    reply = _format_thread(response)
                    reply_blocks = _response_blocks(reply, event_id)
                    visual_job_id = response.get("visual_job_id")
                    purpose = "response-retry"
                    result_status = "completed"
                else:
                    reply = _response_did_not_pass()
                    reply_blocks = _response_blocks(reply, event_id, retry_only=True)
                    purpose = "response-retry-rejected"
                    result_status = "rejected"
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
                    if message.thread_id:
                        session_id = self.service.record_thread_response(
                            binding_id=binding_id,
                            team_id=message.team_id,
                            channel_id=message.channel_id,
                            thread_id=message.thread_id,
                            child_id=child_id,
                            user_text=question.strip(),
                            result=response,
                        )
                    reply_blocks = (
                        _semantic_response_blocks(
                            self.db_path,
                            binding_id=binding_id,
                            session_id=session_id,
                            event_id=event_id,
                            text=reply,
                        )
                        if session_id
                        else _response_blocks(reply, event_id)
                    )
                    visual_job_id = response.get("visual_job_id")
                    purpose = "answer"
                    result_status = "completed"
                else:
                    reply = _response_did_not_pass()
                    reply_blocks = _response_blocks(reply, event_id, retry_only=True)
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
                    if message.thread_id:
                        session_id = self.service.record_thread_response(
                            binding_id=binding_id,
                            team_id=message.team_id,
                            channel_id=message.channel_id,
                            thread_id=message.thread_id,
                            child_id=child_id,
                            user_text=str(row["text"]),
                            result=response,
                        )
                    reply_blocks = (
                        _semantic_response_blocks(
                            self.db_path,
                            binding_id=binding_id,
                            session_id=session_id,
                            event_id=event_id,
                            text=reply,
                        )
                        if session_id
                        else _response_blocks(reply, event_id)
                    )
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
            elif message.thread_id and _existing_thread_session(self.db_path, message, binding_id):
                resource_mode = household_resource_context_mode(self.db_path)
                response = self.service.chat(
                    binding_id=binding_id,
                    team_id=message.team_id,
                    channel_id=message.channel_id,
                    thread_id=message.thread_id,
                    text=text,
                    include_private_excerpts=resource_mode == "selected_excerpts",
                )
                reply = str(response.get("message") or "I followed that thread.")
                session_id = str(response["session_id"])
                event_id = response.get("event_id")
                visual_job_id = response.get("visual_job_id")
                artifact_to_deliver = response.get("artifact")
                if response.get("interaction"):
                    presented = create_interaction(
                        self.db_path,
                        binding_id=binding_id,
                        session_id=session_id,
                        plan=InteractionPlan.model_validate(response["interaction"]),
                    )
                    reply_blocks = [
                        *[
                            {"type": "section", "text": {"type": "mrkdwn", "text": section}}
                            for section in _mrkdwn_sections(reply)
                        ],
                        *interaction_blocks(presented),
                    ]
                else:
                    reply_blocks = [
                        {"type": "section", "text": {"type": "mrkdwn", "text": section}}
                        for section in _mrkdwn_sections(reply)
                    ]
                purpose = "parent-chat"
                result_status = "completed"
                inbox_id = None
            else:
                capture = create_unassigned_capture(self.db_path, message, str(binding["parent_id"]))
                inbox_id = str(capture["inbox_id"])
                children = self.service.children()
                reply = (
                    f"Saved as `{inbox_id}` without choosing a child.\n\n"
                    f"Reply `assign {inbox_id} CHILD_ID` to attach it and get a learning thread, or "
                    f"`dismiss {inbox_id}`. Use `children` to see IDs."
                )
                reply_blocks = _inbox_controls(inbox_id, children)
                purpose = "unassigned"
                result_status = "unassigned"
                event_id = None
            outbound_id = self._queue(
                message,
                binding_id,
                reply,
                purpose=purpose,
                blocks=reply_blocks,
            )
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
            if artifact_to_deliver:
                enqueue_artifact_delivery(
                    self.db_path,
                    artifact=artifact_to_deliver,
                    binding_id=binding_id,
                    depends_on_delivery_id=outbound_id,
                    channel_id=message.channel_id,
                    thread_id=message.thread_id,
                    idempotency_key=f"slack:{message.external_event_id}:{purpose}:artifact:{artifact_to_deliver['artifact_id']}",
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


def flush_slack_artifact_outbox(
    client: Any,
    db_path: str | Path,
    output_dir: str | Path,
    *,
    uploader: Callable[[str, bytes], None] | None = None,
) -> list[dict[str, str]]:
    """Upload reviewed printable PDFs after their explanatory Slack message."""

    results: list[dict[str, str]] = []
    allowed_root = (Path(output_dir).resolve() / "artifacts").resolve()
    upload = uploader or _upload_bytes
    for row in ready_artifact_deliveries(db_path):
        delivery_id = str(row["id"])
        if not claim_artifact_delivery(db_path, delivery_id):
            continue
        stage = "ticket_acquiring"
        try:
            path = Path(str(row["path"])).resolve()
            if not path.is_relative_to(allowed_root) or not path.is_file():
                raise ValueError("artifact path is outside the private artifact root")
            data = path.read_bytes()
            if len(data) != int(row["byte_count"]) or sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("artifact bytes changed after validation")
            if row["mime_type"] != "application/pdf" or not data.startswith(b"%PDF-"):
                raise ValueError("artifact is not a validated PDF")
            ticket = client.files_getUploadURLExternal(filename=row["filename"], length=len(data))
            upload_url = str(ticket.get("upload_url") or "")
            file_id = str(ticket.get("file_id") or "")
            if not upload_url.startswith("https://") or not file_id:
                raise RuntimeError("Slack did not return a valid artifact upload ticket")
            stage = "ticket_acquired"
            mark_artifact_delivery(db_path, delivery_id, status=stage, slack_file_id=file_id)
            stage = "uploading"
            mark_artifact_delivery(db_path, delivery_id, status=stage, slack_file_id=file_id)
            upload(upload_url, data)
            stage = "bytes_uploaded"
            mark_artifact_delivery(db_path, delivery_id, status=stage, slack_file_id=file_id)
            stage = "completing"
            mark_artifact_delivery(db_path, delivery_id, status=stage, slack_file_id=file_id)
            kwargs: dict[str, Any] = {
                "files": [{"id": file_id, "title": row["title"]}],
                "channel_id": row["channel_id"],
                "initial_comment": row["comment"],
            }
            if row.get("thread_id"):
                kwargs["thread_ts"] = row["thread_id"]
            response = client.files_completeUploadExternal(**kwargs)
            response_files = response.get("files") or []
            external_id = str(response_files[0].get("id") or file_id) if response_files else file_id
            mark_artifact_delivery(
                db_path,
                delivery_id,
                status="sent",
                slack_file_id=file_id,
                external_message_id=external_id,
            )
            results.append({"delivery_id": delivery_id, "status": "sent"})
        except Exception as exc:
            slack_rejection = exc.__class__.__name__ == "SlackApiError"
            status = "unknown" if stage == "completing" and not slack_rejection else "expired" if slack_rejection or isinstance(exc, ValueError) else "failed"
            mark_artifact_delivery(db_path, delivery_id, status=status, error=exc.__class__.__name__)
            results.append({"delivery_id": delivery_id, "status": status})
    return results


def _run_visual_worker(client: Any, db_path: str | Path, output_dir: str | Path, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            flush_slack_outbox(client, db_path)
            process_visual_jobs(db_path, output_dir)
            flush_slack_file_outbox(client, db_path, output_dir)
            flush_slack_artifact_outbox(client, db_path, output_dir)
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
        thread_id = str(event.get("thread_ts") or "") or None
        if not thread_id and event_type == "app_mention":
            # Keep public-family-channel conversations compact: the parent's
            # mention is the thread root and every bot artifact follows there.
            thread_id = str(event.get("ts") or event.get("event_ts") or "") or None
        external_event_id = str(body.get("event_id") or event.get("client_msg_id") or event.get("ts") or "")
        incoming = InboundMessage(
            external_event_id=external_event_id,
            team_id=str(body.get("team_id") or event.get("team") or ""),
            user_id=str(event.get("user") or ""),
            channel_id=str(event.get("channel") or ""),
            text=text,
            thread_id=thread_id,
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


def _make_slack_action_receiver(
    transport: SlackTransport,
    db_path: str | Path,
    *,
    action_id: str,
) -> Any:
    # Interactive actions must be acknowledged before any local or model work begins.
    def receive(ack: Any, body: dict[str, Any], action: dict[str, Any], client: Any) -> None:
        ack()
        try:
            block_match = INBOX_BLOCK_RE.fullmatch(str(action.get("block_id") or ""))
            if not block_match:
                raise ValueError("invalid inbox action context")
            inbox_id = block_match.group(1)
            selected_child_name: str | None = None
            if action_id == INBOX_DISMISS_ACTION:
                if str(action.get("value") or "") != inbox_id:
                    raise ValueError("invalid inbox action value")
                command = f"dismiss {inbox_id}"
            elif action_id == INBOX_ASSIGN_ACTION:
                selected = action.get("selected_option") or {}
                child_token = str(selected.get("value") or "")
                matches = [
                    child
                    for child in transport.service.children()
                    if child.get("id") and _child_action_token(str(child["id"])) == child_token
                ]
                if len(matches) != 1:
                    raise ValueError("child not found")
                command = f"assign {inbox_id} {matches[0]['id']}"
                selected_child_name = _plain_text(matches[0].get("name"), fallback="your child")
            else:
                raise ValueError("unsupported inbox action")

            team = body.get("team") or {}
            user = body.get("user") or {}
            channel = body.get("channel") or {}
            container = body.get("container") or {}
            source_message = body.get("message") or {}
            team_id = str(team.get("id") or body.get("team_id") or "")
            user_id = str(user.get("id") or body.get("user_id") or "")
            channel_id = str(channel.get("id") or container.get("channel_id") or "")
            message_ts = str(source_message.get("ts") or container.get("message_ts") or "")
            action_material = ":".join(
                (
                    str(body.get("trigger_id") or ""),
                    str(action.get("action_ts") or ""),
                    action_id,
                    inbox_id,
                    user_id,
                    channel_id,
                )
            )
            incoming = InboundMessage(
                external_event_id=f"act_slack_{sha256(action_material.encode()).hexdigest()[:24]}",
                team_id=team_id,
                user_id=user_id,
                channel_id=channel_id,
                text=command,
                thread_id=str(source_message.get("thread_ts"))
                if source_message.get("thread_ts")
                else None,
            )
            if action_id == INBOX_ASSIGN_ACTION and message_ts:
                working_text = f"Making a learning thread for {selected_child_name}…"
                try:
                    client.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text=working_text,
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f":hourglass_flowing_sand: {working_text}",
                                },
                            }
                        ],
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "could not show Slack inbox assignment progress"
                    )
            result = transport.handle(incoming)
            if result.status == "completed":
                if message_ts:
                    resolved_text = (
                        "Dismissed this saved note."
                        if action_id == INBOX_DISMISS_ACTION
                        else "Assigned this saved note. The learning thread follows."
                    )
                    try:
                        client.chat_update(
                            channel=channel_id,
                            ts=message_ts,
                            text=resolved_text,
                            blocks=[
                                {
                                    "type": "section",
                                    "text": {"type": "mrkdwn", "text": f":white_check_mark: {resolved_text}"},
                                }
                            ],
                        )
                    except Exception:
                        logging.getLogger(__name__).exception("could not retire Slack inbox controls")
            elif action_id == INBOX_ASSIGN_ACTION and message_ts:
                retry_text = "I couldn't finish that learning thread. The note is still saved."
                try:
                    client.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text=retry_text,
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f":warning: {retry_text} Choose a child to try again, or dismiss it.",
                                },
                            },
                            _inbox_controls(inbox_id, transport.service.children())[1],
                        ],
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "could not restore Slack inbox controls"
                    )
            flush_slack_outbox(client, db_path)
        except Exception:
            logging.getLogger(__name__).exception("Slack inbox action failed")

    return receive


def _make_slack_response_action_receiver(
    transport: SlackTransport,
    db_path: str | Path,
    *,
    action_id: str,
) -> Any:
    def receive(ack: Any, body: dict[str, Any], action: dict[str, Any], client: Any) -> None:
        ack()
        try:
            block_match = RESPONSE_BLOCK_RE.fullmatch(str(action.get("block_id") or ""))
            if not block_match:
                raise ValueError("invalid response action context")
            source_event_id = block_match.group(1)
            if str(action.get("value") or "") != source_event_id:
                raise ValueError("invalid response action value")
            if action_id == RESPONSE_HELPFUL_ACTION:
                command = f"rate_response {source_event_id} helpful"
            elif action_id == RESPONSE_NOT_HELPFUL_ACTION:
                command = f"rate_response {source_event_id} not_helpful"
            elif action_id == RESPONSE_RETRY_ACTION:
                command = f"retry_response {source_event_id}"
            else:
                raise ValueError("unsupported response action")

            team = body.get("team") or {}
            user = body.get("user") or {}
            channel = body.get("channel") or {}
            container = body.get("container") or {}
            source_message = body.get("message") or {}
            team_id = str(team.get("id") or body.get("team_id") or "")
            user_id = str(user.get("id") or body.get("user_id") or "")
            channel_id = str(channel.get("id") or container.get("channel_id") or "")
            action_material = ":".join(
                (
                    str(body.get("trigger_id") or ""),
                    str(action.get("action_ts") or ""),
                    action_id,
                    source_event_id,
                    user_id,
                    channel_id,
                )
            )
            incoming = InboundMessage(
                external_event_id=f"act_slack_{sha256(action_material.encode()).hexdigest()[:24]}",
                team_id=team_id,
                user_id=user_id,
                channel_id=channel_id,
                text=command,
                thread_id=str(source_message.get("thread_ts") or source_message.get("ts") or "") or None,
            )
            message_ts = str(source_message.get("ts") or container.get("message_ts") or "")
            if action_id == RESPONSE_RETRY_ACTION and message_ts:
                original_blocks = [
                    block
                    for block in (source_message.get("blocks") or [])
                    if str(block.get("block_id") or "")
                    != f"curiosity_response:{source_event_id}"
                ]
                try:
                    client.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text=str(source_message.get("text") or "Trying a different approach…"),
                        blocks=[
                            *original_blocks,
                            {
                                "type": "context",
                                "elements": [
                                    {
                                        "type": "mrkdwn",
                                        "text": ":hourglass_flowing_sand: Trying a different approach…",
                                    }
                                ],
                            },
                        ],
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "could not show Slack response retry progress"
                    )
            result = transport.handle(incoming)
            if result.status == "completed":
                if message_ts:
                    if action_id == RESPONSE_HELPFUL_ACTION:
                        resolved_text = ":white_check_mark: Marked helpful."
                    elif action_id == RESPONSE_NOT_HELPFUL_ACTION:
                        resolved_text = ":white_check_mark: Marked not for us."
                    else:
                        resolved_text = ":sparkles: Another approach is in the thread."
                    original_blocks = [
                        block
                        for block in (source_message.get("blocks") or [])
                        if str(block.get("block_id") or "")
                        != f"curiosity_response:{source_event_id}"
                    ]
                    updated_blocks = [
                        *original_blocks,
                        {"type": "context", "elements": [{"type": "mrkdwn", "text": resolved_text}]},
                    ]
                    if action_id == RESPONSE_NOT_HELPFUL_ACTION:
                        updated_blocks.extend(_response_blocks("", source_event_id, retry_only=True))
                    try:
                        client.chat_update(
                            channel=channel_id,
                            ts=message_ts,
                            text=str(source_message.get("text") or result.message),
                            blocks=updated_blocks,
                        )
                    except Exception:
                        logging.getLogger(__name__).exception("could not retire Slack response controls")
            elif action_id == RESPONSE_RETRY_ACTION and message_ts:
                try:
                    client.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text=str(source_message.get("text") or result.message),
                        blocks=source_message.get("blocks")
                        or _response_blocks("", source_event_id, retry_only=True),
                    )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "could not restore Slack response controls"
                    )
            flush_slack_outbox(client, db_path)
        except Exception:
            logging.getLogger(__name__).exception("Slack response action failed")

    return receive


def _make_semantic_interaction_receiver(
    transport: SlackTransport,
    db_path: str | Path,
) -> Any:
    def receive(ack: Any, body: dict[str, Any], action: dict[str, Any], client: Any) -> None:
        ack()
        try:
            block_id = str(action.get("block_id") or "")
            match = re.fullmatch(r"curiosity_interaction:(ix_[a-f0-9]{20})", block_id)
            if not match:
                raise ValueError("invalid semantic interaction context")
            interaction_id = match.group(1)
            selected = action.get("selected_option") or {}
            token = str(selected.get("value") or action.get("value") or "")
            team = body.get("team") or {}
            user = body.get("user") or {}
            channel = body.get("channel") or {}
            container = body.get("container") or {}
            source_message = body.get("message") or {}
            team_id = str(team.get("id") or body.get("team_id") or "")
            user_id = str(user.get("id") or body.get("user_id") or "")
            channel_id = str(channel.get("id") or container.get("channel_id") or "")
            thread_id = str(source_message.get("thread_ts") or source_message.get("ts") or "") or None
            external_event_id = "ix_slack_" + sha256(
                ":".join(
                    (
                        str(body.get("trigger_id") or ""),
                        str(action.get("action_ts") or ""),
                        interaction_id,
                        user_id,
                        channel_id,
                    )
                ).encode()
            ).hexdigest()[:24]
            incoming = InboundMessage(
                external_event_id=external_event_id,
                team_id=team_id,
                user_id=user_id,
                channel_id=channel_id,
                text="semantic interaction",
                thread_id=thread_id,
            )
            binding = active_binding(db_path, incoming)
            if not binding:
                raise ValueError("paired parent binding not found")
            resolved = resolve_interaction(
                db_path,
                InteractionEvent(
                    interaction_id=interaction_id,
                    option_token=token,
                    team_id=team_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    thread_id=thread_id,
                    external_event_id=external_event_id,
                ),
                binding_id=str(binding["id"]),
                parent_id=str(binding["parent_id"]),
            )
            if resolved.get("duplicate"):
                return
            result = transport.service.handle_interaction_choice(
                resolved=resolved,
                binding_id=str(binding["id"]),
                channel_id=channel_id,
            )
            reply = str(result.get("message") or "Done.")
            blocks: list[dict[str, Any]] = [
                {"type": "section", "text": {"type": "mrkdwn", "text": section}}
                for section in _mrkdwn_sections(reply)
            ]
            if result.get("interaction"):
                presented = create_interaction(
                    db_path,
                    binding_id=str(binding["id"]),
                    session_id=resolved.get("session_id"),
                    plan=InteractionPlan.model_validate(result["interaction"]),
                )
                blocks.extend(interaction_blocks(presented))
            outbound_id = transport._queue(
                incoming,
                str(binding["id"]),
                reply,
                purpose=f"interaction:{interaction_id}",
                blocks=blocks,
            )
            if resolved.get("session_id") and reply:
                SessionStore(db_path).append_message(
                    str(resolved["session_id"]),
                    role="assistant",
                    content=reply,
                    kind="interaction_result",
                    event_id=result.get("event_id"),
                )
            if result.get("visual_job_id"):
                enqueue_visual_delivery(
                    db_path,
                    visual_job_id=str(result["visual_job_id"]),
                    binding_id=str(binding["id"]),
                    depends_on_delivery_id=outbound_id,
                    channel_id=channel_id,
                    thread_id=thread_id,
                    idempotency_key=f"slack:{external_event_id}:visual",
                    purpose="response_visual",
                )
            if result.get("artifact"):
                enqueue_artifact_delivery(
                    db_path,
                    artifact=result["artifact"],
                    binding_id=str(binding["id"]),
                    depends_on_delivery_id=outbound_id,
                    channel_id=channel_id,
                    thread_id=thread_id,
                    idempotency_key=f"slack:{external_event_id}:artifact:{result['artifact']['artifact_id']}",
                )
            message_ts = str(source_message.get("ts") or container.get("message_ts") or "")
            if message_ts:
                original_blocks = [
                    block for block in source_message.get("blocks") or [] if block.get("block_id") != block_id
                ]
                client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=str(source_message.get("text") or reply),
                    blocks=[
                        *original_blocks,
                        {"type": "context", "elements": [{"type": "mrkdwn", "text": ":white_check_mark: Choice received."}]},
                    ],
                )
            flush_slack_outbox(client, db_path)
        except Exception:
            logging.getLogger(__name__).exception("Slack semantic interaction failed")

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
    app.action(INBOX_ASSIGN_ACTION)(
        _make_slack_action_receiver(transport, db_path, action_id=INBOX_ASSIGN_ACTION)
    )
    app.action(INBOX_DISMISS_ACTION)(
        _make_slack_action_receiver(transport, db_path, action_id=INBOX_DISMISS_ACTION)
    )
    for response_action in (
        RESPONSE_HELPFUL_ACTION,
        RESPONSE_NOT_HELPFUL_ACTION,
        RESPONSE_RETRY_ACTION,
    ):
        app.action(response_action)(
            _make_slack_response_action_receiver(
                transport,
                db_path,
                action_id=response_action,
            )
        )
    semantic_receiver = _make_semantic_interaction_receiver(transport, db_path)
    app.action(re.compile(rf"^{re.escape(SLACK_INTERACTION_BUTTON)}_[0-4]$"))(semantic_receiver)
    app.action(SLACK_INTERACTION_SELECT)(semantic_receiver)
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
