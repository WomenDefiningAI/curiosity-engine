from __future__ import annotations

import json
import logging
import re
import stat
import threading
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError

from ..artifact_delivery import (
    claim_artifact_delivery,
    enqueue_artifact_bundle,
    mark_artifact_delivery,
    ready_artifact_deliveries,
)
from ..attachments import (
    attachments_for_inbox,
    discard_inbox_attachments,
    link_event_attachments,
    persist_inbound_attachments,
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
from ..presentation import format_reviewed_response, response_did_not_pass
from ..service import CuriosityService
from ..sessions import SessionStore
from ..visuals import create_synthetic_visual_job, process_visual_jobs
from .contracts import MAX_INBOUND_ATTACHMENTS, InboundAttachment, InboundMessage, OutboundMessage, TransportResult

PAIR_RE = re.compile(r"^pair\s+([A-Z2-9]{8})$", re.IGNORECASE)
ASK_RE = re.compile(r"^ask\s+([A-Za-z0-9_-]{1,120})\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
ASSIGN_RE = re.compile(r"^assign\s+(inbox_[A-Za-z0-9_-]+)\s+([A-Za-z0-9_-]{1,120})$", re.IGNORECASE)
DISMISS_RE = re.compile(r"^dismiss\s+(inbox_[A-Za-z0-9_-]+)$", re.IGNORECASE)
RATE_RESPONSE_RE = re.compile(r"^rate_response\s+([A-Za-z0-9_-]{1,120})\s+(helpful|not_helpful)$", re.IGNORECASE)
RETRY_RESPONSE_RE = re.compile(r"^retry_response\s+([A-Za-z0-9_-]{1,120})$", re.IGNORECASE)
CHALLENGE_RESPONSE_RE = re.compile(r"^challenge_response\s+([A-Za-z0-9_-]{1,120})$", re.IGNORECASE)
INBOX_BLOCK_RE = re.compile(r"^curiosity_inbox:(inbox_[A-Za-z0-9_-]+)$")
RESPONSE_BLOCK_RE = re.compile(r"^curiosity_response:([A-Za-z0-9_-]{1,120})$")
HEALTH_RE = re.compile(
    r"^(?:(?:are\s+you|is\s+(?:this|it))\s+)?(?:still\s+)?(?:working|online|connected)\??$",
    re.IGNORECASE,
)
ADD_CHILD_RE = re.compile(
    r"^(?:(?:please\s+)?(?:add|create|set up)\s+(?:a|another|new)\s+child(?:\s+profile)?|"
    r"we\s+have\s+another\s+child|(?:(?:how\s+(?:can|do)\s+i|can\s+i|i\s+(?:want|need)\s+to)\s+)"
    r"(?:add|create|set up)\s+(?:a|another|new)\s+child(?:\s+profile)?)[.!?]?$",
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
RESPONSE_CHALLENGE_ACTION = "curiosity_response_challenge"
CHILD_ADD_OPEN_ACTION = "curiosity_child_add_open"
CHILD_ADD_MODAL_CALLBACK = "curiosity_child_add_modal"
PROCESSING_REACTION = "eyes"
PROCESSING_COMPLETE_REACTION = "white_check_mark"

MAX_INBOUND_IMAGE_BYTES = 10_000_000
INBOUND_IMAGE_TYPES = {
    "image/jpeg": ("JPEG", ".jpg"),
    "image/png": ("PNG", ".png"),
    "image/webp": ("WEBP", ".webp"),
}


def _command_text(text: str) -> str:
    """Accept Slack inline-code formatting without changing saved note text."""

    value = text.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1].strip()
    return value


def _named_children(text: str, children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve only strong, explicit child-name signals from a natural parent message.

    Identity attribution stays deterministic. A bare name-like word is not enough:
    the name must lead the message, follow a clear family-context cue, or act as the
    subject of a child-reporting phrase. Multiple matches remain ambiguous.
    """

    matched: list[dict[str, Any]] = []
    for child in children:
        name = " ".join(str(child.get("name") or "").split())
        if not child.get("id") or not name:
            continue
        first_name = name.split()[0]
        aliases = {name}
        if len(first_name) >= 2:
            aliases.add(first_name)
        found = False
        for alias in sorted(aliases, key=len, reverse=True):
            token = re.escape(alias).replace(r"\ ", r"\s+")
            boundary = rf"(?<![\w]){token}(?![\w])"
            if " " in alias and re.search(boundary, text, re.IGNORECASE):
                found = True
            elif re.search(
                rf"^\s*{boundary}(?=\s*(?:[:,—–-]|['’]s\b|\b(?:asked|asks|said|says|keeps|wants|wanted|wonders|wondered|is|was|has|had|loves|likes|made|built|drew|found|noticed|brought|tried|thinks|thought|what|why|how|can|could|would|does|do)\b))",
                text,
                re.IGNORECASE,
            ):
                found = True
            elif re.search(
                rf"\b(?:for|with|about|from|show|tell|help|teach)\s+{boundary}",
                text,
                re.IGNORECASE,
            ):
                found = True
            elif re.search(
                rf"{boundary}(?:['’]s)?\s+(?:asked|asks|said|says|keeps|wants|wanted|wonders|wondered|loves|likes|made|built|drew|found|noticed|brought|tried|thinks|thought)\b",
                text,
                re.IGNORECASE,
            ):
                found = True
            if found:
                matched.append(child)
                break
    return matched


def _message_child(text: str, children: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve a named child, or the sole configured child when there is no ambiguity."""

    named = _named_children(text, children)
    if len(named) == 1:
        return named[0]
    configured = [child for child in children if child.get("id")]
    if len(configured) == 1:
        return configured[0]
    return None


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


def _inbox_list_blocks(rows: list[dict[str, Any]], children: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _child_management_blocks(children: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if children:
        summary = "\n".join(
            f"• *{_mrkdwn_text(child.get('name') or 'Child', limit=80)}*"
            + (f" — {_mrkdwn_text(child.get('grade'), limit=40)}" if child.get("grade") else "")
            for child in children[:20]
        )
        text = f"*Children*\n{summary}"
    else:
        text = "*Children*\nNo child profiles yet."
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text[:2_900]}},
        {
            "type": "actions",
            "block_id": "curiosity_children",
            "elements": [
                {
                    "type": "button",
                    "action_id": CHILD_ADD_OPEN_ACTION,
                    "text": {"type": "plain_text", "text": "➕ Add a child", "emoji": True},
                    "value": "open",
                    "style": "primary",
                }
            ],
        },
    ]


def _child_add_modal(
    *, binding_id: str, team_id: str, user_id: str, channel_id: str, thread_id: str | None
) -> dict[str, Any]:
    metadata = json.dumps(
        {
            "binding_id": binding_id,
            "team_id": team_id,
            "user_id": user_id,
            "channel_id": channel_id,
            "thread_id": thread_id,
        },
        separators=(",", ":"),
    )
    return {
        "type": "modal",
        "callback_id": CHILD_ADD_MODAL_CALLBACK,
        "private_metadata": metadata,
        "title": {"type": "plain_text", "text": "Add a child"},
        "submit": {"type": "plain_text", "text": "Add child"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "This creates an empty private profile on the computer running Curiosity Engine. "
                        "It does not infer interests, ability, or mastery."
                    ),
                },
            },
            {
                "type": "input",
                "block_id": "child_name",
                "label": {"type": "plain_text", "text": "Name"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "placeholder": {"type": "plain_text", "text": "The name your family uses"},
                    "max_length": 80,
                },
            },
            {
                "type": "input",
                "block_id": "child_grade",
                "optional": True,
                "label": {"type": "plain_text", "text": "Grade or learning level (optional)"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "placeholder": {"type": "plain_text", "text": "For example: 1st"},
                    "max_length": 40,
                },
            },
            {
                "type": "input",
                "block_id": "child_birth_year",
                "optional": True,
                "label": {"type": "plain_text", "text": "Birth year (optional)"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "value",
                    "placeholder": {"type": "plain_text", "text": "YYYY"},
                    "max_length": 4,
                },
            },
        ],
    }


def _mrkdwn_sections(text: str, *, limit: int = 2_900) -> list[str]:
    # Model providers commonly return standard Markdown links, while Slack's
    # mrkdwn renderer requires <url|label>. Normalize at the transport boundary
    # so every reply path (new answer, retry, and interaction result) renders the
    # same way without teaching each model Slack-specific syntax.
    text = re.sub(
        r"\[([^\]\n]{1,200})\]\((https?://[^\s<>)]+)\)",
        lambda match: (
            "<"
            + match.group(2).replace("|", "%7C")
            + "|"
            + re.sub(r"[<>|\x00-\x1f\x7f]", "", match.group(1)).strip()
            + ">"
        ),
        text,
    )
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


def _response_action_block(
    event_id: str,
    *,
    include_feedback: bool = True,
    include_retry: bool = True,
    include_challenge: bool = True,
) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    if include_feedback:
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
    if include_retry:
        elements.append(
            {
                "type": "button",
                "action_id": RESPONSE_RETRY_ACTION,
                "text": {"type": "plain_text", "text": "✨ Try another", "emoji": True},
                "value": event_id,
            }
        )
    if include_challenge:
        elements.append(
            {
                "type": "button",
                "action_id": RESPONSE_CHALLENGE_ACTION,
                "text": {"type": "plain_text", "text": "🕵️ Make a challenge", "emoji": True},
                "value": event_id,
            }
        )
    return {
        "type": "actions",
        "block_id": f"curiosity_response:{event_id}",
        "elements": elements,
    }


def _response_blocks(text: str, event_id: str, *, retry_only: bool = False) -> list[dict[str, Any]]:
    return [
        *[{"type": "section", "text": {"type": "mrkdwn", "text": section}} for section in _mrkdwn_sections(text)],
        _response_action_block(
            event_id,
            include_feedback=not retry_only,
            include_challenge=not retry_only,
        ),
    ]


def _remaining_response_action_block(
    blocks: list[dict[str, Any]],
    event_id: str,
    consumed_action_ids: set[str],
) -> dict[str, Any] | None:
    """Keep only still-visible controls and never resurrect a completed action."""

    allowed = {
        RESPONSE_HELPFUL_ACTION,
        RESPONSE_NOT_HELPFUL_ACTION,
        RESPONSE_RETRY_ACTION,
        RESPONSE_CHALLENGE_ACTION,
    }
    for block in blocks:
        if str(block.get("block_id") or "") != f"curiosity_response:{event_id}":
            continue
        elements = [
            element
            for element in (block.get("elements") or [])
            if str(element.get("action_id") or "") in allowed
            and str(element.get("action_id") or "") not in consumed_action_ids
            and str(element.get("value") or "") == event_id
        ]
        return {**block, "elements": elements} if elements else None
    return None


def _response_action_receipt_id(
    action_id: str,
    *,
    team_id: str,
    user_id: str,
    channel_id: str,
    event_id: str,
) -> str:
    action_group = (
        "rating"
        if action_id in {RESPONSE_HELPFUL_ACTION, RESPONSE_NOT_HELPFUL_ACTION}
        else "retry"
        if action_id == RESPONSE_RETRY_ACTION
        else "challenge"
    )
    material = ":".join((team_id, user_id, channel_id, event_id, action_group))
    return f"act_slack_response_{sha256(material.encode()).hexdigest()[:24]}"


def _consumed_response_action_ids(
    db_path: str | Path,
    *,
    team_id: str,
    user_id: str,
    channel_id: str,
    event_id: str,
) -> set[str]:
    receipt_ids = {
        group: _response_action_receipt_id(
            action_id,
            team_id=team_id,
            user_id=user_id,
            channel_id=channel_id,
            event_id=event_id,
        )
        for group, action_id in (
            ("rating", RESPONSE_HELPFUL_ACTION),
            ("retry", RESPONSE_RETRY_ACTION),
            ("challenge", RESPONSE_CHALLENGE_ACTION),
        )
    }
    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT external_event_id FROM transport_receipts
               WHERE transport='slack' AND status IN ('received','completed')
                 AND external_event_id IN (?,?,?)""",
            tuple(receipt_ids.values()),
        ).fetchall()
    consumed_groups = {
        group
        for group, receipt_id in receipt_ids.items()
        if any(str(row["external_event_id"]) == receipt_id for row in rows)
    }
    consumed: set[str] = set()
    if "rating" in consumed_groups:
        consumed.update({RESPONSE_HELPFUL_ACTION, RESPONSE_NOT_HELPFUL_ACTION})
    if "retry" in consumed_groups:
        consumed.add(RESPONSE_RETRY_ACTION)
    if "challenge" in consumed_groups:
        consumed.add(RESPONSE_CHALLENGE_ACTION)
    return consumed


def _bound_response_session_id(
    db_path: str | Path,
    *,
    event_id: str,
    binding_id: str,
) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT s.id FROM agent_messages m
               JOIN agent_sessions s ON s.id=m.session_id
               WHERE m.event_id=? AND s.binding_id=?
               ORDER BY m.ordinal DESC LIMIT 1""",
            (event_id, binding_id),
        ).fetchone()
    return str(row["id"]) if row else None


def _response_or_semantic_blocks(
    db_path: str | Path,
    *,
    binding_id: str,
    session_id: str,
    event_id: str,
    text: str,
    interaction: InteractionPlan | dict[str, Any],
) -> list[dict[str, Any]]:
    plan = interaction if isinstance(interaction, InteractionPlan) else InteractionPlan.model_validate(interaction)
    if plan.kind == "rate_output":
        return _response_blocks(text, event_id)
    presented = create_interaction(
        db_path,
        binding_id=binding_id,
        session_id=session_id,
        plan=plan,
    )
    return [
        *[{"type": "section", "text": {"type": "mrkdwn", "text": section}} for section in _mrkdwn_sections(text)],
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
        content=format_reviewed_response({"status": "completed", "event_id": row["event_id"], "output": output}),
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


def _download_slack_bytes(url: str, token: str, *, max_bytes: int = MAX_INBOUND_IMAGE_BYTES) -> bytes:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or not (hostname == "slack.com" or hostname.endswith(".slack.com")):
        raise ValueError("Slack attachment URL is outside the Slack file host")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - exact Slack hosts only
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Slack attachment exceeds the private image limit")
    return data


def _validate_image_bytes(data: bytes, media_type: str) -> None:
    expected_format = INBOUND_IMAGE_TYPES[media_type][0]
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != expected_format:
                raise ValueError("Slack attachment bytes do not match the declared image type")
            width, height = image.size
            if width < 1 or height < 1 or width * height > 30_000_000:
                raise ValueError("Slack image dimensions are outside the supported limit")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Slack attachment is not a readable image") from exc


def _receive_slack_images(
    transport: SlackTransport,
    event: dict[str, Any],
    client: Any,
    *,
    downloader: Callable[[str, str], bytes] | None = None,
) -> list[InboundAttachment]:
    """Download only paired, explicitly delivered image files into private storage."""

    files = event.get("files") or []
    if not isinstance(files, list):
        return []
    token = str(getattr(client, "token", "") or load_slack_tokens().get("SLACK_BOT_TOKEN") or "")
    inbound_root = (transport.output_dir / "inbound").resolve()
    inbound_root.mkdir(parents=True, exist_ok=True)
    inbound_root.chmod(0o700)
    receive_bytes = downloader or (lambda url, bearer: _download_slack_bytes(url, bearer))
    received: list[InboundAttachment] = []
    for raw in files[:MAX_INBOUND_ATTACHMENTS]:
        if not isinstance(raw, dict):
            continue
        external_id = str(raw.get("id") or "")
        if not external_id:
            continue
        ref_hash = sha256(external_id.encode()).hexdigest()
        media_type = str(raw.get("mimetype") or "").casefold()
        if media_type not in INBOUND_IMAGE_TYPES:
            continue
        declared_size = int(raw.get("size") or 0)
        if declared_size < 0 or declared_size > MAX_INBOUND_IMAGE_BYTES:
            received.append(InboundAttachment(external_ref_hash=ref_hash, status="unavailable"))
            continue
        try:
            url = str(raw.get("url_private_download") or raw.get("url_private") or "")
            if not url:
                detail = client.files_info(file=external_id)
                info = detail.get("file") or {}
                url = str(info.get("url_private_download") or info.get("url_private") or "")
            if not token or not url:
                raise PermissionError("Slack file access is not configured")
            data = receive_bytes(url, token)
            _validate_image_bytes(data, media_type)
            digest = sha256(data).hexdigest()
            path = inbound_root / f"{digest}{INBOUND_IMAGE_TYPES[media_type][1]}"
            if not path.exists():
                path.write_bytes(data)
                path.chmod(0o600)
            received.append(
                InboundAttachment(
                    external_ref_hash=ref_hash,
                    status="ready",
                    media_type=media_type,
                    byte_count=len(data),
                    sha256=digest,
                    private_path=str(path),
                )
            )
        except Exception as exc:
            logging.getLogger(__name__).warning("Slack image attachment was unavailable (%s)", exc.__class__.__name__)
            received.append(InboundAttachment(external_ref_hash=ref_hash, status="unavailable"))
    return received


class CuriosityServiceLike(Protocol):
    def children(self) -> list[dict[str, Any]]: ...

    def add_child_from_binding(
        self,
        *,
        binding_id: str,
        request_id: str,
        name: str,
        birth_year: int | None = None,
        grade: str | None = None,
    ) -> dict[str, Any]: ...

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
        attachments: list[dict[str, Any]] | None = None,
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
        attachment_ids: list[str] | None = None,
        attachment_context: list[dict[str, Any]] | None = None,
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
        child_id: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        input_mode: str = "conversation",
        answer_mode_hint: str = "auto",
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
        "• `children` — show and manage child profiles\n"
        "• Say `add a child` — open the private owner-confirmed child setup form\n"
        "• `ask CHILD_ID: Why does ice float?` — get a response now\n"
        "• Start a new message with one configured child's name — attribute it without a chooser\n"
        "• Start a fresh session with a new top-level mention; mention me in its thread to continue it\n"
        "• Attach a photo to a mention or DM — save it as private thread context after choosing a child\n"
        "• Learning threads include Helpful, Not for us, Try another, and Make a challenge controls\n"
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
    return format_reviewed_response(result)


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
        parent_memory_refs: list[dict[str, Any]] | None = None,
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
            parent_memory_refs=parent_memory_refs,
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
        response: dict[str, Any] | None = None
        try:
            event_attachments = persist_inbound_attachments(
                self.db_path,
                self.output_dir,
                binding_id=binding_id,
                message=message,
            )
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
            elif ADD_CHILD_RE.fullmatch(text):
                reply = (
                    "I can add a child with an owner-confirmed private form. Their profile starts empty—no inferred "
                    "interests, ability, or mastery."
                )
                reply_blocks = [
                    {"type": "section", "text": {"type": "mrkdwn", "text": reply}},
                    _child_management_blocks([])[1],
                ]
                purpose = "add-child"
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
                reply_blocks = _child_management_blocks(children)
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
                    "model provider processes only the bounded context selected for that request. A photo explicitly "
                    "sent to the bot is inspected once by the configured vision provider; the private bytes and a "
                    "bounded observation are then kept locally for thread follow-ups. Deterministic "
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
                reply = "Thanks — marked helpful." if rating.casefold() == "helpful" else "Thanks — marked not for us."
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
                source_session_id = _bound_response_session_id(
                    self.db_path,
                    event_id=source_event_id,
                    binding_id=binding_id,
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
                    reply_blocks = [
                        *[
                            {"type": "section", "text": {"type": "mrkdwn", "text": section}}
                            for section in _mrkdwn_sections(reply)
                        ],
                        _response_action_block(
                            event_id,
                            include_challenge=source_session_id is not None,
                        ),
                    ]
                    if source_session_id:
                        SessionStore(self.db_path).append_message(
                            source_session_id,
                            role="assistant",
                            content=reply,
                            kind="learning_thread",
                            event_id=event_id,
                        )
                    visual_job_id = response.get("visual_job_id")
                    artifact_to_deliver = response.get("artifact")
                    purpose = "response-retry"
                    result_status = "completed"
                else:
                    reply = _response_did_not_pass()
                    reply_blocks = _response_blocks(reply, event_id, retry_only=True)
                    purpose = "response-retry-rejected"
                    result_status = "rejected"
                inbox_id = None
            elif match := CHALLENGE_RESPONSE_RE.fullmatch(text):
                source_event_id = match.group(1)
                delivered = delivered_slack_response(
                    self.db_path,
                    event_id=source_event_id,
                    binding_id=binding_id,
                )
                if not delivered or delivered["status"] != "completed" or not delivered["child_id"]:
                    raise ValueError("response not found for this parent")
                source_session_id = _bound_response_session_id(
                    self.db_path,
                    event_id=source_event_id,
                    binding_id=binding_id,
                )
                if not source_session_id:
                    raise ValueError("response thread not found for this parent")
                challenge = self.service.handle_interaction_choice(
                    resolved={
                        "intent": "create_artifact",
                        "payload": {"event_id": source_event_id, "artifact_type": "challenge"},
                        "session_id": source_session_id,
                    },
                    binding_id=binding_id,
                    channel_id=message.channel_id,
                )
                reply = str(challenge.get("message") or "I made a challenge. The printable will follow here.")
                artifact_to_deliver = challenge.get("artifact")
                purpose = "response-challenge"
                result_status = str(challenge.get("status") or "completed")
                event_id = source_event_id
                inbox_id = None
            elif match := ASK_RE.fullmatch(text):
                resource_mode = household_resource_context_mode(self.db_path)
                include_private_excerpts = resource_mode == "selected_excerpts"
                child_id, question = match.groups()
                response = self.service.chat(
                    binding_id=binding_id,
                    team_id=message.team_id,
                    channel_id=message.channel_id,
                    thread_id=message.thread_id or f"root:{message.external_event_id}",
                    child_id=child_id,
                    text=question.strip(),
                    include_private_excerpts=include_private_excerpts,
                    attachments=event_attachments,
                    answer_mode_hint="explore",
                )
                event_id = response.get("event_id")
                if response.get("status") == "completed":
                    reply = str(response.get("message") or "I couldn't make a reliable learning response.")
                    session_id = str(response["session_id"])
                    link_event_attachments(
                        self.db_path,
                        binding_id=binding_id,
                        external_event_id=message.external_event_id,
                        session_id=session_id,
                    )
                    if response.get("interaction"):
                        reply_blocks = _response_or_semantic_blocks(
                            self.db_path,
                            binding_id=binding_id,
                            session_id=session_id,
                            event_id=str(event_id),
                            text=reply,
                            interaction=response["interaction"],
                        )
                    else:
                        reply_blocks = [
                            {"type": "section", "text": {"type": "mrkdwn", "text": section}}
                            for section in _mrkdwn_sections(reply)
                        ]
                    visual_job_id = response.get("visual_job_id")
                    artifact_to_deliver = response.get("artifact")
                    purpose = "answer"
                    result_status = "completed"
                else:
                    reply = _response_did_not_pass()
                    reply_blocks = _response_blocks(reply, event_id, retry_only=True)
                    purpose = "answer_rejected"
                    result_status = "rejected"
                inbox_id = None
            elif (
                not _existing_thread_session(self.db_path, message, binding_id)
                and ASSIGN_RE.fullmatch(text) is None
                and DISMISS_RE.fullmatch(text) is None
                and (message_child := _message_child(text, self.service.children())) is not None
            ):
                child_id = str(message_child["id"])
                resource_mode = household_resource_context_mode(self.db_path)
                include_private_excerpts = resource_mode == "selected_excerpts"
                # Attribution is deterministic; intent and response scale are semantic.
                response = self.service.chat(
                    binding_id=binding_id,
                    team_id=message.team_id,
                    channel_id=message.channel_id,
                    thread_id=message.thread_id or f"root:{message.external_event_id}",
                    text=text,
                    include_private_excerpts=include_private_excerpts,
                    child_id=child_id,
                    attachments=event_attachments,
                    answer_mode_hint="auto",
                )
                reply = str(response.get("message") or "I saved that context for this thread.")
                session_id = str(response["session_id"])
                event_id = response.get("event_id")
                visual_job_id = response.get("visual_job_id")
                artifact_to_deliver = response.get("artifact")
                link_event_attachments(
                    self.db_path,
                    binding_id=binding_id,
                    external_event_id=message.external_event_id,
                    session_id=session_id,
                )
                if response.get("interaction"):
                    reply_blocks = _response_or_semantic_blocks(
                        self.db_path,
                        binding_id=binding_id,
                        session_id=session_id,
                        event_id=str(event_id),
                        text=reply,
                        interaction=response["interaction"],
                    )
                else:
                    reply_blocks = [
                        {"type": "section", "text": {"type": "mrkdwn", "text": section}}
                        for section in _mrkdwn_sections(reply)
                    ]
                purpose = "named-parent-chat"
                result_status = "rejected" if response.get("status") in {"rejected", "failed"} else "completed"
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
                captured_attachments = attachments_for_inbox(self.db_path, inbox_id)
                response = self.service.chat(
                    binding_id=binding_id,
                    team_id=message.team_id,
                    channel_id=message.channel_id,
                    thread_id=message.thread_id or f"root:{message.external_event_id}",
                    text=str(row["text"]),
                    include_private_excerpts=include_private_excerpts,
                    child_id=child_id,
                    attachments=captured_attachments,
                    answer_mode_hint="auto",
                )
                event_id = response.get("event_id")
                if response.get("status") == "completed":
                    resolve_inbox(self.db_path, inbox_id, child_id=child_id)
                    reply = str(response.get("message") or "I saved that context for this thread.")
                    session_id = str(response["session_id"])
                    if response.get("interaction"):
                        reply_blocks = _response_or_semantic_blocks(
                            self.db_path,
                            binding_id=binding_id,
                            session_id=session_id,
                            event_id=str(event_id),
                            text=reply,
                            interaction=response["interaction"],
                        )
                    else:
                        reply_blocks = [
                            {"type": "section", "text": {"type": "mrkdwn", "text": section}}
                            for section in _mrkdwn_sections(reply)
                        ]
                    visual_job_id = response.get("visual_job_id")
                    artifact_to_deliver = response.get("artifact")
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
                discard_inbox_attachments(self.db_path, inbox_id)
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
                    attachments=event_attachments,
                )
                reply = str(
                    response.get("message")
                    or "I couldn't make a reliable change, so I kept the last reviewed answer unchanged."
                )
                session_id = str(response["session_id"])
                event_id = response.get("event_id")
                visual_job_id = response.get("visual_job_id")
                artifact_to_deliver = response.get("artifact")
                if response.get("interaction"):
                    reply_blocks = _response_or_semantic_blocks(
                        self.db_path,
                        binding_id=binding_id,
                        session_id=session_id,
                        event_id=str(event_id),
                        text=reply,
                        interaction=response["interaction"],
                    )
                else:
                    reply_blocks = [
                        {"type": "section", "text": {"type": "mrkdwn", "text": section}}
                        for section in _mrkdwn_sections(reply)
                    ]
                result_status = "rejected" if response.get("status") in {"rejected", "failed"} else "completed"
                purpose = "parent-chat" if result_status == "completed" else "parent-chat-rejected"
                inbox_id = None
            else:
                capture = create_unassigned_capture(self.db_path, message, str(binding["parent_id"]))
                inbox_id = str(capture["inbox_id"])
                link_event_attachments(
                    self.db_path,
                    binding_id=binding_id,
                    external_event_id=message.external_event_id,
                    inbox_id=inbox_id,
                )
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
            parent_memory_refs: list[dict[str, Any]] | None = None
            if response is not None:
                transient_delivery = response.get("transient_delivery")
                if isinstance(transient_delivery, dict) and transient_delivery.get("kind") == (
                    "parent_memory_inspection"
                ):
                    refs = transient_delivery.get("memory_refs")
                    if not isinstance(refs, list):
                        raise ValueError("parent memory inspection delivery lineage is invalid")
                    parent_memory_refs = refs
            outbound_id = self._queue(
                message,
                binding_id,
                reply,
                purpose=purpose,
                blocks=reply_blocks,
                parent_memory_refs=parent_memory_refs,
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
                enqueue_artifact_bundle(
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
            if active_binding(self.db_path, message) is None:
                finish_receipt(
                    self.db_path,
                    message,
                    status="rejected",
                    error="Slack delivery authority changed during processing",
                )
                return TransportResult(
                    status="rejected",
                    message="Slack access changed while I was working, so no reply was sent.",
                    binding_id=binding_id,
                )
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
            logging.getLogger(__name__).exception(
                "Slack processing failed (%s; event=%s)",
                exc.__class__.__name__,
                sha256(message.external_event_id.encode()).hexdigest()[:12],
            )
            if active_binding(self.db_path, message) is None:
                finish_receipt(
                    self.db_path,
                    message,
                    status="rejected",
                    error="Slack delivery authority changed during processing",
                )
                return TransportResult(
                    status="rejected",
                    message="Slack access changed while I was working, so no reply was sent.",
                    binding_id=binding_id,
                )
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
        kwargs: dict[str, Any] = {
            "channel": message.channel_id,
            "text": message.text,
            # Block Kit repeats the same human-readable text as Slack's required
            # fallback. Suppress automatic previews so one URL does not render as
            # duplicate rich cards from the two representations.
            "unfurl_links": False,
            "unfurl_media": False,
        }
        if message.thread_id:
            kwargs["thread_ts"] = message.thread_id
        if message.blocks:
            kwargs["blocks"] = message.blocks
        if not mark_delivery(db_path, row["id"], status="sending"):
            results.append({"delivery_id": row["id"], "status": "unknown"})
            continue
        try:
            response = client.chat_postMessage(**kwargs)
        except Exception as exc:
            # SlackApiError confirms that Slack rejected the request. Other failures may be
            # ambiguous after a network break, so do not retry them automatically.
            if exc.__class__.__name__ == "SlackApiError":
                changed = mark_delivery(db_path, row["id"], status="failed", error=str(exc))
                results.append(
                    {"delivery_id": row["id"], "status": "failed" if changed else "unknown"}
                )
            else:
                mark_delivery(db_path, row["id"], status="unknown", error=exc.__class__.__name__)
                results.append({"delivery_id": row["id"], "status": "unknown"})
            continue
        external_id = str(response.get("ts", ""))
        if not mark_delivery(db_path, row["id"], status="sent", external_message_id=external_id):
            results.append({"delivery_id": row["id"], "status": "unknown"})
            continue
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
        claim = claim_visual_delivery(db_path, delivery_id)
        if not claim:
            continue
        attempt_id = claim["attempt_id"]
        lease_token = claim["lease_token"]
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

            if not mark_visual_delivery(
                db_path,
                delivery_id,
                status="ticket_acquiring",
                attempt_id=attempt_id,
                lease_token=lease_token,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
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
            if not mark_visual_delivery(
                db_path,
                delivery_id,
                status=stage,
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
            stage = "uploading"
            if not mark_visual_delivery(
                db_path,
                delivery_id,
                status=stage,
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
            upload(upload_url, data)
            stage = "bytes_uploaded"
            if not mark_visual_delivery(
                db_path,
                delivery_id,
                status=stage,
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
            stage = "completing"
            if not mark_visual_delivery(
                db_path,
                delivery_id,
                status=stage,
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
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
            if not mark_visual_delivery(
                db_path,
                delivery_id,
                status="sent",
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
                external_message_id=external_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
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
            if stage != "ticket_acquiring" and not slack_rejection:
                status = "unknown"
            elif slack_rejection or isinstance(exc, ValueError):
                status = "expired"
            else:
                status = "failed"
            changed = mark_visual_delivery(
                db_path,
                delivery_id,
                status=status,
                attempt_id=attempt_id,
                lease_token=lease_token,
                error=exc.__class__.__name__,
            )
            results.append({"delivery_id": delivery_id, "status": status if changed else "unknown"})
    return results


def flush_slack_artifact_outbox(
    client: Any,
    db_path: str | Path,
    output_dir: str | Path,
    *,
    uploader: Callable[[str, bytes], None] | None = None,
) -> list[dict[str, str]]:
    """Upload usable page images and printable PDFs after their Slack message."""

    results: list[dict[str, str]] = []
    allowed_root = (Path(output_dir).resolve() / "artifacts").resolve()
    upload = uploader or _upload_bytes
    for row in ready_artifact_deliveries(db_path):
        delivery_id = str(row["id"])
        claim = claim_artifact_delivery(db_path, delivery_id)
        if not claim:
            continue
        attempt_id = claim["attempt_id"]
        lease_token = claim["lease_token"]
        stage = "ticket_acquiring"
        try:
            path = Path(str(row["path"])).resolve()
            if not path.is_relative_to(allowed_root) or not path.is_file():
                raise ValueError("artifact path is outside the private artifact root")
            data = path.read_bytes()
            if len(data) != int(row["byte_count"]) or sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("artifact bytes changed after validation")
            magic = {"application/pdf": b"%PDF-", "image/png": b"\x89PNG\r\n\x1a\n"}.get(row["mime_type"])
            if not magic or not data.startswith(magic):
                raise ValueError("artifact is not a validated PDF or PNG")
            if not mark_artifact_delivery(
                db_path,
                delivery_id,
                status="ticket_acquiring",
                attempt_id=attempt_id,
                lease_token=lease_token,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
            ticket_args = {"filename": row["filename"], "length": len(data)}
            if row["mime_type"] == "image/png":
                ticket_args["alt_txt"] = f"Activity page: {row['title']}. {row['comment']}"
            ticket = client.files_getUploadURLExternal(**ticket_args)
            upload_url = str(ticket.get("upload_url") or "")
            file_id = str(ticket.get("file_id") or "")
            if not upload_url.startswith("https://") or not file_id:
                raise RuntimeError("Slack did not return a valid artifact upload ticket")
            stage = "ticket_acquired"
            if not mark_artifact_delivery(
                db_path,
                delivery_id,
                status=stage,
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
            stage = "uploading"
            if not mark_artifact_delivery(
                db_path,
                delivery_id,
                status=stage,
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
            upload(upload_url, data)
            stage = "bytes_uploaded"
            if not mark_artifact_delivery(
                db_path,
                delivery_id,
                status=stage,
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
            stage = "completing"
            if not mark_artifact_delivery(
                db_path,
                delivery_id,
                status=stage,
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
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
            if not mark_artifact_delivery(
                db_path,
                delivery_id,
                status="sent",
                attempt_id=attempt_id,
                lease_token=lease_token,
                slack_file_id=file_id,
                external_message_id=external_id,
            ):
                results.append({"delivery_id": delivery_id, "status": "unknown"})
                continue
            results.append({"delivery_id": delivery_id, "status": "sent"})
        except Exception as exc:
            slack_rejection = exc.__class__.__name__ == "SlackApiError"
            status = (
                "unknown"
                if stage != "ticket_acquiring" and not slack_rejection
                else "expired"
                if slack_rejection or isinstance(exc, ValueError)
                else "failed"
            )
            changed = mark_artifact_delivery(
                db_path,
                delivery_id,
                status=status,
                attempt_id=attempt_id,
                lease_token=lease_token,
                error=exc.__class__.__name__,
            )
            results.append({"delivery_id": delivery_id, "status": status if changed else "unknown"})
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
        if event.get("bot_id") or (event.get("subtype") and event.get("subtype") != "file_share"):
            return
        event_type = str(event.get("type", ""))
        if event_type == "message" and event.get("channel_type") != "im":
            return
        text = re.sub(r"<@[A-Z0-9]+>", "", str(event.get("text", ""))).strip()
        has_files = bool(event.get("files"))
        if not text and not has_files:
            return
        if not text:
            text = "Shared a photo."
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
        binding = active_binding(db_path, incoming)
        source_message_ts = str(event.get("ts") or event.get("event_ts") or "")
        reaction_added = False
        if binding and source_message_ts:
            try:
                client.reactions_add(
                    channel=incoming.channel_id,
                    timestamp=source_message_ts,
                    name=PROCESSING_REACTION,
                )
                reaction_added = True
            except Exception as exc:
                # A missing optional Slack scope must never block the actual reply.
                logging.getLogger(__name__).warning(
                    "Slack processing reaction was unavailable (%s)", exc.__class__.__name__
                )
        try:
            # Do not fetch bytes for an unpaired identity or conversation. Slack
            # metadata is normalized first; only an exact active binding unlocks
            # the bounded private download.
            if has_files and binding:
                incoming = incoming.model_copy(update={"attachments": _receive_slack_images(transport, event, client)})
            result = transport.handle(incoming)
            if result.status == "rejected" and not result.binding_id:
                direct: dict[str, Any] = {"channel": incoming.channel_id, "text": result.message}
                if incoming.thread_id:
                    direct["thread_ts"] = incoming.thread_id
                client.chat_postMessage(**direct)
            flush_slack_outbox(client, db_path)
        finally:
            if reaction_added:
                try:
                    client.reactions_add(
                        channel=incoming.channel_id,
                        timestamp=source_message_ts,
                        name=PROCESSING_COMPLETE_REACTION,
                    )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Slack completion reaction was unavailable (%s)",
                        exc.__class__.__name__,
                    )
                # Keep the acknowledgement reaction. Slack cannot atomically replace
                # reactions, and removing :eyes: could render before the reply/check
                # on some clients, making a healthy request look abandoned.

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
                thread_id=str(source_message.get("thread_ts")) if source_message.get("thread_ts") else None,
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
                    logging.getLogger(__name__).exception("could not show Slack inbox assignment progress")
            result = transport.handle(incoming)
            if result.status == "completed":
                if message_ts:
                    resolved_text = (
                        "Dismissed this saved note."
                        if action_id == INBOX_DISMISS_ACTION
                        else (
                            f"Saved this context for {selected_child_name}."
                            if result.event_id is None
                            else "Assigned this saved note. The learning thread follows."
                        )
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
                if result.outbound_id:
                    # The interactive message itself carries the retry state.
                    # Retire the queued fallback so one tap cannot produce a
                    # second, contradictory error message in the thread.
                    mark_delivery(
                        db_path,
                        result.outbound_id,
                        status="expired",
                        error="presented inline by Slack action handler",
                    )
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
                    logging.getLogger(__name__).exception("could not restore Slack inbox controls")
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
        channel_id = ""
        message_ts = ""
        source_event_id = ""
        source_message: dict[str, Any] = {}
        original_blocks: list[dict[str, Any]] = []
        team_id = ""
        user_id = ""
        receipt_claimed = False
        action_completed = False
        incoming: InboundMessage | None = None
        authority_message: InboundMessage | None = None
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
            elif action_id == RESPONSE_CHALLENGE_ACTION:
                command = f"challenge_response {source_event_id}"
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
            authority_message = InboundMessage(
                external_event_id=_response_action_receipt_id(
                    action_id,
                    team_id=team_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    event_id=source_event_id,
                ),
                team_id=team_id,
                user_id=user_id,
                channel_id=channel_id,
                text=command,
                thread_id=str(source_message.get("thread_ts") or source_message.get("ts") or "") or None,
            )
            binding = active_binding(db_path, authority_message)
            if not binding:
                raise PermissionError("paired parent binding not found")
            try:
                receipt_claimed = begin_receipt(
                    db_path,
                    authority_message,
                    str(binding["id"]),
                    retry_failed=True,
                )
            except TransportConflict:
                return
            if not receipt_claimed:
                return
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
                thread_id=authority_message.thread_id,
            )
            message_ts = str(source_message.get("ts") or container.get("message_ts") or "")
            original_blocks = [
                block
                for block in (source_message.get("blocks") or [])
                if str(block.get("block_id") or "") != f"curiosity_response:{source_event_id}"
            ]
            progress_text = {
                RESPONSE_HELPFUL_ACTION: ":hourglass_flowing_sand: Saving as helpful…",
                RESPONSE_NOT_HELPFUL_ACTION: ":hourglass_flowing_sand: Saving as not for us…",
                RESPONSE_RETRY_ACTION: ":hourglass_flowing_sand: Trying a different approach…",
                RESPONSE_CHALLENGE_ACTION: ":art: Drawing and building the challenge…",
            }[action_id]
            if message_ts:
                try:
                    client.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text=str(source_message.get("text") or progress_text),
                        blocks=[
                            *original_blocks,
                            {
                                "type": "context",
                                "elements": [{"type": "mrkdwn", "text": progress_text}],
                            },
                        ],
                    )
                except Exception:
                    logging.getLogger(__name__).exception("could not show Slack response action progress")
            result = transport.handle(incoming)
            if result.status == "completed":
                finish_receipt(
                    db_path,
                    authority_message,
                    status="completed",
                    event_id=result.event_id,
                )
                action_completed = True
                if message_ts:
                    if action_id == RESPONSE_HELPFUL_ACTION:
                        resolved_text = ":white_check_mark: Marked helpful."
                    elif action_id == RESPONSE_NOT_HELPFUL_ACTION:
                        resolved_text = ":white_check_mark: Marked not for us."
                    elif action_id == RESPONSE_RETRY_ACTION:
                        resolved_text = ":sparkles: Another approach is in the thread."
                    else:
                        resolved_text = ":white_check_mark: Visual challenge ready; sending the picture and PDF…"
                    updated_blocks = [
                        *original_blocks,
                        {"type": "context", "elements": [{"type": "mrkdwn", "text": resolved_text}]},
                    ]
                    remaining_action_block = _remaining_response_action_block(
                        list(source_message.get("blocks") or []),
                        source_event_id,
                        _consumed_response_action_ids(
                            db_path,
                            team_id=team_id,
                            user_id=user_id,
                            channel_id=channel_id,
                            event_id=source_event_id,
                        ),
                    )
                    if remaining_action_block:
                        updated_blocks.append(remaining_action_block)
                    try:
                        client.chat_update(
                            channel=channel_id,
                            ts=message_ts,
                            text=str(source_message.get("text") or result.message),
                            blocks=updated_blocks,
                        )
                    except Exception:
                        logging.getLogger(__name__).exception("could not retire Slack response controls")
            else:
                finish_receipt(
                    db_path,
                    authority_message,
                    status="failed",
                    error="response action did not complete",
                )
            if result.status != "completed" and message_ts:
                remaining_action_block = _remaining_response_action_block(
                    list(source_message.get("blocks") or []),
                    source_event_id,
                    _consumed_response_action_ids(
                        db_path,
                        team_id=team_id,
                        user_id=user_id,
                        channel_id=channel_id,
                        event_id=source_event_id,
                    ),
                )
                try:
                    client.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text=str(source_message.get("text") or result.message),
                        blocks=[
                            *original_blocks,
                            {
                                "type": "context",
                                "elements": [
                                    {"type": "mrkdwn", "text": ":warning: I couldn't finish that action. Try again."}
                                ],
                            },
                            *([remaining_action_block] if remaining_action_block else []),
                        ],
                    )
                except Exception:
                    logging.getLogger(__name__).exception("could not restore Slack response controls")
            flush_slack_outbox(client, db_path)
        except Exception:
            logging.getLogger(__name__).exception("Slack response action failed")
            if receipt_claimed and not action_completed and authority_message is not None:
                try:
                    finish_receipt(
                        db_path,
                        authority_message,
                        status="failed",
                        error="Slack response action failed",
                    )
                except Exception:
                    logging.getLogger(__name__).exception("could not release failed Slack response action")
            if channel_id and message_ts and source_event_id:
                try:
                    remaining_action_block = _remaining_response_action_block(
                        list(source_message.get("blocks") or []),
                        source_event_id,
                        _consumed_response_action_ids(
                            db_path,
                            team_id=team_id,
                            user_id=user_id,
                            channel_id=channel_id,
                            event_id=source_event_id,
                        ),
                    )
                    client.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text=str(source_message.get("text") or "I couldn't finish that action."),
                        blocks=[
                            *original_blocks,
                            {
                                "type": "context",
                                "elements": [
                                    {"type": "mrkdwn", "text": ":warning: I couldn't finish that action. Try again."}
                                ],
                            },
                            *([remaining_action_block] if remaining_action_block else []),
                        ],
                    )
                except Exception:
                    logging.getLogger(__name__).exception("could not show Slack response action failure")

    return receive


def _interaction_progress_copy(resolved: dict[str, Any], *, completed: bool = False) -> str | None:
    """Return visible progress for semantic choices that can take more than a moment."""

    intent = str(resolved.get("intent") or "")
    payload = dict(resolved.get("payload") or {})
    artifact_type = str(payload.get("artifact_type") or "printable").casefold()
    if intent == "create_artifact":
        return (
            f":white_check_mark: {artifact_type.title()} ready; sending the picture and PDF…"
            if completed
            else f":art: Drawing and building the {artifact_type}…"
        )
    if intent == "revise_artifact":
        return (
            ":white_check_mark: Revision created; sending the picture and PDF…"
            if completed
            else ":eyes: Revising the child-ready printable…"
        )
    if intent == "retry_response":
        return (
            ":white_check_mark: Another approach is below." if completed else ":eyes: Trying a different approach…"
        )
    if intent == "explain_context":
        return ":white_check_mark: Explanation ready." if completed else ":eyes: Explaining that moment…"
    if intent == "suggest_context_idea":
        return ":white_check_mark: Idea ready." if completed else ":eyes: Thinking of one playful idea…"
    if intent == "create_context_artifact":
        return (
            ":white_check_mark: Visual challenge ready; sending the picture and PDF…"
            if completed
            else ":art: Drawing and building the challenge…"
        )
    if intent == "approve_tool_call":
        return ":white_check_mark: Setup finished." if completed else ":eyes: Setting that up…"
    return None


def _make_semantic_interaction_receiver(
    transport: SlackTransport,
    db_path: str | Path,
) -> Any:
    def receive(ack: Any, body: dict[str, Any], action: dict[str, Any], client: Any) -> None:
        ack()
        channel_id = ""
        message_ts = ""
        original_blocks: list[dict[str, Any]] = []
        progress_text: str | None = None
        resolved: dict[str, Any] = {}
        binding: dict[str, Any] | None = None
        incoming: InboundMessage | None = None
        receipt_claimed = False
        receipt_finished = False
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
            external_event_id = (
                "ix_slack_"
                + sha256(
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
            )
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
            receipt_claimed = begin_receipt(db_path, incoming, str(binding["id"]))
            if not receipt_claimed:
                return
            message_ts = str(source_message.get("ts") or container.get("message_ts") or "")
            original_blocks = [
                block for block in source_message.get("blocks") or [] if block.get("block_id") != block_id
            ]
            progress_text = _interaction_progress_copy(resolved)
            if message_ts and progress_text:
                client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=str(source_message.get("text") or progress_text),
                    blocks=[
                        *original_blocks,
                        {"type": "context", "elements": [{"type": "mrkdwn", "text": progress_text}]},
                    ],
                )
            result = transport.service.handle_interaction_choice(
                resolved=resolved,
                binding_id=str(binding["id"]),
                channel_id=channel_id,
            )
            reply = str(result.get("message") or "Done.")
            blocks: list[dict[str, Any]] = [
                {"type": "section", "text": {"type": "mrkdwn", "text": section}} for section in _mrkdwn_sections(reply)
            ]
            if result.get("interaction"):
                next_plan = InteractionPlan.model_validate(result["interaction"])
                if next_plan.kind == "rate_output" and result.get("event_id"):
                    blocks.append(_response_action_block(str(result["event_id"])))
                else:
                    presented = create_interaction(
                        db_path,
                        binding_id=str(binding["id"]),
                        session_id=resolved.get("session_id"),
                        plan=next_plan,
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
                enqueue_artifact_bundle(
                    db_path,
                    artifact=result["artifact"],
                    binding_id=str(binding["id"]),
                    depends_on_delivery_id=outbound_id,
                    channel_id=channel_id,
                    thread_id=thread_id,
                    idempotency_key=f"slack:{external_event_id}:artifact:{result['artifact']['artifact_id']}",
                )
            # The next response button verifies this receipt and its delivered
            # outbox entry. Semantic choices must record the same lineage as chat.
            result_status = str(result.get("status") or "completed")
            finish_receipt(db_path, incoming,
                           status=result_status if result_status in {"completed", "rejected", "failed"} else "completed",
                           event_id=result.get("event_id"))
            receipt_finished = True
            if message_ts:
                completed_text = _interaction_progress_copy(resolved, completed=True)
                client.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=str(source_message.get("text") or reply),
                    blocks=[
                        *original_blocks,
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": completed_text or ":white_check_mark: Choice received.",
                                }
                            ],
                        },
                    ],
                )
            flush_slack_outbox(client, db_path)
        except Exception:
            logging.getLogger(__name__).exception("Slack semantic interaction failed")
            if incoming and receipt_claimed and not receipt_finished:
                finish_receipt(db_path, incoming, status="failed", error="semantic interaction did not complete")
            if channel_id and message_ts and progress_text:
                try:
                    failure_copy = (
                        ":warning: I couldn't make a family-ready printable, so I did not send a template. "
                        "I kept the conversation and the request."
                        if str(resolved.get("intent") or "")
                        in {"create_artifact", "create_context_artifact", "revise_artifact"}
                        else ":warning: I couldn't finish that. I kept the conversation so you can retry."
                    )
                    failure_blocks = [
                        *original_blocks,
                        {"type": "context", "elements": [{"type": "mrkdwn", "text": failure_copy}]},
                    ]
                    if resolved and binding:
                        retry = create_interaction(
                            db_path,
                            binding_id=str(binding["id"]),
                            session_id=resolved.get("session_id"),
                            plan=InteractionPlan(
                                kind="request_revision",
                                title="Try that step again?",
                                prompt="I will rebuild it from the original thread context.",
                                options=[
                                    InteractionOption(
                                        label="Try again",
                                        intent=str(resolved["intent"]),
                                        payload=dict(resolved.get("payload") or {}),
                                        style="primary",
                                    )
                                ],
                            ),
                        )
                        failure_blocks.extend(interaction_blocks(retry))
                    client.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text="I couldn't finish that choice.",
                        blocks=failure_blocks,
                    )
                except Exception:
                    logging.getLogger(__name__).exception("could not show Slack interaction failure state")

    return receive


def _make_child_add_open_receiver(transport: SlackTransport, db_path: str | Path) -> Any:
    """Open a deterministic owner-only modal; no profile field enters model context."""

    def receive(ack: Any, body: dict[str, Any], action: dict[str, Any], client: Any) -> None:
        ack()
        try:
            if str(action.get("value") or "") != "open":
                raise ValueError("invalid child setup action")
            team = body.get("team") or {}
            user = body.get("user") or {}
            channel = body.get("channel") or {}
            container = body.get("container") or {}
            message = body.get("message") or {}
            incoming = InboundMessage(
                external_event_id="child_open_" + sha256(str(body.get("trigger_id") or "").encode()).hexdigest()[:24],
                team_id=str(team.get("id") or body.get("team_id") or ""),
                user_id=str(user.get("id") or body.get("user_id") or ""),
                channel_id=str(channel.get("id") or container.get("channel_id") or ""),
                text="open child profile form",
                thread_id=str(message.get("thread_ts") or message.get("ts") or "") or None,
            )
            binding = active_binding(db_path, incoming)
            if not binding or binding.get("role") != "owner":
                raise PermissionError("only the paired household owner can add a child")
            client.views_open(
                trigger_id=str(body.get("trigger_id") or ""),
                view=_child_add_modal(
                    binding_id=str(binding["id"]),
                    team_id=incoming.team_id,
                    user_id=incoming.user_id,
                    channel_id=incoming.channel_id,
                    thread_id=incoming.thread_id,
                ),
            )
        except Exception:
            logging.getLogger(__name__).exception("Slack child setup form could not open")

    return receive


def _view_input(view: dict[str, Any], block_id: str) -> str:
    values = ((view.get("state") or {}).get("values") or {}).get(block_id) or {}
    field = values.get("value") or {}
    return str(field.get("value") or "").strip()


def _make_child_add_submission_receiver(transport: SlackTransport, db_path: str | Path) -> Any:
    """Validate and store an owner-confirmed child profile entirely in deterministic code."""

    def receive(ack: Any, body: dict[str, Any], view: dict[str, Any], client: Any) -> None:
        errors: dict[str, str] = {}
        try:
            metadata = json.loads(str(view.get("private_metadata") or "{}"))
        except json.JSONDecodeError:
            metadata = {}
            errors["child_name"] = "This form is no longer valid. Open a new child form."
        name = " ".join(_view_input(view, "child_name").split())
        grade = " ".join(_view_input(view, "child_grade").split())
        birth_year_text = _view_input(view, "child_birth_year")
        birth_year: int | None = None
        if not name:
            errors["child_name"] = "Enter the name your family uses."
        if birth_year_text:
            if not birth_year_text.isdigit() or len(birth_year_text) != 4:
                errors["child_birth_year"] = "Use a four-digit year, such as 2020."
            else:
                birth_year = int(birth_year_text)
                current_year = datetime.now(UTC).year
                if not current_year - 25 <= birth_year <= current_year:
                    errors["child_birth_year"] = "Enter a year from the last 25 years."
        incoming = InboundMessage(
            external_event_id="child_submit_"
            + sha256(str(view.get("id") or body.get("trigger_id") or "").encode()).hexdigest()[:24],
            team_id=str((body.get("team") or {}).get("id") or metadata.get("team_id") or ""),
            user_id=str((body.get("user") or {}).get("id") or metadata.get("user_id") or ""),
            channel_id=str(metadata.get("channel_id") or ""),
            text="submit child profile form",
            thread_id=str(metadata.get("thread_id") or "") or None,
        )
        binding = active_binding(db_path, incoming)
        if (
            not binding
            or binding.get("role") != "owner"
            or str(binding.get("id")) != str(metadata.get("binding_id") or "")
        ):
            errors["child_name"] = "Only the paired household owner can submit this form."
        if errors:
            ack(response_action="errors", errors=errors)
            return
        ack()
        try:
            child = transport.service.add_child_from_binding(
                binding_id=str(binding["id"]),
                request_id=str(view.get("id") or incoming.external_event_id),
                name=name,
                birth_year=birth_year,
                grade=grade or None,
            )
            text = (
                f"Added *{_mrkdwn_text(child['name'], limit=80)}*"
                + (f" ({_mrkdwn_text(child['grade'], limit=40)})" if child.get("grade") else "")
                + ". Their private context graph starts empty—nothing was inferred."
            )
            kwargs: dict[str, Any] = {"channel": incoming.channel_id, "text": text}
            if incoming.thread_id:
                kwargs["thread_ts"] = incoming.thread_id
            client.chat_postMessage(**kwargs)
        except Exception:
            logging.getLogger(__name__).exception("Slack child profile creation failed")
            kwargs = {
                "channel": incoming.channel_id,
                "text": "I couldn't add that child profile. Nothing was inferred or sent to the model.",
            }
            if incoming.thread_id:
                kwargs["thread_ts"] = incoming.thread_id
            client.chat_postMessage(**kwargs)

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
    app.action(INBOX_ASSIGN_ACTION)(_make_slack_action_receiver(transport, db_path, action_id=INBOX_ASSIGN_ACTION))
    app.action(INBOX_DISMISS_ACTION)(_make_slack_action_receiver(transport, db_path, action_id=INBOX_DISMISS_ACTION))
    app.action(CHILD_ADD_OPEN_ACTION)(_make_child_add_open_receiver(transport, db_path))
    app.view(CHILD_ADD_MODAL_CALLBACK)(_make_child_add_submission_receiver(transport, db_path))
    for response_action in (
        RESPONSE_HELPFUL_ACTION,
        RESPONSE_NOT_HELPFUL_ACTION,
        RESPONSE_RETRY_ACTION,
        RESPONSE_CHALLENGE_ACTION,
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
