from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import InteractionEvent, InteractionPlan
from .db import connect, init_db, jdump, jload, utcnow

SLACK_INTERACTION_BUTTON = "curiosity_interaction_button"
SLACK_INTERACTION_SELECT = "curiosity_interaction_select"


def _token_hash(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def create_interaction(
    db_path: str | Path,
    *,
    binding_id: str,
    plan: InteractionPlan | dict[str, Any],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Persist semantic choices and return opaque one-use option tokens."""

    init_db(db_path)
    parsed = plan if isinstance(plan, InteractionPlan) else InteractionPlan.model_validate(plan)
    interaction_id = f"ix_{uuid4().hex[:20]}"
    expires_at = (datetime.now(UTC) + timedelta(minutes=parsed.expires_in_minutes)).isoformat()
    presented_options: list[dict[str, Any]] = []
    stored_options: list[dict[str, Any]] = []
    for option in parsed.options:
        token = secrets.token_urlsafe(24)
        presented_options.append({"label": option.label, "style": option.style, "token": token})
        stored_options.append(
            {
                "token_hash": _token_hash(token),
                "label": option.label,
                "intent": option.intent,
                "payload": option.payload,
                "style": option.style,
            }
        )
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO interaction_instances(
                   id,session_id,binding_id,plan_json,options_json,status,expires_at,created_at,updated_at
               ) VALUES(?,?,?,?,?,'active',?,?,?)""",
            (
                interaction_id,
                session_id,
                binding_id,
                jdump(parsed.model_dump(mode="json")),
                jdump(stored_options),
                expires_at,
                now,
                now,
            ),
        )
    return {
        "interaction_id": interaction_id,
        "plan": parsed.model_dump(mode="json"),
        "options": presented_options,
        "expires_at": expires_at,
    }


def resolve_interaction(
    db_path: str | Path,
    event: InteractionEvent | dict[str, Any],
    *,
    binding_id: str,
    parent_id: str | None,
) -> dict[str, Any]:
    """Resolve an opaque option under the paired binding and consume it exactly once."""

    init_db(db_path)
    parsed = event if isinstance(event, InteractionEvent) else InteractionEvent.model_validate(event)
    now = datetime.now(UTC)
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM interaction_instances WHERE id=? AND binding_id=?",
            (parsed.interaction_id, binding_id),
        ).fetchone()
        if not row:
            raise ValueError("interaction not found for this parent")
        duplicate = conn.execute(
            "SELECT intent,payload_json FROM interaction_events WHERE interaction_id=? AND external_event_id=?",
            (parsed.interaction_id, parsed.external_event_id),
        ).fetchone()
        if duplicate:
            return {"intent": duplicate["intent"], "payload": jload(duplicate["payload_json"]), "duplicate": True}
        if row["status"] != "active":
            raise ValueError("interaction is no longer active")
        if datetime.fromisoformat(row["expires_at"]) <= now:
            conn.execute(
                "UPDATE interaction_instances SET status='expired',updated_at=? WHERE id=?",
                (now.isoformat(), parsed.interaction_id),
            )
            raise ValueError("interaction expired")
        token_hash = _token_hash(parsed.option_token)
        option = next(
            (item for item in jload(row["options_json"], []) if secrets.compare_digest(item["token_hash"], token_hash)),
            None,
        )
        if not option:
            raise ValueError("invalid interaction option")
        event_id = f"ixe_{uuid4().hex[:20]}"
        conn.execute(
            """INSERT INTO interaction_events(
                   id,interaction_id,option_token_hash,intent,payload_json,actor_parent_id,external_event_id,created_at
               ) VALUES(?,?,?,?,?,?,?,?)""",
            (
                event_id,
                parsed.interaction_id,
                token_hash,
                option["intent"],
                jdump(option.get("payload") or {}),
                parent_id,
                parsed.external_event_id,
                now.isoformat(),
            ),
        )
        conn.execute(
            "UPDATE interaction_instances SET status='completed',updated_at=? WHERE id=?",
            (now.isoformat(), parsed.interaction_id),
        )
    return {
        "interaction_event_id": event_id,
        "interaction_id": parsed.interaction_id,
        "session_id": row["session_id"],
        "intent": option["intent"],
        "payload": option.get("payload") or {},
        "duplicate": False,
    }


def interaction_blocks(presented: dict[str, Any]) -> list[dict[str, Any]]:
    """Compile a transport-neutral plan into conservative Slack Block Kit."""

    plan = InteractionPlan.model_validate(presented["plan"])
    interaction_id = str(presented["interaction_id"])
    options = list(presented.get("options") or [])
    blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{plan.title}*\n{plan.prompt}"[:2900]},
        }
    ]
    if not options:
        return blocks
    if plan.kind in {"choose_child", "choose_one"} or len(options) > 5:
        blocks.append(
            {
                "type": "actions",
                "block_id": f"curiosity_interaction:{interaction_id}",
                "elements": [
                    {
                        "type": "static_select",
                        "action_id": SLACK_INTERACTION_SELECT,
                        "placeholder": {"type": "plain_text", "text": "Choose one", "emoji": True},
                        "options": [
                            {
                                "text": {"type": "plain_text", "text": str(option["label"])[:75], "emoji": True},
                                "value": str(option["token"]),
                            }
                            for option in options[:100]
                        ],
                    }
                ],
            }
        )
        return blocks
    blocks.append(
        {
            "type": "actions",
            "block_id": f"curiosity_interaction:{interaction_id}",
            "elements": [
                {
                    "type": "button",
                    # Slack requires every action_id in an actions block to be unique.
                    # The receiver matches this stable prefix and trusts only the opaque,
                    # binding-scoped option token as the requested intent.
                    "action_id": f"{SLACK_INTERACTION_BUTTON}_{index}",
                    "text": {"type": "plain_text", "text": str(option["label"])[:75], "emoji": True},
                    "value": str(option["token"]),
                    **({"style": option["style"]} if option.get("style") in {"primary", "danger"} else {}),
                }
                for index, option in enumerate(options[:5])
            ],
        }
    )
    return blocks
