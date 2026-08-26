from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Literal

from pydantic import Field

from ..contracts import StrictModel


class InboundAttachment(StrictModel):
    """A private, validated transport attachment; never a public asset reference."""

    external_ref_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: Literal["ready", "unavailable"]
    media_type: Literal["image/jpeg", "image/png", "image/webp"] | None = None
    byte_count: int = Field(default=0, ge=0, le=10_000_000)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    private_path: str | None = Field(default=None, max_length=1_000)


class InboundMessage(StrictModel):
    transport: Literal["slack"] = "slack"
    external_event_id: str = Field(min_length=1, max_length=240)
    team_id: str = Field(min_length=1, max_length=120)
    user_id: str = Field(min_length=1, max_length=120)
    channel_id: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=20_000)
    thread_id: str | None = Field(default=None, max_length=120)
    occurred_at: str | None = Field(default=None, max_length=80)
    attachments: list[InboundAttachment] = Field(default_factory=list, max_length=3)

    @property
    def payload_hash(self) -> str:
        payload = self.model_dump(mode="json")
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class OutboundMessage(StrictModel):
    channel_id: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=4_000)
    thread_id: str | None = Field(default=None, max_length=120)
    blocks: list[dict[str, Any]] = Field(default_factory=list, max_length=50)


class TransportResult(StrictModel):
    status: Literal["paired", "completed", "unassigned", "ignored", "rejected", "failed", "duplicate"]
    message: str
    event_id: str | None = None
    inbox_id: str | None = None
    binding_id: str | None = None
    outbound_id: str | None = None
