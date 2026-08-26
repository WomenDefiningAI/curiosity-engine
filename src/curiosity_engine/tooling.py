from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .contracts import ToolSpec

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


DEFAULT_TOOL_SPECS = (
    ToolSpec(
        name="record_thread_context",
        version="1",
        description=(
            "Acknowledge and retain parent-shared text or photo context without creating child evidence, a lesson, "
            "an activity, or a visual."
        ),
        risk="low",
        side_effect="local_write",
        data_classes=["thread_history", "parent_context", "private_image_observation"],
        origins=["slack", "cli"],
        timeout_seconds=30,
    ),
    ToolSpec(
        name="continue_learning_thread",
        version="1",
        description="Answer or deepen the current child learning thread using bounded context and prior thread releases.",
        risk="low",
        side_effect="local_write",
        data_classes=["child_context", "thread_history"],
        origins=["slack", "cli"],
        timeout_seconds=180,
    ),
    ToolSpec(
        name="revise_learning_thread",
        version="1",
        description="Revise requested answer, activity, or visual components while preserving the rest.",
        risk="low",
        side_effect="local_write",
        data_classes=["child_context", "thread_history", "parent_feedback"],
        origins=["slack", "cli"],
        timeout_seconds=180,
    ),
    ToolSpec(
        name="create_learning_artifact",
        version="1",
        description="Generate, validate, render, and preview one worksheet, activity, or challenge.",
        risk="low",
        side_effect="local_write",
        data_classes=["child_context", "thread_history"],
        origins=["slack", "cli"],
        timeout_seconds=240,
    ),
    ToolSpec(
        name="propose_weekly_checkin",
        version="1",
        description="Prepare a recurring weekly parent check-in for explicit confirmation.",
        risk="sensitive",
        side_effect="external_write",
        approval="per_call",
        data_classes=["schedule", "slack_binding"],
        origins=["slack", "cli"],
    ),
    ToolSpec(
        name="record_response_feedback",
        version="1",
        description="Record parent output feedback without treating it as child learning evidence.",
        risk="low",
        side_effect="local_write",
        data_classes=["parent_feedback"],
        origins=["slack", "cli"],
    ),
)


@dataclass(frozen=True)
class ToolDecision:
    allowed: bool
    requires_approval: bool
    reason: str


class ToolPolicy:
    """Deny-by-default boundary independent from model wording."""

    @staticmethod
    def decide(spec: ToolSpec, *, origin: str, approved: bool = False) -> ToolDecision:
        if origin not in spec.origins:
            return ToolDecision(False, False, f"{spec.name} is unavailable from {origin}")
        if spec.approval == "per_call" and not approved:
            return ToolDecision(False, True, f"{spec.name} requires parent confirmation")
        if spec.risk == "high":
            return ToolDecision(False, False, "high-risk tools are disabled")
        return ToolDecision(True, False, "allowed by reviewed tool policy")


class ToolRegistry:
    def __init__(self):
        self._specs = {spec.name: spec for spec in DEFAULT_TOOL_SPECS}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        if name not in self._specs:
            raise KeyError(f"unreviewed tool: {name}")
        self._handlers[name] = handler

    def specs(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        specs = self._specs.values()
        if allowed is not None:
            specs = [spec for spec in specs if spec.name in allowed]
        return [spec.model_dump(mode="json") for spec in specs]

    def spec(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"unreviewed tool: {name}") from exc

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        origin: str,
        approved: bool = False,
    ) -> dict[str, Any]:
        spec = self.spec(name)
        decision = ToolPolicy.decide(spec, origin=origin, approved=approved)
        if not decision.allowed:
            return {
                "status": "awaiting_approval" if decision.requires_approval else "denied",
                "reason": decision.reason,
            }
        try:
            handler = self._handlers[name]
        except KeyError as exc:
            raise RuntimeError(f"tool handler is not configured: {name}") from exc
        return handler(arguments)
