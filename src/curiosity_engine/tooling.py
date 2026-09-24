from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .contracts import ToolSpec

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]
LineageResolver = Callable[[dict[str, Any]], str | None]


DEFAULT_TOOL_SPECS = (
    ToolSpec(
        name="search_thread_outputs",
        version="1",
        description=(
            "Search reviewed answers and artifacts only inside the current authorized learning thread. Arguments: "
            "query (a short description of the earlier output). Use when the parent refers to an older item that "
            "is not unambiguous from the recent history."
        ),
        risk="read_only",
        side_effect="none",
        data_classes=["thread_history"],
        origins=["slack", "cli"],
        timeout_seconds=10,
    ),
    ToolSpec(
        name="update_thread_preference",
        version="1",
        description=(
            "Set or clear explicit presentation/workflow guidance for this Slack thread only. Arguments: operation "
            "(set or clear), category (answer_style, visual_style, artifact_style, activity_style, or "
            "interaction_style), and value for set. Never use for a one-off critique, child trait, safety override, "
            "or family-wide rule."
        ),
        risk="low",
        side_effect="local_write",
        data_classes=["thread_history", "parent_preferences"],
        origins=["slack", "cli"],
        timeout_seconds=10,
    ),
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
        version="2",
        description=(
            "Answer or deepen the current child learning thread using bounded context and prior thread releases. "
            "Arguments: message and mode (auto, quick, or explore). Quick is a concise reviewed answer; explore is "
            "a richer investigation and must not be chosen merely because a child's wording is vague."
        ),
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
        self._execution_context_handlers: set[str] = set()
        self._lineage_resolvers: dict[str, LineageResolver] = {}

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        receives_execution_context: bool = False,
        lineage_resolver: LineageResolver | None = None,
    ) -> None:
        if name not in self._specs:
            raise KeyError(f"unreviewed tool: {name}")
        self._handlers[name] = handler
        if receives_execution_context:
            self._execution_context_handlers.add(name)
        else:
            self._execution_context_handlers.discard(name)
        if lineage_resolver is None:
            self._lineage_resolvers.pop(name, None)
        else:
            self._lineage_resolvers[name] = lineage_resolver

    def lineage_event_id(self, name: str, arguments: dict[str, Any]) -> str | None:
        resolver = self._lineage_resolvers.get(name)
        if resolver is None:
            return None
        resolved = str(resolver(dict(arguments)) or "").strip()
        return resolved or None

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
        execution_context: dict[str, str] | None = None,
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
        bound_arguments = dict(arguments)
        if execution_context and name in self._execution_context_handlers:
            # This namespace is overwritten by code after policy validation; it
            # is never accepted as model-authored authority or exposed to a
            # handler that did not explicitly opt into the internal contract.
            bound_arguments["_agent_execution_context"] = dict(execution_context)
        return handler(bound_arguments)
