from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from .capabilities import CapabilityRegistry
from .contracts import InteractionOption, InteractionPlan, ParentAgentToolCall, ParentAgentTurn
from .db import connect, init_db, jdump, utcnow
from .reasoning import ModelBackend
from .sessions import SessionStore
from .tooling import ToolPolicy, ToolRegistry

PARENT_AGENT_POLICY_VERSION = "parent-agent-v3"
MAX_PARENT_AGENT_STEPS = 2
MAX_PARENT_AGENT_TOOL_CALLS = 3


class ParentAgentRuntime:
    """A bounded, provider-neutral semantic tool loop for ordinary parent chat."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        backend: ModelBackend,
        tools: ToolRegistry,
        capabilities: CapabilityRegistry | None = None,
    ):
        self.db_path = str(db_path)
        init_db(self.db_path)
        self.backend = backend
        self.tools = tools
        self.capabilities = capabilities or CapabilityRegistry()
        self.sessions = SessionStore(self.db_path)

    def run(
        self,
        *,
        session_id: str,
        user_message: str,
        child: dict[str, Any] | None,
        latest_event_id: str | None,
        origin: str = "slack",
        current_attachment_context: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        history = self.sessions.history(session_id, limit=16)
        capability = self.capabilities.capability("parent_chat")
        allowed_tools = set(capability.tool_names)
        request = {
            "session": {"id": session_id, "child": child, "latest_event_id": latest_event_id},
            "history": [
                {
                    "role": item["role"],
                    "kind": item["kind"],
                    "content": item["content"],
                    "attachment_context": (item.get("metadata") or {}).get("attachment_context") or [],
                }
                for item in history
            ],
            "message": user_message,
            "current_attachment_context": current_attachment_context or [],
            "workspace": {
                "thread_preferences": self.sessions.active_preferences(session_id),
                "recent_outputs": self.sessions.output_index(session_id, limit=10),
            },
            "capabilities": self.capabilities.capability_cards(),
            "skills": self.capabilities.skill_cards(set(capability.skill_ids)),
            "tools": self.tools.specs(allowed_tools),
        }
        policy_hash = sha256(
            (PARENT_AGENT_POLICY_VERSION + jdump(request["tools"]) + jdump(request["skills"])).encode()
        ).hexdigest()[:20]
        turn_row = self.sessions.start_turn(session_id, request, policy_hash=policy_hash)
        run_id = self._start_capability_run(session_id, latest_event_id)
        try:
            result: dict[str, Any] = {"status": "completed", "message": None, "tool_results": []}
            seen_calls: set[str] = set()
            call_ordinal = 1
            for step in range(1, MAX_PARENT_AGENT_STEPS + 1):
                request["agent_step"] = step
                turn = self._plan(request, allowed_tools)
                if call_ordinal - 1 + len(turn.tool_calls) > MAX_PARENT_AGENT_TOOL_CALLS:
                    raise ValueError("parent agent exceeded its bounded tool-call budget")
                fingerprints = [
                    sha256(f"{call.name}:{jdump(call.arguments)}".encode()).hexdigest()
                    for call in turn.tool_calls
                ]
                if any(fingerprint in seen_calls for fingerprint in fingerprints):
                    result["message"] = (
                        "I could not tell which earlier item you meant. Tell me a few words from it, and I’ll "
                        "look again without changing anything yet."
                    )
                    break
                seen_calls.update(fingerprints)
                step_result = self._execute_turn(
                    turn,
                    session_id=session_id,
                    turn_id=turn_row["id"],
                    origin=origin,
                    call_ordinal=call_ordinal,
                )
                call_ordinal += len(turn.tool_calls)
                result = self._merge_results(result, step_result)
                if turn.done or not turn.tool_calls:
                    break
                if any(self.tools.spec(call.name).side_effect != "none" for call in turn.tool_calls):
                    break
                if step == MAX_PARENT_AGENT_STEPS:
                    result["message"] = result.get("message") or (
                        "I found the earlier thread context but still need a few words about which part to change."
                    )
                    break
                request["tool_results"] = [
                    self._tool_result_for_model(item)
                    for item in result.get("tool_results", [])[-MAX_PARENT_AGENT_TOOL_CALLS:]
                ]
            self._store_release_units(run_id, result)
            self.sessions.finish_turn(
                turn_row["id"],
                response=result,
                provider=self.backend.name,
                model=self.backend.model,
            )
            self._finish_capability_run(run_id, result)
            if result.get("message"):
                result_status = str(result.get("status") or "completed")
                self.sessions.append_message(
                    session_id,
                    role="assistant",
                    content=str(result["message"]),
                    kind="agent_response" if result_status == "completed" else "agent_notice",
                    event_id=result.get("event_id") if result_status == "completed" else None,
                    metadata={"capability_run_id": run_id},
                )
            return {**result, "session_id": session_id, "capability_run_id": run_id}
        except Exception as exc:
            self.sessions.finish_turn(
                turn_row["id"],
                response={"error_type": exc.__class__.__name__},
                provider=self.backend.name,
                model=self.backend.model,
                status="failed",
            )
            with connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE capability_runs SET status='failed',result_json=?,updated_at=? WHERE id=?",
                    (jdump({"error_type": exc.__class__.__name__}), utcnow(), run_id),
                )
            raise

    def _plan(self, request: dict[str, Any], allowed_tools: set[str]) -> ParentAgentTurn:
        if self.backend.name == "deterministic":
            return self._fallback_plan(
                str(request["message"]), bool(request.get("current_attachment_context"))
            )
        system = (
            "You are the parent-conversation planner inside Curiosity Engine. Parents speak naturally; never require "
            "commands or perfectly structured questions. Read the thread history. Choose the smallest useful next "
            "step and call only reviewed tools shown in the payload. A parent turn is not automatically a lesson. "
            "Use record_thread_context when the parent shares a photo, observation, or play context without asking "
            "for an output. Use continue_learning_thread for a genuine child-learning follow-up question, "
            "revise_learning_thread when the parent asks to change a prior answer, activity, or visual, "
            "create_learning_artifact when they ask for a worksheet/activity/challenge/printable, and "
            "propose_weekly_checkin for recurring check-ins. If the parent asks why the engine behaved a certain "
            "way, corrects what it received, or asks about its capabilities, answer directly in one to four sentences "
            "with no tool unless a local record is needed. Never generate a lesson or visual merely to apologize or "
            "explain the interface. Do not repeat a prior activity. A critique of a printable, worksheet, activity "
            "sheet, or challenge artifact must call "
            "create_learning_artifact again with the full parent wording in revision; revise_learning_thread is only "
            "for the conversational answer or its generated image. A model-written message may acknowledge briefly, "
            "but learning work must use a tool. The workspace contains code-owned recent-output references and "
            "explicit preferences for this thread. When an older target is unclear, call search_thread_outputs by "
            "itself with done=false; its bounded results will be returned for one final planning pass. Never set "
            "done=false after a write or request more than one lookup pass. Use update_thread_preference only when "
            "the parent explicitly requests future behavior in this thread; a one-off critique is not a preference. "
            "Block-style interactions are optional "
            "shortcuts and free-form chat must remain allowed. Never invent child IDs, permissions, or tool names.\n\n"
            + self.capabilities.instructions_for("parent_chat")
        )
        candidate = self.backend.complete(
            role="reasoning",
            system=system,
            payload=request,
            response_model=ParentAgentTurn,
        )
        parsed = ParentAgentTurn.model_validate(candidate)
        parsed = self._normalize_context_capture(parsed, request)
        parsed = self._normalize_artifact_revision(parsed, request)
        invalid = [call.name for call in parsed.tool_calls if call.name not in allowed_tools]
        if invalid:
            raise ValueError(f"parent agent proposed unavailable tools: {', '.join(invalid)}")
        return parsed

    @staticmethod
    def _normalize_context_capture(
        turn: ParentAgentTurn, request: dict[str, Any]
    ) -> ParentAgentTurn:
        """A plain photo/context share must not turn into an unsolicited mini-lesson."""

        if not request.get("current_attachment_context"):
            return turn
        message = str(request.get("message") or "").strip().casefold()
        requests_output = bool(
            "?" in message
            or re.search(
                r"\b(?:why|how|what|when|where|who|can|could|would|should|explain|help|ideas?|make|create|show|tell|give|suggest)\b",
                message,
            )
        )
        if requests_output:
            return turn
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="record_thread_context",
                    arguments={"note": str(request.get("message") or "")},
                    rationale="The parent shared photo context without requesting a learning output.",
                )
            ]
        )

    def _normalize_artifact_revision(
        self,
        turn: ParentAgentTurn,
        request: dict[str, Any],
    ) -> ParentAgentTurn:
        """Keep a model routing slip from silently revising prose instead of a named printable."""

        message = str(request.get("message") or "")
        lowered = message.casefold()
        artifact_terms = ("printable", "worksheet", "activity sheet", "challenge sheet", "pdf")
        if not any(term in lowered for term in artifact_terms):
            return turn
        if not any(call.name == "revise_learning_thread" for call in turn.tool_calls):
            return turn
        artifact_type = next(
            (kind for kind in ("worksheet", "activity", "challenge") if kind in lowered),
            None,
        )
        if artifact_type is None:
            session_id = str((request.get("session") or {}).get("id") or "")
            with connect(self.db_path) as conn:
                row = conn.execute(
                    """SELECT arguments_json FROM tool_calls
                       WHERE session_id=? AND tool_name='create_learning_artifact'
                       ORDER BY ordinal DESC,created_at DESC LIMIT 1""",
                    (session_id,),
                ).fetchone()
            if row:
                prior_arguments = json.loads(row["arguments_json"])
                prior_kind = str(prior_arguments.get("artifact_type") or "").casefold()
                if prior_kind in {"worksheet", "activity", "challenge"}:
                    artifact_type = prior_kind
        artifact_type = artifact_type or "challenge"
        calls = [
            ParentAgentToolCall(
                name="create_learning_artifact",
                arguments={
                    "artifact_type": artifact_type,
                    "revision": message,
                    **(
                        {"target_ref": str(call.arguments["target_ref"])}
                        if call.arguments.get("target_ref")
                        else {}
                    ),
                },
                rationale="The parent is revising the current printable artifact.",
            )
            if call.name == "revise_learning_thread"
            else call
            for call in turn.tool_calls
        ]
        return turn.model_copy(update={"tool_calls": calls})

    @staticmethod
    def _fallback_plan(message: str, has_current_attachments: bool = False) -> ParentAgentTurn:
        lowered = message.casefold()
        if has_current_attachments and not (
            "?" in message
            or re.search(
                r"\b(?:why|how|what|when|where|who|can|could|would|should|explain|help|ideas?|make|create|show|tell|give|suggest)\b",
                lowered,
            )
        ):
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="record_thread_context",
                        arguments={"note": message},
                        rationale="The parent shared photo context without asking for an output.",
                    )
                ]
            )
        artifact = next((kind for kind in ("worksheet", "activity", "challenge") if kind in lowered), None)
        if artifact or "printable" in lowered:
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="create_learning_artifact",
                        arguments={"artifact_type": artifact or "challenge", "revision": message},
                        rationale="The parent asked for a child-facing artifact.",
                    )
                ]
            )
        if re.search(r"\b(?:weekly|every week|each week|sunday|saturday|monday|tuesday|wednesday|thursday|friday)\b", lowered) and "check" in lowered:
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="propose_weekly_checkin",
                        arguments={"request": message},
                        rationale="The parent asked for a recurring check-in.",
                    )
                ]
            )
        revision_terms = ("instead", "isn't", "is not", "not helpful", "change", "make it", "diagram", "visual", "picture", "shorter", "harder", "easier", "don't repeat", "do not repeat")
        if any(term in lowered for term in revision_terms):
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="revise_learning_thread",
                        arguments={"revision": message},
                        rationale="The parent is tuning the current thread.",
                    )
                ]
            )
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="continue_learning_thread",
                    arguments={"message": message},
                    rationale="The parent is continuing the learning conversation.",
                )
            ]
        )

    def _execute_turn(
        self,
        turn: ParentAgentTurn,
        *,
        session_id: str,
        turn_id: str,
        origin: str,
        call_ordinal: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "completed",
            "message": turn.message,
            "tool_results": [],
        }
        if turn.interaction:
            result["interaction"] = turn.interaction.model_dump(mode="json")
        for index, call in enumerate(turn.tool_calls, start=call_ordinal):
            spec = self.tools.spec(call.name)
            decision = ToolPolicy.decide(spec, origin=origin)
            if decision.requires_approval:
                tool_result = {
                    "status": "awaiting_approval",
                }
                status = "awaiting_approval"
                stored_decision = "awaiting_approval"
            else:
                tool_result = self.tools.execute(call.name, call.arguments, origin=origin)
                status = "succeeded" if tool_result.get("status") not in {"failed", "denied"} else tool_result["status"]
                stored_decision = "allowed" if decision.allowed else "denied"
            call_id = self.sessions.record_tool_call(
                session_id=session_id,
                turn_id=turn_id,
                ordinal=index,
                tool_name=call.name,
                tool_version=spec.version,
                arguments=call.arguments,
                risk=spec.risk,
                decision=stored_decision,
                status=status,
                result=tool_result,
            )
            if decision.requires_approval:
                scope_hash = sha256(
                    f"{session_id}:{call_id}:{call.name}:{sha256(jdump(call.arguments).encode()).hexdigest()}".encode()
                ).hexdigest()
                now = datetime.now(UTC)
                with connect(self.db_path) as conn:
                    conn.execute(
                        """INSERT INTO approval_requests(id,tool_call_id,scope_hash,expires_at,created_at)
                           VALUES(?,?,?,?,?)""",
                        (
                            f"approve_{uuid4().hex[:20]}",
                            call_id,
                            scope_hash,
                            (now + timedelta(minutes=30)).isoformat(),
                            now.isoformat(),
                        ),
                    )
                tool_result["interaction"] = InteractionPlan(
                    kind="confirm_action",
                    title="Schedule this check-in?",
                    prompt="I can set this up as a recurring family check-in. Nothing will be scheduled until you confirm.",
                    options=[
                        InteractionOption(
                            label="Schedule it",
                            intent="approve_tool_call",
                            payload={"tool_call_id": call_id},
                            style="primary",
                        ),
                        InteractionOption(
                            label="Not now",
                            intent="cancel_tool_call",
                            payload={"tool_call_id": call_id},
                        ),
                    ],
                ).model_dump(mode="json")
                with connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE tool_calls SET result_json=?,updated_at=? WHERE id=?",
                        (jdump(tool_result), utcnow(), call_id),
                    )
            result["tool_results"].append({"tool": call.name, **tool_result})
            tool_status = str(tool_result.get("status") or "completed")
            if tool_status != "completed":
                result["status"] = tool_status
            for key in ("message", "interaction", "event_id", "visual_job_id", "artifact", "schedule"):
                if tool_result.get(key) is not None:
                    result[key] = tool_result[key]
        notices = [str(item["notice"]) for item in result["tool_results"] if item.get("notice")]
        if notices and result.get("message") not in notices:
            result["message"] = (str(result.get("message") or "") + "\n\n" + "\n".join(notices)).strip()
        return result

    @staticmethod
    def _merge_results(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        merged = {**current}
        merged["tool_results"] = [
            *current.get("tool_results", []),
            *update.get("tool_results", []),
        ]
        if update.get("status") != "completed":
            merged["status"] = update["status"]
        for key in ("message", "interaction", "event_id", "visual_job_id", "artifact", "schedule"):
            if update.get(key) is not None:
                merged[key] = update[key]
        return merged

    @staticmethod
    def _tool_result_for_model(result: dict[str, Any]) -> dict[str, Any]:
        """Return only the bounded fields needed for a second planning pass."""

        safe = {
            key: result[key]
            for key in ("tool", "status", "matches", "operation", "category", "preferences", "message")
            if result.get(key) is not None
        }
        if isinstance(result.get("artifact"), dict):
            artifact = result["artifact"]
            safe["artifact"] = {
                key: artifact[key]
                for key in ("artifact_id", "artifact_type", "title")
                if artifact.get(key) is not None
            }
        return safe

    def _start_capability_run(self, session_id: str, source_event_id: str | None) -> str:
        run_id = f"cap_{uuid4().hex[:20]}"
        now = utcnow()
        with connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO capability_runs(
                       id,session_id,capability_id,capability_version,skill_versions_json,status,source_event_id,
                       created_at,updated_at
                   ) VALUES(?,?,?,?,?,'running',?,?,?)""",
                (
                    run_id,
                    session_id,
                    "parent_chat",
                    "2",
                    jdump({card["id"]: card["version"] for card in self.capabilities.skill_cards()}),
                    source_event_id,
                    now,
                    now,
                ),
            )
        return run_id

    def _store_release_units(self, run_id: str, result: dict[str, Any]) -> None:
        units: list[tuple[str, dict[str, Any]]] = []
        if result.get("message"):
            units.append(("answer", {"text": result["message"], "event_id": result.get("event_id")}))
        if result.get("interaction"):
            units.append(("interaction", result["interaction"]))
        if result.get("artifact"):
            units.append(("artifact", result["artifact"]))
        if result.get("visual_job_id"):
            units.append(("visual", {"visual_job_id": result["visual_job_id"]}))
        now = utcnow()
        with connect(self.db_path) as conn:
            for ordinal, (unit_type, payload) in enumerate(units, start=1):
                conn.execute(
                    """INSERT INTO release_units(
                           id,capability_run_id,unit_type,ordinal,status,content_hash,payload_json,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (
                        f"rel_{uuid4().hex[:20]}",
                        run_id,
                        unit_type,
                        ordinal,
                        "reviewed",
                        sha256(jdump(payload).encode()).hexdigest(),
                        jdump(payload),
                        now,
                        now,
                    ),
                )

    def _finish_capability_run(self, run_id: str, result: dict[str, Any]) -> None:
        status = "rejected" if result.get("status") == "rejected" else "completed"
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE capability_runs SET status=?,result_json=?,updated_at=? WHERE id=?",
                (status, jdump(result), utcnow(), run_id),
            )
