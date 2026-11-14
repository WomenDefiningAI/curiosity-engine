from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from .attachments import source_photo_context
from .capabilities import CapabilityRegistry
from .contracts import (
    AnswerMode,
    InteractionOption,
    InteractionPlan,
    ParentAgentToolCall,
    ParentAgentTurn,
)
from .db import connect, init_db, jdump, jload, utcnow
from .reasoning import ModelBackend
from .scheduler import weekly_checkin_preview
from .sessions import SessionStore
from .tooling import ToolPolicy, ToolRegistry

PARENT_AGENT_POLICY_VERSION = "parent-agent-v6-authority-bound"
MAX_PARENT_AGENT_STEPS = 2
MAX_PARENT_AGENT_TOOL_CALLS = 3


def _explicit_learning_output_request(message: str) -> bool:
    """Identify a parent's request for work, not questions quoted inside shared context."""

    lowered = message.strip().casefold()
    return _explicit_artifact_request(message) or bool(
        re.search(r"\b(?:can|could|would|will)\s+you\b", lowered)
        or re.search(
            r"\b(?:make|create|generate|build|explain|show|tell|give|suggest|help)\s+(?:me|us|him|her|them|a|an|the|this)\b",
            lowered,
        )
        or re.search(
            r"\b(?:what|how|why|where|when|who)\s+(?:can|could|should|would|do|does|did|is|are)\s+(?:we|i|you)\b",
            lowered,
        )
        or re.search(r"\b(?:ideas|suggestions|thoughts)\s*\?\s*$", lowered)
        or re.search(r"\bturn\s+(?:this|it)\s+into\b", lowered)
        or re.search(r"\b(?:please\s+)?(?:use|include|add|remove|put|move|label)\b", lowered)
        or re.search(
            r"\bi\s+(?:need|want|would like)\b.{0,100}\b(?:answer|explanation|sentence|line|wording|"
            r"description|definition|way to say)\b",
            lowered,
        )
    )


_ARTIFACT_TERM = re.compile(
    r"\b(?:worksheet|printable|pdf|diagram|recording aid|recording sheet|fact sheet|practice page|activity sheet|"
    r"activity card|instruction card|step card|results box|picture checklist|visual checklist|comparison page|"
    r"child-facing page|checklist|challenge (?:card|sheet|page))\b",
    re.IGNORECASE,
)


def _explicit_artifact_request(message: str) -> bool:
    """Require a positive request for a tangible child-facing output."""

    lowered = message.casefold()
    if re.search(
        r"\b(?:make|create|generate|build|give (?:me|us))\b.{0,45}\bchallenge\b.{0,45}"
        r"\b(?:from|about|for|using)\b",
        lowered,
    ) and not re.search(r"\b(?:instead of|rather than|not another)\b", lowered):
        return True
    for match in _ARTIFACT_TERM.finditer(lowered):
        before = lowered[max(0, match.start() - 32) : match.start()]
        after = lowered[match.end() : match.end() + 28]
        if re.search(r"\b(?:no|not|without|don't|do not|isn't|is not)\b[^.!?]{0,24}$", before):
            continue
        if re.match(r"[^.!?]{0,18}\b(?:isn't|is not|wasn't|was not)\b", after):
            continue
        return True
    return False


def _explicit_artifact_generation_request(message: str) -> bool:
    """Distinguish making an artifact from merely discussing whether one would help."""

    if not _explicit_artifact_request(message):
        return False
    lowered = message.casefold()
    return bool(
        (
            re.search(
                r"\b(?:make|create|generate|build|give (?:me|us))\b.{0,45}\bchallenge\b.{0,45}"
                r"\b(?:from|about|for|using)\b",
                lowered,
            )
            and not re.search(r"\b(?:instead of|rather than|not another)\b", lowered)
        )
        or
        re.search(
            r"\b(?:can|could|would)\s+you\b.{0,90}\b(?:worksheet|printable|diagram|activity card|"
            r"instruction card|picture checklist|visual checklist|checklist|comparison page|pdf|"
            r"challenge (?:card|sheet|page))\b",
            lowered,
        )
        or re.search(
            r"\b(?:make|create|generate|build|turn)\b.{0,90}\b(?:worksheet|printable|diagram|activity card|"
            r"instruction card|picture checklist|visual checklist|checklist|comparison page|pdf|"
            r"challenge (?:card|sheet|page))\b",
            lowered,
        )
        or re.search(
            r"\bi\s+(?:want|need|would like)\b.{0,90}\b(?:worksheet|printable|diagram|activity card|"
            r"instruction card|picture checklist|visual checklist|checklist|comparison page|pdf|"
            r"challenge (?:card|sheet|page))\b",
            lowered,
        )
    )


def _requested_artifact_type(message: str) -> str:
    lowered = message.casefold()
    if re.search(r"\b(?:diagram|comparison page)\b", lowered):
        return "reference"
    if re.search(
        r"\b(?:activity card|instruction card|step card|results box|picture checklist|visual checklist|checklist)\b",
        lowered,
    ):
        return "activity"
    if re.search(r"\bchallenge\b", lowered):
        return "challenge"
    return "worksheet"


def _looks_like_output_feedback(message: str) -> bool:
    lowered = message.casefold()
    return bool(
        re.search(r"\b(?:last|earlier|previous)\s+(?:answer|explanation|response|idea|activity|visual)\b", lowered)
        and re.search(r"\b(?:helpful|unhelpful|worked|didn't work|did not work|heard|listen|busy)\b", lowered)
    )


def _looks_like_ambiguous_shared_fragment(message: str) -> bool:
    lowered = message.casefold()
    return bool(
        re.search(r"[“\"].{1,80}(?:…|\.\.\.|[”\"])", message)
        and re.search(r"\b(?:not sure|unsure|couldn't tell|could not tell|can't tell|cannot tell)\b", lowered)
        and re.search(r"\b(?:mean|meant|asking|pointing|talking about)\b", lowered)
    )


def _looks_like_parent_reported_reaction(message: str) -> bool:
    """Recognize a brief reaction update that is not a fresh request for teaching."""

    lowered = message.casefold()
    if _explicit_learning_output_request(message):
        return False
    return bool(
        re.search(
            r"\b(?:listened|heard (?:that|it)|nodded|paused|thought about it)\b.{0,45}"
            r"\b(?:then\s+)?(?:said|replied|responded|reacted)\b|"
            r"\b(?:her|his|their)\s+(?:reaction|response)\s+(?:was|is)\b",
            lowered,
        )
    )


def _looks_like_note_handling_question(message: str) -> bool:
    """Recognize operational questions about saving notes versus answering them."""

    lowered = message.casefold()
    note_terms = bool(re.search(r"\b(?:note|notes|save|saved|remember|keep track|context)\b", lowered))
    asks_how = bool(
        re.search(
            r"\b(?:how|what kinds? of things|what)\b.{0,80}\b(?:save|saved|remember|keep track|"
            r"handle|retained?)\b|"
            r"\bcan you\b.{0,80}\b(?:save|remember|keep track|tell me whether)\b",
            lowered,
        )
    )
    return note_terms and asks_how


def _looks_like_underspecified_reference(message: str) -> bool:
    """Spot a child quote whose object is still unknown, so we clarify instead of guessing."""

    lowered = message.casefold()
    has_child_quote = bool(re.search(r"[“\"].{2,100}[”\"]", message))
    parent_lacks_referent = bool(
        re.search(
            r"\b(?:did not|didn't|could not|couldn't)\s+(?:see|hear|catch|notice)\s+(?:it|that)|"
            r"\b(?:something|some thing)\s+(?:at|in|near|by)\b|"
            r"\b(?:not sure|unsure)\b.{0,45}\b(?:what|which|where|means?|meant)\b",
            lowered,
        )
    )
    generic_quote = bool(
        re.search(
            r"[“\"][^”\"]{0,80}\b(?:it|they|them|this|that|thing|things|button|buttons)\b[^”\"]*[”\"]",
            message,
            re.IGNORECASE,
        )
    )
    return has_child_quote and parent_lacks_referent and generic_quote


def _explicit_knowledge_visual_request(message: str) -> bool:
    """Route a self-contained explanatory visual to a reviewed reference artifact."""

    lowered = message.casefold()
    if re.search(r"\bpicture\s+(?:tag|symbol|label|icon|mark)\b", lowered) and not re.search(
        r"\b(?:make|create|draw|show)\b.{0,55}\b(?:visual|diagram|image)\b|"
        r"\b(?:visual|diagram|image)\b.{0,30}\b(?:clearer|better|simpler|more useful)\b",
        lowered,
    ):
        return False
    asks_to_make = bool(re.search(r"\b(?:can|could|would)\s+you\b|\b(?:make|create|draw|show)\b", lowered))
    visual = bool(re.search(r"\b(?:visual|diagram|picture)\b", lowered))
    knowledge_shape = bool(
        re.search(
            r"\b(?:life[- ]?cycle|stages?|sequence|step[- ]by[- ]step|parts?|inside|outside|"
            r"compare|comparison|side[- ]by[- ]side|mechanism|how .{0,55}(?:works?|gets?|moves?|"
            r"travels?|flows?|changes?))\b",
            lowered,
        )
    )
    return asks_to_make and visual and knowledge_shape


def _one_reply_override(message: str) -> bool:
    return bool(
        re.search(
            r"\b(?:just|only)\s+(?:for\s+)?(?:this|the)\s+(?:next\s+)?reply\b|"
            r"\bfor\s+(?:this|the)\s+next\s+(?:reply|answer)\b|\bone[- ]reply\s+exception\b",
            message.casefold(),
        )
    )


def _weekly_checkin_draft(message: str, child: dict[str, Any] | None = None) -> str:
    """Create a visible one-question preview without scheduling anything."""

    compact = " ".join(message.split()).strip()
    raw_name = str((child or {}).get("name") or "").strip()
    if not raw_name:
        name_match = re.search(r"\bfor\s+([A-Z][A-Za-z'’-]{1,40})\b", compact)
        raw_name = name_match.group(1) if name_match else ""
    name = raw_name.split()[0] if raw_name else "your child"
    return weekly_checkin_preview(compact, name)


def _looks_like_shared_context(message: str) -> bool:
    """Recognize common wrappers around school updates and other parent observations."""

    lowered = message.strip().casefold()
    return bool(
        re.search(r"\b(?:fyi|for context|sharing|forwarding|pasting|just so you know)\b", lowered)
        or re.search(
            r"\b(?:email|message|note|update|report|report card|newsletter)\s+(?:from|about)\b",
            lowered,
        )
        or re.search(
            r"\b(?:school\s+)?teacher(?:['’]s)?\s+(?:email|message|note|update|report)\b",
            lowered,
        )
        or re.search(
            r"\b(?:teacher|school|tutor|coach|therapist|caregiver)\b.{0,100}\b(?:said|sent|shared|emailed|wrote|noted|reported|mentioned|observed)\b",
            lowered,
        )
    )


def _looks_like_child_learning_request(message: str, child: dict[str, Any] | None) -> bool:
    """Recognize only strong learning intent that must pass through reviewed generation."""

    if not child or (_looks_like_shared_context(message) and not _explicit_learning_output_request(message)):
        return False
    lowered = message.strip().casefold()
    if re.search(
        r"\b(?:write|draft|rewrite)\b.{0,80}\b(?:note|message|email)\b.{0,80}"
        r"\b(?:teacher|coach|instructor|school|caregiver)\b",
        lowered,
    ):
        return False
    if re.search(r"\b(?:what can i say|words? (?:i|we) can say)\b", lowered) and re.search(
        r"\b(?:nervous|worried|scared|afraid|upset|frustrated|anxious|does not want|doesn't want|refuses?)\b",
        lowered,
    ):
        return False
    if re.search(r"\b(?:asked|asks|asking|wonders|wondered|wondering|wants? to know|curious about)\b", lowered):
        return True
    if "?" not in message:
        return False
    if re.search(
        r"\b(?:you|your|bot|engine|slack|reply|response|message|working|online|setup|privacy|context graph)\b",
        lowered,
    ):
        return False
    return bool(re.search(r"\b(?:why|how|what|when|where|who|which|can|could|does|do|is|are)\b", lowered))


def _looks_like_parent_observation(message: str, child: dict[str, Any] | None) -> bool:
    """Recognize a reported child moment without converting it into a requested lesson."""

    if not child or "?" in message or re.search(
        r"\b(?:asked|asks|asking|wonders|wondered|wondering|wants? to know)\b", message, re.I
    ):
        return False
    name = " ".join(str(child.get("name") or "").split())
    first_name = name.split()[0] if name else ""
    if not first_name or not re.search(rf"(?<![\w]){re.escape(first_name)}(?![\w])", message, re.I):
        return False
    return bool(
        re.search(
            r"\b(?:has been|have been|is|was|were|keeps?|started|loves?|likes?|made|built|drew|found|noticed|"
            r"brought|tried|practiced|read|watched|listened|played|wants?)\b",
            message,
            re.IGNORECASE,
        )
    )


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
        current_message_id: str | None = None,
        origin: str = "slack",
        current_attachment_context: list[dict[str, Any]] | None = None,
        input_mode: str = "conversation",
        answer_mode_hint: str = AnswerMode.AUTO.value,
    ) -> dict[str, Any]:
        if input_mode not in {"conversation", "context"}:
            raise ValueError("parent agent input_mode must be conversation or context")
        answer_mode_hint = AnswerMode(answer_mode_hint).value
        continuity = self.sessions.compacted_context(session_id, compact_after=16, keep_recent=6)
        history = continuity["history"]
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
            "input_mode": input_mode,
            "answer_mode_hint": answer_mode_hint,
            "current_attachment_context": current_attachment_context or [],
            "source_photo_context": source_photo_context(self.db_path, latest_event_id),
            "workspace": {
                "thread_preferences": self.sessions.active_preferences(session_id),
                "parent_memories": self.sessions.active_parent_memories(session_id),
                "thread_compaction": continuity.get("runtime_summary"),
                "compacted_current_intent": continuity.get("resolved_current_intent"),
                "recent_outputs": self.sessions.output_index(session_id, limit=10),
            },
            "capabilities": self.capabilities.capability_cards(),
            "skills": self.capabilities.skill_cards(set(capability.skill_ids)),
            "tools": self.tools.specs(allowed_tools),
        }
        policy_hash = sha256(
            (PARENT_AGENT_POLICY_VERSION + jdump(request["tools"]) + jdump(request["skills"])).encode()
        ).hexdigest()[:20]
        durable_request = {
            **request,
            "source_photo_context": [],
            "source_photo_context_event_id": latest_event_id,
            "workspace": {
                **request["workspace"],
                "parent_memories": [
                    {
                        key: item[key]
                        for key in ("memory_key", "version", "scope_type", "category")
                        if item.get(key) is not None
                    }
                    for item in request["workspace"]["parent_memories"]
                ],
                "compacted_current_intent": (
                    {
                        "message_id": request["workspace"]["compacted_current_intent"][
                            "message_id"
                        ]
                    }
                    if request["workspace"].get("compacted_current_intent")
                    else None
                ),
            },
        }
        turn_row = self.sessions.start_turn(session_id, durable_request, policy_hash=policy_hash)
        run_id = self._start_capability_run(session_id, latest_event_id)
        lineage_event_ids: set[str] = set()
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
                    sha256(f"{call.name}:{jdump(call.arguments)}".encode()).hexdigest() for call in turn.tool_calls
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
                    capability_run_id=run_id,
                    source_message_id=current_message_id,
                    origin=origin,
                    call_ordinal=call_ordinal,
                )
                call_ordinal += len(turn.tool_calls)
                result = self._merge_results(result, step_result)
                for lineage_event_id in dict.fromkeys(step_result.get("_lineage_event_ids") or []):
                    lineage_event_ids.add(str(lineage_event_id))
                    self.sessions.bind_event_lineage(
                        session_id=session_id,
                        turn_id=str(turn_row["id"]),
                        capability_run_id=run_id,
                        event_id=str(lineage_event_id),
                        source_message_id=current_message_id,
                        primary=False,
                    )
                candidate_event_id = str(step_result.get("event_id") or "").strip()
                if candidate_event_id:
                    lineage_event_ids.add(candidate_event_id)
                    self.sessions.bind_event_lineage(
                        session_id=session_id,
                        turn_id=str(turn_row["id"]),
                        capability_run_id=run_id,
                        event_id=candidate_event_id,
                        source_message_id=current_message_id,
                        primary=True,
                    )
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
            self._store_release_units(run_id, result, lineage_event_ids=lineage_event_ids)
            self.sessions.finish_turn(
                turn_row["id"],
                response=result,
                provider=self.backend.name,
                model=self.backend.model,
            )
            self._finish_capability_run(run_id, result)
            if result.get("message"):
                result_status = str(result.get("status") or "completed")
                assistant_message_id = self.sessions.append_message(
                    session_id,
                    role="assistant",
                    content=str(result["message"]),
                    kind="agent_response" if result_status == "completed" else "agent_notice",
                    event_id=result.get("event_id") if result_status == "completed" else None,
                    metadata={"capability_run_id": run_id},
                )
                for lineage_event_id in lineage_event_ids:
                    self.sessions.bind_event_lineage(
                        session_id=session_id,
                        turn_id=str(turn_row["id"]),
                        capability_run_id=run_id,
                        event_id=lineage_event_id,
                        source_message_id=assistant_message_id,
                        primary=False,
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
            self._append_lifecycle(run_id, "failed")
            raise

    def _plan(self, request: dict[str, Any], allowed_tools: set[str]) -> ParentAgentTurn:
        if self.backend.name == "deterministic":
            parsed = self._fallback_plan(
                str(request["message"]),
                bool(request.get("current_attachment_context")),
                (request.get("session") or {}).get("child"),
            )
        else:
            system = (
                "You are the parent-conversation planner inside Curiosity Engine. Parents speak naturally; never require "
                "commands or perfectly structured questions. Read the thread history. Choose the smallest useful next "
                "step and call only reviewed tools shown in the payload. A parent turn is not automatically a lesson. "
                "Set requested_output=material when the parent wants an actual page, picture hunt, cards, chart, "
                "or other reusable visual material, however they phrase it, and call create_learning_artifact. "
                "Do not answer such a request with emoji, a text diagram, or instructions to copy a page. "
                "Set requested_output=conversation for spoken help, ordinary play, or a game without a requested "
                "material. An ordinary challenge using toys already out does not imply a document. "
                "A previous printable does not lock the conversation into editing it: answer new why, how, "
                "what-if questions and requests for ideas with continue_learning_thread. Moving a real object "
                "or changing an experiment is different from moving a picture on a page. "
                "The source_photo_context contains earlier photo observations, not the original pixels. "
                "If the newest request depends on an omitted or uncertain visual detail, or corrects the photo "
                "interpretation, set inspect_source_photos=true in the selected learning/artifact tool arguments. "
                "Code will inspect the original photos for this exact parent request before executing that tool. "
                "Do not request reinspection when the existing observations already answer the question. "
                "Use record_thread_context when the parent shares a photo, observation, teacher update, school email, "
                "schedule, or play context without asking for an output. Quoted questions inside an email are still "
                "context; do not treat them as the parent's request. If input_mode is context, always record context and "
                "acknowledge it briefly. Use continue_learning_thread only for a genuine child-learning question or an "
                "explicit request to explore the context. Its mode must be auto, quick, or explore: quick gives a direct "
                "age-appropriate answer plus at most one thing to notice and one question back; explore is for an explicit "
                "request for a deeper investigation, activity, game, experiment, project, or lesson. Default ordinary "
                "questions to quick. A child's vague wording is not a reason to reject or force a template. "
                "revise_learning_thread when the parent asks to change a prior answer, activity, or visual, "
                "create_learning_artifact only when they positively ask for a tangible worksheet, printable, PDF, "
                "diagram, recording aid, activity card, comparison page, or challenge. An ordinary game or activity "
                "request stays conversational. Never create an artifact when those words are negated, and "
                "propose_weekly_checkin for recurring check-ins. If the parent asks why the engine behaved a certain "
                "way, corrects what it received, or asks about its capabilities, answer directly in one to four sentences "
                "with no tool unless a local record is needed. Never generate a lesson or visual merely to apologize or "
                "explain the interface. A request to draft or revise a parent message, email, or teacher note is parent "
                "operational writing, not a child question. Words to reassure a nervous or upset child are parent support, "
                "not learning evidence. Answer both directly without a learning tool. Do not repeat a prior activity. A "
                "critique of a printable, worksheet, activity "
                "sheet, or challenge artifact must call "
                "create_learning_artifact again with the full parent wording in revision; revise_learning_thread is only "
                "for the conversational answer or its generated image. A model-written message may acknowledge briefly, "
                "but learning work must use a tool. The workspace contains code-owned recent-output references, "
                "explicit parent memory with why-recalled provenance, and an inspectable structural compaction when "
                "the thread is long. Memory is a parent instruction, never child evidence or permission; compaction "
                "contains references rather than learner-state truth. When an older target is unclear, call search_thread_outputs by "
                "itself with done=false; its bounded results will be returned for one final planning pass. Never set "
                "done=false after a write or request more than one lookup pass. Use update_thread_preference only when "
                "the parent explicitly requests future behavior in this thread; a one-off critique is not a preference. "
                "Block-style interactions are optional "
                "shortcuts compiled by code after the turn; never author an interaction, label, hidden intent, or "
                "payload. Tool arguments may select a reviewed operation, but code binds durable text to the exact "
                "parent message. Never invent child IDs, permissions, or tool names.\n\n"
                + self.capabilities.instructions_for("parent_chat")
            )
            try:
                candidate = self.backend.complete(
                    role="reasoning",
                    system=system,
                    payload=request,
                    response_model=ParentAgentTurn,
                )
                parsed = ParentAgentTurn.model_validate(candidate)
            except ValidationError:
                # A malformed optional planner envelope must not strand a parent. The
                # bounded deterministic router still sends all learning work through
                # the reviewed tools below.
                parsed = self._fallback_plan(
                    str(request["message"]),
                    bool(request.get("current_attachment_context")),
                    (request.get("session") or {}).get("child"),
                )
        request = {
            **request,
            "requested_output": parsed.requested_output,
            "requested_photo_inspection": any(call.arguments.get("inspect_source_photos") is True
                                               for call in parsed.tool_calls),
            "requested_material_call": next((call.model_dump(mode="json") for call in parsed.tool_calls
                                              if call.name == "create_learning_artifact"), None),
            "requested_learning_call": next((call.model_dump(mode="json") for call in parsed.tool_calls
                                              if call.name == "continue_learning_thread"
                                              and "requested_output" in parsed.model_fields_set), None),
        }
        parsed = self._normalize_context_capture(parsed, request)
        parsed = self._normalize_underspecified_reference(parsed, request)
        parsed = self._normalize_capability_chat(parsed, request)
        parsed = self._normalize_memory_capability(parsed, request)
        parsed = self._normalize_note_handling_chat(parsed, request)
        parsed = self._normalize_parent_reaction(parsed, request)
        parsed = self._normalize_schedule_request(parsed, request)
        parsed = self._normalize_parent_approval(parsed, request)
        parsed = self._normalize_standalone_knowledge_visual(parsed, request)
        parsed = self._normalize_context_artifact(parsed, request)
        parsed = self._normalize_answer_mode(parsed, request)
        parsed = self._normalize_reviewed_learning(parsed, request)
        # A complete semantic decision owns ordinary learning intent. Phrase-based
        # repair is retained only for deterministic/older/malformed planners.
        semantic_learning = (
            self.backend.name != "deterministic"
            and "requested_output" in parsed.model_fields_set
            and any(call.name in {"continue_learning_thread", "revise_learning_thread", "create_learning_artifact"}
                    for call in parsed.tool_calls)
        )
        if not semantic_learning:
            parsed = self._normalize_lightweight_idea(parsed, request)
            parsed = self._normalize_thread_revision(parsed, request)
            parsed = self._normalize_artifact_revision(parsed, request)
        parsed = self._normalize_negative_feedback(parsed, request)
        parsed = self._normalize_artifact_intent(parsed, request)
        parsed = self._normalize_context_artifact(parsed, request)
        parsed = self._normalize_missing_visual_context(parsed, request)
        parsed = self._normalize_missing_revision_target(parsed, request)
        parsed = self._normalize_context_reaffirmation(parsed, request)
        parsed = self._normalize_one_reply_override(parsed, request)
        parsed = self._normalize_preference_with_learning(parsed, request)
        parsed = self._normalize_output_feedback(parsed, request)
        parsed = self._normalize_schedule_clarification(parsed, request)
        parsed = self._preserve_exact_revision_text(parsed, request)
        parsed = self._preserve_exact_schedule_request(parsed, request)
        # These parent-operational intents are final routing boundaries: later
        # revision heuristics must never turn them back into child learning work.
        parsed = self._normalize_context_recall(parsed, request)
        parsed = self._normalize_context_application(parsed, request)
        parsed = self._normalize_context_planning_chat(parsed, request)
        parsed = self._normalize_parent_operational_response(parsed, request)
        parsed = parsed.model_copy(update={"tool_calls": [
            call.model_copy(update={"arguments": {**call.arguments,
                "artifact_type": _requested_artifact_type(str(request["message"]))}})
            if call.name == "create_learning_artifact" and call.arguments.get("artifact_type")
            not in {"worksheet", "reference", "activity", "challenge"} else call
            for call in parsed.tool_calls
        ]})
        if request.get("requested_photo_inspection"):
            parsed = parsed.model_copy(update={"tool_calls": [
                call.model_copy(update={"arguments": {**call.arguments, "inspect_source_photos": True}})
                if call.name in {"continue_learning_thread", "revise_learning_thread", "create_learning_artifact"}
                else call for call in parsed.tool_calls
            ]})
        invalid = [call.name for call in parsed.tool_calls if call.name not in allowed_tools]
        if invalid:
            raise ValueError(f"parent agent proposed unavailable tools: {', '.join(invalid)}")
        return parsed

    @staticmethod
    def _normalize_underspecified_reference(
        turn: ParentAgentTurn,
        request: dict[str, Any],
    ) -> ParentAgentTurn:
        """Ask one useful question when the described object is not yet identifiable."""

        message = str(request.get("message") or "")
        if not _looks_like_underspecified_reference(message):
            return turn
        return ParentAgentTurn(
            message=(
                "What object or place was the child looking at, and which part did they point to? "
                "Once you know that, tell me their exact words and I’ll answer without guessing."
            )
        )

    @staticmethod
    def _normalize_capability_chat(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Give a bounded, useful capability overview without implying background autonomy."""

        message = str(request.get("message") or "")
        lowered = message.casefold()
        if not re.search(
            r"\b(?:what.{0,35}(?:can|could) you help|what (?:can|could) you (?:do|help)|"
            r"how can you help|your capabilities|what are you able to do)\b",
            lowered,
        ):
            return turn
        return ParentAgentTurn(
            message=(
                "I can answer a child’s why/how question, turn an everyday moment into one short game or "
                "investigation, make a child-facing diagram or printable, and revise an idea when it misses. "
                "I can also save parent-shared context separately from evidence about what a child knows. "
                "To save something, say, “Context only—please remember for later:” and then the exact observation, "
                "question, or constraint. You can also ask me to correct or forget it later. "
                "I only work while this service is running and when you message me or approve a scheduled "
                "check-in—I cannot watch your family, contact anyone, or act outside this chat on my own."
            )
        )

    @staticmethod
    def _normalize_memory_capability(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Answer preference-retention questions with the actual thread boundary."""

        message = str(request.get("message") or "")
        lowered = message.casefold()
        if not (
            re.search(r"\b(?:preference|way you should answer|format)\b", lowered)
            and re.search(r"\b(?:keep|save|remember|retain|repeat it each time|later)\b", lowered)
        ):
            return turn
        return ParentAgentTurn(
            message=(
                "Yes—in this family Slack thread, you can share a small parent preference once and I can use it "
                "in later replies here until you change or clear it. In a new or separate thread, repeat it so I "
                "don’t assume the same preference applies there."
            )
        )

    def _latest_context_text(self, request: dict[str, Any]) -> str:
        session_id = str((request.get("session") or {}).get("id") or "")
        if not session_id:
            return ""
        with connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT e.text FROM events e JOIN agent_messages m ON m.event_id=e.id
                   WHERE m.session_id=? AND e.type='context_input' AND e.status='completed'
                   ORDER BY m.ordinal DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
        return str(row["text"] if row else "").strip()

    def _normalize_context_recall(self, turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Recall the saved note faithfully without turning the request into child work."""

        message = str(request.get("message") or "")
        if not re.search(
            r"\b(?:what did you save|what (?:was|is) the (?:note|reminder)|what do you remember|"
            r"what did i (?:ask you to )?(?:save|remember))\b",
            message.casefold(),
        ):
            return turn
        saved = self._latest_context_text(request)
        if not saved:
            return ParentAgentTurn(message="I don’t have a saved context note in this thread yet.")
        return ParentAgentTurn(message=f"I saved this as context: “{saved}”")

    def _normalize_context_application(
        self,
        turn: ParentAgentTurn,
        request: dict[str, Any],
    ) -> ParentAgentTurn:
        """Turn an explicit future-use request into a real, reversible thread preference."""

        message = str(request.get("message") or "")
        if not re.search(
            r"\b(?:use|apply|follow)\s+(?:that|it)\b.{0,80}\b(?:later|future|when you suggest|next time)\b|"
            r"\bwhen you suggest\b.{0,80}\b(?:use|remember|keep)\b",
            message.casefold(),
        ):
            return turn
        saved = self._latest_context_text(request)
        value = re.split(
            r"\b(?:i\s+(?:only|just)\s+want|no\s+advice|not\s+advice)\b",
            saved,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,;:.-")
        value = re.sub(
            r"^(?:please\s+)?(?:remember|save|keep)\s+(?:that\s+)?",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()
        value = f"Use this parent-provided constraint for future suggestions: {value or message}"
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="update_thread_preference",
                    arguments={"operation": "set", "category": "activity_style", "value": value},
                    rationale="The parent explicitly asked that the saved accommodation guide future suggestions.",
                )
            ]
        )

    def _normalize_context_planning_chat(
        self,
        turn: ParentAgentTurn,
        request: dict[str, Any],
    ) -> ParentAgentTurn:
        """Discuss how one saved observation may inform planning without launching a lesson."""

        message = str(request.get("message") or "")
        if not re.search(
            r"\b(?:could|can|should|would)\s+(?:that|this|it)\b.{0,65}"
            r"\b(?:useful|help|inform|remember)\b.{0,65}\b(?:plan|planning|suggest|later|future)\b",
            message.casefold(),
        ):
            return turn
        if not self._latest_context_text(request):
            return turn
        return ParentAgentTurn(
            message=(
                "Yes—as one tentative planning clue, it can help me offer a related option later without making it "
                "an expectation. One moment does not prove a lasting preference, skill, or comfort level, and the "
                "child can always pass."
            )
        )

    def _normalize_parent_operational_response(
        self,
        turn: ParentAgentTurn,
        request: dict[str, Any],
    ) -> ParentAgentTurn:
        """Keep parent writing and emotional-support wording out of the learning/evidence path."""

        message = str(request.get("message") or "")
        history = request.get("history") or []
        history_text = "\n".join(str(item.get("content") or "") for item in history[-10:])
        writing_context = bool(
            re.search(
                r"\b(?:write|draft|rewrite)\b.{0,80}\b(?:note|message|email)\b.{0,80}"
                r"\b(?:teacher|coach|instructor|school|caregiver)\b",
                f"{history_text}\n{message}".casefold(),
            )
        )
        writing_revision = bool(
            writing_context
            and re.search(
                r"\b(?:formal|warmer|friendlier|plain words|shorter|longer|rewrite|make it|keep it|better)\b",
                message.casefold(),
            )
        )
        support_wording = bool(
            re.search(r"\b(?:what can i say|words? (?:i|we) can say|something (?:i|we) can say)\b", message.casefold())
            and re.search(
                r"\b(?:nervous|worried|scared|afraid|upset|frustrated|anxious|does not want|doesn't want|refuses?)\b",
                f"{history_text}\n{message}".casefold(),
            )
        )
        if not (writing_context and (writing_revision or re.search(r"\b(?:write|draft)\b", message.casefold())) or support_wording):
            return turn
        if self.backend.name == "deterministic":
            return ParentAgentTurn(message=message)
        try:
            candidate = self.backend.complete(
                role="reasoning",
                system=(
                    "Answer one parent operational request directly. Use the conversation history to draft or revise "
                    "the requested wording. Return only the usable parent-facing text or a brief acknowledgement when "
                    "that is what the newest turn asks for. Honor sentence-count, tone, and no-pressure boundaries. "
                    "Do not create a lesson, activity, visual, diagnosis, child trait, or claim that this is learning evidence."
                ),
                payload={"history": history[-10:], "newest_parent_message": message},
                response_model=ParentAgentTurn,
            )
            direct = ParentAgentTurn.model_validate(candidate)
            response = str(direct.message or "").strip()
            if not response or direct.tool_calls:
                return turn
        except Exception:
            return turn
        return ParentAgentTurn(message=response)

    @staticmethod
    def _normalize_note_handling_chat(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Explain a simple note/question convention without creating a durable preference."""

        message = str(request.get("message") or "")
        if not _looks_like_note_handling_question(message):
            return turn
        return ParentAgentTurn(
            message=(
                "Yes. In this family thread, start a context-only message with “Save this for later:” or “Just a "
                "note:” and I’ll keep it without turning it into a lesson. Ask a direct question when you want an "
                "answer; if a short message is ambiguous, I’ll ask rather than guess. I won’t take follow-up action "
                "unless you ask or approve it."
            )
        )

    @staticmethod
    def _normalize_parent_reaction(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Acknowledge a reported reaction proportionately instead of reteaching."""

        message = str(request.get("message") or "")
        if not _looks_like_parent_reported_reaction(message):
            return turn
        return ParentAgentTurn(
            message="Got it—I’ll keep that reaction with this conversation without turning it into another lesson."
        )

    @staticmethod
    def _normalize_schedule_request(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Keep recurring-routine planning operational and behind approval."""

        message = str(request.get("message") or "")
        lowered = message.casefold()
        recurring = bool(
            re.search(
                r"\b(?:weekly|every week|each week|every (?:monday|tuesday|wednesday|thursday|friday|"
                r"saturday|sunday)|recurring)\b",
                lowered,
            )
        )
        if not recurring or not re.search(r"\bcheck[- ]?in\b", lowered):
            return turn
        if re.search(r"\b(?:no thanks|cancel|do not start|don't start|never mind|nevermind)\b", lowered) and not re.search(
            r"\b(?:considering|what would|show me|preview|yet)\b",
            lowered,
        ):
            return ParentAgentTurn(message="Understood—I won’t start a recurring check-in.")
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="propose_weekly_checkin",
                    arguments={"request": message},
                    rationale="A recurring check-in is operational work and requires explicit parent approval.",
                )
            ]
        )

    @staticmethod
    def _normalize_parent_approval(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Keep presentation approval out of the child evidence path."""

        message = str(request.get("message") or "").strip()
        lowered = message.casefold()
        if not re.fullmatch(
            r"(?:yes[,! ]*)?(?:that|this|it) (?:version )?(?:works|is better|works better|looks good|is good)"
            r"(?: much better)?[.! ]*(?:thanks|thank you)?[.! ]*",
            lowered,
        ):
            return turn
        return ParentAgentTurn(message="You’re welcome—I’ll treat that as output feedback, not as evidence about your child.")

    @staticmethod
    def _normalize_standalone_knowledge_visual(
        turn: ParentAgentTurn,
        request: dict[str, Any],
    ) -> ParentAgentTurn:
        """A named explanatory visual is a reference artifact even without prior output."""

        message = str(request.get("message") or "")
        if not _explicit_knowledge_visual_request(message):
            return turn
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="create_learning_artifact",
                    arguments={"artifact_type": "reference", "revision": message},
                    rationale="The parent asked for a self-contained child-facing knowledge visual.",
                )
            ]
        )

    def _normalize_context_reaffirmation(
        self,
        turn: ParentAgentTurn,
        request: dict[str, Any],
    ) -> ParentAgentTurn:
        message = str(request.get("message") or "")
        if not re.search(
            r"\b(?:keep|leave)\b.{0,35}\b(?:background|context)\b.{0,20}\b(?:only|for now)\b|"
            r"\b(?:background|context)\s+only\b|"
            r"\b(?:just|only)\s+keep\s+(?:it|that)\b.{0,50}\b(?:one|simple)\s+note\b|"
            r"\bkeep\s+(?:it|that)\s+in mind\b.{0,60}\b(?:no|not|don't|do not)\b.{0,30}"
            r"\b(?:chart|report|plan|activity)\b",
            message.casefold(),
        ):
            return turn
        saved = self._latest_context_text(request)
        subject = f" The saved note remains: “{saved}”" if saved else ""
        return ParentAgentTurn(
            message=(
                "Yes—I’ll keep the original detail as one tentative context note and won’t turn it into a chart, "
                f"report, or activity unless you ask.{subject}"
            )
        )

    @staticmethod
    def _normalize_one_reply_override(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Apply a scoped exception now without mutating the stored thread preference."""

        message = str(request.get("message") or "")
        if not _one_reply_override(message):
            return turn
        lowered = message.casefold()
        mode = (
            AnswerMode.EXPLORE
            if re.search(r"\b(?:activity|game|experiment|challenge|project)\b", lowered)
            else AnswerMode.QUICK
        )
        latest_event_id = (request.get("session") or {}).get("latest_event_id")
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="revise_learning_thread" if latest_event_id else "continue_learning_thread",
                    arguments={"revision": message} if latest_event_id else {"message": message, "mode": mode.value},
                    rationale="The parent requested a one-reply exception; stored thread preferences remain unchanged.",
                )
            ]
        )

    @staticmethod
    def _normalize_missing_visual_context(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Do not regenerate an unknown picture or turn visual feedback into child evidence."""

        if (request.get("session") or {}).get("latest_event_id") or (request.get("workspace") or {}).get(
            "recent_outputs"
        ):
            return turn
        message = str(request.get("message") or "")
        lowered = message.casefold()
        if not re.search(r"\b(?:picture|visual|image|diagram)\b", lowered):
            return turn
        if not re.search(
            r"\b(?:too busy|confusing|unclear|not helpful|couldn't tell|could not tell|can't tell|cannot tell|"
            r"squiggles?|clearer)\b",
            lowered,
        ):
            return turn
        return ParentAgentTurn(
            message=(
                "I don’t have that earlier picture or visual—or its subject—here. Upload it, or tell me what it "
                "was meant to show. Then I can remake it with one clear focus using fewer labels, clearer arrows, "
                "or numbered steps."
            )
        )

    @staticmethod
    def _preserve_exact_revision_text(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Do not let the planner reinterpret a parent's concrete revision."""

        message = str(request.get("message") or "").strip()
        if not message or not any(call.name == "revise_learning_thread" for call in turn.tool_calls):
            return turn
        calls = [
            call.model_copy(
                update={
                    "arguments": {
                        "revision": message,
                        **(
                            {"target_ref": str(call.arguments["target_ref"])}
                            if call.arguments.get("target_ref")
                            else {}
                        ),
                    }
                }
            )
            if call.name == "revise_learning_thread"
            else call
            for call in turn.tool_calls
        ]
        return turn.model_copy(update={"tool_calls": calls})

    @staticmethod
    def _preserve_exact_schedule_request(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Keep the parent's proposed check-in wording visible through the approval boundary."""

        message = str(request.get("message") or "").strip()
        if not message or not any(call.name == "propose_weekly_checkin" for call in turn.tool_calls):
            return turn
        return turn.model_copy(
            update={
                "tool_calls": [
                    call.model_copy(update={"arguments": {"request": message}})
                    if call.name == "propose_weekly_checkin"
                    else call
                    for call in turn.tool_calls
                ]
            }
        )

    @staticmethod
    def _normalize_lightweight_idea(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Keep one-idea requests conversational unless the parent names an artifact."""

        if any(call.name == "propose_weekly_checkin" for call in turn.tool_calls):
            return turn
        message = str(request.get("message") or "").strip()
        lowered = message.casefold()
        if not re.search(
            r"\b(?:one|another)\b.{0,35}\bidea\b|\b(?:what|anything)\s+could\s+we\s+do\b|"
            r"\b(?:wants?|need|give|suggest|looking for)\b.{0,45}\bidea\b",
            lowered,
        ):
            return turn
        if re.search(
            r"\b(?:just (?:remember|save|note)|context only|background only|not asking|no answer|for later)\b",
            lowered,
        ):
            return turn
        if re.search(r"\b(?:worksheet|printable|activity card|challenge|pdf)\b", lowered):
            return turn
        latest_event_id = (request.get("session") or {}).get("latest_event_id")
        if latest_event_id:
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="revise_learning_thread",
                        arguments={"revision": message},
                        rationale="The parent asked for another idea within the current reviewed thread.",
                    )
                ]
            )
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="continue_learning_thread",
                    arguments={"message": message, "mode": AnswerMode.QUICK.value},
                    rationale="The parent asked for one proportionate conversational idea, not a full artifact.",
                )
            ]
        )

    @staticmethod
    def _normalize_negative_feedback(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Record concrete negative feedback and repair the current answer in one turn."""

        latest_event_id = (request.get("session") or {}).get("latest_event_id")
        if not latest_event_id:
            return turn
        message = str(request.get("message") or "").strip()
        if not re.search(
            r"\b(?:thumbs? down|not helpful|isn't helpful|is not helpful|did not make sense|didn't make sense|"
            r"too much reading|too long|too much)\b",
            message.casefold(),
        ):
            return turn
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="record_response_feedback",
                    arguments={"outcome": "not_helpful", "note": message},
                    rationale="The parent gave concrete negative output feedback.",
                ),
                ParentAgentToolCall(
                    name="revise_learning_thread",
                    arguments={"revision": message},
                    rationale="The same turn asks for a proportionate repair of the current answer.",
                ),
            ]
        )

    @staticmethod
    def _normalize_thread_revision(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Keep unmistakable tuning language attached to the existing reviewed output."""

        if any(call.name == "search_thread_outputs" for call in turn.tool_calls):
            return turn
        latest_event_id = (request.get("session") or {}).get("latest_event_id")
        if not latest_event_id:
            return turn
        message = str(request.get("message") or "").strip()
        lowered = message.casefold()
        unmistakable = bool(
            re.search(
                r"\b(?:better\s+(?:viz|visual|image|picture)|(?:viz|visual|image|picture|diagram)\s+please|"
                r"make it (?:shorter|harder|easier|more playful|less structured)|change (?:that|it)|"
                r"use (?:a|an) (?:diagram|picture|visual)|show.{0,30}(?:steps?|variations?)|"
                r"(?:image|visual|picture).{0,20}(?:isn't|is not|not).{0,10}helpful|"
                r"just answer|no activity|no lesson|keep (?:it|this) to (?:one|two|three|1|2|3) sentences?|"
                r"keep (?:it|this) to (?:about )?(?:\d+|one|two|three|four|five|ten) minutes?|ordinary paper|"
                r"answer key|directions?.{0,20}(?:wordy|shorter)|say (?:it|this) aloud|"
                r"(?:we|i)\s+(?:only|just)\s+have|use (?:only|just) (?:these|the)|"
                r"(?:materials?|supplies).{0,30}(?:are|:))\b",
                lowered,
            )
        )
        if not unmistakable:
            return turn
        explicitly_names_artifact = bool(
            re.search(r"\b(?:worksheet|printable|activity card|challenge|pdf)\b", lowered)
            and not re.search(
                r"\b(?:no|not|isn't|is not)\s+(?:a\s+)?(?:worksheet|printable|activity card|challenge|pdf)\b",
                lowered,
            )
        )
        if any(call.name == "revise_learning_thread" for call in turn.tool_calls):
            calls = [
                call.model_copy(
                    update={
                        "arguments": {
                            "revision": message,
                            **(
                                {"target_ref": str(call.arguments["target_ref"])}
                                if call.arguments.get("target_ref")
                                else {}
                            ),
                        }
                    }
                )
                if call.name == "revise_learning_thread"
                else call
                for call in turn.tool_calls
            ]
            return turn.model_copy(update={"tool_calls": calls})
        if explicitly_names_artifact and any(call.name == "create_learning_artifact" for call in turn.tool_calls):
            return turn
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="revise_learning_thread",
                    arguments={"revision": message},
                    rationale="The parent unmistakably asked to tune the current reviewed output.",
                )
            ]
        )

    @staticmethod
    def _normalize_missing_revision_target(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Recover source-dependent work when there is no reviewed source to target."""

        if (request.get("session") or {}).get("latest_event_id"):
            return turn
        if any(
            item.get("event_id") or item.get("artifact_id")
            for item in (request.get("workspace") or {}).get("recent_outputs") or []
        ):
            return turn
        current_message = str(request.get("message") or "")
        if any(call.name == "create_learning_artifact" for call in turn.tool_calls) and (
            _explicit_artifact_request(current_message) or request.get("requested_output") == "material"
            or _explicit_knowledge_visual_request(current_message)
        ):
            return turn
        needs_source = any(
            call.name in {"revise_learning_thread", "create_learning_artifact"} for call in turn.tool_calls
        )
        records_current_context = any(call.name == "record_thread_context" for call in turn.tool_calls)
        if not needs_source or records_current_context:
            return turn
        current = str(request.get("message") or "").strip()
        user_history = [
            str(item.get("content") or "").strip()
            for item in request.get("history") or []
            if item.get("role") == "user" and str(item.get("content") or "").strip() != current
        ]
        if not user_history:
            mode = (
                AnswerMode.EXPLORE
                if re.search(
                    r"\b(?:activity|game|mystery|story|challenge|experiment|project|lesson)\b",
                    current.casefold(),
                )
                else AnswerMode.QUICK
            )
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="continue_learning_thread",
                        arguments={"message": current, "mode": mode.value},
                        rationale="There is no earlier output to revise, so handle the current request as new learning work.",
                    )
                ]
            )
        prior = user_history[-1]
        mode = (
            AnswerMode.EXPLORE
            if re.search(r"\b(?:activity|game|mystery|story|challenge|experiment|project)\b", f"{prior} {current}".casefold())
            else AnswerMode.QUICK
        )
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="continue_learning_thread",
                    arguments={
                        "message": f"Original child question: {prior}\nParent follow-up: {current}",
                        "mode": mode.value,
                        "recovery_of_rejected": True,
                    },
                    rationale="The prior answer failed closed, so rebuild from the original question and parent repair.",
                )
            ]
        )

    @staticmethod
    def _normalize_artifact_intent(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Keep games conversational and honor explicit artifact negation."""

        message = str(request.get("message") or "").strip()
        learning_call = request.get("requested_learning_call")
        if (request.get("requested_output") == "conversation" and learning_call
                and request.get("input_mode") != "context"
                and any(item.name in {"revise_learning_thread", "create_learning_artifact"}
                        for item in turn.tool_calls)
                and not _explicit_artifact_generation_request(message)):
            # Preserve semantic continuation after legacy revision heuristics.
            # A prior picture plus words like "move", "space", or "before" is
            # not a request to regenerate that picture. Operational boundaries
            # still run after this normalization.
            context_calls = [item for item in turn.tool_calls if item.name == "record_thread_context"]
            other_calls = [item for item in turn.tool_calls if item.name not in {
                "record_thread_context", "continue_learning_thread", "revise_learning_thread",
                "create_learning_artifact",
            }]
            return turn.model_copy(update={"requested_output": "conversation", "tool_calls": [
                *context_calls, *other_calls, ParentAgentToolCall.model_validate(learning_call),
            ]})
        material_declined = bool(re.search(
            r"\b(?:no|without|don't|do not)\s+(?:any\s+)?(?:paper|print(?:ing|ables?)?|pdf|worksheets?|pictures?|visuals?)\b"
            r"(?!\s+(?:boxes|frames|borders))",
            message.casefold(),
        ))
        # The semantic planner recognizes requested materials beyond a phrase list.
        # Keep context capture and explicit no-material boundaries authoritative.
        if request.get("requested_output") == "material" and not material_declined and request.get("input_mode") != "context":
            call = request.get("requested_material_call") or {
                "name": "create_learning_artifact",
                "arguments": {"artifact_type": _requested_artifact_type(message), "revision": message},
                "rationale": "Deliver the material requested by the parent.",
            }
            context_calls = [item for item in turn.tool_calls if item.name == "record_thread_context"]
            return turn.model_copy(update={
                "requested_output": "material",
                "tool_calls": [*context_calls, ParentAgentToolCall.model_validate(call)],
            })
        if not any(call.name == "create_learning_artifact" for call in turn.tool_calls):
            if not _explicit_artifact_generation_request(message):
                return turn
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="create_learning_artifact",
                        arguments={
                            "artifact_type": _requested_artifact_type(message),
                            "revision": message,
                        },
                        rationale="The parent explicitly asked the engine to make a tangible child-facing artifact.",
                    )
                ]
            )
        if (
            (request.get("session") or {}).get("latest_event_id")
            and re.search(r"\bchallenge\b", message.casefold())
            and re.search(r"\b(?:revise|instead|turn|make)\b", message.casefold())
            and not re.search(
                r"\b(?:worksheet|printable|pdf|page|sheet|card|diagram|checklist)\b",
                message.casefold(),
            )
        ):
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="revise_learning_thread",
                        arguments={"revision": message},
                        rationale="The parent asked to retheme the conversational activity, not create a document.",
                    )
                ]
            )
        if _explicit_knowledge_visual_request(message):
            return turn
        recent_artifact = next(
            (
                item
                for item in (request.get("workspace") or {}).get("recent_outputs") or []
                if item.get("kind") == "artifact"
            ),
            None,
        )
        if recent_artifact and re.search(
            r"\b(?:use|add|include|remove|move|put|labels?|arrows?|layout|beside|after|before|space|box|row|column|"
            r"black[- ]and[- ]white|one[- ]page|printer[- ]friendly|print[- ]friendly|theme|mystery|larger|"
            r"smaller|bigger|format|portrait|landscape|cut|cut[- ]apart|play|cards?|poster|set)\b",
            message.casefold(),
        ):
            return turn
        if _explicit_artifact_request(message):
            artifact_type = _requested_artifact_type(message)
            calls = [
                call.model_copy(
                    update={
                        "arguments": {
                            **call.arguments,
                            "artifact_type": artifact_type,
                            "revision": message,
                        }
                    }
                )
                if call.name == "create_learning_artifact"
                else call
                for call in turn.tool_calls
            ]
            return turn.model_copy(update={"tool_calls": calls})
        latest_event_id = (request.get("session") or {}).get("latest_event_id")
        replacement = ParentAgentToolCall(
            name="revise_learning_thread" if latest_event_id else "continue_learning_thread",
            arguments=(
                {"revision": message}
                if latest_event_id
                else {"message": message, "mode": AnswerMode.EXPLORE.value}
            ),
            rationale=(
                "The parent requested a conversational game or revision, not a tangible artifact."
            ),
        )
        calls = [replacement if call.name == "create_learning_artifact" else call for call in turn.tool_calls]
        return turn.model_copy(update={"tool_calls": calls})

    @staticmethod
    def _normalize_preference_with_learning(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Apply a temporary style change without dropping the question in the same turn."""

        message = str(request.get("message") or "").strip()
        if _looks_like_note_handling_question(message):
            return turn
        calls = list(turn.tool_calls)
        workspace = request.get("workspace") or {}
        preferences = [
            *list(workspace.get("thread_preferences") or []),
            *[
                item
                for item in workspace.get("parent_memories") or []
                if item.get("scope_type") == "thread"
            ],
        ]
        refines_preference = bool(
            preferences
            and re.search(r"\b(?:only|instead|except|otherwise)\b", message.casefold())
            and re.search(r"\b(?:short|long|answer|reply|activity|visual|quick)\b", message.casefold())
        )
        reverses = bool(
            preferences
            and not refines_preference
            and re.search(
                r"\b(?:normal|usual|regular).{0,30}\b(?:again|fine)|\b(?:forget|clear|stop using|no longer)\b|"
                r"\b(?:switch|go|change|return)\s+(?:that\s+)?back\b(?:.{0,60}\b(?:normal|usual|regular|"
                r"paragraphs?))?\b",
                message.casefold(),
            )
        )
        if reverses:
            calls = [call for call in calls if call.name != "update_thread_preference"]
            calls.insert(
                0,
                ParentAgentToolCall(
                    name="update_thread_preference",
                    arguments={"operation": "clear", "category": str(preferences[0].get("category") or "answer_style")},
                    rationale="The parent explicitly reversed the temporary thread preference.",
                ),
            )
        sets_preference = bool(
            re.search(r"\b(?:replies?|answers?)\b", message.casefold())
            and re.search(r"\b(?:keep|make|skip|use)\b", message.casefold())
            and re.search(
                r"\b(?:short(?:er)?|brief(?:er)?|concise|long(?:er)?|activity|visual|unless i ask)\b",
                message.casefold(),
            )
        )
        future_framing = bool(
            re.search(r"\b(?:if|when)\b.{0,80}\b(?:later|someday|next time|future|try)\b", message.casefold())
            and re.search(r"\b(?:offer|frame|present|treat|describe)\b", message.casefold())
        )
        if (sets_preference or refines_preference or future_framing) and not any(
            call.name == "update_thread_preference" for call in calls
        ):
            calls.insert(
                0,
                ParentAgentToolCall(
                    name="update_thread_preference",
                    arguments={
                        "operation": "set",
                        "category": "activity_style" if future_framing else "answer_style",
                        "value": message,
                    },
                    rationale="The parent explicitly set a reversible response preference for this thread.",
                ),
            )
        has_learning_payload = bool(
            _looks_like_child_learning_request(message, (request.get("session") or {}).get("child"))
            and (
                "?" in message
                or re.search(
                    r"\b(?:asked|asks|wondered|wonders|wants? to know|why|how|what happens|explain)\b",
                    message.casefold(),
                )
            )
        )
        if (sets_preference or refines_preference or reverses or future_framing) and not has_learning_payload:
            calls = [
                call
                for call in calls
                if call.name not in {"continue_learning_thread", "revise_learning_thread", "create_learning_artifact"}
            ]
        if has_learning_payload and any(
            call.name == "update_thread_preference" for call in calls
        ) and not any(call.name in {"continue_learning_thread", "revise_learning_thread"} for call in calls):
            calls.append(
                ParentAgentToolCall(
                    name="continue_learning_thread",
                    arguments={"message": message, "mode": str(request.get("answer_mode_hint") or "quick")},
                    rationale="The same parent turn also contains a child-learning question that needs an answer.",
                )
            )
        return turn.model_copy(update={"tool_calls": calls})

    @staticmethod
    def _normalize_output_feedback(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Do not convert response timing feedback into a child observation."""

        message = str(request.get("message") or "")
        if not _looks_like_output_feedback(message):
            return turn
        return ParentAgentTurn(
            message=(
                "Got it—that tells me about how the response landed, not what the child learned or understood. "
                "I’ll keep those separate."
            )
        )

    def _normalize_schedule_clarification(
        self,
        turn: ParentAgentTurn,
        request: dict[str, Any],
    ) -> ParentAgentTurn:
        """Explain a pending recurring proposal without creating child evidence."""

        raw_message = str(request.get("message") or "")
        message = raw_message.casefold()
        session_id = str((request.get("session") or {}).get("id") or "")
        with connect(self.db_path) as conn:
            pending_row = conn.execute(
                """SELECT t.id,t.arguments_json,t.result_json
                   FROM tool_calls t JOIN approval_requests a ON a.tool_call_id=t.id
                   WHERE t.session_id=? AND t.tool_name='propose_weekly_checkin'
                     AND t.status='awaiting_approval' AND a.decision='pending'
                   ORDER BY t.created_at DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            pending_schedule = bool(pending_row)
        if pending_row and re.search(
            r"\b(?:make|change|revise|update|move|switch|keep)\b|"
            r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|in the (?:morning|afternoon|evening))\b",
            message,
        ):
            arguments = dict(jload(pending_row["arguments_json"]))
            combined_request = f"{str(arguments.get('request') or '').strip()}\n{raw_message}".strip()
            arguments["request"] = combined_request
            argument_hash = sha256(jdump(arguments).encode()).hexdigest()
            scope_hash = sha256(
                f"{session_id}:{pending_row['id']}:propose_weekly_checkin:{argument_hash}".encode()
            ).hexdigest()
            result = dict(jload(pending_row["result_json"]))
            result["message"] = _weekly_checkin_draft(
                combined_request,
                (request.get("session") or {}).get("child"),
            )
            with connect(self.db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """UPDATE tool_calls SET argument_hash=?,arguments_json=?,result_json=?,updated_at=?
                       WHERE id=? AND status='awaiting_approval'""",
                    (argument_hash, jdump(arguments), jdump(result), utcnow(), str(pending_row["id"])),
                )
                conn.execute(
                    "UPDATE approval_requests SET scope_hash=? WHERE tool_call_id=? AND decision='pending'",
                    (scope_hash, str(pending_row["id"])),
                )
            return ParentAgentTurn(message=str(result["message"]))
        if re.search(r"\b(?:every[- ]other|every[- ]second|biweekly|fortnightly)\b", message) and (
            pending_schedule or re.search(r"\b(?:check-?in|sunday|schedule|recurring)\b", message)
        ):
            return ParentAgentTurn(
                message=(
                    "Curiosity Engine currently supports weekly check-ins, not every-other-week cadence, so I have "
                    "not turned that on. I can keep the wording as an unscheduled draft, or schedule it weekly after "
                    "you explicitly approve that change."
                )
            )
        if not re.search(r"\b(?:what|how).{0,45}\b(?:weekly|check-?in|routine).{0,25}\b(?:look like|work)\b", message):
            return turn
        history = request.get("history") or []
        has_schedule_context = any(
            item.get("role") == "assistant"
            and re.search(r"\b(?:schedule|check-?in|recurring)\b", str(item.get("content") or ""), re.IGNORECASE)
            for item in history
        )
        if not has_schedule_context:
            with connect(self.db_path) as conn:
                has_schedule_context = bool(
                    conn.execute(
                        "SELECT 1 FROM tool_calls WHERE session_id=? AND tool_name='propose_weekly_checkin' LIMIT 1",
                        (session_id,),
                    ).fetchone()
                )
        if not has_schedule_context:
            return turn
        return ParentAgentTurn(
            message=(
                "It would be one short message with one five-minute idea and one optional parent reply about what "
                "worked or felt hard. This is still only a preview—nothing repeats until you explicitly confirm the "
                "schedule."
            )
        )

    @staticmethod
    def _normalize_context_capture(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """A context share must not turn into an unsolicited mini-lesson."""

        message = str(request.get("message") or "").strip()
        if re.search(r"\b(?:replies?|answers?)\b", message.casefold()) and re.search(
            r"\b(?:keep|make|switch|change|skip|use)\b.{0,90}\b(?:short|long|regular|activity|visual|"
            r"optional|unless i ask)\b",
            message.casefold(),
        ):
            return turn
        is_context = bool(
            request.get("input_mode") == "context"
            or request.get("current_attachment_context")
            or _looks_like_shared_context(message)
            or _looks_like_parent_observation(message, (request.get("session") or {}).get("child"))
            or _looks_like_ambiguous_shared_fragment(message)
        )
        if not is_context:
            return turn
        if request.get("input_mode") != "context" and _explicit_learning_output_request(message):
            return turn
        if _looks_like_ambiguous_shared_fragment(message):
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="record_thread_context",
                        arguments={
                            "note": message,
                            "clarification_prompt": (
                                "I saved that as an unfinished thought. What did the child seem to mean or point to?"
                            ),
                        },
                        rationale="Preserve the ambiguous fragment and ask one focused clarification without guessing.",
                    )
                ]
            )
        if re.search(r"\b(?:for|in) this (?:chat|thread)\b", message.casefold()) and re.search(
            r"\b(?:replies?|answers?)\b", message.casefold()
        ):
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="record_thread_context",
                        arguments={"note": message},
                        rationale="Preserve the parent-shared moment without treating it as evidence.",
                    ),
                    ParentAgentToolCall(
                        name="update_thread_preference",
                        arguments={"operation": "set", "category": "answer_style", "value": message},
                        rationale="The same turn explicitly sets a reversible style for this thread.",
                    ),
                ]
            )
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="record_thread_context",
                    arguments={"note": message},
                    rationale="The parent shared context without requesting a learning output.",
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
        artifact_terms = (
            "printable",
            "worksheet",
            "activity sheet",
            "checklist",
            "instruction card",
            "results box",
            "challenge sheet",
            "comparison page",
            "diagram",
            "pdf",
        )
        recent_artifact = next(
            (
                item
                for item in (request.get("workspace") or {}).get("recent_outputs") or []
                if item.get("kind") == "artifact"
            ),
            None,
        )
        artifact_tuning = bool(
            recent_artifact
            and re.search(
                r"\b(?:use|add|include|remove|move|put|labels?|arrows?|layout|beside|after|before|space|box|row|column|"
                r"black[- ]and[- ]white|one[- ]page|printer[- ]friendly|print[- ]friendly|theme|mystery|larger|"
            r"smaller|bigger|format|portrait|landscape|cut|cut[- ]apart|play|cards?|poster|set)\b",
                lowered,
            )
        )
        if not any(term in lowered for term in artifact_terms) and not artifact_tuning:
            return turn
        artifact_is_negated = bool(
            re.search(
            r"\b(?:not|isn't|is not)\s+(?:a\s+)?(?:printable|worksheet|activity sheet|challenge sheet|pdf)\b",
            lowered,
            )
        )
        if artifact_is_negated:
            replacement_calls: list[ParentAgentToolCall] = []
            for call in turn.tool_calls:
                if call.name != "create_learning_artifact":
                    replacement_calls.append(call)
                    continue
                replacement_calls.append(
                    ParentAgentToolCall(
                        name="revise_learning_thread",
                        arguments={"revision": message},
                        rationale="The named artifact was explicitly rejected, so revise in conversation.",
                    )
                )
            if replacement_calls != turn.tool_calls:
                return turn.model_copy(update={"tool_calls": replacement_calls})
            return turn
        if not any(call.name == "revise_learning_thread" for call in turn.tool_calls) and not artifact_tuning:
            return turn
        artifact_type = next(
            (kind for kind in ("worksheet", "reference", "activity", "challenge") if kind in lowered),
            None,
        )
        if re.search(r"\b(?:cut[- ]and[- ]play|cut[- ]apart|shuffle|draw one)\b|\b(?:play|action) cards?\b", lowered):
            artifact_type = "activity"
        if "checklist" in lowered:
            artifact_type = "activity"
        if artifact_type is None and "diagram" in lowered:
            artifact_type = "reference"
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
                if prior_kind in {"worksheet", "reference", "activity", "challenge"}:
                    artifact_type = prior_kind
        if artifact_type is None and recent_artifact:
            prior_kind = str(recent_artifact.get("snippet") or "").split(":", 1)[0].casefold()
            if prior_kind in {"worksheet", "reference", "activity", "challenge"}:
                artifact_type = prior_kind
        artifact_type = artifact_type or "worksheet"
        if artifact_tuning and not any(call.name == "revise_learning_thread" for call in turn.tool_calls):
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="create_learning_artifact",
                        arguments={
                            "artifact_type": artifact_type,
                            "revision": message,
                            "target_ref": str(recent_artifact["ref_id"]),
                        },
                        rationale="The parent is tuning the current child-facing artifact.",
                    )
                ]
            )
        calls = [
            ParentAgentToolCall(
                name="create_learning_artifact",
                arguments={
                    "artifact_type": artifact_type,
                    "revision": message,
                    **({"target_ref": str(call.arguments["target_ref"])} if call.arguments.get("target_ref") else {}),
                },
                rationale="The parent is revising the current printable artifact.",
            )
            if call.name == "revise_learning_thread"
            else call
            for call in turn.tool_calls
        ]
        return turn.model_copy(update={"tool_calls": calls})

    @staticmethod
    def _normalize_context_artifact(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Save shared source material before making an artifact from it in the same bounded turn."""

        message = str(request.get("message") or "").strip()
        child = (request.get("session") or {}).get("child")
        explicit_context = bool(
            request.get("input_mode") == "context"
            or request.get("current_attachment_context")
            or _looks_like_shared_context(message)
            or _looks_like_ambiguous_shared_fragment(message)
        )
        reported_child_moment = _looks_like_parent_observation(message, child)
        initial_artifact_brief = bool(
            _explicit_artifact_generation_request(message)
            and not (request.get("session") or {}).get("latest_event_id")
            and not any(
                item.get("event_id") or item.get("artifact_id")
                for item in (request.get("workspace") or {}).get("recent_outputs") or []
            )
        )
        if _explicit_learning_output_request(message) and not explicit_context:
            name_parts = str((child or {}).get("name") or "").split()
            name = name_parts[0] if name_parts else ""
            reported_child_moment = bool(
                name
                and re.search(
                    rf"(?<![\w]){re.escape(name)}(?![\w]).{{0,80}}\b(?:noticed|built|made|drew|found|watched|"
                    r"listened|played|started|brought|tried|practiced)\b",
                    message,
                    re.IGNORECASE,
                )
            )
        is_context = explicit_context or reported_child_moment or initial_artifact_brief
        work_requested = any(
            call.name in {"create_learning_artifact", "continue_learning_thread"} for call in turn.tool_calls
        )
        already_records = any(call.name == "record_thread_context" for call in turn.tool_calls)
        if not is_context or not work_requested or already_records:
            return turn
        if _explicit_learning_output_request(message) and any(
            item.get("kind") == "artifact"
            for item in (request.get("workspace") or {}).get("recent_outputs") or []
        ):
            return turn
        if (
            request.get("input_mode") != "context"
            and not request.get("current_attachment_context")
            and re.search(
                r"\b(?:that|earlier|previous|already shared)\s+(?:teacher\s+)?(?:update|email|message|note|photo|context)\b",
                message,
                re.IGNORECASE,
            )
        ):
            return turn
        return turn.model_copy(
            update={
                "tool_calls": [
                    ParentAgentToolCall(
                        name="record_thread_context",
                        arguments={"note": message},
                        rationale="Preserve the parent-provided source without treating it as child evidence.",
                    ),
                    *turn.tool_calls,
                ]
            }
        )

    @staticmethod
    def _normalize_answer_mode(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        hint = AnswerMode(str(request.get("answer_mode_hint") or AnswerMode.AUTO.value))
        if hint != AnswerMode.AUTO and not turn.tool_calls:
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="continue_learning_thread",
                        arguments={"message": str(request.get("message") or ""), "mode": hint.value},
                        rationale="The transport supplied an explicit reviewed child-answer depth.",
                    )
                ]
            )
        calls: list[ParentAgentToolCall] = []
        for call in turn.tool_calls:
            if call.name != "continue_learning_thread":
                calls.append(call)
                continue
            arguments = dict(call.arguments)
            proposed = AnswerMode(str(arguments.get("mode") or AnswerMode.AUTO.value))
            arguments["mode"] = (hint if hint != AnswerMode.AUTO else proposed).value
            calls.append(call.model_copy(update={"arguments": arguments}))
        return turn.model_copy(update={"tool_calls": calls})

    @staticmethod
    def _normalize_reviewed_learning(turn: ParentAgentTurn, request: dict[str, Any]) -> ParentAgentTurn:
        """Do not let an unmistakable child question escape factual and pedagogy review."""

        if turn.tool_calls or not _looks_like_child_learning_request(
            str(request.get("message") or ""),
            (request.get("session") or {}).get("child"),
        ):
            return turn
        return ParentAgentTurn(
            tool_calls=[
                ParentAgentToolCall(
                    name="continue_learning_thread",
                    arguments={
                        "message": str(request.get("message") or ""),
                        "mode": AnswerMode.QUICK.value,
                    },
                    rationale="A clear child-learning question must use the reviewed answer path.",
                )
            ]
        )

    @staticmethod
    def _fallback_plan(
        message: str,
        has_current_attachments: bool = False,
        child: dict[str, Any] | None = None,
    ) -> ParentAgentTurn:
        lowered = message.casefold()
        if (
            _looks_like_shared_context(message) or _looks_like_parent_observation(message, child)
        ) and not _explicit_learning_output_request(message):
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="record_thread_context",
                        arguments={"note": message},
                        rationale="The parent shared context without asking for an output.",
                    )
                ]
            )
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
        if _explicit_artifact_request(message):
            artifact = _requested_artifact_type(message)
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="create_learning_artifact",
                        arguments={"artifact_type": artifact, "revision": message},
                        rationale="The parent asked for a child-facing artifact.",
                    )
                ]
            )
        if (
            re.search(
                r"\b(?:weekly|every week|each week|sunday|saturday|monday|tuesday|wednesday|thursday|friday)\b", lowered
            )
            and "check" in lowered
        ):
            return ParentAgentTurn(
                tool_calls=[
                    ParentAgentToolCall(
                        name="propose_weekly_checkin",
                        arguments={"request": message},
                        rationale="The parent asked for a recurring check-in.",
                    )
                ]
            )
        revision_terms = (
            "instead",
            "isn't",
            "is not",
            "not helpful",
            "thumbs down",
            "too much reading",
            "too long",
            "too much",
            "make it",
            "diagram",
            "visual",
            "picture",
            "shorter",
            "harder",
            "easier",
            "don't repeat",
            "do not repeat",
        )
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
                    arguments={"message": message, "mode": AnswerMode.AUTO.value},
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
        capability_run_id: str,
        source_message_id: str | None,
        origin: str,
        call_ordinal: int,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "completed",
            "message": turn.message,
            "tool_results": [],
            "_lineage_event_ids": [],
        }
        if turn.interaction:
            result["interaction"] = turn.interaction.model_dump(mode="json")
        for index, call in enumerate(turn.tool_calls, start=call_ordinal):
            spec = self.tools.spec(call.name)
            decision = ToolPolicy.decide(spec, origin=origin)
            existing_event_id = self.tools.lineage_event_id(call.name, call.arguments)
            stored_decision = (
                "awaiting_approval" if decision.requires_approval else "allowed" if decision.allowed else "denied"
            )
            call_id = self.sessions.record_tool_call(
                session_id=session_id,
                turn_id=turn_id,
                ordinal=index,
                tool_name=call.name,
                tool_version=spec.version,
                arguments=call.arguments,
                risk=spec.risk,
                decision=stored_decision,
                status="awaiting_approval" if decision.requires_approval else "running",
                event_id=existing_event_id,
                source_message_id=source_message_id,
                capability_run_id=capability_run_id,
            )
            if decision.requires_approval:
                tool_result = {
                    "status": "awaiting_approval",
                }
                status = "awaiting_approval"
            else:
                try:
                    tool_result = self.tools.execute(
                        call.name,
                        call.arguments,
                        origin=origin,
                        execution_context={
                            "session_id": session_id,
                            "turn_id": turn_id,
                            "capability_run_id": capability_run_id,
                            "tool_call_id": call_id,
                            **({"source_message_id": source_message_id} if source_message_id else {}),
                        },
                    )
                except Exception as exc:
                    self.sessions.finish_tool_call(
                        call_id,
                        status="failed",
                        result={"error_type": exc.__class__.__name__},
                    )
                    raise
                status = (
                    "succeeded" if tool_result.get("status") not in {"failed", "denied"} else tool_result["status"]
                )
            tool_event_id = str(tool_result.get("event_id") or "").strip() or None
            self.sessions.finish_tool_call(
                call_id,
                status=status,
                result=tool_result,
                event_id=tool_event_id,
            )
            if tool_event_id:
                result["_lineage_event_ids"].append(tool_event_id)
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
                if call.name == "propose_weekly_checkin":
                    session = self.sessions.session(session_id)
                    child_name = "your child"
                    child_id = str(session.get("child_id") or "")
                    if child_id:
                        with connect(self.db_path) as conn:
                            child_row = conn.execute("SELECT name FROM children WHERE id=?", (child_id,)).fetchone()
                        if child_row:
                            child_name = str(child_row["name"]).split()[0]
                    tool_result["message"] = _weekly_checkin_draft(
                        str(call.arguments.get("request") or ""),
                        {"name": child_name},
                    )
                self.sessions.finish_tool_call(
                    call_id,
                    status="awaiting_approval",
                    result=tool_result,
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
        for key in (
            "message",
            "interaction",
            "event_id",
            "visual_job_id",
            "artifact",
            "schedule",
            "reused_reviewed_response",
        ):
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
                key: artifact[key] for key in ("artifact_id", "artifact_type", "title") if artifact.get(key) is not None
            }
        return safe

    def _start_capability_run(self, session_id: str, source_event_id: str | None) -> str:
        run_id = f"cap_{uuid4().hex[:20]}"
        now = utcnow()
        capability = self.capabilities.capability("parent_chat")
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
                    capability.version,
                    jdump(
                        {
                            card["id"]: card["version"]
                            for card in self.capabilities.skill_cards(set(capability.skill_ids))
                        }
                    ),
                    source_event_id,
                    now,
                    now,
                ),
            )
        self._append_lifecycle(run_id, "accepted")
        self._append_lifecycle(run_id, "working")
        return run_id

    def _append_lifecycle(self, run_id: str, stage: str) -> None:
        if stage not in {"accepted", "working", "release_ready", "completed", "failed"}:
            raise ValueError("unsupported capability lifecycle stage")
        now = utcnow()
        payload = {"stage": stage}
        with connect(self.db_path) as conn:
            run = conn.execute(
                "SELECT produced_event_id FROM capability_runs WHERE id=?",
                (run_id,),
            ).fetchone()
            event_id = str(run["produced_event_id"]) if run and run["produced_event_id"] else None
            ordinal = int(
                conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM release_units WHERE capability_run_id=?",
                    (run_id,),
                ).fetchone()[0]
            )
            release_id = f"rel_{uuid4().hex[:20]}"
            conn.execute(
                """INSERT INTO release_units(
                       id,capability_run_id,unit_type,ordinal,status,content_hash,payload_json,created_at,
                       updated_at,event_id
                   ) VALUES(?,?,'status',?,'reviewed',?,?,?,?,?)""",
                (
                    release_id,
                    run_id,
                    ordinal,
                    sha256(jdump(payload).encode()).hexdigest(),
                    jdump(payload),
                    now,
                    now,
                    event_id,
                ),
            )
            if event_id is not None:
                conn.execute(
                    """INSERT INTO release_unit_event_lineage(release_unit_id,event_id)
                       VALUES(?,?)""",
                    (release_id, event_id),
                )

    def _store_release_units(
        self,
        run_id: str,
        result: dict[str, Any],
        *,
        lineage_event_ids: set[str],
    ) -> None:
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
            run = conn.execute(
                "SELECT produced_event_id FROM capability_runs WHERE id=?",
                (run_id,),
            ).fetchone()
            event_id = str(run["produced_event_id"]) if run and run["produced_event_id"] else None
            start = int(
                conn.execute(
                    "SELECT COALESCE(MAX(ordinal),0)+1 FROM release_units WHERE capability_run_id=?",
                    (run_id,),
                ).fetchone()[0]
            )
            for ordinal, (unit_type, payload) in enumerate(units, start=start):
                release_id = f"rel_{uuid4().hex[:20]}"
                conn.execute(
                    """INSERT INTO release_units(
                           id,capability_run_id,unit_type,ordinal,status,content_hash,payload_json,created_at,
                           updated_at,event_id
                       ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        release_id,
                        run_id,
                        unit_type,
                        ordinal,
                        "reviewed",
                        sha256(jdump(payload).encode()).hexdigest(),
                        jdump(payload),
                        now,
                        now,
                        event_id,
                    ),
                )
                if event_id is not None:
                    conn.execute(
                        """INSERT INTO release_unit_event_lineage(release_unit_id,event_id)
                           VALUES(?,?)""",
                        (release_id, event_id),
                    )
                for lineage_event_id in lineage_event_ids:
                    conn.execute(
                        """INSERT OR IGNORE INTO release_unit_event_lineage(release_unit_id,event_id)
                           VALUES(?,?)""",
                        (release_id, lineage_event_id),
                    )
        self._append_lifecycle(run_id, "release_ready")

    def _finish_capability_run(self, run_id: str, result: dict[str, Any]) -> None:
        status = "rejected" if result.get("status") == "rejected" else "completed"
        with connect(self.db_path) as conn:
            conn.execute(
                "UPDATE capability_runs SET status=?,result_json=?,updated_at=? WHERE id=?",
                (status, jdump(result), utcnow(), run_id),
            )
        self._append_lifecycle(run_id, "completed" if status == "completed" else "failed")
