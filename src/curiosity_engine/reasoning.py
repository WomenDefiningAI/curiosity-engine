from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import AppConfig
from .construction_patterns import construction_context
from .contracts import (
    CriticResult,
    CriticVerdict,
    DirectorOutput,
    FeedbackOutput,
    GenericOutput,
    InterestSignalOutput,
    ObservationCandidate,
    PhysicalExtension,
    PullThreadOutput,
    QuickAnswerOutput,
    ReasoningEnvelope,
)
from .pedagogy import (
    ApprovedPedagogyPlan,
    PedagogyCompiler,
    PedagogyPlanProposal,
    PedagogyRegistry,
    _contains_child_activity_directive,
    _is_self_harm,
    _requires_direct_answer,
    _requires_safety_refusal,
    default_plan_proposal,
    safety_response_requirements_for_event,
    safety_response_requirements_for_text,
    validate_generated_against_plan,
)
from .presentation import format_reviewed_response, is_generic_response_only
from .visuals import normalize_response_visual

OutputModel = TypeVar("OutputModel", bound=BaseModel)

logger = logging.getLogger(__name__)

_RETRY_COMPARISON_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "can",
    "do",
    "does",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "with",
    "you",
}

_RETRY_GENERIC_PREFIX = re.compile(
    r"^(?:(?:here|this)\s+(?:is|comes)\s+|try\s+)"
    r"(?:(?:a|an|another|fresh|new|different|really|interesting|thoughtful)\s+){0,8}"
    r"(?:answer|explanation|response|way|angle)"
    r"(?:\s+to\s+(?:think|look)\s+(?:about|at)\s+(?:it|this|that))?\s*[:—,\-]?\s*",
    re.IGNORECASE,
)


def _retry_explanation_text(value: dict[str, Any] | PullThreadOutput | QuickAnswerOutput) -> str:
    fields = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    # Compare the explanation itself. A newly added prompt, example, or activity
    # is useful, but it must not disguise an unchanged explanatory answer.
    keys = (("answer",) if fields.get("answer") else
            ("parent_response",) if fields.get("parent_response") else ("hook", "nugget"))
    return " ".join(str(fields.get(key) or "").strip() for key in keys if fields.get(key)).strip()


def _retry_explanation_is_distinct(
    previous: dict[str, Any],
    candidate: dict[str, Any] | PullThreadOutput | QuickAnswerOutput,
) -> bool:
    """Reject exact or lightly padded repeats while allowing a genuinely new example or explanation."""

    previous_text = " ".join(_retry_explanation_text(previous).casefold().split())
    candidate_text = " ".join(_retry_explanation_text(candidate).casefold().split())
    candidate_text = _RETRY_GENERIC_PREFIX.sub("", candidate_text, count=1).strip()
    if not previous_text or not candidate_text:
        return True
    previous_word_text = " ".join(re.findall(r"[^\W_]+", previous_text, flags=re.UNICODE))
    candidate_word_text = " ".join(re.findall(r"[^\W_]+", candidate_text, flags=re.UNICODE))
    if (
        previous_word_text == candidate_word_text
        or previous_word_text in candidate_word_text
        or candidate_word_text in previous_word_text
    ):
        return False
    previous_tokens = {
        token
        for token in re.findall(r"[^\W_]+", previous_text, flags=re.UNICODE)
        if token not in _RETRY_COMPARISON_STOPWORDS
    }
    candidate_tokens = {
        token
        for token in re.findall(r"[^\W_]+", candidate_text, flags=re.UNICODE)
        if token not in _RETRY_COMPARISON_STOPWORDS
    }
    if not previous_tokens or not candidate_tokens:
        return previous_text != candidate_text
    overlap = len(previous_tokens & candidate_tokens) / min(len(previous_tokens), len(candidate_tokens))
    sequence_similarity = SequenceMatcher(None, previous_text, candidate_text).ratio()
    return overlap < 0.85 and sequence_similarity < 0.82


def _activity_needs_printable(extension: Any) -> bool:
    """Spot activities that otherwise make a parent manufacture reusable play pieces."""

    if extension is None or extension.printable is not None:
        return False
    instructions = " ".join(extension.instructions).casefold()
    piece = r"(?:cards?|targets?|leaves|labels?|pieces?|tokens?|signs?|shapes?|recording sheets?|grids?|charts?|game boards?)"
    return bool(
        re.search(rf"\b(?:draw|make|create|cut(?: out)?|write)\b.{{0,90}}\b{piece}\b", instructions)
        or re.search(r"\brecord\b.{0,70}\b(?:result|observation|prediction|score)s?\b", instructions)
    )


_REPRESENTED_HIGH_RISK_OBJECT = re.compile(
    r"\b(?:picture|photo|drawing|diagram|illustration|worksheet|page|screen|card)\s+"
    r"(?:shows?|depicts?|pictures?|illustrates?|contains?)\s+"
    r"(?:(?:the|a|an|this|that|my|our|your)\s+)?"
    r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"\d+|some|several|many|multiple)\s+)?"
    r"(?:nail\s*guns?|chainsaws?|power\s+(?:drills?|saws?|tools?)|"
    r"knives|knife|scissors|matches|lighters?|candles?|"
    r"gas\s+(?:stoves?|cookers?|ovens?|burners?|knobs?|dials?|valves?)|"
    r"bleach|chemicals?|cleaners?|outlets?|sockets?|exposed\s+wires?)\b"
)


class ModelBackend(Protocol):
    name: str
    model: str | None

    def complete(
        self,
        *,
        role: str,
        system: str,
        payload: dict[str, Any],
        response_model: type[OutputModel],
    ) -> dict[str, Any]: ...


class ReasoningRejected(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        candidate: dict[str, Any] | None = None,
        critiques: list[dict[str, Any]] | None = None,
        critique_rounds: list[list[dict[str, Any]]] | None = None,
        recovery_strategy: str | None = None,
    ):
        super().__init__(message)
        self.candidate = candidate or {}
        self.critiques = critiques or []
        self.critique_rounds = critique_rounds or []
        self.recovery_strategy = recovery_strategy


class StubBackend:
    """Useful deterministic backend for local setup, tests, and offline evaluations."""

    name = "deterministic"
    model = "curiosity-stub-v2"

    def complete(
        self,
        *,
        role: str,
        system: str,
        payload: dict[str, Any],
        response_model: type[OutputModel],
    ) -> dict[str, Any]:
        del system
        if response_model is PedagogyPlanProposal:
            event = payload.get("event") or {}
            workflow = str(payload.get("workflow") or "pull_thread")
            return default_plan_proposal(event, workflow).model_dump(mode="json")
        if response_model is CriticResult:
            candidate = payload.get("candidate") or {}
            if "UNSUPPORTED" in str(candidate):
                return {
                    "verdict": "reject",
                    "concerns": ["Candidate contains an explicitly unsupported claim."],
                    "required_changes": ["Remove or support the claim."],
                }
            return {"verdict": "pass", "concerns": [], "required_changes": []}
        event = payload.get("event") or {}
        context = payload.get("context") or {}
        text = str(event.get("text") or "").strip()
        if response_model is PullThreadOutput:
            result = self._pull_thread(text, context)
            metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
            if isinstance(metadata.get("response_retry"), dict) and "robot" in text.casefold():
                result.update(
                    {
                        "hook": (
                            "Try tracing one message through the robot: a sensor notices a change, then sends "
                            "information to the controller."
                        ),
                        "show": (
                            "Compare a night-light sensor in bright and dim places and notice when its response "
                            "changes."
                        ),
                        "ask": "What change should the sensor notice before the robot acts?",
                        "nugget": (
                            "The sensor is the robot's input; the controller uses that input to choose an output, "
                            "such as turning, stopping, or switching on a light."
                        ),
                        "physical_extension": None,
                        "visual": None,
                    }
                )
            elif isinstance(metadata.get("response_retry"), dict) and any(
                word in text.casefold() for word in ("ice", "freeze", "float")
            ):
                result.update(
                    {
                        "hook": (
                            "Picture equal scoops of liquid water and frozen water: the frozen scoop spreads its "
                            "particles across more space."
                        ),
                        "show": "Compare an ice cube with the water it displaces in a clear cup.",
                        "ask": "Can you spot how much of the cube stays above the waterline?",
                        "nugget": (
                            "That open crystal arrangement makes each volume of ice weigh less than the same volume "
                            "of liquid water, so the surrounding water holds it up."
                        ),
                        "physical_extension": {
                            "title": "Watch the waterline",
                            "instructions": [
                                "Put water and one ice cube in a clear cup.",
                                "Mark the waterline with a washable marker.",
                                "Predict, then check the line after the cube melts.",
                            ],
                            "materials": ["clear cup", "water", "ice cube", "washable marker"],
                            "parent_effort": "low",
                        },
                        "visual": None,
                    }
                )
            plan = payload.get("approved_pedagogy_plan") or {}
            result = self._apply_plan_strategy(result, plan, text)
            if plan.get("question_intent") in {"direct_answer", "sensitive_support"}:
                combined_answer = " ".join(
                    value for value in (str(result.get("hook") or ""), str(result.get("nugget") or "")) if value
                )[:500]
                if not _contains_child_activity_directive(combined_answer):
                    result["hook"] = combined_answer
                result["nugget"] = "That is the complete answer for now."
                result["physical_extension"] = None
                result["actions"] = []
                result["visual"] = None
                result["ask"] = ""
                result["show"] = ""
            return result
        if response_model is QuickAnswerOutput:
            result = self._quick_answer(text, context)
            plan = payload.get("approved_pedagogy_plan") or {}
            if plan.get("question_intent") in {"direct_answer", "sensitive_support"}:
                combined_answer = str(result.get("answer") or "").strip()
                answer_sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+", combined_answer)
                    if sentence.strip() and not _contains_child_activity_directive(sentence)
                ]
                if answer_sentences:
                    result["answer"] = " ".join(answer_sentences)[:1200]
                elif plan.get("question_intent") == "sensitive_support":
                    result["answer"] = (
                        "I’m sorry this feels so hard. Tell a trusted grown-up right now so they can stay with you "
                        "and help keep you safe."
                    )
                else:
                    result["answer"] = "Here is a concise answer to that question."
                if "no_writing" in set((plan.get("accessibility") or {}).get("requested_supports") or []):
                    result["answer"] += " No writing is needed; you can point, say, or move something instead."
                result["notice"] = None
                result["question_back"] = None
                result["actions"] = []
                result["visual"] = None
            return result
        if response_model is InterestSignalOutput:
            return {
                "summary": f"Captured as a tentative interest signal: {text}",
                "observation_candidates": [{"kind": "concept_mention", "label": text}],
                "actions": [],
            }
        if response_model is FeedbackOutput:
            return {
                "acknowledged": "Feedback saved. It will be treated as one observation, not a durable trait.",
                "observation_candidates": [{"kind": "learning_process", "text": text}],
                "actions": [],
            }
        if response_model is DirectorOutput:
            context = payload.get("context") or {}
            curiosity = next(
                (
                    item
                    for item in context.get("observations", [])
                    if item.get("kind") in {"curiosity", "interest_signal"}
                ),
                None,
            )
            if curiosity:
                label = str(curiosity.get("text") or "a recent question")
                return {
                    "choice": {
                        "kind": "pull_thread",
                        "title": f"Revisit: {label[:110]}",
                        "rationale": "A recent child-led question is still timely and can be revisited with almost no setup.",
                        "parent_effort": "very_low",
                        "priority": 0.68,
                        "payload": {"question": label},
                    },
                    "considered": [],
                }
            return {
                "choice": {
                    "kind": "do_nothing",
                    "title": "Leave room for curiosity",
                    "rationale": "No unusually timely, low-effort opportunity is strong enough this week.",
                    "parent_effort": "very_low",
                    "priority": 0.1,
                    "payload": {},
                },
                "considered": [],
            }
        return {"summary": text or "No additional response.", "observation_candidates": [], "actions": []}

    @classmethod
    def _quick_answer(cls, text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        deep = cls._pull_thread(text, context)
        return {
            "answer": " ".join(
                value for value in (str(deep.get("hook") or ""), str(deep.get("nugget") or "")) if value
            ),
            "notice": deep.get("show"),
            "question_back": deep.get("ask"),
            "next_possible_concepts": deep.get("next_possible_concepts") or [],
            "visual": deep.get("visual"),
            "observation_candidates": deep.get("observation_candidates") or [],
            "actions": deep.get("actions") or [],
            "resource_refs": deep.get("resource_refs") or [],
        }

    @staticmethod
    def _apply_plan_strategy(result: dict[str, Any], plan: dict[str, Any], text: str) -> dict[str, Any]:
        """Keep deterministic/offline output honest enough to exercise Phase 2 gates."""

        intent = str(plan.get("question_intent") or "")
        overlay = str((plan.get("domain_overlay") or {}).get("id") or "")
        updated = dict(result)
        if _requires_safety_refusal(text):
            updated.update(
                {
                    "hook": (
                        "A metal object must never go inside a toaster or other powered appliance, even when it "
                        "looks switched off."
                    ),
                    "show": "",
                    "ask": "",
                    "nugget": (
                        "Electricity inside can cause severe shock, burns, or fire. The safe next step is to move "
                        "away and tell a trusted grown-up, who can unplug the appliance."
                    ),
                    "physical_extension": None,
                    "visual": None,
                    "actions": [],
                }
            )
        elif intent == "sensitive_support" and _is_self_harm(text):
            updated.update(
                {
                    "hook": (
                        "I’m really sorry this feels so hard. You deserve help and you should not be alone with "
                        "this feeling."
                    ),
                    "show": "",
                    "ask": "",
                    "nugget": (
                        "A trusted grown-up needs to know right now so they can stay with you and help keep you safe."
                    ),
                    "physical_extension": None,
                    "visual": None,
                    "actions": [],
                }
            )
        elif intent == "guided_inquiry":
            instructions = [
                "Notice one thing you can change and one thing you will keep the same.",
                "Predict what will happen, then run one safe test.",
                "Observe and compare what actually happened.",
                "Explain what the result supports and what you would test next.",
            ]
            if "boring" in text.casefold() or "different way" in text.casefold():
                instructions[0] = "Change the mechanic: use a paper ball on the ramp and choose one new ramp height."
            updated["physical_extension"] = {
                "title": "Predict–test–explain",
                "instructions": instructions,
                "materials": ["paper", "one book"] if "ramp" in text.casefold() else [],
                "parent_effort": "very_low",
            }
            updated["ask"] = (
                "What do you predict, what did you observe, and how does the result support your explanation?"
            )
        elif intent == "compare_connect":
            updated["physical_extension"] = {
                "title": "Compare one visible attribute",
                "instructions": [
                    "Choose two or three examples and name one visible attribute.",
                    "Point, say, move, or draw to sort what is the same and different.",
                    "Compare the groups and explain why each example belongs.",
                ],
                "materials": [],
                "parent_effort": "very_low",
            }
            updated["ask"] = "What is the same or different? Point, say, move, or draw, then explain your grouping."
        elif intent == "practice_with_scaffolds":
            example = (
                "Worked example: two counters and one more counter make three counters."
                if overlay == "early-math-k2"
                else "Worked example: watch one small version of the skill, with the choice explained aloud."
            )
            updated["show"] = example
            updated["ask"] = "Can you try one together with help, then say, point, move, or draw your choice?"
            updated["physical_extension"] = {
                "title": "Example, together, then less help",
                "instructions": [
                    example,
                    "Try one similar example together with a grown-up and name the helpful step.",
                    "Try the same idea once with less help or on your own.",
                    "If it still feels frustrating, choose an easier response mode or stop warmly for now.",
                ],
                "materials": [],
                "parent_effort": "very_low",
            }
        elif intent == "dialogic_reading":
            updated["show"] = "A grown-up reads one short part aloud while the child looks, listens, or points."
            updated["ask"] = "What do you notice or think might happen next? You may point, say, draw, or act it out."
            updated["physical_extension"] = {
                "title": "Read, notice, and talk",
                "instructions": [
                    "A grown-up reads a short page aloud.",
                    "Pause for the child to point, say, draw, or act out one meaning clue.",
                    "Ask one open question and connect the clue back to the words or picture.",
                ],
                "materials": ["one short book or story"],
                "parent_effort": "low",
            }
        elif intent == "teach_back":
            updated["ask"] = "Can you teach the idea back by explaining, showing, drawing, or acting it out?"
        elif intent == "retrieval_revisit":
            updated["show"] = "Before looking back, pause and recall one thing you remember."
            updated["ask"] = "How could you apply that remembered idea to one new or different example?"
        elif intent == "make_design_iterate":
            updated["physical_extension"] = {
                "title": "Make, test, and improve",
                "instructions": [
                    "Choose one design goal and make a small first version.",
                    "Test it once and observe what works or does not work.",
                    "Change one feature, try again, and explain whether it improved.",
                ],
                "materials": ["paper", "writing utensil"],
                "parent_effort": "low",
            }
            updated["ask"] = "What will you make, how will you test it, and what might you change to improve it?"

        # These two code-owned fields are the complete Phase 5 application
        # surface. Enact them observably in the deterministic route as well as
        # exposing them to hosted models, so promotion compares behavior rather
        # than approving inert metadata.
        challenge = str(plan.get("challenge_assumption") or "grade_baseline_uncalibrated")
        challenge_move = {
            "familiar": "Begin with one small, familiar step and offer help before adding another step.",
            "stretch_with_help": "Try one stretch step together, with a hint ready if it is needed.",
            "independent_transfer": "Try the idea in one genuinely different example, without assuming success.",
        }.get(challenge)
        if challenge_move:
            updated["ask"] = f"{challenge_move} {str(updated.get('ask') or '').strip()}".strip()
        interest_support = str(plan.get("interest_support") or "neutral")
        interest_move = {
            "situational_trigger": "If the child wants, let them choose one of two short ways to explore.",
            "maintenance": "Offer one related choice that keeps this exploration going without pressure.",
            "recurring_depth": "Offer one deeper connection, then let the child choose whether to continue or stop.",
        }.get(interest_support)
        if interest_move:
            updated["ask"] = f"{str(updated.get('ask') or '').strip()} {interest_move}".strip()

        accessibility = plan.get("accessibility") or {}
        requested = set(accessibility.get("requested_supports") or [])
        if "no_writing" in requested:
            updated["ask"] = str(updated.get("ask") or "") + " No writing is needed: point, say, move, or draw."
        if "read_aloud" in requested:
            updated["show"] = str(updated.get("show") or "") + " A grown-up can read every short direction aloud."
        extension = updated.get("physical_extension")
        if extension and "low_fine_motor" in requested:
            extension = dict(extension)
            extension["materials"] = [
                item
                for item in extension.get("materials") or []
                if not re.search(r"\b(?:scissors|tiny|cut)\b", str(item), re.IGNORECASE)
            ]
            extension["instructions"] = [
                "Use pointing or whole objects; no cutting or tiny pieces are needed.",
                *list(extension.get("instructions") or []),
            ][:8]
            updated["physical_extension"] = extension
        safety = plan.get("safety") or {}
        if updated.get("physical_extension") and safety.get("risk_level") in {"elevated", "sensitive"}:
            extension = dict(updated["physical_extension"])
            extension["instructions"] = [
                "A grown-up directly supervises and handles the exact hazardous step; the child only watches it.",
                *list(extension.get("instructions") or []),
            ][:8]
            updated["physical_extension"] = extension
        return updated

    @staticmethod
    def _pull_thread(text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        del context  # Canned demo cases stay intentionally narrow; hosted models use the full bounded context.
        lowered = text.casefold()
        if _REPRESENTED_HIGH_RISK_OBJECT.search(lowered):
            return {
                "hook": "This task is about noticing or marking a represented object, not handling real equipment.",
                "show": "Keep your attention on the picture, page, or screen.",
                "ask": "Which visible detail will you point to, describe, or mark in the representation?",
                "nugget": "A label or mark changes only the representation and can help another person find the same detail.",
                "next_possible_concepts": ["visual evidence", "labels", "representation"],
                "physical_extension": None,
                "visual": None,
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "moon" in lowered and ("follow" in lowered or "move" in lowered):
            return {
                "hook": "The Moon can seem like a quiet travel companion—even though it is extremely far away.",
                "show": "Look at the Moon from two spots, or use a faraway tree or building in daylight.",
                "ask": "What changes when you move: the Moon, the nearby things, or both?",
                "nugget": "Nearby objects slide across our view quickly as we move; very distant objects change position so little that they can seem to follow us.",
                "next_possible_concepts": ["distance and perspective", "parallax", "the Moon's orbit"],
                "physical_extension": {
                    "title": "Near finger, far wall",
                    "instructions": [
                        "Hold one finger close to your face and choose a mark on a far wall.",
                        "Move your head slowly side to side.",
                        "Compare how much the finger and far mark seem to move.",
                    ],
                    "materials": ["your finger", "a far wall"],
                    "parent_effort": "very_low",
                },
                "observation_candidates": [{"kind": "concept_mention", "label": "distance and perspective"}],
                "actions": [],
                "resource_refs": [],
            }
        if any(word in lowered for word in ("plane", "airplane", "fly")):
            return {
                "hook": "A heavy airplane does something surprising: it pushes air and the air pushes back.",
                "show": "Hold a flat sheet of paper below your lower lip and blow across its top.",
                "ask": "Which way do you predict the paper will move when the fast air passes over it?",
                "nugget": "Moving wings redirect air downward, and the air pushes the wings upward; engines keep the airplane moving through the air.",
                "next_possible_concepts": ["forces", "air pressure", "wing shapes"],
                "physical_extension": {
                    "title": "Paper wing test",
                    "instructions": [
                        "Blow across a strip of paper.",
                        "Try again with a faster and slower breath.",
                        "Compare what changes.",
                    ],
                    "materials": ["strip of paper"],
                    "parent_effort": "very_low",
                },
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if any(word in lowered for word in ("ice", "freeze", "float")):
            return {
                "hook": "Most solids sink in their own liquid, but ice breaks that pattern.",
                "show": "Float an ice cube in a clear glass and mark the water level.",
                "ask": "What do you predict will happen to the water level after the ice melts?",
                "nugget": "When water freezes, its molecules settle into a roomier structure, making ice less dense than liquid water, so it floats.",
                "next_possible_concepts": ["density", "states of matter", "freezing and melting"],
                "physical_extension": {
                    "title": "Melt-level prediction",
                    "instructions": [
                        "Mark the water level.",
                        "Let the ice melt.",
                        "Compare the final level with your prediction.",
                    ],
                    "materials": ["clear glass", "water", "ice cube", "washable marker"],
                    "parent_effort": "low",
                },
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "bubble solution" in lowered and ("plain water" in lowered or "where" in lowered):
            return {
                "hook": (
                    "Bubble solution lasts longer than plain water because soap molecules help make a stretchy "
                    "film instead of letting the water pull itself into one drop."
                ),
                "show": "A bubble is a thin film with air inside; plain water usually collapses too quickly.",
                "ask": "Which part do you think the soap changes: the air or the watery film around it?",
                "nugget": (
                    "Families can usually find ready-made bubble solution at toy, grocery, or party stores. "
                    "Availability changes, so check or call before making a special trip."
                ),
                "next_possible_concepts": ["surface tension", "thin films", "air pressure"],
                "physical_extension": None,
                "visual": None,
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "dinosaur" in lowered and "color" in lowered:
            return {
                "hook": "Some dinosaur colors may be recoverable from microscopic fossil clues, but a universal dinosaur color is not known.",
                "show": "Compare several modern birds: close relatives can have very different colors and patterns.",
                "ask": "Which parts of a dinosaur's color could fossil evidence support, and which parts might stay unknown?",
                "nugget": "Scientists have inferred colors for a few feathered dinosaurs from preserved pigment structures, but evidence does not reveal the colors of every dinosaur.",
                "next_possible_concepts": ["pigment", "fossil evidence", "scientific uncertainty"],
                "physical_extension": None,
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "sauropod" in lowered or (
            "dinosaur" in lowered and re.search(r"\blong[- ]neck", lowered)
        ):
            return {
                "hook": "Sauropods were long-necked plant eaters that could reach leaves at different heights.",
                "show": (
                    "Keep paper leaf targets flat on a table, then compare which ones a short and a long paper neck can touch."
                ),
                "ask": "Which paper neck reached a farther leaf, and what made that possible?",
                "nugget": "A long neck helped a sauropod reach plants without moving its whole heavy body each time.",
                "next_possible_concepts": ["body adaptations", "plant eaters", "height and reach"],
                "physical_extension": {
                    "title": "Sauropod leaf reach",
                    "instructions": [
                        "Keep the printed comparison whole and flat on a table.",
                        "Trace the SHORT NECK and LONG NECK lines from SAME START.",
                        "Point to the fixed leaves each line reaches, compare by length, and tell why.",
                    ],
                    "materials": ["printed Long-Neck Leaf Test page"],
                    "parent_effort": "low",
                    "printable": {
                        "kind": "reach_comparison",
                        "title": "Long-Neck Leaf Test",
                        "child_directions": (
                            "Trace both printed necks from SAME START. Point to the fixed leaves each reaches, "
                            "compare by length, and tell why."
                        ),
                        "parent_setup": (
                            "Keep this page whole and flat; the START, leaf distances, and neck lines are built in."
                        ),
                        "pedagogical_value": (
                            "The fixed same-start page safely compares how neck length changes reach."
                        ),
                        "pieces": [
                            {"label": "NEAR LEAF", "prompt": "fixed at 2 units", "shape": "leaf"},
                            {"label": "MIDDLE LEAF", "prompt": "fixed at 4 units", "shape": "leaf"},
                            {"label": "FAR LEAF", "prompt": "fixed at 6 units", "shape": "leaf"},
                            {"label": "SHORT NECK", "prompt": "3 units from SAME START", "shape": "card"},
                            {"label": "LONG NECK", "prompt": "6 units from SAME START", "shape": "card"},
                            {"label": "HELP USED?", "prompt": "point: YES / NO", "shape": "card"},
                        ],
                    },
                },
                "visual": None,
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "dinosaur" in lowered or "fossil" in lowered:
            return {
                "hook": "Nobody has ever seen a living non-bird dinosaur, yet scientists can still investigate them like detectives.",
                "show": "Compare a footprint, a tooth, and a skeleton picture and ask what each could reveal.",
                "ask": "Which clue would help you learn what a dinosaur ate, and why?",
                "nugget": "Fossils are evidence: bones, teeth, footprints, nests, and even droppings can support different ideas about ancient lives.",
                "next_possible_concepts": ["fossil evidence", "predators and herbivores", "birds as dinosaurs"],
                "physical_extension": {
                    "title": "Mystery track detective",
                    "instructions": [
                        "Make a track in play dough with a toy or household object.",
                        "Hide the object.",
                        "Ask someone to infer what made it and explain their evidence.",
                    ],
                    "materials": ["play dough", "small object"],
                    "parent_effort": "low",
                },
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "plant" in lowered and re.search(r"\b(?:die|dying|dead)\b", lowered):
            return {
                "hook": "Plants do not want things the way people and animals do.",
                "show": "A plant stays alive by using light, water, air, and nutrients in the right conditions.",
                "ask": "Which condition do you think changed for the plant?",
                "nugget": (
                    "A plant can die when damage, disease, or missing resources keep its cells from doing the jobs "
                    "that sustain life."
                ),
                "next_possible_concepts": ["plant needs", "living things", "plant cells"],
                "physical_extension": None,
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "robot" in lowered and any(word in lowered for word in ("biggest", "largest", "tallest", "heaviest")):
            return {
                "hook": (
                    "There is not one single biggest robot: one can be tallest, another heaviest, and another "
                    "strongest. Some person-shaped robots are taller than a house, while automated mining "
                    "machines can be as big as buildings."
                ),
                "show": (
                    "Compare pictures of a tall person-shaped robot and an automated mining machine. Look for a "
                    "nearby person, wheel, door, or ladder that shows how large each machine is."
                ),
                "ask": "Which should win the word biggest: the tallest, the heaviest, or the one that lifts the most?",
                "nugget": (
                    "Engineers compare height, weight, and lifting power. Different robots can win each contest, "
                    "so a good biggest answer says what was measured."
                ),
                "next_possible_concepts": ["measurement", "scale", "categories", "automation"],
                "physical_extension": {
                    "title": "Make a biggest-robot scoreboard",
                    "instructions": [
                        "Fold a sheet of paper into three columns: tallest, heaviest, and strongest.",
                        "Draw or write one robot contender in each column.",
                        "Circle your winner and explain which measurement you chose.",
                    ],
                    "materials": ["paper", "writing utensil"],
                    "parent_effort": "very_low",
                },
                "visual": {
                    "kind": "comparison_cards",
                    "purpose": "compare",
                    "knowledge_role": "supportive",
                    "title": "What can BIGGEST mean for a robot?",
                    "pedagogical_value": "Lets an early reader compare three meanings of biggest without implying exact scale.",
                    "caption": "A robot can win one kind of big without winning them all.",
                    "alt_text": "Three illustrated cards compare tallest, heaviest, and strongest as different meanings of the biggest robot. The cards are not to scale.",
                    "panels": [
                        {"label": "TALLEST", "detail": "Reaches the highest.", "icon": "height"},
                        {"label": "HEAVIEST", "detail": "Has the most weight.", "icon": "weight"},
                        {"label": "STRONGEST", "detail": "Can lift the most.", "icon": "strength"},
                    ],
                    "source_refs": [],
                    "not_to_scale": True,
                },
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "robot" in lowered:
            return {
                "hook": "A robot can look clever even when it is following a tiny loop: sense, decide, act, repeat.",
                "show": "Use a phone's auto-rotate or a motion-activated light as an example of a sensor changing a response.",
                "ask": "What would the machine need to sense before it could choose its next action?",
                "nugget": "A sensor collects information, a controller applies a rule, and an actuator changes something in the world.",
                "next_possible_concepts": ["sensors", "feedback loops", "control"],
                "physical_extension": {
                    "title": "Be the robot",
                    "instructions": [
                        "Choose one signal, such as a clap.",
                        "Choose one response, such as turn around.",
                        "Add a second rule and see when the system gets confused.",
                    ],
                    "materials": [],
                    "parent_effort": "very_low",
                },
                "visual": {
                    "kind": "activity_sequence",
                    "purpose": "sequence",
                    "knowledge_role": "supportive",
                    "title": "Give your paper robot a plan",
                    "pedagogical_value": "Turns an abstract command idea into three visible actions an early reader can use.",
                    "caption": "Point to one card at a time and follow it exactly.",
                    "alt_text": "Three colorful cards show GO with a curved arrow, STOP with a red stop shape, and TURN with a turning arrow for a pretend robot game.",
                    "panels": [
                        {"label": "GO", "detail": "Move forward.", "icon": "go"},
                        {"label": "STOP", "detail": "Freeze in place.", "icon": "stop"},
                        {"label": "TURN", "detail": "Change direction.", "icon": "turn"},
                    ],
                    "source_refs": [],
                    "not_to_scale": False,
                },
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "element" in lowered or ("atom" in lowered and "different" in lowered):
            return {
                "hook": "Atoms can gain or lose electrons, but one number in the nucleus decides which element they are.",
                "show": "Line up three cards labeled 1 proton, 2 protons, and 3 protons: hydrogen, helium, and lithium.",
                "ask": "If an atom changes its electrons but keeps the same number of protons, do you predict it is still the same element?",
                "nugget": "The number of protons defines an element; changing electrons makes an ion, while changing protons makes a different element.",
                "next_possible_concepts": ["protons", "atomic number", "ions"],
                "physical_extension": None,
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        if any(word in lowered for word in ("friend", "mean", "mad at me")):
            return {
                "hook": "One action can have several possible explanations, and we usually need more clues before deciding which is true.",
                "show": "Name only what was seen or heard, without guessing what the other person meant.",
                "ask": "What are two possible reasons it happened, and what could you ask to learn more?",
                "nugget": "We can be certain about an action we observed while staying uncertain about another person's motive.",
                "next_possible_concepts": ["observation versus inference", "repair questions"],
                "physical_extension": None,
                "observation_candidates": [],
                "actions": [],
                "resource_refs": [],
            }
        return {
            "hook": f"There is something worth noticing inside this question: {text}",
            "show": "Find one nearby object, trustworthy image, or simple demonstration connected to the question.",
            "ask": "What do you notice first, and what do you predict might be happening?",
            "nugget": "Keep the first explanation small: make one observation, test one prediction, and use the result to choose the next question.",
            "next_possible_concepts": ["observation", "prediction", "evidence"],
            "physical_extension": {
                "title": "Notice–predict–check",
                "instructions": [
                    "Write or say one thing you notice.",
                    "Make one prediction.",
                    "Find a safe way to check it.",
                ],
                "materials": [],
                "parent_effort": "very_low",
            },
            "observation_candidates": [],
            "actions": [],
            "resource_refs": [],
        }


@dataclass(frozen=True)
class ReasoningPolicy:
    workflow: str
    context_depth: int
    budget: str = "fast"
    generator_role: str = "reasoning"
    critic_roles: tuple[str, ...] = ()
    max_revision_rounds: int = 0
    final_recovery: str | None = None
    allowed_tools: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["critic_roles"] = list(self.critic_roles)
        data["allowed_tools"] = list(self.allowed_tools)
        return data


CRITIC_ROLE_ALIASES = {
    "factual": "critic_factual",
    "family_usefulness": "critic_family_usefulness",
    "pedagogy": "critic_pedagogy",
    "context": "critic_context",
    "parent_effort": "critic_parent_effort",
    "epistemic": "critic_epistemic",
    "visual": "critic_visual",
}


def runtime_critic_roles(
    policy: ReasoningPolicy,
    pedagogy_plan: ApprovedPedagogyPlan | None,
) -> tuple[str, ...]:
    """Choose proportional model review for the current family-visible workflow.

    Quick answers and learning threads already pass deterministic contract, plan,
    evidence, accessibility, and safety validation. Re-running every one of those
    checks as a separate hosted critic made an ordinary answer require up to nine
    serial model calls. The trusted-parent beta uses one combined usefulness review
    for a quick answer and adds one pedagogy review for a requested exploration.
    Stronger non-conversational workflows retain their plan-declared critic set.
    """

    roles = list(policy.critic_roles)
    if pedagogy_plan is not None and policy.workflow not in {"quick_answer", "pull_thread"}:
        roles.extend(pedagogy_plan.required_critics)
    return tuple(dict.fromkeys(roles))


def load_policies(config: AppConfig | None = None) -> dict[str, ReasoningPolicy]:
    app = config or AppConfig.load()
    result: dict[str, ReasoningPolicy] = {}
    for workflow, entry in app.reasoning["workflows"].items():
        critics = tuple(CRITIC_ROLE_ALIASES.get(x, x) for x in entry.get("critics", []))
        result[workflow] = ReasoningPolicy(
            workflow=workflow,
            context_depth=int(entry["context_depth"]),
            budget=str(entry["budget"]),
            generator_role=str(entry["generator_role"]),
            critic_roles=critics,
            max_revision_rounds=int(entry.get("max_revision_rounds", 0)),
            final_recovery=entry.get("final_recovery"),
            allowed_tools=tuple(entry.get("allowed_tools", [])),
        )
    result.setdefault("interest_signal", ReasoningPolicy("interest_signal", 2, generator_role="structured_extraction"))
    result.setdefault("feedback", ReasoningPolicy("feedback", 2, generator_role="structured_extraction"))
    result.setdefault("generic", ReasoningPolicy("generic", 1))
    return result


POLICIES = load_policies()


def default_policy_for_event(event_type: str) -> ReasoningPolicy:
    if event_type in {"child_question", "curiosity"}:
        return POLICIES["pull_thread"]
    return POLICIES.get(event_type, POLICIES["generic"])


WORKFLOW_MODELS: dict[str, type[BaseModel]] = {
    "pull_thread": PullThreadOutput,
    "quick_answer": QuickAnswerOutput,
    "interest_signal": InterestSignalOutput,
    "feedback": FeedbackOutput,
    "weekly_reflection": DirectorOutput,
    "generic": GenericOutput,
}


def _rejects_activity(text: str) -> bool:
    """Reject an activity category, not just the previous version of a game."""
    kind = r"(?:game|activity|experiment|challenge|plan|lesson|quiz)\b"
    return bool(re.search(
        r"\b(?:no|without)\s+(?:(?:an?|any|another|more|new)\s+)?" + kind
        + r"|\b(?:drop|remove|skip)\s+(?:(?:the|this|that|an?|any)\s+)?" + kind
        + r"|\bnot\s+(?:an?|any|another)\s+" + kind
        + r"|\b(?:don't|do not)\s+(?:add|include|suggest|give(?:\s+(?:me|us))?)\s+(?:(?:an?|any|another)\s+)?" + kind,
        text.casefold(),
    ))


def _release_gate_requirements(event: dict[str, Any]) -> list[str]:
    """Surface semantic obligations that are easy for a prose critic to overlook."""

    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    source_context = metadata.get("source_context") if isinstance(metadata.get("source_context"), dict) else {}
    revision = metadata.get("response_revision") if isinstance(metadata.get("response_revision"), dict) else {}
    retry = metadata.get("response_retry") if isinstance(metadata.get("response_retry"), dict) else {}
    thread_history = revision.get("thread_history") if isinstance(revision.get("thread_history"), list) else []
    history_text = "\n".join(str(item.get("content") or "") for item in thread_history if isinstance(item, dict))
    newest = str(event.get("text") or "")
    source = str(source_context.get("text") or "")
    parent_request = str(revision.get("parent_request") or "")
    previous_reviewed = (
        revision.get("previous_reviewed_response")
        if isinstance(revision.get("previous_reviewed_response"), dict)
        else {}
    )
    previous_reviewed_text = "\n".join(str(value or "") for value in previous_reviewed.values())
    combined = "\n".join(part for part in (newest, source, parent_request, history_text) if part).casefold()
    current_request = "\n".join(part for part in (newest, source, parent_request) if part).casefold()
    requirements: list[str] = []
    if retry:
        requirements.append(
            "This is a parent-requested 'try another' response. Give a complete, substantively different answer, "
            "not a transition, acknowledgement, or paraphrase. End with one concrete optional way to continue the "
            "learning—a prediction, observation, comparison, or tiny test when that naturally fits the question."
        )
    if re.search(r"\b(?:usually|often|not always|sometimes|mostly|generally)\b", combined):
        requirements.append(
            "A qualified comparison needs one explicit important exception or condition; repeating 'usually' alone "
            "does not satisfy this."
        )
    if re.search(r"[“\"]?[^\n.!?]{1,100}\b(?:is|are)\b[^\n.!?]{1,100}[”\"]?", source.casefold()):
        requirements.append(
            "For a child category analogy or fact check, plainly define the category and correct any material "
            "reverse implication (some X may be Y does not mean every Y is X)."
        )
    if re.search(r"\b(?:why|what (?:makes|causes|changes)|how (?:does|do|can|could|would))\b", newest.casefold()):
        requirements.append(
            "Answer the causal question with the main mechanism or variables in concrete child-ready language, not "
            "only examples or a vague statement that something changes. If a common process name helps the family "
            "recognize the idea later, give the plain explanation first and then name the process."
        )
    if re.search(r"\bwho (?:invented|made|discovered|created|started)\b", combined):
        requirements.append(
            "For an origin-history question, do not force a single-inventor story. Say whether the idea developed "
            "over time, name multiple places or traditions when relevant, and distinguish an important contributor "
            "from a sole inventor."
        )
    # Topic words such as shape, order and color do not establish a human
    # convention. Models interpret that distinction from the actual context;
    # code must not turn a keyword guess into a mandatory causal explanation.
    if parent_request and re.search(
        r"\b(?:shorter|simpler|two|three|one|1|2|3)\b.{0,40}\b(?:sentence|answer|version)|"
        r"\b(?:sentence|answer|version)\b.{0,40}\b(?:shorter|simpler)\b",
        parent_request.casefold(),
    ):
        requirements.append(
            "A concise revision must preserve the prior reviewed answer's central mechanism, qualification, and "
            "historical or safety reason. Remove jargon and extras before removing the fact that answers why."
        )
    if parent_request and re.search(r"\b(?:one|1)\s+(?:clear\s+)?sentence\b", parent_request.casefold()):
        requirements.append(
            "Return exactly one easy-to-say sentence of at most 150 characters. Keep only the central answer, use "
            "plain words, and do not append an unrequested warning, follow-up question, activity, or new term."
        )
    if parent_request and re.search(r"\b(?:short|brief)\s+answer\b", parent_request.casefold()):
        prior_questions = [
            item
            for item in thread_history
            if isinstance(item, dict) and item.get("role") == "user" and "?" in str(item.get("content") or "")
        ]
        if len(prior_questions) >= 2:
            requirements.append(
                "The parent asked for one short spoken answer after multiple connected child questions. Preserve a "
                "plain answer to each still-active question; concision is not permission to drop the earlier why."
            )
    if parent_request and re.search(r"\b(?:it|that|those|this)\b", parent_request.casefold()):
        requirements.append(
            "Resolve revision pronouns to the intended subject of the reviewed answer. Name it only when needed "
            "to make the reply clear; an obvious reference does not require a repeated title. Do "
            "not switch to a visual, artifact, or nearby noun merely because it appears in the source wording."
        )
    if parent_request and re.search(
        r"\b(?:prompts?|clues?|orders?|cards?|rows?|labels?|examples?)\b", parent_request.casefold()
    ):
        requirements.append(
            "Supply the parent's requested child-usable components verbatim in the response; do not merely describe "
            "or promise them."
        )
    if parent_request and re.search(r"\b(?:genuinely|completely|substantially) different\b", parent_request.casefold()):
        requirements.append(
            "The revision must change the core child action and play loop while still following every named direction. "
            "Reuse materials or objects already available in the thread, or use ordinary paper/writing tools; do not "
            "introduce a specialized toy, kit, or supply unless the parent offered it."
        )
    if (
        parent_request
        and re.search(r"\b(?:draw|drawing|diagram|map|arrows?)\b", parent_request.casefold())
        and re.search(
            r"\b(?:because|helps?|makes?|causes?|lets?|so that)\b",
            previous_reviewed_text.casefold(),
        )
    ):
        requirements.append(
            "A revised drawing or diagram must state in child-ready words what its arrows, colors, or parts mean and "
            "preserve the prior answer's central causal mechanism. Unlabeled arrows or decorative cues are not enough."
        )
    if re.search(r"\b(?:vote|voting|elected|election|leader)\b", combined) and re.search(
        r"\b(?:boss|control|most votes|deeper|rules?)\b", combined
    ):
        requirements.append(
            "For a young child's voting/leader misconception, say that votes may choose different things under "
            "different rules, and that elected leaders have a public job and must follow laws and limits; winning "
            "does not make someone everyone's unlimited boss."
        )
    if re.search(r"\b(?:skin|shade|melanin)\b", combined) and re.search(
        r"\b(?:better|worse|equal|same kindness|no worksheet)\b", combined
    ):
        requirements.append(
            "A short equality-focused follow-up must preserve one plain causal fact from the prior explanation—people "
            "naturally have different amounts of melanin—while clearly saying no shade is better than another."
        )
    if re.search(
        r"\b(?:gentle|open[- ]ended|do the thinking|notice|noticing)\b.{0,80}\bquestion\b|"
        r"\bquestion\b.{0,80}\b(?:gentle|open[- ]ended|do the thinking|notice|noticing)\b",
        combined,
    ):
        requirements.append(
            "A child-led observation question must ask what the child sees, notices, compares, or wonders. Do not "
            "presume that a visible feature has a purpose, embed the explanation, or steer toward one right answer."
        )
    if parent_request and re.search(
        r"\b(?:what|which)\s+(?:pulls?|pushes?|moves?|makes?)\b|"
        r"\b(?:asking|wants? to know)\b.{0,45}\b(?:what|which)\b.{0,25}\b(?:pulls?|pushes?|moves?)\b",
        parent_request.casefold(),
    ):
        requirements.append(
            "Name the immediate actor and causal direction the parent focused on. Say what physically pushes or "
            "moves what; avoid vague substitutes such as 'it gets carried' or a misleading pulling metaphor."
        )
    if parent_request and re.search(
        r"\b(?:build on|use)\b.{0,50}\b(?:idea|guess|hypothesis|what (?:he|she|they) said)\b|"
        r"\b(?:one step|a little) deeper\b",
        parent_request.casefold(),
    ):
        requirements.append(
            "Explicitly reuse the child's stated idea and the key relationship from the prior reviewed explanation, "
            "then add one causal layer. Do not replace that relationship with a vaguer phrase."
        )
    if re.search(r"\b(?:salsa|spicy|pepper|capsaicin)\b", combined) and re.search(
        r"\b(?:hot|burn|unpleasant|too spicy)\b", combined
    ):
        requirements.append(
            "Clarify that the spicy sensation is real even when the food is not physically hot. Add one calm, "
            "proportionate comfort choice: stop eating it, choose something milder, and tell a grown-up if it hurts."
        )
    if re.search(r"\b(?:heart|breath|breathing|chest|faint|dizzy|pain|hurt|symptom)\w*\b", current_request):
        requirements.append(
            "A calm body/health answer must stay non-diagnostic and include one proportionate tell-a-grown-up "
            "boundary for severe, painful, unusual, lasting, fainting, or trouble-breathing symptoms."
        )
    if re.search(r"\b(?:fridge|refrigerator|freezer|washer|dryer|dishwasher|appliance)\b", combined) and re.search(
        r"\b(?:sound|noise|hum|buzz|brr|bang|rattle|loud)\w*\b",
        combined,
    ):
        requirements.append(
            "For an appliance sound, name the likely ordinary working parts without diagnosing this machine, and "
            "briefly distinguish a normal hum from a new, unusually loud, banging, or concerning sound a grown-up "
            "should check."
        )
    if re.search(r"\binvisible\b", combined) and re.search(
        r"\b(?:light|signal|detect|sensor|receiver|work)\w*\b", combined
    ):
        requirements.append(
            "Explain that a purpose-built sensor or receiver can detect a signal human senses cannot. If suggesting "
            "a camera observation, say results vary because some cameras or modes filter that signal and keep the "
            "observation safely away from eyes."
        )
    if re.search(r"\bremote(?: control)?\b", combined) and re.search(
        r"\b(?:invisible|light|signal|tv|television|camera)\b",
        combined,
    ):
        requirements.append(
            "For an ordinary TV remote, plainly name infrared as the common invisible-light signal after explaining "
            "it in child-ready words. Do not leave the mechanism as only 'a code' or 'invisible light.'"
        )
    if re.search(r"\b(?:rotary|round dial|dial phone|old telephone)\b", combined):
        requirements.append(
            "For a rotary telephone, connect every visible part to the action: a finger goes in a numbered hole, "
            "turns the dial to the stop, and releases it; the returning dial sends counted pulses. Plainly say this "
            "describes older dial phones and that most current phones send number choices electronically instead."
        )
    if re.search(r"\b(?:mushroom|fungus|fungi)\b", combined) and re.search(
        r"\b(?:deeper|more|explore|umbrella|hidden|mulch)\b",
        combined,
    ):
        requirements.append(
            "Explain that the visible mushroom is the spore-making part of a usually larger fungus whose fine "
            "threadlike body is mostly hidden in the material below. Preserve a no-touch/no-eating boundary and do "
            "not claim an exact species."
        )
        if re.search(r"\b(?:deeper|explore|way to look|without picking|no[- ]touch)\b", combined):
            requirements.append(
                "A no-touch mushroom exploration must center on visible evidence the child can actually observe, "
                "draw, or compare over time with a grown-up. Do not ask the child to infer a specific fungus's food "
                "or underground job from nearby objects. Restate that the visible mushroom makes spores."
            )
    if re.search(r"\bhole punch\b", combined):
        requirements.append(
            "A child hole-punch activity must say that a grown-up supervises and operates or helps operate a stiff "
            "punch, and must keep fingers clear. Do not suggest disassembly."
        )
    if re.search(r"\b(?:give|need|want)\b.{0,40}\b(?:answer|explanation)\b", newest.casefold()):
        requirements.append(
            "Deliver the answer itself. Recalling or restating what the child wondered is not an explanation."
        )
    if parent_request and _rejects_activity(parent_request):
        requirements.append(
            "Honor the tone/format repair in straightforward prose. Omit the rejected game, activity, quiz, cutesy "
            "voice, physical_extension, and game-like question; keep the central explanation and any needed safety "
            "reassurance in the main answer fields."
        )
    return requirements


def _needs_visible_public_sources(question: str) -> bool:
    """Keep source lists for claims where recency or stakes make them parent-useful."""

    normalized = re.sub(r"\bfirst[- ]grade(?:r)?\b", "", question.casefold())
    return bool(
        re.search(
            r"\b(?:biggest|smallest|tallest|fastest|oldest|youngest|first|record|current|currently|latest|"
            r"today|recent|medical|medicine|health|doctor|symptom|treatment|dose|dosage)\b",
            normalized,
        )
    )


class ReasoningEngine:
    def __init__(self, backend: ModelBackend | None = None, prompt_dir: str | Path | None = None, *, text_reviewer=None):
        self.backend = backend or StubBackend()
        self.text_reviewer = text_reviewer
        self.prompt_dir = Path(prompt_dir) if prompt_dir else AppConfig.load().root / "prompts"
        self.pedagogy_registry = PedagogyRegistry()
        self.pedagogy_compiler = PedagogyCompiler(self.pedagogy_registry)

    def propose_pedagogy(
        self,
        *,
        policy: ReasoningPolicy,
        context: dict[str, Any],
        event: dict[str, Any],
    ) -> PedagogyPlanProposal:
        candidate = self.backend.complete(
            role="pedagogy_planner",
            system=(
                "Propose semantic choices for one K–2 learning episode. You do not own identifiers, provenance, "
                "learner state, difficulty claims, evidence outcomes, schedules, or side effects. Select only IDs "
                "from the supplied reviewed cards. The current event and bounded context are the only adaptation "
                "inputs.\n\n" + self.pedagogy_registry.router_instruction()
            ),
            payload={
                "event": event,
                "context": context,
                "workflow": policy.workflow,
                "registry_cards": self.pedagogy_registry.planner_cards(),
            },
            response_model=PedagogyPlanProposal,
        )
        return PedagogyPlanProposal.model_validate(self._normalize_pedagogy_proposal(candidate))

    @staticmethod
    def _normalize_pedagogy_proposal(candidate: dict[str, Any]) -> dict[str, Any]:
        """Remove model-authored evidence noise without discarding teaching intent."""

        normalized = dict(candidate)
        evidence = normalized.get("evidence")
        if not isinstance(evidence, dict):
            return normalized
        collection = evidence.get("collection")
        incomplete_future_observation = collection == "future_observation_only" and not (
            evidence.get("observable_behavior") and evidence.get("response_modes")
        )
        if collection == "not_collected" or incomplete_future_observation:
            normalized["evidence"] = {
                "collection": "not_collected",
                "template_id": "not-collected",
                "observable_behavior": None,
                "response_modes": [],
            }
        return normalized

    def approve_pedagogy(
        self,
        proposal: PedagogyPlanProposal,
        *,
        event: dict[str, Any],
        workflow: str,
        personalization_decision: dict[str, Any] | None = None,
    ) -> ApprovedPedagogyPlan:
        return self.pedagogy_compiler.compile(
            proposal,
            event=event,
            workflow=workflow,
            provenance="model_proposal",
            personalization_decision=personalization_decision,
        )

    def run(
        self,
        *,
        policy: ReasoningPolicy,
        context: dict[str, Any],
        event: dict[str, Any],
        pedagogy_plan: ApprovedPedagogyPlan | None = None,
        authorization_preflight: Callable[[], None] | None = None,
    ) -> ReasoningEnvelope:
        def complete(**kwargs: Any) -> dict[str, Any]:
            if authorization_preflight is not None:
                authorization_preflight()
            return self.backend.complete(**kwargs)

        response_model = WORKFLOW_MODELS.get(policy.workflow, GenericOutput)
        supports_code_owned_safety = response_model in {
            PullThreadOutput,
            QuickAnswerOutput,
            InterestSignalOutput,
            FeedbackOutput,
            GenericOutput,
        }
        safety_requirements = set()
        if supports_code_owned_safety:
            event_text = str(event.get("text") or "")
            safety_requirements = (
                safety_response_requirements_for_event(pedagogy_plan, event_text)
                if pedagogy_plan is not None
                else safety_response_requirements_for_text(event_text)
            )
        if safety_requirements:
            parsed = self._code_owned_safety_response(response_model, safety_requirements)
            if parsed is None:
                raise ReasoningRejected(
                    "safety-critical workflow lacks a deterministic response contract",
                    candidate={},
                    critiques=[{"workflow": policy.workflow}],
                )
            self._validate_plan_candidate(
                parsed,
                pedagogy_plan,
                policy.workflow,
                event_text=str(event.get("text") or ""),
            )
            return ReasoningEnvelope(
                workflow=policy.workflow,
                output=parsed.model_dump(mode="json"),
                critiques=[],
                critique_rounds=[],
                revision_rounds=0,
                recovery_strategy=None,
                backend="code_owned",
                model="curiosity-safety-v1",
            )
        release_gate_requirements = _release_gate_requirements(event)
        base_payload = {
            "context": context,
            "event": event,
            "policy": policy.to_dict(),
            "release_gate_requirements": release_gate_requirements,
        }
        if pedagogy_plan is not None:
            base_payload["approved_pedagogy_plan"] = pedagogy_plan.model_dump(mode="json")
        if policy.workflow == "pull_thread":
            base_payload["construction_patterns"] = construction_context()
        candidate = complete(
            role=policy.generator_role,
            system=self._generator_system(policy, pedagogy_plan),
            payload=base_payload,
            response_model=response_model,
        )
        revision_rounds = 0
        validation_repair_rounds = 0
        while True:
            try:
                parsed = self._validate_and_normalize_candidate(candidate, response_model, event)
                parsed = self._with_code_owned_safety_response(parsed, pedagogy_plan, event)
                self._validate_plan_candidate(
                    parsed,
                    pedagogy_plan,
                    policy.workflow,
                    event_text=str(event.get("text") or ""),
                )
                break
            except ReasoningRejected as exc:
                if validation_repair_rounds >= policy.max_revision_rounds:
                    raise
                validation_repair_rounds += 1
                revision_rounds += 1
                candidate = complete(
                    role=policy.generator_role,
                    system=self._generator_system(policy, pedagogy_plan)
                    + " Repair the candidate so it exactly satisfies the response contract.",
                    payload={
                        **base_payload,
                        "candidate": exc.candidate,
                        "validation_errors": exc.critiques,
                    },
                    response_model=response_model,
                )
        has_code_owned_safety_response = bool(
            pedagogy_plan
            and safety_response_requirements_for_event(
                pedagogy_plan,
                str(event.get("text") or ""),
            )
        )
        if isinstance(parsed, (PullThreadOutput, QuickAnswerOutput)) and not has_code_owned_safety_response:
            # Complete an explicitly requested visual before the normal critics so
            # the same review round sees every child-visible field.
            parsed = self._complete_missing_visual(
                policy=policy,
                context=context,
                event=event,
                parsed=parsed,
                pedagogy_plan=pedagogy_plan,
                authorization_preflight=authorization_preflight,
            )
        critic_roles = list(runtime_critic_roles(policy, pedagogy_plan))
        use_text_reviewer = self.text_reviewer is not None and policy.workflow in {"quick_answer", "pull_thread"}
        if use_text_reviewer:
            critic_roles = ["jev_text"]
        critiques: list[CriticResult] = []
        critique_rounds: list[list[CriticResult]] = []
        accumulated_nonpass: list[CriticResult] = []
        semantic_revision_rounds = 0
        recovery_strategy: str | None = None
        resource_only_recovery_applied = False
        while True:
            if use_text_reviewer:
                if authorization_preflight is not None:
                    authorization_preflight()
                critiques = [self.text_reviewer.review(
                    parent_view=format_reviewed_response({"workflow": policy.workflow, "output": parsed.model_dump(mode="json")}),
                    event=event, context=context,
                )]
            else:
                critiques = [
                    CriticResult.model_validate(
                        complete(
                            role=critic,
                            system=self._critic_system(critic, pedagogy_plan),
                            payload={
                                "candidate": parsed.model_dump(mode="json"),
                                "parent_view": format_reviewed_response({
                                    "workflow": policy.workflow, "output": parsed.model_dump(mode="json"),
                                }) if isinstance(parsed, (PullThreadOutput, QuickAnswerOutput)) else None,
                                "context": context,
                                "event": event,
                                "release_gate_requirements": release_gate_requirements,
                                "previous_reviews": [
                                    round_items[critic_roles.index(critic)].model_dump(mode="json")
                                    for round_items in critique_rounds
                                ],
                                "approved_pedagogy_plan": (
                                    pedagogy_plan.model_dump(mode="json") if pedagogy_plan is not None else None
                                ),
                            },
                            response_model=CriticResult,
                        )
                    )
                    for critic in critic_roles
                ]
            critique_rounds.append(critiques)
            if all(review.verdict == CriticVerdict.PASS for review in critiques):
                break
            accumulated_nonpass.extend(review for review in critiques if review.verdict != CriticVerdict.PASS)
            nonpass = [review for review in critiques if review.verdict != CriticVerdict.PASS]
            critique_texts = [
                text for review in nonpass for text in [*review.concerns, *review.required_changes] if text.strip()
            ]
            current_refs = list(getattr(parsed, "resource_refs", []) or [])
            resource_only_failure = bool(
                critique_texts
                and current_refs
                and all(
                    re.search(r"\b(?:resource|source|reference|citation|link|url)s?\b", text, re.IGNORECASE)
                    for text in critique_texts
                )
            )
            if (
                resource_only_failure
                and not resource_only_recovery_applied
                and semantic_revision_rounds >= policy.max_revision_rounds
            ):
                parsed = parsed.model_copy(update={"resource_refs": []})
                resource_only_recovery_applied = True
                recovery_strategy = "remove_unrelated_sources"
                continue
            if (
                any(review.verdict == CriticVerdict.REJECT for review in critiques)
                and semantic_revision_rounds >= policy.max_revision_rounds
            ):
                raise ReasoningRejected(
                    "candidate rejected by critic",
                    candidate=parsed.model_dump(mode="json"),
                    critiques=[x.model_dump(mode="json") for x in critiques],
                    critique_rounds=[
                        [item.model_dump(mode="json") for item in round_items] for round_items in critique_rounds
                    ],
                    recovery_strategy=recovery_strategy,
                )
            if semantic_revision_rounds >= policy.max_revision_rounds:
                raise ReasoningRejected(
                    "candidate did not pass critics within revision budget",
                    candidate=parsed.model_dump(mode="json"),
                    critiques=[x.model_dump(mode="json") for x in critiques],
                    critique_rounds=[
                        [item.model_dump(mode="json") for item in round_items] for round_items in critique_rounds
                    ],
                    recovery_strategy=recovery_strategy,
                )
            semantic_revision_rounds += 1
            revision_rounds += 1
            rebuild_from_scratch = bool(
                policy.final_recovery == "rebuild_from_scratch"
                and semantic_revision_rounds == policy.max_revision_rounds
            )
            if rebuild_from_scratch:
                recovery_strategy = "rebuild_from_scratch"
            revision_payload = {
                **base_payload,
                "critiques": [x.model_dump(mode="json") for x in accumulated_nonpass],
                "required_changes": list(
                    dict.fromkeys(change for critique in accumulated_nonpass for change in critique.required_changes)
                ),
                "critic_concerns": list(
                    dict.fromkeys(concern for critique in accumulated_nonpass for concern in critique.concerns)
                ),
                "rebuild_from_scratch": rebuild_from_scratch,
            }
            if not rebuild_from_scratch:
                revision_payload["candidate"] = parsed.model_dump(mode="json")
            candidate = complete(
                role=policy.generator_role,
                system=self._generator_system(policy, pedagogy_plan)
                + (
                    " FINAL RECOVERY: discard the prior draft and rebuild a smaller answer from scratch. "
                    "Treat every required change as a hard constraint. Remove the specific invalid materials "
                    "and claims. If the critic supplies a workable correction to the activity, use that correction; "
                    "do not abandon it for a more complicated construction. Prefer no optional activity or visual over one "
                    "that conflicts with the context. If the parent explicitly asks for something to do or build, "
                    "preserve the requested number of ideas and answer every open question, using exactly their supplies. "
                    "Simplify the mechanics, not the parent's requested scope. Mentally perform each step: "
                    "state the starting orientation, the movement, and what it should look like. Avoid a named craft "
                    "whose essential steps you cannot explain briefly. Never substitute a lesson about the activity "
                    "for the requested activity. Return only the new structured answer."
                    if rebuild_from_scratch
                    else " Revise the candidate to address every supplied critic concern; do not merely explain the changes."
                ),
                payload=revision_payload,
                response_model=response_model,
            )
            semantic_validation_repairs = 0
            while True:
                try:
                    parsed = self._validate_and_normalize_candidate(candidate, response_model, event)
                    parsed = self._with_code_owned_safety_response(parsed, pedagogy_plan, event)
                    self._validate_plan_candidate(
                        parsed,
                        pedagogy_plan,
                        policy.workflow,
                        event_text=str(event.get("text") or ""),
                    )
                    if isinstance(parsed, (PullThreadOutput, QuickAnswerOutput)) and not has_code_owned_safety_response:
                        # A critic revision may legitimately rebuild the response, but it
                        # may not silently erase a diagram the parent explicitly requested.
                        # Re-complete here so the next critic round reviews the restored
                        # visual together with the revised text.
                        parsed = self._complete_missing_visual(
                            policy=policy,
                            context=context,
                            event=event,
                            parsed=parsed,
                            pedagogy_plan=pedagogy_plan,
                            authorization_preflight=authorization_preflight,
                        )
                    break
                except ReasoningRejected as exc:
                    if semantic_validation_repairs >= 1:
                        exc.critique_rounds = [
                            [item.model_dump(mode="json") for item in round_items] for round_items in critique_rounds
                        ]
                        exc.recovery_strategy = recovery_strategy
                        raise
                    semantic_validation_repairs += 1
                    revision_rounds += 1
                    candidate = complete(
                        role=policy.generator_role,
                        system=self._generator_system(policy, pedagogy_plan)
                        + " Repair only the response contract while preserving every supplied critic constraint.",
                        payload={
                            **revision_payload,
                            "candidate": exc.candidate,
                            "validation_errors": exc.critiques,
                        },
                        response_model=response_model,
                    )
        return ReasoningEnvelope(
            workflow=policy.workflow,
            output=parsed.model_dump(mode="json"),
            critiques=critiques,
            critique_rounds=critique_rounds,
            revision_rounds=revision_rounds,
            recovery_strategy=recovery_strategy,
            backend=self.backend.name,
            model=self.backend.model,
        )

    def _complete_missing_visual(
        self,
        *,
        policy: ReasoningPolicy,
        context: dict[str, Any],
        event: dict[str, Any],
        parsed: PullThreadOutput | QuickAnswerOutput,
        pedagogy_plan: ApprovedPedagogyPlan | None = None,
        authorization_preflight: Callable[[], None] | None = None,
    ) -> PullThreadOutput | QuickAnswerOutput:
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        mode = str(metadata.get("response_visual_mode") or "")
        question = str(event.get("text") or "")
        visual_requested = metadata.get("response_visual_requested") is True
        needs_imagination_visual = visual_requested and parsed.visual is None and mode != "off"
        if not needs_imagination_visual:
            return parsed

        child = context.get("child") if isinstance(context.get("child"), dict) else {}
        completion_model = PullThreadOutput if isinstance(parsed, PullThreadOutput) else QuickAnswerOutput
        try:
            if authorization_preflight is not None:
                authorization_preflight()
            candidate = self.backend.complete(
                role=policy.generator_role,
                system=self._generator_system(policy, pedagogy_plan)
                + (
                    " VISUAL COMPLETION PASS: Copy the supplied structured answer exactly and add only the requested "
                    "primary visual. Choose researched_photo if seeing a real "
                    "public animal, plant, landform, spacecraft, planet, museum object, artwork, or observable "
                    "phenomenon materially helps; otherwise add comparison_cards or an activity_sequence whose "
                    "code-rendered labels and panels carry the learning. Generated art may decorate that structure, "
                    "but never use a standalone decorative illustration. A research query and decorative subject "
                    "must be generic and public. Generated art must not teach facts, labels, counts, scale, sequence, "
                    "anatomy, or a scientific mechanism. Exclude names, relationships, private resources, school, "
                    "home, health, URLs, brands, and copyrighted characters. A printable and one primary visual may "
                    "both be present only when they do different useful jobs."
                ),
                payload={
                    "candidate": parsed.model_dump(mode="json"),
                    "context": {"child": {"grade": child.get("grade")}},
                    "event": {"type": event.get("type"), "text": question},
                    "visual_completion": {
                        "mode": mode,
                        "preserve_reviewed_text": True,
                        "imagination_visual_needed": needs_imagination_visual,
                        "accepted_visual_kinds": ["researched_photo", "comparison_cards", "activity_sequence"],
                    },
                },
                response_model=completion_model,
            )
            completed = self._validate_and_normalize_candidate(candidate, completion_model, event)
        except Exception as exc:
            logger.warning("optional visual completion failed error_type=%s", exc.__class__.__name__)
            return parsed
        if not isinstance(completed, (PullThreadOutput, QuickAnswerOutput)) or type(completed) is not type(parsed):
            logger.warning("optional visual completion returned no usable output")
            return parsed
        try:
            self._validate_plan_candidate(
                completed,
                pedagogy_plan,
                policy.workflow,
                event_text=question,
            )
        except ReasoningRejected as exc:
            logger.warning(
                "discarded visual completion that violated the approved plan errors=%s",
                exc.critiques,
            )
            return parsed
        visual = completed.visual
        if visual is None:
            logger.warning("optional visual completion returned no safe imagination visual")
        return parsed.model_copy(update={"visual": visual})

    @staticmethod
    def _validate_and_normalize_candidate(
        candidate: dict[str, Any],
        response_model: type[OutputModel],
        event: dict[str, Any],
    ) -> OutputModel:
        candidate = ReasoningEngine._without_invalid_observation_candidates(candidate)
        if response_model not in {PullThreadOutput, QuickAnswerOutput}:
            return ReasoningEngine._validate_candidate(candidate, response_model)
        candidate_without_visual = {**candidate, "visual": None}
        if not _needs_visible_public_sources(str(event.get("text") or "")):
            candidate_without_visual["resource_refs"] = []
        parsed = ReasoningEngine._validate_candidate(candidate_without_visual, response_model)
        if not isinstance(parsed, (PullThreadOutput, QuickAnswerOutput)):  # pragma: no cover
            return parsed
        question = str(event.get("text") or "")
        if isinstance(parsed, PullThreadOutput) and _activity_needs_printable(parsed.physical_extension):
            logger.warning("discarded incomplete optional activity aid before review")
            parsed = parsed.model_copy(update={"physical_extension": None})
        if isinstance(parsed, QuickAnswerOutput) and is_generic_response_only(parsed.answer, question):
            raise ReasoningRejected(
                "quick answer contained only generic transition filler",
                candidate=candidate,
                critiques=[
                    {
                        "validation": (
                            "Replace the transition with a complete, concrete answer to the parent's question."
                        )
                    }
                ],
            )
        if isinstance(parsed, PullThreadOutput):
            explanation = parsed.parent_response or " ".join(part for part in (parsed.hook, parsed.nugget) if part.strip())
            if is_generic_response_only(explanation, question):
                raise ReasoningRejected(
                    "learning thread contained only generic transition filler",
                    candidate=candidate,
                    critiques=[
                        {
                            "validation": (
                                "Replace the transition with a concrete explanation and one useful learning move."
                            )
                        }
                    ],
                )
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        retry = metadata.get("response_retry") if isinstance(metadata.get("response_retry"), dict) else {}
        previous_answer = (
            retry.get("previous_answer_to_avoid")
            if isinstance(retry.get("previous_answer_to_avoid"), dict)
            else {}
        )
        if retry and previous_answer and not _retry_explanation_is_distinct(previous_answer, parsed):
            raise ReasoningRejected(
                "retry repeated the previous explanation",
                candidate=candidate,
                critiques=[
                    {
                        "validation": (
                            "Use a substantively different explanation or example, not the previous answer with "
                            "minor wording or an added activity."
                        )
                    }
                ],
            )
        explicit_small_try = bool(
            re.search(
                r"\b(?:small|little|quick|simple|tiny)?\s*(?:thing|way)\s+(?:we|i)\s+can\s+try\b|"
                r"\b(?:can|could)\s+(?:we|i)\s+try\s+(?:one\s+)?(?:small|little|quick|simple|tiny)?\s*(?:thing|way)\b|"
                r"\btry\s+(?:it|this)\s+with\b",
                question.casefold(),
            )
        )
        explicit_play_request = bool(
            re.search(
                r"\b(?:please|can|could|make|want|wants|give)\b.{0,90}\b(?:activity|game|experiment|challenge)\b",
                question.casefold(),
            )
            and not re.search(
                r"\b(?:not|no|without|don't|do not)\b.{0,35}\b(?:activity|game|experiment|challenge)\b|"
                r"\b(?:activity|game|experiment|challenge)\b.{0,18}\b(?:not|no)\b",
                question.casefold(),
            )
        )
        if _requires_direct_answer(question, "pull_thread"):
            explicit_play_request = False
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        response_revision = (
            metadata.get("response_revision") if isinstance(metadata.get("response_revision"), dict) else {}
        )
        parent_revision = str(response_revision.get("parent_request") or "")
        rejects_activity = _rejects_activity(parent_revision)
        if isinstance(parsed, QuickAnswerOutput) and rejects_activity:
            # A concise factual repair should not fail closed merely because a model
            # appended an unrequested imperative mini-activity. Preserve descriptive
            # comparisons and explanations; remove only sentences that tell the family
            # to perform something, plus optional prompts/actions that recreate it.
            sentences = re.split(r"(?<=[.!?])\s+", parsed.answer.strip())
            kept = [
                sentence
                for sentence in sentences
                if not re.match(
                    r"^(?:please\s+)?(?:put|place|take|try|use|make|draw|write|hold|stand|move|"
                    r"blow|pour|cut|tape|build|choose|pick|pretend|ask|watch|look|touch|set)\b",
                    sentence.strip(),
                    re.IGNORECASE,
                )
            ]
            if kept:
                parsed = parsed.model_copy(
                    update={
                        "answer": " ".join(kept),
                        "question_back": None,
                        "actions": [],
                    }
                )
        if (
            isinstance(parsed, PullThreadOutput)
            and not explicit_play_request
            and re.search(r"\b(?:go|take|one level|a little)\b.{0,30}\bdeeper\b", question.casefold())
        ):
            parsed = parsed.model_copy(update={"physical_extension": None})
        if rejects_activity:
            explicit_play_request = False
        prior_reviewed = (
            response_revision.get("previous_reviewed_response")
            if isinstance(response_revision.get("previous_reviewed_response"), dict)
            else {}
        )
        constrained_activity_revision = bool(
            parent_revision
            and prior_reviewed.get("physical_extension")
            and re.search(
                r"\b(?:minutes?|ordinary paper|materials?|supplies|we only have|use only|keep it to)\b",
                parent_revision.casefold(),
            )
        )
        different_activity_revision = bool(
            parent_revision
            and re.search(r"\b(?:genuinely|completely|totally)?\s*different\b", parent_revision.casefold())
            and re.search(r"\b(?:activity|game|thing|idea)\b", parent_revision.casefold())
        )
        if (
            isinstance(parsed, PullThreadOutput)
            and (different_activity_revision or constrained_activity_revision)
            and parsed.physical_extension is None
            and re.search(
                r"\b(?:put|place|take|try|use|make|draw|write|fold|mark|circle|cut|tape|spin|move|slide|build)\b",
                parsed.show.casefold(),
            )
        ):
            # Models occasionally place a complete hands-on revision in `show`.
            # Preserve the useful idea, but put it in the structured activity slot
            # so Slack can render it as an actionable replacement.
            sentences = [
                sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", parsed.show.strip()) if sentence.strip()
            ]
            parsed = parsed.model_copy(
                update={
                    "show": "Start with the steps below.",
                    "physical_extension": PhysicalExtension(
                        title="Try the different hands-on idea",
                        materials=["ordinary household items named in the steps"],
                        instructions=sentences[:6] or [parsed.show.strip()],
                        prompts=["What does this model help you notice?"],
                        parent_effort="very_low",
                    ),
                }
            )
        if (isinstance(parsed, PullThreadOutput) and constrained_activity_revision
                and parsed.physical_extension is None and not parsed.parent_response):
            raise ReasoningRejected(
                "activity constraint revision dropped the usable activity",
                candidate=candidate,
                critiques=[
                    {
                        "validation": (
                            "Preserve the prior activity as physical_extension and supply the actual revised materials "
                            "and steps. A prose summary or promise is not the requested constrained activity."
                        )
                    }
                ],
            )
        component_request = "\n".join(part for part in (question, parent_revision) if part)
        explicit_prompt_request = bool(
            re.search(r"\b(?:prompts?|clues?|orders?|things? to say)\b", component_request.casefold())
        )
        explicit_single_question_request = bool(
            re.search(
                r"\b(?:one|a|single)\s+(?:follow[- ]up\s+)?question\s+(?:i|we)\s+can\s+ask\b",
                component_request.casefold(),
            )
        )
        if (
            isinstance(parsed, PullThreadOutput)
            and (explicit_small_try or explicit_play_request)
            and parsed.physical_extension is None
            and not parsed.parent_response
        ):
            raise ReasoningRejected(
                "explicit hands-on request has no actionable physical extension",
                candidate=candidate,
                critiques=[
                    {"validation": "Add a tiny physical_extension with the named material and concrete first action."}
                ],
            )
        if (isinstance(parsed, PullThreadOutput) and explicit_single_question_request
                and not parsed.ask and not parsed.parent_response):
            raise ReasoningRejected(
                "requested parent question is missing",
                candidate=candidate,
                critiques=[{"validation": "Put the one requested ready-to-use question in ask."}],
            )
        source_context = metadata.get("source_context") if isinstance(metadata.get("source_context"), dict) else {}
        scenario_text = f"{question}\n{str(source_context.get('text') or '')}".casefold()
        if (
            isinstance(parsed, (PullThreadOutput, QuickAnswerOutput))
            and re.search(
                r"\b(?:vote|voting|election)\b",
                scenario_text,
            )
            and re.search(r"\b(?:boss|choose|pick)\b", scenario_text)
        ):
            if isinstance(parsed, PullThreadOutput):
                answer_text = " ".join((parsed.parent_response or "", parsed.hook, parsed.show, parsed.ask, parsed.nugget)).casefold()
            else:
                answer_text = parsed.answer.casefold()
            names_person_choice = bool(re.search(r"\b(?:person|people|leader|representative|candidate)\b", answer_text))
            names_idea_choice = bool(re.search(r"\b(?:idea|plan|rule|question|proposal|law)\b", answer_text))
            names_rule_variation = bool(
                re.search(
                    r"\b(?:different|some)\b.{0,45}\b(?:elections?|votes?|rules?)\b|"
                    r"\b(?:elections?|votes?|rules?)\b.{0,45}\b(?:different|vary|change)\b",
                    answer_text,
                )
            )
            names_limits = bool(re.search(r"\b(?:law|court|constitution|limit|rule)s?\b", answer_text))
            if not all((names_person_choice, names_idea_choice, names_rule_variation, names_limits)):
                raise ReasoningRejected(
                    "voting explanation omitted a required distinction",
                    candidate=candidate,
                    critiques=[
                        {
                            "validation": (
                                "Say explicitly that some votes choose a person while others choose an idea, plan, "
                                "rule, or question; different elections can use different rules; and elected leaders "
                                "still have to follow laws or other limits. Keep all three ideas in the visible answer."
                            )
                        }
                    ],
                )
        if (
            isinstance(parsed, PullThreadOutput)
            and parsed.physical_extension is not None
            and "camera" in " ".join(parsed.physical_extension.instructions).casefold()
            and re.search(r"\b(?:remote|infrared|invisible light)\b", scenario_text)
        ):
            extension_text = " ".join(
                [
                    *parsed.physical_extension.instructions,
                    *parsed.physical_extension.prompts,
                ]
            ).casefold()
            if not re.search(r"\b(?:some|may|might|can)\b.{0,45}\b(?:camera|filter|show|see)\b", extension_text):
                parsed = parsed.model_copy(
                    update={
                        "physical_extension": parsed.physical_extension.model_copy(
                            update={
                                "instructions": [
                                    *parsed.physical_extension.instructions,
                                    (
                                        "If no flash appears, that is okay—some cameras or camera modes filter "
                                        "infrared light."
                                    ),
                                ]
                            }
                        )
                    }
                )
        if (
            isinstance(parsed, PullThreadOutput)
            and explicit_prompt_request
            and (parsed.physical_extension is None or not parsed.physical_extension.prompts)
            and not parsed.parent_response
        ):
            raise ReasoningRejected(
                "requested ready-to-use prompts are missing",
                candidate=candidate,
                critiques=[
                    {
                        "validation": (
                            "Put the actual requested prompts, clues, orders, questions, or lines to say in "
                            "physical_extension.prompts. Do not refer to items 'below' unless they are present."
                        )
                    }
                ],
            )
        output = parsed.model_dump(mode="json")
        output["visual"] = candidate.get("visual")
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        visual_policy = str(metadata.get("response_visual_policy") or "")
        visual_allowed = metadata.get("response_visual_requested") is True or visual_policy in {
            "explicit",
            "supportive",
        }
        visual = normalize_response_visual(question, output) if visual_allowed else None
        if visual_policy == "supportive" and visual is not None and visual.kind not in {
            "comparison_cards",
            "activity_sequence",
        }:
            visual = None
        return parsed.model_copy(update={"visual": visual})

    @staticmethod
    def _without_invalid_observation_candidates(candidate: dict[str, Any]) -> dict[str, Any]:
        if "graph_updates" in candidate:
            logger.warning(
                "discarded legacy model graph updates count=%s",
                len(candidate.get("graph_updates") or []) if isinstance(candidate.get("graph_updates"), list) else 0,
            )
            candidate = {key: value for key, value in candidate.items() if key != "graph_updates"}
        updates = candidate.get("observation_candidates")
        if not isinstance(updates, list):
            return candidate
        valid_updates: list[Any] = []
        for index, update in enumerate(updates):
            try:
                ObservationCandidate.model_validate(update)
            except ValidationError as exc:
                logger.warning(
                    "discarded invalid observation candidate index=%s validation_types=%s",
                    index,
                    [item["type"] for item in exc.errors()],
                )
                continue
            valid_updates.append(update)
        if len(valid_updates) == len(updates):
            return candidate
        return {**candidate, "observation_candidates": valid_updates}

    @staticmethod
    def _validate_candidate(candidate: dict[str, Any], response_model: type[OutputModel]) -> OutputModel:
        try:
            return response_model.model_validate(candidate)
        except ValidationError as exc:
            raise ReasoningRejected(
                "generator returned invalid structured output",
                candidate=candidate,
                critiques=[{"validation": exc.errors(include_context=False)}],
            ) from exc

    def _prompt(self, name: str, fallback: str) -> str:
        path = self.prompt_dir / name
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return fallback

    @staticmethod
    def _code_owned_safety_response(
        response_model: type[OutputModel],
        requirements: set[str],
    ) -> OutputModel | None:
        """Construct the reviewed response without calling a generator or critic."""

        if not requirements:
            return None
        opening: list[str] = []
        guidance: list[str] = []
        if "immediate_trusted_grown_up" in requirements:
            opening.append(
                "I’m really sorry this feels so hard. You deserve help and should not be alone with this feeling."
            )
            guidance.append(
                "Tell a trusted grown-up right now so they can stay with you and help keep you safe."
            )
        if "explicit_hazard_prohibition" in requirements:
            opening.append(
                "Fingers and objects must never go inside an electrical outlet, toaster, or other powered appliance."
            )
            guidance.append(
                "Electricity can cause severe shock, burns, or fire. Move away and tell a trusted grown-up, who can "
                "disconnect the power and keep everyone away."
            )
        if "explicit_appliance_water_prohibition" in requirements:
            opening.append(
                "A powered appliance must not be wet or in or near water. "
                "Do not plug in a wet powered appliance. Do not turn on a wet powered appliance. "
                "Do not use a powered appliance near water."
            )
            guidance.append(
                "Water and electricity can cause severe shock, burns, or fire. Move away and tell a trusted grown-up, "
                "who can disconnect the power and keep everyone away."
            )
        if "explicit_ignition_prohibition" in requirements:
            opening.append("Do not light or handle matches, lighters, candles, flames, a gas stove, or a gas burner.")
            guidance.append(
                "They can cause serious burns or fire. Move away and tell a trusted grown-up right now."
            )
        if "explicit_chemical_prohibition" in requirements:
            opening.append("Never taste bleach or household cleaners; stay away from them.")
            guidance.append(
                "They can poison, burn, or make harmful fumes. Move away and tell a trusted grown-up right now."
            )
        if "explicit_sharp_tool_prohibition" in requirements:
            opening.append(
                "Do not use or handle knives, scissors, razor blades, box cutters, saws, chainsaws, axes, nail guns, "
                "needles, or powered drills."
            )
            guidance.append(
                "They can cause serious cuts or punctures. Move away and tell a trusted grown-up right now."
            )
        if response_model is PullThreadOutput:
            return PullThreadOutput(
                hook=" ".join(opening),
                show="",
                ask="",
                nugget=" ".join(guidance),
                next_possible_concepts=[],
                physical_extension=None,
                visual=None,
                observation_candidates=[],
                actions=[],
                resource_refs=[],
            )
        if response_model is QuickAnswerOutput:
            return QuickAnswerOutput(
                answer=" ".join([*opening, *guidance]),
                notice=None,
                question_back=None,
                next_possible_concepts=[],
                visual=None,
                observation_candidates=[],
                actions=[],
                resource_refs=[],
            )
        safety_text = " ".join([*opening, *guidance])
        if response_model is InterestSignalOutput:
            return InterestSignalOutput(summary=safety_text, observation_candidates=[], actions=[])
        if response_model is FeedbackOutput:
            return FeedbackOutput(acknowledged=safety_text, observation_candidates=[], actions=[])
        if response_model is GenericOutput:
            return GenericOutput(summary=safety_text, observation_candidates=[], actions=[])
        return None

    @classmethod
    def _with_code_owned_safety_response(
        cls,
        parsed: OutputModel,
        plan: ApprovedPedagogyPlan | None,
        event: dict[str, Any],
    ) -> OutputModel:
        """Defense in depth if a future route reaches model parsing before the safety branch."""

        if plan is None:
            return parsed
        requirements = safety_response_requirements_for_event(plan, str(event.get("text") or ""))
        replacement = cls._code_owned_safety_response(type(parsed), requirements)
        return replacement or parsed

    @staticmethod
    def _validate_plan_candidate(
        parsed: OutputModel,
        plan: ApprovedPedagogyPlan | None,
        workflow: str,
        *,
        event_text: str | None = None,
    ) -> None:
        if plan is None:
            return
        candidate = parsed.model_dump(mode="json")
        errors = validate_generated_against_plan(
            plan,
            candidate,
            workflow=workflow,
            event_text=event_text,
        )
        blocking_errors = ReasoningEngine._blocking_plan_validation_errors(plan, errors)
        if blocking_errors:
            raise ReasoningRejected(
                "candidate violates approved pedagogy plan",
                candidate=candidate,
                critiques=[{"pedagogy_plan_validation": blocking_errors}],
            )
        if errors:
            logger.info(
                "accepted ordinary pedagogy guidance deviations workflow=%s findings=%s",
                workflow,
                errors,
            )

    @staticmethod
    def _blocking_plan_validation_errors(
        plan: ApprovedPedagogyPlan,
        errors: list[str],
    ) -> list[str]:
        """Reserve hard rejection for concrete child-visible contract failures."""

        exact = {
            "approved plan requires a visible answer first",
            "response violates healthy closure",
            "sensitive support may not be trivialized with playful mechanics",
            "self-harm support lacks immediate trusted-grown-up guidance",
            "physical-hazard answer lacks an explicit prohibition, danger, or trusted-grown-up handoff",
            "appliance-water answer does not explicitly prohibit a wet or water-related appliance action",
            "generated response directs the child toward self-harm",
            "generated response introduces an appliance-water action absent from the approved safety plan",
            "elevated-risk activity lacks the approved grown-up safety boundary",
            "generated activity assigns hazardous handling to the child",
            "generated response assigns hazardous handling to the child",
            "no-writing request still requires writing",
            "no-writing request lacks a visible non-writing response mode",
            "fine-motor accommodation still requires cutting or tiny pieces",
            "read-aloud accommodation is not visible in the child-facing output",
            "quick answer exceeds its reading-load ceiling",
            "learning thread exceeds its reading-load ceiling",
            "non-collection plan may not manufacture evidence",
        }
        if plan.question_intent == "sensitive_support":
            exact.update(
                {
                    "direct or sensitive plan forbids an unsolicited activity",
                    "direct or sensitive plan forbids artifact actions",
                    "direct or sensitive plan forbids a compulsory question-back",
                    "direct or sensitive plan forbids an activity hidden in prose",
                }
            )
        return [error for error in errors if error in exact]

    def _generator_system(
        self,
        policy: ReasoningPolicy,
        pedagogy_plan: ApprovedPedagogyPlan | None = None,
    ) -> str:
        base = self._prompt(
            "generator-v1.md",
            "You are one semantic component inside Curiosity Engine, not the orchestrator.",
        )
        visual_policy = self._prompt(
            "visual-policy-v1.md",
            "Choose visuals only when they materially help the child observe, understand, imagine, or do.",
        )
        system = (
            f"{base}\n\n{visual_policy}\nWorkflow: {policy.workflow}. "
            f"Allowed semantic tools: {', '.join(policy.allowed_tools) or 'none'}."
            " When the parent asks for a game, something to build, or another activity, give a playable action: "
            "name only available supplies, say exactly how to start and what the child does, and give a simple "
            "ending or next turn. A title plus a question or scientific explanation is not an activity. "
            "An 'another idea' revision keeps the original request for a usable game while changing its main action."
            " For pull_thread, write the complete parent-facing reply in parent_response, leaving hook/show/ask/nugget "
            "empty. Choose the shape the request needs: two sentences, a short explanation, several distinct ideas, "
            "or a sequence of actions. Match the requested number and length; do not force one lesson or add a quiz. "
            "For a few ideas give two or three concrete choices the parent can start, rather than topic labels. "
            "Resolve references using the supplied thread history and source context; the latest question controls "
            "what to answer, even after a printable or menu. A new topic supersedes the old topic. "
            "For a tiny pretend game put its complete playable loop in parent_response and leave physical_extension "
            "null to avoid duplicate instructions. For a construction_pattern, put only a short introduction in "
            "parent_response: code appends the exact steps. Other activities may use either prose or structured "
            "steps, never repeat both. Follow the construction_patterns guidance when a pattern fits."
        )
        if pedagogy_plan is not None:
            system += (
                "\n\nUse the approved pedagogy plan as a teaching guide, preserving the parent's requested job and "
                "explicit constraints. Its authority bindings are immutable; do not emit, "
                "rename, or replace authority IDs, evidence outcomes, schedules, or learner-state claims.\n\n"
                + self.pedagogy_registry.instruction_bundle(pedagogy_plan)
            )
        return system

    def _critic_system(self, role: str, pedagogy_plan: ApprovedPedagogyPlan | None = None) -> str:
        rubrics = {
            "critic_family_usefulness": (
                "Judge the response as a busy trusted parent deciding whether it is worth sharing with an early-"
                "elementary child. Inspect parent_view as the actual delivered text; empty legacy fields do not "
                "indicate missing content. Require a direct, accurate answer to every material part of the request, one "
                "concrete grade-ready mechanism or comparison, and a proportionate next step only when it helps. "
                "Reject generic filler, repeated UI/progress language, decorative visual suggestions, needless "
                "lesson structure, unexplained jargon, missed practical questions, or an answer that makes the "
                "parent do more work to discover what was asked. Preserve relevant context and uncertainty without "
                "burying the answer in caveats. Do not invent exceptions from a different, unstated setup. "
                "Evaluate the whole answer against the "
                "actual source context, including the photographed arrangement; read qualifications in the next "
                "sentence before alleging a factual error. On repair, check previous_reviews: if the candidate "
                "satisfies the requested correction, pass it unless a concrete material problem remains. Do not "
                "reverse a previous correction based on hypothetical edge cases or demand optional precision. "
                "For a requested build, mentally execute every construction step: "
                "does it actually produce the claimed object and function? Request repair if an essential fold, "
                "connection or setup step is missing. A craft name is not a substitute for instructions. "
                "Pass forgiving constructions with understandable start-and-repeat cues; exact dimensions or "
                "extra precision are optional unless the object cannot function without them. An invitation to "
                "try and observe an uncertain result is valid. Do not block usable play for a nicer wording. "
                "Check every causal claim against the actual setting and equipment: a mechanism in another "
                "setup does not explain this one unless the relevant variables still apply. "
                "For a game, check that the parent knows the first move and what the child does next. A tiny "
                "imitation or pretend loop can be complete in one sentence; do not demand scoring or extra rules."
            ),
            "critic_factual": (
                "Find materially unsupported, overstated, misleading, or uncertain factual claims. Reject a model "
                "whose visible behavior is caused by a different mechanism while being presented as evidence for the "
                "target mechanism; superficial visual similarity is not a valid demonstration. Treat every supplied "
                "release_gate_requirement as a hard check: a qualifier without its important exception, a list of "
                "examples without the causal variables asked for, or a one-way category correction is incomplete. "
                "Reject vague phrases such as 'it changes,' 'it knows,' or 'the computer remembers' when the child "
                "asked why or how and a concrete grade-ready mechanism can be named. Reject a shortened revision "
                "that drops the causal, historical, qualification, or safety fact that made the prior answer useful."
            ),
            "critic_pedagogy": (
                "Coach for an age-appropriate, interesting learning response. The approved pedagogy plan is guidance, "
                "not an exact script: pass a useful response that reaches the learning goal through a sound alternative "
                "move, and do not reject it solely for omitting an optional activity or plan step. Use revise only "
                "when a concrete problem would make the response misleading or unusable for this child; optional "
                "questions, comparisons, tests and examples are not blocking requirements. A concise "
                "direct answer to a child's factual question is appropriate. Challenge should be one conceptual rung, "
                "while support should scaffold the current idea rather than lower expectations or solve it for the child."
                " When a response revision requests a mystery, story, adventure, or different game, reject a merely "
                "renamed or reworded explanation. Require a concrete new child action or role that changes the play. "
                "A story, rhythm or pretend game does not need an experiment, prediction, or evidence check."
                " Treat the parent's explicit no-worksheet, no-competition, no-counting, no-quiz, time, length, and "
                "one-step constraints as hard requirements; reject a renamed version of a forbidden mechanic."
            ),
            "critic_context": (
                "Find materially missed relevant context, harmful use of irrelevant retrieved context, or unsupported "
                "child-model assumptions. Reject above-level or below-level adaptation based only on mention count, "
                "exposure, completion, or repeated repair turns. Private context may be correctly omitted when it is "
                "not relevant. When source_context is supplied, reject work that changes its distinctive child action, "
                "sensory medium, named objects, or practical constraint without a parent request to do so."
            ),
            "critic_parent_effort": "Attack anything a busy parent is unlikely to do at the stated moment.",
            "critic_epistemic": "Reject durable child-state claims that lack multiple attributable pieces of evidence.",
            "critic_plan_fidelity": (
                "Compare the candidate to the exact approved pedagogy plan. Reject answer evasion, a different "
                "strategy, theme-only play, unapproved activity blocks, missing frustration/stop behavior, or closure "
                "that creates urgency, streaks, cliffhangers, scheduling, or required continuation."
            ),
            "critic_accessibility": (
                "Reject K–2 reading or action overload and mechanics that depend only on color, fluent reading, "
                "writing, speech, or fine-motor cutting. Require equivalent response choices and usable substitutions."
            ),
            "critic_evidence": (
                "Reject generated scores, mastery claims, evidence outcomes, or implied independence. An evidence "
                "opportunity is only a future observation template; generation, help, completion, delivery, printing, "
                "feedback, and critic revision are never learner evidence."
            ),
            "critic_safety": (
                "For sensitive content, require a calm concrete non-diagnostic answer and proportionate grown-up "
                "support. Reject invented motives, playful mystery mechanics, assessment, spectacle, or unnecessary "
                "detail. For every typed physical hazard in the approved plan, reject any instruction that assigns "
                "handling to the child; a separate sentence promising adult supervision does not make a child-owned "
                "hazardous step safe. Require the grown-up to own the exact hazardous action."
            ),
            "critic_visual": (
                "Find visuals that are misleading, illegible, malformed, or merely decorative when the proposed "
                "activity has an obvious child-usable printout, diagram, play piece, comparison, or recording need. "
                "A pleasant topic picture does not satisfy an activity's visual need."
            ),
        }
        system = (
            self._prompt("critic-v1.md", "Return a strict verdict: pass, revise, or reject.")
            + "\n"
            + rubrics.get(role, "Adversarially find reasons this candidate should not ship.")
        )
        if pedagogy_plan is not None and role in {
            "critic_plan_fidelity",
            "critic_accessibility",
            "critic_evidence",
            "critic_safety",
        }:
            system += (
                "\n\nThe following selected, version-bound teaching and domain instructions are also the "
                "critic checklist. Reject a candidate that names the strategy but does not enact its child-facing "
                "moves, accessibility choices, scaffold/fade sequence, stop rule, or safety boundary.\n\n"
                + self.pedagogy_registry.instruction_bundle(pedagogy_plan)
            )
        return system
