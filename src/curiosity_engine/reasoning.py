from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import AppConfig
from .contracts import (
    CriticResult,
    CriticVerdict,
    DirectorOutput,
    FeedbackOutput,
    GenericOutput,
    GraphMutation,
    InterestSignalOutput,
    PullThreadOutput,
    ReasoningEnvelope,
)
from .visuals import normalize_response_visual, should_attempt_decorative_visual

OutputModel = TypeVar("OutputModel", bound=BaseModel)

logger = logging.getLogger(__name__)


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
            return self._pull_thread(text, context)
        if response_model is InterestSignalOutput:
            return {
                "summary": f"Captured as a tentative interest signal: {text}",
                "graph_updates": [
                    {
                        "kind": "upsert_node",
                        "node_kind": "interest",
                        "label": text,
                        "confidence": 0.55,
                        "state": {"epistemic_state": "observation"},
                    }
                ],
                "actions": [],
            }
        if response_model is FeedbackOutput:
            return {
                "acknowledged": "Feedback saved. It will be treated as one observation, not a durable trait.",
                "graph_updates": [
                    {
                        "kind": "add_observation",
                        "observation_kind": "experience_feedback",
                        "text": text,
                        "confidence": 1.0,
                        "state": {},
                    }
                ],
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
        return {"summary": text or "No additional response.", "graph_updates": [], "actions": []}

    @staticmethod
    def _pull_thread(text: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        del context  # Canned demo cases stay intentionally narrow; hosted models use the full bounded context.
        lowered = text.casefold()
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
                "graph_updates": [
                    {
                        "kind": "upsert_node",
                        "node_kind": "concept",
                        "label": "distance and perspective",
                        "confidence": 0.55,
                        "state": {"knowledge_state": "exposed", "epistemic_state": "observation"},
                    }
                ],
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
                "graph_updates": [],
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
                "graph_updates": [],
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
                "graph_updates": [],
                "actions": [],
                "resource_refs": [],
            }
        if "sauropod" in lowered or ("dinosaur" in lowered and "long neck" in lowered):
            return {
                "hook": "Sauropods were long-necked plant eaters that could reach leaves at different heights.",
                "show": "Place three paper leaves low, medium, and high, then compare which one your pretend sauropod can reach.",
                "ask": "Which leaf was easiest for your dinosaur to eat, and what made it easier?",
                "nugget": "A long neck helped a sauropod reach plants without moving its whole heavy body each time.",
                "next_possible_concepts": ["body adaptations", "plant eaters", "height and reach"],
                "physical_extension": {
                    "title": "Sauropod leaf reach",
                    "instructions": [
                        "Color and cut out the three leaf targets.",
                        "Place them low, medium, and high around one room.",
                        "Keep your feet planted, reach like a long neck, and compare the targets.",
                    ],
                    "materials": ["printed leaf page", "coloring utensils", "tape", "grown-up scissors"],
                    "parent_effort": "low",
                    "printable": {
                        "kind": "target_set",
                        "title": "Sauropod Leaf Reach",
                        "child_directions": "Color and cut out the leaves. Put them at three heights, then reach like a sauropod!",
                        "parent_setup": "Help with cutting if needed, then tape the leaves low, medium, and high.",
                        "pedagogical_value": "The page becomes the three targets used to compare reach instead of merely illustrating a dinosaur.",
                        "pieces": [
                            {"label": "LOW", "prompt": "easy snack", "shape": "leaf"},
                            {"label": "MEDIUM", "prompt": "stretch snack", "shape": "leaf"},
                            {"label": "HIGH", "prompt": "sky snack", "shape": "leaf"},
                        ],
                    },
                },
                "visual": None,
                "graph_updates": [],
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
                "graph_updates": [],
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
                "graph_updates": [],
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
                "graph_updates": [],
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
                "graph_updates": [],
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
                "graph_updates": [],
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
            "graph_updates": [],
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
    "pedagogy": "critic_pedagogy",
    "context": "critic_context",
    "parent_effort": "critic_parent_effort",
    "epistemic": "critic_epistemic",
    "visual": "critic_visual",
}


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
    "interest_signal": InterestSignalOutput,
    "feedback": FeedbackOutput,
    "weekly_reflection": DirectorOutput,
    "generic": GenericOutput,
}


class ReasoningEngine:
    def __init__(self, backend: ModelBackend | None = None, prompt_dir: str | Path | None = None):
        self.backend = backend or StubBackend()
        self.prompt_dir = Path(prompt_dir) if prompt_dir else AppConfig.load().root / "prompts"

    def run(self, *, policy: ReasoningPolicy, context: dict[str, Any], event: dict[str, Any]) -> ReasoningEnvelope:
        response_model = WORKFLOW_MODELS.get(policy.workflow, GenericOutput)
        base_payload = {"context": context, "event": event, "policy": policy.to_dict()}
        candidate = self.backend.complete(
            role=policy.generator_role,
            system=self._generator_system(policy),
            payload=base_payload,
            response_model=response_model,
        )
        revision_rounds = 0
        validation_repair_rounds = 0
        while True:
            try:
                parsed = self._validate_and_normalize_candidate(candidate, response_model, event)
                break
            except ReasoningRejected as exc:
                if validation_repair_rounds >= policy.max_revision_rounds:
                    raise
                validation_repair_rounds += 1
                revision_rounds += 1
                candidate = self.backend.complete(
                    role=policy.generator_role,
                    system=self._generator_system(policy)
                    + " Repair the candidate so it exactly satisfies the response contract.",
                    payload={
                        **base_payload,
                        "candidate": exc.candidate,
                        "validation_errors": exc.critiques,
                    },
                    response_model=response_model,
                )
        critiques: list[CriticResult] = []
        critique_rounds: list[list[CriticResult]] = []
        accumulated_nonpass: list[CriticResult] = []
        semantic_revision_rounds = 0
        recovery_strategy: str | None = None
        while True:
            critiques = [
                CriticResult.model_validate(
                    self.backend.complete(
                        role=critic,
                        system=self._critic_system(critic),
                        payload={"candidate": parsed.model_dump(mode="json"), "context": context, "event": event},
                        response_model=CriticResult,
                    )
                )
                for critic in policy.critic_roles
            ]
            critique_rounds.append(critiques)
            if all(review.verdict == CriticVerdict.PASS for review in critiques):
                break
            accumulated_nonpass.extend(
                review for review in critiques if review.verdict != CriticVerdict.PASS
            )
            if (
                any(review.verdict == CriticVerdict.REJECT for review in critiques)
                and semantic_revision_rounds >= policy.max_revision_rounds
            ):
                raise ReasoningRejected(
                    "candidate rejected by critic",
                    candidate=parsed.model_dump(mode="json"),
                    critiques=[x.model_dump(mode="json") for x in critiques],
                    critique_rounds=[
                        [item.model_dump(mode="json") for item in round_items]
                        for round_items in critique_rounds
                    ],
                    recovery_strategy=recovery_strategy,
                )
            if semantic_revision_rounds >= policy.max_revision_rounds:
                raise ReasoningRejected(
                    "candidate did not pass critics within revision budget",
                    candidate=parsed.model_dump(mode="json"),
                    critiques=[x.model_dump(mode="json") for x in critiques],
                    critique_rounds=[
                        [item.model_dump(mode="json") for item in round_items]
                        for round_items in critique_rounds
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
                    dict.fromkeys(
                        change
                        for critique in accumulated_nonpass
                        for change in critique.required_changes
                    )
                ),
                "critic_concerns": list(
                    dict.fromkeys(
                        concern
                        for critique in accumulated_nonpass
                        for concern in critique.concerns
                    )
                ),
                "rebuild_from_scratch": rebuild_from_scratch,
            }
            if not rebuild_from_scratch:
                revision_payload["candidate"] = parsed.model_dump(mode="json")
            candidate = self.backend.complete(
                role=policy.generator_role,
                system=self._generator_system(policy)
                + (
                    " FINAL RECOVERY: discard the prior draft and rebuild a smaller answer from scratch. "
                    "Treat every required change as a hard constraint. Do not repeat criticized examples, "
                    "materials, claims, or activity mechanics. Prefer no optional activity or visual over one "
                    "that conflicts with the context. Return only the new structured answer."
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
                    break
                except ReasoningRejected as exc:
                    if semantic_validation_repairs >= 1:
                        exc.critique_rounds = [
                            [item.model_dump(mode="json") for item in round_items]
                            for round_items in critique_rounds
                        ]
                        exc.recovery_strategy = recovery_strategy
                        raise
                    semantic_validation_repairs += 1
                    revision_rounds += 1
                    candidate = self.backend.complete(
                        role=policy.generator_role,
                        system=self._generator_system(policy)
                        + " Repair only the response contract while preserving every supplied critic constraint.",
                        payload={
                            **revision_payload,
                            "candidate": exc.candidate,
                            "validation_errors": exc.critiques,
                        },
                        response_model=response_model,
                    )
        if isinstance(parsed, PullThreadOutput):
            parsed = self._complete_missing_visual(
                policy=policy,
                context=context,
                event=event,
                parsed=parsed,
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
        parsed: PullThreadOutput,
    ) -> PullThreadOutput:
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        mode = str(metadata.get("response_visual_mode") or "")
        question = str(event.get("text") or "")
        needs_imagination_visual = parsed.visual is None and should_attempt_decorative_visual(question, mode)
        needs_activity_aid = _activity_needs_printable(parsed.physical_extension)
        if not needs_imagination_visual and not needs_activity_aid:
            return parsed

        child = context.get("child") if isinstance(context.get("child"), dict) else {}
        try:
            candidate = self.backend.complete(
                role=policy.generator_role,
                system=self._generator_system(policy)
                + (
                    " VISUAL COMPLETION PASS: Copy the supplied structured answer exactly. Treat imagination art "
                    "and a functional activity aid as independent outputs. When activity_aid_needed is true, add "
                    "a physical_extension.printable plan that supplies the cards, targets, pieces, or recording "
                    "surface the activity asks the parent to make; preserve the activity title, instructions, and "
                    "materials. When imagination_visual_needed is true, add one decorative_illustration visual. "
                    "The decorative scene may evoke only the broad public topic; it must not teach facts, labels, "
                    "counts, scale, sequence, anatomy, or a scientific mechanism. Use lowercase generic nouns in "
                    "subject and exclude names, relationships, private resources, school, home, health, URLs, "
                    "brands, and copyrighted characters. Both a printable and decorative visual may be present."
                ),
                payload={
                    "candidate": parsed.model_dump(mode="json"),
                    "context": {"child": {"grade": child.get("grade")}},
                    "event": {"type": event.get("type"), "text": question},
                    "visual_completion": {
                        "mode": mode,
                        "preserve_reviewed_text": True,
                        "imagination_visual_needed": needs_imagination_visual,
                        "activity_aid_needed": needs_activity_aid,
                        "accepted_decorative_kind": "decorative_illustration",
                    },
                },
                response_model=PullThreadOutput,
            )
            completed = self._validate_and_normalize_candidate(candidate, PullThreadOutput, event)
        except Exception as exc:
            logger.warning("optional visual completion failed error_type=%s", exc.__class__.__name__)
            return parsed
        if not isinstance(completed, PullThreadOutput):
            logger.warning("optional visual completion returned no usable output")
            return parsed
        visual = completed.visual if needs_imagination_visual else parsed.visual
        extension = parsed.physical_extension
        if (
            needs_activity_aid
            and extension is not None
            and completed.physical_extension is not None
            and completed.physical_extension.printable is not None
        ):
            extension = extension.model_copy(
                update={"printable": completed.physical_extension.printable}
            )
        if needs_imagination_visual and visual is None:
            logger.warning("optional visual completion returned no safe imagination visual")
        if needs_activity_aid and (extension is None or extension.printable is None):
            logger.warning("optional visual completion returned no usable activity aid")
        return parsed.model_copy(update={"visual": visual, "physical_extension": extension})

    @staticmethod
    def _validate_and_normalize_candidate(
        candidate: dict[str, Any],
        response_model: type[OutputModel],
        event: dict[str, Any],
    ) -> OutputModel:
        candidate = ReasoningEngine._without_invalid_graph_updates(candidate)
        if response_model is not PullThreadOutput:
            return ReasoningEngine._validate_candidate(candidate, response_model)
        candidate_without_visual = {**candidate, "visual": None}
        parsed = ReasoningEngine._validate_candidate(candidate_without_visual, response_model)
        if not isinstance(parsed, PullThreadOutput):  # pragma: no cover - type narrowing guard
            return parsed
        output = parsed.model_dump(mode="json")
        output["visual"] = candidate.get("visual")
        visual = normalize_response_visual(str(event.get("text") or ""), output)
        return parsed.model_copy(update={"visual": visual})

    @staticmethod
    def _without_invalid_graph_updates(candidate: dict[str, Any]) -> dict[str, Any]:
        updates = candidate.get("graph_updates")
        if not isinstance(updates, list):
            return candidate
        valid_updates: list[Any] = []
        for index, update in enumerate(updates):
            try:
                GraphMutation.model_validate(update)
            except ValidationError as exc:
                logger.warning(
                    "discarded invalid optional graph update index=%s validation_types=%s",
                    index,
                    [item["type"] for item in exc.errors()],
                )
                continue
            valid_updates.append(update)
        if len(valid_updates) == len(updates):
            return candidate
        return {**candidate, "graph_updates": valid_updates}

    @staticmethod
    def _validate_candidate(candidate: dict[str, Any], response_model: type[OutputModel]) -> OutputModel:
        try:
            return response_model.model_validate(candidate)
        except ValidationError as exc:
            raise ReasoningRejected(
                "generator returned invalid structured output",
                candidate=candidate,
                critiques=[{"validation": exc.errors()}],
            ) from exc

    def _prompt(self, name: str, fallback: str) -> str:
        path = self.prompt_dir / name
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return fallback

    def _generator_system(self, policy: ReasoningPolicy) -> str:
        base = self._prompt(
            "generator-v1.md",
            "You are one semantic component inside Curiosity Engine, not the orchestrator.",
        )
        return (
            f"{base}\nWorkflow: {policy.workflow}. Allowed semantic tools: {', '.join(policy.allowed_tools) or 'none'}."
        )

    def _critic_system(self, role: str) -> str:
        rubrics = {
            "critic_factual": "Find materially unsupported, overstated, misleading, or uncertain factual claims.",
            "critic_pedagogy": (
                "Attack spoon-feeding, premature theory, shallow theming, and weak productive struggle. "
                "A concise direct answer to a child's direct factual question is appropriate; require the show and ask "
                "to reopen curiosity instead of rejecting the answer merely because it answers."
            ),
            "critic_context": (
                "Find materially missed relevant context, harmful use of irrelevant retrieved context, or unsupported "
                "child-model assumptions. Private context may be correctly omitted when it is not relevant."
            ),
            "critic_parent_effort": "Attack anything a busy parent is unlikely to do at the stated moment.",
            "critic_epistemic": "Reject durable child-state claims that lack multiple attributable pieces of evidence.",
            "critic_visual": (
                "Find visuals that are misleading, illegible, malformed, or merely decorative when the proposed "
                "activity has an obvious child-usable printout, diagram, play piece, comparison, or recording need. "
                "A pleasant topic picture does not satisfy an activity's visual need."
            ),
        }
        return (
            self._prompt("critic-v1.md", "Return a strict verdict: pass, revise, or reject.")
            + "\n"
            + rubrics.get(role, "Adversarially find reasons this candidate should not ship.")
        )
