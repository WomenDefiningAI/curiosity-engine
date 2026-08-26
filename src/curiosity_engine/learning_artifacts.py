from __future__ import annotations

import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from .artifact_validation import render_pdf_preview, validate_rendered_file
from .artifacts import deterministic_visual_qa
from .capabilities import CapabilityRegistry
from .contracts import ActivitySpec, ChallengeSpec, LearningArtifactSpec, WorksheetSpec
from .db import connect, init_db, jdump, jload, utcnow
from .reasoning import ModelBackend

logger = logging.getLogger(__name__)


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _paragraph(canvas: Canvas, text: str, *, x: float, y: float, width: float, size: float = 12, leading: float = 15, bold: bool = False) -> float:
    font = "Helvetica-Bold" if bold else "Helvetica"
    canvas.setFont(font, size)
    for line in _wrap(text, font, size, width):
        canvas.drawString(x, y, line)
        y -= leading
    return y


def validate_learning_artifact(spec: LearningArtifactSpec) -> list[str]:
    errors: list[str] = []
    if len(spec.title) > 90:
        errors.append("title is too long for a child-facing one-page artifact")
    if spec.estimated_minutes > 30 and spec.target_grade.casefold() in {"k", "kindergarten", "1st", "first", "2nd", "second"}:
        errors.append("early-elementary artifact should fit within 30 minutes")
    if isinstance(spec, WorksheetSpec):
        kinds = {task.kind for task in spec.tasks}
        if len(kinds) < 2:
            errors.append("worksheet must use at least two task mechanics")
        if sum(len(task.instruction) for task in spec.tasks) > 760:
            errors.append("worksheet reading load is too dense")
        if not any(task.response_lines > 0 or task.kind in {"match", "sort", "sequence", "circle", "draw", "label"} for task in spec.tasks):
            errors.append("worksheet has no usable response mechanism")
    elif isinstance(spec, ActivitySpec):
        if len(spec.steps) < 3 or not spec.observation_prompts:
            errors.append("activity needs a playable loop and observation prompt")
        if not spec.cleanup:
            errors.append("activity cleanup is required")
    elif isinstance(spec, ChallengeSpec):
        if len(spec.hints) < 2:
            errors.append("challenge needs a graduated hint ladder")
        if not spec.constraints or not spec.evidence_of_completion:
            errors.append("challenge needs constraints and visible evidence of completion")
        if len(spec.steps) < 3:
            errors.append("challenge needs an actionable play sequence")
        if len(_wrap(spec.scenario, "Helvetica-Bold", 12, 478)) > 4:
            errors.append("challenge scenario does not fit its child-facing story panel")
        if sum(len(step) for step in spec.steps) > 560:
            errors.append("challenge play sequence is too dense")
        if len(_wrap(spec.goal, "Helvetica-Bold", 13, 508)) > 3:
            errors.append("challenge goal is too tall")
        if sum(len(_wrap(item, "Helvetica", 10, 508)) for item in spec.constraints) > 6:
            errors.append("challenge constraints do not fit")
        if len(_wrap(", ".join(spec.materials), "Helvetica", 10, 508)) > 2:
            errors.append("challenge materials do not fit")
        if sum(len(_wrap(step, "Helvetica", 9, 508)) for step in spec.steps[:5]) > 9:
            errors.append("challenge steps do not fit")
        if len(_wrap(spec.evidence_of_completion, "Helvetica", 10, 478)) > 2:
            errors.append("challenge evidence prompt does not fit")
        if sum(len(_wrap(hint, "Helvetica", 9, 508)) for hint in spec.hints[:3]) > 6:
            errors.append("challenge hints do not fit")
    return errors


def _page_shell(canvas: Canvas, spec: LearningArtifactSpec) -> tuple[float, float, float]:
    width, height = letter
    navy = HexColor("#17324D")
    cream = HexColor("#FFF9EC")
    coral = HexColor("#F26B5B")
    canvas.setFillColor(cream)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setFillColor(navy)
    canvas.roundRect(24, 24, width - 48, height - 48, 20, stroke=0, fill=1)
    canvas.setFillColor(white)
    canvas.roundRect(32, 32, width - 64, height - 64, 16, stroke=0, fill=1)
    canvas.setFillColor(coral)
    canvas.circle(66, height - 65, 18, stroke=0, fill=1)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 20)
    canvas.drawCentredString(66, height - 72, {"worksheet": "W", "activity": "!", "challenge": "?"}[spec.artifact_type])
    canvas.setFillColor(navy)
    canvas.setFont("Helvetica-Bold", 25)
    y = height - 58
    for line in _wrap(spec.title, "Helvetica-Bold", 25, width - 155)[:2]:
        canvas.drawString(98, y, line)
        y -= 27
    canvas.setFillColor(HexColor("#557080"))
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawRightString(width - 52, 52, f"{spec.target_grade} • about {spec.estimated_minutes} min • parent preview")
    return 52, min(y - 16, height - 112), width - 104


def _section_label(canvas: Canvas, label: str, x: float, y: float) -> float:
    canvas.setFillColor(HexColor("#2A9D8F"))
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(x, y, label.upper())
    return y - 15


def _draw_worksheet(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    navy = HexColor("#17324D")
    y = _section_label(canvas, "Your mission", x, y)
    canvas.setFillColor(navy)
    y = _paragraph(canvas, spec.directions, x=x, y=y, width=width, size=13, leading=16, bold=True) - 8
    task_height = min(92, max(62, (y - 78) / len(spec.tasks)))
    for number, task in enumerate(spec.tasks, start=1):
        bottom = y - task_height + 5
        canvas.setFillColor(HexColor("#F2F8F7") if number % 2 else HexColor("#FFF2ED"))
        canvas.roundRect(x, bottom, width, task_height - 3, 9, stroke=0, fill=1)
        canvas.setFillColor(HexColor("#F26B5B"))
        canvas.circle(x + 17, y - 17, 10, stroke=0, fill=1)
        canvas.setFillColor(white)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawCentredString(x + 17, y - 21, str(number))
        canvas.setFillColor(navy)
        text_y = _paragraph(canvas, task.instruction, x=x + 35, y=y - 12, width=width - 48, size=11, leading=13, bold=True)
        if task.choices:
            choice_text = "   ○ " + "     ○ ".join(task.choices)
            text_y = _paragraph(canvas, choice_text, x=x + 35, y=text_y - 3, width=width - 48, size=10, leading=12)
        if task.kind in {"draw", "label"}:
            canvas.setStrokeColor(HexColor("#BFD2D9"))
            canvas.roundRect(x + 35, bottom + 8, width - 52, max(20, text_y - bottom - 10), 6, stroke=1, fill=0)
        elif task.response_lines:
            canvas.setStrokeColor(HexColor("#BFD2D9"))
            line_y = text_y - 3
            for _ in range(task.response_lines):
                if line_y <= bottom + 8:
                    break
                canvas.line(x + 35, line_y, x + width - 16, line_y)
                line_y -= 14
        y = bottom - 5


def _draw_activity(canvas: Canvas, spec: ActivitySpec, x: float, y: float, width: float) -> None:
    navy = HexColor("#17324D")
    canvas.setFillColor(HexColor("#FFF2ED"))
    canvas.roundRect(x, y - 60, width, 60, 10, stroke=0, fill=1)
    canvas.setFillColor(navy)
    _paragraph(canvas, spec.mission, x=x + 14, y=y - 18, width=width - 28, size=12, leading=15, bold=True)
    y -= 75
    left_width = width * 0.31
    y_left = _section_label(canvas, "Gather", x, y)
    canvas.setFillColor(navy)
    for item in spec.materials:
        y_left = _paragraph(canvas, f"• {item}", x=x, y=y_left, width=left_width, size=10, leading=13)
    if spec.substitutions:
        y_left -= 5
        y_left = _section_label(canvas, "Swap if needed", x, y_left)
        canvas.setFillColor(navy)
        for item in spec.substitutions[:3]:
            y_left = _paragraph(canvas, f"• {item}", x=x, y=y_left, width=left_width, size=9, leading=12)
    right_x = x + left_width + 18
    right_width = width - left_width - 18
    y_right = _section_label(canvas, "Play", right_x, y)
    canvas.setFillColor(navy)
    for index, step in enumerate(spec.steps, start=1):
        y_right = _paragraph(canvas, f"{index}. {step}", x=right_x, y=y_right, width=right_width, size=10, leading=13)
        y_right -= 2
    y = min(y_left, y_right) - 8
    canvas.setFillColor(HexColor("#E8F5F2"))
    canvas.roundRect(x, y - 88, width, 88, 10, stroke=0, fill=1)
    canvas.setFillColor(navy)
    obs_y = _section_label(canvas, "Detective notes", x + 14, y - 14)
    for prompt in spec.observation_prompts[:2]:
        obs_y = _paragraph(canvas, f"□ {prompt}", x=x + 14, y=obs_y, width=width - 28, size=10, leading=13)
    canvas.setStrokeColor(HexColor("#BFD2D9"))
    canvas.line(x + 16, y - 73, x + width - 16, y - 73)
    y -= 102
    if spec.variations:
        y = _section_label(canvas, "Level up", x, y)
        canvas.setFillColor(navy)
        _paragraph(canvas, spec.variations[0], x=x, y=y, width=width, size=10, leading=13)
    canvas.setFillColor(HexColor("#557080"))
    _paragraph(canvas, f"Cleanup: {spec.cleanup}", x=x, y=74, width=width, size=8, leading=10)


def _draw_challenge(canvas: Canvas, spec: ChallengeSpec, x: float, y: float, width: float) -> None:
    navy = HexColor("#17324D")
    canvas.setFillColor(HexColor("#E8F5F2"))
    canvas.roundRect(x, y - 72, width, 72, 11, stroke=0, fill=1)
    canvas.setFillColor(navy)
    _paragraph(canvas, spec.scenario, x=x + 15, y=y - 18, width=width - 30, size=12, leading=15, bold=True)
    y -= 88
    y = _section_label(canvas, "Win when", x, y)
    canvas.setFillColor(navy)
    y = _paragraph(canvas, spec.goal, x=x, y=y, width=width, size=13, leading=16, bold=True) - 7
    y = _section_label(canvas, "Rules of the mission", x, y)
    canvas.setFillColor(navy)
    for constraint in spec.constraints:
        y = _paragraph(canvas, f"◆ {constraint}", x=x, y=y, width=width, size=10, leading=13)
    if spec.materials:
        y -= 4
        y = _section_label(canvas, "Mission kit", x, y)
        canvas.setFillColor(navy)
        y = _paragraph(canvas, ", ".join(spec.materials), x=x, y=y, width=width, size=10, leading=13)
    y -= 6
    y = _section_label(canvas, "Mission steps", x, y)
    canvas.setFillColor(navy)
    for index, step in enumerate(spec.steps[:5], start=1):
        y = _paragraph(canvas, f"{index}. {step}", x=x, y=y, width=width, size=9, leading=11)
    y -= 8
    evidence_height = max(142 if spec.evidence_rows else 85, y - 245)
    canvas.setFillColor(HexColor("#FFF2ED"))
    canvas.roundRect(x, y - evidence_height, width, evidence_height, 10, stroke=0, fill=1)
    canvas.setFillColor(navy)
    note_y = _section_label(canvas, "Show your evidence", x + 15, y - 15)
    note_y = _paragraph(
        canvas,
        spec.evidence_of_completion,
        x=x + 15,
        y=note_y,
        width=width - 30,
        size=10,
        leading=13,
    )
    canvas.setStrokeColor(HexColor("#D8C8C0"))
    if spec.evidence_rows:
        table_top = min(note_y - 7, y - 54)
        label_width = width * 0.42
        col_width = (width - label_width) / 3
        canvas.setFillColor(HexColor("#557080"))
        canvas.setFont("Helvetica-Bold", 7)
        for index, label in enumerate(("TRY 1", "TRY 2", "BEST?")):
            canvas.drawCentredString(x + label_width + col_width * (index + 0.5), table_top, label)
        row_y = table_top - 16
        canvas.setFillColor(navy)
        canvas.setFont("Helvetica-Bold", 8)
        for label in spec.evidence_rows[:4]:
            canvas.drawString(x + 15, row_y, label[:32])
            canvas.setStrokeColor(HexColor("#D8C8C0"))
            for index in range(3):
                left = x + label_width + col_width * index + 4
                canvas.rect(left, row_y - 4, col_width - 8, 12, stroke=1, fill=0)
            row_y -= 20
    else:
        line_y = y - 62
        while line_y > y - evidence_height + 16:
            canvas.line(x + 16, line_y, x + width - 16, line_y)
            line_y -= 22
    hint_y = y - evidence_height - 15
    hint_y = _section_label(canvas, "Hint ladder — uncover one at a time", x, hint_y)
    canvas.setFillColor(navy)
    for index, hint in enumerate(spec.hints[:3], start=1):
        hint_y = _paragraph(canvas, f"{index}. {hint}", x=x, y=hint_y, width=width, size=9, leading=11)


def render_learning_pdf(spec: LearningArtifactSpec, output_path: str | Path) -> Path:
    errors = validate_learning_artifact(spec)
    if errors:
        raise ValueError("invalid learning artifact: " + "; ".join(errors))
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(out), pagesize=letter, pageCompression=1)
    canvas.setTitle(spec.title)
    canvas.setAuthor("Curiosity Engine")
    canvas.setSubject(f"{spec.artifact_type}; trust tier {spec.trust_tier}; parent preview")
    x, y, width = _page_shell(canvas, spec)
    if isinstance(spec, WorksheetSpec):
        _draw_worksheet(canvas, spec, x, y, width)
    elif isinstance(spec, ActivitySpec):
        _draw_activity(canvas, spec, x, y, width)
    else:
        _draw_challenge(canvas, spec, x, y, width)
    canvas.showPage()
    canvas.save()
    return out


class LearningArtifactService:
    def __init__(
        self,
        db_path: str | Path,
        output_dir: str | Path,
        *,
        backend: ModelBackend,
        capabilities: CapabilityRegistry | None = None,
    ):
        self.db_path = str(db_path)
        self.output_dir = (Path(output_dir).resolve() / "artifacts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.chmod(0o700)
        self.backend = backend
        self.capabilities = capabilities or CapabilityRegistry()
        init_db(self.db_path)

    def create_from_event(
        self,
        *,
        event_id: str,
        artifact_type: str,
        revision: str = "",
    ) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT e.child_id,e.text,c.grade,r.output_json,r.status
                   FROM events e JOIN responses r ON r.event_id=e.id
                   JOIN children c ON c.id=e.child_id WHERE e.id=?""",
                (event_id,),
            ).fetchone()
        if not row or row["status"] != "completed" or not row["child_id"]:
            raise ValueError("completed child response not found")
        artifact_type = artifact_type if artifact_type in {"worksheet", "activity", "challenge"} else "challenge"
        response = jload(row["output_json"])
        spec = self._design(
            artifact_type,
            question=str(row["text"]),
            response=response,
            grade=str(row["grade"] or "early elementary"),
            source_event_id=event_id,
            revision=revision,
        )
        return self._render_and_store(child_id=str(row["child_id"]), spec=spec)

    def _design(
        self,
        artifact_type: str,
        *,
        question: str,
        response: dict[str, Any],
        grade: str,
        source_event_id: str,
        revision: str,
    ) -> LearningArtifactSpec:
        model_cls: type[WorksheetSpec | ActivitySpec | ChallengeSpec] = {
            "worksheet": WorksheetSpec,
            "activity": ActivitySpec,
            "challenge": ChallengeSpec,
        }[artifact_type]
        capability_id = f"artifact_{artifact_type}"
        if self.backend.name == "deterministic":
            return self._fallback_spec(
                artifact_type,
                question,
                response,
                grade,
                source_event_id,
                revision=revision,
            )
        design_payload = {
            "question": question,
            "reviewed_response": {
                key: response.get(key)
                for key in ("hook", "show", "ask", "nugget", "physical_extension", "resource_refs")
            },
            "target_grade": grade,
            "source_event_id": source_event_id,
            "parent_revision": revision,
            "artifact_brief": revision.strip() or question.strip(),
            "comparison_evidence_requirement": (
                "For a comparison, evidence_rows must name each version so the child can mark two trials and a best result."
            ),
        }
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                candidate = self.backend.complete(
                role="reasoning",
                system=(
                    "Create exactly one child-ready artifact matching the requested structured contract. It must be a "
                    "real playable/markable experience, not the prior prose copied into a PDF. Use ordinary household "
                    "materials, preserve factual accuracy, keep K–2 reading load small, and never include 3D printing. "
                    "Treat artifact_brief and parent_revision as the primary request: preserve every named comparison, "
                    "variation, object, and constraint in the actual child actions. Use a short playful title, not the "
                    "parent's sentence. A challenge must tell the child exactly what to make/do, how to run the fair "
                    "test or puzzle, and what evidence to record. "
                    "Generated imagery is not part of this contract; code owns layout and knowledge-bearing visuals.\n\n"
                    + self.capabilities.instructions_for(capability_id)
                ),
                payload={
                    **design_payload,
                    "attempt": attempt + 1,
                    "repair_instruction": (
                        "The prior structured draft was invalid or too generic. Return every required field, keep the "
                        "title under 70 characters, and explicitly include the named items in the steps."
                        if attempt
                        else ""
                    ),
                },
                response_model=model_cls,
            )
                candidate = self._bound_candidate_lists(candidate, artifact_type)
                parsed = model_cls.model_validate(candidate)
                parsed = parsed.model_copy(update={"source_event_id": source_event_id, "target_grade": grade})
                errors = validate_learning_artifact(parsed)
                errors.extend(self._alignment_errors(parsed, revision.strip() or question))
                if errors:
                    raise ValueError("designed artifact failed quality checks: " + "; ".join(errors))
                return parsed
            except (ValidationError, ValueError) as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                break
        error_type = last_error.__class__.__name__ if last_error else "unknown error"
        issue_codes = ""
        if isinstance(last_error, ValidationError):
            issue_codes = ",".join(
                f"{'.'.join(str(part) for part in issue['loc'])}:{issue['type']}"
                for issue in last_error.errors(include_input=False, include_url=False)
            )[:500]
        elif isinstance(last_error, ValueError) and str(last_error).startswith(
            "designed artifact failed quality checks:"
        ):
            issue_codes = str(last_error).removeprefix(
                "designed artifact failed quality checks:"
            ).strip()[:500]
        logger.warning(
            "artifact design fell back after %s%s",
            error_type,
            f" ({issue_codes})" if issue_codes else "",
        )
        return self._fallback_spec(
            artifact_type,
            question,
            response,
            grade,
            source_event_id,
            revision=revision,
        )

    @staticmethod
    def _bound_candidate_lists(candidate: dict[str, Any], artifact_type: str) -> dict[str, Any]:
        """Trim surplus model suggestions to reviewed contract limits before strict validation."""

        bounded = dict(candidate)
        caps = {
            "worksheet": {"tasks": 8},
            "activity": {
                "materials": 10,
                "substitutions": 6,
                "setup": 4,
                "steps": 8,
                "observation_prompts": 4,
                "variations": 3,
            },
            "challenge": {
                "constraints": 5,
                "materials": 8,
                "steps": 5,
                "hints": 3,
                "evidence_rows": 4,
            },
        }[artifact_type]
        for key, maximum in caps.items():
            value = bounded.get(key)
            if isinstance(value, list) and len(value) > maximum:
                bounded[key] = value[:maximum]
        scalar_caps = {
            "title": 90,
            "learning_objective": 260,
            "story_theme": 100,
            "accessibility_text": 500,
        }
        if artifact_type == "challenge":
            scalar_caps.update(
                {
                    "scenario": 225,
                    "goal": 150,
                    "evidence_of_completion": 130,
                    "reflection": 180,
                }
            )
            list_text_caps = {
                "constraints": 90,
                "materials": 60,
                "steps": 110,
                "hints": 90,
                "evidence_rows": 32,
            }
        elif artifact_type == "activity":
            scalar_caps.update({"mission": 240, "cleanup": 180, "safety": 220})
            list_text_caps = {
                "materials": 60,
                "substitutions": 120,
                "setup": 120,
                "steps": 140,
                "observation_prompts": 120,
                "variations": 140,
            }
        else:
            scalar_caps.update({"directions": 260, "celebration": 80})
            list_text_caps = {}
        for key, maximum in scalar_caps.items():
            if isinstance(bounded.get(key), str):
                clip = (
                    LearningArtifactService._clip_at_word
                    if key in {"title", "story_theme"}
                    else LearningArtifactService._clip_at_sentence
                )
                bounded[key] = clip(bounded[key], maximum)
        for key, maximum in list_text_caps.items():
            if isinstance(bounded.get(key), list):
                clip = (
                    LearningArtifactService._clip_at_word
                    if key in {"materials", "evidence_rows"}
                    else LearningArtifactService._clip_at_sentence
                )
                bounded[key] = [
                    clip(
                        LearningArtifactService._compact_evidence_row(str(item))
                        if key == "evidence_rows"
                        else str(item),
                        maximum,
                    )
                    for item in bounded[key]
                ]
        if artifact_type == "challenge":
            bounded["steps"] = [
                re.sub(r"^\s*(?:step\s*)?\d+[.)]\s*", "", step, flags=re.IGNORECASE)
                for step in bounded.get("steps") or []
            ]
            while len(bounded.get("steps") or []) > 3 and sum(
                len(step) for step in bounded["steps"]
            ) > 560:
                bounded["steps"].pop()
            while len(bounded.get("constraints") or []) > 1 and sum(
                len(_wrap(item, "Helvetica", 10, 508)) for item in bounded["constraints"]
            ) > 6:
                bounded["constraints"].pop()
            while len(bounded.get("materials") or []) > 1 and len(
                _wrap(", ".join(bounded["materials"]), "Helvetica", 10, 508)
            ) > 2:
                bounded["materials"].pop()
            while len(bounded.get("steps") or []) > 3 and sum(
                len(_wrap(step, "Helvetica", 9, 508)) for step in bounded["steps"]
            ) > 9:
                bounded["steps"].pop()
        if artifact_type == "worksheet":
            for task in bounded.get("tasks") or []:
                if isinstance(task, dict) and isinstance(task.get("choices"), list):
                    task["choices"] = task["choices"][:8]
        return bounded

    @staticmethod
    def _clip_at_word(text: str, maximum: int) -> str:
        compact = " ".join(text.split())
        if len(compact) <= maximum:
            return compact
        clipped = compact[: maximum + 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
        return f"{clipped}." if clipped else compact[:maximum]

    @staticmethod
    def _clip_at_sentence(text: str, maximum: int) -> str:
        compact = " ".join(text.split())
        if len(compact) <= maximum:
            return compact
        sentences = re.findall(r".*?[.!?;](?:\s+|$)", compact)
        kept = ""
        for sentence in sentences:
            candidate = f"{kept} {sentence.strip()}".strip()
            if len(candidate) > maximum:
                break
            kept = candidate
        if kept:
            return f"{kept[:-1]}." if kept.endswith(";") else kept
        return LearningArtifactService._clip_at_word(compact, maximum)

    @staticmethod
    def _compact_evidence_row(text: str) -> str:
        compact = " ".join(text.split())
        compact = re.sub(
            r"\b(?:both\s+)?(?:(?:back|outer)\s+)?wing\s+edges?\s+",
            "",
            compact,
            flags=re.IGNORECASE,
        )
        return " ".join(compact.split())

    @staticmethod
    def _alignment_errors(spec: LearningArtifactSpec, brief: str) -> list[str]:
        stop = {
            "about", "activity", "already", "challenge", "create", "different", "earlier",
            "make", "more", "mystery", "printable", "repeat", "tell", "that", "this",
            "three", "what", "with", "worksheet",
        }
        terms = {
            token
            for token in re.findall(r"[a-z]{4,}", brief.casefold())
            if token not in stop
        }
        if not terms:
            return []
        artifact_text = jdump(spec.model_dump(mode="json")).casefold()
        matches = {term for term in terms if term in artifact_text}
        required = min(2, len(terms))
        return [] if len(matches) >= required else ["artifact dropped the parent's named topic or comparison"]

    @staticmethod
    def _fallback_spec(
        artifact_type: str,
        question: str,
        response: dict[str, Any],
        grade: str,
        event_id: str,
        *,
        revision: str = "",
    ) -> LearningArtifactSpec:
        brief = re.sub(r"<@[^>]+>|@Curiosity Engine", "", revision.strip() or question.strip()).strip()
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", brief) if part.strip()]
        if len(sentences) > 1 and re.search(
            r"\b(?:printable|worksheet|challenge|revision)\b",
            sentences[0],
            re.IGNORECASE,
        ):
            brief = " ".join(sentences[1:])
        focus = re.sub(
            r"^(?:please\s+)?(?:make|create)\s+(?:me\s+)?(?:a\s+)?(?:printable\s+)?(?:mystery\s+)?(?:worksheet|activity|challenge)\s+(?:for|about|to)\s+",
            "",
            brief,
            flags=re.IGNORECASE,
        )
        focus = re.split(r"\b(?:do not|don't) repeat\b", focus, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .")
        focus = focus or question.strip()
        comparison = bool(
            re.search(r"\b(?:compar|versus|different|variation|shape|which|each)\w*\b", focus, re.IGNORECASE)
        )
        evidence_rows = LearningArtifactService._comparison_rows(focus) if comparison else []
        common = {
            "title": ("The Fair-Test Mystery" if comparison else "The Curiosity Mission"),
            "target_grade": grade,
            "learning_objective": str(response.get("nugget") or "Notice, predict, test, and explain."),
            "estimated_minutes": 15,
            "parent_effort": "low",
            "story_theme": "detective mission",
            "accessibility_text": "A one-page child mission with large headings, short directions, and visible response spaces.",
            "source_event_id": event_id,
            "source_refs": list(response.get("resource_refs") or [])[:8],
        }
        if artifact_type == "worksheet":
            return WorksheetSpec(
                **common,
                directions="Become a question detective. Circle, draw, and record what you discover.",
                tasks=[
                    {"id": "predict", "kind": "circle", "instruction": "Circle your prediction before you test.", "choices": ["Yes", "No", "Not sure"], "response_lines": 0},
                    {"id": "draw", "kind": "draw", "instruction": "Draw the most important thing you notice.", "response_lines": 0},
                    {"id": "explain", "kind": "short_response", "instruction": "What changed your mind?", "response_lines": 2},
                ],
            )
        if artifact_type == "activity":
            return ActivitySpec(
                **common,
                mission="Test one idea, change one thing, and see what happens.",
                materials=["paper", "pencil or crayons", "one safe nearby object"],
                substitutions=["Say or act out the prediction if writing feels slow."],
                setup=["Choose one thing to notice."],
                steps=["Make a prediction.", "Test or observe one small thing.", "Change one detail and try again.", "Draw or tell what changed."],
                observation_prompts=["What stayed the same?", "What surprised you?"],
                variations=["Turn it into a two-player prediction game."],
                cleanup="Put the object back and save the paper for your next clue.",
            )
        return ChallengeSpec(
            **common,
            scenario=f"Flight Lab Mystery: {focus[:160].rstrip(' .')}. Can you solve it with evidence?",
            goal=(
                "Make or choose the named versions, test each one the same way, and compare what happens."
                if comparison
                else "Turn the mission question into one safe test and collect a result you can show."
            ),
            constraints=[
                "Use the same starting place and test method every time.",
                "Change only the named feature; keep the other parts the same.",
                "Give every version at least two tries.",
            ] if comparison else ["Use only safe things already nearby.", "Change only one thing at a time."],
            materials=["paper", "pencil or crayons"],
            steps=(
                [
                    "Circle or draw the versions named in the mission question.",
                    "Predict which version will work best and say why.",
                    "Test every version from the same starting place.",
                    "Mark or measure each result, then choose the strongest evidence.",
                ]
                if comparison
                else [
                    "Draw what you predict will happen.",
                    "Choose one safe thing to observe or change.",
                    "Run the test and mark what happened.",
                ]
            ),
            hints=["Start by drawing what you think will happen.", "Keep the test fair so the results can be compared.", "Look for a result that could change your first guess."],
            evidence_of_completion=(
                "Mark both tries for every version, then put a star by the strongest result."
                if comparison
                else "Show a drawing, result, or explanation that connects your test to the question."
            ),
            evidence_rows=evidence_rows or (["Version A", "Version B", "Version C"] if comparison else []),
            reflection="What new question appeared after your test?",
        )

    @staticmethod
    def _comparison_rows(focus: str) -> list[str]:
        match = re.search(
            r"\b(?:compare|comparing|test|tests|testing|keep|for)\s+(?:the\s+)?(.{3,150}?)\s+wing(?:\s+shapes?)?s?\b",
            focus,
            re.IGNORECASE,
        )
        if not match:
            return []
        raw = re.sub(r"\s+and\s+", ",", match.group(1), flags=re.IGNORECASE)
        parts = [re.sub(r"^(?:the\s+)", "", part.strip(" ,.-"), flags=re.IGNORECASE) for part in raw.split(",")]
        parts = [part for part in parts if part and len(part.split()) <= 4]
        if not 2 <= len(parts) <= 4:
            return []
        return [f"{part.capitalize()} wing" for part in parts]

    def _render_and_store(self, *, child_id: str, spec: LearningArtifactSpec) -> dict[str, Any]:
        artifact_id = f"art_{uuid4().hex[:16]}"
        experience_id = f"exp_{uuid4().hex[:16]}"
        slug = re.sub(r"[^a-z0-9]+", "-", spec.title.casefold()).strip("-")[:50] or spec.artifact_type
        pdf_path = self.output_dir / f"{artifact_id}-{slug}.pdf"
        preview_path = self.output_dir / f"{artifact_id}-{slug}.png"
        render_learning_pdf(spec, pdf_path)
        pdf_path.chmod(0o600)
        errors = validate_rendered_file(pdf_path)
        if errors:
            raise RuntimeError("rendered learning artifact failed validation: " + "; ".join(errors))
        render_pdf_preview(pdf_path, preview_path)
        preview_path.chmod(0o600)
        visual = deterministic_visual_qa(preview_path)
        if visual.verdict != "pass":
            raise RuntimeError("learning artifact visual QA failed: " + "; ".join(visual.reasons))
        digest = sha256(pdf_path.read_bytes()).hexdigest()
        validation = {
            "structural": "pass",
            "pedagogy": "pass",
            "mechanics": spec.artifact_type,
            "deterministic_visual_qa": visual.model_dump(mode="json"),
            "page_limit": 1,
        }
        now = utcnow()
        payload = spec.model_dump(mode="json")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """INSERT INTO experiences(id,child_id,experience_type,title,spec_json,status,created_at,source_event_id)
                   VALUES(?,?,?,?,?,'generated',?,?)""",
                (experience_id, child_id, spec.artifact_type, spec.title, jdump(payload), now, spec.source_event_id),
            )
            conn.execute(
                """INSERT INTO artifacts(id,experience_id,child_id,artifact_type,path,spec_json,created_at,validated_at,
                   sha256,validation_json,approval_status) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    artifact_id,
                    experience_id,
                    child_id,
                    spec.artifact_type,
                    str(pdf_path),
                    jdump(payload),
                    now,
                    now,
                    digest,
                    jdump(validation),
                    "unreviewed",
                ),
            )
        return {
            "artifact_id": artifact_id,
            "experience_id": experience_id,
            "artifact_type": spec.artifact_type,
            "title": spec.title,
            "pdf_path": str(pdf_path),
            "preview_path": str(preview_path),
            "sha256": digest,
            "validation": validation,
            "approval_status": "unreviewed",
        }
