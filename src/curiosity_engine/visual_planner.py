from __future__ import annotations

from .contracts import (
    VisualIntent,
    VisualPlan,
    VisualPlanUnit,
    WorksheetArchetype,
    WorksheetTask,
)

ARCHETYPE_TASK_COMPATIBILITY: dict[WorksheetArchetype, frozenset[str]] = {
    "look_notice_wonder": frozenset({"draw", "short_response", "circle"}),
    "compare_circle_explain": frozenset({"circle", "draw", "short_response", "record"}),
    "match_two_columns": frozenset({"match"}),
    "cut_sort_categories": frozenset({"sort"}),
    "sequence_story_cards": frozenset({"sequence"}),
    "label_parts": frozenset({"label", "draw"}),
    "observation_checklist": frozenset({"circle", "record", "short_response"}),
    "predict_test_record": frozenset({"predict", "record", "short_response"}),
    "measure_tally_graph": frozenset({"measure", "record", "circle"}),
    "build_test_improve": frozenset({"draw", "record", "short_response"}),
    "mystery_clue_case": frozenset({"circle", "short_response", "record"}),
    "mini_field_guide": frozenset({"label", "draw", "circle", "short_response"}),
    "play_pieces_targets": frozenset(
        {"match", "sort", "sequence", "circle", "label", "draw", "record", "short_response"}
    ),
    "legacy_mixed": frozenset(
        {"match", "sort", "sequence", "circle", "label", "draw", "predict", "measure", "record", "short_response"}
    ),
}


def build_visual_plan(
    intent: VisualIntent | None,
    *,
    parent_only: bool = False,
    printable_type: str | None = None,
) -> VisualPlan:
    """Compile a reviewed semantic intent into at most two independent release units."""

    if parent_only or (intent is None and printable_type is None):
        return VisualPlan(role="none", rationale="This turn does not benefit from a child-facing visual.")
    units: list[VisualPlanUnit] = []
    role = "do" if printable_type else "wonder"
    printable_embeds_photo = bool(
        intent is not None and intent.kind == "researched_photo" and printable_type == "worksheet"
    )
    if intent is not None and not printable_embeds_photo:
        if intent.kind == "researched_photo":
            role = "observe" if intent.purpose in {"notice", "observe"} else "wonder"
            units.append(
                VisualPlanUnit(
                    kind="learning_photo" if role == "observe" else "wonder_photo",
                    purpose="observe" if role == "observe" else "spark",
                    visual_intent=intent,
                )
            )
        elif intent.kind == "decorative_illustration":
            role = "wonder"
            units.append(VisualPlanUnit(kind="story_art", purpose="spark", visual_intent=intent))
        else:
            role = "compare" if intent.purpose == "compare" else "explain"
            units.append(
                VisualPlanUnit(
                    kind="learning_graphic",
                    purpose="compare" if role == "compare" else "explain",
                    visual_intent=intent,
                )
            )
    if printable_type:
        units.append(
            VisualPlanUnit(
                kind="printable",
                purpose="do",
                artifact_type=printable_type,  # type: ignore[arg-type]
            )
        )
    return VisualPlan(
        role=role,  # type: ignore[arg-type]
        rationale=(
            "Embed the researched observation image in the worksheet instead of delivering it twice."
            if printable_embeds_photo
            else "Use one primary learning visual and keep any distinct printable independently deliverable."
        ),
        units=units,
    )


def normalize_archetype(archetype: WorksheetArchetype, tasks: list[WorksheetTask]) -> WorksheetArchetype:
    """Keep a requested archetype when its task mechanics can actually render it."""

    kinds = {task.kind for task in tasks}
    compatible = ARCHETYPE_TASK_COMPATIBILITY[archetype]
    if archetype != "legacy_mixed" and kinds & compatible:
        return archetype
    priority: tuple[tuple[str, WorksheetArchetype], ...] = (
        ("match", "match_two_columns"),
        ("sort", "cut_sort_categories"),
        ("sequence", "sequence_story_cards"),
        ("measure", "measure_tally_graph"),
        ("predict", "predict_test_record"),
        ("label", "label_parts"),
        ("draw", "look_notice_wonder"),
        ("record", "observation_checklist"),
    )
    for task_kind, candidate in priority:
        if task_kind in kinds:
            return candidate
    return "legacy_mixed"


def validate_archetype(archetype: WorksheetArchetype, tasks: list[WorksheetTask]) -> list[str]:
    kinds = {task.kind for task in tasks}
    if archetype == "legacy_mixed":
        return []
    if not kinds & ARCHETYPE_TASK_COMPATIBILITY[archetype]:
        return [f"{archetype} has no compatible task mechanic"]
    if archetype == "match_two_columns":
        matching = [task for task in tasks if task.kind == "match"]
        if len(matching) < 2 or any(len(task.choices) < 2 for task in matching[:2]):
            return ["two-column matching needs separate left and right item sets"]
    if archetype == "cut_sort_categories":
        sorting = [task for task in tasks if task.kind == "sort"]
        if not sorting or len(sorting[0].choices) < 3:
            return ["cut-and-sort needs at least three child-usable cards"]
    if archetype == "sequence_story_cards":
        sequencing = [task for task in tasks if task.kind == "sequence"]
        if not sequencing or not 3 <= len(sequencing[0].choices) <= 6:
            return ["sequence cards need three to six movable steps"]
    if archetype == "play_pieces_targets":
        pieces = [choice for task in tasks for choice in task.choices]
        if len(pieces) < 3:
            return ["play-pieces pages need at least three child-usable targets or cards"]
    return []
