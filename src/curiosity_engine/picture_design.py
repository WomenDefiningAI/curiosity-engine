"""A compact visual task contract, independent of legacy lesson-plan fields."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from .contracts import (
    ActivitySpec,
    ChallengeSpec,
    PictureAnnotation,
    PictureCell,
    PicturePage,
    PictureSymbol,
    ReferencePanel,
    ReferenceSpec,
    StrictModel,
    WorksheetSpec,
    WorksheetTask,
)


class SubjectDraft(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    description: str = Field(min_length=1, max_length=450)
    labels: list[str] = Field(default_factory=list, max_length=4,
                             description="Short names of visible parts needing pointer labels, e.g. root or wing. "
                                         "No step numbers, instructions, or captions; use cell.label/action for those.")


class PictureDraft(StrictModel):
    title: str = Field(default="", max_length=65)
    directions: str = Field(min_length=1, max_length=320)
    display: Literal["paper", "screen"] = "paper"
    layout_locked: bool = False
    minutes: int = Field(default=5, ge=1, le=30)
    columns: int = Field(default=2, ge=1, le=3)
    show_borders: bool = False
    preserve_artwork: bool = False
    enlarge_pictures: bool = False
    subjects: list[SubjectDraft] = Field(default_factory=list, max_length=8)
    cells: list[PictureCell] = Field(min_length=2, max_length=6)


class PictureObstacle(StrictModel):
    kind: Literal["picture", "action", "content", "revision", "child_writing", "cosmetic"]
    detail: str = Field(min_length=1, max_length=650)


class PictureReview(StrictModel):
    observations: list[str] = Field(default_factory=list, max_length=8)
    verdict: Literal["ready", "repair"]
    # A repair needs a specific obstacle, not a preference for another layout.
    obstacles: list[PictureObstacle] = Field(default_factory=list, max_length=4)
    optional_improvements: list[str] = Field(default_factory=list, max_length=3)


DESIGN_INSTRUCTIONS = """
Design ONE usable page for this parent's request. Return only the compact page contract.
Think through the child's FIRST ACTION before choosing a layout. The delivered page must supply
the actual pictures, items or workspace that action needs. Use only the available supplies.
When source_photos are supplied, use their actual learning concepts, tasks and relationships.
The caption may only say 'schoolwork': it is not the topic. The parent's newest request selects
which photographed concepts to extend. Do not substitute a generic school-day activity or infer
mastery from completed answers. A challenge should extend the pictured idea through a new action.
For a hands-on challenge, include enough setup and action to use it, with pictures of the equipment
or steps and blank places for predictions/results. Preserve the reviewed activity when supplied.
Do not fill in an observation the child is meant to discover or substitute decorative pictures for doing.
Do not invent required properties of unspecified supplies. For example, if paint colors are not named,
use neutral empty paint containers or blank swatch spaces that work with whatever colors the family has.
Result prompts must accept every possible outcome, including no change or no effect; do not ask only
for the objects/results that fit an expected positive outcome.
Printed pictures cannot move unless someone cuts them out. With no scissors, make grouping playable
by pointing, tracing/circling groups, or placing the supplied toys beside the pictures. Never replace
the requested pictured items with new counters the parent must find. Do not print a solved grouping.
For a story, let the pictures carry distinct actions with consistent characters and objects.
Define a subject for each COMPLETE scene, and reference those scene IDs in cells. Do not define
separate character/location symbols and then refer to undefined scene IDs. Do not invent a pair
or matching copies of a single story object: keep only the requested object through each scene.
Treat a singular story prop as ONE total object in each scene where it appears. An additional
matching item already worn/carried is still an extra copy. Spell out total count and absence in each scene.
Searching is not finding: show only what is present at each moment. A missing object must not
already be visible at a search location when it is discovered somewhere else later. Avoid tricky
midair transitions; clear before/after poses are easier to read. Shuffle story cells without printed
sequence cues. Show the completed final action: an object put on is visibly worn, not merely held.
Never add first/next/last hints. When the parent requests cards to arrange, use show_borders=true and tell
the grown-up to cut the cards apart; the child arranges them. If cutting is explicitly unavailable,
give a pointing alternative instead. Do not silently replace requested movable cards with a worksheet.
Those hidden-order rules apply to story puzzles only. For instructional diagrams, put the steps
IN ORDER with short numbered labels. Show the starting state, the movement and the resulting state.
Each after-state must physically follow from the preceding action using the same object. An assembly
or fold is not complete until the final orientation, support and attachment are visible.

Subjects describe the actual artwork, not vectors or a lesson plan. Use ONE reusable subject for
identical objects and item.count for exact quantities. An individual counted object must contain
one object only. Name distinguishing details. Give recurring characters the same appearance in
every scene. Do not put family names, private context, text, numbers, arrows or answer boxes into
art descriptions. For labeled-part diagrams, list short part names on the subject; code places
pointer labels on actual art. Step numbers, instructions and scene captions belong in cell.label
or cell.action, NEVER subject.labels. Instructions such as 'fold and press' have no pointer target
and duplicate the cell caption while shrinking useful art. Leave labels=[] for instructional steps.
Code generates all new subjects together in a sheet for coherent art. No need to describe the page
frame or duplicate prompts. Keep each description short and concrete.
The image generator receives ONLY the new descriptions, not the earlier pictures. For a replacement
that must match an earlier object, spell out the visible attributes from the supplied page; a reference
such as 'the same colors as subject X' cannot work by itself. Unknown real-world properties stay unknown:
'unnamed visible colors' still invents paint colors. Show neutral empty containers and blank swatch areas
when the family's actual colors are not specified; do not offer fixed printed color guesses.

Keep directions short enough for one adult read-aloud. A nonreader can point, compare, draw or play.
For phone-only use set display=screen, omit blank work areas and printing/cutting directions. Use
two or three large choices side by side (columns=2 or 3) and one short spoken instruction so all
choices are visible together. Avoid duplicate captions and annotation labels on simple movement poses.
Cell labels are brief headings, preferably under 45 characters; actions are short sentences,
preferably under 90 characters. The larger schema limits allow an essential clarification.
Set layout_locked=true only when the parent explicitly specifies positions/columns or says to keep
the layout. Otherwise code may rearrange one labeled diagram and its drawing box to make the diagram large.
For repeated trials, show shared prediction/comparison pictures ONCE, large enough to point to.
Put blank result spaces separately. Do not squeeze copies of a labeled diagram into every trial row.
Use separate subjects for distinct comparison choices so code can space them clearly.
When the parent asks for one short thing to say, supply an actual child-ready explanation of the
idea shown, not just an instruction to look, point or draw.
Do not add recording, a quiz, another game, headings on every picture, or invented supplies.
Use two columns for three/four large pictures. response_space=draw prints a RECTANGULAR blank box
to draw/paint in; refer to it as a box or space, not a circle or another shape the renderer does not supply.
For a common set of objects to share, put the whole unsolved set in one cell with column_span=2
so the actual objects are large. Put two blank spaces below for the supplied toys or optional drawing.
Give a usable pointing alternative to tracing/drawing. Do not require extra supplies to use the page.
mark adds a checkbox; none means no answer area. show_borders=false removes picture frames;
response spaces have their own outlines. Titles/captions are optional adult support, not child reading.

On a layout/text revision set preserve_artwork=true and reuse subject IDs; code will keep the exact
old images. Omit unchanged subjects from subjects when preserving artwork; the old definitions remain
available. To replace one subject, give it a NEW ID and update only its cell references; preserve other IDs.
Set enlarge_pictures=true only when the PARENT requests bigger pictures; code increases the printed scale.
Remove unnecessary cell text and frames to gain space. Only change art descriptions if the parent
requests a content change or review identifies a specific incorrect picture. Preserve the activity.
For repair, change ONLY the obstacles in feedback, preserving valid subjects and the requested job.
The prior draft and delivered page are supplied when available. Do not restart the lesson.
"""

REVIEW_INSTRUCTIONS = """
Inspect this actual delivered-page candidate for the parent's request. Decide whether it can be
used now. Return ready unless you identify a specific obstacle requiring parent repair. Separate
optional improvements from obstacles; another preferred layout or style is not a blocker.
FIRST record concrete observations from the pixels: for each scene, the action and the total count
of each recurring prop, including objects already worn or carried. Then decide the verdict.
Judge the actual pictures, counts, actions, factual relationships, label targets and blank spaces.
When source_photos are present, check that the task extends their visible concepts and relationships,
not merely the caption or a generic school-day theme. Do not infer mastery from photographed marks.
First check the parent's supplies and unspecified properties: depicted colors, sizes or equipment may
not silently become required materials. Blank input/result spaces can accommodate unknown values.
Trace recurring story objects through each scene: do not accept an object shown in the wrong place
or extra copies of a supposedly single object. A matching item already worn is an extra copy too;
if the parent's story has a singular prop, two copies IN ONE SCENE require repair even if the art looks consistent.
The same object shown once in each separate step, comparison or story scene is valid: repeated
illustrations describe one real object and do not imply that the family needs multiple copies.
Check that the child can physically do the described
action with the page and supplied materials. Printed objects cannot be moved without separation;
pointing, drawing groups or placing supplied toys is a valid alternative. No new counters needed.
A nonreader may hear one short instruction from an adult. Picture borders are not answer boxes.
Pictures on separate parts of ONE PAGE are visible choices together; do not require a shared scene.
Do not invent a quiz, experiment, evidence phase, adult setup task or risk absent from the request.
Flag concrete unsafe child actions or material factual errors. Ordinary supervised play and an
adult preparing paper cards are fine. No writing means no required child writing, not no adult text.
Classify every obstacle. Use child_writing for objections about a 'no writing' request: code checks
whether the page actually asks the child to write. A printed title, short adult instruction, badge
or footer is not child writing. Only an explicit request for no printed text/images only can make
removing such text a revision requirement. Use cosmetic for unrequested style preferences.
For a revision compare the old page (image 2) and measured sizes. Art preservation is verified by
code. A 20% or greater picture-size increase is visible; tiny changes are not an enlargement.
Return concise, actionable obstacles. Do not call a usable page incomplete for cosmetic preferences.
"""


def compile_draft(draft: PictureDraft, *, kind: str, grade: str, event_id: str, previous=None):
    old = {s.id: s for s in previous.picture_page.symbols} if previous and previous.picture_page else {}
    symbols = list(old.values()) if draft.preserve_artwork else []
    for subject in draft.subjects:
        if draft.preserve_artwork and subject.id in old:
            continue
        else:
            symbols.append(PictureSymbol(
                id=subject.id, description=subject.description,
                annotations=[PictureAnnotation(text=text, position=(10, 90), target=(50, 50))
                             for text in subject.labels],
            ))
    used = {item.symbol_id for cell in draft.cells for item in cell.items}
    symbols = [symbol for symbol in symbols if symbol.id in used]
    scale = previous.picture_page.picture_scale if previous and previous.picture_page else .8
    if draft.enlarge_pictures and previous and previous.picture_page:
        scale = min(1, scale*1.25)
    columns = draft.columns
    if (not draft.layout_locked and draft.display == "paper" and columns > 2
            and 3 <= len(draft.cells) <= 4 and all(cell.items for cell in draft.cells)):
        columns = 2
    # One labeled diagram beside an empty drawing box leaves the main image
    # width-constrained while most of the page is blank. Stack those two regions
    # so the diagram has useful detail and the child still has drawing space.
    if (not draft.layout_locked and draft.display == "paper" and len(draft.cells) == 2
            and sum(bool(cell.items) for cell in draft.cells) == 1
            and any(not cell.items and cell.response_space == "draw" for cell in draft.cells)
            and any(symbol.annotations for symbol in symbols)):
        columns = 1
    page = PicturePage(directions=draft.directions, display=draft.display, columns=columns, symbols=symbols,
                       cells=draft.cells, show_borders=draft.show_borders, picture_scale=scale,
                       layout_locked=draft.layout_locked)
    common = dict(title=draft.title or "Look and Play", target_grade=grade, estimated_minutes=max(3, draft.minutes),
                  learning_objective=draft.directions[:300], accessibility_text=draft.directions,
                  source_event_id=event_id, picture_page=page)
    # Compatibility fields are code-owned and never sent to the visual designer.
    if kind == "worksheet":
        return WorksheetSpec(**common, archetype="play_pieces_targets", directions=draft.directions[:300],
                             tasks=[WorksheetTask(id="look", kind="draw", instruction=draft.directions[:220])])
    if kind == "reference":
        return ReferenceSpec(**common, layout="observation", directions=draft.directions[:220],
                             panels=[ReferencePanel(heading="Look", facts=[draft.directions]) for _ in range(2)])
    if kind == "activity":
        return ActivitySpec(**common, mission=draft.directions[:300], materials=["this page"],
                            steps=[draft.directions]*3, observation_prompts=["What do you notice?"],
                            cleanup="Save the page if you want to play again.")
    return ChallengeSpec(**common, scenario=draft.directions[:240], goal=draft.directions[:240],
                         constraints=["Use the available materials."], steps=[draft.directions],
                         hints=["Look at the pictures.", "Try another way."],
                         evidence_of_completion="The child tries the activity.", reflection="What did you notice?")
