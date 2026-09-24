from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from .contracts import ResearchedImageAsset, WorksheetSpec, WorksheetTask

INK = HexColor("#17324D")
TEAL = HexColor("#2A9D8F")
CORAL = HexColor("#F26B5B")
GOLD = HexColor("#F4B942")
PALE_TEAL = HexColor("#E8F5F2")
PALE_CORAL = HexColor("#FFF2ED")
PALE_GOLD = HexColor("#FFF9EC")
LINE = HexColor("#BFD2D9")


def _wrap(text: str, font: str, size: float, width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _text(
    canvas: Canvas,
    value: str,
    *,
    x: float,
    y: float,
    width: float,
    size: float = 10,
    leading: float = 13,
    bold: bool = False,
    max_lines: int | None = None,
) -> float:
    font = "Helvetica-Bold" if bold else "Helvetica"
    canvas.setFillColor(INK)
    canvas.setFont(font, size)
    lines = _wrap(value, font, size, width)
    if max_lines is not None:
        lines = lines[:max_lines]
    for line in lines:
        canvas.drawString(x, y, line)
        y -= leading
    return y


def draw_theme_border(canvas: Canvas, spec: WorksheetSpec) -> None:
    """Draw small margin-only motifs; learning and writing space remain untouched."""

    if spec.print_mode == "ink_saver":
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(1)
        canvas.roundRect(37, 37, 538, 718, 14, stroke=1, fill=0)
        return
    motifs = {
        "space": ("star", CORAL, GOLD),
        "nature": ("leaf", TEAL, GOLD),
        "ocean": ("wave", TEAL, HexColor("#70B7D0")),
        "workshop": ("gear", CORAL, TEAL),
        "mystery": ("clue", HexColor("#75549A"), GOLD),
        "storybook": ("spark", CORAL, TEAL),
    }
    motif, first, second = motifs[spec.theme]
    canvas.saveState()
    positions = (
        (44, 92),
        (568, 112),
        (46, 205),
        (566, 235),
        (46, 355),
        (566, 390),
        (46, 515),
        (566, 545),
        (48, 690),
        (564, 700),
    )
    for index, (px, py) in enumerate(positions):
        canvas.setFillColor(first if index % 2 == 0 else second)
        if motif in {"star", "spark"}:
            canvas.circle(px, py, 4 if motif == "spark" else 5, stroke=0, fill=1)
            canvas.setLineWidth(2)
            canvas.setStrokeColor(first if index % 2 == 0 else second)
            canvas.line(px - 8, py, px + 8, py)
            canvas.line(px, py - 8, px, py + 8)
        elif motif == "leaf":
            canvas.ellipse(px - 7, py - 4, px + 7, py + 4, stroke=0, fill=1)
            canvas.setStrokeColor(INK)
            canvas.line(px - 5, py - 6, px + 5, py + 6)
        elif motif == "wave":
            canvas.setStrokeColor(first if index % 2 == 0 else second)
            canvas.setLineWidth(3)
            canvas.arc(px - 10, py - 5, px + 2, py + 7, 0, 180)
            canvas.arc(px, py - 5, px + 12, py + 7, 180, 180)
        elif motif == "gear":
            canvas.circle(px, py, 7, stroke=0, fill=1)
            canvas.setFillColor(white)
            canvas.circle(px, py, 3, stroke=0, fill=1)
        else:
            canvas.setFont("Helvetica-Bold", 12)
            canvas.drawCentredString(px, py - 4, "?")
    canvas.restoreState()


def _card(canvas: Canvas, x: float, y: float, width: float, height: float, *, fill=PALE_TEAL, dashed=False) -> None:
    canvas.saveState()
    canvas.setFillColor(fill)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(1.4)
    if dashed:
        canvas.setDash(5, 4)
    canvas.roundRect(x, y, width, height, 9, stroke=1, fill=1)
    canvas.restoreState()


def _lines(canvas: Canvas, x: float, y: float, width: float, count: int, *, gap: float = 16) -> None:
    canvas.setStrokeColor(LINE)
    for index in range(count):
        canvas.line(x, y - index * gap, x + width, y - index * gap)


def _checkbox(canvas: Canvas, x: float, y: float, label: str, width: float) -> None:
    canvas.setStrokeColor(INK)
    canvas.rect(x, y - 9, 11, 11, stroke=1, fill=0)
    _text(canvas, label, x=x + 18, y=y, width=width - 18, size=9, leading=11, max_lines=2)


def _task_prompt(canvas: Canvas, task: WorksheetTask, x: float, y: float, width: float, *, number: int) -> float:
    canvas.setFillColor(CORAL)
    canvas.circle(x + 9, y - 4, 9, stroke=0, fill=1)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawCentredString(x + 9, y - 7, str(number))
    return _text(canvas, task.instruction, x=x + 25, y=y, width=width - 25, size=10, leading=12, bold=True)


def _verified_image(asset: ResearchedImageAsset | None) -> Path | None:
    if asset is None:
        return None
    path = Path(asset.local_path).resolve()
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != asset.sha256:
        raise ValueError("researched worksheet image changed after review")
    return path


def _draw_photo(canvas: Canvas, asset: ResearchedImageAsset | None, x: float, y: float, width: float, height: float) -> None:
    path = _verified_image(asset)
    _card(canvas, x, y, width, height, fill=HexColor("#F4F7F8"))
    if path is None:
        canvas.setFillColor(PALE_TEAL)
        canvas.circle(x + width / 2, y + height / 2 + 7, min(width, height) * 0.18, stroke=0, fill=1)
        canvas.setFillColor(TEAL)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawCentredString(x + width / 2, y + height / 2, "LOOK AT THE REAL THING")
        return
    reader = ImageReader(str(path))
    iw, ih = reader.getSize()
    scale = min((width - 8) / iw, (height - 8) / ih)
    dw, dh = iw * scale, ih * scale
    canvas.drawImage(reader, x + (width - dw) / 2, y + (height - dh) / 2, dw, dh, preserveAspectRatio=True)


def _render_look_notice(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    _draw_photo(canvas, spec.researched_image, x, y - 205, width * 0.57, 190)
    right_x = x + width * 0.61
    _card(canvas, right_x, y - 95, width * 0.39, 80, fill=PALE_GOLD)
    _text(canvas, "I NOTICE", x=right_x + 12, y=y - 36, width=width * 0.34, size=10, bold=True)
    _lines(canvas, right_x + 12, y - 57, width * 0.32, 2)
    _card(canvas, right_x, y - 205, width * 0.39, 95, fill=PALE_CORAL)
    _text(canvas, "I WONDER", x=right_x + 12, y=y - 132, width=width * 0.34, size=10, bold=True)
    _lines(canvas, right_x + 12, y - 154, width * 0.32, 3)
    draw_task = next((task for task in spec.tasks if task.kind in {"draw", "label"}), spec.tasks[0])
    panel_bottom = y - 500
    _card(canvas, x, panel_bottom, width, 270, fill=white)
    _text(canvas, "DRAW ONE CLUE", x=x + 14, y=panel_bottom + 244, width=width - 28, size=10, bold=True)
    _text(canvas, draw_task.instruction, x=x + 14, y=panel_bottom + 224, width=width - 28, size=9, max_lines=2)


def _render_compare(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    labels = spec.subject_labels[:3]
    if len(labels) < 2:
        labels = [choice for task in spec.tasks for choice in task.choices][:2] or ["EXAMPLE A", "EXAMPLE B"]
    columns = max(2, min(3, len(labels)))
    gap = 10
    cell = (width - gap * (columns - 1)) / columns
    for index, label in enumerate(labels[:columns]):
        left = x + index * (cell + gap)
        _card(canvas, left, y - 190, cell, 175, fill=PALE_TEAL if index % 2 == 0 else PALE_CORAL)
        canvas.setFillColor(TEAL if index % 2 == 0 else CORAL)
        canvas.circle(left + cell / 2, y - 78, min(28, cell * 0.18), stroke=0, fill=1)
        canvas.setFillColor(white)
        canvas.setFont("Helvetica-Bold", 17)
        canvas.drawCentredString(left + cell / 2, y - 84, chr(ord("A") + index))
        _text(canvas, label, x=left + 10, y=y - 126, width=cell - 20, size=10, bold=True, max_lines=2)
        _text(canvas, "Draw or circle one visible clue", x=left + 10, y=y - 163, width=cell - 20, size=8)
    lower = y - 215
    for index, task in enumerate(spec.tasks[:3], 1):
        lower = _task_prompt(canvas, task, x, lower, width, number=index)
        if task.kind in {"draw", "short_response", "record"}:
            _lines(canvas, x + 25, lower - 3, width - 25, max(1, min(task.response_lines or 1, 3)))
            lower -= 18 * max(1, min(task.response_lines or 1, 3))
        lower -= 8


def _shuffled_matching_choices(spec: WorksheetSpec, choices: list[str]) -> list[str]:
    """Return a stable non-positional order; original indexes remain the answer mapping."""

    if len(choices) < 2:
        return list(choices)
    seed = "|".join([spec.title, *(task.id for task in spec.tasks)])
    ordered = [
        choice
        for _, choice in sorted(
            enumerate(choices),
            key=lambda item: sha256(f"{seed}:{item[0]}".encode()).digest(),
        )
    ]
    if ordered == choices:
        ordered = [*ordered[1:], ordered[0]]
    return ordered


def _render_matching(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    matching = [task for task in spec.tasks if task.kind == "match"]
    left_items = matching[0].choices if matching else ["Item 1", "Item 2", "Item 3"]
    right_items = _shuffled_matching_choices(
        spec,
        matching[1].choices if len(matching) > 1 else ["Job A", "Job B", "Job C"],
    )
    _text(canvas, "Draw a line across the middle.", x=x, y=y, width=width, size=11, bold=True)
    top = y - 38
    count = min(4, max(len(left_items), len(right_items)))
    row_h = min(86, 330 / max(1, count))
    for index in range(count):
        row_y = top - index * row_h
        _card(canvas, x, row_y - row_h + 8, width * 0.39, row_h - 12, fill=PALE_TEAL)
        _card(canvas, x + width * 0.61, row_y - row_h + 8, width * 0.39, row_h - 12, fill=PALE_CORAL)
        if index < len(left_items):
            _text(canvas, left_items[index], x=x + 10, y=row_y - 25, width=width * 0.35, size=10, bold=True, max_lines=3)
        if index < len(right_items):
            _text(
                canvas,
                right_items[index],
                x=x + width * 0.61 + 10,
                y=row_y - 25,
                width=width * 0.35,
                size=10,
                bold=True,
                max_lines=3,
            )
        canvas.setFillColor(GOLD)
        canvas.circle(x + width * 0.45, row_y - row_h / 2, 4, stroke=0, fill=1)
        canvas.circle(x + width * 0.55, row_y - row_h / 2, 4, stroke=0, fill=1)
    _text(canvas, "CHECK: Explain one match.", x=x, y=top - count * row_h - 12, width=width, size=10, bold=True)
    _lines(canvas, x, top - count * row_h - 35, width, 2)


def _render_cut_sort(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    sorting = next((task for task in spec.tasks if task.kind == "sort"), spec.tasks[0])
    categories = spec.category_labels[:2]
    if len(categories) < 2:
        categories = ["CATEGORY A", "CATEGORY B"]
    half = (width - 12) / 2
    for index, label in enumerate(categories):
        left = x + index * (half + 12)
        _card(canvas, left, y - 145, half, 128, fill=PALE_TEAL if index == 0 else PALE_CORAL)
        _text(canvas, label, x=left + 10, y=y - 42, width=half - 20, size=11, bold=True, max_lines=3)
        _text(canvas, "Place cards here", x=left + 10, y=y - 118, width=half - 20, size=8)
    items = sorting.choices[:6]
    card_w = (width - 20) / 3
    card_h = 72
    top = y - 175
    for index, item in enumerate(items):
        row, col = divmod(index, 3)
        left = x + col * (card_w + 10)
        bottom = top - row * (card_h + 10) - card_h
        _card(canvas, left, bottom, card_w, card_h, fill=PALE_GOLD, dashed=True)
        _text(canvas, item, x=left + 8, y=bottom + card_h - 22, width=card_w - 16, size=10, bold=True, max_lines=3)
    _text(canvas, "CUT OR POINT - then explain one choice.", x=x, y=top - 2 * (card_h + 10) - 15, width=width, size=9, bold=True)
    _lines(canvas, x, top - 2 * (card_h + 10) - 38, width, 2)


def _render_sequence(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    task = next((task for task in spec.tasks if task.kind == "sequence"), spec.tasks[0])
    items = task.choices[:6]
    columns = 3
    gap = 10
    card_w = (width - gap * 2) / 3
    card_h = 118
    for index, item in enumerate(items):
        row, col = divmod(index, columns)
        left = x + col * (card_w + gap)
        bottom = y - 25 - row * (card_h + gap) - card_h
        _card(canvas, left, bottom, card_w, card_h, fill=PALE_TEAL if index % 2 == 0 else PALE_CORAL, dashed=True)
        canvas.setFillColor(GOLD)
        canvas.circle(left + 18, bottom + card_h - 18, 10, stroke=0, fill=1)
        _text(canvas, "#", x=left + 14, y=bottom + card_h - 21, width=10, size=9, bold=True)
        _text(canvas, item, x=left + 12, y=bottom + card_h - 45, width=card_w - 24, size=10, bold=True, max_lines=4)
    base = y - 25 - 2 * (card_h + gap) - 12
    _text(canvas, "Tell the story in order:", x=x, y=base, width=width, size=10, bold=True)
    _lines(canvas, x, base - 22, width, 3)


def _diagram_kind(spec: WorksheetSpec, labels: list[str]) -> str:
    if spec.diagram_kind != "auto":
        return spec.diagram_kind
    words = " ".join(labels).casefold()
    if all(word in words for word in ("root", "stem", "leaf")):
        return "plant"
    if all(word in words for word in ("head", "thorax", "abdomen")):
        return "insect"
    if all(word in words for word in ("hull", "sail", "mast")):
        return "boat"
    return "generic"


def _draw_label_diagram(canvas: Canvas, kind: str, cx: float, cy: float) -> None:
    canvas.setLineWidth(3)
    canvas.setStrokeColor(TEAL)
    canvas.setFillColor(PALE_TEAL)
    if kind == "plant":
        canvas.line(cx, cy - 63, cx, cy + 45)
        canvas.setFillColor(GOLD)
        canvas.circle(cx, cy + 60, 22, stroke=1, fill=1)
        canvas.setFillColor(TEAL)
        canvas.ellipse(cx - 45, cy + 5, cx - 5, cy + 28, stroke=1, fill=1)
        canvas.ellipse(cx + 5, cy - 2, cx + 45, cy + 21, stroke=1, fill=1)
        canvas.line(cx, cy - 63, cx - 28, cy - 92)
        canvas.line(cx, cy - 63, cx + 28, cy - 92)
        canvas.line(cx, cy - 58, cx - 43, cy - 70)
        canvas.line(cx, cy - 58, cx + 43, cy - 70)
    elif kind == "insect":
        canvas.circle(cx - 58, cy, 31, stroke=1, fill=1)
        canvas.ellipse(cx - 25, cy - 35, cx + 35, cy + 35, stroke=1, fill=1)
        canvas.ellipse(cx + 31, cy - 48, cx + 107, cy + 48, stroke=1, fill=1)
        canvas.line(cx - 75, cy + 22, cx - 105, cy + 54)
        canvas.line(cx - 55, cy + 26, cx - 70, cy + 66)
        for offset in (-22, 0, 22):
            canvas.line(cx + offset, cy - 20, cx + offset - 30, cy - 62)
            canvas.line(cx + offset + 25, cy + 20, cx + offset + 45, cy + 62)
    elif kind == "boat":
        path = canvas.beginPath()
        path.moveTo(cx - 100, cy - 30)
        path.lineTo(cx + 100, cy - 30)
        path.lineTo(cx + 65, cy - 72)
        path.lineTo(cx - 65, cy - 72)
        path.close()
        canvas.drawPath(path, stroke=1, fill=1)
        canvas.line(cx, cy - 28, cx, cy + 90)
        sail = canvas.beginPath()
        sail.moveTo(cx + 5, cy + 82)
        sail.lineTo(cx + 5, cy)
        sail.lineTo(cx + 78, cy)
        sail.close()
        canvas.setFillColor(PALE_GOLD)
        canvas.drawPath(sail, stroke=1, fill=1)
    else:
        canvas.ellipse(cx - 90, cy - 70, cx + 90, cy + 70, stroke=1, fill=1)


def _label_target(kind: str, label: str, index: int, cx: float, cy: float) -> tuple[float, float]:
    word = label.casefold().strip()
    if kind == "plant":
        if "root" in word:
            return cx, cy - 72
        if "stem" in word:
            return cx, cy - 20
        if "leaf" in word:
            return (cx - 25, cy + 16) if index % 2 == 0 else (cx + 25, cy + 9)
        if "flower" in word or "petal" in word:
            return cx, cy + 60
    elif kind == "insect":
        if "head" in word:
            return cx - 58, cy
        if "thorax" in word:
            return cx + 5, cy
        if "abdomen" in word:
            return cx + 70, cy
        if "anten" in word:
            return cx - 73, cy + 42
        if "leg" in word:
            return cx + 15, cy - 43
        if "wing" in word:
            return cx + 28, cy + 38
    elif kind == "boat":
        if "hull" in word:
            return cx, cy - 48
        if "mast" in word:
            return cx, cy + 38
        if "sail" in word:
            return cx + 42, cy + 30
        if "deck" in word:
            return cx - 30, cy - 27
    angles = ((-72, 40), (72, 40), (-85, 0), (85, 0), (-65, -45), (65, -45))
    dx, dy = angles[index % len(angles)]
    return cx + dx, cy + dy


def _label_box_position(
    kind: str,
    label: str,
    fallback: tuple[float, float],
    *,
    x: float,
    width: float,
    cy: float,
) -> tuple[float, float]:
    word = label.casefold().strip()
    if kind == "plant":
        if "flower" in word or "petal" in word:
            return x + width - 135, cy + 75
        if "leaf" in word:
            return x + 10, cy + 35
        if "stem" in word:
            return x + width - 135, cy - 10
        if "root" in word:
            return x + 10, cy - 75
    if kind == "insect":
        if "anten" in word:
            return x + 10, cy + 75
        if "wing" in word:
            return x + width - 135, cy + 75
        if "head" in word or "thorax" in word:
            return x + 10, cy - 5 if "head" in word else cy - 70
        if "abdomen" in word or "leg" in word:
            return x + width - 135, cy - 5 if "abdomen" in word else cy - 70
    if kind == "boat":
        if "mast" in word:
            return x + 10, cy + 75
        if "sail" in word:
            return x + width - 135, cy + 75
        if "deck" in word:
            return x + 10, cy - 30
        if "hull" in word:
            return x + width - 135, cy - 65
    return fallback


def _render_label_parts(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    labels = [choice for task in spec.tasks for choice in task.choices][:6]
    if not labels:
        labels = ["part 1", "part 2", "part 3"]
    kind = _diagram_kind(spec, labels)
    _text(
        canvas,
        "WORD BANK: " + "  -  ".join(labels),
        x=x + 8,
        y=y - 4,
        width=width - 16,
        size=9,
        bold=True,
        max_lines=2,
    )
    center_x = x + width / 2
    center_y = y - 185
    _draw_label_diagram(canvas, kind, center_x, center_y)
    positions = ((x + 10, center_y + 75), (x + width - 135, center_y + 75), (x + 10, center_y), (x + width - 135, center_y), (x + 10, center_y - 75), (x + width - 135, center_y - 75))
    for index, label in enumerate(labels):
        left, line_y = _label_box_position(
            kind,
            label,
            positions[index],
            x=x,
            width=width,
            cy=center_y,
        )
        right_side = left > center_x
        target_x, target_y = _label_target(kind, label, index, center_x, center_y)
        canvas.setStrokeColor(LINE)
        canvas.line(target_x, target_y, left if right_side else left + 125, line_y)
        _card(canvas, left, line_y - 19, 125, 34, fill=PALE_CORAL if index % 2 else PALE_GOLD)
        _text(canvas, "________________", x=left + 7, y=line_y, width=111, size=8, bold=False, max_lines=1)
    lower = y - 315
    for index, task in enumerate(spec.tasks[:2], 1):
        lower = _task_prompt(canvas, task, x, lower, width, number=index) - 5
        _lines(canvas, x + 25, lower, width - 25, max(1, min(task.response_lines or 1, 2)))
        lower -= 38


def _render_checklist(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    _text(canvas, "LOOK FOR THESE CLUES", x=x, y=y, width=width, size=11, bold=True)
    items = [choice for task in spec.tasks for choice in task.choices]
    if not items:
        items = [task.instruction for task in spec.tasks]
    top = y - 30
    columns = 2
    cell_w = (width - 12) / 2
    for index, item in enumerate(items[:8]):
        row, col = divmod(index, columns)
        left = x + col * (cell_w + 12)
        bottom = top - row * 58 - 48
        _card(canvas, left, bottom, cell_w, 46, fill=PALE_TEAL if index % 2 == 0 else PALE_GOLD)
        _checkbox(canvas, left + 10, bottom + 28, item, cell_w - 20)
    lower = top - 4 * 58 - 15
    _text(canvas, "DRAW YOUR BEST CLUE", x=x, y=lower, width=width, size=10, bold=True)
    _card(canvas, x, lower - 145, width, 125, fill=white)
    _text(canvas, "One thing I still wonder:", x=x, y=lower - 165, width=width, size=9, bold=True)
    _lines(canvas, x, lower - 187, width, 2)


def _render_predict_test(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    predict = next((task for task in spec.tasks if task.kind == "predict"), spec.tasks[0])
    record = next((task for task in spec.tasks if task.kind == "record"), spec.tasks[-1])
    sections = (
        ("1  PREDICT", PALE_GOLD, 108, predict.instruction, predict.choices),
        ("2  TEST", PALE_TEAL, 148, "Do the test once. Change only the thing you predicted about.", []),
        ("3  RECORD", PALE_CORAL, 168, record.instruction, record.choices),
    )
    current = y
    for label, fill, height, instruction, choices in sections:
        _card(canvas, x, current - height, width, height - 10, fill=fill)
        _text(canvas, label, x=x + 12, y=current - 26, width=width - 24, size=11, bold=True)
        content_y = _text(canvas, instruction, x=x + 12, y=current - 48, width=width - 24, size=9, max_lines=3)
        if choices:
            for offset, choice in enumerate(choices[:3]):
                _checkbox(canvas, x + 14 + offset * width / 3, content_y - 6, choice, width / 3 - 10)
        else:
            _lines(canvas, x + 14, content_y - 8, width - 28, 2)
        current -= height


def _render_tally(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    labels = [choice for task in spec.tasks for choice in task.choices][:5] or ["TRY 1", "TRY 2", "TRY 3"]
    row_h = 48
    label_w = width * 0.42
    col_w = (width - label_w) / 3
    headers = ("MEASURE", "TALLY", "NOTICE")
    canvas.setFillColor(PALE_TEAL)
    canvas.rect(x, y - 38, width, 32, stroke=0, fill=1)
    for index, header in enumerate(headers):
        _text(canvas, header, x=x + label_w + index * col_w + 6, y=y - 19, width=col_w - 12, size=8, bold=True)
    top = y - 38
    canvas.setStrokeColor(LINE)
    for row, label in enumerate(labels):
        bottom = top - (row + 1) * row_h
        canvas.rect(x, bottom, width, row_h, stroke=1, fill=0)
        _text(canvas, label, x=x + 8, y=bottom + 28, width=label_w - 16, size=9, bold=True, max_lines=2)
        for col in range(1, 4):
            canvas.line(x + label_w + (col - 1) * col_w, bottom, x + label_w + (col - 1) * col_w, bottom + row_h)
    chart_bottom = top - len(labels) * row_h - 165
    _card(canvas, x, chart_bottom, width, 145, fill=PALE_GOLD)
    _text(canvas, "DRAW A BAR OR DOT FOR EACH RESULT", x=x + 12, y=chart_bottom + 122, width=width - 24, size=9, bold=True)
    for index in range(1, 5):
        canvas.setStrokeColor(LINE)
        canvas.line(x + 18, chart_bottom + 18 + index * 20, x + width - 18, chart_bottom + 18 + index * 20)


def _render_build(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    left_w = width * 0.58
    _card(canvas, x, y - 235, left_w, 220, fill=white)
    _text(canvas, "SKETCH YOUR IDEA", x=x + 12, y=y - 38, width=left_w - 24, size=10, bold=True)
    _text(canvas, "Label one part you can change.", x=x + 12, y=y - 58, width=left_w - 24, size=8)
    right_x = x + left_w + 12
    _card(canvas, right_x, y - 108, width - left_w - 12, 93, fill=PALE_GOLD)
    _text(canvas, "MISSION", x=right_x + 10, y=y - 38, width=width - left_w - 32, size=10, bold=True)
    _text(canvas, spec.directions, x=right_x + 10, y=y - 58, width=width - left_w - 32, size=8, max_lines=4)
    _card(canvas, right_x, y - 235, width - left_w - 12, 112, fill=PALE_TEAL)
    _text(canvas, "RULES", x=right_x + 10, y=y - 146, width=width - left_w - 32, size=10, bold=True)
    for index, task in enumerate(spec.tasks[:3], 1):
        _text(canvas, f"{index}. {task.instruction}", x=right_x + 10, y=y - 166 - (index - 1) * 24, width=width - left_w - 32, size=8, max_lines=2)
    lower = y - 255
    for label, fill in (("TEST 1", PALE_CORAL), ("CHANGE ONE THING", PALE_GOLD), ("TEST 2", PALE_TEAL)):
        _card(canvas, x, lower - 62, width, 52, fill=fill)
        _text(canvas, label, x=x + 12, y=lower - 30, width=130, size=9, bold=True)
        _lines(canvas, x + 145, lower - 31, width - 160, 1)
        lower -= 65


def _render_mystery(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    clues = [choice for task in spec.tasks for choice in task.choices][:6]
    if not clues:
        clues = [task.instruction for task in spec.tasks][:6]
    _text(canvas, "CASE FILE: Read one clue at a time.", x=x, y=y, width=width, size=11, bold=True)
    card_w = (width - 20) / 3
    for index, clue in enumerate(clues):
        row, col = divmod(index, 3)
        left = x + col * (card_w + 10)
        bottom = y - 35 - row * 112 - 100
        _card(canvas, left, bottom, card_w, 96, fill=PALE_GOLD if index % 2 == 0 else PALE_TEAL, dashed=True)
        canvas.setFillColor(CORAL)
        canvas.circle(left + 18, bottom + 77, 10, stroke=0, fill=1)
        _text(canvas, str(index + 1), x=left + 15, y=bottom + 80, width=8, size=8, bold=True)
        _text(canvas, clue, x=left + 12, y=bottom + 57, width=card_w - 24, size=9, bold=True, max_lines=4)
    lower = y - 265
    _card(canvas, x, lower - 112, width, 100, fill=PALE_CORAL)
    _text(canvas, "MY BEST SOLUTION", x=x + 12, y=lower - 34, width=width - 24, size=10, bold=True)
    _lines(canvas, x + 12, lower - 56, width - 24, 3)
    _text(canvas, "Clue that helped most: ______", x=x, y=lower - 137, width=width, size=9, bold=True)


def _render_field_guide(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    photo_w = width * 0.48
    _draw_photo(canvas, spec.researched_image, x, y - 235, photo_w, 220)
    right_x = x + photo_w + 12
    _card(canvas, right_x, y - 235, width - photo_w - 12, 220, fill=PALE_TEAL)
    _text(canvas, "FIELD MARKS", x=right_x + 12, y=y - 38, width=width - photo_w - 36, size=10, bold=True)
    items = [choice for task in spec.tasks for choice in task.choices]
    if not items:
        items = [task.instruction for task in spec.tasks]
    for index, item in enumerate(items[:5]):
        _checkbox(canvas, right_x + 12, y - 66 - index * 30, item, width - photo_w - 36)
    lower = y - 255
    _card(canvas, x, lower - 155, width * 0.58, 140, fill=white)
    _text(canvas, "DRAW ONE DETAIL", x=x + 12, y=lower - 38, width=width * 0.54, size=10, bold=True)
    _card(canvas, x + width * 0.61, lower - 155, width * 0.39, 140, fill=PALE_GOLD)
    _text(canvas, "WHERE MIGHT IT LIVE?", x=x + width * 0.61 + 10, y=lower - 38, width=width * 0.35, size=9, bold=True)
    _lines(canvas, x + width * 0.61 + 10, lower - 64, width * 0.35, 4)


def _render_play_pieces(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    pieces = [choice for task in spec.tasks for choice in task.choices]
    if len(pieces) < 3:
        pieces = [task.instruction for task in spec.tasks]
    pieces = pieces[:6]
    _text(canvas, "CUT, PLACE, POINT, OR PLAY", x=x, y=y, width=width, size=11, bold=True)
    columns = 3
    gap = 10
    card_w = (width - gap * 2) / columns
    card_h = 142 if len(pieces) <= 3 else 116
    top = y - 28
    for index, piece in enumerate(pieces):
        row, col = divmod(index, columns)
        left = x + col * (card_w + gap)
        bottom = top - row * (card_h + gap) - card_h
        _card(canvas, left, bottom, card_w, card_h, fill=PALE_GOLD if index % 2 == 0 else PALE_TEAL, dashed=True)
        canvas.setFillColor(TEAL if index % 2 == 0 else CORAL)
        shape = spec.piece_shape
        if shape == "auto":
            shape = "leaf" if "leaf" in piece.casefold() else "target" if "target" in piece.casefold() else "circle"
        icon_x = left + card_w / 2
        icon_y = bottom + card_h * 0.62
        radius = min(34, card_w * 0.20)
        if shape == "leaf":
            canvas.ellipse(icon_x - radius, icon_y - radius * 0.58, icon_x + radius, icon_y + radius * 0.58, stroke=0, fill=1)
            canvas.setStrokeColor(INK)
            canvas.line(icon_x - radius * 0.75, icon_y - radius * 0.75, icon_x + radius * 0.55, icon_y + radius * 0.55)
        elif shape == "target":
            canvas.circle(icon_x, icon_y, radius, stroke=0, fill=1)
            canvas.setFillColor(white)
            canvas.circle(icon_x, icon_y, radius * 0.60, stroke=0, fill=1)
            canvas.setFillColor(CORAL)
            canvas.circle(icon_x, icon_y, radius * 0.28, stroke=0, fill=1)
        elif shape == "arrow":
            canvas.setFont("Helvetica-Bold", 34)
            canvas.drawCentredString(icon_x, icon_y - 11, "->")
        else:
            canvas.circle(icon_x, icon_y, radius, stroke=0, fill=1)
        _text(canvas, piece, x=left + 10, y=bottom + 33, width=card_w - 20, size=11, bold=True, max_lines=2)
    lower = top - ((len(pieces) + 2) // 3) * (card_h + gap) - 10
    _text(canvas, "PLAY RULE", x=x, y=lower, width=width, size=9, bold=True)
    _text(canvas, spec.tasks[-1].instruction, x=x, y=lower - 20, width=width, size=9, max_lines=3)


def _render_legacy(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    task_height = min(92, max(62, (y - 78) / len(spec.tasks)))
    for number, task in enumerate(spec.tasks, start=1):
        bottom = y - task_height + 5
        _card(canvas, x, bottom, width, task_height - 3, fill=PALE_TEAL if number % 2 else PALE_CORAL)
        text_y = _task_prompt(canvas, task, x + 8, y - 12, width - 16, number=number)
        if task.choices:
            _text(canvas, "   ○ " + "     ○ ".join(task.choices), x=x + 35, y=text_y - 3, width=width - 48, size=9)
        if task.kind in {"draw", "label"}:
            canvas.setStrokeColor(LINE)
            canvas.roundRect(x + 35, bottom + 8, width - 52, max(20, text_y - bottom - 10), 6, stroke=1, fill=0)
        elif task.response_lines:
            _lines(canvas, x + 35, text_y - 3, width - 52, task.response_lines, gap=14)
        y = bottom - 5


RENDERERS = {
    "look_notice_wonder": _render_look_notice,
    "compare_circle_explain": _render_compare,
    "match_two_columns": _render_matching,
    "cut_sort_categories": _render_cut_sort,
    "sequence_story_cards": _render_sequence,
    "label_parts": _render_label_parts,
    "observation_checklist": _render_checklist,
    "predict_test_record": _render_predict_test,
    "measure_tally_graph": _render_tally,
    "build_test_improve": _render_build,
    "mystery_clue_case": _render_mystery,
    "mini_field_guide": _render_field_guide,
    "play_pieces_targets": _render_play_pieces,
    "legacy_mixed": _render_legacy,
}


def render_worksheet(canvas: Canvas, spec: WorksheetSpec, x: float, y: float, width: float) -> None:
    draw_theme_border(canvas, spec)
    _text(canvas, spec.directions, x=x, y=y, width=width, size=12, leading=15, bold=True, max_lines=3)
    content_y = y - min(54, 15 * len(_wrap(spec.directions, "Helvetica-Bold", 12, width)) + 12)
    RENDERERS[spec.archetype](canvas, spec, x, content_y, width)
    if spec.researched_image:
        if spec.researched_image.safety.context_note:
            _text(
                canvas,
                "Image note: " + spec.researched_image.safety.context_note,
                x=x,
                y=82,
                width=width,
                size=6.5,
                leading=8,
                max_lines=2,
            )
        credit = f"Image: {spec.researched_image.rights.attribution} - {spec.researched_image.rights.license_name}"
        _text(canvas, credit, x=x, y=63, width=width, size=6.5, leading=8, max_lines=1)
