"""Generated subject artwork with code-owned text, counts, layout and work space."""
from __future__ import annotations

import math
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth

from .contracts import PicturePage, PictureSymbol

PICTURE_PAGE_INSTRUCTIONS = """
For a picture-based, nonreading, visual, card, or prediction/results request, supply picture_page.
This renders INSTEAD OF the legacy text worksheet/challenge body. Put ALL instructions needed by
the child in picture_page.directions and cell.action. Keep those short enough to say aloud.
Cells are large picture cards or work areas, in row-major order; choose 1-3 columns and 2-6 cells.
For three or four large subject pictures, prefer two columns: three shallow full-width rows waste picture space.
Use two columns for pairs/comparisons. Leave results/predictions blank when the child should discover them.
response_space=draw supplies a large empty drawing/paint area, mark supplies an empty mark box,
and none adds no response box or writing lines. Never tell the parent to copy or draw the layout.
Define recognizable subject pictures as reusable symbols. The configured image model draws each
symbol from its description: describe a specific recognizable object or story scene with the
distinguishing details needed by the child. These are the MAIN useful pictures, not corner decoration.
For story cards repeat character appearance in each description and make actions visibly different.
Use generic subjects only, with no family names, locations, private history or identifying likenesses.
Do not ask the image model to draw text, labels, arrows, answer boxes or a whole worksheet.
Leave illustration=null; code generates and binds the actual asset. Text and repeated quantities are
overlaid by the renderer. Preserve a previous symbol's description verbatim when only changing size,
labels or response spaces: that reuses the same artwork without another image call.
For a labeled-part diagram, supply annotations on the symbol: short text, position=[x,y] for
the label and target=[x,y] for the pictured part, percentages of the square with origin BOTTOM LEFT.
Code draws the text on a white backing with a leader to that part. Describe the part's placement
in the image description so the artwork and annotation agree. A caption saying 'point to the root'
is not a root label. Use annotations=[] on picture clues, story cards and tokens without requested labels.
When image_generation_available=true, use strokes=[]: describe the art instead of encoding it as coordinates.
Only when image_generation_available=false provide a simple vector version: 1-16 strokes in a 100x100 square,
with origin at bottom left. Ellipse/rect use two opposite corners. Polygon is a closed contour;
line is an open polyline. Use fill=none for an unfilled outline. Use simple silhouettes plus distinguishing details, generous margins,
and high contrast. These are actual vector pictures, not emoji, text labels, generic circles,
or suggestions to draw the subject. Do not write a symbol name inside its drawing.
Each cell's items references symbol_id and count; code draws EXACTLY that many copies.
For a counted set draw ONE object in the symbol and let count supply the quantity. Do not encode
multiple objects inside a symbol or reveal a solution that the child is supposed to find.
Use pictures to supply the requested activity, story, objects or relationships, not decoration.
For story scenes, check the state at each moment: an object that will be discovered later in
one place must not already be shown found in another search scene. Describe only the objects
actually present at that moment. Do not print 'first', 'next' or 'last' clues on shuffled story cards.
Use most of each card for its picture. Avoid repeating the same directions in every cell.
Do not add checklists, recording, scoring or extra review cards unless the parent needs them.
When the child only needs to point or look, use response_space=none; marking is not required.
The ordinary legacy fields remain required as semantic descriptions, but only picture_page is
printed in this mode. Do not put necessary child directions only in a legacy field.
When a previous page image is supplied, revise that actual delivered page. Preserve what the
parent asks to keep and verify the requested change at printed size, not just in the title.
For larger pictures, reduce repeated text/unused cells and choose a grid whose picture area is
actually larger. A wide shallow row can shrink a picture compared with a taller card.
previous_page_grid_options measures the old content in each possible grid using the real
renderer. picture_sizes_pt gives each picture's longest printed side (72 points = one inch).
Use these measurements to choose a genuinely larger layout when the parent requests enlargement.
"""

PICTURE_PAGE_REVIEW_INSTRUCTIONS = """
Review the actual illustrated page for a busy parent with a young child. Image 1 is the candidate;
image 2, when present, is the previously delivered page. Return pass, revise or reject and concrete repairs.
The current parent brief determines the task. Do not require a quiz, experiment phase, recording,
extra precaution or added adult explanation unless the requested activity actually needs it.
Separate cards on the SAME PAGE can be compared by pointing; do not demand one shared scene or a
particular grid. No writing means no required child writing, not no printed words. One adult-read-aloud
instruction, short captions, title and footer are acceptable. Do not reject a page for these alone.

Inspect the pictures themselves before reading captions. Are the subjects recognizable and inviting,
their distinguishing details visible at printed size, and the requested actions/relationships actually
depicted? For story cards, trace where every recurring object is at each moment. Searching and finding
must be visibly different; an object first found later cannot already appear in another search location.
Captions cannot repair a contradictory image. Do not reveal sequence answers in shuffled card headings.
For labeled parts, each leader endpoint must touch the correct pictured part, not empty white space or
an adjacent part; the label backing must not cover what it explains. Count actual objects, including
any extra objects inside a repeated image. Check the specific factual relationship a diagram teaches.

Can the nonreader do the stated activity by looking, pointing, moving or drawing, with one short adult
read-aloud instruction if needed? Preserve supplies, no/without constraints and large blank response areas.
Do not leave the parent to invent or redraw missing materials. Reject concrete unsafe child actions,
material misinformation and unsupported learning claims. Ordinary supervised play is not a reason to refuse.

For revisions preserve the requested subjects and artwork. Compare actual picture_sizes_pt, not the
title: an enlargement requires a larger printed picture. Judge only what is printed, not hidden legacy
fields. Cosmetic preferences alone do not require repair, but crude placeholders do not fulfill a request
for rich illustrations. State the main obstacle first; a beautiful picture with wrong content is not a pass.
"""


def _annotations(canvas, symbol: PictureSymbol, x: float, y: float, size: float) -> None:
    font_size = min(12, max(8, size * .06))
    if symbol.annotations:
        required = max(60, max(stringWidth(a.text, "Helvetica-Bold", font_size) for a in symbol.annotations) * 1.6)
        if size < required:
            raise ValueError(
                f"A labeled picture is only {size:.0f} pt wide; it needs at least {required:.0f} pt for visible art "
                "and readable labels. Show shared choices once in a full-width cell without a response box, "
                "and put the blank results area in separate cells. Reduce repeated panels or text."
            )
    for label in symbol.annotations:
        width = stringWidth(label.text, "Helvetica-Bold", font_size)
        lx = min(x + size - width/2 - 3, max(x + width/2 + 3, x + size * label.position[0]/100))
        ly = y + size * label.position[1]/100
        tx, ty = x + size * label.target[0]/100, y + size * label.target[1]/100
        canvas.saveState()
        canvas.setStrokeColor(HexColor("#18354D"))
        canvas.setFillColor(HexColor("#18354D"))
        canvas.setLineWidth(.8)
        canvas.line(lx, ly, tx, ty)
        canvas.circle(tx, ty, 1.8, stroke=0, fill=1)
        canvas.setFillColor(HexColor("#FFFFFF"))
        canvas.roundRect(lx-width/2-3, ly-3, width+6, font_size+5, 2, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#18354D"))
        canvas.setFont("Helvetica-Bold", font_size)
        canvas.drawCentredString(lx, ly, label.text)
        canvas.restoreState()


def _symbol(canvas, symbol: PictureSymbol, x: float, y: float, size: float) -> None:
    if symbol.illustration is not None:
        asset = symbol.illustration
        path = Path(asset.local_path)
        if path.is_symlink():
            raise ValueError("picture artwork file was replaced")
        data = path.read_bytes()
        if len(data) != asset.byte_count or sha256(data).hexdigest() != asset.sha256:
            raise ValueError("picture artwork no longer matches the reviewed asset")
        canvas.drawImage(ImageReader(BytesIO(data)), x, y, width=size, height=size,
                         preserveAspectRatio=True, anchor="c", mask="auto")
        _annotations(canvas, symbol, x, y, size)
        return
    if not symbol.strokes:
        raise ValueError("picture has no generated artwork or vector drawing")
    points = [point for stroke in symbol.strokes for point in stroke.points]
    left, right = min(p[0] for p in points), max(p[0] for p in points)
    bottom, top = min(p[1] for p in points), max(p[1] for p in points)
    extent = max(right-left, top-bottom, 1)
    scale = size / extent
    canvas.saveState()
    # Fit the actual drawing, not the model's unused coordinate margins. This
    # keeps small source glyphs large and recognizable on a nonreader's page.
    canvas.translate(x+(size-(right-left)*scale)/2-left*scale,
                     y+(size-(top-bottom)*scale)/2-bottom*scale)
    canvas.scale(scale, scale)
    canvas.setLineWidth(extent*.018)
    canvas.setLineJoin(1)
    canvas.setLineCap(1)
    for stroke in symbol.strokes:
        filled = int(stroke.fill != "none")
        if filled:
            canvas.setFillColor(HexColor(stroke.fill))
        canvas.setStrokeColor(HexColor(stroke.stroke))
        points = stroke.points
        if stroke.kind in {"ellipse", "rect"}:
            x1, x2 = sorted([points[0][0], points[1][0]])
            y1, y2 = sorted([points[0][1], points[1][1]])
            if stroke.kind == "ellipse":
                canvas.ellipse(x1, y1, x2, y2, fill=filled, stroke=1)
            else:
                canvas.roundRect(x1, y1, x2-x1, y2-y1, min(3, (x2-x1)/4, (y2-y1)/4), fill=filled, stroke=1)
        else:
            path = canvas.beginPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            if stroke.kind == "polygon":
                path.close()
            canvas.drawPath(path, stroke=1, fill=filled if stroke.kind == "polygon" else 0)
    canvas.restoreState()
    _annotations(canvas, symbol, x, y, size)


def render_picture_page(canvas, page: PicturePage, x: float, y: float, width: float) -> list[float]:
    picture_sizes = []
    ink, border = HexColor("#18354D"), HexColor("#ADC3CD")
    canvas.setFillColor(ink)
    canvas.setFont("Helvetica", 12)
    for line in simpleSplit(page.directions, "Helvetica", 12, width):
        canvas.drawString(x, y, line)
        y -= 16
    y -= 12
    placements = []
    row, column = 0, 0
    for cell in page.cells:
        span = min(cell.column_span, page.columns)
        if column + span > page.columns:
            row, column = row + 1, 0
        placements.append((row, column, span, cell))
        column += span
    rows = row + 1
    gap = 12
    column_w = (width - (page.columns-1)*gap) / page.columns
    # Reserve the actual text height, then give remaining space to pictures and
    # drawing. A heading-only row must not consume an equal share of the page.
    row_bases, row_weights = [0.0]*rows, [0.0]*rows
    for row, _column, span, cell in placements:
        cell_w = column_w * span + gap * (span - 1)
        text_h = (41 + 16*len(simpleSplit(cell.label, "Helvetica-Bold", 13, cell_w-24))
                  + 13*len(simpleSplit(cell.action, "Helvetica", 10, cell_w-24)))
        if cell.response_space == "mark":
            text_h += 30
        row_bases[row] = max(row_bases[row], text_h)
        weight = (2 if cell.items else 0) + (1.25 if cell.response_space == "draw" else 0)
        row_weights[row] = max(row_weights[row], weight)
    remaining = max(0, y - (24 if page.display == "screen" else 76) - (rows-1)*gap - sum(row_bases))
    total_weight = sum(row_weights) or 1
    row_heights = [base + remaining*weight/total_weight for base, weight in zip(row_bases, row_weights, strict=True)]
    symbols = {symbol.id: symbol for symbol in page.symbols}
    for row, column, span, cell in placements:
        cell_h = row_heights[row]
        cell_w = column_w * span + gap * (span - 1)
        left = x + column*(column_w+gap)
        bottom = y - sum(row_heights[:row+1]) - row*gap
        canvas.setStrokeColor(border)
        canvas.setLineWidth(1)
        if page.show_borders:
            canvas.roundRect(left, bottom, cell_w, cell_h, 12, stroke=1, fill=0)
        top = bottom + cell_h - 22
        canvas.setFillColor(ink)
        canvas.setFont("Helvetica-Bold", 13)
        for line in simpleSplit(cell.label, "Helvetica-Bold", 13, cell_w-24):
            canvas.drawCentredString(left+cell_w/2, top, line)
            top -= 16
        canvas.setFont("Helvetica", 10)
        for line in simpleSplit(cell.action, "Helvetica", 10, cell_w-24):
            canvas.drawCentredString(left+cell_w/2, top, line)
            top -= 13
        object_floor = bottom+14
        if cell.response_space == "draw":
            blank_h = max(45, (top-bottom-20)*0.42) if cell.items else top-bottom-22
            canvas.setStrokeColor(border)
            canvas.roundRect(left+12, bottom+12, cell_w-24, blank_h, 8, stroke=1, fill=0)
            object_floor = bottom+blank_h+22
        elif cell.response_space == "mark":
            canvas.roundRect(left+cell_w-38, bottom+12, 24, 24, 5, stroke=1, fill=0)
            object_floor += 30
        objects = [symbols[item.symbol_id] for item in cell.items for _ in range(item.count)]
        if objects:
            available_h = max(1, top-object_floor-5)
            columns = min(len(objects), max(1, math.ceil(math.sqrt(len(objects)*(cell_w-24)/available_h))))
            object_rows = math.ceil(len(objects)/columns)
            size = min((cell_w-24)/columns, available_h/object_rows)
            for position, symbol in enumerate(objects):
                row, column = divmod(position, columns)
                in_row = min(columns, len(objects)-row*columns)
                sx = left+(cell_w-in_row*size)/2+column*size
                sy = object_floor+(available_h-object_rows*size)/2+(object_rows-row-1)*size
                drawn = size*.92*page.picture_scale
                minimum = 30 if cell.response_space == "draw" and not symbol.annotations else 60
                if len(objects) == 1 and drawn < minimum:
                    raise ValueError("The main picture is too small to use. Remove repeated panels or text and give the picture more space.")
                _symbol(canvas, symbol, sx+(size-drawn)/2, sy+(size-drawn)/2, drawn)
                picture_sizes.append(round(drawn, 1))
    return picture_sizes
