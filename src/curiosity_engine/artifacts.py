from __future__ import annotations

import html
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from .artifact_validation import validate_artifact_spec
from .contracts import ArtifactSpec, VisualQAResult
from .openai_backend import OpenAIBackend, data_url_for_image


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _validated(spec: dict[str, Any] | ArtifactSpec, *, allow_tier_c: bool = False) -> ArtifactSpec:
    parsed = spec if isinstance(spec, ArtifactSpec) else ArtifactSpec.model_validate(spec)
    errors = validate_artifact_spec(parsed.model_dump(mode="json", exclude_none=True))
    if errors:
        raise ValueError("invalid artifact spec: " + "; ".join(errors))
    if parsed.trust_tier == "C" and not allow_tier_c:
        raise ValueError("Tier C artifact generation is outside the MVP trust boundary and fails closed")
    return parsed


def render_html(spec: dict[str, Any], output_path: str | Path, *, strict: bool = False) -> Path:
    if strict:
        parsed = _validated(spec)
        spec = parsed.model_dump(mode="json")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = spec.get("title", "Wonder Page")
    kicker = spec.get("kicker", "CURIOUS?")
    prompt = spec.get("prompt", "What do you notice?")
    body = spec.get("body", [])
    body_html = "".join(f"<li>{_esc(x)}</li>" for x in body)
    footer = spec.get("footer", "Predict first. Then investigate.")
    trust = spec.get("trust_tier", "A")
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_esc(title)}</title>
<meta name="curiosity-trust-tier" content="{_esc(trust)}">
<style>
@page {{ size: Letter; margin: 0.55in; }}
body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; color:#111; margin:0; }}
.sheet {{ min-height: 9.7in; border: 4px solid #111; padding: 28px; box-sizing:border-box; display:flex; flex-direction:column; }}
.kicker {{ font-weight:800; letter-spacing:.12em; font-size:14px; }}
h1 {{ font-size:38px; line-height:1.02; margin:18px 0 24px; }}
.prompt {{ font-size:26px; font-weight:700; padding:18px 0; border-top:2px solid #111; border-bottom:2px solid #111; }}
ul {{ font-size:20px; line-height:1.45; }}
.footer {{ margin-top:auto; font-size:16px; font-weight:700; }}
</style></head>
<body><main class="sheet">
<div class="kicker">{_esc(kicker)}</div>
<h1>{_esc(title)}</h1>
<div class="prompt">{_esc(prompt)}</div>
<ul>{body_html}</ul>
<div class="footer">{_esc(footer)}</div>
</main></body></html>"""
    out.write_text(html_doc, encoding="utf-8")
    return out


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


def render_pdf(spec: dict[str, Any] | ArtifactSpec, output_path: str | Path) -> Path:
    """Render the supported one-page, child-facing Tier A/B PDF."""

    parsed = _validated(spec)
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height = letter
    canvas = Canvas(str(out), pagesize=letter, pageCompression=1)
    canvas.setTitle(parsed.title)
    canvas.setAuthor("Curiosity Engine")
    canvas.setSubject(f"Trust tier {parsed.trust_tier}; parent review required")

    cream = HexColor("#FFF9EC")
    coral = HexColor("#F26B5B")
    navy = HexColor("#17324D")
    teal = HexColor("#2A9D8F")
    canvas.setFillColor(cream)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setFillColor(navy)
    canvas.roundRect(30, 30, width - 60, height - 60, 18, stroke=0, fill=1)
    canvas.setFillColor(white)
    canvas.roundRect(38, 38, width - 76, height - 76, 14, stroke=0, fill=1)

    left = 64
    content_width = width - 128
    y = height - 72
    canvas.setFillColor(coral)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(left, y, parsed.kicker.upper())
    canvas.setFillColor(navy)
    y -= 27
    title_size = 27 if len(parsed.title) <= 80 else 23
    title_lines = _wrap(parsed.title, "Helvetica-Bold", title_size, content_width)
    if len(title_lines) > 3:
        raise ValueError("artifact title is too long for the one-page MVP template")
    canvas.setFont("Helvetica-Bold", title_size)
    for line in title_lines:
        canvas.drawString(left, y, line)
        y -= title_size * 1.08

    prompt_lines = _wrap(parsed.prompt, "Helvetica-Bold", 16, content_width - 38)
    if len(prompt_lines) > 6:
        raise ValueError("artifact prompt is too long for the one-page MVP template")
    box_height = max(76, 30 + len(prompt_lines) * 20)
    y -= 10
    canvas.setFillColor(HexColor("#E8F5F2"))
    canvas.roundRect(left, y - box_height, content_width, box_height, 12, stroke=0, fill=1)
    canvas.setFillColor(teal)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(left + 18, y - 21, "NOTICE + PREDICT")
    canvas.setFillColor(navy)
    canvas.setFont("Helvetica-Bold", 16)
    text_y = y - 44
    for line in prompt_lines:
        canvas.drawString(left + 18, text_y, line)
        text_y -= 20
    y -= box_height + 27

    canvas.setFillColor(navy)
    body_size = 14
    body_lines: list[tuple[bool, str]] = []
    for item in parsed.body:
        wrapped = _wrap(item, "Helvetica", body_size, content_width - 35)
        body_lines.extend((index == 0, line) for index, line in enumerate(wrapped))
    available_height = y - 98
    needed = len(body_lines) * 22
    if needed > available_height:
        body_size = 12
        body_lines = []
        for item in parsed.body:
            wrapped = _wrap(item, "Helvetica", body_size, content_width - 35)
            body_lines.extend((index == 0, line) for index, line in enumerate(wrapped))
        needed = len(body_lines) * 19
    if needed > available_height:
        raise ValueError("artifact body does not fit the one-page MVP template")
    canvas.setFont("Helvetica", body_size)
    for is_first, line in body_lines:
        if is_first:
            canvas.setFillColor(coral)
            canvas.circle(left + 6, y + 4, 4, stroke=0, fill=1)
        canvas.setFillColor(navy)
        canvas.drawString(left + 22, y, line)
        y -= 22 if body_size == 14 else 19

    if y > 245:
        box_top = y - 12
        box_bottom = 118
        gutter = 14
        box_width = (content_width - gutter) / 2
        canvas.setLineWidth(1)
        canvas.setStrokeColor(HexColor("#BFD2D9"))
        canvas.setFillColor(HexColor("#F7FAFB"))
        canvas.roundRect(left, box_bottom, box_width, box_top - box_bottom, 10, stroke=1, fill=1)
        canvas.roundRect(
            left + box_width + gutter,
            box_bottom,
            box_width,
            box_top - box_bottom,
            10,
            stroke=1,
            fill=1,
        )
        canvas.setFillColor(teal)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(left + 14, box_top - 22, "MY PREDICTION")
        canvas.drawString(left + box_width + gutter + 14, box_top - 22, "WHAT I NOTICED")
        canvas.setStrokeColor(HexColor("#D8E2E8"))
        line_y = box_top - 48
        while line_y > box_bottom + 18:
            canvas.line(left + 14, line_y, left + box_width - 14, line_y)
            canvas.line(
                left + box_width + gutter + 14,
                line_y,
                left + content_width - 14,
                line_y,
            )
            line_y -= 28

    canvas.setStrokeColor(HexColor("#D8E2E8"))
    canvas.line(left, 86, left + content_width, 86)
    canvas.setFillColor(navy)
    canvas.setFont("Helvetica-Bold", 10)
    footer_lines = _wrap(parsed.footer, "Helvetica-Bold", 10, content_width - 100)
    canvas.drawString(left, 68, footer_lines[0])
    canvas.setFillColor(HexColor("#667784"))
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(left + content_width, 68, f"Trust {parsed.trust_tier} • Parent review required")
    canvas.showPage()
    canvas.save()
    return out


def load_spec(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_visual_qa(preview_path: str | Path) -> VisualQAResult:
    image = Image.open(preview_path).convert("RGB")
    reasons: list[str] = []
    if image.width < 600 or image.height < 700:
        reasons.append("rendered preview resolution is too small")
    extrema = ImageChops.difference(image, Image.new("RGB", image.size, image.getpixel((0, 0)))).getbbox()
    if extrema is None:
        reasons.append("rendered page appears blank")
    return VisualQAResult(verdict="fail" if reasons else "pass", reasons=reasons, inspected_pages=1)


def model_visual_qa(preview_path: str | Path, spec: ArtifactSpec, backend: OpenAIBackend) -> VisualQAResult:
    result = backend.complete(
        role="visual_qa",
        system=(
            "Inspect the final child-facing page. Fail for clipping, overlap, illegibility, confusing hierarchy, "
            "age-inappropriate density, factual inconsistency, bad labels, or misleading knowledge-bearing visuals."
        ),
        payload={
            "artifact": spec.model_dump(mode="json"),
            "image_data_urls": [data_url_for_image(preview_path)],
        },
        response_model=VisualQAResult,
    )
    return VisualQAResult.model_validate(result)
