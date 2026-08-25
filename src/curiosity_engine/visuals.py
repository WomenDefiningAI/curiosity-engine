from __future__ import annotations

import io
import os
import re
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageChops, ImageDraw, ImageFont

from .contracts import VisualIntent
from .db import connect, init_db, jdump, jload, utcnow
from .openai_image_backend import ImageBackend, configured_image_backend
from .trust import validate_response_visual_intent

RENDERER_VERSION = "response-card-v1"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MIN_DIMENSION = 600
MAX_DIMENSION = 2_048
MAX_VISUAL_JOB_ATTEMPTS = 3
STALE_VISUAL_JOB_MINUTES = 10

PUBLIC_PROPER_NOUNS = {"earth", "mars", "moon", "sun"}

PALETTE = {
    "cream": "#FFF9EC",
    "ink": "#17324D",
    "teal": "#2A9D8F",
    "coral": "#F26B5B",
    "gold": "#F4B942",
    "sky": "#DDF2F8",
    "mint": "#E8F5F2",
    "white": "#FFFFFF",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(filename, size=size)
    except OSError:  # pragma: no cover - Pillow wheels normally bundle DejaVu
        return ImageFont.load_default(size=size)


def _wrapped(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _draw_icon(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], icon: str) -> None:
    left, top, right, bottom = box
    cx, cy = (left + right) // 2, (top + bottom) // 2
    ink = PALETTE["ink"]
    accent = PALETTE["coral"]
    teal = PALETTE["teal"]
    width = 8
    if icon == "height":
        draw.line((cx, top + 8, cx, bottom - 8), fill=teal, width=width)
        draw.polygon([(cx, top), (cx - 14, top + 22), (cx + 14, top + 22)], fill=teal)
        draw.polygon([(cx, bottom), (cx - 14, bottom - 22), (cx + 14, bottom - 22)], fill=teal)
        draw.rectangle((left + 8, cy, cx - 20, bottom), outline=ink, width=6)
    elif icon == "weight":
        draw.rounded_rectangle((left + 15, top + 28, right - 15, bottom - 10), 16, fill=teal, outline=ink, width=6)
        draw.arc((cx - 24, top + 2, cx + 24, top + 48), 180, 360, fill=ink, width=7)
    elif icon == "strength":
        draw.line((left + 18, cy, right - 18, cy), fill=ink, width=12)
        for x in (left + 8, left + 24, right - 24, right - 8):
            draw.line((x, cy - 26, x, cy + 26), fill=accent, width=10)
    elif icon in {"go", "turn"}:
        draw.arc((left + 12, top + 12, right - 12, bottom - 12), 35, 315, fill=teal, width=10)
        draw.polygon([(right - 7, cy), (right - 34, cy - 15), (right - 30, cy + 17)], fill=teal)
    elif icon == "stop":
        points = [(cx - 34, top + 6), (cx + 34, top + 6), (right - 6, cy - 34), (right - 6, cy + 34),
                  (cx + 34, bottom - 6), (cx - 34, bottom - 6), (left + 6, cy + 34), (left + 6, cy - 34)]
        draw.polygon(points, fill=accent, outline=ink)
    elif icon == "look":
        draw.ellipse((left + 4, cy - 32, right - 4, cy + 32), outline=ink, width=7)
        draw.ellipse((cx - 16, cy - 16, cx + 16, cy + 16), fill=teal)
    elif icon == "predict":
        draw.ellipse((left + 10, top + 18, right - 10, bottom - 20), fill=PALETTE["sky"], outline=ink, width=6)
        draw.ellipse((left + 5, bottom - 22, left + 25, bottom - 2), fill=PALETTE["sky"], outline=ink, width=4)
        draw.text((cx, cy - 5), "?", font=_font(48, bold=True), fill=accent, anchor="mm")
    elif icon == "try":
        draw.polygon([(cx - 12, top + 5), (cx + 22, top + 5), (cx + 4, cy - 2),
                      (cx + 30, cy - 2), (cx - 22, bottom - 5), (cx - 4, cy + 10), (cx - 30, cy + 10)], fill=PALETTE["gold"])
    elif icon == "robot":
        draw.rounded_rectangle((left + 12, top + 18, right - 12, bottom - 8), 18, fill=PALETTE["sky"], outline=ink, width=6)
        draw.line((cx, top + 18, cx, top + 2), fill=ink, width=5)
        draw.ellipse((cx - 7, top - 3, cx + 7, top + 11), fill=accent)
        draw.ellipse((cx - 28, cy - 12, cx - 14, cy + 2), fill=teal)
        draw.ellipse((cx + 14, cy - 12, cx + 28, cy + 2), fill=teal)
        draw.arc((cx - 24, cy - 2, cx + 24, cy + 28), 15, 165, fill=ink, width=5)
    else:
        glyph = "?" if icon == "question" else "✦"
        draw.text((cx, cy), glyph, font=_font(64, bold=True), fill=accent, anchor="mm")


def render_deterministic_visual(intent: VisualIntent, output_path: str | Path) -> Path:
    if intent.kind not in {"comparison_cards", "activity_sequence"}:
        raise ValueError("deterministic renderer supports comparison and activity cards only")
    errors = validate_response_visual_intent(intent.model_dump(mode="json"))
    if errors:
        raise ValueError("unsafe response visual: " + "; ".join(errors))

    image = Image.new("RGB", (1200, 900), PALETTE["cream"])
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((34, 34, 1166, 866), 28, fill=PALETTE["white"], outline=PALETTE["ink"], width=6)
    draw.text((70, 68), "LOOK  +  COMPARE  +  WONDER", font=_font(24, bold=True), fill=PALETTE["teal"])
    title_lines = _wrapped(draw, intent.title, _font(48, bold=True), 1_040)[:2]
    y = 112
    for line in title_lines:
        draw.text((70, y), line, font=_font(48, bold=True), fill=PALETTE["ink"])
        y += 58

    panels = intent.panels
    gap = 22
    panel_left = 70
    panel_top = max(235, y + 8)
    panel_bottom = 690
    panel_width = (1_060 - gap * (len(panels) - 1)) // len(panels)
    fills = (PALETTE["mint"], PALETTE["sky"], "#FFF0E8", "#FFF6D9")
    for index, panel in enumerate(panels):
        left = panel_left + index * (panel_width + gap)
        right = left + panel_width
        draw.rounded_rectangle((left, panel_top, right, panel_bottom), 24, fill=fills[index], outline=PALETTE["ink"], width=4)
        icon_size = min(120, panel_width - 60)
        icon_left = left + (panel_width - icon_size) // 2
        _draw_icon(draw, (icon_left, panel_top + 38, icon_left + icon_size, panel_top + 38 + icon_size), panel.icon)
        label_font = _font(31, bold=True)
        label_lines = _wrapped(draw, panel.label, label_font, panel_width - 40)[:2]
        label_y = panel_top + 185
        for line in label_lines:
            draw.text((left + panel_width // 2, label_y), line, font=label_font, fill=PALETTE["ink"], anchor="ma")
            label_y += 39
        detail_font = _font(24)
        detail_lines = _wrapped(draw, panel.detail, detail_font, panel_width - 44)[:5]
        detail_y = label_y + 18
        for line in detail_lines:
            draw.text((left + 22, detail_y), line, font=detail_font, fill=PALETTE["ink"])
            detail_y += 33

    footer_y = 724
    caption_lines = _wrapped(draw, intent.caption, _font(25, bold=True), 900)[:3]
    for line in caption_lines:
        draw.text((70, footer_y), line, font=_font(25, bold=True), fill=PALETTE["ink"])
        footer_y += 34
    if intent.not_to_scale:
        draw.rounded_rectangle((918, 735, 1128, 790), 18, fill=PALETTE["gold"])
        draw.text((1023, 763), "NOT TO SCALE", font=_font(20, bold=True), fill=PALETTE["ink"], anchor="mm")

    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG", optimize=True)
    target.chmod(0o600)
    return target


def _validated_png(data: bytes) -> tuple[bytes, int, int, dict[str, Any]]:
    if not data or len(data) > MAX_IMAGE_BYTES * 2:
        raise ValueError("image bytes are empty or exceed the decode limit")
    try:
        source = Image.open(io.BytesIO(data))
        width, height = source.size
        if min(width, height) < MIN_DIMENSION or max(width, height) > MAX_DIMENSION:
            raise ValueError("visual dimensions are outside the supported range")
        source.verify()
        source = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise ValueError("visual is not a readable raster image") from exc
    width, height = source.size
    extrema = ImageChops.difference(source, Image.new("RGB", source.size, source.getpixel((0, 0)))).getbbox()
    if extrema is None:
        raise ValueError("visual appears blank")
    clean = io.BytesIO()
    source.save(clean, format="PNG", optimize=True)
    normalized = clean.getvalue()
    if len(normalized) > MAX_IMAGE_BYTES:
        raise ValueError("normalized visual exceeds the upload size limit")
    return normalized, width, height, {"format": "PNG", "metadata_stripped": True, "nonblank": True}


def _write_private_png(output_dir: str | Path, data: bytes) -> tuple[Path, int, int, dict[str, Any]]:
    normalized, width, height, validation = _validated_png(data)
    root = Path(output_dir).resolve() / "visuals"
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    target = root / f"visual_{uuid4().hex[:20]}.png"
    temporary = root / f".{target.name}.tmp"
    temporary.write_bytes(normalized)
    temporary.chmod(0o600)
    os.replace(temporary, target)
    target.chmod(0o600)
    return target, width, height, {**validation, "byte_count": len(normalized)}


def _visual_method(intent: VisualIntent, mode: str) -> str | None:
    errors = validate_response_visual_intent(intent.model_dump(mode="json"))
    if errors:
        raise ValueError("unsafe response visual: " + "; ".join(errors))
    if mode == "off":
        return None
    if intent.kind in {"comparison_cards", "activity_sequence"}:
        return "deterministic"
    if intent.kind == "decorative_illustration" and mode == "decorative":
        return "generative"
    return None


def enqueue_response_visual(
    db_path: str | Path,
    *,
    event_id: str,
    visual: dict[str, Any] | VisualIntent | None,
    mode: str,
) -> str | None:
    if visual is None:
        return None
    intent = visual if isinstance(visual, VisualIntent) else VisualIntent.model_validate(visual)
    method = _visual_method(intent, mode)
    if method is None:
        return None
    init_db(db_path)
    if method == "generative":
        _validate_decorative_privacy(db_path, event_id, intent)
    now = utcnow()
    job_id = f"visual_job_{uuid4().hex[:16]}"
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT id FROM visual_jobs WHERE event_id=?", (event_id,)).fetchone()
        if existing:
            return str(existing["id"])
        conn.execute(
            """INSERT INTO visual_jobs(
                 id,event_id,intent_json,method,status,attempts,available_at,created_at,updated_at
               ) VALUES(?,?,?,?,'queued',0,?,?,?)""",
            (job_id, event_id, intent.model_dump_json(), method, now, now, now),
        )
    return job_id


def synthetic_visual_intent() -> VisualIntent:
    return VisualIntent(
        kind="activity_sequence",
        purpose="sequence",
        knowledge_role="supportive",
        title="Visual connection works",
        pedagogical_value="Proves accessible image delivery without family or model data.",
        caption="A fixed test card: look, predict, then try.",
        alt_text="Three colorful cards labeled Look, Predict, and Try show that visual Slack delivery works.",
        panels=[
            {"label": "LOOK", "detail": "Notice one thing.", "icon": "look"},
            {"label": "PREDICT", "detail": "Guess what comes next.", "icon": "predict"},
            {"label": "TRY", "detail": "Check your idea.", "icon": "try"},
        ],
    )


def create_synthetic_visual_job(db_path: str | Path, event_id: str) -> str:
    init_db(db_path)
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO events(
                 id,type,child_id,text,source,metadata_json,created_at,status,processed_at,result_json
               ) VALUES(?,'visual_connection',NULL,'synthetic visual connection','system','{}',?,'completed',?,'{}')""",
            (event_id, now, now),
        )
    job_id = enqueue_response_visual(db_path, event_id=event_id, visual=synthetic_visual_intent(), mode="deterministic")
    if not job_id:  # pragma: no cover - fixed intent is always eligible
        raise RuntimeError("could not create synthetic visual job")
    return job_id


def _decorative_prompt(intent: VisualIntent) -> str:
    subject = re.sub(r"\s+", " ", str(intent.subject or "")).strip()
    if not subject:
        raise ValueError("decorative illustration subject is empty")
    return (
        "Create one playful, warm children's-book illustration for an elementary learner. "
        f"Scene: {subject}. No words, letters, numbers, captions, labels, diagrams, charts, measurements, "
        "or exact-count teaching. Do not depict a real child, real family, logo, worksheet, or copyrighted character. "
        "Use simple shapes, cheerful colors, clear focus, and generous empty space."
    )


def _validate_decorative_privacy(db_path: str | Path, event_id: str, intent: VisualIntent) -> None:
    """Reject known household identities and proper nouns before a scene reaches an image provider."""

    subject = str(intent.subject or "").casefold()
    with connect(db_path) as conn:
        event = conn.execute("SELECT text FROM events WHERE id=?", (event_id,)).fetchone()
        private_names = [
            str(row[0])
            for row in conn.execute("SELECT name FROM children UNION ALL SELECT display_name FROM parent_principals")
        ]
    for name in private_names:
        tokens = [token for token in re.findall(r"[a-z]+", name.casefold()) if len(token) >= 3]
        if any(re.search(rf"\b{re.escape(token)}\b", subject) for token in tokens):
            raise ValueError("decorative subject overlaps a private household identity")
    if event:
        proper_nouns = {
            token.casefold()
            for token in re.findall(r"\b[A-Z][a-z]{2,}\b", str(event["text"]))
            if token.casefold() not in PUBLIC_PROPER_NOUNS
            and token.casefold() not in {"can", "could", "does", "draw", "how", "is", "please", "show", "what", "why"}
        }
        if any(re.search(rf"\b{re.escape(token)}\b", subject) for token in proper_nouns):
            raise ValueError("decorative subject overlaps a proper name from the family request")


def process_visual_jobs(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    image_backend: ImageBackend | None = None,
    limit: int = 4,
) -> list[dict[str, str]]:
    init_db(db_path)
    current = datetime.now(UTC)
    now = current.isoformat()
    stale = (current - timedelta(minutes=STALE_VISUAL_JOB_MINUTES)).isoformat()
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """UPDATE visual_jobs SET status='queued',available_at=?,
                 last_error='recovered after interrupted local render',updated_at=?
               WHERE status='processing' AND method='deterministic' AND updated_at<?
                 AND attempts<?""",
            (now, now, stale, MAX_VISUAL_JOB_ATTEMPTS),
        )
        conn.execute(
            """UPDATE visual_jobs SET status='failed',
                 last_error=CASE WHEN method='generative'
                   THEN 'interrupted provider request was not retried automatically'
                   ELSE 'local render exhausted its recovery attempts' END,updated_at=?
               WHERE status='processing' AND updated_at<?
                 AND (method='generative' OR attempts>=?)""",
            (now, stale, MAX_VISUAL_JOB_ATTEMPTS),
        )
        conn.execute(
            """UPDATE slack_file_outbox SET status='expired',
                 last_error='image preparation outcome was ambiguous',updated_at=?
               WHERE status='waiting_asset' AND visual_job_id IN(
                 SELECT id FROM visual_jobs WHERE status='failed'
               )""",
            (now,),
        )
        rows = conn.execute(
            """SELECT * FROM visual_jobs WHERE status='queued' AND available_at<=?
                 AND attempts<? ORDER BY created_at LIMIT ?""",
            (now, MAX_VISUAL_JOB_ATTEMPTS, limit),
        ).fetchall()
    results: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        with connect(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            claimed = conn.execute(
                """UPDATE visual_jobs SET status='processing',attempts=attempts+1,updated_at=?
                   WHERE id=? AND status='queued'""",
                (utcnow(), row["id"]),
            )
        if claimed.rowcount != 1:
            continue
        path: Path | None = None
        try:
            intent = VisualIntent.model_validate(jload(row["intent_json"]))
            provenance: dict[str, Any] = {"family_data_sent": False}
            if row["method"] == "deterministic":
                temporary = Path(output_dir).resolve() / "visuals" / f".render-{uuid4().hex}.png"
                temporary.parent.mkdir(parents=True, exist_ok=True)
                temporary.parent.chmod(0o700)
                try:
                    render_deterministic_visual(intent, temporary)
                    raw_bytes = temporary.read_bytes()
                finally:
                    temporary.unlink(missing_ok=True)
                renderer = RENDERER_VERSION
            else:
                backend = image_backend or configured_image_backend()
                if backend is None:
                    raise RuntimeError("decorative image generation is not configured")
                generated = backend.generate(_decorative_prompt(intent))
                raw_bytes = generated.data
                renderer = f"{backend.name}:{generated.model}"
                provenance = {
                    "private_context_sent": False,
                    "response_topic_sent": True,
                    "provider": backend.name,
                    "model": generated.model,
                    "request_id": generated.request_id,
                    "prompt_policy": "decorative-minimized-v1",
                }
            path, width, height, validation = _write_private_png(output_dir, raw_bytes)
            asset_id = f"visual_asset_{uuid4().hex[:16]}"
            asset_hash = sha256(path.read_bytes()).hexdigest()
            created = utcnow()
            with connect(db_path) as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """INSERT INTO visual_assets(
                       id,job_id,event_id,method,trust_tier,renderer_version,path,filename,mime_type,
                       width,height,byte_count,sha256,title,caption,alt_text,provenance_json,validation_json,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        asset_id,
                        row["id"],
                        row["event_id"],
                        row["method"],
                        "A" if row["method"] == "generative" else "B",
                        renderer,
                        str(path),
                        path.name,
                        "image/png",
                        width,
                        height,
                        path.stat().st_size,
                        asset_hash,
                        intent.title,
                        intent.caption,
                        intent.alt_text,
                        jdump(provenance),
                        jdump(validation),
                        created,
                    ),
                )
                conn.execute(
                    "UPDATE visual_jobs SET status='completed',last_error=NULL,updated_at=? WHERE id=?",
                    (created, row["id"]),
                )
                conn.execute(
                    """UPDATE slack_file_outbox SET visual_asset_id=?,status='queued',updated_at=?
                       WHERE visual_job_id=? AND status='waiting_asset'""",
                    (asset_id, created, row["id"]),
                )
            results.append({"job_id": str(row["id"]), "status": "completed", "asset_id": asset_id})
        except Exception as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            with connect(db_path) as conn:
                conn.execute(
                    "UPDATE visual_jobs SET status='failed',last_error=?,updated_at=? WHERE id=?",
                    (f"{exc.__class__.__name__}: {str(exc)[:500]}", utcnow(), row["id"]),
                )
                conn.execute(
                    """UPDATE slack_file_outbox SET status='expired',last_error='visual preparation failed',updated_at=?
                       WHERE visual_job_id=? AND status='waiting_asset'""",
                    (utcnow(), row["id"]),
                )
            results.append({"job_id": str(row["id"]), "status": "failed"})
    return results
