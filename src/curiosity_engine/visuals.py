from __future__ import annotations

import io
import os
import re
from base64 import b64encode
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from .contracts import VisualIntent, VisualQAResult
from .db import connect, init_db, jdump, jload, utcnow
from .openai_image_backend import ImageBackend, configured_image_backend
from .trust import validate_response_visual_intent

RENDERER_VERSION = "response-card-v2"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MIN_DIMENSION = 600
MAX_DIMENSION = 2_048
MAX_VISUAL_JOB_ATTEMPTS = 3
STALE_VISUAL_JOB_MINUTES = 10
_AUTO_VISUAL_QA = object()

PUBLIC_PROPER_NOUNS = {"earth", "mars", "moon", "sun"}

SOCIAL_VISUAL_TERMS = {
    "bully",
    "classmate",
    "friend",
    "sibling",
    "teacher",
}

SOCIAL_INFERENCE_VERBS = {
    "avoid",
    "exclude",
    "hate",
    "ignore",
    "invite",
    "like",
    "mean",
    "tease",
}

ROBOT_ACTIVITY_VERBS = {
    "build",
    "building",
    "craft",
    "crafting",
    "create",
    "creating",
    "design",
    "designing",
    "draw",
    "drawing",
    "make",
    "making",
    "plan",
    "planning",
    "play",
    "playing",
    "pretend",
    "pretending",
    "program",
    "programming",
}

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


def render_deterministic_visual(
    intent: VisualIntent,
    output_path: str | Path,
    *,
    decorative_art: Image.Image | None = None,
) -> Path:
    if intent.kind not in {"comparison_cards", "activity_sequence"}:
        raise ValueError("deterministic renderer supports comparison and activity cards only")
    if not _is_curated_deterministic_intent(intent):
        raise ValueError("deterministic response cards must match a reviewed local template")
    errors = validate_response_visual_intent(intent.model_dump(mode="json"))
    if errors:
        raise ValueError("unsafe response visual: " + "; ".join(errors))

    image = Image.new("RGB", (1200, 900), PALETTE["cream"])
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((34, 34, 1166, 866), 28, fill=PALETTE["white"], outline=PALETTE["ink"], width=6)
    if decorative_art is not None:
        tile = ImageOps.fit(decorative_art.convert("RGB"), (450, 258), method=Image.Resampling.LANCZOS)
        mask = Image.new("L", tile.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, 449, 257), 28, fill=255)
        image.paste(tile, (70, 62), mask)
        draw.rounded_rectangle((88, 270, 274, 306), 12, fill=PALETTE["gold"])
        draw.text((181, 288), "IMAGINARY ART", font=_font(16, bold=True), fill=PALETTE["ink"], anchor="mm")
        draw.rounded_rectangle((542, 62, 1130, 320), 28, fill=PALETTE["sky"])
        draw.text((580, 92), "LOOK  +  COMPARE  +  WONDER", font=_font(21, bold=True), fill=PALETTE["teal"])
        title_font = _font(42, bold=True)
        title_lines = _wrapped(draw, intent.title, title_font, 510)[:3]
        y = 137
        for line in title_lines:
            draw.text((580, y), line, font=title_font, fill=PALETTE["ink"])
            y += 51
        panel_top = 350
        panel_bottom = 705
        footer_y = 738
    else:
        draw.text((70, 68), "LOOK  +  COMPARE  +  WONDER", font=_font(24, bold=True), fill=PALETTE["teal"])
        title_lines = _wrapped(draw, intent.title, _font(48, bold=True), 1_040)[:2]
        y = 112
        for line in title_lines:
            draw.text((70, y), line, font=_font(48, bold=True), fill=PALETTE["ink"])
            y += 58
        panel_top = max(235, y + 8)
        panel_bottom = 690
        footer_y = 724

    panels = intent.panels
    gap = 22
    panel_left = 70
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
        if not _is_curated_deterministic_intent(intent):
            raise ValueError("deterministic response cards must match a reviewed local template")
        if mode == "decorative" and intent.subject:
            return "generative"
        return "deterministic"
    if intent.kind == "decorative_illustration" and mode == "decorative":
        return "generative"
    return None


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _contains_any_word(text: str, words: set[str]) -> bool:
    return any(_contains_word(text, word) for word in words)


def _is_social_inference_question(question: str) -> bool:
    lowered = question.casefold()
    if _contains_any_word(lowered, SOCIAL_VISUAL_TERMS):
        return True
    if any(phrase in lowered for phrase in ("left me out", "mad at me", "mean to me")):
        return True
    has_person = _contains_any_word(lowered, {"he", "her", "him", "kid", "person", "she", "someone", "they"})
    return has_person and _contains_any_word(lowered, SOCIAL_INFERENCE_VERBS)


def _robot_comparison_intent() -> VisualIntent:
    return VisualIntent(
        kind="comparison_cards",
        purpose="compare",
        knowledge_role="supportive",
        title="What can BIGGEST mean for a robot?",
        pedagogical_value="Helps an early reader compare meanings of big without implying exact scale.",
        caption="One robot can be big in one way and not in every way.",
        alt_text=(
            "Playful imaginary robot art decorates three colorful cards comparing height, weight, and strength "
            "as different meanings of biggest. The cards are not to scale."
        ),
        subject="a playful group of whimsical imaginary robots posing together like friendly storybook explorers",
        panels=[
            {"label": "TALLEST", "detail": "Compare height.", "icon": "height"},
            {"label": "HEAVIEST", "detail": "Compare weight.", "icon": "weight"},
            {"label": "STRONGEST", "detail": "Compare the job it can do.", "icon": "strength"},
        ],
        not_to_scale=True,
    )


def _robot_activity_intent() -> VisualIntent:
    return VisualIntent(
        kind="activity_sequence",
        purpose="sequence",
        knowledge_role="supportive",
        title="Give your paper robot a plan",
        pedagogical_value="Turns an abstract command idea into three visible actions for an early reader.",
        caption="Point to one card at a time and follow it exactly.",
        alt_text=(
            "Playful imaginary robot art decorates three colorful cards showing Go with a curved arrow, Stop "
            "with a red stop shape, and Turn with a turning arrow for a pretend robot game."
        ),
        subject="a cheerful imaginary storybook robot ready to play a movement game in a colorful paper world",
        panels=[
            {"label": "GO", "detail": "Move forward.", "icon": "go"},
            {"label": "STOP", "detail": "Freeze in place.", "icon": "stop"},
            {"label": "TURN", "detail": "Change direction.", "icon": "turn"},
        ],
    )


def infer_safe_response_visual(question: str, output: dict[str, Any]) -> VisualIntent | None:
    """Provide a small deterministic fallback when a model omits or exceeds the visual boundary.

    The v0.1 fallback is intentionally tiny: one robot-size comparison and one pretend-robot
    command activity. New templates require code review and tests rather than broad keyword rules.
    """

    del output
    lowered = question.casefold()
    if _is_social_inference_question(question):
        return None

    is_robot = _contains_word(lowered, "robot") or _contains_word(lowered, "robots")
    is_size_question = _contains_any_word(lowered, {"biggest", "largest"}) or "how big" in lowered
    if is_robot and is_size_question:
        return _robot_comparison_intent()
    if is_robot and _contains_any_word(lowered, ROBOT_ACTIVITY_VERBS):
        return _robot_activity_intent()
    return None


def normalize_response_visual(question: str, output: dict[str, Any]) -> VisualIntent | None:
    """Keep a safe decorative proposal or select one reviewed deterministic template."""

    if _is_social_inference_question(question):
        return None
    proposed = output.get("visual")
    if proposed is not None:
        try:
            intent = proposed if isinstance(proposed, VisualIntent) else VisualIntent.model_validate(proposed)
        except (TypeError, ValueError):
            intent = None
        if (
            intent is not None
            and intent.kind == "decorative_illustration"
            and not validate_response_visual_intent(intent.model_dump(mode="json"))
        ):
            return intent
    return infer_safe_response_visual(question, output)


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


def _is_curated_deterministic_intent(intent: VisualIntent) -> bool:
    """Require exact reviewed content at the final deterministic rendering boundary."""

    return intent in (synthetic_visual_intent(), _robot_comparison_intent(), _robot_activity_intent())


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
    hybrid_boundary = (
        " This art is decorative beside a code-rendered learning card. Keep every robot similar in visual size, "
        "avoid factual comparisons or real robot designs, and leave the teaching to the separate card."
        if intent.kind in {"comparison_cards", "activity_sequence"}
        else ""
    )
    return (
        "Create one playful, warm children's-book illustration for an elementary learner. "
        f"Scene: {subject}. No words, letters, numbers, captions, labels, diagrams, charts, measurements, "
        "or exact-count teaching. Do not depict a real child, real family, logo, worksheet, or copyrighted character. "
        "Use a lively cut-paper collage style, expressive faces, cheerful colors, tactile shapes, clear focus, "
        "and generous empty space."
        + hybrid_boundary
    )


def _configured_visual_qa_backend() -> Any | None:
    # Lazy imports avoid coupling the renderer to runtime initialization.
    from .config import AppConfig
    from .runtime import configured_backend

    return configured_backend(AppConfig.load(), role="visual_qa")


def _review_generated_art(intent: VisualIntent, png_bytes: bytes, backend: Any | None) -> VisualQAResult:
    if backend is None:
        raise RuntimeError("generated art requires a configured visual-QA route")
    result = backend.complete(
        role="visual_qa",
        system=(
            "Inspect this generated decorative region before it can reach an elementary-age child. Fail if it "
            "contains words, letters, numbers, logos, recognizable copyrighted characters, frightening or unsafe "
            "imagery, real-person likenesses, factual labels, diagrams, measurements, or an apparent real robot "
            "design. For a robot learning card, also fail if robot size differences imply a factual comparison."
        ),
        payload={
            "visual_intent": {
                "kind": intent.kind,
                "knowledge_role": intent.knowledge_role,
                "subject": intent.subject,
            },
            "image_data_urls": ["data:image/png;base64," + b64encode(png_bytes).decode("ascii")],
        },
        response_model=VisualQAResult,
    )
    review = VisualQAResult.model_validate(result)
    if review.verdict != "pass":
        raise ValueError("generated art did not pass visual QA")
    return review


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
    visual_qa_backend: Any = _AUTO_VISUAL_QA,
    job_id: str | None = None,
    limit: int = 4,
) -> list[dict[str, str]]:
    init_db(db_path)
    resolved_visual_qa_backend = visual_qa_backend
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
                 AND (? IS NULL OR id=?) AND attempts<? ORDER BY created_at LIMIT ?""",
            (now, job_id, job_id, MAX_VISUAL_JOB_ATTEMPTS, limit),
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
            asset_method = str(row["method"])
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
                hybrid = intent.kind in {"comparison_cards", "activity_sequence"}
                try:
                    backend = image_backend or configured_image_backend()
                    if backend is None:
                        raise RuntimeError("decorative image generation is not configured")
                    generated = backend.generate(_decorative_prompt(intent))
                    normalized_art, _art_width, _art_height, _art_validation = _validated_png(generated.data)
                    if resolved_visual_qa_backend is _AUTO_VISUAL_QA:
                        resolved_visual_qa_backend = _configured_visual_qa_backend()
                    reviewer = resolved_visual_qa_backend
                    qa = _review_generated_art(intent, normalized_art, reviewer)
                    if hybrid:
                        art = Image.open(io.BytesIO(normalized_art)).convert("RGB")
                        temporary = Path(output_dir).resolve() / "visuals" / f".render-{uuid4().hex}.png"
                        temporary.parent.mkdir(parents=True, exist_ok=True)
                        temporary.parent.chmod(0o700)
                        try:
                            render_deterministic_visual(intent, temporary, decorative_art=art)
                            raw_bytes = temporary.read_bytes()
                        finally:
                            temporary.unlink(missing_ok=True)
                        renderer = f"{RENDERER_VERSION}+{backend.name}:{generated.model}"
                        prompt_policy = "hybrid-decorative-minimized-v2"
                    else:
                        raw_bytes = normalized_art
                        renderer = f"{backend.name}:{generated.model}"
                        prompt_policy = "decorative-minimized-v2"
                    provenance = {
                        "private_context_sent": False,
                        "response_topic_sent": True,
                        "provider": backend.name,
                        "model": generated.model,
                        "request_id": generated.request_id,
                        "prompt_policy": prompt_policy,
                        "generated_art_embedded": hybrid,
                        "visual_qa": qa.verdict,
                        "visual_qa_provider": getattr(reviewer, "name", "unknown"),
                        "visual_qa_model": getattr(reviewer, "model", None),
                    }
                except Exception:
                    if not hybrid:
                        raise
                    temporary = Path(output_dir).resolve() / "visuals" / f".render-{uuid4().hex}.png"
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    temporary.parent.chmod(0o700)
                    try:
                        render_deterministic_visual(intent, temporary)
                        raw_bytes = temporary.read_bytes()
                    finally:
                        temporary.unlink(missing_ok=True)
                    asset_method = "deterministic"
                    renderer = RENDERER_VERSION
                    provenance = {
                        "private_context_sent": False,
                        "response_topic_sent": True,
                        "fallback_from": "generative",
                        "fallback_reason": "generated art was unavailable or did not pass visual QA",
                        "generated_art_embedded": False,
                        "visual_qa": "not_passed",
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
                        asset_method,
                        (
                            "A"
                            if asset_method == "generative" and intent.kind == "decorative_illustration"
                            else "B"
                        ),
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
