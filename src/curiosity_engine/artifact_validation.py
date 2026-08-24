from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .trust import validate_artifact_trust

MIN_KID_FONT_PX = 16


def validate_artifact_spec(spec: dict[str, Any]) -> list[str]:
    errors = validate_artifact_trust(spec)
    if not spec.get("title"):
        errors.append("title is required")
    if spec.get("artifact_type") not in {
        "wonder_page",
        "reference_page",
        "mini_poster",
        "challenge_card",
        "case_file",
        "field_guide",
        "cut_and_build",
        "mini_book",
        "follow_the_thread",
        "worksheet",
    }:
        errors.append("artifact_type is missing or unsupported")
    if spec.get("target_age") is None and not spec.get("target_grade"):
        errors.append("target_age or target_grade is required")
    return errors


def validate_rendered_file(path: str | Path) -> list[str]:
    p = Path(path)
    errors: list[str] = []
    if not p.exists():
        return ["rendered artifact does not exist"]
    if p.stat().st_size < 100:
        errors.append("rendered artifact is unexpectedly small")
    if p.suffix.lower() == ".html":
        text = p.read_text(encoding="utf-8", errors="replace")
        if "<html" not in text.lower() or "</html>" not in text.lower():
            errors.append("HTML artifact is malformed")
        if "{{" in text or "}}" in text:
            errors.append("unresolved template markers found")
    elif p.suffix.lower() == ".pdf":
        if p.read_bytes()[:5] != b"%PDF-":
            errors.append("PDF signature is invalid")
        if shutil.which("pdfinfo") is None:
            errors.append("pdfinfo is required for page-count validation")
        else:
            proc = subprocess.run(["pdfinfo", str(p)], capture_output=True, text=True, check=False, timeout=20)
            if proc.returncode:
                errors.append(proc.stderr.strip() or "pdfinfo could not inspect PDF")
            else:
                pages = next(
                    (line.split(":", 1)[1].strip() for line in proc.stdout.splitlines() if line.startswith("Pages:")),
                    None,
                )
                if pages != "1":
                    errors.append(f"MVP artifacts must be exactly one page; got {pages or 'unknown'}")
    return errors


def render_pdf_preview(pdf_path: str | Path, output_path: str | Path, dpi: int = 144) -> Path:
    source = Path(pdf_path).resolve()
    target = Path(output_path).resolve()
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm is required for artifact preview and visual QA")
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = target.with_suffix("")
    proc = subprocess.run(
        ["pdftoppm", "-f", "1", "-singlefile", "-png", "-r", str(dpi), str(source), str(prefix)],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or "could not render PDF preview")
    generated = prefix.with_suffix(".png")
    if generated != target:
        generated.replace(target)
    return target


def visual_qa_request(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_role": "visual_qa",
        "task": "Inspect the rendered child-facing artifact for factual inconsistencies, malformed imagery, incorrect labels, ambiguous arrows, bad counts, clipping, illegible text, age-inappropriate complexity, or anything likely to confuse a child. Return PASS or FAIL with concise reasons.",
        "trust_tier": spec.get("trust_tier"),
        "artifact_type": spec.get("artifact_type"),
    }
