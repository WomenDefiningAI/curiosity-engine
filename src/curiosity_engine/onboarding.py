from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from .brain_config import brain_config_fingerprint, brain_status
from .config import (
    AppConfig,
    ConfigurationError,
    configuration_root,
    family_home,
    private_root,
    repository_root,
)
from .db import SCHEMA_VERSION, connect, init_db, jload, utcnow
from .interaction import onboarding_checkpoint, onboarding_status, record_onboarding_checkpoint
from .openai_image_backend import configured_image_backend
from .runtime import configured_backend
from .transports.slack import load_slack_tokens
from .visuals import enqueue_response_visual, infer_safe_response_visual, process_visual_jobs


class BrainProbeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    marker: str
    count: int


IMAGE_PROBE_VERSION = "synthetic-hybrid-image-v4"


def _check(name: str, status: str, detail: str, *, required: bool) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "required": required}


def _installed_layout_is_safe(root: Path, public_config_root: Path) -> bool:
    installed_root = (Path(sys.prefix) / "share" / "curiosity-engine").resolve()
    private = private_root().resolve()
    home = family_home().resolve()
    return (
        public_config_root.resolve() == installed_root
        and not (root / ".git").exists()
        and private.is_relative_to(home)
        and not private.is_relative_to(installed_root)
    )


def _answer_quality_summary(db: Path, *, limit: int = 20) -> dict[str, Any]:
    """Report redacted outcome rates without reading family question or answer text."""

    summary = {
        "status": "not_enough_data",
        "terminal_questions": 0,
        "completed": 0,
        "rejected": 0,
        "failed": 0,
        "rejection_rate": 0.0,
        "unsuccessful_rate": 0.0,
        "window": limit,
    }
    if not db.is_file():
        return summary
    try:
        with connect(db) as conn:
            rows = conn.execute(
                """SELECT status FROM events WHERE type='child_question'
                     AND status IN ('completed','rejected','failed')
                     ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    except sqlite3.DatabaseError:
        return summary
    statuses = [str(row["status"]) for row in rows]
    completed = statuses.count("completed")
    rejected = statuses.count("rejected")
    failed = statuses.count("failed")
    terminal_questions = len(statuses)
    quality_decisions = completed + rejected
    rejection_rate = rejected / quality_decisions if quality_decisions else 0.0
    unsuccessful_rate = (rejected + failed) / terminal_questions if terminal_questions else 0.0
    if terminal_questions >= 5:
        status = "attention" if unsuccessful_rate >= 0.2 else "healthy"
    else:
        status = "not_enough_data"
    return {
        **summary,
        "status": status,
        "terminal_questions": terminal_questions,
        "completed": completed,
        "rejected": rejected,
        "failed": failed,
        "rejection_rate": round(rejection_rate, 3),
        "unsuccessful_rate": round(unsuccessful_rate, 3),
    }


def doctor(db_path: str | Path) -> dict[str, Any]:
    root = repository_root()
    db = Path(db_path).resolve()
    checks: list[dict[str, Any]] = []
    python_ok = sys.version_info >= (3, 11)
    checks.append(
        _check(
            "python",
            "pass" if python_ok else "fail",
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            required=True,
        )
    )
    model_ready = False
    response_mode = "offline_demo"
    try:
        brain = brain_status()
        model_ready = bool(brain["configured"])
        if model_ready:
            response_mode = "api_brain"
            model_status = "pass"
            model_detail = "API brain stack configured; live verification is reported separately"
        elif brain["runtime"] == "coding_agent_attended":
            response_mode = "coding_agent_attended"
            model_status = "experimental"
            model_detail = "attended coding-agent operation is not a reliable Slack runtime"
        elif brain["runtime"] == "deterministic":
            model_status = "demo_mode"
            model_detail = "offline canned responses only; configure the brain stack for tailored answers"
        else:
            model_status = "not_configured"
            model_detail = "; ".join(brain["blockers"][:4])
    except PermissionError:
        brain = {"configured": False, "multimodal_stack_configured": False, "blockers": ["private file permissions"]}
        model_status = "not_configured"
        model_detail = "model credential file permissions are too broad; values were not read or displayed"
    checks.append(_check("reasoning_provider", model_status, model_detail, required=False))
    try:
        public_config_root = configuration_root()
    except ConfigurationError:
        public_config_root = root
    config_ok = all(
        (public_config_root / "configs" / name).is_file()
        for name in ("production.json", "reasoning-policy.json")
    )
    checks.append(
        _check("public_config", "pass" if config_ok else "fail", "versioned public configuration", required=True)
    )
    ignored = False
    boundary_detail = "run inside the cloned repository with private/ ignored"
    if (root / ".git").is_dir() and shutil.which("git"):
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "private/example"],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=10,
        )
        ignored = proc.returncode == 0
        if ignored:
            boundary_detail = "private/ is ignored"
    elif _installed_layout_is_safe(root, public_config_root):
        ignored = True
        boundary_detail = "installed runtime keeps private family state outside the package"
    else:
        ignore_file = root / ".gitignore"
        ignored = ignore_file.is_file() and "private/" in ignore_file.read_text(
            encoding="utf-8", errors="replace"
        )
        if ignored:
            boundary_detail = "private/ is ignored"
    checks.append(
        _check(
            "private_git_boundary",
            "pass" if ignored else "fail",
            boundary_detail,
            required=True,
        )
    )
    db_detail = "not initialized"
    db_ok = True
    db_version: int | None = None
    integrity: str | None = None
    if db.is_file():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            db_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        finally:
            conn.close()
        db_ok = db_version <= SCHEMA_VERSION and integrity == "ok"
        db_detail = f"schema v{db_version}; integrity {integrity}"
    checks.append(_check("private_database", "pass" if db_ok else "fail", db_detail, required=True))
    slack_extra = importlib.util.find_spec("slack_bolt") is not None
    checks.append(
        _check(
            "slack_dependency",
            "pass" if slack_extra else "not_configured",
            "slack-bolt installed" if slack_extra else "install the optional slack dependency",
            required=False,
        )
    )
    try:
        tokens = load_slack_tokens()
        token_shapes = bool(tokens["SLACK_APP_TOKEN"].startswith("xapp-")) and bool(
            tokens["SLACK_BOT_TOKEN"].startswith("xoxb-")
        )
        token_detail = (
            "both token variables have expected prefixes"
            if token_shapes
            else "tokens are absent or malformed; values were not displayed"
        )
    except PermissionError:
        token_shapes = False
        token_detail = "token file permissions are too broad; values were not read or displayed"
    checks.append(
        _check(
            "slack_tokens",
            "pass" if token_shapes else "not_configured",
            token_detail,
            required=False,
        )
    )
    poppler = bool(shutil.which("pdfinfo") and shutil.which("pdftoppm"))
    checks.append(
        _check(
            "paper_pdf_tools",
            "pass" if poppler else "not_configured",
            "Poppler available" if poppler else "optional: install Poppler for PDF preview/QA",
            required=False,
        )
    )
    configured = False
    bindings = 0
    state: dict[str, Any] = {
        "checkpoints": {},
        "family_lens_configured": False,
        "quality_review_accepted": False,
    }
    if db.is_file() and db_version == SCHEMA_VERSION:
        state = onboarding_status(db)
        configured = bool(state["configured"])
        bindings = len([item for item in state["bindings"] if item["status"] == "active"])
    checks.append(
        _check(
            "household_setup",
            "pass" if configured else "not_configured",
            "household owner configured" if configured else "run curiosity setup",
            required=False,
        )
    )
    checks.append(
        _check(
            "slack_pairing",
            "pass" if bindings else "not_configured",
            f"{bindings} active binding(s)" if bindings else "no Slack parent/channel binding yet",
            required=False,
        )
    )
    required_pass = all(item["status"] == "pass" for item in checks if item["required"])
    slack_ready = required_pass and slack_extra and token_shapes and configured and bindings > 0
    checkpoints = state.get("checkpoints") or {}
    transport_verified = slack_ready and (checkpoints.get("transport_verified") or {}).get("status") == "pass"
    brain_check = onboarding_checkpoint(db, "brain_verified") if db.is_file() and db_version == SCHEMA_VERSION else None
    current_brain_hash = brain_config_fingerprint()
    brain_verified = bool(
        model_ready
        and brain_check
        and brain_check["status"] == "pass"
        and (brain_check.get("evidence") or {}).get("config_hash") == current_brain_hash
    )
    family_lens_ready = bool(state.get("family_lens_configured"))
    visual_mode = str((state.get("settings") or {}).get("visual_mode") or "deterministic")
    visual_check = (
        onboarding_checkpoint(db, "visual_delivery_verified")
        if db.is_file() and db_version == SCHEMA_VERSION
        else None
    )
    visual_delivery_verified = bool(
        slack_ready and visual_check and visual_check["status"] == "pass"
    )
    image_check = (
        onboarding_checkpoint(db, "image_generation_verified")
        if db.is_file() and db_version == SCHEMA_VERSION
        else None
    )
    image_generation_verified = bool(
        brain.get("image_generation_runtime_ready")
        and image_check
        and image_check["status"] == "pass"
        and (image_check.get("evidence") or {}).get("probe") == IMAGE_PROBE_VERSION
        and (image_check.get("evidence") or {}).get("config_hash") == current_brain_hash
    )
    deterministic_visual_ready = importlib.util.find_spec("PIL") is not None
    visual_ready = bool(
        visual_mode == "off"
        or (
            deterministic_visual_ready
            and visual_delivery_verified
            and (visual_mode != "decorative" or image_generation_verified)
        )
    )
    quality_review_pending = brain_verified and family_lens_ready and not bool(state.get("quality_review_accepted"))
    end_to_end_ready = (
        transport_verified
        and visual_ready
        and brain_verified
        and family_lens_ready
        and bool(state.get("quality_review_accepted"))
    )
    answer_quality = _answer_quality_summary(db)
    if not required_pass:
        next_action = "fix required local checks"
    elif not configured:
        next_action = "run curiosity setup and add a child"
    elif not slack_ready:
        next_action = "finish Slack installation, credentials, and pairing"
    elif not transport_verified:
        next_action = "send `connection` in the paired Slack conversation"
    elif visual_mode != "off" and not visual_delivery_verified:
        next_action = "send `visual connection` in the paired Slack conversation after installing files:write"
    elif not model_ready:
        next_action = "run curiosity brain configure, then paste provider credentials privately"
    elif not brain_verified:
        next_action = "run curiosity brain test --live"
    elif visual_mode == "decorative" and not image_generation_verified:
        next_action = "run curiosity visual test --live"
    elif not family_lens_ready:
        next_action = "run curiosity family-lens configure"
    elif not end_to_end_ready:
        next_action = "ask one real Slack question and record curiosity onboard review"
    else:
        next_action = "run the Slack connector for everyday use"
    return {
        "status": "pass" if required_pass else "fail",
        "core_ready": required_pass,
        "slack_ready": slack_ready,
        "transport_verified": transport_verified,
        "brain_configured": model_ready,
        "brain_verified": brain_verified,
        "multimodal_stack_configured": bool(brain.get("multimodal_stack_configured")),
        "visual_mode": visual_mode,
        "deterministic_visual_ready": deterministic_visual_ready,
        "visual_delivery_verified": visual_delivery_verified,
        "image_generation_configured": bool(brain.get("image_generation_configured")),
        "image_generation_verified": image_generation_verified,
        "visual_ready": visual_ready,
        "family_lens_ready": family_lens_ready,
        "quality_review_pending": quality_review_pending,
        "end_to_end_ready": end_to_end_ready,
        "answer_ready": slack_ready and brain_verified,
        "answer_quality": answer_quality,
        "response_mode": response_mode,
        "next_action": next_action,
        "checks": checks,
        "privacy_note": "No token values, child names, questions, or licensed excerpts are included in this report.",
    }


def run_brain_probe(db_path: str | Path, *, live: bool) -> dict[str, Any]:
    if not live:
        raise ValueError("brain probe makes a billable network request; pass --live after reviewing disclosure")
    app = AppConfig.load()
    backend = configured_backend(app, role="reasoning")
    if backend is None:
        raise RuntimeError("an API reasoning provider is not configured")
    try:
        result = backend.complete(
            role="reasoning",
            system=(
                "This is a synthetic Curiosity Engine connectivity test. Return only the requested schema. "
                "No family, child, Slack, or private-resource data is present."
            ),
            payload={
                "probe": "curiosity-engine-synthetic-v1",
                "instruction": "Return marker ready and count 3.",
                "policy": {"allowed_tools": []},
            },
            response_model=BrainProbeOutput,
        )
        if result != {"marker": "ready", "count": 3}:
            raise RuntimeError("provider returned a valid schema but failed the fixed semantic probe")
    except Exception:
        record_onboarding_checkpoint(db_path, "brain_verified", status="fail", evidence={"probe": "synthetic-v1"})
        raise
    config_hash = brain_config_fingerprint()
    record_onboarding_checkpoint(
        db_path,
        "brain_verified",
        status="pass",
        evidence={
            "probe": "synthetic-v1",
            "provider": backend.name,
            "model": backend.model,
            "config_hash": config_hash,
            "family_data_sent": False,
        },
    )
    return {
        "status": "pass",
        "provider": backend.name,
        "model": backend.model,
        "family_data_sent": False,
        "note": "This verifies structured reasoning only; vision/OCR/image generation still require the eval checklist.",
    }


def run_image_generation_probe(
    db_path: str | Path,
    output_dir: str | Path,
    *,
    live: bool,
) -> dict[str, Any]:
    if not live:
        raise ValueError("image probe makes a billable network request; pass --live after reviewing disclosure")
    backend = configured_image_backend()
    if backend is None:
        raise RuntimeError("an OpenAI image-generation route is not configured")
    config_hash = brain_config_fingerprint()
    init_db(db_path)
    event_id = f"evt_image_probe_v4_{config_hash}_{uuid4().hex[:8]}"
    now = utcnow()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT OR IGNORE INTO events(
                 id,type,child_id,text,source,metadata_json,created_at,status,processed_at,result_json
               ) VALUES(?,'image_generation_probe',NULL,'synthetic hybrid robot card','system','{}',
                        ?,'completed',?,'{}')""",
            (event_id, now, now),
        )
    visual = infer_safe_response_visual("Can we make an imaginary paper robot?", {})
    if visual is None:  # pragma: no cover - fixed public probe always selects the reviewed template
        raise RuntimeError("could not build the synthetic hybrid visual intent")
    job_id = enqueue_response_visual(db_path, event_id=event_id, visual=visual, mode="decorative")
    if not job_id:
        raise RuntimeError("could not create the image-generation probe")
    process_visual_jobs(db_path, output_dir, image_backend=backend, job_id=job_id, limit=1)
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT j.status,j.last_error,a.id AS asset_id,a.sha256,a.method,a.provenance_json
               FROM visual_jobs j LEFT JOIN visual_assets a ON a.job_id=j.id WHERE j.id=?""",
            (job_id,),
        ).fetchone()
    provenance = jload(row["provenance_json"]) if row and row["provenance_json"] else {}
    if (
        not row
        or row["status"] != "completed"
        or not row["asset_id"]
        or row["method"] != "generative"
        or not provenance.get("generated_art_embedded")
        or provenance.get("visual_qa") != "pass"
    ):
        record_onboarding_checkpoint(
            db_path,
            "image_generation_verified",
            status="fail",
            evidence={"probe": IMAGE_PROBE_VERSION, "config_hash": config_hash, "family_data_sent": False},
        )
        raise RuntimeError("image-generation probe failed; run curiosity doctor for redacted status")
    record_onboarding_checkpoint(
        db_path,
        "image_generation_verified",
        status="pass",
        evidence={
            "probe": IMAGE_PROBE_VERSION,
            "provider": backend.name,
            "model": backend.model,
            "config_hash": config_hash,
            "family_data_sent": False,
        },
    )
    return {
        "status": "pass",
        "provider": backend.name,
        "model": backend.model,
        "asset_id": row["asset_id"],
        "family_data_sent": False,
        "note": "The paid probe used one fixed synthetic hybrid card and stored it only in private output.",
    }


def write_setup_report(db_path: str | Path, report: dict[str, Any] | None = None) -> Path:
    root = repository_root()
    setup_dir = root / "private" / "setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    setup_dir.chmod(0o700)
    target = setup_dir / "status.json"
    payload = report or doctor(db_path)
    payload = {**payload, "generated_at": datetime.now(UTC).isoformat()}
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    target.chmod(0o600)
    return target
