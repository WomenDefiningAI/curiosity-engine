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

from .config import repository_root
from .db import SCHEMA_VERSION
from .interaction import onboarding_status
from .openai_backend import load_model_settings, model_key_is_configured
from .transports.slack import load_slack_tokens


def _check(name: str, status: str, detail: str, *, required: bool) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "required": required}


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
        model_settings = load_model_settings()
        backend = model_settings["CURIOSITY_BACKEND"].casefold()
        if backend == "openai" and model_key_is_configured(model_settings["OPENAI_API_KEY"]):
            model_ready = True
            response_mode = "hosted_model"
            model_status = "pass"
            model_detail = "hosted reasoning configured; bounded context is disclosed to the provider"
        elif backend == "deterministic":
            model_status = "demo_mode"
            model_detail = "offline canned responses only; configure a model for question-specific answers"
        else:
            model_status = "not_configured"
            model_detail = "model backend is enabled but its credential is absent or malformed"
    except PermissionError:
        model_status = "not_configured"
        model_detail = "model credential file permissions are too broad; values were not read or displayed"
    checks.append(_check("reasoning_provider", model_status, model_detail, required=False))
    config_ok = all((root / "configs" / name).is_file() for name in ("production.json", "reasoning-policy.json"))
    checks.append(
        _check("public_config", "pass" if config_ok else "fail", "versioned public configuration", required=True)
    )
    ignored = False
    if (root / ".git").is_dir() and shutil.which("git"):
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "private/example"],
            cwd=root,
            check=False,
            capture_output=True,
            timeout=10,
        )
        ignored = proc.returncode == 0
    else:
        ignored = "private/" in (root / ".gitignore").read_text(encoding="utf-8", errors="replace")
    checks.append(
        _check("private_git_boundary", "pass" if ignored else "fail", "private/ is ignored", required=True)
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
    return {
        "status": "pass" if required_pass else "fail",
        "core_ready": required_pass,
        "slack_ready": slack_ready,
        "answer_ready": slack_ready and model_ready,
        "response_mode": response_mode,
        "checks": checks,
        "privacy_note": "No token values, child names, questions, or licensed excerpts are included in this report.",
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
