from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from .artifacts import file_sha256
from .db import connect, init_db, jdump, utcnow


def _printer_name(value: str | None) -> str | None:
    if value is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
        raise ValueError("printer name contains unsupported characters")
    return value


def print_file(
    path: str | Path,
    printer: str | None = None,
    dry_run: bool = True,
    *,
    allowed_root: str | Path | None = None,
) -> dict:
    """Low-level preview helper. Live printing requires a bounded root."""

    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    if target.suffix.casefold() != ".pdf":
        raise ValueError("only validated PDF artifacts can be printed")
    if not dry_run:
        if allowed_root is None:
            raise PermissionError("live printing requires a registered artifact or an explicit allowed_root")
        try:
            target.relative_to(Path(allowed_root).resolve())
        except ValueError as exc:
            raise PermissionError("print path is outside the allowed artifact directory") from exc
    command = ["lp"]
    printer = _printer_name(printer)
    if printer:
        command += ["-d", printer]
    command += [str(target)]
    if dry_run:
        return {"status": "dry-run", "command": command}
    if shutil.which("lp") is None:
        raise RuntimeError("'lp' command not found. Configure CUPS or use dry-run.")
    proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "printing failed")
    return {"status": "sent", "stdout": proc.stdout.strip(), "command": command}


def approve_artifact(
    db_path: str | Path,
    artifact_id: str,
    *,
    actor: str = "parent",
    decision: str = "approved",
    note: str | None = None,
) -> dict:
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be approved or rejected")
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT path,sha256 FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row:
            raise KeyError(artifact_id)
        digest = file_sha256(row["path"])
        if not row["sha256"] or digest != row["sha256"]:
            conn.execute("UPDATE artifacts SET approval_status='stale' WHERE id=?", (artifact_id,))
            raise ValueError("artifact bytes changed after validation; regenerate before approval")
        approval_id = f"apr_{uuid4().hex[:16]}"
        now = utcnow()
        conn.execute(
            """INSERT INTO approvals(id,artifact_id,artifact_sha256,decision,actor,created_at,note)
               VALUES(?,?,?,?,?,?,?)""",
            (approval_id, artifact_id, digest, decision, actor, now, note),
        )
        conn.execute("UPDATE artifacts SET approval_status=? WHERE id=?", (decision, artifact_id))
    return {"approval_id": approval_id, "artifact_id": artifact_id, "artifact_sha256": digest, "decision": decision}


def print_artifact(
    db_path: str | Path,
    artifact_id: str,
    approval_id: str,
    *,
    printer: str | None = None,
    send: bool = False,
) -> dict:
    """Print only the exact validated bytes approved by a parent; dry-run unless send=True."""

    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute(
            """SELECT a.path,a.sha256,a.validated_at,a.approval_status,p.artifact_sha256,p.decision
               FROM artifacts a JOIN approvals p ON p.artifact_id=a.id
               WHERE a.id=? AND p.id=?""",
            (artifact_id, approval_id),
        ).fetchone()
        if not row:
            raise ValueError("approval does not match artifact")
        if not row["validated_at"] or row["decision"] != "approved" or row["approval_status"] != "approved":
            raise PermissionError("artifact needs a current explicit approval")
        digest = file_sha256(row["path"])
        if digest != row["sha256"] or digest != row["artifact_sha256"]:
            conn.execute("UPDATE artifacts SET approval_status='stale' WHERE id=?", (artifact_id,))
            raise PermissionError("artifact changed after approval")
        target = Path(row["path"]).resolve()
    command_result: dict
    error: Exception | None = None
    try:
        command_result = print_file(target, printer, dry_run=not send, allowed_root=target.parent)
    except Exception as exc:
        command_result = {"status": "failed", "error": repr(exc)}
        error = exc
    attempt_id = f"print_{uuid4().hex[:16]}"
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO print_attempts(id,artifact_id,artifact_sha256,approval_id,printer,status,command_json,result_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                attempt_id,
                artifact_id,
                digest,
                approval_id,
                _printer_name(printer),
                command_result["status"],
                jdump(command_result.get("command", [])),
                jdump(command_result),
                utcnow(),
            ),
        )
    if error:
        raise error
    return {"attempt_id": attempt_id, "artifact_id": artifact_id, **command_result}
