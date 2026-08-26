from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import stat
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .config import family_home, repository_root

BACKUP_FORMAT = "curiosity-family-backup"
BACKUP_FORMAT_VERSION = 1
BACKUP_ROOT_MARKER = ".curiosity-backup-root.json"
MANIFEST_NAME = "manifest.json"
SNAPSHOT_PREFIX = "curiosity-backup-"
SETUP_ALLOWLIST = ("brain.json", "status.json")
SECRET_MARKERS = (
    b"OPENAI_API_KEY",
    b"ANTHROPIC_API_KEY",
    b"OPENROUTER_API_KEY",
    b"SLACK_APP_TOKEN",
    b"SLACK_BOT_TOKEN",
    b"xapp-",
    b"xoxb-",
    b"sk-ant-",
    b"sk-or-",
)


class BackupError(RuntimeError):
    """A backup could not be created, verified, or restored safely."""


def default_backup_root(repo_root: str | Path | None = None) -> Path:
    configured = os.environ.get("CURIOSITY_BACKUP_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(repo_root).resolve() if repo_root else family_home()
    return root.parent / f"{root.name}-family-backups"


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _ensure_external_backup_root(path: str | Path, repo_root: str | Path, *, create: bool) -> Path:
    repo = Path(repo_root).resolve()
    raw_candidate = Path(path).expanduser()
    if raw_candidate.is_symlink():
        raise BackupError("backup destination may not be a symbolic link")
    candidate = raw_candidate.resolve()
    if _is_within(candidate, repo):
        raise BackupError("backup destination must be outside the repository")
    existed = candidate.exists()
    if existed and not candidate.is_dir():
        raise BackupError("backup destination must be a directory")
    if not existed:
        if not create:
            return candidate
        candidate.mkdir(parents=True, mode=0o700)
    if create:
        candidate.chmod(0o700)
    marker = candidate / BACKUP_ROOT_MARKER
    expected = {"format": BACKUP_FORMAT, "format_version": BACKUP_FORMAT_VERSION}
    if marker.exists():
        if marker.is_symlink():
            raise BackupError("backup root marker may not be a symbolic link")
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackupError("backup root marker is unreadable") from exc
        if current != expected:
            raise BackupError("destination is not a compatible Curiosity Engine backup root")
    elif existed and any(candidate.iterdir()):
        raise BackupError("destination is not an initialized Curiosity Engine backup root")
    elif create:
        _write_json(marker, expected)
    return candidate


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _owner_only_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _create_owner_only_parents(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in missing:
        created.chmod(0o700)


def _harden_tree(root: Path) -> None:
    root.chmod(0o700)
    for item in root.rglob("*"):
        item.chmod(0o700 if item.is_dir() else 0o600)


def _copy_private_file(source: Path, destination: Path) -> dict[str, Any]:
    if source.is_symlink():
        raise BackupError(f"symbolic links are not allowed in backup sources: {source.name}")
    if not source.is_file():
        raise BackupError(f"backup source is not a regular file: {source.name}")
    _owner_only_directory(destination.parent)
    shutil.copyfile(source, destination)
    destination.chmod(0o600)
    return {
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _tree_files(source: Path) -> list[Path]:
    if not source.exists():
        return []
    if source.is_symlink() or not source.is_dir():
        raise BackupError(f"backup source must be a regular directory: {source.name}")
    files: list[Path] = []
    for item in sorted(source.rglob("*")):
        if item.is_symlink():
            raise BackupError(f"symbolic links are not allowed in backup sources: {item.name}")
        if item.is_file():
            if item.suffix.casefold() == ".env":
                raise BackupError("credential-like .env files must not be stored in resources or output")
            files.append(item)
        elif not item.is_dir():
            raise BackupError(f"special files are not allowed in backup sources: {item.name}")
    return files


def _copy_tree(source: Path, destination: Path, snapshot_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in _tree_files(source):
        target = destination / item.relative_to(source)
        record = _copy_private_file(item, target)
        record["path"] = target.relative_to(snapshot_root).as_posix()
        records.append(record)
    return records


def _sqlite_backup(source: Path, destination: Path) -> int:
    if source.is_symlink() or not source.is_file():
        raise BackupError("family database does not exist or is not a regular file")
    _owner_only_directory(destination.parent)
    source_connection = sqlite3.connect(source, timeout=30)
    target_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(target_connection)
    finally:
        target_connection.close()
        source_connection.close()
    destination.chmod(0o600)
    with _open_sqlite_read_only(destination) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if integrity != "ok":
        raise BackupError("database snapshot failed its integrity check")
    return version


def _open_sqlite_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)


def _snapshot_id() -> tuple[str, str]:
    now = datetime.now(UTC)
    created_at = now.isoformat()
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return f"{SNAPSHOT_PREFIX}{stamp}-{secrets.token_hex(3)}", created_at


def create_snapshot(
    *,
    db_path: str | Path,
    private_dir: str | Path,
    output_dir: str | Path,
    backup_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root else repository_root()
    root = _ensure_external_backup_root(backup_root or default_backup_root(repo), repo, create=True)
    raw_private = Path(private_dir).expanduser()
    raw_output = Path(output_dir).expanduser()
    raw_database = Path(db_path).expanduser()
    if raw_private.is_symlink() or raw_output.is_symlink() or raw_database.is_symlink():
        raise BackupError("backup sources may not be symbolic links")
    private = raw_private.resolve()
    output = raw_output.resolve()
    database = raw_database.resolve()
    for source in (private, output, database):
        if _is_within(source, root) or _is_within(root, source):
            raise BackupError("backup sources and destination may not contain one another")
    if _is_within(private / "setup", output) or _is_within(output, private / "setup"):
        raise BackupError("output directory may not contain private setup credentials")
    if _is_within(database, output):
        raise BackupError("output directory may not contain the family database")

    snapshot_id, created_at = _snapshot_id()
    final = root / snapshot_id
    partial = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
    partial.chmod(0o700)
    try:
        database_target = partial / "data" / "curiosity.db"
        schema_version = _sqlite_backup(database, database_target)
        files = [
            {
                "path": "data/curiosity.db",
                "bytes": database_target.stat().st_size,
                "sha256": _sha256(database_target),
            }
        ]
        files.extend(_copy_tree(private / "resources", partial / "resources", partial))
        files.extend(_copy_tree(output, partial / "output", partial))
        for name in SETUP_ALLOWLIST:
            source = private / "setup" / name
            if source.exists():
                content = source.read_bytes()
                if any(marker in content for marker in SECRET_MARKERS):
                    raise BackupError(f"non-secret setup file appears to contain credentials: {name}")
                target = partial / "setup" / name
                record = _copy_private_file(source, target)
                record["path"] = target.relative_to(partial).as_posix()
                files.append(record)

        files.sort(key=lambda item: item["path"])
        manifest = {
            "format": BACKUP_FORMAT,
            "format_version": BACKUP_FORMAT_VERSION,
            "app_version": __version__,
            "created_at": created_at,
            "snapshot_id": snapshot_id,
            "database_schema_version": schema_version,
            "source_roots": {
                "private": str(private),
                "output": str(output),
            },
            "files": files,
            "totals": {
                "bytes": sum(int(item["bytes"]) for item in files),
                "files": len(files),
            },
            "credentials_included": False,
        }
        _write_json(partial / MANIFEST_NAME, manifest)
        _harden_tree(partial)
        os.replace(partial, final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    return {
        "status": "created",
        "snapshot": snapshot_id,
        "backup_root": str(root),
        "files": manifest["totals"]["files"],
        "bytes": manifest["totals"]["bytes"],
        "credentials_included": False,
        "next_action": "run curiosity backup verify",
    }


def _snapshots(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (
            item
            for item in root.iterdir()
            if item.is_dir() and not item.is_symlink() and item.name.startswith(SNAPSHOT_PREFIX)
        ),
        key=lambda item: (item.stat().st_mtime_ns, item.name),
    )


def _select_snapshot(root: Path, snapshot: str | None) -> Path:
    snapshots = _snapshots(root)
    if snapshot is None:
        if not snapshots:
            raise BackupError("no family backups exist yet")
        return snapshots[-1]
    if Path(snapshot).name != snapshot or not snapshot.startswith(SNAPSHOT_PREFIX):
        raise BackupError("snapshot must be a backup ID from curiosity backup status")
    selected = root / snapshot
    if not selected.is_dir() or selected.is_symlink():
        raise BackupError("backup snapshot was not found")
    return selected


def backup_status(
    *, backup_root: str | Path | None = None, repo_root: str | Path | None = None
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root else repository_root()
    root = _ensure_external_backup_root(backup_root or default_backup_root(repo), repo, create=False)
    snapshots = _snapshots(root)
    if not snapshots:
        return {
            "status": "empty",
            "backup_root": str(root),
            "snapshots": 0,
            "next_action": "run curiosity backup create",
        }
    latest = snapshots[-1]
    try:
        manifest = _load_manifest(latest)
        totals = manifest.get("totals")
        created_at = manifest.get("created_at")
        if (
            manifest.get("format") != BACKUP_FORMAT
            or manifest.get("format_version") != BACKUP_FORMAT_VERSION
            or manifest.get("snapshot_id") != latest.name
            or not isinstance(created_at, str)
            or not isinstance(totals, dict)
            or not isinstance(totals.get("files"), int)
            or not isinstance(totals.get("bytes"), int)
        ):
            raise BackupError("backup summary metadata is invalid")
        datetime.fromisoformat(created_at)
        summary = {
            "snapshot": latest.name,
            "created_at": created_at,
            "files": totals["files"],
            "bytes": totals["bytes"],
        }
    except (BackupError, ValueError):
        summary = {"snapshot": latest.name, "manifest": "invalid"}
    return {
        "status": "available",
        "backup_root": str(root),
        "snapshots": len(snapshots),
        "latest": summary,
        "next_action": "run curiosity backup verify",
    }


def _load_manifest(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise BackupError("backup manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError("backup manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise BackupError("backup manifest must be a JSON object")
    return manifest


def _safe_manifest_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise BackupError("backup manifest contains a non-text file path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        raise BackupError("backup manifest contains an unsafe file path")
    if path.suffix.casefold() == ".env":
        raise BackupError("backup manifest may not include credential files")
    if path.parts[0] not in {"data", "resources", "output", "setup"}:
        raise BackupError("backup manifest contains an unsupported file location")
    if path.parts[0] == "data" and path != PurePosixPath("data/curiosity.db"):
        raise BackupError("backup manifest contains an unsupported database file")
    if path.parts[0] == "setup" and (len(path.parts) != 2 or path.name not in SETUP_ALLOWLIST):
        raise BackupError("backup manifest contains a non-allowlisted setup file")
    return path


def verify_snapshot(
    snapshot: str | None = None,
    *,
    backup_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root else repository_root()
    root = _ensure_external_backup_root(backup_root or default_backup_root(repo), repo, create=False)
    selected = _select_snapshot(root, snapshot)
    errors: list[str] = []
    try:
        manifest = _load_manifest(selected)
        if manifest.get("format") != BACKUP_FORMAT:
            errors.append("unsupported backup format")
        if manifest.get("format_version") != BACKUP_FORMAT_VERSION:
            errors.append("unsupported backup format version")
        if manifest.get("snapshot_id") != selected.name:
            errors.append("snapshot ID does not match its directory")
        if manifest.get("credentials_included") is not False:
            errors.append("credential exclusion marker is missing")
        source_roots = manifest.get("source_roots")
        if not isinstance(source_roots, dict) or not all(
            isinstance(source_roots.get(name), str) and Path(source_roots[name]).is_absolute()
            for name in ("private", "output")
        ):
            errors.append("backup source-root metadata is invalid")
        entries = manifest.get("files")
        if not isinstance(entries, list):
            raise BackupError("backup manifest file list is invalid")
        expected_paths: set[str] = set()
        expected_bytes = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise BackupError("backup manifest file entry is invalid")
            relative = _safe_manifest_path(entry.get("path"))
            relative_text = relative.as_posix()
            if relative_text in expected_paths:
                errors.append(f"duplicate file entry: {relative_text}")
                continue
            expected_paths.add(relative_text)
            if not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0:
                errors.append(f"invalid size: {relative_text}")
                continue
            if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
                errors.append(f"invalid checksum: {relative_text}")
                continue
            expected_bytes += entry["bytes"]
            target = selected.joinpath(*relative.parts)
            if target.is_symlink() or not target.is_file():
                errors.append(f"missing or unsafe file: {relative_text}")
                continue
            if target.stat().st_size != entry.get("bytes"):
                errors.append(f"size mismatch: {relative_text}")
            if _sha256(target) != entry.get("sha256"):
                errors.append(f"checksum mismatch: {relative_text}")
            if target.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                errors.append(f"file is not owner-only: {relative_text}")

        actual_paths: set[str] = set()
        for item in selected.rglob("*"):
            if item.is_symlink():
                errors.append(f"symbolic link found: {item.relative_to(selected).as_posix()}")
            elif item.is_file() and item.name != MANIFEST_NAME:
                actual_paths.add(item.relative_to(selected).as_posix())
            elif not item.is_dir() and not item.is_file():
                errors.append(f"special file found: {item.relative_to(selected).as_posix()}")
        for unexpected in sorted(actual_paths - expected_paths):
            errors.append(f"unexpected file: {unexpected}")
        totals = manifest.get("totals")
        if not isinstance(totals, dict):
            errors.append("backup totals are missing")
        else:
            if totals.get("files") != len(entries):
                errors.append("backup file total does not match the manifest")
            if totals.get("bytes") != expected_bytes:
                errors.append("backup byte total does not match the manifest")

        protected_paths = [root, root / BACKUP_ROOT_MARKER, selected, selected / MANIFEST_NAME]
        protected_paths.extend(item for item in selected.rglob("*") if item.is_dir())
        for protected in protected_paths:
            if protected.exists() and protected.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                errors.append(f"path is not owner-only: {protected.relative_to(root).as_posix()}")

        database = selected / "data" / "curiosity.db"
        if database.is_file() and not database.is_symlink():
            with _open_sqlite_read_only(database) as connection:
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if integrity != "ok":
                errors.append("database integrity check failed")
            if schema_version != manifest.get("database_schema_version"):
                errors.append("database schema version does not match the manifest")
        else:
            errors.append("database snapshot is missing")
    except (BackupError, OSError, sqlite3.DatabaseError) as exc:
        errors.append(str(exc))

    return {
        "status": "pass" if not errors else "fail",
        "snapshot": selected.name,
        "errors": errors,
        "credentials_included": False,
    }


def _rebase_path(value: str, old_root: Path, new_root: Path) -> str:
    candidate = Path(value)
    try:
        relative = candidate.relative_to(old_root)
    except ValueError:
        return value
    return str(new_root / relative)


def _rebase_restored_database(database: Path, manifest: dict[str, Any], target_private: Path) -> int:
    source_roots = manifest["source_roots"]
    old_private = Path(source_roots["private"])
    old_output = Path(source_roots["output"])
    new_output = target_private / "output"
    changed = 0
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        for table in ("artifacts", "visual_assets"):
            if table not in tables:
                continue
            rows = connection.execute(f"SELECT id,path FROM {table}").fetchall()
            for row_id, value in rows:
                rebased = _rebase_path(str(value), old_output, new_output)
                if rebased == value:
                    rebased = _rebase_path(str(value), old_private, target_private)
                if rebased != value:
                    connection.execute(f"UPDATE {table} SET path=? WHERE id=?", (rebased, row_id))
                    changed += 1

        rows = (
            connection.execute("SELECT id,source_path,metadata_json FROM resource_documents").fetchall()
            if "resource_documents" in tables
            else []
        )
        for row_id, source_path, metadata_json in rows:
            rebased_source = _rebase_path(str(source_path), old_private, target_private)
            try:
                metadata = json.loads(metadata_json)
            except (TypeError, json.JSONDecodeError):
                metadata = None
            rebased_metadata = metadata_json
            if isinstance(metadata, dict) and isinstance(metadata.get("text_extract_path"), str):
                original_text_path = metadata["text_extract_path"]
                metadata["text_extract_path"] = _rebase_path(
                    original_text_path, old_private, target_private
                )
                if metadata["text_extract_path"] != original_text_path:
                    rebased_metadata = json.dumps(
                        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
            if rebased_source != source_path or rebased_metadata != metadata_json:
                connection.execute(
                    "UPDATE resource_documents SET source_path=?,metadata_json=? WHERE id=?",
                    (rebased_source, rebased_metadata, row_id),
                )
                changed += 1
        connection.commit()
    finally:
        connection.close()
    database.chmod(0o600)
    with _open_sqlite_read_only(database) as check:
        if check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise BackupError("restored database failed its integrity check after path rebasing")
    return changed


def restore_snapshot(
    snapshot: str | None = None,
    *,
    target_private: str | Path | None = None,
    backup_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root else repository_root()
    root = _ensure_external_backup_root(backup_root or default_backup_root(repo), repo, create=False)
    selected = _select_snapshot(root, snapshot)
    verification = verify_snapshot(selected.name, backup_root=root, repo_root=repo)
    if verification["status"] != "pass":
        raise BackupError("backup verification failed; nothing was restored")

    raw_target = (
        Path(target_private).expanduser()
        if target_private
        else repo / "private" / "restores" / selected.name
    )
    if raw_target.is_symlink():
        raise BackupError("restore target may not be a symbolic link")
    target = raw_target.resolve()
    if _is_within(target, root):
        raise BackupError("restore target may not be inside the backup destination")
    if target.exists() or target.is_symlink():
        raise BackupError("restore target already exists; choose a new empty path")
    _create_owner_only_parents(target.parent)
    partial = target.parent / f".{target.name}.partial-{secrets.token_hex(3)}"
    _owner_only_directory(partial)
    target_reserved = False
    try:
        manifest = _load_manifest(selected)
        for entry in manifest["files"]:
            relative = _safe_manifest_path(entry["path"])
            source = selected.joinpath(*relative.parts)
            destination = partial.joinpath(*relative.parts)
            copied = _copy_private_file(source, destination)
            if copied["bytes"] != entry["bytes"] or copied["sha256"] != entry["sha256"]:
                raise BackupError("backup changed during restore; nothing was restored")
        paths_rebased = _rebase_restored_database(partial / "data" / "curiosity.db", manifest, target)
        _harden_tree(partial)
        target.mkdir(mode=0o700)
        target_reserved = True
        os.replace(partial, target)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        if target_reserved:
            try:
                target.rmdir()
            except OSError:
                pass
        raise

    active_private = repo / "private"
    active = target == active_private
    return {
        "status": "restored",
        "snapshot": selected.name,
        "restore_path": str(target),
        "database": str(target / "data" / "curiosity.db"),
        "active": active,
        "database_paths_rebased": paths_rebased,
        "credentials_restored": False,
        "next_action": (
            "re-enter Slack and model credentials, then run curiosity doctor"
            if active
            else "current family data is unchanged; inspect this recovery copy before activating it"
        ),
    }
