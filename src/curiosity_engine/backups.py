from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import sqlite3
import stat
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .config import family_home, repository_root
from .db import init_db
from .deletion_ledger import (
    BACKUP_IDENTITY_NAME,
    BACKUP_SOURCES_NAME,
    LEDGER_NAME,
    MANAGED_RESTORES_NAME,
    RESTORE_MARKER_NAME,
    assert_backup_root_registration_complete,
    backup_root_database_identity,
    backup_root_deletion_manifest,
    backup_root_lifecycle,
    backup_roots_lifecycle,
    database_deletion_conflicts,
    deletion_manifest,
    import_replay_tombstones,
    initialize_backup_root_metadata,
    managed_data_lifecycle,
    managed_restore_marker,
    mark_backup_root_registration_pending,
    merge_deletion_manifest,
    prepare_restored_database_registration,
    reconcile_database_deletions,
    recover_interrupted_restore_paths,
    register_backup_root,
    register_managed_restore,
    sync_registered_backup_sources,
    unregister_managed_restore,
)
from .research_authority_ledger import (
    assert_research_authority_image,
    backup_authority_roots,
    install_backup_research_authority,
    sync_registered_backup_authority,
)
from .research_authority_ledger import ledger_path as research_authority_path

BACKUP_FORMAT = "curiosity-family-backup"
BACKUP_FORMAT_VERSION = 2
LEGACY_BACKUP_FORMAT_VERSION = 1
BACKUP_ROOT_MARKER = ".curiosity-backup-root.json"
MANIFEST_NAME = "manifest.json"
SNAPSHOT_PREFIX = "curiosity-backup-"
RESEARCH_AUTHORITY_BACKUP_PATH = PurePosixPath(
    "data/curiosity.db.research-authority.json"
)
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
        compatible = current == expected or current == {
            "format": BACKUP_FORMAT,
            "format_version": LEGACY_BACKUP_FORMAT_VERSION,
        }
        if not compatible:
            raise BackupError("destination is not a compatible Curiosity Engine backup root")
    elif existed and any(candidate.iterdir()):
        raise BackupError("destination is not an initialized Curiosity Engine backup root")
    elif create:
        _write_json(marker, expected)
        try:
            initialize_backup_root_metadata(candidate)
        except RuntimeError as exc:
            marker.unlink(missing_ok=True)
            raise BackupError("backup root authority metadata could not be initialized") from exc
    return candidate


def _backup_root_marker(root: Path) -> dict[str, Any]:
    marker = root / BACKUP_ROOT_MARKER
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError("backup root marker is unreadable") from exc
    if not isinstance(value, dict):
        raise BackupError("backup root marker is invalid")
    return value


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


def _tree_files(source: Path, *, excluded_top_level: frozenset[str] = frozenset()) -> list[Path]:
    if not source.exists():
        return []
    if source.is_symlink() or not source.is_dir():
        raise BackupError(f"backup source must be a regular directory: {source.name}")
    files: list[Path] = []
    for item in sorted(source.rglob("*")):
        if item.is_symlink():
            raise BackupError(f"symbolic links are not allowed in backup sources: {item.name}")
        relative = item.relative_to(source)
        if relative.parts and relative.parts[0] in excluded_top_level:
            continue
        if item.is_file():
            if item.suffix.casefold() == ".env":
                raise BackupError("credential-like .env files must not be stored in resources or output")
            files.append(item)
        elif not item.is_dir():
            raise BackupError(f"special files are not allowed in backup sources: {item.name}")
    return files


def _copy_tree(
    source: Path,
    destination: Path,
    snapshot_root: Path,
    *,
    excluded_top_level: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in _tree_files(source, excluded_top_level=excluded_top_level):
        target = destination / item.relative_to(source)
        record = _copy_private_file(item, target)
        record["path"] = target.relative_to(snapshot_root).as_posix()
        records.append(record)
    return records


def _sqlite_backup(source: Path, destination: Path) -> tuple[int, str]:
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
        identities = connection.execute(
            "SELECT singleton,database_id FROM family_database_identity"
        ).fetchall()
    if integrity != "ok":
        raise BackupError("database snapshot failed its integrity check")
    if len(identities) != 1 or int(identities[0][0]) != 1:
        raise BackupError("database snapshot has no stable family identity")
    return version, str(identities[0][1])


def _open_sqlite_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)


def _snapshot_id() -> tuple[str, str]:
    now = datetime.now(UTC)
    created_at = now.isoformat()
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return f"{SNAPSHOT_PREFIX}{stamp}-{secrets.token_hex(3)}", created_at


def _database_identity(database: Path) -> str:
    try:
        with _open_sqlite_read_only(database) as connection:
            rows = connection.execute(
                "SELECT singleton,database_id FROM family_database_identity"
            ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise BackupError("family database identity is unreadable") from exc
    if len(rows) != 1 or int(rows[0][0]) != 1 or not isinstance(rows[0][1], str):
        raise BackupError("family database has no stable identity")
    return str(rows[0][1])


def _adopt_legacy_backup_root(
    root: Path,
    *,
    database: Path,
    private: Path,
    output: Path,
    repo: Path,
) -> None:
    """Authenticate a v1 root before making it deletion-authoritative."""

    marker = _backup_root_marker(root)
    current = {"format": BACKUP_FORMAT, "format_version": BACKUP_FORMAT_VERSION}
    legacy = {"format": BACKUP_FORMAT, "format_version": LEGACY_BACKUP_FORMAT_VERSION}
    if marker == current:
        return
    if marker != legacy:
        raise BackupError("destination is not a compatible Curiosity Engine backup root")

    database_identity = _database_identity(database)
    for partial in root.iterdir():
        if partial.is_dir() and partial.name.startswith(".partial-"):
            raise BackupError("legacy backup root has an interrupted snapshot; adoption refused")
    for snapshot in _snapshots(root):
        manifest = _load_manifest(snapshot)
        if manifest.get("format_version") != LEGACY_BACKUP_FORMAT_VERSION:
            raise BackupError("legacy backup root contains a non-legacy snapshot")
        verification = verify_snapshot(snapshot.name, backup_root=root, repo_root=repo)
        if verification["status"] != "pass":
            raise BackupError("legacy backup verification failed; adoption refused")
        source_roots = manifest.get("source_roots", {})
        if source_roots.get("private") != str(private) or source_roots.get("output") != str(output):
            raise BackupError("legacy backup sources do not match this family")
        manifest_identity = manifest.get("database_identity")
        if manifest_identity is not None and manifest_identity != database_identity:
            raise BackupError("legacy backup belongs to a different family database")

    sources_path = root / BACKUP_SOURCES_NAME
    restores_path = root / MANAGED_RESTORES_NAME
    ledger_path = root / LEDGER_NAME
    authority_paths = (sources_path, restores_path, ledger_path)
    if any(path.exists() for path in authority_paths) and not all(
        path.exists() for path in authority_paths
    ):
        raise BackupError("legacy backup authority metadata is incomplete")
    if not sources_path.exists():
        try:
            initialize_backup_root_metadata(root)
        except RuntimeError as exc:
            raise BackupError("legacy backup authority metadata could not be initialized") from exc
    else:
        try:
            sources = json.loads(sources_path.read_text(encoding="utf-8"))
            restores = json.loads(restores_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackupError("legacy backup authority metadata is unreadable") from exc
        if sources != {
            "format": "curiosity-backup-source-registry",
            "version": 1,
            "databases": [],
        } or restores != {
            "format": "curiosity-managed-restore-registry",
            "version": 1,
            "restores": [],
        }:
            raise BackupError("legacy backup authority metadata is not safely adoptable")
        try:
            legacy_deletions = backup_root_deletion_manifest(root)
        except RuntimeError as exc:
            raise BackupError("legacy backup deletion metadata is invalid") from exc
        if any(legacy_deletions[kind] for kind in legacy_deletions):
            raise BackupError("legacy backup deletion metadata is not safely adoptable")
    if (root / BACKUP_IDENTITY_NAME).exists():
        try:
            if backup_root_database_identity(root) != database_identity:
                raise BackupError("legacy backup belongs to a different family database")
        except RuntimeError as exc:
            raise BackupError("legacy backup identity is invalid") from exc
    # A current-format marker makes this root eligible for restore. Publish a
    # durable barrier first; registration will clear it only after the active
    # database's deletion manifest has been synchronized.
    mark_backup_root_registration_pending(root, database, database_identity)
    _write_json(root / BACKUP_ROOT_MARKER, current)


def create_snapshot(
    *,
    db_path: str | Path,
    private_dir: str | Path,
    output_dir: str | Path,
    backup_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    _test_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root else repository_root()
    database = Path(db_path).expanduser().resolve()
    private = Path(private_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    init_db(database)
    root = _ensure_external_backup_root(backup_root or default_backup_root(repo), repo, create=True)
    with backup_root_lifecycle(root):
        _adopt_legacy_backup_root(
            root,
            database=database,
            private=private,
            output=output,
            repo=repo,
        )
    with managed_data_lifecycle(database, extra_backup_roots=(root,)):
        # A killed deletion can leave verified bytes in the private quarantine
        # after its database transaction committed. Recover them before any
        # source walk, then still exclude the internal journal defensively.
        init_db(database)
        return _create_snapshot_locked(
            db_path=database,
            private_dir=private_dir,
            output_dir=output_dir,
            backup_root=root,
            repo_root=repo,
            _test_hook=_test_hook,
        )


def _create_snapshot_locked(
    *,
    db_path: str | Path,
    private_dir: str | Path,
    output_dir: str | Path,
    backup_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    _test_hook: Callable[[str], None] | None = None,
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
    register_backup_root(
        database,
        root,
        _test_hook=_test_hook,
        _before_complete=lambda: sync_registered_backup_authority(
            database,
            allow_registration_pending=True,
        ),
    )
    sync_registered_backup_authority(database)
    # Registration makes an adopted root deletion-authoritative. Reconcile it
    # immediately so deletions made while a v1 root was undiscoverable erase
    # its legacy images before this command can report success.
    reconcile_database_deletions(database)

    snapshot_id, created_at = _snapshot_id()
    final = root / snapshot_id
    partial = Path(tempfile.mkdtemp(prefix=".partial-", dir=root))
    partial.chmod(0o700)
    try:
        database_target = partial / "data" / "curiosity.db"
        schema_version, database_identity = _sqlite_backup(database, database_target)
        if _test_hook is not None:
            _test_hook("after_database_copy")
        files = [
            {
                "path": "data/curiosity.db",
                "bytes": database_target.stat().st_size,
                "sha256": _sha256(database_target),
            }
        ]
        authority_source = research_authority_path(database)
        if not authority_source.is_file() or authority_source.is_symlink():
            raise BackupError("research-authority ledger is unavailable for backup")
        authority_target = research_authority_path(database_target)
        authority_record = _copy_private_file(authority_source, authority_target)
        authority_record["path"] = authority_target.relative_to(partial).as_posix()
        files.append(authority_record)
        try:
            assert_research_authority_image(database_target, authority_target)
        except RuntimeError as exc:
            raise BackupError(
                "database and research-authority ledger changed during backup"
            ) from exc
        files.extend(_copy_tree(private / "resources", partial / "resources", partial))
        files.extend(
            _copy_tree(
                output,
                partial / "output",
                partial,
                excluded_top_level=frozenset({".family-delete-journal"}),
            )
        )
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
            "database_identity": database_identity,
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
        sync_registered_backup_sources(root)
        conflicts = database_deletion_conflicts(
            database_target,
            backup_root_deletion_manifest(root),
            snapshot_created_at=created_at,
        )
        if conflicts:
            raise BackupError("source contains data already covered by a deletion tombstone")
        _harden_tree(partial)
        if _test_hook is not None:
            _test_hook("before_publish")
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
    if root.is_dir() and _backup_root_marker(root).get("format_version") == LEGACY_BACKUP_FORMAT_VERSION:
        return {
            "status": "upgrade_required",
            "backup_root": str(root),
            "snapshots": len(_snapshots(root)),
            "next_action": "run curiosity backup create to authenticate and adopt this legacy root",
        }
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
            or manifest.get("format_version") not in {
                LEGACY_BACKUP_FORMAT_VERSION,
                BACKUP_FORMAT_VERSION,
            }
            or manifest.get("snapshot_id") != latest.name
            or not isinstance(created_at, str)
            or (
                manifest.get("format_version") == BACKUP_FORMAT_VERSION
                and not isinstance(manifest.get("database_identity"), str)
            )
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
    if path.parts[0] == "data" and path not in {
        PurePosixPath("data/curiosity.db"),
        RESEARCH_AUTHORITY_BACKUP_PATH,
    }:
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
        format_version = manifest.get("format_version")
        if manifest.get("format") != BACKUP_FORMAT:
            errors.append("unsupported backup format")
        if format_version not in {LEGACY_BACKUP_FORMAT_VERSION, BACKUP_FORMAT_VERSION}:
            errors.append("unsupported backup format version")
        if manifest.get("snapshot_id") != selected.name:
            errors.append("snapshot ID does not match its directory")
        if format_version == BACKUP_FORMAT_VERSION and not isinstance(
            manifest.get("database_identity"), str
        ):
            errors.append("database identity is missing")
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
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                identities = (
                    connection.execute(
                        "SELECT singleton,database_id FROM family_database_identity"
                    ).fetchall()
                    if "family_database_identity" in tables
                    else []
                )
            if integrity != "ok":
                errors.append("database integrity check failed")
            if schema_version != manifest.get("database_schema_version"):
                errors.append("database schema version does not match the manifest")
            if format_version == BACKUP_FORMAT_VERSION:
                if (
                    len(identities) != 1
                    or int(identities[0][0]) != 1
                    or str(identities[0][1]) != manifest.get("database_identity")
                ):
                    errors.append("database identity does not match the manifest")
            elif identities and (
                len(identities) != 1
                or int(identities[0][0]) != 1
                or manifest.get("database_identity") not in {None, str(identities[0][1])}
            ):
                errors.append("legacy database identity is invalid")
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


def _bind_restored_database_identity(database: Path, database_identity: str) -> None:
    """Attach an adopted legacy image to its authenticated backup root."""

    connection = sqlite3.connect(database)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS family_database_identity (
               singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
               database_id TEXT NOT NULL UNIQUE,
               created_at TEXT NOT NULL
               )"""
        )
        rows = connection.execute(
            "SELECT singleton,database_id FROM family_database_identity ORDER BY singleton"
        ).fetchall()
        if not rows:
            connection.execute(
                "INSERT INTO family_database_identity(singleton,database_id,created_at) VALUES(1,?,?)",
                (database_identity, datetime.now(UTC).isoformat()),
            )
        elif len(rows) != 1 or int(rows[0][0]) != 1 or str(rows[0][1]) != database_identity:
            raise BackupError("legacy database belongs to a different family")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


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
        for table in ("artifacts", "visual_assets", "inbound_assets"):
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
    _test_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root else repository_root()
    root = _ensure_external_backup_root(backup_root or default_backup_root(repo), repo, create=False)
    if _backup_root_marker(root).get("format_version") != BACKUP_FORMAT_VERSION:
        raise BackupError("legacy backup root must be adopted with curiosity backup create before restore")
    # Preserve the original registration barrier even when a process died
    # before it could create the research-authority peer ledger.
    try:
        assert_backup_root_registration_complete(root)
    except RuntimeError as exc:
        raise BackupError("backup root registration is incomplete; restore refused") from exc
    try:
        authority_roots = backup_authority_roots(root)
    except RuntimeError as exc:
        raise BackupError("backup research-authority ledger is unreadable") from exc
    if root not in authority_roots:
        raise BackupError("backup research-authority peer-root membership is incomplete")
    try:
        with backup_roots_lifecycle(authority_roots):
            try:
                assert_backup_root_registration_complete(root)
                if backup_authority_roots(root) != authority_roots:
                    raise RuntimeError("backup research-authority peer-root membership changed")
            except RuntimeError as exc:
                raise BackupError("backup root registration is incomplete; restore refused") from exc
            return _restore_snapshot_locked(
                snapshot,
                target_private=target_private,
                backup_root=root,
                repo_root=repo,
                _test_hook=_test_hook,
            )
    except BackupError:
        raise
    except RuntimeError as exc:
        if "deletion" in str(exc):
            raise BackupError("backup deletion ledger is unreadable; restore refused") from exc
        raise BackupError("backup root authority metadata is unreadable; restore refused") from exc


def _restore_snapshot_locked(
    snapshot: str | None = None,
    *,
    target_private: str | Path | None = None,
    backup_root: str | Path | None = None,
    repo_root: str | Path | None = None,
    _test_hook: Callable[[str], None] | None = None,
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
    try:
        recover_interrupted_restore_paths(root, target, snapshot=selected.name)
    except RuntimeError as exc:
        raise BackupError("interrupted restore metadata is unsafe; restore refused") from exc
    if target.exists() or target.is_symlink():
        raise BackupError("restore target already exists; choose a new empty path")
    _create_owner_only_parents(target.parent)
    partial = target.parent / f".{target.name}.partial-{secrets.token_hex(3)}"
    staging_registered = False
    target_registered = False
    target_reserved = False
    target_published = False
    try:
        manifest = _load_manifest(selected)
        format_version = int(manifest["format_version"])
        database_identity = (
            str(manifest["database_identity"])
            if format_version == BACKUP_FORMAT_VERSION
            else backup_root_database_identity(root)
        )
        try:
            sync_registered_backup_sources(root)
            current_deletions = backup_root_deletion_manifest(root)
        except RuntimeError as exc:
            raise BackupError("backup deletion ledger is unreadable; restore refused") from exc
        default_database = repo / "private" / "data" / "curiosity.db"
        if default_database.parent.exists():
            active_manifest = deletion_manifest(default_database)
            for kind, records in active_manifest.items():
                current_deletions[kind].update(records)
        snapshot_created_at = str(manifest["created_at"])
        if database_deletion_conflicts(
            selected / "data" / "curiosity.db",
            current_deletions,
            snapshot_created_at=snapshot_created_at,
            expected_database_identity=(
                database_identity if format_version == BACKUP_FORMAT_VERSION else None
            ),
            require_database_identity=format_version == BACKUP_FORMAT_VERSION,
        ):
            raise BackupError("backup contains data deleted after the snapshot; restore refused")
        _owner_only_directory(partial)
        restore_marker = managed_restore_marker(
            root,
            snapshot=selected.name,
            snapshot_created_at=snapshot_created_at,
            database_identity=database_identity,
        )
        _write_json(partial / RESTORE_MARKER_NAME, restore_marker)
        register_managed_restore(
            root,
            partial,
            snapshot=selected.name,
            snapshot_created_at=snapshot_created_at,
            database_identity=database_identity,
            state="staging",
        )
        staging_registered = True
        for entry in manifest["files"]:
            relative = _safe_manifest_path(entry["path"])
            source = selected.joinpath(*relative.parts)
            destination = partial.joinpath(*relative.parts)
            copied = _copy_private_file(source, destination)
            if copied["bytes"] != entry["bytes"] or copied["sha256"] != entry["sha256"]:
                raise BackupError("backup changed during restore; nothing was restored")
        if _test_hook is not None:
            _test_hook("before_rebase")
        if format_version == LEGACY_BACKUP_FORMAT_VERSION:
            _bind_restored_database_identity(
                partial / "data" / "curiosity.db",
                database_identity,
            )
        paths_rebased = _rebase_restored_database(partial / "data" / "curiosity.db", manifest, target)
        try:
            restored_database = partial / "data" / "curiosity.db"
            authority_roots = backup_authority_roots(root)
            with managed_data_lifecycle(
                restored_database,
                extra_backup_roots=authority_roots,
            ):
                install_backup_research_authority(
                    restored_database,
                    root,
                )
        except RuntimeError as exc:
            raise BackupError("backup research-authority ledger is unreadable") from exc
        try:
            sync_registered_backup_sources(root)
            final_deletions = backup_root_deletion_manifest(root)
        except RuntimeError as exc:
            raise BackupError("backup deletion ledger is unreadable; restore refused") from exc
        for kind, records in current_deletions.items():
            final_deletions[kind].update(records)
        if database_deletion_conflicts(
            partial / "data" / "curiosity.db",
            final_deletions,
            snapshot_created_at=snapshot_created_at,
            expected_database_identity=database_identity,
        ):
            raise BackupError("backup contains data deleted during restore; restore refused")
        merge_deletion_manifest(partial / "data" / "curiosity.db", final_deletions)
        with sqlite3.connect(partial / "data" / "curiosity.db") as restored_connection:
            import_replay_tombstones(restored_connection, final_deletions)
        active_private = repo / "private"
        active = target == active_private
        if active:
            try:
                prepare_restored_database_registration(
                    partial / "data" / "curiosity.db",
                    target / "data" / "curiosity.db",
                    root,
                )
            except RuntimeError as exc:
                raise BackupError("active restore could not inherit its backup-root registration") from exc
        register_managed_restore(
            root,
            target,
            snapshot=selected.name,
            snapshot_created_at=snapshot_created_at,
            database_identity=database_identity,
            state="pending",
        )
        target_registered = True
        _harden_tree(partial)
        if _test_hook is not None:
            _test_hook("before_publish")
        target.mkdir(mode=0o700)
        target_reserved = True
        if _test_hook is not None:
            _test_hook("after_target_reservation")
        os.replace(partial, target)
        target_published = True
        if _test_hook is not None:
            _test_hook("after_publish")
        register_managed_restore(
            root,
            target,
            snapshot=selected.name,
            snapshot_created_at=snapshot_created_at,
            database_identity=database_identity,
            state="published",
        )
        unregister_managed_restore(root, partial)
        staging_registered = False
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        if target_published:
            shutil.rmtree(target, ignore_errors=True)
        elif target_reserved:
            try:
                target.rmdir()
            except OSError:
                pass
        if target_registered:
            unregister_managed_restore(root, target)
        if staging_registered:
            unregister_managed_restore(root, partial)
        raise

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
