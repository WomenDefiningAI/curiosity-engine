from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from threading import local
from typing import Any

LEDGER_FORMAT = "curiosity-monotonic-deletions"
LEDGER_VERSION = 4
FAMILY_DATABASE_IDENTITY_SCHEMA_VERSION = 23
SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")
LEDGER_NAME = "family-deletions.json"
REGISTRY_NAME = "family-backup-roots.json"
BACKUP_SOURCES_NAME = "family-backup-sources.json"
BACKUP_IDENTITY_NAME = "family-backup-identity.json"
BACKUP_REGISTRATION_PENDING_NAME = "family-backup-registration-pending.json"
MANAGED_RESTORES_NAME = "family-managed-restores.json"
RESTORE_MARKER_NAME = ".curiosity-managed-restore.json"
LIFECYCLE_LOCK_NAME = ".family-data-lifecycle.lock"
_LIFECYCLE_LOCKS = local()
DELETION_KINDS = (
    "events",
    "transport_events",
    "learning_sources",
    "learner_models",
    "parent_memories",
)
MANAGED_RESTORE_STATES = ("staging", "pending", "published")
REDACTED_PARENT_MEMORY_INSPECTION = (
    "Parent memory values were displayed to the parent and omitted from durable thread history."
)
PARENT_MEMORY_INSPECTION_PREFIX = (
    "Here’s what I currently remember as active parent guidance:\n"
)


def event_id_hash(event_id: str) -> str:
    return hashlib.sha256(event_id.encode()).hexdigest()


def _has_deleted_memory_inspection(
    conn: sqlite3.Connection,
    memory_manifest: dict[str, dict[str, str]],
) -> bool:
    """Detect a legacy durable inspection that can replay a forgotten value."""

    tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "agent_messages" not in tables:
        return False
    for content, metadata_json in conn.execute(
        """SELECT content,metadata_json FROM agent_messages
           WHERE kind='parent_memory_inspection'"""
    ).fetchall():
        if str(content) == REDACTED_PARENT_MEMORY_INSPECTION:
            continue
        try:
            metadata = json.loads(str(metadata_json))
            refs = metadata.get("memory_refs") if isinstance(metadata, dict) else None
        except json.JSONDecodeError:
            return True
        if not isinstance(refs, list):
            return True
        for ref in refs:
            if not isinstance(ref, dict):
                return True
            memory_key = ref.get("memory_key")
            version = ref.get("version")
            if not isinstance(memory_key, str) or not isinstance(version, int):
                return True
            if event_id_hash(f"{memory_key}\0{version}") in memory_manifest:
                return True
    return False


def _redacted_parent_memory_delivery_payload(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    channel_id = payload.get("channel_id")
    if not isinstance(channel_id, str) or not channel_id:
        channel_id = "redacted-parent-memory-delivery"
    thread_id = payload.get("thread_id")
    return json.dumps(
        {
            "blocks": [],
            "channel_id": channel_id,
            "text": REDACTED_PARENT_MEMORY_INSPECTION,
            "thread_id": thread_id if isinstance(thread_id, str) else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _legacy_parent_memory_delivery_ids(
    conn: sqlite3.Connection,
    memory_keys: set[str] | None,
    *,
    delivery_ids: set[str] | None = None,
) -> set[str]:
    """Locate pre-lineage outbox rows through code-owned inspection records."""

    if memory_keys is not None and not memory_keys:
        return set()
    rows = conn.execute(
        """SELECT m.session_id,m.ordinal,m.metadata_json,m.created_at,
                  s.binding_id,s.conversation_ref,s.thread_ref,b.team_id,b.channel_id
           FROM agent_messages m
           JOIN agent_sessions s ON s.id=m.session_id
           JOIN transport_bindings b ON b.id=s.binding_id
           WHERE m.kind='parent_memory_inspection' AND s.transport='slack'
             AND s.binding_id IS NOT NULL
           ORDER BY m.created_at,m.id"""
    ).fetchall()
    selected: set[str] = set()
    for row in rows:
        try:
            metadata = json.loads(str(row["metadata_json"]))
        except json.JSONDecodeError:
            continue
        refs = metadata.get("memory_refs") if isinstance(metadata, dict) else None
        if not isinstance(refs, list) or not refs:
            continue
        valid_refs = [
            ref
            for ref in refs
            if isinstance(ref, dict)
            and isinstance(ref.get("memory_key"), str)
            and type(ref.get("version")) is int
            and int(ref["version"]) > 0
        ]
        if not valid_refs or (memory_keys is not None and not any(
            ref["memory_key"] in memory_keys for ref in valid_refs
        )):
            continue
        team_id = str(row["team_id"])
        channel_id = str(row["channel_id"])
        if event_id_hash(f"{team_id}:{channel_id}")[:20] != str(row["conversation_ref"]):
            continue
        next_message = conn.execute(
            """SELECT created_at FROM agent_messages
               WHERE session_id=? AND ordinal>? ORDER BY ordinal LIMIT 1""",
            (row["session_id"], row["ordinal"]),
        ).fetchone()
        params: list[Any] = [row["binding_id"], row["created_at"]]
        before = ""
        if next_message:
            before = " AND o.created_at<?"
            params.append(next_message["created_at"])
        for outbox in conn.execute(
            f"""SELECT o.id,o.payload_json,o.idempotency_key
                FROM delivery_outbox o
                WHERE o.binding_id=? AND o.created_at>=? {before}
                  AND NOT EXISTS(
                    SELECT 1 FROM delivery_outbox_parent_memories pm
                    WHERE pm.delivery_id=o.id
                  )
                ORDER BY o.created_at,o.id""",
            tuple(params),
        ).fetchall():
            if delivery_ids is not None and str(outbox["id"]) not in delivery_ids:
                continue
            try:
                payload = json.loads(str(outbox["payload_json"]))
            except json.JSONDecodeError:
                continue
            text = payload.get("text") if isinstance(payload, dict) else None
            if (
                not isinstance(text, str)
                or not text.startswith(PARENT_MEMORY_INSPECTION_PREFIX)
                or payload.get("channel_id") != channel_id
            ):
                continue
            thread_id = payload.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                thread_ref = event_id_hash(f"{team_id}:{channel_id}:{thread_id}")[:20]
            else:
                idempotency_key = str(outbox["idempotency_key"])
                if not idempotency_key.startswith("slack:") or ":" not in idempotency_key[6:]:
                    continue
                external_event_id = idempotency_key[6:].rsplit(":", 1)[0]
                thread_ref = event_id_hash(
                    f"{team_id}:{channel_id}:root:{external_event_id}"
                )[:20]
            if thread_ref == str(row["thread_ref"]):
                selected.add(str(outbox["id"]))
                break
    return selected


def redact_parent_memory_deliveries(
    conn: sqlite3.Connection,
    *,
    memory_keys: set[str] | None = None,
    delivery_ids: set[str] | None = None,
) -> int:
    """Redact typed inspection payloads and fence undelivered copies."""

    tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if not {"delivery_outbox", "delivery_outbox_parent_memories"} <= tables:
        return 0
    selected: dict[str, tuple[str, str]] = {}
    if memory_keys:
        placeholders = ",".join("?" for _ in memory_keys)
        for row in conn.execute(
            f"""SELECT DISTINCT o.id,o.payload_json,o.status FROM delivery_outbox o
                JOIN delivery_outbox_parent_memories m ON m.delivery_id=o.id
                WHERE m.memory_key IN ({placeholders})""",
            tuple(sorted(memory_keys)),
        ).fetchall():
            selected[str(row[0])] = (str(row[1]), str(row[2]))
        for delivery_id in _legacy_parent_memory_delivery_ids(conn, memory_keys):
            row = conn.execute(
                "SELECT payload_json,status FROM delivery_outbox WHERE id=?",
                (delivery_id,),
            ).fetchone()
            if row:
                selected[delivery_id] = (str(row[0]), str(row[1]))
    if delivery_ids:
        placeholders = ",".join("?" for _ in delivery_ids)
        for row in conn.execute(
            f"""SELECT id,payload_json,status FROM delivery_outbox
                WHERE id IN ({placeholders})""",
            tuple(sorted(delivery_ids)),
        ).fetchall():
            if conn.execute(
                "SELECT 1 FROM delivery_outbox_parent_memories WHERE delivery_id=? LIMIT 1",
                (row[0],),
            ).fetchone():
                selected[str(row[0])] = (str(row[1]), str(row[2]))
        for delivery_id in _legacy_parent_memory_delivery_ids(
            conn,
            None,
            delivery_ids=delivery_ids,
        ):
            row = conn.execute(
                "SELECT payload_json,status FROM delivery_outbox WHERE id=?",
                (delivery_id,),
            ).fetchone()
            if row:
                selected[delivery_id] = (str(row[0]), str(row[1]))
    for delivery_id, (payload_json, status) in selected.items():
        next_status = (
            "expired"
            if status in {"queued", "failed"}
            else "unknown"
            if status == "sending"
            else status
        )
        error = (
            "parent memory erased before delivery"
            if status in {"queued", "failed", "sending"}
            else None
        )
        conn.execute(
            """UPDATE delivery_outbox
               SET payload_json=?,status=?,last_error=COALESCE(?,last_error),updated_at=?
               WHERE id=?""",
            (
                _redacted_parent_memory_delivery_payload(payload_json),
                next_status,
                error,
                datetime.now().astimezone().isoformat(),
                delivery_id,
            ),
        )
    return len(selected)


def _has_deleted_memory_delivery(
    conn: sqlite3.Connection,
    memory_manifest: dict[str, dict[str, str]],
) -> bool:
    tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if not {"delivery_outbox", "delivery_outbox_parent_memories"} <= tables:
        return False
    for row in conn.execute(
        """SELECT o.payload_json,m.memory_key,m.memory_version
           FROM delivery_outbox o JOIN delivery_outbox_parent_memories m
             ON m.delivery_id=o.id"""
    ).fetchall():
        if event_id_hash(f"{row[1]}\0{row[2]}") not in memory_manifest:
            continue
        try:
            payload = json.loads(str(row[0]))
        except json.JSONDecodeError:
            return True
        if (
            not isinstance(payload, dict)
            or payload.get("text") != REDACTED_PARENT_MEMORY_INSPECTION
            or payload.get("blocks") != []
        ):
            return True
    return False


def migration_backup_path(database: str | Path, version: int, stamp: str) -> Path:
    """Return the one canonical automatic schema-migration backup name."""

    path = Path(database)
    return path.with_name(f"{path.name}.backup-v{version}-{stamp}")


def migration_backup_candidates(database: str | Path) -> Iterator[Path]:
    path = Path(database).expanduser().resolve()
    for candidate in path.parent.glob(f"{path.name}.backup-v*-*"):
        # SQLite may create transient sidecars when an old WAL-mode snapshot is
        # inspected.  They are not standalone migration backups and must not be
        # subjected to database-image validation.
        if candidate.name.endswith(SQLITE_SIDECAR_SUFFIXES):
            continue
        yield candidate


def migration_backup_sidecars(database: str | Path) -> Iterator[Path]:
    path = Path(database).expanduser().resolve()
    for suffix in SQLITE_SIDECAR_SUFFIXES:
        candidate = Path(f"{path}{suffix}")
        if candidate.exists() or candidate.is_symlink():
            yield candidate


def ledger_path(db_path: str | Path) -> Path:
    database = Path(db_path).expanduser().resolve(strict=False)
    return database.parent / LEDGER_NAME


def _registry_path(db_path: str | Path) -> Path:
    database = Path(db_path).expanduser().resolve(strict=False)
    return database.parent / REGISTRY_NAME


@contextmanager
def _metadata_locked(path: Path) -> Iterator[None]:
    lock = path.with_name(f".{path.name}.lock")
    with _file_locked(lock):
        yield


@contextmanager
def _file_locked(path: Path) -> Iterator[None]:
    resolved = path.expanduser().resolve(strict=False)
    held: dict[str, int] = getattr(_LIFECYCLE_LOCKS, "held", {})
    key = str(resolved)
    if key in held:
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        held[key] = 1
        _LIFECYCLE_LOCKS.held = held
        try:
            yield
        finally:
            held.pop(key, None)


def _read(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe deletion metadata: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unreadable deletion metadata: {path.name}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid deletion metadata: {path.name}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


def _empty_manifest() -> dict[str, Any]:
    return {
        "format": LEDGER_FORMAT,
        "version": LEDGER_VERSION,
        **{kind: {} for kind in DELETION_KINDS},
    }


def _manifest_at(
    path: Path,
    *,
    required: bool = False,
) -> dict[str, dict[str, dict[str, str]]]:
    if required and not path.exists():
        raise RuntimeError("required monotonic-deletion ledger is missing")
    value = _read(path, default=_empty_manifest())
    version = value.get("version")
    if value.get("format") != LEDGER_FORMAT or version not in {1, 2, 3, LEDGER_VERSION}:
        raise RuntimeError("unsupported monotonic-deletion ledger")
    result: dict[str, dict[str, dict[str, str]]] = {}
    for kind in DELETION_KINDS:
        records = value.get(kind, {})
        if not isinstance(records, dict):
            raise RuntimeError(f"invalid monotonic-deletion ledger {kind}")
        normalized: dict[str, dict[str, str]] = {}
        for digest, record in records.items():
            if not isinstance(digest, str) or len(digest) != 64 or not isinstance(record, dict):
                raise RuntimeError("invalid monotonic-deletion ledger record")
            normalized[digest] = {
                str(key): str(item) for key, item in record.items() if item is not None
            }
        result[kind] = normalized
    return result


def deletion_manifest(db_path: str | Path) -> dict[str, dict[str, dict[str, str]]]:
    return _manifest_at(ledger_path(db_path))


def deletion_records(db_path: str | Path) -> dict[str, dict[str, str]]:
    """Compatibility view of deleted raw-event records."""

    return deletion_manifest(db_path)["events"]


def backup_root_deletion_manifest(
    backup_root: str | Path,
) -> dict[str, dict[str, dict[str, str]]]:
    return _manifest_at(
        Path(backup_root).expanduser().resolve() / LEDGER_NAME,
        required=True,
    )


def backup_root_deletion_records(backup_root: str | Path) -> dict[str, dict[str, str]]:
    return backup_root_deletion_manifest(backup_root)["events"]


def _merge_manifest_at(path: Path, incoming: dict[str, dict[str, dict[str, str]]]) -> None:
    with _metadata_locked(path):
        existing = _manifest_at(path)
        for kind in DELETION_KINDS:
            existing[kind].update(incoming.get(kind, {}))
        _write(path, {"format": LEDGER_FORMAT, "version": LEDGER_VERSION, **existing})


def merge_deletion_manifest(
    db_path: str | Path,
    manifest: dict[str, dict[str, dict[str, str]]],
) -> None:
    _merge_manifest_at(ledger_path(db_path), manifest)


def merge_deletion_records(db_path: str | Path, records: dict[str, dict[str, str]]) -> None:
    merge_deletion_manifest(
        db_path,
        {
            "events": records,
            "transport_events": {},
            "learning_sources": {},
            "learner_models": {},
            "parent_memories": {},
        },
    )


def record_event_deletion(
    db_path: str | Path,
    *,
    event_hash: str,
    child_hash: str,
    deleted_at: str,
    actor_type: str,
    transport_hashes: tuple[str, ...] = (),
) -> None:
    merge_deletion_manifest(
        db_path,
        {
            "events": {
                event_hash: {
                    "child_id_hash": child_hash,
                    "deleted_at": deleted_at,
                    "actor_type": actor_type,
                }
            },
            "transport_events": {
                digest: {
                    "child_id_hash": child_hash,
                    "deleted_at": deleted_at,
                    "actor_type": actor_type,
                }
                for digest in transport_hashes
            },
            "learning_sources": {},
            "learner_models": {},
            "parent_memories": {},
        },
    )


def _database_manifest(conn: sqlite3.Connection) -> dict[str, dict[str, dict[str, str]]]:
    tables = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    result = {kind: {} for kind in DELETION_KINDS}
    if "deleted_event_tombstones" in tables:
        result["events"] = {
            str(row[0]): {
                "child_id_hash": str(row[1] or ""),
                "deleted_at": str(row[2]),
                "actor_type": str(row[3]),
            }
            for row in conn.execute(
                """SELECT event_id_hash,original_child_id_hash,deleted_at,actor_type
                   FROM deleted_event_tombstones"""
            ).fetchall()
        }
    if "deleted_transport_event_tombstones" in tables:
        result["transport_events"] = {
            str(row[0]): {
                "child_id_hash": str(row[1] or ""),
                "deleted_at": str(row[2]),
                "actor_type": str(row[3]),
            }
            for row in conn.execute(
                """SELECT transport_event_hash,original_child_id_hash,deleted_at,actor_type
                   FROM deleted_transport_event_tombstones"""
            ).fetchall()
        }
    if "evidence_source_tombstones" in tables:
        result["learning_sources"] = {
            str(row[0]): {
                "child_id_hash": event_id_hash(str(row[1])),
                "deleted_at": str(row[2]),
                "actor_type": str(row[3]),
                "reason": str(row[4]),
            }
            for row in conn.execute(
                """SELECT target_id_hash,child_id,deleted_at,actor_type,reason
                   FROM evidence_source_tombstones
                   WHERE target_kind='learning_opportunity'"""
            ).fetchall()
        }
    if "family_model_audit_log" in tables:
        result["learner_models"] = {
            str(row[0]): {
                "deleted_at": str(row[1]),
                "actor_type": str(row[2]),
            }
            for row in conn.execute(
                """SELECT target_hash,created_at,actor_type FROM family_model_audit_log
                   WHERE action='model_deleted' AND target_kind='child_model'"""
            ).fetchall()
        }
    if "parent_memory_tombstones" in tables:
        result["parent_memories"] = {
            str(row[0]): {
                "memory_key_hash": str(row[1]),
                "version": str(row[2]),
                "deleted_at": str(row[3]),
                "actor_type": str(row[4]),
            }
            for row in conn.execute(
                """SELECT memory_record_hash,memory_key_hash,version,deleted_at,actor_type
                   FROM parent_memory_tombstones"""
            ).fetchall()
        }
    return result


def sync_database_tombstones(db_path: str | Path, conn: sqlite3.Connection) -> None:
    manifest = _database_manifest(conn)
    if any(manifest[kind] for kind in DELETION_KINDS):
        merge_deletion_manifest(db_path, manifest)


def import_replay_tombstones(
    conn: sqlite3.Connection,
    manifest: dict[str, dict[str, dict[str, str]]],
) -> None:
    """Seed hash-only replay authority from the external monotonic ledger."""

    tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "deleted_event_tombstones" in tables:
        for digest, record in manifest.get("events", {}).items():
            actor = record.get("actor_type", "operator")
            if actor not in {"parent", "operator"}:
                actor = "operator"
            conn.execute(
                """INSERT OR IGNORE INTO deleted_event_tombstones(
                   tombstone_id,event_id_hash,original_child_id_hash,reason,actor_type,deleted_at
                   ) VALUES(?,?,?,?,?,?)""",
                (
                    f"evt_tomb_{digest[:24]}",
                    digest,
                    record.get("child_id_hash") or None,
                    "external_deletion_ledger",
                    actor,
                    record.get("deleted_at") or datetime.now().astimezone().isoformat(),
                ),
            )
    if "parent_memory_tombstones" in tables:
        for digest, record in manifest.get("parent_memories", {}).items():
            actor = record.get("actor_type", "operator")
            if actor not in {"parent", "operator"}:
                actor = "operator"
            try:
                version = int(record.get("version", "0"))
            except ValueError:
                version = 0
            if version < 1:
                continue
            conn.execute(
                """INSERT OR IGNORE INTO parent_memory_tombstones(
                       tombstone_id,memory_record_hash,memory_key_hash,version,actor_type,deleted_at
                   ) VALUES(?,?,?,?,?,?)""",
                (
                    f"memory_tomb_{digest[:24]}",
                    digest,
                    record.get("memory_key_hash") or event_id_hash("external-memory-key"),
                    version,
                    actor,
                    record.get("deleted_at") or datetime.now().astimezone().isoformat(),
                ),
            )
    if "deleted_transport_event_tombstones" in tables:
        for digest, record in manifest.get("transport_events", {}).items():
            actor = record.get("actor_type", "operator")
            if actor not in {"parent", "operator"}:
                actor = "operator"
            conn.execute(
                """INSERT OR IGNORE INTO deleted_transport_event_tombstones(
                   tombstone_id,transport_event_hash,original_child_id_hash,reason,actor_type,deleted_at
                   ) VALUES(?,?,?,?,?,?)""",
                (
                    f"transport_tomb_{digest[:24]}",
                    digest,
                    record.get("child_id_hash") or None,
                    "external_deletion_ledger",
                    actor,
                    record.get("deleted_at") or datetime.now().astimezone().isoformat(),
                ),
            )
    if "evidence_source_tombstones" in tables and "children" in tables:
        children = {
            event_id_hash(str(row[0])): str(row[0])
            for row in conn.execute("SELECT id FROM children").fetchall()
        }
        for digest, record in manifest.get("learning_sources", {}).items():
            child_id = children.get(record.get("child_id_hash", ""))
            if child_id is None:
                continue
            actor = record.get("actor_type", "operator")
            if actor not in {"parent", "operator"}:
                actor = "operator"
            conn.execute(
                """INSERT OR IGNORE INTO evidence_source_tombstones(
                   tombstone_id,child_id,target_kind,target_id_hash,reason,actor_type,deleted_at
                   ) VALUES(?,?,'learning_opportunity',?,?,?,?)""",
                (
                    f"source_tomb_{digest[:24]}",
                    child_id,
                    digest,
                    record.get("reason") or "external_deletion_ledger",
                    actor,
                    record.get("deleted_at") or datetime.now().astimezone().isoformat(),
                ),
            )


def assert_no_resurrected_events(db_path: str | Path, conn: sqlite3.Connection) -> None:
    manifest = deletion_manifest(db_path)
    if manifest["events"]:
        for row in conn.execute("SELECT id FROM events").fetchall():
            if event_id_hash(str(row[0])) in manifest["events"]:
                raise RuntimeError("database contains an event recorded in the external deletion ledger")
    if manifest["learning_sources"]:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "learning_opportunities" in tables:
            for row in conn.execute("SELECT opportunity_id FROM learning_opportunities").fetchall():
                if event_id_hash(str(row[0])) in manifest["learning_sources"]:
                    raise RuntimeError(
                        "database contains a learning source recorded in the external deletion ledger"
                    )
    if manifest["transport_events"]:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "transport_receipts" in tables:
            for row in conn.execute(
                "SELECT transport,external_event_id FROM transport_receipts"
            ).fetchall():
                if event_id_hash(f"{row[0]}\0{row[1]}") in manifest["transport_events"]:
                    raise RuntimeError(
                        "database contains a transport event recorded in the external deletion ledger"
                    )
    if manifest["parent_memories"]:
        for row in conn.execute(
            """SELECT memory_key,version FROM parent_memories
               WHERE status!='forgotten' OR value!='' OR provenance!='forgotten'
                  OR why_saved!='forgotten'"""
        ).fetchall():
            if event_id_hash(f"{row[0]}\0{row[1]}") in manifest["parent_memories"]:
                raise RuntimeError(
                    "database contains parent memory recorded in the external deletion ledger"
                )
        if _has_deleted_memory_inspection(conn, manifest["parent_memories"]):
            raise RuntimeError(
                "database contains a parent-memory inspection recorded before erasure"
            )
        if _has_deleted_memory_delivery(conn, manifest["parent_memories"]):
            raise RuntimeError(
                "database contains a parent-memory delivery recorded before erasure"
            )
    import_replay_tombstones(conn, manifest)


def _registry_value(
    path: Path,
    *,
    format_name: str,
    list_name: str,
    required: bool = False,
) -> list[str]:
    if required and not path.exists():
        raise RuntimeError(f"required {format_name} registry is missing")
    value = _read(path, default={"format": format_name, "version": 1, list_name: []})
    if value.get("format") != format_name or value.get("version") != 1:
        raise RuntimeError(f"unsupported {format_name} registry")
    raw = value.get(list_name)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise RuntimeError(f"invalid {format_name} registry")
    return [str(item) for item in raw]


def initialize_backup_root_metadata(backup_root: str | Path) -> None:
    """Create required empty registries and deletion authority for a new root."""

    root = Path(backup_root).expanduser().resolve()
    registries = (
        (
            root / BACKUP_SOURCES_NAME,
            {
                "format": "curiosity-backup-source-registry",
                "version": 1,
                "databases": [],
            },
        ),
        (
            root / MANAGED_RESTORES_NAME,
            {
                "format": "curiosity-managed-restore-registry",
                "version": 1,
                "restores": [],
            },
        ),
        (root / LEDGER_NAME, _empty_manifest()),
    )
    created: list[Path] = []
    try:
        for path, value in registries:
            with _metadata_locked(path):
                if path.exists():
                    raise RuntimeError("new backup root authority metadata already exists")
                _write(path, value)
                created.append(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def _backup_root_database_identity(root: Path, *, required: bool) -> str | None:
    path = root / BACKUP_IDENTITY_NAME
    if not path.exists():
        if required:
            raise RuntimeError("required backup-root database identity is missing")
        return None
    value = _read(path, default={})
    identity = value.get("database_identity")
    if (
        value.get("format") != "curiosity-backup-identity"
        or value.get("version") != 1
        or not isinstance(identity, str)
        or not 16 <= len(identity) <= 128
    ):
        raise RuntimeError("backup-root database identity is invalid")
    return identity


def backup_root_database_identity(backup_root: str | Path) -> str:
    root = Path(backup_root).expanduser().resolve()
    identity = _backup_root_database_identity(root, required=True)
    assert identity is not None
    return identity


def _bind_backup_root_database_identity(root: Path, database_identity: str) -> None:
    path = root / BACKUP_IDENTITY_NAME
    with _metadata_locked(path):
        existing = _backup_root_database_identity(root, required=False)
        if existing is not None and existing != database_identity:
            raise RuntimeError("backup root belongs to a different family database")
        if existing is None:
            _write(
                path,
                {
                    "format": "curiosity-backup-identity",
                    "version": 1,
                    "database_identity": database_identity,
                },
            )


def _backup_root_registration_pending(root: Path) -> dict[str, str] | None:
    path = root / BACKUP_REGISTRATION_PENDING_NAME
    if not path.exists():
        return None
    value = _read(path, default={})
    database_identity = value.get("database_identity")
    database_path = value.get("database_path")
    if (
        value.get("format") != "curiosity-backup-registration-pending"
        or value.get("version") != 1
        or not isinstance(database_identity, str)
        or not 16 <= len(database_identity) <= 128
        or not isinstance(database_path, str)
        or not Path(database_path).is_absolute()
    ):
        raise RuntimeError("backup-root pending registration is invalid")
    return {
        "database_identity": database_identity,
        "database_path": database_path,
    }


def mark_backup_root_registration_pending(
    backup_root: str | Path,
    database: str | Path,
    database_identity: str,
) -> None:
    """Publish a fail-closed barrier before a root can become restore-authorizing."""

    root = Path(backup_root).expanduser().resolve()
    database_path = Path(database).expanduser().resolve()
    path = root / BACKUP_REGISTRATION_PENDING_NAME
    with _metadata_locked(path):
        existing = _backup_root_registration_pending(root)
        expected = {
            "database_identity": database_identity,
            "database_path": str(database_path),
        }
        if existing is not None and existing != expected:
            raise RuntimeError("backup root has a different pending family registration")
        _write(
            path,
            {
                "format": "curiosity-backup-registration-pending",
                "version": 1,
                **expected,
            },
        )


def assert_backup_root_registration_complete(backup_root: str | Path) -> None:
    root = Path(backup_root).expanduser().resolve()
    pending = _backup_root_registration_pending(root)
    sources = _registered_backup_sources(root)
    identity = _backup_root_database_identity(root, required=False)
    if pending is not None or not sources or identity is None:
        raise RuntimeError("backup root registration is incomplete")


def _clear_backup_root_registration_pending(root: Path) -> None:
    path = root / BACKUP_REGISTRATION_PENDING_NAME
    with _metadata_locked(path):
        _backup_root_registration_pending(root)
        path.unlink(missing_ok=True)


def register_backup_root(
    db_path: str | Path,
    backup_root: str | Path,
    *,
    _test_hook: Callable[[str], None] | None = None,
    _before_complete: Callable[[], None] | None = None,
) -> None:
    path = _registry_path(db_path)
    database = Path(db_path).expanduser().resolve()
    root = Path(backup_root).expanduser().resolve()
    database_identity = _assert_managed_database_image_readable(database, wal_aware=True)
    root_identity = _backup_root_database_identity(root, required=False)
    if root_identity is not None and root_identity != database_identity:
        raise RuntimeError("backup root belongs to a different family database")
    sources_path = root / BACKUP_SOURCES_NAME
    # Preflight root ownership without publishing an identity. A root-side
    # pending barrier first prevents restore; the database-local pointer then
    # makes the root discoverable before identity publication.
    with _metadata_locked(sources_path):
        sources = {
            str(Path(item).expanduser().resolve())
            for item in _registry_value(
                sources_path,
                format_name="curiosity-backup-source-registry",
                list_name="databases",
                required=True,
            )
        }
        if sources and str(database) not in sources:
            raise RuntimeError("backup root is already bound to a different family database")
    registration_complete = str(database) in sources and root_identity == database_identity
    if not registration_complete or _backup_root_registration_pending(root) is not None:
        mark_backup_root_registration_pending(root, database, database_identity)
        if _test_hook is not None:
            _test_hook("after_backup_registration_pending")
    with _metadata_locked(path):
        roots = {
            str(Path(item).expanduser().resolve())
            for item in _registry_value(
                path,
                format_name="curiosity-backup-root-registry",
                list_name="roots",
            )
        }
        roots.add(str(root))
        _write(
            path,
            {
                "format": "curiosity-backup-root-registry",
                "version": 1,
                "roots": sorted(roots),
            },
        )
    if _test_hook is not None:
        _test_hook("after_local_backup_registration")
    _bind_backup_root_database_identity(root, database_identity)
    if _test_hook is not None:
        _test_hook("after_backup_identity_binding")
    with _metadata_locked(sources_path):
        sources = {
            str(Path(item).expanduser().resolve())
            for item in _registry_value(
                sources_path,
                format_name="curiosity-backup-source-registry",
                list_name="databases",
                required=True,
            )
        }
        if sources and str(database) not in sources:
            raise RuntimeError("backup root is already bound to a different family database")
        sources.add(str(database))
        _write(
            sources_path,
            {
                "format": "curiosity-backup-source-registry",
                "version": 1,
                "databases": sorted(sources),
            },
        )
    sync_backup_root_deletions(database, root)
    if _before_complete is not None:
        _before_complete()
    _clear_backup_root_registration_pending(root)


def _recover_pending_backup_registration(database: Path, root: Path) -> None:
    """Complete a local-first registration left by abrupt process death."""

    sources = set(_registered_backup_sources(root))
    pending = _backup_root_registration_pending(root)
    if database in sources and pending is None:
        return
    if sources - {database}:
        raise RuntimeError("backup root is not reciprocally authorized for this family database")
    database_identity = _assert_managed_database_image_readable(database, wal_aware=True)
    root_identity = _backup_root_database_identity(root, required=False)
    if root_identity is not None and database_identity != root_identity:
        raise RuntimeError("backup root belongs to a different family database")
    mark_backup_root_registration_pending(root, database, database_identity)
    _bind_backup_root_database_identity(root, database_identity)
    path = root / BACKUP_SOURCES_NAME
    with _metadata_locked(path):
        current = set(_registered_backup_sources(root))
        if current and database not in current:
            raise RuntimeError("backup root is not reciprocally authorized for this family database")
        current.add(database)
        _write(
            path,
            {
                "format": "curiosity-backup-source-registry",
                "version": 1,
                "databases": sorted(str(item) for item in current),
            },
        )
    sync_backup_root_deletions(database, root)
    # Schema-v26 registration is not complete until the root has joined the
    # external research-authority peer set as well.  Import locally to avoid a
    # module cycle: the authority layer authenticates roots through this file.
    from .research_authority_ledger import ledger_path as research_authority_path
    from .research_authority_ledger import sync_registered_backup_authority

    if research_authority_path(database).exists():
        sync_registered_backup_authority(
            database,
            allow_registration_pending=True,
        )
    _clear_backup_root_registration_pending(root)


def prepare_restored_database_registration(
    staged_db_path: str | Path,
    final_db_path: str | Path,
    backup_root: str | Path,
) -> None:
    """Prebind an active disaster restore to its root before publication."""

    staged = Path(staged_db_path).expanduser().resolve()
    final = Path(final_db_path).expanduser().resolve()
    root = Path(backup_root).expanduser().resolve()
    root_identity = backup_root_database_identity(root)
    restored_identity = _assert_managed_database_image_readable(staged, wal_aware=True)
    if restored_identity != root_identity:
        raise RuntimeError("restored database belongs to a different family database")
    sources_path = root / BACKUP_SOURCES_NAME
    with _metadata_locked(sources_path):
        sources = {
            Path(item).expanduser().resolve()
            for item in _registry_value(
                sources_path,
                format_name="curiosity-backup-source-registry",
                list_name="databases",
                required=True,
            )
        }
        live_others = [source for source in sources if source != final and source.is_file()]
        if live_others:
            raise RuntimeError("backup root is already bound to a different live family database")
        _write(
            sources_path,
            {
                "format": "curiosity-backup-source-registry",
                "version": 1,
                "databases": [str(final)],
            },
        )
    registry = _registry_path(staged)
    with _metadata_locked(registry):
        _write(
            registry,
            {
                "format": "curiosity-backup-root-registry",
                "version": 1,
                "roots": [str(root)],
            },
        )


def _registered_backup_roots(db_path: str | Path) -> list[Path]:
    path = _registry_path(db_path)
    return [
        Path(item).expanduser().resolve()
        for item in _registry_value(
            path,
            format_name="curiosity-backup-root-registry",
            list_name="roots",
        )
    ]


def _registered_backup_sources(backup_root: str | Path) -> list[Path]:
    root = Path(backup_root).expanduser().resolve()
    path = root / BACKUP_SOURCES_NAME
    return [
        Path(item).expanduser().resolve()
        for item in _registry_value(
            path,
            format_name="curiosity-backup-source-registry",
            list_name="databases",
            required=True,
        )
    ]


def _has_symlink_component(path: Path) -> bool:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _assert_backup_root(root: Path) -> None:
    if _has_symlink_component(root) or not root.is_dir():
        raise RuntimeError("registered backup root is unavailable or unsafe")
    marker = root / ".curiosity-backup-root.json"
    expected = {"format": "curiosity-family-backup", "format_version": 2}
    if _read(marker, default={}) != expected:
        raise RuntimeError("registered backup root marker is missing or incompatible")
    # These registries are part of the root's deletion authority. Treat their
    # disappearance as damage, never as an empty registry that forgets copies.
    sources = _registered_backup_sources(root)
    _managed_restore_entries(root)
    backup_root_deletion_manifest(root)
    _backup_root_registration_pending(root)
    identity = _backup_root_database_identity(root, required=bool(sources))
    if sources and identity is None:  # pragma: no cover - required=True owns this branch
        raise RuntimeError("registered backup root has no family identity")


def _assert_reciprocal_backup_authority(database: Path, root: Path) -> None:
    if database not in set(_registered_backup_sources(root)):
        raise RuntimeError("backup root registry is not reciprocally authorized")
    database_identity = _assert_managed_database_image_readable(database, wal_aware=True)
    if database_identity != backup_root_database_identity(root):
        raise RuntimeError("backup root belongs to a different family database")


def _assert_registered_restore_entries_safe(root: Path) -> None:
    for entry in _managed_restore_entries(root):
        state = entry.get("state", "published")
        if state not in MANAGED_RESTORE_STATES:
            raise RuntimeError("registered managed restore has an invalid state")
        raw_target = Path(entry.get("path", "")).expanduser()
        if not raw_target.is_absolute() or _has_symlink_component(raw_target):
            raise RuntimeError("registered managed restore path is unsafe")
        target = raw_target.resolve()
        if target == root or target.is_relative_to(root):
            raise RuntimeError("registered managed restore path crosses the backup root")
        if not target.exists():
            continue
        if not target.is_dir():
            raise RuntimeError("registered managed restore target is not a directory")
        if state == "pending" and not any(target.iterdir()):
            continue
        expected = managed_restore_marker(
            root,
            snapshot=entry.get("snapshot", ""),
            snapshot_created_at=entry.get("snapshot_created_at", ""),
            database_identity=entry.get("database_identity", ""),
        )
        if _read(target / RESTORE_MARKER_NAME, default={}) != expected:
            raise RuntimeError("registered managed restore marker is missing or incompatible")


def authenticated_research_authority_targets(
    db_path: str | Path,
    *,
    claimed_roots: tuple[str | Path, ...] = (),
    allow_registration_pending: bool = False,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Authenticate every root and published database sharing research authority.

    Callers hold ``managed_data_lifecycle`` while using the returned paths.
    A database is authorized either as the reciprocal live source or as an
    authenticated managed restore registered by that root.
    """

    database = Path(db_path).expanduser().resolve()
    registered = set(_registered_backup_roots(database))
    roots = registered | {
        Path(item).expanduser().resolve() for item in claimed_roots
    }
    if not roots:
        return (), ()
    database_identity = _assert_managed_database_image_readable(database, wal_aware=True)
    restores: set[Path] = set()
    database_authorized = False
    for root in sorted(roots, key=str):
        _assert_backup_root(root)
        if (
            _backup_root_registration_pending(root) is not None
            and not allow_registration_pending
        ):
            raise RuntimeError("backup root registration is incomplete")
        if backup_root_database_identity(root) != database_identity:
            raise RuntimeError("backup root belongs to a different family database")
        sources = set(_registered_backup_sources(root))
        entries = _managed_restore_entries(root)
        authorized_restore = any(
            Path(entry.get("path", "")).expanduser().resolve() / "data" / "curiosity.db"
            == database
            and entry.get("state", "published") in {"staging", "published"}
            and entry.get("database_identity") == database_identity
            for entry in entries
        )
        database_authorized = database_authorized or database in sources or authorized_restore
        _assert_registered_restore_entries_safe(root)
        for entry in entries:
            if (
                entry.get("state", "published") != "published"
                or entry.get("database_identity") != database_identity
            ):
                continue
            target = Path(entry.get("path", "")).expanduser().resolve()
            target_database = target / "data" / "curiosity.db"
            if target_database == database or not target.exists():
                continue
            _assert_managed_database_image_readable(
                target_database,
                expected_database_identity=database_identity,
                wal_aware=True,
            )
            restores.add(target_database)
    if not database_authorized:
        raise RuntimeError(
            "database is not authorized by its research backup-root membership"
        )
    return tuple(sorted(roots, key=str)), tuple(sorted(restores, key=str))


def _open_managed_database(path: Path, *, wal_aware: bool) -> sqlite3.Connection:
    query = "mode=ro" if wal_aware else "mode=ro&immutable=1"
    return sqlite3.connect(f"{path.resolve().as_uri()}?{query}", uri=True)


def _assert_managed_database_image_readable(
    path: Path,
    *,
    expected_database_identity: str | None = None,
    wal_aware: bool = False,
    require_curiosity_schema: bool = True,
    require_database_identity: bool = True,
    allow_legacy_without_database_identity: bool = False,
) -> str | None:
    """Require one managed copy to be an authentic, readable Curiosity database."""

    raw = path.expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if _has_symlink_component(raw) or not raw.is_file():
        raise RuntimeError("managed database image is missing or unsafe")
    database = raw.resolve()
    try:
        conn = _open_managed_database(database, wal_aware=wal_aware)
        try:
            check = [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                ).fetchall()
            }
            identity_required = require_database_identity and not (
                allow_legacy_without_database_identity
                and version < FAMILY_DATABASE_IDENTITY_SCHEMA_VERSION
            )
            required_tables: set[str] = set()
            if identity_required:
                required_tables.add("family_database_identity")
            if require_curiosity_schema:
                required_tables.update({"children", "events"})
            if not required_tables.issubset(tables):
                raise RuntimeError("managed database image is not a Curiosity database")
            identities = (
                conn.execute(
                    "SELECT singleton,database_id FROM family_database_identity"
                ).fetchall()
                if "family_database_identity" in tables
                else []
            )
        finally:
            conn.close()
    except RuntimeError:
        raise
    except (OSError, sqlite3.DatabaseError, TypeError, ValueError) as exc:
        raise RuntimeError("managed database image is unreadable") from exc
    if check != ["ok"]:
        raise RuntimeError("managed database image failed its integrity check")
    if version < 0:
        raise RuntimeError("managed database image has an invalid schema version")
    if not identities and not identity_required:
        return None
    if len(identities) != 1 or int(identities[0][0]) != 1:
        raise RuntimeError("managed database image has an invalid family identity")
    database_identity = str(identities[0][1])
    if not 16 <= len(database_identity) <= 128:
        raise RuntimeError("managed database image has an invalid family identity")
    if expected_database_identity is not None and database_identity != expected_database_identity:
        raise RuntimeError("managed database image belongs to a different family database")
    return database_identity


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_managed_snapshot_database(
    candidate: Path,
    *,
    database_identity: str,
) -> tuple[Path, int]:
    """Authenticate a static snapshot database against its immutable manifest."""

    manifest = _read(candidate / "manifest.json", default={})
    format_version = manifest.get("format_version")
    if manifest.get("format") != "curiosity-family-backup" or format_version not in {1, 2}:
        raise RuntimeError("managed backup snapshot manifest is missing or incompatible")
    if format_version == 2 and manifest.get("database_identity") != database_identity:
        raise RuntimeError("managed backup snapshot belongs to a different family database")
    if format_version == 1 and manifest.get("database_identity") not in {None, database_identity}:
        raise RuntimeError("legacy backup snapshot belongs to a different family database")
    if candidate.name.startswith("curiosity-backup-") and manifest.get("snapshot_id") != candidate.name:
        raise RuntimeError("managed backup snapshot identity does not match its directory")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("managed backup snapshot file manifest is invalid")
    database_entries = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("path") == "data/curiosity.db"
    ]
    if len(database_entries) != 1:
        raise RuntimeError("managed backup snapshot database manifest is invalid")
    database = candidate / "data" / "curiosity.db"
    _assert_managed_database_image_readable(
        database,
        expected_database_identity=(database_identity if manifest.get("database_identity") else None),
        require_database_identity=bool(manifest.get("database_identity")),
    )
    entry = database_entries[0]
    if (
        not isinstance(entry.get("bytes"), int)
        or int(entry["bytes"]) != database.stat().st_size
        or not isinstance(entry.get("sha256"), str)
        or _file_sha256(database) != entry["sha256"]
    ):
        raise RuntimeError("managed backup snapshot database does not match its manifest")
    return database, int(format_version)


def _assert_managed_database_images_safe(database: Path, roots: set[Path]) -> None:
    """Validate every deletion-authoritative database copy before live mutation."""

    migration_backups = list(migration_backup_candidates(database))
    if not migration_backups and not roots:
        return
    database_identity = _assert_managed_database_image_readable(database, wal_aware=True)
    for candidate in migration_backups:
        _assert_managed_database_image_readable(
            candidate,
            expected_database_identity=database_identity,
            wal_aware=True,
            require_curiosity_schema=False,
            allow_legacy_without_database_identity=True,
        )

    for root in sorted(roots, key=str):
        for candidate in sorted(root.iterdir(), key=lambda item: item.name):
            if not (
                candidate.name.startswith("curiosity-backup-")
                or candidate.name.startswith(".partial-")
            ):
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise RuntimeError("managed backup snapshot is missing or unsafe")
            _assert_managed_snapshot_database(candidate, database_identity=database_identity)

        for entry in _managed_restore_entries(root):
            raw_target = Path(entry.get("path", "")).expanduser()
            target = raw_target.resolve()
            if not target.exists():
                continue
            # A pending empty directory is only a crash-safe publication
            # reservation. It contains no family bytes and is removed during
            # reconciliation or the next restore retry.
            if entry.get("state", "published") == "pending" and not any(target.iterdir()):
                continue
            if entry.get("database_identity") != database_identity:
                raise RuntimeError("registered managed restore belongs to a different family database")
            _assert_managed_database_image_readable(
                target / "data" / "curiosity.db",
                expected_database_identity=database_identity,
                wal_aware=True,
            )


@contextmanager
def managed_data_lifecycle(
    db_path: str | Path,
    *,
    extra_backup_roots: tuple[str | Path, ...] = (),
) -> Iterator[None]:
    """Serialize mutations with snapshot publication across processes."""

    database = Path(db_path).expanduser().resolve(strict=False)
    with ExitStack() as stack:
        stack.enter_context(_file_locked(database.parent / LIFECYCLE_LOCK_NAME))
        registered = set(_registered_backup_roots(database))
        roots = registered | {
            Path(item).expanduser().resolve() for item in extra_backup_roots
        }
        for root in sorted(roots, key=str):
            _assert_backup_root(root)
            stack.enter_context(_file_locked(root / LIFECYCLE_LOCK_NAME))
            _assert_backup_root(root)
            _assert_registered_restore_entries_safe(root)
            if root in registered:
                _recover_pending_backup_registration(database, root)
                _assert_reciprocal_backup_authority(database, root)
        _assert_managed_database_images_safe(database, registered)
        yield


@contextmanager
def backup_root_lifecycle(backup_root: str | Path) -> Iterator[None]:
    root = Path(backup_root).expanduser().resolve()
    with _file_locked(root / LIFECYCLE_LOCK_NAME):
        yield


@contextmanager
def backup_roots_lifecycle(backup_roots: tuple[str | Path, ...]) -> Iterator[None]:
    """Lock an authenticated peer-root set in the shared canonical order."""

    roots = sorted(
        {Path(item).expanduser().resolve() for item in backup_roots},
        key=str,
    )
    with ExitStack() as stack:
        for root in roots:
            _assert_backup_root(root)
            stack.enter_context(_file_locked(root / LIFECYCLE_LOCK_NAME))
            _assert_backup_root(root)
        yield


def sync_backup_root_deletions(db_path: str | Path, backup_root: str | Path) -> None:
    database = Path(db_path).expanduser().resolve()
    root = Path(backup_root).expanduser().resolve()
    _assert_backup_root(root)
    _assert_reciprocal_backup_authority(database, root)
    manifest = deletion_manifest(database)
    if any(manifest[kind] for kind in DELETION_KINDS):
        _merge_manifest_at(root / LEDGER_NAME, manifest)


def sync_registered_backup_sources(backup_root: str | Path) -> None:
    """Reconcile committed in-database tombstones before backup use."""

    root = Path(backup_root).expanduser().resolve()
    for database in _registered_backup_sources(root):
        if database.is_symlink() or not database.is_file():
            continue
        _assert_reciprocal_backup_authority(database, root)
        try:
            conn = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
            try:
                manifest = _database_manifest(conn)
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("registered backup source database is unreadable") from exc
        if any(manifest[kind] for kind in DELETION_KINDS):
            merge_deletion_manifest(database, manifest)
            _merge_manifest_at(root / LEDGER_NAME, manifest)


def _database_contains_hash(
    path: Path,
    table: str,
    column: str,
    digest: str,
    *,
    wal_aware: bool,
) -> bool:
    if _has_symlink_component(path) or not path.is_file():
        raise RuntimeError("managed database image is missing or unsafe")
    try:
        conn = _open_managed_database(path, wal_aware=wal_aware)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                return False
            return any(
                event_id_hash(str(row[0])) == digest
                for row in conn.execute(f'SELECT "{column}" FROM "{table}"')
            )
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError("managed database image is unreadable") from exc


_MODEL_TABLES = (
    "research_proactivity_candidates",
    "research_uncertainty_traces",
    "research_track_control_events",
    "research_track_controls",
    "personalization_feedback_signals",
    "personalization_decisions",
    "personalization_control_events",
    "evidence_corrections",
    "evidence_annotations",
    "learning_opportunities",
    "learning_episode_plans",
    "child_reflection_responses",
    "derived_claims_v2",
    "learner_state_projections",
    "interest_projections",
    "observations",
    "claims",
    "nodes",
    "edges",
    "school_signals",
    "opportunities",
    "evidence",
    "episodes",
    "projection_runs",
    "context_corrections",
)


def database_deletion_conflicts(
    path: str | Path,
    manifest: dict[str, dict[str, dict[str, str]]],
    *,
    snapshot_created_at: str | None = None,
    wal_aware: bool = False,
    expected_database_identity: str | None = None,
    require_curiosity_schema: bool = True,
    require_database_identity: bool = True,
    allow_legacy_without_database_identity: bool = False,
    current_database: bool = False,
) -> list[str]:
    """Return deleted-data categories still present in one database image."""

    raw_database = Path(path).expanduser()
    _assert_managed_database_image_readable(
        raw_database,
        expected_database_identity=expected_database_identity,
        wal_aware=wal_aware,
        require_curiosity_schema=require_curiosity_schema,
        require_database_identity=require_database_identity,
        allow_legacy_without_database_identity=allow_legacy_without_database_identity,
    )
    database = raw_database.resolve()
    conflicts: list[str] = []
    for digest in manifest["events"]:
        if _database_contains_hash(database, "events", "id", digest, wal_aware=wal_aware):
            conflicts.append("event")
            break
    for digest in manifest["learning_sources"]:
        if _database_contains_hash(
            database,
            "learning_opportunities",
            "opportunity_id",
            digest,
            wal_aware=wal_aware,
        ):
            conflicts.append("learning_source")
            break
    if manifest.get("transport_events"):
        try:
            conn = _open_managed_database(database, wal_aware=wal_aware)
            try:
                tables = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "transport_receipts" in tables and any(
                    event_id_hash(f"{row[0]}\0{row[1]}") in manifest["transport_events"]
                    for row in conn.execute(
                        "SELECT transport,external_event_id FROM transport_receipts"
                    )
                ):
                    conflicts.append("transport_event")
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("managed database image is unreadable") from exc
    if manifest["learner_models"]:
        try:
            conn = _open_managed_database(database, wal_aware=wal_aware)
            try:
                tables = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                children = (
                    [str(row[0]) for row in conn.execute("SELECT id FROM children").fetchall()]
                    if "children" in tables
                    else []
                )
                for child_id in children:
                    record = manifest["learner_models"].get(event_id_hash(child_id))
                    if not record:
                        continue
                    deleted_at = (
                        datetime.fromisoformat(record["deleted_at"])
                        if record.get("deleted_at")
                        else None
                    )
                    predates_deletion = True
                    if snapshot_created_at and deleted_at is not None:
                        predates_deletion = datetime.fromisoformat(snapshot_created_at) <= deleted_at
                    # A historical image containing the child predates the
                    # whole-model deletion unless its timestamp proves
                    # otherwise. This includes control-only and audit-only
                    # state, and the default-enabled absence of a control row.
                    if predates_deletion and not current_database:
                        conflicts.append("learner_model")
                        break
                    # A static image created after the latest whole-model
                    # deletion represents an explicitly restarted epoch.
                    if not predates_deletion and not current_database:
                        continue
                    control = (
                        conn.execute(
                            """SELECT collection_enabled,updated_at
                               FROM learner_model_controls WHERE child_id=?""",
                            (child_id,),
                        ).fetchone()
                        if "learner_model_controls" in tables
                        else None
                    )
                    # The append-only resume audit establishes a new live model
                    # epoch even if the parent later pauses collection again.
                    # Current control state is therefore not epoch evidence.
                    if current_database and deleted_at is not None and "family_model_audit_log" in tables:
                        resumed_after_deletion = any(
                            datetime.fromisoformat(str(row[0])) > deleted_at
                            for row in conn.execute(
                                """SELECT created_at FROM family_model_audit_log
                                   WHERE child_id=? AND action='collection_resumed'
                                     AND target_kind='child_model'""",
                                (child_id,),
                            ).fetchall()
                        )
                        if resumed_after_deletion:
                            continue
                    present = any(
                        table in tables
                        and conn.execute(
                            f'SELECT 1 FROM "{table}" WHERE child_id=? LIMIT 1',
                            (child_id,),
                        ).fetchone()
                        for table in _MODEL_TABLES
                    )
                    if not present and "events" in tables:
                        present = any(
                            table in tables
                            and conn.execute(
                                f'''SELECT 1 FROM "{table}" r JOIN events e ON e.id=r.event_id
                                    WHERE e.child_id=? LIMIT 1''',
                                (child_id,),
                            ).fetchone()
                            for table in ("graph_effects", "runs")
                        )
                    if control is None or bool(control[0]):
                        present = True
                    if "family_model_audit_log" in tables:
                        for audit in conn.execute(
                            """SELECT action,target_kind,created_at FROM family_model_audit_log
                               WHERE child_id=?""",
                            (child_id,),
                        ).fetchall():
                            if audit[0] == "model_deleted" and audit[1] == "child_model":
                                continue
                            if deleted_at is None or datetime.fromisoformat(str(audit[2])) <= deleted_at:
                                present = True
                                break
                    if present:
                        conflicts.append("learner_model")
                        break
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("managed database image is unreadable") from exc
    if manifest["parent_memories"]:
        try:
            conn = _open_managed_database(database, wal_aware=wal_aware)
            try:
                tables = {
                    str(row[0])
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "parent_memories" in tables and any(
                    event_id_hash(f"{row[0]}\0{row[1]}") in manifest["parent_memories"]
                    for row in conn.execute(
                        """SELECT memory_key,version FROM parent_memories
                           WHERE status!='forgotten' OR value!='' OR provenance!='forgotten'
                              OR why_saved!='forgotten'"""
                    )
                ):
                    conflicts.append("parent_memory")
                if _has_deleted_memory_inspection(conn, manifest["parent_memories"]):
                    conflicts.append("parent_memory_inspection")
                if _has_deleted_memory_delivery(conn, manifest["parent_memories"]):
                    conflicts.append("parent_memory_delivery")
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            raise RuntimeError("managed database image is unreadable") from exc
    return conflicts


def _managed_restore_entries(root: Path) -> list[dict[str, str]]:
    path = root / MANAGED_RESTORES_NAME
    if not path.exists():
        raise RuntimeError("required managed-restore registry is missing")
    value = _read(
        path,
        default={"format": "curiosity-managed-restore-registry", "version": 1, "restores": []},
    )
    if value.get("format") != "curiosity-managed-restore-registry" or value.get("version") != 1:
        raise RuntimeError("unsupported managed-restore registry")
    entries = value.get("restores")
    if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
        raise RuntimeError("invalid managed-restore registry")
    return [
        {str(key): str(item) for key, item in entry.items() if item is not None}
        for entry in entries
    ]


def register_managed_restore(
    backup_root: str | Path,
    target: str | Path,
    *,
    snapshot: str,
    snapshot_created_at: str,
    database_identity: str,
    state: str = "published",
) -> None:
    if state not in MANAGED_RESTORE_STATES:
        raise ValueError("invalid managed restore state")
    root = Path(backup_root).expanduser().resolve()
    if database_identity != backup_root_database_identity(root):
        raise RuntimeError("managed restore belongs to a different family database")
    destination = Path(target).expanduser().resolve()
    path = root / MANAGED_RESTORES_NAME
    with _metadata_locked(path):
        entries = _managed_restore_entries(root)
        entries = [entry for entry in entries if entry.get("path") != str(destination)]
        entries.append(
            {
                "path": str(destination),
                "snapshot": snapshot,
                "snapshot_created_at": snapshot_created_at,
                "database_identity": database_identity,
                "state": state,
            }
        )
        _write(
            path,
            {
                "format": "curiosity-managed-restore-registry",
                "version": 1,
                "restores": sorted(entries, key=lambda item: item["path"]),
            },
        )


def unregister_managed_restore(backup_root: str | Path, target: str | Path) -> None:
    root = Path(backup_root).expanduser().resolve()
    destination = Path(target).expanduser().resolve()
    path = root / MANAGED_RESTORES_NAME
    with _metadata_locked(path):
        entries = _managed_restore_entries(root)
        retained = [entry for entry in entries if entry.get("path") != str(destination)]
        if retained != entries:
            _write(
                path,
                {
                    "format": "curiosity-managed-restore-registry",
                    "version": 1,
                    "restores": retained,
                },
            )


def managed_restore_marker(
    backup_root: str | Path,
    *,
    snapshot: str,
    snapshot_created_at: str,
    database_identity: str,
) -> dict[str, str]:
    root = Path(backup_root).expanduser().resolve()
    return {
        "format": "curiosity-managed-restore",
        "backup_root_hash": event_id_hash(str(root)),
        "snapshot": snapshot,
        "snapshot_created_at": snapshot_created_at,
        "database_identity": database_identity,
    }


def recover_interrupted_restore_paths(
    backup_root: str | Path,
    target: str | Path,
    *,
    snapshot: str,
) -> int:
    """Remove authenticated staging and empty reservations left by a killed restore."""

    root = Path(backup_root).expanduser().resolve()
    destination = Path(target).expanduser().resolve()
    path = root / MANAGED_RESTORES_NAME
    removed = 0
    with _metadata_locked(path):
        entries = _managed_restore_entries(root)
        retained: list[dict[str, str]] = []
        for entry in entries:
            state = entry.get("state", "published")
            raw_candidate = Path(entry.get("path", "")).expanduser()
            candidate = raw_candidate.resolve()
            is_target_reservation = (
                candidate == destination
                and state == "pending"
                and entry.get("snapshot") == snapshot
            )
            is_target_staging = (
                candidate.parent == destination.parent
                and candidate.name.startswith(f".{destination.name}.partial-")
                and state == "staging"
                and entry.get("snapshot") == snapshot
            )
            if not (is_target_reservation or is_target_staging):
                retained.append(entry)
                continue
            if not raw_candidate.is_absolute() or _has_symlink_component(raw_candidate):
                raise RuntimeError("interrupted restore path is unsafe")
            if candidate == root or candidate.is_relative_to(root):
                raise RuntimeError("interrupted restore path crosses the backup root")
            if not candidate.exists():
                removed += 1
                continue
            if not candidate.is_dir():
                raise RuntimeError("interrupted restore path is not a directory")
            if is_target_reservation:
                if any(candidate.iterdir()):
                    retained.append(entry)
                    continue
                candidate.rmdir()
            else:
                expected = managed_restore_marker(
                    root,
                    snapshot=str(entry.get("snapshot", "")),
                    snapshot_created_at=str(entry.get("snapshot_created_at", "")),
                    database_identity=str(entry.get("database_identity", "")),
                )
                if _read(candidate / RESTORE_MARKER_NAME, default={}) != expected:
                    raise RuntimeError("interrupted restore marker is missing or incompatible")
                shutil.rmtree(candidate)
            removed += 1
        if retained != entries:
            _write(
                path,
                {
                    "format": "curiosity-managed-restore-registry",
                    "version": 1,
                    "restores": retained,
                },
            )
    return removed


def _propagate_replay_authority(
    database: Path,
    manifest: dict[str, dict[str, dict[str, str]]],
) -> bool:
    """Install future replay guards in a retained managed database."""

    try:
        conn = sqlite3.connect(database, timeout=30)
        try:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            required = {
                "events": "deleted_event_tombstones",
                "transport_events": "deleted_transport_event_tombstones",
                "learning_sources": "evidence_source_tombstones",
            }
            if any(manifest[kind] and table not in tables for kind, table in required.items()):
                return False
            conn.execute("BEGIN IMMEDIATE")
            import_replay_tombstones(conn, manifest)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError("managed restore replay authority could not be updated") from exc
    merge_deletion_manifest(database, manifest)
    return True


def _purge_managed_restores(
    root: Path,
    manifest: dict[str, dict[str, dict[str, str]]],
    *,
    active_database: Path,
    database_identity: str,
) -> int:
    path = root / MANAGED_RESTORES_NAME
    entries = _managed_restore_entries(root)
    retained: list[dict[str, str]] = []
    purged = 0
    for entry in entries:
        state = entry.get("state", "published")
        if state not in MANAGED_RESTORE_STATES:
            raise RuntimeError("registered managed restore has an invalid state")
        raw_target = Path(entry.get("path", "")).expanduser()
        if not raw_target.is_absolute() or _has_symlink_component(raw_target):
            raise RuntimeError("registered managed restore path is unsafe")
        target = raw_target.resolve()
        marker_path = target / RESTORE_MARKER_NAME
        expected = managed_restore_marker(
            root,
            snapshot=entry.get("snapshot", ""),
            snapshot_created_at=entry.get("snapshot_created_at", ""),
            database_identity=entry.get("database_identity", ""),
        )
        if target == root or target.is_relative_to(root):
            raise RuntimeError("registered managed restore path crosses the backup root")
        if not target.exists():
            continue
        if not target.is_dir():
            raise RuntimeError("registered managed restore target is not a directory")
        if state == "pending" and not any(target.iterdir()):
            target.rmdir()
            purged += 1
            continue
        try:
            marker = _read(marker_path, default={})
        except RuntimeError as exc:
            raise RuntimeError("registered managed restore marker is unreadable") from exc
        if marker != expected:
            raise RuntimeError("registered managed restore marker is missing or incompatible")
        if state == "staging":
            shutil.rmtree(target)
            purged += 1
            continue
        restored_database = target / "data" / "curiosity.db"
        is_active_database = restored_database.resolve() == active_database
        conflicts = database_deletion_conflicts(
            restored_database,
            manifest,
            snapshot_created_at=entry.get("snapshot_created_at"),
            wal_aware=True,
            expected_database_identity=database_identity,
            current_database=is_active_database,
        )
        if is_active_database:
            if conflicts:
                raise RuntimeError("active family database conflicts with its deletion manifest")
            retained.append(entry)
            continue
        if conflicts:
            shutil.rmtree(target)
            purged += 1
        elif not _propagate_replay_authority(restored_database, manifest):
            shutil.rmtree(target)
            purged += 1
        else:
            retained.append(entry)
    if entries != retained:
        with _metadata_locked(path):
            _write(
                path,
                {
                    "format": "curiosity-managed-restore-registry",
                    "version": 1,
                    "restores": retained,
                },
            )
    return purged


def purge_managed_backups(db_path: str | Path, event_hash: str | None = None) -> dict[str, int]:
    """Remove managed snapshots, migration copies, and restores that retain deleted data."""

    del event_hash
    database = Path(db_path).expanduser().resolve()
    database_identity = _assert_managed_database_image_readable(database, wal_aware=True)
    manifest = deletion_manifest(database)
    migration_backups = 0
    for candidate in migration_backup_candidates(database):
        if database_deletion_conflicts(
            candidate,
            manifest,
            snapshot_created_at=datetime.fromtimestamp(candidate.stat().st_mtime).astimezone().isoformat(),
            wal_aware=True,
            expected_database_identity=database_identity,
            require_curiosity_schema=False,
            allow_legacy_without_database_identity=True,
        ):
            sidecars = tuple(migration_backup_sidecars(candidate))
            for sidecar in sidecars:
                sidecar.unlink(missing_ok=True)
            # Keep the discoverable base until every sidecar is gone.  If a
            # sidecar unlink fails, the next reconciliation can find and retry
            # the complete migration image instead of stranding family bytes.
            candidate.unlink()
            migration_backups += 1

    snapshots = 0
    partials = 0
    managed_restores = 0
    for root in _registered_backup_roots(database):
        _assert_backup_root(root)
        _assert_reciprocal_backup_authority(database, root)
        _assert_registered_restore_entries_safe(root)
        marker = root / ".curiosity-backup-root.json"
        try:
            marker_value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("registered backup root marker is unreadable") from exc
        if marker_value != {"format": "curiosity-family-backup", "format_version": 2}:
            raise RuntimeError("registered backup root marker is incompatible")
        sync_backup_root_deletions(database, root)
        root_manifest = backup_root_deletion_manifest(root)
        for snapshot in sorted(root.iterdir()):
            if snapshot.is_symlink() or not snapshot.is_dir():
                continue
            if snapshot.name.startswith("curiosity-backup-"):
                _, snapshot_format_version = _assert_managed_snapshot_database(
                    snapshot,
                    database_identity=database_identity,
                )
                manifest_path = snapshot / "manifest.json"
                try:
                    snapshot_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    created_at = str(snapshot_manifest["created_at"])
                except (OSError, KeyError, json.JSONDecodeError):
                    created_at = None
                if (
                    snapshot_format_version == 1
                    and any(root_manifest[kind] for kind in DELETION_KINDS)
                ) or database_deletion_conflicts(
                    snapshot / "data" / "curiosity.db",
                    root_manifest,
                    snapshot_created_at=created_at,
                    expected_database_identity=(
                        database_identity if snapshot_format_version == 2 else None
                    ),
                    require_database_identity=snapshot_format_version == 2,
                ):
                    shutil.rmtree(snapshot)
                    snapshots += 1
            elif snapshot.name.startswith(".partial-"):
                partial_database, partial_format_version = _assert_managed_snapshot_database(
                    snapshot,
                    database_identity=database_identity,
                )
                if (
                    partial_format_version == 1
                    and any(root_manifest[kind] for kind in DELETION_KINDS)
                ) or database_deletion_conflicts(
                    partial_database,
                    root_manifest,
                    expected_database_identity=(
                        database_identity if partial_format_version == 2 else None
                    ),
                    require_database_identity=partial_format_version == 2,
                ):
                    shutil.rmtree(snapshot)
                    partials += 1
        managed_restores += _purge_managed_restores(
            root,
            root_manifest,
            active_database=database,
            database_identity=database_identity,
        )
    return {
        "migration_backups_purged": migration_backups,
        "managed_snapshots_purged": snapshots,
        "partial_snapshots_purged": partials,
        "managed_restores_purged": managed_restores,
    }


def reconcile_managed_backups(db_path: str | Path, conn: sqlite3.Connection) -> dict[str, int]:
    sync_database_tombstones(db_path, conn)
    return purge_managed_backups(db_path)


def reconcile_database_deletions(db_path: str | Path) -> dict[str, int]:
    database = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        sync_database_tombstones(database, conn)
    finally:
        conn.close()
    return purge_managed_backups(database)
