from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .deletion_ledger import authenticated_research_authority_targets

LEDGER_FORMAT = "curiosity-monotonic-research-authority"
LEDGER_VERSION = 3
LEDGER_SUFFIX = ".research-authority.json"
ROOT_LEDGER_NAME = "family-research-authority.json"


def ledger_path(db_path: str | Path) -> Path:
    database = Path(db_path).expanduser().resolve(strict=False)
    return database.with_name(f"{database.name}{LEDGER_SUFFIX}")


def root_ledger_path(backup_root: str | Path) -> Path:
    return Path(backup_root).expanduser().resolve() / ROOT_LEDGER_NAME


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _subject_hash(*parts: Any) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode()).hexdigest()


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError("unsafe monotonic research-authority ledger parent")
    lock = path.with_name(f".{path.name}.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "manifest_hash"}
    return {**unsigned, "manifest_hash": _digest(unsigned)}


def _empty(
    database_id: str,
    *,
    backup_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    return _seal(
        {
            "format": LEDGER_FORMAT,
            "version": LEDGER_VERSION,
            "database_id": database_id,
            "backup_roots": sorted(
                str(Path(item).expanduser().resolve()) for item in backup_roots
            ),
            "policies": {},
            "family_controls": {},
            "pending": None,
        }
    )


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)
    path.chmod(0o600)


POLICY_FIELDS = {
    "subject_hash",
    "policy_version",
    "authorization_epoch",
    "active",
    "event_id",
    "record_hash",
}
CONTROL_FIELDS = {
    "subject_hash",
    "family_authorization_epoch",
    "enabled",
    "binding_hash",
    "event_id",
    "lineage_floor",
    "record_hash",
}


def _validate_records(kind: str, records: Any) -> None:
    expected_fields = POLICY_FIELDS if kind == "policies" else CONTROL_FIELDS
    if not isinstance(records, dict):
        raise RuntimeError("invalid monotonic research-authority ledger")
    for key, record in records.items():
        if (
            not isinstance(key, str)
            or len(key) != 64
            or not isinstance(record, dict)
            or set(record) != expected_fields
            or record.get("subject_hash") != key
            or record.get("record_hash")
            != _digest({name: item for name, item in record.items() if name != "record_hash"})
        ):
            raise RuntimeError("invalid monotonic research-authority ledger record")
        epoch_field = (
            "authorization_epoch" if kind == "policies" else "family_authorization_epoch"
        )
        if type(record.get(epoch_field)) is not int or int(record[epoch_field]) < 1:
            raise RuntimeError("invalid monotonic research-authority ledger epoch")
        state_field = "active" if kind == "policies" else "enabled"
        if type(record.get(state_field)) is not bool or not isinstance(record.get("event_id"), str):
            raise RuntimeError("invalid monotonic research-authority ledger state")
        if kind == "family_controls" and (
            type(record.get("lineage_floor")) is not int
            or not 0 <= int(record["lineage_floor"]) <= int(record[epoch_field])
            or (record["enabled"] and record["lineage_floor"] == record[epoch_field])
        ):
            raise RuntimeError("invalid monotonic research-authority lineage floor")


def _read(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.exists():
        if required:
            raise RuntimeError("required monotonic research-authority ledger is missing")
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("unsafe monotonic research-authority ledger")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("unreadable monotonic research-authority ledger") from exc
    if not isinstance(value, dict) or set(value) != {
        "format",
        "version",
        "database_id",
        "backup_roots",
        "policies",
        "family_controls",
        "pending",
        "manifest_hash",
    }:
        raise RuntimeError("invalid monotonic research-authority ledger")
    unsigned = {key: item for key, item in value.items() if key != "manifest_hash"}
    roots = value.get("backup_roots")
    if (
        value["format"] != LEDGER_FORMAT
        or value["version"] != LEDGER_VERSION
        or not isinstance(value["database_id"], str)
        or not isinstance(roots, list)
        or not all(
            isinstance(item, str)
            and Path(item).is_absolute()
            and str(Path(item).expanduser().resolve()) == item
            for item in roots
        )
        or roots != sorted(set(roots))
        or value["manifest_hash"] != _digest(unsigned)
    ):
        raise RuntimeError("invalid monotonic research-authority ledger")
    _validate_records("policies", value["policies"])
    _validate_records("family_controls", value["family_controls"])
    pending = value["pending"]
    if pending is not None:
        if (
            not isinstance(pending, dict)
            or set(pending) != {"operation_id", "policies", "family_controls"}
            or not isinstance(pending["operation_id"], str)
            or pending["operation_id"]
            != _digest(
                {
                    "policies": pending["policies"],
                    "family_controls": pending["family_controls"],
                }
            )
        ):
            raise RuntimeError("invalid pending research-authority transition")
        _validate_records("policies", pending["policies"])
        _validate_records("family_controls", pending["family_controls"])
    return value


def authority_backup_roots(db_path: str | Path) -> tuple[Path, ...]:
    """Return the immutable peer-root claims needed before lifecycle locking."""

    path = ledger_path(db_path)
    if not path.exists():
        return ()
    value = _read(path, required=True)
    assert value is not None
    return tuple(Path(item) for item in value["backup_roots"])


def backup_authority_roots(backup_root: str | Path) -> tuple[Path, ...]:
    """Return authenticated-ledger peer claims before restore locking."""

    value = _read(root_ledger_path(backup_root), required=True)
    assert value is not None
    return tuple(Path(item) for item in value["backup_roots"])


def _database_id(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT database_id FROM family_database_identity WHERE singleton=1"
    ).fetchone()
    if not row:
        raise RuntimeError("research authority requires a stable family database identity")
    return str(row[0])


def _record(unsigned: dict[str, Any]) -> dict[str, Any]:
    return {**unsigned, "record_hash": _digest(unsigned)}


def _policy_record(conn: sqlite3.Connection, event_id: str) -> tuple[str, dict[str, Any]]:
    event = conn.execute(
        "SELECT * FROM research_track_policy_events WHERE event_id=?", (event_id,)
    ).fetchone()
    if not event or str(event["action"]) not in {"activated", "disabled", "rolled_back"}:
        raise RuntimeError("research policy authority requires an activation-state event")
    policy = conn.execute(
        "SELECT * FROM research_track_policies WHERE policy_id=? AND version=?",
        (event["policy_id"], event["policy_version"]),
    ).fetchone()
    if not policy:
        raise RuntimeError("research policy authority event has no policy")
    active = str(policy["status"]) == "active"
    if active != (str(event["action"]) == "activated"):
        raise RuntimeError("research policy authority event does not match current state")
    key = _subject_hash(event["policy_id"], event["policy_version"])
    return key, _record(
        {
            "subject_hash": key,
            "policy_version": int(event["policy_version"]),
            "authorization_epoch": int(policy["authorization_epoch"]),
            "active": active,
            "event_id": str(event["event_id"]),
        }
    )


def _control_record(
    conn: sqlite3.Connection,
    event_id: str,
    *,
    lineage_floor: int = 0,
) -> tuple[str, dict[str, Any]]:
    event = conn.execute(
        "SELECT * FROM research_track_control_events WHERE event_id=?", (event_id,)
    ).fetchone()
    if not event:
        raise RuntimeError("family research authority requires a control event")
    control = conn.execute(
        "SELECT * FROM research_track_controls WHERE child_id=?", (event["child_id"],)
    ).fetchone()
    if not control:
        raise RuntimeError("family research authority event has no control")
    binding = (
        control["policy_id"],
        control["policy_version"],
        control["definition_hash"],
        control["release_content_hash"],
    )
    event_binding = (
        event["policy_id"],
        event["policy_version"],
        event["definition_hash"],
        event["release_content_hash"],
    )
    if (
        int(event["family_authorization_epoch"])
        != int(control["family_authorization_epoch"])
        or bool(event["enabled"]) != bool(control["enabled"])
        or event_binding != binding
    ):
        raise RuntimeError("family research authority event does not match current state")
    key = _subject_hash(event["child_id"])
    return key, _record(
        {
            "subject_hash": key,
            "family_authorization_epoch": int(control["family_authorization_epoch"]),
            "enabled": bool(control["enabled"]),
            "binding_hash": _digest(list(binding)) if bool(control["enabled"]) else None,
            "event_id": str(event["event_id"]),
            "lineage_floor": lineage_floor,
        }
    )


def _transition_is_valid(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
    *,
    epoch_field: str,
) -> bool:
    epoch = int(incoming[epoch_field])
    if existing is None:
        return epoch == 1
    if incoming == existing:
        return True
    if epoch == int(existing[epoch_field]) + 1:
        return True
    return bool(
        epoch_field == "family_authorization_epoch"
        and epoch == int(existing[epoch_field])
        and not incoming["enabled"]
        and int(incoming["lineage_floor"]) == epoch
        and int(existing["lineage_floor"]) < epoch
    )


def record_research_authority(
    db_path: str | Path,
    conn: sqlite3.Connection,
    *,
    policy_event_id: str | None = None,
    control_event_ids: Sequence[str] = (),
    erased_control_event_ids: Sequence[str] = (),
) -> None:
    """Stage an external transition before SQLite commits it.

    Startup/finalization commits a matching transition, rolls back an
    interrupted grant, or rolls forward an interrupted revocation.
    """

    database = Path(db_path).expanduser().resolve()
    path = ledger_path(database)
    with _locked(path):
        value = _read(path, required=True)
    assert value is not None
    if value["database_id"] != _database_id(conn):
        raise RuntimeError("research-authority ledger belongs to another family database")
    if value["pending"] is not None:
        raise RuntimeError("a research-authority transition is already pending recovery")
    policies: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    if policy_event_id is not None:
        key, item = _policy_record(conn, policy_event_id)
        if not _transition_is_valid(
            value["policies"].get(key), item, epoch_field="authorization_epoch"
        ):
            raise RuntimeError("research policy authority epoch is not monotonic")
        policies[key] = item
    erased = set(erased_control_event_ids)
    for event_id in (*control_event_ids, *erased_control_event_ids):
        event = conn.execute(
            "SELECT child_id,family_authorization_epoch FROM research_track_control_events "
            "WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if not event:
            raise RuntimeError("family research authority requires a control event")
        key = _subject_hash(event["child_id"])
        existing = value["family_controls"].get(key)
        inherited_floor = int(existing["lineage_floor"]) if existing else 0
        floor = int(event["family_authorization_epoch"]) if event_id in erased else inherited_floor
        key, item = _control_record(conn, event_id, lineage_floor=floor)
        if not _transition_is_valid(
            existing, item, epoch_field="family_authorization_epoch"
        ):
            raise RuntimeError("family research authority epoch is not monotonic")
        controls[key] = item
    pending_unsigned = {"policies": policies, "family_controls": controls}
    pending = {
        "operation_id": _digest(pending_unsigned),
        **pending_unsigned,
    }
    roots, restores = authenticated_research_authority_targets(
        database,
        claimed_roots=tuple(value["backup_roots"]),
    )
    root_names = [str(root) for root in roots]
    target_paths = [root_ledger_path(root) for root in roots]
    target_paths.extend(ledger_path(restore) for restore in restores)
    targets: list[tuple[Path, dict[str, Any]]] = []
    for target_path in target_paths:
        with _locked(target_path):
            target = _read(target_path, required=True)
        assert target is not None
        if target["database_id"] != value["database_id"]:
            raise RuntimeError("research-authority target belongs to another family")
        if target["backup_roots"] != root_names:
            raise RuntimeError("research-authority peer-root membership is incomplete")
        if target["pending"] is not None and target["pending"] != pending:
            raise RuntimeError("research-authority target has another pending transition")
        _assert_grants_are_newer(target, pending)
        target["pending"] = pending
        targets.append((target_path, target))

    # Lifecycle locks make the ordered fanout exclusive. Root peers are written
    # before the database-local journal; every peer already knows the full root
    # set, so a partial process death remains discoverable from any survivor.
    for target_path, target in targets:
        with _locked(target_path):
            _write(target_path, _seal(target))
    value["backup_roots"] = root_names
    value["pending"] = pending
    with _locked(path):
        _write(path, _seal(value))


def _binding_hash(row: sqlite3.Row) -> str | None:
    if not bool(row["enabled"]):
        return None
    return _digest(
        [
            row["policy_id"],
            row["policy_version"],
            row["definition_hash"],
            row["release_content_hash"],
        ]
    )


def _policy_matches(conn: sqlite3.Connection, record: dict[str, Any]) -> bool:
    policies = [
        row
        for row in conn.execute("SELECT * FROM research_track_policies").fetchall()
        if _subject_hash(row["policy_id"], row["version"]) == record["subject_hash"]
    ]
    if len(policies) != 1:
        return False
    policy = policies[0]
    event = conn.execute(
        "SELECT action FROM research_track_policy_events WHERE event_id=?", (record["event_id"],)
    ).fetchone()
    return bool(
        int(policy["version"]) == int(record["policy_version"])
        and int(policy["authorization_epoch"]) == int(record["authorization_epoch"])
        and (str(policy["status"]) == "active") == bool(record["active"])
        and event is not None
        and (str(event["action"]) == "activated") == bool(record["active"])
    )


def _control_rows(conn: sqlite3.Connection, subject_hash: str) -> list[sqlite3.Row]:
    return [
        row
        for row in conn.execute("SELECT * FROM research_track_controls").fetchall()
        if _subject_hash(row["child_id"]) == subject_hash
    ]


def _control_matches(conn: sqlite3.Connection, record: dict[str, Any]) -> bool:
    controls = _control_rows(conn, str(record["subject_hash"]))
    if not controls:
        return bool(
            not record["enabled"]
            and int(record["lineage_floor"])
            == int(record["family_authorization_epoch"])
        )
    if len(controls) != 1:
        return False
    if int(record["lineage_floor"]) == int(record["family_authorization_epoch"]):
        return False
    control = controls[0]
    event = conn.execute(
        "SELECT enabled FROM research_track_control_events WHERE event_id=?",
        (record["event_id"],),
    ).fetchone()
    return bool(
        int(control["family_authorization_epoch"])
        == int(record["family_authorization_epoch"])
        and bool(control["enabled"]) == bool(record["enabled"])
        and _binding_hash(control) == record["binding_hash"]
        and event is not None
        and bool(event["enabled"]) == bool(record["enabled"])
    )


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _policy_event_id(
    policy_id: Any,
    policy_version: Any,
    action: Any,
    actor_type: Any,
    operator_id: Any,
    reason: Any,
    created_at: Any,
) -> str:
    return "rtev_" + hashlib.sha256(
        (
            f"{policy_id}|{int(policy_version)}|{action}|{actor_type}|"
            f"{operator_id or ''}|{reason}|{created_at}"
        ).encode()
    ).hexdigest()[:24]


def _control_event_id(
    child_id: Any,
    enabled: Any,
    policy_id: Any,
    policy_version: Any,
    definition_hash: Any,
    release_content_hash: Any,
    family_authorization_epoch: Any,
    actor_type: Any,
    reason: Any,
    created_at: Any,
) -> str:
    binding = (
        policy_id,
        int(policy_version) if policy_version is not None else None,
        definition_hash,
        release_content_hash,
    )
    return "rtctl_" + hashlib.sha256(
        (
            f"{child_id}|{int(enabled)}|{binding}|{int(family_authorization_epoch)}|"
            f"{actor_type}|{reason}|{created_at}"
        ).encode()
    ).hexdigest()[:24]


def _recover_revocations(
    conn: sqlite3.Connection,
    value: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    recovered_policies: dict[str, Any] = {}
    recovered_controls: dict[str, Any] = {}
    conn.create_function("research_write_authorized", 0, lambda: 1, deterministic=True)
    try:
        for key, target in pending["policies"].items():
            matches = [
                row
                for row in conn.execute("SELECT * FROM research_track_policies").fetchall()
                if _subject_hash(row["policy_id"], row["version"]) == key
            ]
            if len(matches) != 1 or target["active"]:
                raise RuntimeError("pending research-policy authority cannot be recovered")
            policy = matches[0]
            prior = value["policies"].get(key)
            target_epoch = int(target["authorization_epoch"])
            if int(policy["authorization_epoch"]) == target_epoch - 1 and str(
                policy["status"]
            ) == "active":
                now = _utcnow()
                reason = "Recovered an interrupted research-policy revocation."
                conn.execute(
                    """UPDATE research_track_policies
                       SET status='disabled',authorization_epoch=?,updated_at=?
                       WHERE policy_id=? AND version=?""",
                    (target_epoch, now, policy["policy_id"], policy["version"]),
                )
                event_id = _policy_event_id(
                    policy["policy_id"],
                    policy["version"],
                    "disabled",
                    "system",
                    None,
                    reason,
                    now,
                )
                conn.execute(
                    """INSERT INTO research_track_policy_events(
                         event_id,policy_id,policy_version,action,actor_type,operator_id,reason,created_at
                       ) VALUES(?,?,?,'disabled','system',NULL,?,?)""",
                    (event_id, policy["policy_id"], policy["version"], reason, now),
                )
                recovered_policies[key] = _policy_record(conn, event_id)[1]
            elif _policy_matches(conn, target):
                recovered_policies[key] = target
            elif (
                prior is not None
                and not bool(prior["active"])
                and _policy_matches(conn, prior)
            ):
                # This recovery image deliberately skipped an intervening
                # grant. Keep its older verified closed state; never invent
                # the skipped activation merely to reach the newer revocation.
                recovered_policies[key] = prior
            else:
                raise RuntimeError("pending research-policy revocation conflicts with the database")

        for key, target in pending["family_controls"].items():
            controls = _control_rows(conn, key)
            prior = value["family_controls"].get(key)
            if target["enabled"]:
                raise RuntimeError("pending family research grant cannot be rolled forward")
            erases_lineage = int(target["lineage_floor"]) == int(
                target["family_authorization_epoch"]
            )
            if erases_lineage and controls:
                child_id = str(controls[0]["child_id"])
                conn.execute(
                    "DELETE FROM research_track_control_events WHERE child_id=?",
                    (child_id,),
                )
                conn.execute(
                    "DELETE FROM research_track_controls WHERE child_id=?",
                    (child_id,),
                )
                controls = []
            if not controls and erases_lineage:
                recovered_controls[key] = target
                continue
            if len(controls) != 1:
                raise RuntimeError("pending family research revocation cannot identify its child")
            control = controls[0]
            target_epoch = int(target["family_authorization_epoch"])
            if bool(control["enabled"]) and int(control["family_authorization_epoch"]) == (
                target_epoch - 1
            ):
                now = _utcnow()
                reason = "Recovered an interrupted family research revocation."
                conn.execute(
                    """UPDATE research_track_controls
                       SET enabled=0,policy_id=NULL,policy_version=NULL,definition_hash=NULL,
                           release_content_hash=NULL,family_authorization_epoch=?,updated_by='operator',
                           reason=?,updated_at=? WHERE child_id=?""",
                    (target_epoch, reason, now, control["child_id"]),
                )
                event_id = _control_event_id(
                    control["child_id"],
                    False,
                    None,
                    None,
                    None,
                    None,
                    target_epoch,
                    "system",
                    reason,
                    now,
                )
                conn.execute(
                    """INSERT INTO research_track_control_events(
                         event_id,child_id,enabled,policy_id,policy_version,definition_hash,
                         release_content_hash,family_authorization_epoch,actor_type,reason,created_at
                       ) VALUES(?,?,0,NULL,NULL,NULL,NULL,?,'system',?,?)""",
                    (event_id, control["child_id"], target_epoch, reason, now),
                )
                prior_floor = int(prior["lineage_floor"]) if prior else 0
                recovered_controls[key] = _control_record(
                    conn, event_id, lineage_floor=prior_floor
                )[1]
            elif _control_matches(conn, target):
                recovered_controls[key] = target
            elif (
                not bool(control["enabled"])
                and prior is not None
                and int(control["family_authorization_epoch"])
                == int(prior["family_authorization_epoch"])
            ):
                recovered_controls[key] = prior
            else:
                raise RuntimeError("pending family research revocation conflicts with the database")
        conn.commit()
    finally:
        conn.create_function("research_write_authorized", 0, lambda: 0, deterministic=True)
    return recovered_policies, recovered_controls


def _resolve_pending_locked(
    path: Path,
    value: dict[str, Any],
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    pending = value["pending"]
    if pending is None:
        return value
    targets_match = all(
        _policy_matches(conn, record) for record in pending["policies"].values()
    ) and all(
        _control_matches(conn, record) for record in pending["family_controls"].values()
    )
    if targets_match:
        recovered_policies = dict(pending["policies"])
        recovered_controls = dict(pending["family_controls"])
    else:
        is_grant = any(record["active"] for record in pending["policies"].values()) or any(
            record["enabled"] for record in pending["family_controls"].values()
        )
        if is_grant:
            value["pending"] = None
            _write(path, _seal(value))
            return value
        recovered_policies, recovered_controls = _recover_revocations(conn, value, pending)
    value["policies"].update(recovered_policies)
    value["family_controls"].update(recovered_controls)
    value["pending"] = None
    _write(path, _seal(value))
    return value


def _same_authority_state(left: dict[str, Any], right: dict[str, Any], *, kind: str) -> bool:
    if kind == "policies":
        fields = ("policy_version", "authorization_epoch", "active")
    else:
        fields = (
            "family_authorization_epoch",
            "enabled",
            "binding_hash",
        )
    return all(left.get(field) == right.get(field) for field in fields)


def _merge_committed(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for kind, epoch_field in (
        ("policies", "authorization_epoch"),
        ("family_controls", "family_authorization_epoch"),
    ):
        for key, record in incoming[kind].items():
            existing = target[kind].get(key)
            if existing is None or int(record[epoch_field]) > int(existing[epoch_field]):
                target[kind][key] = record
            elif int(record[epoch_field]) == int(existing[epoch_field]):
                if not _same_authority_state(existing, record, kind=kind):
                    raise RuntimeError("backup research-authority high-water state conflicts")
                if kind == "family_controls" and int(record["lineage_floor"]) > int(
                    existing["lineage_floor"]
                ):
                    target[kind][key] = record


def _resolve_root_pending(root: dict[str, Any], local: dict[str, Any]) -> None:
    pending = root["pending"]
    if pending is None:
        return
    for kind, epoch_field, state_field in (
        ("policies", "authorization_epoch", "active"),
        ("family_controls", "family_authorization_epoch", "enabled"),
    ):
        for key, target in pending[kind].items():
            current = local[kind].get(key)
            if (
                current is not None
                and int(current[epoch_field]) == int(target[epoch_field])
                and current[state_field] == target[state_field]
                and (kind == "policies" or current["binding_hash"] == target["binding_hash"])
                and (
                    kind == "policies"
                    or int(current["lineage_floor"]) >= int(target["lineage_floor"])
                )
            ):
                root[kind][key] = current
            elif target[state_field] and (
                current is None or int(current[epoch_field]) < int(target[epoch_field])
            ):
                # SQLite did not commit the staged grant. Keeping the earlier
                # committed root state is the fail-closed outcome.
                continue
            else:
                raise RuntimeError(
                    "backup research-authority pending transition conflicts with source"
                )
    root["pending"] = None


def _pending_is_grant(pending: dict[str, Any]) -> bool:
    return any(record["active"] for record in pending["policies"].values()) or any(
        record["enabled"] for record in pending["family_controls"].values()
    )


def _assert_grants_are_newer(
    authority: dict[str, Any],
    pending: dict[str, Any],
) -> None:
    """Reject stale grants against every target before fanout changes bytes."""

    for kind, epoch_field, state_field in (
        ("policies", "authorization_epoch", "active"),
        ("family_controls", "family_authorization_epoch", "enabled"),
    ):
        for key, incoming in pending[kind].items():
            if not incoming[state_field]:
                continue
            committed = authority[kind].get(key)
            if committed is None:
                continue
            committed_epoch = int(committed[epoch_field])
            incoming_epoch = int(incoming[epoch_field])
            if committed_epoch > incoming_epoch or (
                committed_epoch == incoming_epoch and committed != incoming
            ):
                raise RuntimeError(
                    "research-authority grant is obsolete relative to a recovery peer"
                )


def _newer_records(
    incoming: dict[str, Any],
    local: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    policies = {
        key: record
        for key, record in incoming["policies"].items()
        if not record["active"]
        and (
            key not in local["policies"]
            or int(record["authorization_epoch"])
            > int(local["policies"][key]["authorization_epoch"])
        )
    }
    controls = {
        key: record
        for key, record in incoming["family_controls"].items()
        if not record["enabled"]
        and (
            key not in local["family_controls"]
            or int(record["family_authorization_epoch"])
            > int(local["family_controls"][key]["family_authorization_epoch"])
            or (
                int(record["family_authorization_epoch"])
                == int(local["family_controls"][key]["family_authorization_epoch"])
                and int(record["lineage_floor"])
                > int(local["family_controls"][key]["lineage_floor"])
            )
        )
    }
    return policies, controls


def _merge_root_revocations(
    database: Path,
    local: dict[str, Any],
    *,
    allow_registration_pending: bool = False,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    roots, restores = authenticated_research_authority_targets(
        database,
        claimed_roots=tuple(local["backup_roots"]),
        allow_registration_pending=allow_registration_pending,
    )
    if not roots:
        return roots, restores
    root_names = [str(root) for root in roots]
    aggregate = _empty(str(local["database_id"]), backup_roots=roots)
    for root in roots:
        with _locked(root_ledger_path(root)):
            value = _read(root_ledger_path(root), required=True)
        assert value is not None
        if value["database_id"] != local["database_id"]:
            raise RuntimeError("backup research-authority ledger belongs to another family")
        if value["backup_roots"] != root_names:
            raise RuntimeError("backup research-authority peer-root membership is incomplete")
        _merge_committed(aggregate, value)
        if value["pending"] is not None and not _pending_is_grant(value["pending"]):
            pending_value = _empty(str(local["database_id"]), backup_roots=roots)
            pending_value["policies"] = dict(value["pending"]["policies"])
            pending_value["family_controls"] = dict(value["pending"]["family_controls"])
            _merge_committed(aggregate, pending_value)
    # A newer grant is never copied into an older image. An image that is
    # already closed may remain closed, but an older active grant must not be
    # mistaken for the newer consent merely because both states are enabled.
    for kind, epoch_field, state_field in (
        ("policies", "authorization_epoch", "active"),
        ("family_controls", "family_authorization_epoch", "enabled"),
    ):
        for key, incoming in aggregate[kind].items():
            if not incoming[state_field]:
                continue
            existing = local[kind].get(key)
            pending = local["pending"][kind].get(key) if local["pending"] is not None else None
            effective = (
                pending
                if pending is not None and not pending[state_field]
                else existing
            )
            if (
                effective is not None
                and effective[state_field]
                and int(incoming[epoch_field]) > int(effective[epoch_field])
            ):
                raise RuntimeError(
                    "database active research authority is behind a newer external grant"
                )
    policies, controls = _newer_records(aggregate, local)
    if policies or controls:
        pending_unsigned = {"policies": policies, "family_controls": controls}
        incoming_pending = {
            "operation_id": _digest(pending_unsigned),
            **pending_unsigned,
        }
        if local["pending"] is not None and local["pending"] != incoming_pending:
            raise RuntimeError("local and backup research-authority transitions conflict")
        local["pending"] = incoming_pending
    local["backup_roots"] = root_names
    return roots, restores


def sync_registered_backup_authority(
    db_path: str | Path,
    *,
    allow_registration_pending: bool = False,
) -> None:
    database = Path(db_path).expanduser().resolve()
    path = ledger_path(database)
    with _locked(path):
        local = _read(path, required=True)
    assert local is not None
    if local["pending"] is not None:
        raise RuntimeError("cannot back up a pending research-authority transition")
    roots, restores = authenticated_research_authority_targets(
        database,
        claimed_roots=tuple(local["backup_roots"]),
        allow_registration_pending=allow_registration_pending,
    )
    root_names = [str(root) for root in roots]
    root_values: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        root_path = root_ledger_path(root)
        with _locked(root_path):
            value = _read(root_path, required=not allow_registration_pending) or _empty(
                str(local["database_id"]), backup_roots=roots
            )
        if value["database_id"] != local["database_id"]:
            raise RuntimeError("backup research-authority ledger belongs to another family")
        _resolve_root_pending(value, local)
        _merge_committed(value, local)
        value["backup_roots"] = root_names
        root_values.append((root_path, value))
    for restore in restores:
        restore_path = ledger_path(restore)
        restore_conn = sqlite3.connect(restore, timeout=30)
        restore_conn.row_factory = sqlite3.Row
        try:
            with _locked(restore_path):
                value = _read(restore_path, required=True)
                assert value is not None
                if (
                    value["database_id"] != local["database_id"]
                    or _database_id(restore_conn) != local["database_id"]
                ):
                    raise RuntimeError("managed restore authority belongs to another family")
                value = _resolve_pending_locked(
                    restore_path,
                    value,
                    restore_conn,
                )
                policies, controls = _newer_records(local, value)
                if policies or controls:
                    pending_unsigned = {
                        "policies": policies,
                        "family_controls": controls,
                    }
                    value["pending"] = {
                        "operation_id": _digest(pending_unsigned),
                        **pending_unsigned,
                    }
                    _write(restore_path, _seal(value))
                    value = _resolve_pending_locked(
                        restore_path,
                        value,
                        restore_conn,
                    )
                _assert_matches_database(value, restore_conn)
                value["backup_roots"] = root_names
                _write(restore_path, _seal(value))
        finally:
            restore_conn.close()
    for root_path, value in root_values:
        with _locked(root_path):
            _write(root_path, _seal(value))
    local["backup_roots"] = root_names
    with _locked(path):
        _write(path, _seal(local))


def assert_research_authority_image(
    db_path: str | Path,
    authority_path: str | Path,
) -> None:
    """Verify a copied database and sidecar form one committed authority image."""

    database = Path(db_path).expanduser().resolve()
    authority = Path(authority_path).expanduser().resolve()
    value = _read(authority, required=True)
    assert value is not None
    if value["pending"] is not None:
        raise RuntimeError("backup contains a pending research-authority transition")
    conn = sqlite3.connect(database, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        if value["database_id"] != _database_id(conn):
            raise RuntimeError("backup research-authority ledger belongs to another family")
        _assert_matches_database(value, conn)
    finally:
        conn.close()


def finalize_research_authority(
    db_path: str | Path,
    *,
    required: bool = False,
) -> None:
    database = Path(db_path).expanduser().resolve()
    path = ledger_path(database)
    if not path.exists():
        if required:
            raise RuntimeError("required monotonic research-authority ledger is missing")
        return
    conn = sqlite3.connect(database, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        with _locked(path):
            value = _read(path, required=True)
        assert value is not None
        if value["database_id"] != _database_id(conn):
            raise RuntimeError("research-authority ledger belongs to another family database")
        roots, _restores = _merge_root_revocations(database, value)
        with _locked(path):
            _write(path, _seal(value))
            _resolve_pending_locked(path, value, conn)
            _assert_matches_database(value, conn)
    finally:
        conn.close()
    if roots:
        sync_registered_backup_authority(database)


def family_authorization_high_water(db_path: str | Path, child_id: str) -> tuple[int, int]:
    path = ledger_path(db_path)
    with _locked(path):
        value = _read(path, required=True)
        assert value is not None
        if value["pending"] is not None:
            raise RuntimeError("family research authority is pending recovery")
        record = value["family_controls"].get(_subject_hash(child_id))
        if record is None:
            return 0, 0
        return int(record["family_authorization_epoch"]), int(record["lineage_floor"])


def _assert_matches_database(value: dict[str, Any], conn: sqlite3.Connection) -> None:
    policies = {
        _subject_hash(row["policy_id"], row["version"]): row
        for row in conn.execute("SELECT * FROM research_track_policies").fetchall()
    }
    for key, record in value["policies"].items():
        if key not in policies or not _policy_matches(conn, record):
            raise RuntimeError("database research policy is behind external authorization authority")
    for key, policy in policies.items():
        if int(policy["authorization_epoch"]) > 0 and key not in value["policies"]:
            raise RuntimeError("database research policy lacks external authorization authority")

    controls = {
        _subject_hash(row["child_id"]): row
        for row in conn.execute("SELECT * FROM research_track_controls").fetchall()
    }
    for key, record in value["family_controls"].items():
        control = controls.get(key)
        if control is None:
            if not record["enabled"] and int(record["lineage_floor"]) == int(
                record["family_authorization_epoch"]
            ):
                continue
            raise RuntimeError("database family research control is missing external authority")
        if not _control_matches(conn, record):
            raise RuntimeError("database family research control is behind external authorization authority")
        epochs = [
            int(row[0])
            for row in conn.execute(
                """SELECT family_authorization_epoch FROM research_track_control_events
                   WHERE child_id=? ORDER BY family_authorization_epoch,event_id""",
                (control["child_id"],),
            ).fetchall()
        ]
        expected = list(
            range(
                int(record["lineage_floor"]) + 1,
                int(record["family_authorization_epoch"]) + 1,
            )
        )
        if epochs != expected:
            raise RuntimeError("database family research control lineage disagrees with erasure floor")
    for key, control in controls.items():
        if int(control["family_authorization_epoch"]) > 0 and key not in value["family_controls"]:
            raise RuntimeError("database family research control lacks external authorization authority")


def initialize_or_assert_research_authority(
    db_path: str | Path, conn: sqlite3.Connection
) -> None:
    """Create fresh authority, recover an interrupted transition, then verify it."""

    path = ledger_path(db_path)
    with _locked(path):
        value = _read(path, required=False)
        if value is None:
            has_authority = bool(
                conn.execute(
                    "SELECT 1 FROM research_track_policies WHERE authorization_epoch>0 LIMIT 1"
                ).fetchone()
                or conn.execute("SELECT 1 FROM research_track_controls LIMIT 1").fetchone()
                or conn.execute(
                    "SELECT 1 FROM research_track_control_events LIMIT 1"
                ).fetchone()
            )
            if has_authority:
                raise RuntimeError("required monotonic research-authority ledger is missing")
            value = _empty(_database_id(conn))
            _write(path, value)
        if value["database_id"] != _database_id(conn):
            raise RuntimeError("research-authority ledger belongs to another family database")
        value = _resolve_pending_locked(path, value, conn)
        _assert_matches_database(value, conn)


def install_backup_research_authority(
    db_path: str | Path,
    backup_root: str | Path,
) -> None:
    """Merge a backup root's latest high-water state into one restored database."""

    database = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect(database, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        database_id = _database_id(conn)
    finally:
        conn.close()
    local_path = ledger_path(database)
    root_path = root_ledger_path(backup_root)
    with _locked(root_path):
        root = _read(root_path, required=True)
    assert root is not None
    if root["database_id"] != database_id:
        raise RuntimeError("backup research-authority ledger is invalid for this family")
    root_names = list(root["backup_roots"])
    selected_root = str(Path(backup_root).expanduser().resolve())
    if selected_root not in root_names:
        raise RuntimeError("selected backup root is absent from authority peer membership")
    with _locked(local_path):
        local = _read(local_path, required=False) or _empty(
            database_id,
            backup_roots=root_names,
        )
        if local["database_id"] != database_id or local["pending"] is not None:
            raise RuntimeError("snapshot research-authority ledger is invalid for this family")
        if local["backup_roots"] and not set(local["backup_roots"]).issubset(root_names):
            raise RuntimeError("snapshot research-authority peer membership is incomplete")
        local["backup_roots"] = root_names
        _write(local_path, _seal(local))
    finalize_research_authority(database)
