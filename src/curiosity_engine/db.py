from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 11

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS children (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  birth_year INTEGER,
  grade TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  label TEXT NOT NULL,
  canonical_key TEXT NOT NULL,
  state_json TEXT NOT NULL DEFAULT '{}',
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  evidence_count INTEGER NOT NULL DEFAULT 1,
  confidence REAL NOT NULL DEFAULT 0.5 CHECK(confidence >= 0 AND confidence <= 1),
  UNIQUE(child_id, kind, canonical_key),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT NOT NULL,
  source_node_id INTEGER NOT NULL,
  relation TEXT NOT NULL,
  target_node_id INTEGER NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  evidence_count INTEGER NOT NULL DEFAULT 1,
  UNIQUE(child_id, source_node_id, relation, target_node_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(source_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
  FOREIGN KEY(target_node_id) REFERENCES nodes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  text TEXT NOT NULL,
  source TEXT,
  confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0 AND confidence <= 1),
  occurred_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  event_id TEXT,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS school_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT NOT NULL,
  category TEXT NOT NULL,
  value TEXT NOT NULL,
  source_ref TEXT,
  observed_at TEXT NOT NULL,
  expires_at TEXT,
  confidence REAL NOT NULL DEFAULT 0.8 CHECK(confidence >= 0 AND confidence <= 1),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS experiences (
  id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  experience_type TEXT NOT NULL,
  title TEXT NOT NULL,
  spec_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'generated',
  created_at TEXT NOT NULL,
  offered_at TEXT,
  completed_at TEXT,
  feedback TEXT,
  source_event_id TEXT,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
  id TEXT PRIMARY KEY,
  experience_id TEXT,
  child_id TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  path TEXT NOT NULL,
  spec_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  validated_at TEXT,
  sha256 TEXT,
  validation_json TEXT NOT NULL DEFAULT '{}',
  approval_status TEXT NOT NULL DEFAULT 'unreviewed',
  FOREIGN KEY(experience_id) REFERENCES experiences(id) ON DELETE SET NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  child_id TEXT,
  text TEXT NOT NULL,
  source TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'received',
  payload_hash TEXT,
  processed_at TEXT,
  result_json TEXT,
  error TEXT,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS claims (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT NOT NULL,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'hypothesis',
  confidence REAL NOT NULL DEFAULT 0.5,
  supporting_evidence_json TEXT NOT NULL DEFAULT '[]',
  contradicting_evidence_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(child_id, subject, predicate, object),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'corrected', 'closed')),
  topic_key TEXT NOT NULL,
  summary TEXT NOT NULL,
  clustering_version TEXT NOT NULL,
  opened_at TEXT NOT NULL,
  last_event_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT,
  event_id TEXT,
  episode_id TEXT,
  evidence_type TEXT NOT NULL,
  content TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 1.0,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE SET NULL,
  FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS episode_memberships (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id TEXT NOT NULL,
  event_id TEXT NOT NULL UNIQUE,
  relation TEXT NOT NULL,
  independence_status TEXT NOT NULL CHECK(independence_status IN ('eligible', 'same_episode', 'diagnostic', 'system', 'excluded', 'uncertain')),
  learning_scope TEXT NOT NULL CHECK(learning_scope IN ('family_signal', 'diagnostic', 'system')),
  confidence REAL NOT NULL DEFAULT 1.0 CHECK(confidence >= 0 AND confidence <= 1),
  rationale TEXT NOT NULL,
  classifier_source TEXT NOT NULL DEFAULT 'deterministic',
  classifier_version TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(episode_id) REFERENCES episodes(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS context_corrections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  action TEXT NOT NULL CHECK(action IN ('retry', 'deepening', 'new_episode', 'exclude')),
  related_event_id TEXT,
  previous_json TEXT NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(related_event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS node_evidence (
  node_id INTEGER NOT NULL,
  evidence_id INTEGER NOT NULL,
  stance TEXT NOT NULL DEFAULT 'supports',
  created_at TEXT NOT NULL,
  PRIMARY KEY(node_id, evidence_id, stance),
  FOREIGN KEY(node_id) REFERENCES nodes(id) ON DELETE CASCADE,
  FOREIGN KEY(evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  attempts INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_error TEXT,
  idempotency_key TEXT,
  event_id TEXT,
  available_at TEXT,
  leased_at TEXT,
  lease_owner TEXT,
  run_id INTEGER,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  details_json TEXT NOT NULL DEFAULT '{}',
  event_id TEXT,
  policy_json TEXT,
  result_json TEXT,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS responses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  run_id INTEGER,
  workflow TEXT NOT NULL,
  status TEXT NOT NULL,
  output_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS graph_effects (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL,
  run_id INTEGER,
  mutation_json TEXT NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  applied_at TEXT,
  UNIQUE(event_id, mutation_json),
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS actions (
  id TEXT PRIMARY KEY,
  event_id TEXT,
  run_id INTEGER,
  action_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  rationale TEXT,
  status TEXT NOT NULL DEFAULT 'proposed',
  idempotency_key TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  result_json TEXT,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE SET NULL,
  FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  child_id TEXT NOT NULL,
  event_id TEXT,
  experience_id TEXT,
  artifact_id TEXT,
  outcome TEXT NOT NULL,
  note TEXT,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE SET NULL,
  FOREIGN KEY(experience_id) REFERENCES experiences(id) ON DELETE SET NULL,
  FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS opportunities (
  id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  rationale TEXT NOT NULL,
  priority REAL NOT NULL,
  parent_effort TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL DEFAULT 'suggested',
  dedupe_key TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT,
  UNIQUE(child_id, dedupe_key),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS schedules (
  id TEXT PRIMARY KEY,
  schedule_type TEXT NOT NULL,
  child_id TEXT,
  cadence TEXT NOT NULL,
  next_run_at TEXT NOT NULL,
  last_run_at TEXT,
  enabled INTEGER NOT NULL DEFAULT 1,
  payload_json TEXT NOT NULL DEFAULT '{}',
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS approvals (
  id TEXT PRIMARY KEY,
  artifact_id TEXT NOT NULL,
  artifact_sha256 TEXT NOT NULL,
  decision TEXT NOT NULL,
  actor TEXT NOT NULL,
  created_at TEXT NOT NULL,
  note TEXT,
  FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS print_attempts (
  id TEXT PRIMARY KEY,
  artifact_id TEXT NOT NULL,
  artifact_sha256 TEXT NOT NULL,
  approval_id TEXT NOT NULL,
  printer TEXT,
  status TEXT NOT NULL,
  command_json TEXT NOT NULL,
  result_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE,
  FOREIGN KEY(approval_id) REFERENCES approvals(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS household_settings (
  id TEXT PRIMARY KEY CHECK(id = 'default'),
  timezone TEXT NOT NULL,
  quiet_start TEXT,
  quiet_end TEXT,
  proactive_enabled INTEGER NOT NULL DEFAULT 0 CHECK(proactive_enabled IN (0, 1)),
  weekly_suggestion_limit INTEGER NOT NULL DEFAULT 1 CHECK(weekly_suggestion_limit BETWEEN 0 AND 1),
  resource_context_mode TEXT NOT NULL DEFAULT 'metadata_only'
    CHECK(resource_context_mode IN ('metadata_only', 'selected_excerpts')),
  visual_mode TEXT NOT NULL DEFAULT 'deterministic'
    CHECK(visual_mode IN ('off', 'deterministic', 'decorative')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parent_principals (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('owner', 'parent')),
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'revoked')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_owner
ON parent_principals(role) WHERE role = 'owner' AND status = 'active';

CREATE TABLE IF NOT EXISTS transport_pairing_codes (
  code_hash TEXT PRIMARY KEY,
  transport TEXT NOT NULL,
  parent_id TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  used_at TEXT,
  FOREIGN KEY(parent_id) REFERENCES parent_principals(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS transport_bindings (
  id TEXT PRIMARY KEY,
  transport TEXT NOT NULL,
  team_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  channel_id TEXT NOT NULL,
  parent_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'revoked')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(transport, team_id, user_id, channel_id),
  FOREIGN KEY(parent_id) REFERENCES parent_principals(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS transport_receipts (
  transport TEXT NOT NULL,
  external_event_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  binding_id TEXT,
  status TEXT NOT NULL CHECK(status IN ('received', 'completed', 'rejected', 'failed')),
  event_id TEXT,
  received_at TEXT NOT NULL,
  processed_at TEXT,
  error TEXT,
  PRIMARY KEY(transport, external_event_id),
  FOREIGN KEY(binding_id) REFERENCES transport_bindings(id) ON DELETE SET NULL,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS capture_inbox (
  id TEXT PRIMARY KEY,
  parent_id TEXT NOT NULL,
  transport TEXT NOT NULL,
  external_event_id TEXT NOT NULL,
  text TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'unassigned' CHECK(status IN ('unassigned', 'assigned', 'dismissed')),
  child_id TEXT,
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  UNIQUE(transport, external_event_id),
  FOREIGN KEY(parent_id) REFERENCES parent_principals(id) ON DELETE CASCADE,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS delivery_outbox (
  id TEXT PRIMARY KEY,
  transport TEXT NOT NULL,
  binding_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'queued' CHECK(status IN ('queued', 'sending', 'sent', 'failed', 'unknown', 'expired')),
  attempts INTEGER NOT NULL DEFAULT 0,
  available_at TEXT NOT NULL,
  expires_at TEXT,
  external_message_id TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(binding_id) REFERENCES transport_bindings(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS visual_jobs (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL UNIQUE,
  intent_json TEXT NOT NULL,
  method TEXT NOT NULL CHECK(method IN ('deterministic', 'generative')),
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK(status IN ('queued', 'processing', 'completed', 'failed')),
  attempts INTEGER NOT NULL DEFAULT 0,
  available_at TEXT NOT NULL,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS visual_assets (
  id TEXT PRIMARY KEY,
  job_id TEXT NOT NULL UNIQUE,
  event_id TEXT NOT NULL,
  method TEXT NOT NULL CHECK(method IN ('deterministic', 'generative')),
  trust_tier TEXT NOT NULL CHECK(trust_tier IN ('A', 'B')),
  renderer_version TEXT NOT NULL,
  path TEXT NOT NULL,
  filename TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL,
  byte_count INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  title TEXT NOT NULL,
  caption TEXT NOT NULL,
  alt_text TEXT NOT NULL,
  provenance_json TEXT NOT NULL DEFAULT '{}',
  validation_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(job_id) REFERENCES visual_jobs(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS slack_file_outbox (
  id TEXT PRIMARY KEY,
  visual_job_id TEXT NOT NULL,
  visual_asset_id TEXT,
  binding_id TEXT NOT NULL,
  depends_on_delivery_id TEXT NOT NULL,
  channel_id TEXT NOT NULL,
  thread_id TEXT,
  idempotency_key TEXT NOT NULL UNIQUE,
  purpose TEXT NOT NULL DEFAULT 'response_visual',
  status TEXT NOT NULL DEFAULT 'waiting_asset'
    CHECK(status IN ('waiting_asset', 'queued', 'ticket_acquiring', 'ticket_acquired', 'uploading',
                     'bytes_uploaded', 'completing', 'sent', 'failed', 'unknown', 'expired')),
  attempts INTEGER NOT NULL DEFAULT 0,
  available_at TEXT NOT NULL,
  expires_at TEXT,
  slack_file_id TEXT,
  external_message_id TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(visual_job_id) REFERENCES visual_jobs(id) ON DELETE CASCADE,
  FOREIGN KEY(visual_asset_id) REFERENCES visual_assets(id) ON DELETE SET NULL,
  FOREIGN KEY(binding_id) REFERENCES transport_bindings(id) ON DELETE CASCADE,
  FOREIGN KEY(depends_on_delivery_id) REFERENCES delivery_outbox(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS interaction_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor_parent_id TEXT,
  action TEXT NOT NULL,
  subject_type TEXT,
  subject_id TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(actor_parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS onboarding_checkpoints (
  checkpoint TEXT PRIMARY KEY,
  status TEXT NOT NULL CHECK(status IN ('pending', 'pass', 'fail')),
  evidence_json TEXT NOT NULL DEFAULT '{}',
  verified_at TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_preferences (
  scope_type TEXT NOT NULL CHECK(scope_type IN ('household', 'child')),
  scope_id TEXT NOT NULL,
  profile_json TEXT NOT NULL DEFAULT '{}',
  source TEXT NOT NULL DEFAULT 'parent',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(scope_type, scope_id)
);

CREATE TABLE IF NOT EXISTS onboarding_reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL,
  brain_config_hash TEXT NOT NULL,
  response_hash TEXT NOT NULL,
  workflow TEXT NOT NULL,
  factuality TEXT NOT NULL CHECK(factuality IN ('pass', 'retry')),
  grade_fit TEXT NOT NULL CHECK(grade_fit IN ('pass', 'retry')),
  curiosity_value TEXT NOT NULL CHECK(curiosity_value IN ('pass', 'retry')),
  parent_effort TEXT NOT NULL CHECK(parent_effort IN ('pass', 'retry')),
  note TEXT,
  decision TEXT NOT NULL CHECK(decision IN ('pass', 'retry')),
  created_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS resource_collections (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  provider TEXT NOT NULL,
  license_scope TEXT NOT NULL,
  source_url TEXT,
  private INTEGER NOT NULL DEFAULT 1,
  indexed_at TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS resource_units (
  id TEXT PRIMARY KEY,
  collection_id TEXT NOT NULL,
  title TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  source_url TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(collection_id, ordinal),
  FOREIGN KEY(collection_id) REFERENCES resource_collections(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS resource_documents (
  id TEXT PRIMARY KEY,
  unit_id TEXT NOT NULL,
  title TEXT NOT NULL,
  document_type TEXT NOT NULL,
  source_path TEXT NOT NULL,
  source_url TEXT,
  sha256 TEXT NOT NULL,
  page_count INTEGER,
  byte_count INTEGER,
  content_access TEXT NOT NULL DEFAULT 'metadata_only',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  indexed_at TEXT NOT NULL,
  UNIQUE(unit_id, sha256),
  FOREIGN KEY(unit_id) REFERENCES resource_units(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS resource_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  heading TEXT,
  content TEXT NOT NULL,
  token_estimate INTEGER NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  UNIQUE(document_id, ordinal),
  FOREIGN KEY(document_id) REFERENCES resource_documents(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS resource_chunks_fts USING fts5(
  content,
  heading,
  content='resource_chunks',
  content_rowid='id',
  tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS resource_chunks_ai AFTER INSERT ON resource_chunks BEGIN
  INSERT INTO resource_chunks_fts(rowid, content, heading) VALUES (new.id, new.content, new.heading);
END;
CREATE TRIGGER IF NOT EXISTS resource_chunks_ad AFTER DELETE ON resource_chunks BEGIN
  INSERT INTO resource_chunks_fts(resource_chunks_fts, rowid, content, heading)
  VALUES ('delete', old.id, old.content, old.heading);
END;
CREATE TRIGGER IF NOT EXISTS resource_chunks_au AFTER UPDATE ON resource_chunks BEGIN
  INSERT INTO resource_chunks_fts(resource_chunks_fts, rowid, content, heading)
  VALUES ('delete', old.id, old.content, old.heading);
  INSERT INTO resource_chunks_fts(rowid, content, heading) VALUES (new.id, new.content, new.heading);
END;
"""


LEGACY_COLUMNS: dict[str, dict[str, str]] = {
    "children": {"updated_at": "TEXT"},
    "observations": {"event_id": "TEXT"},
    "evidence": {"episode_id": "TEXT"},
    "feedback": {"event_id": "TEXT"},
    "experiences": {"source_event_id": "TEXT"},
    "artifacts": {
        "sha256": "TEXT",
        "validation_json": "TEXT NOT NULL DEFAULT '{}'",
        "approval_status": "TEXT NOT NULL DEFAULT 'unreviewed'",
    },
    "events": {
        "payload_hash": "TEXT",
        "processed_at": "TEXT",
        "result_json": "TEXT",
        "error": "TEXT",
    },
    "jobs": {
        "idempotency_key": "TEXT",
        "event_id": "TEXT",
        "available_at": "TEXT",
        "leased_at": "TEXT",
        "lease_owner": "TEXT",
        "run_id": "INTEGER",
    },
    "runs": {"event_id": "TEXT", "policy_json": "TEXT", "result_json": "TEXT"},
    "household_settings": {
        "resource_context_mode": "TEXT NOT NULL DEFAULT 'metadata_only' CHECK(resource_context_mode IN ('metadata_only', 'selected_excerpts'))",
        "visual_mode": "TEXT NOT NULL DEFAULT 'deterministic' CHECK(visual_mode IN ('off', 'deterministic', 'decorative'))",
    },
    "onboarding_reviews": {
        "brain_config_hash": "TEXT NOT NULL DEFAULT 'legacy'",
        "response_hash": "TEXT NOT NULL DEFAULT 'legacy'",
        "workflow": "TEXT NOT NULL DEFAULT 'legacy'",
    },
}


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def jload(value: str | None, default: Any | None = None) -> Any:
    if value is None or value == "":
        return {} if default is None else default
    return json.loads(value)


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _open(path)
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _backup_existing(path: Path, version: int) -> Path | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    source = sqlite3.connect(path)
    try:
        if not _tables(source):
            return None
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_path = path.with_name(f"{path.name}.backup-v{version}-{stamp}")
        target = sqlite3.connect(backup_path)
        try:
            source.backup(target)
        finally:
            target.close()
        shutil.copymode(path, backup_path)
        return backup_path
    finally:
        source.close()


def _ensure_legacy_columns(conn: sqlite3.Connection) -> None:
    tables = _tables(conn)
    for table, columns in LEGACY_COLUMNS.items():
        if table not in tables:
            continue
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, declaration in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def init_db(db_path: str | Path) -> Path | None:
    """Create or migrate the database, returning a migration backup path when made."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    version = 0
    has_tables = False
    if path.exists() and path.stat().st_size:
        probe = sqlite3.connect(path)
        try:
            version = int(probe.execute("PRAGMA user_version").fetchone()[0])
            has_tables = bool(_tables(probe))
        finally:
            probe.close()
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"database schema v{version} is newer than supported v{SCHEMA_VERSION}")
    backup = _backup_existing(path, version) if has_tables and version < SCHEMA_VERSION else None
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        _ensure_legacy_columns(conn)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs(idempotency_key) WHERE idempotency_key IS NOT NULL"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_ready ON jobs(status, available_at, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_child_created ON events(child_id, created_at DESC)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_child_time ON observations(child_id, occurred_at DESC)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_child_time ON nodes(child_id, last_seen DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_child_time ON episodes(child_id, last_event_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episode_members_episode ON episode_memberships(episode_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_episode ON evidence(episode_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_resources_unit ON resource_documents(unit_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bindings_lookup ON transport_bindings(transport,team_id,user_id,status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inbox_status ON capture_inbox(status,created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outbox_ready ON delivery_outbox(transport,status,available_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visual_jobs_ready ON visual_jobs(status,available_at,created_at)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_slack_files_ready ON slack_file_outbox(status,available_at,created_at)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_onboarding_reviews_event ON onboarding_reviews(event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_event ON feedback(event_id,created_at)")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    path.chmod(0o600)
    if backup:
        backup.chmod(0o600)
    return backup
