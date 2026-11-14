from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any

from .deletion_ledger import (
    assert_no_resurrected_events,
    managed_data_lifecycle,
    migration_backup_path,
    reconcile_managed_backups,
    sync_database_tombstones,
)
from .personalization_policy import FEATURE_SCHEMA_VERSION as PERSONALIZATION_FEATURE_SCHEMA_VERSION
from .personalization_policy import POLICY_DEFINITION as PERSONALIZATION_POLICY_DEFINITION
from .personalization_policy import POLICY_DEFINITION_HASH as PERSONALIZATION_POLICY_DEFINITION_HASH
from .personalization_policy import POLICY_ID as PERSONALIZATION_POLICY_ID
from .personalization_policy import POLICY_VERSION as PERSONALIZATION_POLICY_VERSION
from .personalization_policy import replay_case_payload as personalization_replay_case_payload
from .research_authority_ledger import (
    authority_backup_roots,
    finalize_research_authority,
    initialize_or_assert_research_authority,
    record_research_authority,
)
from .research_policy import (
    RESEARCH_FEATURE_SCHEMA_VERSION,
    RESEARCH_POLICY_DEFINITION,
    RESEARCH_POLICY_DEFINITION_HASH,
    RESEARCH_POLICY_ID,
    RESEARCH_POLICY_VERSION,
    archival_research_evaluation_is_valid,
    calibration_case_hash,
    canonical_research_evaluation,
    history_from_evidence_rows,
    posterior_from_history,
    research_evaluation_matches,
)

SCHEMA_VERSION = 26
_INIT_LOCK = RLock()
_INITIALIZED_DATABASES: set[tuple[str, int, int, str]] = set()


def _database_file_identity(path: Path) -> tuple[str, int, int] | None:
    """Return a cheap process-local identity for an existing SQLite file."""

    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not stat.st_size:
        return None
    return (str(path.resolve()), int(stat.st_dev), int(stat.st_ino))


def _database_initialization_key(
    path: Path,
    repo_root: str | Path | None,
) -> tuple[str, int, int, str] | None:
    identity = _database_file_identity(path)
    if identity is None:
        return None
    root = str(Path(repo_root).resolve()) if repo_root is not None else ""
    return (*identity, root)


def _remember_initialized_database(path: Path, repo_root: str | Path | None) -> None:
    key = _database_initialization_key(path, repo_root)
    if key is None:
        return
    resolved, root = key[0], key[3]
    _INITIALIZED_DATABASES.difference_update(
        {cached for cached in _INITIALIZED_DATABASES if cached[0] == resolved and cached[3] == root}
    )
    _INITIALIZED_DATABASES.add(key)


def _has_pending_family_file_deletions(path: Path) -> bool:
    """Keep deletion recovery active while skipping repeated schema work."""

    try:
        probe = sqlite3.connect(path, timeout=0.1)
        try:
            return bool(
                probe.execute(
                    "SELECT 1 FROM family_file_deletion_journal LIMIT 1"
                ).fetchone()
            )
        finally:
            probe.close()
    except sqlite3.DatabaseError:
        # A cached current schema should contain this table. Fall through to the
        # complete initializer when that assumption cannot be confirmed.
        return True

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

CREATE TABLE IF NOT EXISTS family_database_identity (
  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
  database_id TEXT NOT NULL UNIQUE CHECK(length(database_id) BETWEEN 16 AND 128),
  created_at TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS output_evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL,
  visual_asset_id TEXT,
  artifact_id TEXT,
  response_rating TEXT NOT NULL
    CHECK(response_rating IN ('useful','needs_work','not_useful')),
  visual_rating TEXT NOT NULL
    CHECK(visual_rating IN ('useful','pretty_not_useful','missing_needed','misleading','not_needed')),
  preferred_response_shape TEXT
    CHECK(preferred_response_shape IN ('parent_chat','quick_answer','learning_thread')),
  preferred_visual_mix TEXT
    CHECK(preferred_visual_mix IN ('none','imagination','activity_aid','both')),
  note TEXT,
  source TEXT NOT NULL DEFAULT 'local_eval_dashboard',
  created_at TEXT NOT NULL,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(visual_asset_id) REFERENCES visual_assets(id) ON DELETE SET NULL,
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
  weekly_checkins_enabled INTEGER NOT NULL DEFAULT 0
    CHECK(weekly_checkins_enabled IN (0, 1)),
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
  strategy TEXT NOT NULL DEFAULT 'deterministic',
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
  strategy TEXT NOT NULL DEFAULT 'deterministic',
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

CREATE TABLE IF NOT EXISTS researched_image_cache (
  query_sha256 TEXT PRIMARY KEY,
  query_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('approved', 'failed')),
  asset_sha256 TEXT,
  asset_json TEXT,
  rights_json TEXT,
  review_json TEXT,
  source_name TEXT,
  source_item_id TEXT,
  canonical_url TEXT,
  failure_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
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
  active_attempt_id TEXT,
  lease_token_hash TEXT,
  available_at TEXT NOT NULL,
  expires_at TEXT,
  slack_file_id TEXT,
  external_message_id TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  CHECK((active_attempt_id IS NULL AND lease_token_hash IS NULL) OR
        (active_attempt_id IS NOT NULL AND length(lease_token_hash)=64)),
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

CREATE TABLE IF NOT EXISTS agent_sessions (
  id TEXT PRIMARY KEY,
  origin TEXT NOT NULL CHECK(origin IN ('slack','schedule','cli')),
  transport TEXT,
  binding_id TEXT,
  conversation_ref TEXT NOT NULL,
  thread_ref TEXT NOT NULL,
  child_id TEXT,
  capability_id TEXT NOT NULL DEFAULT 'parent_chat',
  state TEXT NOT NULL DEFAULT 'active' CHECK(state IN ('active','waiting','completed','failed')),
  summary_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(origin,binding_id,conversation_ref,thread_ref),
  FOREIGN KEY(binding_id) REFERENCES transport_bindings(id) ON DELETE SET NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS agent_messages (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('user','assistant','tool','system')),
  kind TEXT NOT NULL DEFAULT 'message',
  content TEXT NOT NULL,
  event_id TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  UNIQUE(session_id,ordinal),
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS agent_session_preference_events (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  category TEXT NOT NULL CHECK(category IN (
    'answer_style','visual_style','artifact_style','activity_style','interaction_style'
  )),
  operation TEXT NOT NULL CHECK(operation IN ('set','clear')),
  value TEXT,
  source_message_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(session_id,ordinal),
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(source_message_id) REFERENCES agent_messages(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS inbound_assets (
  id TEXT PRIMARY KEY,
  binding_id TEXT NOT NULL,
  external_event_id TEXT NOT NULL,
  external_ref_hash TEXT NOT NULL,
  inbox_id TEXT,
  session_id TEXT,
  status TEXT NOT NULL CHECK(status IN ('ready','unavailable')),
  media_type TEXT CHECK(media_type IS NULL OR media_type IN ('image/jpeg','image/png','image/webp')),
  byte_count INTEGER NOT NULL DEFAULT 0 CHECK(byte_count >= 0 AND byte_count <= 10000000),
  sha256 TEXT,
  path TEXT,
  observation_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(binding_id,external_event_id,external_ref_hash),
  FOREIGN KEY(binding_id) REFERENCES transport_bindings(id) ON DELETE CASCADE,
  FOREIGN KEY(inbox_id) REFERENCES capture_inbox(id) ON DELETE SET NULL,
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_turns (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  provider TEXT,
  model TEXT,
  status TEXT NOT NULL CHECK(status IN ('running','waiting','completed','failed')),
  request_hash TEXT NOT NULL,
  response_json TEXT NOT NULL DEFAULT '{}',
  usage_json TEXT NOT NULL DEFAULT '{}',
  policy_hash TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  UNIQUE(session_id,ordinal),
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tool_calls (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  tool_name TEXT NOT NULL,
  tool_version TEXT NOT NULL,
  argument_hash TEXT NOT NULL,
  arguments_json TEXT NOT NULL,
  risk TEXT NOT NULL,
  decision TEXT NOT NULL CHECK(decision IN ('allowed','awaiting_approval','denied')),
  status TEXT NOT NULL CHECK(status IN ('proposed','awaiting_approval','running','succeeded','failed','unknown','denied','expired')),
  result_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(session_id,turn_id,ordinal),
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(turn_id) REFERENCES agent_turns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS approval_requests (
  id TEXT PRIMARY KEY,
  tool_call_id TEXT NOT NULL UNIQUE,
  scope_hash TEXT NOT NULL,
  decision TEXT NOT NULL DEFAULT 'pending' CHECK(decision IN ('pending','approved','denied','expired')),
  actor_parent_id TEXT,
  expires_at TEXT NOT NULL,
  decided_at TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(tool_call_id) REFERENCES tool_calls(id) ON DELETE CASCADE,
  FOREIGN KEY(actor_parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS capability_runs (
  id TEXT PRIMARY KEY,
  session_id TEXT,
  capability_id TEXT NOT NULL,
  capability_version TEXT NOT NULL,
  skill_versions_json TEXT NOT NULL DEFAULT '{}',
  status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','rejected')),
  source_event_id TEXT,
  result_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE SET NULL,
  FOREIGN KEY(source_event_id) REFERENCES events(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS release_units (
  id TEXT PRIMARY KEY,
  capability_run_id TEXT NOT NULL,
  parent_release_unit_id TEXT,
  unit_type TEXT NOT NULL CHECK(unit_type IN ('answer','interaction','artifact','visual','status')),
  ordinal INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('draft','reviewed','queued','delivered','failed','superseded')),
  content_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(capability_run_id,ordinal),
  FOREIGN KEY(capability_run_id) REFERENCES capability_runs(id) ON DELETE CASCADE,
  FOREIGN KEY(parent_release_unit_id) REFERENCES release_units(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS interaction_instances (
  id TEXT PRIMARY KEY,
  session_id TEXT,
  binding_id TEXT NOT NULL,
  plan_json TEXT NOT NULL,
  options_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','completed','expired','cancelled')),
  expires_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE SET NULL,
  FOREIGN KEY(binding_id) REFERENCES transport_bindings(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS interaction_events (
  id TEXT PRIMARY KEY,
  interaction_id TEXT NOT NULL,
  option_token_hash TEXT NOT NULL,
  intent TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  actor_parent_id TEXT,
  external_event_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(interaction_id,external_event_id),
  FOREIGN KEY(interaction_id) REFERENCES interaction_instances(id) ON DELETE CASCADE,
  FOREIGN KEY(actor_parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS schedule_runs (
  id TEXT PRIMARY KEY,
  schedule_id TEXT NOT NULL,
  due_at TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','skipped')),
  session_id TEXT,
  lease_owner TEXT,
  leased_at TEXT,
  result_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(schedule_id,due_at),
  FOREIGN KEY(schedule_id) REFERENCES schedules(id) ON DELETE CASCADE,
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS artifact_file_outbox (
  id TEXT PRIMARY KEY,
  artifact_id TEXT NOT NULL,
  binding_id TEXT NOT NULL,
  depends_on_delivery_id TEXT,
  channel_id TEXT NOT NULL,
  thread_id TEXT,
  path TEXT NOT NULL,
  filename TEXT NOT NULL,
  mime_type TEXT NOT NULL CHECK(mime_type IN ('application/pdf','image/png')),
  byte_count INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  title TEXT NOT NULL,
  comment TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'queued' CHECK(status IN ('queued','ticket_acquiring','ticket_acquired','uploading','bytes_uploaded','completing','sent','failed','unknown','expired')),
  attempts INTEGER NOT NULL DEFAULT 0,
  active_attempt_id TEXT,
  lease_token_hash TEXT,
  available_at TEXT NOT NULL,
  slack_file_id TEXT,
  external_message_id TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  CHECK((active_attempt_id IS NULL AND lease_token_hash IS NULL) OR
        (active_attempt_id IS NOT NULL AND length(lease_token_hash)=64)),
  FOREIGN KEY(artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE,
  FOREIGN KEY(binding_id) REFERENCES transport_bindings(id) ON DELETE CASCADE,
  FOREIGN KEY(depends_on_delivery_id) REFERENCES delivery_outbox(id) ON DELETE SET NULL
);
"""


# Schema v18 deliberately lives outside the legacy aggregate schema. It is an
# additive, transactionally applied migration: old graph/evidence rows remain
# available for audit, but no row is silently reinterpreted as learner evidence.
MIGRATION_18_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  applied_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_events_id_child ON events(id,child_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_id_child ON episodes(id,child_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_id_event ON runs(id,event_id);

CREATE TABLE IF NOT EXISTS concepts (
  concept_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version > 0),
  label TEXT NOT NULL,
  domain TEXT NOT NULL,
  grade_band TEXT,
  description TEXT NOT NULL DEFAULT '',
  framework TEXT NOT NULL DEFAULT 'household',
  source_uri TEXT,
  definition_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(concept_id,version)
);

CREATE TABLE IF NOT EXISTS concept_aliases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  alias TEXT NOT NULL,
  normalized_alias TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'registered',
  created_at TEXT NOT NULL,
  UNIQUE(concept_id,concept_version,normalized_alias),
  FOREIGN KEY(concept_id,concept_version) REFERENCES concepts(concept_id,version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS concept_relations (
  relation_id TEXT PRIMARY KEY,
  source_concept_id TEXT NOT NULL,
  source_concept_version INTEGER NOT NULL,
  relation TEXT NOT NULL CHECK(relation IN ('part_of','prerequisite_of','contrasts_with','applies_in')),
  target_concept_id TEXT NOT NULL,
  target_concept_version INTEGER NOT NULL,
  provenance_json TEXT NOT NULL DEFAULT '{}',
  relation_version TEXT NOT NULL,
  definition_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(source_concept_id,source_concept_version)
    REFERENCES concepts(concept_id,version) ON DELETE CASCADE,
  FOREIGN KEY(target_concept_id,target_concept_version)
    REFERENCES concepts(concept_id,version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rubrics (
  rubric_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version > 0),
  label TEXT NOT NULL,
  criteria_json TEXT NOT NULL,
  evidence_rule_id TEXT NOT NULL,
  evidence_rule_version TEXT NOT NULL,
  definition_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(rubric_id,version)
);

CREATE TABLE IF NOT EXISTS learning_tasks (
  task_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version > 0),
  title TEXT NOT NULL,
  learning_goal TEXT NOT NULL,
  strategy TEXT NOT NULL,
  difficulty TEXT NOT NULL CHECK(difficulty IN ('trivial','familiar','stretch','advanced','unknown')),
  difficulty_provenance TEXT NOT NULL
    CHECK(difficulty_provenance IN ('reviewed_task','parent_selected','estimated','unknown')),
  modalities_json TEXT NOT NULL DEFAULT '[]',
  context_key TEXT NOT NULL,
  representation_key TEXT NOT NULL,
  rubric_id TEXT NOT NULL,
  rubric_version INTEGER NOT NULL,
  definition_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(task_id,version),
  FOREIGN KEY(rubric_id,rubric_version) REFERENCES rubrics(rubric_id,version)
);

CREATE TABLE IF NOT EXISTS learning_task_concepts (
  task_id TEXT NOT NULL,
  task_version INTEGER NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('target','prerequisite','supporting')),
  PRIMARY KEY(task_id,task_version,concept_id,concept_version,role),
  FOREIGN KEY(task_id,task_version) REFERENCES learning_tasks(task_id,version) ON DELETE CASCADE,
  FOREIGN KEY(concept_id,concept_version) REFERENCES concepts(concept_id,version) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS learning_episode_plans (
  plan_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  episode_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  task_version INTEGER NOT NULL,
  source_event_id TEXT NOT NULL,
  attempt_group_id TEXT NOT NULL,
  context_key TEXT NOT NULL,
  representation_key TEXT NOT NULL,
  transfer_from_plan_id TEXT,
  status TEXT NOT NULL CHECK(status IN ('planned','offered','started','completed','cancelled')),
  plan_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(plan_id,child_id),
  UNIQUE(attempt_group_id,child_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(episode_id,child_id) REFERENCES episodes(id,child_id) ON DELETE CASCADE,
  FOREIGN KEY(source_event_id,child_id) REFERENCES events(id,child_id) ON DELETE CASCADE,
  FOREIGN KEY(task_id,task_version) REFERENCES learning_tasks(task_id,version),
  FOREIGN KEY(transfer_from_plan_id,child_id)
    REFERENCES learning_episode_plans(plan_id,child_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_annotations (
  annotation_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  run_id INTEGER,
  ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
  kind TEXT NOT NULL CHECK(kind IN ('concept_mention','learning_process')),
  label TEXT,
  text TEXT,
  author_type TEXT NOT NULL CHECK(author_type IN ('model','code','parent')),
  authority TEXT NOT NULL CHECK(authority IN ('audit_only','reviewed')),
  purpose TEXT NOT NULL CHECK(purpose IN ('semantic_audit')),
  retention_class TEXT NOT NULL CHECK(retention_class IN ('transient')),
  expires_at TEXT NOT NULL,
  annotation_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(event_id,run_id,ordinal),
  UNIQUE(annotation_id,child_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id,child_id) REFERENCES events(id,child_id) ON DELETE CASCADE,
  FOREIGN KEY(run_id,event_id) REFERENCES runs(id,event_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS learning_opportunities (
  opportunity_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  episode_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  task_version INTEGER NOT NULL,
  plan_id TEXT NOT NULL,
  rubric_id TEXT NOT NULL,
  rubric_version INTEGER NOT NULL,
  evidence_rule_id TEXT NOT NULL,
  evidence_rule_version TEXT NOT NULL,
  attempt_group_id TEXT NOT NULL,
  attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
  actor TEXT NOT NULL CHECK(actor IN ('child','parent_report','system')),
  initiation TEXT NOT NULL CHECK(initiation IN ('child','parent','engine','unknown')),
  response_mode TEXT NOT NULL CHECK(response_mode IN (
    'spoken','written','drawn','gesture_or_manipulative','selection','demonstration','symbolic','mixed','parent_report'
  )),
  assistance_level TEXT NOT NULL CHECK(assistance_level IN (
    'none','clarifying_prompt','general_prompt','hint','worked_example','joint_completion','direct_answer','unknown'
  )),
  result TEXT NOT NULL CHECK(result IN ('not_observed','correct','partial','incorrect','abandoned')),
  score REAL CHECK(score IS NULL OR (score >= 0 AND score <= 1)),
  criterion_results_json TEXT NOT NULL,
  scorer TEXT NOT NULL,
  scorer_version TEXT NOT NULL,
  source_type TEXT NOT NULL CHECK(source_type IN (
    'direct_observation','parent_report','system_generated','model_annotation','output_feedback'
  )),
  independence_status TEXT NOT NULL
    CHECK(independence_status IN ('eligible','same_episode','diagnostic','system','excluded','uncertain')),
  context_key TEXT NOT NULL,
  representation_key TEXT NOT NULL,
  is_transfer INTEGER NOT NULL DEFAULT 0 CHECK(is_transfer IN (0,1)),
  is_trivial INTEGER NOT NULL DEFAULT 0 CHECK(is_trivial IN (0,1)),
  child_continued INTEGER NOT NULL DEFAULT 0 CHECK(child_continued IN (0,1)),
  child_new_question INTEGER NOT NULL DEFAULT 0 CHECK(child_new_question IN (0,1)),
  retention_class TEXT NOT NULL CHECK(retention_class IN ('transient','learning_record','parent_confirmed')),
  opportunity_hash TEXT NOT NULL,
  opportunity_id_hash TEXT,
  occurred_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(opportunity_id,child_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id,child_id) REFERENCES events(id,child_id) ON DELETE CASCADE,
  FOREIGN KEY(episode_id,child_id) REFERENCES episodes(id,child_id) ON DELETE CASCADE,
  FOREIGN KEY(task_id,task_version) REFERENCES learning_tasks(task_id,version),
  FOREIGN KEY(plan_id,child_id) REFERENCES learning_episode_plans(plan_id,child_id),
  FOREIGN KEY(rubric_id,rubric_version) REFERENCES rubrics(rubric_id,version)
);

CREATE TABLE IF NOT EXISTS opportunity_assistance_events (
  assistance_id TEXT PRIMARY KEY,
  opportunity_id TEXT NOT NULL,
  child_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK(ordinal > 0),
  assistance_level TEXT NOT NULL CHECK(assistance_level IN (
    'clarifying_prompt','general_prompt','hint','worked_example','joint_completion','direct_answer'
  )),
  provider TEXT NOT NULL CHECK(provider IN ('parent','engine','other')),
  occurred_at TEXT NOT NULL,
  UNIQUE(opportunity_id,ordinal),
  FOREIGN KEY(opportunity_id,child_id)
    REFERENCES learning_opportunities(opportunity_id,child_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_records (
  evidence_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  opportunity_id TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  construct TEXT NOT NULL CHECK(construct IN ('exposure','knowledge_performance','interest')),
  observation_kind TEXT NOT NULL,
  stance TEXT NOT NULL CHECK(stance IN ('supports','contradicts','context_only')),
  result TEXT NOT NULL CHECK(result IN ('not_observed','correct','partial','incorrect','abandoned')),
  score REAL CHECK(score IS NULL OR (score >= 0 AND score <= 1)),
  rubric_id TEXT NOT NULL,
  rubric_version INTEGER NOT NULL,
  scored_criteria_json TEXT NOT NULL,
  source_type TEXT NOT NULL,
  source_reliability TEXT NOT NULL CHECK(source_reliability IN ('direct','reported','generated','unknown')),
  evidence_strength TEXT NOT NULL CHECK(evidence_strength IN ('none','weak','moderate','strong')),
  projection_eligible INTEGER NOT NULL CHECK(projection_eligible IN (0,1)),
  episode_id TEXT NOT NULL,
  context_key TEXT NOT NULL,
  representation_key TEXT NOT NULL,
  assistance_level TEXT NOT NULL,
  attempt_number INTEGER NOT NULL,
  is_transfer INTEGER NOT NULL CHECK(is_transfer IN (0,1)),
  is_trivial INTEGER NOT NULL CHECK(is_trivial IN (0,1)),
  rule_id TEXT NOT NULL,
  rule_version TEXT NOT NULL,
  retention_class TEXT NOT NULL,
  expires_at TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(evidence_id,child_id),
  UNIQUE(opportunity_id,concept_id,concept_version,construct,observation_kind),
  FOREIGN KEY(opportunity_id,child_id)
    REFERENCES learning_opportunities(opportunity_id,child_id) ON DELETE CASCADE,
  FOREIGN KEY(concept_id,concept_version) REFERENCES concepts(concept_id,version) ON DELETE RESTRICT,
  FOREIGN KEY(episode_id,child_id) REFERENCES episodes(id,child_id) ON DELETE RESTRICT,
  FOREIGN KEY(rubric_id,rubric_version) REFERENCES rubrics(rubric_id,version) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS evidence_corrections (
  correction_id TEXT PRIMARY KEY,
  evidence_id TEXT NOT NULL,
  target_evidence_hash TEXT NOT NULL,
  child_id TEXT NOT NULL,
  action TEXT NOT NULL CHECK(action IN ('exclude','restore','correct')),
  replacement_json TEXT NOT NULL DEFAULT '{}',
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator','system')),
  reason TEXT NOT NULL,
  correction_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(correction_id,child_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS projection_runs (
  run_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  projector_version TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  output_digest TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS learner_state_projections (
  child_id TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  state TEXT NOT NULL CHECK(state IN (
    'not_observed_in_harness','exposure_recorded','practicing_with_support','demonstrated_in_context',
    'independently_demonstrated','transfer_observed'
  )),
  evidence_strength TEXT NOT NULL CHECK(evidence_strength IN ('none','weak','moderate','strong')),
  uncertainty REAL NOT NULL CHECK(uncertainty >= 0 AND uncertainty <= 1),
  support_level TEXT NOT NULL CHECK(support_level IN ('not_observed','independent','some_support','substantial_support','unknown')),
  eligible_opportunity_count INTEGER NOT NULL,
  eligible_episode_count INTEGER NOT NULL,
  context_diversity INTEGER NOT NULL,
  representation_diversity INTEGER NOT NULL,
  contradicting_count INTEGER NOT NULL,
  last_observed_at TEXT,
  rule_id TEXT NOT NULL,
  rule_version TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  rationale TEXT NOT NULL,
  computed_at TEXT NOT NULL,
  PRIMARY KEY(child_id,concept_id,concept_version),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(concept_id,concept_version) REFERENCES concepts(concept_id,version) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS learner_projection_evidence (
  child_id TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  evidence_id TEXT NOT NULL,
  stance TEXT NOT NULL CHECK(stance IN ('supports','contradicts','context_only')),
  PRIMARY KEY(child_id,concept_id,concept_version,evidence_id),
  FOREIGN KEY(child_id,concept_id,concept_version)
    REFERENCES learner_state_projections(child_id,concept_id,concept_version) ON DELETE CASCADE,
  FOREIGN KEY(evidence_id,child_id) REFERENCES evidence_records(evidence_id,child_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS interest_projections (
  child_id TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  phase TEXT NOT NULL CHECK(phase IN (
    'not_observed','triggered_situational','maintained_situational','emerging_recurrent','sustained_parent_confirmed'
  )),
  child_initiated_count INTEGER NOT NULL,
  continued_or_question_count INTEGER NOT NULL,
  eligible_episode_count INTEGER NOT NULL,
  context_diversity INTEGER NOT NULL,
  uncertainty REAL NOT NULL CHECK(uncertainty >= 0 AND uncertainty <= 1),
  next_review_at TEXT,
  rule_id TEXT NOT NULL,
  rule_version TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  rationale TEXT NOT NULL,
  computed_at TEXT NOT NULL,
  PRIMARY KEY(child_id,concept_id,concept_version),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(concept_id,concept_version) REFERENCES concepts(concept_id,version) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS interest_projection_evidence (
  child_id TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  evidence_id TEXT NOT NULL,
  stance TEXT NOT NULL CHECK(stance IN ('supports','contradicts','context_only')),
  PRIMARY KEY(child_id,concept_id,concept_version,evidence_id),
  FOREIGN KEY(child_id,concept_id,concept_version)
    REFERENCES interest_projections(child_id,concept_id,concept_version) ON DELETE CASCADE,
  FOREIGN KEY(evidence_id,child_id) REFERENCES evidence_records(evidence_id,child_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS derived_claims_v2 (
  claim_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  construct TEXT NOT NULL CHECK(construct IN ('knowledge','interest')),
  value TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('shadow','parent_confirmed','expired')),
  rule_id TEXT NOT NULL,
  rule_version TEXT NOT NULL,
  input_digest TEXT NOT NULL,
  rationale TEXT NOT NULL,
  computed_at TEXT NOT NULL,
  UNIQUE(child_id,concept_id,concept_version,construct),
  UNIQUE(claim_id,child_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(concept_id,concept_version) REFERENCES concepts(concept_id,version) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS derived_claim_evidence_v2 (
  claim_id TEXT NOT NULL,
  child_id TEXT NOT NULL,
  evidence_id TEXT NOT NULL,
  stance TEXT NOT NULL CHECK(stance IN ('supports','contradicts','context_only')),
  PRIMARY KEY(claim_id,evidence_id),
  FOREIGN KEY(claim_id,child_id) REFERENCES derived_claims_v2(claim_id,child_id) ON DELETE CASCADE,
  FOREIGN KEY(evidence_id,child_id) REFERENCES evidence_records(evidence_id,child_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS preference_support_records (
  record_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('explicit_preference','support_need')),
  value_json TEXT NOT NULL,
  source_event_id TEXT,
  source_type TEXT NOT NULL CHECK(source_type IN ('parent_statement','child_statement','operator')),
  status TEXT NOT NULL CHECK(status IN ('active','expired','withdrawn')),
  sensitivity TEXT NOT NULL CHECK(sensitivity IN ('ordinary','sensitive')),
  review_at TEXT,
  expires_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(record_id,child_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(source_event_id,child_id) REFERENCES events(id,child_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_source_tombstones (
  tombstone_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  target_kind TEXT NOT NULL CHECK(target_kind IN ('learning_opportunity')),
  target_id_hash TEXT NOT NULL,
  reason TEXT NOT NULL,
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator')),
  deleted_at TEXT NOT NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_task_concepts_concept
  ON learning_task_concepts(concept_id,concept_version,role);
CREATE INDEX IF NOT EXISTS idx_opportunities_child_time
  ON learning_opportunities(child_id,occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_opportunities_episode
  ON learning_opportunities(episode_id,occurred_at);
CREATE INDEX IF NOT EXISTS idx_evidence_child_concept
  ON evidence_records(child_id,concept_id,concept_version,created_at);
CREATE INDEX IF NOT EXISTS idx_evidence_opportunity ON evidence_records(opportunity_id);
CREATE INDEX IF NOT EXISTS idx_corrections_evidence
  ON evidence_corrections(evidence_id,created_at,correction_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tombstone_target
  ON evidence_source_tombstones(child_id,target_kind,target_id_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_opportunity_attempt
  ON learning_opportunities(plan_id,attempt_number);

INSERT INTO schema_migrations(version,name,applied_at)
VALUES(18,'versioned evidence spine',strftime('%Y-%m-%dT%H:%M:%fZ','now'))
ON CONFLICT(version) DO NOTHING;
PRAGMA user_version = 18;
"""


MIGRATION_19_SQL = """
CREATE TABLE IF NOT EXISTS pedagogy_plans (
  plan_id TEXT PRIMARY KEY,
  plan_hash TEXT NOT NULL UNIQUE,
  event_id TEXT NOT NULL,
  child_id TEXT NOT NULL,
  initial_run_id INTEGER,
  workflow TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('approved','generated','rejected','superseded')),
  proposal_json TEXT NOT NULL,
  approved_json TEXT NOT NULL,
  registry_version TEXT NOT NULL,
  registry_digest TEXT NOT NULL,
  contract_version TEXT NOT NULL,
  validator_version TEXT NOT NULL,
  routing_provenance TEXT NOT NULL,
  supersedes_plan_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE(event_id,workflow,contract_version),
  UNIQUE(plan_id,child_id),
  UNIQUE(plan_id,event_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(initial_run_id) REFERENCES runs(id) ON DELETE SET NULL,
  FOREIGN KEY(supersedes_plan_id,child_id)
    REFERENCES pedagogy_plans(plan_id,child_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_pedagogy_plans_child_time
  ON pedagogy_plans(child_id,created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pedagogy_plans_event
  ON pedagogy_plans(event_id,workflow,status);

CREATE TRIGGER IF NOT EXISTS trg_pedagogy_plan_authority_insert
BEFORE INSERT ON pedagogy_plans
WHEN NOT EXISTS (
  SELECT 1 FROM events e WHERE e.id=NEW.event_id AND e.child_id=NEW.child_id
) OR (
  NEW.initial_run_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM runs r WHERE r.id=NEW.initial_run_id AND r.event_id=NEW.event_id
  )
)
BEGIN SELECT RAISE(ABORT,'invalid pedagogy plan authority binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_pedagogy_plan_authority_update
BEFORE UPDATE OF event_id,child_id,initial_run_id,supersedes_plan_id ON pedagogy_plans
WHEN NOT EXISTS (
  SELECT 1 FROM events e WHERE e.id=NEW.event_id AND e.child_id=NEW.child_id
) OR (
  NEW.initial_run_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM runs r WHERE r.id=NEW.initial_run_id AND r.event_id=NEW.event_id
  )
)
BEGIN SELECT RAISE(ABORT,'invalid pedagogy plan authority binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_pedagogy_plan_supersession_insert
BEFORE INSERT ON pedagogy_plans
WHEN (
  NEW.routing_provenance='artifact_derivation' AND (
    NEW.supersedes_plan_id IS NULL OR NOT EXISTS (
      SELECT 1 FROM pedagogy_plans parent
      WHERE parent.plan_id=NEW.supersedes_plan_id
        AND parent.event_id=NEW.event_id AND parent.child_id=NEW.child_id
    )
  )
) OR (
  NEW.supersedes_plan_id IS NOT NULL
  AND NEW.routing_provenance NOT IN ('artifact_derivation','visual_revision')
)
BEGIN SELECT RAISE(ABORT,'invalid pedagogy plan supersession binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_pedagogy_plan_supersession_update
BEFORE UPDATE OF event_id,child_id,routing_provenance,supersedes_plan_id ON pedagogy_plans
WHEN (
  NEW.routing_provenance='artifact_derivation' AND (
    NEW.supersedes_plan_id IS NULL OR NOT EXISTS (
      SELECT 1 FROM pedagogy_plans parent
      WHERE parent.plan_id=NEW.supersedes_plan_id
        AND parent.event_id=NEW.event_id AND parent.child_id=NEW.child_id
    )
  )
) OR (
  NEW.supersedes_plan_id IS NOT NULL
  AND NEW.routing_provenance NOT IN ('artifact_derivation','visual_revision')
)
BEGIN SELECT RAISE(ABORT,'invalid pedagogy plan supersession binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_pedagogy_plan_immutable
BEFORE UPDATE OF plan_hash,event_id,child_id,workflow,proposal_json,approved_json,
  registry_version,registry_digest,contract_version,validator_version,routing_provenance,
  supersedes_plan_id,created_at ON pedagogy_plans
BEGIN SELECT RAISE(ABORT,'approved pedagogy plan semantics are immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_pedagogy_plan_lineage_delete
BEFORE DELETE ON pedagogy_plans
BEGIN
  DELETE FROM feedback
    WHERE pedagogy_plan_id=OLD.plan_id AND pedagogy_plan_hash=OLD.plan_hash;
  DELETE FROM artifacts
    WHERE pedagogy_plan_id=OLD.plan_id AND pedagogy_plan_hash=OLD.plan_hash;
  DELETE FROM experiences
    WHERE pedagogy_plan_id=OLD.plan_id AND pedagogy_plan_hash=OLD.plan_hash;
  DELETE FROM visual_jobs
    WHERE pedagogy_plan_id=OLD.plan_id AND pedagogy_plan_hash=OLD.plan_hash;
  DELETE FROM actions
    WHERE pedagogy_plan_id=OLD.plan_id AND pedagogy_plan_hash=OLD.plan_hash;
  DELETE FROM responses
    WHERE pedagogy_plan_id=OLD.plan_id AND pedagogy_plan_hash=OLD.plan_hash;
  UPDATE runs SET pedagogy_plan_id=NULL,pedagogy_plan_hash=NULL
    WHERE pedagogy_plan_id=OLD.plan_id AND pedagogy_plan_hash=OLD.plan_hash;
END;

CREATE TRIGGER IF NOT EXISTS trg_plan_bound_event_semantics_immutable
BEFORE UPDATE OF id,type,child_id,text,source,metadata_json,payload_hash,created_at ON events
WHEN EXISTS (SELECT 1 FROM pedagogy_plans p WHERE p.event_id=OLD.id)
BEGIN SELECT RAISE(ABORT,'plan-bound event semantics are immutable'); END;

CREATE TRIGGER IF NOT EXISTS trg_event_pedagogy_lineage_delete
BEFORE DELETE ON events
BEGIN
  DELETE FROM feedback WHERE event_id=OLD.id;
  DELETE FROM artifacts WHERE experience_id IN (
    SELECT id FROM experiences WHERE source_event_id=OLD.id
  );
  DELETE FROM experiences WHERE source_event_id=OLD.id;
  DELETE FROM actions WHERE event_id=OLD.id;
  UPDATE runs SET pedagogy_plan_id=NULL,pedagogy_plan_hash=NULL WHERE event_id=OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_child_private_events_delete
BEFORE DELETE ON children
BEGIN
  DELETE FROM events WHERE child_id=OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_response_pedagogy_binding_insert
BEFORE INSERT ON responses
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.event_id=NEW.event_id
)
BEGIN SELECT RAISE(ABORT,'invalid response pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_response_pedagogy_binding_update
BEFORE UPDATE OF pedagogy_plan_id,pedagogy_plan_hash,event_id ON responses
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.event_id=NEW.event_id
)
BEGIN SELECT RAISE(ABORT,'invalid response pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_run_pedagogy_binding_insert
BEFORE INSERT ON runs
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL) OR (
  NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM pedagogy_plans p
    WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
      AND p.event_id=NEW.event_id
  )
)
BEGIN SELECT RAISE(ABORT,'invalid run pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_run_pedagogy_binding_update
BEFORE UPDATE OF event_id,pedagogy_plan_id,pedagogy_plan_hash ON runs
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL) OR (
  NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM pedagogy_plans p
    WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
      AND p.event_id=NEW.event_id
  )
)
BEGIN SELECT RAISE(ABORT,'invalid run pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_response_pedagogy_pair_insert
BEFORE INSERT ON responses
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete response pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_response_pedagogy_pair_update
BEFORE UPDATE OF pedagogy_plan_id,pedagogy_plan_hash ON responses
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete response pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_action_pedagogy_pair_insert
BEFORE INSERT ON actions
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete action pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_action_pedagogy_pair_update
BEFORE UPDATE OF pedagogy_plan_id,pedagogy_plan_hash ON actions
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete action pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_experience_pedagogy_pair_insert
BEFORE INSERT ON experiences
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete experience pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_experience_pedagogy_pair_update
BEFORE UPDATE OF pedagogy_plan_id,pedagogy_plan_hash ON experiences
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete experience pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_artifact_pedagogy_pair_insert
BEFORE INSERT ON artifacts
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete artifact pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_artifact_pedagogy_pair_update
BEFORE UPDATE OF pedagogy_plan_id,pedagogy_plan_hash ON artifacts
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete artifact pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_visual_job_pedagogy_pair_insert
BEFORE INSERT ON visual_jobs
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete visual job pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_visual_job_pedagogy_pair_update
BEFORE UPDATE OF pedagogy_plan_id,pedagogy_plan_hash ON visual_jobs
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete visual job pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_visual_asset_pedagogy_pair_insert
BEFORE INSERT ON visual_assets
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete visual asset pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_visual_asset_pedagogy_pair_update
BEFORE UPDATE OF pedagogy_plan_id,pedagogy_plan_hash ON visual_assets
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete visual asset pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_feedback_pedagogy_pair_insert
BEFORE INSERT ON feedback
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete feedback pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_feedback_pedagogy_pair_update
BEFORE UPDATE OF pedagogy_plan_id,pedagogy_plan_hash ON feedback
WHEN (NEW.pedagogy_plan_id IS NULL) != (NEW.pedagogy_plan_hash IS NULL)
BEGIN SELECT RAISE(ABORT,'incomplete feedback pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_action_pedagogy_binding_insert
BEFORE INSERT ON actions
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.event_id=NEW.event_id
)
BEGIN SELECT RAISE(ABORT,'invalid action pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_action_pedagogy_binding_update
BEFORE UPDATE OF event_id,pedagogy_plan_id,pedagogy_plan_hash ON actions
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.event_id=NEW.event_id
)
BEGIN SELECT RAISE(ABORT,'invalid action pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_experience_pedagogy_binding_insert
BEFORE INSERT ON experiences
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.event_id=NEW.source_event_id AND p.child_id=NEW.child_id
)
BEGIN SELECT RAISE(ABORT,'invalid experience pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_experience_pedagogy_binding_update
BEFORE UPDATE OF source_event_id,child_id,pedagogy_plan_id,pedagogy_plan_hash ON experiences
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.event_id=NEW.source_event_id AND p.child_id=NEW.child_id
)
BEGIN SELECT RAISE(ABORT,'invalid experience pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_artifact_pedagogy_binding_insert
BEFORE INSERT ON artifacts
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p JOIN experiences e ON e.id=NEW.experience_id
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.child_id=NEW.child_id
    AND e.child_id=NEW.child_id
    AND e.pedagogy_plan_id=NEW.pedagogy_plan_id
    AND e.pedagogy_plan_hash=NEW.pedagogy_plan_hash
)
BEGIN SELECT RAISE(ABORT,'invalid artifact pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_artifact_pedagogy_binding_update
BEFORE UPDATE OF experience_id,child_id,pedagogy_plan_id,pedagogy_plan_hash ON artifacts
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p JOIN experiences e ON e.id=NEW.experience_id
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.child_id=NEW.child_id
    AND e.child_id=NEW.child_id
    AND e.pedagogy_plan_id=NEW.pedagogy_plan_id
    AND e.pedagogy_plan_hash=NEW.pedagogy_plan_hash
)
BEGIN SELECT RAISE(ABORT,'invalid artifact pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_visual_job_pedagogy_binding_insert
BEFORE INSERT ON visual_jobs
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.event_id=NEW.event_id
)
BEGIN SELECT RAISE(ABORT,'invalid visual job pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_visual_job_pedagogy_binding_update
BEFORE UPDATE OF event_id,pedagogy_plan_id,pedagogy_plan_hash ON visual_jobs
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.event_id=NEW.event_id
)
BEGIN SELECT RAISE(ABORT,'invalid visual job pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_visual_asset_pedagogy_binding_insert
BEFORE INSERT ON visual_assets
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM visual_jobs j
  WHERE j.id=NEW.job_id AND j.event_id=NEW.event_id
    AND j.pedagogy_plan_id=NEW.pedagogy_plan_id
    AND j.pedagogy_plan_hash=NEW.pedagogy_plan_hash
)
BEGIN SELECT RAISE(ABORT,'invalid visual asset pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_visual_asset_pedagogy_binding_update
BEFORE UPDATE OF job_id,event_id,pedagogy_plan_id,pedagogy_plan_hash ON visual_assets
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM visual_jobs j
  WHERE j.id=NEW.job_id AND j.event_id=NEW.event_id
    AND j.pedagogy_plan_id=NEW.pedagogy_plan_id
    AND j.pedagogy_plan_hash=NEW.pedagogy_plan_hash
)
BEGIN SELECT RAISE(ABORT,'invalid visual asset pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_feedback_pedagogy_binding_insert
BEFORE INSERT ON feedback
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.child_id=NEW.child_id
    AND (NEW.event_id IS NULL OR p.event_id=NEW.event_id)
    AND (NEW.experience_id IS NULL OR EXISTS (
      SELECT 1 FROM experiences e WHERE e.id=NEW.experience_id AND e.child_id=NEW.child_id
        AND e.pedagogy_plan_id=NEW.pedagogy_plan_id AND e.pedagogy_plan_hash=NEW.pedagogy_plan_hash
    ))
    AND (NEW.artifact_id IS NULL OR EXISTS (
      SELECT 1 FROM artifacts a WHERE a.id=NEW.artifact_id AND a.child_id=NEW.child_id
        AND a.pedagogy_plan_id=NEW.pedagogy_plan_id AND a.pedagogy_plan_hash=NEW.pedagogy_plan_hash
    ))
)
BEGIN SELECT RAISE(ABORT,'invalid feedback pedagogy binding'); END;

CREATE TRIGGER IF NOT EXISTS trg_feedback_pedagogy_binding_update
BEFORE UPDATE OF child_id,event_id,experience_id,artifact_id,pedagogy_plan_id,pedagogy_plan_hash ON feedback
WHEN NEW.pedagogy_plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.pedagogy_plan_id AND p.plan_hash=NEW.pedagogy_plan_hash
    AND p.child_id=NEW.child_id
    AND (NEW.event_id IS NULL OR p.event_id=NEW.event_id)
    AND (NEW.experience_id IS NULL OR EXISTS (
      SELECT 1 FROM experiences e WHERE e.id=NEW.experience_id AND e.child_id=NEW.child_id
        AND e.pedagogy_plan_id=NEW.pedagogy_plan_id AND e.pedagogy_plan_hash=NEW.pedagogy_plan_hash
    ))
    AND (NEW.artifact_id IS NULL OR EXISTS (
      SELECT 1 FROM artifacts a WHERE a.id=NEW.artifact_id AND a.child_id=NEW.child_id
        AND a.pedagogy_plan_id=NEW.pedagogy_plan_id AND a.pedagogy_plan_hash=NEW.pedagogy_plan_hash
    ))
)
BEGIN SELECT RAISE(ABORT,'invalid feedback pedagogy binding'); END;

INSERT INTO schema_migrations(version,name,applied_at)
VALUES(19,'approved pedagogy plan authority',strftime('%Y-%m-%dT%H:%M:%fZ','now'))
ON CONFLICT(version) DO NOTHING;
PRAGMA user_version = 19;
"""


MIGRATION_21_SQL = """
CREATE TABLE IF NOT EXISTS learner_model_controls (
  child_id TEXT PRIMARY KEY,
  collection_enabled INTEGER NOT NULL DEFAULT 1 CHECK(collection_enabled IN (0,1)),
  default_retention_class TEXT NOT NULL DEFAULT 'learning_record'
    CHECK(default_retention_class IN ('transient','learning_record','parent_confirmed')),
  updated_by TEXT NOT NULL CHECK(updated_by IN ('parent','operator')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS learner_retention_events (
  retention_event_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  opportunity_id TEXT NOT NULL,
  retain_until TEXT NOT NULL,
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator')),
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(retention_event_id,child_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(opportunity_id,child_id)
    REFERENCES learning_opportunities(opportunity_id,child_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS family_model_audit_log (
  audit_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  action TEXT NOT NULL CHECK(action IN (
    'collection_paused','collection_resumed','evidence_corrected','retention_changed',
    'source_deleted','event_deleted','model_exported','model_deleted','child_reflection_recorded'
  )),
  target_kind TEXT NOT NULL CHECK(target_kind IN (
    'child_model','evidence','learning_opportunity','event','concept','export'
  )),
  target_hash TEXT NOT NULL,
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator')),
  details_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS child_reflection_responses (
  response_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  response TEXT NOT NULL CHECK(response IN ('yes','not_yet','unsure','skip')),
  relayed_by TEXT NOT NULL CHECK(relayed_by IN ('parent','operator')),
  note TEXT,
  authority TEXT NOT NULL DEFAULT 'review_only' CHECK(authority='review_only'),
  created_at TEXT NOT NULL,
  UNIQUE(response_id,child_id),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(concept_id,concept_version) REFERENCES concepts(concept_id,version) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS deleted_event_tombstones (
  tombstone_id TEXT PRIMARY KEY,
  event_id_hash TEXT NOT NULL UNIQUE,
  original_child_id_hash TEXT,
  reason TEXT NOT NULL,
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator')),
  deleted_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retention_events_opportunity
  ON learner_retention_events(child_id,opportunity_id,created_at DESC,retention_event_id DESC);
CREATE INDEX IF NOT EXISTS idx_family_model_audit_child
  ON family_model_audit_log(child_id,created_at DESC,audit_id DESC);
CREATE INDEX IF NOT EXISTS idx_child_reflection_child
  ON child_reflection_responses(child_id,created_at DESC,response_id DESC);

INSERT INTO schema_migrations(version,name,applied_at)
VALUES(21,'open family learner model controls',strftime('%Y-%m-%dT%H:%M:%fZ','now'))
ON CONFLICT(version) DO NOTHING;
PRAGMA user_version = 21;
"""


LEGACY_COLUMNS: dict[str, dict[str, str]] = {
    "children": {"updated_at": "TEXT"},
    "observations": {"event_id": "TEXT"},
    "evidence": {"episode_id": "TEXT"},
    "feedback": {
        "event_id": "TEXT",
        "pedagogy_plan_id": "TEXT",
        "pedagogy_plan_hash": "TEXT",
    },
    "experiences": {
        "source_event_id": "TEXT",
        "pedagogy_plan_id": "TEXT",
        "pedagogy_plan_hash": "TEXT",
    },
    "artifacts": {
        "sha256": "TEXT",
        "validation_json": "TEXT NOT NULL DEFAULT '{}'",
        "approval_status": "TEXT NOT NULL DEFAULT 'unreviewed'",
        "pedagogy_plan_id": "TEXT",
        "pedagogy_plan_hash": "TEXT",
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
    "slack_file_outbox": {
        "active_attempt_id": "TEXT",
        "lease_token_hash": "TEXT",
    },
    "artifact_file_outbox": {
        "active_attempt_id": "TEXT",
        "lease_token_hash": "TEXT",
    },
    "runs": {
        "event_id": "TEXT",
        "policy_json": "TEXT",
        "result_json": "TEXT",
        "pedagogy_plan_id": "TEXT",
        "pedagogy_plan_hash": "TEXT",
    },
    "responses": {"pedagogy_plan_id": "TEXT", "pedagogy_plan_hash": "TEXT"},
    "actions": {"pedagogy_plan_id": "TEXT", "pedagogy_plan_hash": "TEXT"},
    "household_settings": {
        "resource_context_mode": "TEXT NOT NULL DEFAULT 'metadata_only' CHECK(resource_context_mode IN ('metadata_only', 'selected_excerpts'))",
        "visual_mode": "TEXT NOT NULL DEFAULT 'deterministic' CHECK(visual_mode IN ('off', 'deterministic', 'decorative'))",
        "weekly_checkins_enabled": "INTEGER NOT NULL DEFAULT 0 CHECK(weekly_checkins_enabled IN (0, 1))",
    },
    "learner_model_controls": {
        "recommendation_use_enabled": "INTEGER NOT NULL DEFAULT 0 CHECK(recommendation_use_enabled IN (0,1))",
        "recommendation_policy_id": "TEXT",
        "recommendation_policy_version": "INTEGER CHECK(recommendation_policy_version IS NULL OR recommendation_policy_version > 0)",
        "recommendation_policy_hash": "TEXT CHECK(recommendation_policy_hash IS NULL OR length(recommendation_policy_hash)=64)",
    },
    "onboarding_reviews": {
        "brain_config_hash": "TEXT NOT NULL DEFAULT 'legacy'",
        "response_hash": "TEXT NOT NULL DEFAULT 'legacy'",
        "workflow": "TEXT NOT NULL DEFAULT 'legacy'",
    },
    "output_evaluations": {
        "preferred_response_shape": (
            "TEXT CHECK(preferred_response_shape IN ('parent_chat','quick_answer','learning_thread'))"
        ),
        "preferred_visual_mix": ("TEXT CHECK(preferred_visual_mix IN ('none','imagination','activity_aid','both'))"),
    },
    "visual_jobs": {
        "strategy": "TEXT NOT NULL DEFAULT 'deterministic'",
        "pedagogy_plan_id": "TEXT",
        "pedagogy_plan_hash": "TEXT",
    },
    "visual_assets": {
        "strategy": "TEXT NOT NULL DEFAULT 'deterministic'",
        "pedagogy_plan_id": "TEXT",
        "pedagogy_plan_hash": "TEXT",
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


def authority_hash(*values: Any) -> str:
    """Hash an ordered SQLite/Python authority tuple with canonical JSON encoding."""

    # SQLite exposes bound booleans to user-defined functions as integers. Make
    # direct Python calls use that same representation so both trust boundaries
    # compute an identical digest.
    normalized = [int(value) if isinstance(value, bool) else value for value in values]
    return sha256(jdump(normalized).encode()).hexdigest()


def research_policy_event_id(
    policy_id: Any,
    policy_version: Any,
    action: Any,
    actor_type: Any,
    operator_id: Any,
    reason: Any,
    created_at: Any,
) -> str:
    """Return the canonical identifier for a research-policy audit event."""

    return "rtev_" + sha256(
        (
            f"{policy_id}|{int(policy_version)}|{action}|{actor_type}|"
            f"{operator_id or ''}|{reason}|{created_at}"
        ).encode()
    ).hexdigest()[:24]


def research_control_event_id(
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
    """Return the canonical identifier for a parent/operator research-control event."""

    binding = (
        policy_id,
        int(policy_version) if policy_version is not None else None,
        definition_hash,
        release_content_hash,
    )
    return "rtctl_" + sha256(
        (
            f"{child_id}|{int(enabled)}|{binding}|{int(family_authorization_epoch)}|"
            f"{actor_type}|{reason}|{created_at}"
        ).encode()
    ).hexdigest()[:24]


def research_authorization_snapshot_hash(*values: Any) -> str:
    """Hash the complete immutable authority captured at one policy activation."""

    return authority_hash(*values)


def research_authorization_snapshot_id(snapshot_hash: str) -> str:
    """Derive the public identifier from an authorization snapshot's content hash."""

    return "rtauth_" + sha256(snapshot_hash.encode()).hexdigest()[:24]


def research_collection_pause_event_id(child_id: Any, created_at: Any) -> str:
    """Return the canonical identifier for an automatic collection-pause revocation."""

    return "rtctl_" + sha256(f"{child_id}|{created_at}|collection_pause".encode()).hexdigest()[:24]


def _reset_stale_research_authority(
    conn: sqlite3.Connection, current_release_hash: str
) -> None:
    """Fail closed on release drift while retaining historical audit evidence."""

    policy = conn.execute(
        "SELECT * FROM research_track_policies WHERE policy_id=? AND version=?",
        (RESEARCH_POLICY_ID, RESEARCH_POLICY_VERSION),
    ).fetchone()
    if policy is None:
        return
    referenced_evaluation = (
        conn.execute(
            "SELECT release_content_hash FROM research_track_evaluations WHERE evaluation_id=?",
            (policy["evaluation_id"],),
        ).fetchone()
        if policy["evaluation_id"] is not None
        else None
    )
    stale = (
        policy["release_content_hash"] is not None
        and str(policy["release_content_hash"]) != current_release_hash
    ) or (
        referenced_evaluation is not None
        and str(referenced_evaluation["release_content_hash"]) != current_release_hash
    )
    if not stale:
        return

    now = utcnow()
    reset_reason = "Research release changed; current authorization was revoked for fresh evaluation."
    control_reason = "Research release changed; family opt-in was revoked."
    was_active = str(policy["status"]) == "active"
    control_event_ids: list[str] = []
    conn.create_function("research_write_authorized", 0, lambda: 1, deterministic=True)
    try:
        if was_active:
            controls = conn.execute(
                "SELECT child_id,family_authorization_epoch FROM research_track_controls "
                "WHERE enabled=1 AND policy_id=? AND policy_version=?",
                (policy["policy_id"], policy["version"]),
            ).fetchall()
            conn.execute(
                """UPDATE research_track_policies
                   SET status='disabled',authorization_epoch=authorization_epoch+1,updated_at=?
                   WHERE policy_id=? AND version=?""",
                (now, policy["policy_id"], policy["version"]),
            )
            for control in controls:
                epoch = int(control["family_authorization_epoch"]) + 1
                conn.execute(
                    """UPDATE research_track_controls
                       SET enabled=0,policy_id=NULL,policy_version=NULL,definition_hash=NULL,
                           release_content_hash=NULL,family_authorization_epoch=?,
                           updated_by='operator',reason=?,updated_at=?
                       WHERE child_id=?""",
                    (epoch, control_reason, now, control["child_id"]),
                )
                event_id = research_control_event_id(
                    control["child_id"],
                    False,
                    None,
                    None,
                    None,
                    None,
                    epoch,
                    "system",
                    control_reason,
                    now,
                )
                conn.execute(
                    """INSERT INTO research_track_control_events(
                         event_id,child_id,enabled,policy_id,policy_version,definition_hash,
                         release_content_hash,family_authorization_epoch,actor_type,reason,created_at
                       ) VALUES(?,?,0,NULL,NULL,NULL,NULL,?,'system',?,?)""",
                    (event_id, control["child_id"], epoch, control_reason, now),
                )
                control_event_ids.append(event_id)
        conn.execute(
            """UPDATE research_track_policies
               SET status='candidate',release_content_hash=NULL,evaluation_id=NULL,
                   promotion_review_id=NULL,approved_by_operator=NULL,approval_reason=NULL,
                   activation_reason=NULL,approved_at=NULL,activated_at=NULL,updated_at=?
               WHERE policy_id=? AND version=?""",
            (now, policy["policy_id"], policy["version"]),
        )
        event_id = research_policy_event_id(
            policy["policy_id"],
            policy["version"],
            "rolled_back",
            "system",
            None,
            reset_reason,
            now,
        )
        conn.execute(
            """INSERT INTO research_track_policy_events(
                 event_id,policy_id,policy_version,action,actor_type,operator_id,reason,created_at
               ) VALUES(?,?,?,?,?,NULL,?,?)""",
            (
                event_id,
                policy["policy_id"],
                policy["version"],
                "rolled_back",
                "system",
                reset_reason,
                now,
            ),
        )
        if was_active:
            database_file = str(conn.execute("PRAGMA database_list").fetchone()[2])
            record_research_authority(
                database_file,
                conn,
                policy_event_id=event_id,
                control_event_ids=tuple(control_event_ids),
            )
    finally:
        conn.create_function("research_write_authorized", 0, lambda: 0, deterministic=True)


def transport_event_hash(transport: str, external_event_id: str) -> str:
    """Return the content-free, binding-independent identity of one transport event."""

    return sha256(f"{transport}\0{external_event_id}".encode()).hexdigest()


def _canonical_json_text(value: Any) -> str:
    try:
        return jdump(json.loads(str(value)))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""


def _personalization_replay_hash(
    case_id: Any,
    source_kind: Any,
    license_status: Any,
    privacy_approval_id: Any,
    feature_schema_version: Any,
    feature_vector_json: Any,
    outcome_json: Any,
    source_digest: Any,
) -> str:
    try:
        payload = personalization_replay_case_payload(
            case_id=str(case_id),
            source_kind=str(source_kind),
            license_status=str(license_status),
            privacy_approval_id=(str(privacy_approval_id) if privacy_approval_id is not None else None),
            feature_schema_version=str(feature_schema_version),
            feature_vector=json.loads(str(feature_vector_json)),
            outcome=json.loads(str(outcome_json)),
            source_digest=str(source_digest),
        )
        return sha256(jdump(payload).encode()).hexdigest()
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""


def _research_calibration_case_hash(
    case_id: Any,
    source_kind: Any,
    license_status: Any,
    privacy_approval_id: Any,
    corpus_id: Any,
    split: Any,
    group_digest: Any,
    task_signature_digest: Any,
    domain: Any,
    history_json: Any,
    observed_success: Any,
    source_digest: Any,
) -> str:
    try:
        return calibration_case_hash(
            case_id=str(case_id),
            source_kind=str(source_kind),
            license_status=str(license_status),
            privacy_approval_id=(str(privacy_approval_id) if privacy_approval_id is not None else None),
            corpus_id=(str(corpus_id) if corpus_id is not None else None),
            split=str(split),
            group_digest=str(group_digest),
            task_signature_digest=str(task_signature_digest),
            domain=str(domain),
            history=json.loads(str(history_json)),
            observed_success=int(observed_success),
            source_digest=str(source_digest),
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""


def _research_posterior_json(history_json: Any) -> str:
    try:
        return jdump(posterior_from_history(json.loads(str(history_json))))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""


def _research_history_from_rows_json(rows_json: Any) -> str:
    try:
        rows = json.loads(str(rows_json))
        if not isinstance(rows, list):
            return ""
        return jdump(history_from_evidence_rows(rows))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""


def _open(path: Path, *, research_authorized: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.create_function(
        "sha256_hex",
        1,
        lambda value: sha256(str(value).encode()).hexdigest(),
        deterministic=True,
    )
    conn.create_function("canonical_json_text", 1, _canonical_json_text, deterministic=True)
    conn.create_function(
        "research_write_authorized", 0, lambda: int(research_authorized), deterministic=True
    )
    conn.create_function("authority_hash", -1, authority_hash, deterministic=True)
    conn.create_function(
        "personalization_replay_hash",
        8,
        _personalization_replay_hash,
        deterministic=True,
    )
    conn.create_function(
        "research_calibration_case_hash",
        12,
        _research_calibration_case_hash,
        deterministic=True,
    )
    conn.create_function("research_posterior_json", 1, _research_posterior_json, deterministic=True)
    conn.create_function(
        "research_history_from_rows_json",
        1,
        _research_history_from_rows_json,
        deterministic=True,
    )
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


@contextmanager
def research_connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open the private service-only capability used by schema-v26 write guards."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    roots = authority_backup_roots(path)
    with managed_data_lifecycle(path, extra_backup_roots=roots):
        # A peer may have staged a revocation since this service object was
        # constructed. Reconcile it before exposing an authorized connection,
        # so the first post-revocation call cannot observe stale consent.
        finalize_research_authority(path, required=True)
        conn = _open(path, research_authorized=True)
        try:
            yield conn
        except BaseException as exc:
            conn.rollback()
            conn.close()
            try:
                finalize_research_authority(path)
            except BaseException as authority_error:
                exc.add_note(
                    "research-authority cleanup also failed: "
                    f"{type(authority_error).__name__}: {authority_error}"
                )
            raise
        else:
            conn.commit()
            conn.close()
            finalize_research_authority(path)
        finally:
            conn.close()


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _ensure_family_database_identity(conn: sqlite3.Connection) -> str:
    """Create once and validate the stable identity copied into every managed image."""

    conn.execute(
        """CREATE TABLE IF NOT EXISTS family_database_identity (
          singleton INTEGER PRIMARY KEY CHECK(singleton=1),
          database_id TEXT NOT NULL UNIQUE CHECK(length(database_id) BETWEEN 16 AND 128),
          created_at TEXT NOT NULL
        )"""
    )
    rows = conn.execute(
        "SELECT singleton,database_id FROM family_database_identity ORDER BY singleton"
    ).fetchall()
    if not rows:
        database_id = f"db_{secrets.token_hex(24)}"
        conn.execute(
            "INSERT INTO family_database_identity(singleton,database_id,created_at) VALUES(1,?,?)",
            (database_id, utcnow()),
        )
        return database_id
    if len(rows) != 1 or int(rows[0][0]) != 1:
        raise RuntimeError("family database identity is invalid")
    database_id = str(rows[0][1])
    if not 16 <= len(database_id) <= 128:
        raise RuntimeError("family database identity is invalid")
    return database_id


def _backup_existing(path: Path, version: int) -> Path | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    source = sqlite3.connect(path)
    try:
        if not _tables(source):
            return None
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_path = migration_backup_path(path, version, stamp)
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


def _apply_migration_18(conn: sqlite3.Connection) -> None:
    """Apply the evidence-spine migration atomically and verify its ledger."""

    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    restore_version = f"PRAGMA user_version = {current};" if current > 18 else ""
    try:
        conn.executescript(f"BEGIN IMMEDIATE;\n{MIGRATION_18_SQL}\n{restore_version}\nCOMMIT;")
    except BaseException:
        conn.rollback()
        raise
    required = {
        "schema_migrations",
        "concepts",
        "concept_aliases",
        "concept_relations",
        "rubrics",
        "learning_tasks",
        "learning_task_concepts",
        "learning_episode_plans",
        "evidence_annotations",
        "learning_opportunities",
        "opportunity_assistance_events",
        "evidence_records",
        "evidence_corrections",
        "projection_runs",
        "learner_state_projections",
        "learner_projection_evidence",
        "interest_projections",
        "interest_projection_evidence",
        "derived_claims_v2",
        "derived_claim_evidence_v2",
        "preference_support_records",
        "evidence_source_tombstones",
    }
    missing = required - _tables(conn)
    required_indexes = {
        "idx_events_id_child",
        "idx_episodes_id_child",
        "idx_runs_id_event",
        "idx_task_concepts_concept",
        "idx_opportunities_child_time",
        "idx_opportunities_episode",
        "idx_evidence_child_concept",
        "idx_evidence_opportunity",
        "idx_corrections_evidence",
        "idx_tombstone_target",
        "idx_opportunity_attempt",
    }
    indexes = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    missing_indexes = required_indexes - indexes
    opportunity_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(learning_opportunities)")}
    missing_columns = {"plan_id", "rubric_id", "criterion_results_json"} - opportunity_columns
    ledger = conn.execute("SELECT name FROM schema_migrations WHERE version=18").fetchone()
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    foreign_key_issues = conn.execute("PRAGMA foreign_key_check").fetchall()
    if (
        missing
        or missing_indexes
        or missing_columns
        or not ledger
        or ledger["name"] != "versioned evidence spine"
        or version < 18
        or foreign_key_issues
    ):
        raise RuntimeError(
            f"incomplete schema v18 migration: missing={sorted(missing)!r}, "
            f"missing_indexes={sorted(missing_indexes)!r}, "
            f"missing_columns={sorted(missing_columns)!r}, ledger={bool(ledger)}, "
            f"user_version={version}, foreign_key_issues={len(foreign_key_issues)}"
        )


def _apply_migration_19(conn: sqlite3.Connection) -> None:
    """Apply and verify immutable pedagogy-plan authority and lineage."""

    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    restore_version = f"PRAGMA user_version = {current};" if current > 19 else ""
    try:
        conn.executescript(f"BEGIN IMMEDIATE;\n{MIGRATION_19_SQL}\n{restore_version}\nCOMMIT;")
    except BaseException:
        conn.rollback()
        raise
    required_columns = {
        "runs": {"pedagogy_plan_id", "pedagogy_plan_hash"},
        "responses": {"pedagogy_plan_id", "pedagogy_plan_hash"},
        "actions": {"pedagogy_plan_id", "pedagogy_plan_hash"},
        "experiences": {"pedagogy_plan_id", "pedagogy_plan_hash"},
        "artifacts": {"pedagogy_plan_id", "pedagogy_plan_hash"},
        "visual_jobs": {"pedagogy_plan_id", "pedagogy_plan_hash"},
        "visual_assets": {"pedagogy_plan_id", "pedagogy_plan_hash"},
        "feedback": {"pedagogy_plan_id", "pedagogy_plan_hash"},
    }
    missing_columns = {
        table: sorted(columns - {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")})
        for table, columns in required_columns.items()
    }
    missing_columns = {table: columns for table, columns in missing_columns.items() if columns}
    required_triggers = {
        "trg_pedagogy_plan_authority_insert",
        "trg_pedagogy_plan_authority_update",
        "trg_pedagogy_plan_supersession_insert",
        "trg_pedagogy_plan_supersession_update",
        "trg_pedagogy_plan_immutable",
        "trg_plan_bound_event_semantics_immutable",
    }
    present_triggers = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    missing_triggers = required_triggers - present_triggers
    ledger = conn.execute("SELECT name FROM schema_migrations WHERE version=19").fetchone()
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    foreign_key_issues = conn.execute("PRAGMA foreign_key_check").fetchall()
    if (
        "pedagogy_plans" not in _tables(conn)
        or missing_columns
        or missing_triggers
        or not ledger
        or ledger["name"] != "approved pedagogy plan authority"
        or version != max(current, 19)
        or foreign_key_issues
    ):
        raise RuntimeError(
            "incomplete schema v19 migration: "
            f"table={('pedagogy_plans' in _tables(conn))}, missing_columns={missing_columns!r}, "
            f"missing_triggers={sorted(missing_triggers)!r}, "
            f"ledger={bool(ledger)}, user_version={version}, foreign_key_issues={len(foreign_key_issues)}"
        )


def _pedagogy_plan_fk_repair_needed(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='pedagogy_plans'").fetchone()
    normalized = " ".join(str(row["sql"] or "").casefold().split()) if row else ""
    return not all(
        fragment in normalized
        for fragment in (
            "foreign key(child_id) references children(id) on delete cascade",
            "foreign key(event_id) references events(id) on delete cascade",
            "foreign key(initial_run_id) references runs(id) on delete set null",
            "foreign key(supersedes_plan_id,child_id) references pedagogy_plans(plan_id,child_id) on delete cascade",
        )
    )


def _apply_migration_20(conn: sqlite3.Connection) -> None:
    """Repair Phase 2 ownership FKs so privacy deletion remains cascading and complete."""

    if _pedagogy_plan_fk_repair_needed(conn):
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            trigger_names = [
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' AND sql LIKE '%pedagogy_plans%'"
                )
            ]
            for name in trigger_names:
                conn.execute(f'DROP TRIGGER "{name.replace(chr(34), chr(34) * 2)}"')
            conn.execute("ALTER TABLE pedagogy_plans RENAME TO pedagogy_plans_v19")
            conn.execute(
                """CREATE TABLE pedagogy_plans (
                  plan_id TEXT PRIMARY KEY,
                  plan_hash TEXT NOT NULL UNIQUE,
                  event_id TEXT NOT NULL,
                  child_id TEXT NOT NULL,
                  initial_run_id INTEGER,
                  workflow TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('approved','generated','rejected','superseded')),
                  proposal_json TEXT NOT NULL,
                  approved_json TEXT NOT NULL,
                  registry_version TEXT NOT NULL,
                  registry_digest TEXT NOT NULL,
                  contract_version TEXT NOT NULL,
                  validator_version TEXT NOT NULL,
                  routing_provenance TEXT NOT NULL,
                  supersedes_plan_id TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  UNIQUE(event_id,workflow,contract_version),
                  UNIQUE(plan_id,child_id),
                  UNIQUE(plan_id,event_id),
                  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
                  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
                  FOREIGN KEY(initial_run_id) REFERENCES runs(id) ON DELETE SET NULL,
                  FOREIGN KEY(supersedes_plan_id,child_id)
                    REFERENCES pedagogy_plans(plan_id,child_id) ON DELETE CASCADE
                )"""
            )
            conn.execute(
                """INSERT INTO pedagogy_plans SELECT
                     plan_id,plan_hash,event_id,child_id,initial_run_id,workflow,status,proposal_json,approved_json,
                     registry_version,registry_digest,contract_version,validator_version,routing_provenance,
                     supersedes_plan_id,created_at,updated_at
                   FROM pedagogy_plans_v19"""
            )
            conn.execute("DROP TABLE pedagogy_plans_v19")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(
            f"""BEGIN IMMEDIATE;
{MIGRATION_19_SQL}
INSERT INTO schema_migrations(version,name,applied_at)
VALUES(20,'repair pedagogy ownership cascades',strftime('%Y-%m-%dT%H:%M:%fZ','now'))
ON CONFLICT(version) DO NOTHING;
PRAGMA user_version = 20;
COMMIT;"""
        )
    except BaseException:
        conn.rollback()
        raise
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    ledger = conn.execute("SELECT name FROM schema_migrations WHERE version=20").fetchone()
    if (
        version != 20
        or not ledger
        or ledger["name"] != "repair pedagogy ownership cascades"
        or _pedagogy_plan_fk_repair_needed(conn)
        or conn.execute("PRAGMA foreign_key_check").fetchall()
    ):
        raise RuntimeError("incomplete schema v20 pedagogy ownership repair")


def _apply_migration_21(conn: sqlite3.Connection) -> None:
    """Add the parent-visible learner-model control and audit plane."""

    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    restore_version = f"PRAGMA user_version = {current};" if current > 21 else ""
    try:
        conn.executescript(f"BEGIN IMMEDIATE;\n{MIGRATION_21_SQL}\n{restore_version}\nCOMMIT;")
    except BaseException:
        conn.rollback()
        raise
    required_tables = {
        "learner_model_controls",
        "learner_retention_events",
        "family_model_audit_log",
        "child_reflection_responses",
        "deleted_event_tombstones",
    }
    required_indexes = {
        "idx_retention_events_opportunity",
        "idx_family_model_audit_child",
        "idx_child_reflection_child",
    }
    present_indexes = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    ledger = conn.execute("SELECT name FROM schema_migrations WHERE version=21").fetchone()
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    missing_tables = required_tables - _tables(conn)
    missing_indexes = required_indexes - present_indexes
    if (
        missing_tables
        or missing_indexes
        or not ledger
        or ledger["name"] != "open family learner model controls"
        or version != max(current, 21)
        or conn.execute("PRAGMA foreign_key_check").fetchall()
    ):
        raise RuntimeError(
            "incomplete schema v21 family learner-model migration: "
            f"missing_tables={sorted(missing_tables)!r}, missing_indexes={sorted(missing_indexes)!r}, "
            f"ledger={bool(ledger)}, user_version={version}"
        )


def _event_tombstone_repair_needed(conn: sqlite3.Connection) -> bool:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(deleted_event_tombstones)")}
    if "child_id" in columns or "original_child_id_hash" not in columns:
        return True
    for index in conn.execute("PRAGMA index_list(deleted_event_tombstones)"):
        if not bool(index[2]):
            continue
        indexed = [str(row[2]) for row in conn.execute(f'PRAGMA index_info("{index[1]}")')]
        if indexed == ["event_id_hash"]:
            return False
    return True


def _apply_migration_22(conn: sqlite3.Connection) -> None:
    """Make raw-event replay tombstones global and independent of child deletion."""

    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    restore_version = f"PRAGMA user_version = {current};" if current > 22 else ""
    if _event_tombstone_repair_needed(conn):
        old_rows = [dict(row) for row in conn.execute("SELECT * FROM deleted_event_tombstones")]
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ALTER TABLE deleted_event_tombstones RENAME TO deleted_event_tombstones_v21")
            conn.execute(
                """CREATE TABLE deleted_event_tombstones (
                  tombstone_id TEXT PRIMARY KEY,
                  event_id_hash TEXT NOT NULL UNIQUE,
                  original_child_id_hash TEXT,
                  reason TEXT NOT NULL,
                  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator')),
                  deleted_at TEXT NOT NULL
                )"""
            )
            for row in old_rows:
                child_hash = (
                    sha256(str(row["child_id"]).encode()).hexdigest()
                    if row.get("child_id") is not None
                    else row.get("original_child_id_hash")
                )
                conn.execute(
                    """INSERT OR IGNORE INTO deleted_event_tombstones(
                       tombstone_id,event_id_hash,original_child_id_hash,reason,actor_type,deleted_at
                       ) VALUES(?,?,?,?,?,?)""",
                    (
                        row["tombstone_id"],
                        row["event_id_hash"],
                        child_hash,
                        row["reason"],
                        row["actor_type"],
                        row["deleted_at"],
                    ),
                )
            conn.execute("DROP TABLE deleted_event_tombstones_v21")
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(
            f"""BEGIN IMMEDIATE;
CREATE INDEX IF NOT EXISTS idx_deleted_event_child_hash
  ON deleted_event_tombstones(original_child_id_hash,deleted_at DESC,tombstone_id DESC);
INSERT INTO schema_migrations(version,name,applied_at)
VALUES(22,'global event replay tombstones',strftime('%Y-%m-%dT%H:%M:%fZ','now'))
ON CONFLICT(version) DO NOTHING;
PRAGMA user_version = 22;
{restore_version}
COMMIT;"""
        )
    except BaseException:
        conn.rollback()
        raise
    ledger = conn.execute("SELECT name FROM schema_migrations WHERE version=22").fetchone()
    if (
        int(conn.execute("PRAGMA user_version").fetchone()[0]) != max(current, 22)
        or not ledger
        or ledger["name"] != "global event replay tombstones"
        or _event_tombstone_repair_needed(conn)
        or conn.execute("PRAGMA foreign_key_check").fetchall()
    ):
        raise RuntimeError("incomplete schema v22 global event tombstone migration")


def _apply_migration_23(conn: sqlite3.Connection) -> None:
    """Add complete agent lineage, transport replay guards, and durable file-erasure recovery."""

    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    additions = {
        "agent_turns": ("event_id", "TEXT REFERENCES events(id) ON DELETE CASCADE"),
        "tool_calls": ("event_id", "TEXT REFERENCES events(id) ON DELETE CASCADE"),
        "capability_runs": ("produced_event_id", "TEXT REFERENCES events(id) ON DELETE CASCADE"),
        "release_units": ("event_id", "TEXT REFERENCES events(id) ON DELETE CASCADE"),
        "learning_opportunities": ("opportunity_id_hash", "TEXT"),
    }
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_family_database_identity(conn)
        # Repair runs must be able to replace a malformed legacy hash before
        # reinstating the immutable identity guards below.
        conn.execute("DROP TRIGGER IF EXISTS guard_learning_opportunity_id_hash_insert")
        conn.execute("DROP TRIGGER IF EXISTS guard_learning_opportunity_id_hash_update")
        for table, (column, definition) in additions.items():
            columns = {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
            if column not in columns:
                conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {definition}')
        for row in conn.execute(
            """SELECT opportunity_id FROM learning_opportunities
               WHERE opportunity_id_hash IS NULL
                  OR opportunity_id_hash != sha256_hex(opportunity_id)"""
        ).fetchall():
            opportunity_id = str(row[0])
            conn.execute(
                "UPDATE learning_opportunities SET opportunity_id_hash=? WHERE opportunity_id=?",
                (sha256(opportunity_id.encode()).hexdigest(), opportunity_id),
            )
        # Opportunity identifiers are globally unique. A deletion tombstone
        # must therefore remain global even if a stale replay changes child.
        duplicate_tombstones = conn.execute(
            """SELECT target_kind,target_id_hash,MIN(rowid) AS keep_rowid
               FROM evidence_source_tombstones
               GROUP BY target_kind,target_id_hash HAVING COUNT(*) > 1"""
        ).fetchall()
        for duplicate in duplicate_tombstones:
            conn.execute(
                """DELETE FROM evidence_source_tombstones
                   WHERE target_kind=? AND target_id_hash=? AND rowid != ?""",
                (duplicate[0], duplicate[1], duplicate[2]),
            )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_tombstone_target_global
               ON evidence_source_tombstones(target_kind,target_id_hash)"""
        )
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS guard_learning_opportunity_id_hash_insert
            BEFORE INSERT ON learning_opportunities
            WHEN NEW.opportunity_id_hash IS NULL OR length(NEW.opportunity_id_hash) != 64
              OR NEW.opportunity_id_hash != sha256_hex(NEW.opportunity_id)
              OR EXISTS(
                SELECT 1 FROM evidence_source_tombstones t
                WHERE t.target_kind='learning_opportunity'
                  AND t.target_id_hash=NEW.opportunity_id_hash
              )
            BEGIN SELECT RAISE(ABORT,'deleted learning opportunity cannot be replayed'); END"""
        )
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS guard_learning_opportunity_id_hash_update
            BEFORE UPDATE OF opportunity_id,opportunity_id_hash ON learning_opportunities
            WHEN NEW.opportunity_id != OLD.opportunity_id
              OR NEW.opportunity_id_hash != OLD.opportunity_id_hash
            BEGIN SELECT RAISE(ABORT,'learning opportunity identity is immutable'); END"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS deleted_transport_event_tombstones (
              tombstone_id TEXT PRIMARY KEY,
              transport_event_hash TEXT NOT NULL UNIQUE,
              original_child_id_hash TEXT,
              reason TEXT NOT NULL,
              actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator')),
              deleted_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS family_file_deletion_journal (
              journal_id TEXT PRIMARY KEY,
              deletion_id TEXT NOT NULL,
              event_id_hash TEXT NOT NULL,
              original_path TEXT NOT NULL,
              quarantine_path TEXT NOT NULL,
              output_root TEXT NOT NULL,
              content_sha256 TEXT NOT NULL,
              state TEXT NOT NULL CHECK(state IN ('planned','staged','db_committed')),
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(deletion_id,original_path)
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS family_event_deletion_locks (
              event_id TEXT PRIMARY KEY,
              child_id TEXT NOT NULL,
              deletion_id TEXT NOT NULL UNIQUE,
              created_at TEXT NOT NULL,
              FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
              FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
            )"""
        )
        lineage_tables = {
            "agent_message_event_lineage": ("message_id", "agent_messages"),
            "agent_turn_event_lineage": ("turn_id", "agent_turns"),
            "tool_call_event_lineage": ("tool_call_id", "tool_calls"),
            "capability_run_event_lineage": ("capability_run_id", "capability_runs"),
            "release_unit_event_lineage": ("release_unit_id", "release_units"),
            "interaction_instance_event_lineage": ("interaction_id", "interaction_instances"),
        }
        for table, (record_column, record_table) in lineage_tables.items():
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {table} (
                  {record_column} TEXT NOT NULL,
                  event_id TEXT NOT NULL,
                  PRIMARY KEY({record_column},event_id),
                  FOREIGN KEY({record_column}) REFERENCES {record_table}(id) ON DELETE CASCADE,
                  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE
                )"""
            )
        conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_deleted_transport_event_hash
               ON deleted_transport_event_tombstones(transport_event_hash)"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_deleted_transport_child_hash
               ON deleted_transport_event_tombstones(
                 original_child_id_hash,deleted_at DESC,tombstone_id DESC
               )"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_file_deletion_journal_operation
               ON family_file_deletion_journal(deletion_id,state,journal_id)"""
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_family_event_deletion_child
               ON family_event_deletion_locks(child_id,created_at)"""
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_turns_event ON agent_turns(event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_event ON tool_calls(event_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_capability_runs_produced_event ON capability_runs(produced_event_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_release_units_event ON release_units(event_id)")
        for table in lineage_tables:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_event ON {table}(event_id)")

        # Malformed legacy direct links are detached rather than allowing one
        # child's session records to inherit another child's event.
        direct_session_links = {
            "agent_messages": ("event_id", "session_id"),
            "agent_turns": ("event_id", "session_id"),
            "tool_calls": ("event_id", "session_id"),
            "capability_runs": ("produced_event_id", "session_id"),
        }
        for table, (event_column, session_column) in direct_session_links.items():
            conn.execute(
                f"""UPDATE {table} SET {event_column}=NULL
                    WHERE {event_column} IS NOT NULL AND NOT EXISTS(
                      SELECT 1 FROM agent_sessions s JOIN events e ON e.id={table}.{event_column}
                      WHERE s.id={table}.{session_column} AND s.child_id=e.child_id
                    )"""
            )
        conn.execute(
            """UPDATE capability_runs SET source_event_id=NULL
               WHERE source_event_id IS NOT NULL AND session_id IS NOT NULL AND NOT EXISTS(
                 SELECT 1 FROM agent_sessions s JOIN events e ON e.id=capability_runs.source_event_id
                 WHERE s.id=capability_runs.session_id AND s.child_id=e.child_id
               )"""
        )
        conn.execute(
            """UPDATE release_units SET event_id=NULL
               WHERE event_id IS NOT NULL AND NOT EXISTS(
                 SELECT 1 FROM capability_runs cr JOIN agent_sessions s ON s.id=cr.session_id
                 JOIN events e ON e.id=release_units.event_id
                 WHERE cr.id=release_units.capability_run_id AND s.child_id=e.child_id
               )"""
        )

        # Backfill only authoritative event IDs already emitted in stored JSON
        # and owned by the same child-bound session.
        # Rows with no provable relationship remain NULL rather than guessing by
        # timestamp or session proximity.
        conn.execute(
            """UPDATE agent_turns
               SET event_id=json_extract(response_json,'$.event_id')
               WHERE event_id IS NULL AND json_valid(response_json)
                 AND json_type(response_json,'$.event_id')='text'
                 AND EXISTS(
                   SELECT 1 FROM events e JOIN agent_sessions s ON s.id=agent_turns.session_id
                   WHERE e.id=json_extract(agent_turns.response_json,'$.event_id')
                     AND e.child_id=s.child_id
                 )"""
        )
        conn.execute(
            """UPDATE tool_calls
               SET event_id=json_extract(result_json,'$.event_id')
               WHERE event_id IS NULL AND json_valid(result_json)
                 AND json_type(result_json,'$.event_id')='text'
                 AND EXISTS(
                   SELECT 1 FROM events e JOIN agent_sessions s ON s.id=tool_calls.session_id
                   WHERE e.id=json_extract(tool_calls.result_json,'$.event_id')
                     AND e.child_id=s.child_id
                 )"""
        )
        conn.execute(
            """UPDATE capability_runs
               SET produced_event_id=json_extract(result_json,'$.event_id')
               WHERE produced_event_id IS NULL AND json_valid(result_json)
                 AND json_type(result_json,'$.event_id')='text'
                 AND EXISTS(
                   SELECT 1 FROM events e JOIN agent_sessions s ON s.id=capability_runs.session_id
                   WHERE e.id=json_extract(capability_runs.result_json,'$.event_id')
                     AND e.child_id=s.child_id
                 )"""
        )
        conn.execute(
            """UPDATE capability_runs
               SET produced_event_id=(
                 SELECT m.event_id FROM agent_messages m
                 WHERE m.session_id=capability_runs.session_id
                   AND m.event_id IS NOT NULL
                   AND json_valid(m.metadata_json)
                   AND json_extract(m.metadata_json,'$.capability_run_id')=capability_runs.id
                   AND EXISTS(
                     SELECT 1 FROM events e JOIN agent_sessions s ON s.id=capability_runs.session_id
                     WHERE e.id=m.event_id AND e.child_id=s.child_id
                   )
                 ORDER BY m.ordinal DESC LIMIT 1
               )
               WHERE produced_event_id IS NULL AND EXISTS(
                 SELECT 1 FROM agent_messages m
                 WHERE m.session_id=capability_runs.session_id
                   AND m.event_id IS NOT NULL
                   AND json_valid(m.metadata_json)
                   AND json_extract(m.metadata_json,'$.capability_run_id')=capability_runs.id
                   AND EXISTS(
                     SELECT 1 FROM events e JOIN agent_sessions s ON s.id=capability_runs.session_id
                     WHERE e.id=m.event_id AND e.child_id=s.child_id
                   )
               )"""
        )
        conn.execute(
            """UPDATE release_units
               SET event_id=COALESCE(
                 CASE
                   WHEN json_valid(payload_json) AND json_type(payload_json,'$.event_id')='text'
                   THEN json_extract(payload_json,'$.event_id')
                 END,
                 (SELECT cr.produced_event_id FROM capability_runs cr
                  WHERE cr.id=release_units.capability_run_id)
               )
               WHERE event_id IS NULL AND EXISTS(
                 SELECT 1 FROM events e
                 JOIN capability_runs cr ON cr.id=release_units.capability_run_id
                 JOIN agent_sessions s ON s.id=cr.session_id
                 WHERE e.id=COALESCE(
                   CASE
                     WHEN json_valid(release_units.payload_json)
                       AND json_type(release_units.payload_json,'$.event_id')='text'
                     THEN json_extract(release_units.payload_json,'$.event_id')
                   END,
                   (SELECT cr.produced_event_id FROM capability_runs cr
                    WHERE cr.id=release_units.capability_run_id)
                 ) AND e.child_id=s.child_id
               )"""
        )
        conn.execute(
            """INSERT OR IGNORE INTO agent_message_event_lineage(message_id,event_id)
               SELECT id,event_id FROM agent_messages WHERE event_id IS NOT NULL"""
        )
        conn.execute(
            """INSERT OR IGNORE INTO agent_turn_event_lineage(turn_id,event_id)
               SELECT id,event_id FROM agent_turns WHERE event_id IS NOT NULL"""
        )
        conn.execute(
            """INSERT OR IGNORE INTO tool_call_event_lineage(tool_call_id,event_id)
               SELECT id,event_id FROM tool_calls WHERE event_id IS NOT NULL"""
        )
        conn.execute(
            """INSERT OR IGNORE INTO capability_run_event_lineage(capability_run_id,event_id)
               SELECT id,produced_event_id FROM capability_runs WHERE produced_event_id IS NOT NULL"""
        )
        conn.execute(
            """INSERT OR IGNORE INTO release_unit_event_lineage(release_unit_id,event_id)
               SELECT id,event_id FROM release_units WHERE event_id IS NOT NULL"""
        )
        # Interaction payloads predate relational lineage. Recover only explicit
        # event-id fields whose event belongs to the interaction session's child.
        for row in conn.execute(
            """SELECT i.id,i.plan_json,i.options_json,s.child_id
               FROM interaction_instances i LEFT JOIN agent_sessions s ON s.id=i.session_id"""
        ).fetchall():
            payloads = (jload(row["plan_json"], {}), jload(row["options_json"], []))
            candidates: set[str] = set()

            def collect_event_ids(
                value: Any,
                key: str | None = None,
                target: set[str] = candidates,
            ) -> None:
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        collect_event_ids(nested_value, str(nested_key))
                elif isinstance(value, list):
                    for nested_value in value:
                        collect_event_ids(nested_value, key)
                elif key and key.endswith("event_id") and isinstance(value, str):
                    target.add(value)

            for payload in payloads:
                collect_event_ids(payload)
            for event_id in candidates:
                owned = conn.execute(
                    "SELECT 1 FROM events WHERE id=? AND child_id=?",
                    (event_id, row["child_id"]),
                ).fetchone()
                if owned:
                    conn.execute(
                        """INSERT OR IGNORE INTO interaction_instance_event_lineage(interaction_id,event_id)
                           VALUES(?,?)""",
                        (row["id"], event_id),
                    )

        # Database triggers are the hard floor: application validation gives a
        # clearer error, while direct SQL and future callers cannot cross-bind
        # one child's event to another child's session.
        session_guard_specs = {
            "agent_messages": ("session_id", "event_id"),
            "agent_turns": ("session_id", "event_id"),
            "tool_calls": ("session_id", "event_id"),
            "capability_runs": ("session_id", "produced_event_id"),
        }
        for table, (session_column, event_column) in session_guard_specs.items():
            for operation in ("INSERT", "UPDATE"):
                trigger = f"guard_{table}_{event_column}_{operation.casefold()}"
                conn.execute(
                    f"""CREATE TRIGGER IF NOT EXISTS {trigger}
                    BEFORE {operation} ON {table}
                    WHEN NEW.{event_column} IS NOT NULL AND NOT EXISTS(
                      SELECT 1 FROM agent_sessions s JOIN events e ON e.id=NEW.{event_column}
                      WHERE s.id=NEW.{session_column} AND s.child_id=e.child_id
                    )
                    BEGIN SELECT RAISE(ABORT,'cross-child agent event lineage'); END"""
                )
        for operation in ("INSERT", "UPDATE"):
            conn.execute(
                f"""CREATE TRIGGER IF NOT EXISTS guard_capability_source_event_{operation.casefold()}
                BEFORE {operation} ON capability_runs
                WHEN NEW.source_event_id IS NOT NULL AND NEW.session_id IS NOT NULL AND NOT EXISTS(
                  SELECT 1 FROM agent_sessions s JOIN events e ON e.id=NEW.source_event_id
                  WHERE s.id=NEW.session_id AND s.child_id=e.child_id
                )
                BEGIN SELECT RAISE(ABORT,'cross-child capability source event'); END"""
            )
            conn.execute(
                f"""CREATE TRIGGER IF NOT EXISTS guard_release_unit_event_{operation.casefold()}
                BEFORE {operation} ON release_units
                WHEN NEW.event_id IS NOT NULL AND NOT EXISTS(
                  SELECT 1 FROM capability_runs cr JOIN agent_sessions s ON s.id=cr.session_id
                  JOIN events e ON e.id=NEW.event_id
                  WHERE cr.id=NEW.capability_run_id AND s.child_id=e.child_id
                )
                BEGIN SELECT RAISE(ABORT,'cross-child release event lineage'); END"""
            )
        lineage_session_joins = {
            "agent_message_event_lineage": "agent_messages r JOIN agent_sessions s ON s.id=r.session_id AND r.id=NEW.message_id",
            "agent_turn_event_lineage": "agent_turns r JOIN agent_sessions s ON s.id=r.session_id AND r.id=NEW.turn_id",
            "tool_call_event_lineage": "tool_calls r JOIN agent_sessions s ON s.id=r.session_id AND r.id=NEW.tool_call_id",
            "capability_run_event_lineage": "capability_runs r JOIN agent_sessions s ON s.id=r.session_id AND r.id=NEW.capability_run_id",
            "release_unit_event_lineage": (
                "release_units r JOIN capability_runs cr ON cr.id=r.capability_run_id "
                "JOIN agent_sessions s ON s.id=cr.session_id AND r.id=NEW.release_unit_id"
            ),
            "interaction_instance_event_lineage": (
                "interaction_instances r JOIN agent_sessions s ON s.id=r.session_id AND r.id=NEW.interaction_id"
            ),
        }
        for table, session_join in lineage_session_joins.items():
            for operation in ("INSERT", "UPDATE"):
                conn.execute(
                    f"""CREATE TRIGGER IF NOT EXISTS guard_{table}_{operation.casefold()}
                    BEFORE {operation} ON {table}
                    WHEN NOT EXISTS(
                      SELECT 1 FROM {session_join} JOIN events e ON e.id=NEW.event_id
                      WHERE s.child_id=e.child_id
                    )
                    BEGIN SELECT RAISE(ABORT,'cross-child agent event lineage'); END"""
                )

        # Guard both sides of lineage. Valid relationships must not become
        # cross-child later by mutating the owning session, event, or parent
        # record after the lineage row has already passed validation.
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS guard_agent_session_child_immutable
            BEFORE UPDATE OF child_id ON agent_sessions
            WHEN OLD.child_id IS NOT NULL AND NEW.child_id IS NOT OLD.child_id
            BEGIN SELECT RAISE(ABORT,'agent session child ownership is immutable'); END"""
        )
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS guard_event_child_immutable
            BEFORE UPDATE OF child_id ON events
            WHEN NEW.child_id IS NOT OLD.child_id
            BEGIN SELECT RAISE(ABORT,'event child ownership is immutable'); END"""
        )
        ownership_guard_specs = {
            "agent_messages": ("session_id",),
            "agent_turns": ("session_id",),
            "tool_calls": ("session_id", "turn_id"),
            "capability_runs": ("session_id",),
            "release_units": ("capability_run_id",),
            "interaction_instances": ("session_id",),
        }
        for table, columns in ownership_guard_specs.items():
            for column in columns:
                conn.execute(
                    f"""CREATE TRIGGER IF NOT EXISTS guard_{table}_{column}_immutable
                    BEFORE UPDATE OF {column} ON {table}
                    WHEN NEW.{column} IS NOT NULL AND NEW.{column} IS NOT OLD.{column}
                    BEGIN SELECT RAISE(ABORT,'agent lineage ownership is immutable'); END"""
                )
        for operation in ("INSERT", "UPDATE"):
            conn.execute(
                f"""CREATE TRIGGER IF NOT EXISTS guard_tool_call_turn_session_{operation.casefold()}
                BEFORE {operation} ON tool_calls
                WHEN NOT EXISTS(
                  SELECT 1 FROM agent_turns t
                  WHERE t.id=NEW.turn_id AND t.session_id=NEW.session_id
                )
                BEGIN SELECT RAISE(ABORT,'tool call turn belongs to another session'); END"""
            )

        # Paused collection is also a database hard floor. Corrections,
        # preferences, raw conversation, and operational output remain allowed.
        paused_direct_tables = (
            "nodes",
            "edges",
            "observations",
            "school_signals",
            "claims",
            "episodes",
            "evidence",
            "opportunities",
            "learning_episode_plans",
            "evidence_annotations",
            "learning_opportunities",
            "evidence_records",
        )
        for table in paused_direct_tables:
            conn.execute(
                f"""CREATE TRIGGER IF NOT EXISTS block_paused_profile_{table}
                BEFORE INSERT ON {table}
                WHEN EXISTS(
                  SELECT 1 FROM learner_model_controls c
                  WHERE c.child_id=NEW.child_id AND c.collection_enabled=0
                )
                BEGIN SELECT RAISE(ABORT,'learner-model collection is paused'); END"""
            )
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS block_paused_profile_episode_memberships
            BEFORE INSERT ON episode_memberships
            WHEN EXISTS(
              SELECT 1 FROM episodes e JOIN learner_model_controls c ON c.child_id=e.child_id
              WHERE e.id=NEW.episode_id AND c.collection_enabled=0
            )
            BEGIN SELECT RAISE(ABORT,'learner-model collection is paused'); END"""
        )

        # Once event deletion is reserved, no new durable output may acquire
        # lineage to it. Writers that render before persistence must clean up a
        # rejected file; the final deletion transaction rechecks the closure.
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS block_deleting_event_visual_jobs_insert
            BEFORE INSERT ON visual_jobs
            WHEN EXISTS(SELECT 1 FROM family_event_deletion_locks l WHERE l.event_id=NEW.event_id)
            BEGIN SELECT RAISE(ABORT,'event deletion is pending'); END"""
        )
        for operation in ("INSERT", "UPDATE"):
            conn.execute(
                f"""CREATE TRIGGER IF NOT EXISTS block_deleting_event_visual_assets_{operation.casefold()}
                BEFORE {operation} ON visual_assets
                WHEN EXISTS(SELECT 1 FROM family_event_deletion_locks l WHERE l.event_id=NEW.event_id)
                BEGIN SELECT RAISE(ABORT,'event deletion is pending'); END"""
            )
        for operation in ("INSERT", "UPDATE"):
            conn.execute(
                f"""CREATE TRIGGER IF NOT EXISTS block_deleting_event_artifacts_{operation.casefold()}
                BEFORE {operation} ON artifacts
                WHEN EXISTS(
                  SELECT 1 FROM family_event_deletion_locks l
                  WHERE EXISTS(
                    SELECT 1
                    FROM experiences x LEFT JOIN pedagogy_plans p ON p.plan_id=NEW.pedagogy_plan_id
                    WHERE x.id=NEW.experience_id
                      AND (l.event_id=x.source_event_id OR l.event_id=p.event_id)
                  )
                )
                BEGIN SELECT RAISE(ABORT,'event deletion is pending'); END"""
            )
        conn.execute("DROP TRIGGER IF EXISTS block_deleting_event_artifact_outbox_update")
        conn.execute(
            """CREATE TRIGGER IF NOT EXISTS block_deleting_event_artifact_outbox_insert
            BEFORE INSERT ON artifact_file_outbox
            WHEN EXISTS(
              SELECT 1 FROM artifacts a LEFT JOIN experiences x ON x.id=a.experience_id
              LEFT JOIN pedagogy_plans p ON p.plan_id=a.pedagogy_plan_id
              JOIN family_event_deletion_locks l
                ON l.event_id=x.source_event_id OR l.event_id=p.event_id
              WHERE a.id=NEW.artifact_id
            )
            BEGIN SELECT RAISE(ABORT,'event deletion is pending'); END"""
        )
        conn.execute(
            """INSERT INTO schema_migrations(version,name,applied_at)
               VALUES(23,'complete deletion lineage and recovery',?)
               ON CONFLICT(version) DO NOTHING""",
            (utcnow(),),
        )
        conn.execute(f"PRAGMA user_version = {max(current, 23)}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    required_tables = {
        "family_database_identity",
        "deleted_transport_event_tombstones",
        "family_file_deletion_journal",
        "family_event_deletion_locks",
        *lineage_tables,
    }
    required_indexes = {
        "idx_deleted_transport_event_hash",
        "idx_deleted_transport_child_hash",
        "idx_file_deletion_journal_operation",
        "idx_family_event_deletion_child",
        "idx_agent_turns_event",
        "idx_tool_calls_event",
        "idx_capability_runs_produced_event",
        "idx_release_units_event",
        "idx_tombstone_target_global",
        *(f"idx_{table}_event" for table in lineage_tables),
    }
    required_triggers = {
        "guard_agent_session_child_immutable",
        "guard_event_child_immutable",
        "guard_learning_opportunity_id_hash_insert",
        "guard_learning_opportunity_id_hash_update",
        "guard_tool_call_turn_session_insert",
        "guard_tool_call_turn_session_update",
        *(
            f"guard_{table}_{column}_immutable"
            for table, columns in ownership_guard_specs.items()
            for column in columns
        ),
    }
    columns_ok = all(
        column in {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
        for table, (column, _definition) in additions.items()
    )
    present_indexes = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    present_triggers = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    }
    ledger = conn.execute("SELECT name FROM schema_migrations WHERE version=23").fetchone()
    identity = conn.execute(
        "SELECT singleton,database_id FROM family_database_identity"
    ).fetchall()
    if (
        required_tables - _tables(conn)
        or required_indexes - present_indexes
        or required_triggers - present_triggers
        or not columns_ok
        or not ledger
        or ledger["name"] != "complete deletion lineage and recovery"
        or len(identity) != 1
        or int(identity[0]["singleton"]) != 1
        or not 16 <= len(str(identity[0]["database_id"])) <= 128
        or conn.execute(
            """SELECT 1 FROM learning_opportunities
               WHERE opportunity_id_hash IS NULL
                  OR opportunity_id_hash != sha256_hex(opportunity_id) LIMIT 1"""
        ).fetchone()
        or int(conn.execute("PRAGMA user_version").fetchone()[0]) != max(current, 23)
        or conn.execute("PRAGMA foreign_key_check").fetchall()
    ):
        raise RuntimeError("incomplete schema v23 deletion-lineage migration")


def _apply_migration_24(conn: sqlite3.Connection) -> None:
    """Add the governed adaptation control plane without learner-evidence edges."""

    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    try:
        conn.executescript(
            """BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS parent_memory_settings (
  scope_type TEXT NOT NULL CHECK(scope_type IN ('household','thread','child_support')),
  scope_id TEXT NOT NULL,
  enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
  updated_by_parent_id TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(scope_type,scope_id),
  FOREIGN KEY(updated_by_parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS parent_memories (
  memory_key TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version > 0),
  scope_type TEXT NOT NULL CHECK(scope_type IN ('household','thread','child_support')),
  scope_id TEXT NOT NULL,
  child_id TEXT,
  category TEXT NOT NULL CHECK(category IN (
    'answer_style','visual_style','artifact_style','activity_style','interaction_style',
    'accessibility_support','delivery_preference'
  )),
  value TEXT NOT NULL CHECK(length(value) <= 1000),
  provenance TEXT NOT NULL CHECK(length(provenance) BETWEEN 1 AND 500),
  creator_type TEXT NOT NULL CHECK(creator_type IN ('parent','agent_proposal')),
  parent_id TEXT,
  source_message_id TEXT,
  sensitivity TEXT NOT NULL CHECK(sensitivity IN ('ordinary','sensitive')),
  status TEXT NOT NULL CHECK(status IN ('proposed','active','superseded','disabled','forgotten')),
  review_at TEXT,
  expires_at TEXT,
  why_saved TEXT NOT NULL CHECK(length(why_saved) BETWEEN 1 AND 500),
  content_hash TEXT NOT NULL CHECK(length(content_hash)=64),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(memory_key,version),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL,
  FOREIGN KEY(source_message_id) REFERENCES agent_messages(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_parent_memory_one_current
  ON parent_memories(memory_key) WHERE status IN ('proposed','active','disabled');
CREATE INDEX IF NOT EXISTS idx_parent_memory_recall
  ON parent_memories(scope_type,scope_id,status,category,updated_at DESC);

CREATE TABLE IF NOT EXISTS parent_memory_events (
  event_id TEXT PRIMARY KEY,
  memory_key TEXT NOT NULL,
  action TEXT NOT NULL CHECK(action IN (
    'remember','propose','approve','update','disable','enable','forget','undo'
  )),
  prior_version INTEGER,
  new_version INTEGER,
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator','system')),
  actor_parent_id TEXT,
  reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 500),
  prior_content_hash TEXT,
  new_content_hash TEXT,
  undone_by_event_id TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(actor_parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL,
  FOREIGN KEY(undone_by_event_id) REFERENCES parent_memory_events(event_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_parent_memory_events_key
  ON parent_memory_events(memory_key,created_at DESC,event_id DESC);

CREATE TABLE IF NOT EXISTS parent_memory_tombstones (
  tombstone_id TEXT PRIMARY KEY,
  memory_record_hash TEXT NOT NULL UNIQUE CHECK(length(memory_record_hash)=64),
  memory_key_hash TEXT NOT NULL CHECK(length(memory_key_hash)=64),
  version INTEGER NOT NULL CHECK(version > 0),
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator')),
  deleted_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS delivery_outbox_parent_memories (
  delivery_id TEXT NOT NULL,
  memory_key TEXT NOT NULL,
  memory_version INTEGER NOT NULL CHECK(memory_version > 0),
  content_hash TEXT NOT NULL CHECK(length(content_hash)=64),
  created_at TEXT NOT NULL,
  PRIMARY KEY(delivery_id,memory_key,memory_version),
  FOREIGN KEY(delivery_id) REFERENCES delivery_outbox(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_delivery_outbox_parent_memory
  ON delivery_outbox_parent_memories(memory_key,memory_version);

CREATE TABLE IF NOT EXISTS family_recipes (
  recipe_id TEXT PRIMARY KEY,
  capability_id TEXT NOT NULL,
  title TEXT NOT NULL CHECK(length(title) BETWEEN 1 AND 160),
  description TEXT NOT NULL CHECK(length(description) BETWEEN 1 AND 500),
  created_by_parent_id TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(created_by_parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS family_recipe_versions (
  recipe_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version > 0),
  instructions TEXT NOT NULL CHECK(length(instructions) BETWEEN 1 AND 12000),
  instructions_hash TEXT NOT NULL CHECK(length(instructions_hash)=64),
  tool_names_json TEXT NOT NULL DEFAULT '[]',
  status TEXT NOT NULL CHECK(status IN (
    'draft','linted','tested','shadow','approved','active','disabled','retired'
  )),
  test_report_json TEXT NOT NULL DEFAULT '{}',
  shadow_report_json TEXT NOT NULL DEFAULT '{}',
  test_evaluation_id TEXT,
  shadow_evaluation_id TEXT,
  approved_by_parent_id TEXT,
  approved_at TEXT,
  activated_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(recipe_id,version),
  FOREIGN KEY(recipe_id) REFERENCES family_recipes(recipe_id) ON DELETE CASCADE,
  FOREIGN KEY(approved_by_parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_family_recipe_one_active
  ON family_recipe_versions(recipe_id) WHERE status='active';

CREATE TABLE IF NOT EXISTS family_recipe_evaluations (
  evaluation_id TEXT PRIMARY KEY,
  recipe_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  instructions_hash TEXT NOT NULL CHECK(length(instructions_hash)=64),
  evaluation_kind TEXT NOT NULL CHECK(evaluation_kind IN ('tests','shadow')),
  suite_version TEXT NOT NULL,
  fixture_digest TEXT NOT NULL CHECK(length(fixture_digest)=64),
  case_count INTEGER NOT NULL CHECK(case_count > 0),
  passed_count INTEGER NOT NULL CHECK(passed_count BETWEEN 0 AND case_count),
  status TEXT NOT NULL CHECK(status IN ('pass','fail')),
  evaluator TEXT NOT NULL,
  results_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(recipe_id,version)
    REFERENCES family_recipe_versions(recipe_id,version) ON DELETE CASCADE
);

CREATE TRIGGER IF NOT EXISTS guard_family_recipe_identity_immutable
BEFORE UPDATE OF capability_id ON family_recipes
BEGIN SELECT RAISE(ABORT,'family recipe capability is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_family_recipe_definition_immutable
BEFORE UPDATE OF recipe_id,version,instructions,instructions_hash,tool_names_json
ON family_recipe_versions
BEGIN SELECT RAISE(ABORT,'family recipe version definition is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_family_recipe_evaluation_immutable
BEFORE UPDATE ON family_recipe_evaluations
BEGIN SELECT RAISE(ABORT,'family recipe evaluation is immutable'); END;

CREATE TABLE IF NOT EXISTS family_recipe_events (
  event_id TEXT PRIMARY KEY,
  recipe_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  action TEXT NOT NULL CHECK(action IN (
    'drafted','linted','tested','shadowed','approved','activated','disabled','retired','rolled_back'
  )),
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator','system')),
  actor_parent_id TEXT,
  details_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(recipe_id,version) REFERENCES family_recipe_versions(recipe_id,version) ON DELETE CASCADE,
  FOREIGN KEY(actor_parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS capability_run_recipes (
  capability_run_id TEXT NOT NULL,
  recipe_id TEXT NOT NULL,
  recipe_version INTEGER NOT NULL,
  instructions_hash TEXT NOT NULL CHECK(length(instructions_hash)=64),
  created_at TEXT NOT NULL,
  PRIMARY KEY(capability_run_id,recipe_id),
  FOREIGN KEY(capability_run_id) REFERENCES capability_runs(id) ON DELETE CASCADE,
  FOREIGN KEY(recipe_id,recipe_version)
    REFERENCES family_recipe_versions(recipe_id,version) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS capability_run_parent_memories (
  capability_run_id TEXT NOT NULL,
  memory_key TEXT NOT NULL,
  memory_version INTEGER NOT NULL CHECK(memory_version > 0),
  content_hash TEXT NOT NULL CHECK(length(content_hash)=64),
  created_at TEXT NOT NULL,
  PRIMARY KEY(capability_run_id,memory_key,memory_version),
  FOREIGN KEY(capability_run_id) REFERENCES capability_runs(id) ON DELETE CASCADE,
  FOREIGN KEY(memory_key,memory_version)
    REFERENCES parent_memories(memory_key,version) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS capability_run_legacy_preferences (
  capability_run_id TEXT NOT NULL,
  preference_event_id TEXT NOT NULL,
  content_hash TEXT NOT NULL CHECK(length(content_hash)=64),
  created_at TEXT NOT NULL,
  PRIMARY KEY(capability_run_id,preference_event_id),
  FOREIGN KEY(capability_run_id) REFERENCES capability_runs(id) ON DELETE CASCADE,
  FOREIGN KEY(preference_event_id)
    REFERENCES agent_session_preference_events(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS thread_compactions (
  compaction_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  anchor_message_id TEXT,
  covers_through_ordinal INTEGER NOT NULL CHECK(covers_through_ordinal > 0),
  summary_json TEXT NOT NULL,
  summary_hash TEXT NOT NULL CHECK(length(summary_hash)=64),
  created_at TEXT NOT NULL,
  UNIQUE(session_id,covers_through_ordinal),
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE CASCADE,
  FOREIGN KEY(anchor_message_id) REFERENCES agent_messages(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_thread_compactions_latest
  ON thread_compactions(session_id,covers_through_ordinal DESC);

CREATE TABLE IF NOT EXISTS background_runs (
  run_id TEXT PRIMARY KEY,
  run_kind TEXT NOT NULL CHECK(length(run_kind) BETWEEN 1 AND 80),
  session_id TEXT,
  child_id TEXT,
  source_event_id TEXT,
  capability_run_id TEXT,
  capability_id TEXT,
  recipe_id TEXT,
  recipe_version INTEGER,
  status TEXT NOT NULL CHECK(status IN (
    'queued','running','completed','failed','cancelled'
  )),
  overlap_key TEXT,
  idempotency_key TEXT NOT NULL UNIQUE,
  request_hash TEXT NOT NULL CHECK(length(request_hash)=64),
  workspace_json TEXT NOT NULL DEFAULT '{}',
  policy_snapshot_json TEXT NOT NULL,
  policy_hash TEXT NOT NULL CHECK(length(policy_hash)=64),
  input_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT NOT NULL DEFAULT '{}',
  max_attempts INTEGER NOT NULL DEFAULT 3 CHECK(max_attempts BETWEEN 1 AND 10),
  active_attempt_id TEXT,
  lease_token_hash TEXT,
  lease_owner TEXT,
  leased_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  completed_at TEXT,
  FOREIGN KEY(session_id) REFERENCES agent_sessions(id) ON DELETE SET NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(source_event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(capability_run_id) REFERENCES capability_runs(id) ON DELETE CASCADE,
  FOREIGN KEY(recipe_id,recipe_version)
    REFERENCES family_recipe_versions(recipe_id,version) ON DELETE RESTRICT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_background_run_active_overlap
  ON background_runs(overlap_key)
  WHERE overlap_key IS NOT NULL AND status IN ('queued','running');
CREATE INDEX IF NOT EXISTS idx_background_run_ready
  ON background_runs(status,created_at);

CREATE TABLE IF NOT EXISTS background_overlap_receipts (
  idempotency_key TEXT PRIMARY KEY,
  request_hash TEXT NOT NULL CHECK(length(request_hash)=64),
  run_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES background_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS background_run_attempts (
  attempt_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
  worker_id TEXT NOT NULL,
  lease_token_hash TEXT NOT NULL CHECK(length(lease_token_hash)=64),
  status TEXT NOT NULL CHECK(status IN ('running','completed','failed','abandoned')),
  error_code TEXT,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  UNIQUE(run_id,attempt_number),
  FOREIGN KEY(run_id) REFERENCES background_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS background_run_events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  event_type TEXT NOT NULL CHECK(event_type IN (
    'queued','started','completed','failed','cancelled','wake_ready','wake_consumed',
    'delivery_requested','delivery_confirmed','delivery_sent','delivery_unknown','skipped_overlap'
  )),
  details_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES background_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS background_run_waiters (
  run_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('waiting','ready','consumed','cancelled')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(run_id,turn_id),
  FOREIGN KEY(run_id) REFERENCES background_runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY(turn_id) REFERENCES agent_turns(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS background_delivery_outbox (
  delivery_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  binding_id TEXT NOT NULL,
  audience TEXT NOT NULL DEFAULT 'parent' CHECK(audience='parent'),
  payload_pointer_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN (
    'awaiting_confirmation','queued','sending','sent','unknown','failed','cancelled'
  )),
  confirmed_by_parent_id TEXT,
  confirmed_at TEXT,
  external_message_id TEXT,
  idempotency_key TEXT NOT NULL UNIQUE,
  request_hash TEXT NOT NULL CHECK(length(request_hash)=64),
  max_attempts INTEGER NOT NULL DEFAULT 3 CHECK(max_attempts BETWEEN 1 AND 10),
  active_attempt_id TEXT,
  lease_owner TEXT,
  leased_at TEXT,
  lease_token_hash TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES background_runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY(binding_id) REFERENCES transport_bindings(id) ON DELETE CASCADE,
  FOREIGN KEY(confirmed_by_parent_id) REFERENCES parent_principals(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS background_delivery_attempts (
  attempt_id TEXT PRIMARY KEY,
  delivery_id TEXT NOT NULL,
  attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
  worker_id TEXT NOT NULL,
  lease_token_hash TEXT NOT NULL CHECK(length(lease_token_hash)=64),
  status TEXT NOT NULL CHECK(status IN ('sending','sent','unknown','failed','abandoned')),
  error_code TEXT,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  UNIQUE(delivery_id,attempt_number),
  FOREIGN KEY(delivery_id) REFERENCES background_delivery_outbox(delivery_id) ON DELETE CASCADE
);

INSERT INTO schema_migrations(version,name,applied_at)
VALUES(24,'governed adaptation control plane',strftime('%Y-%m-%dT%H:%M:%fZ','now'))
ON CONFLICT(version) DO NOTHING;
COMMIT;"""
        )
        background_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(background_runs)")
        }
        if "max_attempts" not in background_columns:
            conn.execute(
                """ALTER TABLE background_runs ADD COLUMN max_attempts INTEGER NOT NULL
                   DEFAULT 3 CHECK(max_attempts BETWEEN 1 AND 10)"""
            )
        for column, definition in {
            "child_id": "TEXT",
            "source_event_id": "TEXT",
            "capability_run_id": "TEXT",
            "request_hash": "TEXT NOT NULL DEFAULT ''",
            "active_attempt_id": "TEXT",
            "lease_token_hash": "TEXT",
        }.items():
            if column not in background_columns:
                conn.execute(f"ALTER TABLE background_runs ADD COLUMN {column} {definition}")
        delivery_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(background_delivery_outbox)")
        }
        for column, definition in {
            "request_hash": "TEXT NOT NULL DEFAULT ''",
            "max_attempts": "INTEGER NOT NULL DEFAULT 3",
            "active_attempt_id": "TEXT",
            "lease_owner": "TEXT",
            "leased_at": "TEXT",
            "lease_token_hash": "TEXT",
        }.items():
            if column not in delivery_columns:
                conn.execute(
                    f"ALTER TABLE background_delivery_outbox ADD COLUMN {column} {definition}"
                )
        attempt_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(background_run_attempts)")
        }
        if "lease_token_hash" not in attempt_columns:
            conn.execute(
                """ALTER TABLE background_run_attempts ADD COLUMN lease_token_hash TEXT
                   NOT NULL DEFAULT ''"""
            )
        compaction_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(thread_compactions)")
        }
        if "anchor_message_id" not in compaction_columns:
            conn.execute("ALTER TABLE thread_compactions ADD COLUMN anchor_message_id TEXT")
        recipe_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(family_recipes)")
        }
        if "capability_id" not in recipe_columns:
            conn.execute(
                "ALTER TABLE family_recipes ADD COLUMN capability_id TEXT NOT NULL DEFAULT 'parent_chat'"
            )
        recipe_version_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(family_recipe_versions)")
        }
        if "test_evaluation_id" not in recipe_version_columns:
            conn.execute("ALTER TABLE family_recipe_versions ADD COLUMN test_evaluation_id TEXT")
        if "shadow_evaluation_id" not in recipe_version_columns:
            conn.execute("ALTER TABLE family_recipe_versions ADD COLUMN shadow_evaluation_id TEXT")
        conn.execute(f"PRAGMA user_version = {max(current, 24)}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    required = {
        "parent_memory_settings",
        "parent_memories",
        "parent_memory_events",
        "parent_memory_tombstones",
        "delivery_outbox_parent_memories",
        "family_recipes",
        "family_recipe_versions",
        "family_recipe_evaluations",
        "family_recipe_events",
        "capability_run_recipes",
        "capability_run_parent_memories",
        "capability_run_legacy_preferences",
        "thread_compactions",
        "background_runs",
        "background_overlap_receipts",
        "background_run_attempts",
        "background_run_events",
        "background_run_waiters",
        "background_delivery_outbox",
        "background_delivery_attempts",
    }
    required_columns = {
        "parent_memory_settings": {
            "scope_type",
            "scope_id",
            "enabled",
            "updated_by_parent_id",
            "updated_at",
        },
        "parent_memories": {
            "memory_key",
            "version",
            "scope_type",
            "scope_id",
            "child_id",
            "category",
            "value",
            "provenance",
            "creator_type",
            "parent_id",
            "source_message_id",
            "sensitivity",
            "status",
            "review_at",
            "expires_at",
            "why_saved",
            "content_hash",
            "created_at",
            "updated_at",
        },
        "parent_memory_events": {
            "event_id",
            "memory_key",
            "action",
            "prior_version",
            "new_version",
            "actor_type",
            "actor_parent_id",
            "reason",
            "prior_content_hash",
            "new_content_hash",
            "undone_by_event_id",
            "created_at",
        },
        "parent_memory_tombstones": {
            "tombstone_id",
            "memory_record_hash",
            "memory_key_hash",
            "version",
            "actor_type",
            "deleted_at",
        },
        "delivery_outbox_parent_memories": {
            "delivery_id",
            "memory_key",
            "memory_version",
            "content_hash",
            "created_at",
        },
        "slack_file_outbox": {
            "active_attempt_id",
            "lease_token_hash",
        },
        "artifact_file_outbox": {
            "active_attempt_id",
            "lease_token_hash",
        },
        "family_recipes": {
            "recipe_id",
            "capability_id",
            "title",
            "description",
            "created_by_parent_id",
            "created_at",
        },
        "family_recipe_versions": {
            "recipe_id",
            "version",
            "instructions",
            "instructions_hash",
            "tool_names_json",
            "status",
            "test_report_json",
            "shadow_report_json",
            "test_evaluation_id",
            "shadow_evaluation_id",
            "approved_by_parent_id",
            "approved_at",
            "activated_at",
            "created_at",
            "updated_at",
        },
        "family_recipe_evaluations": {
            "evaluation_id",
            "recipe_id",
            "version",
            "instructions_hash",
            "evaluation_kind",
            "suite_version",
            "fixture_digest",
            "case_count",
            "passed_count",
            "status",
            "evaluator",
            "results_json",
            "created_at",
        },
        "family_recipe_events": {
            "event_id",
            "recipe_id",
            "version",
            "action",
            "actor_type",
            "actor_parent_id",
            "details_json",
            "created_at",
        },
        "capability_run_recipes": {
            "capability_run_id",
            "recipe_id",
            "recipe_version",
            "instructions_hash",
            "created_at",
        },
        "capability_run_parent_memories": {
            "capability_run_id",
            "memory_key",
            "memory_version",
            "content_hash",
            "created_at",
        },
        "capability_run_legacy_preferences": {
            "capability_run_id",
            "preference_event_id",
            "content_hash",
            "created_at",
        },
        "thread_compactions": {
            "compaction_id",
            "session_id",
            "anchor_message_id",
            "covers_through_ordinal",
            "summary_json",
            "summary_hash",
            "created_at",
        },
        "background_runs": {
            "run_id",
            "run_kind",
            "session_id",
            "child_id",
            "source_event_id",
            "capability_run_id",
            "capability_id",
            "recipe_id",
            "recipe_version",
            "status",
            "overlap_key",
            "idempotency_key",
            "request_hash",
            "workspace_json",
            "policy_snapshot_json",
            "policy_hash",
            "input_json",
            "result_json",
            "max_attempts",
            "active_attempt_id",
            "lease_token_hash",
            "lease_owner",
            "leased_at",
            "created_at",
            "updated_at",
            "completed_at",
        },
        "background_overlap_receipts": {
            "idempotency_key",
            "request_hash",
            "run_id",
            "created_at",
        },
        "background_run_attempts": {
            "attempt_id",
            "run_id",
            "attempt_number",
            "worker_id",
            "lease_token_hash",
            "status",
            "error_code",
            "started_at",
            "completed_at",
        },
        "background_run_events": {
            "event_id",
            "run_id",
            "event_type",
            "details_json",
            "created_at",
        },
        "background_run_waiters": {
            "run_id",
            "turn_id",
            "status",
            "created_at",
            "updated_at",
        },
        "background_delivery_outbox": {
            "delivery_id",
            "run_id",
            "binding_id",
            "audience",
            "payload_pointer_json",
            "status",
            "confirmed_by_parent_id",
            "confirmed_at",
            "external_message_id",
            "idempotency_key",
            "request_hash",
            "max_attempts",
            "active_attempt_id",
            "lease_owner",
            "leased_at",
            "lease_token_hash",
            "created_at",
            "updated_at",
        },
        "background_delivery_attempts": {
            "attempt_id",
            "delivery_id",
            "attempt_number",
            "worker_id",
            "lease_token_hash",
            "status",
            "error_code",
            "started_at",
            "completed_at",
        },
    }
    missing_columns = {
        table: sorted(
            columns
            - {
                str(row[1])
                for row in conn.execute(f'PRAGMA table_info("{table}")')
            }
        )
        for table, columns in required_columns.items()
        if table in _tables(conn)
    }
    missing_columns = {table: columns for table, columns in missing_columns.items() if columns}
    required_indexes = {
        "idx_parent_memory_one_current",
        "idx_parent_memory_recall",
        "idx_parent_memory_events_key",
        "idx_family_recipe_one_active",
        "idx_thread_compactions_latest",
        "idx_background_run_active_overlap",
        "idx_background_run_ready",
    }
    present_indexes = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    missing_indexes = required_indexes - present_indexes
    required_triggers = {
        "guard_family_recipe_identity_immutable",
        "guard_family_recipe_definition_immutable",
        "guard_family_recipe_evaluation_immutable",
    }
    present_triggers = {
        str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    }
    missing_triggers = required_triggers - present_triggers
    # Foreign keys are authenticated as grouped constraints, not as independent
    # columns. Otherwise two forged single-column keys can impersonate one
    # composite lineage edge.
    foreign_key = lambda table, on_delete, *columns: (  # noqa: E731
        table,
        on_delete,
        tuple(columns),
    )
    required_foreign_keys = {
        "parent_memory_settings": {
            foreign_key("parent_principals", "SET NULL", ("updated_by_parent_id", "id")),
        },
        "parent_memories": {
            foreign_key("children", "CASCADE", ("child_id", "id")),
            foreign_key("parent_principals", "SET NULL", ("parent_id", "id")),
            foreign_key("agent_messages", "CASCADE", ("source_message_id", "id")),
        },
        "parent_memory_events": {
            foreign_key("parent_principals", "SET NULL", ("actor_parent_id", "id")),
            foreign_key("parent_memory_events", "SET NULL", ("undone_by_event_id", "event_id")),
        },
        "delivery_outbox_parent_memories": {
            foreign_key("delivery_outbox", "CASCADE", ("delivery_id", "id")),
        },
        "family_recipes": {
            foreign_key("parent_principals", "SET NULL", ("created_by_parent_id", "id")),
        },
        "family_recipe_versions": {
            foreign_key("family_recipes", "CASCADE", ("recipe_id", "recipe_id")),
            foreign_key("parent_principals", "SET NULL", ("approved_by_parent_id", "id")),
        },
        "family_recipe_evaluations": {
            foreign_key(
                "family_recipe_versions",
                "CASCADE",
                ("recipe_id", "recipe_id"),
                ("version", "version"),
            ),
        },
        "family_recipe_events": {
            foreign_key(
                "family_recipe_versions",
                "CASCADE",
                ("recipe_id", "recipe_id"),
                ("version", "version"),
            ),
            foreign_key("parent_principals", "SET NULL", ("actor_parent_id", "id")),
        },
        "capability_run_recipes": {
            foreign_key("capability_runs", "CASCADE", ("capability_run_id", "id")),
            foreign_key(
                "family_recipe_versions",
                "RESTRICT",
                ("recipe_id", "recipe_id"),
                ("recipe_version", "version"),
            ),
        },
        "capability_run_parent_memories": {
            foreign_key("capability_runs", "CASCADE", ("capability_run_id", "id")),
            foreign_key(
                "parent_memories",
                "RESTRICT",
                ("memory_key", "memory_key"),
                ("memory_version", "version"),
            ),
        },
        "capability_run_legacy_preferences": {
            foreign_key("capability_runs", "CASCADE", ("capability_run_id", "id")),
            foreign_key(
                "agent_session_preference_events",
                "CASCADE",
                ("preference_event_id", "id"),
            ),
        },
        "thread_compactions": {
            foreign_key("agent_sessions", "CASCADE", ("session_id", "id")),
            foreign_key("agent_messages", "CASCADE", ("anchor_message_id", "id")),
        },
        "background_runs": {
            foreign_key("agent_sessions", "SET NULL", ("session_id", "id")),
            foreign_key("children", "CASCADE", ("child_id", "id")),
            foreign_key("events", "CASCADE", ("source_event_id", "id")),
            foreign_key("capability_runs", "CASCADE", ("capability_run_id", "id")),
            foreign_key(
                "family_recipe_versions",
                "RESTRICT",
                ("recipe_id", "recipe_id"),
                ("recipe_version", "version"),
            ),
        },
        "background_overlap_receipts": {
            foreign_key("background_runs", "CASCADE", ("run_id", "run_id")),
        },
        "background_run_attempts": {
            foreign_key("background_runs", "CASCADE", ("run_id", "run_id")),
        },
        "background_run_events": {
            foreign_key("background_runs", "CASCADE", ("run_id", "run_id")),
        },
        "background_run_waiters": {
            foreign_key("background_runs", "CASCADE", ("run_id", "run_id")),
            foreign_key("agent_turns", "CASCADE", ("turn_id", "id")),
        },
        "background_delivery_outbox": {
            foreign_key("background_runs", "CASCADE", ("run_id", "run_id")),
            foreign_key("transport_bindings", "CASCADE", ("binding_id", "id")),
            foreign_key("parent_principals", "SET NULL", ("confirmed_by_parent_id", "id")),
        },
        "background_delivery_attempts": {
            foreign_key(
                "background_delivery_outbox",
                "CASCADE",
                ("delivery_id", "delivery_id"),
            ),
        },
    }
    missing_foreign_keys: dict[str, list[tuple[Any, ...]]] = {}
    for table, expected in required_foreign_keys.items():
        grouped: dict[int, list[sqlite3.Row]] = {}
        for item in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            grouped.setdefault(int(item[0]), []).append(item)
        present = {
            (
                str(items[0][2]),
                str(items[0][6]).upper(),
                tuple(
                    (str(item[3]), str(item[4]))
                    for item in sorted(items, key=lambda value: int(value[1]))
                ),
            )
            for items in grouped.values()
        }
        missing = sorted(expected - present)
        if missing:
            missing_foreign_keys[table] = missing

    required_primary_keys = {
        "parent_memory_settings": ("scope_type", "scope_id"),
        "parent_memories": ("memory_key", "version"),
        "parent_memory_events": ("event_id",),
        "parent_memory_tombstones": ("tombstone_id",),
        "delivery_outbox_parent_memories": (
            "delivery_id",
            "memory_key",
            "memory_version",
        ),
        "family_recipes": ("recipe_id",),
        "family_recipe_versions": ("recipe_id", "version"),
        "family_recipe_evaluations": ("evaluation_id",),
        "family_recipe_events": ("event_id",),
        "capability_run_recipes": ("capability_run_id", "recipe_id"),
        "capability_run_parent_memories": (
            "capability_run_id",
            "memory_key",
            "memory_version",
        ),
        "capability_run_legacy_preferences": (
            "capability_run_id",
            "preference_event_id",
        ),
        "thread_compactions": ("compaction_id",),
        "background_runs": ("run_id",),
        "background_overlap_receipts": ("idempotency_key",),
        "background_run_attempts": ("attempt_id",),
        "background_run_events": ("event_id",),
        "background_run_waiters": ("run_id", "turn_id"),
        "background_delivery_outbox": ("delivery_id",),
        "background_delivery_attempts": ("attempt_id",),
    }
    invalid_primary_keys = {
        table: {
            "expected": expected,
            "present": tuple(
                str(row[1])
                for row in sorted(
                    conn.execute(f'PRAGMA table_info("{table}")').fetchall(),
                    key=lambda value: int(value[5]) if int(value[5]) else 10_000,
                )
                if int(row[5])
            ),
        }
        for table, expected in required_primary_keys.items()
        if tuple(
            str(row[1])
            for row in sorted(
                conn.execute(f'PRAGMA table_info("{table}")').fetchall(),
                key=lambda value: int(value[5]) if int(value[5]) else 10_000,
            )
            if int(row[5])
        )
        != expected
    }

    def index_shape(table: str, name: str) -> tuple[tuple[str, ...], bool, bool] | None:
        index = next(
            (row for row in conn.execute(f'PRAGMA index_list("{table}")') if str(row[1]) == name),
            None,
        )
        if index is None:
            return None
        columns = tuple(
            str(row[2])
            for row in sorted(
                conn.execute(f'PRAGMA index_info("{name}")').fetchall(),
                key=lambda value: int(value[0]),
            )
        )
        return columns, bool(index[2]), bool(index[4])

    required_index_shapes = {
        ("parent_memories", "idx_parent_memory_one_current"): (("memory_key",), True, True),
        ("parent_memories", "idx_parent_memory_recall"): (
            ("scope_type", "scope_id", "status", "category", "updated_at"),
            False,
            False,
        ),
        ("parent_memory_events", "idx_parent_memory_events_key"): (
            ("memory_key", "created_at", "event_id"),
            False,
            False,
        ),
        ("delivery_outbox_parent_memories", "idx_delivery_outbox_parent_memory"): (
            ("memory_key", "memory_version"),
            False,
            False,
        ),
        ("family_recipe_versions", "idx_family_recipe_one_active"): (
            ("recipe_id",),
            True,
            True,
        ),
        ("thread_compactions", "idx_thread_compactions_latest"): (
            ("session_id", "covers_through_ordinal"),
            False,
            False,
        ),
        ("background_runs", "idx_background_run_active_overlap"): (
            ("overlap_key",),
            True,
            True,
        ),
        ("background_runs", "idx_background_run_ready"): (
            ("status", "created_at"),
            False,
            False,
        ),
    }
    invalid_indexes = {
        name: {"expected": expected, "present": index_shape(table, name)}
        for (table, name), expected in required_index_shapes.items()
        if index_shape(table, name) != expected
    }

    required_unique_shapes = {
        "thread_compactions": {("session_id", "covers_through_ordinal")},
        "background_runs": {("idempotency_key",)},
        "background_run_attempts": {("run_id", "attempt_number")},
        "background_delivery_outbox": {("idempotency_key",)},
        "background_delivery_attempts": {("delivery_id", "attempt_number")},
        "parent_memory_tombstones": {("memory_record_hash",)},
    }
    missing_unique_constraints: dict[str, list[tuple[str, ...]]] = {}
    for table, expected in required_unique_shapes.items():
        present = {
            tuple(
                str(info[2])
                for info in sorted(
                    conn.execute(f'PRAGMA index_info("{row[1]}")').fetchall(),
                    key=lambda value: int(value[0]),
                )
            )
            for row in conn.execute(f'PRAGMA index_list("{table}")')
            if bool(row[2]) and not bool(row[4])
        }
        missing = sorted(expected - present)
        if missing:
            missing_unique_constraints[table] = missing

    normalized_table_sql = {
        table: re.sub(
            r"\s+",
            "",
            str(
                conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()[0]
            ).lower(),
        )
        for table in required
    }
    required_check_fragments = {
        "parent_memory_settings": ["check(enabledin(0,1))"],
        "parent_memories": [
            "check(version>0)",
            "check(length(value)<=1000)",
            "check(length(content_hash)=64)",
            "'proposed','active','superseded','disabled','forgotten'",
        ],
        "parent_memory_events": ["'remember','propose','approve','update','disable','enable','forget','undo'"],
        "parent_memory_tombstones": [
            "check(length(memory_record_hash)=64)",
            "check(length(memory_key_hash)=64)",
            "check(version>0)",
            "check(actor_typein('parent','operator'))",
        ],
        "delivery_outbox_parent_memories": [
            "check(memory_version>0)",
            "check(length(content_hash)=64)",
        ],
        "family_recipes": ["check(length(title)between1and160)", "check(length(description)between1and500)"],
        "family_recipe_versions": [
            "check(version>0)",
            "check(length(instructions)between1and12000)",
            "check(length(instructions_hash)=64)",
            "'draft','linted','tested','shadow','approved','active','disabled','retired'",
        ],
        "family_recipe_evaluations": [
            "check(evaluation_kindin('tests','shadow'))",
            "check(case_count>0)",
            "check(passed_countbetween0andcase_count)",
        ],
        "family_recipe_events": ["'drafted','linted','tested','shadowed','approved','activated','disabled','retired','rolled_back'"],
        "capability_run_recipes": ["check(length(instructions_hash)=64)"],
        "capability_run_parent_memories": [
            "check(memory_version>0)",
            "check(length(content_hash)=64)",
        ],
        "capability_run_legacy_preferences": ["check(length(content_hash)=64)"],
        "thread_compactions": ["check(covers_through_ordinal>0)", "check(length(summary_hash)=64)"],
        "background_runs": [
            "'queued','running','completed','failed','cancelled'",
            "check(length(request_hash)=64)",
            "check(length(policy_hash)=64)",
            "check(max_attemptsbetween1and10)",
        ],
        "background_overlap_receipts": ["check(length(request_hash)=64)"],
        "background_run_attempts": [
            "check(attempt_number>0)",
            "check(length(lease_token_hash)=64)",
            "'running','completed','failed','abandoned'",
        ],
        "background_run_events": ["'delivery_sent','delivery_unknown','skipped_overlap'"],
        "background_run_waiters": ["'waiting','ready','consumed','cancelled'"],
        "background_delivery_outbox": [
            "check(audience='parent')",
            "'awaiting_confirmation','queued','sending','sent','unknown','failed','cancelled'",
            "check(length(request_hash)=64)",
            "check(max_attemptsbetween1and10)",
        ],
        "background_delivery_attempts": [
            "check(attempt_number>0)",
            "check(length(lease_token_hash)=64)",
            "'sending','sent','unknown','failed','abandoned'",
        ],
    }
    missing_check_fragments = {
        table: [
            fragment
            for fragment in fragments
            if re.sub(r"\s+", "", fragment.lower()) not in normalized_table_sql[table]
        ]
        for table, fragments in required_check_fragments.items()
    }
    missing_check_fragments = {
        table: fragments for table, fragments in missing_check_fragments.items() if fragments
    }
    invalid_hash_rows = any(
        conn.execute(
            f"SELECT 1 FROM {table} WHERE {column} IS NULL OR length({column})!=64 LIMIT 1"
        ).fetchone()
        for table, column in (
            ("parent_memories", "content_hash"),
            ("parent_memory_tombstones", "memory_record_hash"),
            ("parent_memory_tombstones", "memory_key_hash"),
            ("delivery_outbox_parent_memories", "content_hash"),
            ("family_recipe_versions", "instructions_hash"),
            ("family_recipe_evaluations", "instructions_hash"),
            ("family_recipe_evaluations", "fixture_digest"),
            ("capability_run_parent_memories", "content_hash"),
            ("capability_run_legacy_preferences", "content_hash"),
            ("thread_compactions", "summary_hash"),
            ("background_runs", "request_hash"),
            ("background_runs", "policy_hash"),
            ("background_overlap_receipts", "request_hash"),
            ("background_run_attempts", "lease_token_hash"),
            ("background_delivery_outbox", "request_hash"),
            ("background_delivery_attempts", "lease_token_hash"),
        )
    )
    ledger = conn.execute("SELECT name FROM schema_migrations WHERE version=24").fetchone()
    if (
        required - _tables(conn)
        or missing_columns
        or missing_indexes
        or missing_triggers
        or missing_foreign_keys
        or invalid_primary_keys
        or invalid_indexes
        or missing_unique_constraints
        or missing_check_fragments
        or invalid_hash_rows
        or not ledger
        or ledger["name"] != "governed adaptation control plane"
        or int(conn.execute("PRAGMA user_version").fetchone()[0]) != max(current, 24)
        or conn.execute("PRAGMA foreign_key_check").fetchall()
    ):
        raise RuntimeError(
            "incomplete schema v24 adaptation-control migration: "
            f"missing_tables={sorted(required - _tables(conn))!r}, "
            f"missing_columns={missing_columns!r}, "
            f"missing_indexes={sorted(missing_indexes)!r}, "
            f"missing_triggers={sorted(missing_triggers)!r}, "
            f"missing_foreign_keys={missing_foreign_keys!r}, "
            f"invalid_primary_keys={invalid_primary_keys!r}, "
            f"invalid_indexes={invalid_indexes!r}, "
            f"missing_unique_constraints={missing_unique_constraints!r}, "
            f"missing_check_fragments={missing_check_fragments!r}, "
            f"invalid_hash_rows={invalid_hash_rows}"
        )


def _apply_migration_25(conn: sqlite3.Connection) -> None:
    """Add evaluated personalization while keeping replay separate from learner evidence."""

    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    try:
        control_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(learner_model_controls)")
        }
        for column, definition in {
            "recommendation_use_enabled": (
                "INTEGER NOT NULL DEFAULT 0 CHECK(recommendation_use_enabled IN (0,1))"
            ),
            "recommendation_policy_id": "TEXT",
            "recommendation_policy_version": (
                "INTEGER CHECK(recommendation_policy_version IS NULL OR recommendation_policy_version > 0)"
            ),
            "recommendation_policy_hash": (
                "TEXT CHECK(recommendation_policy_hash IS NULL OR length(recommendation_policy_hash)=64)"
            ),
            "personalization_epoch": "INTEGER NOT NULL DEFAULT 0 CHECK(personalization_epoch >= 0)",
        }.items():
            if column not in control_columns:
                conn.execute(f"ALTER TABLE learner_model_controls ADD COLUMN {column} {definition}")
        conn.executescript(
            """BEGIN IMMEDIATE;
DROP TRIGGER IF EXISTS guard_personalization_policy_definition_immutable;
DROP TRIGGER IF EXISTS guard_personalization_policy_promotion_update;
DROP TRIGGER IF EXISTS guard_personalization_policy_epoch_update;
DROP TRIGGER IF EXISTS guard_personalization_control_immutable;
DROP TRIGGER IF EXISTS guard_personalization_policy_event_immutable;
DROP TRIGGER IF EXISTS guard_personalization_control_binding_insert;
DROP TRIGGER IF EXISTS guard_personalization_control_binding_update;
DROP TRIGGER IF EXISTS guard_personalization_control_epoch_update;
DROP TRIGGER IF EXISTS guard_personalization_evaluation_immutable;
DROP TRIGGER IF EXISTS guard_personalization_evaluation_delete;
DROP TRIGGER IF EXISTS guard_personalization_promotion_review_immutable;
DROP TRIGGER IF EXISTS guard_personalization_promotion_review_delete;
DROP TRIGGER IF EXISTS guard_personalization_decision_immutable;
DROP TRIGGER IF EXISTS guard_personalization_decision_binding_insert;
DROP TRIGGER IF EXISTS delete_personalization_decision_with_plan;
DROP TRIGGER IF EXISTS guard_personalization_signal_immutable;
DROP TRIGGER IF EXISTS guard_personalization_signal_binding_insert;
DROP TRIGGER IF EXISTS guard_personalization_replay_insert;
DROP TRIGGER IF EXISTS guard_personalization_replay_immutable;
CREATE TABLE IF NOT EXISTS personalization_policies (
  policy_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version > 0),
  feature_schema_version TEXT NOT NULL,
  definition_json TEXT NOT NULL,
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  authorization_epoch INTEGER NOT NULL DEFAULT 0 CHECK(authorization_epoch >= 0),
  status TEXT NOT NULL CHECK(status IN (
    'candidate','evaluated','approved','active','disabled','retired'
  )),
  evaluation_id TEXT,
  promotion_review_id TEXT,
  approved_by_operator TEXT,
  approval_reason TEXT,
  activation_reason TEXT,
  approved_at TEXT,
  activated_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(policy_id,version)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_personalization_one_active
  ON personalization_policies((1)) WHERE status='active';

CREATE TABLE IF NOT EXISTS personalization_control_events (
  event_id TEXT NOT NULL PRIMARY KEY,
  child_id TEXT NOT NULL,
  enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
  policy_id TEXT,
  policy_version INTEGER CHECK(policy_version IS NULL OR policy_version > 0),
  definition_hash TEXT CHECK(definition_hash IS NULL OR length(definition_hash)=64),
  authorization_epoch INTEGER NOT NULL CHECK(authorization_epoch > 0),
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator')),
  reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 500),
  created_at TEXT NOT NULL,
  CHECK(
    (enabled=1 AND policy_id IS NOT NULL AND policy_version IS NOT NULL AND definition_hash IS NOT NULL)
    OR (enabled=0 AND policy_id IS NULL AND policy_version IS NULL AND definition_hash IS NULL)
  ),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_personalization_control_child
  ON personalization_control_events(child_id,created_at DESC);

CREATE TABLE IF NOT EXISTS personalization_policy_events (
  event_id TEXT NOT NULL PRIMARY KEY,
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  action TEXT NOT NULL CHECK(action IN (
    'evaluated','promotion_reviewed','approved','activated','disabled','rolled_back'
  )),
  actor_type TEXT NOT NULL CHECK(actor_type IN ('system','operator')),
  operator_id TEXT,
  reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 500),
  created_at TEXT NOT NULL,
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES personalization_policies(policy_id,version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS personalization_evaluations (
  evaluation_id TEXT NOT NULL PRIMARY KEY,
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  suite_version TEXT NOT NULL,
  fixture_digest TEXT NOT NULL CHECK(length(fixture_digest)=64),
  case_count INTEGER NOT NULL CHECK(case_count > 0),
  benefit_delta REAL NOT NULL,
  baseline_calibration_error REAL NOT NULL CHECK(baseline_calibration_error BETWEEN 0 AND 1),
  candidate_calibration_error REAL NOT NULL CHECK(candidate_calibration_error BETWEEN 0 AND 1),
  safety_pass INTEGER NOT NULL CHECK(safety_pass IN (0,1)),
  privacy_pass INTEGER NOT NULL CHECK(privacy_pass IN (0,1)),
  fairness_pass INTEGER NOT NULL CHECK(fairness_pass IN (0,1)),
  golden_pass INTEGER NOT NULL CHECK(golden_pass IN (0,1)),
  status TEXT NOT NULL CHECK(status IN ('pass','fail')),
  evaluator TEXT NOT NULL,
  evaluator_digest TEXT NOT NULL CHECK(length(evaluator_digest)=64),
  results_json TEXT NOT NULL,
  results_hash TEXT NOT NULL CHECK(length(results_hash)=64),
  created_at TEXT NOT NULL,
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES personalization_policies(policy_id,version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS personalization_promotion_reviews (
  review_id TEXT NOT NULL PRIMARY KEY,
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  evaluation_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL CHECK(length(reviewer_id) BETWEEN 1 AND 160),
  lab_report_json TEXT NOT NULL,
  lab_report_hash TEXT NOT NULL UNIQUE CHECK(length(lab_report_hash)=64),
  created_at TEXT NOT NULL,
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES personalization_policies(policy_id,version) ON DELETE CASCADE,
  FOREIGN KEY(evaluation_id) REFERENCES personalization_evaluations(evaluation_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS personalization_decisions (
  decision_id TEXT NOT NULL PRIMARY KEY,
  child_id TEXT NOT NULL,
  event_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  event_text_hash TEXT NOT NULL CHECK(length(event_text_hash)=64),
  workflow TEXT NOT NULL,
  proposal_hash TEXT NOT NULL CHECK(length(proposal_hash)=64),
  baseline_plan_id TEXT NOT NULL,
  baseline_plan_hash TEXT NOT NULL CHECK(length(baseline_plan_hash)=64),
  concept_ids_json TEXT NOT NULL,
  concept_refs_json TEXT NOT NULL CHECK(json_valid(concept_refs_json)),
  question_intent TEXT NOT NULL,
  plan_id TEXT NOT NULL,
  plan_hash TEXT NOT NULL CHECK(length(plan_hash)=64),
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  family_authorization_epoch INTEGER NOT NULL CHECK(family_authorization_epoch >= 0),
  policy_authorization_epoch INTEGER NOT NULL CHECK(policy_authorization_epoch >= 0),
  evaluation_id TEXT,
  promotion_review_id TEXT,
  mode TEXT NOT NULL CHECK(mode IN ('shadow','active')),
  feature_vector_json TEXT NOT NULL,
  feature_vector_hash TEXT NOT NULL CHECK(length(feature_vector_hash)=64),
  baseline_json TEXT NOT NULL,
  candidate_json TEXT NOT NULL,
  decision_hash TEXT NOT NULL UNIQUE CHECK(length(decision_hash)=64),
  created_at TEXT NOT NULL,
  UNIQUE(event_id,policy_id,policy_version),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(event_id) REFERENCES events(id) ON DELETE CASCADE,
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES personalization_policies(policy_id,version) ON DELETE RESTRICT,
  FOREIGN KEY(evaluation_id) REFERENCES personalization_evaluations(evaluation_id) ON DELETE RESTRICT,
  FOREIGN KEY(promotion_review_id) REFERENCES personalization_promotion_reviews(review_id) ON DELETE RESTRICT,
  CHECK(
    (mode='active' AND evaluation_id IS NOT NULL AND promotion_review_id IS NOT NULL)
    OR (mode='shadow' AND promotion_review_id IS NULL)
  )
);
CREATE INDEX IF NOT EXISTS idx_personalization_decision_child
  ON personalization_decisions(child_id,created_at DESC);

CREATE TABLE IF NOT EXISTS personalization_feedback_signals (
  signal_id TEXT NOT NULL PRIMARY KEY,
  feedback_id INTEGER NOT NULL UNIQUE,
  child_id TEXT NOT NULL,
  plan_id TEXT,
  plan_hash TEXT,
  outcome TEXT NOT NULL CHECK(outcome IN (
    'loved','engaged','neutral','too_easy','too_hard','not_used','disliked','helpful','not_helpful'
  )),
  used INTEGER CHECK(used IN (0,1)),
  chose_to_continue INTEGER CHECK(chose_to_continue IN (0,1)),
  made_prediction INTEGER CHECK(made_prediction IN (0,1)),
  gave_explanation INTEGER CHECK(gave_explanation IN (0,1)),
  help_level TEXT CHECK(help_level IN (
    'none','clarifying_prompt','general_prompt','hint','worked_example','joint_completion','direct_answer','unknown'
  )),
  asked_new_question INTEGER CHECK(asked_new_question IN (0,1)),
  parent_correction INTEGER CHECK(parent_correction IN (0,1)),
  created_at TEXT NOT NULL,
  FOREIGN KEY(feedback_id) REFERENCES feedback(id) ON DELETE CASCADE,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_personalization_signal_child
  ON personalization_feedback_signals(child_id,created_at DESC);

CREATE TABLE IF NOT EXISTS personalization_replay_cases (
  case_id TEXT NOT NULL PRIMARY KEY,
  source_kind TEXT NOT NULL CHECK(source_kind IN ('public_fixture','synthetic','approved_deidentified')),
  license_status TEXT NOT NULL CHECK(license_status IN ('public','synthetic','approved')),
  privacy_approval_id TEXT CHECK(
    privacy_approval_id IS NULL OR (
      length(privacy_approval_id) BETWEEN 1 AND 160
      AND privacy_approval_id NOT GLOB '*[^A-Za-z0-9_.:-]*'
    )
  ),
  feature_schema_version TEXT NOT NULL CHECK(feature_schema_version='phase5-v2'),
  feature_vector_json TEXT NOT NULL CHECK(json_valid(feature_vector_json)),
  outcome_json TEXT NOT NULL CHECK(json_valid(outcome_json)),
  source_digest TEXT NOT NULL CHECK(
    length(source_digest)=64 AND source_digest NOT GLOB '*[^0-9a-f]*'
  ),
  case_hash TEXT NOT NULL UNIQUE CHECK(
    length(case_hash)=64 AND case_hash NOT GLOB '*[^0-9a-f]*'
  ),
  created_by_operator TEXT NOT NULL CHECK(
    length(created_by_operator) BETWEEN 1 AND 160
    AND created_by_operator NOT GLOB '*[^A-Za-z0-9_.:-]*'
  ),
  created_at TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS guard_personalization_policy_definition_immutable
BEFORE UPDATE OF policy_id,version,feature_schema_version,definition_json,definition_hash,created_at
ON personalization_policies
BEGIN SELECT RAISE(ABORT,'personalization policy definition is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_policy_promotion_update
BEFORE UPDATE OF status,evaluation_id,promotion_review_id,approved_by_operator,approved_at,activated_at
ON personalization_policies
WHEN NEW.status IN ('approved','active') AND (
  NEW.evaluation_id IS NULL OR NEW.promotion_review_id IS NULL OR NEW.approved_by_operator IS NULL
  OR NEW.approved_at IS NULL
  OR NOT EXISTS(
    SELECT 1 FROM personalization_evaluations e
    WHERE e.evaluation_id=NEW.evaluation_id AND e.policy_id=NEW.policy_id
      AND e.policy_version=NEW.version AND e.definition_hash=NEW.definition_hash AND e.status='pass'
  )
  OR NOT EXISTS(
    SELECT 1 FROM personalization_promotion_reviews r
    WHERE r.review_id=NEW.promotion_review_id AND r.policy_id=NEW.policy_id
      AND r.policy_version=NEW.version AND r.definition_hash=NEW.definition_hash
      AND r.evaluation_id=NEW.evaluation_id
  )
  OR (NEW.status='active' AND NEW.activated_at IS NULL)
)
BEGIN SELECT RAISE(ABORT,'personalization promotion evidence is incomplete'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_policy_epoch_update
BEFORE UPDATE OF status,authorization_epoch ON personalization_policies
WHEN (
  ((OLD.status='active') IS NOT (NEW.status='active'))
  AND NEW.authorization_epoch!=OLD.authorization_epoch+1
) OR (
  ((OLD.status='active') IS (NEW.status='active'))
  AND NEW.authorization_epoch!=OLD.authorization_epoch
)
BEGIN SELECT RAISE(ABORT,'personalization policy authorization epoch mismatch'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_control_immutable
BEFORE UPDATE ON personalization_control_events
BEGIN SELECT RAISE(ABORT,'personalization control event is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_policy_event_immutable
BEFORE UPDATE ON personalization_policy_events
BEGIN SELECT RAISE(ABORT,'personalization policy event is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_control_binding_insert
BEFORE INSERT ON learner_model_controls
WHEN (
  NEW.recommendation_use_enabled=1 AND (
    NEW.recommendation_policy_id IS NULL OR NEW.recommendation_policy_version IS NULL
    OR NEW.recommendation_policy_hash IS NULL OR NOT EXISTS(
      SELECT 1 FROM personalization_policies p
      WHERE p.policy_id=NEW.recommendation_policy_id
        AND p.version=NEW.recommendation_policy_version
        AND p.definition_hash=NEW.recommendation_policy_hash AND p.status='active'
    )
  )
) OR (
  NEW.recommendation_use_enabled=0 AND (
    NEW.recommendation_policy_id IS NOT NULL OR NEW.recommendation_policy_version IS NOT NULL
    OR NEW.recommendation_policy_hash IS NOT NULL
  )
)
BEGIN SELECT RAISE(ABORT,'invalid family personalization policy binding'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_control_binding_update
BEFORE UPDATE OF recommendation_use_enabled,recommendation_policy_id,
  recommendation_policy_version,recommendation_policy_hash ON learner_model_controls
WHEN (
  NEW.recommendation_use_enabled=1 AND (
    NEW.recommendation_policy_id IS NULL OR NEW.recommendation_policy_version IS NULL
    OR NEW.recommendation_policy_hash IS NULL OR NOT EXISTS(
      SELECT 1 FROM personalization_policies p
      WHERE p.policy_id=NEW.recommendation_policy_id
        AND p.version=NEW.recommendation_policy_version
        AND p.definition_hash=NEW.recommendation_policy_hash AND p.status='active'
    )
  )
) OR (
  NEW.recommendation_use_enabled=0 AND (
    NEW.recommendation_policy_id IS NOT NULL OR NEW.recommendation_policy_version IS NOT NULL
    OR NEW.recommendation_policy_hash IS NOT NULL
  )
)
BEGIN SELECT RAISE(ABORT,'invalid family personalization policy binding'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_control_epoch_update
BEFORE UPDATE OF collection_enabled,recommendation_use_enabled,recommendation_policy_id,
  recommendation_policy_version,recommendation_policy_hash,personalization_epoch
ON learner_model_controls
WHEN (
  (
    OLD.collection_enabled IS NOT NEW.collection_enabled
    OR OLD.recommendation_use_enabled IS NOT NEW.recommendation_use_enabled
    OR OLD.recommendation_policy_id IS NOT NEW.recommendation_policy_id
    OR OLD.recommendation_policy_version IS NOT NEW.recommendation_policy_version
    OR OLD.recommendation_policy_hash IS NOT NEW.recommendation_policy_hash
  ) AND NEW.personalization_epoch!=OLD.personalization_epoch+1
) OR (
  OLD.collection_enabled IS NEW.collection_enabled
  AND OLD.recommendation_use_enabled IS NEW.recommendation_use_enabled
  AND OLD.recommendation_policy_id IS NEW.recommendation_policy_id
  AND OLD.recommendation_policy_version IS NEW.recommendation_policy_version
  AND OLD.recommendation_policy_hash IS NEW.recommendation_policy_hash
  AND NEW.personalization_epoch!=OLD.personalization_epoch
)
BEGIN SELECT RAISE(ABORT,'family personalization authorization epoch mismatch'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_evaluation_immutable
BEFORE UPDATE ON personalization_evaluations
BEGIN SELECT RAISE(ABORT,'personalization evaluation is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_evaluation_delete
BEFORE DELETE ON personalization_evaluations
WHEN EXISTS(
  SELECT 1 FROM personalization_policies p WHERE p.evaluation_id=OLD.evaluation_id
)
BEGIN SELECT RAISE(ABORT,'referenced personalization evaluation cannot be deleted'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_promotion_review_immutable
BEFORE UPDATE ON personalization_promotion_reviews
BEGIN SELECT RAISE(ABORT,'personalization promotion review is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_promotion_review_delete
BEFORE DELETE ON personalization_promotion_reviews
WHEN EXISTS(
  SELECT 1 FROM personalization_policies p WHERE p.promotion_review_id=OLD.review_id
) OR EXISTS(
  SELECT 1 FROM personalization_decisions d WHERE d.promotion_review_id=OLD.review_id
)
BEGIN SELECT RAISE(ABORT,'referenced personalization promotion review cannot be deleted'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_decision_immutable
BEFORE UPDATE ON personalization_decisions
BEGIN SELECT RAISE(ABORT,'personalization decision is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_decision_binding_insert
BEFORE INSERT ON personalization_decisions
WHEN NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p JOIN events e ON e.id=NEW.event_id
  WHERE p.plan_id=NEW.plan_id AND p.plan_hash=NEW.plan_hash
    AND p.event_id=NEW.event_id AND p.child_id=NEW.child_id AND e.child_id=NEW.child_id
    AND p.workflow=NEW.workflow AND e.type=NEW.event_type
) OR NOT EXISTS (
  SELECT 1 FROM personalization_policies p
  WHERE p.policy_id=NEW.policy_id AND p.version=NEW.policy_version
    AND p.definition_hash=NEW.definition_hash AND p.evaluation_id IS NEW.evaluation_id
    AND p.authorization_epoch=NEW.policy_authorization_epoch
    AND (
      NEW.mode='shadow'
      OR (p.status='active' AND p.promotion_review_id=NEW.promotion_review_id)
    )
) OR (
  json_type(NEW.concept_refs_json)!='array'
  OR json_array_length(NEW.concept_refs_json)<1
  OR json_array_length(NEW.concept_refs_json)>3
  OR NOT EXISTS (
    SELECT 1 FROM pedagogy_plans p
    WHERE p.plan_id=NEW.plan_id AND p.plan_hash=NEW.plan_hash
      AND json_array_length(json_extract(p.approved_json,'$.concepts'))
          =json_array_length(NEW.concept_refs_json)
      AND NOT EXISTS (
        SELECT 1 FROM json_each(NEW.concept_refs_json) ref
        WHERE NOT EXISTS (
          SELECT 1 FROM json_each(p.approved_json,'$.concepts') concept
          WHERE json_extract(concept.value,'$.id')=json_extract(ref.value,'$.id')
            AND json_extract(concept.value,'$.version')=json_extract(ref.value,'$.version')
        )
      )
  )
) OR (
  NEW.mode='active' AND NOT EXISTS (
    SELECT 1 FROM learner_model_controls c
    WHERE c.child_id=NEW.child_id AND c.collection_enabled=1
      AND c.recommendation_use_enabled=1
      AND c.recommendation_policy_id=NEW.policy_id
      AND c.recommendation_policy_version=NEW.policy_version
      AND c.recommendation_policy_hash=NEW.definition_hash
      AND c.personalization_epoch=NEW.family_authorization_epoch
  )
) OR (
  NEW.mode='active' AND NOT EXISTS (
    SELECT 1 FROM pedagogy_plans p
    WHERE p.plan_id=NEW.plan_id
      AND json_extract(p.approved_json,'$.personalization.decision_id')=NEW.decision_id
      AND json_extract(p.approved_json,'$.personalization.decision_hash')=NEW.decision_hash
      AND json_extract(p.approved_json,'$.personalization.child_id')=NEW.child_id
      AND json_extract(p.approved_json,'$.personalization.event_id')=NEW.event_id
      AND json_extract(p.approved_json,'$.personalization.workflow')=NEW.workflow
      AND json_extract(p.approved_json,'$.personalization.proposal_hash')=NEW.proposal_hash
      AND json_extract(p.approved_json,'$.personalization.baseline_plan_id')=NEW.baseline_plan_id
      AND json_extract(p.approved_json,'$.personalization.baseline_plan_hash')=NEW.baseline_plan_hash
      AND canonical_json_text(json_extract(p.approved_json,'$.personalization.concept_refs'))
          =NEW.concept_refs_json
      AND json_extract(p.approved_json,'$.personalization.question_intent')=NEW.question_intent
      AND json_extract(p.approved_json,'$.personalization.family_authorization_epoch')=NEW.family_authorization_epoch
      AND json_extract(p.approved_json,'$.personalization.policy_authorization_epoch')=NEW.policy_authorization_epoch
      AND json_extract(p.approved_json,'$.personalization.evaluation_id')=NEW.evaluation_id
      AND json_extract(p.approved_json,'$.personalization.promotion_review_id')=NEW.promotion_review_id
      AND json_extract(p.approved_json,'$.question_intent') NOT IN ('direct_answer','sensitive_support')
      AND json_extract(p.approved_json,'$.primary_strategy.id')=json_extract(NEW.baseline_json,'$.strategy_id')
      AND json_extract(p.approved_json,'$.challenge_assumption')=json_extract(NEW.candidate_json,'$.challenge_assumption')
      AND json_extract(p.approved_json,'$.interest_support')=json_extract(NEW.candidate_json,'$.interest_support')
  )
) OR (
  NEW.mode='shadow' AND NOT EXISTS (
    SELECT 1 FROM pedagogy_plans p
    WHERE p.plan_id=NEW.baseline_plan_id AND p.plan_hash=NEW.baseline_plan_hash
      AND p.plan_id=NEW.plan_id
      AND json_extract(p.approved_json,'$.personalization') IS NULL
      AND json_extract(p.approved_json,'$.question_intent')=NEW.question_intent
      AND json_extract(p.approved_json,'$.challenge_assumption')='grade_baseline_uncalibrated'
      AND json_extract(p.approved_json,'$.interest_support')='neutral'
  )
)
BEGIN SELECT RAISE(ABORT,'invalid personalization decision binding'); END;

CREATE TRIGGER IF NOT EXISTS delete_personalization_decision_with_plan
BEFORE DELETE ON pedagogy_plans
BEGIN DELETE FROM personalization_decisions WHERE plan_id=OLD.plan_id; END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_signal_immutable
BEFORE UPDATE ON personalization_feedback_signals
BEGIN SELECT RAISE(ABORT,'personalization feedback signal is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_signal_binding_insert
BEFORE INSERT ON personalization_feedback_signals
WHEN NOT EXISTS (
  SELECT 1 FROM feedback f
  WHERE f.id=NEW.feedback_id AND f.child_id=NEW.child_id
    AND f.outcome=NEW.outcome
    AND f.pedagogy_plan_id IS NEW.plan_id AND f.pedagogy_plan_hash IS NEW.plan_hash
) OR (NEW.plan_id IS NOT NULL AND NOT EXISTS (
  SELECT 1 FROM pedagogy_plans p
  WHERE p.plan_id=NEW.plan_id AND p.plan_hash=NEW.plan_hash AND p.child_id=NEW.child_id
))
BEGIN SELECT RAISE(ABORT,'invalid personalization feedback binding'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_replay_immutable
BEFORE UPDATE ON personalization_replay_cases
BEGIN SELECT RAISE(ABORT,'personalization replay case is immutable'); END;

CREATE TRIGGER IF NOT EXISTS guard_personalization_replay_insert
BEFORE INSERT ON personalization_replay_cases
WHEN json_type(NEW.feature_vector_json)!='object'
  OR canonical_json_text(NEW.feature_vector_json)!=NEW.feature_vector_json
  OR (SELECT COUNT(*) FROM json_each(NEW.feature_vector_json))!=8
  OR EXISTS(
    SELECT 1 FROM json_each(NEW.feature_vector_json)
    WHERE key NOT IN (
      'question_intent','baseline_strategy_id','domain_overlay_id','prior_feedback_band',
      'recent_challenge_signal','knowledge_band','interest_band','response_mode'
    ) OR type!='text'
  )
  OR json_extract(NEW.feature_vector_json,'$.question_intent') NOT IN (
    'direct_answer','answer_and_wonder','guided_inquiry','compare_connect','practice_with_scaffolds',
    'dialogic_reading','teach_back','retrieval_revisit','make_design_iterate','sensitive_support'
  )
  OR json_extract(NEW.feature_vector_json,'$.baseline_strategy_id') NOT IN (
    'answer-and-wonder-k2','guided-inquiry-k2','compare-connect-k2','model-practice-fade-k2',
    'dialogic-reading-k2','teach-back-k2','retrieval-revisit-k2','make-design-iterate-k2'
  )
  OR json_extract(NEW.feature_vector_json,'$.domain_overlay_id') NOT IN (
    'general-learning-k2','early-science-k2','early-math-k2','early-literacy-k2',
    'early-writing-k2','source-sense-k2','sensitive-support-k2'
  )
  OR json_extract(NEW.feature_vector_json,'$.prior_feedback_band') NOT IN ('none','one','multiple')
  OR json_extract(NEW.feature_vector_json,'$.recent_challenge_signal') NOT IN (
    'none','positive','too_easy','too_hard','mixed'
  )
  OR json_extract(NEW.feature_vector_json,'$.knowledge_band') NOT IN (
    'not_observed','exposure_recorded','practicing_with_support','demonstrated_in_context',
    'independently_demonstrated','transfer_observed'
  )
  OR json_extract(NEW.feature_vector_json,'$.interest_band') NOT IN (
    'not_observed','triggered_situational','maintained_situational','emerging_recurrent',
    'sustained_parent_confirmed'
  )
  OR json_extract(NEW.feature_vector_json,'$.response_mode') NOT IN (
    'none','multiple','spoken','written','drawn','gesture_or_manipulative','selection',
    'demonstration','symbolic'
  )
  OR json_type(NEW.outcome_json)!='object'
  OR canonical_json_text(NEW.outcome_json)!=NEW.outcome_json
  OR (SELECT COUNT(*) FROM json_each(NEW.outcome_json))!=4
  OR EXISTS(
    SELECT 1 FROM json_each(NEW.outcome_json)
    WHERE key NOT IN (
      'baseline_utility','candidate_utility','baseline_calibration_error',
      'candidate_calibration_error'
    ) OR type NOT IN ('integer','real') OR value<0 OR value>1
  )
  OR (NEW.source_kind='public_fixture' AND (
    NEW.license_status!='public' OR NEW.privacy_approval_id IS NOT NULL
  ))
  OR (NEW.source_kind='synthetic' AND (
    NEW.license_status!='synthetic' OR NEW.privacy_approval_id IS NOT NULL
  ))
  OR (NEW.source_kind='approved_deidentified' AND (
    NEW.license_status!='approved' OR NEW.privacy_approval_id IS NULL
  ))
  OR NEW.case_hash!=personalization_replay_hash(
    NEW.case_id,NEW.source_kind,NEW.license_status,NEW.privacy_approval_id,
    NEW.feature_schema_version,NEW.feature_vector_json,NEW.outcome_json,NEW.source_digest
  )
BEGIN SELECT RAISE(ABORT,'invalid personalization replay case'); END;

INSERT INTO schema_migrations(version,name,applied_at)
VALUES(25,'evaluated personalization',strftime('%Y-%m-%dT%H:%M:%fZ','now'))
ON CONFLICT(version) DO NOTHING;
COMMIT;"""
        )
        definition_json = jdump(PERSONALIZATION_POLICY_DEFINITION)
        now = utcnow()
        conn.execute(
            """INSERT OR IGNORE INTO personalization_policies(
                 policy_id,version,feature_schema_version,definition_json,definition_hash,status,
                 created_at,updated_at
               ) VALUES(?,?,?,?,?,'candidate',?,?)""",
            (
                PERSONALIZATION_POLICY_ID,
                PERSONALIZATION_POLICY_VERSION,
                PERSONALIZATION_FEATURE_SCHEMA_VERSION,
                definition_json,
                PERSONALIZATION_POLICY_DEFINITION_HASH,
                now,
                now,
            ),
        )
        conn.execute(f"PRAGMA user_version = {max(current, 25)}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    required = {
        "personalization_policies",
        "personalization_control_events",
        "personalization_policy_events",
        "personalization_evaluations",
        "personalization_promotion_reviews",
        "personalization_decisions",
        "personalization_feedback_signals",
        "personalization_replay_cases",
    }
    required_columns = {
        "personalization_control_events": {
            "event_id",
            "child_id",
            "enabled",
            "policy_id",
            "policy_version",
            "definition_hash",
            "authorization_epoch",
            "actor_type",
            "reason",
            "created_at",
        },
        "personalization_policy_events": {
            "event_id",
            "policy_id",
            "policy_version",
            "action",
            "actor_type",
            "operator_id",
            "reason",
            "created_at",
        },
        "personalization_policies": {
            "policy_id",
            "version",
            "feature_schema_version",
            "definition_json",
            "definition_hash",
            "authorization_epoch",
            "status",
            "evaluation_id",
            "promotion_review_id",
            "approved_by_operator",
            "approval_reason",
            "activation_reason",
            "approved_at",
            "activated_at",
            "created_at",
            "updated_at",
        },
        "personalization_evaluations": {
            "evaluation_id",
            "policy_id",
            "policy_version",
            "definition_hash",
            "suite_version",
            "fixture_digest",
            "case_count",
            "benefit_delta",
            "baseline_calibration_error",
            "candidate_calibration_error",
            "safety_pass",
            "privacy_pass",
            "fairness_pass",
            "golden_pass",
            "status",
            "evaluator",
            "evaluator_digest",
            "results_json",
            "results_hash",
            "created_at",
        },
        "personalization_promotion_reviews": {
            "review_id",
            "policy_id",
            "policy_version",
            "definition_hash",
            "evaluation_id",
            "reviewer_id",
            "lab_report_json",
            "lab_report_hash",
            "created_at",
        },
        "personalization_decisions": {
            "decision_id",
            "child_id",
            "event_id",
            "event_type",
            "event_text_hash",
            "workflow",
            "proposal_hash",
            "baseline_plan_id",
            "baseline_plan_hash",
            "concept_ids_json",
            "concept_refs_json",
            "question_intent",
            "plan_id",
            "plan_hash",
            "policy_id",
            "policy_version",
            "definition_hash",
            "family_authorization_epoch",
            "policy_authorization_epoch",
            "evaluation_id",
            "promotion_review_id",
            "mode",
            "feature_vector_json",
            "feature_vector_hash",
            "baseline_json",
            "candidate_json",
            "decision_hash",
            "created_at",
        },
        "personalization_feedback_signals": {
            "signal_id",
            "feedback_id",
            "child_id",
            "plan_id",
            "plan_hash",
            "outcome",
            "used",
            "chose_to_continue",
            "made_prediction",
            "gave_explanation",
            "help_level",
            "asked_new_question",
            "parent_correction",
            "created_at",
        },
        "personalization_replay_cases": {
            "case_id",
            "source_kind",
            "license_status",
            "privacy_approval_id",
            "feature_schema_version",
            "feature_vector_json",
            "outcome_json",
            "source_digest",
            "case_hash",
            "created_by_operator",
            "created_at",
        },
    }
    missing_columns = {
        table: sorted(columns - {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")})
        for table, columns in required_columns.items()
        if table in _tables(conn)
    }
    missing_columns = {table: columns for table, columns in missing_columns.items() if columns}
    required_triggers = {
        "guard_personalization_policy_definition_immutable",
        "guard_personalization_policy_promotion_update",
        "guard_personalization_control_immutable",
        "guard_personalization_policy_event_immutable",
        "guard_personalization_control_binding_insert",
        "guard_personalization_control_binding_update",
        "guard_personalization_control_epoch_update",
        "guard_personalization_evaluation_immutable",
        "guard_personalization_evaluation_delete",
        "guard_personalization_promotion_review_immutable",
        "guard_personalization_promotion_review_delete",
        "guard_personalization_decision_immutable",
        "guard_personalization_decision_binding_insert",
        "delete_personalization_decision_with_plan",
        "guard_personalization_signal_immutable",
        "guard_personalization_signal_binding_insert",
        "guard_personalization_replay_immutable",
        "guard_personalization_replay_insert",
        "guard_personalization_policy_epoch_update",
    }
    triggers = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    required_indexes = {
        "idx_personalization_one_active",
        "idx_personalization_control_child",
        "idx_personalization_decision_child",
        "idx_personalization_signal_child",
    }
    indexes = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    table_sql = {
        table: re.sub(
            r"\s+",
            "",
            str(
                conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
                ).fetchone()[0]
            ).casefold(),
        )
        for table in required & _tables(conn)
    }
    required_fragments = {
        "personalization_control_events": [
            "check(enabledin(0,1))",
            "check(definition_hashisnullorlength(definition_hash)=64)",
            "check(authorization_epoch>0)",
        ],
        "personalization_policies": ["check(version>0)", "check(length(definition_hash)=64)", "check(authorization_epoch>=0)", "'candidate','evaluated','approved','active','disabled','retired'"],
        "personalization_evaluations": ["check(case_count>0)", "check(length(results_hash)=64)", "check(length(evaluator_digest)=64)", "check(safety_passin(0,1))"],
        "personalization_promotion_reviews": ["check(length(definition_hash)=64)", "check(length(lab_report_hash)=64)", "check(length(reviewer_id)between1and160)"],
        "personalization_decisions": ["check(modein('shadow','active'))", "check(length(decision_hash)=64)", "check(length(feature_vector_hash)=64)", "check(length(event_text_hash)=64)", "check(length(proposal_hash)=64)", "check(json_valid(concept_refs_json))", "check(family_authorization_epoch>=0)", "check(policy_authorization_epoch>=0)"],
        "personalization_feedback_signals": ["check(usedin(0,1))", "'too_easy','too_hard'", "'worked_example','joint_completion'"],
        "personalization_replay_cases": ["'public_fixture','synthetic','approved_deidentified'", "check(feature_schema_version='phase5-v2')", "check(json_valid(feature_vector_json))", "check(json_valid(outcome_json))", "case_hashnotglob'*[^0-9a-f]*'"],
    }
    missing_fragments = {
        table: [fragment for fragment in fragments if fragment not in table_sql.get(table, "")]
        for table, fragments in required_fragments.items()
    }
    missing_fragments = {table: fragments for table, fragments in missing_fragments.items() if fragments}

    foreign_key = lambda table, on_delete, *columns: (  # noqa: E731
        table,
        on_delete,
        tuple(columns),
    )
    required_foreign_keys = {
        "personalization_control_events": {
            foreign_key("children", "CASCADE", ("child_id", "id")),
        },
        "personalization_policy_events": {
            foreign_key(
                "personalization_policies",
                "CASCADE",
                ("policy_id", "policy_id"),
                ("policy_version", "version"),
            ),
        },
        "personalization_evaluations": {
            foreign_key(
                "personalization_policies",
                "CASCADE",
                ("policy_id", "policy_id"),
                ("policy_version", "version"),
            ),
        },
        "personalization_promotion_reviews": {
            foreign_key(
                "personalization_policies",
                "CASCADE",
                ("policy_id", "policy_id"),
                ("policy_version", "version"),
            ),
            foreign_key("personalization_evaluations", "RESTRICT", ("evaluation_id", "evaluation_id")),
        },
        "personalization_decisions": {
            foreign_key("children", "CASCADE", ("child_id", "id")),
            foreign_key("events", "CASCADE", ("event_id", "id")),
            foreign_key(
                "personalization_policies",
                "RESTRICT",
                ("policy_id", "policy_id"),
                ("policy_version", "version"),
            ),
            foreign_key("personalization_evaluations", "RESTRICT", ("evaluation_id", "evaluation_id")),
            foreign_key(
                "personalization_promotion_reviews",
                "RESTRICT",
                ("promotion_review_id", "review_id"),
            ),
        },
        "personalization_feedback_signals": {
            foreign_key("feedback", "CASCADE", ("feedback_id", "id")),
            foreign_key("children", "CASCADE", ("child_id", "id")),
        },
    }
    missing_foreign_keys: dict[str, list[tuple[Any, ...]]] = {}
    for table, expected in required_foreign_keys.items():
        grouped: dict[int, list[sqlite3.Row]] = {}
        for item in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            grouped.setdefault(int(item[0]), []).append(item)
        present = {
            (
                str(items[0][2]),
                str(items[0][6]).upper(),
                tuple(
                    (str(item[3]), str(item[4]))
                    for item in sorted(items, key=lambda value: int(value[1]))
                ),
            )
            for items in grouped.values()
        }
        missing = sorted(expected - present)
        if missing:
            missing_foreign_keys[table] = missing

    required_primary_keys = {
        "personalization_policies": ("policy_id", "version"),
        "personalization_control_events": ("event_id",),
        "personalization_policy_events": ("event_id",),
        "personalization_evaluations": ("evaluation_id",),
        "personalization_promotion_reviews": ("review_id",),
        "personalization_decisions": ("decision_id",),
        "personalization_feedback_signals": ("signal_id",),
        "personalization_replay_cases": ("case_id",),
    }
    invalid_primary_keys = {}
    for table, expected in required_primary_keys.items():
        present = tuple(
            str(row[1])
            for row in sorted(
                conn.execute(f'PRAGMA table_info("{table}")').fetchall(),
                key=lambda value: int(value[5]) if int(value[5]) else 10_000,
            )
            if int(row[5])
        )
        if present != expected:
            invalid_primary_keys[table] = {"expected": expected, "present": present}

    required_not_null = {
        "personalization_policies": {
            "policy_id", "version", "feature_schema_version", "definition_json", "definition_hash",
            "authorization_epoch", "status", "created_at", "updated_at",
        },
        "personalization_control_events": {
            "event_id", "child_id", "enabled", "authorization_epoch", "actor_type", "reason", "created_at",
        },
        "personalization_policy_events": {
            "event_id", "policy_id", "policy_version", "action", "actor_type", "reason", "created_at",
        },
        "personalization_evaluations": set(required_columns["personalization_evaluations"]),
        "personalization_promotion_reviews": set(required_columns["personalization_promotion_reviews"]),
        "personalization_decisions": set(required_columns["personalization_decisions"]) - {
            "evaluation_id", "promotion_review_id"
        },
        "personalization_feedback_signals": {
            "signal_id", "feedback_id", "child_id", "outcome", "created_at",
        },
        "personalization_replay_cases": set(required_columns["personalization_replay_cases"]) - {"privacy_approval_id"},
    }
    invalid_not_null = {
        table: sorted(
            column
            for column in columns
            if not next(
                (bool(row[3]) for row in conn.execute(f'PRAGMA table_info("{table}")') if str(row[1]) == column),
                False,
            )
        )
        for table, columns in required_not_null.items()
    }
    invalid_not_null = {table: columns for table, columns in invalid_not_null.items() if columns}

    required_unique_shapes = {
        "personalization_decisions": {
            ("decision_hash",),
            ("event_id", "policy_id", "policy_version"),
        },
        "personalization_feedback_signals": {("feedback_id",)},
        "personalization_promotion_reviews": {("lab_report_hash",)},
        "personalization_replay_cases": {("case_hash",)},
    }
    missing_unique_constraints: dict[str, list[tuple[str, ...]]] = {}
    for table, expected in required_unique_shapes.items():
        present = {
            tuple(
                str(info[2])
                for info in sorted(
                    conn.execute(f'PRAGMA index_info("{row[1]}")').fetchall(),
                    key=lambda value: int(value[0]),
                )
            )
            for row in conn.execute(f'PRAGMA index_list("{table}")')
            if bool(row[2]) and not bool(row[4])
        }
        missing = sorted(expected - present)
        if missing:
            missing_unique_constraints[table] = missing

    def index_shape(table: str, name: str) -> tuple[tuple[str, ...], bool, bool] | None:
        index = next(
            (row for row in conn.execute(f'PRAGMA index_list("{table}")') if str(row[1]) == name),
            None,
        )
        if index is None:
            return None
        columns = tuple(
            str(row[2])
            for row in sorted(
                conn.execute(f'PRAGMA index_info("{name}")').fetchall(),
                key=lambda value: int(value[0]),
            )
        )
        return columns, bool(index[2]), bool(index[4])

    required_index_shapes = {
        ("personalization_policies", "idx_personalization_one_active"): (
            ("None",), True, True,
        ),
        ("personalization_control_events", "idx_personalization_control_child"): (
            ("child_id", "created_at"), False, False,
        ),
        ("personalization_decisions", "idx_personalization_decision_child"): (
            ("child_id", "created_at"), False, False,
        ),
        ("personalization_feedback_signals", "idx_personalization_signal_child"): (
            ("child_id", "created_at"), False, False,
        ),
    }
    invalid_indexes = {
        name: {"expected": expected, "present": index_shape(table, name)}
        for (table, name), expected in required_index_shapes.items()
        if index_shape(table, name) != expected
    }
    invalid_hash_rows = any(
        conn.execute(
            f"SELECT 1 FROM {table} WHERE {column} IS NULL OR length({column})!=64 LIMIT 1"
        ).fetchone()
        for table, column in (
            ("personalization_policies", "definition_hash"),
            ("personalization_evaluations", "definition_hash"),
            ("personalization_evaluations", "fixture_digest"),
            ("personalization_evaluations", "results_hash"),
            ("personalization_evaluations", "evaluator_digest"),
            ("personalization_promotion_reviews", "definition_hash"),
            ("personalization_promotion_reviews", "lab_report_hash"),
            ("personalization_decisions", "decision_hash"),
            ("personalization_decisions", "event_text_hash"),
            ("personalization_decisions", "proposal_hash"),
            ("personalization_decisions", "baseline_plan_hash"),
            ("personalization_decisions", "feature_vector_hash"),
            ("personalization_replay_cases", "source_digest"),
            ("personalization_replay_cases", "case_hash"),
        )
    )
    invalid_replay_rows: list[str] = []
    for row in conn.execute("SELECT * FROM personalization_replay_cases ORDER BY case_id"):
        try:
            feature_vector = json.loads(str(row["feature_vector_json"]))
            outcome = json.loads(str(row["outcome_json"]))
            payload = personalization_replay_case_payload(
                case_id=str(row["case_id"]),
                source_kind=str(row["source_kind"]),
                license_status=str(row["license_status"]),
                privacy_approval_id=(
                    str(row["privacy_approval_id"])
                    if row["privacy_approval_id"] is not None
                    else None
                ),
                feature_schema_version=str(row["feature_schema_version"]),
                feature_vector=feature_vector,
                outcome=outcome,
                source_digest=str(row["source_digest"]),
            )
            source_kind = str(row["source_kind"])
            expected_license = {
                "public_fixture": "public",
                "synthetic": "synthetic",
                "approved_deidentified": "approved",
            }.get(source_kind)
            privacy_valid = (
                bool(row["privacy_approval_id"])
                if source_kind == "approved_deidentified"
                else row["privacy_approval_id"] is None
            )
            valid = all(
                (
                    str(row["feature_schema_version"])
                    == PERSONALIZATION_FEATURE_SCHEMA_VERSION,
                    expected_license == str(row["license_status"]),
                    privacy_valid,
                    jdump(feature_vector) == str(row["feature_vector_json"]),
                    jdump(outcome) == str(row["outcome_json"]),
                    sha256(jdump(payload).encode()).hexdigest() == str(row["case_hash"]),
                    bool(re.fullmatch(r"[0-9a-f]{64}", str(row["source_digest"]))),
                    bool(re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", str(row["created_by_operator"]))),
                    row["privacy_approval_id"] is None
                    or bool(
                        re.fullmatch(
                            r"[A-Za-z0-9_.:-]{1,160}",
                            str(row["privacy_approval_id"]),
                        )
                    ),
                )
            )
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalid_replay_rows.append(str(row["case_id"]))
    policy = conn.execute(
        """SELECT feature_schema_version,definition_json,definition_hash
           FROM personalization_policies WHERE policy_id=? AND version=?""",
        (PERSONALIZATION_POLICY_ID, PERSONALIZATION_POLICY_VERSION),
    ).fetchone()
    control_info = {
        str(row[1]): row for row in conn.execute("PRAGMA table_info(learner_model_controls)")
    }
    controls = set(control_info)
    epoch_column = control_info.get("personalization_epoch")
    control_sql = re.sub(
        r"\s+",
        "",
        str(
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='learner_model_controls'"
            ).fetchone()[0]
        ).casefold(),
    )
    invalid_control_epoch = bool(
        not epoch_column
        or not bool(epoch_column[3])
        or str(epoch_column[4]).strip("'\"") != "0"
        or "check(personalization_epoch>=0)" not in control_sql
    )
    if (
        required - _tables(conn)
        or missing_columns
        or required_triggers - triggers
        or required_indexes - indexes
        or missing_fragments
        or missing_foreign_keys
        or invalid_primary_keys
        or invalid_not_null
        or invalid_indexes
        or missing_unique_constraints
        or invalid_hash_rows
        or invalid_replay_rows
        or invalid_control_epoch
        or {
            "recommendation_use_enabled",
            "recommendation_policy_id",
            "recommendation_policy_version",
            "recommendation_policy_hash",
            "personalization_epoch",
        }
        - controls
        or not policy
        or str(policy["feature_schema_version"]) != PERSONALIZATION_FEATURE_SCHEMA_VERSION
        or str(policy["definition_json"]) != jdump(PERSONALIZATION_POLICY_DEFINITION)
        or str(policy["definition_hash"]) != PERSONALIZATION_POLICY_DEFINITION_HASH
        or sha256(str(policy["definition_json"]).encode()).hexdigest() != str(policy["definition_hash"])
        or not conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version=25 AND name='evaluated personalization'"
        ).fetchone()
        or int(conn.execute("PRAGMA user_version").fetchone()[0]) != max(current, 25)
        or conn.execute("PRAGMA foreign_key_check").fetchall()
    ):
        raise RuntimeError(
            "incomplete schema v25 evaluated-personalization migration: "
            f"missing_tables={sorted(required - _tables(conn))!r}, "
            f"missing_columns={missing_columns!r}, "
            f"missing_foreign_keys={missing_foreign_keys!r}, "
            f"invalid_primary_keys={invalid_primary_keys!r}, "
            f"invalid_not_null={invalid_not_null!r}, "
            f"invalid_indexes={invalid_indexes!r}, "
            f"missing_unique_constraints={missing_unique_constraints!r}, "
            f"missing_check_fragments={missing_fragments!r}"
            f", invalid_replay_rows={invalid_replay_rows!r}"
            f", invalid_control_epoch={invalid_control_epoch!r}"
        )


def _apply_migration_26(conn: sqlite3.Connection, repo_root: Path | None = None) -> None:
    """Add a gated, evidence-linked uncertainty and proactivity research plane."""

    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    try:
        conn.executescript(
            """BEGIN IMMEDIATE;
DROP TRIGGER IF EXISTS guard_research_policy_definition_immutable;
DROP TRIGGER IF EXISTS guard_research_policy_insert;
DROP TRIGGER IF EXISTS guard_research_policy_authorized_update;
DROP TRIGGER IF EXISTS guard_research_policy_transition;
DROP TRIGGER IF EXISTS guard_research_policy_promotion_update;
DROP TRIGGER IF EXISTS guard_research_policy_epoch_update;
DROP TRIGGER IF EXISTS guard_research_calibration_case_insert;
DROP TRIGGER IF EXISTS guard_research_calibration_case_update;
DROP TRIGGER IF EXISTS guard_research_calibration_case_delete;
DROP TRIGGER IF EXISTS guard_research_corpus_insert;
DROP TRIGGER IF EXISTS guard_research_corpus_update;
DROP TRIGGER IF EXISTS guard_research_corpus_delete;
DROP TRIGGER IF EXISTS guard_research_evaluation_insert;
DROP TRIGGER IF EXISTS guard_research_evaluation_immutable;
DROP TRIGGER IF EXISTS guard_research_evaluation_delete;
DROP TRIGGER IF EXISTS guard_research_review_immutable;
DROP TRIGGER IF EXISTS guard_research_review_delete;
DROP TRIGGER IF EXISTS guard_research_review_insert;
DROP TRIGGER IF EXISTS guard_research_policy_event_immutable;
DROP TRIGGER IF EXISTS guard_research_policy_event_insert;
DROP TRIGGER IF EXISTS guard_research_policy_event_delete;
DROP TRIGGER IF EXISTS guard_research_control_binding_insert;
DROP TRIGGER IF EXISTS guard_research_control_binding_update;
DROP TRIGGER IF EXISTS guard_research_control_authorized_update;
DROP TRIGGER IF EXISTS guard_research_control_epoch_update;
DROP TRIGGER IF EXISTS guard_research_control_event_immutable;
DROP TRIGGER IF EXISTS guard_research_control_event_insert;
DROP TRIGGER IF EXISTS guard_research_control_event_delete;
DROP TRIGGER IF EXISTS guard_research_authorization_snapshot_insert;
DROP TRIGGER IF EXISTS guard_research_authorization_snapshot_immutable;
DROP TRIGGER IF EXISTS guard_research_authorization_snapshot_delete;
DROP TRIGGER IF EXISTS guard_research_trace_insert;
DROP TRIGGER IF EXISTS guard_research_trace_immutable;
DROP TRIGGER IF EXISTS delete_research_trace_with_evidence;
DROP TRIGGER IF EXISTS delete_research_trace_with_projection_evidence;
DROP TRIGGER IF EXISTS delete_research_trace_with_projection_change;
DROP TRIGGER IF EXISTS delete_research_trace_with_projection_delete;
DROP TRIGGER IF EXISTS guard_research_candidate_insert;
DROP TRIGGER IF EXISTS guard_research_candidate_update;
DROP TRIGGER IF EXISTS invalidate_research_candidates_on_optout;
DROP TRIGGER IF EXISTS invalidate_research_candidates_on_policy_stop;
DROP TRIGGER IF EXISTS invalidate_research_candidates_on_interest_change;
DROP TRIGGER IF EXISTS disable_research_on_collection_pause;
DROP VIEW IF EXISTS research_effective_knowledge_evidence;

CREATE TABLE IF NOT EXISTS research_track_policies (
  policy_id TEXT NOT NULL,
  version INTEGER NOT NULL CHECK(version > 0),
  feature_schema_version TEXT NOT NULL,
  definition_json TEXT NOT NULL CHECK(json_valid(definition_json)),
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  release_content_hash TEXT CHECK(
    release_content_hash IS NULL OR (
      length(release_content_hash)=64 AND release_content_hash NOT GLOB '*[^0-9a-f]*'
    )
  ),
  authorization_epoch INTEGER NOT NULL DEFAULT 0 CHECK(authorization_epoch >= 0),
  status TEXT NOT NULL CHECK(status IN (
    'candidate','evaluated','approved','active','disabled','retired'
  )),
  evaluation_id TEXT,
  promotion_review_id TEXT,
  approved_by_operator TEXT,
  approval_reason TEXT,
  activation_reason TEXT,
  approved_at TEXT,
  activated_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(policy_id,version)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_track_one_active
  ON research_track_policies((1)) WHERE status='active';

CREATE TABLE IF NOT EXISTS research_calibration_corpora (
  corpus_id TEXT PRIMARY KEY CHECK(
    length(corpus_id) BETWEEN 1 AND 160 AND corpus_id NOT GLOB '*[^A-Za-z0-9_.:-]*'
  ),
  feature_schema_version TEXT NOT NULL CHECK(feature_schema_version='phase6-v1'),
  manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json)),
  manifest_hash TEXT NOT NULL UNIQUE CHECK(
    length(manifest_hash)=64 AND manifest_hash NOT GLOB '*[^0-9a-f]*'
  ),
  case_count INTEGER NOT NULL CHECK(case_count > 0),
  status TEXT NOT NULL CHECK(status IN ('pending_review','approved','withdrawn')),
  proposed_by_operator TEXT NOT NULL CHECK(
    length(proposed_by_operator) BETWEEN 1 AND 160
    AND proposed_by_operator NOT GLOB '*[^A-Za-z0-9_.:-]*'
  ),
  approved_by_reviewer TEXT,
  approval_reason TEXT,
  approved_at TEXT,
  created_at TEXT NOT NULL,
  CHECK(
    (status='pending_review' AND approved_by_reviewer IS NULL AND approval_reason IS NULL AND approved_at IS NULL)
    OR (status='approved' AND approved_by_reviewer IS NOT NULL AND approval_reason IS NOT NULL AND approved_at IS NOT NULL)
    OR status='withdrawn'
  )
);

CREATE TABLE IF NOT EXISTS research_calibration_cases (
  case_id TEXT PRIMARY KEY CHECK(
    length(case_id) BETWEEN 1 AND 160 AND case_id NOT GLOB '*[^A-Za-z0-9_.:-]*'
  ),
  source_kind TEXT NOT NULL CHECK(source_kind IN ('public_fixture','synthetic','approved_deidentified')),
  license_status TEXT NOT NULL CHECK(license_status IN ('public','synthetic','approved')),
  privacy_approval_id TEXT CHECK(
    privacy_approval_id IS NULL OR (
      length(privacy_approval_id) BETWEEN 1 AND 160
      AND privacy_approval_id NOT GLOB '*[^A-Za-z0-9_.:-]*'
    )
  ),
  corpus_id TEXT,
  feature_schema_version TEXT NOT NULL CHECK(feature_schema_version='phase6-v1'),
  split TEXT NOT NULL CHECK(split IN ('train','heldout')),
  group_digest TEXT NOT NULL CHECK(length(group_digest)=64 AND group_digest NOT GLOB '*[^0-9a-f]*'),
  task_signature_digest TEXT NOT NULL CHECK(
    length(task_signature_digest)=64 AND task_signature_digest NOT GLOB '*[^0-9a-f]*'
  ),
  domain TEXT NOT NULL CHECK(domain IN ('general','science','math','literacy','writing','source_sense')),
  history_json TEXT NOT NULL CHECK(json_valid(history_json)),
  observed_success INTEGER NOT NULL CHECK(observed_success IN (0,1)),
  source_digest TEXT NOT NULL CHECK(length(source_digest)=64 AND source_digest NOT GLOB '*[^0-9a-f]*'),
  case_hash TEXT NOT NULL UNIQUE CHECK(length(case_hash)=64 AND case_hash NOT GLOB '*[^0-9a-f]*'),
  created_by_operator TEXT NOT NULL CHECK(
    length(created_by_operator) BETWEEN 1 AND 160
    AND created_by_operator NOT GLOB '*[^A-Za-z0-9_.:-]*'
  ),
  created_at TEXT NOT NULL
  ,FOREIGN KEY(corpus_id) REFERENCES research_calibration_corpora(corpus_id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_research_calibration_split
  ON research_calibration_cases(split,source_kind,domain);

CREATE TABLE IF NOT EXISTS research_track_evaluations (
  evaluation_id TEXT PRIMARY KEY,
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  release_content_hash TEXT NOT NULL CHECK(
    length(release_content_hash)=64 AND release_content_hash NOT GLOB '*[^0-9a-f]*'
  ),
  dataset_digest TEXT NOT NULL CHECK(length(dataset_digest)=64),
  case_count INTEGER NOT NULL CHECK(case_count >= 0),
  real_case_count INTEGER NOT NULL CHECK(real_case_count >= 0),
  train_count INTEGER NOT NULL CHECK(train_count >= 0),
  heldout_count INTEGER NOT NULL CHECK(heldout_count >= 0),
  distinct_task_count INTEGER NOT NULL CHECK(distinct_task_count >= 0),
  distinct_domain_count INTEGER NOT NULL CHECK(distinct_domain_count >= 0),
  heldout_positive_count INTEGER NOT NULL CHECK(heldout_positive_count >= 0),
  heldout_negative_count INTEGER NOT NULL CHECK(heldout_negative_count >= 0),
  split_overlap_count INTEGER NOT NULL CHECK(split_overlap_count >= 0),
  brier_score REAL NOT NULL CHECK(brier_score BETWEEN 0 AND 1),
  expected_calibration_error REAL NOT NULL CHECK(expected_calibration_error BETWEEN 0 AND 1),
  gates_json TEXT NOT NULL CHECK(json_valid(gates_json)),
  results_hash TEXT NOT NULL CHECK(length(results_hash)=64),
  evaluator TEXT NOT NULL,
  evaluator_digest TEXT NOT NULL CHECK(length(evaluator_digest)=64),
  status TEXT NOT NULL CHECK(status IN ('pass','fail')),
  created_at TEXT NOT NULL,
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES research_track_policies(policy_id,version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS research_track_promotion_reviews (
  review_id TEXT PRIMARY KEY,
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  release_content_hash TEXT NOT NULL CHECK(
    length(release_content_hash)=64 AND release_content_hash NOT GLOB '*[^0-9a-f]*'
  ),
  evaluation_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL CHECK(length(reviewer_id) BETWEEN 1 AND 160),
  lab_report_json TEXT NOT NULL CHECK(json_valid(lab_report_json)),
  lab_report_hash TEXT NOT NULL UNIQUE CHECK(length(lab_report_hash)=64),
  created_at TEXT NOT NULL,
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES research_track_policies(policy_id,version) ON DELETE CASCADE,
  FOREIGN KEY(evaluation_id) REFERENCES research_track_evaluations(evaluation_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS research_track_policy_events (
  event_id TEXT PRIMARY KEY,
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  action TEXT NOT NULL CHECK(action IN (
    'evaluated','promotion_reviewed','approved','activated','disabled','rolled_back'
  )),
  actor_type TEXT NOT NULL CHECK(actor_type IN ('system','operator')),
  operator_id TEXT,
  reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 500),
  created_at TEXT NOT NULL,
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES research_track_policies(policy_id,version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS research_track_controls (
  child_id TEXT PRIMARY KEY,
  enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
  policy_id TEXT,
  policy_version INTEGER CHECK(policy_version IS NULL OR policy_version > 0),
  definition_hash TEXT CHECK(definition_hash IS NULL OR length(definition_hash)=64),
  release_content_hash TEXT CHECK(
    release_content_hash IS NULL OR (
      length(release_content_hash)=64 AND release_content_hash NOT GLOB '*[^0-9a-f]*'
    )
  ),
  family_authorization_epoch INTEGER NOT NULL DEFAULT 0 CHECK(family_authorization_epoch >= 0),
  updated_by TEXT NOT NULL CHECK(updated_by IN ('parent','operator')),
  reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 500),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  CHECK(
    (enabled=1 AND policy_id IS NOT NULL AND policy_version IS NOT NULL
      AND definition_hash IS NOT NULL AND release_content_hash IS NOT NULL)
    OR (enabled=0 AND policy_id IS NULL AND policy_version IS NULL
      AND definition_hash IS NULL AND release_content_hash IS NULL)
  ),
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS research_track_control_events (
  event_id TEXT PRIMARY KEY,
  child_id TEXT NOT NULL,
  enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
  policy_id TEXT,
  policy_version INTEGER,
  definition_hash TEXT,
  release_content_hash TEXT,
  family_authorization_epoch INTEGER NOT NULL CHECK(family_authorization_epoch > 0),
  actor_type TEXT NOT NULL CHECK(actor_type IN ('parent','operator','system')),
  reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 500),
  created_at TEXT NOT NULL,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_research_control_events_child
  ON research_track_control_events(child_id,created_at DESC);

CREATE TABLE IF NOT EXISTS research_track_authorization_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  snapshot_hash TEXT NOT NULL UNIQUE CHECK(
    length(snapshot_hash)=64 AND snapshot_hash NOT GLOB '*[^0-9a-f]*'
  ),
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL CHECK(policy_version > 0),
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  release_content_hash TEXT NOT NULL CHECK(
    length(release_content_hash)=64 AND release_content_hash NOT GLOB '*[^0-9a-f]*'
  ),
  evaluation_id TEXT NOT NULL,
  promotion_review_id TEXT NOT NULL,
  policy_authorization_epoch INTEGER NOT NULL CHECK(
    policy_authorization_epoch > 0 AND policy_authorization_epoch % 2 = 1
  ),
  approved_by_operator TEXT NOT NULL,
  approval_reason TEXT NOT NULL,
  activation_reason TEXT NOT NULL,
  approved_at TEXT NOT NULL,
  activated_at TEXT NOT NULL,
  approval_event_id TEXT NOT NULL UNIQUE,
  activation_event_id TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  UNIQUE(policy_id,policy_version,policy_authorization_epoch),
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES research_track_policies(policy_id,version) ON DELETE RESTRICT,
  FOREIGN KEY(evaluation_id) REFERENCES research_track_evaluations(evaluation_id) ON DELETE RESTRICT,
  FOREIGN KEY(promotion_review_id)
    REFERENCES research_track_promotion_reviews(review_id) ON DELETE RESTRICT,
  FOREIGN KEY(approval_event_id) REFERENCES research_track_policy_events(event_id) ON DELETE RESTRICT,
  FOREIGN KEY(activation_event_id) REFERENCES research_track_policy_events(event_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS research_uncertainty_traces (
  trace_id TEXT PRIMARY KEY,
  trace_hash TEXT NOT NULL UNIQUE CHECK(length(trace_hash)=64),
  child_id TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  construct TEXT NOT NULL CHECK(construct='knowledge_performance'),
  projection_input_digest TEXT NOT NULL CHECK(length(projection_input_digest)=64),
  projection_rule_id TEXT NOT NULL,
  projection_rule_version TEXT NOT NULL,
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  release_content_hash TEXT NOT NULL CHECK(
    length(release_content_hash)=64 AND release_content_hash NOT GLOB '*[^0-9a-f]*'
  ),
  evaluation_id TEXT NOT NULL,
  promotion_review_id TEXT NOT NULL,
  authorization_snapshot_id TEXT NOT NULL,
  family_authorization_epoch INTEGER NOT NULL CHECK(family_authorization_epoch > 0),
  policy_authorization_epoch INTEGER NOT NULL CHECK(policy_authorization_epoch > 0),
  evidence_ids_json TEXT NOT NULL CHECK(json_valid(evidence_ids_json)),
  evidence_digest TEXT NOT NULL CHECK(length(evidence_digest)=64),
  history_json TEXT NOT NULL CHECK(json_valid(history_json)),
  alpha REAL NOT NULL CHECK(alpha > 0),
  beta REAL NOT NULL CHECK(beta > 0),
  posterior_mean REAL NOT NULL CHECK(posterior_mean BETWEEN 0 AND 1),
  variance REAL NOT NULL CHECK(variance BETWEEN 0 AND 1),
  interval_90_lower REAL NOT NULL CHECK(interval_90_lower BETWEEN 0 AND 1),
  interval_90_upper REAL NOT NULL CHECK(interval_90_upper BETWEEN 0 AND 1),
  uncertainty REAL NOT NULL CHECK(uncertainty BETWEEN 0 AND 1),
  method_version TEXT NOT NULL,
  advisory_only INTEGER NOT NULL CHECK(advisory_only=1),
  rationale TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(child_id,concept_id,concept_version)
    REFERENCES learner_state_projections(child_id,concept_id,concept_version) ON DELETE CASCADE,
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES research_track_policies(policy_id,version) ON DELETE RESTRICT,
  FOREIGN KEY(evaluation_id) REFERENCES research_track_evaluations(evaluation_id) ON DELETE RESTRICT,
  FOREIGN KEY(promotion_review_id) REFERENCES research_track_promotion_reviews(review_id) ON DELETE RESTRICT,
  FOREIGN KEY(authorization_snapshot_id)
    REFERENCES research_track_authorization_snapshots(snapshot_id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_research_trace_child
  ON research_uncertainty_traces(child_id,concept_id,concept_version,created_at DESC);

CREATE TABLE IF NOT EXISTS research_proactivity_candidates (
  candidate_id TEXT PRIMARY KEY,
  candidate_hash TEXT NOT NULL UNIQUE CHECK(length(candidate_hash)=64),
  child_id TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  concept_id TEXT NOT NULL,
  concept_version INTEGER NOT NULL,
  interest_input_digest TEXT NOT NULL CHECK(
    length(interest_input_digest)=64 AND interest_input_digest NOT GLOB '*[^0-9a-f]*'
  ),
  interest_rule_id TEXT NOT NULL,
  interest_rule_version TEXT NOT NULL,
  interest_phase TEXT NOT NULL CHECK(interest_phase IN (
    'maintained_situational','emerging_recurrent','sustained_parent_confirmed'
  )),
  policy_id TEXT NOT NULL,
  policy_version INTEGER NOT NULL,
  definition_hash TEXT NOT NULL CHECK(length(definition_hash)=64),
  release_content_hash TEXT NOT NULL CHECK(
    length(release_content_hash)=64 AND release_content_hash NOT GLOB '*[^0-9a-f]*'
  ),
  family_authorization_epoch INTEGER NOT NULL CHECK(family_authorization_epoch > 0),
  policy_authorization_epoch INTEGER NOT NULL CHECK(policy_authorization_epoch > 0),
  title TEXT NOT NULL CHECK(length(title) BETWEEN 1 AND 160),
  rationale TEXT NOT NULL CHECK(length(rationale) BETWEEN 1 AND 1000),
  payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
  status TEXT NOT NULL CHECK(status IN ('candidate_only','invalidated')),
  created_at TEXT NOT NULL,
  invalidated_at TEXT,
  FOREIGN KEY(child_id) REFERENCES children(id) ON DELETE CASCADE,
  FOREIGN KEY(trace_id) REFERENCES research_uncertainty_traces(trace_id) ON DELETE CASCADE,
  FOREIGN KEY(child_id,concept_id,concept_version)
    REFERENCES interest_projections(child_id,concept_id,concept_version) ON DELETE CASCADE,
  FOREIGN KEY(policy_id,policy_version)
    REFERENCES research_track_policies(policy_id,version) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_research_candidate_child
  ON research_proactivity_candidates(child_id,status,created_at DESC);

CREATE VIEW research_effective_knowledge_evidence AS
SELECT pe.child_id,pe.concept_id,pe.concept_version,e.evidence_id,e.episode_id,
       COALESCE(
         (SELECT json_extract(ec.replacement_json,'$.result')
          FROM evidence_corrections ec
          WHERE ec.evidence_id=e.evidence_id AND ec.child_id=e.child_id
            AND ec.action='correct' AND json_type(ec.replacement_json,'$.result')='text'
          ORDER BY ec.created_at DESC,ec.correction_id DESC LIMIT 1),
         e.result
       ) AS result,
       COALESCE(
         (SELECT json_extract(ec.replacement_json,'$.assistance_level')
          FROM evidence_corrections ec
          WHERE ec.evidence_id=e.evidence_id AND ec.child_id=e.child_id
            AND ec.action='correct' AND json_type(ec.replacement_json,'$.assistance_level')='text'
          ORDER BY ec.created_at DESC,ec.correction_id DESC LIMIT 1),
         e.assistance_level
       ) AS assistance_level
FROM learner_projection_evidence pe JOIN evidence_records e
  ON e.evidence_id=pe.evidence_id AND e.child_id=pe.child_id
WHERE e.construct='knowledge_performance' AND pe.stance!='context_only';

CREATE TRIGGER guard_research_policy_insert
BEFORE INSERT ON research_track_policies
WHEN NEW.policy_id!='uncertainty-context-k2' OR NEW.version!=1
  OR NEW.feature_schema_version!='phase6-v1'
  OR canonical_json_text(NEW.definition_json)!=NEW.definition_json
  OR NEW.definition_hash!=sha256_hex(NEW.definition_json)
  OR NEW.release_content_hash IS NOT NULL OR NEW.authorization_epoch!=0
  OR NEW.status!='candidate' OR NEW.evaluation_id IS NOT NULL
  OR NEW.promotion_review_id IS NOT NULL OR NEW.approved_by_operator IS NOT NULL
  OR NEW.approval_reason IS NOT NULL OR NEW.activation_reason IS NOT NULL
  OR NEW.approved_at IS NOT NULL OR NEW.activated_at IS NOT NULL
BEGIN SELECT RAISE(ABORT,'invalid seeded research policy'); END;

CREATE TRIGGER guard_research_policy_authorized_update
BEFORE UPDATE ON research_track_policies
WHEN research_write_authorized()!=1
BEGIN SELECT RAISE(ABORT,'research policy update requires service authority'); END;

CREATE TRIGGER guard_research_policy_transition
BEFORE UPDATE OF status ON research_track_policies
WHEN NOT (
  (OLD.status='candidate' AND NEW.status IN ('candidate','evaluated'))
  OR (OLD.status='evaluated' AND NEW.status IN ('candidate','evaluated','approved'))
  OR (OLD.status='approved' AND NEW.status IN ('candidate','approved','active'))
  OR (OLD.status='active' AND NEW.status IN ('active','disabled'))
  OR (OLD.status='disabled' AND NEW.status IN ('candidate','disabled','retired'))
  OR (OLD.status='retired' AND NEW.status IN ('candidate','retired'))
)
BEGIN SELECT RAISE(ABORT,'invalid research policy transition'); END;

CREATE TRIGGER guard_research_policy_definition_immutable
BEFORE UPDATE OF policy_id,version,feature_schema_version,definition_json,definition_hash,created_at
ON research_track_policies
BEGIN SELECT RAISE(ABORT,'research policy definition is immutable'); END;

CREATE TRIGGER guard_research_policy_promotion_update
BEFORE UPDATE OF status,evaluation_id,promotion_review_id,approved_by_operator,approved_at,activated_at,
  release_content_hash
ON research_track_policies
WHEN NEW.status IN ('approved','active') AND (
  NEW.evaluation_id IS NULL OR NEW.promotion_review_id IS NULL OR NEW.approved_by_operator IS NULL
  OR NEW.release_content_hash IS NULL
  OR NEW.approved_at IS NULL
  OR NOT EXISTS(
    SELECT 1 FROM research_track_evaluations e
    WHERE e.evaluation_id=NEW.evaluation_id AND e.policy_id=NEW.policy_id
      AND e.policy_version=NEW.version AND e.definition_hash=NEW.definition_hash AND e.status='pass'
      AND e.release_content_hash=NEW.release_content_hash
  )
  OR NOT EXISTS(
    SELECT 1 FROM research_track_promotion_reviews r
    WHERE r.review_id=NEW.promotion_review_id AND r.policy_id=NEW.policy_id
      AND r.policy_version=NEW.version AND r.definition_hash=NEW.definition_hash
      AND r.evaluation_id=NEW.evaluation_id
      AND r.release_content_hash=NEW.release_content_hash
      AND r.reviewer_id!=NEW.approved_by_operator
  )
  OR (NEW.status='active' AND NEW.activated_at IS NULL)
)
BEGIN SELECT RAISE(ABORT,'research policy promotion evidence is incomplete'); END;

CREATE TRIGGER guard_research_policy_epoch_update
BEFORE UPDATE OF status,authorization_epoch ON research_track_policies
WHEN (
  ((OLD.status='active') IS NOT (NEW.status='active'))
  AND NEW.authorization_epoch!=OLD.authorization_epoch+1
) OR (
  ((OLD.status='active') IS (NEW.status='active'))
  AND NEW.authorization_epoch!=OLD.authorization_epoch
)
BEGIN SELECT RAISE(ABORT,'research policy authorization epoch mismatch'); END;

CREATE TRIGGER guard_research_corpus_insert
BEFORE INSERT ON research_calibration_corpora
WHEN research_write_authorized()!=1
  OR EXISTS(SELECT 1 FROM research_track_policies
            WHERE status IN ('approved','active','disabled','retired'))
  OR NEW.status!='pending_review' OR NEW.feature_schema_version!='phase6-v1'
  OR canonical_json_text(NEW.manifest_json)!=NEW.manifest_json
  OR NEW.manifest_hash!=sha256_hex(NEW.manifest_json)
  OR json_extract(NEW.manifest_json,'$.corpus_id')!=NEW.corpus_id
  OR json_extract(NEW.manifest_json,'$.feature_schema_version')!=NEW.feature_schema_version
  OR json_type(NEW.manifest_json,'$.case_hashes')!='array'
  OR json_array_length(NEW.manifest_json,'$.case_hashes')!=NEW.case_count
BEGIN SELECT RAISE(ABORT,'invalid research calibration corpus proposal'); END;

CREATE TRIGGER guard_research_corpus_update
BEFORE UPDATE ON research_calibration_corpora
WHEN research_write_authorized()!=1
  OR EXISTS(SELECT 1 FROM research_track_policies
            WHERE status IN ('approved','active','disabled','retired'))
  OR OLD.corpus_id!=NEW.corpus_id OR OLD.feature_schema_version!=NEW.feature_schema_version
  OR OLD.manifest_json!=NEW.manifest_json OR OLD.manifest_hash!=NEW.manifest_hash
  OR OLD.case_count!=NEW.case_count OR OLD.proposed_by_operator!=NEW.proposed_by_operator
  OR OLD.created_at!=NEW.created_at
  OR NOT (
    (OLD.status='pending_review' AND NEW.status IN ('approved','withdrawn'))
    OR (OLD.status='approved' AND NEW.status='withdrawn')
  )
  OR (NEW.status='approved' AND (
    NEW.approved_by_reviewer IS NULL OR NEW.approved_by_reviewer=NEW.proposed_by_operator
    OR NEW.approval_reason IS NULL OR NEW.approved_at IS NULL
    OR (SELECT COUNT(*) FROM research_calibration_cases c WHERE c.corpus_id=NEW.corpus_id)!=NEW.case_count
    OR EXISTS(
      SELECT 1 FROM research_calibration_cases c WHERE c.corpus_id=NEW.corpus_id
        AND c.case_hash NOT IN (
          SELECT value FROM json_each(NEW.manifest_json,'$.case_hashes')
        )
    )
    OR EXISTS(
      SELECT 1 FROM json_each(NEW.manifest_json,'$.case_hashes') hashes
      WHERE NOT EXISTS(
        SELECT 1 FROM research_calibration_cases c
        WHERE c.corpus_id=NEW.corpus_id AND c.case_hash=hashes.value
      )
    )
  ))
  OR (OLD.status='approved' AND NEW.status='withdrawn'
      AND EXISTS(SELECT 1 FROM research_track_policies
                 WHERE status IN ('approved','active','disabled','retired')))
BEGIN SELECT RAISE(ABORT,'invalid research calibration corpus transition'); END;

CREATE TRIGGER guard_research_corpus_delete
BEFORE DELETE ON research_calibration_corpora
WHEN EXISTS(SELECT 1 FROM research_calibration_cases WHERE corpus_id=OLD.corpus_id)
BEGIN SELECT RAISE(ABORT,'research calibration corpus with cases cannot be deleted'); END;

CREATE TRIGGER guard_research_calibration_case_insert
BEFORE INSERT ON research_calibration_cases
WHEN research_write_authorized()!=1
  OR EXISTS(SELECT 1 FROM research_track_policies
            WHERE status IN ('approved','active','disabled','retired'))
  OR NEW.feature_schema_version!='phase6-v1'
  OR canonical_json_text(NEW.history_json)!=NEW.history_json
  OR json_type(NEW.history_json)!='object'
  OR (SELECT COUNT(*) FROM json_each(NEW.history_json))!=5
  OR EXISTS(
    SELECT 1 FROM json_each(NEW.history_json)
    WHERE key NOT IN (
      'independent_successes','independent_failures','partial_results',
      'supported_successes','supported_failures'
    ) OR type!='integer' OR value<0 OR value>10000
  )
  OR (NEW.source_kind='public_fixture' AND (
    NEW.license_status!='public' OR NEW.privacy_approval_id IS NOT NULL OR NEW.corpus_id IS NOT NULL
  ))
  OR (NEW.source_kind='synthetic' AND (
    NEW.license_status!='synthetic' OR NEW.privacy_approval_id IS NOT NULL OR NEW.corpus_id IS NOT NULL
  ))
  OR (NEW.source_kind='approved_deidentified' AND (
    NEW.license_status!='approved' OR NEW.privacy_approval_id IS NULL
    OR NEW.corpus_id IS NULL OR NEW.privacy_approval_id!=NEW.corpus_id
    OR NOT EXISTS(
      SELECT 1 FROM research_calibration_corpora corpus
      WHERE corpus.corpus_id=NEW.corpus_id AND corpus.status='pending_review'
        AND NEW.case_hash IN (SELECT value FROM json_each(corpus.manifest_json,'$.case_hashes'))
    )
  ))
  OR NEW.case_hash!=research_calibration_case_hash(
    NEW.case_id,NEW.source_kind,NEW.license_status,NEW.privacy_approval_id,NEW.corpus_id,NEW.split,
    NEW.group_digest,NEW.task_signature_digest,NEW.domain,NEW.history_json,
    NEW.observed_success,NEW.source_digest
  )
BEGIN SELECT RAISE(ABORT,'invalid research calibration case'); END;

CREATE TRIGGER guard_research_calibration_case_update
BEFORE UPDATE ON research_calibration_cases
BEGIN SELECT RAISE(ABORT,'research calibration case is immutable'); END;

CREATE TRIGGER guard_research_calibration_case_delete
BEFORE DELETE ON research_calibration_cases
WHEN EXISTS(SELECT 1 FROM research_track_policies
            WHERE status IN ('approved','active','disabled','retired'))
  OR EXISTS(SELECT 1 FROM research_calibration_corpora corpus
            WHERE corpus.corpus_id=OLD.corpus_id AND corpus.status='approved')
BEGIN SELECT RAISE(ABORT,'approved research calibration data cannot be deleted'); END;

CREATE TRIGGER guard_research_evaluation_insert
BEFORE INSERT ON research_track_evaluations
WHEN research_write_authorized()!=1
  OR NOT EXISTS(
    SELECT 1 FROM research_track_policies p
    WHERE p.policy_id=NEW.policy_id AND p.version=NEW.policy_version
      AND p.definition_hash=NEW.definition_hash
  )
  OR NEW.evaluator!='heldout-calibration-v1'
  OR (NEW.status='pass' AND (
    NEW.real_case_count<40 OR NEW.train_count<20 OR NEW.heldout_count<20
    OR NEW.distinct_task_count<8 OR NEW.distinct_domain_count<3
    OR NEW.heldout_positive_count<5 OR NEW.heldout_negative_count<5
    OR NEW.split_overlap_count!=0 OR NEW.brier_score>0.12
    OR NEW.expected_calibration_error>0.15
    OR json_type(NEW.gates_json)!='object'
    OR (SELECT COUNT(*) FROM json_each(NEW.gates_json))!=10
    OR EXISTS(SELECT 1 FROM json_each(NEW.gates_json) WHERE type!='true')
  ))
BEGIN SELECT RAISE(ABORT,'invalid research evaluation insert'); END;

CREATE TRIGGER guard_research_evaluation_immutable
BEFORE UPDATE ON research_track_evaluations
BEGIN SELECT RAISE(ABORT,'research evaluation is immutable'); END;

CREATE TRIGGER guard_research_evaluation_delete
BEFORE DELETE ON research_track_evaluations
WHEN EXISTS(SELECT 1 FROM research_track_policies WHERE evaluation_id=OLD.evaluation_id)
  OR EXISTS(SELECT 1 FROM research_track_authorization_snapshots
            WHERE evaluation_id=OLD.evaluation_id)
BEGIN SELECT RAISE(ABORT,'referenced research evaluation cannot be deleted'); END;

CREATE TRIGGER guard_research_review_immutable
BEFORE UPDATE ON research_track_promotion_reviews
BEGIN SELECT RAISE(ABORT,'research promotion review is immutable'); END;

CREATE TRIGGER guard_research_review_insert
BEFORE INSERT ON research_track_promotion_reviews
WHEN research_write_authorized()!=1
  OR canonical_json_text(NEW.lab_report_json)!=NEW.lab_report_json
  OR NEW.lab_report_hash!=sha256_hex(NEW.lab_report_json)
  OR json_extract(NEW.lab_report_json,'$.status')!='pass'
  OR json_extract(NEW.lab_report_json,'$.promotion_eligible')!=1
  OR json_extract(NEW.lab_report_json,'$.live_judge.status')!='pass'
  OR json_extract(NEW.lab_report_json,'$.live_judge.required_for_promotion')!=1
  OR json_extract(NEW.lab_report_json,'$.live_judge.identity.independent_from_generation')!=1
  OR json_extract(NEW.lab_report_json,'$.personalization_policy.promotion_content_hash')
     !=NEW.release_content_hash
  OR json_extract(NEW.lab_report_json,'$.suites.research_track.cases')<1
  OR json_extract(NEW.lab_report_json,'$.suites.research_track.failed')!=0
  OR json_extract(NEW.lab_report_json,'$.suites.research_track.passed')
     !=json_extract(NEW.lab_report_json,'$.suites.research_track.cases')
  OR NOT EXISTS(
    SELECT 1 FROM research_track_evaluations e
    WHERE e.evaluation_id=NEW.evaluation_id AND e.policy_id=NEW.policy_id
      AND e.policy_version=NEW.policy_version AND e.definition_hash=NEW.definition_hash
      AND e.release_content_hash=NEW.release_content_hash AND e.status='pass'
  )
BEGIN SELECT RAISE(ABORT,'invalid research promotion review insert'); END;

CREATE TRIGGER guard_research_review_delete
BEFORE DELETE ON research_track_promotion_reviews
WHEN EXISTS(SELECT 1 FROM research_track_policies WHERE promotion_review_id=OLD.review_id)
  OR EXISTS(SELECT 1 FROM research_track_authorization_snapshots
            WHERE promotion_review_id=OLD.review_id)
BEGIN SELECT RAISE(ABORT,'referenced research promotion review cannot be deleted'); END;

CREATE TRIGGER guard_research_policy_event_immutable
BEFORE UPDATE ON research_track_policy_events
BEGIN SELECT RAISE(ABORT,'research policy event is immutable'); END;

CREATE TRIGGER guard_research_policy_event_insert
BEFORE INSERT ON research_track_policy_events
WHEN research_write_authorized()!=1
BEGIN SELECT RAISE(ABORT,'research policy event requires service authority'); END;

CREATE TRIGGER guard_research_policy_event_delete
BEFORE DELETE ON research_track_policy_events
BEGIN SELECT RAISE(ABORT,'research policy event is immutable'); END;

CREATE TRIGGER guard_research_control_binding_insert
BEFORE INSERT ON research_track_controls
WHEN research_write_authorized()!=1 OR (NEW.enabled=1 AND NOT EXISTS(
  SELECT 1 FROM research_track_policies p JOIN learner_model_controls l ON l.child_id=NEW.child_id
  WHERE p.policy_id=NEW.policy_id AND p.version=NEW.policy_version
    AND p.definition_hash=NEW.definition_hash
    AND p.release_content_hash=NEW.release_content_hash AND p.status='active'
    AND p.evaluation_id IS NOT NULL AND p.promotion_review_id IS NOT NULL
    AND l.collection_enabled=1
))
BEGIN SELECT RAISE(ABORT,'invalid family research-policy binding'); END;

CREATE TRIGGER guard_research_control_binding_update
BEFORE UPDATE OF enabled,policy_id,policy_version,definition_hash,release_content_hash
ON research_track_controls
WHEN NEW.enabled=1 AND NOT EXISTS(
  SELECT 1 FROM research_track_policies p JOIN learner_model_controls l ON l.child_id=NEW.child_id
  WHERE p.policy_id=NEW.policy_id AND p.version=NEW.policy_version
    AND p.definition_hash=NEW.definition_hash AND p.status='active'
    AND p.release_content_hash=NEW.release_content_hash
    AND p.evaluation_id IS NOT NULL AND p.promotion_review_id IS NOT NULL
    AND l.collection_enabled=1
)
BEGIN SELECT RAISE(ABORT,'invalid family research-policy binding'); END;

CREATE TRIGGER guard_research_control_authorized_update
BEFORE UPDATE ON research_track_controls
WHEN research_write_authorized()!=1 AND NOT (
  OLD.enabled=1 AND NEW.enabled=0 AND NEW.policy_id IS NULL
  AND NEW.policy_version IS NULL AND NEW.definition_hash IS NULL
  AND NEW.release_content_hash IS NULL
  AND EXISTS(
    SELECT 1 FROM learner_model_controls l
    WHERE l.child_id=NEW.child_id AND l.collection_enabled=0
  )
)
BEGIN SELECT RAISE(ABORT,'research control update requires service authority'); END;

CREATE TRIGGER guard_research_control_epoch_update
BEFORE UPDATE OF enabled,policy_id,policy_version,definition_hash,release_content_hash,
  family_authorization_epoch
ON research_track_controls
WHEN (
  (
    OLD.enabled IS NOT NEW.enabled OR OLD.policy_id IS NOT NEW.policy_id
    OR OLD.policy_version IS NOT NEW.policy_version OR OLD.definition_hash IS NOT NEW.definition_hash
    OR OLD.release_content_hash IS NOT NEW.release_content_hash
  ) AND NEW.family_authorization_epoch!=OLD.family_authorization_epoch+1
) OR (
  OLD.enabled IS NEW.enabled AND OLD.policy_id IS NEW.policy_id
  AND OLD.policy_version IS NEW.policy_version AND OLD.definition_hash IS NEW.definition_hash
  AND OLD.release_content_hash IS NEW.release_content_hash
  AND NEW.family_authorization_epoch!=OLD.family_authorization_epoch
)
BEGIN SELECT RAISE(ABORT,'family research authorization epoch mismatch'); END;

CREATE TRIGGER guard_research_control_event_immutable
BEFORE UPDATE ON research_track_control_events
BEGIN SELECT RAISE(ABORT,'research control event is immutable'); END;

CREATE TRIGGER guard_research_control_event_insert
BEFORE INSERT ON research_track_control_events
WHEN research_write_authorized()!=1 AND NOT (
  NEW.actor_type='system' AND NEW.enabled=0 AND NEW.policy_id IS NULL
  AND NEW.policy_version IS NULL AND NEW.definition_hash IS NULL
  AND NEW.release_content_hash IS NULL
  AND NEW.reason='Learner-model collection was paused.'
  AND EXISTS(
    SELECT 1 FROM learner_model_controls l
    WHERE l.child_id=NEW.child_id AND l.collection_enabled=0
  )
)
BEGIN SELECT RAISE(ABORT,'research control event requires service authority'); END;

CREATE TRIGGER guard_research_control_event_delete
BEFORE DELETE ON research_track_control_events
WHEN research_write_authorized()!=1
BEGIN SELECT RAISE(ABORT,'research control event deletion requires service authority'); END;

CREATE TRIGGER guard_research_authorization_snapshot_insert
BEFORE INSERT ON research_track_authorization_snapshots
WHEN research_write_authorized()!=1
  OR NEW.created_at!=NEW.activated_at
  OR NEW.snapshot_hash!=authority_hash(
    NEW.policy_id,NEW.policy_version,NEW.definition_hash,NEW.release_content_hash,
    NEW.evaluation_id,NEW.promotion_review_id,NEW.policy_authorization_epoch,
    NEW.approved_by_operator,NEW.approval_reason,NEW.activation_reason,
    NEW.approved_at,NEW.activated_at,NEW.approval_event_id,NEW.activation_event_id,
    NEW.created_at
  )
  OR NEW.snapshot_id!='rtauth_'||substr(sha256_hex(NEW.snapshot_hash),1,24)
  OR NOT EXISTS(
    SELECT 1 FROM research_track_policies p
    WHERE p.policy_id=NEW.policy_id AND p.version=NEW.policy_version
      AND p.status='active' AND p.definition_hash=NEW.definition_hash
      AND p.release_content_hash=NEW.release_content_hash
      AND p.evaluation_id=NEW.evaluation_id
      AND p.promotion_review_id=NEW.promotion_review_id
      AND p.authorization_epoch=NEW.policy_authorization_epoch
      AND p.approved_by_operator=NEW.approved_by_operator
      AND p.approval_reason=NEW.approval_reason
      AND p.activation_reason=NEW.activation_reason
      AND p.approved_at=NEW.approved_at AND p.activated_at=NEW.activated_at
  )
  OR NOT EXISTS(
    SELECT 1 FROM research_track_evaluations e
    WHERE e.evaluation_id=NEW.evaluation_id AND e.policy_id=NEW.policy_id
      AND e.policy_version=NEW.policy_version AND e.definition_hash=NEW.definition_hash
      AND e.release_content_hash=NEW.release_content_hash AND e.status='pass'
  )
  OR NOT EXISTS(
    SELECT 1 FROM research_track_promotion_reviews r
    WHERE r.review_id=NEW.promotion_review_id AND r.policy_id=NEW.policy_id
      AND r.policy_version=NEW.policy_version AND r.definition_hash=NEW.definition_hash
      AND r.release_content_hash=NEW.release_content_hash
      AND r.evaluation_id=NEW.evaluation_id
  )
  OR NOT EXISTS(
    SELECT 1 FROM research_track_policy_events e
    WHERE e.event_id=NEW.approval_event_id AND e.policy_id=NEW.policy_id
      AND e.policy_version=NEW.policy_version AND e.action='approved'
      AND e.actor_type='operator' AND e.operator_id=NEW.approved_by_operator
      AND e.reason=NEW.approval_reason
  )
  OR NOT EXISTS(
    SELECT 1 FROM research_track_policy_events e
    WHERE e.event_id=NEW.activation_event_id AND e.policy_id=NEW.policy_id
      AND e.policy_version=NEW.policy_version AND e.action='activated'
      AND e.actor_type='operator' AND e.operator_id=NEW.approved_by_operator
      AND e.reason=NEW.activation_reason
  )
BEGIN SELECT RAISE(ABORT,'invalid research authorization snapshot'); END;

CREATE TRIGGER guard_research_authorization_snapshot_immutable
BEFORE UPDATE ON research_track_authorization_snapshots
BEGIN SELECT RAISE(ABORT,'research authorization snapshot is immutable'); END;

CREATE TRIGGER guard_research_authorization_snapshot_delete
BEFORE DELETE ON research_track_authorization_snapshots
BEGIN SELECT RAISE(ABORT,'research authorization snapshot is immutable'); END;

CREATE TRIGGER guard_research_trace_insert
BEFORE INSERT ON research_uncertainty_traces
WHEN research_write_authorized()!=1
  OR NEW.advisory_only!=1
  OR NEW.trace_hash!=authority_hash(
    NEW.child_id,NEW.concept_id,NEW.concept_version,NEW.construct,
    NEW.projection_input_digest,NEW.projection_rule_id,NEW.projection_rule_version,
    NEW.policy_id,NEW.policy_version,NEW.definition_hash,NEW.release_content_hash,NEW.evaluation_id,
    NEW.promotion_review_id,NEW.authorization_snapshot_id,
    NEW.family_authorization_epoch,NEW.policy_authorization_epoch,
    NEW.evidence_digest,NEW.history_json,NEW.alpha,NEW.beta,NEW.posterior_mean,
    NEW.variance,NEW.interval_90_lower,NEW.interval_90_upper,NEW.uncertainty,
    NEW.method_version,NEW.advisory_only,NEW.rationale
  )
  OR json_type(NEW.evidence_ids_json)!='array'
  OR json_array_length(NEW.evidence_ids_json)<3
  OR canonical_json_text(NEW.evidence_ids_json)!=NEW.evidence_ids_json
  OR NEW.evidence_digest!=sha256_hex(NEW.evidence_ids_json)
  OR canonical_json_text(NEW.history_json)!=NEW.history_json
  OR NEW.trace_id!='rtrace_'||substr(sha256_hex(NEW.trace_hash),1,24)
  OR NEW.history_json!=research_history_from_rows_json(
    (SELECT json_group_array(json_object(
       'episode_id',rows.episode_id,'result',rows.result,'assistance_level',rows.assistance_level
     )) FROM (
       SELECT episode_id,result,assistance_level
       FROM research_effective_knowledge_evidence
       WHERE child_id=NEW.child_id AND concept_id=NEW.concept_id
         AND concept_version=NEW.concept_version
       ORDER BY evidence_id
     ) rows)
  )
  OR research_posterior_json(NEW.history_json)=''
  OR json_extract(research_posterior_json(NEW.history_json),'$.method')!=NEW.method_version
  OR json_extract(research_posterior_json(NEW.history_json),'$.alpha')!=NEW.alpha
  OR json_extract(research_posterior_json(NEW.history_json),'$.beta')!=NEW.beta
  OR json_extract(research_posterior_json(NEW.history_json),'$.posterior_mean')!=NEW.posterior_mean
  OR json_extract(research_posterior_json(NEW.history_json),'$.variance')!=NEW.variance
  OR json_extract(research_posterior_json(NEW.history_json),'$.interval_90_lower')!=NEW.interval_90_lower
  OR json_extract(research_posterior_json(NEW.history_json),'$.interval_90_upper')!=NEW.interval_90_upper
  OR json_extract(research_posterior_json(NEW.history_json),'$.uncertainty')!=NEW.uncertainty
  OR NOT EXISTS(
    SELECT 1 FROM learner_state_projections p
    WHERE p.child_id=NEW.child_id AND p.concept_id=NEW.concept_id
      AND p.concept_version=NEW.concept_version AND p.input_digest=NEW.projection_input_digest
      AND p.rule_id=NEW.projection_rule_id AND p.rule_version=NEW.projection_rule_version
  )
  OR EXISTS(
    SELECT 1 FROM json_each(NEW.evidence_ids_json) ids
    WHERE NOT EXISTS(
      SELECT 1 FROM learner_projection_evidence pe JOIN evidence_records e
        ON e.evidence_id=pe.evidence_id AND e.child_id=pe.child_id
      WHERE pe.child_id=NEW.child_id AND pe.concept_id=NEW.concept_id
        AND pe.concept_version=NEW.concept_version AND pe.evidence_id=ids.value
    )
  )
  OR 3>(
    SELECT COUNT(DISTINCT episode_id) FROM research_effective_knowledge_evidence
    WHERE child_id=NEW.child_id AND concept_id=NEW.concept_id
      AND concept_version=NEW.concept_version
  )
  OR EXISTS(
    SELECT 1 FROM learner_projection_evidence pe
    WHERE pe.child_id=NEW.child_id AND pe.concept_id=NEW.concept_id
      AND pe.concept_version=NEW.concept_version
      AND pe.evidence_id NOT IN (SELECT value FROM json_each(NEW.evidence_ids_json))
  )
  OR json_array_length(NEW.evidence_ids_json)!=(
    SELECT COUNT(*) FROM learner_projection_evidence pe
    WHERE pe.child_id=NEW.child_id AND pe.concept_id=NEW.concept_id
      AND pe.concept_version=NEW.concept_version
  )
  OR NOT EXISTS(
    SELECT 1 FROM research_track_authorization_snapshots s
    WHERE s.snapshot_id=NEW.authorization_snapshot_id
      AND s.policy_id=NEW.policy_id AND s.policy_version=NEW.policy_version
      AND s.definition_hash=NEW.definition_hash
      AND s.release_content_hash=NEW.release_content_hash
      AND s.evaluation_id=NEW.evaluation_id
      AND s.promotion_review_id=NEW.promotion_review_id
      AND s.policy_authorization_epoch=NEW.policy_authorization_epoch
  )
  OR NOT EXISTS(
    SELECT 1 FROM research_track_controls c JOIN research_track_policies p
      ON p.policy_id=c.policy_id AND p.version=c.policy_version
      JOIN learner_model_controls l ON l.child_id=c.child_id
    WHERE c.child_id=NEW.child_id AND c.enabled=1
      AND c.family_authorization_epoch=NEW.family_authorization_epoch
      AND c.definition_hash=NEW.definition_hash
      AND c.release_content_hash=NEW.release_content_hash
      AND p.policy_id=NEW.policy_id AND p.version=NEW.policy_version
      AND p.definition_hash=NEW.definition_hash AND p.status='active'
      AND p.release_content_hash=NEW.release_content_hash
      AND p.authorization_epoch=NEW.policy_authorization_epoch
      AND p.evaluation_id=NEW.evaluation_id AND p.promotion_review_id=NEW.promotion_review_id
      AND l.collection_enabled=1
  )
BEGIN SELECT RAISE(ABORT,'invalid research uncertainty trace'); END;

CREATE TRIGGER guard_research_trace_immutable
BEFORE UPDATE ON research_uncertainty_traces
BEGIN SELECT RAISE(ABORT,'research uncertainty trace is immutable'); END;

CREATE TRIGGER delete_research_trace_with_evidence
BEFORE DELETE ON evidence_records
BEGIN
  DELETE FROM research_uncertainty_traces
  WHERE child_id=OLD.child_id
    AND EXISTS(SELECT 1 FROM json_each(evidence_ids_json) WHERE value=OLD.evidence_id);
END;

CREATE TRIGGER delete_research_trace_with_projection_change
BEFORE UPDATE OF input_digest,rule_id,rule_version ON learner_state_projections
WHEN OLD.input_digest!=NEW.input_digest OR OLD.rule_id!=NEW.rule_id OR OLD.rule_version!=NEW.rule_version
BEGIN
  DELETE FROM research_uncertainty_traces
  WHERE child_id=OLD.child_id AND concept_id=OLD.concept_id
    AND concept_version=OLD.concept_version;
END;

CREATE TRIGGER delete_research_trace_with_projection_delete
BEFORE DELETE ON learner_state_projections
BEGIN
  DELETE FROM research_uncertainty_traces
  WHERE child_id=OLD.child_id AND concept_id=OLD.concept_id
    AND concept_version=OLD.concept_version;
END;

CREATE TRIGGER guard_research_candidate_insert
BEFORE INSERT ON research_proactivity_candidates
WHEN research_write_authorized()!=1
  OR NEW.status!='candidate_only'
  OR NEW.candidate_id!='rtcand_'||substr(sha256_hex(NEW.candidate_hash),1,24)
  OR canonical_json_text(NEW.payload_json)!=NEW.payload_json
  OR json_type(NEW.payload_json)!='object'
  OR (SELECT COUNT(*) FROM json_each(NEW.payload_json))!=8
  OR EXISTS(
    SELECT 1 FROM json_each(NEW.payload_json)
    WHERE key NOT IN (
      'concept','trace_id','evidence_ids','deterministic_state','posterior_mean','uncertainty',
      'interest_projection','delivery'
    )
  )
  OR json_extract(NEW.payload_json,'$.delivery')!='disabled'
  OR json_extract(NEW.payload_json,'$.trace_id')!=NEW.trace_id
  OR json_extract(NEW.payload_json,'$.concept.id')!=NEW.concept_id
  OR json_extract(NEW.payload_json,'$.concept.version')!=NEW.concept_version
  OR json_extract(NEW.payload_json,'$.interest_projection.input_digest')!=NEW.interest_input_digest
  OR json_extract(NEW.payload_json,'$.interest_projection.rule_id')!=NEW.interest_rule_id
  OR json_extract(NEW.payload_json,'$.interest_projection.rule_version')!=NEW.interest_rule_version
  OR json_extract(NEW.payload_json,'$.interest_projection.phase')!=NEW.interest_phase
  OR NEW.candidate_hash!=authority_hash(
    NEW.child_id,NEW.trace_id,NEW.concept_id,NEW.concept_version,
    NEW.interest_input_digest,NEW.interest_rule_id,NEW.interest_rule_version,NEW.interest_phase,
    NEW.policy_id,NEW.policy_version,NEW.definition_hash,NEW.release_content_hash,
    NEW.family_authorization_epoch,NEW.policy_authorization_epoch,
    NEW.title,NEW.rationale,NEW.payload_json,NEW.status
  )
  OR NOT EXISTS(
    SELECT 1 FROM research_uncertainty_traces t JOIN research_track_controls c
      ON c.child_id=t.child_id JOIN research_track_policies p
      ON p.policy_id=t.policy_id AND p.version=t.policy_version
      JOIN learner_model_controls l ON l.child_id=t.child_id
      JOIN learner_state_projections projection
        ON projection.child_id=t.child_id AND projection.concept_id=t.concept_id
       AND projection.concept_version=t.concept_version
       AND projection.input_digest=t.projection_input_digest
    WHERE t.trace_id=NEW.trace_id AND t.child_id=NEW.child_id
      AND t.concept_id=NEW.concept_id AND t.concept_version=NEW.concept_version
      AND EXISTS(
        SELECT 1 FROM interest_projections i
        WHERE i.child_id=NEW.child_id AND i.concept_id=NEW.concept_id
          AND i.concept_version=NEW.concept_version
          AND i.input_digest=NEW.interest_input_digest AND i.rule_id=NEW.interest_rule_id
          AND i.rule_version=NEW.interest_rule_version AND i.phase=NEW.interest_phase
      )
      AND t.policy_id=NEW.policy_id AND t.policy_version=NEW.policy_version
      AND t.definition_hash=NEW.definition_hash
      AND t.release_content_hash=NEW.release_content_hash
      AND canonical_json_text(json_extract(NEW.payload_json,'$.evidence_ids'))=t.evidence_ids_json
      AND json_extract(NEW.payload_json,'$.deterministic_state')=projection.state
      AND json_extract(NEW.payload_json,'$.posterior_mean')=t.posterior_mean
      AND json_extract(NEW.payload_json,'$.uncertainty')=t.uncertainty
      AND t.family_authorization_epoch=NEW.family_authorization_epoch
      AND t.policy_authorization_epoch=NEW.policy_authorization_epoch
      AND c.enabled=1 AND c.family_authorization_epoch=NEW.family_authorization_epoch
      AND c.release_content_hash=NEW.release_content_hash
      AND p.status='active' AND p.definition_hash=NEW.definition_hash
      AND p.release_content_hash=NEW.release_content_hash
      AND p.authorization_epoch=NEW.policy_authorization_epoch
      AND l.collection_enabled=1
  )
BEGIN SELECT RAISE(ABORT,'invalid research proactivity candidate'); END;

CREATE TRIGGER guard_research_candidate_update
BEFORE UPDATE ON research_proactivity_candidates
WHEN OLD.status!='candidate_only' OR NEW.status!='invalidated'
  OR OLD.candidate_id!=NEW.candidate_id OR OLD.candidate_hash!=NEW.candidate_hash
  OR OLD.child_id!=NEW.child_id OR OLD.trace_id!=NEW.trace_id
  OR OLD.concept_id!=NEW.concept_id OR OLD.concept_version!=NEW.concept_version
  OR OLD.interest_input_digest!=NEW.interest_input_digest
  OR OLD.interest_rule_id!=NEW.interest_rule_id
  OR OLD.interest_rule_version!=NEW.interest_rule_version
  OR OLD.interest_phase!=NEW.interest_phase
  OR OLD.policy_id!=NEW.policy_id OR OLD.policy_version!=NEW.policy_version
  OR OLD.definition_hash!=NEW.definition_hash
  OR OLD.release_content_hash!=NEW.release_content_hash
  OR OLD.family_authorization_epoch!=NEW.family_authorization_epoch
  OR OLD.policy_authorization_epoch!=NEW.policy_authorization_epoch
  OR OLD.title!=NEW.title OR OLD.rationale!=NEW.rationale OR OLD.payload_json!=NEW.payload_json
  OR OLD.created_at!=NEW.created_at OR NEW.invalidated_at IS NULL
BEGIN SELECT RAISE(ABORT,'research candidate is immutable except invalidation'); END;

CREATE TRIGGER invalidate_research_candidates_on_optout
AFTER UPDATE OF enabled ON research_track_controls
WHEN OLD.enabled=1 AND NEW.enabled=0
BEGIN
  UPDATE research_proactivity_candidates SET status='invalidated',invalidated_at=NEW.updated_at
  WHERE child_id=NEW.child_id AND status='candidate_only';
END;

CREATE TRIGGER invalidate_research_candidates_on_policy_stop
AFTER UPDATE OF status ON research_track_policies
WHEN OLD.status='active' AND NEW.status!='active'
BEGIN
  UPDATE research_proactivity_candidates SET status='invalidated',invalidated_at=NEW.updated_at
  WHERE policy_id=NEW.policy_id AND policy_version=NEW.version AND status='candidate_only';
END;

CREATE TRIGGER invalidate_research_candidates_on_interest_change
BEFORE UPDATE OF input_digest,rule_id,rule_version,phase ON interest_projections
WHEN OLD.input_digest!=NEW.input_digest OR OLD.rule_id!=NEW.rule_id
  OR OLD.rule_version!=NEW.rule_version OR OLD.phase!=NEW.phase
BEGIN
  UPDATE research_proactivity_candidates SET status='invalidated',invalidated_at=NEW.computed_at
  WHERE child_id=OLD.child_id AND concept_id=OLD.concept_id
    AND concept_version=OLD.concept_version AND status='candidate_only';
END;

CREATE TRIGGER disable_research_on_collection_pause
AFTER UPDATE OF collection_enabled ON learner_model_controls
WHEN OLD.collection_enabled=1 AND NEW.collection_enabled=0
BEGIN
  UPDATE research_track_controls
  SET enabled=0,policy_id=NULL,policy_version=NULL,definition_hash=NULL,release_content_hash=NULL,
      family_authorization_epoch=family_authorization_epoch+1,
      updated_by=NEW.updated_by,reason='Learner-model collection was paused.',updated_at=NEW.updated_at
  WHERE child_id=NEW.child_id AND enabled=1;
  INSERT OR IGNORE INTO research_track_control_events(
    event_id,child_id,enabled,policy_id,policy_version,definition_hash,release_content_hash,
    family_authorization_epoch,actor_type,reason,created_at
  )
  SELECT 'rtctl_'||substr(sha256_hex(NEW.child_id||'|'||NEW.updated_at||'|collection_pause'),1,24),
    NEW.child_id,0,NULL,NULL,NULL,NULL,c.family_authorization_epoch,'system',
    'Learner-model collection was paused.',NEW.updated_at
  FROM research_track_controls c WHERE c.child_id=NEW.child_id
    AND c.enabled=0 AND c.updated_at=NEW.updated_at
    AND c.reason='Learner-model collection was paused.';
END;

INSERT INTO schema_migrations(version,name,applied_at)
VALUES(26,'gated uncertainty and proactivity research track',strftime('%Y-%m-%dT%H:%M:%fZ','now'))
ON CONFLICT(version) DO NOTHING;
COMMIT;"""
        )
        now = utcnow()
        conn.execute(
            """INSERT OR IGNORE INTO research_track_policies(
                 policy_id,version,feature_schema_version,definition_json,definition_hash,status,
                 created_at,updated_at
               ) VALUES(?,?,?,?,?,'candidate',?,?)""",
            (
                RESEARCH_POLICY_ID,
                RESEARCH_POLICY_VERSION,
                RESEARCH_FEATURE_SCHEMA_VERSION,
                jdump(RESEARCH_POLICY_DEFINITION),
                RESEARCH_POLICY_DEFINITION_HASH,
                now,
                now,
            ),
        )
        conn.execute(f"PRAGMA user_version = {max(current, 26)}")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise

    required_tables = {
        "research_track_policies",
        "research_calibration_corpora",
        "research_calibration_cases",
        "research_track_evaluations",
        "research_track_promotion_reviews",
        "research_track_policy_events",
        "research_track_controls",
        "research_track_control_events",
        "research_track_authorization_snapshots",
        "research_uncertainty_traces",
        "research_proactivity_candidates",
    }
    required_triggers = {
        "guard_research_policy_insert",
        "guard_research_policy_authorized_update",
        "guard_research_policy_transition",
        "guard_research_policy_definition_immutable",
        "guard_research_policy_promotion_update",
        "guard_research_policy_epoch_update",
        "guard_research_corpus_insert",
        "guard_research_corpus_update",
        "guard_research_corpus_delete",
        "guard_research_calibration_case_insert",
        "guard_research_calibration_case_update",
        "guard_research_calibration_case_delete",
        "guard_research_evaluation_insert",
        "guard_research_evaluation_immutable",
        "guard_research_evaluation_delete",
        "guard_research_review_insert",
        "guard_research_review_immutable",
        "guard_research_review_delete",
        "guard_research_policy_event_insert",
        "guard_research_policy_event_immutable",
        "guard_research_policy_event_delete",
        "guard_research_control_binding_insert",
        "guard_research_control_binding_update",
        "guard_research_control_authorized_update",
        "guard_research_control_epoch_update",
        "guard_research_control_event_immutable",
        "guard_research_control_event_delete",
        "guard_research_authorization_snapshot_insert",
        "guard_research_authorization_snapshot_immutable",
        "guard_research_authorization_snapshot_delete",
        "guard_research_trace_insert",
        "guard_research_trace_immutable",
        "delete_research_trace_with_evidence",
        "delete_research_trace_with_projection_change",
        "delete_research_trace_with_projection_delete",
        "guard_research_candidate_insert",
        "guard_research_candidate_update",
        "invalidate_research_candidates_on_optout",
        "invalidate_research_candidates_on_policy_stop",
        "invalidate_research_candidates_on_interest_change",
        "disable_research_on_collection_pause",
    }
    required_indexes = {
        "idx_research_track_one_active",
        "idx_research_calibration_split",
        "idx_research_control_events_child",
        "idx_research_trace_child",
        "idx_research_candidate_child",
    }
    tables = _tables(conn)
    triggers = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    indexes = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    views = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")}
    required_columns = {
        "research_track_policies": {
            "policy_id", "version", "feature_schema_version", "definition_json",
            "definition_hash", "release_content_hash", "authorization_epoch", "status",
            "evaluation_id", "promotion_review_id", "approved_by_operator",
            "approval_reason", "activation_reason", "approved_at", "activated_at",
            "created_at", "updated_at",
        },
        "research_calibration_corpora": {
            "corpus_id", "feature_schema_version", "manifest_json", "manifest_hash",
            "case_count", "status", "proposed_by_operator", "approved_by_reviewer",
            "approval_reason", "approved_at", "created_at",
        },
        "research_calibration_cases": {
            "case_id", "source_kind", "license_status", "privacy_approval_id", "corpus_id",
            "feature_schema_version", "split", "group_digest", "task_signature_digest",
            "domain", "history_json", "observed_success", "source_digest", "case_hash",
            "created_by_operator", "created_at",
        },
        "research_track_evaluations": {
            "evaluation_id", "policy_id", "policy_version", "definition_hash",
            "release_content_hash", "dataset_digest", "case_count", "real_case_count",
            "train_count", "heldout_count", "distinct_task_count", "distinct_domain_count",
            "heldout_positive_count", "heldout_negative_count", "split_overlap_count",
            "brier_score", "expected_calibration_error", "gates_json", "results_hash",
            "evaluator", "evaluator_digest", "status", "created_at",
        },
        "research_track_promotion_reviews": {
            "review_id", "policy_id", "policy_version", "definition_hash",
            "release_content_hash", "evaluation_id", "reviewer_id", "lab_report_json",
            "lab_report_hash", "created_at",
        },
        "research_track_policy_events": {
            "event_id", "policy_id", "policy_version", "action", "actor_type",
            "operator_id", "reason", "created_at",
        },
        "research_track_controls": {
            "child_id", "enabled", "policy_id", "policy_version", "definition_hash",
            "release_content_hash", "family_authorization_epoch", "updated_by", "reason",
            "created_at", "updated_at",
        },
        "research_track_control_events": {
            "event_id", "child_id", "enabled", "policy_id", "policy_version",
            "definition_hash", "release_content_hash", "family_authorization_epoch",
            "actor_type", "reason", "created_at",
        },
        "research_track_authorization_snapshots": {
            "snapshot_id", "snapshot_hash", "policy_id", "policy_version",
            "definition_hash", "release_content_hash", "evaluation_id",
            "promotion_review_id", "policy_authorization_epoch",
            "approved_by_operator", "approval_reason", "activation_reason",
            "approved_at", "activated_at", "approval_event_id",
            "activation_event_id", "created_at",
        },
        "research_uncertainty_traces": {
            "trace_id", "trace_hash", "child_id", "concept_id", "concept_version",
            "construct", "projection_input_digest", "projection_rule_id",
            "projection_rule_version", "policy_id", "policy_version", "definition_hash",
            "release_content_hash", "evaluation_id", "promotion_review_id",
            "authorization_snapshot_id",
            "family_authorization_epoch", "policy_authorization_epoch", "evidence_ids_json",
            "evidence_digest", "history_json", "alpha", "beta", "posterior_mean",
            "variance", "interval_90_lower", "interval_90_upper", "uncertainty",
            "method_version", "advisory_only", "rationale", "created_at",
        },
        "research_proactivity_candidates": {
            "candidate_id", "candidate_hash", "child_id", "trace_id", "concept_id",
            "concept_version", "interest_input_digest", "interest_rule_id",
            "interest_rule_version", "interest_phase", "policy_id", "policy_version",
            "definition_hash", "release_content_hash", "family_authorization_epoch",
            "policy_authorization_epoch", "title", "rationale", "payload_json", "status",
            "created_at", "invalidated_at",
        },
    }
    missing_columns = {
        table: sorted(columns - {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')})
        for table, columns in required_columns.items()
        if table in tables
    }
    missing_columns = {table: columns for table, columns in missing_columns.items() if columns}
    required_primary_keys = {
        "research_track_policies": ("policy_id", "version"),
        "research_calibration_corpora": ("corpus_id",),
        "research_calibration_cases": ("case_id",),
        "research_track_evaluations": ("evaluation_id",),
        "research_track_promotion_reviews": ("review_id",),
        "research_track_policy_events": ("event_id",),
        "research_track_controls": ("child_id",),
        "research_track_control_events": ("event_id",),
        "research_track_authorization_snapshots": ("snapshot_id",),
        "research_uncertainty_traces": ("trace_id",),
        "research_proactivity_candidates": ("candidate_id",),
    }
    invalid_primary_keys = {}
    for table, expected in required_primary_keys.items():
        if table not in tables:
            continue
        present = tuple(
            str(row[1])
            for row in sorted(
                conn.execute(f'PRAGMA table_info("{table}")').fetchall(),
                key=lambda value: int(value[5]) if int(value[5]) else 10_000,
            )
            if int(row[5])
        )
        if present != expected:
            invalid_primary_keys[table] = {"expected": expected, "present": present}
    nullable = {
        "research_track_policies": {
            "release_content_hash", "evaluation_id", "promotion_review_id",
            "approved_by_operator", "approval_reason", "activation_reason", "approved_at",
            "activated_at",
        },
        "research_calibration_corpora": {
            "approved_by_reviewer", "approval_reason", "approved_at",
        },
        "research_calibration_cases": {"privacy_approval_id", "corpus_id"},
        "research_track_policy_events": {"operator_id"},
        "research_track_controls": {
            "policy_id", "policy_version", "definition_hash", "release_content_hash",
        },
        "research_track_control_events": {
            "policy_id", "policy_version", "definition_hash", "release_content_hash",
        },
        "research_proactivity_candidates": {"invalidated_at"},
    }
    invalid_not_null = {
        table: sorted(
            column
            for column in columns - nullable.get(table, set())
            if not next(
                (bool(row[3]) or bool(row[5]) for row in conn.execute(f'PRAGMA table_info("{table}")') if str(row[1]) == column),
                False,
            )
        )
        for table, columns in required_columns.items()
        if table in tables
    }
    invalid_not_null = {table: columns for table, columns in invalid_not_null.items() if columns}

    def research_foreign_keys(table: str) -> set[tuple[str, str, tuple[tuple[str, str], ...]]]:
        grouped: dict[int, list[sqlite3.Row]] = {}
        for item in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            grouped.setdefault(int(item[0]), []).append(item)
        return {
            (
                str(items[0][2]),
                str(items[0][6]).upper(),
                tuple(
                    (str(item[3]), str(item[4]))
                    for item in sorted(items, key=lambda value: int(value[1]))
                ),
            )
            for items in grouped.values()
        }

    required_foreign_keys = {
        "research_calibration_cases": {
            ("research_calibration_corpora", "RESTRICT", (("corpus_id", "corpus_id"),)),
        },
        "research_track_evaluations": {
            ("research_track_policies", "CASCADE", (("policy_id", "policy_id"), ("policy_version", "version"))),
        },
        "research_track_promotion_reviews": {
            ("research_track_policies", "CASCADE", (("policy_id", "policy_id"), ("policy_version", "version"))),
            ("research_track_evaluations", "RESTRICT", (("evaluation_id", "evaluation_id"),)),
        },
        "research_track_policy_events": {
            ("research_track_policies", "CASCADE", (("policy_id", "policy_id"), ("policy_version", "version"))),
        },
        "research_track_controls": {
            ("children", "CASCADE", (("child_id", "id"),)),
        },
        "research_track_control_events": {
            ("children", "CASCADE", (("child_id", "id"),)),
        },
        "research_track_authorization_snapshots": {
            ("research_track_policies", "RESTRICT", (("policy_id", "policy_id"), ("policy_version", "version"))),
            ("research_track_evaluations", "RESTRICT", (("evaluation_id", "evaluation_id"),)),
            ("research_track_promotion_reviews", "RESTRICT", (("promotion_review_id", "review_id"),)),
            ("research_track_policy_events", "RESTRICT", (("approval_event_id", "event_id"),)),
            ("research_track_policy_events", "RESTRICT", (("activation_event_id", "event_id"),)),
        },
        "research_uncertainty_traces": {
            ("learner_state_projections", "CASCADE", (("child_id", "child_id"), ("concept_id", "concept_id"), ("concept_version", "concept_version"))),
            ("research_track_policies", "RESTRICT", (("policy_id", "policy_id"), ("policy_version", "version"))),
            ("research_track_evaluations", "RESTRICT", (("evaluation_id", "evaluation_id"),)),
            ("research_track_promotion_reviews", "RESTRICT", (("promotion_review_id", "review_id"),)),
            ("research_track_authorization_snapshots", "RESTRICT", (("authorization_snapshot_id", "snapshot_id"),)),
        },
        "research_proactivity_candidates": {
            ("children", "CASCADE", (("child_id", "id"),)),
            ("research_uncertainty_traces", "CASCADE", (("trace_id", "trace_id"),)),
            ("interest_projections", "CASCADE", (("child_id", "child_id"), ("concept_id", "concept_id"), ("concept_version", "concept_version"))),
            ("research_track_policies", "RESTRICT", (("policy_id", "policy_id"), ("policy_version", "version"))),
        },
    }
    missing_foreign_keys = {
        table: sorted(expected - research_foreign_keys(table))
        for table, expected in required_foreign_keys.items()
        if table in tables and expected - research_foreign_keys(table)
    }

    def research_index_shape(table: str, name: str) -> tuple[tuple[str, ...], bool, bool] | None:
        index = next(
            (row for row in conn.execute(f'PRAGMA index_list("{table}")') if str(row[1]) == name),
            None,
        )
        if index is None:
            return None
        columns = tuple(
            str(row[2])
            for row in sorted(
                conn.execute(f'PRAGMA index_info("{name}")').fetchall(), key=lambda value: int(value[0])
            )
        )
        return columns, bool(index[2]), bool(index[4])

    required_index_shapes = {
        ("research_track_policies", "idx_research_track_one_active"): (("None",), True, True),
        ("research_calibration_cases", "idx_research_calibration_split"): (("split", "source_kind", "domain"), False, False),
        ("research_track_control_events", "idx_research_control_events_child"): (("child_id", "created_at"), False, False),
        ("research_uncertainty_traces", "idx_research_trace_child"): (("child_id", "concept_id", "concept_version", "created_at"), False, False),
        ("research_proactivity_candidates", "idx_research_candidate_child"): (("child_id", "status", "created_at"), False, False),
    }
    invalid_indexes = {
        name: {"expected": expected, "present": research_index_shape(table, name)}
        for (table, name), expected in required_index_shapes.items()
        if research_index_shape(table, name) != expected
    }
    required_unique_shapes = {
        "research_calibration_corpora": {("manifest_hash",)},
        "research_calibration_cases": {("case_hash",)},
        "research_track_promotion_reviews": {("lab_report_hash",)},
        "research_track_authorization_snapshots": {
            ("snapshot_hash",), ("approval_event_id",), ("activation_event_id",),
            ("policy_id", "policy_version", "policy_authorization_epoch"),
        },
        "research_uncertainty_traces": {("trace_hash",)},
        "research_proactivity_candidates": {("candidate_hash",)},
    }
    missing_unique_constraints: dict[str, list[tuple[str, ...]]] = {}
    for table, expected in required_unique_shapes.items():
        present = {
            tuple(
                str(info[2])
                for info in sorted(
                    conn.execute(f'PRAGMA index_info("{row[1]}")').fetchall(),
                    key=lambda value: int(value[0]),
                )
            )
            for row in conn.execute(f'PRAGMA index_list("{table}")')
            if bool(row[2]) and not bool(row[4])
        }
        missing = sorted(expected - present)
        if missing:
            missing_unique_constraints[table] = missing
    table_sql = {
        table: re.sub(
            r"\s+", "", str(conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()[0]).casefold()
        )
        for table in required_tables & tables
    }
    required_check_fragments = {
        "research_track_policies": ["check(version>0)", "release_content_hashnotglob'*[^0-9a-f]*'", "'candidate','evaluated','approved','active','disabled','retired'"],
        "research_calibration_corpora": ["check(feature_schema_version='phase6-v1')", "'pending_review','approved','withdrawn'", "check(json_valid(manifest_json))"],
        "research_calibration_cases": ["'public_fixture','synthetic','approved_deidentified'", "check(feature_schema_version='phase6-v1')", "check(json_valid(history_json))"],
        "research_track_evaluations": ["check(brier_scorebetween0and1)", "check(expected_calibration_errorbetween0and1)", "check(json_valid(gates_json))"],
        "research_track_controls": ["check(enabledin(0,1))", "check(family_authorization_epoch>=0)", "release_content_hashisnotnull"],
        "research_track_authorization_snapshots": [
            "check(policy_authorization_epoch>0andpolicy_authorization_epoch%2=1)",
            "snapshot_hashnotglob'*[^0-9a-f]*'",
        ],
        "research_uncertainty_traces": ["check(construct='knowledge_performance')", "check(advisory_only=1)", "check(json_valid(evidence_ids_json))"],
        "research_proactivity_candidates": ["'candidate_only','invalidated'", "check(json_valid(payload_json))", "release_content_hashnotglob'*[^0-9a-f]*'"],
    }
    missing_check_fragments = {
        table: [fragment for fragment in fragments if fragment not in table_sql.get(table, "")]
        for table, fragments in required_check_fragments.items()
    }
    missing_check_fragments = {
        table: fragments for table, fragments in missing_check_fragments.items() if fragments
    }
    view_sql = re.sub(
        r"\s+",
        "",
        str(
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='view' "
                "AND name='research_effective_knowledge_evidence'"
            ).fetchone()[0]
        ).casefold(),
    ) if "research_effective_knowledge_evidence" in views else ""
    invalid_view = not all(
        fragment in view_sql
        for fragment in (
            "fromlearner_projection_evidencepejoinevidence_recordse",
            "fromevidence_correctionsec",
            "ec.action='correct'",
            "pe.stance!='context_only'",
        )
    )

    invalid_corpora: list[str] = []
    invalid_cases: list[str] = []
    invalid_evaluations: list[str] = []
    invalid_reviews: list[str] = []
    invalid_policy_events: list[str] = []
    invalid_authorization_snapshots: list[str] = []
    invalid_controls: list[str] = []
    invalid_traces: list[str] = []
    invalid_candidates: list[str] = []
    shape_valid = not any(
        (
            required_tables - tables,
            missing_columns,
            missing_foreign_keys,
            invalid_primary_keys,
            invalid_not_null,
            invalid_indexes,
            missing_unique_constraints,
            missing_check_fragments,
            invalid_view,
        )
    )
    if repo_root is None:
        source_root = Path(__file__).resolve().parents[2]
        if (source_root / "src" / "curiosity_engine").is_dir() and (
            source_root / "configs"
        ).is_dir():
            current_repo_root = source_root
        else:
            from .config import configuration_root

            current_repo_root = configuration_root().resolve()
    else:
        current_repo_root = repo_root.resolve()
    from .personalization_policy import promotion_manifest
    from .promotion_validation import validate_full_lab_report

    current_release_hash = str(promotion_manifest(current_repo_root)["promotion_content_hash"])
    if shape_valid:
        _reset_stale_research_authority(conn, current_release_hash)
        corpus_by_id = {
            str(row["corpus_id"]): row
            for row in conn.execute("SELECT * FROM research_calibration_corpora")
        }
        for corpus_id, row in corpus_by_id.items():
            case_hashes = [
                str(item[0])
                for item in conn.execute(
                    "SELECT case_hash FROM research_calibration_cases "
                    "WHERE corpus_id=? ORDER BY case_hash",
                    (corpus_id,),
                )
            ]
            expected_manifest = {
                "corpus_id": corpus_id,
                "feature_schema_version": RESEARCH_FEATURE_SCHEMA_VERSION,
                "case_hashes": sorted(case_hashes),
            }
            status = str(row["status"])
            approval_valid = (
                status == "pending_review"
                and row["approved_by_reviewer"] is None
                and row["approval_reason"] is None
                and row["approved_at"] is None
            ) or (
                status == "approved"
                and bool(row["approved_by_reviewer"])
                and row["approved_by_reviewer"] != row["proposed_by_operator"]
                and bool(row["approval_reason"])
                and bool(row["approved_at"])
            ) or status == "withdrawn"
            if not all(
                (
                    str(row["feature_schema_version"]) == RESEARCH_FEATURE_SCHEMA_VERSION,
                    int(row["case_count"]) == len(case_hashes) > 0,
                    str(row["manifest_json"]) == jdump(expected_manifest),
                    str(row["manifest_hash"])
                    == sha256(jdump(expected_manifest).encode()).hexdigest(),
                    approval_valid,
                )
            ):
                invalid_corpora.append(corpus_id)
        for row in conn.execute("SELECT * FROM research_calibration_cases ORDER BY case_id"):
            try:
                history = jload(str(row["history_json"]), {})
                expected_hash = calibration_case_hash(
                    case_id=str(row["case_id"]),
                    source_kind=str(row["source_kind"]),
                    license_status=str(row["license_status"]),
                    privacy_approval_id=(
                        str(row["privacy_approval_id"])
                        if row["privacy_approval_id"] is not None
                        else None
                    ),
                    corpus_id=str(row["corpus_id"]) if row["corpus_id"] is not None else None,
                    split=str(row["split"]),
                    group_digest=str(row["group_digest"]),
                    task_signature_digest=str(row["task_signature_digest"]),
                    domain=str(row["domain"]),
                    history=history,
                    observed_success=int(row["observed_success"]),
                    source_digest=str(row["source_digest"]),
                )
                source_kind = str(row["source_kind"])
                corpus = corpus_by_id.get(str(row["corpus_id"])) if row["corpus_id"] else None
                source_valid = (
                    source_kind == "public_fixture"
                    and row["license_status"] == "public"
                    and row["privacy_approval_id"] is None
                    and row["corpus_id"] is None
                ) or (
                    source_kind == "synthetic"
                    and row["license_status"] == "synthetic"
                    and row["privacy_approval_id"] is None
                    and row["corpus_id"] is None
                ) or (
                    source_kind == "approved_deidentified"
                    and row["license_status"] == "approved"
                    and row["privacy_approval_id"] == row["corpus_id"]
                    and corpus is not None
                    and str(row["case_hash"])
                    in set(jload(str(corpus["manifest_json"]), {}).get("case_hashes") or [])
                )
                valid = all(
                    (
                        source_valid,
                        str(row["feature_schema_version"]) == RESEARCH_FEATURE_SCHEMA_VERSION,
                        str(row["history_json"]) == jdump(history),
                        str(row["case_hash"]) == expected_hash,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                valid = False
            if not valid:
                invalid_cases.append(str(row["case_id"]))
        evaluation_case_rows = conn.execute(
            """SELECT c.*,corpus.status AS corpus_status
               FROM research_calibration_cases c
               LEFT JOIN research_calibration_corpora corpus ON corpus.corpus_id=c.corpus_id
               WHERE c.source_kind!='approved_deidentified' OR corpus.status='approved'
               ORDER BY c.case_id"""
        ).fetchall()
        approved_corpus_rows = conn.execute(
            "SELECT * FROM research_calibration_corpora WHERE status='approved' ORDER BY corpus_id"
        ).fetchall()
        for row in conn.execute("SELECT * FROM research_track_evaluations ORDER BY evaluation_id"):
            try:
                evaluation_policy = conn.execute(
                    "SELECT * FROM research_track_policies WHERE policy_id=? AND version=?",
                    (row["policy_id"], row["policy_version"]),
                ).fetchone()
                valid = (
                    evaluation_policy is not None
                    and (
                        research_evaluation_matches(
                            row,
                            canonical_research_evaluation(
                                evaluation_policy,
                                list(evaluation_case_rows),
                                list(approved_corpus_rows),
                                current_release_hash,
                            ),
                        )
                        if str(row["release_content_hash"]) == current_release_hash
                        else archival_research_evaluation_is_valid(row, evaluation_policy)
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                valid = False
            if not valid:
                invalid_evaluations.append(str(row["evaluation_id"]))
        for row in conn.execute("SELECT * FROM research_track_promotion_reviews ORDER BY review_id"):
            try:
                report = jload(str(row["lab_report_json"]), {})
                validate_full_lab_report(
                    report,
                    current_repo_root,
                    require_current_release=(
                        str(row["release_content_hash"]) == current_release_hash
                    ),
                )
                evaluation = conn.execute(
                    "SELECT * FROM research_track_evaluations WHERE evaluation_id=?",
                    (row["evaluation_id"],),
                ).fetchone()
                identity = {
                    "policy_id": str(row["policy_id"]),
                    "policy_version": int(row["policy_version"]),
                    "definition_hash": str(row["definition_hash"]),
                    "release_content_hash": str(row["release_content_hash"]),
                    "evaluation_id": str(row["evaluation_id"]),
                    "reviewer_id": str(row["reviewer_id"]),
                    "lab_report_hash": str(row["lab_report_hash"]),
                }
                valid = all(
                    (
                        str(row["lab_report_json"]) == jdump(report),
                        str(row["lab_report_hash"])
                        == sha256(jdump(report).encode()).hexdigest(),
                        str(
                            (report.get("personalization_policy") or {}).get(
                                "promotion_content_hash"
                            )
                            or ""
                        )
                        == str(row["release_content_hash"]),
                        str(row["review_id"])
                        == "rtreview_" + sha256(jdump(identity).encode()).hexdigest()[:24],
                        evaluation is not None,
                        evaluation is not None
                        and str(evaluation["release_content_hash"])
                        == str(row["release_content_hash"]),
                        evaluation is not None
                        and str(evaluation["definition_hash"])
                        == str(row["definition_hash"]),
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                valid = False
            if not valid:
                invalid_reviews.append(str(row["review_id"]))
        for row in conn.execute("SELECT * FROM research_track_policy_events ORDER BY event_id"):
            valid = all(
                (
                    str(row["event_id"])
                    == research_policy_event_id(
                        row["policy_id"],
                        row["policy_version"],
                        row["action"],
                        row["actor_type"],
                        row["operator_id"],
                        row["reason"],
                        row["created_at"],
                    ),
                    (row["actor_type"] == "system" and row["operator_id"] is None)
                    or (row["actor_type"] == "operator" and bool(row["operator_id"])),
                )
            )
            if not valid:
                invalid_policy_events.append(str(row["event_id"]))
        snapshot_rows = conn.execute(
            "SELECT * FROM research_track_authorization_snapshots "
            "ORDER BY policy_id,policy_version,policy_authorization_epoch"
        ).fetchall()
        for row in snapshot_rows:
            try:
                policy = conn.execute(
                    "SELECT * FROM research_track_policies WHERE policy_id=? AND version=?",
                    (row["policy_id"], row["policy_version"]),
                ).fetchone()
                evaluation = conn.execute(
                    "SELECT * FROM research_track_evaluations WHERE evaluation_id=?",
                    (row["evaluation_id"],),
                ).fetchone()
                review = conn.execute(
                    "SELECT * FROM research_track_promotion_reviews WHERE review_id=?",
                    (row["promotion_review_id"],),
                ).fetchone()
                approval_event = conn.execute(
                    "SELECT * FROM research_track_policy_events WHERE event_id=?",
                    (row["approval_event_id"],),
                ).fetchone()
                activation_event = conn.execute(
                    "SELECT * FROM research_track_policy_events WHERE event_id=?",
                    (row["activation_event_id"],),
                ).fetchone()
                snapshot_authority = (
                    row["policy_id"], row["policy_version"], row["definition_hash"],
                    row["release_content_hash"], row["evaluation_id"],
                    row["promotion_review_id"], row["policy_authorization_epoch"],
                    row["approved_by_operator"], row["approval_reason"],
                    row["activation_reason"], row["approved_at"], row["activated_at"],
                    row["approval_event_id"], row["activation_event_id"], row["created_at"],
                )
                valid = all(
                    (
                        str(row["snapshot_hash"])
                        == research_authorization_snapshot_hash(*snapshot_authority),
                        str(row["snapshot_id"])
                        == research_authorization_snapshot_id(str(row["snapshot_hash"])),
                        str(row["created_at"]) == str(row["activated_at"]),
                        int(row["policy_authorization_epoch"]) > 0,
                        int(row["policy_authorization_epoch"]) % 2 == 1,
                        policy is not None,
                        policy is not None
                        and int(policy["authorization_epoch"])
                        >= int(row["policy_authorization_epoch"]),
                        evaluation is not None,
                        evaluation is not None
                        and str(evaluation["evaluation_id"]) not in invalid_evaluations,
                        evaluation is not None
                        and str(evaluation["policy_id"]) == str(row["policy_id"]),
                        evaluation is not None
                        and int(evaluation["policy_version"]) == int(row["policy_version"]),
                        evaluation is not None
                        and str(evaluation["definition_hash"]) == str(row["definition_hash"]),
                        evaluation is not None
                        and str(evaluation["release_content_hash"])
                        == str(row["release_content_hash"]),
                        review is not None,
                        review is not None and str(review["review_id"]) not in invalid_reviews,
                        review is not None and str(review["evaluation_id"]) == str(row["evaluation_id"]),
                        review is not None
                        and str(review["release_content_hash"])
                        == str(row["release_content_hash"]),
                        approval_event is not None,
                        approval_event is not None
                        and str(approval_event["event_id"]) not in invalid_policy_events,
                        approval_event is not None and approval_event["action"] == "approved",
                        approval_event is not None and approval_event["actor_type"] == "operator",
                        approval_event is not None
                        and approval_event["operator_id"] == row["approved_by_operator"],
                        approval_event is not None
                        and approval_event["reason"] == row["approval_reason"],
                        activation_event is not None,
                        activation_event is not None
                        and str(activation_event["event_id"]) not in invalid_policy_events,
                        activation_event is not None and activation_event["action"] == "activated",
                        activation_event is not None and activation_event["actor_type"] == "operator",
                        activation_event is not None
                        and activation_event["operator_id"] == row["approved_by_operator"],
                        activation_event is not None
                        and activation_event["reason"] == row["activation_reason"],
                    )
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                invalid_authorization_snapshots.append(str(row["snapshot_id"]))
        snapshot_groups: dict[tuple[str, int], list[sqlite3.Row]] = {}
        for row in snapshot_rows:
            snapshot_groups.setdefault(
                (str(row["policy_id"]), int(row["policy_version"])), []
            ).append(row)
        activated_identities = {
            (str(row["policy_id"]), int(row["policy_version"]))
            for row in conn.execute(
                "SELECT policy_id,policy_version FROM research_track_policy_events "
                "WHERE action='activated'"
            )
        }
        for identity in activated_identities | set(snapshot_groups):
            rows = snapshot_groups.get(identity, [])
            epochs = [int(row["policy_authorization_epoch"]) for row in rows]
            activation_event_ids = {
                str(row[0])
                for row in conn.execute(
                    "SELECT event_id FROM research_track_policy_events "
                    "WHERE policy_id=? AND policy_version=? AND action='activated'",
                    identity,
                )
            }
            if epochs != list(range(1, len(rows) * 2, 2)) or activation_event_ids != {
                str(row["activation_event_id"]) for row in rows
            }:
                invalid_authorization_snapshots.extend(
                    [str(row["snapshot_id"]) for row in rows]
                    or [f"missing:{identity[0]}:{identity[1]}"]
                )
        for row in conn.execute("SELECT * FROM research_uncertainty_traces ORDER BY trace_id"):
            try:
                evidence_ids = jload(str(row["evidence_ids_json"]), [])
                history = jload(str(row["history_json"]), {})
                effective_rows = [
                    dict(item)
                    for item in conn.execute(
                        """SELECT episode_id,result,assistance_level
                           FROM research_effective_knowledge_evidence
                           WHERE child_id=? AND concept_id=? AND concept_version=?
                           ORDER BY evidence_id""",
                        (row["child_id"], row["concept_id"], row["concept_version"]),
                    )
                ]
                linked_ids = [
                    str(item[0])
                    for item in conn.execute(
                        """SELECT evidence_id FROM learner_projection_evidence
                           WHERE child_id=? AND concept_id=? AND concept_version=?
                           ORDER BY evidence_id""",
                        (row["child_id"], row["concept_id"], row["concept_version"]),
                    )
                ]
                expected_history = history_from_evidence_rows(effective_rows)
                posterior = posterior_from_history(expected_history)
                projection = conn.execute(
                    """SELECT * FROM learner_state_projections
                       WHERE child_id=? AND concept_id=? AND concept_version=?""",
                    (row["child_id"], row["concept_id"], row["concept_version"]),
                ).fetchone()
                evaluation = conn.execute(
                    "SELECT * FROM research_track_evaluations WHERE evaluation_id=?",
                    (row["evaluation_id"],),
                ).fetchone()
                review = conn.execute(
                    "SELECT * FROM research_track_promotion_reviews WHERE review_id=?",
                    (row["promotion_review_id"],),
                ).fetchone()
                authorization_snapshot = conn.execute(
                    "SELECT * FROM research_track_authorization_snapshots WHERE snapshot_id=?",
                    (row["authorization_snapshot_id"],),
                ).fetchone()
                control_event = conn.execute(
                    """SELECT 1 FROM research_track_control_events
                       WHERE child_id=? AND enabled=1 AND policy_id=? AND policy_version=?
                         AND definition_hash=? AND release_content_hash=?
                         AND family_authorization_epoch=? AND actor_type='parent' LIMIT 1""",
                    (
                        row["child_id"],
                        row["policy_id"],
                        row["policy_version"],
                        row["definition_hash"],
                        row["release_content_hash"],
                        row["family_authorization_epoch"],
                    ),
                ).fetchone()
                activation_event = conn.execute(
                    """SELECT 1 FROM research_track_policy_events
                       WHERE policy_id=? AND policy_version=? AND action='activated'
                         AND actor_type='operator' LIMIT 1""",
                    (row["policy_id"], row["policy_version"]),
                ).fetchone()
                episode_count = len({str(item["episode_id"]) for item in effective_rows})
                expected_rationale = (
                    f"Advisory estimate from {episode_count} distinct eligible episodes linked to "
                    f"deterministic projection {row['projection_rule_id']}@"
                    f"{row['projection_rule_version']}; it cannot change state or deliver."
                )
                trace_authority = (
                    row["child_id"], row["concept_id"], row["concept_version"], row["construct"],
                    row["projection_input_digest"], row["projection_rule_id"],
                    row["projection_rule_version"], row["policy_id"], row["policy_version"],
                    row["definition_hash"], row["release_content_hash"], row["evaluation_id"],
                    row["promotion_review_id"], row["authorization_snapshot_id"],
                    row["family_authorization_epoch"],
                    row["policy_authorization_epoch"], row["evidence_digest"], row["history_json"],
                    row["alpha"], row["beta"], row["posterior_mean"], row["variance"],
                    row["interval_90_lower"], row["interval_90_upper"], row["uncertainty"],
                    row["method_version"], row["advisory_only"], row["rationale"],
                )
                valid = all(
                    (
                        projection is not None,
                        projection is not None
                        and str(projection["input_digest"]) == str(row["projection_input_digest"]),
                        projection is not None
                        and str(projection["rule_id"]) == str(row["projection_rule_id"]),
                        projection is not None
                        and str(projection["rule_version"]) == str(row["projection_rule_version"]),
                        str(row["construct"]) == "knowledge_performance",
                        str(row["definition_hash"]) == RESEARCH_POLICY_DEFINITION_HASH,
                        evaluation is not None,
                        evaluation is not None
                        and str(evaluation["evaluation_id"]) not in invalid_evaluations,
                        evaluation is not None
                        and str(evaluation["policy_id"]) == str(row["policy_id"]),
                        evaluation is not None
                        and int(evaluation["policy_version"]) == int(row["policy_version"]),
                        evaluation is not None
                        and str(evaluation["definition_hash"]) == str(row["definition_hash"]),
                        evaluation is not None
                        and str(evaluation["release_content_hash"])
                        == str(row["release_content_hash"]),
                        review is not None,
                        review is not None and str(review["review_id"]) not in invalid_reviews,
                        review is not None
                        and str(review["policy_id"]) == str(row["policy_id"]),
                        review is not None
                        and int(review["policy_version"]) == int(row["policy_version"]),
                        review is not None
                        and str(review["definition_hash"]) == str(row["definition_hash"]),
                        review is not None
                        and str(review["release_content_hash"])
                        == str(row["release_content_hash"]),
                        review is not None
                        and str(review["evaluation_id"]) == str(row["evaluation_id"]),
                        authorization_snapshot is not None,
                        authorization_snapshot is not None
                        and str(authorization_snapshot["snapshot_id"])
                        not in invalid_authorization_snapshots,
                        authorization_snapshot is not None
                        and str(authorization_snapshot["policy_id"]) == str(row["policy_id"]),
                        authorization_snapshot is not None
                        and int(authorization_snapshot["policy_version"])
                        == int(row["policy_version"]),
                        authorization_snapshot is not None
                        and str(authorization_snapshot["definition_hash"])
                        == str(row["definition_hash"]),
                        authorization_snapshot is not None
                        and str(authorization_snapshot["release_content_hash"])
                        == str(row["release_content_hash"]),
                        authorization_snapshot is not None
                        and str(authorization_snapshot["evaluation_id"])
                        == str(row["evaluation_id"]),
                        authorization_snapshot is not None
                        and str(authorization_snapshot["promotion_review_id"])
                        == str(row["promotion_review_id"]),
                        authorization_snapshot is not None
                        and int(authorization_snapshot["policy_authorization_epoch"])
                        == int(row["policy_authorization_epoch"]),
                        control_event is not None,
                        activation_event is not None,
                        int(row["family_authorization_epoch"]) > 0,
                        int(row["policy_authorization_epoch"]) > 0,
                        int(row["policy_authorization_epoch"]) % 2 == 1,
                        evidence_ids == linked_ids,
                        episode_count >= 3,
                        str(row["evidence_ids_json"]) == jdump(evidence_ids),
                        str(row["evidence_digest"])
                        == sha256(jdump(evidence_ids).encode()).hexdigest(),
                        history == expected_history,
                        str(row["history_json"]) == jdump(history),
                        str(row["method_version"]) == str(posterior["method"]),
                        all(float(row[key]) == float(posterior[key]) for key in (
                            "alpha", "beta", "posterior_mean", "variance",
                            "interval_90_lower", "interval_90_upper", "uncertainty",
                        )),
                        str(row["trace_hash"]) == authority_hash(*trace_authority),
                        str(row["trace_id"])
                        == "rtrace_" + sha256(str(row["trace_hash"]).encode()).hexdigest()[:24],
                        int(row["advisory_only"]) == 1,
                        str(row["rationale"]) == expected_rationale,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                valid = False
            if not valid:
                invalid_traces.append(str(row["trace_id"]))
        for row in conn.execute("SELECT * FROM research_proactivity_candidates ORDER BY candidate_id"):
            try:
                payload = jload(str(row["payload_json"]), {})
                trace = conn.execute(
                    "SELECT * FROM research_uncertainty_traces WHERE trace_id=?",
                    (row["trace_id"],),
                ).fetchone()
                interest = conn.execute(
                    """SELECT * FROM interest_projections
                       WHERE child_id=? AND concept_id=? AND concept_version=?""",
                    (row["child_id"], row["concept_id"], row["concept_version"]),
                ).fetchone()
                projection = conn.execute(
                    """SELECT state FROM learner_state_projections
                       WHERE child_id=? AND concept_id=? AND concept_version=?""",
                    (row["child_id"], row["concept_id"], row["concept_version"]),
                ).fetchone()
                concept = conn.execute(
                    "SELECT label FROM concepts WHERE concept_id=? AND version=?",
                    (row["concept_id"], row["concept_version"]),
                ).fetchone()
                control_event = conn.execute(
                    """SELECT 1 FROM research_track_control_events
                       WHERE child_id=? AND enabled=1 AND policy_id=? AND policy_version=?
                         AND definition_hash=? AND release_content_hash=?
                         AND family_authorization_epoch=? AND actor_type='parent' LIMIT 1""",
                    (
                        row["child_id"],
                        row["policy_id"],
                        row["policy_version"],
                        row["definition_hash"],
                        row["release_content_hash"],
                        row["family_authorization_epoch"],
                    ),
                ).fetchone()
                expected_title = (
                    f"Possible revisit: {str(concept['label'])[:120]}" if concept is not None else ""
                )
                expected_rationale = (
                    f"Candidate only: the current interest projection is {row['interest_phase']} "
                    f"and the advisory uncertainty interval is {float(trace['uncertainty']):.2f} "
                    "wide. A parent must inspect any future delivery design."
                    if trace is not None
                    else ""
                )
                candidate_authority = (
                    row["child_id"], row["trace_id"], row["concept_id"], row["concept_version"],
                    row["interest_input_digest"], row["interest_rule_id"],
                    row["interest_rule_version"], row["interest_phase"], row["policy_id"],
                    row["policy_version"], row["definition_hash"], row["release_content_hash"],
                    row["family_authorization_epoch"], row["policy_authorization_epoch"],
                    row["title"], row["rationale"], row["payload_json"], "candidate_only",
                )
                active_candidate = str(row["status"]) == "candidate_only"
                provenance_valid = all(
                    (
                        set(payload) == {
                            "concept", "trace_id", "evidence_ids", "deterministic_state",
                            "posterior_mean", "uncertainty", "interest_projection", "delivery",
                        },
                        payload.get("delivery") == "disabled",
                        payload.get("trace_id") == row["trace_id"],
                        dict(payload.get("concept") or {})
                        == {"id": row["concept_id"], "version": row["concept_version"]},
                        trace is not None,
                        concept is not None,
                        projection is not None,
                        control_event is not None,
                        trace is not None
                        and str(trace["child_id"]) == str(row["child_id"]),
                        trace is not None
                        and str(trace["concept_id"]) == str(row["concept_id"]),
                        trace is not None
                        and int(trace["concept_version"]) == int(row["concept_version"]),
                        trace is not None and str(trace["policy_id"]) == str(row["policy_id"]),
                        trace is not None
                        and int(trace["policy_version"]) == int(row["policy_version"]),
                        trace is not None
                        and str(trace["definition_hash"]) == str(row["definition_hash"]),
                        trace is not None
                        and str(trace["release_content_hash"])
                        == str(row["release_content_hash"]),
                        trace is not None
                        and int(trace["family_authorization_epoch"])
                        == int(row["family_authorization_epoch"]),
                        trace is not None
                        and int(trace["policy_authorization_epoch"])
                        == int(row["policy_authorization_epoch"]),
                        trace is not None and payload.get("evidence_ids") == jload(trace["evidence_ids_json"], []),
                        trace is not None and float(payload.get("posterior_mean")) == float(trace["posterior_mean"]),
                        trace is not None and float(payload.get("uncertainty")) == float(trace["uncertainty"]),
                        projection is not None and payload.get("deterministic_state") == projection["state"],
                        dict(payload.get("interest_projection") or {}) == {
                            "input_digest": row["interest_input_digest"],
                            "rule_id": row["interest_rule_id"],
                            "rule_version": row["interest_rule_version"],
                            "phase": row["interest_phase"],
                        },
                        str(row["title"]) == expected_title,
                        str(row["rationale"]) == expected_rationale,
                    )
                )
                current_interest_valid = bool(
                    interest is not None
                    and dict(payload.get("interest_projection") or {})
                    == {
                        "input_digest": interest["input_digest"],
                        "rule_id": interest["rule_id"],
                        "rule_version": interest["rule_version"],
                        "phase": interest["phase"],
                    }
                )
                valid = all(
                    (
                        str(row["payload_json"]) == jdump(payload),
                        str(row["candidate_hash"]) == authority_hash(*candidate_authority),
                        str(row["candidate_id"])
                        == "rtcand_" + sha256(str(row["candidate_hash"]).encode()).hexdigest()[:24],
                        provenance_valid,
                        not active_candidate or current_interest_valid,
                        row["invalidated_at"] is None
                        if active_candidate
                        else row["invalidated_at"] is not None,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                valid = False
            if not valid:
                invalid_candidates.append(str(row["candidate_id"]))
    policy = conn.execute(
        """SELECT *
           FROM research_track_policies WHERE policy_id=? AND version=?""",
        (RESEARCH_POLICY_ID, RESEARCH_POLICY_VERSION),
    ).fetchone()
    invalid_policy_state = True
    if (
        policy
        and not missing_columns
        and conn.execute("SELECT COUNT(*) FROM research_track_policies").fetchone()[0] == 1
    ):
        evaluation = (
            conn.execute(
                "SELECT * FROM research_track_evaluations WHERE evaluation_id=?",
                (policy["evaluation_id"],),
            ).fetchone()
            if policy["evaluation_id"] is not None
            else None
        )
        review = (
            conn.execute(
                "SELECT * FROM research_track_promotion_reviews WHERE review_id=?",
                (policy["promotion_review_id"],),
            ).fetchone()
            if policy["promotion_review_id"] is not None
            else None
        )
        status = str(policy["status"])
        evaluation_matches = bool(
            evaluation
            and str(evaluation["evaluation_id"]) not in invalid_evaluations
            and evaluation["policy_id"] == policy["policy_id"]
            and int(evaluation["policy_version"]) == int(policy["version"])
            and evaluation["definition_hash"] == policy["definition_hash"]
        )
        review_matches = bool(
            review
            and str(review["review_id"]) not in invalid_reviews
            and evaluation_matches
            and review["policy_id"] == policy["policy_id"]
            and int(review["policy_version"]) == int(policy["version"])
            and review["definition_hash"] == policy["definition_hash"]
            and review["evaluation_id"] == policy["evaluation_id"]
            and review["release_content_hash"] == policy["release_content_hash"]
            and review["reviewer_id"] != policy["approved_by_operator"]
        )
        if status == "candidate":
            status_valid = (
                policy["promotion_review_id"] is None
                and policy["approved_by_operator"] is None
                and policy["release_content_hash"] is None
                and (evaluation is None or str(evaluation["status"]) == "fail")
            )
        elif status == "evaluated":
            status_valid = bool(
                evaluation_matches
                and str(evaluation["status"]) == "pass"
                and policy["promotion_review_id"] is None
                and policy["approved_by_operator"] is None
                and policy["release_content_hash"] is None
            )
        else:
            status_valid = bool(
                status in {"approved", "active", "disabled", "retired"}
                and evaluation_matches
                and str(evaluation["status"]) == "pass"
                and review_matches
                and bool(policy["approved_at"])
                and (status != "active" or bool(policy["activated_at"]))
            )
        invalid_policy_state = not status_valid
    if shape_valid and policy:
        from .research_track import ResearchTrackAuthorizationChanged, ResearchTrackService

        validator = object.__new__(ResearchTrackService)
        validator.db_path = ""
        validator.repo_root = current_repo_root
        status = str(policy["status"])
        if not ResearchTrackService._policy_lifecycle_is_canonical(conn, policy):
            invalid_policy_state = True
        if status in {"approved", "active", "disabled", "retired"}:
            try:
                validator._trusted_promoted_policy(  # noqa: SLF001 - shared recovery gate
                    conn,
                    policy,
                    allowed_statuses={status},
                )
            except (ResearchTrackAuthorizationChanged, TypeError, ValueError):
                invalid_policy_state = True
        controls = conn.execute(
            "SELECT * FROM research_track_controls ORDER BY child_id"
        ).fetchall()
        controls_by_child = {str(row["child_id"]): row for row in controls}
        for row in controls:
            valid = ResearchTrackService._control_lineage_is_canonical(conn, row)
            if bool(row["enabled"]):
                collection = conn.execute(
                    "SELECT collection_enabled FROM learner_model_controls WHERE child_id=?",
                    (row["child_id"],),
                ).fetchone()
                valid = bool(
                    valid
                    and status == "active"
                    and not invalid_policy_state
                    and row["policy_id"] == policy["policy_id"]
                    and int(row["policy_version"]) == int(policy["version"])
                    and row["definition_hash"] == policy["definition_hash"]
                    and row["release_content_hash"] == policy["release_content_hash"]
                    and row["release_content_hash"] == current_release_hash
                    and row["updated_by"] == "parent"
                    and collection
                    and bool(collection["collection_enabled"])
                )
            if not valid:
                invalid_controls.append(str(row["child_id"]))
        orphan_control_events = {
            str(row["child_id"])
            for row in conn.execute("SELECT child_id FROM research_track_control_events")
            if str(row["child_id"]) not in controls_by_child
        }
        invalid_controls.extend(sorted(orphan_control_events))
    if (
        required_tables - tables
        or required_triggers - triggers
        or required_indexes - indexes
        or "research_effective_knowledge_evidence" not in views
        or missing_columns
        or missing_foreign_keys
        or invalid_primary_keys
        or invalid_not_null
        or invalid_indexes
        or missing_unique_constraints
        or missing_check_fragments
        or invalid_view
        or invalid_corpora
        or invalid_cases
        or invalid_evaluations
        or invalid_reviews
        or invalid_policy_events
        or invalid_authorization_snapshots
        or invalid_controls
        or invalid_traces
        or invalid_candidates
        or invalid_policy_state
        or not policy
        or str(policy["feature_schema_version"]) != RESEARCH_FEATURE_SCHEMA_VERSION
        or str(policy["definition_json"]) != jdump(RESEARCH_POLICY_DEFINITION)
        or str(policy["definition_hash"]) != RESEARCH_POLICY_DEFINITION_HASH
        or sha256(str(policy["definition_json"]).encode()).hexdigest() != str(policy["definition_hash"])
        or not conn.execute(
            """SELECT 1 FROM schema_migrations
               WHERE version=26 AND name='gated uncertainty and proactivity research track'"""
        ).fetchone()
        or int(conn.execute("PRAGMA user_version").fetchone()[0]) != max(current, 26)
        or conn.execute("PRAGMA foreign_key_check").fetchall()
    ):
        raise RuntimeError(
            "incomplete schema v26 research-track migration: "
            f"missing_tables={sorted(required_tables - tables)!r}, "
            f"missing_triggers={sorted(required_triggers - triggers)!r}, "
            f"missing_indexes={sorted(required_indexes - indexes)!r}, "
            f"missing_columns={missing_columns!r}, "
            f"missing_foreign_keys={missing_foreign_keys!r}, "
            f"invalid_primary_keys={invalid_primary_keys!r}, "
            f"invalid_not_null={invalid_not_null!r}, "
            f"invalid_indexes={invalid_indexes!r}, "
            f"missing_unique_constraints={missing_unique_constraints!r}, "
            f"missing_check_fragments={missing_check_fragments!r}, "
            f"invalid_view={invalid_view!r}, invalid_corpora={invalid_corpora!r}, "
            f"invalid_cases={invalid_cases!r}, invalid_evaluations={invalid_evaluations!r}, "
            f"invalid_reviews={invalid_reviews!r}, invalid_policy_events={invalid_policy_events!r}, "
            f"invalid_authorization_snapshots={invalid_authorization_snapshots!r}, "
            f"invalid_controls={invalid_controls!r}, invalid_traces={invalid_traces!r}, "
            f"invalid_candidates={invalid_candidates!r}, "
            f"invalid_policy_state={invalid_policy_state!r}"
        )


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _journal_path(value: str, root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise RuntimeError("refusing to recover a symlinked deletion-journal path")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise RuntimeError("deletion-journal path escaped its recorded private output root")
    return resolved


def _verify_journal_file(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("deletion-journal target is not a regular file")
    if _file_digest(path) != expected_sha256:
        raise RuntimeError("deletion-journal file digest changed; refusing destructive recovery")


def _recover_family_file_deletions(conn: sqlite3.Connection) -> None:
    """Finish or roll back every interrupted private-file erasure operation."""

    if "family_file_deletion_journal" not in _tables(conn):
        return
    rows = [
        dict(row)
        for row in conn.execute(
            """SELECT journal_id,deletion_id,event_id_hash,original_path,quarantine_path,
                      output_root,content_sha256,state
               FROM family_file_deletion_journal ORDER BY created_at,journal_id"""
        ).fetchall()
    ]
    if not rows:
        return
    live_event_hashes = {
        sha256(str(row[0]).encode()).hexdigest() for row in conn.execute("SELECT id FROM events").fetchall()
    }
    restored_deletions: set[str] = set()
    for row in rows:
        root = Path(str(row["output_root"])).expanduser().resolve(strict=False)
        original = _journal_path(str(row["original_path"]), root)
        quarantine = _journal_path(str(row["quarantine_path"]), root)
        expected = str(row["content_sha256"])
        event_exists = str(row["event_id_hash"]) in live_event_hashes
        original_exists = original.exists()
        quarantine_exists = quarantine.exists()

        if event_exists:
            restored_deletions.add(str(row["deletion_id"]))
            # The database deletion never committed. Restore the exact staged
            # bytes, or discard only a verified duplicate if restoration had
            # already completed before a second crash.
            if quarantine_exists:
                _verify_journal_file(quarantine, expected)
                if original_exists:
                    _verify_journal_file(original, expected)
                    quarantine.unlink()
                else:
                    original.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(quarantine, original)
            elif not original_exists:
                raise RuntimeError("interrupted deletion lost both original and quarantined private file")
        else:
            # The database deletion committed. Erase every verified surviving
            # copy, including the original if a crash happened during staging.
            if quarantine_exists:
                _verify_journal_file(quarantine, expected)
                quarantine.unlink()
            if original_exists:
                _verify_journal_file(original, expected)
                original.unlink()

        conn.execute(
            "DELETE FROM family_file_deletion_journal WHERE journal_id=?",
            (row["journal_id"],),
        )
        try:
            quarantine.parent.rmdir()
        except OSError:
            pass
    for deletion_id in restored_deletions:
        conn.execute(
            "DELETE FROM family_event_deletion_locks WHERE deletion_id=?",
            (deletion_id,),
        )


def init_db(db_path: str | Path, *, repo_root: str | Path | None = None) -> Path | None:
    """Create or migrate the database, returning a migration backup path when made."""

    path = Path(db_path)
    cache_key = _database_initialization_key(path, repo_root)
    with _INIT_LOCK:
        if (
            cache_key is not None
            and cache_key in _INITIALIZED_DATABASES
            and not _has_pending_family_file_deletions(path)
        ):
            return None
    version = 0
    if path.exists() and path.stat().st_size:
        probe = sqlite3.connect(path)
        try:
            version = int(probe.execute("PRAGMA user_version").fetchone()[0])
        finally:
            probe.close()
    if version >= 26:
        roots = authority_backup_roots(path)
        # Lifecycle serialization is the cross-process outer boundary. Taking
        # the process-only initialization lock first lets an ordinary artifact
        # thread and the visual worker deadlock while each waits for the other.
        with managed_data_lifecycle(path, extra_backup_roots=roots):
            with _INIT_LOCK:
                cache_key = _database_initialization_key(path, repo_root)
                if (
                    cache_key is not None
                    and cache_key in _INITIALIZED_DATABASES
                    and not _has_pending_family_file_deletions(path)
                ):
                    return None
                backup = _init_db_locked(path, repo_root=repo_root)
                _remember_initialized_database(path, repo_root)
                return backup
    with _INIT_LOCK:
        backup = _init_db_locked(path, repo_root=repo_root)
        _remember_initialized_database(path, repo_root)
        return backup


def _init_db_locked(
    db_path: str | Path, *, repo_root: str | Path | None = None
) -> Path | None:
    """Serialize backup and migration work inside this process."""

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
    # Bind the stable database identity before copying a legacy image so every
    # automatic migration backup can be authenticated against its live family.
    if has_tables:
        identity_conn = sqlite3.connect(path, timeout=30)
        try:
            identity_conn.execute("BEGIN IMMEDIATE")
            _ensure_family_database_identity(identity_conn)
            identity_conn.commit()
        except BaseException:
            identity_conn.rollback()
            raise
        finally:
            identity_conn.close()
    backup = _backup_existing(path, version) if has_tables and version < SCHEMA_VERSION else None
    if version >= 26:
        finalize_research_authority(path)
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        _ensure_legacy_columns(conn)
        _apply_migration_18(conn)
        _apply_migration_19(conn)
        _apply_migration_20(conn)
        _apply_migration_21(conn)
        _apply_migration_22(conn)
        _apply_migration_23(conn)
        _apply_migration_24(conn)
        _apply_migration_25(conn)
        _apply_migration_26(
            conn,
            Path(repo_root).resolve() if repo_root is not None else None,
        )
        conn.commit()
        finalize_research_authority(path)
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episode_members_episode ON episode_memberships(episode_id, created_at)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_episode ON evidence(episode_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_resources_unit ON resource_documents(unit_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bindings_lookup ON transport_bindings(transport,team_id,user_id,status)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inbox_status ON capture_inbox(status,created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outbox_ready ON delivery_outbox(transport,status,available_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visual_jobs_ready ON visual_jobs(status,available_at,created_at)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_slack_files_ready ON slack_file_outbox(status,available_at,created_at)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_onboarding_reviews_event ON onboarding_reviews(event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_event ON feedback(event_id,created_at)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_output_evaluations_event ON output_evaluations(event_id,created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_sessions_thread ON agent_sessions(binding_id,conversation_ref,thread_ref,updated_at)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_messages_session ON agent_messages(session_id,ordinal)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_preferences_session ON agent_session_preference_events(session_id,ordinal)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inbound_assets_event ON inbound_assets(binding_id,external_event_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inbound_assets_inbox ON inbound_assets(inbox_id,created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inbound_assets_session ON inbound_assets(session_id,created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id,created_at)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_status ON interaction_instances(binding_id,status,expires_at)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_schedule_runs_status ON schedule_runs(status,created_at)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_artifact_files_ready ON artifact_file_outbox(status,available_at,created_at)"
        )
        _recover_family_file_deletions(conn)
        initialize_or_assert_research_authority(path, conn)
        assert_no_resurrected_events(path, conn)
        sync_database_tombstones(path, conn)
        reconcile_managed_backups(path, conn)
    path.chmod(0o600)
    if backup:
        backup.chmod(0o600)
    return backup
