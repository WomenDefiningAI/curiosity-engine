# Evidence-spine migration and rebuild runbook

Schema 18 adds the Phase 1 evidence spine without rewriting legacy learner data.

## Automatic migration

`init_db()` performs these steps:

1. reject databases newer than the running code;
2. create an owner-only SQLite backup when an older non-empty database is present;
3. apply the aggregate legacy schema and required historical columns;
4. apply the v18 evidence-spine DDL inside one `BEGIN IMMEDIATE` transaction;
5. record `schema_migrations.version=18`, set `PRAGMA user_version=18`, and verify every evidence-spine table, required
   index, key opportunity columns, the exact ledger entry, and all foreign keys;
6. retain the backup beside the database with a `backup-v17-...` suffix.

The v18 DDL is idempotent. If a database claims v18 but is missing an evidence-spine object or ledger row,
initialization re-applies the additive migration and verifies it. Initialization is serialized inside the process so
concurrent startup cannot race backup creation or DDL. Any migration error rolls the v18 transaction back. No legacy
observation, graph node, edge, claim, or raw `evidence` row is copied into the new tables.

## Verification

Use an offline copy or the normal test database; do not inspect family text for a schema check.

```bash
sqlite3 private/data/curiosity.db 'PRAGMA user_version; PRAGMA integrity_check; PRAGMA foreign_key_check;'
sqlite3 private/data/curiosity.db \
  "SELECT version,name,applied_at FROM schema_migrations WHERE version=18;"
```

Expected results are version `18`, integrity `ok`, no foreign-key rows, and one migration-ledger row. The automated
suite additionally migrates a synthetic v17 database and verifies preservation, backup creation, the full object set,
integrity, foreign keys, concurrent initialization, partial-state repair, idempotent resume, and rollback on injected
DDL failure.

## Rebuild

`recompute_child_projections(db_path, child_id)` performs a full deterministic rebuild from effective Phase 1 evidence.
It is safe to repeat. The public API records an input and output digest in `projection_runs`; an unchanged input produces
the same result. Corrections and deletion APIs call the in-transaction form automatically.

Do not repair projections by editing their rows. Correct or exclude source evidence, or delete the learning source, and
let the projector rebuild. Expiry-aware reads accept an explicit `as_of`; `expire_learning_records()` performs scheduled
cleanup of expired audit annotations and transient opportunities, cancels their plans, and rebuilds affected children.

## Recovery

Stop the service before database recovery. Verify the automatic backup before using it. Restore to a new path first,
run integrity and foreign-key checks, then deliberately change the configured database path. Never overwrite the only
copy or delete the newer database during diagnosis. A v17 recovery copy will be backed up and migrated again when
opened by v18 code.
