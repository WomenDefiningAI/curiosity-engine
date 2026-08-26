# Local operations

## Start and stop

```bash
curiosity host install
curiosity host status
```

On Linux, this installs restartable owner-level Slack and scheduler services. Sleep, shutdown, or lost networking still
stops replies. Without systemd, run `curiosity slack run` and `curiosity worker --forever` in supervised terminals.
The optional local console is loopback-only:

```bash
curiosity serve --host 127.0.0.1 --port 8766
```

## Diagnose

```bash
curiosity doctor --write-report
curiosity slack status
curiosity slack bindings
curiosity inbox list
curiosity brain doctor
curiosity onboard status
```

These commands are redacted except `slack bindings`, which contains private IDs. Do not paste private command output into public issues.

`curiosity doctor` includes a rolling `answer_quality` summary with no question or answer text. `attention` means at least 20% of five or more recent questions were rejected or failed; inspect before asking a parent to retry.

Revoke access with `curiosity slack revoke --binding ID`. If a token may be exposed, rotate it at the provider and update only the owner-readable ignored env file.

## Back up and recover

```bash
curiosity backup create
curiosity backup status
curiosity backup verify                    # latest snapshot
curiosity backup restore                   # separate recovery copy
```

Snapshots default to a sibling of the family home, normally `~/.curiosity-engine-family-backups/`. Set
`CURIOSITY_BACKUP_DIR` or pass `--destination` to use another location. Each snapshot contains the SQLite database,
private resources, generated output, and non-secret setup state. It excludes Slack/model credentials, uses owner-only
permissions, and has file checksums plus a SQLite integrity check.

Restore verifies first, rebases stored resource/output paths for the new location, and never overwrites an existing path. By default it creates `private/restores/SNAPSHOT_ID`; use `--target-private NEW_PATH` for a different new path. To recover an empty clone directly, pass its not-yet-created `private/` path, then re-enter credentials and run `curiosity doctor`.

These snapshots protect against accidental repository cleanup, not disk loss. Include the backup directory in an encrypted system backup such as Time Machine.

## Jobs and database migrations

```bash
curiosity worker --drain
```

Jobs use leases, bounded retries, and independent idempotency boundaries. Setup commands migrate the database in place and first make a database-only migration backup beside it as `curiosity.db.backup-v<old>-<timestamp>`. That is not a substitute for a full family snapshot.

Inspect context locally with `curiosity context --child CHILD_ID`. It contains family text. Context-driven model calls and unsolicited suggestions remain disabled in v1.

## Public audit

```bash
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
curiosity ecosystem status
git status --short --ignored
git ls-files private .env data output .Codex
```

The last command must print nothing. Use `curiosity ecosystem check --live` only as a public-metadata alarm; integrations still require [manual review](PUBLIC_PROJECTS.md).
