# Local operations

## Start and stop

```bash
source .venv/bin/activate
curiosity slack run       # leave in foreground
```

Control-C, terminal closure, sleep, shutdown, or lost networking stops replies. The optional local console is loopback-only:

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

## Jobs and recovery

```bash
curiosity worker --drain
```

Jobs use leases, bounded retries, and independent idempotency boundaries. Setup commands migrate the database in place and first back up an older non-empty database beside it as `curiosity.db.backup-v<old>-<timestamp>`.

For manual recovery, stop the connector/console, preserve the current database, copy a backup to a new filename, and point `CURIOSITY_DB` to it.

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
