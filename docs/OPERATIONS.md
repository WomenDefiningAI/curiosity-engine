# Local operations

## Install and diagnose

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack]'
curiosity doctor
```

Poppler is optional for PDF preview/QA (`brew install poppler` on macOS; `poppler-utils` on common Linux distributions). A conventional print command is needed only if a parent chooses direct printing; PDFs can always be downloaded instead.

## Configure and run

```bash
curiosity setup --owner-name "PRIVATE NAME" --timezone "America/New_York"
curiosity child add --id kid-a --name "PRIVATE NAME"
curiosity slack pair-code
curiosity slack run
```

Follow `SLACK_SETUP.md` before pairing. The connector remains in the foreground and should be stopped with Control-C. It does not expose the local console publicly.

For local setup/review:

```bash
curiosity serve --host 127.0.0.1 --port 8766
```

## Status and access recovery

```bash
curiosity doctor --write-report
curiosity slack status
curiosity slack bindings
curiosity inbox list
```

Revoke one exact Slack binding with `curiosity slack revoke --binding ID`. If a token may be exposed, rotate it in Slack and update only `private/setup/slack.env` with owner-only permissions.

## Durable work

Interactive requests synchronously process their own job. Resume any ready jobs and due parent-enabled schedules with:

```bash
curiosity worker --drain
```

Jobs have leases, bounded retries, and exponential delay. Events, graph effects, actions, Slack receipts, and Slack deliveries have independent idempotency boundaries.

## Database migration and recovery

`curiosity init` and setup commands migrate in place. Before an older non-empty database is changed, the engine creates a sibling backup named `curiosity.db.backup-v<old>-<UTC timestamp>`. Stop the connector and console before manual recovery. Preserve the current database, copy a chosen backup to a new filename, and point `CURIOSITY_DB` at it.

## Optional model mode

Deterministic mode is the default setup/demo mode and is visibly labeled in Slack. OpenAI mode reads `CURIOSITY_BACKEND` and `OPENAI_API_KEY` from owner-only `private/setup/model.env` or the process environment. A rejected model response is never shown as completed and its proposed effects are not applied. Enabling a provider discloses the bounded request context to it; private excerpts remain separately opt-in.

## Verification and open-source audit

```bash
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
python -m compileall -q src
git status --short --ignored
git ls-files private .env data output .Codex
```

The last command must print nothing. Before sharing logs or issues, search tracked files for actual family names, emails, workspace identifiers, provider-specific purchased-resource names, and copied excerpts. Do not paste private `doctor`-adjacent command output if another command includes raw binding IDs or family state.
