# Local operations

## Install and diagnose

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack,all-providers]'
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

## Brain configuration and verification

The fixed Slack `connection` test is model-free. Supported API brains use ignored `private/setup/brain.json` plus owner-only `private/setup/model.env`; direct OpenAI, native Anthropic, and OpenRouter are available. Legacy `CURIOSITY_BACKEND`, `CURIOSITY_MODEL`, and OpenAI-only env files remain readable for one compatibility window.

```bash
curiosity brain doctor
curiosity brain test --live
curiosity onboard status
curiosity onboard pending
```

The live probe is explicit and billable but family-data-free. A rejected/invalid model response is never completed and proposed effects are not applied. Enabling providers discloses bounded request context to them; private excerpts remain separately opt-in. See `BRAIN_SETUP.md` for multimodal evals that the structured probe does not cover.

The attended coding-agent path still needs a custom backend and a live authenticated terminal; it is not an operational substitute for the API path.

Inspect episode-aware context locally with `curiosity context --child CHILD_ID`. This output contains family text and should not be pasted into public issues. Context-driven reflection is release-disabled regardless of an older household opt-in.

## Verification and open-source audit

```bash
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
curiosity ecosystem status
python -m compileall -q src
git status --short --ignored
git ls-files private .env data output .Codex
```

When the curated registry is due or an upstream project changes ownership, license, archive state, or major version, run `curiosity ecosystem check --live` and complete the manual gate in `PUBLIC_PROJECTS.md`. The command checks public metadata only and never authorizes an install or upgrade.

The last command must print nothing. Before sharing logs or issues, search tracked files for actual family names, emails, workspace identifiers, provider-specific purchased-resource names, and copied excerpts. Do not paste private `doctor`-adjacent command output if another command includes raw binding IDs or family state.
