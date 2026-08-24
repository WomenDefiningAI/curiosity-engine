# Contributing

Thank you for helping families turn everyday questions into thoughtful learning threads. Curiosity Engine is early-stage, local-first software; privacy and parent control are release requirements, not optional polish.

## Set up a development checkout

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack,all-providers]'
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
```

All ordinary tests must be offline, deterministic, and safe without credentials. Live provider or public-metadata checks must be explicit and must never use family data.

## Public-data boundary

Use synthetic people, events, questions, worksheets, screenshots, and outputs in code, tests, docs, issues, and pull requests. Never contribute:

- real names, child details, questions, Slack/workspace identifiers, or contact information;
- tokens, keys, `.env` contents, databases, logs, or generated family artifacts;
- purchased-resource names, private URLs, copied excerpts, screenshots, or derivative answer keys;
- council reviews, coding-agent working notes, or private evaluation reports.

Confirm `git ls-files private .env data output .Codex` prints nothing before committing. If a secret or family detail reaches Git history, rotate the credential when relevant and report privately; deleting the current file is not enough.

## Behavioral changes

- Add a de-identified regression for every safely reproducible behavior fix.
- Treat node/edge frequency as mention count, not interest or mastery evidence.
- Durable claims need independent eligible episodes; retries, clarification, and answer repair must not inflate them.
- Keep context-driven unsolicited suggestions disabled unless a future reviewed proposal includes shadow-mode evidence, parent controls, and safety tests.
- Preserve code-owned schemas, critics, privacy gates, idempotency, and side-effect approval boundaries.

## Dependencies and external projects

Do not add, install, or execute a public project merely because it appears useful. Follow `docs/PUBLIC_PROJECTS.md`: verify canonical ownership, license, maintenance, releases, supply-chain posture, privacy implications, and elementary-family fit. Prefer reference-only integration and pinned declared dependencies.

## Pull-request checklist

- Explain the family problem and the general solution without private examples.
- State privacy/security impact and whether provider disclosure changes.
- Add or update tests and public docs.
- Run Ruff, pytest, the offline Lab, package build, and secret/privacy checks.
- Keep provider-specific live results and family quality reviews local unless a separate synthetic public fixture exists.

By contributing, you agree that your contribution is licensed under the repository's MIT License.
