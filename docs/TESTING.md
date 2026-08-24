# Testing and evaluation

## Local quality gate

```bash
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
```

`pytest` covers contracts, migrations, event idempotency, critic rejection, context expiry, episode grouping, answer repair, diagnostic exclusion, claim independence, parent corrections, disabled proactivity, private resource gating, distinct provider request shapes, PDF/hash approval, the web flow, household setup, staged onboarding, confirmed-delivery review binding, exact Slack pairing, fixed non-model connection handling, replay/conflict handling, unassigned attribution, outbox ambiguity, token redaction, and least-privilege manifest scope.

Tests force deterministic mode and ignored empty provider paths. They never inherit a family's `private/setup/brain.json` or `model.env`, and never make billable provider calls.

The Lab separately executes 32 public behavioral cases across:

- golden behavior;
- curiosity preservation;
- context use;
- harness policy;
- bounded autonomy;
- factual uncertainty;
- parent effort;
- social safety;
- artifact trust;
- permanent regressions.

The regression suite includes duplicate-event effects, conflicting idempotency keys, graph-preserving child updates, failed critics, expired school state, private excerpt opt-in, action proposal boundaries, print approvals, one-page PDFs, Tier C fail-closed behavior, claim evidence thresholds, Director bounds, and migration backups. Deterministic episode invariants live in `tests/test_context_episodes.py`; they are release-blocking even though they are not model-scored Lab cases.

Credential-gated live verification is local-only and ordered: paired Slack `connection`; family-data-free provider probe; synthetic vision/OCR/image checklist; then one real question with parent review. Tokens, Slack IDs, family messages, images, and responses never belong in CI artifacts.

The curated public-project catalog is schema-validated offline. `curiosity ecosystem check --live` is a separate explicit public-metadata alarm: it must never clone, install, import, execute, or send private data to an upstream project. A passing metadata check does not promote a project or replace the manual gate in `PUBLIC_PROJECTS.md`.

## What an offline pass means

Offline mode runs deterministic workflows and executable invariants. It verifies orchestration and known semantic properties, but it is not an independent assessment of a production model. The report therefore records the live judge as `not_run` and sets `promotion_eligible=false`.

## Live semantic judge

```bash
export OPENAI_API_KEY='...'
export CURIOSITY_EVAL_LIVE=1
curiosity-lab --repo . --live-judge --json-out private/live-eval-report.json
```

Only public eval inputs and candidate outputs are sent. Private family rows and licensed excerpts never enter the Lab. The judge uses the case’s `must` / `must_not` rubric and independently fails factual, safety, curiosity, context, or parent-effort misses.

## Promotion discipline

The Lab compares release metadata and refuses automatic promotion. A viable promotion requires:

1. no required suite missing;
2. every deterministic case passing;
3. at least the configured minimum case count;
4. the live judge passing;
5. deliberate operator approval.

Any production miss becomes a public, de-identified regression fixture whenever licensing and privacy permit. Copyrighted resource excerpts and family-identifying material belong only in ignored local audits, never the public suite.

## CI

GitHub Actions installs the editable package with dev dependencies, runs Ruff, runs pytest, executes the Lab, and uploads the JSON report. No model key is required in pull-request CI.
