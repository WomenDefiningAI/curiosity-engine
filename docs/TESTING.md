# Testing and evaluation

## Local quality gate

```bash
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
```

`pytest` covers contracts, migrations, event idempotency, critic rejection, context expiry, private resource gating, the OpenAI request shape, PDF/hash approval, the web flow, household setup, weekly opt-in, exact Slack pairing, replay/conflict handling, unassigned attribution, outbox ambiguity, token redaction, and least-privilege manifest scope.

The Lab separately executes 30 public behavioral cases across:

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

The regression suite includes duplicate-event effects, conflicting idempotency keys, graph-preserving child updates, failed critics, expired school state, private excerpt opt-in, action proposal boundaries, print approvals, one-page PDFs, Tier C fail-closed behavior, claim evidence thresholds, Director bounds, and migration backups.

Credential-gated live Slack verification is local-only: pair a test parent/conversation, send `children`, an explicit `ask`, and an unattributed note, then revoke the binding. Tokens, Slack IDs, family messages, and responses never belong in CI artifacts.

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
