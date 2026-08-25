# Testing and evaluation

## Public quality gate

```bash
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
```

Tests cover contracts, migrations, idempotency, retries, context episodes/corrections, disabled proactivity, private-resource gates, provider request shapes, artifact trust, visual policy/rendering, Slack text/file crash boundaries, staged onboarding, redaction, and the loopback console.

Tests force deterministic mode, ignore private provider configuration, and make no billable calls. The Lab runs public behavioral cases for golden behavior, curiosity, context, safety, uncertainty, parent effort, artifact trust, and regressions.

An offline pass proves orchestration and known invariants—not production-model quality. It records the independent live judge as `not_run`, so model promotion remains ineligible.

## Live checks

Run live verification locally and in this order:

1. paired Slack `connection`;
2. paired Slack `visual connection`;
3. family-data-free provider probe;
4. optional paid `curiosity visual test --live` for decorative mode;
5. synthetic vision/OCR checks;
6. one real question with parent review.

Private messages, files, responses, keys, Slack IDs, and licensed excerpts never belong in CI artifacts. Live Lab judging uses only public fixtures:

```bash
CURIOSITY_EVAL_LIVE=1 curiosity-lab \
  --repo . --live-judge \
  --json-out private/live-eval-report.json
```

Behavior promotion requires all deterministic suites, the minimum case count, an independent live judge, no factual/safety/golden regression, and operator approval. Convert real failures to de-identified public regressions only when privacy and licensing allow.

CI runs Ruff, pytest, the offline Lab, and CodeQL without provider keys. `curiosity ecosystem check --live` is a separate metadata alarm and never approves an integration.
