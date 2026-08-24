# Open-source stewardship loop

Curiosity Engine is designed to improve through real family use without turning family life into public product telemetry. Coding agents therefore alternate between two roles defined in `AGENTS.md`: the Family Operator and the Open-Source Steward.

## When to run the steward review

Run it after a meaningful setup session, a real answer that needed tuning, a privacy or operations failure, a new family resource pattern, or before ending a development session with tracked changes. It is also reasonable to run once at the end of a day. Do not run it continuously against private family activity.

## Review inputs

Use the current conversation, `git status`, the tracked diff, recent local commits, public tests/evals, and feedback the parent has deliberately surfaced. Read private material only when necessary to understand a specific issue. Never enumerate or summarize private resources merely because they exist.

## Classification

Put every finding in one bucket:

1. **Public contribution prepared** — a generic code, documentation, test, schema, provider, or extension improvement with synthetic evidence.
2. **Family-only customization retained** — preferences, names, schedules, provider keys/routes, purchased resources, private evals, or a change too specific to generalize safely.
3. **Needs more evidence** — a possible pattern seen only once or one that cannot yet be tested without private details.

## Contribution workflow

1. State the abstract problem without family identifiers.
2. Create the smallest generic contract or extension point that solves it.
3. Add a synthetic regression or public eval.
4. Run the full public quality and privacy gates.
5. Prepare a local commit and pull-request description that names no private source.
6. Ask before pushing, publishing, opening a pull request, or creating an issue.

Suggested handoff:

```text
Public contribution prepared:
- ...

Family-only customization retained:
- ...

Ideas needing more evidence:
- ...

Verification:
- ...
```

The open-source project should learn from the shape of family problems, never from publishing the family itself.
