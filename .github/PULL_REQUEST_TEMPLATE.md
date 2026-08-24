## What and why

Describe the general family problem and the change. Use synthetic examples only.

## Safety and privacy

- [ ] No family data, Slack IDs, credentials, licensed-resource details/excerpts, generated family outputs, council reviews, or coding-agent working notes are included.
- [ ] Provider disclosure, permissions, storage, and side effects are unchanged or explained.
- [ ] Context frequency is not treated as independent interest/mastery evidence.

## Verification

- [ ] `ruff check .`
- [ ] `pytest`
- [ ] `curiosity-lab --repo . --json-out private/eval-report.json`
- [ ] Relevant docs and synthetic regressions were updated.
- [ ] `git ls-files private .env data output .Codex` prints nothing.
