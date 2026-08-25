# Publishing checklist

Public changes must contain reusable code and synthetic evidence only. Never publish `private/`, credentials, family details, purchased-resource details, generated family output, or private evals.

## Before a push or release

```bash
git status --short
git ls-files private .env data output .Codex
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
python -m compileall -q src tests
python -m pip check
python -m pip wheel . --no-deps --wheel-dir private/release-wheel
gitleaks git --redact --no-banner
```

The tracked-private-path command must print nothing. Review the complete diff for identifiers, secrets, local paths, licensed text, and private examples. An offline Lab result with `promotion_eligible=false` can support an honestly labeled alpha source change, but not a model promotion.

## Human authorization

A coding agent may prepare a branch, commit, and PR text locally. It must ask before pushing, publishing, opening a PR/issue, changing GitHub settings, creating a repository/release, or contacting anyone.

For a new public repository, confirm the exact owner/name, public visibility, license, description, topics, conduct contact, and administrators before creation.

## GitHub safeguards

- private vulnerability reporting;
- secret scanning, push protection, and validity checks;
- Dependabot alerts and updates;
- CodeQL and behavioral CI;
- read-only Actions by default;
- protected `main`: PRs, required checks/review, no force push or deletion.

After publication, inspect as a logged-out visitor and test a clean clone. Search for family identifiers, keys, Slack IDs, purchased-resource references, and local paths. If a secret or private data reached history, stop, rotate affected credentials, and use a deliberate history-rewrite/disclosure process.

Tag a release only after hosted CI, clean-clone setup, required provider probes, and operator approval pass. Family answer reviews remain local.
