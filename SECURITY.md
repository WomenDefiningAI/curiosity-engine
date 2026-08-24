# Security and privacy reports

Curiosity Engine handles parent messages, child context, provider credentials, Slack credentials, and optionally licensed resources. Please do not disclose vulnerabilities—or any real family data—in a public issue.

## Report privately

Use [GitHub private vulnerability reporting](https://github.com/WomenDefiningAI/curiosity-engine/security/advisories/new). Include a minimal synthetic reproduction, affected version/commit, expected impact, and suggested mitigation if known.

Never attach real names, child profiles, questions, Slack IDs, tokens, provider keys, local databases, purchased-resource titles/URLs/excerpts, generated family outputs, or private evaluation reports. Replace them with synthetic fixtures.

If private vulnerability reporting is temporarily unavailable, open a public issue containing only “Private security contact needed” and no vulnerability details or private data.

## Supported version

Before the first stable release, only the latest commit on `main` receives security fixes. This policy will be revised when versioned releases are published.

## Scope reminders

- `private/`, `.env`, and `.Codex/` must remain untracked.
- Slack and model credentials belong only in owner-readable ignored files.
- The local connector must not expose its setup console beyond loopback.
- Provider privacy depends on both this harness's request boundary and the current terms of every selected provider/downstream route.
