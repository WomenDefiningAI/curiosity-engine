# Coding-agent role

Codex or Claude Code is the setup and customization operator, not the always-on brain behind Slack.

The no-clone installer creates `~/.curiosity-engine/workspace/` with private `AGENTS.md` and `SETUP_HANDOFF.md` files.
The coding agent should operate the terminal, explain choices, configure Slack and model APIs, run tests, install the
local services, and verify a real family answer. Credentials must be entered only into owner-only ignored files.

The supported runtime calls model APIs through provider adapters. This remains responsive while the coding-agent
terminal is closed; only the family computer and local services must stay running.

An advanced family may build a coding-agent `ModelBackend`, but it must accept only bounded context, run unattended
with explicit timeouts, satisfy the same typed contracts, have no direct side-effect authority, and pass the same
public evals. A consumer coding-agent subscription should never be assumed to grant application API or unattended use.

At session end, the agent’s second role is Open Source Steward: extract general, privacy-safe improvements and
synthetic regressions. Family transcripts, purchased resources, credentials, and private context never become a
contribution.
