# Experimental attended coding-agent operation

The supported Curiosity Engine runtime calls LLM APIs through explicit provider adapters. A family can instead experiment with an already authenticated Codex, Claude Code, or similar coding-agent session, but this is a customization path—not a dependable unattended Slack backend in v1.

## What works without a custom backend

Point the coding agent at the entire repository and ask it to:

- read `AGENTS.md` and `docs/AGENT_ONBOARDING.md`;
- install and diagnose the local harness;
- configure and run Slack;
- run `curiosity worker --drain`, tests, and evals;
- inspect public code plus redacted status;
- implement family-specific changes only under ignored private paths;
- propose generic public improvements through the stewardship loop.

The agent can operate the repository while its session is alive. It does not automatically become the model behind every Slack answer.

## What a custom coding-agent backend would require

A family choosing this route must have its coding agent implement and maintain a `ModelBackend` adapter that:

1. receives only the harness's bounded request context;
2. invokes the coding CLI non-interactively under an authenticated session;
3. enforces the Pydantic response schema and fails closed;
4. applies explicit timeouts, concurrency limits, and error handling;
5. never grants the model direct authority over SQLite or side effects;
6. records the CLI/model/version used without storing prompts or responses in tracked files;
7. passes the same public evals and real parent review as an API route.

Do not assume a consumer coding-agent subscription grants application API use or unattended automation. Check that product's current terms and account behavior. Do not work around login, rate, or session controls.

## Reliability boundary

This route adds one more live dependency:

```text
family computer awake
  + Slack connector process alive
  + coding-agent terminal/session alive and authenticated
  + custom adapter compatible with the current CLI
```

Closing the terminal, logging out, session expiry, CLI updates, or a stopped computer can break replies. The API runtime still needs the local computer and Slack connector, but it does not need an attended coding-agent login. That is why APIs remain the project recommendation.

If a coding-agent adapter reveals a generally useful pattern, the Open-Source Steward may prepare a generic provider extension and synthetic tests. Never contribute the family's session data, transcripts, credentials, or private model context.
