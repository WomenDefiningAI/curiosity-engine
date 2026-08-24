# Curiosity Engine roadmap

## v0.1 — Agent-led Slack MVP

- [x] New-family instructions for Codex, Claude Code, and comparable coding agents
- [x] Parent-readable architecture, privacy, setup, restart, and customization walkthrough
- [x] Deterministic `doctor` and `setup` commands with privacy-safe JSON output
- [x] Household owner, co-parent records, timezone, quiet hours, and weekly opt-in setting
- [x] Free family Slack workspace guide and least-privilege app manifest
- [x] Slack Socket Mode connector for parent DMs and explicit mentions
- [x] Short-lived pairing, exact user/conversation binding, revocation, and duplicate receipts
- [x] Unassigned capture inbox; child attribution is explicit and never guessed
- [x] Durable response outbox with bounded retry and ambiguous-delivery state
- [x] Shared service behind Slack, CLI, and loopback setup/review console
- [x] Bounded local family context and ignored purchased-resource collection
- [x] Ordinary household-material activities and optional downloadable paper PDFs
- [x] Parent-enabled weekly reflection with at most one active family suggestion
- [x] Public privacy, migration, Slack, workflow, and behavioral regression tests

## v0.2 — Learning-loop hardening

- [x] Lightweight parent feedback from Slack
- [ ] Slack accept/dismiss interactions for offered suggestions
- [ ] A parent-controlled paper-page request from Slack
- [ ] Background connector installer for supported home-computer platforms
- [ ] Clear local export and deletion walkthroughs
- [ ] Topic entity resolution and a visible curiosity ladder
- [ ] Trusted current-resource research with source provenance
- [ ] De-identified regression capture from parent corrections
- [ ] Broader factual, context, safety, and parent-effort evaluation coverage

## Later, after evidence from real family use

- Telegram through the same transport contracts
- Optional external context sources selected and authorized by each family
- Broader parent roles or hosted operation under a separate threat model
- Longitudinal context audits and release-comparison automation

WhatsApp, cloud multi-tenancy, child-facing chat, autonomous purchasing, automatic outbound campaigns, and specialized fabrication are outside this MVP.

## v1 outcome

A parent can clone the repository, point a coding agent at it, understand what stays local, connect a free family Slack workspace, add only the context they choose, capture a real question, receive a useful next thread, optionally download a trustworthy paper activity, and give feedback without exposing private family or purchased-resource data in the public repository.
