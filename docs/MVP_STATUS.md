# MVP status

Release target: `0.1.0-slack-mvp`

## Implemented vertical slice

| Requirement | Implementation | Verification |
|---|---|---|
| New-family setup | Coding-agent contract, parent guide, deterministic setup/doctor commands | CLI and privacy tests |
| Slack parent entry | Socket Mode connector for DMs and explicit mentions | Transport unit tests; live workspace smoke test remains local |
| Parent authorization | Short-lived pairing code bound to workspace, user, and conversation | Expiry, replay, exact-binding tests |
| Uncertain attribution | Durable unassigned inbox with explicit assign/dismiss | Inbox and Slack command tests |
| Durable capture | Immutable event, raw evidence, job, run, response | Unit and regression cases |
| Exactly-once behavior | Payload hashes, transport receipts, event/action/outbox idempotency | Duplicate and conflict tests |
| Family context | Depth 0–4, relevance ranking, expiry and evidence gates | Context and regression suites |
| Purchased resources | Generic private catalog and metadata/excerpt gate | Inventory and retrieval tests |
| Learning response | Strict Pull-the-Thread output and bounded critic/revision policy | Golden, factual, safety suites |
| Paper extension | Ordinary materials and optional one-page PDF | Contract and PDF validation |
| Parent control | Parent-selected artifacts, exact-byte approval, printing optional | Browser and tamper tests |
| Weekly suggestion | At most one low-effort result or do nothing; opt-in during setup | Autonomy tests |
| Local review | Loopback setup/review console and full CLI | HTTP and service tests |

## Family setup state

Family state is deliberately not reported in this public document. Run `curiosity doctor` locally for privacy-safe readiness categories and `curiosity slack status` for local onboarding state. Those commands do not display token values, child questions, or licensed excerpts.

## Intentionally outside v1

Telegram may be added later through the normalized transport boundary. WhatsApp, cloud multi-tenancy, child-facing accounts, autonomous purchasing, automatic outbound campaigns, specialized fabrication, and automatic release promotion require separate acceptance plans.

An offline test pass proves deterministic workflow invariants. It does not prove production-model family quality. Live Slack verification, optional model evaluation, and parent review remain explicit local checkpoints.
