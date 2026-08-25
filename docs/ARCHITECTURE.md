# Architecture

```text
parent in Slack ─► Socket Mode shell ─► normalized InboundMessage
                                              │
local console / CLI ──────────────────────────┤
                                              ▼
                                  shared CuriosityService
                                              │
                 pairing • receipts • inbox • immutable Event
                                              ▼
┌──────────────────────── SQLite boundary ───────────────────────┐
│ private family graph • jobs/runs • responses • durable outbox  │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
             bounded context builder (depth + relevance)
       family evidence • active school state • private metadata
                              │
                              ▼
          generator → adversarial critics → bounded revision
                              │
                    strict local validation
                              ▼
                 allowlisted effects and response
                              │
            optional paper PDF • parent review • feedback
                              │
                              └────────► Slack text outbox
                                           │
                    visual intent → trust policy → private PNG
                                           └────► dependent Slack file outbox
```

The fixed paired `connection` command exits at the Slack transport/outbox boundary. It does not construct the service, build family context, search resources, or invoke a model. This lets setup prove transport before the brain exists.

## Interaction ownership

Slack owns no family truth. It converts an authorized message to a strict transport contract and delivers queued replies. Pairing binds an exact `(workspace, user, conversation)` tuple. Transport receipts prevent duplicate Slack delivery from duplicating state; the durable outbox separates processing from delivery.

Messages that do not explicitly name a child are saved to `capture_inbox`. The core does not infer attribution. Future transports must reuse the same contracts, authorization, receipts, inbox, service, and outbox semantics.

## Core ownership

- `contracts.py` defines model-visible and state-transition shapes. Extra keys fail validation.
- `db.py` owns schema versioning, backups, SQLite safety settings, pairing, receipts, inbox, and outbox tables.
- `interaction.py` owns household setup and transport-independent authorization/delivery state.
- `transports/contracts.py` normalizes messages; `transports/slack.py` is the Slack shell.
- `repository.py` owns idempotent events, job leases, run state, atomic effects, and proposed actions.
- `context_builder.py` returns a controlled projection rather than the whole family database.
- `resources.py` owns the licensed-content boundary and FTS5 retrieval.
- `reasoning.py` owns generator/critic/revision topology. Backends cannot execute effects.
- `brain_config.py` owns ignored non-secret role selections and redacted credential/config readiness.
- `providers/` owns distinct OpenAI Responses, Anthropic Messages, and OpenRouter Chat Completions adapters.
- `artifacts.py` owns deterministic one-page paper output and validation.
- `episodes.py` conservatively groups raw events into inspectable learning occasions and records append-only parent corrections.
- `director.py` owns the bounded reflection path; public policy keeps context-driven suggestions disabled in v0.1.
- `service.py` is the shared application boundary; `web.py` is a loopback setup/review console.

## Transactions and retries

Network/model calls occur outside SQLite transactions. Raw evidence is committed before reasoning. Final response, validated graph effects, action proposals, and completion statuses commit together. Duplicate event IDs return the original result; an ID reused with different content fails.

Slack events receive an independent payload hash. A confirmed Slack API rejection can be retried with a bound attempt limit. An ambiguous network failure is marked `unknown` instead of automatically risking a duplicate parent message.

Visuals are dependent deliveries: accessible text is confirmed first, then one validated asset may upload. In v0.1, model-proposed deterministic cards are replaced by exact reviewed local templates; the available fallbacks cover robot-size comparisons and pretend-robot command activities. Safe decorative proposals may continue through the separate opt-in image route. The file state machine persists the Slack file ID before byte transfer and marks `completing` before the single-use completion call. An ambiguous completion is never retried. Deterministic cards are Tier B and explicitly non-scale; Tier C instructional visuals remain disabled.

## Deployment boundary

The MVP is a local family service. SQLite and private source paths are deliberate. Slack Socket Mode establishes an outbound WebSocket from that service, so no inbound public endpoint or cloud database is required. The loopback console is not a LAN service. Cloud hosting and remote workers need a separate threat model.

Coding agents are setup/customization operators. An experimental attended coding-agent backend would still have to implement the same bounded `ModelBackend` contract; the coding session itself does not own family truth or side effects.
