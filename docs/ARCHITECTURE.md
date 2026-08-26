# Architecture

```text
parent in Slack
      │ paired identity + normalized message + durable receipt
      ▼
durable thread session ──► parent-chat planner ──► reviewed capability + skill
      │                                               │
      │ prior releases + bounded child context        ▼
      │                                      typed tool + code policy
      │                                               │
      └───────────────────────────────────────────────┤
                                                      ▼
     answer ──► Block Kit choice ──► visual ──► worksheet/activity/challenge
       │              │                 │                    │
       └──────────── durable ordered release units + outboxes ────────────► Slack thread

confirmed weekly request ──► narrow scheduler ──► interactive parent check-in
```

The harness is provider-neutral. Models interpret natural parent language, select reviewed tools, and produce typed
content. Code owns identity, permissions, approval, transactions, rendering, retries, files, schedules, and external
effects. There is no general shell or computer-control tool in the family runtime.

## Main boundaries

- `sessions.py` persists thread history, agent turns, and tool calls.
- `capabilities.py` progressively loads reviewed workflow and skill instructions from `capabilities/`.
- `parent_agent.py` runs the bounded semantic tool loop; `tooling.py` enforces deny-by-default policy.
- `interactions.py` compiles transport-neutral choices to Block Kit using opaque, bound, expiring, one-use tokens.
- `learning_artifacts.py` owns distinct worksheet, activity, and challenge contracts plus one-page render/QA.
- `scheduler.py` owns confirmed weekly check-ins and durable, deduplicated runs.
- `service.py` binds tools to family-safe operations; `transports/slack.py` only handles Slack normalization/delivery.
- `db.py` owns the versioned local SQLite state; text, visual, and artifact outboxes isolate external delivery.

Every model action produces an inspectable capability run and ordered release units. This lets the text arrive first,
while a visual or printable finishes later, without losing thread identity or repeating work.

## Existing learning pipeline

New questions still pass through bounded context, generator, adversarial critics, revision, and strict local validation.
Raw evidence is committed before model calls. A retry or parent revision is diagnostic evidence, not a second interest
signal. Private-resource retrieval requires household opt-in and relevance; Slack never receives purchased excerpts.

The fixed `connection` response stops at the transport/outbox boundary and contacts no model. Visuals deliver only
after accessible text. Knowledge-bearing diagrams remain deterministic and code-labelled; optional decorative art uses
a minimized prompt and visual QA, falling back safely when generation fails.

## Reliability

Network/model calls occur outside SQLite transactions. Duplicate inputs return the original result; conflicting reuse
fails. Confirmed provider rejections may retry within bounds. Ambiguous external completion becomes `unknown` so the
harness does not silently duplicate a parent message or file.

The MVP runs on one family computer. Slack Socket Mode needs no public inbound endpoint. `curiosity host install`
creates restartable Linux user services for Slack and scheduled work. Cloud or multi-family hosting needs a separate
threat model.
