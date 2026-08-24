# Curiosity Engine

A local-first, parent-facing harness that catches a child’s questions and turns them into small, hands-on learning threads.

> AI does the planning. Parents create the conditions. Kids do the thinking.

Curiosity Engine is not a child-facing chatbot. It keeps durable family context locally, controls what context reaches a model, retrieves parent-owned resources, proposes low-effort activities, makes optional one-page PDFs, and learns from parent feedback. Slack is the v1 daily parent interface; the local website is a setup and review console.

## Start with your coding agent

This MVP assumes a parent clones the repository, opens it in VS Code, and points Codex, Claude Code, or another coding agent at it. Tell the agent:

> Read `AGENTS.md` and `docs/AGENT_ONBOARDING.md` completely. Then walk me through setup one checkpoint at a time, explain what remains local, and do not put family data or credentials in tracked files.

Parent-readable orientation is in [Getting started](docs/GETTING_STARTED.md). The complete family-workspace and bot walkthrough is in [Slack setup](docs/SLACK_SETUP.md).

## What the MVP includes

- Agent-guided, deterministic onboarding and a privacy-safe `curiosity doctor`.
- A parent-only Slack bot using outbound Socket Mode: DMs and explicit mentions, no public server.
- Exact parent/workspace/conversation pairing with short-lived, single-use codes.
- An unassigned inbox so the engine never guesses which child asked a question.
- A loopback-only local setup/review console and CLI over the same service layer.
- SQLite schema v7 with backup-aware migrations, idempotent events, receipts, jobs, and delivery outbox.
- Bounded family context and generic indexing of purchased resources beneath ignored `private/` paths.
- A structured Pull-the-Thread response with factual, pedagogy, and context critics plus fail-closed validation.
- Deterministic one-page PDFs for ordinary paper, plus optional conventional home/library printing.
- A bounded weekly suggestion—or an explicit choice to do nothing—disabled until the parent enables it.
- Public behavioral and regression evaluations with an optional live-model judge.

The deterministic backend verifies setup and transport wiring without an AI key, but its Slack replies are visibly labeled as offline demos. Question-specific daily answers require a configured reasoning provider and an explicit disclosure choice.

## Manual quick start

Python 3.11 or newer is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack]'
curiosity doctor
curiosity setup --owner-name "YOUR DISPLAY NAME" --timezone "America/New_York"
curiosity child add --id kid-a --name "PRIVATE NAME"
```

Next follow [Slack setup](docs/SLACK_SETUP.md), then:

```bash
curiosity slack pair-code
curiosity slack run
```

The connector runs on the family computer and connects outward to Slack. The computer must be awake and the process running for replies. Start the optional local console with `curiosity serve` and open [http://127.0.0.1:8766](http://127.0.0.1:8766).

## The daily Slack loop

```text
children
ask kid-a: Why does ice float?
We saw a shiny beetle today
inbox
assign inbox_... kid-a
feedback kid-a engaged: Kept asking about it
privacy
help
```

Free-form notes remain unassigned until a parent names a child. Pairing is per adult and per conversation; being a member of the Slack workspace is not enough.

## Public code, private family context

Everything family-specific belongs beneath `private/`, which Git ignores:

```text
private/data/curiosity.db       family graph, questions, receipts, feedback
private/resources/              purchased or family-authored resources
private/output/                 generated PDFs and previews
private/setup/slack.env         Slack credentials
private/setup/model.env         optional hosted-model credential and mode
private/setup/status.json       privacy-safe local setup report
```

Public files contain generic code, schemas, prompts, configs, examples, and tests. Never force-add `private/`. Search results expose purchased-resource metadata by default. A household may explicitly opt in to bounded relevant excerpts for Slack-generated answers; those excerpts enter the configured provider request but are paraphrased rather than copied into Slack. Retrieval never forces an unrelated purchased unit into an answer.

See [Privacy](docs/PRIVACY.md) before publishing a fork or sharing logs.

## Question-specific reasoning

The offline backend is the safe setup default. For tailored daily answers, copy the private template, paste an [OpenAI API key](https://platform.openai.com/api-keys) directly into the ignored file, and protect it:

```bash
cp integrations/openai/model.env.example private/setup/model.env
chmod 600 private/setup/model.env
curiosity doctor
```

Do not paste the key into agent chat. Provider requests disable response storage, but the selected provider still processes the bounded context sent with a request. The Responses API supports structured output and bounded built-in tools such as web search; the harness enables search for workflows that need current factual examples. See the [official Responses API reference](https://developers.openai.com/api/reference/resources/responses/methods/create).

Private purchased-resource excerpts remain a separate household choice:

```bash
curiosity resource mode --mode selected_excerpts
```

## Paper outputs

Core use assumes paper, writing utensils, and common household materials. A printer is optional. Parents can download one-page PDFs and print them at home or a library; no specialized fabrication equipment is part of this MVP.

PDF preview/QA uses Poppler (`pdfinfo` and `pdftoppm`); on macOS: `brew install poppler`.

## Reliability checks

```bash
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
```

Private family data and licensed excerpts are excluded from public eval fixtures. An offline pass proves deterministic invariants, not family-quality model semantics; promotion remains manual.

## Repository map

```text
src/curiosity_engine/             service, workflows, contracts, local UI
src/curiosity_engine/transports/  normalized transport contracts and Slack shell
integrations/slack/               least-privilege Slack manifest and token template
configs/                          versioned runtime/autonomy policies
evals/                            public behavioral/regression fixtures
private/                          ignored family data and licensed resources
docs/                             onboarding, privacy, operations, architecture
```

## Scope

Slack is the only messaging channel in v1. Telegram may be added later through the transport boundary. WhatsApp, cloud hosting, child chat accounts, autonomous purchasing, automatic printing, and specialized fabrication are outside the MVP.

See [MVP status](docs/MVP_STATUS.md), [Architecture](docs/ARCHITECTURE.md), [Operations](docs/OPERATIONS.md), and [Testing](docs/TESTING.md).
