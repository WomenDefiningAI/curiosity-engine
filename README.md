# Curiosity Engine

A local-first, parent-facing harness that turns children’s questions into short, hands-on learning threads.

> AI plans. Parents create the conditions. Kids do the thinking.

Slack is the v1 family interface. Family context stays in ignored local storage; only bounded request context is sent to the model provider you choose. This is not a child-facing chatbot.

## What you need

- Python 3.11+ on a Mac or Linux computer
- Codex, Claude Code, or another coding agent for guided setup
- an adult-controlled Slack workspace and Slack app
- an API key from OpenAI, Anthropic, or OpenRouter

The family computer and Slack connector must stay running for replies. Paper and writing utensils are enough; a home or library printer is optional.

## Start here

Clone the repo, open it in VS Code, and tell your coding agent:

> Read `AGENTS.md` and `docs/AGENT_ONBOARDING.md`. Act as my Family Operator and walk me through each checkpoint. Keep all family data and credentials private. Do not call setup complete until the fixed Slack test, synthetic model test, and one parent-reviewed real answer pass.

Then let the agent run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack,all-providers]'
curiosity doctor
```

`curiosity doctor` reports the next setup action without printing family details or secrets.

## Setup checkpoints

| Checkpoint | Proof |
|---|---|
| Local | `private/` is ignored; household and child records exist locally |
| Slack | One adult and conversation are paired |
| Connection | Slack replies to `connection` without loading a model or family context |
| Brain | A family-data-free live provider probe passes |
| Family fit | The family lens is set and one real Slack answer passes parent review |

Use [Slack setup](docs/SLACK_SETUP.md) first, then [brain setup](docs/BRAIN_SETUP.md). The model stack should cover structured reasoning, vision/OCR, visual QA, and image generation; one provider does not need to fill every role.

## Daily Slack use

```text
connection
children
ask kid-a: Why does ice float?
We saw a shiny beetle today
inbox
assign inbox_... kid-a
feedback kid-a engaged: Kept asking about it
privacy
help
```

Unattributed notes stay in the inbox until a parent assigns them. In a channel, invite the app and mention it explicitly.

## Privacy boundary

Family data never belongs in tracked files:

```text
private/data/       local database
private/resources/  family-owned and purchased material
private/output/     generated pages
private/setup/      Slack and model configuration
```

Do not paste credentials into agent chat. Enter them directly in the ignored files created by setup. Purchased-resource excerpts are off by default and require both parent opt-in and relevant retrieval. See [Privacy](docs/PRIVACY.md).

## MVP boundaries

Included: Slack capture, bring-your-own model APIs, local context, parent feedback, and optional one-page PDFs.

Not included: unsolicited suggestions, embeddings, a graph database, probabilistic mastery, child-facing accounts, cloud hosting, autonomous purchasing, or 3D printing. Telegram, school email, calendars, and easier printer workflows are future extensions.

## Documentation

- Families and coding agents: [onboarding](docs/AGENT_ONBOARDING.md), [Slack](docs/SLACK_SETUP.md), [brain](docs/BRAIN_SETUP.md), [family lens](docs/FAMILY_LENS.md), [operations](docs/OPERATIONS.md)
- Design and trust: [architecture](docs/ARCHITECTURE.md), [context graph](docs/CONTEXT_GRAPH.md), [artifact trust](docs/ARTIFACT_TRUST.md), [privacy](docs/PRIVACY.md)
- Contributors: [testing](docs/TESTING.md), [public-project policy](docs/PUBLIC_PROJECTS.md), [stewardship](docs/OPEN_SOURCE_STEWARDSHIP.md), [roadmap](ROADMAP.md)

Run the public quality gate with:

```bash
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
```
