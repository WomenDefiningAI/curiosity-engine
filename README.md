# Curiosity Engine

A local-first family AI harness that turns children’s questions into accurate, playful learning threads, visuals,
games, and printables. Parents use it in Slack; children do not chat with the model directly.

> AI plans. Parents create the conditions. Kids do the thinking.

## Install

On a Mac or Linux family computer with Python 3.11+:

```bash
curl -fsSL https://raw.githubusercontent.com/WomenDefiningAI/curiosity-engine/main/scripts/install.sh | sh
```

The installer creates an isolated runtime and private family home at `~/.curiosity-engine`; it does not clone a
source repository. If Codex or Claude Code is installed, `curiosity setup` opens a private setup workspace and asks
the coding agent to operate the terminal, explain decisions, and verify every checkpoint. Otherwise run
`curiosity setup --no-launch`, then point your coding agent to `~/.curiosity-engine/workspace/SETUP_HANDOFF.md`.

Setup covers six decisions:

1. household and child profiles;
2. an adult-controlled Slack workspace and app;
3. a fixed, non-AI Slack connection test;
4. a bring-your-own model stack for reasoning, vision/OCR, visual QA, and image generation;
5. family pedagogy, practical constraints, and licensed-resource consent;
6. always-on local services plus one parent-reviewed real answer.

Use [Slack setup](docs/SLACK_SETUP.md) and [brain setup](docs/BRAIN_SETUP.md) if you want to inspect the manual steps.

## How families use it

Mention the bot with a child ID for a new question:

```text
@Curiosity Engine ask kid-a: Why do airplanes stay up?
```

The answer stays in a thread. Continue naturally there: “show different wing shapes,” “turn this into a mystery,”
or “make a printable challenge.” Buttons are shortcuts, not commands the parent must memorize. Unattributed channel
notes remain in a private inbox until a parent chooses a child or dismisses them.

Review recent Slack answers and visuals at `http://127.0.0.1:8766/evals`. Evaluations stay in a separate private
table and do not change a child’s context graph. For a remote family computer, forward port 8766 over SSH first.

## What makes it a harness

- Durable sessions keep the thread’s prior ideas and parent revisions.
- Reviewed capabilities and skills guide worksheets, activities, challenges, visuals, and check-ins.
- A provider-neutral model planner chooses only typed, policy-checked tools.
- Code owns permissions, approval, persistence, retries, rendering, and external effects.
- Responses, visuals, interactions, and PDFs release independently as they are ready.
- Feedback and regeneration attempts improve private family fit and create opt-in, de-identified eval candidates.

Weekly check-ins are available only after explicit confirmation. The harness has no general shell or computer-control
tool; scheduling is a narrow reviewed capability.

## Privacy

Family state is separate from the installed code:

```text
~/.curiosity-engine/private/    database, resources, output, credentials
~/.curiosity-engine/workspace/  private coding-agent setup handoff
```

Credentials are read from owner-only files and never printed by `curiosity doctor`. Purchased-resource excerpts stay
local unless the parent opts into bounded, relevant model context. Slack and the selected model provider still
process the content sent to them. Back up family state with:

```bash
curiosity backup create
curiosity backup verify
```

## Scope

The MVP includes Slack, local child context, parent feedback, visual responses, three real one-page artifact types,
and confirmed weekly check-ins. It assumes paper, writing tools, and optional access to a home or library printer.

It does not yet include unsolicited suggestions, embeddings, a graph database, probabilistic mastery, child accounts,
cloud hosting, Telegram, school email, calendars, autonomous purchasing, or 3D printing.

## Develop

Clone only if you want to change the open-source code:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack,all-providers]'
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
```

Read [AGENTS.md](AGENTS.md) before using a coding agent. Family files belong only in ignored local storage. Start with
[architecture](docs/ARCHITECTURE.md), [privacy](docs/PRIVACY.md), [testing](docs/TESTING.md), and
[open-source stewardship](docs/OPEN_SOURCE_STEWARDSHIP.md).
