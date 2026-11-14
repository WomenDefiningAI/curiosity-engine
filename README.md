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
The wheel includes every public Lab fixture and an immutable manifest that startup checks against the installed
runtime, configuration, prompts, schemas, and capabilities.

Setup covers six decisions:

1. household and child profiles;
2. an adult-controlled Slack workspace and app;
3. a fixed, non-AI Slack connection test;
4. a bring-your-own model stack for reasoning, vision/OCR, visual QA, and image generation;
5. family pedagogy, practical constraints, and licensed-resource consent;
6. always-on local services plus one parent-reviewed real answer.

Use [Slack setup](docs/SLACK_SETUP.md) and [brain setup](docs/BRAIN_SETUP.md) if you want to inspect the manual steps.

For faster ordinary text review, [optionally enable JEV](docs/BRAIN_SETUP.md#optional-fast-text-review-with-jev).
The [parent review lab](docs/HARNESS_EXPERIMENTS.md) lets you compare your ratings with JEV on complete conversations,
including separate human checks of source photos and finished worksheets.

## How families use it

Talk naturally and include a child’s name:

```text
@Curiosity Engine Ethan asked why airplanes stay up
```

Ordinary questions get a short, reviewed answer. Ask for an activity, game, experiment, project, or lesson when you
want a deeper exploration. The answer stays in a thread, where you can say “show different wing shapes,” “turn this
into a mystery,” or “make a printable challenge.” Buttons are optional shortcuts.

Say `add a child` to open an owner-only form. If a message has no clear child, choose one from the private inbox; the
engine never guesses identity.

Review recent Slack answers at `http://127.0.0.1:8766/evals`, rating response shape separately from imagination art
and functional activity aids. Evals stay outside the child graph. For a remote computer, forward port 8766 over SSH.

## What makes it a harness

- Durable sessions keep the thread’s prior ideas and parent revisions.
- Scoped parent memory is inspectable, reversible, and kept separate from evidence about a child.
- Long threads use reference-only structured compaction, and durable background work resumes exactly once.
- Parent chat routes to the smallest useful response: conversation, context capture, quick answer, exploration, or artifact.
- Grade is the baseline; an exactly evaluated policy may calibrate challenge and interest support only after full-Lab
  independent review, separate operator promotion, and reversible per-child family opt-in. Alternative strategy
  selection remains shadow-only.
- Reviewed capabilities and skills guide worksheets, activities, challenges, visuals, and check-ins.
- A provider-neutral model planner chooses only typed, policy-checked tools.
- Private family recipes must change the real typed artifact contract in code-owned baseline/candidate tests, pass
  independent shadow judgment, and receive human approval before bind-before-work activation.
- Code owns permissions, approval, persistence, retries, rendering, and external effects.
- Responses, visuals, interactions, and PDFs release independently as they are ready.
- Typed feedback supports private shadow comparisons. Replay accepts only reviewed categorical features and numeric
  labels; private notes and identity are never copied automatically.
- A Phase 6 research track can compute an evidence-linked uncertainty companion and save inspectable proactive
  candidates only after independently approved deidentified-corpus calibration, internal full-Lab review, operator
  activation, exact release binding, and per-child family opt-in. Evidence corrections apply and repeated attempts in
  one episode count once. It cannot change learner state or enter any delivery queue.

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

The MVP includes Slack, local child context, parent feedback, vetted public photos, visual responses, four one-page
artifact types, 13 worksheet mechanics, and confirmed weekly check-ins. It assumes paper, writing tools, and optional
access to a home or library printer.

It does not include delivered unsolicited suggestions, embeddings, a graph database, production probabilistic mastery,
child accounts, cloud hosting, Telegram, school email, calendars, autonomous purchasing, or 3D printing. The optional
uncertainty/proactivity research plane is closed by default and remains non-deliverable even if separately activated.

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
[architecture](docs/ARCHITECTURE.md), [privacy](docs/PRIVACY.md), [testing](docs/TESTING.md),
[adaptation controls](docs/ADAPTATION_CONTROL_PLANE.md),
[evaluated personalization](docs/EVALUATED_PERSONALIZATION.md),
[research-track controls](docs/RESEARCH_TRACK.md), and
[open-source stewardship](docs/OPEN_SOURCE_STEWARDSHIP.md).
