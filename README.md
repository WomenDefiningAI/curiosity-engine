# Curiosity Engine

A local-first, parent-facing harness that catches a child’s questions and turns them into small, hands-on learning threads.

> AI does the planning. Parents create the conditions. Kids do the thinking.

Curiosity Engine is not a child-facing chatbot. It keeps durable family context locally, controls what reaches outside providers, retrieves parent-owned resources, proposes low-effort activities, creates optional one-page PDFs, and learns from parent feedback. Slack is the v1 daily parent interface.

## Before you clone

Cloning the code is only the first part. A working family setup also needs:

| You will need | Why | Cost / operational reality |
|---|---|---|
| Python 3.11+ and a Mac/Linux computer | Runs the local database, harness, and Slack connector | The computer must be awake and the connector process running for Slack replies |
| Codex, Claude Code, or another coding agent | Walks a nontechnical parent through setup and customization | Recommended for setup; not the dependable production brain by default |
| A family Slack workspace and Slack app | Gives parents one familiar capture/response surface | A free workspace is sufficient for the MVP; Slack still processes messages and replies |
| An LLM API account and key | Produces arbitrary question-specific answers | Usage may be billed by the provider; the parent brings and controls the account |
| Model choices for reasoning, vision/OCR, and image generation | Elementary learning is visual and often starts with worksheets, photos, diagrams, or PDFs | One provider may cover all roles, or the family may mix providers by role |
| A few family choices | Sets pedagogy, grade/readiness, parent effort, themes, and private-resource disclosure | Do not upload a complete family history; setup gathers only bounded useful context |

Paper and writing utensils are enough for core use. A conventional home/library printer is optional. No specialized fabrication equipment is part of the MVP.

## Start with your coding agent

Clone/open the repository in VS Code and tell Codex, Claude Code, or an equivalent agent:

> Read `AGENTS.md` and `docs/AGENT_ONBOARDING.md` completely. Act as my Family Operator: walk me through every checkpoint, run and test the repository, explain every outside service before credentials, keep family data private, and do not call setup complete until a real Slack answer passes my review. At the end, run the Open-Source Steward review and prepare any safe general improvements locally.

The agent has two repository personas:

- **Family Operator:** accountable for getting this family cloned, private, connected, customized, tested, and restartable.
- **Open-Source Steward:** at the end of a meaningful session, abstracts safe reusable patterns into synthetic public code/docs/tests without exposing the family. It never publishes or opens a pull request without approval.

See [agent onboarding](docs/AGENT_ONBOARDING.md) and [open-source stewardship](docs/OPEN_SOURCE_STEWARDSHIP.md).

## The five setup gates

```text
local/private → Slack paired → fixed non-model connection proof
       → API brain + synthetic probe → family lens + real parent-reviewed answer
```

1. **Local/private:** install dependencies, verify `private/` is ignored, initialize the database, add the minimum household/child information.
2. **Slack:** create/install the app, store tokens privately, pair an exact adult/workspace/conversation.
3. **Transport proof:** send `connection`. Its fixed reply does not invoke a model, load child context, or read resources.
4. **Brain and family lens:** choose the model stack, paste API keys directly into ignored files, run a family-data-free probe, then accept/customize pedagogy and resource rules.
5. **Real quality check:** ask one real question in Slack and have the parent review factuality, grade fit, curiosity value, and effort. Only then is `end_to_end_ready` true.

`curiosity doctor` reports the current stage and `next_action` without displaying keys, child names, questions, Slack IDs, or licensed resource titles.

## Recommended brain architecture

The stable, recommended runtime is direct API access: Slack calls the local deterministic harness, and the harness calls explicitly configured model APIs. Code—not a free-running agent—owns context selection, schemas, critics, retries, persistence, and side effects.

Do not choose only a good chat model. A Curiosity Engine brain stack must cover:

| Role | Required capability | Curiosity Engine evaluation |
|---|---|---|
| Reasoning and critics | Strict structured output; strong factual and pedagogical reasoning; web grounding when freshness matters | Elementary accuracy, grade calibration, curiosity preservation, parent effort |
| Vision and extraction | High-resolution image input, scanned PDF/worksheet understanding, OCR/layout/table comprehension, structured extraction | Worksheets, small labels, handwriting tolerance, charts and diagrams |
| Visual QA | Image input and structured issue reporting | Clipping, tiny text, incorrect counts/labels, misleading visual geometry, print legibility |
| Image generation/editing | Good instruction following, child-appropriate illustration, reference-image editing | Delight and relevance; never the source of truth for instructional text, exact counts, maps, or scientific geometry |

One model does not have to do every role. For example, Claude supports image and PDF understanding, but a Claude reasoning route may be paired with an OpenAI or OpenRouter image-generation route. OpenRouter can expose many vision and image-output models, but each selected route still needs capability and privacy checks. See [Brain setup](docs/BRAIN_SETUP.md), [model routing](docs/MODEL_ROUTING.md), and [family lens](docs/FAMILY_LENS.md).

### Provisional starting candidate, not yet the project champion

Until family evals establish a recommendation, the documented OpenAI starting candidate is a current balanced vision-capable frontier model for reasoning/vision plus a dedicated current image model. As of this README revision, that means `gpt-5.6-terra` and `gpt-image-2`; both names are examples to evaluate, not permanent defaults. OpenAI documents current text models as vision-capable and lists GPT Image separately for image generation. Review the current [OpenAI model catalog](https://developers.openai.com/api/docs/models) before setup.

The project will label a route `family_recommended` only after it passes public regression/eval gates plus repeated real-family worksheet, visual, and first-grade answer reviews. Other families remain free to choose a direct provider or [OpenRouter](https://openrouter.ai/docs/quickstart).

## Two ways to operate the intelligence

### 1. API brain — recommended

Supported in the harness: direct OpenAI, native Anthropic, or OpenRouter, with private per-role model choices. This is the most deterministic interface, supports explicit schemas/timeouts/capabilities, and does not depend on an interactive coding session remaining logged in.

### 2. Coding-agent-operated — experimental and attended

A family may instead point an already authenticated Codex or Claude Code session at the whole repository and ask it to run jobs or customize a local adapter. This can be useful for experimentation with an existing subscription, but it is not the supported unattended Slack brain:

- closing the terminal, logging out, expiring the session, or shutting down the computer stops it;
- a coding-agent subscription is not automatically equivalent to an application API entitlement;
- structured output, timeouts, retries, cost controls, and model availability can differ from provider APIs;
- the agent sees whatever bounded context the custom adapter gives it, so the same privacy review is required;
- the family owns and maintains any adapter-specific changes.

This option and the API path share one limitation: because v1 runs locally, shutting down the family computer stops Slack replies. The API path removes the extra dependency on a live interactive coding-agent session. See [attended coding-agent operation](docs/CODING_AGENT_RUNTIME.md).

## Manual setup outline

Python 3.11 or newer is required:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack,all-providers]'
curiosity doctor
curiosity setup --owner-name "YOUR DISPLAY NAME" --timezone "America/New_York"
curiosity child add --id kid-a --name "PRIVATE NAME" --grade "1st"
```

Follow [Slack setup](docs/SLACK_SETUP.md). Pair the conversation, run the connector, and send `connection` before configuring any model.

Then choose the brain stack. This example is a starting candidate only:

```bash
curiosity brain configure \
  --provider openai \
  --model gpt-5.6-terra \
  --vision-model gpt-5.6-terra \
  --image-provider openai \
  --image-model gpt-image-2 \
  --web-search \
  --recommendation-status family_evaluating
```

Paste the requested key directly into `private/setup/model.env`, never into agent chat, then:

```bash
chmod 600 private/setup/model.env
curiosity brain doctor
curiosity brain test --live
curiosity family-lens configure
curiosity doctor
```

Finally ask a real question in Slack and record the parent's assessment:

```bash
curiosity onboard pending
curiosity onboard review \
  --latest \
  --factuality pass \
  --grade-fit pass \
  --curiosity-value pass \
  --parent-effort pass
```

`pending` shows only private event IDs, timestamps, workflow names, review state, and whether each answer came from the current answer stack. `--latest` selects the newest current-stack delivery; a synthetic, stale-stack, or undelivered event cannot satisfy this gate. Changing model routes, answer prompts/policies, the family lens, or resource-disclosure mode requires a new real review.

## Daily Slack loop

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

Free-form notes stay unassigned until a parent names a child. Pairing is per adult and per conversation.

Inspect the local context model at any time with `curiosity context --child kid-a`. Repeated messages in one learning occasion are grouped into an episode so retries are not mistaken for durable interest. Even a later exact repeat remains ambiguous until content or a parent correction distinguishes a renewed thread from retry/dissatisfaction. The inspector shows episode membership, evidence eligibility, uncertainty, and corrections; it stays local because it includes family text.

## Public code, private family context

Everything family-specific belongs beneath ignored `private/`:

```text
private/data/curiosity.db       family graph, questions, receipts, feedback, setup evidence
private/resources/              purchased or family-authored resources
private/output/                 generated PDFs and previews
private/setup/slack.env         Slack credentials
private/setup/brain.json        non-secret provider/model/role choices
private/setup/model.env         provider API credentials
private/setup/status.json       redacted local setup report
```

Purchased-resource availability is not evidence that a child has seen or understood it. Metadata and bounded excerpts enter a provider request only under the household's explicit mode and relevance gate. See [Privacy](docs/PRIVACY.md).

## Paper outputs

Core activities use ordinary paper, writing utensils, and household materials. The harness already creates downloadable one-page PDFs and can optionally send an exact parent-approved file to a conventional system printer. Future printer integrations may simplify discovery, queues, and remote family workflows; automatic printing remains outside v1.

## Curated open-source ecosystem

The project maintains a reviewed, machine-readable catalog for artifact creation, OCR, worksheet/question generation, educational interactions, and simple game engines. It points to canonical upstream documentation and release feeds instead of copying static snapshots. Listing is not permission to install: projects are labeled `integrated`, `approved_reference`, `evaluation_candidate`, or `watch_only`, and the default is reference-only.

```bash
curiosity ecosystem status
curiosity ecosystem list --category worksheet_generation
curiosity ecosystem check --live
```

The explicit live check reads public GitHub/PyPI metadata only; it never clones or executes upstream code or sends family data. See the [curated project research and vetting gate](docs/PUBLIC_PROJECTS.md).

## Reliability checks

```bash
ruff check .
pytest
curiosity-lab --repo . --json-out private/eval-report.json
```

Tests never inherit private provider configuration or make live provider calls. Credential-gated live probes are explicit and local. Private family data and licensed excerpts never belong in public eval fixtures.

## Repository map

```text
src/curiosity_engine/             service, workflows, providers, contracts, local UI
src/curiosity_engine/transports/  normalized transport contracts and Slack shell
integrations/                     provider and Slack templates
configs/                          public requirements, policies, starter family lens
evals/                            public behavioral/regression fixtures
private/                          ignored family data, credentials, resources, outputs
docs/                             onboarding, privacy, operations, architecture
```

## MVP and future integrations

Slack is the only daily messaging transport in v1. Near-term integrations may include parent-authorized Gmail/school-email forwarding for background extraction, calendar context, easier conventional-printer workflows, and Telegram through the existing transport boundary. Each requires its own consent, least-privilege scopes, provenance, expiry, and evals. They are roadmap items—not enabled accounts or promises in the current MVP.

WhatsApp, cloud multi-tenancy, child-facing chat accounts, autonomous purchasing, automatic outbound campaigns, and specialized fabrication remain outside the MVP.

Context-driven unsolicited suggestions are also disabled in this release. The repository records episode-aware evidence and permits parent corrections, but does not yet add embeddings, a graph database, probabilistic mastery, or unsolicited suggestions.

See [MVP status](docs/MVP_STATUS.md), [Architecture](docs/ARCHITECTURE.md), [Operations](docs/OPERATIONS.md), [Testing](docs/TESTING.md), and the [Roadmap](ROADMAP.md).
