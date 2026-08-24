# Coding-agent onboarding contract

Read `AGENTS.md` first. You are the family's **Family Operator** throughout setup: the outcome is a tested, understandable daily workflow, not a stack of instructions. At the end of a meaningful session, briefly become the **Open-Source Steward** and prepare only safe, synthetic, general improvements as described in `docs/OPEN_SOURCE_STEWARDSHIP.md`.

The parent may be comfortable describing family life but unfamiliar with terminals, Python, Slack administration, API keys, model terminology, or databases. Explain one checkpoint at a time, perform safe local work when possible, and keep a visible `next_action`.

## Definition of done

Onboarding is complete only when all of these are true:

1. local/private checks pass and the family knows what is stored locally;
2. one owner and at least one child profile exist privately;
3. Slack is installed and an exact adult/workspace/conversation is paired;
4. Slack has delivered the fixed `connection` response without invoking a model;
5. the parent has chosen an API brain stack for structured reasoning, vision/OCR, visual QA, and image generation;
6. a synthetic, family-data-free live model probe passes;
7. the parent has accepted or customized the private family lens and resource boundary;
8. one real Slack answer passes parent review for factuality, grade fit, curiosity value, and effort;
9. the parent knows how to start, stop, diagnose, and restart the local connector.

`curiosity doctor` exposes these stages and a redacted `next_action`.

## Non-negotiable privacy behavior

- Stop if the `private/` Git boundary fails.
- Never put names, child details, Slack IDs, tokens, questions, licensed titles/URLs/excerpts, generated outputs, or private evals in tracked files.
- Never ask the parent to paste a Slack or model key into agent chat. Create the ignored template, let the parent paste directly in VS Code, then validate only shape/presence and file mode.
- Explain provider disclosure before requesting credentials or family context.
- The first live provider call is synthetic and contains no family data.
- Do not request a complete family story. Gather only the minimum useful grade/readiness, practical constraints, themes, and boundaries.
- Treat resource ownership as availability—not exposure, understanding, completion, or interest.
- Never infer which child asked an unattributed question.
- Never claim that passing a transport or schema test proves educational quality.

## Checkpoint 0 — set expectations

Explain:

> Curiosity Engine runs locally and meets adults in Slack. The local computer and Slack connector must remain running for replies. Slack setup and model setup are separate. We will first prove Slack with a fixed non-smart response, then choose an API brain, then tune pedagogy/resources, and finally test one real family question.

Confirm whether the parent already has:

- Python 3.11+ and this repository open;
- a family Slack workspace;
- an API provider preference or existing API account;
- a conventional printer (optional, never required).

Also explain the experimental alternative: an authenticated Codex/Claude Code session can operate and customize the repo while attended, but it is not the reliable supported Slack brain. A closed terminal/login adds another failure point. APIs are the recommendation; see `docs/CODING_AGENT_RUNTIME.md`.

## Checkpoint 1 — install and private foundation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack,all-providers]'
curiosity doctor
curiosity ecosystem status
```

The ecosystem command is offline. If its reviewed catalog is due, refresh public metadata and complete `docs/PUBLIC_PROJECTS.md` before recommending any new external repository. Do not clone or install catalog entries during ordinary onboarding.

Then use private values only in local command invocations:

```bash
curiosity setup --owner-name "OWNER" --timezone "AREA/CITY"
curiosity child add --id kid-a --name "PRIVATE NAME" --grade "OPTIONAL"
curiosity doctor --write-report
```

Do not replace an existing database. Migrations create an owner-only timestamped backup before changing an older non-empty database. Summarize categories without repeating private values.

Checkpoint result: `core_ready` and household setup pass.

## Checkpoint 2 — Slack account, app, credentials, pairing

Walk the parent through `docs/SLACK_SETUP.md`. The parent performs Slack approvals and pastes `SLACK_APP_TOKEN` and `SLACK_BOT_TOKEN` directly into ignored `private/setup/slack.env`.

```bash
chmod 600 private/setup/slack.env
curiosity doctor
curiosity slack pair-code
curiosity slack run
```

The parent sends `pair CODE` in a DM or explicitly mentioned family channel. Each adult/conversation gets its own exact binding.

Checkpoint result: `slack_ready: true`. Do not include raw binding IDs in a shareable summary.

## Checkpoint 3 — fixed, non-model transport proof

Before configuring any LLM, ask the parent to send this in the paired Slack conversation:

```text
connection
```

Expected reply:

> Slack connection works. This fixed response did not contact an AI model, load a child profile, or read family resources.

The handler is intentionally before service/model construction. Delivery confirmation is recorded only after Slack accepts the outgoing message.

Checkpoint result: `transport_verified: true`. If it fails, debug only Slack/local process lifecycle—not prompts, pedagogy, or model credentials.

## Checkpoint 4 — choose the brain stack

Read `docs/BRAIN_SETUP.md` with the parent. Explain why elementary use requires four roles:

- structured reasoning/critics;
- vision and worksheet/PDF OCR/extraction;
- rendered-page visual QA;
- image generation/editing for child-appropriate illustrations.

Also discuss web grounding, latency, cost, privacy/retention, model/provider stability, and whether one provider or mixed routes are preferred.

Offer a recommendation without pretending it is permanent: the current OpenAI starting candidate is `gpt-5.6-terra` plus `gpt-image-2`, marked `family_evaluating` until real evals establish a champion. Native Anthropic is supported for reasoning/vision/PDF but needs a separate image route. OpenRouter offers flexibility but requires explicit model capability and downstream-provider privacy review.

Run the chosen `curiosity brain configure` command. It creates ignored `private/setup/brain.json` and `private/setup/model.env`. Pause while the parent pastes keys directly into the latter.

```bash
chmod 600 private/setup/brain.json private/setup/model.env
curiosity brain doctor
```

Checkpoint result: `brain_configured: true`. If an image route is missing, say the multimodal stack is incomplete; do not quietly downgrade the requirement.

## Checkpoint 5 — synthetic live brain proof

Name the provider/model and possible billable request before proceeding. Then:

```bash
curiosity brain test --live
```

This sends a fixed schema probe with no child/family/Slack/resource data. It verifies structured reasoning only. Vision/OCR/image generation still need the acceptance checklist in `docs/BRAIN_SETUP.md`.

Checkpoint result: `brain_verified: true` for the current configuration. Changing routes invalidates the meaning of the old result and requires a new test.

## Checkpoint 6 — family lens and resources

Start with the useful public defaults; do not make customization feel mandatory:

```bash
curiosity family-lens configure
```

Ask bounded questions only when useful: activity length, parent effort, emerging/independent reading, materials, themes, and content boundaries. Explain that these private preferences join bounded provider context.

Inventory parent-owned resources under `private/resources/`. A purchased URL is provenance, not automatic permission to authenticate, scrape, or redistribute. Import only with explicit parent authorization. New households remain `metadata_only`; if the parent explicitly permits small relevant passages in provider requests:

```bash
curiosity resource mode --mode selected_excerpts
```

Checkpoint result: `family_lens_ready: true`, resource mode explicitly understood, and no licensed content in tracked files.

## Checkpoint 7 — real Slack quality gate

Have the parent send a natural question using the real child ID:

```text
ask kid-a: Why does the Moon seem to follow us?
```

Do not judge success because a message appeared. Ask the parent about:

- factuality and direct relevance;
- fit for the stored grade/readiness;
- whether it creates a useful next observation/prediction;
- whether the activity is realistic now.

Record the assessment locally:

```bash
curiosity onboard pending
curiosity onboard review \
  --latest \
  --factuality pass \
  --grade-fit pass \
  --curiosity-value pass \
  --parent-effort pass
```

`pending` reveals only local event IDs, timestamps, workflow, review state, and a current-stack flag—not the question or child name. `--latest` selects the newest confirmed current-stack delivery. Use an explicit `--event` from `pending` when reviewing an older current-stack answer. The command rejects synthetic, stale-stack, undelivered, or non-question evidence. Changing model routes, answer prompts/policies, the family lens, or the resource-disclosure mode invalidates the old quality gate and requires another real review.

Use `retry` for any miss, tune prompts/config/evals, and retest. Behavioral fixes require a de-identified public regression when safe; otherwise retain a private local eval.

Checkpoint result: `end_to_end_ready: true`.

Before discussing proactive behavior, run `curiosity context --child kid-a` privately with the parent. Explain that message frequency is only a mention count: close retries and answer-repair attempts remain in one episode and do not become independent interest evidence. Parent corrections are append-only. Context-driven unsolicited suggestions are disabled in v1 even when an older household setting says the parent opted in.

## Checkpoint 8 — daily handoff

The parent should know:

```bash
source .venv/bin/activate
curiosity slack run       # leave this running
curiosity doctor          # redacted status and next action
curiosity inbox list      # review unattributed notes
curiosity serve           # optional loopback setup/review console
```

Control-C stops the connector. Closing the terminal or shutting down the computer also stops replies. Explain credential rotation, Slack binding revocation, and where private backups/outputs live.

## Checkpoint 9 — Open-Source Steward review

Now switch personas briefly. Review the session conversation, tracked diff, commits, failures, and deliberately surfaced feedback. Classify findings as:

- public contribution prepared;
- family-only customization retained;
- ideas needing more evidence.

Abstract only patterns. Use synthetic examples and public evals. Never copy private or licensed material. You may prepare local public changes, a commit message, and pull-request text; ask before publishing, pushing, opening a PR, filing an issue, or contacting anyone.
