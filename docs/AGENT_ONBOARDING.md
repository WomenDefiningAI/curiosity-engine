# Coding-agent onboarding

Read `AGENTS.md` first. Act as the **Family Operator** until the family has a working, understandable daily loop. Guide one checkpoint at a time; run safe local commands yourself and leave the parent with a clear next action.

Setup is complete only when local privacy, Slack pairing, the model-free text and visual connection tests, a synthetic model probe, the family lens, and one parent-reviewed real answer all pass. `curiosity doctor` is the redacted source of truth.

## Rules

- Stop if the `private/` Git boundary fails.
- Never put family details, purchased-resource details, tokens, private outputs, or private evals in tracked files or public text.
- Never ask for a credential in chat. Let the parent paste it directly into an ignored owner-only file.
- Explain what Slack or a model provider will receive before requesting access.
- Do not infer which child asked an unattributed question.
- Resource ownership means availability, not exposure, understanding, or interest.
- A transport or schema pass does not prove educational quality.

## 1. Explain the system

Tell the parent:

> Curiosity Engine runs on this computer and uses Slack as the parent interface. Slack setup and model setup are separate. We will prove Slack with a fixed response, test the model with synthetic data, then review one real family answer. The computer and connector must remain running for replies.

Provider APIs are the supported brain. An authenticated coding-agent session can help operate or customize the repo while attended, but it stops when the terminal or login stops; see [coding-agent runtime](CODING_AGENT_RUNTIME.md).

## 2. Install and create private state

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack,all-providers]'
curiosity doctor
curiosity setup --owner-name "OWNER" --timezone "AREA/CITY"
curiosity child add --id kid-a --name "PRIVATE NAME" --grade "OPTIONAL"
curiosity doctor --write-report
```

Do not replace an existing database. Summarize readiness without repeating private values.

## 3. Connect Slack

Follow [Slack setup](SLACK_SETUP.md). The parent pastes both tokens into `private/setup/slack.env`.

```bash
chmod 600 private/setup/slack.env
curiosity slack pair-code
curiosity slack run
```

Pair the exact adult/conversation, then send `connection`. Its fixed reply must arrive before any model is configured. Check `curiosity doctor` for `transport_verified: true`.

Then send `visual connection`. Its fixed local card must arrive with alt text before family questions rely on visuals. This requires `files:write` and app reinstallation; check `visual_delivery_verified: true`.

## 4. Configure and test the brain

Use [Brain setup](BRAIN_SETUP.md) to choose structured reasoning, vision/OCR, visual-QA, and image-generation routes. The parent pastes keys into `private/setup/model.env`.

```bash
chmod 600 private/setup/brain.json private/setup/model.env
curiosity brain doctor
curiosity brain test --live
```

The live probe may be billable but contains no family, Slack, or resource data. Changing routes requires another probe.

Deterministic cards need no image provider. If the parent explicitly chooses decorative generation, run `curiosity visual mode --mode decorative` and `curiosity visual test --live`; the latter is a separate billable, family-free image probe.

## 5. Set the family lens

Start with public defaults:

```bash
curiosity family-lens configure
```

Ask only for useful constraints such as grade/readiness, activity length, parent effort, materials, themes, and boundaries. See [Family lens](FAMILY_LENS.md).

Purchased material stays in `private/resources/`. Default to metadata only. Enable bounded relevant excerpts only with explicit permission:

```bash
curiosity resource mode --mode selected_excerpts
```

## 6. Review one real answer

Ask a real question in paired Slack, then review factuality, grade fit, curiosity value, and parent effort:

```bash
curiosity onboard pending
curiosity onboard review --latest \
  --factuality pass \
  --grade-fit pass \
  --curiosity-value pass \
  --parent-effort pass
```

Only a delivered, current-stack real question can satisfy this gate. If any dimension misses, tune, retry, and review again. Behavioral changes should gain a synthetic public regression when that can be done safely.

Privately inspect context with `curiosity context --child kid-a`. Explain that repeated messages are mention counts, not proof of interest or mastery; retries and answer repair remain one episode. Unsolicited suggestions are disabled in v1.

## 7. Hand off daily operation

```bash
source .venv/bin/activate
curiosity slack run       # leave running
curiosity doctor          # redacted status
curiosity inbox list      # unattributed notes
```

Control-C, closing the terminal, sleep, or shutdown stops Slack replies. Explain token rotation, binding revocation, backups, and private output locations; see [Operations](OPERATIONS.md).

## 8. Run the steward review

At session end, follow [Open-source stewardship](OPEN_SOURCE_STEWARDSHIP.md). Report: **public contribution prepared**, **family-only customization retained**, and **ideas needing more evidence**. Prepare generic work locally, but ask before pushing, publishing, opening a PR, filing an issue, or contacting anyone.
