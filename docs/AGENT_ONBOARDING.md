# Coding-agent onboarding contract

This is the first file a coding agent should use after reading `AGENTS.md`. The parent may be comfortable describing family life but unfamiliar with terminals, Python, Slack administration, tokens, or databases. Explain each checkpoint in plain language and perform safe local steps for them when possible.

## The result you are guiding them toward

The family will have:

1. a local Curiosity Engine database beneath ignored `private/`;
2. one owner/parent and one or more private child profiles;
3. a Slack workspace and minimal-permission Slack app;
4. one explicitly paired Slack identity/conversation per participating parent;
5. a working daily loop: capture a question, assign it to a child, receive a small learning thread, and optionally download a one-page PDF.

Slack is the daily parent interface. The loopback website is an optional setup and review console. Neither is a child-facing chatbot.

## Non-negotiable privacy behavior

- First confirm that `private/` is ignored. Stop if `curiosity doctor` reports the Git boundary as failed.
- Never put family names, child details, Slack IDs, tokens, questions, licensed excerpts, or generated outputs in tracked files.
- Do not print token values in commands, logs, setup reports, or your summary. Ask the parent to paste tokens directly into `private/setup/slack.env` in their editor.
- Do not paste purchased curriculum into public prompts or fixtures. Index only a parent-owned copy stored beneath `private/resources/`; excerpt use defaults off and requires an explicit household opt-in.
- Describe provider disclosure before enabling a hosted model. Slack processes messages sent through Slack. A configured model provider processes the bounded request context sent to it. The local database remains on this computer.
- Do not infer which child asked an unattributed question. Leave it in the unassigned inbox.

## Checkpoint 1 — orient the parent

Explain this in your own words before installing anything:

> Curiosity Engine is a small local service, not a new social app. It remembers family context on this computer and uses Slack as the familiar place where parents capture questions. The bot runs only while its process is running. Its first safe mode is deterministic and needs no paid AI account, but that mode is only for testing the wiring and labels its replies as demos.

Ask only for the details needed for the next checkpoint. A good sequence is:

- the owner's display name;
- the household IANA timezone (for example, `America/New_York`);
- whether a family Slack workspace already exists;
- a simple private ID and display name for each child, plus optional grade or birth year.

Do not ask the parent to share their full family story up front.

## Checkpoint 2 — verify and initialize

Run from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,slack]'
curiosity doctor
```

Then use the parent's actual values only in the local command invocation:

```bash
curiosity setup --owner-name "OWNER" --timezone "AREA/CITY"
curiosity child add --id kid-a --name "PRIVATE NAME" --grade "OPTIONAL"
```

The setup command is idempotent. Database migrations create a timestamped sibling backup before modifying an older non-empty database. Do not replace an existing database just to make onboarding appear clean.

Checkpoint: run `curiosity doctor --write-report`. Summarize pass/fail categories without repeating private details.

## Checkpoint 3 — create Slack safely

Walk through `docs/SLACK_SETUP.md` with the parent. They must perform the Slack account/app approvals and paste both tokens directly into the ignored owner-only file. You may create the empty directory and placeholder file, set permissions, and validate token *shapes*, but never echo values.

Checkpoint: `curiosity doctor` should show the Slack dependency and token checks passing. Do not claim Slack is ready until a parent/conversation is paired.

## Checkpoint 4 — pair one exact conversation

Create a short-lived code. It defaults to the configured owner:

```bash
curiosity slack pair-code
```

Start the connector in another terminal (or as a supervised coding-agent process):

```bash
curiosity slack run
```

Then ask the parent to DM the Slack app with `pair CODE`. The code expires after 15 minutes and can be used once. Pairing binds the exact Slack workspace, user, and conversation. Each co-parent must have a local parent record and use their own code; each family channel pairing is separate from a DM pairing.

Checkpoint: `curiosity slack bindings` shows an active binding and `curiosity doctor` reports `slack_ready: true`. Do not include raw Slack IDs in a shareable summary.

## Checkpoint 5 — choose the reasoning and resource boundary

Before asking a real family question, explain that arbitrary question-specific answers need a reasoning provider. Slack will process the parent message and bot reply. The configured model provider will process the bounded child context selected for that request. The local database itself stays on the family computer.

The first supported hosted backend is OpenAI. Ask the parent to create their own API key and paste it directly into an ignored owner-only file—never into agent chat:

```bash
cp integrations/openai/model.env.example private/setup/model.env
chmod 600 private/setup/model.env
curiosity doctor
```

Do not call tailored answers ready until `answer_ready` is true. If the parent stays in deterministic mode, describe it as a workflow demo, not a general answer engine.

Licensed excerpts are a second, independent choice. If the parent explicitly permits small relevant passages to enter hosted-model requests, run:

```bash
curiosity resource mode --mode selected_excerpts
```

Explain that output is paraphrased, excerpts are not posted verbatim to Slack, and retrieval uses a purchased unit only when it is relevant. The default remains `metadata_only` for every new family.

## Checkpoint 6 — teach the daily loop

Have the parent try these in Slack:

```text
children
ask kid-a: Why does the Moon seem to follow us?
We saw a very shiny beetle today
inbox
assign inbox_... kid-a
feedback kid-a engaged: Kept asking about it
privacy
help
```

Explain that free-form notes remain unassigned until the parent names a child. The response uses ordinary paper, writing tools, and common household materials; a conventional printer is optional. Ask the parent to judge whether the answer directly addressed the question and fit the stored grade—not merely whether the bot replied.

Show the optional local console with `curiosity serve`, but frame it as setup, review, PDF download, and feedback—not a second daily chat product.

## Checkpoint 7 — handoff

End with the parent knowing:

- start daily Slack access with `curiosity slack run`;
- stop it with Control-C;
- inspect setup safely with `curiosity doctor`;
- review unassigned notes with `curiosity inbox list` or Slack `inbox`;
- open the private local console with `curiosity serve`;
- customize public behavior through versioned configs/prompts/evals, and family context only beneath `private/`;
- ask a coding agent to read `AGENTS.md` and this file before making future changes.

Do not call onboarding complete until the parent has successfully sent one paired Slack command and understands that the connector must be running.
