# Getting started as a parent

You are not setting up a new place for your family to remember to check. Curiosity Engine runs privately on your computer and meets the adults in your family Slack.

## What the pieces do

- **Curiosity Engine** keeps questions, learning context, and generated pages in a local private database.
- **Slack** is where a parent quickly captures a question and receives a response.
- **The Slack app** is the narrow bridge between your workspace and the local engine. It sees DMs sent to it and messages where it is explicitly mentioned.
- **The local website** is an optional setup/review console for profiles, unassigned questions, PDFs, and feedback.
- **A coding agent** guides setup and helps you safely customize this open-source repository.

Your computer must be awake and `curiosity slack run` must be running for the bot to answer. The Slack connection goes outward from your computer; you do not publish your local website to the internet.

## Start here

1. Open this repository in VS Code.
2. Tell your coding agent: “Read `AGENTS.md` and `docs/AGENT_ONBOARDING.md`, then walk me through setup one checkpoint at a time.”
3. Follow the agent through the local privacy check and family setup.
4. Use `docs/SLACK_SETUP.md` to create a free family Slack workspace and the private bot connection.
5. Pair the Slack conversation and send `connection`; this fixed reply proves Slack without using AI.
6. Use `docs/BRAIN_SETUP.md` to choose reasoning, vision/OCR, and image-generation routes, then run the synthetic probe.
7. Accept or customize the family lens and resource boundary.
8. Ask one real question and review factuality, grade fit, curiosity value, and parent effort.

The transport test requires no AI key and never reads child context. Tailored answers require a bring-your-own API brain and should be enabled only after you understand which bounded context each selected provider will receive. `curiosity doctor` reports the next checkpoint.

If a coding agent recommends an OCR tool, worksheet generator, educational library, or game engine, ask it to show the project's status with `curiosity ecosystem show --id PROJECT`. The reviewed catalog defaults to reference-only; it does not let an agent install a public repository merely because it looks useful or popular.

## Everyday Slack commands

```text
children
connection
ask kid-a: Why does ice float?
inbox
assign inbox_... kid-a
dismiss inbox_...
feedback kid-a engaged: Kept asking about it
privacy
help
```

You can also send a quick note without a child ID. It will be saved as unassigned. This is deliberate: the engine will not guess which child a question belongs to.

## What stays private

The `private/` directory is excluded from Git. It holds family profiles, the local database, setup status, Slack tokens, purchased resources, and generated outputs. Public code, generic examples, schemas, and tests stay outside it. Read `docs/PRIVACY.md` before publishing a fork or opening an issue with logs.
