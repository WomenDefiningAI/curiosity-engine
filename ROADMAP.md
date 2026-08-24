# Curiosity Engine roadmap

## v0.1 — Agent-led Slack MVP

- [x] New-family instructions for Codex, Claude Code, and comparable coding agents
- [x] Parent-readable architecture, privacy, setup, restart, and customization walkthrough
- [x] Separate Slack transport proof from bring-your-own OpenAI/Anthropic/OpenRouter brain setup
- [x] Public multimodal requirements for reasoning, worksheet/PDF OCR, visual QA, and image generation
- [x] Curated public-project registry with canonical release feeds, review expiry, and no-execution defaults
- [x] External-project identity, license, maintenance, supply-chain, and elementary-fit vetting gate
- [x] Family Operator and privacy-safe Open-Source Steward agent personas
- [x] Deterministic `doctor` and `setup` commands with privacy-safe JSON output
- [x] Household owner, co-parent records, timezone, quiet hours, and weekly opt-in setting
- [x] Free family Slack workspace guide and least-privilege app manifest
- [x] Slack Socket Mode connector for parent DMs and explicit mentions
- [x] Short-lived pairing, exact user/conversation binding, revocation, and duplicate receipts
- [x] Unassigned capture inbox; child attribution is explicit and never guessed
- [x] Durable response outbox with bounded retry and ambiguous-delivery state
- [x] Shared service behind Slack, CLI, and loopback setup/review console
- [x] Bounded local family context and ignored purchased-resource collection
- [x] Ordinary household-material activities and optional downloadable paper PDFs
- [x] Episode-aware evidence foundation with retry/repair grouping and parent corrections
- [x] Safety gate that keeps context-driven suggestions disabled in v0.1
- [x] Public privacy, migration, Slack, workflow, and behavioral regression tests

## v0.2 — Learning-loop hardening

- [x] Lightweight parent feedback from Slack
- [ ] Slack accept/dismiss interactions for offered suggestions
- [ ] A parent-controlled paper-page request from Slack
- [ ] Background connector installer for supported home-computer platforms
- [ ] Clear local export and deletion walkthroughs
- [ ] Topic entity resolution and a visible curiosity ladder
- [ ] Shadow-mode evaluation for episode evidence and suggestion timing
- [ ] Trusted current-resource research with source provenance
- [ ] De-identified regression capture from parent corrections
- [ ] Broader factual, context, safety, and parent-effort evaluation coverage
- [ ] Repeated family evals that promote a recommended reasoning/vision/image stack
- [ ] Synthetic worksheet, phone-photo OCR, printable visual-QA, and image-edit eval suite
- [ ] Sandboxed OCR bake-off using only synthetic/public worksheets before selecting a local OCR integration
- [ ] Elementary interaction/game eval rubric before promoting any public engine beyond reference-only

## Later, after evidence from real family use

- Telegram through the same transport contracts
- Optional external context sources selected and authorized by each family
- Parent-authorized Gmail/school-email forwarding with least-privilege scopes, provenance, expiry, and background extraction
- Calendar context with explicit source controls and time-bounded retention
- Easier conventional-printer discovery/queues and parent-approved remote print workflows
- Broader parent roles or hosted operation under a separate threat model
- Longitudinal context audits and release-comparison automation

### Context-intelligence research track

These are possible future improvements, not dependencies or promises for the current MVP:

- **Embedding-assisted retrieval and episode comparison**, only after family-free evals show that it improves semantic matching without collapsing retries into interest. Families must be able to choose a local embedding model or explicitly authorize a provider.
- **An optional graph database**, only if measured scale or query needs outgrow the SQLite context projection. SQLite remains the default so a family can inspect, back up, and operate the harness locally.
- **Probabilistic knowledge or mastery estimation**, only after the harness captures meaningful child performance—not answer delivery or repetition—and can expose uncertainty, evidence, and parent corrections.
- **Evidence-gated proactive suggestions**, first evaluated in shadow mode and then explicitly enabled by a parent. They must require independent learning episodes, honor quiet hours and cooldowns, explain “why now,” and default to doing nothing when evidence is ambiguous or recent activity looks like repair or dissatisfaction.

The temporal evidence-and-episode foundation should be built and evaluated before any of these capabilities. The initial context-graph slice should not add embeddings, a graph database, probabilistic mastery, or new unsolicited suggestions.

WhatsApp, cloud multi-tenancy, child-facing chat, autonomous purchasing, automatic outbound campaigns, and specialized fabrication are outside this MVP.

## v1 outcome

A parent can clone the repository, point a coding agent at it, understand what stays local, connect a free family Slack workspace, add only the context they choose, capture a real question, receive a useful next thread, optionally download a trustworthy paper activity, and give feedback without exposing private family or purchased-resource data in the public repository.
