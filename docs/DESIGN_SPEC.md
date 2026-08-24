# Curiosity Engine — Full MVP Design Specification

## 1. Product thesis

Curiosity Engine is an open-source, parent-facing autonomous AI harness that helps families cultivate curiosity, resilience, reasoning, and age-appropriate academic growth through physical-world learning.

Children do not need to use AI directly. The system turns school, questions, interests, relationships, seasons, weather, family life, and prior experiences into timely parent prompts, resources, experiments, games, printables, books, and other physical experiences.

North stars:

> AI does the planning. Parents create the conditions. Kids do the thinking.

> Digital intelligence → physical childhood.

## 2. Why own the harness

The harness is the machinery surrounding models: event handling, state, context retrieval, model/tool routing, autonomous loops, validation, actions, logging, evaluation, and release control.

Curiosity Engine owns this layer because its quality depends on domain-specific controls that general agent runtimes treat generically:

- child-state epistemics;
- longitudinal concept/interest graphs;
- explicit context depth;
- adversarial pedagogical/factual/context review;
- parent-effort optimization;
- safe bounded autonomy;
- physical artifact trust;
- regression control across every behavioral component.

Messaging transports are optional input/output surfaces. The channel-neutral harness remains the source of truth and orchestrator.

## 3. Two core product loops

### Reactive: Pull the Thread

Child asks/shows interest → parent captures locally or in paired Slack → question persists immediately → relevant graph projection → curiosity response → optional accurate physical artifact → feedback → graph update.

The objective is not maximum explanation. It is the probability that the child wants to know one more thing.

Preferred sequence:

WOW → NOTICE → PREDICT → NUGGET → NEXT QUESTION → DO

### Proactive: Curiosity Producer

The autonomous Director considers only parent-authorized school, life, interest, environment, and experience context. It creates multiple candidate opportunities internally, critiques them, and surfaces very few. `do_nothing` is always a candidate.

## 4. Runtime boundary

### Code owns

- events and routing;
- database and transactions;
- context-depth policy;
- deterministic retrieval mechanics;
- schemas and validation;
- jobs/retries/idempotency;
- artifact rendering;
- printable generation, validation, and download mechanics;
- permissions/approval boundaries;
- run logs;
- release manifests;
- evaluation orchestration.

### Models own

- semantic extraction;
- concept/entity interpretation;
- curiosity hooks;
- conceptual next-step reasoning;
- opportunity generation;
- pedagogical synthesis;
- ambiguity interpretation;
- adversarial critique.

Rule: if a step must reliably happen every time, code owns whether it happens.

## 5. Context Graph

The graph represents questions, topics, concepts, school context, life events, experiences, and evidence—not merely a text summary of the child.

Each child can accumulate:

- exact questions;
- topics/interests and recurrence;
- concepts encountered;
- concept prerequisite/next-thread relationships;
- school concepts;
- teacher language;
- SEL themes;
- life events;
- experience history;
- observed engagement;
- candidate misconceptions;
- evidence-backed claims.

### Knowledge state

`unseen → exposed → emerging → demonstrated`, plus `unknown` when evidence is insufficient.

An explanation being delivered only proves exposure.

### Epistemic state

`observation → hypothesis → established_pattern`

No automatic trait inference from isolated events.

### Provenance

Durable claims carry source, timestamp, confidence, supporting evidence, contradicting evidence, and producing extractor/version where available.

## 6. Context Builder

Models never receive the entire family history by default.

Depth 0 — identity/current request.

Depth 1 — direct recent observations, school signals, direct topic matches.

Depth 2 — related concepts/questions/interests plus recent experiences.

Depth 3 — broader graph neighborhood and edges.

Depth 4 — longitudinal evidence, claims, hypotheses, contradictions.

Workflows pin depth. Context policy is itself versioned and evaluated.

## 7. Reasoning architecture

Reasoning topology is policy.

### FAST

Immediate parent interaction. Example: `pull_thread` uses depth 2, one generator, factual/pedagogy/context critics, and at most one revision.

### NORMAL

Artifact/experience production. Generator plus factual/visual validation.

### DEEP

Weekly reflection. Multiple lenses (school, curiosity, life/SEL), broader context, parent-effort/pedagogy/context/factual critics, selector.

### LAB

Background challenger experiments, context audits, evaluator calibration, model/resource radar.

## 8. Adversarial reasoning

Independent critic roles attack output:

- factual: unsupported/overstated/uncertain claims;
- pedagogy: spoon-feeding, theory-first instruction, cosmetic theming;
- context: missed relevant state, irrelevant retrieval, unsupported child inference;
- parent-effort: unrealistic weekday burden;
- epistemic: confusion of exposure/understanding or observation/trait;
- visual: malformed/misleading artifacts.

When useful, different provider/model families may fill generator, critic, and judge roles to reduce correlated failure.

## 9. Autonomous Director

The Director has four responsibilities:

1. **Sense** — ingest changes/signals.
2. **Reflect** — determine what changed and what may matter.
3. **Generate** — create candidate opportunities.
4. **Select/delegate** — choose very few and dispatch known workflows.

It may not directly bypass approval boundaries or mutate production releases.

### Opportunity dimensions

- timeliness;
- developmental value;
- school connection;
- child interest;
- curiosity potential;
- SEL relevance;
- physical-world engagement;
- novelty;
- sibling compatibility;
- available materials;
- schedule fit;
- parent effort;
- recent repetition.

## 10. Inputs

Transport code normalizes explicit parent inputs into Event objects. It never owns family state or context policy.

MVP event sources:

- parent text in the local setup/review console or CLI;
- a direct message or explicit mention from a paired parent in the configured Slack workspace;
- parent-entered observations and school context.

Future sources require an explicit product, privacy, provenance, and evaluation decision. Placeholder adapters are not architecture. The core runtime does not depend on Slack-specific types.

## 11. School context

School ingestion is separate from weekly planning.

Extract:

- academic concepts;
- current books/themes/projects;
- teacher terminology;
- classroom culture/resilience phrases;
- upcoming school events;
- source/provenance/freshness.

Planner consumes normalized school state, not a giant mailbox dump.

## 12. Curiosity ladders

Topics may have conceptual neighborhoods/prerequisite ladders, but the engine does not auto-march children through a curriculum.

The Context Graph answers: what has this child encountered, apparently understood, wondered about, and what is one compelling next rung?

The child controls whether the thread deepens.

## 13. Experience types

Core reusable structures:

- LAB — phenomenon → prediction → investigation → observation → iteration → explanation;
- MISSION — constrained design/engineering challenge;
- CASE FILE — narrative reading/logic/math/science mystery;
- FIELD QUEST — outside observation/classification/measurement;
- MAKE — art/design/invention/model-making;
- REAL WORLD JOB — cooking, measuring, budgeting, fixing, organizing, navigating;
- WONDER — one unusually good question/phenomenon;
- CONVERSATION — one reasoning/SEL prompt.

## 14. Parent-effort budget

Weekday defaults:

- prep ≤5 minutes;
- activity roughly 10–30 minutes;
- low cleanup;
- no special shopping.

The system optimizes for parent attention as the scarce resource.

## 15. Physical artifacts and trust

Ordinary paper, writing utensils, and common household materials are the physical MVP. A validated one-page PDF is an optional output that a parent may download and print at home or a library; core learning threads do not require a printer.

Artifact types include reference pages, wonder pages, mini posters, challenge cards, case files, field guides, cut-and-build pages, mini books, games, and intentional practice worksheets.

Trust pipeline:

verified facts → trust tier → asset strategy → structured artifact spec → deterministic layout → machine validation → multimodal visual QA → parent preview/download.

Generative imagery is appropriate for decorative/atmospheric illustration. Trusted sources or deterministic diagrams are preferred when position, count, labels, anatomy, maps, scientific structure, or geometry convey knowledge.

Text is rendered by code, not baked into generated images.

The MVP has no 3D-printer integration, 3D-model output, slicer files, filament instructions, or 3D-printing roadmap scope.

## 16. Model routing

The runtime defines roles rather than one global model:

- reasoning;
- structured_extraction;
- image_generation;
- visual_qa;
- cheap_classifier;
- critic_factual;
- critic_pedagogy;
- critic_context;
- critic_parent_effort;
- critic_epistemic;
- judge.

Operator config maps roles to provider/models. Production releases pin the mapping. Lab challengers may change one role at a time.

## 17. Job/runtime reliability

Every workflow run records event, workflow/version, context policy/depth, models/roles, tools, state reads/writes, actions, latency/cost where available, validation, and error status.

Known mechanics use deterministic jobs with explicit states: queued/running/completed/failed/retrying.

Jobs are idempotent and safe to retry.

## 18. Reliability Lab

Self-improvement occurs in the Lab, never by silent production mutation.

Any change to models, prompts, schemas, graph rules, retrieval/context policy, reasoning topology, tools, renderer, resources, evaluators, or orchestration creates a challenger.

Lifecycle:

CHAMPION → CHALLENGER → TARGETED EVAL → GLOBAL GOLDEN/REGRESSION → SHADOW → CANARY(optional) → OPERATOR PROMOTION → ROLLBACKABLE RELEASE.

Every real-world regression becomes a permanent fixture.

## 19. Eval suites

At minimum:

- curiosity quality;
- cumulative context use;
- factuality/uncertainty;
- parent effort;
- safety/social inference;
- artifact correctness;
- visual quality;
- autonomous Director opportunity selection;
- context retrieval/depth;
- historical regressions.

Evaluators combine deterministic assertions, trusted facts, independently prompted judge models, human-labeled examples, and real parent feedback.

## 20. Background improvement

Lab jobs may periodically:

- run regressions;
- inspect recent failures/corrections;
- benchmark new models against specific roles;
- test prompt/skill/retrieval/orchestration challengers;
- refresh vetted current-world resource candidates;
- audit context claims for weak/stale evidence;
- generate promotion reports.

No automatic production promotion in MVP.

## 21. MVP success criteria

A parent can tell the paired Slack bot:

> Kid A wants to know why airplanes stay up.

The system must reliably:

1. identify/persist the event;
2. retrieve the appropriate child context depth;
3. not forget prior aerodynamics knowledge;
4. generate a curiosity-first response;
5. run required critics;
6. update graph state transactionally;
7. optionally produce a trustworthy downloadable one-page paper artifact;
8. keep the activity usable with paper and writing utensils even without direct printer access;
9. log the complete run;
10. turn later corrections into regression cases.

The parent should not manage model prompts, file paths, state synchronization, transport internals, or context selection.
