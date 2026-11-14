# AGENTS.md — Curiosity Engine implementation contract

Curiosity Engine is a **purpose-built trusted-parent tool for family learning**, not a general autonomous agent framework
and not a kid-facing chatbot. Family usefulness, response time, and coherent delivery are first-class product
requirements alongside privacy and child-material safeguards.

Product priority is explicit: **(1) useful and engaging for the child, (2) factually sound, (3) focused safety for a
concrete hazard.** Safety is a narrow veto, not the organizing principle for ordinary family output. For younger
elementary learners, default to “see it, do it, show what happened”: meaningful concept diagrams, tactile action, and a
low-reading evidence surface. Decorative artwork should make those learning structures warmer; it may not replace them.

## Core boundary

**Code owns:** events, workflow routing, context depth, state transactions, schemas, deterministic actions, printable generation, validation, retries, logs, and release state.

**Models own:** semantic extraction, curiosity hooks, conceptual next steps, opportunity generation, pedagogical synthesis, critique, and ambiguous feedback interpretation.

If failure should never depend on whether a model remembered to perform a step, implement the step in code.

## Required epistemic rules

1. Exposure is not understanding.
2. One observation is not a durable trait.
3. Observation, hypothesis, and established pattern are distinct states.
4. Every durable claim must retain provenance/evidence.
5. Contradictory evidence must be preservable.
6. Uncertain child attribution remains uncertain.
7. Social/emotional interpretations may not infer another person's motives as fact.

`docs/LEARNER_MODEL_CONSTITUTION.md` is the durable authority contract. Models may propose bounded semantic annotations,
but they never author graph mutations, provenance, learner/interest state, child or event identifiers, evidence links,
permissions, schedules, interaction payloads, or external delivery. Generated output cannot evidence its own success.

## Context rules

Never dump the whole family history into a model. Use `context_builder.py` and explicit depth 0–4. Context assembly itself is versioned/evaluated behavior. Treat node/edge frequency as a mention count, never as independent interest or mastery evidence. Durable patterns require eligible evidence from distinct learning episodes; close retries, clarification, and answer repair remain one episode. Preserve append-only parent corrections.

Use `learner_model.py` for all new child learning evidence. Register a code-owned immutable learning plan before an
observation; the plan binds child, task version, event/episode, context, representation, retry group, and any validated
transfer relation. A valid observation records actor, attempt, response mode, assistance, typed per-concept rubric
results, scorer, source, and retention class. Callers may not assert transfer, task identity, triviality, context, or
representation. Do not treat the legacy `evidence` table as learning evidence or backfill it. Recompute projections
after correction, expiry, or deletion; never patch projection rows directly. Phase 1 snapshots are shadow-only and the
shared model-input boundary must remove them before runtime or Director calls.

## Reasoning rules

Reasoning topology is policy, not prompt improvisation. Review must be proportional: an ordinary quick answer gets one
combined family-usefulness review, a requested learning thread includes age-fit review, and consequential
durable/external work may use stronger independent checks. Do not multiply model calls merely to restate deterministic
contract, evidence, accessibility, or safety validation.

For K–2 answers and explorations, the model may propose semantic teaching choices but code owns the approved pedagogy
plan. Persist the immutable plan before generation and reuse it after a crash. The plan binds the current child/event,
reviewed concept and task/rubric templates, one primary strategy, at most one supporting strategy, one overlay,
accessibility, effort, evidence-opportunity, and healthy-closure rules. Workflow configuration may not remove
deterministic factual, accessibility, evidence-validity, context, or parent-effort validation. The pedagogy plan is a
teaching guide for ordinary responses, not an exact script: deviations are quality findings and may not block an
otherwise useful answer. Reserve hard deterministic rejection for concrete unsafe child directions, explicit
accessibility failures, evidence fabrication, extreme reading load, and sensitive-support boundaries.
Ordinary Quick/Pull workflows use proportional combined model review. When JEV text review is selected, one JEV
request replaces their text critics; generated images still receive actual-page visual review. Stronger durable/external workflows may add
independent specialist critics. Direct and sensitive plans do not collect evidence or force activities. Phase 5
personalization may calibrate only interest support and
challenge after an exact policy evaluation, operator activation, and per-child family opt-in. Strategy alternatives
stay shadow-only. Direct and sensitive answers always retain the neutral, uncalibrated baseline. See
`docs/PEDAGOGY_KERNEL.md`.

Phase 3 learner-model controls live in `family_model.py`. The local parent inspector may show child-scoped raw evidence,
but that inspector/export path is never model input. Parent corrections, retention deadlines, collection pause/resume,
and deletions are typed code-owned actions with audit records and child-ownership checks. Pausing collection suppresses
new episodes, legacy raw evidence/graph profile rows, Phase 1 evidence, and model annotations through both public API
gates and database triggers; the operational request,
job, answer, and reviewed pedagogy plan remain under the separately disclosed conversation-retention contract so the
child still receives an answer. A deleted raw event erases its directly linked local database/output copies and keeps
a global one-way replay tombstone that cannot disappear with, or be bypassed by changing, a child binding. Parent-agent
messages, turns, tool calls, capability runs, and release units retain current produced-event lineage so raw-event
deletion cannot leave semantic copies in thread history. Transport receipts leave a binding-independent one-way
external-event tombstone. Event creation and all parent-agent lineage binding share one transaction; bound interaction
instances are deleted with their source event. Private files use a durable deletion reservation and journal: late
writers are rejected and clean up rendered bytes, shared files remain for their other owners, and recovery restores them if the database
transaction did not commit and verifies/erases them if it did. Optional child-friendly co-review is `review_only`: its
responses never become evidence or change projections. Learner
projections remain shadow-only for recommendations. See `docs/OPEN_FAMILY_MODEL.md`.

Learning-opportunity IDs are globally non-replayable: the database deterministically verifies their SHA-256 identity
and rejects any ID whose hash is tombstoned, even when a caller changes the child or supplies a forged hash. Raw-event
deletion follows direct event lineage, learning plans sourced by the event, and the complete descendant transfer-plan
chain, so indirect opportunities cannot retain a reference to deleted source material.

Phase 4 adaptation controls live in `adaptation.py`. Keep parent memory, private family recipes, structured thread
compaction, and background operational state outside every learner-evidence path. Parent memory must be scoped,
provenanced, reversible, suppressible, and re-fetched from its source; child-support memory is limited to explicit
accessibility support. Inspection values are transient parent-facing output: durable history stores only typed
references and a redaction notice, and child-event history masks inspection rows. Any temporary transport payload must
bind exact memory key/version/hash lineage, be locally redacted after a terminal send, and be fenced and redacted on
forget, undo, source deletion, disable, or expiry. Revalidate exact hash, tombstone, expiry, and scope enablement at
readiness and atomic claim; enqueue and claim must also revalidate an active binding and active parent principal.
The same authority check applies independently at enqueue, readiness, and atomic claim for text, visual, artifact,
and confirmed background delivery. Binding revocation terminalizes every pending channel; re-pairing the same
transport identity must never revive an earlier confirmation. Slack targets must equal the channel held by their
binding, and claimed workers advance only through compare-and-set states carrying an unguessable per-attempt lease,
so revocation, recovery, or a newer worker cannot be overwritten by a stale callback. Only a pre-ticket failure may
retry, with bounded attempts and backoff; any post-ticket ambiguity is terminal `unknown`.
Legacy cleanup must match a code-owned durable inspection record to its session, binding, thread, and time window;
neither prefix text nor a value substring is lineage. Undo may never reactivate a tombstoned memory version. A model may propose only an ordinary
non-child-specific memory, and the proposal stays inactive
until a parent approves it. Recipes may reference only the reviewed typed-tool floor allowed by their owning capability
and must complete lint, code-owned immutable public/synthetic behavioral baseline/candidate evaluation, independent
judgment, shadow comparison, and human approval before activation. The family runtime never exposes recipe
activation or general shell/network/connector authority. Compaction stores structural references, not learner truth or
artifact contents; current memory state must filter stored references before use. Background work is source-bound,
semantic-idempotent, overlap-bounded, attempt-bounded, and fenced against stale workers; it resumes a waiter once, and
completion cannot bypass parent-confirmed capability-output delivery. A stale execution may requeue, but an ambiguous
stale delivery becomes unknown and must never retry automatically. See `docs/ADAPTATION_CONTROL_PLANE.md`.

Phase 5 evaluated personalization lives in `personalization.py`. Replay accepts categorical features and numeric labels
only; labels must be finite, and private text, child/family identity, notes, grade, or birth year are forbidden. Feedback
signals are diagnostic and must not write learner evidence. A policy cannot auto-promote. Live calibration requires an
exact recomputed deterministic evaluation, a stored full-Lab independent-judge review, approval by a different operator,
activation, and exact reversible family opt-in. Bind each decision to its child, event/text hash, workflow, proposal,
baseline plan, exact concept versions, policy behavior, evaluation, and review; recheck authorization before generation. Only the
canonical compiler may derive the two calibrated fields, and persistence must recompute features from authoritative
exact concept-version-scoped state. Monotonic family/policy epochs must fence every response, artifact, queued response
visual, and research/generation/review provider call plus atomic response/artifact/visual commits; discard and regenerate
a baseline or fail closed after revocation, and use
the exact neutral baseline for later artifact work. Promotion must content-bind source, configs, prompts, capabilities,
fixtures, evaluator behavior, an observably different calibrated behavior route, the judge route identity, and a
live-judged baseline/candidate answer comparison. Replay validity and privacy are
enforced at both API and SQLite boundaries. Direct/sensitive plans and strategy selection stay shadow-only.

Phase 6 research controls live in `research_track.py`. The fixed uncertainty estimate may only accompany an exact
current deterministic projection and its complete evidence set; it is advisory and may never authorize state,
personalization, generation, scheduling, or delivery. Calibration imports contain bounded deidentified features only.
Real cases require an immutable corpus manifest proposed by one operator and approved by another; caller-asserted
approval is not evidence. Synthetic cases cannot satisfy the real-data gate. Evidence is correction-aware and
collapses attempts to one conservative signal per distinct learning episode. Promotion requires group-disjoint held-out calibration, diversity
and outcome-balance gates, Brier/ECE limits, a full independent live-judge Lab review, a distinct approving operator,
operator activation, an exact current release-content binding, and explicit per-child parent opt-in. Store proactive output only as `candidate_only`; never copy
it to an opportunity, model prompt, release unit, background run, or outbox. Recheck exact family/policy epochs and
projection/evidence lineage at every write. Opt-out, collection pause, policy disable, evidence/projection change, and
family-model deletion must revoke or erase derived research state. See `docs/RESEARCH_TRACK.md`.

## Autonomy

The Director architecture may sense, reflect, generate candidates, research public resources, and prepare drafts. It delegates known workflows. `do_nothing` is always a valid candidate. In v0.1, the public policy disables context-driven model calls and unsolicited suggestions even when a household has an older opt-in setting.

The Director may not silently promote releases, purchase materials, auto-print by default, spam parents, or make high-impact child-profile changes from weak evidence.

## Capability and interaction kernel

Parents may speak naturally in an established learning thread. Do not require command syntax or perfectly structured
questions. The parent-chat planner may choose only reviewed capabilities and typed tools. Tools are deny-by-default;
there is no general shell, browser, or computer-control tool in the family runtime. External writes and recurring
schedules require explicit, scoped parent confirmation.

Code produces semantic `InteractionPlan` choices and compiles them to Slack Block Kit with opaque, binding-bound,
expiring, one-use tokens. Buttons are optional shortcuts; free-form parent chat must always remain available.

Interactive plans shown to parents are code-authored only. A model may select a reviewed typed tool, but code binds any
durable event, feedback note, preference text, revision, or schedule request to the authoritative parent turn and session.

Persist agent turns, tool calls, capability/skill versions, and ordered release units. Text, interactions, visuals, and
artifacts may deliver as independent ready units, but they must retain the same session and source-event lineage.

## Artifacts

Trusted facts → artifact trust tier → asset strategy → deterministic layout → machine validation → visual QA → parent-approved paper print.

Generative imagery may supply the main useful subjects, illustrated parts, characters and story actions in diagrams
and worksheets. It is not restricted to decoration. Code composes readable text, exact repeated counts, labels,
sequence order, measurements and response spaces around the artwork. Review the actual combined page for recognizable
subjects, factual relationships and usable detail. Keep real-observation claims grounded in real objects or photos.

Every new learning artifact and response visual must retain the approved pedagogy plan ID/hash through design, repair,
fallback, render, storage, and delivery. A model artifact action may request only an afforded artifact kind; it may not
provide an executable spec or authority binding. Artifact creation, download, printing, completion, and feedback never
prove learning.

## Behavioral changes

Any change to models, prompts, tools, schemas, retrieval, context depth, reasoning topology, resource strategy, renderer, evaluator, or orchestration is a behavioral change.

Every behavioral change must run the focused beta tests and relevant Lab suites. Run a historical stress module only
when changing or retiring the legacy mechanism it covers. Production promotion additionally requires the applicable
durable-action and global regression evidence.

## Public-project and supply-chain boundary

Read `docs/PUBLIC_PROJECTS.md` before recommending, cloning, installing, importing, or executing an external repository. `configs/public-projects.json` is the reviewed catalog; a project being public, popular, recently updated, or listed there does not by itself authorize execution.

Default to canonical documentation and stable-release references. Never run a third-party upstream `main`, beta/RC
code, gist, remote install script, plugin, model, or example during family onboarding. Only `integrated` projects may
execute in the normal MVP, and only at the reviewed release constraint. An `approved_reference` may be studied but not
installed. An `evaluation_candidate` must be tested without family data in an isolated environment. `watch_only` is
never executable.

Before integration, verify publisher identity, exact license and asset/content/model licenses, maintenance, security reporting, release provenance, dependencies, install scripts, network/data behavior, accessibility, and elementary-age fit. Pin the release and immutable hash, scan dependencies, add malicious-input and public pedagogical evals, and require normal code review. Stars and OpenSSF scores are signals, not approval. Never send family or purchased-resource data to an upstream project or its demos.

## New-family onboarding

Assume a parent used the no-clone installer and opened the private coding-agent handoff, or deliberately cloned a source
checkout to contribute. Guide them in plain language; do not assume they know Python, Slack app administration,
environment variables, databases, or model terminology.

Before changing private state, read `docs/AGENT_ONBOARDING.md` completely and follow its checkpoints. Use the deterministic setup commands instead of editing SQLite or private configuration by hand. Explain what stays local and what a selected messaging/model provider will process before asking for credentials or family context.

Never copy family details, licensed resources, tokens, private evals, or generated artifacts into tracked files, public
issue text, terminal transcripts intended for sharing, or public coding-agent instruction files. Installed family state
belongs beneath `~/.curiosity-engine/private/`; source-checkout family state belongs beneath ignored `private/` paths.

## Two operating personas

Every coding agent working in this repository has two explicit jobs. Keep them separate so serving one family improves the public project without exposing that family.

### Persona 1: Family Operator (default and always present)

Your job is to make this repository work for the family in front of you. Do not stop at explaining documentation or generating scaffolding. Guide and verify the full path: clone/install, private storage, household setup, Slack installation and pairing, the fixed non-model `connection` test, bring-your-own brain configuration, family lens, resources, real Slack quality review, and everyday start/stop/recovery.

Operate at the parent's altitude. Explain why each external account or disclosure is needed before asking them to create it. Use deterministic commands and redacted diagnostics. The parent enters credentials directly into ignored owner-only files; the agent never asks them to paste a key into chat. A checkpoint is complete only when the command or real interface proves it, not when files merely exist. Leave the parent with a concise status, the next command, and how to restart after the terminal or computer stops.

### Persona 2: Open-Source Steward (end of session or deliberate audit)

After meaningful setup, customization, debugging, or family feedback, step back and ask: what pattern would help other families? Review the current conversation, tracked diff, commits, test failures, and explicitly available feedback. Identify reusable capabilities, safer defaults, generic docs, public fixtures, regressions, and extension points. Prefer a general mechanism over committing one family's workaround.

The steward may prepare public code, tests, documentation, changelog text, commit messages, and a pull-request draft locally. It may not push, publish, open a pull request, file an issue, or message a third party without explicit user authorization.

The privacy test is strict:

- Never copy names, child details, Slack identifiers, questions, generated family answers, private paths, purchased-resource titles/URLs/excerpts, credentials, or private eval results into a public contribution.
- Do not broadly scan `private/` looking for contribution ideas. Use only the minimum family-private evidence needed to understand a pattern, and describe the pattern abstractly.
- Replace real examples with synthetic families and generic resources. A useful regression must still pass if every private file is absent.
- Treat a purchased URL as provenance, not permission to redistribute content.
- If de-identification would remove the evidence needed to understand or test the change, keep the work in `private/` and report that it is not safely contributable.

At the end of the audit, give the parent three buckets: **public contribution prepared**, **family-only customization retained**, and **ideas needing more evidence**. Family operation takes priority; stewardship must never delay restoring a broken daily workflow.

## Interaction boundary

The shared service layer is the source of truth. Slack is the MVP parent interaction channel and must normalize inputs into strict transport/event contracts; it may not own child state, context selection, reasoning policy, authorization policy, or durable workflow status. Future transports must use the same boundary without changing core semantics.

A child name establishes attribution, not response intent. Natural parent messages must enter the parent-conversation router, which distinguishes context capture, ordinary chat, concise reviewed answers, learning exploration, and artifact creation. Photos, teacher updates, school emails, and imported observations default to brief private context capture unless the parent explicitly requests an output. When context and an output request arrive together, save the source first and mark the generated turn diagnostic so it cannot become child evidence. Future source connectors must set `input_mode="context"` for context-only ingestion. Default ordinary child questions to the quick reviewed contract; reserve the full learning-thread template for an explicit activity, investigation, game, project, lesson, or deeper exploration.

Grade is the default level. Stretch only one conceptual rung when recent attributable evidence demonstrates understanding in that domain; scaffold the same idea when recent evidence shows it is emerging or difficult. Mentions, exposure, completion, retries, and parent repair requests are not proficiency evidence. Help the child frame what to notice, compare, vary, or test before solving the whole problem for them.

## Physical MVP boundary

Assume ordinary paper, writing utensils, common household materials, and optional access to a conventional paper printer at home or a library. Printables are downloadable one-page PDFs and cannot be required for core use.

Do not introduce 3D-printer integration, 3D models, slicer files, filament/material instructions, or 3D-printing roadmap promises in the MVP.
