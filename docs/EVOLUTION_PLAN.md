# Curiosity Engine: context, pedagogy, and adaptive-harness plan

**Decision brief · 2 September 2026 · Research snapshot**

 **Canonical use:** implementation playbook for the live deployment checkout. Before each implementation slice, re-read the live branch, `AGENTS.md`, recent commits, and dirty diff; the repository was four commits ahead of the pinned public baseline when this research was completed. Never inspect or copy private family content unless the task explicitly requires it.

**Implementation status (9 September 2026):** Commit `4e6dce4` preserves the complete Phase 0–6 software candidate,
its 1,915-case validation evidence, and two independent approvals of the same frozen tree. It is a recovery point, not
a declaration that every implemented control belongs in the beta. The current work deliberately simplifies from that
point. Phase 6 remains closed and non-deliverable, and consented parent/child co-design has not occurred.

## Trusted-parent beta reset

Curiosity Engine is a tool for an authenticated, trusted parent who may use its output with a child. The beta does not
need to defend against a malicious parent, deliberate SQLite manipulation, hostile backup replay, or arbitrary
self-harm and physical-hazard paraphrases entered to defeat the harness. Its credible risks are model mistakes,
private-family data leaking into the wrong context or artifact, developmentally unsuitable child-visible material,
and a durable or external action occurring without informed parent consent.

This reset changes the amount of machinery attached to each action, not the core learner-model commitments:

| Tier | Beta behavior | Required evidence |
|---|---|---|
| **1 · Ordinary parent assistance** | Answer, explain, brainstorm, or prepare a draft for the parent. | Lightweight contract, factual, privacy, pedagogy, and regression checks. No adversarial-parent language gate. |
| **2 · Child-usable material** | Material the parent may show, read, print, or do with a child. | Focused developmental, accessibility, model-output, and physical-safety checks plus parent review before delivery/use. |
| **3 · Durable or external action** | Save or delete family state, activate a recipe/policy, contact a service, schedule work, or otherwise persist/act beyond the current draft. | Explicit consent, provenance, revocation/rollback behavior, and stronger targeted testing. |

The executable promotion policy now partitions every active Lab suite into exactly one of these tiers. Active safety
validation shrinks from 1,829 mandatory cases to five representative child-material cases: two unsafe model-output
rejections, two protective contrasts, and one social-uncertainty check. The complete 1,829-case
input-routing corpus remains at `evals/history/safety/`; it is excluded from active discovery and the release-content
manifest. The active offline Lab is now 91 cases, including seven family-usefulness and six performance/coherence
contracts. The default pytest gate is 268 representative cases instead of 3,754; full phase, routing, capability,
Slack, artifact, and visual stress modules remain explicitly runnable but are excluded from default discovery. A
collection-budget check caps the beta gate at 275 and requires
usefulness plus performance coverage to exceed focused safety coverage.

An ordinary quick answer now uses one combined family-usefulness critic rather than separate factual, pedagogy,
context, plan-fidelity, accessibility, evidence, and parent-effort model calls. Its healthy path is bounded to one
pedagogy-plan call, one generation, and one review. Quick answers remain text-first unless a parent explicitly requests
a visual. An exploratory compare/test/observe/sequence question may produce one optional learning diagram from the
main generation without adding a hidden completion call. Code-rendered labels, panels, ordering, and geometry carry
the learning; image generation may decorate that diagram or a worksheet to make it warm and engaging. Standalone
topic illustrations are not shipped. These constraints directly address observed beta latency and decorative-output
failure without weakening deterministic contract, privacy, evidence, accessibility, or model-output safety validation.

A 10 September live beta retry exposed another over-strict boundary: an otherwise useful answer was rejected because
an adult-owned equipment precaution introduced a hazard word absent from the pre-generated pedagogy plan. Ordinary
plan fidelity is now advisory. The plan continues to supply age-appropriate concepts, strategies, representations, and
learning moves, but a sound alternative or an omitted optional activity cannot suppress the answer. Hard response gates
remain for concrete unsafe child directions, explicit accessibility failures, evidence fabrication, extreme reading
load, and sensitive-support boundaries. Malformed planner evidence metadata is normalized to non-collection while its
useful teaching intent is preserved. In Slack, a household with exactly one configured child now attributes new root
messages to that child automatically; the assignment menu remains for genuinely ambiguous multi-child households.

A subsequent live “Make a challenge” click exposed the more important acceptance gap: two slow model attempts ended in
an invalid response, then the local fallback rejected its own placeholder copy and delivered nothing. The repair makes
structurally malformed artifact output fall back after one provider attempt, replaces placeholder challenge mechanics
with child-facing choices, and adds a concrete gas-and-container investigation for the observed class of question. The
default gate now exercises the complete context-to-PDF path and requires persisted output, actionable child steps, a
markable evidence surface, and passing visual QA. Hot processes also remember that the current SQLite file is already
initialized, avoiding repeated backup/schema lifecycle work on every Slack and visual-worker loop while a lightweight
journal check preserves interrupted private-file deletion recovery.

The existing cross-cutting `regressions` suite mixes ordinary, child-material, and consequential state-transition
cases. It is assigned as a whole to Tier 3, the strongest gate, instead of adding case-level policy machinery. Tier
case totals must equal the complete Lab summary, and a current report that omits regressions is rejected.

### What remains product-defining

- Evidence, explicit preferences, and inferences remain separate and provenance-bearing.
- Learner state remains conservative, recomputable, parent-inspectable, correctable, exportable, and deletable.
- A model cannot silently promote an observation into a trait, permission, or durable procedure.
- Child-usable output retains focused developmental, accessibility, privacy, and physical-safety validation.
- Durable/external actions retain explicit parent consent, an inspectable reason, and rollback or deletion behavior.

### Complexity retirement register

Completed code is not automatically permanent. Each item below is a compatibility candidate and should be removed in
a small, reversible slice only after its replacement regression passes:

| Existing complexity | Beta disposition now | Measured follow-up |
|---|---|---|
| Exhaustive linguistic self-harm/hazard parsing | Stop expanding it; archive its mandatory corpus. Keep runtime compatibility for this release. | Remove inbound-only branches that have no observed parent workflow, retaining generated-output rejection and protective-language acceptance tests. |
| Tamper-evident authorization chains and deliberate direct-SQL defenses | Keep ordinary API validation and provenance; stop treating a hostile local database owner as an actor. | Collapse redundant hashes/epochs behind a migration; prove normal consent, correction, deletion, restart, and rollback still work. |
| External high-water ledgers and multi-root backup authority | Freeze feature growth. Do not make them a beta launch condition. | Replace with one understandable local backup/restore contract and a parent-visible stale-backup warning; test one current restore and one post-deletion stale restore. |
| Pervasive process-kill matrix | Preserve as historical stress evidence while changes overlap the old code. | Reduce the required matrix to one interruption/recovery test per consequential state transition; move implementation-specific permutations to an optional legacy lane. |
| Phase 5/6 promotion machinery | Tier 3 only. It must not burden an ordinary answer or parent-reviewed draft. | Collapse duplicated reviewer/operator/epoch plumbing around disabled research and shadow personalization while retaining explicit activation consent and provenance. |
| Oversized mandatory safety corpus | **Reduced now:** 1,829 → 5 active cases; full corpus archived. | Add a case only for a concrete child-visible model failure or demonstrated false positive, never merely another hostile-parent paraphrase. |

For every retirement slice, record before/after code and test counts, run the affected tier plus parent-facing smoke
flow, verify data migration/rollback where applicable, and keep the preservation commit reachable. Removal is complete
only when evidence/preferences/inferences, parent controls, child-output quality, and durable-action consent remain
covered.

## Implementation handoff

This section preserves the original implementation sequence. For a new checkout or reimplementation, start with
**Phase 0 only**: codify the Learner Model Constitution and close the direct model-authored state-mutation seam. Treat
each later phase as gated work, not an automatic backlog. Every slice should include its contract/migration changes,
deterministic tests, evaluation fixtures, documentation, rollback or recomputation behavior, and an explicit review of
privacy and durable authority.

Recommended first three implementation slices:

1. **State-authority hardening:** remove `SET_KNOWLEDGE_STATE` from model-authored contracts; emit only bounded observation/evidence candidates; add negative contract and runtime tests; document the invariant.
2. **Evidence envelope:** introduce versioned task, opportunity, assistance, rubric, provenance, and evidence records behind a migration; make retry and correction behavior idempotent; add recomputation tests before changing recommendations.
3. **Projection and context seam:** separate concepts from per-child projections, derive conservative learner states in deterministic code, and expose a budgeted snapshot plus evidence rationale to `context_builder.py`; keep projection use in shadow mode until calibration passes.

After each slice, run the affected tier and its deterministic regression set. Add a safety case only for a concrete
model-output, privacy, developmental, or durable-action failure. Use live judging as advisory evidence for child-usable
candidate comparisons, not as authority over deterministic gates or ordinary parent requests. Update architecture and
testing documentation in the same change.

## Decision

**Evolve the current engine; do not rewrite it.** Preserve Curiosity Engine’s strongest constraints, replace its generic child-state seams with an evidence-centered learner model, add a typed pedagogy-selection layer, and borrow selected OpenWorker control-plane patterns without inheriting OpenWorker’s broad trust surface.

The target is a Curiosity Engine that is **self-improving, not self-modifying**. Preserve its existing episodes, evidence independence, parent corrections, bounded context, typed tools, and promotion gates. Strengthen the system in three coordinated streams:

1. **Child context graph v2:** immutable learning events, explicit task and evidence models, stable concept identifiers, conservative code-owned learner-state projections, provenance, uncertainty, retention, and parent correction.
2. **Pedagogy kernel:** a typed learning-episode plan that selects from a researched K–2 repertoire—guided play, inquiry, explanation, dialogic interaction, multiple representations, scaffold-and-fade, teach-back, playful retrieval, reflection, and domain-specific overlays.
3. **Adaptive harness:** a parent-visible memory ledger, structured thread compaction, reviewed private family recipes, durable background runs, and shadow-mode experience replay inspired by OpenWorker.

**Immediate correction:** stop accepting a model-authored `set_knowledge_state` as a direct graph mutation. At the pinned public commit, the output contract permits that mutation and `graph.py` applies it without an evidence threshold. The model should emit bounded observation/evidence candidates; deterministic code should recompute every derived learner state.

**Confidence:** high on the architecture and safety direction; medium-high on the pedagogy repertoire because evidence varies by domain, age, and outcome; medium on ecosystem impact because public popularity signals are not measured user outcomes.

## Executive answer

OpenWorker validates an important product intuition. It is unusually flexible because it composes a provider-neutral model loop with tools, live-discovered skills, explicit memory, persistent sessions, scheduled work, event wakes, connectors, and a layered permission system. That architecture lets behavior change without retraining or replacing the model.

“Self-learning” needs a precise meaning here. At the pinned OpenWorker commit, no model-training, reinforcement-learning, or weight-update loop was found. The harness adapts through controlled state and procedure: explicit memories, user-owned rules, user-approved skill files, live skill discovery, compaction summaries, and durable scheduled or self-waking runs. This is good news for Curiosity Engine: the most useful mechanisms are inspectable, reversible, testable, and compatible with its local-first design.

Curiosity Engine already has a better domain foundation than a generic agent. Its context graph distinguishes events, episodes, hypotheses, and established patterns; established patterns need evidence from at least two eligible episodes; parent corrections are append-only; retries do not masquerade as independent interest. Feedback and evaluation are deliberately kept from silently rewriting child context. Replacing that with generic memory would be a regression.

The graph is nevertheless incomplete as a learner model. It does not yet represent tasks, opportunities, assistance, item difficulty, response modality, rubrics, context diversity, or separate kinds of uncertainty. Generic nodes mix ontology and per-child state, and model output can directly propose a knowledge-state update. Those gaps matter more than whether the storage engine is technically a graph database.

The pedagogy layer is also narrower than the research supports. Current skills do a good job with answer-first curiosity, mysteries, hands-on comparisons, hints, worksheets, and functional visuals. They do not yet provide a typed strategy router, evidence opportunities, spaced revisit plans, dialogic/shared-reading moves, teach-back, representation choice, scaffold fading, or domain progressions.

The architectural move is additive and staged: strengthen the evidence graph, introduce a **pedagogy kernel**, and build an **adaptation control plane** beside them.

## 1. Harness adaptation: what OpenWorker actually does

### Harness shape

OpenWorker’s turn engine assembles an agent prompt, tool registry, permission policy, workspace instructions, memories, a progressive skill catalog, and a live per-turn context provider. A turn may span multiple model-to-tool iterations. Low-risk authorized tools can execute concurrently, while tool calls, interrupts, resumes, and compaction checkpoints are persisted. Provider adapters and MCP connectors widen what the same loop can do.

The flexibility comes from separation of concerns:

| Plane | OpenWorker mechanism | Why it matters |
|---|---|---|
| Reasoning | Provider-neutral model router and multi-step turn engine | Models can change without rewriting the harness. |
| Capability | Stable platform catalog, MCP adapters, progressive skills | The model sees a compact menu and loads detail when needed. |
| Adaptation | Explicit memory plus user-approved skill authoring | Preferences and procedures persist without model retraining. |
| Continuity | Durable messages/tool calls, compaction, scheduled runs, self-wake | Work survives long contexts and asynchronous boundaries. |
| Authority | Risk classes, hard floors, approvals, reviewer, audit | Capability and permission are separate decisions. |

### The adaptation mechanisms

**Explicit memory.** The agent is instructed to save conservatively, ask before saving sensitive information, visibly acknowledge a save, update duplicates, and forget obsolete entries. Global and workspace memories are injected until they exceed a bounded budget; beyond that, summaries and recent items are shown with on-demand reads. Turning memory saving off stops writes but does not pretend existing context vanished. Persistent user rules remain separate and outrank learned memory.

**Progressive, live skills.** Only a skill’s name and description enter the initial context. The model loads full instructions on demand. The catalog is re-scanned rather than frozen for the life of the process. A model may propose a finished skill, but saving or replacing it is human-approved and path-restricted.

**Structured compaction.** When context is compacted, the summary is told to preserve the primary request, key decisions and reasons, artifacts and file pointers, errors and fixes, user corrections, and the current next step. It is explicitly told not to carry old file contents forward as truth; the agent should re-read them.

**Durable initiative.** Scheduled tasks create fresh run records and working folders, catch up once after downtime, and avoid overlapping runs. Self-wake records let a suspended task resume on a timer, completion, or event. Persistence is the mechanism; there is no implication that a model trained itself.

### Governance is part of the architecture

OpenWorker places non-negotiable hard floors ahead of user permission modes. Authority that persists—saving a skill or creating, changing, or deleting a scheduled task—remains human-only. Deferred-execution surfaces such as workflows and hooks are specially treated. A model-based reviewer can allow only ask-level actions; it cannot override hard denials and is designed to fail closed.

That is the conceptual lesson to retain: **a more capable agent needs a clearer authority lattice, not a larger prompt asking it to be careful.**

### How this compares with Curiosity Engine

Curiosity Engine is deliberately narrower and, for a family system, better bounded:

- Parents use Slack; children do not chat with the model directly.
- The model selects typed, policy-checked tools; the family runtime has no general shell or computer-control tool.
- The parent loop is bounded, can perform only one read-only recall pass, records a policy hash and capability/skill versions, and stops after side effects.
- One-off criticism remains revision feedback. Only an explicit “remember/from now on/for this thread” request becomes a reversible thread preference.
- Child context uses evidence states and episode independence, not free-form memory. Established patterns require two eligible episodes and can be downgraded when a correction excludes evidence.
- Evaluation and regeneration improve private fit, while public regression proposals are de-identified and opt-in. Promotion requires deterministic suites, an independent live judge, no key regressions, and operator approval.
- Context-driven proactive suggestions remain disabled pending shadow-mode and family review.

The main gaps are continuity and parent-owned procedural adaptation. The ordinary parent loop currently reads a bounded recent history (16 items in the pinned public code) and has thread preferences, but it does not yet have OpenWorker-style structured compaction or a full household memory ledger. The capability registry already supports `family_private` skill paths and records skill versions, but it does not yet expose a governed lifecycle through which the agent can draft and parents can evaluate and activate a family-specific recipe.

### Recommended architecture: an adaptation control plane

Keep three kinds of durable state separate:

| Layer | Meaning | Who may create or promote it | Model use |
|---|---|---|---|
| **Evidence layer** | What has happened in a child’s learning: events, episodes, claims, confidence, corrections | Code-owned ingestion; parent corrections; evidence thresholds | Bounded pedagogical context. Existing Curiosity Engine graph remains authoritative. |
| **Preference layer** | Explicit parent/household working rules: tone, format, accommodations, delivery preferences | Parent request or parent confirmation; visible and reversible | Shapes responses, never becomes evidence about the child. |
| **Procedure layer** | A reusable family workflow: “when we ask for a worksheet, use this shape and review checklist” | Agent may draft; tests plus parent/operator approval activate a version | Loaded progressively by an authorized capability; immutable version recorded per run. |

This separation prevents the most dangerous collapse: turning an inference about a child into an invisible instruction that repeatedly shapes future interactions.

### 1. Extend thread preferences into an explicit memory ledger

Build on the existing append-only preference events rather than introducing an unscoped vector-memory bucket. Support thread, household, and narrowly defined child-support scopes. Every item should include source/provenance, creator, timestamp, sensitivity class, status, review/expiry time where relevant, and a visible reason it was recalled.

Required behaviors:

- “Remember,” “update,” “forget,” and “show what you remember” are first-class parent actions.
- The system visibly confirms every durable save and offers undo.
- Parent-authored rules outrank inferred or proposed preferences.
- Sensitive facts and durable descriptions of a child are never silently saved.
- Memory off blocks new writes; disabling or deleting an item takes effect in the next turn.
- Preference records never write into the child evidence graph.

### 2. Add governed, private “family recipes”

Use the existing `family_private` skill seam, progressive disclosure, capability/skill version recording, and promotion framework. The model may propose a recipe after an explicit parent request—or suggest drafting one after repeated feedback—but may not activate, overwrite, or broaden it on its own.

Recommended lifecycle:

**draft → contract lint → public-fixture and synthetic-family tests → shadow comparison → parent/operator approval → active version → disable/retire**

Activation is durable authority and should therefore always require a human. A run should record the exact recipe version, and rollback should be one operation. Recipes should reference only reviewed typed tools; a recipe must never be able to introduce a shell, arbitrary network destination, or new connector permission.

### 3. Add structured compaction for long Slack threads

Compaction should preserve:

- the parent’s current intent and constraints;
- explicit preferences and their source IDs;
- decisions and why they were made;
- released output IDs and pointers, not copied artifact contents;
- errors, fixes, and parent corrections;
- pending action and the next step.

It must not infer child traits, upgrade hypotheses, or become evidence. On resume, the runtime should re-fetch code-owned records and output metadata rather than treating the compacted prose as truth. Keep the last few raw turns alongside the summary and make the summary inspectable in the local dashboard.

### 4. Introduce durable background runs and event wakes

Long visual generation, printable rendering, evaluation, and future local-model jobs should have their own persistent run entity, attempt history, status, workspace, policy snapshot, and completion event. A suspended parent turn can wake on completion and continue from code-owned state.

Borrow OpenWorker’s “catch up once” and “skip overlap” semantics. Do not borrow generic autonomous scheduling. A wake may resume work, but delivery still obeys Curiosity Engine’s outbox and parent-confirmation rules; it must not create unsolicited child-facing messages.

### 5. Extract a small harness kernel only where it reduces risk

Curiosity Engine’s large parent/service modules contain a reusable execution kernel and domain policy in close proximity. Extract interfaces for turn lifecycle, tool execution records, interrupt/resume, compaction, and background-run events, while leaving identity, child evidence, pedagogy, delivery, and family policy in domain modules.

Do not make OpenWorker a runtime dependency yet. It is a fast-moving open beta with a substantially broader trust surface. A future spike can compare interfaces, but migration would trade away the domain constraints that are Curiosity Engine’s advantage.

### 6. Turn feedback into governed experience replay

Curiosity Engine already has the right promotion stance. Extend it with shadow comparisons for candidate memory behavior, recipes, routing, and compaction. Use only public fixtures or explicitly de-identified, licensed cases. A successful candidate may become eligible for operator approval; it may never self-promote.

OpenWorker’s own reviewer reports show the value of layered benign, dangerous, and injection corpora, but they are internal project evaluations rather than independent safety proof. Curiosity Engine should borrow the method, not the conclusion.

### 7. Add an optional probabilistic-verifier research lane

The September 2026 review of **LLM-as-a-Verifier** supports a narrow addition to the harness plan, not a dependency
change. Its most useful contribution is a more informative way to compare already-safe candidate outputs: score A/B
pairs on one explicit criterion at a time, read the expected value of an ordered score-token distribution, repeat the
comparison with swapped positions, and preserve the per-repeat distribution instead of collapsing immediately to a
single judge label. In the paper’s Terminal-Bench ablations, greater score granularity, repeated evaluation, and a
three-criterion ensemble each improved pairwise selection; however, those results cover coding, robotics, and medical
agent trajectories—not K–2 pedagogy or family privacy.

Keep this lane **off by default, research-only, and downstream of every deterministic safety and privacy gate**. It may
rank public-fixture or explicitly de-identified baseline/candidate responses, expose disagreement and variance to a
reviewer, and help prioritize adversarial cases. It must not approve delivery, create learner evidence, tune a child’s
state, activate personalization or the research track, replace operator approval, or substitute for the two
independent converge reviews required for a release candidate.

The first pilot should implement a small Curiosity-owned adapter rather than importing the package:

1. Accept only explicit, bounded, public or approved de-identified evaluation inputs; never read `.env`, arbitrary
   filesystem paths, URLs, family images, child context, or production traces implicitly.
2. Run hard safety, privacy, evidence-validity, and authorization checks before probabilistic comparison; a candidate
   that fails a hard gate is ineligible rather than assigned an averageable score.
3. Compare only survivors using versioned criteria such as request fidelity, factual clarity, developmental fit,
   accessibility, evidence quality, and parent effort. Keep each criterion separate in the stored result.
4. Use at least one A/B position swap and measure order sensitivity, repeat variance, abstention/error rate, latency,
   tokens, cached tokens, and cost. Choose repeat count and any tournament budget from Curiosity-specific calibration,
   not the upstream defaults.
5. Treat a missing score, missing logprobs, malformed output, provider error, cache-lineage mismatch, or budget overrun
   as an invalid run. Upstream’s `0.5` fallback and default `on_error="tie"` are unsuitable for promotion evidence.
6. Bind the provider, model, endpoint class, prompt/criterion versions, input hashes, candidate order, repeats, seed,
   score distribution, error state, usage, adapter version, and release-manifest hash into an inspectable trace.
7. Validate against labeled Curiosity Engine public fixtures and an independent human/model review set. Report
   pairwise accuracy, calibration, false-safe and false-preference rates by category, subgroup parity, inter-reviewer
   agreement, and bootstrap uncertainty before expanding its role.

Progress scores over trajectory prefixes are also worth testing for **engineering-run observability**: they could flag
a stalled research or generation job for human inspection. They are not valid measures of a child’s learning,
curiosity, motivation, or mastery, and must never enter the child context graph.

The detailed evidence matrix, limitations, and future propagation map are preserved in
[`docs/report-source.md`](report-source.md). The current implementation deliberately adds no verifier package,
provider call, promotion signal, or production behavior.

## What not to copy

| OpenWorker pattern | Decision for Curiosity Engine | Reason |
|---|---|---|
| General shell/computer control | **Decline** | The family runtime’s typed-tool boundary is a core safety property. |
| Broad MCP/provider/connector surface | **Decline by default** | Each connector enlarges data exposure and durable authority. Add only named, reviewed family use cases. |
| Generic memory injected as agent context | **Adapt narrowly** | Parent preferences are useful; generic facts about children can become sticky, incorrect labels. |
| Model reviewer that auto-allows ask-level actions | **Shadow only** | Use it to measure policy quality, not to approve live family-data or external-write actions. |
| Agent-authored skills activated immediately | **Decline** | Drafting is fine; activation/overwrite requires tests, provenance, approval, versioning, and rollback. |
| Generic scheduled autonomy | **Decline** | Durable runs are useful; unsolicited work or delivery conflicts with current family-review gates. |
| Wholesale OpenWorker migration | **Decline now** | Mismatched trust surfaces, rapid beta churn, and loss of domain-specific evidence semantics outweigh reuse. |

The warning is not theoretical. OpenWorker’s repository includes 36 permission-gate corpus cases explicitly tagged as known gaps at the pinned commit, covering areas such as unsandboxed shell effects, deferred execution, broad grants, host-only egress rules, unknown tools, MCP relaxation, browser uploads, and high-impact connector effects. This is healthy engineering transparency, but it means OpenWorker should be treated as a rich design reference—not a safety-certified base for a child-centered system.

## 2. Child context graph v2

### Research synthesis

There is no settled standard for a child context graph. The strongest approach combines several mature ideas:

- **Evidence-centered design** separates the claims a system wants to make, the observations that would support those claims, and the tasks that create valid opportunities to observe them. This maps cleanly to a proficiency/learner model, evidence model, and task model.
- **xAPI and Caliper** offer useful event vocabulary patterns: actor, action, activity, context, result, and timestamp. Curiosity Engine should borrow the shape without adopting an enterprise Learning Record Store.
- **CASE** supplies stable, source-owned identifiers for competencies and relationships. It is a better interoperability seam than using free-text topic labels as curricular identity.
- **W3C PROV** reinforces that every inference should retain the entities, activities, agents, and versions that produced it.
- **Knowledge tracing** shows why observations need guess/slip, item difficulty, partial credit, hints, and temporal/domain structure. It also shows the risk of overconfidence: classic models can fit average performance while remaining less sensitive to individual differences.
- **Open and negotiated learner models** show the value of making a model inspectable and correctable. Human-centred learning-analytics research adds a stronger lesson: design with the affected users, keep explanations contextual, and preserve human control.
- **Child-centered AI guidance** adds privacy, fairness, participation, explainability, data minimization, limited retention, and high-privacy defaults. A learner model is still profiling even when its purpose is benevolent.

The conclusion is not “adopt a graph database” or “implement deep knowledge tracing.” It is: **represent the validity chain explicitly, make all projections recomputable, and expose them to the family in language that does not label the child.**

### What the current graph gets right—and what to change

| Current design | Assessment | Planned change |
|---|---|---|
| Immutable events plus observations/evidence | Strong foundation | Keep; add an explicit learning-opportunity envelope and retention class. |
| Conservative episodes and independence status | Unusually strong | Generalize beyond interest so mastery evidence also distinguishes retry, practice, and independent transfer. |
| Claims with support, contradiction, and downgrade | Strong epistemic shape | Add claim type/evidence-rule/version; derive projections instead of accepting model-set state. |
| Parent corrections are append-only | Strong | Recompute every affected projection and make the lineage visible in the dashboard. |
| Bounded context depths and relevance ranking | Good privacy boundary | Rank derived learner snapshots and task-relevant evidence; keep raw text out unless specifically needed. |
| Nodes combine topic identity and child state | Fragile | Separate global/versioned concepts from child-specific learner-state projections. |
| Free-text canonical keys | Adequate for MVP | Add stable concept IDs, aliases, source framework/version, and explicit concept relations. |
| One confidence scalar | Ambiguous | Split source reliability, evidence strength, and model uncertainty. |
| `evidence_count` increments on upsert/mention | Easy to misread | Rename it to mention count or compute counts from linked evidence; never use it as mastery or interest confidence. |
| Knowledge states include `unseen` | Overclaims the system’s view | Use `not_observed_in_harness`; the child may know or have encountered it elsewhere. |
| Model may emit `set_knowledge_state` | Too much authority | Remove from model-authored output. Code derives state from evidence rules. |
| No explicit task/difficulty/assistance model | Major validity gap | Version the learning task and record opportunity, scaffold, attempt, response mode, and rubric. |
| Interest and knowledge share generic graph machinery | Risk of construct leakage | Give knowledge, interest, preference, strategy, and transient affect different schemas and promotion rules. |

### Recommended data model

Retain SQLite and the event-sourced approach. Introduce a versioned relational graph rather than a new database technology.

| Entity | Required content | Notes |
|---|---|---|
| **concepts** | stable ID, label, domain, grade band, description, aliases, framework/source/version | Global or household-curated ontology; not child state. Optional CASE URIs can be added later. |
| **concept_relations** | source concept, relation, target concept, provenance, version | Start with `part_of`, `prerequisite_of`, `contrasts_with`, and `applies_in`; do not infer mastery across edges. |
| **learning_tasks** | task ID/version, target concepts, learning goal, strategy, difficulty features, modalities, rubric/evidence rule | This is the task model. Generated artifacts receive immutable task metadata. |
| **learning_opportunities** | event/episode/task IDs, actor, attempt, response mode, assistance level, result, scorer, timestamp | One observable opportunity; retries remain linked and non-independent. |
| **evidence_records** | observation, concept, stance, score/quality, source reliability, privacy/retention class, provenance | Raw evidence is append-only; text is a pointer where practical. |
| **learner_state_projections** | child, concept/construct, state, evidence strength, uncertainty, support level, episode/context diversity, last observed, rule/version | Recomputable materialized projection; never the source of truth. |
| **interest_projections** | topic, phase, child-initiation, recurrence, context diversity, recency, uncertainty, next review | Separate from mastery. Defaults to situational and expires without renewed evidence. |
| **preference/support records** | explicit parent statement, scope, source, status, sensitivity, review/expiry | Separate from child evidence and from inferred interest. |
| **corrections and lineage** | target record, prior state, action, actor, reason, timestamp | Append-only. Deleting source data invalidates/recomputes derivatives. |

### Learner-state semantics

Replace “mastery” as a single score with conservative, contextual states:

1. `not_observed_in_harness`
2. `exposure_recorded`
3. `practicing_with_support`
4. `demonstrated_in_context`
5. `independently_demonstrated`
6. `transfer_observed`

Every state must show the target concept, task context, assistance level, number of eligible episodes, context diversity, recency, and uncertainty. “Demonstrated” means demonstrated on the recorded tasks; it never means a global judgment about intelligence or school attainment.

Promotion rules should be construct-specific. For example, an independent concept demonstration may require two eligible opportunities across separate episodes, at least one without help, and no unresolved contradiction. Transfer requires a meaningfully different representation or application context. A parent report may be valuable evidence but should retain its source type rather than being treated as a system-scored task.

Do not add Bayesian or neural knowledge tracing until the engine has enough versioned tasks, assistance labels, outcome data, and held-out validation to calibrate the model. A simple, inspectable evidence rule is superior to a statistically sophisticated but unvalidated probability. Probabilistic estimates, when introduced, should accompany—not replace—the evidence ledger and should never directly authorize delivery or profile changes.

### Interest and curiosity semantics

Adapt the four-phase interest model as a projection heuristic:

- **triggered situational:** one question, surprise, or parent-reported engagement;
- **maintained situational:** engagement continues within or across a bounded experience;
- **emerging recurrent:** the child independently returns across eligible episodes or contexts and seeks greater depth;
- **sustained, parent-confirmed:** a long-running interest the family recognizes and wants used for personalization.

Do not automatically assign a fixed “well-developed interest” label. Frequency alone is not interest; parent prompts, retries, and the engine’s own suggestions can create a self-confirming loop. Record who initiated the topic, whether the child chose to continue, whether the context changed, and whether the child generated a new question. Keep inferred interest profiling off for unsolicited recommendations until the existing shadow/family-review gate is met.

### Privacy, fairness, and inspectability

Apply a purpose and retention class to every durable field. Keep raw child text only where it is necessary for the family’s requested service; otherwise retain structured evidence and a local source pointer. Derived projections must be deletable and recomputable when their sources are removed. Never export private learner records into public evals or model training by default.

The parent dashboard should distinguish **what happened**, **what the engine inferred**, and **what the family explicitly asked it to remember**. Each inference needs a short “why this appears” explanation, supporting and contradicting evidence, uncertainty, and controls to correct, exclude, expire, or delete it. Add an optional child-friendly co-review view—“things we explored” and “things you might try next”—without grades, deficits, or personality labels.

These are product/safety recommendations, not a jurisdiction-specific legal opinion. COPPA, the ICO Children’s Code, and other obligations still require review against the actual deployment, data flows, and users.

## 3. Early-elementary teaching and curiosity system

### Design doctrine

No single pedagogy should dominate. Developmentally appropriate practice for birth through age eight emphasizes joyful, strengths-based, play-based learning; attention to common developmental patterns, individual variation, and family/cultural context; challenge just beyond current performance; responsive scaffolding; reflection; and repeated opportunities across time. Technology should support active, hands-on learning rather than replace it.

For Curiosity Engine, translate that into a repeatable interaction rhythm:

**orient → open a meaningful gap → child predicts/chooses → active investigation or construction → graduated help → explanation/representation → specific feedback → new question or later revisit**

The rhythm is flexible. A direct factual question deserves a direct, bounded answer. Foundational knowledge sometimes benefits from explicit modeling and practice. Exploration and playful inquiry then reopen agency and reveal understanding. Preschool experiments showing that pedagogical demonstration can narrow spontaneous exploration are a caution against making every answer exhaustive, not an argument against teaching.

### Strategy repertoire

| Strategy | Engine behavior | Best use / caution |
|---|---|---|
| **Answer + wonder** | Give the child-sized answer first, then one surprising contrast or open question. | Default for direct factual questions. Do not hide the answer behind a lesson. |
| **Guided play** | Set a clear learning goal and a playful environment; give bounded choices; guide with questions, hints, modeling, or materials. | Strong fit for K–2 and home use. “Fun” alone is not guided play; preserve agency. |
| **Predict–test–explain** | Ask for a prediction, run a fair observation/comparison, record what happened, explain, and generate a next test. | Science, measurement, cause, and everyday phenomena. Record help and task conditions. |
| **Compare–classify–connect** | Contrast examples/nonexamples, sort by a visible rule, link concrete and abstract representations. | Vocabulary, concepts, math, science, and misconceptions. Avoid irrelevant surface similarity. |
| **Model–practice–fade** | Show one compact worked example or model, try together, then remove support through a hint ladder. | New procedures and foundational skills. Do not confuse assisted success with independent evidence. |
| **Self-explain / teach-back** | Ask the child to explain a mechanism, teach a toy/character, draw a causal path, or correct the agent’s incomplete account. | Constructive engagement and causal understanding. Use short prompts; evidence is promising but younger-child fit must be evaluated. |
| **Dialogic talk / shared reading** | Use open prompts, prediction, recall, expansion, turn-taking, text structure, and discussion of meaning. | Language, vocabulary, comprehension, nonfiction, and parent-child co-learning. Do not interrogate after every sentence. |
| **Playful retrieval and spaced revisit** | Revisit a prior idea after a delay through a game, drawing from memory, “what do you remember?”, or applying it in a new context, with feedback. | Retention and transfer. Low-stakes only; curiosity conversations should not feel like constant testing. |
| **Story, role, and mission** | Place a real concept inside a coherent role or problem with an authentic purpose and multiple paths. | Motivation and meaning. Story must serve the concept, not bury or distort it. |
| **Make, design, and iterate** | Build/draw/arrange something, observe consequences, revise, and explain a design choice. | ICAP-style constructive engagement and hands-on learning. Keep materials ordinary and parent effort low. |
| **Multiple representations and expression** | Offer speech, drawing, gesture/movement, manipulatives, diagrams, words, or symbols; let the child show thinking in more than one way. | UDL/accessibility and transfer. Do not treat response mode as ability. |
| **Plan–check–reflect** | Ask a brief before/after question: “What is your plan?”, “What changed?”, “What would you try next?” | Metacognition and strategy awareness. Never infer an executive-function diagnosis from a single interaction. |

The research base is strongest for the underlying components rather than for this exact product package. IES/WWC guidance supports spacing, worked examples paired with problem solving, graphic-plus-verbal and concrete-plus-abstract representations, retrieval with feedback, deep explanatory questions, early math progressions and mathematical talk, shared reading, comprehension discussion, and the writing process. The guided-play evidence is positive but heterogeneous. ICAP is a useful engagement framework, not a guarantee that every “interactive” activity is well designed. Teachable-agent results support selective pilots, not a universal default.

### Curiosity mechanics

Curiosity should be cultivated, not optimized as engagement.

- **Information gap:** identify what the child already notices, then expose one answerable uncertainty.
- **Goldilocks challenge:** offer something neither obvious nor incomprehensible, but treat this as a calibration heuristic rather than a measured trait.
- **Question agency:** let the child choose among two or three genuine paths and generate the next question.
- **Surprise with integrity:** use anomalies and counterexamples only when factually accurate.
- **Trustworthy answers:** help children distinguish observation, explanation, uncertainty, and source; “I’m not sure” is a useful model behavior.
- **Interest development:** match support to situational versus recurring interest instead of decorating everything with a presumed favorite topic.
- **Healthy stopping:** celebrate closure, invite off-screen action, and avoid streaks, infinite continuation, urgency, or notifications designed to maximize time-on-task.

### Domain overlays

The pedagogy kernel should load a small domain overlay when relevant:

- **Early science/inquiry:** observation vs. inference, prediction, fair comparison, repeat trials, evidence, causal explanation, and a next test.
- **Early mathematics:** developmental progression, concrete–visual–symbolic connections, comparison, patterns, number/operation talk, and math in the child’s world.
- **Early literacy and reading:** oral language, vocabulary in context, letters/sounds where appropriate, shared reading, narrative/informational structure, retelling, inference, and focused discussion.
- **Writing and creative expression:** authentic purpose, child choice, model–practice–reflect, planning/drafting/revision, and multiple modes before transcription demands dominate.
- **World knowledge and source sense:** build coherent background knowledge, compare perspectives, separate fact from claim, and model how a source earns trust.
- **Social-emotional or sensitive questions:** parent-mediated, concrete, validating, non-diagnostic, culturally humble, and routed through stricter safety/review rules.

### Typed learning-episode plan

Do not add all of this to the base prompt. Add a compact, typed plan produced before child-facing content:

| Field | Purpose |
|---|---|
| `question_intent` | Direct answer, explanation, exploration, practice, artifact, or sensitive support. |
| `learning_goal` and `concept_ids` | What the child may understand or do; linked to stable concepts. |
| `strategy` and `domain_overlay` | One primary strategy, at most one supporting strategy, and relevant domain rules. |
| `interest_support` | Situational trigger, maintenance, recurring-depth, or neutral; includes provenance. |
| `challenge_band` | Familiar, stretch-with-help, or independent transfer; never derived from grade alone. |
| `representations` | Concrete, visual, verbal, movement, symbolic, and child response choices. |
| `scaffold_policy` | Hint ladder, worked example, joint attempt, fade condition, and stop/frustration rule. |
| `evidence_opportunities` | What observable behavior would support which claim, under what assistance and rubric. |
| `closure` and `revisit` | Reflection/new question and whether a later low-stakes revisit is appropriate. |
| `safety_and_parent_effort` | Materials, supervision, sensitivity, time, cleanup, and delivery constraints. |

The planner proposes pedagogy; deterministic code validates allowed strategies, evidence rules, risk, and context use. The generation model then creates the answer/activity against the approved plan. Critics evaluate factual accuracy, pedagogy-plan fidelity, cognitive/reading load, accessibility, evidence validity, context use, and parent effort.

## 4. Propagation through the current engine

### Code and product surfaces

| Surface | Planned update | Verification |
|---|---|---|
| `contracts.py` | Add `LearningEpisodePlan`, `LearningOpportunity`, `EvidenceCandidate`, assistance/rubric enums, concept references, and projection types. Remove `SET_KNOWLEDGE_STATE` from model-authored mutations. | Contract tests reject direct state changes, unknown strategies, unversioned tasks, and evidence without provenance. |
| `db.py` migrations | Add concepts/relations, learning tasks/opportunities, evidence lineage, learner-state and interest projections, retention classes, and explicit parent memory. | Migration/rollback fixtures; delete/recompute tests; no cross-child joins. |
| `runtime.py` / `repository.py` | Persist the task plan before generation; accept only observation/evidence candidates; validate and record opportunity conditions; trigger deterministic projection. | Crash/retry idempotency; model output cannot promote state or durable interest. |
| `graph.py` | Split ontology from child projections; implement recompute-by-rule/version; deprecate mention-count semantics as confidence. | Property tests for correction, contradiction, deletion, support level, and episode independence. |
| `episodes.py` / `claims.py` | Generalize independence and evidence rules across mastery, interest, and transfer; retain construct-specific thresholds. | Retries/helped attempts never count as independent transfer; correction downgrades dependent state. |
| `context_builder.py` | Return a budgeted learner snapshot, relevant concept neighborhood, evidence rationale, preferences, and pedagogy cards; minimize raw text. | Relevance tests, token budgets, stale/expired exclusion, privacy snapshots, adversarial leakage cases. |
| `reasoning.py` / prompts | Two-stage plan→generate flow; strategy-specific critics; factual answer still precedes extension. | Golden cases across direct fact, inquiry, sensitive, bored, frustrated, and “just tell me” requests. |
| `capabilities/catalog.json` / skills | Add progressive pedagogy-router, guided-play/inquiry, dialogic/shared-reading, teach-back, retrieval/revisit, scaffold/fade, UDL/representation, and domain overlay skills. Update existing worksheet/activity/challenge/visual skills to consume the plan. | Skill contract lint, collision tests, public fixtures, version recording, disable behavior. |
| `feedback.py` / parent tools | Add explicit outcome signals: used/not used, chose to continue, prediction, explanation, help level, too hard/easy, new question, and parent correction. | Feedback never mutates state directly; all claims retain source type and episode. |
| `sessions.py` / `parent_agent.py` | Add structured compaction and the scoped parent memory ledger; expose provenance/undo; keep memory separate from child evidence. | Long-thread replay, immediate disable/delete, no memory-to-evidence path. |
| `learning_artifacts.py` / `visuals.py` | Carry task/concept/strategy/evidence metadata; add response-mode choice, retrieval/revisit cards, teach-back prompts, shared-reading guides, accessible alternatives, and kid-friendly learning diagrams. Keep instructional text/geometry code-rendered; use generated art only to make diagrams and worksheets engaging. | Child-usable layout QA plus pedagogy, usefulness, accessibility, and generated-decoration privacy rubrics. |
| `tooling.py` / policies | Add durable-authority and learner-profile risk classes; require approval for memory/recipe/schedule changes; retain typed-tool/no-shell boundary. | Permission matrices and hard-floor tests. |
| Lab, golden fixtures, promotion | Add graph validity, pedagogy selection, challenge calibration, accessibility, interest, retention/deletion, fairness, and self-confirmation suites. | No promotion without deterministic suites, independent judge, no safety/factual/golden regression, and operator approval. |
| Local dashboard / Slack | Three views: events, inferences, explicit memories; why-used explanation; correct/exclude/expire/delete; optional child-friendly reflection card. | Parent usability study; child/parent co-design sessions; accessibility review. |
| Documentation | Add Learner Model Constitution, Pedagogy Strategy Registry, evidence-rule specs, privacy/retention map, and migration/runbooks. | Architecture and test docs must change in the same release as behavior. |

### Proposed skill set

Keep progressive disclosure. The base `parent_chat` capability sees only compact strategy cards and loads the chosen instructions.

**New core skills:** `pedagogy-router-k2`, `answer-and-wonder-k2`, `guided-inquiry-k2`, `compare-connect-k2`, `model-practice-fade-k2`, `dialogic-reading-k2`, `teach-back-k2`, `retrieval-revisit-k2`, `metacognitive-closure-k2`, and `udl-representation-k2`.

**New domain overlays:** `early-science-k2`, `early-math-k2`, `early-literacy-k2`, `early-writing-k2`, and `source-sense-k2`.

**Update existing skills:** make `hands-on-activity-k2`, `mystery-challenge-k2`, `worksheet-k2`, and `visual-selection-k2` consume the typed learning plan, preserve representation choice, declare evidence opportunities, and distinguish supported from independent performance.

Do not load more than one primary pedagogy skill plus one domain overlay and the visual/accessibility rules for a normal turn. Prompt growth would otherwise erase the benefit of progressive skills.

### Immediate hybrid-challenge rollout (implemented 2026-09-11)

The first rollout deliberately improves three prompts already exercised by the family instead of adding a generic
visual framework with no product evidence: cup-shaped ice, helium/open cups, and paper-airplane wing tips.

- `ChallengeVisualBrief` now fixes the illustration role, teaching diagram, answer timing, art slot, and response
  layout before rendering. Generated images cannot supply labels, measurements, causal arrows, trial structure, or
  answers.
- The configured OpenAI `image_generation` route supplies one high-quality landscape illustration from a versioned,
  family-free prompt. Assets and reviews are private and cached by content-relevant hashes. Provider failure takes the
  complete deterministic path; it does not fail the Slack action.
- Ice pages keep both prediction areas equal and hide the result until reveal. Air pages show the required
  upside-down/straight-down/tilt sequence. Airplane pages show that only the outer tips bend and make the child choose
  a distance or direction target before two trials per setting.
- Final-page review sees the composed page, not just a structured spec. It scores family usefulness and engagement
  first, then factuality, learning value, K–2 usability, and visual integrity. It can remove misleading generated art
  without rerunning generation.
- Slack acknowledges the action immediately with `Drawing and building the challenge…`, then sends the validated PDF.
  The old vague `Making a child-ready challenge…` message is retired.

Next expansion is evidence-driven: add a visual brief only after a parent-used prompt exposes a reusable diagram or
workspace need. The private comparison lab may generate multiple art candidates; runtime remains one candidate plus
fallback. Parent Helpful/Not for us/revision outcomes, successful delivery, wall time, and rendered-page scores decide
which templates generalize. Completed safety machinery is not a reason to delay these usefulness iterations.

### Evaluation matrix

Add deterministic and judged cases for:

- **Validity:** output text and system suggestions cannot become evidence of their own success.
- **Independence:** retries, reformulations, and same-session enthusiasm cannot inflate durable interest or mastery.
- **Assistance:** a correct answer after a strong hint is recorded differently from independent performance.
- **Difficulty:** success on a trivial task does not promote state; failure on an overly difficult task does not demote it.
- **Transfer:** a genuinely different application is required before `transfer_observed`.
- **Uncertainty:** sparse or conflicting evidence remains visibly uncertain.
- **Correction/deletion:** excluding a source recomputes every dependent projection and explanation.
- **Pedagogy fit:** direct questions receive direct answers; inquiry requests create observable action; sensitive questions avoid playful trivialization.
- **Challenge and frustration:** hint ladders adapt without shame; repeated failure causes simplification or a stop option.
- **Accessibility:** reading load, alt text, response modes, color/visual dependence, fine-motor demands, multilingual respect, and material substitutions.
- **Curiosity integrity:** no fake mystery, manipulative cliffhanger, streak, urgency, or unsupported surprise.
- **Fairness:** equivalent evidence yields equivalent state across names, dialects, response modes, and family contexts.
- **Privacy:** no private text in public fixtures, external prompts beyond the selected context, logs, generated imagery, or telemetry.

Use shadow-mode family trials before any state affects recommendations. Evaluate both pedagogical quality and calibration: how often the engine’s prediction of “familiar,” “stretch,” or “independent” matches parent-observed outcomes. A model that produces delightful activities but systematically mislabels child state should not promote.

### Integrated phased adoption (historical implementation record)

The table records what was built and why. Under the trusted-parent reset, its completed exit criteria are evidence to
preserve, not permanent beta requirements. The retirement register above controls what remains mandatory.

| Phase | Deliverable | Exit criteria |
|---|---|---|
| **0 · Constitution and hardening** | Learner Model Constitution; remove direct model-set state; define durable-authority, privacy, and delivery invariants. | No model path can silently create child state, traits, tools, recipes, schedules, or external writes. Existing tests remain green. |
| **1 · Evidence spine** | Versioned concepts, tasks, opportunities, assistance/rubrics, evidence lineage, and recomputable projections. | Corrections/deletions deterministically recompute state; helped/retry/trivial evidence cannot over-promote. |
| **2 · Pedagogy kernel** | Typed plan→generate runtime, strategy registry, domain overlays, critics, and upgraded artifacts. | Golden matrix passes for strategy fit, factual quality, reading load, accessibility, parent effort, and evidence validity. |
| **3 · Open family model** | Implemented software candidate: parent inspector/correction/retention controls and optional child-friendly co-review. Consented family co-design has not yet occurred. | **Software exit:** synthetic authority, correction, deletion, accessibility, and comprehension surfaces pass. **Production exit:** parent/child sessions show families understand and can control inferences, and observed usability, accessibility, and emotional-safety issues are resolved. |
| **4 · Adaptive harness** | Structured thread compaction, scoped parent memory, draft-only family recipes, and durable background runs. | Memory stays separate from evidence; recipes cannot self-activate; resume/outbox/overlap tests pass. |
| **5 · Evaluated personalization** | Preserved compatibility candidate: shadow strategy selection, bounded challenge/interest calibration, typed outcome capture, replay, and consented activation. | Tier-3 consent/provenance and observable behavior benefit remain; duplicated promotion authority and adversarial-local-state defenses are retirement candidates. |
| **6 · Research track** | Preserved, disabled compatibility candidate: fixed-Beta uncertainty plus inspectable, non-deliverable context candidates. | It stays off in beta. Any later trial needs parent opt-in and real calibration; its multi-party promotion, high-water, and multi-root mechanisms are not presumed permanent. |
| **Post-6 verifier pilot** | Disabled Curiosity-owned pairwise verifier adapter for public/de-identified Lab candidates only. | Hard gates run first; failures abstain/fail closed; criteria and per-repeat scores remain inspectable; order-bias, calibration, category false-preference, privacy, cost, and independence gates pass. It remains advisory until a separately reviewed policy explicitly changes its authority. |

Phases 0–2 are the recommended near-term program. Phase 3 should begin in design alongside them because family understanding is part of model validity, not post-launch polish. OpenWorker-inspired continuity belongs after the evidence and pedagogy contracts are stable, so the system does not become better at persisting the wrong abstraction.

## Acceptance checklist

Before any adaptive feature ships:

- A parent can inspect, correct, disable, export, and delete the durable state.
- Every event, inference, remembered item, task, evidence rule, and recipe version has provenance.
- The model can propose observations but cannot directly set learner state, interest phase, permissions, or durable procedure.
- Learner state is recomputable from source evidence after correction, expiry, or deletion.
- Assistance, attempt, task difficulty, response mode, and episode independence are represented before evidence is promoted.
- No inference about a child becomes a durable rule without explicit parent confirmation.
- No grade, score, trait, diagnosis, or global “mastery” label is inferred from sparse household interaction.
- Durable authority changes always require human approval.
- Side effects remain code-owned, typed, bounded, and auditable.
- A compacted summary cannot create evidence, permissions, or delivery.
- A recipe cannot add tools, hosts, connectors, or scopes.
- Background wakes cannot bypass outbox or parent-review gates.
- Every generated learning experience declares a goal, strategy, challenge assumption, representation, scaffold policy, and valid evidence opportunity.
- Direct questions still receive direct answers; child agency is not used to evade explanation.
- Curiosity features optimize learning agency and healthy closure—not time-on-task, streaks, or dependency.
- Accessibility and family/cultural context are planned at generation time, not added as cleanup.
- Public evals contain no private family text; real failures are de-identified only with privacy/licensing clearance.
- Promotion is measured, regression-gated, reversible, and operator-approved.
- Probabilistic or model-based verifier output is advisory, never averaged across a hard safety failure, and cannot
  replace deterministic gates, independent converge review, or human activation authority.

## Evidence of adoption—and its limits

As checked on 2 September 2026 PDT, GitHub’s repository API reported 17,255 stars, 2,413 forks, and 123 subscribers for OpenWorker, which was created on 20 July 2026. That is exceptional early attention and a credible signal that the design resonates with builders. It does not establish how many households or teams run it successfully, retention, task success, or safety outcomes. No independent outcome study was found in this scoped review.

OpenWorker’s rapid development and open-beta status further argue for importing concepts and tests before importing code. If substantial source code is later copied, its MIT license and attribution requirements should be preserved.

## Research method and limitations

This review prioritized primary evidence. OpenWorker was inspected at commit `fb1bfc627201f1e159a9380ada25e954174271a5` (30 August 2026). Curiosity Engine’s reproducible citations are pinned to public commit `b8b0b544b0aa74d99cc7d25045024ba23f201173` (27 August 2026). The live DGX checkout was also read on 2 September 2026; it was four commits ahead of that public upstream and had uncommitted work, so claims in this brief avoid relying on non-public diffs. No private family database, child content, or question history was inspected.

The learner-model and pedagogy synthesis assumes Curiosity Engine’s current K–2 scope, roughly ages five to eight. Some cited evidence covers preschool, all K–12 grades, or older primary students; transfer into K–2 is identified as a design hypothesis where appropriate and must be evaluated. The analysis covers architecture and repository evidence, not a full penetration test, third-party dependency audit, legal opinion, longitudinal user study, or controlled benchmark of end-user outcomes.

## Selected primary sources

### OpenWorker

- [Repository and README](https://github.com/andrewyng/openworker/tree/fb1bfc627201f1e159a9380ada25e954174271a5)
- [Agent assembly and memory/skill integration](https://github.com/andrewyng/openworker/blob/fb1bfc627201f1e159a9380ada25e954174271a5/coworker/agent.py)
- [Memory tools](https://github.com/andrewyng/openworker/blob/fb1bfc627201f1e159a9380ada25e954174271a5/coworker/memory/tools.py)
- [Skill store and user-approved authoring](https://github.com/andrewyng/openworker/blob/fb1bfc627201f1e159a9380ada25e954174271a5/coworker/skills/store.py)
- [Permissions and hard floors](https://github.com/andrewyng/openworker/blob/fb1bfc627201f1e159a9380ada25e954174271a5/coworker/permissions.py)
- [Structured compaction](https://github.com/andrewyng/openworker/blob/fb1bfc627201f1e159a9380ada25e954174271a5/coworker/compaction.py)
- [Durable self-wake](https://github.com/andrewyng/openworker/blob/fb1bfc627201f1e159a9380ada25e954174271a5/coworker/selfwake.py)
- [Permission-gate corpus](https://github.com/andrewyng/openworker/blob/fb1bfc627201f1e159a9380ada25e954174271a5/tests/corpora/permission_gate.jsonl)
- [Current repository metadata](https://api.github.com/repos/andrewyng/openworker)

### Curiosity Engine

- [Repository and README](https://github.com/WomenDefiningAI/curiosity-engine/tree/b8b0b544b0aa74d99cc7d25045024ba23f201173)
- [Architecture](https://github.com/WomenDefiningAI/curiosity-engine/blob/b8b0b544b0aa74d99cc7d25045024ba23f201173/docs/ARCHITECTURE.md)
- [Context graph and evidence rules](https://github.com/WomenDefiningAI/curiosity-engine/blob/b8b0b544b0aa74d99cc7d25045024ba23f201173/docs/CONTEXT_GRAPH.md)
- [Autonomy contract](https://github.com/WomenDefiningAI/curiosity-engine/blob/b8b0b544b0aa74d99cc7d25045024ba23f201173/docs/AUTONOMY.md)
- [Capability registry and private skill loading](https://github.com/WomenDefiningAI/curiosity-engine/blob/b8b0b544b0aa74d99cc7d25045024ba23f201173/src/curiosity_engine/capabilities.py)
- [Bounded parent-agent loop](https://github.com/WomenDefiningAI/curiosity-engine/blob/b8b0b544b0aa74d99cc7d25045024ba23f201173/src/curiosity_engine/parent_agent.py)
- [Durable thread preferences and run records](https://github.com/WomenDefiningAI/curiosity-engine/blob/b8b0b544b0aa74d99cc7d25045024ba23f201173/src/curiosity_engine/sessions.py)
- [Testing and promotion criteria](https://github.com/WomenDefiningAI/curiosity-engine/blob/b8b0b544b0aa74d99cc7d25045024ba23f201173/docs/TESTING.md)
- [Promotion policy](https://github.com/WomenDefiningAI/curiosity-engine/blob/b8b0b544b0aa74d99cc7d25045024ba23f201173/configs/promotion-policy.json)

### LLM-as-a-Verifier

- [Paper, arXiv v2](https://arxiv.org/abs/2607.05391v2)
- [Repository and README at inspected commit](https://github.com/llm-as-a-verifier/llm-as-a-verifier/tree/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770)
- [Fine-grained score extraction, clients, cache, and error behavior](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py)
- [Probabilistic Pivot Tournament](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/pivot_tournament.py)
- [Progress scoring](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/progress.py)
- [Version, beta status, and dependencies](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/pyproject.toml)

### Child context, learner modeling, and governance

- [ETS: Assessment as Evidential Reasoning](https://www.ets.org/Media/Research/pdf/gorin_assessment_evidential_reasoning.pdf)
- [1EdTech: Caliper Analytics](https://www.1edtech.org/standards/caliper)
- [1EdTech: CASE competency standard](https://standards.1edtech.org/case/)
- [ADL: xAPI/cmi5 best-practices guide](https://www.adlnet.gov/assets/uploads/cmi5%20Best%20Practices%20Guide%20-%20From%20Conception%20to%20Conformance.pdf)
- [W3C: PROV overview](https://www.w3.org/TR/prov-overview/)
- [Corbett et al.: Individual Differences and Predictive Validity in Student Modeling](https://escholarship.org/uc/item/5bs755pr)
- [Pardos & Heffernan: item difficulty in knowledge tracing](https://doi.org/10.1007/978-3-642-22362-4_21)
- [Ait Chabane: domain-informed, uncertainty-aware knowledge tracing](https://educationaldatamining.org/edm2026/proceedings/2026.EDM.full-papers.153/index.html)
- [Kerly & Bull: children’s interactions with inspectable and negotiated learner models](https://doi.org/10.1007/978-3-540-69132-7_18)
- [Human-centred learning analytics and AI in education: systematic review](https://doi.org/10.1016/j.caeai.2024.100215)
- [Co-designing privacy, transparency, and trust in K–12 learning analytics](https://doi.org/10.1145/3448139.3448145)
- [UNICEF: Guidance on AI and Children 3.0](https://www.unicef.org/innocenti/reports/policy-guidance-ai-children)
- [FTC: 2025 COPPA amendments](https://www.ftc.gov/news-events/news/press-releases/2025/01/ftc-finalizes-changes-childrens-privacy-rule-limiting-companies-ability-monetize-kids-data)
- [ICO: profiling for automated decision-making](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/childrens-information/childrens-code-guidance-and-resources/how-to-use-our-guidance-for-standard-one-best-interests-of-the-child/best-interests-framework/profiling-for-automated-decision-making/)

### Early-elementary learning and curiosity

- [NAEYC: Developmentally Appropriate Practice principles](https://www.naeyc.org/resources/position-statements/dap/principles)
- [Guided play systematic review and meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC9545698/)
- [CAST: Universal Design for Learning Guidelines 3.0](https://udlguidelines.cast.org/)
- [IES/WWC: Organizing Instruction and Study to Improve Student Learning](https://ies.ed.gov/ncee/wwc/PracticeGuide/1)
- [IES/WWC: Preparing Young Children for School](https://ies.ed.gov/ncee/wwc/practiceguide/30)
- [IES/WWC: Teaching Math to Young Children](https://ies.ed.gov/ncee/WWC/PracticeGuide/18/Published)
- [IES/WWC: Improving Reading Comprehension in K–3](https://ies.ed.gov/ncee/wwc/PracticeGuide/14)
- [IES/WWC: Teaching Elementary Students to Be Effective Writers](https://ies.ed.gov/ncee/wwc/PracticeGuide/17/Published)
- [Chi & Wylie: ICAP framework](https://doi.org/10.1080/00461520.2014.965823)
- [Hidi & Renninger: four-phase model of interest development](https://doi.org/10.1207/s15326985ep4102_4)
- [Jirout & Klahr: children’s scientific curiosity](https://www.cmu.edu/dietrich/psychology/pdf/klahr/PDFs/Scientific%20curiosity.pdf)
- [Bonawitz et al.: instruction and spontaneous exploration](https://pmc.ncbi.nlm.nih.gov/articles/PMC3369499/)
- [Kidd, Piantadosi & Aslin: Goldilocks effect](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0036399)
- [Retrieval-based learning in elementary school children](https://pmc.ncbi.nlm.nih.gov/articles/PMC4786565/)
- [Selective effects of explanation in early childhood](https://doi.org/10.1016/j.jecp.2014.03.001)
- [Teachable-agent game for primary mathematics](https://eric.ed.gov/?id=EJ1036909)
- [UNESCO: Guidance for Generative AI in Education and Research](https://www.unesco.org/en/articles/guidance-generative-ai-education-and-research?hub=66973)
