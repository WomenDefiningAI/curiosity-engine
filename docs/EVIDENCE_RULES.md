# Versioned evidence rules

Phase 1 implements the Learner Model Constitution as a separate, relational evidence spine. It does not promote or
backfill the legacy `evidence`, `observations`, `nodes`, `edges`, or `claims` tables. Those tables remain available to
the local inspector only.

## Validity chain

```text
immutable concept/task/rubric
              │
authoritative event ──► code-resolved episode ──► immutable learning plan
              │                                             │
              └──────────────────────────────► observed learning opportunity
                               │
                   deterministic evidence compiler
                               │
                 evidence records + normalized lineage
                               │
                  deterministic shadow projections
```

A learning opportunity is the only entry to the compiler, and it must belong to a previously registered learning plan.
The plan fixes the child, task version, source episode, context, representation, retry group, and any transfer relation.
Code verifies all child/event/episode ownership again when the observation arrives and derives occurrence time from the
authoritative event. Callers cannot supply task identity, transfer status, triviality, context, representation,
occurrence time, projection state, confidence, evidence IDs, rule versions, or provenance.

Model `observation_candidates` are stored in `evidence_annotations` with `authority=audit_only`. Generated output,
model annotations, output feedback, and system-authored activity completion may be retained for audit, but they are
always ineligible to support a learner or interest projection. The old `evidence` table is a raw source ledger, not
learning evidence.

## Physical and logical constructs

- `concepts`, aliases, relations, tasks, and rubrics define versioned educational objects. They contain no child state.
- `learning_episode_plans` are immutable, reviewed task placements. A transfer plan must point to a completed plan for
  the same child, in a distinct episode, sharing a target concept while varying context or representation.
- `learning_opportunities` describe what was actually observable: actor, initiation, attempt, response mode,
  assistance, per-criterion results, scorer, plan, source, and retention class. Their task, episode, context,
  representation, transfer status, and triviality are derived from code-owned lineage.
- `evidence_records` are deterministic compiler outputs. They separately represent exposure, knowledge performance,
  and interest, retaining the exact rubric and scored criterion lineage for each target concept.
- `learner_state_projections` and `interest_projections` are disposable materialized views.
- `derived_claims_v2` adds a normalized claim/evidence seam. It does not use JSON evidence lists.
- `preference_support_records` is a separate foundation for explicit family input; it is not evidence of knowledge or
  interest.

Speech, writing, drawing, gesture/manipulatives, selection, demonstration, symbols, and mixed responses are parallel
modes—not an ability hierarchy.

## Knowledge projection

The deterministic rule is `ce.knowledge.evidence@1.0.0`.

1. `not_observed_in_harness`: no eligible observation. This never claims that a child has not learned elsewhere.
2. `exposure_recorded`: an eligible direct observation or parent report shows an actual opportunity.
3. `practicing_with_support`: an eligible substantive response is correct with help or partially correct.
4. `demonstrated_in_context`: at least one substantive, correct, direct child response with no assistance.
5. `independently_demonstrated`: correct evidence spans at least two eligible episodes and includes an unassisted
   success.
6. `transfer_observed`: an unassisted success from a validated transfer plan follows another unassisted success in a
   distinct episode and deliberately varies context or representation.

A performance observation is substantive only when it is a first attempt in an eligible episode, comes from direct
child observation, uses a reviewed or parent-selected task difficulty, is not trivial, and has a known assistance
level. Retried, same-episode, diagnostic, system, excluded, uncertain, trivial, generated, model-authored, and
output-feedback records cannot promote performance. Helped work can show practice but cannot establish an independent
demonstration on its own. An incorrect response to an uncalibrated `advanced` task remains context-only rather than
demoting the child. Other incorrect substantive responses remain linked as contradiction; they do not create a global
deficit label. A task-wide result can never promote every target: required, typed rubric criteria are scored separately
for each concept, and missing criterion results reject the observation.

Response mode never changes promotion thresholds. Difficulty and help describe the opportunity, not the child.

## Interest projection

The deterministic rule is `ce.interest.four-phase@1.0.0`.

- `triggered_situational`: one eligible, child-initiated signal.
- `maintained_situational`: the child chose to continue or produced a new question.
- `emerging_recurrent`: eligible child-attributed signals span at least two episodes and two contexts.
- `sustained_parent_confirmed`: an explicit parent report is retained as `parent_confirmed`.

Retries, same-episode message volume, engine suggestions, generated activities, and output feedback are excluded.
Situational interest evidence expires after 90 days without renewal. `parent_confirmed` interest has no automatic
expiry; transient evidence expires after 30 days. `learner_snapshot(..., as_of=...)` applies these rules lazily and
`expire_learning_records()` performs physical retention cleanup. Interest remains separate from knowledge and stays
shadow-only for recommendations.

## Corrections, deletion, and recomputation

Evidence corrections are append-only and survive deletion of their source evidence without retaining response text or
the raw opportunity identifier. `exclude`, `restore`, and bounded field `correct` operations rebuild every
projection and claim for the child inside the same transaction. Episode corrections also rebind Phase 1 opportunity
and evidence episode lineage before rebuilding.

Deleting a Phase 1 learning source physically removes its opportunity, compiled evidence, and normalized links via
foreign-key cascades. A tombstone retains only the child, deletion category, actor category, time, and a one-way hash of
the removed opportunity ID—never response text or the raw identifier. Deletion also cancels the bound plan, preventing
the same attempt from being recreated under an alternate opportunity ID. The same transaction performs a full rebuild.
Phase 3 implements raw event deletion and the family-facing deletion/export UI through the same rebuild seam. It adds
one-way deleted-event replay tombstones, child-scoped inspection, collection pause, and append-only parent retention
deadlines without changing the evidence rules or enabling recommendation use. See `OPEN_FAMILY_MODEL.md`.

Repeated recomputation over the same effective evidence produces the same input/output digest and does not change a
projection's `computed_at`. SQLite composite foreign keys enforce child ownership in addition to application checks.

## Shadow release boundary

`learner_snapshot()` returns bounded labels, categorical states, counts, uncertainty, rule versions, rationales, and
evidence IDs; it never returns raw evidence text. `context_builder.py` exposes that snapshot for local diagnostics, but
one shared model-input boundary physically removes both the snapshot and its summary before runtime and Director model
calls, de-duplicates the event, and enforces a hard 48,000-character budget across event plus context. The snapshot has
`may_affect_recommendations=false` until the evaluated-personalization release gate is completed.

The thresholds above are conservative policy choices, not validated developmental measurements. Green local tests do
not establish fairness, longitudinal validity, legal compliance, or real-family usefulness.
