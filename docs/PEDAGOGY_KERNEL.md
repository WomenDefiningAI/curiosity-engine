# Pedagogy kernel

Phase 2 turns teaching choices into an inspectable, versioned contract before any child-facing content is generated.
The kernel is deliberately narrower than learner-state adaptation: it uses the current request, bounded authorized
context, and conservative grade defaults. Phase 1 shadow projections remain physically absent from every planner,
generator, and critic payload.

## Authority flow

```text
bounded event/context
        │
        ▼
model semantic proposal (no child/event/plan/task/rubric/evidence authority)
        │
        ▼
code compiler ──► reviewed strategy/overlay/concept/evidence registry
        │
        ▼
immutable approved plan, persisted before generation
        │
        ├──► safety requirement: reviewed code-owned response, no model call
        ├──► ordinary route: generator with selected instructions only
        ├──► ordinary route: mandatory code-owned critic-role floor
        └──► response/action/artifact/visual/feedback lineage
```

Recognized urgent self-harm or explicit electrical, ignition, chemical, or sharp-tool no-go events take a shorter code-owned branch before this ordinary
proposal flow: deterministic classification builds the safe fallback plan when a child is bound, then returns the
reviewed response. An unbound event receives the same minimum response without creating child data.

`PedagogyPlanProposal` contains semantic choices only. `ApprovedPedagogyPlan` binds those choices to the child, event,
workflow, registry and instruction digests, task template, reviewed concepts and rubric, validation versions, and a
canonical plan hash. A crash after approval reuses the exact plan. The 2.1 contract can read a stored 2.0 plan, but if
the authoritative event now requires a safety response, code recompiles and separately persists a current 2.1 plan
before generation; validator version participates in new plan identity. Concurrent identical compiles converge on the
winner's stored audit timestamp; every authority-bearing field must still match. A conflicting replay fails, and a
plan-bound event's identity or semantic source content cannot be mutated after approval.

The schema-v19 `pedagogy_plans` table is separate from Phase 1 `learning_episode_plans`; schema v20 repairs its ownership
cascades for safe plan, run, event, and child deletion. The latter table controls attributable learning observations. A
pedagogy plan may describe a future evidence opportunity, but planning, generation, critic approval, delivery, artifact
creation, printing, completion, and ordinary output feedback create no Phase 1 opportunity or evidence record. Approved
concept, task, rubric, and evidence-template references are materialized as reviewed Phase 1 definitions so no plan can
carry a dangling or model-invented learning reference.

## Reviewed repertoire

The versioned registry is `configs/pedagogy-registry.json`. It validates unique enabled IDs, versions, compatibility,
instruction paths, and a registry digest. At runtime it discloses compact cards to the planner and loads only:

- one primary strategy;
- at most one compatible supporting strategy;
- one domain overlay;
- the accessibility and healthy-closure foundations.

Primary strategies cover answer-and-wonder, guided predict-test-explain inquiry, compare/classify/connect,
model-practice-fade, dialogic reading, teach-back, retrieval/revisit, and make/design/iterate. Domain overlays cover
general learning, early science, math, literacy, writing, source sense, and sensitive support. The capability skill files
are the reviewed procedural instructions; their hashes are captured in every approved plan.

## Conservative Phase 2 defaults

- `interest_support` is always `neutral`.
- Challenge is `grade_baseline_uncalibrated`; no mastery or preference is inferred.
- Direct and sensitive requests use explicit non-collection and prohibit unsolicited activities or artifact actions.
- Sensitive requests force the sensitive overlay and prohibit playful mystery, spectacle, diagnosis, and invented motives.
- Every activity has bounded time/material/effort, accessible response alternatives, a fade policy, and a warm stop path.
- Code derives typed safety risk, hazards, supervision, and explicit accessibility floors from the request; a proposal
  cannot lower those constraints. Heat, ignition, electrical, sharp-tool, chemical, water, small-part, outdoor, and
  body/health risks are explicit plan types. Ordinary response mentions that differ from the plan are review guidance;
  adult-owned or protective additions do not suppress an otherwise useful answer. Concrete hazardous actions still
  must be assigned to the grown-up rather than merely mention supervision elsewhere, and artifacts retain their stricter
  execution contract. Safety-critical turns are classified before any provider call and construct a reviewed deterministic
  response before planning, generation, or semantic criticism, so a slow model, model outage, or invalid model envelope
  cannot suppress the minimum safety answer; deterministic plan validation still checks that response. The reviewed
  no-go routes cover fingers or objects entering outlets, conductive objects with powered appliances, energizing an
  appliance around a conductive object, carrying damp/soaked/splashed or in-water appliance state across sentence
  boundaries (including hyphenated/water-logged wet state, modified submerged state, dripping/leaking, underwater,
  simple/perfect-passive or knocked/fallen placement, floating, submerged, immersed, sitting-in-water,
  water-inside-device, and water-covered states), binding named actions and
  nearest-antecedent pronouns to the same wet appliance, using
  powered equipment near or over water, putting
  common powered devices or appliance aliases into
  open-vocabulary water containers, ignition and gas controls (including valves, knobs, and dials), hazardous cleaner
  handling, and sharp or powered tools, including ordinary arming and trigger-operation variants. Wet-device energizing
  includes recharge, charge-up, and power-up wording. Direct operations on
  these high-risk controls fail closed even when the verb is novel; explicitly observational, descriptive, and
  adult-owned references remain ordinary.
  Appliance-water
  events bind a separate typed response duty whose code-owned answer explicitly says not to put, use, plug in, or turn
  on wet or water-adjacent powered equipment; candidate validation preserves that subtype independently of the broader
  electrical category. Hazard
  actions must bind to the named object or a clear cross-sentence referent; polarity-aware parsing distinguishes safe
  negation from inability, refusal, or a failed attempt to stop, recognizes child-group ownership even near an adult
  mention, preserves a child subject across a nearby adult object, resolves adult-to-child recipients using grammatical
  and gender evidence, follows finite or progressive nested child reporting verbs and pronouns back to an explicit owner,
  and fails closed
  when an owner cannot be authenticated, resolves
  bidirectional contextual appliance aliases and parts, and treats
  any named non-appliance destination—or an external `beside`/`near`/`next to`/`behind`/`across from`/`outside`
  relation in either grammatical direction—as ordinary placement, while evaluating a later coordinated action
  independently
  rather than relying on a household-noun whitelist.
  Explicitly authenticated adult and third-party narrative actions, along with unrelated nearby objects, do not trigger the
  emergency-style route. Clause-bound validators also reject negated help,
  secrecy, unsafe reassurance, and unrelated keyword
  stuffing. Validation pre-scans every child-visible response, visual, and artifact field for concrete referents, then
  resolves conductive objects, appliance parts, wet or water-adjacent appliance state, action relations, and
  cross-field pronouns regardless of serialization
  order while preserving explicit safe-separation wording. Common hot tools,
powered tools, hair/blow dryers, curling irons, appliance compartments, and open-water containers are
  included in the conservative ontology. A dedicated self-harm phrase-family boundary always enters sensitive,
  non-assessing support, including present, written/completed/finalized/adjective-modified/dated/all-ready-or-set/planned,
  preparing/chosen, persistent
  method-thought,
  intentional-injury,
  prior-attempt, past, and recalled
  disclosures plus common stop/keep-living,
  stop-breathing, and never/not-wake constructions; prospective accidental injury, phrasal uses, and benign compound
  nouns such as fire trucks and fire stations, as well as explicit game/story death contexts, remain ordinary.
- Artifact instructions visibly enact a non-writing response choice even when no evidence is collected. Explicit
  `say`, `point`, `draw`, `move`, `choose`, or `build` mechanics are required; ambiguous phrases such as “explain in
  writing” do not count. Explicit no-writing, read-aloud, and low-fine-motor requests are also enacted by deterministic
  fallbacks.
- Composite evidence templates are complete mechanics: predict-test-explain requires prediction, test, observation, and
  explanation; compare/classify, retrieval/application, and design/iterate similarly require every named move. Reading
  or pointing to those words is not evidence-template enactment, and completion or rewards cannot stand in for mastery.
- Closure is complete and non-urgent; no streaks, cliffhangers, automatic schedules, or required continuation.
- Unknown concepts fall back to a reviewed general concept with evidence collection disabled.

## Release gates and lineage

On model-generated routes, deterministic validation runs before semantic critics. For ordinary responses, the pedagogy
plan guides age-appropriate quality rather than serving as an exact release script: missing or alternative teaching moves
are advisory and receive proportional semantic review. Concrete unsafe child directions, explicit accessibility failures,
evidence fabrication, extreme reading load, and sensitive-support boundaries remain hard gates. A typed safety-response requirement selects
the reviewed code-owned response path above and does not depend on a generator or model critic. Artifact release adds a comprehensive plan-bound
semantic gate spanning exact plan fidelity, every evidence phase, executable access modes, typed hazard ownership,
healthy closure, and color-independent mechanics. Revisions receive the same approved plan.

New responses, actions, experiences, learning artifacts, response visuals, and attributable feedback retain the plan ID
and hash. SQLite triggers reject wrong-event, cross-child, wrong-hash, artifact/experience, and visual job/asset
mismatches. The source event's authority-bearing semantics are immutable once a plan exists. When a parent explicitly
requests an artifact that the source response plan did not afford, code compiles a request-digest-specific artifact plan
with an immutable `supersedes_plan_id`; an artifact derivation can supersede only a plan for that exact event, enforced
both in persistence code and a database trigger. It never rewrites the source answer plan. Legacy responses
receive an explicit non-inferential backfill plan only when a parent requests a new artifact; legacy arbitrary artifact
action payloads cannot execute.

A code-reviewed visual-only revision follows the same rule: it copies semantic choices into a new-event plan without
copying task/rubric authority, records the source plan as superseded lineage, and binds the revised response and visual
job to the new plan.

## Evaluation

`tests/test_phase2_pedagogy_kernel.py` attacks proposal authority, registry compatibility, stored-plan integrity,
pre-generation persistence, crash and concurrent replay, critic removal, direct/sensitive boundaries, phrase-family
self-harm routing (including direct intent, suicidality, persistent method thoughts, prior attempts, past/recalled disclosure, indirect loss-of-meaning language, and object-aware
  compound/residence/streaming contrasts), a promotion-required candidate-safety category covering direction-prefix,
  target-first polarity, nominal endorsements, and negation-shaped reminders through persisted response rejection for
  response prose, visuals, and extensions. The artifact case uses the real generator and persistence boundary to prove
  unsafe drafts are discarded, only a safe unreviewed fallback persists, and nothing enters delivery; protective
  warnings and immediate trusted-grown-up handoffs remain benign contrasts. Pre-provider
  deterministic child-facing safety responses,
polarity-aware negation validation, legacy-plan safety upgrade, and relation-bound electrical/contact refusal plus
appliance-water subtype preservation, contextual-alias, ignition/chemical/sharp-tool no-go handling and benign contrasts across response, visual, and
artifact surfaces, cross-step
hazard ownership, all-field artifact safety, universal accessibility floors, complete
evidence mechanics, executable action payloads, deletion and cross-event lineage, instruction disclosure, healthy closure, and evidence
exclusion. The Lab adds public `pedagogy`, `accessibility`, and
`evidence_validity` suites to the promotion-required matrix. These deterministic gates establish contract behavior, not
educational effectiveness; live semantic judging and parent review remain required for promotion.
