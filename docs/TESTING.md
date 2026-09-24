# Testing and evaluation

## Default beta gate

```bash
ruff check .
python scripts/check_test_budget.py
pytest
curiosity-lab --repo . --json-out private/eval-report.json
```

The default suite is intentionally small enough to run during normal development. It currently collects **293 pytest
cases** and is capped at **293**. The active offline Lab contains **91 public/synthetic cases**. CI reports the slowest
20 tests so performance regressions are visible instead of hiding inside a single pass/fail result.

This gate represents an authenticated, trusted-parent beta. It protects against model mistakes, privacy leakage,
developmentally poor child-visible material, and unapproved consequential actions. It does not spend the default test
budget defending against a malicious parent, deliberate SQLite manipulation, hostile backup replay, or thousands of
wording variants intended to defeat a safety parser.

## What the default suite optimizes

The order of product importance is explicit:

1. A parent receives output that is useful and engaging for an early-elementary child in a reasonable amount of time.
2. The explanation, diagram, and activity are factually sound and preserve the family's stated constraints.
3. A challenge defaults to “see it, do it, show what happened”: a meaningful concept visual, tactile action, and a
   low-reading evidence surface. Decorative artwork warms the learning structure instead of replacing it.
4. Family context is private, correctly attributed, inspectable, correctable, exportable, and deletable.
5. Child-usable material passes focused developmental, accessibility, and physical-safety boundaries.
6. Durable or external actions require explicit parent consent, provenance, revocation, and rollback.

Passing schema validation is not the same as being useful. Public cases in `evals/family_usefulness/` therefore carry
machine-executable `quality_contract` fields. These can require visible ideas, accept equivalent wording, reject filler,
bound response size, and forbid an unrequested activity or visual. Checks inspect only parent/child-visible fields;
private reasoning metadata cannot make a weak answer pass.

Parent-facing controls are held to an end-to-end outcome contract too. A “Make a challenge” regression must exercise
the context-to-artifact path, tolerate a malformed or unusable model response within the bounded repair budget, render and persist
the PDF, pass visual QA, and assert concrete child actions plus a usable response surface. An internal schema object or
successful safety review alone does not count as a working control.
Recognized, fully specified challenge patterns may use a tested deterministic activity with zero generation calls. Interactive
challenges use focused code-owned checks plus parent review. Reusable picture pages receive one combined review of
the actual rendered page, with the previous delivered page included for a revision. A failed picture request may not
be replaced with an unrelated text worksheet and reported as complete.

The [live parent-usefulness pack](../lab/parent_usefulness/README.md) separately tests 24 synthetic everyday parent
situations through the configured model and the normal conversation service. It covers spoken help, play, siblings,
picture pages and multi-turn changes. Generation receives only parent inputs. One sequential JEV call classifies each
complete conversation's delivered text and extracted PDF instructions. Parents rate the same text beside JEV and check
source photos and rendered pages separately. Picture design allows at most five drafts: up to two structural corrections and three rendered-page reviews.
Repairs retain semantic feedback, previous drafts and successfully generated artwork even when layout fails. Its use-as-is ratings, remaining parent work and latency must
be reported alongside offline checks. A Lab `--live-judge` run judges offline candidates and does not replace this
live service evaluation. Model scores remain proxies until calibrated by parent review.

The [photo follow-up pack](../lab/parent_usefulness/photos/README.md) adds real source images, five-photo intake,
rendered Slack buttons, duplicate-click replay and open questions after challenge delivery. It exercises the Slack
handlers and outboxes with local network I/O; review includes every input photo and final page. The complete frozen
release run includes all six packs and six predeclared repeats (56 conversations). Real Slack delivery is checked
separately. Text-only conversations cannot establish photo grounding or working chained buttons.

For hybrid challenge pages, focused regressions additionally assert that the generated prompt contains no family text,
art is cached, provider failure preserves the printable, final-page review receives the rendered PNG, rejected art is
removed without losing the challenge, and diagram mechanics match the instructions. The current used-prompt set checks
equal ice prediction spaces with no revealed shape, explicit upside-down cup orientation, and outer-wing-tip-only bends
with two trials for each setting. A colored-pixel count alone is not an acceptable visual assertion.

The active balance is deliberate:

- 7 family-usefulness cases;
- 6 performance/coherence cases;
- 5 focused safety cases.

The test-budget script fails unless usefulness plus performance coverage exceeds focused safety coverage.

## Performance and delivery contracts

An ordinary quick answer has a three-call healthy-path budget:

1. select a bounded pedagogy plan;
2. generate the answer;
3. run one combined family-usefulness review.

Deterministic code still enforces response shape, approved-plan binding, evidence separation, accessibility floors, and
targeted output safety. A requested richer learning thread adds one pedagogy critic. Stronger independent review is
reserved for consequential durable/external work.

All ordinary responses are text-first. An exploratory thread may start one optional visual job when the question has a
strong comparison, prediction, test, observation, sequence, or mechanism opportunity. That visual must be a useful
diagram, comparison card, activity sequence, or printable whose code-rendered structure carries the learning. Image
generation may make the artifact warm and engaging, but it does not create the labels, facts, ordering, or geometry.
Standalone topic illustrations do not qualify. “Try another” does not silently request a visual and must return a
substantively different explanation with a concrete optional learning move, not a paraphrase or status line. Missing or
generic-only answer content becomes an actionable failure; it must never be rendered as friendly-sounding filler such
as “Let's look at that together.” A parent’s text-only/no-visual instruction overrides automatic diagram cues. The main
generation must supply any optional cards, labels, or recording sheet it proposes; an incomplete aid is dropped rather
than triggering a hidden repair call.

Questions shaped as “what might happen if…?” or “how could we test…?” route to a focused exploration so the response
can pair the explanation with a prediction, observation, comparison, or tiny test. Ordinary factual “why” questions
remain concise unless the parent asks to explore.

Wall-clock provider quality is checked separately in consented live review because CI uses a deterministic backend and
makes no billable calls. Live reports should record time to first acknowledgement, time to reviewed text, time to each
optional asset, provider calls, revisions, and failures. A decorative asset never compensates for a slow or incomplete
answer.

## Proportional validation tiers

### 1. Ordinary parent assistance

Run the core pytest suite and ordinary Lab suites. Validate routing, direct answers, factual clarity, contextual
relevance, parent effort, family usefulness, performance, pedagogy, and evidence validity.

### 2. Child-usable material

Additionally run focused safety, accessibility, artifact, and visual cases. Parent review remains part of the product
boundary. Add a safety case only for a concrete child-visible model failure or demonstrated false positive.

### 3. Durable or external action

Run the affected family-model, adaptation, personalization, autonomy, and regression cases. Require explicit consent,
provenance, and one representative interruption/recovery path for each consequential transition.

## Historical stress evidence

The large `tests/legacy_*.py` modules preserve exhaustive phase, routing, capability, Slack, artifact, visual,
adversarial, fault-injection, backup-authority, and promotion matrices. Their filenames deliberately do not match
default pytest discovery. Run one explicitly when changing or retiring the matching legacy mechanism:

```bash
pytest tests/legacy_phase2_pedagogy_kernel.py
pytest tests/legacy_phase3_family_model.py
pytest tests/legacy_phase4_adaptation_control.py
pytest tests/legacy_phase5_personalization.py
pytest tests/legacy_phase6_research_track.py
pytest tests/legacy_natural_routing_v2.py
pytest tests/legacy_slack_onboarding.py
pytest tests/legacy_visuals.py
```

The former 1,829-case safety Lab corpus remains under `evals/history/safety/`. Git history and the frozen preservation
commit retain the earlier validation evidence. Historical failures are useful migration information; they are not an
automatic reason to restore an obsolete beta requirement.

## Adding a regression

A new default test should protect at least one of these:

- an observed family workflow or quality failure;
- factual completeness or developmental usefulness;
- latency/model-call/asset-work budget;
- privacy or cross-child isolation;
- evidence/preference/inference separation;
- parent inspection, correction, export, or deletion;
- focused child-visible safety or accessibility;
- explicit consent and rollback for a consequential action.

Do not add another paraphrase merely because it can be generated. Prefer one representative boundary test plus a
useful contrast. Prefer assertions on visible behavior and durable outcomes over hashes, internal SQL shape, or every
possible process-death location.

## Live family-quality checks

Use only with explicit family consent and configured provider credentials:

1. paired Slack `connection`;
2. one ordinary text-first family question;
3. one explicit visual request where a diagram genuinely helps;
4. one correction or “try another” interaction;
5. `curiosity doctor` after the run.

Review the complete thread for latency, duplicate or late messages, generic filler, missed subquestions, unnecessary
caveats, visual instructional value, and parent effort. Do not copy private Slack messages, child details, files,
provider keys, or licensed resources into tracked fixtures. Convert a reusable failure only into a distinct public
synthetic case that remains meaningful after all family details are removed.

An offline pass proves known deterministic mechanisms; it does not establish production-model quality. Live judging
is advisory and cannot authorize delivery, create learner evidence, or activate personalization/research policy.

Generated picture pages compose image-model artwork with code-rendered text, labels, exact repeated quantities and work spaces. Label placement inspects the actual generated pixels before the combined page review. A focused regression verifies real PDF image embedding, selectable overlaid labels, repeated counts and artwork reuse on layout revisions. The [illustrated-material supplement](../lab/parent_usefulness/illustrated/README.md) evaluates the pictures themselves.

## Complete parent release evaluation

Run `lab/parent_usefulness/run_release.py` with a frozen `--source`, configured `--auth-root`, and a new `--run-dir`. It combines the 24 parent cases, four illustrated cases, four transfer cases, twelve everyday/open-conversation cases, three photo conversations and three complete conversation cases, plus two additional repetitions each of PU12, EU05 and EU06. All 56 captures must carry the same runtime hash; the harness and reviewer pack are hashed separately. JEV is the sole default text evaluator; it does not judge images or replace the generator's composed-page review. A completed capture run is not a quality pass: report JEV text ratings, separate human picture checks and failures, alongside actual Slack transport evidence.

Conversation checks include a new question after an artifact: the prior picture must not lock the next turn into
artifact revision. Semantic continuation and depth survive legacy routing heuristics; operational and context-only
boundaries still apply. A successful transport check must include this return to conversation, alongside PNG/PDF
delivery in the same thread and duplicate-event replay.

Natural replies can use `parent_response` for short games, multiple ideas and open follow-ups. Legacy lesson fields remain supported for stored answers. The reviewer sees the actual rendered parent reply. Explorations allow one focused correction and one final recovery; quick answers retain their one-revision limit. Construction patterns bind a few simple physical mechanics while the model selects a suitable activity. Their inclusion does not establish a physical trial or child learning outcome.

Slack bundles put the picture first and the PDF alongside, with image alt text, shared thread binding, normal text-delivery dependency and idempotent delivery. Phone-only material can use a compact screen layout. Optional buttons must never limit free-form replies.
