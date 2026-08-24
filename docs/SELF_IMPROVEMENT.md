# Self-Improvement, Regression Control, and Release Discipline

## Principle
Agent behavior is software behavior. Every behavioral change must be versioned, evaluated, compared, and reversible.

**Self-improvement happens in the Lab. Production only receives proven releases.**

This applies to model changes AND skill edits, prompts, tools, context schemas, retrieval policies, orchestration, renderers, resource curation, and evaluator rubrics.

## Channels
- `stable`: known-good production release.
- `candidate`: passed offline gates; eligible for shadow/canary.
- `experimental`: unrestricted Lab work; never affects family production by default.

## Required loop
1. Propose a change as a challenger release.
2. Detect changed components and dependent behavior surfaces.
3. Run static/unit checks.
4. Run component-specific eval suites.
5. Run the full golden regression suite.
6. Compare challenger to the current champion, not just an absolute score.
7. Block on factual, safety, or golden regression.
8. Shadow against live-like inputs when useful; champion remains user-facing.
9. Produce a promotion report.
10. Require operator approval initially.
11. Keep the previous champion immediately rollbackable.

## Regression capture
Any parent correction or observed failure can become a regression fixture containing input, relevant context, expected behavior, forbidden behavior, actual behavior, and release ID. Minimize private family data before committing to a public suite.

## Model/resource radar
Background jobs may discover new models, skills, resources, or world events and register challengers. Discovery never changes production. A new model is evaluated only for relevant model roles. A new resource is vetted for provenance, age range, authority, freshness, and curiosity value.

## Evaluator discipline
Use deterministic assertions whenever possible. For subjective dimensions, use separate judge models calibrated against human-ranked examples. Do not let a generator's self-score be the sole promotion signal.

## Shadowing
For eligible low-risk requests, run champion and challenger. Deliver champion only. Store minimized comparison records and evaluator preference. Never duplicate physical printing during shadow runs.

## Promotion
Promotion is evidence-based and atomic: the production manifest points to a complete release. Do not edit production skills in place. `auto_promote` is false in the MVP.
