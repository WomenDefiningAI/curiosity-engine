# Learner Model Constitution

This document is the non-negotiable authority contract for Curiosity Engine's child context and learner model.

## Authority chain

Durable understanding follows this one-way chain:

`raw event → bounded annotation → evidence record → claim → derived projection`

- **Raw events** preserve parent-supplied or code-captured facts with source, child attribution, time, and episode lineage.
- **Annotations** are bounded semantic proposals. Model-authored annotations have no durable learner-state authority.
- **Evidence records** are created only by versioned code from a declared learning opportunity, rubric, assistance level, and provenance.
- **Claims** retain supporting and contradicting evidence. They remain scoped and uncertain.
- **Projections** are disposable, recomputable views produced by deterministic, versioned rules.

Skipping a layer is a contract violation. A generated answer, activity, visual, model annotation, retry, or compacted summary cannot evidence its own effectiveness.

## Learning distinctions

The engine must keep these conditions distinct:

1. availability of an answer or activity;
2. exposure to an idea;
3. task completion;
4. performance with support;
5. independent performance in a recorded context;
6. transfer to a meaningfully different context.

None of these establishes intelligence, school attainment, diagnosis, personality, or a global trait. Repeated messages, retries, reformulations, and turns in the same episode are not independent evidence.

## Code-owned provenance

Code binds the child, source event, reporter, source type, task/rule version, episode, assistance, and retention class. A model may not supply or override child IDs, event IDs, sources, node or edge IDs, evidence IDs, confidence, learner state, interest phase, permissions, tool scopes, schedules, or delivery targets.

Only deterministic, versioned projection code may derive learner or interest state. Model output may be retained as an audit-only proposal, but it is excluded from the model-facing learner profile until a reviewed evidence compiler accepts it.

## Correction, uncertainty, and legacy data

Raw records and corrections are append-only. Contradictions remain visible. Excluding, expiring, correcting, or deleting source evidence must cause every dependent claim and projection to recompute.

Legacy model-authored learner-state keys remain locally inspectable for audit and rollback. They are untrusted, are never silently rewritten as valid evidence, and are omitted from model-facing context. A future migration may supersede them only through an explicit, versioned projection rule.

## Separate durable state

Parent preferences and household memory are explicit, visible, reversible instructions. They are not child evidence. Child interests, knowledge, preferences, temporary affect, support needs, and parent working rules use separate schemas and promotion rules.

Thread compaction preserves intent and pointers; it cannot create evidence, preferences, permissions, or delivery authority. Family recipes may be drafted by a model but require tests and human activation. Tools, schedules, permissions, persistent procedures, and external delivery remain typed, code-owned, scoped, audited, and human-approved at the existing gate.

## Family rights and release gate

A parent must be able to inspect, understand, correct, exclude, expire, export, and delete durable family state. Every inference must say what evidence supports it, what contradicts it, how uncertain it is, and which rule/version produced it.

No adaptive behavior ships merely because tests are green. It must pass relevant deterministic suites, public behavioral evaluation, privacy and fairness review, an independent live judge when promotion requires one, operator approval, and a reversible rollout.
