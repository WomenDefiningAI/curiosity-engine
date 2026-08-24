# AGENTS.md — Curiosity Engine implementation contract

Curiosity Engine is a **purpose-built autonomous AI harness for family learning**, not a general agent framework and not a kid-facing chatbot.

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

## Context rules

Never dump the whole family history into a model. Use `context_builder.py` and explicit depth 0–4. Context assembly itself is versioned/evaluated behavior.

## Reasoning rules

Reasoning topology is policy, not prompt improvisation. Important workflows may use independent adversarial critics. Critics should attempt to disprove or reject candidate output, not merely praise it.

## Autonomy

The autonomous Director may sense, reflect, generate candidates, research public resources, and prepare drafts. It delegates known workflows. `do_nothing` is always a valid candidate.

The Director may not silently promote releases, purchase materials, auto-print by default, spam parents, or make high-impact child-profile changes from weak evidence.

## Artifacts

Trusted facts → artifact trust tier → asset strategy → deterministic layout → machine validation → visual QA → parent-approved paper print.

Generative imagery creates wonder; it is not the source of truth for exact counts, labels, anatomy, maps, number lines, or scientific diagrams where geometry conveys knowledge.

## Behavioral changes

Any change to models, prompts, tools, schemas, retrieval, context depth, reasoning topology, resource strategy, renderer, evaluator, or orchestration is a behavioral change.

Every behavioral change must run relevant eval suites and the global golden/regression suite before production promotion.

## New-family onboarding

Assume a parent has cloned this repository and pointed a coding agent at it. Guide them in plain language; do not assume they know Python, Slack app administration, environment variables, databases, or model terminology.

Before changing private state, read `docs/AGENT_ONBOARDING.md` completely and follow its checkpoints. Use the deterministic setup commands instead of editing SQLite or private configuration by hand. Explain what stays local and what a selected messaging/model provider will process before asking for credentials or family context.

Never copy family details, licensed resources, tokens, private evals, or generated artifacts into tracked files, public issue text, terminal transcripts intended for sharing, or coding-agent instruction files. Store family-specific setup output only beneath ignored `private/` paths.

## Interaction boundary

The shared service layer is the source of truth. Slack is the MVP parent interaction channel and must normalize inputs into strict transport/event contracts; it may not own child state, context selection, reasoning policy, authorization policy, or durable workflow status. Future transports must use the same boundary without changing core semantics.

## Physical MVP boundary

Assume ordinary paper, writing utensils, common household materials, and optional access to a conventional paper printer at home or a library. Printables are downloadable one-page PDFs and cannot be required for core use.

Do not introduce 3D-printer integration, 3D models, slicer files, filament/material instructions, or 3D-printing roadmap promises in the MVP.
