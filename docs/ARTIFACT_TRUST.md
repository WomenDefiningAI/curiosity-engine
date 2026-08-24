# Artifact Trust Pipeline

Educational printouts must be trustworthy enough to put in front of a child. The engine separates **what is true** from **how it looks**.

## Pipeline

```text
research / trusted source
→ structured fact model
→ artifact trust tier
→ asset-method selection
→ artifact spec
→ deterministic layout
→ render
→ machine checks
→ multimodal visual QA
→ parent approval / print
```

## Golden rule

> **AI may illustrate truth; it must not invent the truth that a knowledge-bearing visual teaches.**

## Trust tiers

### Tier A — Decorative
Narrative covers, friendly characters, atmospheric art. Generative imagery is allowed. Basic visual QA is sufficient.

### Tier B — Learning-supporting
The visual supports a real concept but is not itself the authoritative diagram. Facts must be established first. Generative illustration can be used when it cannot silently change the taught fact. Final visual QA is required by default.

### Tier C — Knowledge-bearing
Position, count, labels, geometry, sequence, scale, or structure conveys the lesson. Examples: anatomy, maps, number lines, graphs, clocks, electron configurations, periodic tables, life-cycle sequences.

Tier C requires:
- structured fact model with provenance;
- trusted-source or deterministic knowledge-bearing assets;
- deterministic text/labels/counts;
- machine validation;
- multimodal visual QA before printing.

The v0.1 MVP validates Tier C specs for the trust test suite but refuses to generate them. This is an intentional fail-closed scope boundary until provenance, deterministic knowledge-bearing renderers, and mandatory model visual QA are complete end to end.

## Asset hierarchy

Prefer, in order:

1. trusted existing scientific/educational asset;
2. deterministic SVG/HTML/CSS/programmatic diagram;
3. generative illustration grounded in verified facts;
4. no visual.

No visual is better than a misleading visual.

## Hard rules

- Never rely on generative images for exact counts.
- Never bake instructional text or labels into generated images.
- Never use generative imagery as the source of truth for maps, anatomy, number lines, charts, clocks, coins, geometry, life-cycle order, or scientific labels.
- Code owns quantity and layout.
- Text is rendered separately from imagery.
- Unknown/debated facts remain marked unknown/debated.
- Simplified scientific models must be described as models when the simplification could create a misconception.

## Provenance

A fact model stores `claim`, `certainty`, and `source`. Certainty is one of:

- established
- likely
- debated
- unknown

For `unknown`, the system should turn uncertainty into curiosity instead of making up an answer.

## Final visual QA

A model routed to `visual_qa` inspects the rendered page, not merely the source spec. It checks:

- factual inconsistencies;
- malformed or misleading imagery;
- label/arrows mismatches;
- incorrect counts;
- clipped or overlapping content;
- illegible typography;
- age-inappropriate density;
- visual ambiguity likely to confuse a child.

The MVP always runs structural PDF inspection and rendered-image checks for Tier A/B. It can additionally route the rendered page through the configured multimodal `visual_qa` model. Parent approval is still mandatory and cannot override a changed file hash. Tier C has no print path in this release.
