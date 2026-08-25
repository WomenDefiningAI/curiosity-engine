# Artifact trust

Educational pages separate truth from decoration:

```text
trusted facts → trust tier → asset method → deterministic layout
→ machine checks → rendered-page visual QA → parent approval
```

> AI may illustrate truth; it must not invent the truth a visual teaches.

## Tiers

| Tier | Meaning | Rule |
|---|---|---|
| A | Decorative | Generative art allowed; basic QA |
| B | Supports a known concept | Establish facts first; final visual QA required |
| C | Position, count, labels, scale, or geometry teaches the lesson | Provenance, deterministic structure, machine checks, and visual QA required |

Tier C includes maps, anatomy, charts, clocks, number lines, scientific diagrams, and life-cycle order. v0.1 validates Tier C test specs but refuses to generate them until the complete trust pipeline exists.

Prefer trusted assets, then deterministic diagrams, then fact-grounded illustration, then no visual. No visual is better than a misleading one.

## Hard rules

- Code renders readable text, labels, quantities, and layout.
- Generative images never establish exact counts, maps, anatomy, geometry, number lines, clocks, coins, charts, or scientific labels.
- Fact models retain `claim`, `source`, and certainty: `established`, `likely`, `debated`, or `unknown`.
- Uncertainty stays visible; simplified models are labeled when they could mislead.
- Parent approval applies to the exact validated file hash.

Visual QA inspects the rendered page for factual conflicts, misleading imagery, label/arrow errors, wrong counts, clipping, overlap, tiny type, age-inappropriate density, and ambiguity. Structural PDF checks always run for Tier A/B; configured multimodal QA can add model review. Parent approval cannot override a changed hash.

## Slack response visuals

Slack uses the same truth boundary with a smaller surface: at most one local PNG card or decorative illustration per answer. Comparison cards are qualitative and marked **not to scale**; activity cards show parent/child actions, not scientific sequence facts. Tier C response diagrams remain disabled.

Decorative generation is household opt-in and receives only a minimized generic scene prompt; the broad topic may be present, but known household identities and private-context categories fail closed. Code supplies captions and alt text. Every upload rechecks the private output path, PNG signature, byte count, and validated hash; useful answer text is delivered first and survives any visual failure.
