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

Prefer a vetted real photo for observation, deterministic diagrams for teaching structure, and generated art for atmosphere. No visual is better than a misleading one.

## Hard rules

- Code renders readable text, labels, quantities, and layout.
- Generative images never establish exact counts, maps, anatomy, geometry, number lines, clocks, coins, charts, or scientific labels.
- Fact models retain `claim`, `source`, and certainty: `established`, `likely`, `debated`, or `unknown`.
- Uncertainty stays visible; simplified models are labeled when they could mislead.
- Parent approval applies to the exact validated file hash.

## One recoverable artifact run

Artifact creation follows one bounded state machine:

```text
parent-visible thread lineage -> structured draft -> deterministic validation
-> one error-guided repair -> eligible topic-specific fallback -> render and inspect -> release
```

Internal router instructions are never artifact subject matter. A fallback may release only when it is a reviewed,
topic-specific mechanic; unresolved labels such as `Version A`, generic "named feature" instructions, contradictory
trial counts, and instructions asking the child to recreate boxes supplied by the PDF fail closed. Provider or semantic
failure preserves the original thread and presents one fresh retry action instead of branching into unrelated error
messages. A rendered file is not success until content, mechanics, glyph support, layout, hash, and delivery state pass.

## Public research images

Research photos come only through source-specific adapters with allowlisted HTTPS hosts. NASA, Smithsonian Open Access, and compatible Wikimedia Commons items are supported; candidates must carry a canonical item page and a conservative reuse license or official use policy. NASA policy items never claim commercial or modification rights; explicit item restrictions are rejected. The engine downloads a bounded raster file, strips metadata, re-encodes it, verifies dimensions and content hash, then uses vision to reject people, unsafe or frightening content, ads/watermarks, private information, weak query matches, and unverified illustrations. False-color, composite, or otherwise processed science imagery is allowed only with a visible child-readable context note grounded in source metadata. Approved bytes, rights metadata, safety review, representation type, and purpose-specific alt text stay in the ignored private cache and are revalidated after 30 days. A failed candidate is never shown.

Source preference depends on the topic. Official science and cultural collections come first when an adapter is available. Missing adapters or failed checks fall through to another vetted source, a deterministic diagram, or no visual; the system does not scrape arbitrary search results.

Visual QA inspects the rendered page for factual conflicts, misleading imagery, label/arrow errors, wrong counts, clipping, overlap, tiny type, age-inappropriate density, and ambiguity. Structural PDF checks always run for Tier A/B; configured multimodal QA can add model review. Parent approval cannot override a changed hash.

## Hybrid challenge pages

Supported K–2 challenges use a typed visual brief that separates four jobs: generated illustration for invitation,
deterministic diagrams for concepts and actions, deterministic text for directions, and deterministic geometry for
the child's marks or drawings. The image prompt is selected from a small code-owned subject key and never receives
the source event, child name, or family wording. One high-quality landscape candidate is cached by prompt and model
hash; generation failure leaves the complete deterministic challenge intact.

The page is judged only after the illustration, diagram, directions, and response surfaces are composed. The compact
rubric scores usefulness, engagement, factuality, learning value, child usability, and visual integrity. Answer
leakage, misleading visual metaphors, missing action orientation, diagram/instruction disagreement, and unusable
response geometry are explicit critical defects. If the generated art itself causes a critical defect, the exact page
is rerendered without it rather than delaying the parent with a second image-generation call. Repeated identical
pages reuse the private rendered-page review.

## Slack response visuals

Slack uses the same truth boundary with a smaller surface: at most one local PNG card or decorative illustration per answer. Comparison cards are qualitative and marked **not to scale**; activity cards show parent/child actions, not scientific sequence facts. Tier C response diagrams remain disabled.

Model-proposed deterministic cards are never rendered directly. The runtime may substitute an exact reviewed local template for the supported robot-size comparison or pretend-robot activity; otherwise it stays text-only. It never relabels an instructional proposal as supportive. New deterministic templates require code review and tests.

Decorative generation is household opt-in and receives only a minimized generic scene prompt; the broad topic may be present, but known household identities and private-context categories fail closed. Code supplies captions and alt text. Every upload rechecks the private output path, PNG signature, byte count, and validated hash; useful answer text is delivered first and survives any visual failure.

Reviewed response cards may embed generated decoration in opt-in mode. The generated region is marked **imaginary art**, may not carry labels or scale, and must pass the configured vision review. Failure falls back to the Tier B code-rendered card.
