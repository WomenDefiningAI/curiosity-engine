# Curiosity Graph and epistemic state

The Context Graph is not generic chat memory. It is an explicit, inspectable model of what has happened, what may be true, and how confident the system should be.

## Raw events, episodes, and claims

The local model separates three layers:

1. **Raw events** preserve what arrived, its source, time, and transport-scoped conversation references.
2. **Episodes** group turns that appear to belong to one learning occasion. An exact repeat, close follow-up, or retry after a failed answer stays in the same episode. Elapsed time alone does not make an exact repeat independent. A meaningfully developed related question on a later occasion may open a provisional new episode; the parent can correct either decision.
3. **Claims** express interpretations such as a possible durable interest. An `established_pattern` needs supporting evidence from at least two eligible episodes—not merely two messages.

Time narrows candidates but never establishes interest by itself. The deterministic v1 classifier prefers under-counting to turning frustration, clarification, or answer repair into a learner trait. Diagnostic/system events remain visible in the local inspector but are ineligible as family evidence and excluded from later family model context.

## Core node families

child, question, topic, concept, interest, school_concept, teacher_language, SEL_theme, life_event, experience, resource, media_resource.

## Knowledge states

- `unseen`
- `exposed`
- `emerging`
- `demonstrated`
- `unknown`

Never promote `exposed` to `demonstrated` merely because an explanation was delivered.

## Epistemic states

- **observation** — something actually reported/observed
- **hypothesis** — a plausible interpretation requiring evidence
- **established_pattern** — repeated/strong evidence supports durable use

## Claims

Claims retain confidence plus supporting and contradicting evidence. Context audits should challenge stale or weak claims rather than letting them silently harden into truth.

`nodes.evidence_count` and `edges.evidence_count` are mention counts for navigation and retrieval. They are not independent-evidence counts, interest scores, or mastery estimates.

## Parent inspection and correction

```bash
curiosity context --child kid-a
curiosity context --child kid-a \
  --correct-event EVENT_ID \
  --classification new_episode \
  --note "Independent return; the earlier message was recorded late."
```

Valid corrections are `retry`, `deepening`, `new_episode`, and `exclude`. Retry/deepening corrections also require `--related-event`. Corrections are appended to `context_corrections`; raw events are never erased. The active episode projection changes while retaining who/what made the correction. Excluded evidence leaves the model-facing projection, and any established claim that no longer has two eligible episodes is automatically downgraded to a hypothesis.

The inspector includes family text and therefore stays local. Its graph-health summary reports eligible episodes, same-episode turns, non-family/excluded evidence, uncertain classifications, corrections, and pre-episode legacy evidence.

## Retrieval depths

0. identity/current request only
1. direct recent observations + direct school/topic state
2. related questions/concepts/interests + recent experiences
3. broader graph neighborhood + edges
4. longitudinal evidence + claims/hypotheses/contradictions

Depth is selected by workflow policy and is itself evaluated.

## Current limits

Schema upgrades do not silently reinterpret older events. Historical evidence without episode membership is reported as `legacy_unreviewed_count` and cannot establish a new durable pattern until explicitly revisited. Context-driven proactive suggestions remain disabled.

This release does not add embeddings, a graph database, probabilistic mastery, or unsolicited suggestions. Those are research-track options only after the episode foundation is trustworthy and auditable.
