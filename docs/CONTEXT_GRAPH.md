# Curiosity Graph and epistemic state

The Context Graph is not generic chat memory. It is an explicit, inspectable model of what has happened, what may be true, and how confident the system should be.

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

## Retrieval depths

0. identity/current request only
1. direct recent observations + direct school/topic state
2. related questions/concepts/interests + recent experiences
3. broader graph neighborhood + edges
4. longitudinal evidence + claims/hypotheses/contradictions

Depth is selected by workflow policy and is itself evaluated.
