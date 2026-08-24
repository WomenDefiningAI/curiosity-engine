# Reasoning architecture

Curiosity Engine treats model topology as code/configuration rather than an implicit single-agent trajectory.

## Budgets

**FAST** — immediate parent interaction. Small context, one generator, and bounded factual/pedagogy/context review.

**NORMAL** — artifact/experience production. Generator + relevant validation/critics.

**DEEP** — weekly reflection. Broad graph context, multiple opportunity lenses, critics, selector.

**LAB** — background experiments, shadow runs, context audits, model/prompt challengers.

## Adversarial roles

- factual critic: unsupported/overstated/misleading claims
- pedagogy critic: spoon-feeding, premature theory, shallow theming
- context critic: missed context, irrelevant context, unsupported child inference
- parent-effort critic: unrealistic prep/supervision/cleanup
- epistemic critic: exposure vs understanding, weak trait claims, missing uncertainty
- visual critic: malformed or misleading print artifact

Important critics should be independently prompted and may use different model families when configured.
