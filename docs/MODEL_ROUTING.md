# Model routing

Curiosity Engine owns model routing. No host agent's global model selection controls production behavior.

## Roles

Core roles: reasoning, structured_extraction, image_generation, visual_qa, cheap_classifier, critic_factual, critic_pedagogy, critic_context, critic_parent_effort, critic_epistemic, judge.

Deployment config maps roles to providers/models. Production release manifests pin known-good routes. The Reliability Lab can benchmark challengers per role.

## Principle

Use the strongest model where semantic complexity warrants it and cheaper/smaller models for routing/extraction when evals show they are sufficient.

For important adversarial review, different model families/providers may be used to reduce correlated error.

A newer model is not promoted because it is newer; it must beat the champion on the relevant eval suites without blocked regressions.
