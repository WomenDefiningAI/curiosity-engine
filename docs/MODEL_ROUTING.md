# Model routing

Curiosity Engine owns model routing. A coding agent's global model choice does not silently control the supported API runtime.

## Roles

Core roles: reasoning, structured_extraction, image_generation, visual_qa, cheap_classifier, critic_factual, critic_pedagogy, critic_context, critic_parent_effort, critic_epistemic, judge.

Public deployment config declares role requirements. Ignored `private/setup/brain.json` maps those roles to the family's providers/models. The Reliability Lab benchmarks challengers per role; model names are promoted only after evals.

The elementary-family minimum is a stack with strict structured reasoning, high-resolution vision, worksheet/PDF OCR and layout understanding, rendered-page visual QA, and a dedicated image-generation/editing route. One provider need not supply every role.

## Principle

Use the strongest model where semantic complexity warrants it and cheaper/smaller models for routing/extraction when evals show they are sufficient.

For important adversarial review, different model families/providers may be used to reduce correlated error.

A newer model is not promoted because it is newer; it must beat the champion on the relevant eval suites without blocked regressions.

The current OpenAI route in `README.md` is explicitly `family_evaluating`, not a champion. Direct Anthropic uses its native Messages protocol. OpenRouter uses its Chat Completions protocol plus capability/privacy routing controls; it is not an OpenAI Responses base-URL substitution.

Generated images can decorate or inspire. Deterministic code remains responsible for worksheet text, exact counts, labels, maps, number lines, and knowledge-bearing geometry.
