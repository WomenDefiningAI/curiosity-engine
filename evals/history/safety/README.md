# Historical safety review rounds

These numbered adversarial rounds preserve the evidence that produced the September 2026 implementation snapshot.
They are intentionally outside the active `evals/<suite>/*.json` discovery and release-manifest paths.

The beta is a trusted-parent tool, not an adversarial child-facing agent. Active release validation therefore uses the
focused model-output and developmental fixtures in `evals/safety/child-material.json`. The original core/adversarial
input corpus and every numbered round remain here. They may be run manually when changing the legacy linguistic
classifier, but they do not expand product authority, require a live judge, or block an ordinary parent-assistance
release.

Moving a case back into the active corpus requires a concrete child-visible model-output, developmental, privacy, or
physical-safety rationale. Mere discovery of another inbound paraphrase is not sufficient.
