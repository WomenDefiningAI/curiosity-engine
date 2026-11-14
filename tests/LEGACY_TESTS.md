# Historical stress tests

The `legacy_*.py` modules preserve the exhaustive phase, routing, capability, Slack, artifact, visual, adversarial,
fault-injection, and promotion matrices. Their names intentionally do not match pytest's default `test_*.py`
discovery pattern.

The trusted-parent beta gate runs focused `test_*.py` wrappers. Run a historical module explicitly only when changing
or retiring the corresponding mechanism, for example:

```bash
pytest tests/legacy_phase2_pedagogy_kernel.py
pytest tests/legacy_phase3_family_model.py
pytest tests/legacy_natural_routing_v2.py
pytest tests/legacy_capability_kernel.py
pytest tests/legacy_slack_onboarding.py
pytest tests/legacy_visuals.py
```

Historical evidence also remains reachable in git. New tests belong in the default suite only when they protect an
observed family workflow, child-visible output, privacy boundary, or consequential parent-authorized action.
