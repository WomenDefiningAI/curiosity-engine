# Evaluated personalization

Phase 5 makes adaptation measurable without allowing the engine to rewrite itself. The system compares a fixed
grade-baseline plan with a candidate recommendation, stores the comparison as immutable local lineage, and applies only
two bounded plan fields—`challenge_assumption` and `interest_support`—after separate operator and family gates.
Alternative strategy selection remains shadow-only in this phase.

## Authority model

Four independent conditions are required before a new plan may be calibrated:

1. The exact immutable policy version passes the public replay suite with positive benefit, improved or equal challenge
   calibration, and no safety, privacy, fairness, or golden regression.
2. A named reviewer runs the complete promotion-required Lab matrix with the independent live judge. The matrix
   generates both baseline and personalized child-facing answers through the real reasoning/validator path, and the
   judge must pass their exact comparison. The resulting immutable review binds the policy definition, deterministic
   evaluation, fixtures, configs, prompts, capabilities, engine source, and Lab report through one content manifest.
3. A different named local operator approves that exact evaluation and review, then activates the exact policy version.
   Neither evaluation nor review promotes a policy automatically.
4. A parent opts the individual child into that exact active policy version and definition hash while learner-model collection
   remains enabled. A replacement policy returns the child to shadow mode until a new family opt-in.

Removing either active human gate has immediate effect for new work. If authorization changes after a personalized plan
is persisted, monotonic family/policy epochs fence every response, artifact, queued response visual, and public-image-
research provider call and each atomic response, artifact, visual-claim, or visual-asset completion transaction. A
response change during generation discards that result and
regenerates from a separately bound conservative baseline; a last-moment change fails closed before delivery and queues
the baseline retry. Later artifact requests resolve the exact bound neutral baseline after family opt-out, policy disable,
or model deletion; a mid-generation or pre-commit change fails closed and removes uncommitted private files. Disabling
the active policy returns every family to shadow mode. Direct-answer and sensitive-support plans remain unpersonalized
even when both gates are present.

## Data planes

`personalization_feedback_signals` stores typed outcome labels such as used/not used, continuation, prediction,
explanation, assistance level, challenge fit, and a new question. Free-form parent notes remain in `feedback` and never
enter the personalization signal or replay tables. These signals are diagnostic outcome data, not learner evidence.

`personalization_decisions` binds one categorical feature vector and candidate to the exact child, source event and its
text hash, workflow, proposal hash, baseline plan and hash, canonical concept `{id, version}` references, question intent, policy definition,
evaluation, promotion review, and approved pedagogy plan. The compiler accepts only this complete canonical decision;
it cannot accept caller-authored calibration fields. Shadow feature values are never sent to the generation model, and
the strict response/artifact trace exposes only their hash plus authority lineage. Child deletion removes child-specific
decisions, signals, and control history.

`personalization_replay_cases` accepts only reviewed categorical feature values and four finite numeric outcome labels. Its
canonical JSON, exact keys and enums, source/license/privacy coupling, identifier shapes, and content hash are enforced
again by SQLite on insert and revalidated at startup. It has
no raw-text, name, child, family, message, event, grade, birth-year, or note field. Public and synthetic cases carry their
source digest. A separately de-identified case additionally requires an explicit privacy-approval identifier and an
approved license status; there is no automatic path from private family feedback into replay.

## Evaluation and operation

The built-in `balanced-k2@1` candidate is intentionally small and inspectable. Its declarative selector rules, fallback
behavior, categorical schema, replay validation, feature extraction, and selector implementation are inside one
canonical definition hash. The deterministic evaluation separately binds its evaluator implementation and thresholds;
promotion binds the complete release content manifest. For multi-concept work, challenge uses the least-supported
concept version, and feedback is attributed and recency-bounded per exact concept version before conservative aggregation; interest may use
the strongest supported interest signal. The policy recommends a familiar band after `too_hard`, independent transfer after
`too_easy`, and measured interest support only for evidence-backed interest phases. Its suggested pedagogy strategy is
recorded for shadow comparison but does not replace the approved primary strategy.

```bash
curiosity personalization evaluate --db private/data/curiosity.db
curiosity personalization review --db private/data/curiosity.db \
  --policy balanced-k2 --version 1 --evaluation <evaluation-id> \
  --reviewer <reviewer-id>
curiosity personalization approve --db private/data/curiosity.db \
  --policy balanced-k2 --version 1 --evaluation <evaluation-id> \
  --promotion-review <review-id> \
  --operator <operator-id> --reason "Reviewed every Phase 5 gate."
curiosity personalization activate --db private/data/curiosity.db \
  --policy balanced-k2 --version 1 --operator <operator-id> \
  --reason "Start opt-in family trial."
curiosity learner-model personalization --db private/data/curiosity.db \
  --child <child-id> --enabled on --reason "Family chose the evaluated trial."
```

Use `curiosity personalization disable ...` for immediate operator rollback and the same learner-model command with
`--enabled off` for immediate family opt-out. `curiosity personalization status` reports policy and aggregate decision
counts without returning family text or feature vectors.

The Lab's required `personalization` suite replays public/synthetic cases against the fixed baseline and generates real
baseline and candidate answers through the strict compiler, reasoning engine, critics, and response validators. The
candidate route must observably enact both calibrated fields, differ from the non-neutral baseline case, bind its exact
implementation digest, and record an independently identified judge provider/model/prompt route. It verifies benefit, calibration, golden behavior, equivalent
decisions across response modes, absence of private-shaped data, and zero writes to learner evidence. An ordinary
offline Lab run is a required development gate but is not promotion eligible; `personalization review` itself runs the
full matrix with the live independent judge, requires the candidate-answer comparison to pass that judge, and stores
the authenticated result.
