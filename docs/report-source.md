# LLM-as-a-Verifier research source report

**Research snapshot:** 7 September 2026 PDT

**Decision owner:** Curiosity Engine evolution plan

**Inspected upstream revision:** `8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770`

**Scope:** whether `llm-as-a-verifier` should influence Curiosity Engine’s harness, evaluation, personalization, or
research track. No private family data, child content, local secrets, or production traces were inspected or sent to
an external model.

## Executive decision

Adopt the **evaluation pattern** as a future, disabled research pilot; do not adopt the package as a runtime dependency
and do not give its output authority over safety, delivery, learner state, personalization, promotion, or release.

The pattern is useful because it preserves more signal than one discrete judge verdict. It compares candidates in
pairs, scores one criterion at a time, derives a continuous value from ordered score-token probabilities, repeats the
judgment, and swaps candidate positions. Those mechanisms could make Curiosity Engine’s public-fixture shadow
comparisons more diagnostic. The evidence does not establish validity for K–2 pedagogy, family privacy, or child
modeling, and the upstream implementation has permissive failure and input-discovery behavior that conflicts with
Curiosity Engine’s controls.

## Research questions and gap matrix

| Question | Primary evidence | Finding | Confidence and limitation | Curiosity Engine implication |
|---|---|---|---|---|
| Does probabilistic scoring improve comparison quality? | [Paper §§3–5](https://arxiv.org/html/2607.05391v2#S3) | The verifier takes an expectation over ordered score-token logits. On a 200-pair Terminal-Bench ablation, granularity 20, repeated evaluation, and criterion decomposition improved pairwise accuracy. | High for the reported experiment; external validity to K–2 content is untested. | Pilot it only on labeled public/de-identified Lab pairs and preserve uncertainty. |
| Does repeating and swapping help? | [Paper §4.2](https://arxiv.org/html/2607.05391v2#S4.SS2), [directed-cache implementation](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py), and [pivot tournament](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/pivot_tournament.py) | Repeats reduce estimator variance but not bias; odd repetitions swap A/B, and a ring pass gives every candidate both positions. | High for mechanism; correlated model bias and diminishing returns remain. | Require position swaps and report order sensitivity/variance; do not claim independence merely from repeated calls. |
| Are decomposed criteria better than one rubric? | [Paper §4.3](https://arxiv.org/html/2607.05391v2#S4.SS3) | Three separate code-agent criteria individually scored 75.2–76.4% pairwise accuracy and averaged to 78.3%. | Medium-high; criteria were hand-designed for code trajectories. | Version Curiosity-specific criteria, retain criterion-level results, and validate each rather than assuming the ensemble transfers. |
| Can it rank many candidates efficiently? | [README algorithm](https://github.com/llm-as-a-verifier/llm-as-a-verifier/tree/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770#probabilistic-pivot-tournament) and [PPT implementation](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/pivot_tournament.py) | The pivot tournament uses `N + k(N-k) + C(k,2)` comparisons rather than a full round robin. | High for complexity; selection quality depends on model, pivots, and data. | Consider only after a small exhaustive round-robin pilot establishes a reference; optimize cost later. |
| Can progress scoring represent child learning? | [Paper §6](https://arxiv.org/html/2607.05391v2#S6) and [progress implementation](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/progress.py) | The tool scores prefixes of agent trajectories and reports correlations with chronological task progress. | High for what was tested; no evidence connects it to curiosity, mastery, or child development. | Allow an advisory engineering-run monitor only. Never write its scores into learner evidence or the child context graph. |
| Does it fail safely enough for promotion? | [Score/error paths](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py) | Missing/malformed scores can become `0.5`; missing cache entries default to `0.5`; batch scoring defaults failed calls to a temporary `0.5/0.5` tie, although `on_error="raise"` exists. | High from source inspection. | A Curiosity adapter must abstain/fail closed for every missing score, error, budget breach, or lineage mismatch. |
| Does direct package adoption fit privacy boundaries? | [Client/input implementation](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py), [criteria lookup](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/prompts.py) | The package discovers `.env` and criteria from the working directory, can use hosted endpoints, accepts local/URL/raw images, and defaults to high concurrency. | High from source inspection; these are general-purpose conveniences, not reported vulnerabilities. | Do not import it into family runtime. Use explicit clients, bounded inputs/concurrency, no implicit path/URL/env discovery, and public/de-identified data only. |
| Can same-model self-verification replace independent review? | [Version 0.2.0 README](https://github.com/llm-as-a-verifier/llm-as-a-verifier/tree/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770#self-verification-terminal-bench-21) | Upstream reports improved best-of-N selection when one model verifies its own trajectories. | Medium for that benchmark; shared model biases remain and the result is not evidence of reviewer independence. | Self-verification may rank candidates for research, but it does not satisfy Curiosity Engine’s independent-review or operator gates. |
| Is the implementation stable enough to depend on? | [Package metadata](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/pyproject.toml), [changelog](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/CHANGELOG.md), and [license](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/LICENSE) | The package is version 0.2.0, classified beta, depends on hosted/self-hosted model clients, and is MIT-licensed. | High as of the snapshot; it can change rapidly. | Borrow concepts now. If code is later copied, pin a revision, audit it, preserve MIT notice, and vendor only the smallest reviewed portion. |

## Reconciled evidence

The headline results are substantial but answer a narrower question than Curiosity Engine has. The paper reports
strong benchmark selection and progress signals across coding, robotics, and medical tasks. It also explicitly lists
limitations: the core method needs score-token logits; criteria are hand-designed; repeated evaluation retains bias;
and the reported reinforcement-learning experiments are single-turn. None of the evaluated benchmarks establishes
developmental appropriateness, child-safety discrimination, household privacy, calibration across ages or response
modes, or resistance to correlated evaluator bias.

The upstream source contains both useful robustness ideas and incompatible defaults. Directed cache keys, A/B swaps,
criteria-at-the-tail prefix reuse, raw token usage, and non-caching of failed scores are good patterns. Silent neutral
scores, tie-on-error batch behavior, implicit `.env`/working-directory discovery, arbitrary image sources, and 50/500
worker defaults are acceptable choices for a generic benchmark tool but unacceptable defaults for a family system or
promotion gate.

## Planned propagation through the current engine

This is a future, separately reviewed slice. The current candidate changes documentation only.

| Surface | Planned update | Authority boundary |
|---|---|---|
| `lab.py` | Add an optional `PairwiseVerifier` protocol and a disabled adapter that consumes already-redacted, bounded candidate records. | It may emit a research observation; it cannot determine pass/fail. |
| Lab result schema | Store candidate hashes, A/B order, criterion ID/version, per-repeat scores, aggregate, dispersion, abstention/error, provider/model/endpoint class, seed, latency, and token/cost usage. | Append-only evaluation evidence; never learner evidence. |
| `promotion_validation.py` | Verify trace completeness, manifest binding, data-class eligibility, calibrated adapter version, and zero hard-gate bypass. | An invalid/missing verifier trace cannot be treated as a tie or pass. Initially the whole lane is advisory and optional. |
| `configs/promotion-policy.json` | Add a disabled feature declaration, strict public/de-identified input class, bounded calls/repeats/concurrency/tokens/cost, and explicit `advisory_only=true`. | Config cannot grant delivery, learner-state, recipe, personalization, research-track, or release authority. |
| Public eval corpus | Add labeled baseline/candidate pairs, reversed-order duplicates, deliberate evaluator-injection text, malformed-score/provider-error simulations, and category/subgroup slices. | No private family or child-derived text. |
| Calibration report | Measure pairwise accuracy, Brier/log loss where labels support it, reliability, false-safe/false-preference rates, A/B order sensitivity, variance, abstention, category/subgroup parity, human agreement, latency, and cost with uncertainty intervals. | Thresholds must be chosen before a held-out run; synthetic-only success cannot expand authority. |
| Reviewer workflow | Show criterion-level distributions and disagreement beside deterministic evidence; use low-confidence cases to seed adversarial fixtures. | Two independent reviewers must still approve the same frozen release digest. Repeated calls or a self-verifier do not create independence. |
| Research/job monitor | Optionally score public engineering trajectory prefixes for stall/regression signals. | Human-visible advisory only; no automatic rollback or child-facing consequence. |
| Documentation | Update Testing, Architecture, Privacy, Research Track, and Open Source Stewardship if the pilot is implemented. | Documentation and release manifest must describe the exact inactive/active state. |

## Pilot acceptance gates

1. Deterministic safety, privacy, authorization, evidence-validity, and manifest checks always run first and cannot be
   weakened, averaged, or overridden.
2. Input provenance proves every evaluated byte is public or explicitly approved and de-identified; prompt-injection
   strings in candidate content cannot alter the evaluation contract.
3. No missing score, malformed output, provider exception, partial repeat set, usage omission, budget exceedance, cache
   mismatch, or unrecognized model/version is represented as `0.5`, a tie, or a pass.
4. Every comparison includes an A/B reversal or equivalent balanced design. Order sensitivity and per-repeat variance
   are reported, not hidden by the mean.
5. Criteria remain independently visible. A good average cannot compensate for a hard safety failure.
6. Calibration uses predeclared thresholds and group-disjoint held-out fixtures; the report includes uncertainty and
   false-preference rates, not only aggregate accuracy.
7. Provider/model independence is declared honestly. Same-model self-verification and repeated stochastic calls count
   as evidence samples, not independent release reviewers.
8. No output affects a child, family delivery, learner state, memory, recipe activation, personalization selection,
   research-track activation, or release decision during the pilot.
9. The adapter, criteria, model identity, calibration corpus, and policy are release-manifest-bound and independently
   reviewed before any authority change.

## Claim-to-source ledger

| Claim family | Canonical source | Status |
|---|---|---|
| Probabilistic expected-score method and three scaling axes | [Kwok et al., arXiv:2607.05391v2](https://arxiv.org/abs/2607.05391v2) | Directly reported; benchmark transfer to Curiosity Engine is unproven. |
| Granularity/repetition/criteria ablations and stated limitations | [Paper §§4 and Appendix A](https://arxiv.org/html/2607.05391v2#S4) | Directly reported. |
| Pivot tournament complexity and position balancing | [README](https://github.com/llm-as-a-verifier/llm-as-a-verifier/tree/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770#probabilistic-pivot-tournament), [implementation](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/pivot_tournament.py) | Directly documented and implemented. |
| Neutral fallbacks, tie-on-error, implicit environment/path discovery, concurrency | [Fine-grained reward source](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/fine_grained_reward.py) | Direct source inspection. Risk conclusions are Curiosity Engine-specific inference. |
| Progress API semantics | [Progress source](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/llm_verifier/progress.py), [paper §6](https://arxiv.org/html/2607.05391v2#S6) | Directly reported; prohibition on child-state use is a governance decision. |
| Package maturity, dependencies, and reuse terms | [Package metadata](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/pyproject.toml), [MIT license](https://github.com/llm-as-a-verifier/llm-as-a-verifier/blob/8db8a114355a9d7fdf9a8d1d5c87f6aeebd18770/LICENSE) | Directly reported as of snapshot. |

## Research limitations

This was a bounded architecture review, not an installation, benchmark reproduction, security audit, legal opinion,
privacy impact assessment, or child/user study. The upstream repository and paper are authored by the same project
team; no independent replication of the 2026 results was located in this scoped review. The recommended pilot exists
to test transfer, calibration, privacy, and correlated-bias risk before Curiosity Engine relies on the technique.
