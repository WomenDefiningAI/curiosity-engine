# Phase 6 uncertainty and proactivity research track

Phase 6 implements the final evolution-plan slice as a research-only companion to the deterministic learner model. It
does not enable unsolicited delivery or probabilistic mastery.

## What it does

For one child and one versioned concept, `ResearchTrackService.compute_uncertainty_trace()` reads the current
code-owned knowledge projection and its complete linked evidence set. At least three eligible knowledge-performance
observations from three distinct learning episodes are required; retries within one episode collapse to one
conservative signal. Parent corrections are applied before aggregation. A fixed Beta prior then weights independent
successes/failures fully, partial results at one-half on each side, and supported results at one-quarter. The immutable trace records the posterior mean, variance,
90% interval, interval width, exact evidence IDs, deterministic projection digest/rule, policy identity, evaluation and
review IDs, both authorization epochs, and the exact reviewed release-content hash. A changed source, configuration,
prompt, capability, or evaluation fixture automatically revokes live policy and family authority, returns the policy
to `candidate`, and requires a fresh evaluation and review before reactivation.

Each activation also creates an immutable authorization snapshot. The snapshot binds its odd-numbered policy epoch to
the exact release, calibration evaluation, independent Lab review, operator approval, and activation audit events.
Traces name that snapshot directly, and trace/candidate identifiers are derived from their content hashes. Recovery
reconstructs the activation sequence and rejects a self-hashed row whose epoch, snapshot, or derived identifier was
rewritten offline. The ordered activation/disable/rollback history must also derive the current status and epoch, so
rewinding a disabled row to an earlier active snapshot cannot resurrect authority.

The latest policy and per-child authorization state is also staged before the matching database transaction in a
database-identity-bound, hash-sealed sidecar journal. Startup resolves the two-phase transition: an interrupted grant
is discarded, while an interrupted revocation is rolled forward with a system audit event. It then requires the
database to match the external high-water mark. Deleting revocation events and rewinding otherwise self-consistent
database rows therefore remains closed. A missing, foreign, or malformed ledger is accepted only for a fresh database
that has never recorded research authority. Every registered backup root and published managed restore participates
in one authenticated authority family. Revocations are staged to the full root/restore set before the live database
may commit, and every root records the exact peer set so a partial fanout is discoverable after loss of the active
database. Each privileged research transaction reconciles external authority before its connection is exposed, and
root synchronization completes pending restore revocations so an unopened restore cannot block the family's next
consent or erasure transition. Restoring or reopening an older snapshot merges newer revocations before the image can
open; an older active grant is rejected rather than treated as a newer re-enrollment, while a verifiably closed image
may remain at its older epoch when it deliberately skipped an intervening grant. A retained service also requires its
database-local ledger on every privileged call. Any grant is compared with every authenticated recovery peer before
fanout, so a stale restore cannot overwrite a newer revocation even temporarily. Missing peers,
pending registration, cross-family registries, damaged restore markers, and partial authority metadata all fail
closed under the same ordered lifecycle locks used by backup publication and deletion.

This number is an **advisory companion**. The evidence ledger and deterministic projection remain authoritative. The
trace is not a grade, diagnosis, global mastery score, or permission to change a plan.

An authorized trace may be combined with an existing maintained, recurrent, or parent-confirmed interest projection.
The result is stored only in `research_proactivity_candidates` with status `candidate_only` and `delivery=disabled`.
There is intentionally no promotion method from that table and no write to opportunities, response/Slack outboxes,
artifact/visual outboxes, schedules, background runs, model prompts, or release units.

## Gates

The seeded `uncertainty-context-k2@1` policy starts in `candidate` status. The following sequence is mandatory:

1. A corpus operator proposes an immutable manifest of approved, deidentified categorical cases. Private prose and
   family identity have no fields. A different privacy reviewer approves the exact manifest before any row counts as
   real calibration evidence; an individual caller cannot assert its own approval.
2. Pass fixed minimums: 40 real cases, 20 train, 20 heldout, eight task signatures, three domains, and at least five
   positive and five negative heldout outcomes.
3. Keep train and heldout group digests disjoint and pass Brier score ≤ 0.12 and five-bin ECE ≤ 0.15.
4. Run a separate promotion review over the complete Lab with an independent live judge, including this suite.
5. Have a different operator approve and activate the exact policy/evaluation/review tuple.
6. Record explicit parent opt-in separately for each child.

Public fixtures may help test mechanics. Synthetic cases are excluded from every real-data count and can never unlock
promotion. Each corpus and its cases are frozen as soon as its independent approval is recorded. A pre-promotion
calibration change clears stale evaluation authority for a fresh evaluation, and operator approval freezes the full
selected dataset before activation. Evaluation reads, recomputation, persistence, and policy binding share one write
transaction, so a concurrent corpus change cannot create a stale snapshot. No policy can self-promote.

The current repository contains no approved real calibration corpus and no live-judge promotion report, so the seeded
policy remains a non-promotable candidate after ordinary tests.

## Revocation and lifecycle

Every trace and candidate is bound to monotonic policy and family authorization epochs and the reviewed release.
On database reopen, current-release evaluations are recomputed from the exact approved corpus manifests and current
evaluator, while older evaluations and full-Lab reviews are retained only after archival identity and structural
integrity checks, including an exact match between each review row's release and the embedded Lab manifest. Historical
records never authorize the current release. Promoted policy, trace/candidate provenance,
deterministically derived candidate copy, and append-only family-control lineage are authenticated. The same current
chain is rechecked before activation, family opt-in, trace creation, candidate creation, and candidate listing.
Structurally plausible self-hashed database rows therefore cannot manufacture promotion or parental consent. Release
drift fails closed without bricking inspection, export, deletion, or the fresh evaluation lifecycle. Opt-out and policy disable
immediately invalidate visible candidates. Pausing learner-model collection disables the child research control.
Deleting or changing linked evidence or the deterministic projection removes stale traces and their dependent
candidates. Whole-model deletion advances an external erasure floor before removing child controls, control history,
traces, and candidates, and managed-backup
conflict detection treats each of those four tables as learner-model state. Family inspection
and export include child-scoped research lineage; global calibration data is not included unless referenced policy
metadata is needed to interpret the child record. The activating authorization snapshot is included as soon as a
child opts in, even before the first uncertainty trace is created. A later explicit re-enrollment starts at the next
external family epoch and validates the new post-erasure event suffix rather than recreating deleted history.

Source checkouts compute the release manifest from the complete runtime, configuration, prompt, capability, schema,
and evaluation surface. Built wheels carry that immutable manifest and every hashed Lab fixture; startup verifies the
installed bytes before database initialization. Source-manifest mode is permitted only when the selected checkout owns
the package actually executing; an installed wheel pointed at or launched inside another checkout fails closed. A wheel
is smoke-tested both from an empty directory and against this mixed-root attack.

## Operator surface

Use `curiosity research-track status` to inspect the closed state. Public/synthetic mechanical cases use `import-case`;
real deidentified data uses the separate `propose-corpus` and `approve-corpus` commands. Other explicit subcommands
evaluate, review, approve, activate/disable, set child opt-in, compute a trace, create a candidate, and list candidates. There is
no send or deliver command. `review` intentionally runs the full live-judge Lab and may incur provider cost; ordinary
offline tests never call it. The review command always launches that Lab internally and does not accept a caller-made
report.

## Promotion interpretation

A deterministic and offline-Lab pass means only that the fences and public mechanics work. It does not establish that
the uncertainty values are calibrated for children, that proactive timing is welcome, or that a family-facing design
is usable. Those require approved real data, consented family research, accessibility/usability review, independent
judgment, and a separate product decision for any future delivery path.
