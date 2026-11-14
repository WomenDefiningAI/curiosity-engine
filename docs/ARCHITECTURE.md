# Architecture

```text
parent in Slack
      │ paired identity + normalized message + durable receipt
      ▼
durable thread session ──► conversational manager ──► direct parent reply
      │                                               │
      │ prior releases + bounded child context        ├──► context-only record
      │                                               └──► reviewed typed tool
      │                                               │
      └───────────────────────────────────────────────┤
                                                      ▼
     approved pedagogy plan (persisted before generation)
       │
       ▼
     answer ──► Block Kit choice ──► visual ──► worksheet/activity/challenge
       │              │                 │                    │
       └──────────── durable ordered release units + outboxes ────────────► Slack thread

confirmed weekly request ──► narrow scheduler ──► interactive parent check-in
```

The harness is provider-neutral. Models interpret natural parent language, select reviewed tools, and produce typed
content. The parent agent may take one bounded read-only recall step before acting, so “change that earlier diagram”
works without an open-ended tool loop. Code owns identity, permissions, approval, transactions, rendering, retries,
files, schedules, and external effects. There is no general shell or computer-control tool in the family runtime.

## Main boundaries

- `sessions.py` persists thread history, agent turns, tool calls, code-owned output references, and reversible
  presentation preferences scoped to one thread.
- `capabilities.py` progressively loads reviewed workflow and skill instructions from `capabilities/`.
- `parent_agent.py` runs the bounded semantic tool loop; `tooling.py` enforces deny-by-default policy.
- `interactions.py` compiles transport-neutral choices to Block Kit using opaque, bound, expiring, one-use tokens.
- `learning_artifacts.py` owns distinct worksheet, activity, and challenge contracts plus one-page render/QA.
- `scheduler.py` owns confirmed weekly check-ins and durable, deduplicated runs.
- `service.py` binds tools to family-safe operations; `transports/slack.py` only handles Slack normalization/delivery.
- `db.py` owns the versioned local SQLite state; text, visual, and artifact outboxes isolate external delivery.
- `learner_model.py` owns immutable concepts/tasks/rubrics and learning plans, plan-bound opportunities,
  per-concept evidence compilation, expiry/corrections/deletion, and separate knowledge/interest projections.
- `family_model.py` owns the child-scoped local inspector, plain-language rationale/provenance, parent controls,
  export, replay-safe deletion, and non-evidentiary child co-review.
- `pedagogy.py` owns the proposal/approval boundary, reviewed strategy registry, selected instruction bundle,
  deterministic plan-fidelity gates, and compact release trace.

Every model action produces an inspectable capability run and ordered release units. This lets the text arrive first,
while a visual or printable finishes later, without losing thread identity or repeating work.

Models do not author interactive plans or graph effects. Code produces every displayed interaction, binds its opaque
choice to the current parent/session, and rechecks the target before execution. Learning generators may return only
bounded observation candidates. Those candidates persist as audit-only annotations and cannot enter the Phase 1
evidence compiler.

Because historical observation/node/edge/claim rows predate that authority distinction, the context builder omits the
entire legacy graph projection and output-feedback summaries from model context. The local inspector retains them for
audit. Phase 1 builds code-owned, rule-versioned shadow snapshots, but a shared input boundary physically removes both
the snapshot and its summary before runtime or Director model calls and budgets the complete event-plus-context input.
Phase 5 adds a separate evaluated-personalization plane. Shadow decisions compare the fixed baseline with a bounded
candidate without entering model context. Every active decision is bound to the exact child, event, workflow, proposal,
baseline plan, concepts, immutable policy behavior, deterministic evaluation, and full-Lab independent-judge review.
Only separately reviewed and operator-promoted policies plus exact per-child family opt-in may calibrate
`interest_support` and `challenge_assumption`; the compiler derives those values from the canonical decision, direct and
sensitive work remains unpersonalized, and suggested strategy changes remain shadow-only. Monotonic policy/family
epochs fence every response, artifact, queued response visual, and research/generation/review provider call plus atomic
response/artifact/visual commits; revocation discards in-flight personalized output or terminalizes queued visual work,
and later artifact work resolves the separately bound
baseline. Concept-derived features bind exact ontology versions. The Lab requires an observable effect from both
calibrated fields and records an independent judge route. A content manifest invalidates promotion after any engine, config, prompt,
capability, or fixture change. See
`docs/EVALUATED_PERSONALIZATION.md`.

Phase 6 adds a separate research plane rather than changing the deterministic learner model. A fixed, versioned Beta
companion summarizes exact linked knowledge observations and records its interval beside the current projection. The
trace is immutable, advisory-only, and invalidated when its evidence or projection changes. It cannot authorize a
state transition, plan, model prompt, release unit, or delivery. An optional proactivity pass can turn a current trace
plus a recurrent-interest projection into an inspectable `candidate_only` record; it has no path to `opportunities` or
any text/file outbox.

The research policy begins as `candidate`. Activation requires enough real deidentified calibration cases bound to
an immutable corpus manifest approved by a reviewer other than its proposer,
group-disjoint train/heldout splits, task/domain and outcome diversity, Brier/ECE limits, the normal complete Lab with
an independent live judge, a distinct reviewer and operator, operator activation, and explicit per-child parent
opt-in. The review runs the Lab internally, and evaluation through candidate writes all bind the exact release-content
hash. Corrected evidence is authoritative and multiple attempts in one learning episode collapse to one conservative
signal. Independently approved corpus manifests freeze their cases immediately; pre-promotion calibration changes
clear stale evaluation authority, and operator approval freezes the selected dataset before activation. Evaluation
uses one write transaction from input read through policy binding. Database recovery canonically recomputes
current-release evaluations, structurally authenticates historical evaluations and full-Lab reviews without granting
them current authority, and validates promoted policy, trace/candidate provenance, derived candidate copy, plus
append-only parent-control lineage. Release drift revokes active policy and family authority and returns the policy to
a fresh candidate lifecycle while preserving inspection, export, and deletion. Runtime
authorization repeats that chain check. Policy and family authorization epochs fence trace and candidate writes. Pausing collection, opting out,
disabling the policy, changing projection evidence, or deleting the family model revokes or removes derived research
state. Backup conflict detection includes research controls, control events, traces, and candidates as child-model
state. Synthetic fixtures exercise the mechanism but can never satisfy the real-data gate. See
`docs/RESEARCH_TRACK.md`.

Every activation is preserved as an immutable authorization snapshot binding its epoch, release, evaluation, review,
operator approval, and activation audit events. Traces reference the snapshot, and recovery derives snapshot, trace,
and candidate identities from their hashes while reconstructing the odd activation-epoch sequence. The packaged
runtime ships and verifies the same complete release manifest used by source checkouts, including all Lab fixtures.
Policy and family authorization high-water marks are additionally staged in a database-specific, hash-sealed
two-phase sidecar before the corresponding database commit. Startup discards interrupted grants, rolls interrupted
revocations forward, and rejects missing revocation events, row rewinds, foreign database identity, or a missing
ledger after authority has ever been granted. Every registered root and published managed restore is authenticated
against the stable family identity, carries the complete peer-root membership, and is locked in deterministic order
with backup publication and deletion. Revocations fan out to those recovery authorities before the live database
commit; a root that observed only part of an interrupted fanout can therefore identify the full peer set and recover
the newest revocation even after the active database is lost. Restore and ordinary reopen merge that root-level
high-water mark so an older or already-published image cannot resurrect consent. Reconciliation also runs before every
privileged research connection is yielded and resolves pending revocations in authenticated published restores; an
unopened restore therefore neither gets one stale transaction nor blocks a later grant or model erasure. Newer grants
are never copied into stale images, an older still-active grant is rejected, and an older verifiably closed image may
remain closed across a skipped grant epoch. The database-local ledger is required again before each privileged
connection is exposed, and grant preflight checks every authenticated target's committed epoch before changing any
authority bytes. Missing peers, pending registration, foreign family identity, unsafe paths, and damaged or
nonreciprocal registries fail closed. Model erasure retains only
a hashed subject and lineage floor, allowing an explicit later opt-in to begin a new event suffix without restoring
deleted history. Installed runtimes may use source-manifest mode
only when that source root owns the imported package, preventing authority from being borrowed from another checkout.

Schema 18 uses composite foreign keys to bind events, episodes, opportunities, evidence, and projections to the same
child. Concepts, tasks, rubrics, and plans are immutable/versioned definitions, not inferred child state. Corrections
are append-only and survive source deletion; deletion cancels the plan, cascades compiled lineage, and triggers a
deterministic rebuild in the same transaction.

Schema 19 adds a distinct immutable pedagogy-plan authority. A semantic planner proposes registry choices without IDs;
code binds and persists the approved plan before the content generator runs. The generator and an unremovable critic
floor receive the same plan and only the selected strategy instructions. Responses, plan-compiled actions,
experiences, artifacts, visuals, and attributable feedback retain its ID/hash; database triggers reject mismatched
lineage. Pedagogy evidence templates describe possible future observations but create no Phase 1 evidence.
An explicit later artifact request may create a derivative artifact plan linked by `supersedes_plan_id`; the original
response plan remains immutable. Artifact supersession is event-bound in both repository validation and SQLite triggers;
the separate `visual_revision` provenance is the only plan lineage allowed to cross into its explicit new revision
event. Schema 20 repairs plan ownership foreign keys so deleting a run clears only its optional plan pointer, while
deleting a plan, event, or child removes all dependent plan-bound output without dangling lineage.

Schema 21 adds a family control plane beside the evidence plane, schema 22 hardens deleted-event replay identity, and
schema 23 makes deletion complete across the conversational harness and process crashes. Agent turns, tool calls,
capability runs, release units, and interactions retain child-bound produced-event lineage; event creation and lineage
binding commit atomically. Existing-event tool calls prebind their source message, turn, and capability run in the same
transaction before handler execution, and database INSERT/UPDATE guards reject cross-child lineage changes. External
ownership-side guards also make event/session child and session/turn/run relationships immutable after binding, so a
previously valid lineage cannot be made cross-child by mutating its owner. External transport event IDs leave
binding-independent one-way tombstones. A durable
reservation rejects late output writers. Exclusively owned private files are moved only after a journal record exists,
and initialization restores or erases verified bytes according to whether the event deletion committed. Managed
snapshots, migration copies, and registered restores are purged, with a typed hash-only monotonic deletion manifest at
each registered backup root covering events, learning sources/expiry, and whole-model state. Snapshot creation, restore,
and deletion share process-safe lifecycle locks and reconcile committed tombstones before publication. Every managed
migration copy, snapshot, partial, and non-empty restore database is integrity-checked before live deletion can commit;
stable database identity and snapshot-manifest checks reject valid-but-unrelated images. Mutable restores are read with
WAL-aware semantics, and an active restored database is never a purge target. Restore authenticates every present
registered source database against the root identity before importing its deletion authority. Missing authority registries, unsafe,
unreadable, or damaged images fail closed. Restore staging and destination records are durable before publication and are
recoverable after process death. When the destination is the active `private/` directory, its staged database receives
the reciprocal backup-root registry before the atomic rename, preserving deletion propagation after disaster recovery.
The current backup-root format authenticates a stable family identity. A legacy v1 root can be upgraded only after
every snapshot checksum, database, and recorded source root verifies; legacy snapshots are treated as stale after any
later deletion. The external ledger also carries binding-independent transport hashes, and restore seeds both internal
and transport replay tombstones into the staged database before it can become active.
The root deletion ledger is mandatory authority metadata, retained recovery copies receive future replay tombstones,
and source-deletion/retention tombstones are imported as well. Learning opportunities retain a SHA-256 ID whose exact
derivation the database verifies; a trigger rejects a tombstoned source identity globally rather than trusting either a
caller-provided child binding or hash. Raw-event erasure closes opportunities directly linked to the event, indirectly
linked through a learning plan whose source is the event, and reachable through its recursive transfer-plan chain.
Model deletion establishes an epoch boundary: all earlier child-containing images are stale,
while snapshots created after an explicit collection restart remain valid even if collection is later paused, because
the append-only resume audit record proves the newer epoch.
Automatic schema-migration copies use one shared canonical naming helper for creation and purge discovery.
Deletion preflight validates reciprocal root/database authority plus all registered root and restore markers before the
database transaction. Snapshot preflight completes a committed private-file deletion journal, and the copier excludes
its internal quarantine defensively. Pending empty target reservations and authenticated staging trees are recovered
after process death. A root-side pending-registration barrier is durable before a legacy root becomes restore-capable.
The database-local association is then written before root identity publication. Restore requires the complete
identity-plus-source state and refuses a pending root or an identity with an empty source registry; if
the active database survives, deletion completes identity binding, reciprocal registration, and tombstone synchronization
before clearing the barrier and running its preflight.
Collection pause, corrections, retention deadlines,
exports, and deletions are typed and audited. Exports are uncapped and include the underlying child-owned records and
referenced provenance definitions. The inspector may show raw evidence to the parent locally, but it is not a
context-builder input. Raw event deletion cascades linked outputs and leaves one-way internal and transport replay
tombstones. Public profile writers and database triggers jointly enforce collection pause. Whole-model deletion removes
both Phase 1 state and the quarantined legacy graph/claim plane while preserving explicit family-authored support
preferences. Child-friendly
reflection responses remain in a separate `review_only` table with no projection edge. None of these surfaces promote
the shadow learner model into recommendation use.

One-off parent criticism remains revision feedback. Only an explicit “remember/from now on/for this thread” request
can become a thread working preference; it never becomes child evidence and the parent can clear it in ordinary chat.

Schema 24 adds a separate adaptation control plane. `ParentMemoryLedger` supplies scoped, versioned parent working
preferences; `FamilyRecipeService` supplies human-promoted private procedures limited to reviewed typed tools;
`ThreadCompactor` replaces old raw thread turns with hashed structural references plus a bounded recent tail; and
`BackgroundRunService` persists source-bound idempotent attempts, fenced leases, one-shot wakes, and parent-confirmed
capability-output delivery. The parent-agent resolves current memory values ephemerally on each turn, and governed
artifact runs resolve relevant presentation preferences plus an exact active recipe before generation, while durable
events and compactions retain only memory references. Capability runs bind exact parent-memory and legacy-preference
hash lineage before artifact work; automatic printables use the same path, and a deterministic runtime fails closed when
an active recipe cannot be enacted. An automatic printable without parent-session lineage also fails closed. Memory
inspection values are returned transiently to the parent while durable history stores only a redaction notice and typed
references. Slack's temporary delivery payload has exact memory key/version/hash lineage and is locally redacted after
a terminal send or erasure; raw-source deletion scrubs sourced memory without breaking hash-only lineage held by later
governed runs, and undo cannot reactivate a tombstoned version.
The local dashboard shows immutable compactions beside refreshed source status.
Background overlap skips retain idempotency receipts, and a terminal run can create an immediately ready one-shot
waiter. None of these tables has a path into observations, learning
opportunities, evidence records, or projections. See `ADAPTATION_CONTROL_PLANE.md`.

## Existing learning pipeline

Ordinary child questions use a concise answer contract; explicit requests for exploration use the richer learning-thread
contract. Both pass through bounded context, typed pedagogy planning, pre-generation persistence, selected strategy
instructions, deterministic validation, mandatory critics, and bounded revision. Phase 2 uses a conservative grade
baseline only; learner-state challenge and interest adaptation remain disabled until their later promotion gate.

Raw events are committed before model calls. A retry or parent revision is diagnostic context, not a second interest
signal or proficiency observation. Teacher emails, photos, and observations can source a parent-requested artifact, but the source alone is not proof
of the child’s interest, readiness, or mastery. Private-resource retrieval requires household opt-in and relevance; Slack
never receives purchased excerpts.

The fixed `connection` response stops at the transport/outbox boundary and contacts no model. Visuals deliver only
after accessible text. Knowledge-bearing diagrams remain deterministic and code-labelled; optional decorative art uses
a minimized prompt and visual QA, falling back safely when generation fails.

## Reliability

Network/model calls occur outside SQLite transactions. Duplicate inputs return the original result; conflicting reuse
fails. Confirmed provider rejections may retry within bounds. Ambiguous external completion becomes `unknown` so the
harness does not silently duplicate a parent message or file.

The MVP runs on one family computer. Slack Socket Mode needs no public inbound endpoint. `curiosity host install`
creates restartable Linux user services for Slack and scheduled work. Cloud or multi-family hosting needs a separate
threat model.
