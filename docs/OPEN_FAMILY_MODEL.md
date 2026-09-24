# Open family learner model

Phase 3 makes the evidence-centered learner model inspectable and correctable without promoting it into live
personalization. The local dashboard and `curiosity learner-model` commands show what the engine has observed, what it
currently infers, why each item counted or did not count, the rule and provenance that produced it, and the available
parent controls.

## Trusted-parent beta scope

The authenticated parent is trusted. Parent inspection, correction, export, deletion, and separation of evidence from
preferences/inferences remain product requirements. Defending against deliberate SQLite forgery, malicious backup
replay, a hostile parent, or every multi-root crash ordering does not. The extensive hash-chain, registry,
high-water-ledger, and process-kill implementation below is preserved as a compatibility record after the Phase 0–6
snapshot; it is explicitly eligible for measured simplification to a clear local backup/restore contract. Normal
consent, provenance, stale-backup warning, deletion, restart, and rollback behavior must remain covered during removal.

## Authority and scope

- The view is private, local, child-scoped, and versioned as `phase3-v1`.
- Learner projections remain shadow-only. Inspecting, exporting, or co-reviewing them never enables recommendation use.
- A model cannot call these controls. Pause/resume, correction, retention, export, and deletion are code-owned parent or
  operator actions.
- Raw evidence excerpts appear only in the local parent inspector and export. They do not enter model context through
  this view.
- Inferences are described as revisable observations, never grades, diagnoses, fixed traits, or global mastery labels.

## Parent controls

The parent may:

1. inspect knowledge and interest projections with plain-language rationales, uncertainty, evidence counts, and exact
   rule versions;
2. inspect source evidence, assistance, response mode, task difficulty, retention, corrections, and why the evidence
   was or was not eligible;
3. pause or resume new learner/profile collection without deleting existing history; while paused, answering continues
   but no new episode, legacy raw-evidence/graph profile, Phase 1 evidence, or model-annotation row is recorded. The
   operational request, job, answer, and reviewed pedagogy plan remain subject to the separate conversation-retention
   contract;
4. append an `exclude`, `restore`, or bounded result/assistance correction and immediately recompute every dependent
   projection;
5. set an earlier deletion deadline for a learning source;
6. delete one compiled learning source, delete its raw source event and linked outputs, or delete all child-specific
   learner-model data after an exact confirmation;
7. export a complete, uncapped child-scoped JSON record containing the friendly inspector plus the underlying learner
   records, correction/retention history, source lineage, referenced definitions, and content-free tombstones. CLI
   exports are owner-only (`0600`) and refuse overwrites unless explicitly asked.

Deleting a raw event removes its directly linked local event, response, run, review, graph/evidence, session,
capability, interaction, transport/outbox, artifact, visual, and private-file copies. The configured private output root is a hard
filesystem boundary; deletion refuses to proceed if a linked file is outside it. A durable file-deletion journal makes
the operation recoverable across process or power loss: startup restores verified files when the database deletion did
not commit, and finishes verified erasure when it did. A deletion reservation prevents late output writers from
acquiring the event; real artifact writers remove rendered bytes when persistence is rejected. Files referenced by a
different event remain for that owner. Slack or another provider may retain already delivered material
under that provider's own policy. The engine creates both a global one-way internal event-ID tombstone and a one-way
transport/external-event tombstone, so neither an internal replay nor a fresh receipt under a changed binding can
recreate deleted content. Current-event lineage on parent messages, turns, tool calls, capability runs, and release units
is child-bound and committed atomically with each new event, allowing all local semantic copies and cached preferences
to be erased or rebuilt deterministically. Event and session child ownership plus the owning session/turn/run columns
are immutable after binding, preventing a valid lineage row from becoming cross-child through its other side.

Managed family snapshots, registered restores, and automatic schema-migration database copies containing deleted data
are removed. Every registered backup root receives an owner-only, hash-only monotonic deletion manifest covering raw
events, learning sources (including retention expiry), and whole learner-model deletions. Snapshot creation, restore,
and deletion share process-safe lifecycle locks and reconcile committed database tombstones before publication, so a
crash or race cannot publish a stale managed copy. A stale snapshot that later reappears in a managed root is refused.
Before live deletion, every migration copy, snapshot, partial, and non-empty restore is integrity-checked; a missing,
unsafe, unreadable, or damaged database leaves the requested family operation uncommitted. Restore staging and final destinations are registered before bytes can
be published, so deletion can recover copies left by a process kill. A disaster restore activated at `private/`
inherits the originating root association before publication; later event, source, expiry, or model deletion therefore
reaches the old snapshots, recovery copies, and canonically named automatic migration backups. Copies made outside a
registered root are outside the engine's control and must be governed by the household's storage-retention process.
Every destructive operation revalidates the root's reciprocal database authority and authenticates all registered
root/restore metadata before its database transaction. Missing, corrupt, mismatched, symlinked, or cross-family
metadata fails closed. Snapshot creation first recovers any committed private-file deletion journal and always excludes
the internal quarantine, so a post-commit process crash cannot recapture deleted bytes. Empty restore reservations and
authenticated staging trees left by a killed process are recoverable on retry or the next deletion.
A backup root's source and restore registries are required authority metadata from initialization onward; disappearance
never means "no copies." A stable family-database identity binds snapshots, migration copies, and restores to the active
database, while snapshot file hashes reject same-family image swaps. Registered restores are read with SQLite WAL-aware
semantics, and the active restored database path is explicitly protected from recovery-copy deletion.
Legacy v1 backup roots are adopted only when every snapshot verifies and its recorded private/output roots match this
family. Their snapshots remain restorable before deletion, but are purged conservatively after any deletion because
the older format cannot prove absence of newer protected state. Restores import external internal-event and
binding-independent transport tombstones before publication, so restoring an image from before an event existed does
not reopen either replay path. Any historical image that still contains a tombstoned child is stale, including images
whose only learner-model state is collection control or audit history.
The root ledger is required and cannot disappear into an implicit empty state. Deletions are also propagated into
already-published recovery copies, and learning-source/retention tombstones block opportunity replay globally across
children through a source-ID hash whose exact SHA-256 derivation is verified by a database trigger. Raw-event deletion
also closes any opportunity reached indirectly through a plan sourced by that event and its recursive transfer-plan
chain. After a full model deletion, explicit
parent resumption writes an append-only epoch proof that can be backed up normally and remains valid after a later pause.
Backup-root registration uses a durable root-side pending barrier before adoption can authorize restore, followed by a
database-local association before root identity publication. Restore requires identity plus a non-empty source registry,
so a missing pending file cannot turn an incomplete root into an authorized one. If the active database
survives, deletion recovers identity binding, reciprocal registration, and tombstone synchronization before clearing
the barrier and running its fail-closed preflight. Restore identity-checks every present registered source before
accepting any of its deletion authority.
Deleting a learning source preserves only the pre-existing content-free replay tombstone and code-owned audit trail.
Deleting the whole learner model removes both the Phase 1 evidence/projection plane and the legacy nodes, edges,
claims, observations, signals, episodes, model runs, and inferred opportunity state. It pauses future collection and
intentionally leaves raw question/transport and reviewed-answer history to its separate retention contract. Explicit
family preferences are a separate authority layer and are not silently deleted with learner evidence; the UI says this
before confirmation.

## Optional child-friendly co-review

The parent may open a separate reflection page and review a small set of careful notes with the child. Copy emphasizes
that the child is the expert on their experience and may answer “yes,” “not yet,” “I’m not sure,” or skip. Responses are
stored as `review_only`; they do not become evidence, change a projection, or alter a recommendation. A parent who learns
that a source was wrong uses the explicit correction control separately.

This page is an instrument for future family co-design, not a claim that usability research has already occurred.
Before production promotion, conduct consented parent/child sessions and resolve observed comprehension, accessibility,
and emotional-safety issues. Never put real family responses into public fixtures.

The software implementation and its synthetic authority tests are complete; the human-validation exit is not. Status
reports must preserve that distinction and must not call Phase 3 production-validated until those sessions and fixes
are documented.

## Retention behavior

Built-in evidence expiry still applies: transient sources expire after 30 days and situational interest after 90 days.
A parent deadline may only shorten the effective window. The earliest previously accepted parent deadline is permanent;
a later request cannot lengthen it or resurrect evidence. Projections ignore the source after that deadline, and
`expire_learning_records()` physically removes it, creates a replay tombstone, and recomputes the child model. Setting a
deadline is append-only and audited; equal deadlines are idempotent and later deadlines are rejected.

## CLI examples

```text
curiosity learner-model inspect --child kid-a
curiosity learner-model pause --child kid-a --reason "Family privacy pause"
curiosity learner-model correct --child kid-a --evidence ev2_... --action exclude --reason "That was adult help"
curiosity learner-model retain --child kid-a --opportunity opp_... --days 30 --reason "Short retention"
curiosity learner-model export --child kid-a --output /private/path/kid-a-model.json
curiosity learner-model child-card --child kid-a
```

The local dashboard exposes the same operations with CSRF-protected forms and exact confirmation for destructive
actions.

## Verification and promotion

`tests/test_phase3_family_model.py` covers migration repair, cross-child isolation, explanations/provenance,
correction/recomputation, API-and-database collection pause, retention cleanup, replay-safe deletion, owner-only export,
non-evidentiary child review, real parent-agent and interaction lineage erasure, uncapped export, atomic event lineage,
process-kill/rollback/unlink recovery, late artifact-writer cleanup, shared-file ownership, typed managed-backup
deletion manifests, snapshot/restore races, registered-restore cleanup, pre-commit fail-closed database checks across
every managed copy class and destructive family operation, process-killed
restore staging/publication, active disaster-recovery rebinding, real migration-backup naming, local-only web controls,
reciprocal cross-family authority rejection, damaged-metadata preflight, ownership-side lineage immutability,
quarantine-free post-crash snapshots, required-registry loss, valid-but-wrong SQLite substitution, pinned-WAL active
restore protection, WAL-only model-copy detection, empty-target recovery, CSRF, and CLI discovery. The promotion-required
`family_model` Lab suite repeats
the central authority and privacy invariants on synthetic public fixtures.

Real family usability and accessibility sessions are a separate promotion gate. Their absence must never be represented
as a completed study or silently replaced by synthetic tests.
