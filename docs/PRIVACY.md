# Privacy and licensed-resource boundary

## Evaluated personalization

Phase 5 keeps free-form feedback notes in the private feedback table and copies only typed outcome labels into the
personalization plane. Replay rows accept enumerated categories and finite numeric labels only, never family text or
identity. There is no automatic private-feedback export. Separately de-identified replay requires explicit privacy and
licensing approval. Shadow feature values stay local and are not model-facing. The local parent inspector shows a
redacted child-scoped decision/control history, while the explicit owner export includes complete child-scoped lineage
and only the global policy/evaluation/review definitions referenced by that child. Child-model deletion removes
child-specific decisions, signals, and opt-in history. See `docs/EVALUATED_PERSONALIZATION.md`.

## Uncertainty and proactivity research track

Schema 26 keeps research calibration outside the family evidence plane. Imports accept only bounded categorical
counts, a binary outcome, coarse domain, split, and one-way source/group/task digests. A corpus operator proposes an
immutable exact manifest of approved deidentified cases and a different privacy reviewer must approve it before the
rows count as real calibration data; individual callers cannot claim approval. Private text, child IDs, family IDs,
notes, grade, birth year, and resource excerpts have no calibration fields. Public and synthetic cases cannot claim
private approval, and never count toward promotion minimums.

Child uncertainty traces are local, evidence-linked derivatives. They retain exact evidence IDs and policy/family
authorization epochs plus exact policy, evaluation, review, evidence, and release-content lineage so parents can
inspect and delete them with the learner model. They do not enter model context or
change the deterministic projection. Context-driven output is an inspectable `candidate_only` row with delivery marked
disabled; no Slack, artifact, visual, opportunity, or background outbox is written. Policy activation and explicit
per-child family opt-in are separate gates, and opt-out, collection pause, evidence/projection change, policy disable,
or model deletion revokes the derived state. Approved calibration is frozen before activation. Startup and each live
authorization recompute or revalidate the evaluation, full independent Lab review, promoted policy events, and exact
parent control-event epoch. Managed-backup conflict checks include all four child-scoped research tables. See
`docs/RESEARCH_TRACK.md`.

## Storage

Family profiles, events, evidence, curriculum source files, extracted text, indexes, eval reports, and artifacts are stored under `private/` by default. The repository ignores that entire directory. `.env`, `data/`, and `output/` are also ignored.

`curiosity backup create` copies family data to an owner-only directory outside the repository and records checksums. It includes only allowlisted non-secret setup files; Slack and model credential files are excluded and must be re-entered after recovery. A registered backup root also holds a required hash-only deletion manifest for raw events, transport events, learning sources and expiry, and whole-model deletions. Losing that manifest is authority damage and fails closed even when the active database is gone. Snapshots, registered restores, and automatic migration copies containing subsequently deleted data are purged; restore refuses a stale snapshot that reappears under that managed root and imports external internal-event, transport-event, and learning-source replay tombstones into an older valid image. Retained recovery copies receive future replay tombstones as deletions occur. Creation, restore, and deletion use shared process-safe lifecycle locks and reconcile committed tombstones before publication. Required source and restore registries are initialized with the root; losing either is authority damage, never an empty list. Restore staging and destination paths are registered before publication, and an activated disaster restore inherits the originating backup root, so process death or loss of the old `private/` directory does not sever future deletion propagation. Every family database has a stable random identity copied into managed images and snapshot manifests. Before deletion commits, the engine validates that identity and SQLite integrity across every managed migration copy, snapshot, partial snapshot, and non-empty restore, revalidates reciprocal database/root authority, and authenticates every registered root and restore marker. Mutable restores are read with WAL-aware semantics, and the active database path is never purged as a recovery copy. Missing, unsafe, unreadable, corrupt, mismatched, symlinked, or cross-family managed state stops the operation without changing live family rows. Version-1 roots are adopted only after all snapshots verify and their recorded source roots match the current family; because the old format lacks complete identity and deletion metadata, any legacy snapshot is conservatively purged once the root has a deletion record, including deletions made before adoption. A parent may explicitly resume collection after a whole-model reset; an append-only resume audit record proves that new epoch, so images created within it remain valid even after a later pause. Backup-root registration is database-local-first under the lifecycle lock. A durable root-side pending barrier is published before adoption makes a legacy root restore-capable, and restore refuses to cross it. If the process dies before reciprocal root metadata and deletions are synchronized, the next deletion completes that pending registration before preflight. Before snapshot copying the engine recovers committed file-deletion journals and excludes the internal quarantine. Copies moved elsewhere cannot be recalled by the engine. Keep the backup directory on an encrypted disk and include it in a retention process that honors family deletions.

Restore authorization also requires both a root identity and a non-empty source registry. Losing the pending barrier
during an incomplete registration therefore fails closed rather than turning an unsynchronized root into a valid one.
Every present registered source is identity-checked before its deletion records can affect the backup root.

Slack tokens live in owner-only `private/setup/slack.env`; non-secret private routes live in `private/setup/brain.json`; provider keys live in owner-only `private/setup/model.env`. Setup reports contain readiness categories only: never credential values, child names, questions, Slack IDs, provider keys, purchased-resource titles/URLs, or licensed excerpts.

Response PNGs use opaque filenames under `private/output/visuals/`, owner-only permissions, stripped metadata, bounded dimensions, and a recorded hash. Uploading a card sends that image, its caption, and alt text to Slack, where Slack retains it under the workspace's policies.

Researched public images are queried only with a lowercased public topic of at most 12 plain words. Queries containing known household identities, proper names copied from the source message, named schools/addresses, identifiers, file paths, URLs, or family, relationship, school, health, and location markers are rejected before any collection call. Candidate bytes are quarantined under `private/`, stripped of metadata, rights-checked, and vision-reviewed before use. The local cache stores provenance and the exact approved content hash; Slack receives only an approved image, its learning caption, alt text, and visible source credit.

The Slack app can read only files explicitly delivered to it in a paired DM or mention. Supported images are size-, type-, host-, hash-, and pixel-bounded before owner-only storage under `private/output/inbound/`. The configured vision provider inspects a photo once; the engine stores a bounded evidence summary locally for threaded follow-ups instead of resending the source image on every turn. Visible image text is treated as untrusted content, never as an instruction. A photo is not automatically child-interest or mastery evidence. Dismissing an unassigned photo removes its local copy when no other record references it.

The resource indexer refuses catalogs outside `private/` and requires a catalog declaration of:

- `access.scope = family_private`
- `access.redistribution_allowed = false`

The database stores absolute source paths because it is itself private. Public fixtures, logs, and release configs must never contain family identifiers or copyrighted excerpts.

## Retrieval and disclosure

Private resource search has two modes:

- `metadata_only` (default): collection/unit/document metadata and family-authored summaries only;
- `selected_excerpts`: up to a few bounded relevant excerpts after an explicit household opt-in, applied only when retrieval finds a relevant match.

When a provider backend is enabled, any opted-in excerpt in the context is disclosed to every configured provider used for that request. Request storage controls do not mean that no provider processing, safety retention, routing metadata, or downstream-provider handling occurs. Review current terms for direct providers and both OpenRouter plus its selected downstream endpoints.

The first `curiosity brain test --live` sends a fixed synthetic payload with no child, family, Slack, or resource content. Real context is sent only after that check and explicit family-lens/resource choices.

Deterministic response cards are rendered locally. Optional `decorative` mode sends a minimized scene prompt that may reflect the broad question topic, but code rejects known household identities, proper names, personal relationships, and school/home/health details. It never sends the family profile, licensed excerpts, source images, labels, measurements, or exact counts. Its explicit `curiosity visual test --live` probe is billable but synthetic.

## Epistemic separation

The engine records resource ownership as availability. Availability is not evidence that a child has seen, completed, understood, remembered, or enjoyed material.

A single observation may create an observation or hypothesis. `established_pattern` claims require support from at least two eligible learning episodes; two messages or events from one retry/repair episode are not enough. Contradicting evidence and parent corrections are retained.

Schema 18 adds purpose-specific retention classes for observed learning opportunities: `transient`, `learning_record`,
and `parent_confirmed`. Transient records expire after 30 days, situational interest evidence after 90 days, and explicit
parent confirmation has no automatic expiry. Concepts, tasks, rubrics, and plans are versioned definitions rather than
inferred child state. Child-bound evidence retains source type, assistance, task/episode/context lineage, per-concept
rubric results, and rule version; model annotations are 30-day `audit_only` records and generated output is
projection-ineligible. Learner snapshots contain IDs, bounded labels, categorical states, counts, uncertainty, and
short rationales—never raw evidence text.

An evidence correction is append-only and causes an in-transaction full rebuild. Deleting a Phase 1 learning source
physically cascades its compiled evidence and links, cancels its immutable plan, rebuilds projections, and retains only
append-only content-free corrections plus a tombstone with a one-way target hash and categorical deletion reason. The
Phase 1 source API deliberately does not claim raw-event erasure; the separate confirmed raw-event control removes the
local source and directly linked copies. Neither control can remove material already delivered to an external provider.
Learning-source deletion is global by one-way source-ID hash rather than child-scoped. The database verifies the exact
SHA-256 derivation and rejects reuse under any child even if a direct caller supplies a forged hash. Raw-event deletion
also follows a source event through its learning plan and the recursive transfer-plan chain to indirect opportunities.

Schema 21 adds the local open-family-model control plane, schema 22 makes deleted-event replay tombstones global and
non-cascading, and schema 23 adds child-bound atomic produced-event lineage, interaction lineage,
binding-independent transport tombstones, deletion reservations, and a durable private-file deletion journal. Parents can inspect child-scoped evidence and rationales,
pause new learner-evidence collection, correct or restore evidence, choose an earlier retention deadline, export a
child-scoped JSON record, and delete a learning source, its raw event, or the whole learner model with explicit
confirmation. A pause suppresses new derived profile records while preserving operational request/answer history under
its separate retention contract. Raw-event deletion removes directly linked local database, transport/outbox,
artifact/visual, interaction, parent-agent, and exclusively owned private-file copies and creates global internal and transport replay tombstones;
startup safely completes or rolls back interrupted file erasure, while already delivered provider copies remain
governed by that provider. New output is blocked after deletion is reserved, shared files remain for other owning
events, and managed backup roots receive typed monotonic hash-only deletion manifests. Full learner-model deletion erases both Phase 1 and legacy inferred state, pauses collection, and
retains raw question/transport history and explicit family-authored support preferences under their separately disclosed retention contracts. The optional child co-review
stores `review_only` responses that never enter evidence or recommendations. See `OPEN_FAMILY_MODEL.md`.

Schema 24 keeps adaptive-harness data in a separate control plane. Parent memory is explicit, scoped, versioned, and
immediately suppressible; forgetting or undoing a save scrubs removed values, provenance, save reasons, and
caller-authored audit text, retaining only content-free hashes and code-owned tombstone reasons. Typed memory
tombstones join the monotonic deletion manifest, purge stale snapshots/migration copies/managed restores, and are
enforced during restore and startup recovery. Legacy preference source-and-reply spans are excluded from child-event
context, and clearing a legacy category nulls its historical set values.
Natural memory inspection returns values only in the immediate parent-facing response. Its local thread record is
value-free, the child-event history boundary masks inspection records, and the Slack outbox records exact typed
key/version/hash lineage only while its payload awaits delivery. A confirmed or ambiguous send redacts the local
payload; erasure, expiry, or exhausted attempts redact and fence an unsent payload. Erasure also scrubs value-bearing
history and authenticated inspection outbox rows created by older versions before registered backups or restores can
survive reconciliation. Readiness and the atomic send claim recheck memory expiry, tombstones, exact hashes, and scope
switches, plus the active binding and parent principal, so a restored or already-readied queue cannot send disabled or
revoked guidance. Revoking a binding proactively fences its pending deliveries, and enqueue rechecks authority if
revocation occurs while a response is being prepared. Legacy cleanup requires a durable code-owned inspection record
matched to the same session, binding, thread, and time window; copied prefix text is not provenance and cannot cause an
unrelated delivery to expire.
This authorization fence applies to text, visual, printable-artifact, and parent-confirmed background outboxes.
Revocation cancels pending background confirmations rather than allowing them to become sendable after re-pairing.
Slack channel targets must match their binding at enqueue, readiness, claim, and worker preflight. File and text
workers use terminal compare-and-set states, preventing late success or failure callbacks from undoing revocation.
Every visual and printable attempt has an unguessable hashed lease that recovery invalidates before another claim.
Only pre-ticket failures can retry, under a three-attempt cap and backoff; possible post-ticket or post-byte effects
are recorded as terminal `unknown` rather than risking duplicate disclosure.
Slack may retain a message it already accepted under the workspace's policy. Deleting a raw source event similarly
scrubs and tombstones its sourced memory versions while retaining only one-way hash lineage for later governed runs;
undo cannot reactivate a source-deleted or otherwise tombstoned version.
Only parent-authored accessibility support may use child scope. Governed artifact runs persist the exact memory
key/version/hash and legacy preference event/hash they consumed, never the preference value. Structured compaction persists memory/output identifiers
and decision metadata, not memory values, artifact contents, child traits, or learner evidence. Artifact generation
resolves relevant preference values and active recipe text only as transient runtime modifiers; durable generation
lineage stores their governed identifiers and hashes rather than copying the raw control text. Private recipe text and
trusted public-fixture evaluation records remain in the local family database and cannot introduce connector or tool
authority. When a human runs recipe evaluation, the exact private recipe text and public synthetic cases are sent to
the configured behavior and judge model providers; no child context or family evidence is included. Background state
stores bounded identifiers rather than raw prompts; delivery accepts a capability-owned release-unit identifier only,
remains parent-only and unsendable until the bound parent confirms it, and never automatically retries an ambiguous
stale send.
Automatic printables without authenticated parent-session lineage fail closed instead of using an ungoverned direct
persistence path.
See `ADAPTATION_CONTROL_PLANE.md`.

## Network and actions

The web app binds to loopback and rejects non-loopback clients. Mutating forms use a per-process CSRF token and responses are `no-store` with a restrictive Content Security Policy.

The Slack connector uses an outbound Socket Mode connection. It accepts only direct messages or explicit mentions and then requires a local pairing for the exact Slack workspace, adult user, and conversation. The app manifest requests no user scopes and no ambient channel-history scopes. It requests `files:read` only for photos explicitly sent to the paired bot. Slack still processes the messages, files, and replies sent through the Slack service; do not treat Slack itself as local storage.

New households never opt into licensed-resource excerpts automatically. An owner may enable the setting locally with `curiosity resource mode --mode selected_excerpts`. A future transport must preserve the same authorization and disclosure behavior.

Model output cannot print, purchase, share data, or alter a release. The only model-originating MVP action is `propose_artifact`. Creation is parent-selected; printing additionally requires validation, exact-byte approval, and an explicit live-send flag.

The Open-Source Steward may derive a reusable pattern from family use, but it may not copy private evidence into a public contribution. Public regressions use synthetic families and generic resources. Publishing, pushing, opening a pull request, filing an issue, or contacting anyone requires explicit authorization.

## Before open-sourcing

Run the completion audit command documented in `OPERATIONS.md`. Review ignored/untracked state with Git in the eventual repository. Never force-add `private/`, `.env`, `data/`, `output/`, or `.Codex/`.
