# Adaptation control plane

Phase 4 adds durable adaptation without letting memory, generated procedure, or conversation continuity become learner
evidence. The central invariant is:

> Learning evidence, parent preferences, private procedures, and operational continuity are separate authority planes.

## Parent memory

`ParentMemoryLedger` stores explicit working preferences in three bounded scopes:

- `household` for general family working preferences;
- `thread` for one parent conversation; and
- `child_support` only for parent-authored accessibility support.

Every item records its scope, category, value, provenance, creator, parent principal where applicable, sensitivity,
status, review/expiry time, version, content hash, and reason for saving. Recall adds a code-owned `why_recalled` reason.
Parent-authored memory outranks an approved agent proposal. An agent proposal is limited to ordinary, non-child-specific
preferences and remains inactive until a parent approves it.

Remember and update operations return a visible confirmation and an event that can undo the most recent save. Disable
is reversible. Forget and undo scrub the removed version while retaining only content-free audit hashes and a typed
memory tombstone. That tombstone is propagated to every registered backup root; stale snapshots, managed restores, and
automatic migration copies are purged, and restore refuses an image that reintroduces the memory. The shared lifecycle
lock serializes artifact use, deletion, snapshot publication, and restore, while startup reconciles a deletion that
committed before backup cleanup completed. Undo never reactivates a version whose source was deleted or whose exact
key/version identity is tombstoned; if no eligible prior version remains, recall stays empty.
Household, thread, or child-support memory can be turned off immediately. The parent-agent re-fetches active memory on
each turn; compacted memory references never substitute for that read.

“What do you remember?” is a code-owned inspection route that returns the current effective ledger to the parent for
that response only. The durable thread row records typed references and a redaction notice, never the displayed values;
the ordinary Slack outbox binds every displayed item by exact memory key/version/hash while it is queued. After a
confirmed or ambiguous send, the local payload is replaced by the same redaction notice. Forget, undo, source-event
deletion, expiry, scope/item disable, or exhausted delivery attempts redact and fence any unsent copy. Readiness and
the atomic send claim both revalidate the exact reference, hash, expiry, tombstone, and scope switches, including after
a restore. Enqueue, readiness, binding revocation, and atomic claim also require a live binding and parent principal.
That rule covers text, visual, and printable-artifact outboxes independently, so a file that was ready before
revocation cannot be claimed afterward. Every target channel is matched to the binding rather than trusted from an
outbound payload. Claimed senders use compare-and-set stage transitions and stop when revocation wins the race, so an
already-running worker cannot resurrect an `unknown` or `expired` row. Visual and artifact claims also carry a unique
hashed lease through every stage; recovery clears it before retry, preventing a stale worker from advancing a later
attempt that happens to occupy the same status. Artifact readiness performs crash recovery itself. Both file channels
retry only bounded, backed-off failures before ticket acquisition; any ambiguity after a ticket or byte upload is
terminal `unknown` and is never sent again automatically.
Legacy untyped cleanup requires a durable code-owned inspection record matched to the same session, binding, thread,
and message-time window; text alone is never provenance. This local erasure cannot recall a message Slack already accepted, which remains
governed by the workspace's retention policy. The evidence-safe history boundary also masks inspection rows, and
erasure scrubs any legacy value-bearing history or authenticated inspection outbox row before backup reconciliation.
The older compatibility preference log is likewise
evidence-safe: its source turn and confirmation span are masked before child-event context is assembled, and clearing a
category scrubs every superseded legacy value.

Parent memory is an instruction about how to work. It never inserts an event, observation, claim, learning opportunity,
evidence record, or learner projection.

## Private family recipes

`FamilyRecipeService` versions private procedures through an explicit lifecycle:

`draft → linted → tested → shadow → approved → active`

Each transition is durable and auditable. Lint permits only the intersection of the global reviewed local/read tool
floor and the recipe's owning capability, and rejects shell, network, MCP, connector, or unknown-tool authority. Test
and shadow transitions require an immutable, exact-instruction-hash evaluation produced by the code-owned public
fixture runner. For each public synthetic case, a behavior route executes the reviewed baseline and exact candidate
through the real capability output contract and deterministic artifact validators. A
distinct judge route must observe a useful, code-confirmed output difference while rejecting safety regressions and
authority expansion. Both routes return typed records, every output is hashed, tool use is checked against both the
recipe and capability floors, and an evaluation fails if it changes learner evidence. The CLI does not accept
caller-authored pass reports. Approval and activation require a parent or human operator. A model cannot activate,
disable, retire, or roll back a recipe.

Every session-scoped artifact path—including an automatic functional printable from a reviewed answer—creates a
governed capability run; automatic printable creation without parent-session lineage fails closed and produces no
artifact. Before model work or output creation, the run binds the exact active recipe version and
instruction hash, each parent-memory key/version/content hash, and any effective legacy preference event/hash. The exact
recipe is then applied to the ordinary typed artifact contract; it does not replace the contract or its validators. A
deterministic backend that cannot apply an active recipe records a failed run and releases nothing. Relevant artifact,
visual, activity, and accessibility values are resolved ephemerally without copying raw values into events, evidence,
compactions, capability results, or release records. Activating a recipe disables any prior active recipe for that capability, and
rollback restores the most recent prior approved version in one operation. A recipe never adds a tool, host, connector,
scope, or permission.

## Structured thread compaction

Long parent threads retain a small raw tail plus an inspectable, deterministic structural summary. Its stored hash and
schema are checked before use. The summary contains:

- a source-message pointer for the last covered parent intent, never copied parent text;
- parent-memory keys and versions, not memory values;
- legacy preference-event identifiers;
- tool decisions and code-owned reasons;
- output identifiers and hashes;
- errors, pending work, and correction-event identifiers; and
- an explicit declaration that child traits, learner evidence, and artifact contents are prohibited.

The model must re-fetch code-owned memory and outputs from their source records. A compaction cannot create evidence,
permission, delivery authority, or a child inference. Stored summaries remain immutable audit history; immediately
before runtime use, their memory, tool-decision, output, legacy-preference, and correction references are resolved
against current source rows. Disabling or forgetting a preference and completing or failing pending work therefore
takes effect even when immutable audit history records the earlier status. Every repeated list is capped, and concurrent
compaction of the same boundary converges on one canonical record.
The private local dashboard exposes the immutable compaction record beside freshly resolved memory, decision, and
release-unit status so an operator can inspect continuity without treating the snapshot as current authority.

## Durable background work

`BackgroundRunService` persists each logical run, a canonical request hash, policy snapshot and hash, bounded
identifier-only input, logical workspace descriptor, source/capability lineage, recipe version, attempts, lease,
completion pointer, wake state, and delivery state. Reusing an idempotency key with different semantics fails closed;
overlap keys skip concurrent copies and persist a durable receipt, so retrying that skipped request after the original
finishes still returns the same logical outcome. Schedule catch-up materializes only the latest missed run with an existing
session and running capability-run lineage. Every attempt has an unguessable lease token, so a stale worker cannot
complete or fail work after reassignment. Stale execution leases are requeued with the interrupted attempt marked
abandoned; a bounded attempt limit fails closed.

A parent-agent turn can register before or after success or failure; a terminal run creates an immediately ready waiter,
and every waiter can still be resumed exactly once. Completion does not send a message.
Delivery accepts only the exact release-unit pointer owned by the completed capability run, is bound to the session's
active parent transport, and stays unsendable until that parent confirms it. Delivery attempts use the same fenced,
bounded attempt model, and claiming rechecks that the binding and parent principal are still active. Because a crashed
sender may have reached the provider before losing its lease, a stale sending attempt becomes `unknown` and is never
automatically retried. Completion uses a typed provider outcome: only a confirmed rejection can retry, while an
ambiguous result becomes terminal `unknown`. A parent must explicitly request and confirm a new delivery under a new idempotency key. This
is also true after transport revocation: every waiting or queued confirmation is cancelled, an in-flight attempt is
terminally unknown, and re-pairing the same Slack identity cannot revive either one. This
outbox is intentionally separate from learner evidence and from the ordinary transport outbox.

## Operator commands

Use `curiosity memory …` to remember, update, approve, forget, undo, inspect, or switch memory. Ordinary explicit thread
preference requests in parent chat use the same ledger. Use `curiosity recipe …` to draft and govern a recipe.
`recipe run-tests` and `recipe run-shadow` execute the pinned public fixture runner and record the resulting trusted
evaluation before advancing the exact version. They require configured behavior and independently identified judge
routes and may send the private recipe instructions, along with public synthetic cases only, to those configured model
providers. Configure a separately identified judge with `curiosity brain configure --judge-model ...` and, when
needed, `--judge-provider ...` and `--judge-reasoning-effort ...`. These commands expose authority to the local human
operator; they are not registered as family-runtime model tools.

## Verification and promotion

Schema 24 and `tests/test_phase4_adaptation_control.py` cover validated v23 migration, full-column forged-constraint
failure, memory separation/reversal/deletion, backup purge and post-commit recovery, transient natural inspection plus
inspection-to-child-context redaction, typed Slack outbox lineage and terminal redaction, source-memory deletion after
a later governed consumer, restored queue expiry/disable revalidation, no tombstoned-version resurrection through undo,
structurally authenticated legacy redaction without spoofed/unrelated-delivery effects, binding-revocation races,
actual-contract baseline/candidate recipe evaluation and rollback, exact adaptation lineage, automatic
printable governance and deterministic fail-closed behavior, installed fixture discovery, bounded concurrent compaction
with dashboard-visible live status refresh, exact session/event background lineage, attempt fencing, execution recovery,
ambiguity-safe delivery, durable overlap suppression, pre/post-completion one-shot wake, binding-owned release pointers,
and parent-confirmed delivery. `evals/adaptation/` is a
required offline Lab suite. Production promotion still requires all deterministic suites, live independent judgment,
and operator approval.
