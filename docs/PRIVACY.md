# Privacy and licensed-resource boundary

## Storage

Family profiles, events, evidence, curriculum source files, extracted text, indexes, eval reports, and artifacts are stored under `private/` by default. The repository ignores that entire directory. `.env`, `data/`, and `output/` are also ignored.

Slack tokens live in owner-only `private/setup/slack.env`; an optional model key lives in owner-only `private/setup/model.env`. Setup reports contain readiness categories only: never credential values, child names, questions, Slack IDs, or licensed excerpts.

The resource indexer refuses catalogs outside `private/` and requires a catalog declaration of:

- `access.scope = family_private`
- `access.redistribution_allowed = false`

The database stores absolute source paths because it is itself private. Public fixtures, logs, and release configs must never contain family identifiers or copyrighted excerpts.

## Retrieval and disclosure

Private resource search has two modes:

- `metadata_only` (default): collection/unit/document metadata and family-authored summaries only;
- `selected_excerpts`: up to a few bounded relevant excerpts after an explicit household opt-in, applied only when retrieval finds a relevant match.

When a provider backend is enabled, any opted-in excerpt in the context is disclosed to that provider. Responses API storage is disabled, but provider processing still occurs. The UI names this choice and leaves it unchecked.

## Epistemic separation

The engine records resource ownership as availability. Availability is not evidence that a child has seen, completed, understood, remembered, or enjoyed material.

A single observation may create an observation or hypothesis. `established_pattern` claims require at least two evidence records attributable to two different events. Contradicting evidence is retained.

## Network and actions

The web app binds to loopback and rejects non-loopback clients. Mutating forms use a per-process CSRF token and responses are `no-store` with a restrictive Content Security Policy.

The Slack connector uses an outbound Socket Mode connection. It accepts only direct messages or explicit mentions and then requires a local pairing for the exact Slack workspace, adult user, and conversation. The app manifest requests no user scopes and no ambient channel-history scopes. Slack still processes the messages and replies sent through the Slack service; do not treat Slack itself as local storage.

New households never opt into licensed-resource excerpts automatically. An owner may enable the setting locally with `curiosity resource mode --mode selected_excerpts`. A future transport must preserve the same authorization and disclosure behavior.

Model output cannot print, purchase, share data, or alter a release. The only model-originating MVP action is `propose_artifact`. Creation is parent-selected; printing additionally requires validation, exact-byte approval, and an explicit live-send flag.

## Before open-sourcing

Run the completion audit command documented in `OPERATIONS.md`. Review ignored/untracked state with Git in the eventual repository. Never force-add `private/`, `.env`, `data/`, `output/`, or `.Codex/`.
