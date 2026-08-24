# Interaction model

## Why Slack first

The harness needs a place parents already notice in the middle of family life. Slack provides DMs, private family channels, explicit mentions, adult identity, and an outbound Socket Mode connection without requiring a public webhook. For this MVP, those advantages outweigh adding another custom daily interface.

Slack is still a transport, not the product's memory or reasoning layer. The local service remains authoritative. Telegram can later reuse the same normalized messages, pairing, receipts, inbox, service, and delivery state. WhatsApp is not an MVP target because its business-platform setup and hosted webhook requirements would make a zero-start family walkthrough substantially heavier.

## First-run surfaces

1. **Coding agent:** reads the repository contract, explains the system in parent language, runs deterministic setup, and pauses for Slack approvals and private values.
2. **Local CLI/console:** owns setup, privacy review, profiles, bindings, purchased-resource indexing, downloadable files, and recovery.
3. **Slack:** handles the small daily loop after an adult and conversation are paired.

This is why the project contains a local web page without trying to become another family app: it is the safe control room, while Slack is the front door.

## Daily loop

```text
parent question or note
        │
        ├─ explicit `ask CHILD_ID: ...` ─► answer now
        │
        └─ no child attribution ─────────► unassigned inbox
                                                   │
                                   parent assigns or dismisses
                                                   │
                                                   ▼
                                           answer after assignment
```

Responses are short parent prompts: a direct, age-calibrated hook, something to show or notice, a question to ask, one small explanation, and an optional activity using ordinary materials. Purchased-resource excerpts default off; a household opt-in permits only bounded relevant excerpts to enter the hosted-model request, and source passages are not reproduced in Slack.

## Authorization model

A Slack workspace membership does not grant access. A short-lived local code binds one parent principal to one exact workspace, Slack user, and conversation. A DM and a private family channel are separate bindings. Co-parents have separate local parent records and pairing codes. Bindings can be listed and revoked locally.

The bot listens only to DMs and explicit mentions. It does not ingest ambient workspace conversation.

## Proactive behavior

Weekly reflection is off by default. When a parent enables it, the core may retain at most one active low-effort family suggestion in a week, and doing nothing is valid. The first Slack MVP does not send autonomous campaigns; proactive Slack delivery needs separate feedback and quiet-hour product testing.
