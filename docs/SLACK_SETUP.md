# Slack setup for a family

This guide proves only the messaging layer. Do not configure an LLM until the fixed `connection` response succeeds. Keeping those checks separate makes failures understandable.

Slack is the only v1 messaging interface. The local connector uses outbound Socket Mode, so no public web server is required.

## 1. Create or choose the family workspace

If the family already has a workspace, continue to step 2. Otherwise use Slack's official [workspace guide](https://slack.com/help/articles/206845317-Create-a-Slack-workspace-Create-a-Slack-workspace) with an adult-controlled email and a family-neutral workspace name. Children do not need accounts.

A private channel such as `#family-curiosity` is optional; a DM to the app is the simplest start. Slack's free plan may limit message/file history, so Slack is an interaction surface—not the durable family record. The local ignored SQLite database is the durable record.

## 2. Install local Slack support

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,slack]'
curiosity doctor
```

## 3. Create the app from the public manifest

1. Open [Your Slack Apps](https://api.slack.com/apps) and select **Create New App**.
2. Choose **From an app manifest** and select the family workspace.
3. Choose YAML and paste `integrations/slack/manifest.yaml`.
4. Review and create the app.

The manifest asks only for:

- `chat:write` to reply;
- `im:history` to receive app DMs;
- `app_mentions:read` to receive explicit mentions.

It subscribes only to `message.im` and `app_mention`. It does not request ambient channel history, files, users, or user-token scopes.

## 4. Create and install the two credentials

These are different tokens:

1. Under **Basic Information → App-Level Tokens**, create `curiosity-socket` with `connections:write`. Copy the `xapp-...` token.
2. Under **OAuth & Permissions**, select **Install to Workspace**, approve the bot scopes, and copy the `xoxb-...` Bot User OAuth Token.

Installing through OAuth is required; creating the app alone is not enough.

Never paste tokens into coding-agent chat, a README, issue, or tracked file. In VS Code, create `private/setup/slack.env` from `integrations/slack/slack.env.example`, paste both values directly, and run:

```bash
chmod 600 private/setup/slack.env
curiosity doctor
```

The report validates only presence, prefixes, and permissions. It never prints token values.

## 5. Pair one exact adult and conversation

Configure the household first if necessary:

```bash
curiosity setup --owner-name "YOUR DISPLAY NAME" --timezone "America/New_York"
curiosity slack pair-code
curiosity slack run
```

Leave the connector running. In Slack, DM the app with `pair CODE`. For a family channel, invite the app, explicitly mention it with the pair command, and pair that channel separately.

The code expires after 15 minutes and is single-use. Pairing authorizes one exact workspace, adult Slack user, and conversation. Each co-parent needs a local parent record and their own pairing.

Checkpoint:

```bash
curiosity slack bindings
curiosity doctor
```

`slack_ready` should now be true. Do not share raw binding output publicly.

## 6. Prove transport without a model

In the paired Slack conversation, send:

```text
connection
```

Expected response:

> Slack connection works. This fixed response did not contact an AI model, load a child profile, or read family resources. Delivery confirmation is being recorded locally.

The handler runs before model/service construction. The checkpoint passes only after Slack confirms delivery.

```bash
curiosity doctor
```

`transport_verified` should be true. If not, debug the Slack tokens, app installation, pairing, connector process, and delivery—not model prompts or API keys.

## 7. Hand off to brain setup

Messaging setup is now finished. Continue with [Brain setup](BRAIN_SETUP.md). The next live model test is synthetic and family-data-free; a real child question comes only after the brain and family lens are configured.

## Running Slack later

```bash
source .venv/bin/activate
curiosity slack run
```

Leave the process running. Control-C, closing the terminal, logging out of the computer, sleeping/shutting down the computer, or losing network connectivity stops replies. Background startup helpers are a later operational feature.

## Troubleshooting

- **No response:** confirm `curiosity slack run` is still running; then run `curiosity doctor`.
- **App exists but cannot post:** install/reinstall it from **OAuth & Permissions** and update the local bot token.
- **Channel mention ignored:** invite the app, mention it explicitly, and pair that adult/channel combination.
- **Pair rejected:** generate a new code; codes expire and are single-use.
- **Token error:** rotate/reinstall in Slack and replace only ignored `private/setup/slack.env`.
- **Remove access:** inspect `curiosity slack bindings`, revoke with `curiosity slack revoke --binding ID`, and rotate tokens if compromise is possible.
