# Slack setup

Slack is the v1 parent interface. This guide proves messaging with a fixed `connection` reply before any model is configured. The connector uses outbound Socket Mode, so no public server is needed.

## 1. Prepare the workspace

Use an existing adult-controlled workspace or follow Slack’s [workspace guide](https://slack.com/help/articles/206845317-Create-a-Slack-workspace-Create-a-Slack-workspace). A free workspace is enough; children do not need accounts.

Slack is an interaction surface, not the durable record. Messages sent through it are processed by Slack; family state remains in the ignored local database.

## 2. Create the app

1. Open [Your Slack Apps](https://api.slack.com/apps) and choose **Create New App**.
2. Select **From an app manifest** and the family workspace.
3. Paste `integrations/slack/manifest.yaml` as YAML and create the app.

The manifest requests `chat:write`, `files:write`, `im:history`, and `app_mentions:read`, subscribes only to `message.im` and `app_mention`, and enables Block Kit interactions for parent controls. It does not request file reads, ambient channel history, users, or user-token scopes. `files:write` is used only for response visuals.

## 3. Install and save both tokens

1. Under **Basic Information → App-Level Tokens**, create a token with `connections:write`; copy the `xapp-...` value.
2. Under **OAuth & Permissions**, choose **Install to Workspace**; copy the `xoxb-...` Bot User OAuth Token.

Creating the app is not enough: the OAuth installation is required.

If the app was installed before visual responses were added, add `files:write` under **OAuth & Permissions → Bot Token Scopes**, then choose **Reinstall to Workspace**. Reinstallation can change the bot token; update the ignored token file if needed.

If the app was installed before interactive inbox controls were added, open **Interactivity & Shortcuts**, turn **Interactivity** on, and save. Socket Mode carries button and menu actions, so no Request URL is needed.

Copy `integrations/slack/slack.env.example` to ignored `private/setup/slack.env`, then paste both values there directly in VS Code. Never paste them into agent chat.

```bash
chmod 600 private/setup/slack.env
curiosity doctor
```

## 4. Pair an adult and conversation

```bash
curiosity slack pair-code
curiosity slack run
```

Leave the connector running. In Slack:

- DM the app with `pair CODE`; or
- invite it to a family channel, then explicitly mention it with `pair CODE`.

Codes are single-use and expire after 15 minutes. Pairing binds one exact workspace, adult, and conversation. Pair each co-parent/conversation separately.

```bash
curiosity slack bindings
curiosity doctor
```

Do not share raw binding output publicly.

## 5. Prove the connection

In the paired conversation, send:

```text
connection
```

The reply states that Slack works without contacting a model, loading a child profile, or reading resources. `curiosity doctor` should then report `transport_verified: true`.

Next send:

```text
visual connection
```

This creates a fixed card locally and uploads it with alt text. It uses no model, child profile, family message, or resource. `curiosity doctor` should report `visual_delivery_verified: true`.

Every unassigned note includes a child dropdown and Dismiss button. Completed learning threads include Helpful, Not for us, and Try another controls. These actions remain restricted to the paired parent and conversation; retries stay in the same learning episode instead of becoming new interest evidence.

Continue with [brain setup](BRAIN_SETUP.md).

## Restart and troubleshoot

```bash
source .venv/bin/activate
curiosity slack run
```

No reply usually means the connector stopped. A text reply without its card usually means `files:write` was not authorized or the visual worker failed; run `curiosity doctor`. Also check app installation, current tokens, pairing, channel invitation, and explicit mention. Revoke access with `curiosity slack revoke --binding ID`; rotate both tokens if exposure is possible.
