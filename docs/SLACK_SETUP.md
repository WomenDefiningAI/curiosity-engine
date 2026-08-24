# Slack setup for a family

This guide starts at zero: first create a free family workspace, then create the Curiosity Engine Slack app. Slack is the first supported messaging interface; no public web server is required because the connector uses Socket Mode.

## 1. Create the family Slack workspace

If your family already has a workspace, skip to step 2.

1. Open Slack's official [create-a-workspace guide](https://slack.com/help/articles/206845317-Create-a-Slack-workspace-Create-a-Slack-workspace) and choose **Create a new workspace**.
2. Use an adult-controlled email address and a family-neutral workspace name.
3. Invite a co-parent if wanted. Children do not need Slack accounts for this MVP.
4. Optionally create a private channel such as `#family-curiosity`. DMs to the bot are the simplest starting point.

Slack's free plan has message/file history limits. Treat Slack as the interaction surface, not the family record; Curiosity Engine's durable record is the local ignored database. Review Slack's current [plans and features](https://slack.com/help/articles/115003205446-Slack-plans-and-features) and [free-workspace usage limits](https://slack.com/help/articles/115002422943-Usage-limits-for-free-workspaces).

## 2. Install local Slack support

From the repository terminal:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,slack]'
curiosity doctor
```

The Slack dependency may still read `not_configured` until installation finishes. Tokens should be absent at this point.

## 3. Create the Slack app from the checked-in manifest

1. Go to [Your Apps](https://api.slack.com/apps) and choose **Create New App**.
2. Choose **From an app manifest**, then choose the family workspace.
3. Select YAML and paste the contents of `integrations/slack/manifest.yaml`.
4. Review and create the app.

The manifest requests only these bot scopes:

- `chat:write` to answer;
- `im:history` to receive direct messages;
- `app_mentions:read` to receive explicit mentions.

It subscribes only to `message.im` and `app_mention`. It does not request user scopes or ambient public/private channel history.

## 4. Create the two tokens

These are two different credentials:

1. In **Basic Information → App-Level Tokens**, create a token named `curiosity-socket` with only `connections:write`. This is the `xapp-...` app token used by [Socket Mode](https://docs.slack.dev/tools/bolt-python/concepts/socket-mode/).
2. In **OAuth & Permissions**, choose **Install to Workspace**, approve the three bot scopes, and copy the **Bot User OAuth Token**. This is the `xoxb-...` bot token.

Do not paste either token into a coding-agent chat, README, issue, or tracked file.

## 5. Store tokens in the ignored local file

In VS Code:

1. Create `private/setup/slack.env`.
2. Copy the two lines from `integrations/slack/slack.env.example`.
3. Replace the placeholders with the tokens and save.
4. In the terminal, protect the file:

```bash
chmod 600 private/setup/slack.env
curiosity doctor
```

The doctor reports only whether the prefixes and permissions look right. It never displays token values. Environment variables with the same names also work and take precedence over the file.

## 6. Configure question-specific reasoning

Slack can now reach the local harness, but the safe default backend produces only labeled offline demo responses. Before using the bot as the family's primary question interface, create an [OpenAI API key](https://platform.openai.com/api-keys), then:

1. Copy `integrations/openai/model.env.example` to `private/setup/model.env`.
2. Paste the key directly into that ignored file; never paste it into agent chat.
3. Protect and verify the file:

```bash
chmod 600 private/setup/model.env
curiosity doctor
```

`answer_ready` becomes true only after Slack is paired as well. The provider processes the bounded request context, while the durable database remains local. Responses API storage is disabled by the harness.

Purchased-resource excerpts are independently off for every new household. If the owner explicitly allows small relevant excerpts to enter hosted-model requests, run locally:

```bash
curiosity resource mode --mode selected_excerpts
```

The response paraphrases useful context rather than posting source passages to Slack, and irrelevant units are not forced into an answer.

## 7. Pair the parent, not the whole workspace

First configure the household if you have not already:

```bash
curiosity setup --owner-name "YOUR DISPLAY NAME" --timezone "America/New_York"
```

Create a one-time code:

```bash
curiosity slack pair-code
```

Start the local connector in the terminal:

```bash
curiosity slack run
```

In Slack, open the Curiosity Engine app's **Messages** tab and send `pair CODE`, using the code printed locally. It expires after 15 minutes. Pairing authorizes only that exact workspace, Slack user, and DM. To use a private family channel, invite the app, mention it with the pair command, and create a separate pairing. Each co-parent gets a local parent record and their own code.

## 8. Verify the daily flow

Send:

```text
children
ask kid-a: Why do birds have different beaks?
```

Then try an unattributed note:

```text
We found a feather on our walk
```

The bot should save it without guessing a child and tell you how to assign or dismiss it. Finally run locally:

```bash
curiosity slack bindings
curiosity doctor --write-report
```

Transport setup is complete when `slack_ready` is true. Tailored daily use is complete when `answer_ready` is true and a paired Slack question receives a relevant, grade-appropriate response.

## Running it later

Whenever you want Slack replies, open the repository terminal and run:

```bash
source .venv/bin/activate
curiosity slack run
```

Leave that process running. Stop it with Control-C. Automatic startup on a family computer is a later, optional operational choice—not required for the MVP walkthrough.

## Troubleshooting

- **Bot does not answer:** confirm `curiosity slack run` is still running, then run `curiosity doctor`.
- **Not paired:** create a new code; codes are single-use and expire.
- **Channel mention is ignored:** invite the app, mention it explicitly, and pair that parent/channel combination.
- **`invalid_auth` or token error:** reinstall the Slack app if its scopes changed and replace only the local token file.
- **Concern about access:** run `curiosity slack bindings`, revoke an entry with `curiosity slack revoke --binding ID`, and rotate tokens in Slack if necessary.
