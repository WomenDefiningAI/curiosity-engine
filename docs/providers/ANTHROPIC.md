# Anthropic provider

Maturity: **beta**. The native request shape and schema handling have deterministic tests, but the public project has not recorded a credentialed live contract probe. Each family must run `curiosity brain test --live` before use.

The native Anthropic adapter uses the Messages API and structured outputs. Claude accepts image inputs, and its PDF support analyzes extracted text together with page images; see [Messages](https://platform.claude.com/docs/en/api/messages/create), [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), and [PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support).

Choose current model IDs from the official Claude model documentation. Configure a separate OpenAI or OpenRouter `image_generation` route because Curiosity Engine does not assume that the native Anthropic API supplies that role.

Paste `ANTHROPIC_API_KEY`—and the second provider key when applicable—directly into ignored `private/setup/model.env`. The harness uses native Anthropic request shapes, not an OpenAI compatibility layer, and validates every structured result locally.
