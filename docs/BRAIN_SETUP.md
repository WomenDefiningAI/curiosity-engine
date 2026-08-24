# Set up the LLM brain

Slack proves that messages can reach the local harness. It does not make the harness intelligent. The brain step deliberately comes after the fixed Slack `connection` response so a family can debug transport and model problems separately.

## What the brain is

Curiosity Engine itself remains the agent. Local code selects bounded context, routes roles, validates schemas, runs critics, persists evidence, and controls every side effect. A model provider supplies semantic reasoning, visual understanding, OCR/document interpretation, or image generation.

The recommended runtime is a provider API because it gives the harness an explicit request contract, model ID, schema, timeout/error boundary, and usage account. Codex and Claude Code are the setup/customization agents; using an interactive coding-agent subscription as the live brain is an [experimental attended option](CODING_AGENT_RUNTIME.md).

## Choose a brain stack, not just a chat model

Elementary-family work is multimodal. The configured stack must include:

1. **Reasoning:** strict structured output and strong elementary factual/pedagogical behavior.
2. **Vision/OCR:** high-resolution image input, scanned worksheet/PDF understanding, layout/tables, and structured extraction.
3. **Visual QA:** inspect the rendered printable, including small text and misleading diagrams.
4. **Image generation/editing:** child-appropriate illustrative output and reliable instruction following.
5. **Freshness when needed:** web search with usable sources for current records, products, events, or superlatives.

These may be different models or providers. Image generation is a separate role because a strong reasoning/vision model does not necessarily generate images.

Generated imagery is never trusted for instructional text, exact counts, number lines, maps, anatomy, or scientific geometry. Code renders knowledge-bearing structure and all readable worksheet text; the visual model adds wonder and the visual-QA route checks the finished page.

## Provider choices

| Choice | Adapter maturity | Good fit | Important constraint |
|---|---|---|---|
| OpenAI direct | Family-evaluating | One account can provide structured reasoning, vision, hosted search, and a separate image model | Still configure and evaluate reasoning and image roles independently |
| Anthropic direct | Beta; synthetic contract-tested | Native structured Messages plus strong image/PDF understanding | Run the explicit live probe; Curiosity Engine does not assume an Anthropic image-generation endpoint, so pair it with an OpenAI or OpenRouter image route |
| OpenRouter | Beta; synthetic contract-tested | One gateway can expose many text, vision, OCR/PDF, and image-output models | Run the explicit live probe; pin models, require parameters, disable implicit fallback, and review router plus downstream privacy |

Current source documentation: [OpenAI models and modalities](https://developers.openai.com/api/docs/models), [Claude PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support), [OpenRouter multimodal overview](https://openrouter.ai/docs/guides/overview/multimodal/overview), and [OpenRouter image models](https://openrouter.ai/docs/guides/overview/multimodal/image-generation).

## Provisional reference setup

The project has not yet promoted a family-tested champion. The OpenAI starting candidate is currently:

```text
reasoning / critics: gpt-5.6-terra
vision / OCR / visual QA: gpt-5.6-terra
image generation: gpt-image-2
web grounding: enabled for workflows whose facts may have changed
status: family_evaluating
```

This is a candidate because current OpenAI documentation describes the latest text family as vision-capable and Terra as its balanced model, while GPT Image is the specialized image family. It becomes a Curiosity Engine recommendation only after repeated worksheet/OCR, printable-visual, first-grade factuality, latency, and cost evals.

## Configure

Install the provider clients:

```bash
python -m pip install -e '.[dev,slack,all-providers]'
```

OpenAI candidate:

```bash
curiosity brain configure \
  --provider openai \
  --model gpt-5.6-terra \
  --vision-model gpt-5.6-terra \
  --image-provider openai \
  --image-model gpt-image-2 \
  --web-search \
  --recommendation-status family_evaluating
```

Native Anthropic plus a separate image provider:

```bash
curiosity brain configure \
  --provider anthropic \
  --model YOUR_CLAUDE_MODEL_ID \
  --vision-model YOUR_CLAUDE_MODEL_ID \
  --image-provider openai \
  --image-model YOUR_OPENAI_IMAGE_MODEL_ID \
  --web-search
```

OpenRouter:

```bash
curiosity brain configure \
  --provider openrouter \
  --model AUTHOR/STRUCTURED_VISION_MODEL \
  --vision-model AUTHOR/STRUCTURED_VISION_MODEL \
  --image-provider openrouter \
  --image-model AUTHOR/IMAGE_MODEL \
  --web-search
```

The command writes non-secret choices to ignored `private/setup/brain.json` and creates an owner-only `private/setup/model.env` template. Paste keys into that file directly in VS Code. Do not paste them into coding-agent chat.

```bash
chmod 600 private/setup/brain.json private/setup/model.env
curiosity brain doctor
```

`brain doctor` is offline. It checks private file permissions, required routes, declared capabilities, and credential presence without displaying key values.

## Run the first live probe

Review the chosen provider/model and possible cost, then explicitly run:

```bash
curiosity brain test --live
```

This first request contains a fixed synthetic marker and no child name, family context, Slack message, or purchased resource. It proves the reasoning route can return the required structured schema. It does not yet prove worksheet OCR, image quality, or educational usefulness.

Do not describe a beta adapter as live-verified for the project merely because its mocked request-shape tests pass. Record live results only in ignored family/local reports until a privacy-safe public provider contract test is designed.

## Multimodal acceptance checklist

Before labeling a route `family_recommended`, evaluate it on de-identified or synthetic material:

- a clean printed first-grade worksheet;
- a phone photo with skew, shadows, and small text;
- a mixed text/picture worksheet and a simple table;
- a handwritten response sample with no real child name;
- a one-page printable checked for clipping, legibility, exact counts, and misleading geometry;
- an illustration request plus edit request using a generic reference image;
- current factual questions that require web grounding and sources;
- several real family questions reviewed for factuality, grade fit, curiosity value, effort, latency, and cost.

Model support is never inferred from a familiar family name alone. OpenRouter in particular exposes capability fields per model/endpoint; configure strict structured output and fail rather than silently dropping parameters. The adapter also denies data-collecting routes, requests ZDR, disables implicit provider fallback, and locally validates every result.

## Readiness meanings

- `brain_configured`: reasoning, vision/OCR, visual-QA, and image-generation routes plus credentials are present.
- `brain_verified`: the family-data-free structured reasoning probe passed for the current private configuration.
- `family_lens_ready`: the parent accepted or customized pedagogy and practical constraints.
- `answer_ready`: compatibility field meaning Slack is paired and the reasoning probe passed.
- `end_to_end_ready`: Slack delivery, brain probe, family lens, and one parent-approved real answer all passed.

Run `curiosity doctor` for the exact next action.
