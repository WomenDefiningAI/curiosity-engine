# LLM brain setup

Slack is only the transport. The brain is one or more model APIs; local code still controls context, schemas, critics, retries, storage, and side effects.

Provider APIs are the supported runtime because they have explicit model IDs, request contracts, timeouts, and billing. Coding-agent operation is an [experimental attended option](CODING_AGENT_RUNTIME.md).

## Required roles

| Role | Requirement |
|---|---|
| Reasoning | Structured output, factual and elementary-age reasoning |
| Vision/OCR | Worksheets, photos, PDFs, tables, and small labels |
| Visual QA | Detect clipping, illegibility, wrong counts, and misleading diagrams |
| Image generation | Child-appropriate illustrations and edits |

Web grounding is also needed for facts that may have changed. Different roles may use different providers. Generated images must never be the source of truth for instructional text, exact counts, maps, anatomy, number lines, or scientific geometry.

## Choose a provider

| Provider | Fit | Constraint |
|---|---|---|
| OpenAI | Reasoning, vision, search, and separate image models | Evaluate each route independently |
| Anthropic | Reasoning plus image/PDF understanding | Pair with another image-generation route |
| OpenRouter | Flexible access to many routes | Pin capabilities and review both router and downstream privacy |

The project has no family-tested champion yet. The current OpenAI starting candidate is `gpt-5.6-terra` for reasoning/vision and `gpt-image-2` for images, marked `family_evaluating`. Check the provider’s current model catalog before choosing.

## Configure

OpenAI example:

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

For Anthropic or OpenRouter, use the same command with the chosen provider/model IDs. Provider-specific notes: [OpenAI](providers/OPENAI.md), [Anthropic](providers/ANTHROPIC.md), [OpenRouter](providers/OPENROUTER.md).

The command writes ignored `private/setup/brain.json` and creates `private/setup/model.env`. Paste keys directly into the latter, never into agent chat.

```bash
chmod 600 private/setup/brain.json private/setup/model.env
curiosity brain doctor
curiosity brain test --live
```

`brain doctor` is offline and redacted. The explicit live probe may be billable, uses a fixed synthetic payload, and tests only structured reasoning—not OCR, images, or family usefulness.

Visual response cards default to deterministic local rendering. To enable optional decorative generation and test the exact image route:

```bash
curiosity visual mode --mode decorative
curiosity visual test --live
```

The paid test sends one fixed, family-free scene prompt and stores the result under ignored private output. OpenAI is the maintained image-delivery adapter in this release; another configured image provider is reported as not runtime-ready instead of silently falling back to the reasoning model.

## Acceptance before recommendation

Use only synthetic or de-identified material to check:

- clean and phone-photographed elementary worksheets;
- small text, tables, mixed text/pictures, and generic handwriting;
- one-page print legibility, exact counts, and visual geometry;
- illustration generation plus editing;
- current factual questions with sources;
- repeated parent reviews for grade fit, usefulness, latency, and cost.

`brain_verified` means only that the current reasoning route passed its synthetic probe. `visual_delivery_verified` proves Slack accepted a synthetic card; `image_generation_verified` proves the current paid image route when decorative mode is enabled. `end_to_end_ready` additionally requires the configured visual mode, family lens, and a parent-approved real answer. Run `curiosity doctor` for the next action.
