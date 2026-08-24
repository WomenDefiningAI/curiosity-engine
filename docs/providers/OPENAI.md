# OpenAI provider

OpenAI can cover Curiosity Engine's structured reasoning, image input/vision, hosted web search, and a separate image-generation role. Current model IDs and availability change; use the [official model catalog](https://developers.openai.com/api/docs/models) and [image-generation documentation](https://developers.openai.com/api/docs/guides/image-generation) during setup.

The current provisional family-eval candidate is `gpt-5.6-terra` for reasoning/vision and `gpt-image-2` for image generation. It is not yet the project champion.

Create the private selection with `curiosity brain configure`, paste `OPENAI_API_KEY` into ignored `private/setup/model.env`, and run `curiosity brain test --live`. The adapter uses the Responses API, disables response storage for its request, bounds hosted search, and validates the result locally. Storage controls do not replace reading the provider's current data terms.
