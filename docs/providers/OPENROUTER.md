# OpenRouter provider

Maturity: **beta**. The native request, structured-output, web-search, and routing controls have deterministic tests, but the public project has not recorded a credentialed live contract probe. Each family must run `curiosity brain test --live` for the exact model and downstream route selected.

OpenRouter can expose text, image-input, PDF/OCR, and image-output models through one account. Model and endpoint capabilities vary, so choose explicit model slugs and inspect the current [model capabilities](https://openrouter.ai/docs/guides/overview/models), [multimodal input](https://openrouter.ai/docs/guides/overview/multimodal/overview), and [image model](https://openrouter.ai/docs/guides/overview/multimodal/image-generation) documentation.

The adapter uses OpenRouter Chat Completions rather than pretending it is OpenAI Responses. It requires requested parameters, disables implicit provider fallback, sets `data_collection: deny`, requests ZDR routes, and still validates the returned schema locally. These controls can make a route unavailable; failure is safer than silently selecting a less private or incompatible endpoint. Review both OpenRouter's policy and the selected downstream provider.

OpenRouter currently labels server tools such as web search as beta, so their request shape may change. The live probe and the project's provider-version review are the safeguards for detecting that drift before family use.

Paste `OPENROUTER_API_KEY` directly into ignored `private/setup/model.env`, run `curiosity brain doctor`, and explicitly run the synthetic live probe before family context is sent.
