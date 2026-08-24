from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel

from ..brain_config import secret_is_configured

OutputModel = TypeVar("OutputModel", bound=BaseModel)


def _raw(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _sources(message: Any) -> list[str]:
    urls: list[str] = []
    for annotation in _raw(message).get("annotations") or []:
        citation = _raw(annotation).get("url_citation") or {}
        url = str(citation.get("url") or "")
        if url.startswith(("https://", "http://")) and url not in urls:
            urls.append(url)
    return urls[:8]


class OpenRouterBackend:
    """OpenRouter Chat Completions adapter; intentionally separate from OpenAI Responses."""

    name = "openrouter"

    def __init__(
        self,
        model: str,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        max_output_tokens: int = 4_000,
        routes: dict[str, dict[str, Any]] | None = None,
    ):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - base dependency guards this
                raise RuntimeError("Install the 'openai' package to use OpenRouterBackend") from exc
            selected_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
            if not secret_is_configured("openrouter", selected_key):
                raise RuntimeError("OpenRouter mode is enabled but no valid API key is configured")
            client = OpenAI(api_key=selected_key, base_url="https://openrouter.ai/api/v1")
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.routes = routes or {}

    def complete(
        self,
        *,
        role: str,
        system: str,
        payload: dict[str, Any],
        response_model: type[OutputModel],
    ) -> dict[str, Any]:
        route = self.routes.get(role) or {}
        provider = route.get("provider")
        if provider and provider != "openrouter":
            raise ValueError(f"OpenRouterBackend cannot execute provider route {provider!r} for {role}")
        selected_model = str(route.get("model") or self.model)
        self.model = selected_model
        text_payload = {key: value for key, value in payload.items() if key != "image_data_urls"}
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": json.dumps(text_payload, ensure_ascii=False, default=str)}
        ]
        for image in payload.get("image_data_urls", []):
            user_content.append({"type": "image_url", "image_url": {"url": image, "detail": "high"}})
        extra_body: dict[str, Any] = {
            "provider": {
                "require_parameters": True,
                "allow_fallbacks": False,
                "data_collection": "deny",
                "zdr": True,
            }
        }
        allowed_tools = set((payload.get("policy") or {}).get("allowed_tools") or [])
        request: dict[str, Any] = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"curiosity_{role}".replace("-", "_")[:64],
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
            "max_tokens": self.max_output_tokens,
            "extra_body": extra_body,
        }
        if "web_search" in allowed_tools:
            request["tools"] = [
                {"type": "openrouter:web_search", "parameters": {"max_results": 5}}
            ]
        response = self.client.chat.completions.create(**request)
        choice = response.choices[0] if getattr(response, "choices", None) else None
        message = getattr(choice, "message", None)
        text = getattr(message, "content", None)
        if not text:
            raise RuntimeError("OpenRouter response did not contain structured output text")
        data = json.loads(text)
        if "resource_refs" in response_model.model_fields:
            existing = [str(item) for item in data.get("resource_refs") or []]
            data["resource_refs"] = list(dict.fromkeys([*existing, *_sources(message)]))[:8]
        # Contract validation belongs to ReasoningEngine, where a configured
        # repair round can correct the provider candidate before it reaches a
        # family-facing transport.
        return data
