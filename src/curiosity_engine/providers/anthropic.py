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


def _content_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    text_payload = {key: value for key, value in payload.items() if key != "image_data_urls"}
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(text_payload, ensure_ascii=False, default=str)}
    ]
    for value in payload.get("image_data_urls", []):
        prefix, marker, data = str(value).partition(",")
        if not marker or ";base64" not in prefix or not prefix.startswith("data:image/"):
            raise ValueError("Anthropic image inputs must be base64 image data URLs")
        media_type = prefix[5:].split(";", 1)[0]
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        )
    return blocks


def _sources(response: Any) -> list[str]:
    urls: list[str] = []
    for block in getattr(response, "content", []) or []:
        raw = _raw(block)
        for citation in raw.get("citations") or []:
            url = str(_raw(citation).get("url") or "")
            if url.startswith(("https://", "http://")) and url not in urls:
                urls.append(url)
        content = raw.get("content") or []
        if isinstance(content, list):
            for result in content:
                url = str(_raw(result).get("url") or "")
                if url.startswith(("https://", "http://")) and url not in urls:
                    urls.append(url)
    return urls[:8]


class AnthropicBackend:
    """Native Messages API adapter with local schema validation."""

    name = "anthropic"

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
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - guarded by doctor/package extra
                raise RuntimeError("Install Anthropic support with: python -m pip install -e '.[anthropic]'") from exc
            selected_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not secret_is_configured("anthropic", selected_key):
                raise RuntimeError("Anthropic mode is enabled but no valid API key is configured")
            client = Anthropic(api_key=selected_key)
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
        if provider and provider != "anthropic":
            raise ValueError(f"AnthropicBackend cannot execute provider route {provider!r} for {role}")
        selected_model = str(route.get("model") or self.model)
        self.model = selected_model
        request: dict[str, Any] = {
            "model": selected_model,
            "max_tokens": self.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": _content_blocks(payload)}],
            "output_config": {
                "format": {"type": "json_schema", "schema": response_model.model_json_schema()}
            },
        }
        allowed_tools = set((payload.get("policy") or {}).get("allowed_tools") or [])
        if "web_search" in allowed_tools:
            request["tools"] = [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 2,
                    "allowed_callers": ["direct"],
                }
            ]
        response = self.client.messages.create(**request)
        text = next(
            (
                str(_raw(block).get("text") or "")
                for block in getattr(response, "content", []) or []
                if _raw(block).get("type") == "text"
            ),
            "",
        )
        if not text:
            raise RuntimeError("Anthropic response did not contain structured output text")
        data = json.loads(text)
        if "resource_refs" in response_model.model_fields:
            existing = [str(item) for item in data.get("resource_refs") or []]
            data["resource_refs"] = list(dict.fromkeys([*existing, *_sources(response)]))[:8]
        # Contract validation belongs to ReasoningEngine, where a configured
        # repair round can correct the provider candidate before it reaches a
        # family-facing transport.
        return data
