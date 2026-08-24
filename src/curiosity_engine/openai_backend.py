from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import repository_root

OutputModel = TypeVar("OutputModel", bound=BaseModel)

MODEL_SETTING_KEYS = ("CURIOSITY_BACKEND", "OPENAI_API_KEY", "CURIOSITY_MODEL")


def load_model_settings() -> dict[str, str]:
    """Load model settings from the environment or the ignored owner-only setup file."""

    settings = {key: os.environ.get(key, "") for key in MODEL_SETTING_KEYS}
    model_file = repository_root() / "private" / "setup" / "model.env"
    if model_file.is_file():
        mode = stat.S_IMODE(model_file.stat().st_mode)
        if mode & 0o077:
            raise PermissionError("private/setup/model.env must be owner-only; run: chmod 600 private/setup/model.env")
        for raw_line in model_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            normalized = key.strip()
            if normalized in settings and not settings[normalized]:
                settings[normalized] = value.strip().strip("'\"")
    settings["CURIOSITY_BACKEND"] = settings["CURIOSITY_BACKEND"] or "deterministic"
    return settings


def model_key_is_configured(value: str) -> bool:
    return value.startswith("sk-") and "REPLACE_ME" not in value and len(value) >= 20


def _web_sources(response: Any) -> list[str]:
    urls: list[str] = []
    for item in getattr(response, "output", []) or []:
        raw = item if isinstance(item, dict) else item.model_dump() if hasattr(item, "model_dump") else {}
        action = raw.get("action") or {}
        for source in action.get("sources") or []:
            source_raw = source if isinstance(source, dict) else source.model_dump() if hasattr(source, "model_dump") else {}
            url = str(source_raw.get("url") or "")
            if url.startswith(("https://", "http://")) and url not in urls:
                urls.append(url)
    return urls[:8]


class OpenAIBackend:
    """Responses API adapter with schema guidance plus fail-closed local validation."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-5.4",
        *,
        client: Any | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int = 4_000,
        routes: dict[str, dict[str, Any]] | None = None,
    ):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - packaging guards this
                raise RuntimeError("Install the 'openai' package to use OpenAIBackend") from exc
            selected_key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not model_key_is_configured(selected_key):
                raise RuntimeError("OpenAI mode is enabled but no valid API key is configured")
            client = OpenAI(api_key=selected_key)
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
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
        text_payload = {key: value for key, value in payload.items() if key != "image_data_urls"}
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": json.dumps(text_payload, ensure_ascii=False, default=str)}
        ]
        for image in payload.get("image_data_urls", []):
            content.append({"type": "input_image", "image_url": image, "detail": "high"})
        schema = response_model.model_json_schema()
        route = self.routes.get(role) or {}
        provider = route.get("provider")
        if provider and provider != "openai":
            raise ValueError(f"OpenAIBackend cannot execute provider route {provider!r} for {role}")
        selected_model = route.get("model") or self.model
        selected_effort = route.get("reasoning_effort") or self.reasoning_effort
        self.model = selected_model
        request: dict[str, Any] = {
            "model": selected_model,
            "instructions": system,
            "input": [{"role": "user", "content": content}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": f"curiosity_{role}".replace("-", "_")[:64],
                    # Some response contracts contain intentionally bounded free-form metadata.
                    # Pydantic performs the final strict, extra-forbid validation locally.
                    "strict": False,
                    "schema": schema,
                }
            },
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if selected_effort:
            request["reasoning"] = {"effort": selected_effort}
        allowed_tools = set((payload.get("policy") or {}).get("allowed_tools") or [])
        if "web_search" in allowed_tools:
            request["tools"] = [{"type": "web_search"}]
            request["tool_choice"] = "auto"
            request["max_tool_calls"] = 2
            request["include"] = ["web_search_call.action.sources"]
        response = self.client.responses.create(**request)
        text = getattr(response, "output_text", None)
        if not text:
            raise RuntimeError("OpenAI response did not contain structured output text")
        data = json.loads(text)
        if "resource_refs" in response_model.model_fields:
            existing_refs = [str(item) for item in data.get("resource_refs") or []]
            data["resource_refs"] = list(dict.fromkeys([*existing_refs, *_web_sources(response)]))[:8]
        return response_model.model_validate(data).model_dump(mode="json")


def data_url_for_image(path: str | Path) -> str:
    import base64
    import mimetypes

    image = Path(path)
    mime = mimetypes.guess_type(image.name)[0] or "image/png"
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
