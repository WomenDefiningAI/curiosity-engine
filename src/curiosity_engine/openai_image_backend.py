from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Protocol

from .brain_config import SECRET_KEYS, load_brain_config, load_secret_settings, secret_is_configured


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    model: str
    request_id: str | None = None


class ImageBackend(Protocol):
    name: str
    model: str

    def generate(self, prompt: str) -> GeneratedImage: ...


class OpenAIImageBackend:
    """One-shot Image API adapter for opt-in, Tier-A decorative illustrations."""

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        quality: str = "medium",
    ):
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - packaging guard
                raise RuntimeError("Install the 'openai' package to use image generation") from exc
            if not api_key or not secret_is_configured("openai", api_key):
                raise RuntimeError("OpenAI image generation is configured but its API key is unavailable")
            client = OpenAI(api_key=api_key)
        if quality not in {"low", "medium", "high", "auto"}:
            raise ValueError("image quality must be low, medium, high, or auto")
        self.client = client
        self.model = model
        self.quality = quality

    def generate(self, prompt: str) -> GeneratedImage:
        result = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size="1024x1024",
            quality=self.quality,
        )
        entries = getattr(result, "data", None) or []
        encoded = getattr(entries[0], "b64_json", None) if entries else None
        if not encoded:
            raise RuntimeError("image provider returned no image bytes")
        try:
            data = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise RuntimeError("image provider returned invalid base64") from exc
        return GeneratedImage(
            data=data,
            model=self.model,
            request_id=str(getattr(result, "_request_id", "") or "") or None,
        )


def configured_image_backend() -> ImageBackend | None:
    """Resolve only the explicit image route; never fall back to the reasoning model."""

    config = load_brain_config()
    if config is None or config.runtime != "api":
        return None
    route = config.routes.get("image_generation")
    if route is None:
        return None
    if route.provider != "openai":
        raise RuntimeError(
            f"image route provider {route.provider!r} is configured but this release supports OpenAI image delivery only"
        )
    settings = load_secret_settings()
    key = settings[SECRET_KEYS[route.provider]]
    if not secret_is_configured(route.provider, key):
        raise RuntimeError("the configured image provider credential is unavailable")
    return OpenAIImageBackend(route.model, api_key=key)
