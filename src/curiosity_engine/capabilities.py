from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AppConfig, private_root
from .contracts import CapabilityCard, SkillCard


class CapabilityRegistry:
    """Reviewed, progressively disclosed workflows and procedural skills."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).resolve() if root else AppConfig.load().root
        catalog_path = self.root / "capabilities" / "catalog.json"
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        self._capabilities = {
            item["id"]: CapabilityCard.model_validate(item) for item in raw.get("capabilities", [])
        }
        self._skills = {item["id"]: SkillCard.model_validate(item) for item in raw.get("skills", [])}

    def capability_cards(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        cards = self._capabilities.values()
        if allowed is not None:
            cards = [card for card in cards if card.id in allowed]
        return [card.model_dump(mode="json") for card in cards]

    def skill_cards(self, allowed: set[str] | None = None) -> list[dict[str, Any]]:
        cards = self._skills.values()
        if allowed is not None:
            cards = [card for card in cards if card.id in allowed]
        return [card.model_dump(mode="json") for card in cards]

    def capability(self, capability_id: str) -> CapabilityCard:
        try:
            return self._capabilities[capability_id]
        except KeyError as exc:
            raise KeyError(f"unknown capability: {capability_id}") from exc

    def load_skill(self, skill_id: str) -> str:
        try:
            card = self._skills[skill_id]
        except KeyError as exc:
            raise KeyError(f"unknown skill: {skill_id}") from exc
        base = private_root() if card.family_private else self.root
        path = (base / card.instructions_path).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError("skill path escapes its reviewed root")
        return path.read_text(encoding="utf-8").strip()

    def instructions_for(self, capability_id: str) -> str:
        capability = self.capability(capability_id)
        return "\n\n".join(self.load_skill(skill_id) for skill_id in capability.skill_ids)
