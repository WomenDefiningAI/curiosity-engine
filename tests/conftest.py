from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_tests_from_family_provider_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tests must never inherit a family's ignored credentials or spend API money."""

    monkeypatch.setenv("CURIOSITY_BACKEND", "deterministic")
    monkeypatch.setenv("CURIOSITY_BRAIN_CONFIG", str(tmp_path / "no-family-brain.json"))
    monkeypatch.setenv("CURIOSITY_MODEL_ENV", str(tmp_path / "no-family-model.env"))
