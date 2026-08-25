from __future__ import annotations

from pathlib import Path

import pytest

from curiosity_engine.brain_config import (
    brain_config_fingerprint,
    brain_status,
    configure_api_brain,
    ensure_model_env_template,
    write_brain_config,
)
from curiosity_engine.cli import dump
from curiosity_engine.db import connect, init_db, utcnow
from curiosity_engine.interaction import configure_family_lens, onboarding_status, record_onboarding_review
from curiosity_engine.lab import evaluate
from curiosity_engine.onboarding import doctor, run_brain_probe, run_image_generation_probe


def test_private_brain_stack_requires_reasoning_vision_ocr_image_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    brain_path = tmp_path / "private" / "setup" / "brain.json"
    model_path = tmp_path / "private" / "setup" / "model.env"
    monkeypatch.setenv("CURIOSITY_BRAIN_CONFIG", str(brain_path))
    monkeypatch.setenv("CURIOSITY_MODEL_ENV", str(model_path))
    config = configure_api_brain(
        provider="openai",
        model="reasoning-model",
        vision_model="vision-model",
        image_model="image-model",
    )
    write_brain_config(config)
    ensure_model_env_template({"openai"})
    assert brain_status()["configured"] is False
    model_path.write_text("OPENAI_API_KEY=sk-private-example-key-123456789\n", encoding="utf-8")
    model_path.chmod(0o600)
    status = brain_status()
    assert status["configured"] is True
    assert status["multimodal_stack_configured"] is True
    assert status["providers"] == ["openai"]


def test_brain_template_never_rewrites_existing_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "private" / "setup" / "model.env"
    monkeypatch.setenv("CURIOSITY_MODEL_ENV", str(model_path))
    model_path.parent.mkdir(parents=True)
    model_path.write_text("OPENAI_API_KEY=sk-private-example-key-123456789\n", encoding="utf-8")
    model_path.chmod(0o600)

    ensure_model_env_template({"openai", "anthropic"})

    rendered = model_path.read_text(encoding="utf-8")
    assert rendered.count("OPENAI_API_KEY=") == 1
    assert "OPENAI_API_KEY=sk-private-example-key-123456789" in rendered
    assert rendered.count("ANTHROPIC_API_KEY=REPLACE_ME") == 1
    assert model_path.stat().st_mode & 0o077 == 0


def test_cli_json_redacts_provider_and_transport_credentials(capsys: pytest.CaptureFixture[str]):
    dump(
        {
            "status": "configured",
            "OPENAI_API_KEY": "sk-private-example-key-123456789",
            "nested": {"SLACK_APP_TOKEN": "xapp-private-example-token-123456789"},
            "accidental_value": "provider error included xoxb-private-example-token-123456789 in text",
        }
    )
    rendered = capsys.readouterr().out
    assert rendered.count("[redacted]") == 3
    assert "sk-private" not in rendered
    assert "xapp-private" not in rendered
    assert "xoxb-private" not in rendered


def test_brain_probe_is_synthetic_and_records_only_sanitized_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = tmp_path / "db.sqlite"
    brain_path = tmp_path / "brain.json"
    monkeypatch.setenv("CURIOSITY_BRAIN_CONFIG", str(brain_path))
    config = configure_api_brain(provider="openai", model="test-model", image_model="image-model")
    write_brain_config(config)

    class FakeBackend:
        name = "fake-provider"
        model = "fake-model"

        def complete(self, **kwargs):
            assert kwargs["payload"]["probe"] == "curiosity-engine-synthetic-v1"
            rendered = str(kwargs["payload"])
            assert "child" not in rendered.casefold()
            assert "resource" not in rendered.casefold()
            return {"marker": "ready", "count": 3}

    monkeypatch.setattr("curiosity_engine.onboarding.configured_backend", lambda *_args, **_kwargs: FakeBackend())
    result = run_brain_probe(db, live=True)
    assert result["family_data_sent"] is False
    state = onboarding_status(db)
    assert state["checkpoints"]["brain_verified"]["status"] == "pass"


def test_image_probe_is_explicit_paid_synthetic_and_route_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from io import BytesIO

    from PIL import Image

    from curiosity_engine.openai_image_backend import GeneratedImage

    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    brain_path = tmp_path / "private" / "setup" / "brain.json"
    monkeypatch.setenv("CURIOSITY_BRAIN_CONFIG", str(brain_path))
    write_brain_config(configure_api_brain(provider="openai", model="text-test", image_model="image-test"))

    class FakeImageBackend:
        name = "openai"
        model = "image-test"

        def generate(self, prompt: str) -> GeneratedImage:
            assert "cheerful imaginary storybook robot" in prompt
            assert "code-rendered learning card" in prompt
            assert "private resource" not in prompt.casefold()
            image = Image.new("RGB", (1024, 1024), "#AADDCC")
            image.paste("#224466", (100, 100, 900, 900))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return GeneratedImage(buffer.getvalue(), self.model, "request-synthetic")

    monkeypatch.setattr(
        "curiosity_engine.onboarding.configured_image_backend", lambda: FakeImageBackend()
    )
    monkeypatch.setattr(
        "curiosity_engine.visuals._configured_visual_qa_backend",
        lambda: type(
            "PassingVisualQA",
            (),
            {
                "name": "openai",
                "model": "vision-test",
                "complete": lambda self, **_kwargs: {
                    "verdict": "pass",
                    "reasons": [],
                    "inspected_pages": 1,
                },
            },
        )(),
    )
    with pytest.raises(ValueError, match="billable"):
        run_image_generation_probe(db, output, live=False)
    result = run_image_generation_probe(db, output, live=True)
    assert result["status"] == "pass" and result["family_data_sent"] is False
    assert onboarding_status(db)["checkpoints"]["image_generation_verified"]["status"] == "pass"


def test_family_lens_and_parent_review_are_explicit_private_gates(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    configure_family_lens(
        db,
        {
            "pedagogy": ["show before explaining"],
            "themes": [],
            "activity_minutes": 10,
            "parent_effort": "low",
            "reading_load": "early_elementary",
            "materials": ["paper"],
            "content_boundaries": [],
        },
    )
    state = onboarding_status(db)
    assert state["family_lens_configured"] is True
    with pytest.raises(ValueError, match="completed real Slack answer"):
        record_onboarding_review(
            db,
            event_id="evt_not_delivered",
            factuality="pass",
            grade_fit="pass",
            curiosity_value="pass",
            parent_effort="pass",
        )
    assert onboarding_status(db)["quality_review_accepted"] is False


def test_offline_lab_never_inherits_private_provider_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model_path = tmp_path / "model.env"
    model_path.write_text(
        "CURIOSITY_BACKEND=openai\nOPENAI_API_KEY=sk-REPLACE_ME\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CURIOSITY_BACKEND", "openai")
    monkeypatch.setenv("CURIOSITY_MODEL_ENV", str(model_path))
    report = evaluate(Path(__file__).resolve().parents[1], live_judge=False)
    assert report["status"] == "pass"
    assert report["summary"]["failed"] == 0


def test_doctor_fails_safely_outside_a_protected_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("curiosity_engine.onboarding.repository_root", lambda: tmp_path)
    report = doctor(tmp_path / "private" / "data" / "db.sqlite")
    boundary = next(check for check in report["checks"] if check["name"] == "private_git_boundary")
    assert report["core_ready"] is False
    assert boundary["status"] == "fail"
    assert "cloned repository" in boundary["detail"]


def test_doctor_reports_redacted_answer_rejection_rate(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    init_db(db)
    with connect(db) as conn:
        for index, status in enumerate(("completed", "rejected", "rejected", "completed", "rejected")):
            conn.execute(
                """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
                   VALUES(?, 'child_question', ?, 'synthetic', '{}', ?, ?)""",
                (f"evt_quality_{index}", f"synthetic private wording {index}", utcnow(), status),
            )

    report = doctor(db)
    assert report["answer_quality"] == {
        "status": "attention",
        "terminal_questions": 5,
        "completed": 2,
        "rejected": 3,
        "failed": 0,
        "rejection_rate": 0.6,
        "unsuccessful_rate": 0.6,
        "window": 20,
    }
    assert "synthetic private wording" not in str(report)


def test_doctor_treats_runtime_failures_as_unhealthy_answer_quality(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    init_db(db)
    with connect(db) as conn:
        for index, status in enumerate(("completed", "completed", "completed", "failed", "failed")):
            conn.execute(
                """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
                   VALUES(?, 'child_question', 'synthetic', 'synthetic', '{}', ?, ?)""",
                (f"evt_failure_rate_{index}", utcnow(), status),
            )

    quality = doctor(db)["answer_quality"]
    assert quality["status"] == "attention"
    assert quality["rejection_rate"] == 0.0
    assert quality["unsuccessful_rate"] == 0.4


def test_legacy_brain_fingerprint_changes_with_non_secret_route(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CURIOSITY_BACKEND", "openai")
    monkeypatch.setenv("CURIOSITY_MODEL", "model-a")
    first = brain_config_fingerprint()
    monkeypatch.setenv("CURIOSITY_MODEL", "model-b")
    assert brain_config_fingerprint() != first
