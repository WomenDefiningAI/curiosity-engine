from __future__ import annotations

import io
import stat
from base64 import b64encode
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from curiosity_engine.contracts import PullThreadOutput, VisualIntent, VisualPanel
from curiosity_engine.db import connect, init_db, utcnow
from curiosity_engine.openai_image_backend import OpenAIImageBackend
from curiosity_engine.reasoning import ReasoningEngine, ReasoningPolicy, StubBackend
from curiosity_engine.trust import validate_response_visual_intent
from curiosity_engine.visuals import (
    enqueue_response_visual,
    infer_safe_response_visual,
    normalize_response_visual,
    process_visual_jobs,
    render_deterministic_visual,
    synthetic_visual_intent,
)


def add_event(db: Path, event_id: str = "evt_visual") -> None:
    init_db(db)
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
               VALUES(?,'child_question','synthetic robot question','test','{}',?,'completed')""",
            (event_id, utcnow()),
        )


def test_visual_contract_and_policy_fail_closed_for_knowledge_bearing_requests():
    with pytest.raises(ValidationError, match="not to scale"):
        VisualIntent(
            kind="comparison_cards",
            purpose="compare",
            knowledge_role="supportive",
            title="Compare",
            pedagogical_value="Helps compare.",
            caption="Compare these.",
            alt_text="Two cards compare the examples safely.",
            panels=[
                {"label": "ONE", "detail": "First example.", "icon": "look"},
                {"label": "TWO", "detail": "Second example.", "icon": "look"},
            ],
        )
    unsafe = synthetic_visual_intent().model_copy(update={"knowledge_role": "instructional"})
    assert "future Tier C pipeline" in "; ".join(
        validate_response_visual_intent(unsafe.model_dump(mode="json"))
    )
    numbered = synthetic_visual_intent().model_copy(
        update={
            "panels": [
                VisualPanel(label="STEP 1", detail="Exact first value.", icon="look"),
                VisualPanel(label="STEP 2", detail="Exact second value.", icon="try"),
            ]
        }
    )
    assert "exact numeric values" in "; ".join(
        validate_response_visual_intent(numbered.model_dump(mode="json"))
    )


def test_deterministic_renderer_is_accessible_private_and_nonblank(tmp_path: Path):
    target = tmp_path / "visual.png"
    render_deterministic_visual(synthetic_visual_intent(), target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    image = Image.open(target).convert("RGB")
    assert image.size == (1200, 900)
    assert len(image.getcolors(maxcolors=image.width * image.height) or []) > 4


def test_visual_job_writes_opaque_validated_asset_beneath_private_output(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    add_event(db)
    job_id = enqueue_response_visual(
        db,
        event_id="evt_visual",
        visual=synthetic_visual_intent(),
        mode="deterministic",
    )
    assert job_id
    result = process_visual_jobs(db, output)
    assert result == [{"job_id": job_id, "status": "completed", "asset_id": result[0]["asset_id"]}]
    with connect(db) as conn:
        asset = conn.execute("SELECT * FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()
    path = Path(asset["path"]).resolve()
    assert path.is_relative_to((output / "visuals").resolve())
    assert path.name.startswith("visual_") and "robot" not in path.name
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert asset["mime_type"] == "image/png"
    assert len(asset["sha256"]) == 64
    assert "metadata_stripped" in asset["validation_json"]


def test_decorative_generation_uses_minimized_prompt_and_strips_metadata(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    add_event(db, "evt_decorative")
    visual = {
        "kind": "decorative_illustration",
        "purpose": "imagine",
        "knowledge_role": "decorative",
        "title": "Moon garden",
        "pedagogical_value": "Creates wonder without teaching measurements.",
        "caption": "Imagine a robot garden on the Moon.",
        "alt_text": "A friendly imaginary robot waters a tiny Moon garden.",
        "subject": "a friendly imaginary robot watering a tiny garden on the moon",
    }
    job_id = enqueue_response_visual(db, event_id="evt_decorative", visual=visual, mode="decorative")

    class FakeImageBackend:
        name = "fake"
        model = "image-test"

        def __init__(self):
            self.prompt = ""

        def generate(self, prompt: str):
            from curiosity_engine.openai_image_backend import GeneratedImage

            self.prompt = prompt
            image = Image.new("RGB", (1024, 1024), "#88CCAA")
            image.paste("#224466", (100, 100, 900, 900))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", pnginfo=None)
            return GeneratedImage(buffer.getvalue(), self.model, "request-test")

    backend = FakeImageBackend()
    assert process_visual_jobs(db, output, image_backend=backend)[0]["status"] == "completed"
    assert "No words, letters, numbers" in backend.prompt
    assert "child_id" not in backend.prompt and "private" not in backend.prompt.casefold()
    with connect(db) as conn:
        asset = conn.execute("SELECT provenance_json FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()
    assert '"private_context_sent":false' in asset["provenance_json"]
    assert '"response_topic_sent":true' in asset["provenance_json"]


def test_openai_image_backend_decodes_the_default_base64_response():
    class FakeImages:
        def __init__(self):
            self.kwargs = {}

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                data=[SimpleNamespace(b64_json=b64encode(b"synthetic-image").decode())],
                _request_id="request-test",
            )

    images = FakeImages()
    backend = OpenAIImageBackend("gpt-image-test", client=SimpleNamespace(images=images))
    generated = backend.generate("synthetic prompt")
    assert generated.data == b"synthetic-image"
    assert images.kwargs["quality"] == "low" and images.kwargs["size"] == "1024x1024"
    assert "response_format" not in images.kwargs


def test_interrupted_deterministic_job_recovers_but_generative_job_does_not(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    add_event(db, "evt_local_recovery")
    local_job = enqueue_response_visual(
        db,
        event_id="evt_local_recovery",
        visual=synthetic_visual_intent(),
        mode="deterministic",
    )
    add_event(db, "evt_provider_recovery")
    provider_job = enqueue_response_visual(
        db,
        event_id="evt_provider_recovery",
        mode="decorative",
        visual={
            "kind": "decorative_illustration",
            "purpose": "imagine",
            "knowledge_role": "decorative",
            "title": "Provider recovery",
            "pedagogical_value": "Exercises ambiguous provider recovery.",
            "caption": "A synthetic provider recovery test.",
            "alt_text": "A synthetic robot garden used for provider recovery testing.",
            "subject": "an imaginary robot tending a moon garden",
        },
    )
    stale = (datetime.now(UTC) - timedelta(minutes=20)).isoformat()
    with connect(db) as conn:
        conn.execute(
            "UPDATE visual_jobs SET status='processing',attempts=1,updated_at=? WHERE id IN (?,?)",
            (stale, local_job, provider_job),
        )
    results = process_visual_jobs(db, output)
    assert results[0]["job_id"] == local_job and results[0]["status"] == "completed"
    with connect(db) as conn:
        statuses = {
            row["id"]: row["status"]
            for row in conn.execute("SELECT id,status FROM visual_jobs WHERE id IN (?,?)", (local_job, provider_job))
        }
    assert statuses == {local_job: "completed", provider_job: "failed"}


def test_stub_robot_answers_request_the_right_safe_visual():
    biggest = StubBackend._pull_thread("What are the biggest robots in the world?")
    parsed = PullThreadOutput.model_validate(biggest)
    assert parsed.visual and parsed.visual.kind == "comparison_cards"
    assert parsed.visual.not_to_scale is True
    assert [panel.label for panel in parsed.visual.panels] == ["TALLEST", "HEAVIEST", "STRONGEST"]

    build = PullThreadOutput.model_validate(StubBackend._pull_thread("Can we build a robot?"))
    assert build.visual and build.visual.kind == "activity_sequence"
    assert [panel.label for panel in build.visual.panels] == ["GO", "STOP", "TURN"]


def test_unsafe_model_robot_visual_is_replaced_with_curated_comparison():
    unsafe = StubBackend._pull_thread("What are the biggest robots in the world?")
    unsafe["visual"] = {
        **unsafe["visual"],
        "knowledge_role": "instructional",
        "title": "Model-proposed instructional card",
        "panels": [
            {"label": "Walking robot", "detail": "Moves on legs.", "icon": "go"},
            {"label": "Robot truck", "detail": "Carries rocks.", "icon": "strength"},
            {"label": "Space arm", "detail": "Reaches and grabs.", "icon": "try"},
        ],
    }
    visual = normalize_response_visual("What are the biggest robots in the world?", unsafe)
    assert visual and visual.knowledge_role == "supportive"
    assert [panel.label for panel in visual.panels] == ["TALLEST", "HEAVIEST", "STRONGEST"]
    assert validate_response_visual_intent(visual.model_dump(mode="json")) == []


def test_reasoning_pipeline_normalizes_unsafe_visual_before_persistence():
    class UnsafeVisualBackend(StubBackend):
        def complete(self, *, role, system, payload, response_model):
            candidate = super().complete(
                role=role,
                system=system,
                payload=payload,
                response_model=response_model,
            )
            if response_model is PullThreadOutput:
                candidate["visual"] = {**candidate["visual"], "knowledge_role": "instructional"}
            return candidate

    result = ReasoningEngine(UnsafeVisualBackend()).run(
        policy=ReasoningPolicy("pull_thread", 2),
        context={},
        event={"text": "What are the biggest robots in the world?"},
    )
    visual = result.output["visual"]
    assert visual["knowledge_role"] == "supportive"
    assert [panel["label"] for panel in visual["panels"]] == ["TALLEST", "HEAVIEST", "STRONGEST"]


def test_safe_visual_fallbacks_are_narrow_and_exclude_social_inference():
    activity = infer_safe_response_visual("Can we make a paper robot?", {})
    assert activity and activity.kind == "activity_sequence"
    assert [panel.label for panel in activity.panels] == ["GO", "STOP", "TURN"]
    assert infer_safe_response_visual("Why was my friend mad at me?", {}) is None
    assert infer_safe_response_visual("Why did she ignore me?", {}) is None
    assert infer_safe_response_visual("What is the largest number?", {}) is None
    assert infer_safe_response_visual("How do black holes form?", {}) is None
    assert infer_safe_response_visual("Are robots alive?", {}) is None
    assert infer_safe_response_visual("Which country has the most lakes?", {}) is None


def test_unreviewed_deterministic_model_content_is_discarded_and_cannot_render(tmp_path: Path):
    proposed = synthetic_visual_intent().model_copy(
        update={
            "title": "Mix household chemicals",
            "caption": "Try this reaction.",
            "panels": [
                VisualPanel(label="BLEACH", detail="Pour it first.", icon="try"),
                VisualPanel(label="VINEGAR", detail="Add it next.", icon="try"),
            ],
        }
    )
    assert normalize_response_visual("Can we mix these liquids?", {"visual": proposed}) is None
    with pytest.raises(ValueError, match="reviewed local template"):
        render_deterministic_visual(proposed, tmp_path / "unsafe.png")


def test_malformed_optional_visual_does_not_reject_a_good_text_answer():
    class MalformedVisualBackend(StubBackend):
        def complete(self, *, role, system, payload, response_model):
            candidate = super().complete(
                role=role,
                system=system,
                payload=payload,
                response_model=response_model,
            )
            if response_model is PullThreadOutput:
                candidate["visual"] = {"kind": "activity_sequence"}
            return candidate

    result = ReasoningEngine(MalformedVisualBackend()).run(
        policy=ReasoningPolicy("pull_thread", 2),
        context={},
        event={"text": "Can we build a robot?"},
    )
    assert result.output["hook"]
    assert [panel["label"] for panel in result.output["visual"]["panels"]] == ["GO", "STOP", "TURN"]


def test_safe_decorative_visual_proposal_is_retained():
    proposed = {
        "kind": "decorative_illustration",
        "purpose": "imagine",
        "knowledge_role": "decorative",
        "title": "Moon garden",
        "pedagogical_value": "Creates wonder without teaching facts.",
        "caption": "Imagine a robot garden on the moon.",
        "alt_text": "A friendly imaginary robot waters a tiny garden on the moon.",
        "subject": "a friendly imaginary robot watering a tiny garden on the moon",
    }
    visual = normalize_response_visual("Imagine a robot garden on the moon", {"visual": proposed})
    assert visual and visual.kind == "decorative_illustration"


def test_visual_policy_rejects_private_decorative_subject():
    unsafe = {
        "kind": "decorative_illustration",
        "purpose": "imagine",
        "knowledge_role": "decorative",
        "title": "Unsafe",
        "pedagogical_value": "Would leak context.",
        "caption": "Unsafe example.",
        "alt_text": "An unsafe decorative example with private context.",
        "subject": "my child with our family private resource",
    }
    assert validate_response_visual_intent(unsafe)


def test_visual_policy_rejects_proper_names_and_known_household_identity(tmp_path: Path):
    proper_name = {
        "kind": "decorative_illustration",
        "purpose": "imagine",
        "knowledge_role": "decorative",
        "title": "Unsafe",
        "pedagogical_value": "Would expose a proper name.",
        "caption": "Unsafe example.",
        "alt_text": "An unsafe decorative example containing a proper name.",
        "subject": "alice visiting Seattle",
    }
    assert validate_response_visual_intent(proper_name)

    db = tmp_path / "private" / "data" / "db.sqlite"
    init_db(db)
    from curiosity_engine.graph import add_child

    add_child(db, "kid-private", "Alice", grade="1")
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_private_name','child_question','kid-private','Draw Alice with a robot','test','{}',?,'completed')""",
            (utcnow(),),
        )
    with pytest.raises(ValueError, match="private household identity"):
        enqueue_response_visual(
            db,
            event_id="evt_private_name",
            mode="decorative",
            visual={**proper_name, "subject": "alice with an imaginary robot"},
        )
