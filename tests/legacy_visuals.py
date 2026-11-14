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

import curiosity_engine.visuals as visuals_module
from curiosity_engine.contracts import PullThreadOutput, VisualIntent, VisualPanel
from curiosity_engine.db import connect, init_db, utcnow
from curiosity_engine.openai_image_backend import OpenAIImageBackend
from curiosity_engine.reasoning import ReasoningEngine, ReasoningPolicy, StubBackend
from curiosity_engine.trust import validate_response_visual_intent
from curiosity_engine.visuals import (
    enqueue_response_visual,
    infer_safe_response_visual,
    infer_safe_visual_revision,
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
    assert "future Tier C pipeline" in "; ".join(validate_response_visual_intent(unsafe.model_dump(mode="json")))
    numbered = synthetic_visual_intent().model_copy(
        update={
            "panels": [
                VisualPanel(label="STEP 1", detail="Exact first value.", icon="look"),
                VisualPanel(label="STEP 2", detail="Exact second value.", icon="try"),
            ]
        }
    )
    assert "exact numeric values" in "; ".join(validate_response_visual_intent(numbered.model_dump(mode="json")))


def test_deterministic_renderer_is_accessible_private_and_nonblank(tmp_path: Path):
    target = tmp_path / "visual.png"
    render_deterministic_visual(synthetic_visual_intent(), target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    image = Image.open(target).convert("RGB")
    assert image.size == (1200, 900)
    assert len(image.getcolors(maxcolors=image.width * image.height) or []) > 4


def test_explicit_water_freezing_revision_gets_a_reviewed_before_after_diagram(tmp_path: Path):
    intent = infer_safe_visual_revision(
        "Why does water expand when it freezes?",
        {
            "hook": "Ice can rise above the starting water line.",
            "show": "Mark the water line before freezing.",
            "nugget": "Frozen water usually needs a little more room.",
        },
        "How about a diagram?",
    )

    assert intent is not None
    assert [panel.icon for panel in intent.panels] == ["water", "ice", "look"]
    target = tmp_path / "ice-diagram.png"
    render_deterministic_visual(intent, target)
    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_quick_answer_visual_revision_uses_quick_fields_but_does_not_guess_from_better():
    output = {
        "answer": "Put a safe little object in a cup of water and see it after the water freezes.",
        "notice": "Compare the water and ice.",
        "question_back": None,
        "visual": {
            "subject": "a magical ice treasure in a cup",
            "caption": "What is frozen inside?",
        },
    }

    assert infer_safe_visual_revision("Try an ice treasure.", output, "Better viz please") is None
    intent = infer_safe_visual_revision("Try an ice treasure.", output, "Show why the ice changes in a diagram")
    assert intent is not None
    assert intent.title == "Watch the water line"

    floating = infer_safe_visual_revision(
        "Why does ice float?",
        {
            "answer": "Ice has a roomier pattern than liquid water, so it is less dense and floats.",
            "notice": "Most of a floating cube sits below the water line.",
        },
        "Show why in a comparison diagram",
    )
    assert floating is not None
    assert floating.title == "Why does ice float?"


def test_visual_revision_returns_reviewed_wing_comparison(tmp_path: Path):
    visual = infer_safe_visual_revision(
        "How do airplane wings change how a paper airplane flies?",
        {"answer": "Wing shape can change how the air pushes on a plane."},
        "This image isn't helpful. Show wide, narrow, swept, and straight wing variations instead.",
    )

    assert visual is not None
    assert visual.title == "Four ways a wing can change"
    assert [panel.label for panel in visual.panels] == ["WIDE", "NARROW", "SWEPT", "STRAIGHT"]
    rendered = render_deterministic_visual(visual, tmp_path / "wings.png")
    assert rendered.exists()
    assert rendered.stat().st_size > 10_000


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


def test_visual_worker_can_claim_only_the_explicit_probe_job(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    add_event(db, "evt_older_visual")
    older = enqueue_response_visual(
        db,
        event_id="evt_older_visual",
        visual=synthetic_visual_intent(),
        mode="deterministic",
    )
    add_event(db, "evt_explicit_probe")
    probe = enqueue_response_visual(
        db,
        event_id="evt_explicit_probe",
        visual=synthetic_visual_intent(),
        mode="deterministic",
    )

    result = process_visual_jobs(db, output, job_id=probe, limit=1)
    assert result[0]["job_id"] == probe
    with connect(db) as conn:
        statuses = {
            row["id"]: row["status"]
            for row in conn.execute("SELECT id,status FROM visual_jobs WHERE id IN (?,?)", (older, probe))
        }
    assert statuses == {older: "queued", probe: "completed"}


def test_transient_visual_failure_requeues_without_expiring_slack_delivery(tmp_path: Path, monkeypatch):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    add_event(db, "evt_visual_retry")
    job_id = enqueue_response_visual(
        db,
        event_id="evt_visual_retry",
        visual=synthetic_visual_intent(),
        mode="deterministic",
    )
    original = visuals_module.render_deterministic_visual
    calls = 0

    def flaky(intent, path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary renderer outage")
        return original(intent, path)

    monkeypatch.setattr(visuals_module, "render_deterministic_visual", flaky)
    assert process_visual_jobs(db, output) == [{"job_id": job_id, "status": "retrying"}]
    with connect(db) as conn:
        job = conn.execute("SELECT status,attempts FROM visual_jobs WHERE id=?", (job_id,)).fetchone()
        conn.execute("UPDATE visual_jobs SET available_at=? WHERE id=?", (utcnow(), job_id))
    assert dict(job) == {"status": "queued", "attempts": 1}
    assert process_visual_jobs(db, output)[0]["status"] == "completed"


def test_decorative_generation_uses_minimized_prompt_and_strips_metadata(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    add_event(db, "evt_decorative")
    visual = {
        "kind": "comparison_cards",
        "purpose": "compare",
        "knowledge_role": "supportive",
        "title": "Two Moon gardens",
        "pedagogical_value": "Makes two visible plant-growth conditions easy to compare.",
        "caption": "Compare the garden with light and the garden in shade.",
        "alt_text": "Two labeled cards compare a Moon garden in light with one in shade.",
        "subject": "a friendly imaginary robot watering a tiny garden on the moon",
        "panels": [
            {"label": "LIGHT", "detail": "The plant receives light.", "icon": "look"},
            {"label": "SHADE", "detail": "The plant stays shaded.", "icon": "look"},
        ],
        "not_to_scale": True,
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

    class PassingVisualQA:
        name = "fake-vision"
        model = "vision-test"

        def __init__(self):
            self.call = None

        def complete(self, **kwargs):
            self.call = kwargs
            return {"verdict": "pass", "reasons": [], "inspected_pages": 1}

    backend = FakeImageBackend()
    reviewer = PassingVisualQA()
    assert (
        process_visual_jobs(db, output, image_backend=backend, visual_qa_backend=reviewer)[0]["status"]
        == "completed"
    )
    assert "No words, letters, numbers" in backend.prompt
    assert "child_id" not in backend.prompt and "private" not in backend.prompt.casefold()
    assert "intentionally decorative" in reviewer.call["system"]
    assert "code will add the accurate labels" in reviewer.call["system"]
    reviewed_intent = reviewer.call["payload"]["visual_intent"]
    assert reviewed_intent["purpose"] == "compare"
    assert reviewed_intent["usefulness_goal"] == (
        "make the surrounding code-rendered comparison inviting without depicting or teaching the difference"
    )
    assert set(reviewed_intent) == {"kind", "purpose", "knowledge_role", "subject", "usefulness_goal"}
    with connect(db) as conn:
        asset = conn.execute("SELECT provenance_json FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()
    assert '"private_context_sent":false' in asset["provenance_json"]
    assert '"response_topic_sent":true' in asset["provenance_json"]


def test_decorative_mode_embeds_generated_art_inside_reviewed_robot_card(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    add_event(db, "evt_hybrid")
    intent = infer_safe_response_visual("What are the biggest robots?", {})
    assert intent and intent.subject
    job_id = enqueue_response_visual(
        db,
        event_id="evt_hybrid",
        visual=intent,
        mode="decorative",
    )

    class FakeImageBackend:
        name = "fake"
        model = "image-test"

        def __init__(self):
            self.prompt = ""

        def generate(self, prompt: str):
            from curiosity_engine.openai_image_backend import GeneratedImage

            self.prompt = prompt
            image = Image.new("RGB", (1024, 1024), "#FF88AA")
            image.paste("#55CCDD", (100, 100, 924, 924))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return GeneratedImage(buffer.getvalue(), self.model, "request-hybrid")

    class PassingVisualQA:
        name = "fake-vision"
        model = "vision-test"

        def complete(self, **kwargs):
            assert kwargs["payload"]["image_data_urls"][0].startswith("data:image/png;base64,")
            return {"verdict": "pass", "reasons": [], "inspected_pages": 1}

    backend = FakeImageBackend()
    result = process_visual_jobs(db, output, image_backend=backend, visual_qa_backend=PassingVisualQA())
    assert result[0]["status"] == "completed"
    assert "decorative beside a code-rendered learning card" in backend.prompt
    assert "No words, letters, numbers" in backend.prompt
    with connect(db) as conn:
        asset = conn.execute("SELECT * FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()
    assert asset["method"] == "generative"
    assert asset["trust_tier"] == "B"
    assert asset["renderer_version"].startswith("response-card-v2+fake:")
    assert '"generated_art_embedded":true' in asset["provenance_json"]
    assert '"visual_qa":"pass"' in asset["provenance_json"]
    assert Image.open(asset["path"]).size == (1200, 900)


def test_hybrid_visual_falls_back_to_reviewed_card_when_generated_art_is_not_verified(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    add_event(db, "evt_hybrid_fallback")
    intent = infer_safe_response_visual("Can we make a paper robot?", {})
    assert intent
    job_id = enqueue_response_visual(
        db,
        event_id="evt_hybrid_fallback",
        visual=intent,
        mode="decorative",
    )

    class FailingImageBackend:
        name = "fake"
        model = "image-test"

        def generate(self, _prompt: str):
            raise TimeoutError("synthetic provider failure")

    result = process_visual_jobs(
        db,
        output,
        image_backend=FailingImageBackend(),
        visual_qa_backend=None,
    )
    assert result[0]["status"] == "completed"
    with connect(db) as conn:
        asset = conn.execute("SELECT * FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()
    assert asset["method"] == "deterministic"
    assert asset["trust_tier"] == "B"
    assert asset["renderer_version"] == "response-card-v2"
    assert '"fallback_from":"generative"' in asset["provenance_json"]
    assert '"generated_art_embedded":false' in asset["provenance_json"]


def test_hybrid_visual_falls_back_when_visual_qa_rejects_generated_art(tmp_path: Path):
    db = tmp_path / "private" / "data" / "db.sqlite"
    output = tmp_path / "private" / "output"
    add_event(db, "evt_hybrid_qa_fallback")
    intent = infer_safe_response_visual("What are the biggest robots?", {})
    assert intent
    job_id = enqueue_response_visual(
        db,
        event_id="evt_hybrid_qa_fallback",
        visual=intent,
        mode="decorative",
    )

    class ImageBackend:
        name = "fake"
        model = "image-test"

        def generate(self, _prompt: str):
            from curiosity_engine.openai_image_backend import GeneratedImage

            image = Image.new("RGB", (1024, 1024), "#FF88AA")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return GeneratedImage(buffer.getvalue(), self.model, "request-rejected")

    class RejectingVisualQA:
        name = "fake-vision"
        model = "vision-test"

        def complete(self, **_kwargs):
            return {"verdict": "fail", "reasons": ["contains letters"], "inspected_pages": 1}

    assert (
        process_visual_jobs(
            db,
            output,
            image_backend=ImageBackend(),
            visual_qa_backend=RejectingVisualQA(),
        )[0]["status"]
        == "completed"
    )
    with connect(db) as conn:
        asset = conn.execute("SELECT method,provenance_json FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()
    assert asset["method"] == "deterministic"
    assert '"visual_qa":"not_passed"' in asset["provenance_json"]


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
    assert images.kwargs["quality"] == "medium" and images.kwargs["size"] == "1024x1024"
    assert "response_format" not in images.kwargs


def test_openai_image_backend_accepts_high_quality_landscape_challenge_art():
    class FakeImages:
        def __init__(self):
            self.kwargs = {}

        def generate(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(data=[SimpleNamespace(b64_json=b64encode(b"image").decode())])

    images = FakeImages()
    OpenAIImageBackend(
        "gpt-image-test",
        client=SimpleNamespace(images=images),
        quality="high",
        size="1536x1024",
    ).generate("challenge art")
    assert images.kwargs["quality"] == "high" and images.kwargs["size"] == "1536x1024"
    assert images.kwargs["output_format"] == "png" and images.kwargs["background"] == "opaque"


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
    gerund_activity = infer_safe_response_visual("How can we start making robots?", {})
    assert gerund_activity and gerund_activity.kind == "activity_sequence"
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
    with pytest.raises(ValueError, match="household chemicals"):
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
    assert result.output["visual"] is None


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


def test_visual_policy_rejects_private_decorative_subject(tmp_path: Path):
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

    db = tmp_path / "private" / "data" / "db.sqlite"
    init_db(db)
    from curiosity_engine.graph import add_child

    add_child(db, "kid-private-fields", "Example Kid", grade="1")
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_private_fields','child_question','kid-private-fields','Draw a moon robot','test','{}',?,'completed')""",
            (utcnow(),),
        )
    private_caption = {
        **unsafe,
        "kind": "comparison_cards",
        "purpose": "compare",
        "knowledge_role": "supportive",
        "title": "Moon robot",
        "pedagogical_value": "Supports imagination.",
        "caption": "Example Kid visits 123 Main Street.",
        "alt_text": "A friendly imaginary robot explores a colorful moon garden.",
        "subject": "a friendly imaginary robot exploring a moon garden",
        "panels": [
            {"label": "FIRST", "detail": "Look at one scene.", "icon": "look"},
            {"label": "NEXT", "detail": "Compare the other scene.", "icon": "look"},
        ],
        "not_to_scale": True,
    }
    with pytest.raises(
        ValueError,
        match="unsafe response visual|private household identity|private family or identifying context",
    ):
        enqueue_response_visual(
            db,
            event_id="evt_private_fields",
            mode="decorative",
            visual=private_caption,
        )

    add_child(db, "kid-short-name", "Bo", grade="1")
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_short_name','child_question','kid-short-name','Draw a paper airplane','test','{}',?,'completed')""",
            (utcnow(),),
        )
    with pytest.raises(ValueError, match="private household identity"):
        enqueue_response_visual(
            db,
            event_id="evt_short_name",
            mode="decorative",
            visual={
                **private_caption,
                "caption": "Compare the two paper-airplane flights.",
                "subject": "bo flying a cheerful paper airplane in a colorful paper world",
            },
        )

    safe_identity_visual = {
        **private_caption,
        "caption": "Compare the two paper-airplane flights.",
    }
    for index, name in enumerate(("Éva", "李", "Q")):
        child_id = f"kid-unicode-name-{index}"
        event_id = f"evt_unicode_name_{index}"
        add_child(db, child_id, name, grade="1")
        with connect(db) as conn:
            conn.execute(
                """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                   VALUES(?,'child_question',?,'Draw a paper airplane','test','{}',?,'completed')""",
                (event_id, child_id, utcnow()),
            )
        with pytest.raises(ValueError, match="private household identity"):
            enqueue_response_visual(
                db,
                event_id=event_id,
                mode="decorative",
                visual={
                    **safe_identity_visual,
                    "subject": f"{name.casefold()} flying a cheerful paper airplane in a colorful paper world",
                },
            )

    for index, name in enumerate(("小明", "alice")):
        event_id = f"evt_unregistered_name_{index}"
        with connect(db) as conn:
            conn.execute(
                """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
                   VALUES(?,'child_question',?,'test','{}',?,'completed')""",
                (event_id, f"Draw {name} with a robot", utcnow()),
            )
        with pytest.raises(ValueError, match="proper name from the family request"):
            enqueue_response_visual(
                db,
                event_id=event_id,
                mode="decorative",
                visual={
                    **safe_identity_visual,
                    "subject": f"{name} with a cheerful imaginary robot in a colorful paper world",
                },
            )

    possessive_requests = (
        ("evt_possessive_name_ava", "Let's compare ava's two paper airplanes", "ava"),
        (
            "evt_possessive_name_alice",
            "Please make a comparison diagram about alice's favorite birds.",
            "alice",
        ),
        (
            "evt_possessive_name_diagrams",
            "Let's compare diagrams of ava's two paper airplanes.",
            "ava",
        ),
        ("evt_possessive_name_of", "Make a diagram of alice's robot.", "alice"),
        ("evt_possessive_name_inspired", "Make a diagram inspired by alice's robot.", "alice"),
        ("evt_possessive_name_using", "Use alice's favorite birds in the diagram.", "alice"),
    )
    for event_id, request, name in possessive_requests:
        with connect(db) as conn:
            conn.execute(
                """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
                   VALUES(?,'child_question',?,'test','{}',?,'completed')""",
                (event_id, request, utcnow()),
            )
        with pytest.raises(ValueError, match="proper name from the family request"):
            enqueue_response_visual(
                db,
                event_id=event_id,
                mode="decorative",
                visual={
                    **safe_identity_visual,
                    "subject": f"{name}'s favorite objects in a cheerful paper world",
                },
            )

    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
               VALUES('evt_contextual_name','child_question',?,'test','{}',?,'completed')""",
            ("Make a diagram inspired by alice and her birds.", utcnow()),
        )
    with pytest.raises(ValueError, match="proper name from the family request"):
        enqueue_response_visual(
            db,
            event_id="evt_contextual_name",
            mode="decorative",
            visual={
                **safe_identity_visual,
                "subject": "alice with her favorite birds in a cheerful paper world",
            },
        )

    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
               VALUES('evt_picture_about_name','child_question',?,'test','{}',?,'completed')""",
            ("Make a picture about alice exploring birds.", utcnow()),
        )
    with pytest.raises(ValueError, match="proper name from the family request"):
        enqueue_response_visual(
            db,
            event_id="evt_picture_about_name",
            mode="decorative",
            visual={
                **safe_identity_visual,
                "subject": "alice exploring birds in a cheerful paper world",
            },
        )

    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
               VALUES('evt_terminal_name','child_question',?,'test','{}',?,'completed')""",
            ("Make a picture of ava.", utcnow()),
        )
    with pytest.raises(ValueError, match="proper name from the family request"):
        enqueue_response_visual(
            db,
            event_id="evt_terminal_name",
            mode="decorative",
            visual={
                **safe_identity_visual,
                "subject": "ava in a cheerful paper world",
            },
        )

    visual_noun_requests = (
        ("evt_want_picture_name", "I want a picture of alice."),
        ("evt_image_showing_name", "Create an image showing alice."),
        ("evt_image_that_shows_name", "Create an image that shows alice."),
        ("evt_image_depicting_name", "Create an image depicting alice."),
    )
    for event_id, request in visual_noun_requests:
        with connect(db) as conn:
            conn.execute(
                """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
                   VALUES(?,'child_question',?,'test','{}',?,'completed')""",
                (event_id, request, utcnow()),
            )
        with pytest.raises(ValueError, match="proper name from the family request"):
            enqueue_response_visual(
                db,
                event_id=event_id,
                mode="decorative",
                visual={
                    **safe_identity_visual,
                    "subject": "alice in a cheerful paper world",
                },
            )

    add_event(db, "evt_late_source_identity")
    late_source_job = enqueue_response_visual(
        db,
        event_id="evt_late_source_identity",
        mode="decorative",
        visual={
            **safe_identity_visual,
            "subject": "alice in a cheerful paper world",
        },
    )
    with connect(db) as conn:
        conn.execute(
            "UPDATE events SET text=? WHERE id='evt_late_source_identity'",
            ("Create an image showing alice.",),
        )

    add_event(db, "evt_late_identity")
    late_job = enqueue_response_visual(
        db,
        event_id="evt_late_identity",
        mode="decorative",
        visual={
            **safe_identity_visual,
            "subject": "zoë flying a cheerful paper airplane in a colorful paper world",
        },
    )
    add_child(db, "kid-late-identity", "Zoë", grade="1")

    class ProviderMustNotReceiveIdentity:
        name = "must-not-run"
        model = "must-not-run"

        def __init__(self):
            self.calls = 0

        def generate(self, prompt):
            self.calls += 1
            raise AssertionError(f"private prompt reached provider: {prompt}")

    provider = ProviderMustNotReceiveIdentity()
    result = process_visual_jobs(
        db,
        tmp_path / "private" / "output",
        image_backend=provider,
        job_id=late_job,
    )
    assert provider.calls == 0
    assert len(result) == 1 and result[0]["job_id"] == late_job and result[0]["status"] == "completed"
    late_source_result = process_visual_jobs(
        db,
        tmp_path / "private" / "output",
        image_backend=provider,
        job_id=late_source_job,
    )
    assert provider.calls == 0
    assert len(late_source_result) == 1
    assert late_source_result[0]["job_id"] == late_source_job
    assert late_source_result[0]["status"] == "completed"
    with connect(db) as conn:
        late_asset = conn.execute("SELECT method FROM visual_assets WHERE job_id=?", (late_job,)).fetchone()
        late_source_asset = conn.execute(
            "SELECT method FROM visual_assets WHERE job_id=?", (late_source_job,)
        ).fetchone()
    assert late_asset["method"] == "deterministic"
    assert late_source_asset["method"] == "deterministic"


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


def test_explicit_visual_request_completes_in_deterministic_mode_without_rewriting_text():
    question = "How does electricity move through a wire?"
    expected_hook = StubBackend._pull_thread(question)["hook"]

    class VisualCompletionBackend(StubBackend):
        def __init__(self):
            self.pull_thread_calls = 0

        def complete(self, *, role, system, payload, response_model):
            if response_model is PullThreadOutput:
                self.pull_thread_calls += 1
                if payload.get("visual_completion"):
                    result = dict(payload["candidate"])
                    result["visual"] = {
                        "kind": "comparison_cards",
                        "purpose": "compare",
                        "knowledge_role": "supportive",
                        "title": "Open and closed paths",
                        "pedagogical_value": "Makes two circuit-path states easy to compare.",
                        "caption": "Compare the open path with the closed path.",
                        "alt_text": "Two labeled cards compare an open path and a closed path.",
                        "subject": "a cheerful imaginary spark exploring a colorful cut-paper world",
                        "panels": [
                            {"label": "OPEN", "detail": "The path has a gap.", "icon": "look"},
                            {"label": "CLOSED", "detail": "The path connects.", "icon": "look"},
                        ],
                        "not_to_scale": True,
                    }
                    return result
                result = super().complete(
                    role=role,
                    system=system,
                    payload=payload,
                    response_model=response_model,
                )
                result["visual"] = None
                return result
            return super().complete(
                role=role,
                system=system,
                payload=payload,
                response_model=response_model,
            )

    backend = VisualCompletionBackend()
    result = ReasoningEngine(backend).run(
        policy=ReasoningPolicy("pull_thread", 2),
        context={"child": {"grade": "1st"}},
        event={
            "type": "child_question",
            "text": question,
            "metadata": {"response_visual_mode": "deterministic", "response_visual_requested": True},
        },
    )

    assert backend.pull_thread_calls == 2
    assert result.output["hook"] == expected_hook
    assert result.output["visual"]["kind"] == "comparison_cards"
    assert result.output["visual"]["knowledge_role"] == "supportive"


def test_incomplete_activity_aid_is_discarded_without_hidden_completion():
    question = "How did long-necked dinosaurs reach leaves?"
    art = {
        "kind": "comparison_cards",
        "purpose": "compare",
        "knowledge_role": "supportive",
        "title": "Compare leaf reach",
        "pedagogical_value": "Makes near and far leaf targets easy to compare.",
        "caption": "Compare which leaves each neck can reach.",
        "alt_text": "Two labeled cards compare reaching a near leaf and a far leaf.",
        "subject": "a friendly imaginary long-necked dinosaur reaching toward leafy branches",
        "panels": [
            {"label": "NEAR", "detail": "The leaf is close.", "icon": "look"},
            {"label": "FAR", "detail": "The leaf is farther away.", "icon": "look"},
        ],
        "not_to_scale": True,
    }

    class ActivityAidCompletionBackend(StubBackend):
        def __init__(self):
            self.pull_thread_calls = 0

        def complete(self, *, role, system, payload, response_model):
            if response_model is PullThreadOutput:
                self.pull_thread_calls += 1
                if payload.get("visual_completion"):
                    raise AssertionError("an incomplete activity aid must not trigger a hidden completion call")
                result = super().complete(
                    role=role,
                    system=system,
                    payload=payload,
                    response_model=response_model,
                )
                result["physical_extension"] = {
                    "title": "Sauropod leaf reach",
                    "instructions": [
                        "Ask a grown-up to draw one START, two neck lines, and three fixed leaf targets on a whole page."
                    ],
                    "materials": ["paper", "crayon"],
                    "parent_effort": "low",
                }
                result["visual"] = art
                return result
            return super().complete(
                role=role,
                system=system,
                payload=payload,
                response_model=response_model,
            )

    backend = ActivityAidCompletionBackend()
    result = ReasoningEngine(backend).run(
        policy=ReasoningPolicy("pull_thread", 2),
        context={"child": {"grade": "1st"}},
        event={
            "type": "child_question",
            "text": question,
            "metadata": {"response_visual_mode": "decorative", "response_visual_requested": True},
        },
    )

    assert backend.pull_thread_calls == 1
    assert result.output["visual"]["kind"] == "comparison_cards"
    assert result.output["physical_extension"] is None
