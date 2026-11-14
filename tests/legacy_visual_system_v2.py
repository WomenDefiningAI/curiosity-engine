from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

import curiosity_engine.image_research as research_module
from curiosity_engine.artifact_validation import validate_rendered_file
from curiosity_engine.contracts import (
    ImageResearchQuery,
    ImageRights,
    ImageSafetyReview,
    ResearchedImageAsset,
    VisualIntent,
    WorksheetSpec,
    WorksheetTask,
)
from curiosity_engine.db import connect, init_db, utcnow
from curiosity_engine.image_research import ImageCandidate, ImageResearchService
from curiosity_engine.learning_artifacts import render_learning_pdf, validate_learning_artifact
from curiosity_engine.visual_planner import build_visual_plan
from curiosity_engine.visuals import enqueue_response_visual, normalize_response_visual, process_visual_jobs
from curiosity_engine.worksheet_rendering import _shuffled_matching_choices


def _task(kind: str, instruction: str, choices: list[str] | None = None) -> WorksheetTask:
    return WorksheetTask(
        id=f"{kind}_{sha256(instruction.encode()).hexdigest()[:8]}",
        kind=kind,
        instruction=instruction,
        choices=choices or [],
        response_lines=1,
    )


ARCHETYPE_TASKS = {
    "look_notice_wonder": [_task("draw", "Draw one detail."), _task("short_response", "Write one wonder.")],
    "compare_circle_explain": [
        _task("circle", "Circle the best clue.", ["round", "striped", "smooth"]),
        _task("short_response", "Tell how the examples differ."),
    ],
    "match_two_columns": [
        _task("match", "Look at these animals.", ["owl", "frog", "bee"]),
        _task("match", "Match each home.", ["pond", "hive", "tree"]),
    ],
    "cut_sort_categories": [
        _task("sort", "Sort by where it moves.", ["whale", "eagle", "rabbit", "shark"]),
        _task("short_response", "Explain one choice."),
    ],
    "sequence_story_cards": [
        _task("sequence", "Put the seed story in order.", ["seed", "root", "sprout", "leaves"]),
        _task("short_response", "Tell what changes."),
    ],
    "label_parts": [
        _task("label", "Label the plant.", ["root", "stem", "leaf", "flower"]),
        _task("draw", "Add one detail."),
    ],
    "observation_checklist": [
        _task("circle", "Check clues you see.", ["shadow", "edge", "pattern", "color"]),
        _task("record", "Record one surprise."),
    ],
    "predict_test_record": [
        _task("predict", "Which paper bridge will hold more?", ["flat", "folded"]),
        _task("record", "Test and record what happened."),
    ],
    "measure_tally_graph": [
        _task("measure", "Measure three paper-airplane flights.", ["try 1", "try 2", "try 3"]),
        _task("record", "Tally where each plane landed."),
    ],
    "build_test_improve": [
        _task("draw", "Sketch a paper tower."),
        _task("record", "Change one thing after the test."),
    ],
    "mystery_clue_case": [
        _task("circle", "Use the clues.", ["wet tracks", "feather", "tiny seeds", "round print"]),
        _task("short_response", "Name your best solution."),
    ],
    "mini_field_guide": [
        _task("label", "Find these field marks.", ["beak", "wing", "tail", "feet"]),
        _task("draw", "Draw one field mark."),
    ],
    "play_pieces_targets": [
        _task("circle", "Use these leaf targets.", ["low leaf", "middle leaf", "high leaf"]),
        _task("short_response", "Place the leaves, then reach without moving your feet."),
    ],
}


@pytest.mark.parametrize("archetype", list(ARCHETYPE_TASKS))
def test_each_worksheet_archetype_renders_one_valid_page(tmp_path: Path, archetype: str) -> None:
    semantic_fields = {
        "compare_circle_explain": {"subject_labels": ["smooth stone", "rough bark"]},
        "cut_sort_categories": {"category_labels": ["moves in water", "moves in air"]},
        "label_parts": {"diagram_kind": "plant"},
        "play_pieces_targets": {"piece_shape": "leaf"},
    }.get(archetype, {})
    spec = WorksheetSpec(
        title=archetype.replace("_", " ").title(),
        target_grade="1st",
        learning_objective="Notice, compare, and record visible evidence.",
        estimated_minutes=12,
        parent_effort="low",
        trust_tier="B",
        story_theme="a playful nature mission",
        accessibility_text="A one-page child worksheet with large response spaces.",
        archetype=archetype,
        theme="nature",
        directions="Be a clue hunter. Look closely, make a choice, and show what you noticed.",
        tasks=ARCHETYPE_TASKS[archetype],
        **semantic_fields,
    )
    assert validate_learning_artifact(spec) == []
    target = render_learning_pdf(spec, tmp_path / f"{archetype}.pdf")
    assert validate_rendered_file(target) == []
    assert target.stat().st_size > 2_000


def test_visual_plan_suppresses_parent_chat_and_separates_photo_from_printable() -> None:
    assert build_visual_plan(None, parent_only=True).role == "none"
    intent = VisualIntent(
        kind="researched_photo",
        purpose="observe",
        knowledge_role="supportive",
        title="Look at a real owl",
        pedagogical_value="Supports noticing visible adaptations.",
        caption="What details help this owl live at night?",
        alt_text="A public wildlife photo of an owl for close observation.",
        research_query="great horned owl close view",
        preferred_sources=["smithsonian", "wikimedia"],
    )
    plan = build_visual_plan(intent, printable_type="worksheet")
    assert plan.role == "do"
    assert [unit.kind for unit in plan.units] == ["printable"]
    assert [unit.kind for unit in build_visual_plan(intent, printable_type="activity").units] == [
        "learning_photo",
        "printable",
    ]


def test_researched_photo_is_preserved_as_a_safe_model_visual() -> None:
    intent = {
        "kind": "researched_photo",
        "purpose": "observe",
        "knowledge_role": "supportive",
        "title": "Mars up close",
        "pedagogical_value": "Lets a child notice real landforms.",
        "caption": "What shapes and colors do you notice?",
        "alt_text": "A real public space-agency photograph of the surface of Mars.",
        "research_query": "NASA Mars surface panorama",
        "preferred_sources": ["nasa"],
    }
    normalized = normalize_response_visual("What does Mars look like?", {"visual": intent})
    assert normalized is not None
    assert normalized.kind == "researched_photo"


def test_research_service_checks_rights_reviews_and_reuses_private_cache(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "private" / "data.sqlite"
    init_db(db)
    image = Image.new("RGB", (640, 480), "#2A9D8F")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    candidate = ImageCandidate(
        source_name="wikimedia",
        source_item_id="42",
        title="A real owl",
        description="A great horned owl on a branch.",
        creator="Example Photographer",
        canonical_url="https://commons.wikimedia.org/wiki/File:Owl.jpg",
        media_url="https://upload.wikimedia.org/owl.jpg",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="Example Photographer / Wikimedia Commons",
        rights_status="verified",
    )

    class FakeSource:
        name = "wikimedia"

        def __init__(self) -> None:
            self.calls = 0

        def search(self, query: ImageResearchQuery, *, limit: int) -> list[ImageCandidate]:
            self.calls += 1
            return [candidate]

    class FakeReviewer:
        name = "fake-vision"
        model = "fake-vision-1"

        def complete(self, **kwargs):
            payload = ImageSafetyReview(
                verdict="pass",
                reasons=["safe and relevant"],
                subject_matches=True,
                observable_description="A brown owl perches on a branch with its large eyes facing the camera.",
                representation_type="photograph",
                reviewer="fake-vision-1",
            ).model_dump(mode="json")
            payload["reviewed_at"] = "not provided"
            payload["reviewer"] = "untrusted model value"
            return payload

    source = FakeSource()
    monkeypatch.setattr(research_module, "_get_bytes", lambda *args, **kwargs: (buffer.getvalue(), "image/jpeg"))
    service = ImageResearchService(db, tmp_path / "private", review_backend=FakeReviewer(), sources=[source])
    query = ImageResearchQuery(query="great horned owl close view", purpose="observe", preferred_sources=["wikimedia"])
    first = service.find(query)
    second = service.find(query)

    assert first == second
    assert source.calls == 1
    assert first.rights.license_name == "CC BY 4.0"
    assert first.safety.verdict == "pass"
    assert first.alt_text.startswith("A brown owl perches")
    assert first.safety.reviewer == "FakeReviewer"
    assert Path(first.local_path).is_relative_to((tmp_path / "private").resolve())
    assert Image.open(first.local_path).format == "PNG"

    with connect(db) as conn:
        conn.execute(
            "UPDATE researched_image_cache SET updated_at=? WHERE status='approved'",
            ((datetime.now(UTC) - timedelta(days=31)).isoformat(),),
        )
    refreshed = service.find(query)
    assert source.calls == 2
    assert refreshed.sha256 == first.sha256


@pytest.mark.parametrize(
    "flag",
    [
        "people_present",
        "frightening",
        "graphic_or_gory",
        "personal_information_visible",
        "watermark_or_advertising",
    ],
)
def test_pass_review_rejects_every_disqualifying_safety_flag(flag: str) -> None:
    with pytest.raises(ValueError, match="disqualifying"):
        ImageSafetyReview(
            verdict="pass",
            representation_type="photograph",
            **{flag: True},
        )


def test_pass_review_requires_subject_match() -> None:
    with pytest.raises(ValueError, match="disqualifying"):
        ImageSafetyReview(verdict="pass", representation_type="photograph", subject_matches=False)


def test_public_query_rejects_private_context_without_known_family_names(tmp_path: Path) -> None:
    for value in (
        "Lincoln Elementary butterflies",
        "birds at 123 oak street",
        "my friend therapy dog",
    ):
        with pytest.raises(ValueError):
            ImageResearchQuery(query=value, purpose="observe")

    class NeverCalled:
        name = "wikimedia"

        def search(self, query: ImageResearchQuery, *, limit: int):
            raise AssertionError("private query reached a collection")

    service = ImageResearchService(
        tmp_path / "private/db.sqlite",
        tmp_path / "private",
        review_backend=object(),
        sources=[NeverCalled()],
    )
    with pytest.raises(ValueError, match="proper name"):
        service.find(
            ImageResearchQuery(query="lincoln butterflies", purpose="observe"),
            private_source_text="A note from Lincoln about butterflies.",
        )


def test_research_candidate_and_vision_budgets_are_global(tmp_path: Path, monkeypatch) -> None:
    image = Image.new("RGB", (640, 480), "#2A9D8F")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    class ManyCandidates:
        name = "wikimedia"

        def search(self, query: ImageResearchQuery, *, limit: int):
            return [
                ImageCandidate(
                    source_name="wikimedia",
                    source_item_id=str(index),
                    title=f"Candidate {index}",
                    description="A generic candidate.",
                    creator="Example",
                    canonical_url=f"https://commons.wikimedia.org/wiki/File:Candidate_{index}.jpg",
                    media_url="https://upload.wikimedia.org/candidate.jpg",
                    license_name="CC BY 4.0",
                    license_url="https://creativecommons.org/licenses/by/4.0/",
                    attribution="Example / Wikimedia Commons",
                    rights_status="verified",
                )
                for index in range(20)
            ][:limit]

    class RejectingReviewer:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            return ImageSafetyReview(
                verdict="fail",
                subject_matches=False,
                representation_type="photograph",
            ).model_dump(mode="json")

    reviewer = RejectingReviewer()
    monkeypatch.setattr(research_module, "_get_bytes", lambda *args, **kwargs: (buffer.getvalue(), "image/jpeg"))
    service = ImageResearchService(
        tmp_path / "private/db.sqlite",
        tmp_path / "private",
        review_backend=reviewer,
        sources=[ManyCandidates()],
    )
    query = ImageResearchQuery(query="close owl portrait", purpose="observe")
    with pytest.raises(ValueError, match="no public image"):
        service.find(query)
    assert reviewer.calls == research_module.MAX_VISION_REVIEWS_PER_REQUEST
    with pytest.raises(ValueError, match="no approved public image"):
        service.find(query)
    assert reviewer.calls == research_module.MAX_VISION_REVIEWS_PER_REQUEST


def test_nasa_adapter_rejects_explicit_item_restrictions(monkeypatch) -> None:
    monkeypatch.setattr(
        research_module,
        "_get_json",
        lambda _url: {
            "collection": {
                "items": [
                    {
                        "data": [
                            {
                                "nasa_id": "restricted-1",
                                "title": "Restricted example",
                                "description": "A neutral description of the image.",
                                "copyright": "Example Studio",
                            }
                        ],
                        "links": [{"render": "image", "href": "https://images-assets.nasa.gov/restricted.jpg"}],
                    }
                ]
            }
        },
    )
    assert research_module.NasaImageSource().search(
        ImageResearchQuery(query="rocky mars surface", purpose="observe"), limit=4
    ) == []


def test_processed_science_image_requires_visible_context_note() -> None:
    candidate = ImageCandidate(
        source_name="nasa",
        source_item_id="mars-false-color",
        title="Enhanced Mars terrain",
        description="Colors were enhanced to reveal surface materials.",
        creator="NASA",
        canonical_url="https://images.nasa.gov/details/mars-false-color",
        media_url="https://images-assets.nasa.gov/image/mars-false-color/sample.jpg",
        license_name="NASA Media Usage Guidelines",
        license_url="https://www.nasa.gov/nasa-brand-center/images-and-media/",
        attribution="NASA",
        rights_status="policy_based",
    )
    query = ImageResearchQuery(query="enhanced Mars terrain", purpose="observe", preferred_sources=["nasa"])
    policy_rights = research_module._rights(candidate)
    assert policy_rights.rights_status == "policy_based"
    assert policy_rights.modifications_allowed is None
    assert policy_rights.commercial_use_allowed is None

    class Reviewer:
        def __init__(self, note: str) -> None:
            self.note = note

        def complete(self, **kwargs):
            return ImageSafetyReview(
                verdict="pass",
                subject_matches=True,
                observable_description="A cratered surface has blue and tan regions.",
                representation_type="processed_science_image",
                context_note=self.note,
            ).model_dump(mode="json")

    approved = research_module._review_image(candidate, query, b"review-bytes", Reviewer("Colors are enhanced."))
    assert approved.context_note == "Colors are enhanced."
    with pytest.raises(ValueError, match="context note"):
        research_module._review_image(candidate, query, b"review-bytes", Reviewer(""))

    class OutageReviewer:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            raise RuntimeError("provider unavailable")

    outage = OutageReviewer()
    with pytest.raises(research_module.ImageReviewUnavailable):
        research_module._review_image(candidate, query, b"review-bytes", outage)
    assert outage.calls == 1


def test_researched_visual_job_persists_strategy_and_visible_provenance(tmp_path: Path, monkeypatch) -> None:
    db = tmp_path / "private" / "data.sqlite"
    output = tmp_path / "private" / "output"
    init_db(db)
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,text,source,metadata_json,created_at,status)
               VALUES('evt_mars','child_question','What does Mars look like?','test','{}',?,'completed')""",
            (utcnow(),),
        )
    image_path = tmp_path / "reviewed.png"
    Image.new("RGB", (640, 480), "#D97852").save(image_path)
    digest = sha256(image_path.read_bytes()).hexdigest()
    rights = ImageRights(
        source_name="nasa",
        source_item_id="mars-1",
        canonical_url="https://images.nasa.gov/details/mars-1",
        creator="NASA",
        license_name="NASA Media Usage Guidelines",
        license_url="https://www.nasa.gov/nasa-brand-center/images-and-media/",
        attribution="NASA",
        rights_status="policy_based",
        modifications_allowed=None,
        commercial_use_allowed=None,
        metadata_sha256="a" * 64,
    )
    asset = ResearchedImageAsset(
        title="Mars surface",
        caption="A real view of the Martian surface.",
        alt_text="A public NASA image showing rocky ground on the surface of Mars.",
        content_role="observe",
        local_path=str(image_path),
        mime_type="image/png",
        width=640,
        height=480,
        byte_count=image_path.stat().st_size,
        sha256=digest,
        rights=rights,
        safety=ImageSafetyReview(
            verdict="pass",
            subject_matches=True,
            representation_type="photograph",
            reviewer="test",
        ),
    )
    monkeypatch.setattr("curiosity_engine.visuals.ImageResearchService.find", lambda self, query, **kwargs: asset)
    intent = VisualIntent(
        kind="researched_photo",
        purpose="observe",
        knowledge_role="supportive",
        title="Mars up close",
        pedagogical_value="Supports observation of real landforms.",
        caption="Look closely.",
        alt_text="A real public photo of the surface of Mars for close observation.",
        research_query="NASA Mars surface",
        preferred_sources=["nasa"],
    )
    job_id = enqueue_response_visual(db, event_id="evt_mars", visual=intent, mode="deterministic")
    assert job_id
    result = process_visual_jobs(db, output, visual_qa_backend=object())
    assert result[0]["status"] == "completed"
    with connect(db) as conn:
        row = conn.execute("SELECT strategy,caption,provenance_json FROM visual_assets WHERE job_id=?", (job_id,)).fetchone()
    assert row["strategy"] == "researched"
    assert "Source: NASA" in row["caption"]
    assert "images.nasa.gov/details/mars-1" in row["provenance_json"]


def test_matching_choices_are_stably_shuffled_off_the_answer_rows() -> None:
    spec = WorksheetSpec(
        title="Animal Homes",
        target_grade="1st",
        learning_objective="Match animals with their homes.",
        estimated_minutes=8,
        parent_effort="low",
        trust_tier="B",
        accessibility_text="A two-column matching page.",
        archetype="match_two_columns",
        directions="Draw a line from each animal to its home.",
        tasks=[
            _task("match", "Animals", ["owl", "frog", "bee"]),
            _task("match", "Homes", ["tree", "pond", "hive"]),
        ],
    )
    original = spec.tasks[1].choices
    first = _shuffled_matching_choices(spec, original)
    assert first != original
    assert first == _shuffled_matching_choices(spec, original)
    assert sorted(first) == sorted(original)
