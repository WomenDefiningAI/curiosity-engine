from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from curiosity_engine.artifact_validation import render_pdf_preview
from curiosity_engine.artifacts import deterministic_visual_qa
from curiosity_engine.contracts import (
    ActivitySpec,
    ChallengePageReview,
    ChallengeSpec,
    ChallengeVisualBrief,
    ReferencePanel,
    ReferenceSpec,
    WorksheetSpec,
)
from curiosity_engine.db import connect, init_db, jdump, jload, utcnow
from curiosity_engine.graph import add_child
from curiosity_engine.learning_artifacts import (
    LearningArtifactService,
    _pdf_safe_text,
    render_learning_pdf,
    validate_learning_artifact,
)
from curiosity_engine.openai_image_backend import GeneratedImage
from curiosity_engine.reasoning import StubBackend


class _UnusableChallengeBackend(StubBackend):
    name = "unusable-challenge"

    def __init__(self) -> None:
        self.challenge_attempts = 0

    def complete(self, *, role, system, payload, response_model):
        if role == "critic_plan_fidelity":
            raise AssertionError("interactive challenge reached the heavyweight final model critic")
        if response_model is ChallengeSpec:
            self.challenge_attempts += 1
            return {
                "title": "Question Lab",
                "target_grade": "1st",
                "learning_objective": "Notice and test.",
                "estimated_minutes": 10,
                "accessibility_text": "A one-page challenge.",
                "scenario": "A question needs an investigation.",
                "goal": "Turn the mission question into a test.",
                "constraints": ["Change one thing."],
                "materials": ["paper", "pencil"],
                "steps": ["Predict.", "Try it.", "Write what happened."],
                "hints": ["Look closely.", "Try again."],
                "evidence_of_completion": "Show what happened.",
                "reflection": "What did you notice?",
            }
        return super().complete(
            role=role,
            system=system,
            payload=payload,
            response_model=response_model,
        )


class _MisleadingVisualThemeBackend(StubBackend):
    name = "misleading-visual-theme"

    def complete(self, *, role, system, payload, response_model):
        if response_model is ChallengeSpec:
            return {
                "title": "Soda Bubble Detective",
                "target_grade": "1st",
                "learning_objective": "Notice where bubbles form in soda.",
                "estimated_minutes": 8,
                "story_theme": "invisible-gas investigation",
                "accessibility_text": "A short soda-bubble observation challenge.",
                "scenario": "A glass of soda is making tiny bubbles. Look closely to find where they appear.",
                "goal": "Observe the soda twice and draw one bubble clue.",
                "constraints": ["A grown-up pours the soda.", "Look without shaking the glass."],
                "materials": ["clear glass of soda", "paper", "crayon"],
                "steps": [
                    "Predict where the first bubbles will appear.",
                    "Watch the soda for ten slow counts.",
                    "Draw the place where you noticed bubbles.",
                ],
                "hints": ["Look at the sides and bottom.", "Notice whether every bubble starts in the same place."],
                "evidence_of_completion": "Show one drawing of a bubble clue.",
                "evidence_layout": "writing_lines",
                "evidence_rows": [],
                "reflection": "What did the soda bubbles do?",
            }
        return super().complete(
            role=role,
            system=system,
            payload=payload,
            response_model=response_model,
        )


class _SyntheticChallengeImageBackend:
    name = "openai"
    model = "gpt-image-2-test"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> GeneratedImage:
        self.prompts.append(prompt)
        image = Image.new("RGB", (1536, 1024), "#fff6dc")
        image.paste((242, 107, 91), (300, 260, 1236, 764))
        stream = BytesIO()
        image.save(stream, format="PNG")
        return GeneratedImage(stream.getvalue(), model=self.model, request_id="image-request-1")


class _RenderedPageReviewBackend(_UnusableChallengeBackend):
    name = "openai"
    model = "gpt-5.6-terra-test"

    def __init__(self) -> None:
        super().__init__()
        self.review_payload: dict | None = None

    def complete(self, *, role, system, payload, response_model):
        if response_model is ChallengePageReview:
            self.review_payload = payload
            return {
                "usefulness": "4",
                "engagement": 3,
                "factuality": 5,
                "learning_value": 4,
                "kid_usability": 4,
                "visual_integrity": 2,
                "art_verdict": "replace",
                "critical_defects": ["Decorative art suggests the result."],
                "improvements": [
                    "Use neutral objects that do not reveal the outcome.",
                    "Keep the action sequence visible.",
                    "Keep both prediction spaces equal.",
                    "Preserve a distinct reveal space.",
                    "Keep reading load short.",
                    "This sixth suggestion should be clipped.",
                ],
                "provider_note": "This harmless extra field should be ignored.",
            }
        return super().complete(role=role, system=system, payload=payload, response_model=response_model)


def _assert_three_panel_visual(preview_path: str) -> None:
    with Image.open(preview_path).convert("RGB") as preview:
        width, height = preview.size
        visual_band = preview.crop((int(width * 0.08), int(height * 0.215), int(width * 0.92), int(height * 0.36)))
        panel_width = visual_band.width // 3
        concept_panels = [
            visual_band.crop((index * panel_width + 8, 8, (index + 1) * panel_width - 8, visual_band.height - 8))
            for index in range(3)
        ]
    for panel in concept_panels:
        pixels = list(panel.get_flattened_data())
        foreground_pixels = sum(min(pixel) < 200 and max(pixel) - min(pixel) >= 8 for pixel in pixels)
        assert foreground_pixels / len(pixels) > 0.01


def test_known_helium_challenge_delivers_without_model_generation_or_final_critic(
    tmp_path: Path,
) -> None:
    db = tmp_path / "curiosity.db"
    init_db(db)
    add_child(db, "kid-a", "Kid A", grade="1st")
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_helium','context_input','kid-a',?,'test',?,?,'completed')""",
            (
                "We were talking about helium during science time. What might happen if we added helium to cups "
                "(instead of balloons) and where could someone get helium?",
                jdump({"evidence_eligible": False, "learning_scope": "context_only"}),
                utcnow(),
            ),
        )
    backend = _UnusableChallengeBackend()
    artifact = LearningArtifactService(db, tmp_path / "output", backend=backend).create_from_context_event(
        event_id="evt_helium",
        artifact_type="challenge",
        revision=(
            "Create a playful, child-ready challenge from this saved moment. Include a useful child-facing visual, "
            "keep the reading load age-appropriate, and do not treat the observation as proof of interest, skill, "
            "or mastery."
        ),
    )

    assert backend.challenge_attempts == 0
    assert artifact["title"] == "Invisible Air Escape"
    assert Path(artifact["pdf_path"]).is_file()
    assert Path(artifact["preview_path"]).is_file()
    with connect(db) as conn:
        row = conn.execute(
            "SELECT spec_json,validation_json FROM artifacts WHERE id=?",
            (artifact["artifact_id"],),
        ).fetchone()
    spec = ChallengeSpec.model_validate(jload(row["spec_json"]))
    child_actions = " ".join(spec.steps).casefold()
    for required_action in ("paper", "cup", "bowl", "water", "bubbles", "tilt", "pencil", "crayon"):
        assert required_action in child_actions
    assert spec.evidence_rows == ["Cup straight down", "Cup tilted"]
    assert "no helium tank is needed" in " ".join(spec.constraints).casefold()
    assert jload(row["validation_json"])["deterministic_visual_qa"]["verdict"] == "pass"
    assert validate_learning_artifact(spec) == []
    assert LearningArtifactService._text_quality_errors(spec) == []
    assert spec.story_theme == "invisible-gas investigation"
    assert spec.visual_brief is not None
    assert spec.visual_brief.teaching_diagram == "air_escape_test"


def test_giant_ice_context_becomes_a_fast_visual_shape_quest(tmp_path: Path) -> None:
    db = tmp_path / "curiosity.db"
    init_db(db)
    add_child(db, "kid-a", "Kid A", grade="1st")
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_ice','context_input','kid-a',?,'test',?,?,'completed')""",
            (
                "Kid A is still making ice cubes for fun - giant ones in cups.",
                jdump({"evidence_eligible": False, "learning_scope": "context_only"}),
                utcnow(),
            ),
        )
    backend = _UnusableChallengeBackend()

    artifact = LearningArtifactService(db, tmp_path / "output", backend=backend).create_from_context_event(
        event_id="evt_ice",
        artifact_type="challenge",
        revision="Create a playful, child-ready challenge from this saved moment.",
    )

    assert backend.challenge_attempts == 0
    assert artifact["title"] == "Ice Mold Mystery"
    with connect(db) as conn:
        row = conn.execute(
            "SELECT spec_json,validation_json FROM artifacts WHERE id=?",
            (artifact["artifact_id"],),
        ).fetchone()
    spec = ChallengeSpec.model_validate(jload(row["spec_json"]))
    assert spec.story_theme == "giant-ice-shape quest"
    assert spec.visual_brief is not None
    assert spec.visual_brief.answer_timing == "withhold_until_observation"
    assert spec.estimated_minutes == 12
    assert spec.evidence_layout == "design_plan"
    assert spec.evidence_rows == ["WIDE CUP GUESS", "TALL CUP GUESS", "ICE REVEAL"]
    assert len(spec.constraints) == 3
    assert len(spec.materials) == 5
    assert len(spec.steps) == 4
    assert len(spec.hints) == 2
    assert "same scoop" in " ".join([*spec.constraints, *spec.steps]).casefold()
    assert validate_learning_artifact(spec) == []
    assert jload(row["validation_json"])["deterministic_visual_qa"]["verdict"] == "pass"
    assert "expands" not in spec.model_dump_json().casefold()


def test_challenge_art_is_private_cached_and_never_receives_family_text(tmp_path: Path) -> None:
    backend = _SyntheticChallengeImageBackend()
    service = LearningArtifactService(
        tmp_path / "curiosity.db",
        tmp_path / "output",
        backend=StubBackend(),
        image_backend=backend,
    )
    spec = service._giant_ice_challenge_spec(
        response={},
        grade="1st",
        source_event_id="evt-private-family-moment",
    )

    first = service._prepare_challenge_illustration(spec, authorization_preflight=None)
    second = service._prepare_challenge_illustration(spec, authorization_preflight=None)

    assert len(backend.prompts) == 1
    assert "evt-private-family-moment" not in backend.prompts[0]
    assert "Kid A" not in backend.prompts[0]
    assert "no words, letters, numbers" in backend.prompts[0].casefold()
    assert first.visual_brief is not None
    assert first.illustration is not None
    assert second.illustration is not None
    assert first.illustration.local_path == second.illustration.local_path
    assert second.illustration.request_id == "image-request-1"
    assert Path(first.illustration.local_path).is_file()
    rendered = render_learning_pdf(first, tmp_path / "hybrid-ice.pdf")
    assert rendered.is_file()

    tampered = Image.new("RGB", (1536, 1024), "#3d73a8")
    tampered_stream = BytesIO()
    tampered.save(tampered_stream, format="PNG")
    Path(first.illustration.local_path).write_bytes(tampered_stream.getvalue())
    repaired = service._prepare_challenge_illustration(spec, authorization_preflight=None)
    assert len(backend.prompts) == 2
    assert repaired.illustration is not None
    assert repaired.illustration.sha256 == first.illustration.sha256

    class FailingImageBackend:
        name = "openai"
        model = "gpt-image-2-test"

        def generate(self, _prompt: str) -> GeneratedImage:
            raise RuntimeError("provider unavailable")

    fallback = LearningArtifactService(
        tmp_path / "fallback.db",
        tmp_path / "fallback-output",
        backend=StubBackend(),
        image_backend=FailingImageBackend(),
    )._prepare_challenge_illustration(spec, authorization_preflight=None)
    assert fallback.visual_brief is not None
    assert fallback.illustration is None
    assert render_learning_pdf(fallback, tmp_path / "hybrid-ice-fallback.pdf").is_file()


def test_rendered_page_review_can_remove_bad_generated_art_without_losing_challenge(
    tmp_path: Path,
) -> None:
    db = tmp_path / "curiosity.db"
    init_db(db)
    add_child(db, "kid-a", "Kid A", grade="1st")
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_review_ice','context_input','kid-a',?,'test',?,?,'completed')""",
            (
                "Kid A is still making ice cubes for fun - giant ones in cups.",
                jdump({"evidence_eligible": False, "learning_scope": "context_only"}),
                utcnow(),
            ),
        )
    reviewer = _RenderedPageReviewBackend()
    # Retain coverage of stored legacy decorative challenges. The live OpenAI
    # route now designs functional picture pages even for this familiar topic.
    reviewer.name = "legacy-rendered-page-review"
    artifact = LearningArtifactService(
        db,
        tmp_path / "output",
        backend=reviewer,
        image_backend=_SyntheticChallengeImageBackend(),
    ).create_from_context_event(
        event_id="evt_review_ice",
        artifact_type="challenge",
        revision="Make a visual challenge.",
    )

    with connect(db) as conn:
        row = conn.execute(
            "SELECT spec_json,validation_json FROM artifacts WHERE id=?",
            (artifact["artifact_id"],),
        ).fetchone()
    persisted = ChallengeSpec.model_validate(jload(row["spec_json"]))
    validation = jload(row["validation_json"])
    assert reviewer.review_payload is not None
    assert reviewer.review_payload["image_data_urls"][0].startswith("data:image/png;base64,")
    assert persisted.visual_brief is not None
    assert persisted.illustration is None
    assert validation["generated_art_fallback"] is True
    assert validation["family_usefulness_review"]["art_verdict"] == "fallback"
    assert len(validation["family_usefulness_review"]["improvements"]) == 5
    assert Path(artifact["pdf_path"]).is_file()


def test_paper_airplane_wing_tip_prompt_gets_a_real_fair_test_and_diagram(tmp_path: Path) -> None:
    backend = _UnusableChallengeBackend()
    service = LearningArtifactService(
        tmp_path / "curiosity.db",
        tmp_path / "output",
        backend=backend,
        image_backend=None,
    )
    spec = service._design(
        "challenge",
        question="How do bent-up or bent-down wing tips change a paper airplane's flight?",
        response={"resource_refs": []},
        grade="1st",
        source_event_id="evt-plane",
        revision="Make a challenge that tests it.",
    )
    assert isinstance(spec, ChallengeSpec)
    assert backend.challenge_attempts == 0
    assert spec.title == "Wing-Tip Flight Lab"
    assert spec.evidence_rows == ["TIPS UP", "WINGS FLAT", "TIPS DOWN"]
    assert "same airplane" in " ".join(spec.constraints).casefold()
    assert "two finger-widths" in " ".join(spec.steps).casefold()
    prepared = service._prepare_challenge_illustration(spec, authorization_preflight=None)
    assert prepared.visual_brief is not None
    assert prepared.visual_brief.teaching_diagram == "paper_airplane_wing_tips"
    assert prepared.visual_brief.answer_timing == "withhold_until_observation"
    output = render_learning_pdf(prepared, tmp_path / "wing-tip-flight-lab.pdf")
    preview = render_pdf_preview(output, tmp_path / "wing-tip-flight-lab.png")
    assert deterministic_visual_qa(preview).verdict == "pass"


def test_giant_ice_fast_path_does_not_replace_other_ice_and_cup_questions(tmp_path: Path) -> None:
    db = tmp_path / "curiosity.db"
    init_db(db)
    add_child(db, "kid-a", "Kid A", grade="1st")
    questions = (
        "We put ice cubes in big cups and the outside got wet. Why?",
        "Why do ice cubes make big cups sweat?",
        "Can a large ice cube fit inside this cup?",
        "What makes an ice cube melt faster in a cup?",
        "Why does a big cup keep ice from melting?",
        "Can a large container of ice keep food cold?",
        "Kid A made a big drawing of ice cubes in cups.",
    )

    for index, question in enumerate(questions):
        assert not LearningArtifactService._is_giant_ice_cup_context(question)
        event_id = f"evt_ice_control_{index}"
        with connect(db) as conn:
            conn.execute(
                """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                   VALUES(?,'context_input','kid-a',?,'test',?,?,'completed')""",
                (
                    event_id,
                    question,
                    jdump({"evidence_eligible": False, "learning_scope": "context_only"}),
                    utcnow(),
                ),
            )
        backend = _UnusableChallengeBackend()
        artifact = LearningArtifactService(
            db,
            tmp_path / f"output-{index}",
            backend=backend,
        ).create_from_context_event(
            event_id=event_id,
            artifact_type="challenge",
            revision="Make a short child-ready challenge about this question.",
        )

        assert backend.challenge_attempts == 2
        assert artifact["title"] != "Giant Ice Shape Quest"
        with connect(db) as conn:
            spec_json = conn.execute(
                "SELECT spec_json FROM artifacts WHERE id=?",
                (artifact["artifact_id"],),
            ).fetchone()[0]
        spec = ChallengeSpec.model_validate(jload(spec_json))
        assert spec.story_theme != "giant-ice-shape quest"


def test_helium_cup_fallback_does_not_replace_non_containment_questions(tmp_path: Path) -> None:
    db = tmp_path / "curiosity.db"
    init_db(db)
    add_child(db, "kid-a", "Kid A", grade="1st")
    for event_id, question in (
        ("evt_lift", "Could a helium balloon lift an open paper cup?"),
        ("evt_hyphenated", "Could we put the helium-filled balloon next to an open cup?"),
        ("evt_filled", "Place the balloon filled with helium beside a paper cup."),
        ("evt_plural", "What happens if we put two helium balloons into open cups?"),
        ("evt_compare", "Can we place helium balloons next to cups and compare their sizes?"),
        (
            "evt_negated_add",
            "We are not going to add helium to cups; compare a helium balloon with an air balloon instead.",
        ),
        (
            "evt_negated_put",
            "Do not put helium into an open cup. Could a helium balloon lift the cup?",
        ),
        (
            "evt_cancelled",
            "Don't make a challenge about what would happen if we add helium to cups; compare balloon sizes instead.",
        ),
        (
            "evt_changed_mind",
            "What would happen if we add helium to cups? Actually, compare helium and air balloons instead.",
        ),
        (
            "evt_reported",
            "The worksheet asks, 'What would happen if we add helium to cups?' We want a balloon-size comparison.",
        ),
        (
            "evt_not_asking",
            "We are not asking what would happen if we tried to add helium to cups; compare the balloons instead.",
        ),
        ("evt_holder", "What would happen if we put helium into a cup holder?"),
        ("evt_shaped", "What would happen if we put helium into a cup-shaped balloon?"),
        ("evt_ship", "What would happen if we put helium into a container ship?"),
    ):
        with connect(db) as conn:
            conn.execute(
                """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
                   VALUES(?,'context_input','kid-a',?,'test',?,?,'completed')""",
                (
                    event_id,
                    question,
                    jdump({"evidence_eligible": False, "learning_scope": "context_only"}),
                    utcnow(),
                ),
            )
        backend = _UnusableChallengeBackend()
        artifact = LearningArtifactService(
            db,
            tmp_path / event_id,
            backend=backend,
        ).create_from_context_event(
            event_id=event_id,
            artifact_type="challenge",
            revision=(
                "Create a playful, child-ready challenge from this saved moment. Include a useful child-facing "
                "visual, keep the reading load age-appropriate, and do not treat the observation as proof of "
                "interest, skill, or mastery."
            ),
        )

        assert backend.challenge_attempts == 2
        assert artifact["title"] != "Can a Cup Hold a Gas?"
        with connect(db) as conn:
            spec_json = conn.execute(
                "SELECT spec_json FROM artifacts WHERE id=?",
                (artifact["artifact_id"],),
            ).fetchone()[0]
        assert "bubbles" not in spec_json.casefold()


def test_model_cannot_select_the_code_owned_helium_diagram_for_an_unrelated_challenge(tmp_path: Path) -> None:
    service = LearningArtifactService(
        tmp_path / "curiosity.db",
        tmp_path / "output",
        backend=_MisleadingVisualThemeBackend(),
    )

    spec = service._design(
        "challenge",
        question="Why does soda fizz and where do its bubbles appear?",
        response={
            "hook": "Soda contains dissolved carbon dioxide gas.",
            "show": "Bubbles can form and rise after the container opens.",
            "ask": "Where do you notice the first bubbles?",
            "nugget": "A close observation can reveal where bubbles start.",
            "physical_extension": None,
            "resource_refs": [],
        },
        grade="1st",
        source_event_id="evt_soda",
        revision="Make a short soda-bubble observation challenge.",
    )

    assert isinstance(spec, ChallengeSpec)
    assert spec.title == "Soda Bubble Detective"
    assert spec.story_theme == "curiosity mission"
    contaminated = spec.model_copy(
        update={
            "visual_brief": ChallengeVisualBrief(
                subject_key="ice_lab",
                teaching_diagram="ice_mold_mystery",
                answer_timing="withhold_until_observation",
                response_layout="paired_predictions",
            )
        }
    )
    assert "challenge visual brief does not match its code-owned story theme" in validate_learning_artifact(
        contaminated
    )
    cleared = service._prepare_challenge_illustration(contaminated, authorization_preflight=None)
    assert cleared.visual_brief is None
    assert cleared.illustration is None
    pdf_path = render_learning_pdf(spec, tmp_path / "soda.pdf")
    preview_path = render_pdf_preview(pdf_path, tmp_path / "soda.png")
    _assert_three_panel_visual(str(preview_path))

    fake_diagram = spec.model_copy(update={"scenario": "ICE CLUE MAP HALF CUP -> TALL CUP -> BIG ICE DIAGRAM"})
    assert "challenge scenario cannot imitate a visual; the renderer owns diagrams" in validate_learning_artifact(
        fake_diagram
    )


def test_code_owned_artifact_instruction_keeps_the_visible_subject_as_the_brief() -> None:
    brief = LearningArtifactService._artifact_brief(
        "A child is starting a LEGO creature-building class with two teammates.",
        {"answer": "Pick one simple creature build idea to share when the team arrives."},
        "Create a playful, child-ready printable that extends this thread without repeating it.",
    )

    assert "LEGO creature-building class" in brief
    assert "Pick one simple creature build idea" in brief
    assert "extends this thread" not in brief


def test_fallback_does_not_mistake_choice_before_class_or_after_school_for_ice_comparison() -> None:
    spec = LearningArtifactService._fallback_spec(
        "challenge",
        "A child will start LEGO creature class after school with a sibling and friend.",
        {"answer": "Before class, choose one build idea for the team."},
        "1st",
        "evt_build_team",
        revision="Create a playful, child-ready printable that extends this thread without repeating it.",
    )

    assert isinstance(spec, ChallengeSpec)
    assert spec.title == "Build-Team Idea Mission"
    rendered = spec.model_dump_json().casefold()
    assert "water to ice" not in rendered
    assert "version a" not in rendered
    assert "named feature" not in rendered
    assert LearningArtifactService._text_quality_errors(spec) == []


def test_context_idea_without_question_back_does_not_invent_comparison_intent(tmp_path: Path) -> None:
    db = tmp_path / "curiosity.db"
    init_db(db)
    add_child(db, "kid-a", "Kid A", grade="1st")
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt-context','context_input','kid-a',?,'parent',?,?,'completed')""",
            (
                "Kid A will start a LEGO creature class after school with a sibling and friend.",
                jdump({"evidence_eligible": False}),
                now,
            ),
        )
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt-idea','child_question','kid-a',?,'slack_parent_context_choice',?,?,'completed')""",
            (
                "Offer exactly one playful idea from this parent-shared moment.",
                jdump(
                    {
                        "context_source_event_id": "evt-context",
                        "related_event_id": "evt-context",
                        "episode_relation": "context_derived",
                    }
                ),
                now,
            ),
        )
        conn.execute(
            """INSERT INTO responses(event_id,run_id,workflow,status,output_json,created_at,updated_at)
               VALUES('evt-idea',NULL,'quick_answer','completed',?,?,?)""",
            (
                jdump(
                    {
                        "answer": "Before class, choose one simple creature build idea to share with the team.",
                        "notice": None,
                        "question_back": None,
                    }
                ),
                now,
                now,
            ),
        )

    artifact = LearningArtifactService(
        db,
        tmp_path / "output",
        backend=StubBackend(),
    ).create_from_event(
        event_id="evt-idea",
        artifact_type="challenge",
        revision="Create a playful, child-ready printable that extends this thread without repeating it.",
    )

    with connect(db) as conn:
        spec = jload(
            conn.execute("SELECT spec_json FROM artifacts WHERE id=?", (artifact["artifact_id"],)).fetchone()[0]
        )
    assert spec["title"] == "Build-Team Idea Mission"
    rendered = jdump(spec).casefold()
    assert "offer exactly one playful" not in rendered
    assert "what could we notice or compare" not in rendered
    assert "version a" not in rendered


def test_challenge_rows_leave_trial_columns_to_the_renderer() -> None:
    spec = ChallengeSpec(
        title="Two Tiny Builds",
        target_grade="1st",
        learning_objective="Compare two small builds.",
        estimated_minutes=10,
        parent_effort="low",
        trust_tier="B",
        story_theme="build mission",
        accessibility_text="A two-row build comparison.",
        source_event_id="evt-build",
        scenario="Two tiny builds need a careful tester.",
        goal="Try both builds twice and choose one result.",
        constraints=["Use the same gentle test each time."],
        materials=["building blocks"],
        steps=["Build the first idea.", "Test it twice.", "Build and test the second idea."],
        hints=["Keep the test gentle.", "Watch the base."],
        evidence_of_completion="Mark both trials and circle one result.",
        evidence_rows=["Home - Test 1", "Training spot"],
        reflection="What changed?",
    )

    assert "challenge evidence rows must name versions only; the renderer owns trial columns" in (
        validate_learning_artifact(spec)
    )
    wrong_trial_count = spec.model_copy(
        update={"constraints": ["Test each build three times from the same starting spot."]}
    )
    assert "trial-grid challenge must use exactly two trials because the renderer supplies two columns" in (
        validate_learning_artifact(wrong_trial_count)
    )
    assert _pdf_safe_text("stayed up ○ fell down ○") == "stayed up [ ] fell down [ ]"


def test_design_plan_challenge_supplies_child_work_areas(tmp_path: Path) -> None:
    spec = ChallengeSpec(
        title="Creature Build Team",
        target_grade="1st",
        learning_objective="Turn one build idea into a plan the team can discuss.",
        estimated_minutes=10,
        parent_effort="low",
        trust_tier="B",
        story_theme="workshop",
        accessibility_text="A large drawing box and three labeled planning boxes.",
        source_event_id="evt-design",
        scenario="A creature-building team needs one clear plan.",
        goal="Draw the build and choose a job for each teammate.",
        constraints=["Use one main idea."],
        materials=["pencil or crayons"],
        steps=["Draw the build.", "Add its important part.", "Choose the team jobs."],
        hints=["Start with the biggest shape.", "Use arrows to show movement."],
        evidence_of_completion="Complete the drawing and the three planning boxes.",
        evidence_layout="design_plan",
        evidence_rows=["DRAW THE BUILD", "MY JOB", "TEAM JOB", "BUILD FIRST"],
        reflection="What would the team change?",
    )

    assert validate_learning_artifact(spec) == []
    output = render_learning_pdf(spec, tmp_path / "design-plan.pdf")
    assert output.is_file()

    count_mismatch = spec.model_copy(update={"accessibility_text": "A page with three big boxes."})
    assert "design-plan accessibility description contradicts the rendered work-area count" in (
        validate_learning_artifact(count_mismatch)
    )
    broken_hint = spec.model_copy(update={"hints": ["Choose one job, and one.", "Draw the largest shape first."]})
    assert LearningArtifactService._text_quality_errors(broken_hint) == [
        "artifact contains an incomplete child-facing phrase"
    ]
    dangling_preposition = spec.model_copy(
        update={"hints": ["Make sure the bridge reaches from one tower to.", "Keep the supports still."]}
    )
    assert LearningArtifactService._text_quality_errors(dangling_preposition) == [
        "artifact contains an incomplete child-facing phrase"
    ]


def test_fallback_visual_comparison_keeps_every_named_wing_shape() -> None:
    spec = LearningArtifactService._fallback_spec(
        "worksheet",
        "How do airplane wings change how a paper airplane flies?",
        {"nugget": "Wing shape changes how air pushes on a plane."},
        "1st",
        "evt_wings",
        revision=(
            "Create a useful child-facing visual comparison page, not decorative art. "
            "Show wide, narrow, swept, and straight wing variations instead."
        ),
    )

    assert isinstance(spec, WorksheetSpec)
    assert spec.title == "Wing Shape Comparison"
    rendered_contract = spec.model_dump_json().casefold()
    for label in ("wide", "narrow", "swept", "straight"):
        assert label in rendered_contract
    assert validate_learning_artifact(spec) == []


def test_fallback_bounds_a_long_inherited_learning_objective() -> None:
    spec = LearningArtifactService._fallback_spec(
        "activity",
        "What changes when water freezes around an object in a cup?",
        {"nugget": "Water changes into solid ice around the object. " * 20},
        "1st",
        "evt_long_objective",
        revision="show the activity steps",
    )

    assert len(spec.learning_objective) <= 280
    assert validate_learning_artifact(spec) == []


def test_breathing_comparison_fallback_shows_both_lung_size_changes() -> None:
    spec = LearningArtifactService._fallback_spec(
        "reference",
        "What happens when I breathe in and breathe out?",
        {"answer": "Air moves in and out while the diaphragm helps make room."},
        "1st",
        "evt_breathing",
        revision="Make a simple side-by-side diagram comparing breathing in and breathing out.",
    )

    assert isinstance(spec, ReferenceSpec)
    assert spec.title == "Breathing In / Breathing Out"
    rendered = spec.model_dump_json().casefold()
    assert "lungs: fill and get bigger" in rendered
    assert "lungs: empty and get smaller" in rendered
    assert validate_learning_artifact(spec) == []


def test_air_takes_space_fallback_is_a_complete_two_trial_activity_card() -> None:
    spec = LearningArtifactService._fallback_spec(
        "activity",
        "If air is there, why can't I see it? An empty cup must have nothing in it.",
        {"answer": "Clear air is invisible, but it still takes up space."},
        "1st",
        "evt_air_cup",
        revision=(
            "Make a one-page grown-up-and-child activity card using an upside-down cup and water to test whether "
            "air takes up space."
        ),
        require_printable=True,
    )

    assert isinstance(spec, ActivitySpec)
    assert spec.title == "Air Is In the Cup"
    rendered = spec.model_dump_json().casefold()
    for phrase in ("straight down", "tip one edge", "bubbles", "water enter", "paper stay dry"):
        assert phrase in rendered
    assert spec.printable is not None
    assert len(spec.printable.pieces) == 4
    assert validate_learning_artifact(spec) == []


def test_air_takes_space_fallback_uses_reviewed_answer_when_parent_says_that_idea() -> None:
    spec = LearningArtifactService._fallback_spec(
        "activity",
        "She said an empty cup must have nothing in it.",
        {"answer": "The cup looks empty, but it still contains air that takes up space."},
        "1st",
        "evt_air_cup_followup",
        revision="Make a one-page grown-up-and-child activity card to test that idea with ordinary things at home.",
        require_printable=True,
    )

    assert isinstance(spec, ActivitySpec)
    assert spec.title == "Air Is In the Cup"
    assert spec.printable is not None
    assert validate_learning_artifact(spec) == []


def test_handwashing_fallback_is_a_low_reading_five_step_card() -> None:
    spec = LearningArtifactService._fallback_spec(
        "activity",
        "Milo is learning handwashing.",
        {"answer": "Wash with water and soap, then rinse and dry."},
        "1st",
        "evt_handwashing",
        revision="Make an activity card he can practice with; a grown-up checks at the end.",
        require_printable=True,
    )

    assert isinstance(spec, ActivitySpec)
    assert spec.title == "Wash Hands: 5 Steps"
    assert spec.printable is not None
    rendered = spec.model_dump_json().casefold()
    for word in ("wet", "soap", "scrub", "rinse", "dry", "grown-up check"):
        assert word in rendered
    assert validate_learning_artifact(spec) == []


def test_soap_pepper_fallback_is_a_complete_markable_checklist() -> None:
    spec = LearningArtifactService._fallback_spec(
        "activity",
        "Why does soap matter when we wash hands?",
        {
            "answer": "Pepper is not germs; soap changes how the water surface pulls.",
            "physical_extension": {
                "title": "Soap and pepper clue",
                "materials": ["shallow dish", "water", "ground black pepper", "liquid soap"],
                "instructions": ["Touch plain water once.", "Reset, then touch once with soap."],
            },
        },
        "1st",
        "evt_soap_pepper",
        revision="Make a one-page child-markable checklist for the shallow dish, water, pepper, and soap activity.",
        require_printable=True,
    )

    assert isinstance(spec, ActivitySpec)
    assert spec.title == "Soap and Pepper: Step Checklist"
    assert spec.printable is not None
    rendered = spec.model_dump_json().casefold()
    for phrase in ("black pepper", "water touch", "soap touch", "pepper is not germs", "do not taste"):
        assert phrase in rendered
    assert validate_learning_artifact(spec) == []


def test_water_wheel_fallback_uses_unambiguous_tangential_push_arrows() -> None:
    spec = LearningArtifactService._fallback_spec(
        "reference",
        "Why can water hitting near the top or bottom make a water wheel turn?",
        {"answer": "Moving water pushes an off-center paddle and turns the wheel."},
        "1st",
        "evt_water_wheel",
        revision="Make a side-by-side diagram with short labels and turn arrows.",
    )

    assert isinstance(spec, ReferenceSpec)
    assert spec.title == "Water Wheel: Top / Bottom"
    rendered = spec.model_dump_json()
    assert "PUSH → above" in rendered
    assert "PUSH → below" in rendered
    assert not any(symbol in rendered for symbol in ("↗", "↘", "↖", "↙"))
    assert validate_learning_artifact(spec) == []


def test_reference_page_is_a_compact_visual_not_a_worksheet() -> None:
    spec = LearningArtifactService._fallback_spec(
        "reference",
        "Can you compare a smooth stone and rough tree bark?",
        {"nugget": "Surfaces can have different textures."},
        "1st",
        "evt_texture",
        revision="Make a side-by-side diagram comparing a smooth stone and rough tree bark.",
    )

    assert isinstance(spec, ReferenceSpec)
    assert spec.layout == "comparison"
    assert [panel.heading for panel in spec.panels] == ["Smooth Stone", "Rough Tree Bark"]
    assert "tasks" not in spec.model_dump(mode="json")
    assert validate_learning_artifact(spec) == []


def test_reference_fallback_carries_reviewed_causal_facts_into_layer_panels() -> None:
    spec = LearningArtifactService._fallback_spec(
        "reference",
        "Why can an animal stay warm in the cold?",
        {
            "answer": (
                "Penguin feathers trap air close to the body, slowing heat from escaping. "
                "A fluffy winter coat also holds pockets of air that slow heat loss."
            )
        },
        "1st",
        "evt_layers",
        revision=(
            "Make a simple side-by-side diagram for a first grader comparing a penguin's feather layers "
            "with a winter coat."
        ),
    )

    assert isinstance(spec, ReferenceSpec)
    assert [panel.heading for panel in spec.panels] == ["Penguin Feather Layers", "Winter Coat"]
    assert all(panel.cue == "segments" for panel in spec.panels)
    assert spec.panels[0].segment_labels == ["OUTSIDE FEATHERS", "SOFT FEATHERS HOLD AIR", "FAT", "BODY"]
    assert "hold warm air" in " ".join(spec.panels[0].facts).casefold()
    assert "holds warm air" in " ".join(spec.panels[1].facts).casefold()
    assert not spec.footer_prompt.endswith("?")
    assert validate_learning_artifact(spec) == []

    weakened = spec.model_copy(
        update={
            "panels": [
                ReferencePanel(
                    heading=spec.panels[0].heading,
                    cue="segments",
                    segment_labels=["OUTSIDE FEATHERS", "SOFT FEATHERS", "WARM AIR"],
                    facts=["Outer feathers shed water.", "Soft feathers hold warm air."],
                ),
                spec.panels[1],
            ]
        }
    )
    causal_errors = LearningArtifactService._reference_causal_alignment_errors(
        weakened,
        {
            "answer": (
                "Penguin feathers block wind and hold warm air. A winter coat blocks wind and holds warm air too."
            )
        },
    )
    assert any("wind" in error and "Penguin" in error for error in causal_errors)
    insulation_errors = LearningArtifactService._reference_insulation_errors(
        weakened,
        brief="Compare penguin feather layers with a winter coat.",
        source_question="Why do penguins stay warm in the cold?",
    )
    assert any("closely packed" in error and "wind protection" in error for error in insulation_errors)


def test_comparison_printable_fallback_supplies_one_space_per_named_sample() -> None:
    spec = LearningArtifactService._fallback_spec(
        "worksheet",
        "A child has three mystery objects but does not want to run an experiment.",
        {"nugget": "Careful observation helps us compare objects."},
        "1st",
        "evt_samples",
        revision="Make a one-page printable for comparing the three samples.",
    )

    assert isinstance(spec, WorksheetSpec)
    assert [task.id for task in spec.tasks[:3]] == ["sample_1", "sample_2", "sample_3"]
    assert all(task.kind == "draw" for task in spec.tasks[:3])
    assert "test" not in spec.directions.casefold()
    assert validate_learning_artifact(spec) == []


def test_observation_sheet_fallback_supplies_tally_color_and_size_tools() -> None:
    spec = LearningArtifactService._fallback_spec(
        "worksheet",
        "Watch quietly from indoors for about 8 minutes.",
        {"nugget": "Record only what you can observe."},
        "1st",
        "evt_watch",
        revision=(
            "Make a playful observation detective sheet to tally how many visitors appear, note their colors, and "
            "circle whether each looks small, medium, or large."
        ),
    )

    assert isinstance(spec, WorksheetSpec)
    assert spec.estimated_minutes == 8
    assert [task.id for task in spec.tasks[:2]] == ["observation_1", "observation_2"]
    assert spec.tasks[-1].id == "total_seen"
    assert {task.kind for task in spec.tasks} == {"record", "circle"}
    observation_rows = [task for task in spec.tasks if task.id.startswith("observation_")]
    assert all("COLOR" in task.instruction for task in observation_rows)
    assert all(task.choices == ["Small", "Medium", "Large"] for task in observation_rows)
    assert validate_learning_artifact(spec) == []


def test_bird_detective_wording_still_routes_to_observation_sheet() -> None:
    spec = LearningArtifactService._fallback_spec(
        "worksheet",
        "Watch birds quietly from indoors for about 10 minutes.",
        {"nugget": "Record only what you can observe."},
        "1st",
        "evt_birds",
        revision=(
            "Make a playful printable bird detective sheet. Only notice how many, their colors, and whether each "
            "looks small, medium, or large."
        ),
    )

    assert isinstance(spec, WorksheetSpec)
    assert spec.estimated_minutes == 10
    assert len([task for task in spec.tasks if task.id.startswith("observation_")]) == 5
    assert any(task.id == "more_birds" for task in spec.tasks)
    assert spec.tasks[-1].id == "total_seen"
    assert validate_learning_artifact(spec) == []


def test_recording_sheet_fallback_has_named_prediction_and_observation_sections() -> None:
    spec = LearningArtifactService._fallback_spec(
        "worksheet",
        "We have sugar, salt, sand, clear cups, water, and spoons for a small comparison. Nobody will taste.",
        {"nugget": "Dissolving means mixing through water, not disappearing from existence."},
        "1st",
        "evt_mix",
        revision=(
            "Make a one-page recording sheet with a prediction, what we did, and what we noticed for each material."
        ),
    )

    assert isinstance(spec, WorksheetSpec)
    serialized = spec.model_dump_json().casefold()
    for material in ("sugar", "salt", "sand"):
        assert material in serialized
    assert {task.kind for task in spec.tasks} == {"circle", "record"}
    assert "never taste" in spec.directions.casefold()
    assert "not gone" in spec.directions.casefold()
    assert validate_learning_artifact(spec) == []


def test_reviewed_weather_challenge_has_two_trials_at_each_spot() -> None:
    spec = LearningArtifactService._weather_challenge_spec(
        response={"resource_refs": ["Teacher update: weather observations."]},
        grade="1st",
        source_event_id="evt_weather",
    )

    assert spec.evidence_rows == ["OPEN SPOT", "SHELTERED SPOT"]
    assert len(spec.steps) == 4
    assert any("OPEN SPOT" in step and "two 10-count trials" in step for step in spec.steps)
    assert any("SHELTERED SPOT" in step and "same two trials" in step for step in spec.steps)
    assert "two trials" in " ".join(spec.steps).casefold()
    assert "TRY 1 and TRY 2" in spec.evidence_of_completion
    assert all(not material.endswith(" a.") for material in spec.materials)
    assert validate_learning_artifact(spec) == []


def test_fallback_mystery_keeps_float_sink_topic_and_changes_play_loop() -> None:
    spec = LearningArtifactService._fallback_spec(
        "activity",
        "Why do some things float and others sink?",
        {
            "nugget": "Floating depends on density and can also depend on shape.",
            "physical_extension": {
                "materials": ["bowl of water", "safe objects"],
                "instructions": ["Predict.", "Test.", "Notice."],
            },
        },
        "1st",
        "evt_float",
        revision=(
            "Change the actual play loop into a child-ready mystery or story investigation, not just the title. "
            "Do not make a worksheet. Parent request: make it feel like a mystery, not a worksheet."
        ),
    )

    assert isinstance(spec, ActivitySpec)
    assert spec.title == "The Floating Clue Mystery"
    assert "floaters" in spec.mission.casefold()
    assert "clue" in spec.model_dump_json().casefold()
    assert validate_learning_artifact(spec) == []


def test_context_derived_artifact_recovers_source_topic_from_generic_followup(tmp_path) -> None:
    db = tmp_path / "private" / "curiosity.db"
    init_db(db)
    add_child(db, "kid-a", "Kid A", grade="1st")
    now = utcnow()
    with connect(db) as conn:
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_context','context_input','kid-a',?,'test',?,?,'completed')""",
            (
                "Kid A wants to see what happens when a little safe object freezes inside a cup of water.",
                jdump({"evidence_eligible": False, "learning_scope": "context_only"}),
                now,
            ),
        )
        conn.execute(
            """INSERT INTO events(id,type,child_id,text,source,metadata_json,created_at,status)
               VALUES('evt_idea','child_question','kid-a','give us one idea','test',?,?,'completed')""",
            (jdump({"related_event_id": "evt_context", "learning_scope": "diagnostic"}), now),
        )
        conn.execute(
            """INSERT INTO responses(event_id,workflow,status,output_json,created_at,updated_at)
               VALUES('evt_idea','quick_answer','completed',?,?,?)""",
            (
                jdump(
                    {
                        "answer": "Put a safe little object in a cup of water, freeze it, and compare later.",
                        "notice": "Notice the water line before and after.",
                        "question_back": None,
                    }
                ),
                now,
                now,
            ),
        )

    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt_idea",
        artifact_type="activity",
        revision="show the activity steps",
    )

    assert artifact["title"] == "Ice Treasure Steps"
    with connect(db) as conn:
        row = conn.execute("SELECT spec_json FROM experiences WHERE id=?", (artifact["experience_id"],)).fetchone()
    spec = jload(row["spec_json"])
    assert "small plastic cup" in spec["materials"]
    assert "freezer" in " ".join(spec["steps"]).casefold()
