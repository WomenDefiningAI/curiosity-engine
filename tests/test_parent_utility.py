"""Small regression gate for parent materials and revisions; live usability is evaluated separately."""
import subprocess
from io import BytesIO
from pathlib import Path

from legacy_natural_routing_v2 import _family, _offline_service

from curiosity_engine.contracts import ParentAgentTurn, PicturePage, ReferencePanel, ReferenceSpec
from curiosity_engine.db import connect, jdump, jload
from curiosity_engine.learning_artifacts import LearningArtifactService, render_learning_pdf
from curiosity_engine.openai_image_backend import GeneratedImage
from curiosity_engine.parent_agent import ParentAgentRuntime
from curiosity_engine.pedagogy import (
    PedagogyCompiler,
    _approved_safety,
    _candidate_hazards,
    validate_artifact_against_plan,
)
from curiosity_engine.reasoning import StubBackend, _release_gate_requirements
from curiosity_engine.service import _proportionate_thread_message, _resolve_answer_mode
from curiosity_engine.tooling import ToolRegistry


def test_generated_subject_is_embedded_and_reused_for_layout_revision(tmp_path):
    import pytest
    from PIL import Image

    class ImageBackend:
        name, model = "openai", "test-image-model"

        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt)
            data = BytesIO()
            sheet = Image.new("RGB", (1024, 1024), "white")
            for i, color in enumerate(("red", "green", "blue", "yellow")):
                sheet.paste(color, ((i % 2)*512, (i // 2)*512, (i % 2 + 1)*512, (i // 2 + 1)*512))
            sheet.save(data, format="PNG")
            return GeneratedImage(data.getvalue(), self.model, "test-request")

    page = PicturePage.model_validate({
        "directions": "Point to a frog.", "columns": 2,
        "symbols": [{"id": "frog", "description": "One green frog with broad feet and visible eyes.",
                     "annotations": [{"text": "Eye", "position": [20, 80], "target": [50, 60]}]}],
        "cells": [{"label": "First", "items": [{"symbol_id": "frog", "count": 2}]},
                  {"label": "Next", "items": [{"symbol_id": "frog", "count": 1}]}],
    })
    spec = ReferenceSpec(title="Frog friends", target_grade="K", learning_objective="Notice frogs.",
                         estimated_minutes=3, accessibility_text="Point or say what you see.",
                         layout="comparison", directions="Point to a frog.",
                         panels=[ReferencePanel(heading="First", facts=["Look at their feet."]),
                                 ReferencePanel(heading="Next", facts=["Look at their eyes."])],
                         picture_page=page)
    backend = ImageBackend()
    service = LearningArtifactService(tmp_path / "data.db", tmp_path / "output",
                                      backend=StubBackend(), image_backend=backend)
    prepared = service._prepare_picture_illustrations(spec, authorization_preflight=None)
    assert len(backend.prompts) == 1
    assert "No words" in backend.prompts[0]
    asset = prepared.picture_page.symbols[0].illustration
    assert asset.model == backend.model
    assert asset.prompt_version == "picture-sheet-v1"
    sizes = []
    pdf = render_learning_pdf(prepared, tmp_path / "generated.pdf", picture_sizes=sizes)
    assert b"/Subtype /Image" in pdf.read_bytes()
    assert len(sizes) == 3
    screen = prepared.model_copy(update={"picture_page": prepared.picture_page.model_copy(
        update={"display": "screen"})})
    screen_pdf = render_learning_pdf(screen, tmp_path / "screen.pdf")
    screen_text = subprocess.check_output(["pdftotext", str(screen_pdf), "-"], text=True)
    assert "Point to a frog" in screen_text and "parent preview" not in screen_text
    assert b"612 560" in screen_pdf.read_bytes()
    assert subprocess.check_output(["pdftotext", str(pdf), "-"], text=True).count("Eye") == 3
    plan = PedagogyCompiler().fallback(
        event={"id": "evt-picture-cards", "child_id": "kid-a", "text": "Make picture cards to point to."},
        workflow="pull_thread",
    )
    for directions, should_block in (
        ("Grown-up: cut the paper cards with scissors. Child: point to the pictures.", False),
        ("Child: use the scissors to cut the cards. Grown-up: watch.", True),
    ):
        directed = spec.model_copy(update={"picture_page": page.model_copy(update={"directions": directions})})
        errors = validate_artifact_against_plan(plan, directed)
        assert any("introduces hazards absent" in error and "sharp_tools" in error for error in errors) == should_block
    road_plan = plan.model_copy(update={"safety": _approved_safety("A bus on the road", sensitive=False)})
    no_writing_plan = plan.model_copy(update={"accessibility": plan.accessibility.model_copy(
        update={"requested_supports": ["no_writing"]})})
    for directions, should_block in (
        ("Point to the longer pencil.", False),
        ("Compare the two pens.", False),
        ("Say one sentence about the picture.", False),
        ("Write the answer.", True),
        ("Use a pencil to record the answer.", True),
    ):
        directed = spec.model_copy(update={"picture_page": page.model_copy(update={"directions": directions})})
        errors = validate_artifact_against_plan(no_writing_plan, directed)
        assert ("artifact requires writing despite the approved no-writing support" in errors) == should_block
    for directions, should_block in (
        ("Point to the bus picture. It carries people on a road.", False),
        ("Child: cross the road to find a bus.", True),
    ):
        directed = spec.model_copy(update={"picture_page": page.model_copy(update={"directions": directions})})
        errors = validate_artifact_against_plan(road_plan, directed)
        assert any("safety boundary" in error or "hazardous handling" in error for error in errors) == should_block
    revision = spec.model_copy(update={"picture_page": page.model_copy(update={"columns": 1})})
    revised = service._prepare_picture_illustrations(revision, authorization_preflight=None)
    assert len(backend.prompts) == 1
    assert revised.picture_page.symbols[0].illustration.sha256 == asset.sha256
    from curiosity_engine.picture_design import PictureDraft, compile_draft
    diagram_draft = PictureDraft.model_validate({
        "title": "Frog parts", "directions": "Look at the frog, then draw one.",
        "subjects": [{"id": "frog", "description": "One frog", "labels": ["eye"]}],
        "cells": [{"items": [{"symbol_id": "frog"}]}, {"response_space": "draw"}],
    })
    stacked = compile_draft(diagram_draft, kind="reference", grade="K", event_id="synthetic")
    fixed = compile_draft(diagram_draft.model_copy(update={"layout_locked": True}),
                          kind="reference", grade="K", event_id="synthetic")
    assert stacked.picture_page.columns == 1 and fixed.picture_page.columns == 2
    draft = PictureDraft.model_validate({
        "title": "Picture story", "directions": "Point to the order of the story.",
        "subjects": [{"id": f"scene{i}", "description": f"Story moment {i}"} for i in range(4)],
        "cells": [{"items": [{"symbol_id": f"scene{i}"}]} for i in range(4)],
    })
    story = compile_draft(draft, kind="reference", grade="K", event_id="synthetic")
    story = service._prepare_picture_illustrations(story, authorization_preflight=None)
    assert len(backend.prompts) == 2  # Four coherent scenes require only one additional provider call.
    assert len({s.illustration.sha256 for s in story.picture_page.symbols}) == 4
    for i, symbol in enumerate(story.picture_page.symbols):
        with Image.open(symbol.illustration.local_path) as crop:
            assert crop.size == (512, 512)
            assert crop.getpixel((250, 250)) == ((255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 255, 0))[i]
    enlarged = compile_draft(draft.model_copy(update={"enlarge_pictures": True, "preserve_artwork": True}),
                             kind="reference", grade="K", event_id="synthetic", previous=story)
    enlarged = service._prepare_picture_illustrations(enlarged, authorization_preflight=None)
    before_sizes, after_sizes = [], []
    render_learning_pdf(story, tmp_path / "story.pdf", picture_sizes=before_sizes)
    render_learning_pdf(enlarged, tmp_path / "bigger.pdf", picture_sizes=after_sizes)
    assert min(after_sizes) / min(before_sizes) >= 1.24
    assert len(backend.prompts) == 2
    assert [s.illustration.sha256 for s in enlarged.picture_page.symbols] == [s.illustration.sha256 for s in story.picture_page.symbols]
    class PageBackend(StubBackend):
        name = "openai"

        def __init__(self):
            self.designs, self.reviews = 0, 0

        def complete(self, **kwargs):
            assert kwargs["payload"]["parent_constraints"] == ["Pointing only; no child writing."]
            if kwargs["response_model"] is PictureDraft:
                self.designs += 1
                proposed = draft.model_dump(mode="json")
                if self.designs == 2:
                    proposed["directions"] = "Point to the story order."
                    proposed["cells"][0]["action"] = "Too much cell text " * 15
                if self.designs == 3:
                    assert kwargs["payload"]["previous_draft"]["directions"] == "Point to the story order."
                    assert "Use pointing; printed pictures cannot move." in kwargs["payload"]["feedback"]
                    assert any("cells.0.action" in f for f in kwargs["payload"]["feedback"])
                return proposed
            assert kwargs["response_model"].__name__ == "PictureReview"
            assert kwargs["payload"]["image_data_urls"][0].startswith("data:image/png;base64,")
            self.reviews += 1
            if self.reviews == 1:
                return {"verdict": "repair", "obstacles": [{"kind": "action", "detail": "Use pointing; printed pictures cannot move."}]}
            assert kwargs["payload"]["code_verified_child_writing_required"] is False
            return {"verdict": "repair", "obstacles": [{"kind": "revision", "detail": "No writing means remove the printed title and footer."}]}

    service.backend = PageBackend()
    unified = service._design("challenge", question="Still making giant ice cubes in cups.", response={},
                              grade="K", source_event_id="synthetic", revision="Make a visual challenge.",
                              runtime_instructions=["Pointing only; no child writing."])
    assert unified.picture_page is not None  # Known topics use the same reviewed visual pipeline.
    assert unified.illustration is None
    assert all(s.illustration for s in unified.picture_page.symbols)
    class LayoutRepairBackend(StubBackend):
        name = "openai"
        designs = 0

        def complete(self, **kwargs):
            if kwargs["response_model"] is PictureDraft:
                self.designs += 1
                return draft.model_copy(update={"preserve_artwork": self.designs > 1,
                    "subjects": [] if self.designs > 1 else draft.subjects}).model_dump(mode="json")
            return {"verdict": "ready", "obstacles": []}

    repair = LearningArtifactService(tmp_path / "repair.db", tmp_path / "repair-output",
                                     backend=LayoutRepairBackend(), image_backend=backend)
    original_preview, previews = repair._preview_image, []

    def fail_first_layout(spec):
        previews.append(spec)
        if len(previews) == 1:
            raise ValueError("The main picture is too small to use.")
        return original_preview(spec)

    repair._preview_image = fail_first_layout
    recovered = repair._design("reference", question="Make big story pictures.", response={}, grade="K",
                                source_event_id="synthetic-layout-repair", revision="")
    assert len(previews) == 2
    assert [s.illustration.sha256 for s in recovered.picture_page.symbols] == [
        s.illustration.sha256 for s in previews[0].picture_page.symbols]
    Path(asset.local_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="no longer matches"):
        render_learning_pdf(revised, tmp_path / "changed.pdf")


def test_ordinary_art_and_outdoor_play_do_not_become_risk_events():
    assert "body_or_health" not in _approved_safety("Mix washable paint colors on paper.", sensitive=False).hazards
    assert "body_or_health" in _approved_safety("My child has stomach pain.", sensitive=False).hazards
    assert not _approved_safety("Point to leaves outside in the park.", sensitive=False).hazards
    assert "outdoors" in _approved_safety("Cross the road.", sensitive=False).hazards
    assert not _approved_safety("Point to a possible match. Check the socks together.", sensitive=False).hazards
    assert "ignition" in _approved_safety("Light a match.", sensitive=False).hazards
    assert not _approved_safety("Build with paper. We have no scissors or tape.", sensitive=False).hazards
    assert "sharp_tools" in _approved_safety("No knives; use scissors.", sensitive=False).hazards
    assert not _candidate_hazards(materials=[], instructions=["Try one more object. Draw what you saw."])
    assert "sharp_tools" in _candidate_hazards(materials=["saw"], instructions=["Use the saw to cut wood."])
    assert "sharp_tools" in _candidate_hazards(materials=[], instructions=["Draw what you saw. Use a saw to cut wood."])
    assert not _candidate_hazards(materials=["one sheet of paper"], instructions=[
        "No scissors or tape. Fold the top corners down. Press the back fold, then point to the shape."])
    assert not _candidate_hazards(materials=["paper"], instructions=[
        "No tape or scissors. Fold a paper tent. Press its top crease."])
    assert "sharp_tools" in _candidate_hazards(materials=[], instructions=["No knives; use scissors to cut paper."])
    assert "sharp_tools" in _candidate_hazards(materials=[], instructions=["No knives, use scissors to cut paper."])


def test_semantic_material_request_survives_conversation_normalizers(tmp_path, monkeypatch):
    class MaterialBackend(StubBackend):
        name = "semantic-material-test"

        def complete(self, **kwargs):
            assert kwargs["response_model"] is ParentAgentTurn
            return {"requested_output": "material", "tool_calls": [{
                "name": "create_learning_artifact", "arguments": {"artifact_type": "worksheet"},
                "rationale": "The parent requests something to use with the objects.",
            }]}

    db, _ = _family(tmp_path)
    runtime = ParentAgentRuntime(db, backend=MaterialBackend(), tools=ToolRegistry())
    for wording in (
        "Can you make us a mat for putting our finds into groups?",
        "Make a set of little scene cards we can arrange and tell aloud.",
        "Make two big pictures we can copy from my phone. No printer. No worksheet boxes.",
    ):
        turn = runtime._plan({"message": wording, "session": {"id": "test", "child": {"grade": "K"}},
                              "workspace": {}, "history": [], "input_mode": "conversation"},
                             {"create_learning_artifact", "record_thread_context", "continue_learning_thread"})
        assert any(call.name == "create_learning_artifact" for call in turn.tool_calls)

    class ConversationBackend(StubBackend):
        name = "semantic-conversation-test"

        def complete(self, **kwargs):
            return {"requested_output": "conversation", "tool_calls": [{
                "name": "continue_learning_thread", "arguments": {"mode": "explore"},
                "rationale": "The parent is exploring the topic after seeing a picture.",
            }]}

    runtime.backend = ConversationBackend()
    for wording in (
        "What if we move the light instead and leave the toy still?",
        "What are a few ways we could play with that idea?",
        "Why does adding more water change what happens?",
        "Could we use only the toys already out to explore that?",
    ):
        turn = runtime._plan({"message": wording, "session": {
            "id": "test", "child": {"grade": "K"}, "latest_event_id": "synthetic-event"},
            "workspace": {"recent_outputs": [{"kind": "artifact", "ref_id": "synthetic-page"}]},
            "history": [], "input_mode": "conversation"},
            {"create_learning_artifact", "record_thread_context", "continue_learning_thread", "revise_learning_thread"})
        assert [call.name for call in turn.tool_calls] == ["continue_learning_thread"]
        assert turn.tool_calls[0].arguments["mode"] == "explore"

    service, depth_db, binding = _offline_service(tmp_path / "depth", monkeypatch)
    monkeypatch.setattr(ParentAgentRuntime, "_plan", lambda *_: ParentAgentTurn.model_validate({
        "requested_output": "conversation", "tool_calls": [{
            "name": "continue_learning_thread", "arguments": {"mode": "explore"},
            "rationale": "Several observations and checks need a flexible exploration reply.",
        }],
    }))
    result = service.chat(binding_id=binding, team_id="T1", channel_id="C1", thread_id="depth",
                          child_id="kid-a", text="What might make fallen leaves behave differently in water, "
                          "and what are a couple of ways we could check?")
    assert result["status"] == "completed"
    with connect(depth_db) as conn:
        assert conn.execute("SELECT job FROM runs WHERE event_id=?", (result["event_id"],)).fetchone()[0] == "pull_thread"


def test_another_mechanic_does_not_become_a_no_activity_rule():
    from curiosity_engine.construction_patterns import PATTERNS
    from curiosity_engine.contracts import PhysicalExtension, PullThreadOutput
    from curiosity_engine.presentation import format_learning_thread

    brief = "Say 'turn into a tiny cloud' and copy one another's silly shapes. Swap who leads, then wave goodbye."
    output = PullThreadOutput(parent_response=brief, hook="Old repeated introduction", nugget="Old explanation")
    from curiosity_engine.reasoning import ReasoningEngine
    normalized = ReasoningEngine._validate_and_normalize_candidate(
        output.model_dump(), PullThreadOutput,
        {"text": "Give me one silly game to play with my five-year-old. Two short sentences."})
    assert normalized.parent_response == brief and normalized.physical_extension is None
    assert format_learning_thread({"output": output.model_dump()}) == brief
    assert _proportionate_thread_message({"output": output.model_dump()}, "Two short sentences.") == brief
    multiple = "1. Trace a triangle in the air.\n2. Make a square with four big blocks.\n3. Turn a paper square around."
    assert format_learning_thread({"output": PullThreadOutput(parent_response=multiple).model_dump()}) == multiple
    extension = PhysicalExtension(title="Finger tunnel", construction_pattern="ridge_tunnel",
                                  instructions=["Invent four legs and glue them on."], materials=["glue"])
    assert extension.instructions == PATTERNS["ridge_tunnel"]["steps"]
    assert extension.materials == ["one sheet of paper"]
    compiled = format_learning_thread({"output": PullThreadOutput(
        parent_response="Make a tunnel for your fingers.", physical_extension=extension).model_dump()})
    assert compiled.count("Bring one long edge") == 1
    assert "glue" not in compiled and "Walk two fingers" in compiled

    def requirements(text):
        return _release_gate_requirements({"text": "Play a room game.", "metadata": {
            "response_revision": {"parent_request": text}}})

    for wording in ("Another idea, not the same listening game.", "Not a renamed sorting game; something we do differently."):
        assert not any("Omit the rejected game" in rule for rule in requirements(wording))
    assert any("Omit the rejected game" in rule for rule in requirements("No activity. Just explain it."))
    result = {"status": "completed", "workflow": "pull_thread", "output": {
        "hook": "Try a paper drum.", "nugget": "Stretched paper vibrates when tapped.",
        "physical_extension": {"title": "Paper Drum", "materials": ["bowl", "paper", "rubber band"],
                               "instructions": ["Cover the bowl with paper.", "Secure it with the band.",
                                                "Tap the paper, then stretch it tighter and tap again."],
                               "prompts": ["What changed?"]}}}
    for wording in ("Another idea please. Something we do differently, not the same listening game with different sounds.",
                    "Give us a different game, not the same one again."):
        message = _proportionate_thread_message(result, wording)
        assert "Materials: bowl, paper, rubber band" in message
        assert "1. Cover the bowl with paper." in message
        assert "3. Tap the paper" in message
    assert "Materials:" not in _proportionate_thread_message(result, "No activity. Just explain it.")
    assert _resolve_answer_mode("Give us a different game, not the same activity again.", "auto").value == "explore"
    assert _resolve_answer_mode("No activity. Just explain it.", "auto").value == "quick"
    assert _resolve_answer_mode("Give us one playful thing to build with this single sheet of paper. We have no scissors or tape.", "auto").value == "explore"
    assert _resolve_answer_mode("Why do birds build nests?", "auto").value == "quick"


def test_slack_picture_and_pdf_bundle_delivers_once_in_the_same_thread(tmp_path):
    from legacy_capability_kernel import _ArtifactSlackClient, _family_db, _stored_response

    from curiosity_engine.artifact_delivery import enqueue_artifact_bundle
    from curiosity_engine.transports.slack import flush_slack_artifact_outbox

    db, _, binding = _family_db(tmp_path)
    _stored_response(db)
    artifact = LearningArtifactService(db, tmp_path / "output", backend=StubBackend()).create_from_event(
        event_id="evt-source", artifact_type="challenge")
    kwargs = dict(artifact=artifact, binding_id=binding, channel_id="C_FAMILY",
                  thread_id="123.4", idempotency_key="phone-bundle")
    first = enqueue_artifact_bundle(db, **kwargs)
    assert enqueue_artifact_bundle(db, **kwargs) == first
    client, uploaded = _ArtifactSlackClient(), []
    results = flush_slack_artifact_outbox(client, db, tmp_path / "output",
                                         uploader=lambda _url, data: uploaded.append(data))
    assert [row["status"] for row in results] == ["sent", "sent"]
    assert uploaded[0].startswith(b"\x89PNG\r\n\x1a\n") and uploaded[1].startswith(b"%PDF-")
    assert client.tickets[0]["alt_txt"]
    assert all(item["thread_ts"] == "123.4" and item["channel_id"] == "C_FAMILY" for item in client.completed)
    assert flush_slack_artifact_outbox(client, db, tmp_path / "output", uploader=lambda *_: None) == []


def test_requested_sentence_count_reuses_only_the_needed_reviewed_sentences(tmp_path, monkeypatch):
    service, db, binding = _offline_service(tmp_path, monkeypatch)
    first = service.chat(binding_id=binding, team_id="T1", channel_id="C1", thread_id="shorten",
                         child_id="kid-a", text="Why does ice float?", answer_mode_hint="quick")
    with connect(db) as conn:
        row = conn.execute("SELECT output_json FROM responses WHERE event_id=?", (first["event_id"],)).fetchone()
        output = jload(row["output_json"])
        output["answer"] = "Ice takes up more room than the water it came from. That makes it float. This is called density. We can explore it later."
        conn.execute("UPDATE responses SET output_json=? WHERE event_id=?", (jdump(output), first["event_id"]))
    result = service.chat(binding_id=binding, team_id="T1", channel_id="C1", thread_id="shorten",
                          text="Give me two short sentences I can say. No activity.")
    assert result["message"] == "Ice takes up more room than the water it came from. That makes it float."
    assert result["event_id"] == first["event_id"]
    saved = service.chat(binding_id=binding, team_id="T1", channel_id="C1", thread_id="saving",
                         child_id="kid-a", input_mode="context",
                         text="Just saving this: we watched ants carry crumbs at the park.")
    assert saved["status"] == "completed"
    assert "saved" in saved["message"]
    assert not saved.get("interaction")


def test_picture_page_renders_counted_objects_and_omits_legacy_body(tmp_path, monkeypatch):
    import pytest
    from reportlab.pdfgen import canvas

    import curiosity_engine.picture_pages as renderer
    from curiosity_engine.contracts import PictureAnnotation
    page = PicturePage.model_validate({
        "directions": "Move a token to each picture.", "columns": 2,
        "symbols": [{"id": "token", "description": "one square token", "strokes": [
            {"kind": "rect", "points": [[15, 15], [85, 85]], "fill": "none"}]}],
        "cells": [{"label": "One group", "items": [{"symbol_id": "token", "count": 7}]},
                  {"label": "Another group", "items": [{"symbol_id": "token", "count": 8}], "response_space": "draw"}],
    })
    drawn = []
    original = renderer._symbol

    def counted(canvas, symbol, x, y, size):
        drawn.append((symbol.id, size))
        original(canvas, symbol, x, y, size)

    monkeypatch.setattr(renderer, "_symbol", counted)
    spec = ReferenceSpec(title="Token groups", target_grade="K", learning_objective="Explore groups",
                         estimated_minutes=5, accessibility_text="Point to a token.", layout="comparison",
                         directions="Unprinted legacy directions", panels=[ReferencePanel(heading=str(i), facts=["Unprinted fact"])
                                                                            for i in range(2)], picture_page=page)
    picture_sizes = []
    pdf = render_learning_pdf(spec, tmp_path / "tokens.pdf", picture_sizes=picture_sizes)
    assert Path(pdf).read_bytes().startswith(b"%PDF")
    assert len(drawn) == 15
    assert all(size >= 25 for _, size in drawn)
    assert picture_sizes == [round(size, 1) for _, size in drawn]
    from curiosity_engine.contracts import PictureCell
    drawing_page = page.model_copy(update={"columns": 2, "cells": [PictureCell(
        label="Share", items=[{"symbol_id": "token"}], response_space="draw") for _ in range(6)]})
    header_sizes = []
    render_learning_pdf(spec.model_copy(update={"picture_page": drawing_page}),
                        tmp_path / "drawing-headers.pdf", picture_sizes=header_sizes)
    assert len(header_sizes) == 6 and min(header_sizes) >= 30
    # The real picture renderer was exercised, rather than a declared visual intent.
    from curiosity_engine.artifact_validation import render_pdf_preview
    assert render_pdf_preview(pdf, tmp_path / "page.png").is_file()
    labeled = page.symbols[0].model_copy(update={"annotations": [PictureAnnotation(
        text="does not move", position=(50, 90), target=(50, 50))]})
    with pytest.raises(ValueError, match="Show shared choices once"):
        renderer._annotations(canvas.Canvas(BytesIO()), labeled, 0, 0, 30)
