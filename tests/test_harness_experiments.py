from __future__ import annotations

import importlib.util
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from curiosity_engine.model_usage import BudgetExceeded, estimate_cost, metered_call, reservation, usage_summary


def lab_module(name):
    path = Path(__file__).resolve().parents[1] / "lab/parent_usefulness" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_usage_accounts_for_cache_images_and_unknown_usage():
    assert estimate_cost("gpt-5.6-terra", {"input_tokens": 1000, "output_tokens": 200,
                         "input_tokens_details": {"cached_tokens": 500}}) == pytest.approx(.0035)
    assert estimate_cost("gpt-5.6-terra", {"input_tokens": 1000, "output_tokens": 200,
                         "input_tokens_details": {"cached_tokens": 500, "cache_write_tokens": 400}}) == pytest.approx(.0037)
    assert estimate_cost("gpt-5.6-terra", {"input_tokens": 300000, "output_tokens": 200,
                         "input_tokens_details": {"cached_tokens": 100000, "cache_write_tokens": 100000}}) == pytest.approx(.9436)
    assert estimate_cost("gpt-5.6-terra", {"input_tokens": 100, "output_tokens": 1,
                         "input_tokens_details": {"cached_tokens": 90, "cache_write_tokens": 90}}) is None
    assert estimate_cost("gpt-image-2", {"input_tokens": 100, "output_tokens": 200,
                         "input_tokens_details": {"text_tokens": 90, "image_tokens": 10}}, image=True) == pytest.approx(.003265)
    assert estimate_cost("gpt-5.6-terra", None) is None
    assert estimate_cost("unknown-model", {"input_tokens": 1, "output_tokens": 1}) is None
    assert estimate_cost("jev-1.13.0", {"input_tokens": 1000}) == pytest.approx(.000042)
    assert estimate_cost("jev-1.13.0", {"input_tokens": -1}) is None


def test_concurrent_budget_reserves_before_call_and_retains_unknown_charges(tmp_path, monkeypatch):
    path = tmp_path / "usage.sqlite"
    request = {"model": "gpt-5.6-terra", "max_output_tokens": 100, "instructions": "PRIVATE SENTINEL"}
    monkeypatch.setenv("CURIOSITY_USAGE_DB", str(path))
    monkeypatch.setenv("CURIOSITY_BUDGET_USD", str(reservation(request) * 1.5))
    entered, finish = threading.Event(), threading.Event()

    def invoke(budgeted):
        assert budgeted
        entered.set()
        assert finish.wait(5)
        return SimpleNamespace(usage=None)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(metered_call, request, invoke, role="generator")
        assert entered.wait(5)
        with pytest.raises(BudgetExceeded):
            metered_call(request, lambda _: pytest.fail("Over-budget request reached provider"), role="judge")
        finish.set()
        first.result()
    summary = usage_summary(path)
    assert summary["unknown_calls"] == 1
    assert summary["reserved_or_spent_usd"] == pytest.approx(reservation(request))
    assert "PRIVATE SENTINEL" not in json.dumps(summary)
    monkeypatch.setenv("CURIOSITY_BUDGET_USD", "0")
    with pytest.raises(BudgetExceeded):
        metered_call(request, lambda _: None, role="judge")


def test_exposed_inputs_cannot_be_renamed_into_holdouts(tmp_path):
    register = lab_module("holdouts").register_exposure
    cases = [{"id": "first", "input": {"parent_turns": ["Why does that happen?"]}}]
    first = register(tmp_path / "registry.sqlite", cases, "frozen-a", purpose="development", run_id="one")
    assert first["classification"] == "development"
    cases[0]["id"] = "new-label"
    second = register(tmp_path / "registry.sqlite", cases, "frozen-b", purpose="validation", run_id="two")
    assert second["classification"] == "regression" and second["previously_exposed_inputs"] == 1


def test_human_review_contains_original_and_delivered_evidence_with_version_bound_ratings(tmp_path):
    run = tmp_path / "captures"
    folder = run / "CV01"
    folder.mkdir(parents=True)
    Image.new("RGB", (20, 20), "blue").save(folder / "source.png")
    Image.new("RGB", (20, 20), "green").save(folder / "page.png")
    bundle = {"id": "CV01", "source_sha256": "version-a", "status": "captured", "turns": [
        {"number": 1, "parent": "Use this concept </script>", "message": "A complete activity.",
         "input_files": [{"path": "source.png"}], "files": [{"path": "page.png"}]}]}
    (folder / "bundle.json").write_text(json.dumps(bundle))
    build = lab_module("human_review").build_review
    first = tmp_path / "first"
    build([run], first)
    manifest = json.loads((first / "review-manifest.json").read_text())[0]
    assert len(manifest["turns"][0]["sources"]) == len(manifest["turns"][0]["images"]) == 1
    assert "Use this concept </script>" not in (first / "index.html").read_text()
    assert "reviewer_kind:'human'" in (first / "index.html").read_text()
    bundle["turns"][0]["message"] = "A changed activity."
    (folder / "bundle.json").write_text(json.dumps(bundle))
    build([run], tmp_path / "second")
    second = json.loads((tmp_path / "second/review-manifest.json").read_text())[0]
    assert manifest["fingerprint"] != second["fingerprint"]


def test_focused_reinspection_reads_pixels_without_rewriting_original_observation(tmp_path, monkeypatch):
    from legacy_natural_routing_v2 import _offline_service

    from curiosity_engine.contracts import ImageContextOutput

    service, _, _ = _offline_service(tmp_path, monkeypatch)
    path = tmp_path / "photo.png"
    Image.new("RGB", (30, 20), "orange").save(path)
    original = {"status": "ready", "summary": "Objects on a worksheet."}
    monkeypatch.setattr("curiosity_engine.service.source_photo_assets", lambda *_a, **_k: [
        {"status": "ready", "path": str(path), "sha256": "source-hash", "observation": original}])
    observed = []

    class Vision:
        def complete(self, **kwargs):
            observed.append(kwargs["payload"])
            return ImageContextOutput(summary="The objects form equal groups.").model_dump()

    service.visual_backend = Vision()
    result = service._focused_photo_context(session_id="session", source_event_id="event",
                                           parent_request="What relationship do those objects show?")
    assert observed[0]["image_data_urls"][0].startswith("data:image/")
    assert result["status"] == "ready" and result["source_hashes"] == ["source-hash"]
    assert "equal groups" in result["observation"]["summary"]
    assert original == {"status": "ready", "summary": "Objects on a worksheet."}


def test_open_question_preserves_semantic_action_and_inspection_request(tmp_path):
    from curiosity_engine.parent_agent import ParentAgentRuntime
    from curiosity_engine.tooling import ToolRegistry

    class Planner:
        name, model = "openai", "test"

        def complete(self, **_kwargs):
            return {"requested_output": "conversation", "tool_calls": [{"name": "continue_learning_thread",
                    "arguments": {"mode": "quick", "inspect_source_photos": True},
                    "rationale": "Inspect the omitted relationship and answer the new question."}]}

    runtime = ParentAgentRuntime(tmp_path / "db.sqlite", backend=Planner(), tools=ToolRegistry())
    turn = runtime._plan({"message": "What if we move those apart instead?", "session": {"id": "s", "child": {"grade": "K"},
                         "latest_event_id": "evt_previous"}, "history": [], "workspace": {"recent_outputs": []},
                         "input_mode": "conversation", "current_attachment_context": []},
                        {"continue_learning_thread", "create_learning_artifact", "revise_learning_thread"})
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "continue_learning_thread"
    assert turn.tool_calls[0].arguments["inspect_source_photos"] is True


def test_jev_review_uses_one_call_caches_exact_evidence_and_keeps_visuals_for_humans(tmp_path, monkeypatch):
    from reportlab.pdfgen import canvas

    import curiosity_engine.typesafe_backend as provider

    review = lab_module("review")
    folder = tmp_path / "CV01"
    folder.mkdir()
    pdf = canvas.Canvas(str(folder / "artifact.pdf"))
    pdf.drawString(30, 700, "Point to the same number. No writing is needed.")
    pdf.save()
    bundle = {"id": "CV01", "source_sha256": "synthetic", "status": "captured", "input": {}, "turns": [
        {"number": 1, "parent": "Make a pointing challenge.", "message": "Your page is ready.", "status": "completed",
         "files": [{"path": "artifact.pdf"}]}]}
    (folder / "bundle.json").write_text(json.dumps(bundle))
    requests = []

    def call(state, questions, *, key):
        requests.append((state, questions))
        answers = {}
        for name, question in questions.items():
            selected = "use_as_is" if name == "readiness" else "none"
            answers[name] = {"type": "choice", "choice": selected,
                             "probabilities": {choice: float(choice == selected) for choice in question["criteria"]}}
        return {"model": provider.MODEL, "usage": {"input_tokens": 1000}, "answers": answers}

    monkeypatch.setattr(provider, "evaluate", call)
    result = review.review_folder(folder, key="test")
    assert len(requests) == 1 and result["gate"] == "ready" and result["visual_review_required"]
    assert "Point to the same number" in requests[0][0]["turns"][0]["printed_text"][0]
    assert set(requests[0][1]) == {"readiness", "issue"}
    review.review_folder(folder, key="test")
    assert len(requests) == 1
    output = tmp_path / "page"
    lab_module("human_review").build_review([tmp_path], output)
    manifest = json.loads((output / "review-manifest.json").read_text())[0]
    assert manifest["jev"]["gate"] == "ready"
    bundle["turns"][0]["parent"] = "Use blocks instead."
    (folder / "bundle.json").write_text(json.dumps(bundle))
    stale = tmp_path / "stale"
    lab_module("human_review").build_review([tmp_path], stale)
    assert json.loads((stale / "review-manifest.json").read_text())[0]["jev"] is None
    monkeypatch.setattr(provider, "evaluate", lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError()))
    assert review.review_folder(folder, key="test")["gate"] == "review"
    assert len(list(folder.glob("jev-review-*.json"))) == 1


def test_human_jev_comparison_separates_text_disagreement_from_picture_approval():
    summarize = lab_module("summarize_human").summarize
    records = [{"id": "C1", "fingerprint": "a", "source": "s", "status": "captured", "cost": .1,
                "unknown_cost_calls": 0, "turns": [{"images": [{}]}],
                "jev": {"status": "complete", "use_as_is": "use_as_is", "input_sha256": "j", "policy": "v1"}}]
    exported = {"version": 3, "reviewer_kind": "human", "records": [
        {"fingerprint": "a", "jev_input_sha256": "j", "jev_policy": "v1"}],
        "reviews": {"a": {"text_rating": "would_not_use", "jev_seen_before_rating": False}}}
    result = summarize(records, exported)
    assert result["jev_comparison"]["jev_approved_text_human_would_change"] == 1
    assert result["accepted_conversations"] == 0
    exported["reviews"]["a"]["text_rating"] = "use_as_is"
    assert summarize(records, exported)["accepted_conversations"] == 0
    exported["reviews"]["a"]["visual_rating"] = "works"
    assert summarize(records, exported)["accepted_conversations"] == 1
    records[0]["jev"]["input_sha256"] = "new-judge"
    assert summarize(records, exported)["jev_comparison"]["changed_judge_not_compared"] == 1


def test_runtime_jev_replaces_ordinary_critics_and_preserves_bounded_repair(monkeypatch):
    from curiosity_engine import text_review
    from curiosity_engine.reasoning import ReasoningEngine, StubBackend, load_policies

    calls, roles = [], []

    def evaluate(state, questions, *, key):
        calls.append(state)
        choices = {"readiness": "small_edit_needed" if len(calls) == 1 else "use_as_is",
                   "issue": "request" if len(calls) == 1 else "none"}
        return {"model": text_review.MODEL, "answers": {
            name: {"type": "choice", "choice": choices[name],
                   "probabilities": {c: float(c == choices[name]) for c in question["criteria"]}}
            for name, question in questions.items()}}

    class Generator(StubBackend):
        def complete(self, **kwargs):
            roles.append(kwargs["role"])
            assert not kwargs["role"].startswith("critic_")
            return super().complete(**kwargs)

    monkeypatch.setattr(text_review, "evaluate", evaluate)
    engine = ReasoningEngine(Generator(), text_reviewer=text_review.JevTextReviewer("test"))
    policies = load_policies()
    for workflow in ("quick_answer", "pull_thread"):
        calls.clear()
        result = engine.run(policy=policies[workflow], context={"child": {"grade": "1"}},
                            event={"text": "Why does ice melt?", "metadata": {}})
        assert len(calls) == 2 and result.revision_rounds == 1
        assert calls[0]["draft_reply"] and calls[0]["parent_request"] == "Why does ice melt?"
        assert result.critiques[0].verdict == "pass"
    assert roles.count("reasoning") == 4


def test_runtime_jev_is_explicit_and_errors_do_not_trigger_a_second_judge(tmp_path, monkeypatch):
    from curiosity_engine import text_review
    from curiosity_engine.reasoning import ReasoningEngine, StubBackend, load_policies

    monkeypatch.delenv("CURIOSITY_TEXT_REVIEW", raising=False)
    assert text_review.configured_text_reviewer() is None
    key = tmp_path / "typesafe.env"
    key.write_text("TYPESAFE_API_KEY=synthetic-test-key\n")
    key.chmod(0o600)
    monkeypatch.setenv("CURIOSITY_TEXT_REVIEW", "jev")
    monkeypatch.setenv("CURIOSITY_TYPESAFE_KEY_FILE", str(key))
    reviewer = text_review.configured_text_reviewer()
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("synthetic outage")

    monkeypatch.setattr(text_review, "evaluate", unavailable)
    engine = ReasoningEngine(StubBackend(), text_reviewer=reviewer)
    with pytest.raises(TimeoutError):
        engine.run(policy=load_policies()["quick_answer"], context={"child": {"grade": "1"}},
                   event={"text": "Why does ice melt?", "metadata": {}})
    assert len(calls) == 1
    # Artifact and consequential workflows retain their declared critics.
    result = engine.run(policy=load_policies()["context_audit"], context={},
                        event={"text": "Review context.", "metadata": {}})
    assert len(calls) == 1 and result.critiques
