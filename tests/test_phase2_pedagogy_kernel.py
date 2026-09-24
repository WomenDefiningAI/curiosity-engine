"""Focused beta contracts for pedagogy, usefulness, and child-visible boundaries.

The exhaustive linguistic corpus remains reproducible in
``legacy_phase2_pedagogy_kernel.py`` but is intentionally outside default discovery.
"""

import legacy_phase2_pedagogy_kernel as legacy


def test_planner_persists_approved_plan_before_generation_with_proportional_review(tmp_path):
    db = legacy._db(tmp_path)
    backend = legacy.OrderingProbeBackend(db)
    policy = legacy.ReasoningPolicy(
        "pull_thread",
        2,
        critic_roles=("critic_family_usefulness", "critic_pedagogy"),
        max_revision_rounds=1,
        final_recovery="rebuild_from_scratch",
    )
    result = legacy.CuriosityHarness(db, legacy.ReasoningEngine(backend)).dispatch(
        legacy.Event(
            id="evt-order",
            type="child_question",
            child_id="kid-a",
            text="What happens if we predict and test which ice cube melts first?",
        ),
        policy=policy,
    )

    assert result.status == "completed"
    generator_index = next(
        index for index, call in enumerate(backend.calls) if call["response_model"] == "PullThreadOutput"
    )
    assert backend.calls[0]["role"] == "pedagogy_planner"
    assert generator_index > 0
    assert [
        call["role"] for call in backend.calls if call["response_model"] == "CriticResult"
    ] == ["critic_family_usefulness", "critic_pedagogy"]
    trace = result.output["_pedagogy"]
    with legacy.connect(db) as conn:
        response = conn.execute(
            "SELECT pedagogy_plan_id,pedagogy_plan_hash FROM responses WHERE event_id='evt-order'"
        ).fetchone()
        plan = conn.execute("SELECT status FROM pedagogy_plans WHERE event_id='evt-order'").fetchone()
    assert (response["pedagogy_plan_id"], response["pedagogy_plan_hash"]) == (
        trace["plan_id"],
        trace["plan_hash"],
    )
    assert plan["status"] == "generated"
test_planner_sees_cards_and_generator_sees_only_selected_instructions = (
    legacy.test_planner_sees_cards_and_generator_sees_only_selected_instructions
)
test_excluding_one_format_does_not_cancel_requested_learning_activity = (
    legacy.test_excluding_one_format_does_not_cancel_requested_learning_activity
)
test_reviewed_task_and_rubric_refs_materialize_in_phase1_tables = (
    legacy.test_reviewed_task_and_rubric_refs_materialize_in_phase1_tables
)
test_invalid_planner_output_uses_persisted_safe_fallback = (
    legacy.test_invalid_planner_output_uses_persisted_safe_fallback
)
test_accessibility_choices_apply_even_without_evidence_collection = (
    legacy.test_accessibility_choices_apply_even_without_evidence_collection
)
test_artifact_and_visual_preserve_plan_lineage_without_creating_evidence = (
    legacy.test_artifact_and_visual_preserve_plan_lineage_without_creating_evidence
)
