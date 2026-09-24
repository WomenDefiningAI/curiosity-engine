"""Focused shadow-personalization and parent-consent contracts."""

import legacy_phase5_personalization as legacy

test_structured_feedback_is_separate_from_notes_and_evidence = (
    legacy.test_structured_feedback_is_separate_from_notes_and_evidence
)
test_replay_accepts_only_categorical_features_and_numeric_labels = (
    legacy.test_replay_accepts_only_categorical_features_and_numeric_labels
)
test_default_runtime_records_shadow_comparison_without_changing_plan = (
    legacy.test_default_runtime_records_shadow_comparison_without_changing_plan
)
test_cli_exposes_explicit_personalization_governance = (
    legacy.test_cli_exposes_explicit_personalization_governance
)
