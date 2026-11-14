import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from curiosity_engine.artifact_validation import validate_artifact_spec, validate_rendered_file
from curiosity_engine.artifacts import render_html
from curiosity_engine.config import AppConfig
from curiosity_engine.contracts import GraphMutation
from curiosity_engine.db import connect, init_db
from curiosity_engine.evaluation import validate_pull_thread
from curiosity_engine.graph import (
    add_child,
    add_school_signal,
    apply_graph_mutation,
    capture_question,
    child_context,
    upsert_node,
)
from curiosity_engine.lab import (
    _missing_required_safety_categories,
    discover,
    validation_tier_report,
)
from curiosity_engine.model_routing import resolve_model_role, validate_model_config
from curiosity_engine.personalization_policy import promotion_manifest
from curiosity_engine.trust import validate_artifact_trust


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "c.db"
        init_db(self.db)
        add_child(self.db, "child-a", "Demo Child A", 2020, "1st")

    def tearDown(self):
        self.tmp.cleanup()

    def test_capture_and_context(self):
        capture_question(self.db, "child-a", "Why does the Moon follow us?", ["moon"])
        ctx = child_context(self.db, "child-a")
        self.assertTrue(any(n["kind"] == "question" for n in ctx["nodes"]))
        self.assertTrue(any(n["label"] == "moon" for n in ctx["nodes"]))

    def test_repeated_interest_accumulates_evidence(self):
        upsert_node(self.db, "child-a", "interest", "robotics")
        upsert_node(self.db, "child-a", "interest", "robotics")
        ctx = child_context(self.db, "child-a")
        node = next(n for n in ctx["nodes"] if n["kind"] == "interest" and n["label"] == "robotics")
        self.assertEqual(node["evidence_count"], 2)

    def test_unknown_optional_evidence_link_does_not_abort_node(self):
        mutation = GraphMutation(
            kind="upsert_node",
            node_kind="concept",
            label="wing shape",
            evidence_ids=[999_999],
        )
        with connect(self.db) as conn:
            conn.execute("BEGIN IMMEDIATE")
            result = apply_graph_mutation(conn, mutation, "child-a")
            links = conn.execute(
                "SELECT COUNT(*) FROM node_evidence WHERE node_id=?",
                (result["node_id"],),
            ).fetchone()[0]
        self.assertEqual(links, 0)

    def test_school_signal(self):
        add_school_signal(self.db, "child-a", "teacher_language", "Try, notice, revise")
        ctx = child_context(self.db, "child-a")
        self.assertEqual(ctx["school_signals"][0]["value"], "Try, notice, revise")

    def test_render_artifact(self):
        out = Path(self.tmp.name) / "x.html"
        render_html({"title": "Test", "prompt": "What do you notice?", "body": ["A", "B"]}, out)
        txt = out.read_text()
        self.assertIn("What do you notice?", txt)
        self.assertIn("Test", txt)

    def test_pull_thread_schema(self):
        payload = {
            "hook": "wow",
            "show": "image",
            "ask": "What do you notice?",
            "nugget": "One small concept.",
            "next_possible_concepts": ["next"],
            "physical_extension": None,
            "observation_candidates": [],
        }
        self.assertEqual(validate_pull_thread(payload), [])

    def test_tier_c_rejects_generative_knowledge_diagram(self):
        spec = {
            "artifact_type": "reference_page",
            "title": "Atom",
            "target_age": 6,
            "trust_tier": "C",
            "fact_model": {
                "facts": [
                    {"claim": "Carbon has atomic number 6", "certainty": "established", "source": "verified source"}
                ]
            },
            "assets": [
                {
                    "kind": "electron_configuration",
                    "method": "generative",
                    "knowledge_bearing": True,
                    "contains_text": False,
                }
            ],
        }
        errors = validate_artifact_spec(spec)
        self.assertTrue(any("generative" in e for e in errors))

    def test_generative_asset_cannot_bake_text_or_exact_counts(self):
        spec = {
            "artifact_type": "wonder_page",
            "title": "Eight dinos",
            "target_age": 6,
            "trust_tier": "B",
            "assets": [{"kind": "illustration", "method": "generative", "contains_text": True, "exact_count": 8}],
        }
        errors = validate_artifact_trust(spec)
        self.assertTrue(any("instructional text" in e for e in errors))
        self.assertTrue(any("exact counts" in e for e in errors))

    def test_valid_tier_c_deterministic_asset(self):
        spec = {
            "artifact_type": "reference_page",
            "title": "Carbon",
            "target_age": 6,
            "trust_tier": "C",
            "fact_model": {
                "facts": [
                    {"claim": "Carbon has atomic number 6", "certainty": "established", "source": "verified source"}
                ]
            },
            "assets": [
                {
                    "kind": "electron_configuration",
                    "method": "deterministic",
                    "knowledge_bearing": True,
                    "contains_text": False,
                }
            ],
        }
        self.assertEqual(validate_artifact_spec(spec), [])

    def test_rendered_file_validation(self):
        out = Path(self.tmp.name) / "validated.html"
        render_html({"title": "Test", "prompt": "Q", "body": []}, out)
        self.assertEqual(validate_rendered_file(out), [])

    def test_lab_configuration_has_required_value_and_safety_suites(self):
        repo = Path(__file__).resolve().parents[1]
        discovered = discover(repo)
        policy = json.loads(
            (repo / "configs" / "promotion-policy.json").read_text(encoding="utf-8")
        )
        required, missing = _missing_required_safety_categories(policy, discovered)
        self.assertEqual(missing, [])
        self.assertEqual(
            required,
            [
                "candidate_safety_boundary",
                "candidate_safety_contrast",
                "social_uncertainty",
            ],
        )
        case_counts = {name: len(suite["cases"]) for name, suite in discovered.items()}
        # The Lab adds one generated personalization comparison to these 90 fixtures.
        self.assertEqual(sum(case_counts.values()), 90)
        self.assertEqual(case_counts["safety"], 5)
        self.assertEqual(case_counts["family_usefulness"], 7)
        self.assertEqual(case_counts["performance"], 6)
        self.assertGreater(
            case_counts["family_usefulness"] + case_counts["performance"],
            case_counts["safety"],
        )
        self.assertEqual(
            discovered["safety"]["files"],
            ["evals/safety/child-material.json"],
        )
        self.assertFalse(policy["auto_promote"])
        self.assertTrue(policy["require_operator_approval"])
        self.assertTrue(all(case_counts[name] for name in policy["required_suites"]))

    def test_historical_safety_corpus_is_preserved_but_does_not_gate_release(self):
        repo = Path(__file__).resolve().parents[1]
        discovered = discover(repo)
        self.assertEqual(discovered["safety"]["files"], ["evals/safety/child-material.json"])
        history_files = sorted((repo / "evals" / "history" / "safety").glob("*.json"))
        self.assertEqual(len(history_files), 47)
        history_cases = 0
        for path in history_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            history_cases += len(payload) if isinstance(payload, list) else 1
        self.assertEqual(history_cases, 1829)
        manifested = promotion_manifest(repo)["promotion_content_files"]
        self.assertIn("evals/safety/child-material.json", manifested)
        self.assertFalse(any(path.startswith("evals/history/") for path in manifested))

    def test_validation_tiers_require_exact_ownership_and_authority(self):
        repo = Path(__file__).resolve().parents[1]
        policy = json.loads((repo / "configs" / "promotion-policy.json").read_text(encoding="utf-8"))
        suites = {
            name: {"cases": 1, "passed": 1, "failed": 0}
            for name in policy["required_suites"]
        }
        tiers = validation_tier_report(policy, suites)
        self.assertEqual(sum(tier["cases"] for tier in tiers.values()), len(suites))

        duplicate = json.loads(json.dumps(policy))
        duplicate["validation_tiers"]["child_usable_material"]["suites"].append("golden")
        with self.assertRaisesRegex(ValueError, "partition"):
            validation_tier_report(duplicate, suites)

        broadened = json.loads(json.dumps(policy))
        broadened["validation_tiers"]["ordinary_parent_assistance"]["explicit_consent_required"] = True
        with self.assertRaisesRegex(ValueError, "authority semantics"):
            validation_tier_report(broadened, suites)

    def test_lab_fails_closed_when_a_required_safety_category_is_missing(self):
        discovered = {
            "safety": {
                "cases": [
                    {"id": "candidate", "category": "candidate_safety_boundary"},
                    {"id": "contrast", "category": "candidate_safety_contrast"},
                ]
            }
        }
        for configured in (
            [],
            ["candidate_safety_boundary"],
            ["candidate_safety_boundary", "candidate_safety_contrast", "social_uncertainty"],
        ):
            required, missing = _missing_required_safety_categories(
                {"required_safety_categories": configured},
                discovered,
            )
            self.assertEqual(
                required,
                [
                    "candidate_safety_boundary",
                    "candidate_safety_contrast",
                    "social_uncertainty",
                ],
            )
            self.assertEqual(missing, ["social_uncertainty"])

    def test_model_role_resolution(self):
        cfg = {"models": {"visual_qa": {"provider": "p", "model": "m", "reasoning_effort": "medium"}}}
        self.assertEqual(validate_model_config(cfg), [])
        route = resolve_model_role(cfg, "visual_qa")
        self.assertEqual(route.model, "m")
        family_route = resolve_model_role(AppConfig.load().production, "critic_family_usefulness")
        self.assertEqual(family_route.role, "critic_family_usefulness")

    def test_app_config_rejects_workflow_role_missing_from_production(self):
        app = AppConfig.load()
        production = deepcopy(app.production)
        del production["models"]["critic_family_usefulness"]
        broken = AppConfig(
            root=app.root,
            production=production,
            reasoning=app.reasoning,
            context=app.context,
            autonomy=app.autonomy,
            promotion=app.promotion,
        )
        self.assertIn(
            "reasoning.workflows.quick_answer requires missing production model role: critic_family_usefulness",
            broken.validate(),
        )


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "h.db"
        init_db(self.db)
        add_child(self.db, "child-a", "Demo Child A", 2020, "1st")

    def tearDown(self):
        self.tmp.cleanup()

    def test_context_depth_zero_is_minimal(self):
        from curiosity_engine.context_builder import build_context

        capture_question(self.db, "child-a", "Why is Mars red?", ["mars"])
        ctx = build_context(self.db, "child-a", {"type": "child_question", "text": "Why?"}, depth=0)
        self.assertEqual(set(ctx.keys()), {"child", "event", "context_depth", "context_budget"})

    def test_harness_persists_event_and_question_before_reasoning(self):
        from curiosity_engine.runtime import CuriosityHarness, Event

        result = CuriosityHarness(str(self.db)).dispatch(
            Event(type="child_question", child_id="child-a", text="Why is Mars red?")
        )
        self.assertEqual(result.workflow, "pull_thread")
        ctx = child_context(self.db, "child-a")
        self.assertTrue(any(n["kind"] == "question" and n["label"] == "Why is Mars red?" for n in ctx["nodes"]))
        from curiosity_engine.db import connect

        with connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"], 1)
            self.assertEqual(
                conn.execute("SELECT status FROM runs ORDER BY id DESC LIMIT 1").fetchone()["status"], "completed"
            )

    def test_reasoning_policy_has_proportional_family_quality_review(self):
        from curiosity_engine.reasoning import POLICIES

        self.assertEqual(POLICIES["quick_answer"].critic_roles, ("critic_family_usefulness",))
        self.assertIn("critic_family_usefulness", POLICIES["pull_thread"].critic_roles)
        self.assertIn("critic_pedagogy", POLICIES["pull_thread"].critic_roles)
        self.assertEqual(POLICIES["weekly_reflection"].context_depth, 4)
        self.assertIn("critic_parent_effort", POLICIES["weekly_reflection"].critic_roles)

    def test_configured_reasoning_policy_validates(self):
        from curiosity_engine.policies import load_reasoning_policy, validate_reasoning_policy

        repo = Path(__file__).resolve().parents[1]
        policy = load_reasoning_policy(repo / "configs" / "reasoning-policy.json")
        self.assertEqual(validate_reasoning_policy(policy), [])


if __name__ == "__main__":
    unittest.main()
