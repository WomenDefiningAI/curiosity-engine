from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .config import AppConfig
from .contracts import ImageResearchQuery, StrictModel, WorksheetArchetype
from .runtime import configured_backend


class VisualEvalDecision(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    response_shape: Literal["parent_chat", "quick_answer", "learning_thread"]
    primary_visual: Literal["none", "researched_photo", "deterministic_diagram", "decorative_art"]
    printable: bool
    worksheet_archetype: WorksheetArchetype | None = None
    public_research_query: str | None = Field(default=None, max_length=180)
    source_priority: list[Literal["smithsonian", "nasa", "noaa", "usgs", "wikimedia"]] = Field(
        default_factory=list,
        max_length=5,
    )
    decorative_subject: str | None = Field(default=None, max_length=180)
    rationale: str = Field(min_length=1, max_length=500)

    @field_validator("decorative_subject")
    @classmethod
    def normalize_decorative_subject(cls, value: str | None) -> str | None:
        return " ".join(value.split()).casefold() if value else None

    @model_validator(mode="after")
    def coherent_release(self) -> VisualEvalDecision:
        if self.printable and self.worksheet_archetype is None:
            raise ValueError("printable decisions require a worksheet archetype")
        if self.worksheet_archetype == "legacy_mixed":
            raise ValueError("legacy_mixed is not available for new visual plans")
        if not self.printable and self.worksheet_archetype is not None:
            raise ValueError("non-printable decisions may not select a worksheet archetype")
        if self.primary_visual == "researched_photo":
            if not self.public_research_query:
                raise ValueError("researched-photo decisions require a public query")
            ImageResearchQuery(query=self.public_research_query, purpose="observe")
            if self.decorative_subject:
                raise ValueError("one turn may not select researched and generated art")
        elif self.public_research_query or self.source_priority:
            raise ValueError("only researched-photo decisions may include source selection")
        if self.primary_visual == "decorative_art":
            if not self.decorative_subject:
                raise ValueError("decorative-art decisions require a generic scene")
        elif self.decorative_subject:
            raise ValueError("only decorative-art decisions may include a scene")
        if self.response_shape == "parent_chat" and (self.primary_visual != "none" or self.printable):
            raise ValueError("parent chat may not silently become a child-facing lesson")
        return self


class VisualEvalBatch(StrictModel):
    decisions: list[VisualEvalDecision] = Field(min_length=25, max_length=25)


class VisualJudgeVerdict(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    verdict: Literal["pass", "fail"]
    score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(min_length=1, max_length=6)


class VisualJudgeBatch(StrictModel):
    verdicts: list[VisualJudgeVerdict] = Field(min_length=25, max_length=25)


def load_round(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 25:
        raise ValueError(f"{path} must contain exactly 25 visual evaluation cases")
    ids = [str(case.get("id") or "") for case in value if isinstance(case, dict)]
    if len(ids) != 25 or len(set(ids)) != 25 or any(not value for value in ids):
        raise ValueError(f"{path} case IDs must be non-empty and unique")
    return value


def _planner_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": case["id"],
            "scenario": case["scenario"],
            "input": case["input"],
            "thread_context": case.get("thread_context") or [],
            "must": case.get("must") or [],
            "must_not": case.get("must_not") or [],
        }
        for case in cases
    ]


def _complete_batch(backend: Any, *, role: str, system: str, payload: dict[str, Any], model: type[StrictModel]):
    errors: list[str] = []
    for attempt in range(2):
        try:
            result = backend.complete(
                role=role,
                system=system,
                payload={
                    **payload,
                    "format_repair": (
                        "Return exactly 25 unique items in the requested contract, one per supplied case."
                        if attempt
                        else None
                    ),
                },
                response_model=model,
            )
            return model.model_validate(result)
        except Exception as exc:
            errors.append(f"{exc.__class__.__name__}: {str(exc)[:500]}")
    raise RuntimeError("batch model output failed twice: " + " | ".join(errors))


def plan_round(cases: list[dict[str, Any]], backend: Any, visual_policy: str) -> VisualEvalBatch:
    return _complete_batch(
        backend,
        role="reasoning",
        system=(
            "Act as the visual planner inside Curiosity Engine for 25 independent early-elementary family turns. "
            "Return one decision per case. Do not answer the child; decide the proportionate response shape and "
            "visual release units. response_shape=parent_chat means the parent is talking to or operating the system "
            "(setup, context, feedback, tuning, clarification); response_shape=quick_answer means the parent needs a "
            "short child-ready answer, including a careful answer to a sensitive child question; learning_thread "
            "means a richer child-facing exploration or requested activity. Never select legacy_mixed for a new "
            "worksheet. Apply this exact production policy:\n\n" + visual_policy
        ),
        payload={"cases": _planner_cases(cases)},
        model=VisualEvalBatch,
    )


def judge_round(
    cases: list[dict[str, Any]],
    decisions: VisualEvalBatch,
    backend: Any,
    visual_policy: str,
) -> VisualJudgeBatch:
    return _complete_batch(
        backend,
        role="judge",
        system=(
            "Independently judge 25 visual-planning decisions for an elementary-age family harness. Fail decorative "
            "art that replaces a useful worksheet, a worksheet that over-structures parent chat, a generated image "
            "used for facts or diagrams, a researched photo where a mechanism diagram is needed, an unminimized "
            "public query, a poor worksheet archetype, excess parent effort, repetition that ignores thread context, "
            "or any choice that could incorrectly become evidence about the child. Require genuinely useful visuals, "
            "not pictures added by habit. A pass should score at least 0.80. Use this production policy:\n\n"
            + visual_policy
        ),
        payload={"cases": cases, "decisions": decisions.model_dump(mode="json")},
        model=VisualJudgeBatch,
    )


def deterministic_failures(case: dict[str, Any], decision: VisualEvalDecision) -> list[str]:
    expected = case.get("expected") or {}
    failures: list[str] = []
    shapes = expected.get("response_shapes") or []
    visuals = expected.get("primary_visuals") or []
    archetypes = expected.get("worksheet_archetypes") or []
    if shapes and decision.response_shape not in shapes:
        failures.append(f"response shape {decision.response_shape} not in {shapes}")
    if visuals and decision.primary_visual not in visuals:
        failures.append(f"primary visual {decision.primary_visual} not in {visuals}")
    if "printable" in expected and decision.printable != bool(expected["printable"]):
        failures.append(f"printable={decision.printable} expected {bool(expected['printable'])}")
    if archetypes and decision.worksheet_archetype not in archetypes:
        failures.append(f"worksheet archetype {decision.worksheet_archetype} not in {archetypes}")
    if decision.primary_visual == "researched_photo" and decision.public_research_query:
        try:
            ImageResearchQuery(query=decision.public_research_query, purpose="observe")
        except ValueError as exc:
            failures.append(f"unsafe research query: {exc}")
    return failures


def run_round(repo: Path, cases_path: Path, report_path: Path) -> dict[str, Any]:
    cases = load_round(cases_path)
    config = AppConfig.load(repo)
    planner = configured_backend(config, role="reasoning")
    judge = configured_backend(config, role="judge")
    if planner is None or judge is None:
        raise RuntimeError("visual review requires configured API reasoning and judge routes")
    visual_policy = (repo / "prompts" / "visual-policy-v1.md").read_text(encoding="utf-8").strip()
    decisions = plan_round(cases, planner, visual_policy)
    decision_by_id = {item.id: item for item in decisions.decisions}
    if set(decision_by_id) != {case["id"] for case in cases}:
        raise RuntimeError("visual planner omitted or invented case IDs")
    judgments = judge_round(cases, decisions, judge, visual_policy)
    judge_by_id = {item.id: item for item in judgments.verdicts}
    if set(judge_by_id) != set(decision_by_id):
        raise RuntimeError("visual judge omitted or invented case IDs")
    results: list[dict[str, Any]] = []
    for case in cases:
        decision = decision_by_id[case["id"]]
        verdict = judge_by_id[case["id"]]
        hard_failures = deterministic_failures(case, decision)
        passed = not hard_failures and verdict.verdict == "pass" and verdict.score >= 0.8
        results.append(
            {
                "id": case["id"],
                "passed": passed,
                "decision": decision.model_dump(mode="json"),
                "judge": verdict.model_dump(mode="json"),
                "hard_failures": hard_failures,
            }
        )
    passed = sum(int(result["passed"]) for result in results)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "private": True,
        "source_cases": str(cases_path.relative_to(repo)),
        "cases": 25,
        "passed": passed,
        "pass_rate": passed / 25,
        "status": "pass" if passed >= 23 else "fail",
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.chmod(0o600)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="curiosity-visual-eval")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_round(args.repo.resolve(), args.cases.resolve(), args.report.resolve())
    print(
        f"VISUAL REVIEW {report['passed']}/25 pass_rate={report['pass_rate']:.1%} "
        f"status={report['status']} report={args.report}"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
