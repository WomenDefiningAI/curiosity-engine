#!/usr/bin/env python3
"""Evaluate one unchanged source with all parent packs and predeclared fragile-case repeats."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


def digest(source, directories):
    value = hashlib.sha256()
    for directory in directories:
        for path in sorted((source / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                value.update(str(path.relative_to(source)).encode())
                value.update(path.read_bytes())
    return value.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "auth-root", "run-dir"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--usage-db", type=Path)
    parser.add_argument("--pack", type=Path, help="Use one independently prepared conversation pack")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--capture-only", action="store_true", help="Collect for human review without paid model judging")
    parser.add_argument("--typesafe-key-file", type=Path, help="Defaults to auth-root/typesafe.env")
    parser.add_argument("--purpose", choices=["development", "validation"], default="development")
    parser.add_argument("--exposure-registry", type=Path)
    args = parser.parse_args()
    source, run = args.source.resolve(), args.run_dir.resolve()
    if not 0 < args.budget_usd <= 1000:
        parser.error("A positive experiment budget is required")
    run.mkdir(parents=True, exist_ok=False, mode=0o700)
    base = source / "lab/parent_usefulness"
    runtime_hash = digest(source, ("src", "configs", "prompts", "capabilities"))
    harness_hash = digest(source, ("lab/parent_usefulness",))
    cases, inputs, rubrics, calibrations = [], {}, [], []
    directories = [args.pack.resolve()] if args.pack else [base, base / "illustrated", base / "transfer", base / "everyday", base / "photos", base / "conversations"]
    for directory in directories:
        pack = json.loads((directory / "cases.json").read_text())["cases"]
        for case in pack:
            if args.ids and case["id"] not in args.ids:
                continue
            if case["id"] in inputs:
                raise ValueError("Duplicate case identifier")
            cases.append(case)
            inputs[case["id"]] = case["input"]
        rubrics.append((directory / "rubric.md").read_text())
        calibrations.extend(json.loads((directory / "calibration.json").read_text())["examples"])
    base_count = len(cases)
    if not cases or (args.ids and set(args.ids) != set(inputs)):
        parser.error("Requested case IDs must exist in the selected packs")
    repeats = {key: 2 for key in ("PU12", "EU05", "EU06") if key in inputs} if not args.ids else {}
    for original, count in repeats.items():
        case = next(case for case in cases if case["id"] == original)
        for index in range(count):
            repeated = {**case, "id": f"{original}-repeat-{index + 1}"}
            cases.append(repeated)
            inputs[repeated["id"]] = repeated["input"]
    pack = run / "frozen-pack"
    pack.mkdir()
    write(pack / "cases.json", {"cases": cases})
    write(pack / "candidate-inputs.json", inputs)
    write(pack / "calibration.json", {"examples": calibrations})
    (pack / "rubric.md").write_text("\n\n".join(dict.fromkeys(rubrics)))
    manifest = {"source": str(source), "source_sha256": runtime_hash, "harness_sha256": harness_hash,
                "pack_sha256": digest(run, ("frozen-pack",)), "base_cases": base_count,
                "repeats": repeats, "expected_captures": len(cases), "status": "running"}
    from holdouts import register_exposure

    manifest["evaluation_purpose"] = register_exposure(
        args.exposure_registry or source / "private/evals/exposure-registry.sqlite",
        cases, runtime_hash, purpose=args.purpose, run_id=str(run))
    manifest["budget_usd"] = args.budget_usd
    write(run / "release-evaluation.json", manifest)
    capture = run / "captures"
    usage_db = (args.usage_db or run / "usage.sqlite").resolve()
    budget_args = ["--budget-usd", str(args.budget_usd), "--usage-db", str(usage_db)]
    os.environ.update(CURIOSITY_USAGE_DB=str(usage_db), CURIOSITY_BUDGET_USD=str(args.budget_usd))
    collection = subprocess.run([sys.executable, str(base / "collect.py"), "--source", str(source),
                    "--auth-root", str(args.auth_root), "--run-dir", str(capture),
                    "--inputs", str(pack / "candidate-inputs.json"), "--workers", str(args.workers), *budget_args])
    review_failed = False
    if not args.capture_only:
        review = subprocess.run([sys.executable, str(base / "review.py"), "--source", str(source),
                    "--auth-root", str(args.auth_root), "--run-dir", str(capture),
                    "--pack", str(pack), *budget_args,
                    *(["--key-file", str(args.typesafe_key_file)] if args.typesafe_key_file else [])])
        review_failed = review.returncode != 0
    ratings, turns, failures = Counter(), [], []
    for case in cases:
        folder = capture / case["id"]
        if not (folder / "bundle.json").exists():
            failures.append({"id": case["id"], "capture_status": "missing"})
            ratings["not_reviewable"] += 1
            continue
        bundle = json.loads((folder / "bundle.json").read_text())
        review = (json.loads((folder / "jev-review.json").read_text())
                  if (folder / "jev-review.json").exists() else
                  {"use_as_is": "not_reviewed", "reason": "Human/model review pending"})
        if bundle["source_sha256"] != runtime_hash or bundle["status"] != "captured":
            failures.append({"id": case["id"], "capture_status": bundle["status"], "source_mismatch": bundle["source_sha256"] != runtime_hash})
        ratings[review["use_as_is"]] += 1
        if review["use_as_is"] != "use_as_is":
            failures.append({"id": case["id"], "rating": review["use_as_is"], "reason": review["reason"]})
        turns.extend(turn["timing"]["total_seconds"] for turn in bundle["turns"]
                     if turn.get("timing", {}).get("total_seconds") is not None)
    unchanged = (digest(source, ("src", "configs", "prompts", "capabilities")) == runtime_hash
                 and digest(source, ("lab/parent_usefulness",)) == harness_hash
                 and digest(run, ("frozen-pack",)) == manifest["pack_sha256"])
    sys.path.insert(0, str(source / "src"))
    from curiosity_engine.model_usage import usage_summary

    costs = usage_summary(usage_db)
    write(run / "costs.json", costs)
    manifest.update(status=("complete" if collection.returncode == 0 and not review_failed else "incomplete") if unchanged else "invalid_source_changed", text_ratings=dict(ratings),
                    failures=failures, human_reviewed=False,
                    cost={key: value for key, value in costs.items() if key != "calls"},
                    timing={"turns": len(turns), "median_seconds": statistics.median(turns) if turns else None,
                            "p95_seconds": sorted(turns)[min(len(turns)-1, int(len(turns)*.95))] if turns else None},
                    evaluator="jev-1.13.0", evaluation_scope="text_only", visual_review="human_pending",
                    limitation="JEV text judgments only; pictures require human review. Photo cases use local Slack I/O.")
    write(run / "release-evaluation.json", manifest)
    from human_review import build_review

    build_review([capture], run / "human-review")
    print(json.dumps(manifest), flush=True)
    if not unchanged or collection.returncode or review_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
