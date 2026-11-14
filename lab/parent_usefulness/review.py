#!/usr/bin/env python3
"""Fast JEV text review: one call per captured conversation, no OpenAI judge."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from curiosity_engine.text_review import POLICY, QUESTIONS, classify


def prepare(folder):
    bundle = json.loads((folder / "bundle.json").read_text())
    turns, errors, visual = [], [], False
    evidence = {"bundle": hashlib.sha256((folder / "bundle.json").read_bytes()).hexdigest(), "files": {}}
    for turn in bundle.get("turns", []):
        item = {k: turn.get(k) for k in ("number", "parent", "message", "status", "capture_status")}
        item.update(source_photos=len(turn.get("input_files", [])), delivered_files=len(turn.get("files", [])), printed_text=[])
        visual |= bool(turn.get("input_files"))
        for file in turn.get("files", []):
            path = (folder / file["path"]).resolve()
            if not path.is_relative_to(folder.resolve()) or not path.is_file():
                errors.append("A delivered file is missing")
                continue
            evidence["files"][file["path"]] = hashlib.sha256(path.read_bytes()).hexdigest()
            visual |= path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
            if path.suffix.lower() == ".pdf":
                try:
                    result = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True,
                                            capture_output=True, text=True, timeout=10)
                    item["printed_text"].append(result.stdout.strip())
                except (OSError, subprocess.SubprocessError):
                    errors.append("Printable text could not be extracted")
        turns.append(item)
    state = {"children": bundle.get("input", {}).get("children", []), "turns": turns,
             "visuals_need_human_review": visual}
    fingerprint = hashlib.sha256(json.dumps({"state": state, "evidence": evidence, "questions": QUESTIONS,
                                            "policy": POLICY}, sort_keys=True).encode()).hexdigest()
    if not turns:
        errors.append("No conversation was captured")
    return bundle, state, fingerprint, errors, evidence



def review_folder(folder, *, key):
    from curiosity_engine.model_usage import estimate_cost, usage_case
    from curiosity_engine.typesafe_backend import MODEL, evaluate

    bundle, state, fingerprint, errors, evidence = prepare(folder)
    target = folder / "jev-review.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if previous.get("input_sha256") == fingerprint and previous.get("status") == "complete":
            return previous
    result = {"reviewer_kind": "jev_text", "reviewer_model": MODEL, "policy": POLICY,
              "input_sha256": fingerprint, "evidence": evidence, "human_reviewed": False, "scope": "text_only",
              "visual_review_required": state["visuals_need_human_review"], "status": "not_reviewable",
              "use_as_is": "not_reviewable", "gate": "review", "main_issue": None}
    started = time.monotonic()
    if bundle["status"] != "captured" or errors:
        result["reason"] = "; ".join(errors) or "Capture is incomplete"
    else:
        try:
            with usage_case(str(folder.resolve())):
                response = evaluate(state, QUESTIONS, key=key)
            result.update(classify(response), status="complete", usage=response.get("usage"), raw_answers=response["answers"])
            result["estimated_cost_usd"] = estimate_cost(MODEL, response.get("usage"))
        except Exception as exc:
            result.update(reason="Text review unavailable; rate it yourself.", error_type=type(exc).__name__)
    result["seconds"] = time.monotonic() - started
    if target.exists():
        archive = folder / ("jev-review-" + hashlib.sha256(target.read_bytes()).hexdigest()[:16] + ".json")
        if not archive.exists():
            archive.write_bytes(target.read_bytes())
    target.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--runs", nargs="+", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--auth-root", type=Path)
    p.add_argument("--key-file", type=Path)
    p.add_argument("--pack", type=Path, help="Compatibility only; expected labels are never sent")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--usage-db", type=Path)
    args = p.parse_args()
    runs = args.runs or ([args.run_dir] if args.run_dir else [])
    if not runs or not 0 < args.budget_usd <= 1000:
        p.error("Supply capture roots and a positive budget")
    source = args.source.resolve()
    sys.path.insert(0, str(source / "src"))
    from curiosity_engine.model_usage import usage_summary
    from curiosity_engine.typesafe_backend import load_key

    key = load_key(args.key_file or (args.auth_root or source / "private/setup") / "typesafe.env")
    ledger = (args.usage_db or runs[0] / "jev-usage.sqlite").resolve()
    os.environ.update(CURIOSITY_USAGE_DB=str(ledger), CURIOSITY_BUDGET_USD=str(args.budget_usd),
                      CURIOSITY_USAGE_PHASE="evaluation")
    paths = [path for run in runs for path in sorted(run.glob("*/bundle.json")) if not args.ids or path.parent.name in args.ids]
    if not paths:
        p.error("No matching captured conversations")
    failed = False
    for path in paths:  # Sequential. One JEV call, two typed choices.
        result = review_folder(path.parent, key=key)
        failed |= result["status"] != "complete"
        print(json.dumps({"id": path.parent.name, "rating": result["use_as_is"], "seconds": result["seconds"]}), flush=True)
    ledger.with_suffix(".json").write_text(json.dumps(usage_summary(ledger), indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
