#!/usr/bin/env python3
"""Compare captured experiences with source labels hidden and randomized A/B order."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Preference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    choice: Literal["A", "B", "tie", "neither"]
    reason: str
    evidence: list[str] = Field(min_length=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "auth-root", "baseline", "candidate", "pack", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--usage-db", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.budget_usd <= 1000:
        parser.error("A positive experiment budget is required")
    for name in ("source", "auth_root", "baseline", "candidate", "pack", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        CURIOSITY_REPO_ROOT=str(args.source),
        CURIOSITY_HOME=str(args.output),
        CURIOSITY_PRIVATE_ROOT=str(args.output / "private"),
        CURIOSITY_MODEL_ENV=str(args.auth_root / "model.env"),
        CURIOSITY_BRAIN_CONFIG=str(args.auth_root / "brain.json"),
        CURIOSITY_USAGE_DB=str(args.usage_db.resolve()),
        CURIOSITY_BUDGET_USD=str(args.budget_usd),
        CURIOSITY_USAGE_PHASE="comparison",
    )
    sys.path.insert(0, str(args.source / "src"))
    from curiosity_engine.config import AppConfig
    from curiosity_engine.openai_backend import data_url_for_image
    from curiosity_engine.runtime import configured_backend

    cases = json.loads((args.pack / "cases.json").read_text())["cases"]
    rubric = (args.pack / "rubric.md").read_text()

    def compare(case):
        from curiosity_engine.model_usage import CASE

        CASE.set(str(args.output / case["id"]))
        target = args.output / f"{case['id']}.json"
        if target.exists():
            return {"id": case["id"], "status": "existing_review_preserved"}
        # Stable opaque ordering makes the comparison reproducible without
        # revealing the source identity to the judge.
        candidate_first = hashlib.sha256(case["id"].encode()).digest()[0] % 2 == 0
        order = ["candidate", "baseline"] if candidate_first else ["baseline", "candidate"]
        experiences, images, hashes = {}, [], {}
        for label, version in zip(("A", "B"), order, strict=True):
            folder = getattr(args, version) / case["id"]
            bundle_path = folder / "bundle.json"
            hashes[version] = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
            bundle = json.loads(bundle_path.read_text())
            turns = []
            for turn in bundle["turns"]:
                item = {
                    key: turn.get(key)
                    for key in ("number", "parent", "message", "interaction", "status", "capture_status", "timing")
                }
                item["image_indexes"] = []
                item["source_photo_indexes"] = []
                for source_file in turn.get("input_files", []):
                    images.append(data_url_for_image(folder / source_file["path"]))
                    item["source_photo_indexes"].append(len(images))
                for file in turn["files"]:
                    path = folder / file["path"]
                    pages = []
                    if path.suffix == ".pdf":
                        prefix = path.parent / "comparison-page"
                        subprocess.run(
                            ["pdftoppm", "-scale-to", "1700", "-png", str(path), str(prefix)],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                        )
                        pages = sorted(path.parent.glob("comparison-page-*.png"))
                    elif path.suffix == ".png" and "visual-" in path.name:
                        pages = [path]
                    for page in pages:
                        images.append(data_url_for_image(page))
                        item["image_indexes"].append(len(images))
                item["delivered_file_count"] = len(turn["files"])
                turns.append(item)
            if bundle["status"] != "captured":
                raise ValueError(f"{case['id']} has an incomplete capture; do not rank missing evidence")
            experiences[label] = turns
        backend = configured_backend(AppConfig.load(), role="judge")
        if backend is None:
            raise RuntimeError("A real model judge is required")
        value = backend.complete(
            role="judge",
            system=(
                "Compare two actual delivered experiences for a parent with small kids. Choose the one this "
                "parent could use with less work and better fulfillment of their request. Inspect every supplied "
                "page; an attached file or a claim about pictures is not enough. Evaluate the whole conversation "
                "Use the separately indexed source photos to check the actual concepts and relationships; "
                "source photos do not count as newly delivered output. "
                "including revisions and waiting. Neither is appropriate if both leave the central request "
                "unfulfilled. Tie is appropriate if both work comparably. You do not know their source versions. "
                "Use concrete evidence, not stylistic preferences or hypothetical risks. A parent can read a short "
                "instruction aloud to a nonreader. This predicts utility; it is not human feedback.\n\n" + rubric
            ),
            payload={
                "input": case["input"],
                "expectations": case["review"],
                "experiences": experiences,
                "image_data_urls": images,
            },
            response_model=Preference,
        )
        (args.output / f"{case['id']}.raw.json").write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
        result = Preference.model_validate(value).model_dump()
        result.update(
            id=case["id"],
            position_mapping=dict(zip(("A", "B"), order, strict=True)),
            bundle_sha256=hashes,
            reviewer_model=backend.model,
            human_reviewed=False,
            winner=order[("A", "B").index(result["choice"])] if result["choice"] in {"A", "B"} else result["choice"],
        )
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return {"id": case["id"], "winner": result["winner"]}

    selected = [case for case in cases if not args.ids or case["id"] in args.ids]
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(compare, case): case["id"] for case in selected}
        for future in concurrent.futures.as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:
                failure = {"id": futures[future], "status": "not_reviewable", "error": str(exc)}
                failures.append(failure)
                print(json.dumps(failure), flush=True)
    error_path = args.output / "errors.json"
    previous = json.loads(error_path.read_text()) if error_path.exists() else []
    remaining = {
        item["id"]: item for item in [*previous, *failures] if not (args.output / f"{item['id']}.json").exists()
    }
    if remaining:
        error_path.write_text(json.dumps(list(remaining.values()), indent=2) + "\n")
        raise SystemExit(1)
    error_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
