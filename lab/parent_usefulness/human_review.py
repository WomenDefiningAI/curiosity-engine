#!/usr/bin/env python3
"""Build an offline, portable review page from complete captured conversations."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def build_review(runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    assets = output / "assets"
    assets.mkdir()
    records = []
    for run in runs:
        for bundle_path in sorted(Path(run).glob("*/bundle.json")):
            folder = bundle_path.parent
            bundle = json.loads(bundle_path.read_text())
            record = {"id": bundle["id"], "source": bundle.get("source_sha256", "unknown"),
                      "status": bundle["status"], "turns": [], "errors": [],
                      "cost": bundle.get("estimated_cost_usd"), "unknown_cost_calls": bundle.get("unknown_cost_calls"),
                      "delivery": bundle.get("delivery", "Local capture"), "input": bundle.get("input")}

            def copy_asset(name, folder=folder, record=record):
                path = (folder / name).resolve()
                if not path.is_relative_to(folder.resolve()) or not path.is_file():
                    record["errors"].append(f"Missing review file: {name}")
                    return None
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                target = assets / (digest + path.suffix.lower())
                if not target.exists():
                    shutil.copyfile(path, target)
                return {"url": "assets/" + target.name, "sha256": digest}

            for turn in bundle.get("turns", []):
                item = {key: turn.get(key) for key in ("number", "parent", "message", "status", "timing")}
                item.update(sources=[], images=[], pdfs=[])
                for file in turn.get("input_files", []):
                    asset = copy_asset(file["path"])
                    if asset:
                        item["sources"].append(asset)
                for file in turn.get("files", []):
                    name = file["path"]
                    asset = copy_asset(name)
                    if not asset:
                        continue
                    if name.lower().endswith(".pdf"):
                        item["pdfs"].append(asset)
                        prefix = assets / asset["sha256"]
                        try:
                            subprocess.run(["pdftoppm", "-scale-to", "1500", "-png", str(output / asset["url"]),
                                            str(prefix)], check=True, capture_output=True)
                            item["images"].extend({"url": "assets/" + page.name} for page in sorted(assets.glob(prefix.name + "-*.png")))
                        except (OSError, subprocess.CalledProcessError):
                            record["errors"].append("PDF page rendering unavailable; open the PDF to inspect all pages.")
                    elif name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        item["images"].append({**asset, "source_name": name})
                # Prefer rendered PDF pages to a duplicate preview of the same page.
                if item["pdfs"] and any("source_name" not in x for x in item["images"]):
                    item["images"] = [x for x in item["images"] if Path(x.get("source_name", "")).name != "preview.png"]
                record["turns"].append(item)
            record["fingerprint"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            # Judge changes never invalidate the human's rating of the same output.
            review_path = folder / "jev-review.json"
            record["jev"] = None
            if review_path.exists():
                review = json.loads(review_path.read_text())
                evidence = review.get("evidence", {})
                current = evidence.get("bundle") == hashlib.sha256(bundle_path.read_bytes()).hexdigest()
                for name, digest in evidence.get("files", {}).items():
                    path = (folder / name).resolve()
                    current &= (path.is_relative_to(folder.resolve()) and path.is_file()
                                and hashlib.sha256(path.read_bytes()).hexdigest() == digest)
                if current and review.get("scope") == "text_only":
                    record["jev"] = {k: review.get(k) for k in (
                        "status", "use_as_is", "main_issue", "reason", "gate", "seconds", "estimated_cost_usd",
                        "probabilities", "policy", "input_sha256", "reviewer_model", "visual_review_required")}
            records.append(record)
    if not records:
        raise ValueError("No captured conversations found")
    payload = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("human_review.html").read_text()
    (output / "index.html").write_text(template.replace("__RECORDS__", payload))
    (output / "review-manifest.json").write_text(json.dumps(records, indent=2))
    return {"conversations": len(records), "page": str(output / "index.html")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_review(args.runs, args.output)))
