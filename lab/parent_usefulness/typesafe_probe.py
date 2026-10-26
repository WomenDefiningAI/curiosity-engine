#!/usr/bin/env python3
"""Twelve small, synthetic typed-judgment checks. Never changes runtime routing."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PAIRS = [
    ("concept", "The worksheet shows two equal groups of three counters.", "Give us an idea using this concept.",
     "Make two equal groups with six blocks; mix them and rebuild the groups.",
     "Color the blocks in your favorite colors.", "Does the activity use equal grouping?"),
    ("open_question", "We compared a long pencil and a short pencil.", "Can one pencil be both longer and shorter?",
     "Yes, a middle pencil can be longer than a small one and shorter than a big one.",
     "Choose Make a challenge or Give us another idea to continue.", "Does the reply answer the comparison question?"),
    ("topic", "We are making clap-tap repeating patterns.", "Make it harder.",
     "Try clap-clap-tap, then pause for your child to continue the same unit.",
     "Build a tall tower and compare its height with a chair.", "Does the reply retain the repeating-pattern topic?"),
    ("constraint", "The child is not writing today.", "Give us a small counting challenge.",
     "Point to a group of three blocks, then show three fingers.",
     "Count the blocks and write the number on paper.", "Can the child complete the activity without writing?"),
    ("preparation", "We have five minutes and only blocks already on the floor.", "Give us something to do now.",
     "Take turns making a group with one more block than the other person.",
     "Print, laminate, and cut a set of twenty challenge cards before starting.", "Can they start with the stated materials and time?"),
    ("participation", "A young child can point and move blocks but does not read.", "Give us a playable first move.",
     "Put down two blocks and ask your child to move one more beside them.",
     "Explain that addition is an operation that combines quantities.", "Does the reply give a concrete action the child can take?"),
]


def requests():
    for dimension, source, parent, good, bad, instruction in PAIRS:
        for expected, answer in ((True, good), (False, bad)):
            yield {"id": f"{dimension}-{'good' if expected else 'bad'}", "expected": expected,
                   "request": {"model": "jev-1.13.0", "state": {"context": source, "parent_request": parent,
                                "answer": answer}, "questions": {"criterion": {"type": "noul",
                                "instructions": instruction, "criteria": {"true": "The answer satisfies the stated criterion.",
                                "false": "The answer does not satisfy the stated criterion."}}}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, default=Path("private/setup/typesafe.env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-usd", type=float, default=.01)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.max_usd) or not 0 < args.max_usd <= 1:
        parser.error("Use a positive budget no greater than $1 for this probe")
    cases = list(requests())
    report = {"model": "jev-1.13.0", "label_source": "authored synthetic expectations; not human ratings",
              "price_usd_per_million_input_tokens": .042, "pricing_checked": "2026-09-17",
              "pricing_source": "https://docs.typesafe.ai/models", "maximum_requests": 12,
              "budget_usd": args.max_usd, "results": [], "estimated_usd": 0., "status": "prepared"}
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (args.output / "requests.json").write_text(json.dumps(cases, indent=2))
    target = args.output / "report.json"

    def save():
        target.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if args.dry_run:
        print("Prepared 12 synthetic judgments; no API requests made.")
        return
    if not args.key_file.exists() or args.key_file.stat().st_mode & 0o077:
        parser.error("The key file must exist and be owner-only")
    key = next((line.split("=", 1)[1].strip().strip("\"'") for line in args.key_file.read_text().splitlines()
                if line.startswith("TYPESAFE_API_KEY=")), "")
    if not key:
        report["status"] = "awaiting_key"
        save()
        parser.error("TYPESAFE_API_KEY is empty in the private key file")
    for case in cases:
        data = json.dumps(case["request"]).encode()
        reserve = (len(data) + 4096) * .042 / 1e6
        if report["estimated_usd"] + reserve > args.max_usd:
            report["status"] = "budget_stopped"
            break
        started = time.monotonic()
        request = Request("https://api.typesafe.ai/v1/systemone", data=data,
                          headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
            usage = result.get("usage") or {}
            if "input_tokens" not in usage:
                raise ValueError("Usage missing; stopping instead of recording zero cost")
            cost = usage["input_tokens"] * .042 / 1e6
            probability = result["answers"]["criterion"]["noul"]
            report["estimated_usd"] += cost
            report["results"].append({"id": case["id"], "expected": case["expected"],
                                      "probability": probability, "agrees_at_illustrative_half_threshold": (probability >= .5) == case["expected"],
                                      "usage": usage, "estimated_usd": cost, "seconds": time.monotonic() - started,
                                      "response_model": result.get("model")})
            save()
        except (HTTPError, URLError, ValueError, KeyError, TimeoutError) as exc:
            report.update(status="stopped_after_error", unknown_charge_reservation_usd=reserve,
                          error_type=type(exc).__name__)
            break
    else:
        report["status"] = "complete"
    save()
    print(json.dumps({key: report[key] for key in ("status", "estimated_usd")}))


if __name__ == "__main__":
    main()
