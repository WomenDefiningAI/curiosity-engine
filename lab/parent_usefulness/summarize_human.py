#!/usr/bin/env python3
"""Score exported human judgments against the exact reviewed output versions."""
import argparse
import json
from collections import Counter
from pathlib import Path


def summarize(manifest, exported):
    if exported.get("version") not in {2, 3} or exported.get("reviewer_kind") != "human":
        raise ValueError("Expected an exported human-review file")
    text_scope = exported["version"] == 3
    exported_records = {r["fingerprint"]: r for r in exported.get("records", [])}
    records = {row["fingerprint"]: row for row in manifest}
    ratings, costs, durations, accepted, unknown_costs, results = Counter(), 0., [], 0, 0, []
    comparison = Counter()
    visual_ratings = Counter()
    matrix = {}
    for fingerprint, record in records.items():
        vote = exported.get("reviews", {}).get(fingerprint, {})
        rating = vote.get("text_rating" if text_scope else "rating") or "not_reviewed"
        if rating not in {"use_as_is", "small_edit_needed", "would_not_use", "not_reviewable", "not_reviewed"}:
            raise ValueError("Unknown human rating")
        if record["status"] != "captured" and rating in {"use_as_is", "small_edit_needed"}:
            raise ValueError("Incomplete captures cannot become quality passes")
        ratings[rating] += 1
        has_visual = any(t.get("images") or t.get("sources") or t.get("pdfs") for t in record["turns"])
        visual = vote.get("visual_rating") or "not_reviewed"
        if text_scope and has_visual:
            if visual not in {"works", "needs_fix", "not_reviewable", "not_reviewed"}:
                raise ValueError("Unknown picture rating")
            visual_ratings[visual] += 1
        accepted += rating == "use_as_is" and (not text_scope or not has_visual or visual == "works")
        jev = record.get("jev") or {}
        exported_record = exported_records.get(fingerprint, {})
        if text_scope and rating in {"use_as_is", "small_edit_needed", "would_not_use"} and jev.get("status") == "complete":
            if (exported_record.get("jev_input_sha256") == jev.get("input_sha256")
                    and exported_record.get("jev_policy") == jev.get("policy")):
                predicted = jev["use_as_is"]
                comparison["compared"] += 1
                comparison["same_verdict"] += rating == predicted
                comparison["jev_approved_text_human_would_change"] += predicted == "use_as_is" and rating != "use_as_is"
                comparison["jev_flagged_text_human_would_use"] += predicted != "use_as_is" and rating == "use_as_is"
                comparison["rated_before_seeing_jev"] += vote.get("jev_seen_before_rating") is False
                row = matrix.setdefault(rating, {})
                row[predicted] = row.get(predicted, 0) + 1
            else:
                comparison["changed_judge_not_compared"] += 1
        if record.get("cost") is None or record.get("unknown_cost_calls") != 0:
            unknown_costs += 1
        else:
            costs += record["cost"]
        durations.extend(turn["timing"]["total_seconds"] for turn in record["turns"]
                         if (turn.get("timing") or {}).get("total_seconds") is not None)
        results.append({"id": record["id"], "source": record["source"], "fingerprint": fingerprint,
                        "rating": rating, "visual_rating": visual if text_scope and has_visual else None,
                        "main_issue": vote.get("main_issue"), "remaining_parent_work": vote.get("note", "")})
    return {"reviewer_kind": "human", "outputs": len(records), "ratings": dict(ratings),
            "accepted_conversations": accepted, "reviewed_outputs": len(records) - ratings["not_reviewed"],
            "acceptance_rate_among_reviewed": accepted / (len(records) - ratings["not_reviewed"])
            if len(records) > ratings["not_reviewed"] and not visual_ratings["not_reviewed"] else None,
            "rating_scope": "text" if text_scope else "whole_experience", "visual_ratings": dict(visual_ratings),
            "jev_comparison": dict(comparison), "human_rows_jev_columns": matrix,
            "known_generation_cost_usd": costs, "outputs_with_unknown_cost": unknown_costs,
            "generation_cost_per_accepted_conversation_usd": costs / accepted
            if accepted and not unknown_costs and not ratings["not_reviewed"] and not visual_ratings["not_reviewed"] else None,
            "p95_turn_seconds": sorted(durations)[min(len(durations)-1, int(len(durations)*.95))] if durations else None,
            "limitation": "Parent usability judgments, not measured child engagement or learning. Evaluation costs are reported separately in the usage ledger.",
            "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads((args.page / "review-manifest.json").read_text()),
                       json.loads(args.reviews.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}))
