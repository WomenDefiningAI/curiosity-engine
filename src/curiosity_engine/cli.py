from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .actions import list_actions
from .artifact_validation import validate_artifact_spec, validate_rendered_file
from .artifacts import load_spec, render_html, render_pdf
from .config import repository_root
from .contracts import Event, FeedbackInput
from .db import init_db
from .director import AutonomousDirector
from .feedback import record_feedback
from .graph import add_child, add_observation, add_school_signal, capture_question, child_context, upsert_node
from .interaction import (
    add_parent,
    create_pairing_code,
    list_bindings,
    list_inbox,
    onboarding_status,
    revoke_binding,
    set_household_resource_context_mode,
    setup_household,
)
from .onboarding import doctor, write_setup_report
from .printer import approve_artifact, print_artifact
from .resources import discover_private_catalogs, index_collection, resource_inventory, search_resources
from .runtime import CuriosityHarness
from .service import CuriosityService
from .trust import trust_summary


def dump(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _default_db() -> str:
    return os.environ.get("CURIOSITY_DB") or str(repository_root() / "private" / "data" / "curiosity.db")


def _default_output() -> str:
    return os.environ.get("CURIOSITY_OUTPUT") or str(repository_root() / "private" / "output")


def add_db(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default=_default_db())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="curiosity", description="Local-first family curiosity harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    command = sub.add_parser("init")
    add_db(command)

    command = sub.add_parser("doctor", help="Check local setup without printing secrets or family content")
    add_db(command)
    command.add_argument("--write-report", action="store_true")

    command = sub.add_parser("setup", help="Configure the first parent and household defaults")
    add_db(command)
    command.add_argument("--owner-name", required=True)
    command.add_argument("--timezone", required=True, help="IANA timezone, such as America/New_York")
    command.add_argument("--quiet-start", default="20:00")
    command.add_argument("--quiet-end", default="07:00")
    command.add_argument("--enable-weekly", action="store_true")
    command.add_argument(
        "--resource-context-mode",
        choices=["metadata_only", "selected_excerpts"],
        default="metadata_only",
        help="Whether bounded licensed excerpts may enter hosted-model requests",
    )

    parent = sub.add_parser("parent")
    parent_sub = parent.add_subparsers(dest="parent_cmd", required=True)
    command = parent_sub.add_parser("add")
    add_db(command)
    command.add_argument("--name", required=True)
    command = parent_sub.add_parser("list")
    add_db(command)

    slack = sub.add_parser("slack")
    slack_sub = slack.add_subparsers(dest="slack_cmd", required=True)
    command = slack_sub.add_parser("pair-code", help="Create a short-lived code for one parent")
    add_db(command)
    command.add_argument("--parent", help="Parent ID; defaults to the household owner")
    command.add_argument("--ttl", type=int, default=15)
    command = slack_sub.add_parser("bindings")
    add_db(command)
    command = slack_sub.add_parser("revoke")
    add_db(command)
    command.add_argument("--binding", required=True)
    command = slack_sub.add_parser("status")
    add_db(command)
    command = slack_sub.add_parser("run", help="Run the parent-only Slack connector in Socket Mode")
    add_db(command)
    command.add_argument("--output-dir", default=_default_output())

    inbox = sub.add_parser("inbox")
    inbox_sub = inbox.add_subparsers(dest="inbox_cmd", required=True)
    command = inbox_sub.add_parser("list")
    add_db(command)
    command.add_argument("--status", default="unassigned", choices=["unassigned", "assigned", "dismissed", "all"])
    command = inbox_sub.add_parser("assign")
    add_db(command)
    command.add_argument("--id", required=True)
    command.add_argument("--child", required=True)
    command = inbox_sub.add_parser("dismiss")
    add_db(command)
    command.add_argument("--id", required=True)

    child = sub.add_parser("child")
    child_sub = child.add_subparsers(dest="child_cmd", required=True)
    command = child_sub.add_parser("add")
    add_db(command)
    command.add_argument("--id", required=True)
    command.add_argument("--name", required=True)
    command.add_argument("--birth-year", type=int)
    command.add_argument("--grade")

    command = sub.add_parser("ask")
    add_db(command)
    command.add_argument("--child", required=True)
    command.add_argument("--question", required=True)
    command.add_argument("--topic", action="append", default=[])
    command.add_argument("--source", default="parent")
    command.add_argument("--include-private-excerpts", action="store_true")
    command.add_argument("--event-id")

    command = sub.add_parser("capture")
    add_db(command)
    command.add_argument("--child", required=True)
    command.add_argument("--question", required=True)
    command.add_argument("--topic", action="append", default=[])
    command.add_argument("--source", default="parent")

    command = sub.add_parser("observe")
    add_db(command)
    command.add_argument("--child", required=True)
    command.add_argument("--kind", required=True)
    command.add_argument("--text", required=True)
    command.add_argument("--source", default="parent")
    command.add_argument("--confidence", type=float, default=1.0)

    command = sub.add_parser("interest")
    add_db(command)
    command.add_argument("--child", required=True)
    command.add_argument("--topic", required=True)
    command.add_argument("--confidence", type=float, default=0.7)

    command = sub.add_parser("school")
    add_db(command)
    command.add_argument("--child", required=True)
    command.add_argument("--category", required=True)
    command.add_argument("--value", required=True)
    command.add_argument("--source-ref")
    command.add_argument("--expires-at")

    command = sub.add_parser("context")
    add_db(command)
    command.add_argument("--child", required=True)

    command = sub.add_parser("event")
    add_db(command)
    command.add_argument("--type", required=True)
    command.add_argument("--child")
    command.add_argument("--text", required=True)
    command.add_argument("--source", default="parent")
    command.add_argument("--event-id")
    command.add_argument("--metadata-json", default="{}")

    resource = sub.add_parser("resource")
    resource_sub = resource.add_subparsers(dest="resource_cmd", required=True)
    command = resource_sub.add_parser("index")
    add_db(command)
    command.add_argument("--catalog", help="Private catalog path; auto-detected when exactly one exists")
    command.add_argument("--repo", default=str(repository_root()))
    command = resource_sub.add_parser("status")
    add_db(command)
    command = resource_sub.add_parser("mode", help="Set household private-resource disclosure mode")
    add_db(command)
    command.add_argument("--mode", required=True, choices=["metadata_only", "selected_excerpts"])
    command = resource_sub.add_parser("search")
    add_db(command)
    command.add_argument("query")
    command.add_argument("--include-private-excerpts", action="store_true")
    command.add_argument("--limit", type=int, default=5)

    artifact = sub.add_parser("artifact")
    artifact_sub = artifact.add_subparsers(dest="artifact_cmd", required=True)
    command = artifact_sub.add_parser("render")
    command.add_argument("--spec", required=True)
    command.add_argument("--out", required=True)
    command = artifact_sub.add_parser("validate")
    command.add_argument("--spec", required=True)
    command.add_argument("--rendered")
    command = artifact_sub.add_parser("create")
    add_db(command)
    command.add_argument("--child", required=True)
    command.add_argument("--spec", required=True)
    command.add_argument("--output-dir", default=_default_output())
    command = artifact_sub.add_parser("approve")
    add_db(command)
    command.add_argument("--id", required=True)
    command.add_argument("--note")
    command = artifact_sub.add_parser("print")
    add_db(command)
    command.add_argument("--id", required=True)
    command.add_argument("--approval", required=True)
    command.add_argument("--printer")
    command.add_argument(
        "--send", action="store_true", help="Actually send; without this flag the command is a dry-run"
    )

    action = sub.add_parser("action")
    action_sub = action.add_subparsers(dest="action_cmd", required=True)
    command = action_sub.add_parser("list")
    add_db(command)
    command.add_argument("--status")
    command = action_sub.add_parser("execute")
    add_db(command)
    command.add_argument("--id", required=True)
    command.add_argument("--output-dir", default=_default_output())

    command = sub.add_parser("feedback")
    add_db(command)
    command.add_argument("--child", required=True)
    command.add_argument("--outcome", required=True)
    command.add_argument("--note")
    command.add_argument("--experience")
    command.add_argument("--artifact")

    command = sub.add_parser("reflect")
    add_db(command)
    command.add_argument("--child", required=True)

    command = sub.add_parser("worker")
    add_db(command)
    command.add_argument("--drain", action="store_true", help="Process every currently ready job")

    command = sub.add_parser("serve")
    add_db(command)
    command.add_argument("--output-dir", default=_default_output())
    command.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "::1", "localhost"])
    command.add_argument("--port", type=int, default=8766)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "init":
        backup = init_db(args.db)
        dump({"status": "ok", "db": str(Path(args.db).resolve()), "migration_backup": str(backup) if backup else None})
    elif args.cmd == "doctor":
        report = doctor(args.db)
        if args.write_report:
            report["report_path"] = str(write_setup_report(args.db, report))
        dump(report)
        return 0 if report["core_ready"] else 1
    elif args.cmd == "setup":
        result = setup_household(
            args.db,
            owner_name=args.owner_name,
            timezone=args.timezone,
            quiet_start=args.quiet_start,
            quiet_end=args.quiet_end,
            proactive_enabled=args.enable_weekly,
            resource_context_mode=args.resource_context_mode,
        )
        result["report_path"] = str(write_setup_report(args.db))
        dump(result)
    elif args.cmd == "parent" and args.parent_cmd == "add":
        dump(add_parent(args.db, args.name))
    elif args.cmd == "parent" and args.parent_cmd == "list":
        dump(onboarding_status(args.db)["parents"])
    elif args.cmd == "slack" and args.slack_cmd == "pair-code":
        state = onboarding_status(args.db)
        parent_id = args.parent or next(
            (item["id"] for item in state["parents"] if item["role"] == "owner" and item["status"] == "active"),
            None,
        )
        if not parent_id:
            raise ValueError("no active owner; run curiosity setup first")
        dump(create_pairing_code(args.db, parent_id, ttl_minutes=args.ttl))
    elif args.cmd == "slack" and args.slack_cmd == "bindings":
        dump(list_bindings(args.db))
    elif args.cmd == "slack" and args.slack_cmd == "revoke":
        dump(revoke_binding(args.db, args.binding))
    elif args.cmd == "slack" and args.slack_cmd == "status":
        dump({"onboarding": onboarding_status(args.db), "doctor": doctor(args.db)})
    elif args.cmd == "slack" and args.slack_cmd == "run":
        from .transports.slack import run_slack_connector

        run_slack_connector(args.db, args.output_dir)
    elif args.cmd == "inbox" and args.inbox_cmd == "list":
        dump(list_inbox(args.db, status=args.status))
    elif args.cmd == "inbox" and args.inbox_cmd == "assign":
        dump(CuriosityService(args.db, _default_output()).assign_inbox(args.id, args.child))
    elif args.cmd == "inbox" and args.inbox_cmd == "dismiss":
        dump(CuriosityService(args.db, _default_output()).dismiss_inbox(args.id))
    elif args.cmd == "child" and args.child_cmd == "add":
        init_db(args.db)
        add_child(args.db, args.id, args.name, args.birth_year, args.grade)
        AutonomousDirector(args.db).ensure_weekly_schedule(args.id)
        dump({"status": "ok", "child": args.id})
    elif args.cmd == "ask":
        metadata: dict[str, Any] = {"topics": args.topic}
        if args.include_private_excerpts:
            metadata["private_resource_mode"] = "selected_excerpts"
        payload: dict[str, Any] = {
            "type": "child_question",
            "child_id": args.child,
            "text": args.question,
            "source": args.source,
            "metadata": metadata,
        }
        if args.event_id:
            payload["id"] = args.event_id
        dump(CuriosityHarness(args.db).dispatch(Event.model_validate(payload)).model_dump(mode="json"))
    elif args.cmd == "capture":
        init_db(args.db)
        dump(capture_question(args.db, args.child, args.question, args.topic, args.source))
    elif args.cmd == "observe":
        init_db(args.db)
        dump(
            {"observation_id": add_observation(args.db, args.child, args.kind, args.text, args.source, args.confidence)}
        )
    elif args.cmd == "interest":
        init_db(args.db)
        dump({"node_id": upsert_node(args.db, args.child, "interest", args.topic, args.confidence)})
    elif args.cmd == "school":
        init_db(args.db)
        add_school_signal(args.db, args.child, args.category, args.value, args.source_ref, args.expires_at)
        dump({"status": "ok"})
    elif args.cmd == "context":
        init_db(args.db)
        dump(child_context(args.db, args.child))
    elif args.cmd == "event":
        init_db(args.db)
        payload = {
            "type": args.type,
            "child_id": args.child,
            "text": args.text,
            "source": args.source,
            "metadata": json.loads(args.metadata_json),
        }
        if args.event_id:
            payload["id"] = args.event_id
        dump(CuriosityHarness(args.db).dispatch(Event.model_validate(payload)).model_dump(mode="json"))
    elif args.cmd == "resource" and args.resource_cmd == "index":
        catalogs = [Path(args.catalog)] if args.catalog else discover_private_catalogs(args.repo)
        if len(catalogs) != 1:
            raise ValueError(f"expected one private catalog or --catalog; found {len(catalogs)}")
        dump(index_collection(args.db, catalogs[0], repository_root=args.repo).__dict__)
    elif args.cmd == "resource" and args.resource_cmd == "status":
        init_db(args.db)
        dump(resource_inventory(args.db))
    elif args.cmd == "resource" and args.resource_cmd == "mode":
        dump(set_household_resource_context_mode(args.db, args.mode))
    elif args.cmd == "resource" and args.resource_cmd == "search":
        init_db(args.db)
        dump(search_resources(args.db, args.query, limit=args.limit, include_excerpts=args.include_private_excerpts))
    elif args.cmd == "artifact" and args.artifact_cmd == "render":
        spec = load_spec(args.spec)
        output = (
            render_pdf(spec, args.out)
            if Path(args.out).suffix.casefold() == ".pdf"
            else render_html(spec, args.out, strict=True)
        )
        dump({"status": "ok", "path": str(output), "trust": trust_summary(spec)})
    elif args.cmd == "artifact" and args.artifact_cmd == "validate":
        spec = load_spec(args.spec)
        errors = validate_artifact_spec(spec)
        if args.rendered:
            errors.extend(validate_rendered_file(args.rendered))
        dump({"status": "pass" if not errors else "fail", "errors": errors, "trust": trust_summary(spec)})
        return 1 if errors else 0
    elif args.cmd == "artifact" and args.artifact_cmd == "create":
        dump(CuriosityService(args.db, args.output_dir).create_artifact(args.child, load_spec(args.spec)))
    elif args.cmd == "artifact" and args.artifact_cmd == "approve":
        dump(approve_artifact(args.db, args.id, note=args.note))
    elif args.cmd == "artifact" and args.artifact_cmd == "print":
        dump(print_artifact(args.db, args.id, args.approval, printer=args.printer, send=args.send))
    elif args.cmd == "action" and args.action_cmd == "list":
        dump(list_actions(args.db, status=args.status))
    elif args.cmd == "action" and args.action_cmd == "execute":
        dump(CuriosityService(args.db, args.output_dir).execute_action(args.id))
    elif args.cmd == "feedback":
        feedback = FeedbackInput(
            child_id=args.child,
            outcome=args.outcome,
            note=args.note,
            experience_id=args.experience,
            artifact_id=args.artifact,
        )
        dump({"feedback_id": record_feedback(args.db, feedback)})
    elif args.cmd == "reflect":
        dump(AutonomousDirector(args.db).reflect_for_child(args.child))
    elif args.cmd == "worker":
        harness = CuriosityHarness(args.db)
        scheduled = AutonomousDirector(args.db).run_due()
        results = []
        while True:
            result = harness.process_next()
            if result is None:
                break
            results.append(result.model_dump(mode="json"))
            if not args.drain:
                break
        dump({"scheduled_reflections": scheduled, "processed": len(results), "results": results})
    elif args.cmd == "serve":
        import uvicorn

        from .web import create_app

        uvicorn.run(create_app(args.db, args.output_dir), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
