from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .actions import list_actions
from .artifact_validation import validate_artifact_spec, validate_rendered_file
from .artifacts import load_spec, render_html, render_pdf
from .backups import backup_status, create_snapshot, default_backup_root, restore_snapshot, verify_snapshot
from .brain_config import (
    brain_status,
    configure_api_brain,
    ensure_model_env_template,
    write_brain_config,
)
from .config import private_root, repository_root
from .contracts import Event, FeedbackInput
from .db import init_db
from .director import AutonomousDirector
from .episodes import apply_episode_correction
from .feedback import record_feedback
from .graph import add_child, add_observation, add_school_signal, capture_question, child_context, upsert_node
from .host import host_status, install_user_services
from .interaction import (
    add_parent,
    configure_family_lens,
    create_pairing_code,
    list_bindings,
    list_inbox,
    onboarding_status,
    record_onboarding_review,
    reviewable_slack_events,
    revoke_binding,
    set_household_resource_context_mode,
    set_household_visual_mode,
    setup_household,
)
from .onboarding import doctor, run_brain_probe, run_image_generation_probe, write_setup_report
from .printer import approve_artifact, print_artifact
from .public_projects import audit_public_projects, public_project, public_project_catalog, registry_status
from .resources import discover_private_catalogs, index_collection, resource_inventory, search_resources
from .runtime import CuriosityHarness
from .scheduler import SchedulerService
from .service import CuriosityService
from .setup_agent import launch_setup_agent, prepare_agent_setup
from .trust import trust_summary

TERMINAL_SECRET_KEY_SUFFIXES = ("_api_key", "_token", "_password", "_secret")
TERMINAL_SECRET_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:sk-|xapp-|xoxb-)[A-Za-z0-9._-]{12,}")


def _redact_terminal_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: (
                "[redacted]"
                if str(key).casefold().endswith(TERMINAL_SECRET_KEY_SUFFIXES)
                else _redact_terminal_secrets(value)
            )
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [_redact_terminal_secrets(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_redact_terminal_secrets(value) for value in obj)
    if isinstance(obj, str):
        return TERMINAL_SECRET_PATTERN.sub("[redacted]", obj)
    return obj


def dump(obj: Any) -> None:
    print(json.dumps(_redact_terminal_secrets(obj), indent=2, ensure_ascii=False, default=str))


def _default_db() -> str:
    return os.environ.get("CURIOSITY_DB") or str(private_root() / "data" / "curiosity.db")


def _default_output() -> str:
    return os.environ.get("CURIOSITY_OUTPUT") or str(private_root() / "output")


def _default_private() -> str:
    return str(private_root())


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

    backup = sub.add_parser("backup", help="Create and check owner-only family snapshots outside the repository")
    backup_sub = backup.add_subparsers(dest="backup_cmd", required=True)
    command = backup_sub.add_parser("create", help="Snapshot family data without saving Slack or model credentials")
    add_db(command)
    command.add_argument("--private-dir", default=_default_private())
    command.add_argument("--output-dir", default=_default_output())
    command.add_argument("--destination", default=str(default_backup_root()))
    command = backup_sub.add_parser("status", help="Show backup count and the latest snapshot summary")
    command.add_argument("--destination", default=str(default_backup_root()))
    command = backup_sub.add_parser("verify", help="Verify checksums, permissions, and database integrity")
    command.add_argument("snapshot", nargs="?", help="Snapshot ID; defaults to the latest")
    command.add_argument("--destination", default=str(default_backup_root()))
    command = backup_sub.add_parser("restore", help="Restore into a new path without overwriting current family data")
    command.add_argument("snapshot", nargs="?", help="Snapshot ID; defaults to the latest")
    command.add_argument("--destination", default=str(default_backup_root()))
    command.add_argument("--target-private", help="New restore path; must not already exist")

    onboard = sub.add_parser("onboard", help="Show or record end-to-end onboarding checkpoints")
    onboard_sub = onboard.add_subparsers(dest="onboard_cmd", required=True)
    command = onboard_sub.add_parser("status")
    add_db(command)
    command = onboard_sub.add_parser("pending", help="List recent delivered Slack answers by private event ID")
    add_db(command)
    command.add_argument("--limit", type=int, default=5)
    command = onboard_sub.add_parser("review", help="Record the parent's review of one real Slack answer")
    add_db(command)
    review_target = command.add_mutually_exclusive_group(required=True)
    review_target.add_argument("--event")
    review_target.add_argument("--latest", action="store_true")
    for rating in ("factuality", "grade-fit", "curiosity-value", "parent-effort"):
        command.add_argument(f"--{rating}", required=True, choices=["pass", "retry"])
    command.add_argument("--note")

    brain = sub.add_parser("brain", help="Configure and verify the bring-your-own model stack")
    brain_sub = brain.add_subparsers(dest="brain_cmd", required=True)
    command = brain_sub.add_parser("configure")
    command.add_argument("--provider", required=True, choices=["openai", "anthropic", "openrouter"])
    command.add_argument("--model", required=True, help="Structured reasoning model ID")
    command.add_argument("--vision-model", help="Vision/OCR model ID; defaults to --model")
    command.add_argument("--image-provider", choices=["openai", "openrouter"])
    command.add_argument("--image-model", help="Image generation model ID; required for the full visual stack")
    command.add_argument("--web-search", action="store_true")
    command.add_argument("--reasoning-effort")
    command.add_argument(
        "--recommendation-status",
        choices=["family_evaluating", "family_recommended", "custom"],
        default="custom",
    )
    command = brain_sub.add_parser("status")
    command = brain_sub.add_parser("doctor")
    command = brain_sub.add_parser("test", help="Run a family-data-free structured reasoning probe")
    add_db(command)
    command.add_argument("--live", action="store_true")

    visual = sub.add_parser("visual", help="Configure and verify visual Slack responses")
    visual_sub = visual.add_subparsers(dest="visual_cmd", required=True)
    command = visual_sub.add_parser("status")
    add_db(command)
    command = visual_sub.add_parser("mode", help="Choose off, deterministic cards, or opt-in decorative generation")
    add_db(command)
    command.add_argument("--mode", required=True, choices=["off", "deterministic", "decorative"])
    command = visual_sub.add_parser("test", help="Run one billable, family-data-free decorative image probe")
    add_db(command)
    command.add_argument("--output-dir", default=_default_output())
    command.add_argument("--live", action="store_true")

    family_lens = sub.add_parser("family-lens", help="Configure private pedagogy and practical constraints")
    family_lens_sub = family_lens.add_subparsers(dest="family_lens_cmd", required=True)
    command = family_lens_sub.add_parser("configure")
    add_db(command)
    command.add_argument("--pedagogy", action="append", default=[])
    command.add_argument("--theme", action="append", default=[])
    command.add_argument("--activity-minutes", type=int, default=15)
    command.add_argument("--parent-effort", choices=["very_low", "low", "moderate"], default="low")
    command.add_argument("--reading-load", choices=["emerging", "early_elementary", "independent"], default="early_elementary")
    command.add_argument("--material", action="append", default=[])
    command.add_argument("--content-boundary", action="append", default=[])
    command = family_lens_sub.add_parser("status")
    add_db(command)

    command = sub.add_parser("setup", help="Open guided setup, or configure household defaults non-interactively")
    add_db(command)
    command.add_argument("--owner-name")
    command.add_argument("--timezone", help="IANA timezone, such as America/New_York")
    command.add_argument("--quiet-start", default="20:00")
    command.add_argument("--quiet-end", default="07:00")
    command.add_argument(
        "--enable-weekly",
        action="store_true",
        help="Record parent opt-in for a future release; v0.1 public policy keeps suggestions disabled",
    )
    command.add_argument(
        "--resource-context-mode",
        choices=["metadata_only", "selected_excerpts"],
        default="metadata_only",
        help="Whether bounded licensed excerpts may enter hosted-model requests",
    )
    command.add_argument("--agent", choices=["auto", "codex", "claude"], default="auto")
    command.add_argument("--workspace", help="Private setup workspace; defaults under CURIOSITY_HOME")
    command.add_argument("--no-launch", action="store_true", help="Prepare the coding-agent handoff without opening it")

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

    command = sub.add_parser("context", help="Inspect a child's episode-based context or append a correction")
    add_db(command)
    command.add_argument("--child", required=True)
    command.add_argument("--correct-event")
    command.add_argument("--classification", choices=["retry", "deepening", "new_episode", "exclude"])
    command.add_argument("--related-event")
    command.add_argument("--note")

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

    ecosystem = sub.add_parser(
        "ecosystem", help="Inspect vetted public projects without cloning or executing them"
    )
    ecosystem_sub = ecosystem.add_subparsers(dest="ecosystem_cmd", required=True)
    command = ecosystem_sub.add_parser("list")
    command.add_argument("--category")
    command.add_argument(
        "--status",
        choices=["integrated", "approved_reference", "evaluation_candidate", "watch_only"],
    )
    command = ecosystem_sub.add_parser("show")
    command.add_argument("--id", required=True)
    ecosystem_sub.add_parser("status")
    command = ecosystem_sub.add_parser(
        "check", help="Refresh public metadata only; never clone, install, import, or execute upstream code"
    )
    command.add_argument("--live", action="store_true")

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

    command = sub.add_parser("reflect", help="Inspect the disabled reflection path; v0.1 returns do_nothing")
    add_db(command)
    command.add_argument("--child", required=True)

    command = sub.add_parser("worker")
    add_db(command)
    command.add_argument("--drain", action="store_true", help="Process every currently ready job")
    command.add_argument("--forever", action="store_true", help="Continuously materialize schedules and drain jobs")
    command.add_argument("--interval", type=float, default=2.0, help="Worker polling interval in seconds")

    host = sub.add_parser("host", help="Install or inspect always-on local Linux services")
    host_sub = host.add_subparsers(dest="host_cmd", required=True)
    command = host_sub.add_parser("install", help="Install owner-level Slack and scheduler services")
    command.add_argument("--no-start", action="store_true")
    host_sub.add_parser("status")

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
    elif args.cmd == "backup" and args.backup_cmd == "create":
        dump(
            create_snapshot(
                db_path=args.db,
                private_dir=args.private_dir,
                output_dir=args.output_dir,
                backup_root=args.destination,
            )
        )
    elif args.cmd == "backup" and args.backup_cmd == "status":
        dump(backup_status(backup_root=args.destination))
    elif args.cmd == "backup" and args.backup_cmd == "verify":
        result = verify_snapshot(args.snapshot, backup_root=args.destination)
        dump(result)
        return 0 if result["status"] == "pass" else 1
    elif args.cmd == "backup" and args.backup_cmd == "restore":
        dump(
            restore_snapshot(
                args.snapshot,
                target_private=args.target_private,
                backup_root=args.destination,
            )
        )
    elif args.cmd == "onboard" and args.onboard_cmd == "status":
        dump(doctor(args.db))
    elif args.cmd == "onboard" and args.onboard_cmd == "pending":
        dump(reviewable_slack_events(args.db, limit=args.limit))
    elif args.cmd == "onboard" and args.onboard_cmd == "review":
        event_id = args.event
        if args.latest:
            reviewable = [
                item
                for item in reviewable_slack_events(args.db, limit=20)
                if item["current_answer_stack"]
            ]
            if not reviewable:
                raise ValueError("no delivered Slack answer from the current answer stack is available for review")
            event_id = reviewable[0]["event_id"]
        dump(
            record_onboarding_review(
                args.db,
                event_id=event_id,
                factuality=args.factuality,
                grade_fit=args.grade_fit,
                curiosity_value=args.curiosity_value,
                parent_effort=args.parent_effort,
                note=args.note,
            )
        )
    elif args.cmd == "brain" and args.brain_cmd == "configure":
        config = configure_api_brain(
            provider=args.provider,
            model=args.model,
            vision_model=args.vision_model,
            image_provider=args.image_provider,
            image_model=args.image_model,
            web_search=args.web_search,
            reasoning_effort=args.reasoning_effort,
            recommendation_status=args.recommendation_status,
        )
        config_path = write_brain_config(config)
        providers = {route.provider for route in config.routes.values()}
        credential_path = ensure_model_env_template(providers)
        dump(
            {
                "status": "configured",
                "brain_config": str(config_path),
                "credential_file": str(credential_path),
                "credentials_present": False,
                "next_action": "paste provider keys directly into the credential file, chmod 600 it, then run curiosity brain doctor",
            }
        )
    elif args.cmd == "brain" and args.brain_cmd in {"status", "doctor"}:
        dump(brain_status())
    elif args.cmd == "brain" and args.brain_cmd == "test":
        dump(run_brain_probe(args.db, live=args.live))
    elif args.cmd == "visual" and args.visual_cmd == "status":
        report = doctor(args.db)
        dump(
            {
                key: report[key]
                for key in (
                    "visual_mode",
                    "deterministic_visual_ready",
                    "visual_delivery_verified",
                    "image_generation_configured",
                    "image_generation_verified",
                    "visual_ready",
                    "next_action",
                )
            }
        )
    elif args.cmd == "visual" and args.visual_cmd == "mode":
        dump(set_household_visual_mode(args.db, args.mode))
    elif args.cmd == "visual" and args.visual_cmd == "test":
        dump(run_image_generation_probe(args.db, args.output_dir, live=args.live))
    elif args.cmd == "family-lens" and args.family_lens_cmd == "configure":
        pedagogy = args.pedagogy or [
            "follow the child's question",
            "show before explaining",
            "one conceptual rung above",
            "productive struggle without spoon-feeding",
        ]
        materials = args.material or ["paper", "writing utensils", "common household materials"]
        dump(
            configure_family_lens(
                args.db,
                {
                    "pedagogy": pedagogy,
                    "themes": args.theme,
                    "activity_minutes": args.activity_minutes,
                    "parent_effort": args.parent_effort,
                    "reading_load": args.reading_load,
                    "materials": materials,
                    "content_boundaries": args.content_boundary,
                },
            )
        )
    elif args.cmd == "family-lens" and args.family_lens_cmd == "status":
        state = onboarding_status(args.db)
        dump({"configured": state["family_lens_configured"], "private": True})
    elif args.cmd == "setup":
        if bool(args.owner_name) != bool(args.timezone):
            raise ValueError("--owner-name and --timezone must be provided together")
        if args.owner_name and args.timezone:
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
        else:
            result = prepare_agent_setup(preference=args.agent, workspace=args.workspace)
            dump(result)
            if result["agent"] and not args.no_launch:
                launch_setup_agent(str(result["agent"]), str(result["workspace"]))
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
        if args.correct_event:
            if not args.classification:
                raise ValueError("--classification is required with --correct-event")
            dump(
                apply_episode_correction(
                    args.db,
                    child_id=args.child,
                    event_id=args.correct_event,
                    action=args.classification,
                    related_event_id=args.related_event,
                    note=args.note,
                )
            )
        else:
            if args.classification or args.related_event or args.note:
                raise ValueError("--correct-event is required for a context correction")
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
    elif args.cmd == "ecosystem" and args.ecosystem_cmd == "list":
        dump(public_project_catalog(category=args.category, status=args.status))
    elif args.cmd == "ecosystem" and args.ecosystem_cmd == "show":
        dump(public_project(args.id))
    elif args.cmd == "ecosystem" and args.ecosystem_cmd == "status":
        dump(registry_status())
    elif args.cmd == "ecosystem" and args.ecosystem_cmd == "check":
        dump(audit_public_projects(live=args.live))
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
        if args.interval < 0.25 or args.interval > 300:
            raise ValueError("worker interval must be between 0.25 and 300 seconds")
        harness = CuriosityHarness(args.db)
        if args.forever:
            while True:
                SchedulerService(args.db).run_due()
                AutonomousDirector(args.db).run_due()
                while harness.process_next() is not None:
                    pass
                time.sleep(args.interval)
        else:
            checkins = SchedulerService(args.db).run_due()
            scheduled = AutonomousDirector(args.db).run_due()
            results = []
            while True:
                result = harness.process_next()
                if result is None:
                    break
                results.append(result.model_dump(mode="json"))
                if not args.drain:
                    break
            dump(
                {
                    "scheduled_checkins": checkins,
                    "scheduled_reflections": scheduled,
                    "processed": len(results),
                    "results": results,
                }
            )
    elif args.cmd == "host" and args.host_cmd == "install":
        dump(install_user_services(start=not args.no_start))
    elif args.cmd == "host" and args.host_cmd == "status":
        dump(host_status())
    elif args.cmd == "serve":
        import uvicorn

        from .web import create_app

        uvicorn.run(create_app(args.db, args.output_dir), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
