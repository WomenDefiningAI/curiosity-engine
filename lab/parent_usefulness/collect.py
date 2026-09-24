#!/usr/bin/env python3
"""Capture real CuriosityService conversations in disposable synthetic households.

Run with the selected release's Python. Photo/button cases use the Slack transport
with a local network substitute; no messages are sent to a real workspace.
The input file maps opaque case IDs to inputs; it contains no reviewer guidance.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextvars
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def collect(args):
    case_dir = args.run_dir / args.case
    case_dir.mkdir(parents=True, exist_ok=False)
    private = case_dir / "private"
    private.mkdir(mode=0o700)
    os.environ.update(
        CURIOSITY_REPO_ROOT=str(args.source),
        CURIOSITY_HOME=str(case_dir),
        CURIOSITY_PRIVATE_ROOT=str(private),
        CURIOSITY_MODEL_ENV=str(args.auth_root / "model.env"),
        CURIOSITY_USAGE_CASE=str(args.run_dir / args.case),
        CURIOSITY_USAGE_PHASE="generation",
        CURIOSITY_BRAIN_CONFIG=str(args.auth_root / "brain.json"),
    )
    if os.environ.get("CURIOSITY_TEXT_REVIEW") == "jev":
        os.environ.setdefault("CURIOSITY_TYPESAFE_KEY_FILE", str(args.auth_root / "typesafe.env"))
    sys.path.insert(0, str(args.source / "src"))
    from curiosity_engine.brain_config import load_brain_config
    from curiosity_engine.config import AppConfig
    from curiosity_engine.db import connect
    from curiosity_engine.interaction import (
        consume_pairing_code,
        create_pairing_code,
        set_household_visual_mode,
        setup_household,
    )
    from curiosity_engine.runtime import configured_backend
    from curiosity_engine.service import CuriosityService
    from curiosity_engine.transports.contracts import InboundMessage
    from curiosity_engine.visuals import process_visual_jobs

    backend = configured_backend(AppConfig.load())
    if backend is None or backend.name != "openai":
        raise RuntimeError("This collector's telemetry requires the live OpenAI backend")
    calls = []
    designs = []
    role_context = contextvars.ContextVar("eval_role", default=None)
    from curiosity_engine.openai_backend import OpenAIBackend

    original_complete = OpenAIBackend.complete

    def complete(self, **kwargs):
        token = role_context.set(kwargs["role"])
        try:
            result = original_complete(self, **kwargs)
            payload = kwargs.get("payload") or {}
            if ("artifact_brief" in payload and "attempt" in payload) or kwargs["response_model"].__name__ in {"PictureDraft", "PictureReview"}:
                designs.append({"attempt": payload.get("attempt"), "contract": kwargs["response_model"].__name__,
                                "repair_instruction": payload.get("feedback", payload.get("repair_instruction")), "draft": result})
                # Synthetic diagnostic drafts are never passed to the utility judge.
                write(case_dir / "design-diagnostics.json", designs)
            return result
        finally:
            role_context.reset(token)

    OpenAIBackend.complete = complete

    # Observe SDK invocations and returned usage without changing requests/responses.
    from openai.resources.images import Images
    from openai.resources.responses import Responses

    def instrument(cls, name):
        original = getattr(cls, name)

        def observed(self, *pos, **kw):
            record = {
                "role": role_context.get(),
                "method": f"{cls.__name__}.{name}",
                "model": kw.get("model"),
                "started_monotonic": time.monotonic(),
                "usage": None,
                "reported_cost": None,
            }
            calls.append(record)
            try:
                result = original(self, *pos, **kw)
                usage = getattr(result, "usage", None)
                record.update(status="completed", usage=usage.model_dump() if usage else None)
                return result
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
                raise
            finally:
                record["duration_seconds"] = round(time.monotonic() - record.pop("started_monotonic"), 3)
                write(case_dir / "model-calls.json", calls)

        setattr(cls, name, observed)

    instrument(Responses, "create")
    instrument(Images, "generate")
    instrument(Images, "edit")

    input_data = json.loads(args.inputs.read_text())[args.case]
    brain = load_brain_config()
    source_hash = hashlib.sha256()
    for directory in ("src", "configs", "prompts", "capabilities"):
        for path in sorted((args.source / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                source_hash.update(str(path.relative_to(args.source)).encode())
                source_hash.update(path.read_bytes())
    bundle = {
        "id": args.case,
        "input": input_data,
        "status": "collecting",
        "release": args.source.name,
        "source_sha256": source_hash.hexdigest(),
        "routes": {k: v.model_dump() for k, v in brain.routes.items()},
        "visual_mode": args.visual_mode,
        "entrypoint": "CuriosityService.chat",
        "delivery": "local capture; Slack delivery and network latency not exercised",
        "child_attribution": "first listed child; all listed children registered normally",
        "turns": [],
        "reported_cost": None,
    }
    write(case_dir / "bundle.json", bundle)
    db = private / "data" / "curiosity.db"
    db.parent.mkdir(mode=0o700)
    setup = setup_household(db, owner_name="Synthetic Parent", timezone="America/Los_Angeles")
    set_household_visual_mode(db, args.visual_mode)
    service = CuriosityService(db, private / "output")
    for child in input_data["children"]:
        service.add_child(child["label"], child["label"], grade=child["grade"])
    pairing = create_pairing_code(db, setup["owner_id"])
    binding = consume_pairing_code(
        db,
        pairing["pairing_code"],
        InboundMessage(
            external_event_id="eval-pair",
            team_id="EVAL_TEAM",
            user_id="EVAL_PARENT",
            channel_id="EVAL_CHANNEL",
            text="synthetic pairing",
        ),
    )
    slack = None
    if input_data.get("transport_steps"):
        from slack_capture import SlackCapture
        slack = SlackCapture(db, private / "output", service)
        bundle["entrypoint"] = "Slack event receiver and rendered button receivers"
        bundle["delivery"] = "Local Slack network substitute; real image bytes, handlers, receipts, outboxes and button replay"

    def copy_file(path, dest):
        original = Path(path).resolve()
        if not original.is_relative_to(private):
            raise ValueError("Delivery path escaped the synthetic household")
        shutil.copy2(original, dest)
        return {"path": str(dest.relative_to(case_dir)), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}

    steps = input_data.get("transport_steps") or [{"text": text} for text in input_data["parent_turns"]]
    for number, step in enumerate(steps, 1):
        prompt = step.get("text") or f"Click: {step['button']}"
        turn_dir = case_dir / f"turn-{number}"
        turn_dir.mkdir()
        started = time.monotonic()
        turn = {
            "number": number,
            "parent": prompt,
            "files": [],
            "timing": {
                "acknowledgment_seconds": None,
                "first_useful_seconds": None,
                "service_response_seconds": None,
                "artifact_ready_seconds": None,
            },
        }
        bundle["turns"].append(turn)
        stage = "service"
        try:
            if slack:
                photos = []
                turn["input_files"] = []
                for index, value in enumerate(step.get("photos") or []):
                    source_photo = (args.source / value).resolve()
                    path = turn_dir / f"input-photo-{index + 1}{source_photo.suffix}"
                    shutil.copy2(source_photo, path)
                    photos.append(path)
                    turn["input_files"].append({"path": str(path.relative_to(case_dir)),
                                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                result = slack.step(step, number=number, photos=photos)
            else:
                result = service.chat(
                    binding_id=binding["binding_id"], team_id="EVAL_TEAM", channel_id="EVAL_CHANNEL",
                    thread_id="eval-thread", text=prompt, child_id=input_data["children"][0]["label"],
                )
            stage = "capture"
            turn["timing"]["service_response_seconds"] = round(time.monotonic() - started, 3)
            write(turn_dir / "service-result.json", result)
            turn.update(
                status=result.get("status"), message=result.get("message"), interaction=result.get("interaction")
            )
            artifact = result.get("artifact")
            if artifact:
                turn["artifact"] = {k: artifact.get(k) for k in ("artifact_type", "title", "artifact_id")}
                for key, name in (("pdf_path", "artifact.pdf"), ("preview_path", "preview.png")):
                    if artifact.get(key):
                        turn["files"].append(copy_file(artifact[key], turn_dir / name))
                turn["timing"]["artifact_ready_seconds"] = round(time.monotonic() - started, 3)
            if result.get("status") == "completed" and (result.get("message") or artifact):
                turn["timing"]["first_useful_seconds"] = turn["timing"]["service_response_seconds"]
            if result.get("visual_job_id"):
                job_id = result["visual_job_id"]
                visual_started = time.monotonic()
                # Honor the worker's normal retry/backoff; never change job state.
                while True:
                    process_visual_jobs(db, private / "output", job_id=job_id)
                    with connect(db) as conn:
                        job = dict(
                            conn.execute("SELECT status,last_error FROM visual_jobs WHERE id=?", (job_id,)).fetchone()
                        )
                        assets = [
                            dict(r)
                            for r in conn.execute(
                                "SELECT path,title,caption,alt_text FROM visual_assets WHERE job_id=?", (job_id,)
                            )
                        ]
                    if job["status"] in {"completed", "failed"} or time.monotonic() - visual_started > 240:
                        break
                    time.sleep(5)
                turn["visual_job"] = job
                for i, asset in enumerate(assets):
                    turn["files"].append({**copy_file(asset.pop("path"), turn_dir / f"visual-{i + 1}.png"), **asset})
                if assets:
                    turn["timing"]["artifact_ready_seconds"] = round(time.monotonic() - started, 3)
            turn["capture_status"] = "complete"
        except Exception as exc:
            turn.update(
                capture_status="complete" if stage == "service" else "error",
                status="service_error" if stage == "service" else "collection_error",
                error_type=type(exc).__name__,
                error=str(exc),
                error_stage=stage,
            )
            (turn_dir / "error.txt").write_text(traceback.format_exc())
        turn["timing"]["total_seconds"] = round(time.monotonic() - started, 3)
        from curiosity_engine.model_usage import usage_summary

        snapshot = usage_summary(os.environ["CURIOSITY_USAGE_DB"])
        if any(r["case_id"] == os.environ["CURIOSITY_USAGE_CASE"] and r["status"] == "budget_blocked" for r in snapshot["calls"]):
            turn.update(status="budget_stopped", capture_status="budget_stopped")
        write(case_dir / "bundle.json", bundle)
        print(
            json.dumps(
                {
                    "id": args.case,
                    "turn": number,
                    "status": turn.get("status"),
                    "capture": turn["capture_status"],
                    "files": len(turn["files"]),
                    "seconds": turn["timing"]["total_seconds"],
                }
            ),
            flush=True,
        )
        if turn["capture_status"] != "complete" or turn.get("status") == "service_error":
            break
    # A service failure blocks later dependent turns; that is a product failure,
    # not missing measurement. Keep every intended turn visible to the reviewer.
    if bundle["turns"][-1].get("status") == "service_error":
        for number, step in enumerate(steps[len(bundle["turns"]) :], len(bundle["turns"]) + 1):
            bundle["turns"].append(
                {
                    "number": number,
                    "parent": step.get("text") or f"Click: {step['button']}",
                    "files": [],
                    "status": "blocked_by_prior_service_error",
                    "capture_status": "complete",
                    "timing": {},
                }
            )
    bundle["status"] = (
        "captured"
        if all(t["capture_status"] == "complete" for t in bundle["turns"])
        and len(bundle["turns"]) == len(steps)
        else "collection_error"
    )
    ledger = usage_summary(os.environ["CURIOSITY_USAGE_DB"])
    measured = [r for r in ledger["calls"] if r["case_id"] == os.environ["CURIOSITY_USAGE_CASE"] and r["phase"] == "generation"]
    bundle["estimated_cost_usd"] = sum(r["estimated_usd"] or 0 for r in measured)
    bundle["unknown_cost_calls"] = sum(r["estimated_usd"] is None for r in measured)
    bundle["model_calls"] = len(calls)
    if any(r["status"] == "budget_blocked" for r in measured):
        bundle["status"] = "budget_stopped"
    write(case_dir / "bundle.json", bundle)
    write(case_dir / "model-calls.json", calls)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "auth-root", "run-dir", "inputs"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--visual-mode", default="decorative")
    p.add_argument("--case")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--budget-usd", type=float, required=True)
    p.add_argument("--usage-db", type=Path)
    args = p.parse_args()
    for name in ("source", "auth_root", "run_dir", "inputs"):
        setattr(args, name, getattr(args, name).resolve())
    args.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not 0 < args.budget_usd <= 1000:
        p.error("--budget-usd must be positive and at most 1000")
    os.environ["CURIOSITY_USAGE_DB"] = str((args.usage_db or args.run_dir / "usage.sqlite").resolve())
    os.environ["CURIOSITY_BUDGET_USD"] = str(args.budget_usd)
    logging.basicConfig(level=logging.WARNING)
    if args.case:
        collect(args)
        bundle = json.loads((args.run_dir / args.case / "bundle.json").read_text())
        if bundle["status"] != "captured":
            raise SystemExit(1)
        return

    def one(case_id):
        if (args.run_dir / case_id).exists():
            return {"id": case_id, "status": "already_exists_not_overwritten"}
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source",
            str(args.source),
            "--auth-root",
            str(args.auth_root),
            "--run-dir",
            str(args.run_dir),
            "--inputs",
            str(args.inputs),
            "--visual-mode",
            args.visual_mode,
            "--case",
            case_id,
            "--budget-usd", str(args.budget_usd),
            "--usage-db", os.environ["CURIOSITY_USAGE_DB"],
        ]
        with (args.run_dir / f"{case_id}.log").open("w") as log:
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                return {"id": case_id, "exit_code": result.returncode}
            except subprocess.TimeoutExpired:
                return {"id": case_id, "status": "collector_timeout_900_seconds"}

    failed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(one, args.ids or json.loads(args.inputs.read_text()).keys()):
            print(json.dumps(result), flush=True)
            failed |= result.get("exit_code", 0) != 0 or result.get("status") == "collector_timeout_900_seconds"
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
