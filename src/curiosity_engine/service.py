from __future__ import annotations

from pathlib import Path
from typing import Any

from .actions import execute_action, list_actions
from .artifacts import ArtifactService
from .config import AppConfig, repository_root
from .contracts import Event, FeedbackInput
from .db import connect, init_db, jload
from .director import AutonomousDirector, list_opportunities
from .feedback import record_feedback
from .graph import add_child, child_context
from .interaction import list_inbox, onboarding_status, resolve_inbox
from .printer import approve_artifact, print_artifact
from .resources import index_collection, resource_inventory, search_resources
from .runtime import CuriosityHarness, configured_backend


class CuriosityService:
    def __init__(self, db_path: str | Path, output_dir: str | Path):
        self.db_path = Path(db_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.config = AppConfig.load()
        self.visual_backend = configured_backend(self.config, role="visual_qa")
        db_parent_existed = self.db_path.parent.exists()
        output_existed = self.output_dir.exists()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        private_root = repository_root() / "private"
        db_is_private = self.db_path.is_relative_to(private_root)
        output_is_private = self.output_dir.is_relative_to(private_root)
        if not db_parent_existed or db_is_private:
            self.db_path.parent.chmod(0o700)
        if not output_existed or output_is_private:
            self.output_dir.chmod(0o700)
        init_db(self.db_path)

    def add_child(self, child_id: str, name: str, birth_year: int | None = None, grade: str | None = None) -> None:
        add_child(self.db_path, child_id, name, birth_year, grade)
        AutonomousDirector(self.db_path).ensure_weekly_schedule(child_id)

    def children(self) -> list[dict[str, Any]]:
        with connect(self.db_path) as conn:
            return [dict(row) for row in conn.execute("SELECT id,name,birth_year,grade FROM children ORDER BY name")]

    def ask(
        self,
        *,
        child_id: str,
        text: str,
        source: str = "parent",
        topics: list[str] | None = None,
        include_private_excerpts: bool = False,
        event_id: str | None = None,
        context_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {"topics": topics or [], **(context_metadata or {})}
        if include_private_excerpts:
            metadata["private_resource_mode"] = "selected_excerpts"
        payload: dict[str, Any] = {
            "type": "child_question",
            "child_id": child_id,
            "text": text,
            "source": source,
            "metadata": metadata,
        }
        if event_id:
            payload["id"] = event_id
        result = CuriosityHarness(self.db_path).dispatch(Event.model_validate(payload))
        return result.model_dump(mode="json")

    def event_result(self, event_id: str) -> dict[str, Any] | None:
        result = CuriosityHarness(self.db_path).repository.get_response(event_id)
        return result.model_dump(mode="json") if result else None

    def context(self, child_id: str) -> dict[str, Any]:
        return child_context(self.db_path, child_id)

    def index_resources(self, catalog_path: str | Path, repository_root: str | Path) -> dict[str, Any]:
        return index_collection(
            self.db_path,
            catalog_path,
            repository_root=repository_root,
        ).__dict__

    def resource_inventory(self) -> dict[str, Any]:
        return resource_inventory(self.db_path)

    def resource_search(self, query: str, *, include_excerpts: bool = False) -> list[dict[str, Any]]:
        return search_resources(self.db_path, query, include_excerpts=include_excerpts)

    def inbox(self, status: str = "unassigned") -> list[dict[str, Any]]:
        return list_inbox(self.db_path, status=status)

    def assign_inbox(self, inbox_id: str, child_id: str) -> dict[str, Any]:
        row = next((item for item in self.inbox() if item["id"] == inbox_id), None)
        if not row:
            raise ValueError("unassigned inbox item not found")
        response = self.ask(
            child_id=child_id,
            text=row["text"],
            source="local_inbox",
            event_id=f"evt_inbox_{inbox_id}",
        )
        resolved = resolve_inbox(self.db_path, inbox_id, child_id=child_id)
        return {"inbox": resolved, "response": response}

    def dismiss_inbox(self, inbox_id: str) -> dict[str, Any]:
        return resolve_inbox(self.db_path, inbox_id, child_id=None, dismiss=True)

    def create_artifact(self, child_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        return ArtifactService(self.db_path, self.output_dir).create(
            child_id=child_id,
            spec=spec,
            visual_backend=self.visual_backend,
        )

    def create_artifact_from_response(self, event_id: str) -> dict[str, Any]:
        with connect(self.db_path) as conn:
            existing = conn.execute(
                """SELECT a.id,a.experience_id,a.path,a.sha256,a.validation_json,a.approval_status
                   FROM artifacts a JOIN experiences x ON x.id=a.experience_id
                   WHERE x.source_event_id=? ORDER BY a.created_at LIMIT 1""",
                (event_id,),
            ).fetchone()
            if existing and Path(existing["path"]).is_file():
                return {
                    "artifact_id": existing["id"],
                    "experience_id": existing["experience_id"],
                    "pdf_path": existing["path"],
                    "preview_path": str(Path(existing["path"]).with_suffix(".png")),
                    "sha256": existing["sha256"],
                    "validation": jload(existing["validation_json"]),
                    "approval_status": existing["approval_status"],
                    "duplicate": True,
                }
            row = conn.execute(
                """SELECT e.child_id,e.text,r.output_json,r.status FROM responses r
                   JOIN events e ON e.id=r.event_id WHERE r.event_id=?""",
                (event_id,),
            ).fetchone()
        if not row or row["status"] != "completed" or not row["child_id"]:
            raise ValueError("completed child response not found")
        output = jload(row["output_json"])
        extension = output.get("physical_extension") or {}
        body = [output["show"], output["nugget"]]
        body.extend(extension.get("instructions", []))
        spec = {
            "artifact_type": "wonder_page",
            "title": row["text"],
            "trust_tier": "B",
            "target_grade": next(
                (child["grade"] for child in self.children() if child["id"] == row["child_id"]),
                "family",
            ),
            "kicker": "NOTICE • PREDICT • INVESTIGATE",
            "prompt": output["ask"],
            "body": body[:8],
            "footer": "Keep the explanation small. Let the next question lead.",
            "assets": [],
            "source_event_id": event_id,
        }
        return self.create_artifact(row["child_id"], spec)

    def actions(self, status: str | None = None) -> list[dict[str, Any]]:
        return list_actions(self.db_path, status=status)

    def execute_action(self, action_id: str) -> dict[str, Any]:
        return execute_action(
            self.db_path,
            action_id,
            output_dir=self.output_dir,
            visual_backend=self.visual_backend,
        )

    def approve_artifact(self, artifact_id: str, *, note: str | None = None) -> dict[str, Any]:
        return approve_artifact(self.db_path, artifact_id, note=note)

    def print_artifact(
        self,
        artifact_id: str,
        approval_id: str,
        *,
        printer: str | None = None,
        send: bool = False,
    ) -> dict[str, Any]:
        return print_artifact(self.db_path, artifact_id, approval_id, printer=printer, send=send)

    def feedback(self, payload: dict[str, Any]) -> int:
        return record_feedback(self.db_path, FeedbackInput.model_validate(payload))

    def respond_to_opportunity(self, opportunity_id: str, decision: str) -> dict[str, Any]:
        if decision not in {"accepted", "dismissed"}:
            raise ValueError("decision must be accepted or dismissed")
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT child_id,kind,payload_json,status FROM opportunities WHERE id=?", (opportunity_id,)
            ).fetchone()
            if not row:
                raise KeyError(opportunity_id)
            if row["status"] not in {"suggested", decision}:
                raise ValueError(f"opportunity is already {row['status']}")
            conn.execute("UPDATE opportunities SET status=? WHERE id=?", (decision, opportunity_id))
            payload = jload(row["payload_json"])
        result: dict[str, Any] = {"opportunity_id": opportunity_id, "decision": decision}
        if decision == "accepted" and row["kind"] == "pull_thread" and payload.get("question"):
            result["thread"] = self.ask(
                child_id=row["child_id"],
                text=payload["question"],
                source="weekly_director",
                event_id=f"evt_{opportunity_id}",
            )
        return result

    def reflect(self, child_id: str) -> dict[str, Any]:
        return AutonomousDirector(self.db_path).reflect_for_child(child_id)

    def dashboard(self, child_id: str | None = None) -> dict[str, Any]:
        children = self.children()
        selected = child_id or (children[0]["id"] if children else None)
        with connect(self.db_path) as conn:
            artifacts = [
                {**dict(row), "spec": jload(row["spec_json"]), "validation": jload(row["validation_json"])}
                for row in conn.execute(
                    """SELECT artifacts.*,
                       (SELECT id FROM approvals WHERE approvals.artifact_id=artifacts.id AND decision='approved'
                        ORDER BY created_at DESC LIMIT 1) AS approval_id FROM artifacts"""
                    + (" WHERE child_id=?" if selected else "")
                    + " ORDER BY created_at DESC LIMIT 12",
                    (selected,) if selected else (),
                ).fetchall()
            ]
            responses = [
                {**dict(row), "output": jload(row["output_json"])}
                for row in conn.execute(
                    """SELECT r.*,e.child_id,e.text AS question FROM responses r JOIN events e ON e.id=r.event_id"""
                    + (" WHERE e.child_id=?" if selected else "")
                    + " ORDER BY r.created_at DESC LIMIT 12",
                    (selected,) if selected else (),
                ).fetchall()
            ]
        return {
            "children": children,
            "selected_child_id": selected,
            "responses": responses,
            "artifacts": artifacts,
            "actions": self.actions("proposed"),
            "opportunities": list_opportunities(self.db_path, selected) if selected else [],
            "resources": self.resource_inventory(),
            "inbox": self.inbox(),
            "onboarding": onboarding_status(self.db_path),
        }
