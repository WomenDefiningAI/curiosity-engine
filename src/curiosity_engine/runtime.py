from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

from .config import AppConfig
from .context_builder import build_context
from .contracts import ActionRequest, Event, GraphMutation, RunResult
from .openai_backend import OpenAIBackend, load_model_settings, model_key_is_configured
from .reasoning import ModelBackend, ReasoningEngine, ReasoningPolicy, ReasoningRejected, load_policies
from .repository import Job, Repository

__all__ = ["CuriosityHarness", "Event", "RunResult"]


def configured_backend(config: AppConfig, role: str = "reasoning") -> ModelBackend | None:
    settings = load_model_settings()
    if settings["CURIOSITY_BACKEND"].casefold() != "openai":
        return None
    if not model_key_is_configured(settings["OPENAI_API_KEY"]):
        raise RuntimeError("OpenAI mode is enabled but private/setup/model.env does not contain a valid API key")
    routes = config.production.get("models") or {}
    route = routes.get(role) or {}
    override = settings["CURIOSITY_MODEL"]
    if override:
        routes = {name: {**entry, "model": override} for name, entry in routes.items()}
    return OpenAIBackend(
        model=override or route.get("model") or "gpt-5.4",
        api_key=settings["OPENAI_API_KEY"],
        reasoning_effort=route.get("reasoning_effort"),
        routes=routes,
    )


class CuriosityHarness:
    """Durable local orchestrator. Models may propose output; code owns every side effect."""

    def __init__(
        self,
        db_path: str | Path,
        reasoning_engine: ReasoningEngine | None = None,
        *,
        config: AppConfig | None = None,
        worker_id: str | None = None,
    ):
        self.db_path = str(db_path)
        self.config = config or AppConfig.load()
        self.repository = Repository(self.db_path)
        self.policies = load_policies(self.config)
        self.reasoning = reasoning_engine or ReasoningEngine(
            configured_backend(self.config), prompt_dir=self.config.root / "prompts"
        )
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"

    def dispatch(self, raw_event: Event | dict[str, Any], policy: ReasoningPolicy | None = None) -> RunResult:
        event = raw_event if isinstance(raw_event, Event) else Event.model_validate(raw_event)
        submission = self.repository.submit_event(event)
        if submission.result is not None:
            return submission.result.model_copy(update={"duplicate": True})
        job = self.repository.claim_job(self.worker_id, submission.job_id)
        if job is None:
            existing = self.repository.get_response(event.id)
            if existing is not None:
                return existing.model_copy(update={"duplicate": True})
            event_state = self.repository.event(event.id)
            if event_state["status"] == "failed":
                return RunResult(
                    event_id=event.id,
                    workflow=(policy or self._policy_for_event(event.type)).workflow,
                    status="failed",
                    output={"error": event_state.get("error") or "job failed"},
                    duplicate=True,
                )
            return RunResult(
                event_id=event.id, workflow=(policy or self._policy_for_event(event.type)).workflow, status="queued"
            )
        return self._process_job(job, policy)

    def process_next(self) -> RunResult | None:
        job = self.repository.claim_job(self.worker_id)
        return self._process_job(job) if job else None

    def _process_job(self, job: Job, explicit_policy: ReasoningPolicy | None = None) -> RunResult:
        if job.job_type != "process_event" or not job.event_id:
            raise ValueError(f"unsupported job {job.job_type!r}")
        event_row = self.repository.event(job.event_id)
        event = Event.model_validate(
            {
                "id": event_row["id"],
                "type": event_row["type"],
                "child_id": event_row["child_id"],
                "text": event_row["text"],
                "source": event_row["source"] or "parent",
                "metadata": event_row["metadata"],
                "created_at": event_row["created_at"],
            }
        )
        policy = explicit_policy or self._policy_for_event(event.type)
        run_id = self.repository.start_run(event.id, policy.workflow, policy.to_dict(), job.id)
        try:
            if event.child_id:
                initial = self._initial_mutations(event)
                self.repository.record_initial_mutations(event.id, event.child_id, run_id, initial)
                context = build_context(
                    self.db_path,
                    event.child_id,
                    event.model_dump(mode="json"),
                    depth=policy.context_depth,
                    topic_hint=str(event.metadata.get("topic") or "") or None,
                )
            else:
                context = {
                    "event": event.model_dump(mode="json"),
                    "context_depth": 0,
                    "epistemic_rules": ["One observation is not a durable trait."],
                }
            envelope = self.reasoning.run(policy=policy, context=context, event=event.model_dump(mode="json"))
            output = dict(envelope.output)
            output["_reasoning"] = {
                "backend": envelope.backend,
                "model": envelope.model,
                "critics": [x.model_dump(mode="json") for x in envelope.critiques],
                "revision_rounds": envelope.revision_rounds,
                "policy": policy.to_dict(),
                "private_resource_mode": context.get("private_resource_mode", "not_used"),
                "private_resource_matches": len(context.get("private_resources", [])),
            }
            graph_updates = [GraphMutation.model_validate(item) for item in envelope.output.get("graph_updates", [])]
            actions = [ActionRequest.model_validate(item) for item in envelope.output.get("actions", [])]
            return self.repository.complete_event(
                event_id=event.id,
                job_id=job.id,
                run_id=run_id,
                workflow=policy.workflow,
                output=output,
                graph_updates=graph_updates,
                actions=actions,
                child_id=event.child_id,
            )
        except ReasoningRejected as exc:
            return self.repository.reject_event(
                event_id=event.id,
                job_id=job.id,
                run_id=run_id,
                workflow=policy.workflow,
                output={"candidate": exc.candidate, "critiques": exc.critiques},
                reason=str(exc),
            )
        except Exception as exc:
            self.repository.fail_job(job.id, event.id, run_id, exc)
            raise

    @staticmethod
    def _initial_mutations(event: Event) -> list[GraphMutation]:
        if not event.child_id:
            return []
        if event.type in {"child_question", "curiosity"}:
            mutations = [
                GraphMutation(
                    kind="add_observation",
                    observation_kind="curiosity",
                    text=event.text,
                    source=event.source,
                    confidence=1.0,
                    state={"event_id": event.id},
                ),
                GraphMutation(
                    kind="upsert_node",
                    node_kind="question",
                    label=event.text,
                    confidence=1.0,
                    state={"epistemic_state": "observation", "source_event_id": event.id},
                ),
            ]
            topics = event.metadata.get("topics") or ([event.metadata["topic"]] if event.metadata.get("topic") else [])
            for topic in topics[:8]:
                mutations.append(
                    GraphMutation(
                        kind="upsert_node",
                        node_kind="topic",
                        label=str(topic),
                        confidence=0.65,
                        state={"epistemic_state": "observation", "source_event_id": event.id},
                    )
                )
            return mutations
        if event.type == "interest_signal":
            return [
                GraphMutation(
                    kind="add_observation",
                    observation_kind="interest_signal",
                    text=event.text,
                    source=event.source,
                    confidence=0.7,
                    state={"event_id": event.id},
                )
            ]
        return []

    def _policy_for_event(self, event_type: str) -> ReasoningPolicy:
        if event_type in {"child_question", "curiosity"}:
            return self.policies["pull_thread"]
        return self.policies.get(event_type, self.policies["generic"])
