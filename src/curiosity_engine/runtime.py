from __future__ import annotations

import logging
import os
import socket
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .brain_config import SECRET_KEYS, load_brain_config, load_secret_settings, secret_is_configured
from .config import AppConfig
from .context_builder import build_context, fit_context_budget, prepare_model_inputs
from .contracts import ActionRequest, Event, GraphMutation, ObservationCandidate, RunResult
from .episodes import episode_for_event
from .interaction import answer_stack_fingerprint
from .learner_model import record_model_annotations
from .openai_backend import OpenAIBackend
from .pedagogy import (
    ApprovedPedagogyPlan,
    compile_artifact_action,
    plan_requires_current_safety_upgrade,
    proposal_from_approved,
    safety_response_requirements_for_text,
)
from .personalization import (
    PersonalizationAuthorizationChanged,
    PersonalizationService,
    decision_trace,
)
from .providers import AnthropicBackend, OpenRouterBackend
from .reasoning import (
    ModelBackend,
    ReasoningEngine,
    ReasoningPolicy,
    ReasoningRejected,
    load_policies,
    runtime_critic_roles,
)
from .repository import Job, Repository
from .text_review import configured_text_reviewer

__all__ = ["CuriosityHarness", "Event", "RunResult"]
logger = logging.getLogger(__name__)


def _validation_error_summary(exc: Exception) -> list[dict[str, str]]:
    """Keep planner failures diagnosable without persisting family or model input."""

    if not isinstance(exc, ValidationError):
        return []
    return [
        {
            "location": ".".join(str(part) for part in error.get("loc", ())),
            "type": str(error.get("type") or "validation_error"),
            "message": str(error.get("msg") or "Invalid value"),
        }
        for error in exc.errors(include_url=False, include_input=False)[:8]
    ]


def configured_backend(config: AppConfig, role: str = "reasoning") -> ModelBackend | None:
    settings = load_secret_settings()
    brain = load_brain_config()
    if brain is not None:
        if brain.runtime != "api":
            return None
        selected = brain.routes.get(role) or brain.routes.get("reasoning")
        if selected is None:
            return None
        provider = selected.provider
        routes = {name: entry.model_dump(mode="json") for name, entry in brain.routes.items()}
        model = selected.model
        reasoning_effort = selected.reasoning_effort
    else:
        provider = settings["CURIOSITY_BACKEND"].casefold()
        if provider == "deterministic":
            return None
        if provider not in SECRET_KEYS:
            raise RuntimeError(f"unsupported reasoning provider: {provider!r}")
        public_routes = config.production.get("models") or {}
        public_route = public_routes.get(role) or public_routes.get("reasoning") or {}
        model = settings["CURIOSITY_MODEL"] or public_route.get("model") or "gpt-5.4"
        reasoning_effort = public_route.get("reasoning_effort")
        routes = {
            name: {**entry, "provider": provider, "model": settings["CURIOSITY_MODEL"] or entry.get("model") or model}
            for name, entry in public_routes.items()
        }
    key_name = SECRET_KEYS[provider]
    api_key = settings[key_name]
    if not secret_is_configured(provider, api_key):
        raise RuntimeError(
            f"{provider} mode is enabled but private/setup/model.env does not contain a valid {key_name}"
        )
    if provider == "openai":
        return OpenAIBackend(
            model=model,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            routes=routes,
        )
    if provider == "anthropic":
        return AnthropicBackend(model=model, api_key=api_key, routes=routes)
    return OpenRouterBackend(model=model, api_key=api_key, routes=routes)


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
            configured_backend(self.config), prompt_dir=self.config.root / "prompts",
            text_reviewer=configured_text_reviewer(),
        )
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"

    def dispatch(
        self,
        raw_event: Event | dict[str, Any],
        policy: ReasoningPolicy | None = None,
        *,
        agent_lineage: dict[str, str] | None = None,
    ) -> RunResult:
        event = raw_event if isinstance(raw_event, Event) else Event.model_validate(raw_event)
        submission = self.repository.submit_event(event, agent_lineage=agent_lineage)
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
        pedagogy_plan: ApprovedPedagogyPlan | None = None
        personalization_decision: dict[str, Any] | None = None
        try:
            if event.child_id:
                episode = episode_for_event(self.db_path, event.id)
                graph_eligible = bool(episode and episode.get("learning_scope") == "family_signal")
                initial = self._initial_mutations(event, episode) if graph_eligible else []
                self.repository.record_initial_mutations(event.id, event.child_id, run_id, initial)
                context = build_context(
                    self.db_path,
                    event.child_id,
                    event.model_dump(mode="json"),
                    depth=policy.context_depth,
                    topic_hint=str(event.metadata.get("topic") or "") or None,
                )
            else:
                context = fit_context_budget(
                    {
                        "event": event.model_dump(mode="json"),
                        "context_depth": 0,
                        "epistemic_rules": ["One observation is not a durable trait."],
                    }
                )
            # Phase 1 projections remain measurable but cannot influence model
            # recommendations until the later evaluated-personalization gate.
            model_context, model_event, shadow_diagnostic = prepare_model_inputs(context, event.model_dump(mode="json"))
            if event.child_id and policy.workflow in {"quick_answer", "pull_thread"}:
                personalization = PersonalizationService(self.db_path, self.config.root)
                personalization_enabled = bool(
                    ((self.config.production.get("components") or {}).get("pedagogy_kernel") or {}).get(
                        "shadow_personalization", False
                    )
                )
                stored_plan = self.repository.pedagogy_plan(event.id, policy.workflow)
                if stored_plan is not None:
                    stored_plan.pop("_status", None)
                    pedagogy_plan = ApprovedPedagogyPlan.model_validate(stored_plan)
                    if plan_requires_current_safety_upgrade(pedagogy_plan, event.text):
                        pedagogy_plan = self.reasoning.pedagogy_compiler.fallback(
                            event=event.model_dump(mode="json"),
                            workflow=policy.workflow,
                        )
                        persisted_plan = self.repository.persist_pedagogy_plan(
                            plan=pedagogy_plan,
                            proposal={"fallback_reason": "validator_safety_upgrade"},
                            run_id=run_id,
                        )
                        pedagogy_plan = ApprovedPedagogyPlan.model_validate(persisted_plan)
                        personalization_decision = None
                    else:
                        personalization_decision = personalization.decision_for_plan(pedagogy_plan.plan_id)
                        if (
                            pedagogy_plan.personalization is not None
                            and (
                                personalization_decision is None
                                or not personalization.active_decision_is_authorized(personalization_decision)
                            )
                        ):
                            baseline_proposal = proposal_from_approved(pedagogy_plan)
                            pedagogy_plan = self.reasoning.pedagogy_compiler.compile(
                                baseline_proposal,
                                event=event.model_dump(mode="json"),
                                workflow=policy.workflow,
                                provenance="safe_fallback",
                            )
                            persisted_plan = self.repository.persist_pedagogy_plan(
                                plan=pedagogy_plan,
                                proposal={"fallback_reason": "personalization_authorization_changed"},
                                run_id=run_id,
                            )
                            pedagogy_plan = ApprovedPedagogyPlan.model_validate(persisted_plan)
                            personalization_decision = None
                        else:
                            self.repository.bind_pedagogy_plan_to_run(plan=pedagogy_plan, run_id=run_id)
                else:
                    safety_requirements = safety_response_requirements_for_text(event.text)
                    if safety_requirements:
                        pedagogy_plan = self.reasoning.pedagogy_compiler.fallback(
                            event=event.model_dump(mode="json"),
                            workflow=policy.workflow,
                        )
                        proposal_payload = {"fallback_reason": "code_owned_safety_route"}
                        personalization_decision = None
                    else:
                        try:
                            proposal = self.reasoning.propose_pedagogy(
                                policy=policy,
                                context=model_context,
                                event=model_event,
                            )
                            baseline_plan = self.reasoning.approve_pedagogy(
                                proposal,
                                event=event.model_dump(mode="json"),
                                workflow=policy.workflow,
                            )
                            if personalization_enabled:
                                personalization_decision = personalization.prepare_decision(
                                    child_id=event.child_id,
                                    event_id=event.id,
                                    workflow=policy.workflow,
                                    proposal=proposal,
                                    baseline_plan=baseline_plan,
                                )
                            pedagogy_plan = (
                                self.reasoning.approve_pedagogy(
                                    proposal,
                                    event=event.model_dump(mode="json"),
                                    workflow=policy.workflow,
                                    personalization_decision=personalization_decision,
                                )
                                if personalization_decision and personalization_decision["mode"] == "active"
                                else baseline_plan
                            )
                            proposal_payload = proposal.model_dump(mode="json")
                        except Exception as exc:
                            validation_errors = _validation_error_summary(exc)
                            logger.warning(
                                "pedagogy proposal failed closed to grade baseline workflow=%s error_type=%s "
                                "validation_errors=%s",
                                policy.workflow,
                                exc.__class__.__name__,
                                validation_errors,
                            )
                            pedagogy_plan = self.reasoning.pedagogy_compiler.planner_failure_fallback(
                                event=event.model_dump(mode="json"),
                                workflow=policy.workflow,
                            )
                            proposal_payload = {
                                "fallback_reason": exc.__class__.__name__,
                                "validation_errors": validation_errors,
                            }
                            personalization_decision = None
                    persisted_plan = self.repository.persist_pedagogy_plan(
                        plan=pedagogy_plan,
                        proposal=proposal_payload,
                        run_id=run_id,
                        personalization_decision=personalization_decision,
                    )
                    pedagogy_plan = ApprovedPedagogyPlan.model_validate(persisted_plan)
                if personalization_decision and not personalization.active_decision_is_authorized(
                    personalization_decision
                ):
                    baseline_proposal = proposal_from_approved(pedagogy_plan)
                    pedagogy_plan = self.reasoning.pedagogy_compiler.compile(
                        baseline_proposal,
                        event=event.model_dump(mode="json"),
                        workflow=policy.workflow,
                        provenance="safe_fallback",
                    )
                    persisted_plan = self.repository.persist_pedagogy_plan(
                        plan=pedagogy_plan,
                        proposal={"fallback_reason": "personalization_authorization_changed"},
                        run_id=run_id,
                    )
                    pedagogy_plan = ApprovedPedagogyPlan.model_validate(persisted_plan)
                    personalization_decision = None
            authorization_preflight = (
                lambda: personalization.require_active_decision_authorized(personalization_decision)
                if personalization_decision is not None
                else None
            ) if personalization_decision and personalization_decision["mode"] == "active" else None
            try:
                envelope = self.reasoning.run(
                    policy=policy,
                    context=model_context,
                    event=model_event,
                    pedagogy_plan=pedagogy_plan,
                    authorization_preflight=authorization_preflight,
                )
            except PersonalizationAuthorizationChanged:
                envelope = None
            if envelope is None or (
                personalization_decision
                and personalization_decision["mode"] == "active"
                and not personalization.active_decision_is_authorized(personalization_decision)
            ):
                baseline_proposal = proposal_from_approved(pedagogy_plan)
                pedagogy_plan = self.reasoning.pedagogy_compiler.compile(
                    baseline_proposal,
                    event=event.model_dump(mode="json"),
                    workflow=policy.workflow,
                    provenance="safe_fallback",
                )
                persisted_plan = self.repository.persist_pedagogy_plan(
                    plan=pedagogy_plan,
                    proposal={"fallback_reason": "personalization_authorization_changed_during_generation"},
                    run_id=run_id,
                )
                pedagogy_plan = ApprovedPedagogyPlan.model_validate(persisted_plan)
                personalization_decision = None
                envelope = self.reasoning.run(
                    policy=policy,
                    context=model_context,
                    event=model_event,
                    pedagogy_plan=pedagogy_plan,
                )
            output = dict(envelope.output)
            if pedagogy_plan is not None:
                output["_pedagogy"] = pedagogy_plan.trace()
            output["_reasoning"] = {
                "backend": envelope.backend,
                "model": envelope.model,
                "critics": [x.model_dump(mode="json") for x in envelope.critiques],
                "critic_rounds": [
                    [item.model_dump(mode="json") for item in round_items] for round_items in envelope.critique_rounds
                ],
                "revision_rounds": envelope.revision_rounds,
                "recovery_strategy": envelope.recovery_strategy,
                "policy": policy.to_dict(),
                "effective_critic_roles": (
                    []
                    if envelope.backend == "code_owned"
                    else list(runtime_critic_roles(policy, pedagogy_plan))
                ),
                "response_route": (
                    "code_owned_safety" if envelope.backend == "code_owned" else "model_generated"
                ),
                "private_resource_mode": context.get("private_resource_mode", "not_used"),
                "private_resource_matches": len(context.get("private_resources", [])),
                "answer_stack_hash": answer_stack_fingerprint(self.db_path),
                "shadow_learner_snapshot": shadow_diagnostic,
                "pedagogy_plan_id": pedagogy_plan.plan_id if pedagogy_plan else None,
                "pedagogy_plan_hash": pedagogy_plan.plan_hash if pedagogy_plan else None,
            }
            if personalization_decision is not None:
                output["_reasoning"]["shadow_personalization"] = decision_trace(personalization_decision)
            observation_candidates = [
                ObservationCandidate.model_validate(item) for item in envelope.output.get("observation_candidates", [])
            ]
            # Phase 0: model annotations are response-audit data only. They cannot
            # become graph effects until Phase 1 supplies a versioned task/evidence
            # compiler with deterministic provenance and promotion rules.
            graph_updates: list[GraphMutation] = []
            output["observation_candidates"] = [
                candidate.model_dump(mode="json") for candidate in observation_candidates
            ]
            annotation_ids: list[str] = []
            if event.child_id and observation_candidates:
                annotation_ids = record_model_annotations(
                    self.db_path,
                    child_id=event.child_id,
                    event_id=event.id,
                    run_id=run_id,
                    candidates=observation_candidates,
                )
            output["_reasoning"]["observation_candidate_count"] = len(observation_candidates)
            output["_reasoning"]["observation_candidate_status"] = "audit_only_phase1"
            output["_reasoning"]["observation_annotation_ids"] = annotation_ids
            actions: list[ActionRequest] = []
            for item in envelope.output.get("actions", []):
                action = ActionRequest.model_validate(item)
                if pedagogy_plan is not None and action.type.value == "propose_artifact":
                    try:
                        action = action.model_copy(
                            update={"payload": compile_artifact_action(action.payload, pedagogy_plan)}
                        )
                    except ValueError as exc:
                        logger.warning("discarded unapproved artifact action reason=%s", str(exc))
                        continue
                actions.append(action)
            if not (event.child_id and graph_eligible):
                actions = []
            return self.repository.complete_event(
                event_id=event.id,
                job_id=job.id,
                run_id=run_id,
                workflow=policy.workflow,
                output=output,
                graph_updates=graph_updates,
                actions=actions,
                child_id=event.child_id,
                pedagogy_plan=pedagogy_plan,
            )
        except ReasoningRejected as exc:
            logger.warning(
                "reasoning rejected workflow=%s run_id=%s reason=%s critique_verdicts=%s",
                policy.workflow,
                run_id,
                str(exc),
                [str(item.get("verdict") or "unknown") for item in exc.critiques],
            )
            return self.repository.reject_event(
                event_id=event.id,
                job_id=job.id,
                run_id=run_id,
                workflow=policy.workflow,
                output={
                    "candidate": exc.candidate,
                    "critiques": exc.critiques,
                    "_reasoning": {
                        "critic_rounds": exc.critique_rounds,
                        "recovery_strategy": exc.recovery_strategy,
                    },
                },
                reason=str(exc),
                pedagogy_plan=pedagogy_plan,
            )
        except PersonalizationAuthorizationChanged as exc:
            logger.warning(
                "personalization authorization changed at atomic completion; queued baseline retry "
                "event_id=%s run_id=%s",
                event.id,
                run_id,
            )
            self.repository.fail_job(job.id, event.id, run_id, exc)
            return RunResult(
                event_id=event.id,
                run_id=run_id,
                workflow=policy.workflow,
                status="queued",
                output={"retry_reason": "personalization_authorization_changed"},
            )
        except Exception as exc:
            logger.exception(
                "workflow failed event_id=%s job_id=%s run_id=%s workflow=%s",
                event.id,
                job.id,
                run_id,
                policy.workflow,
            )
            self.repository.fail_job(job.id, event.id, run_id, exc)
            raise

    @staticmethod
    def _initial_mutations(event: Event, episode: dict[str, Any] | None = None) -> list[GraphMutation]:
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
                    state={"event_id": event.id, **(episode or {})},
                ),
                GraphMutation(
                    kind="upsert_node",
                    node_kind="question",
                    label=event.text,
                    confidence=1.0,
                    state={
                        "epistemic_state": "observation",
                        "source_event_id": event.id,
                        "count_semantics": "mention_only",
                        **(episode or {}),
                    },
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
                        state={
                            "epistemic_state": "observation",
                            "source_event_id": event.id,
                            "count_semantics": "mention_only",
                            **(episode or {}),
                        },
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
                    state={"event_id": event.id, **(episode or {})},
                )
            ]
        return []

    def _policy_for_event(self, event_type: str) -> ReasoningPolicy:
        if event_type in {"child_question", "curiosity"}:
            return self.policies["pull_thread"]
        return self.policies.get(event_type, self.policies["generic"])
