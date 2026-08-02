from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Mapping

from agno.workflow.loop import Loop
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.steps import Steps
from agno.workflow.types import StepInput, StepOutput


StateHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


def _state(step_input: StepInput) -> Dict[str, Any]:
    previous = step_input.previous_step_content
    if isinstance(previous, Mapping):
        return dict(previous)
    if isinstance(step_input.input, Mapping):
        return dict(step_input.input)
    raise TypeError("Agno workflow step requires dictionary state")


def _step(
    name: str,
    handler: StateHandler,
    error_sink: Dict[str, str],
) -> Step:
    async def execute(step_input: StepInput) -> StepOutput:
        try:
            return StepOutput(content=await handler(_state(step_input)))
        except Exception as exc:
            error_sink["error"] = f"{name}: {exc}"
            raise

    return Step(
        name=name,
        executor=execute,
        max_retries=0,
        skip_on_failure=False,
        on_error="fail",
    )


def build_agno_steps(service: Any, error_sink: Dict[str, str]) -> list[Any]:
    load_session = _step("load_session", service._step_load_session, error_sink)
    conversation_context = _step(
        "conversation_context", service._step_build_conversation_context, error_sink
    )
    intent_and_planner = _step(
        "intent_and_planner",
        service._step_understand_intent_and_slots,
        error_sink,
    )
    clarify_gate = _step(
        "clarify_gate", service._step_run_clarify_gate, error_sink
    )
    build_clarify = _step(
        "build_clarify_response",
        service._step_build_clarify_response,
        error_sink,
    )
    response_cache = _step(
        "response_cache_lookup",
        service._step_lookup_response_cache,
        error_sink,
    )
    retrieve = _step(
        "retrieve_evidence", service._step_retrieve_evidence, error_sink
    )
    evaluate = _step(
        "evaluate_evidence", service._step_evaluate_evidence, error_sink
    )
    retry = _step(
        "retry_retrieval", service._step_retry_retrieval, error_sink
    )
    build_answer = _step(
        "build_answer_response",
        service._step_build_answer_response,
        error_sink,
    )
    persist = _step(
        "persist_response", service._step_persist_response, error_sink
    )

    retry_loop = Loop(
        name="bounded_evidence_retry",
        steps=[retry],
        max_iterations=max(1, int(service.evidence_decision.retry_limit)),
        end_condition=lambda outputs: bool(outputs)
        and service._route_after_retry(dict(outputs[-1].content or {})) == "final",
        forward_iteration_output=True,
    )
    retry_branch = Steps(
        name="retry_then_answer",
        steps=[retry_loop, build_answer, persist],
    )
    final_branch = Steps(
        name="answer_without_retry",
        steps=[build_answer, persist],
    )

    def select_evidence_path(step_input: StepInput) -> list[Any]:
        state = _state(step_input)
        if service._route_after_gate(state) == "retry":
            return [retry_branch]
        return [final_branch]

    evidence_router = Router(
        name="evidence_path",
        choices=[retry_branch, final_branch],
        selector=select_evidence_path,
    )
    cache_miss_branch = Steps(
        name="retrieve_after_cache_miss",
        steps=[retrieve, evaluate, evidence_router],
    )
    cache_hit_branch = Steps(
        name="persist_cached_response",
        steps=[persist],
    )

    def select_cache_path(step_input: StepInput) -> list[Any]:
        if service._route_after_response_cache(_state(step_input)) == "cached":
            return [cache_hit_branch]
        return [cache_miss_branch]

    cache_router = Router(
        name="response_cache_path",
        choices=[cache_hit_branch, cache_miss_branch],
        selector=select_cache_path,
    )
    retrieval_branch = Steps(
        name="lookup_cache_then_retrieve",
        steps=[response_cache, cache_router],
    )
    clarify_branch = Steps(
        name="clarify_response",
        steps=[build_clarify, persist],
    )
    refuse_branch = Steps(
        name="refuse_response",
        steps=[build_answer, persist],
    )

    def select_request_path(step_input: StepInput) -> list[Any]:
        route = service._route_after_clarify_gate(_state(step_input))
        if route == "clarify":
            return [clarify_branch]
        if route == "scope_refuse":
            return [refuse_branch]
        return [retrieval_branch]

    request_router = Router(
        name="request_path",
        choices=[clarify_branch, refuse_branch, retrieval_branch],
        selector=select_request_path,
    )
    return [
        load_session,
        conversation_context,
        intent_and_planner,
        clarify_gate,
        request_router,
    ]
