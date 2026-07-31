from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from service.agent.agno_planner_executor import (
    AgnoPlannerExecutor,
    AgnoStructuredPlanner,
)
from service.agent.agno_stream_workflow import (
    AgnoStreamWorkflow,
    AgnoWorkflowExecutionError,
)
from service.agent.llm_planner import build_safe_fallback_plan
from service.agent.plan_validator import PlanValidator
from service.agent.planner_registry import (
    DEFAULT_SCHEMA_REGISTRY,
    DEFAULT_TOOL_REGISTRY,
)


class _FakeWorkflowService:
    def __init__(
        self,
        *,
        decision: str = "answer",
        fail_step: str = "",
        response_cache_hit: bool = False,
        retry_decision: str = "answer",
    ) -> None:
        self.decision = decision
        self.fail_step = fail_step
        self.response_cache_hit = response_cache_hit
        self.retry_decision = retry_decision
        self.evidence_decision = SimpleNamespace(retry_limit=1)
        self.persisted: list[str] = []
        self.retrieval_calls = 0

    def _initial_workflow_state(self, **kwargs: Any) -> dict[str, Any]:
        return {
            **kwargs,
            "workflow_started_at": 0.0,
            "retry_count": 0,
            "observations": [],
        }

    async def _run(
        self,
        name: str,
        state: dict[str, Any],
        **updates: Any,
    ) -> dict[str, Any]:
        if self.fail_step == name:
            raise RuntimeError(f"{name} failed")
        return {**state, **updates}

    async def _step_load_session(self, state: dict[str, Any]) -> dict[str, Any]:
        sid = str(state.get("session_id") or state.get("question"))
        return await self._run("load_session", state, sid=sid)

    async def _step_build_conversation_context(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._run(
            "conversation_context",
            state,
            original_question=state["question"],
            effective_question=state["question"],
        )

    async def _step_understand_intent_and_slots(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._run(
            "intent_and_planner",
            state,
            query_type="information_extraction",
            planner_result={"execution_manifest": {"status": "ready", "tasks": []}},
        )

    async def _step_run_clarify_gate(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._run(
            "clarify_gate",
            state,
            clarify={"decision": self.decision},
            gate={"decision": self.decision} if self.decision == "refuse" else {},
        )

    def _route_after_clarify_gate(self, state: dict[str, Any]) -> str:
        if (state.get("clarify") or {}).get("decision") == "clarify":
            return "clarify"
        if (state.get("gate") or {}).get("decision") == "refuse":
            return "scope_refuse"
        return "retrieve"

    async def _step_lookup_response_cache(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.response_cache_hit:
            return await self._run(
                "response_cache_lookup",
                state,
                response_cache_hit=False,
            )
        return await self._run(
            "response_cache_lookup",
            state,
            response_cache_hit=True,
            response={"decision": "answer", "session_id": state["sid"]},
            evidence=[{"chunk_id": "cached"}],
            llm_answer_cache_hit=True,
        )

    @staticmethod
    def _route_after_response_cache(state: dict[str, Any]) -> str:
        return "cached" if state.get("response_cache_hit") else "retrieve"

    async def _step_build_clarify_response(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._run(
            "build_clarify_response",
            state,
            response={"decision": "clarify", "session_id": state["sid"]},
        )

    async def _step_retrieve_evidence(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        self.retrieval_calls += 1
        return await self._run(
            "retrieve_evidence",
            state,
            evidence=[{"chunk_id": state["sid"]}],
        )

    async def _step_evaluate_evidence(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._run(
            "evaluate_evidence",
            state,
            gate={"decision": "retry", "reason": "need_more"},
        )

    def _route_after_gate(self, state: dict[str, Any]) -> str:
        return (
            "retry"
            if (state.get("gate") or {}).get("decision") == "retry"
            and int(state.get("retry_count") or 0) < 1
            else "final"
        )

    async def _step_retry_retrieval(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        return await self._run(
            "retry_retrieval",
            state,
            retry_count=int(state.get("retry_count") or 0) + 1,
            gate={"decision": self.retry_decision},
        )

    def _route_after_retry(self, state: dict[str, Any]) -> str:
        return (
            "retry"
            if (state.get("gate") or {}).get("decision") == "retry"
            and int(state.get("retry_count") or 0) < 1
            else "final"
        )

    async def _step_build_answer_response(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        decision = str((state.get("gate") or {}).get("decision") or "answer")
        if decision == "retry":
            decision = "refuse"
        return await self._run(
            "build_answer_response",
            state,
            response={"decision": decision, "session_id": state["sid"]},
        )

    async def _step_persist_response(
        self, state: dict[str, Any]
    ) -> dict[str, Any]:
        self.persisted.append(str(state["sid"]))
        response = dict(state["response"])
        response["workflow_runner"] = "agno"
        response["workflow_run_id"] = state["workflow_run_id"]
        return await self._run("persist_response", state, response=response)


def _run_fake(
    service: _FakeWorkflowService,
    question: str,
    progress: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    async def callback(item: dict[str, Any]) -> None:
        if progress is not None:
            progress.append(item)

    return asyncio.run(
        AgnoStreamWorkflow(service).run(
            question=question,
            collection_name="test",
            session_id=None,
            top_k=5,
            expand_query_num=3,
            enable_cache=False,
            use_llm_intent_slot=False,
            progress_callback=callback,
        )
    )


def test_agno_workflow_runs_retry_and_persists_exactly_once() -> None:
    service = _FakeWorkflowService()
    progress: list[dict[str, Any]] = []

    response = _run_fake(service, "question-a", progress)

    assert response["decision"] == "answer"
    assert response["workflow_runner"] == "agno"
    assert response["workflow_run_id"]
    assert service.persisted == ["question-a"]
    phases = [item["phase"] for item in progress]
    assert "parallel_hybrid_retrieval" in phases
    assert "retry_retrieval" in phases
    assert "finalize_response" in phases


def test_agno_workflow_routes_clarify_without_retrieval() -> None:
    service = _FakeWorkflowService(decision="clarify")
    progress: list[dict[str, Any]] = []

    response = _run_fake(service, "question-b", progress)

    assert response["decision"] == "clarify"
    assert service.persisted == ["question-b"]
    assert "parallel_hybrid_retrieval" not in {
        item["phase"] for item in progress
    }


def test_agno_workflow_routes_scope_refusal_without_retrieval() -> None:
    service = _FakeWorkflowService(decision="refuse")

    response = _run_fake(service, "question-refuse")

    assert response["decision"] == "refuse"
    assert service.retrieval_calls == 0
    assert service.persisted == ["question-refuse"]


def test_agno_workflow_refuses_after_bounded_retry_is_exhausted() -> None:
    service = _FakeWorkflowService(retry_decision="retry")

    response = _run_fake(service, "question-retry-exhausted")

    assert response["decision"] == "refuse"
    assert service.retrieval_calls == 1
    assert service.persisted == ["question-retry-exhausted"]


def test_agno_workflow_uses_final_response_cache_without_retrieval() -> None:
    service = _FakeWorkflowService(response_cache_hit=True)
    progress: list[dict[str, Any]] = []

    response = _run_fake(service, "question-cache", progress)

    assert response["decision"] == "answer"
    assert service.retrieval_calls == 0
    assert service.persisted == ["question-cache"]
    retrieval_events = [
        item for item in progress if item["phase"] == "parallel_hybrid_retrieval"
    ]
    assert len(retrieval_events) == 1
    assert retrieval_events[0]["cache_hit"] is True


def test_agno_step_failure_is_visible_and_never_runs_another_executor() -> None:
    service = _FakeWorkflowService(fail_step="retrieve_evidence")

    with pytest.raises(AgnoWorkflowExecutionError, match="retrieve_evidence failed"):
        _run_fake(service, "question-c")

    assert service.persisted == []


def test_agno_workflow_isolates_concurrent_request_state() -> None:
    service = _FakeWorkflowService()
    workflow = AgnoStreamWorkflow(service)

    async def run(question: str) -> dict[str, Any]:
        return await workflow.run(
            question=question,
            collection_name="test",
            session_id=None,
            top_k=5,
            expand_query_num=3,
            enable_cache=False,
            use_llm_intent_slot=False,
            progress_callback=None,
        )

    async def run_both() -> tuple[dict[str, Any], dict[str, Any]]:
        first, second = await asyncio.gather(
            run("session-one"),
            run("session-two"),
        )
        return first, second

    first, second = asyncio.run(run_both())

    assert first["session_id"] == "session-one"
    assert second["session_id"] == "session-two"
    assert first["workflow_run_id"] != second["workflow_run_id"]
    assert sorted(service.persisted) == ["session-one", "session-two"]


class _StructuredPlanner:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    async def structured_json(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return self.payload


def test_agno_structured_planner_validates_fenced_json() -> None:
    plan = build_safe_fallback_plan(
        "information_extraction",
        {},
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
    )
    schema = DEFAULT_SCHEMA_REGISTRY.get("information_extraction")
    assert schema is not None

    payload = AgnoStructuredPlanner._validated_payload(
        "```json\n" + json.dumps(plan) + "\n```",
        schema,
    )

    assert payload is not None
    assert payload["intent"] == "information_extraction"
    assert len(payload["tasks"]) == 4


def test_planner_manifest_is_validated_ordered_and_audited() -> None:
    plan = build_safe_fallback_plan(
        "comparison",
        {
            "companies": ["腾讯"],
            "years": ["2023", "2024"],
            "metric": "营业收入",
            "compare_targets": ["2023", "2024"],
        },
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
    )
    validator = PlanValidator(
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        max_tasks=8,
    )
    executor = AgnoPlannerExecutor(
        SimpleNamespace(),
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        validator,
        structured_planner=_StructuredPlanner(plan),
    )

    result = asyncio.run(
        executor.plan(
            "比较腾讯2023年和2024年的营业收入",
            "comparison",
            {
                "companies": ["腾讯"],
                "years": ["2023", "2024"],
                "metric": "营业收入",
                "compare_targets": ["2023", "2024"],
            },
        )
    )

    tasks = result["execution_manifest"]["tasks"]
    assert result["validation"]["valid"] is True
    assert [item["task_id"] for item in tasks] == [
        "task_1",
        "task_2",
        "task_3",
        "task_4",
    ]
    finalized = executor.finalize_execution_manifest(
        result,
        {
            "tool_outputs": {
                "parallel_hybrid_retrieval": {},
                "two_stage_hybrid_rerank": {},
                "evidence_gate": {},
                "answer_generator": {},
            }
        },
    )
    assert finalized["execution_manifest"]["status"] == "completed"
    cached = executor.finalize_execution_manifest(
        result,
        {"tool_outputs": {"response_cache": {"cache_hit": True}}},
    )
    assert cached["execution_manifest"]["status"] == "completed_from_cache"
    assert {
        item["status"] for item in cached["execution_manifest"]["tasks"]
    } == {"satisfied_from_cache"}


def test_invalid_planner_tool_is_never_present_in_executable_manifest() -> None:
    invalid = build_safe_fallback_plan(
        "information_extraction",
        {},
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
    )
    invalid["tasks"][0]["tool_name"] = "send_email"
    validator = PlanValidator(
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        max_tasks=8,
    )
    executor = AgnoPlannerExecutor(
        SimpleNamespace(),
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        validator,
        allow_one_repair=False,
        structured_planner=_StructuredPlanner(invalid),
    )

    result = asyncio.run(
        executor.plan(
            "查找营业收入",
            "information_extraction",
            {},
        )
    )

    assert result["source"] == "deterministic_safe_fallback_after_invalid_plan"
    assert all(
        item["tool_name"] != "send_email"
        for item in result["execution_manifest"]["tasks"]
    )
