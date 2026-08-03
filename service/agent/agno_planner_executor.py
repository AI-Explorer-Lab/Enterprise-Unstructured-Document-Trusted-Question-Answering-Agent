from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.models.openai import OpenAIChat
from pydantic import BaseModel

from service.agent.llm_planner import LLMPlanner
from service.agent.plan_validator import PlanValidator
from service.agent.planner_registry import SchemaRegistry, ToolRegistry


_TOOL_PHASES = {
    "query_expander": "parallel_hybrid_retrieval",
    "parallel_hybrid_retrieval": "parallel_hybrid_retrieval",
    "table_prioritized_retrieval": "parallel_hybrid_retrieval",
    "two_stage_hybrid_rerank": "parallel_hybrid_retrieval",
    "evidence_gate": "evidence_decision",
    "answer_generator": "answer_generation",
}


class AgnoStructuredPlanner:
    """Expose the configured chat model through an Agno Agent.

    The Agent may only return the intent-specific Pydantic plan. It receives
    tool descriptions as data and is not given executable tools, so no model
    output can invoke retrieval before PlanValidator approves it.
    """

    def __init__(self, llm_service: Any) -> None:
        self.llm_service = llm_service

    def _model(self, max_tokens: int) -> Any:
        provider = str(getattr(self.llm_service, "provider_name", "") or "").lower()
        common = {
            "id": str(getattr(self.llm_service, "model", "") or "deepseek-chat"),
            "api_key": str(getattr(self.llm_service, "api_key", "") or ""),
            "base_url": getattr(self.llm_service, "base_url", None),
            "temperature": float(getattr(self.llm_service, "temperature", 0.2) or 0.2),
            "max_tokens": max(1, int(max_tokens)),
            "timeout": float(getattr(self.llm_service, "timeout_seconds", 90) or 90),
            "max_retries": 0,
        }
        if provider == "deepseek":
            return DeepSeek(**common, use_thinking=False)
        return OpenAIChat(**common)

    @staticmethod
    def _validated_payload(
        content: Any,
        schema: type[BaseModel],
    ) -> Dict[str, Any] | None:
        if isinstance(content, BaseModel):
            return content.model_dump()
        if isinstance(content, Mapping):
            candidate: Any = dict(content)
        elif isinstance(content, str):
            text = content.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                return None
        else:
            return None
        if not isinstance(candidate, Mapping):
            return None
        try:
            return schema.model_validate(dict(candidate)).model_dump()
        except Exception:
            return None

    async def structured_json(
        self,
        system_prompt: str,
        user_payload: Any,
        schema: type[BaseModel],
        max_tokens: int,
    ) -> Dict[str, Any] | None:
        if not bool(getattr(self.llm_service, "is_available", False)):
            return None
        agent = Agent(
            name="bounded_financial_qa_planner",
            model=self._model(max_tokens),
            instructions=[
                system_prompt,
                "Return exactly one valid JSON object and no markdown or commentary.",
                "Do not execute tools; tool descriptions are planning constraints only.",
                "The JSON object must match this schema exactly: "
                + json.dumps(
                    schema.model_json_schema(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ],
            expected_output="One JSON object matching the supplied schema.",
            use_json_mode=True,
            parse_response=False,
            retries=0,
            telemetry=False,
        )
        try:
            result = await agent.arun(
                input=json.dumps(user_payload, ensure_ascii=False, default=str),
                stream=False,
            )
        except Exception:
            return None
        return self._validated_payload(getattr(result, "content", None), schema)


def _ordered_tasks(plan: Mapping[str, Any]) -> list[Dict[str, Any]]:
    tasks = [
        dict(item)
        for item in list(plan.get("tasks") or [])
        if isinstance(item, Mapping)
    ]
    by_id = {
        str(task.get("task_id") or ""): task
        for task in tasks
        if str(task.get("task_id") or "")
    }
    remaining = list(by_id)
    ordered: list[Dict[str, Any]] = []
    completed: set[str] = set()
    while remaining:
        ready = [
            task_id
            for task_id in remaining
            if set(str(item) for item in by_id[task_id].get("depends_on") or [])
            <= completed
        ]
        if not ready:
            return []
        for task_id in ready:
            task = by_id[task_id]
            ordered.append(
                {
                    "task_id": task_id,
                    "task_type": str(task.get("task_type") or ""),
                    "tool_name": str(task.get("tool_name") or ""),
                    "arguments": dict(task.get("arguments") or {}),
                    "depends_on": [
                        str(item) for item in task.get("depends_on") or []
                    ],
                    "output_key": str(task.get("output_key") or ""),
                    "adapter_phase": _TOOL_PHASES.get(
                        str(task.get("tool_name") or ""), ""
                    ),
                    "status": "ready",
                }
            )
            completed.add(task_id)
            remaining.remove(task_id)
    return ordered


class AgnoPlannerExecutor:
    """Generate, validate and audit a bounded executable plan."""

    def __init__(
        self,
        llm_service: Any,
        schema_registry: SchemaRegistry,
        tool_registry: ToolRegistry,
        validator: PlanValidator,
        allow_one_repair: bool = True,
        structured_planner: Any | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.planner = LLMPlanner(
            structured_planner or AgnoStructuredPlanner(llm_service),
            schema_registry,
            tool_registry,
            validator,
            allow_one_repair=allow_one_repair,
        )

    def attach_execution_manifest(
        self,
        planner_result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        result = dict(planner_result)
        validation = (
            result.get("validation")
            if isinstance(result.get("validation"), Mapping)
            else {}
        )
        plan = result.get("plan") if isinstance(result.get("plan"), Mapping) else {}
        ordered = (
            _ordered_tasks(plan)
            if validation.get("valid") and validation.get("executable")
            else []
        )
        result["execution_manifest"] = {
            "status": "ready" if ordered else "not_executable",
            "task_count": len(ordered),
            "tasks": ordered,
            "stages": list(result.get("execution_stages") or []),
        }
        return result

    async def plan(
        self,
        question: str,
        intent_id: str,
        extracted_slots: Mapping[str, Any],
        conversation_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        result = await self.planner.plan(
            question,
            intent_id,
            extracted_slots,
            conversation_context=conversation_context or {},
        )
        return self.attach_execution_manifest(result)

    @staticmethod
    def planned_tools(planner_result: Mapping[str, Any]) -> frozenset[str]:
        manifest = (
            planner_result.get("execution_manifest")
            if isinstance(planner_result.get("execution_manifest"), Mapping)
            else {}
        )
        return frozenset(
            str(item.get("tool_name") or "")
            for item in list(manifest.get("tasks") or [])
            if isinstance(item, Mapping) and str(item.get("tool_name") or "")
        )

    def require_tools(
        self,
        planner_result: Mapping[str, Any],
        required_tools: set[str],
    ) -> None:
        validation = (
            planner_result.get("validation")
            if isinstance(planner_result.get("validation"), Mapping)
            else {}
        )
        if not bool(validation.get("executable")):
            raise RuntimeError("validated executable plan is required")
        missing = sorted(required_tools - self.planned_tools(planner_result))
        if missing:
            raise RuntimeError(
                "validated plan does not authorize required tools: "
                + ", ".join(missing)
            )

    def finalize_execution_manifest(
        self,
        planner_result: Mapping[str, Any],
        execution_context: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        result = dict(planner_result)
        manifest = (
            dict(result.get("execution_manifest") or {})
            if isinstance(result.get("execution_manifest"), Mapping)
            else {"status": "not_executable", "task_count": 0, "tasks": []}
        )
        outputs = (
            (execution_context or {}).get("tool_outputs")
            if isinstance((execution_context or {}).get("tool_outputs"), Mapping)
            else {}
        )
        response_cache_hit = bool(
            isinstance(outputs.get("response_cache"), Mapping)
            and outputs["response_cache"].get("cache_hit")
        )
        finalized = []
        completed_task_ids: set[str] = set()
        for item in list(manifest.get("tasks") or []):
            if not isinstance(item, Mapping):
                continue
            task = dict(item)
            tool_name = str(task.get("tool_name") or "")
            if response_cache_hit:
                task["status"] = "satisfied_from_cache"
                completed_task_ids.add(str(task.get("task_id") or ""))
            else:
                dependencies = {
                    str(value) for value in task.get("depends_on") or []
                }
                executed = (
                    dependencies <= completed_task_ids and tool_name in outputs
                )
                task["status"] = "executed" if executed else "not_executed"
                if executed:
                    completed_task_ids.add(str(task.get("task_id") or ""))
            finalized.append(task)
        manifest["tasks"] = finalized
        if finalized:
            if response_cache_hit:
                manifest["status"] = "completed_from_cache"
            else:
                manifest["status"] = (
                    "completed"
                    if all(item.get("status") == "executed" for item in finalized)
                    else "incomplete"
                )
        result["execution_manifest"] = manifest
        return result
