from __future__ import annotations

from typing import Any, Dict, List, Mapping

from service.agent.plan_validator import PlanValidationResult, PlanValidator
from service.agent.planner_models import OutputRequirements
from service.agent.planner_registry import SchemaRegistry, ToolRegistry


def _clean_list(value: Any) -> List[str]:
    values = value if isinstance(value, list) else ([value] if value not in (None, "") else [])
    result: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _initial_input_slots(intent_id: str, slots: Mapping[str, Any]) -> Dict[str, Any]:
    companies = _clean_list(slots.get("companies"))
    periods = _clean_list(slots.get("years") or slots.get("periods") or slots.get("period"))
    metrics = _clean_list(slots.get("metrics") or slots.get("metric"))
    document_names = _clean_list(slots.get("document_names"))
    numeric_conditions = [
        dict(item)
        for item in list(slots.get("numeric_conditions") or [])
        if isinstance(item, Mapping)
    ]
    base: Dict[str, Any] = {
        "companies": companies,
        "periods": periods,
        "quarters": _clean_list(slots.get("quarters")),
        "half_years": _clean_list(slots.get("half_years")),
        "report_types": _clean_list(slots.get("report_types")),
        "statement_types": _clean_list(slots.get("statement_types")),
        "requested_pages": [
            int(item)
            for item in list(slots.get("requested_pages") or [])
            if isinstance(item, int) and not isinstance(item, bool)
        ],
        "document_references": _clean_list(slots.get("document_references")),
        "document_name": (
            str(slots.get("document_name") or "").strip()
            or (document_names[0] if document_names else None)
        ),
        "numeric_conditions": numeric_conditions,
    }
    if intent_id != "summarization":
        base["metrics"] = metrics
    if intent_id == "information_extraction":
        base["target"] = str(slots.get("target_statement") or slots.get("metric") or slots.get("scope") or "").strip() or None
    elif intent_id == "metric_calculation":
        base["derived_metric"] = str(slots.get("metric") or "").strip() or None
    elif intent_id == "comparison":
        base["compare_targets"] = _clean_list(slots.get("compare_targets"))
        base["comparison_dimension"] = str(slots.get("metric") or slots.get("focus") or "").strip() or None
    elif intent_id == "analysis":
        base["analysis_topic"] = str(
            slots.get("analysis_topic") or slots.get("metric") or ""
        ).strip() or None
        base["analysis_dimension"] = str(slots.get("focus") or "").strip() or None
    elif intent_id == "summarization":
        base["summary_scope"] = str(
            slots.get("summary_scope") or slots.get("target_statement") or ""
        ).strip() or None
        base["focus"] = str(slots.get("focus") or "").strip() or None
    return base


def _output_requirements(intent_id: str, slots: Mapping[str, Any]) -> Dict[str, Any]:
    requirements = slots.get("requirements") if isinstance(slots.get("requirements"), Mapping) else {}
    output_format = str(slots.get("output_format") or "answer")
    if output_format == "short_report":
        output_format = "report"
    if intent_id == "summarization" and output_format == "answer":
        output_format = "summary"
    return OutputRequirements(
        need_summary=intent_id == "summarization" or output_format in {"summary", "report"},
        need_location=bool(requirements.get("need_location") or requirements.get("need_citation")),
        need_citation=bool(requirements.get("need_citation", True)),
        need_comparison=intent_id == "comparison" or bool(slots.get("compare_targets")),
        output_format=output_format if output_format in {"answer", "summary", "report", "table"} else "answer",
    ).model_dump()


def _task(
    task_id: str,
    task_type: str,
    tool_name: str,
    output_key: str,
    depends_on: List[str] | None = None,
    arguments: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "task_type": task_type,
        "tool_name": tool_name,
        "arguments": dict(arguments or {}),
        "depends_on": list(depends_on or []),
        "output_key": output_key,
        "required": True,
    }


def _protect_user_derived_fields(
    candidate: Mapping[str, Any],
    seed: Mapping[str, Any],
    schema_registry: SchemaRegistry,
    intent_id: str,
) -> Dict[str, Any]:
    protected = dict(candidate)
    candidate_slots = (
        dict(candidate.get("input_slots") or {})
        if isinstance(candidate.get("input_slots"), Mapping)
        else {}
    )
    seed_slots = dict(seed.get("input_slots") or {})
    for key, value in seed_slots.items():
        if value not in (None, "", []):
            candidate_slots[key] = value
    protected["intent"] = intent_id
    protected["input_slots"] = candidate_slots

    candidate_requirements = (
        dict(candidate.get("output_requirements") or {})
        if isinstance(candidate.get("output_requirements"), Mapping)
        else {}
    )
    seed_requirements = dict(seed.get("output_requirements") or {})
    for key in ("need_summary", "need_location", "need_citation", "need_comparison"):
        if seed_requirements.get(key):
            candidate_requirements[key] = True
    if (
        seed_requirements.get("output_format")
        and seed_requirements.get("output_format") != "answer"
    ):
        candidate_requirements["output_format"] = seed_requirements["output_format"]
    protected["output_requirements"] = candidate_requirements

    required = schema_registry.required_input_slots(intent_id)
    protected["missing_required_slots"] = [
        name for name in required if candidate_slots.get(name) in (None, "", [])
    ]
    return protected


def apply_validated_plan_to_execution_slots(
    slots: Mapping[str, Any],
    planner_result: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(slots)
    validation = (
        planner_result.get("validation")
        if isinstance(planner_result.get("validation"), Mapping)
        else {}
    )
    plan = (
        planner_result.get("plan")
        if isinstance(planner_result.get("plan"), Mapping)
        else {}
    )
    if not validation.get("valid") or not plan:
        return merged
    input_slots = (
        plan.get("input_slots")
        if isinstance(plan.get("input_slots"), Mapping)
        else {}
    )
    intent_id = str(plan.get("intent") or "")

    companies = _clean_list(input_slots.get("companies"))
    periods = _clean_list(input_slots.get("periods"))
    metrics = _clean_list(input_slots.get("metrics"))
    if companies and not merged.get("companies"):
        merged["companies"] = companies
    if periods:
        if not merged.get("years"):
            merged["years"] = periods
        if not merged.get("period"):
            merged["period"] = "、".join(periods)
    if metrics and not merged.get("metric"):
        merged["metric"] = "、".join(metrics)

    document_name = str(input_slots.get("document_name") or "").strip()
    if document_name and not merged.get("document_name"):
        merged["document_name"] = document_name
    for key in (
        "quarters",
        "half_years",
        "report_types",
        "statement_types",
        "requested_pages",
        "document_references",
        "numeric_conditions",
    ):
        value = input_slots.get(key)
        if isinstance(value, list) and value and not merged.get(key):
            merged[key] = list(value)
    if intent_id == "information_extraction":
        target = str(input_slots.get("target") or "").strip()
        if target and not merged.get("scope"):
            merged["scope"] = target
    elif intent_id == "metric_calculation":
        derived_metric = str(input_slots.get("derived_metric") or "").strip()
        if derived_metric and not merged.get("metric"):
            merged["metric"] = derived_metric
    elif intent_id == "comparison":
        compare_targets = _clean_list(input_slots.get("compare_targets"))
        if compare_targets and not merged.get("compare_targets"):
            merged["compare_targets"] = compare_targets
        dimension = str(input_slots.get("comparison_dimension") or "").strip()
        if dimension and not merged.get("metric"):
            merged["metric"] = dimension
    elif intent_id == "analysis":
        topic = str(input_slots.get("analysis_topic") or "").strip()
        if topic:
            merged["scope"] = topic
        dimension = str(input_slots.get("analysis_dimension") or "").strip()
        if dimension:
            merged["focus"] = dimension
    elif intent_id == "summarization":
        summary_scope = str(input_slots.get("summary_scope") or "").strip()
        if summary_scope:
            merged["scope"] = summary_scope
        focus = str(input_slots.get("focus") or "").strip()
        if focus:
            merged["focus"] = focus

    output_requirements = (
        plan.get("output_requirements")
        if isinstance(plan.get("output_requirements"), Mapping)
        else {}
    )
    output_format = str(output_requirements.get("output_format") or "").strip()
    if output_format:
        merged["output_format"] = output_format
    requirements = (
        dict(merged.get("requirements") or {})
        if isinstance(merged.get("requirements"), Mapping)
        else {}
    )
    for key in ("need_summary", "need_location", "need_citation", "need_comparison"):
        if key in output_requirements:
            requirements[key] = bool(output_requirements[key])
    merged["requirements"] = requirements
    return merged


def build_safe_fallback_plan(
    intent_id: str,
    slots: Mapping[str, Any],
    schema_registry: SchemaRegistry,
    tool_registry: ToolRegistry,
) -> Dict[str, Any]:
    input_slots = _initial_input_slots(intent_id, slots)
    required = schema_registry.required_input_slots(intent_id)
    missing = [name for name in required if input_slots.get(name) in (None, "", [])]
    requirements = _output_requirements(intent_id, slots)
    tasks: List[Dict[str, Any]] = []
    if not missing:
        tasks.append(_task("task_1", "retrieve", "parallel_hybrid_retrieval", "retrieval"))
        tasks.append(
            _task(
                "task_2",
                "search",
                "two_stage_hybrid_rerank",
                "ranked_evidence",
                ["task_1"],
            )
        )
        tasks.append(
            _task(
                "task_3",
                "locate",
                "evidence_gate",
                "grounded_evidence",
                ["task_2"],
                {
                    "need_citation": bool(requirements["need_citation"]),
                    "need_location": bool(requirements["need_location"]),
                },
            )
        )
        final_type = {
            "metric_calculation": "calculate",
            "comparison": "compare",
            "analysis": "analyze",
            "summarization": "summarize",
        }.get(intent_id, "answer")
        if requirements["output_format"] == "report":
            final_type = "generate_report"
        tasks.append(
            _task(
                "task_4",
                final_type,
                "answer_generator",
                "answer",
                ["task_3"],
                {
                    "output_format": requirements["output_format"],
                    "need_citation": bool(requirements["need_citation"]),
                    "need_location": bool(requirements["need_location"]),
                },
            )
        )
    return {
        "intent": intent_id,
        "input_slots": input_slots,
        "missing_required_slots": missing,
        "output_requirements": requirements,
        "tasks": tasks,
    }


class LLMPlanner:
    def __init__(
        self,
        llm_service: Any,
        schema_registry: SchemaRegistry,
        tool_registry: ToolRegistry,
        validator: PlanValidator,
        allow_one_repair: bool = True,
    ) -> None:
        self.llm_service = llm_service
        self.schema_registry = schema_registry
        self.tool_registry = tool_registry
        self.validator = validator
        self.allow_one_repair = bool(allow_one_repair)

    async def plan(
        self,
        question: str,
        intent_id: str,
        extracted_slots: Mapping[str, Any],
        conversation_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        schema = self.schema_registry.get(intent_id)
        if schema is None:
            return self._failed_result(intent_id, [f"unknown intent schema: {intent_id}"])
        seed = build_safe_fallback_plan(
            intent_id,
            extracted_slots,
            self.schema_registry,
            self.tool_registry,
        )
        routed_sub_intents = _clean_list(extracted_slots.get("intent_sub_intents"))
        payload = await self._call_planner(
            question,
            intent_id,
            schema,
            seed,
            conversation_context or {},
            routed_sub_intents,
        )
        source = "llm"
        if payload is None:
            payload = seed
            source = "deterministic_safe_fallback"
        else:
            payload = _protect_user_derived_fields(
                payload,
                seed,
                self.schema_registry,
                intent_id,
            )
        validation = self.validator.validate(intent_id, payload)
        repair_attempted = False
        if not validation.valid and self.allow_one_repair and source == "llm":
            repair_attempted = True
            repaired = await self._repair_plan(
                question,
                intent_id,
                schema,
                payload,
                validation,
            )
            if repaired is not None:
                payload = _protect_user_derived_fields(
                    repaired,
                    seed,
                    self.schema_registry,
                    intent_id,
                )
                source = "llm_repair"
                validation = self.validator.validate(intent_id, payload)
        if not validation.valid:
            payload = seed
            source = "deterministic_safe_fallback_after_invalid_plan"
            validation = self.validator.validate(intent_id, payload)
        return {
            "plan": payload,
            "validation": validation.as_dict(),
            "source": source,
            "repair_attempted": repair_attempted,
            "routed_sub_intents": routed_sub_intents,
        }

    async def _call_planner(
        self,
        question: str,
        intent_id: str,
        schema: type,
        seed: Mapping[str, Any],
        conversation_context: Mapping[str, Any],
        routed_sub_intents: List[str],
    ) -> Dict[str, Any] | None:
        structured_json = getattr(self.llm_service, "structured_json", None)
        if not callable(structured_json):
            return None
        try:
            payload = await structured_json(
                "Create one bounded execution plan. Extract only user-provided input slots. "
                "Never populate document ids, chunks, pages, tables, evidence, scores, citations, or tool outputs. "
                "Use only the supplied tools and return only the structured schema. "
                "When routed_sub_intents contains multiple operations, cover every operation in that execution order "
                "while keeping the routed terminal intent fixed.",
                {
                    "question": question,
                    "intent": intent_id,
                    "routed_sub_intents": routed_sub_intents,
                    "seed_input": dict(seed),
                    "conversation_context": dict(conversation_context),
                    "available_tools": self.tool_registry.describe_tools(intent_id),
                },
                schema=schema,
                max_tokens=1200,
            )
        except Exception:
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    async def _repair_plan(
        self,
        question: str,
        intent_id: str,
        schema: type,
        plan_payload: Mapping[str, Any],
        validation: PlanValidationResult,
    ) -> Dict[str, Any] | None:
        structured_json = getattr(self.llm_service, "structured_json", None)
        if not callable(structured_json):
            return None
        try:
            payload = await structured_json(
                "Repair only the listed structural validation errors. Keep the routed intent and user meaning fixed. "
                "Do not add execution-only slots or tools outside the supplied whitelist.",
                {
                    "question": question,
                    "intent": intent_id,
                    "validation_errors": list(validation.errors),
                    "current_plan": dict(plan_payload),
                    "available_tools": self.tool_registry.describe_tools(intent_id),
                },
                schema=schema,
                max_tokens=1200,
            )
        except Exception:
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    @staticmethod
    def _failed_result(intent_id: str, errors: List[str]) -> Dict[str, Any]:
        return {
            "plan": {
                "intent": intent_id,
                "input_slots": {},
                "missing_required_slots": [],
                "output_requirements": {},
                "tasks": [],
            },
            "validation": {
                "valid": False,
                "executable": False,
                "errors": errors,
                "missing_required_slots": [],
            },
            "source": "failed",
            "repair_attempted": False,
        }
