from __future__ import annotations

import asyncio
import json
from pathlib import Path

from service.agent.schemas import PRIMARY_INTENT_TYPES
from service.agent.execution_context import update_execution_context
from service.agent.llm_planner import (
    LLMPlanner,
    apply_validated_plan_to_execution_slots,
    build_safe_fallback_plan,
)
from service.agent.plan_validator import PlanValidator
from service.agent.planner_models import EXECUTION_SLOT_NAMES
from service.agent.planner_registry import (
    DEFAULT_SCHEMA_REGISTRY,
    DEFAULT_TOOL_REGISTRY,
)


def _validator() -> PlanValidator:
    return PlanValidator(
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        max_tasks=8,
    )


def _comparison_plan():
    return build_safe_fallback_plan(
        "comparison",
        {
            "companies": ["腾讯"],
            "years": ["2023", "2024"],
            "metric": "营业收入",
            "compare_targets": ["2023", "2024"],
            "requirements": {"need_citation": True},
        },
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
    )


class _PlannerLLM:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    async def structured_json(self, system_prompt, user_payload, schema, max_tokens):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "schema": schema,
                "max_tokens": max_tokens,
            }
        )
        return self.payloads.pop(0)


def test_schema_registry_loads_an_intent_specific_model() -> None:
    comparison_schema = DEFAULT_SCHEMA_REGISTRY.get("comparison")
    summary_schema = DEFAULT_SCHEMA_REGISTRY.get("summarization")

    assert comparison_schema is not None
    assert summary_schema is not None
    comparison_fields = comparison_schema.model_fields["input_slots"].annotation.model_fields
    summary_fields = summary_schema.model_fields["input_slots"].annotation.model_fields
    assert "compare_targets" in comparison_fields
    assert "compare_targets" not in summary_fields
    assert not EXECUTION_SLOT_NAMES.intersection(comparison_fields)


def test_tool_registry_exposes_only_intent_whitelist() -> None:
    calculation_tools = DEFAULT_TOOL_REGISTRY.allowed_names("metric_calculation")
    summary_tools = DEFAULT_TOOL_REGISTRY.allowed_names("summarization")

    assert "table_prioritized_retrieval" in calculation_tools
    assert "table_prioritized_retrieval" not in summary_tools
    assert "send_email" not in calculation_tools


def test_planner_uses_structured_schema_and_intent_tool_subset() -> None:
    plan = _comparison_plan()
    llm = _PlannerLLM([plan])
    planner = LLMPlanner(
        llm,
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        _validator(),
    )

    result = asyncio.run(
        planner.plan(
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

    assert result["source"] == "llm"
    assert result["validation"]["valid"] is True
    assert result["validation"]["executable"] is True
    assert llm.calls[0]["schema"] is DEFAULT_SCHEMA_REGISTRY.get("comparison")
    exposed = {
        item["name"]
        for item in llm.calls[0]["user_payload"]["available_tools"]
    }
    assert exposed == DEFAULT_TOOL_REGISTRY.allowed_names("comparison")


def test_planner_receives_llm_routed_sub_intents_in_execution_order() -> None:
    plan = build_safe_fallback_plan(
        "analysis",
        {
            "metric": "营业收入",
            "focus": "差异原因",
        },
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
    )
    llm = _PlannerLLM([plan])
    planner = LLMPlanner(
        llm,
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        _validator(),
    )

    result = asyncio.run(
        planner.plan(
            "比较两家公司营业收入并分析差异原因",
            "analysis",
            {
                "metric": "营业收入",
                "focus": "差异原因",
                "intent_sub_intents": ["comparison", "analysis"],
            },
        )
    )

    assert result["routed_sub_intents"] == ["comparison", "analysis"]
    assert llm.calls[0]["user_payload"]["routed_sub_intents"] == [
        "comparison",
        "analysis",
    ]
    assert result["plan"]["intent"] == "analysis"


def test_planner_cannot_override_rule_slots_or_required_output_constraints() -> None:
    plan = _comparison_plan()
    plan["input_slots"]["companies"] = ["虚构公司"]
    plan["input_slots"]["periods"] = ["2020"]
    plan["output_requirements"]["need_citation"] = False
    llm = _PlannerLLM([plan])
    planner = LLMPlanner(
        llm,
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        _validator(),
    )

    result = asyncio.run(
        planner.plan(
            "比较腾讯2023年和2024年的营业收入并给出引用",
            "comparison",
            {
                "companies": ["腾讯"],
                "years": ["2023", "2024"],
                "metric": "营业收入",
                "compare_targets": ["2023", "2024"],
                "requirements": {"need_citation": True},
            },
        )
    )

    assert result["plan"]["input_slots"]["companies"] == ["腾讯"]
    assert result["plan"]["input_slots"]["periods"] == ["2023", "2024"]
    assert result["plan"]["output_requirements"]["need_citation"] is True


def test_validator_rejects_wrong_tool_unknown_dependency_and_cycle() -> None:
    plan = _comparison_plan()
    plan["tasks"][0]["tool_name"] = "send_email"
    plan["tasks"][0]["depends_on"] = ["task_4"]
    plan["tasks"][3]["depends_on"] = ["task_1"]

    result = _validator().validate("comparison", plan)

    assert result.valid is False
    assert any("not allowed" in error for error in result.errors)
    assert any("cyclic" in error for error in result.errors)


def test_validator_rejects_execution_slots_in_planner_arguments() -> None:
    plan = _comparison_plan()
    plan["tasks"][0]["arguments"] = {"document_id": "invented-doc"}

    result = _validator().validate("comparison", plan)

    assert result.valid is False
    assert any("execution-only slots" in error for error in result.errors)


def test_validator_checks_tool_argument_types() -> None:
    plan = _comparison_plan()
    plan["tasks"][2]["arguments"] = {
        "need_citation": "yes",
        "need_location": True,
    }

    result = _validator().validate("comparison", plan)

    assert result.valid is False
    assert any("invalid argument types" in error for error in result.errors)


def test_invalid_llm_plan_gets_exactly_one_structured_repair() -> None:
    invalid = _comparison_plan()
    invalid["tasks"][0]["tool_name"] = "send_email"
    repaired = _comparison_plan()
    llm = _PlannerLLM([invalid, repaired])
    planner = LLMPlanner(
        llm,
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        _validator(),
        allow_one_repair=True,
    )

    result = asyncio.run(
        planner.plan(
            "比较腾讯2023年和2024年的营业收入",
            "comparison",
            {
                "compare_targets": ["2023", "2024"],
                "metric": "营业收入",
            },
        )
    )

    assert len(llm.calls) == 2
    assert result["repair_attempted"] is True
    assert result["source"] == "llm_repair"
    assert result["validation"]["valid"] is True


def test_failed_repair_falls_back_without_executing_invalid_plan() -> None:
    invalid = _comparison_plan()
    invalid["tasks"][0]["tool_name"] = "send_email"
    llm = _PlannerLLM([invalid, invalid])
    planner = LLMPlanner(
        llm,
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
        _validator(),
        allow_one_repair=True,
    )

    result = asyncio.run(
        planner.plan(
            "比较腾讯2023年和2024年的营业收入",
            "comparison",
            {
                "compare_targets": ["2023", "2024"],
                "metric": "营业收入",
            },
        )
    )

    assert len(llm.calls) == 2
    assert result["source"] == "deterministic_safe_fallback_after_invalid_plan"
    assert result["validation"]["valid"] is True
    assert all(
        task["tool_name"] != "send_email"
        for task in result["plan"]["tasks"]
    )


def test_missing_user_slot_produces_valid_but_non_executable_plan() -> None:
    plan = build_safe_fallback_plan(
        "comparison",
        {"metric": "营业收入"},
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
    )

    result = _validator().validate("comparison", plan)

    assert result.valid is True
    assert result.executable is False
    assert result.missing_required_slots == ("compare_targets",)
    assert plan["tasks"] == []


def test_validated_planner_slots_feed_executor_without_execution_slots() -> None:
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
    result = {
        "plan": plan,
        "validation": _validator().validate("comparison", plan).as_dict(),
    }

    execution_slots = apply_validated_plan_to_execution_slots({}, result)

    assert execution_slots["companies"] == ["腾讯"]
    assert execution_slots["years"] == ["2023", "2024"]
    assert execution_slots["compare_targets"] == ["2023", "2024"]
    assert execution_slots["requirements"]["need_comparison"] is True
    assert not EXECUTION_SLOT_NAMES.intersection(execution_slots)


def test_deterministic_domain_slots_survive_plan_validation_and_executor_adapter() -> None:
    plan = build_safe_fallback_plan(
        "information_extraction",
        {
            "years": ["2024", "2024Q1"],
            "quarters": ["2024Q1"],
            "report_types": ["annual_report"],
            "statement_types": ["cash_flow_statement"],
            "requested_pages": [27],
            "document_references": ["doc_123"],
            "numeric_conditions": [
                {
                    "kind": "amount",
                    "operator": "gt",
                    "value": 100000000.0,
                    "unit": "CNY",
                    "raw_text": "超过一亿元",
                }
            ],
        },
        DEFAULT_SCHEMA_REGISTRY,
        DEFAULT_TOOL_REGISTRY,
    )
    validation = _validator().validate("information_extraction", plan)

    assert validation.valid is True
    adapted = apply_validated_plan_to_execution_slots(
        {},
        {"plan": plan, "validation": validation.as_dict()},
    )
    assert adapted["quarters"] == ["2024Q1"]
    assert adapted["report_types"] == ["annual_report"]
    assert adapted["statement_types"] == ["cash_flow_statement"]
    assert adapted["requested_pages"] == [27]
    assert adapted["document_references"] == ["doc_123"]
    assert "page_numbers" not in adapted
    assert "document_ids" not in adapted


def test_execution_context_is_populated_only_from_tool_outputs() -> None:
    initial = update_execution_context()
    updated = update_execution_context(
        initial,
        evidence=[
            {
                "doc_id": "doc-1",
                "chunk_id": "chunk-2",
                "table_id": "table-3",
                "score": 0.91,
                "metadata": {"page_idx": 8},
            }
        ],
        citations=[{"doc_id": "doc-1", "metadata": {"page_idx": 8}}],
        tool_name="parallel_hybrid_retrieval",
        tool_output={"evidence_count": 1},
    )

    assert initial["document_ids"] == []
    assert updated["document_ids"] == ["doc-1"]
    assert updated["chunk_ids"] == ["chunk-2"]
    assert updated["page_numbers"] == [8]
    assert updated["table_ids"] == ["table-3"]
    assert updated["retrieval_scores"] == [0.91]


def test_query_understanding_evaluation_samples_cover_all_intents_and_guards() -> None:
    fixture = Path(__file__).parent / "fixtures" / "query_understanding_eval.json"
    samples = json.loads(fixture.read_text(encoding="utf-8"))

    covered_intents = {
        sample["expected_intent"]
        for sample in samples
        if sample.get("expected_intent")
    }
    covered_statuses = {
        sample["expected_route_status"]
        for sample in samples
        if sample.get("expected_route_status")
    }
    assert set(PRIMARY_INTENT_TYPES).issubset(covered_intents)
    assert {"ambiguous", "unknown"}.issubset(covered_statuses)
