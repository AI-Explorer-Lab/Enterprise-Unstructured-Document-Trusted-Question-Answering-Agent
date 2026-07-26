from __future__ import annotations

import asyncio
from datetime import date

import pytest

from service.agent.company_registry import CompanyRegistry
from service.agent.controlled_agents import QuestionUnderstandingAgent
from service.agent.intent_gate import LLMIntentGate, load_intent_gate_config
from service.agent.query_classifier import classify_query_type
from service.agent.schemas import PRIMARY_INTENT_TYPES, normalize_query_type
from service.agent.skill_registry import DEFAULT_SKILL_REGISTRY
from service.agent.structured_understanding import (
    HardSignalExtractor,
    query_type_from_frame,
)
from service.llm.llm_client import _grounded_answer_scope_contract


def _registry() -> CompanyRegistry:
    return CompanyRegistry(
        [
            {
                "company_id": "xindao",
                "company_name": "上海芯导电子科技股份有限公司",
                "aliases": ["芯导", "芯导科技", "芯导科技股份有限公司"],
            }
        ]
    )


class _FixedIntentGate:
    def __init__(
        self,
        decision: str,
        intent_id: str = "",
        sub_intents: list[str] | None = None,
    ) -> None:
        self.decision = decision
        self.intent_id = intent_id
        self.sub_intents = list(sub_intents or [])
        self.questions: list[str] = []

    async def decide(self, question: str):
        self.questions.append(question)
        execution_intent = (
            self.intent_id
            if self.decision == "select"
            else (self.sub_intents[-1] if self.decision == "planner" else "")
        )
        return {
            "valid": True,
            "decision": self.decision,
            "route_status": "accepted" if execution_intent else "unknown",
            "intent_id": self.intent_id if self.decision == "select" else "",
            "sub_intents": self.sub_intents if self.decision == "planner" else [],
            "execution_intent": execution_intent,
            "top_intent": execution_intent,
            "provider": "llm",
            "strategy": "llm_zero_shot",
            "few_shot": False,
            "validation_error": "",
        }


class _StructuredIntentLLM:
    def __init__(self, payload):
        self.payload = payload
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
        return dict(self.payload)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("芯导科技 2025 年营业收入是多少？", "information_extraction"),
        ("芯导科技 2025 年研发投入占营业收入比例是多少？", "information_extraction"),
        ("根据2024年和2025年营业收入计算增长率。", "metric_calculation"),
        ("2025年营业收入相比2024年增长了多少？", "metric_calculation"),
        ("比较芯导科技2024年和2025年的营业收入。", "comparison"),
        ("分析芯导科技经营现金流下降的原因。", "analysis"),
        ("总结芯导科技年报中的主要风险。", "summarization"),
    ],
)
def test_classifier_supports_five_single_intents(question: str, expected: str) -> None:
    assert classify_query_type(question) == expected


def test_classifier_does_not_drop_compound_actions() -> None:
    assert classify_query_type("比较两家公司营收并分析差异原因") == "ambiguous_query"


def test_information_extraction_keeps_table_as_evidence_mode() -> None:
    frame = HardSignalExtractor(_registry()).extract("芯导科技2025年营业收入是多少？")

    assert frame["primary_action"] == "extract"
    assert frame["slots"]["metrics"] == ["营业收入"]
    assert frame["evidence_modes"] == ["table"]
    assert query_type_from_frame(frame) == "information_extraction"


def test_metric_calculation_is_distinct_from_information_extraction() -> None:
    frame = HardSignalExtractor(_registry()).extract("计算芯导科技2025年资产负债率")

    assert frame["primary_action"] == "calculate"
    assert query_type_from_frame(frame) == "metric_calculation"


def test_financial_three_statements_keep_analysis_intent() -> None:
    frame = HardSignalExtractor(_registry()).extract("分析下芯导科技的财务三表")

    assert frame["primary_action"] == "analyze"
    assert frame["domain_objects"] == ["financial_three_statements"]
    assert frame["slots"]["companies"] == ["上海芯导电子科技股份有限公司"]
    assert frame["evidence_modes"] == ["table"]
    assert query_type_from_frame(frame) == "analysis"


def test_flexible_comparison_pattern_is_a_hard_action_signal() -> None:
    frame = HardSignalExtractor(_registry()).extract("中芯国际和华虹半导体谁的收入规模更大？")

    assert frame["primary_action"] == "compare"
    assert frame["slots"]["companies"] == ["中芯国际", "华虹半导体"]
    assert frame["slots"]["metrics"] == ["营业收入"]
    assert query_type_from_frame(frame) == "comparison"


def test_year_comparison_uses_periods_as_compare_targets() -> None:
    frame = HardSignalExtractor(_registry()).extract("比较芯导科技2024年和2025年的营业收入。")

    assert frame["slots"]["periods"] == ["2024", "2025"]
    assert frame["slots"]["compare_targets"] == ["2024", "2025"]
    assert query_type_from_frame(frame) == "comparison"


def test_deterministic_layer_normalizes_report_period_statement_and_metric() -> None:
    frame = HardSignalExtractor(_registry()).extract(
        "查询芯导科技FY2024年度报告中2024Q1资产负债表的总资产和资产负债率。"
    )

    assert frame["slots"]["periods"] == ["2024", "2024Q1"]
    assert frame["slots"]["quarters"] == ["2024Q1"]
    assert frame["slots"]["report_types"] == ["annual_report"]
    assert frame["slots"]["statement_types"] == ["balance_sheet"]
    assert frame["slots"]["metrics"] == ["资产总额", "资产负债率"]


def test_deterministic_layer_keeps_semiannual_report_distinct() -> None:
    frame = HardSignalExtractor(_registry()).extract("总结芯导科技2025年半年度报告。")

    assert frame["slots"]["report_types"] == ["semiannual_report"]
    assert frame["slots"]["half_years"] == ["2025H1"]


def test_deterministic_layer_uses_explicit_reference_date_for_relative_year() -> None:
    frame = HardSignalExtractor(_registry()).extract(
        "查询芯导科技去年的净利润。",
        reference_date=date(2026, 7, 26),
    )

    assert frame["slots"]["periods"] == ["2025"]
    relative_evidence = next(
        item
        for item in frame["field_evidence"]
        if item.get("method") == "relative_year_parser"
    )
    assert relative_evidence["source_text"] == "去年"
    assert relative_evidence["reference_date"] == "2026-07-26"


def test_deterministic_layer_separates_requested_location_from_execution_slots() -> None:
    frame = HardSignalExtractor(_registry()).extract(
        "定位doc_123中现金流量表第27页的原文。"
    )

    assert frame["slots"]["statement_types"] == ["cash_flow_statement"]
    assert frame["slots"]["document_references"] == ["doc_123"]
    assert frame["slots"]["requested_pages"] == [27]
    assert frame["requirements"]["need_location"] is True
    assert frame["requirements"]["need_citation"] is True
    assert "document_ids" not in frame["slots"]
    assert "page_numbers" not in frame["slots"]


def test_deterministic_layer_normalizes_amount_and_percentage_conditions() -> None:
    frame = HardSignalExtractor(_registry()).extract(
        "查询营业收入超过一亿元且同比增长15%的项目。"
    )

    assert frame["slots"]["numeric_conditions"] == [
        {
            "kind": "percentage",
            "operator": "eq",
            "value": 15.0,
            "unit": "percent",
            "raw_text": "15%",
        },
        {
            "kind": "amount",
            "operator": "gt",
            "value": 100000000.0,
            "unit": "CNY",
            "raw_text": "超过一亿元",
        },
    ]


def test_production_intent_router_is_zero_shot_llm_without_prototypes() -> None:
    config = load_intent_gate_config()

    assert config["strategy"] == "llm_zero_shot"
    assert config["few_shot_enabled"] is False
    assert "prototypes" not in config
    assert "min_intent_score" not in config
    assert "min_score_margin" not in config


def test_llm_intent_gate_selects_one_primary_intent_without_few_shot() -> None:
    llm = _StructuredIntentLLM(
        {
            "decision": "select",
            "intent_id": "comparison",
            "sub_intents": [],
        }
    )
    gate = LLMIntentGate(llm, {"enabled": True, "max_tokens": 320})

    route = asyncio.run(gate.decide("比较两家公司2025年的营业收入"))

    assert route["valid"] is True
    assert route["decision"] == "select"
    assert route["execution_intent"] == "comparison"
    assert route["few_shot"] is False
    assert llm.calls[0]["user_payload"] == {
        "question": "比较两家公司2025年的营业收入"
    }


def test_llm_intent_gate_planner_keeps_order_and_uses_terminal_intent() -> None:
    llm = _StructuredIntentLLM(
        {
            "decision": "planner",
            "intent_id": "",
            "sub_intents": ["comparison", "analysis"],
        }
    )
    gate = LLMIntentGate(llm, {"enabled": True})

    route = asyncio.run(gate.decide("比较两家公司营收并分析差异原因"))

    assert route["valid"] is True
    assert route["sub_intents"] == ["comparison", "analysis"]
    assert route["execution_intent"] == "analysis"
    assert route["route_status"] == "accepted"


def test_llm_intent_gate_rejects_invalid_planner_payload() -> None:
    llm = _StructuredIntentLLM(
        {
            "decision": "planner",
            "intent_id": "",
            "sub_intents": ["analysis"],
        }
    )
    gate = LLMIntentGate(llm, {"enabled": True})

    route = asyncio.run(gate.decide("分析一下"))

    assert route["valid"] is False
    assert route["decision"] == "error"
    assert route["validation_error"] == "planner_requires_two_sub_intents"


def test_question_understanding_returns_one_intent_and_one_skill() -> None:
    agent = QuestionUnderstandingAgent(
        llm_service=None,
        intent_gate=_FixedIntentGate("select", "comparison"),
        company_registry=_registry(),
    )

    result = asyncio.run(
        agent.understand(
            "比较中芯国际和华虹半导体的营收。",
            DEFAULT_SKILL_REGISTRY,
            use_llm_intent_slot=False,
        )
    )

    assert result["query_type"] == "comparison"
    assert result["selected_skill"].skill_name == "ComparisonSkill"
    assert "secondary_intents" not in result
    assert "secondary_actions" not in result["structured_frame"]
    assert result["slots"]["companies"] == ["中芯国际", "华虹半导体"]
    assert result["slots"]["metric"] == "营业收入"
    assert result["intent_trace"]["primary_action"] == "compare"


@pytest.mark.parametrize(
    ("question", "query_type", "skill_name"),
    [
        ("芯导科技2025年营业收入是多少？", "information_extraction", "InformationExtractionSkill"),
        ("根据2024年和2025年营业收入计算增长率。", "metric_calculation", "MetricCalculationSkill"),
        ("比较芯导科技2024年和2025年的营业收入。", "comparison", "ComparisonSkill"),
        ("分析芯导科技经营现金流下降的原因。", "analysis", "AnalysisSkill"),
        ("总结芯导科技年报中的主要风险。", "summarization", "SummarizationSkill"),
    ],
)
def test_question_understanding_routes_each_single_intent(
    question: str,
    query_type: str,
    skill_name: str,
) -> None:
    agent = QuestionUnderstandingAgent(
        llm_service=None,
        intent_gate=_FixedIntentGate("select", query_type),
        company_registry=_registry(),
    )

    result = asyncio.run(agent.understand(question, DEFAULT_SKILL_REGISTRY))

    assert result["query_type"] == query_type
    assert result["selected_skill"].skill_name == skill_name
    assert result["slots"]["__missing_required__"] == []


def test_llm_planner_decision_preserves_all_sub_intents_and_terminal_skill() -> None:
    agent = QuestionUnderstandingAgent(
        llm_service=None,
        intent_gate=_FixedIntentGate(
            "planner",
            sub_intents=["comparison", "analysis"],
        ),
        company_registry=_registry(),
    )

    result = asyncio.run(
        agent.understand(
            "比较两家公司营收并分析差异原因。",
            DEFAULT_SKILL_REGISTRY,
        )
    )

    assert result["query_type"] == "analysis"
    assert result["intent_trace"]["routing_state"] == "planner"
    assert result["intent_trace"]["intent_decision"] == "planner"
    assert result["intent_trace"]["sub_intents"] == ["comparison", "analysis"]
    assert result["slots"]["intent_sub_intents"] == ["comparison", "analysis"]


def test_llm_clarify_decision_does_not_force_a_rule_intent() -> None:
    agent = QuestionUnderstandingAgent(
        llm_service=None,
        intent_gate=_FixedIntentGate("clarify"),
        company_registry=_registry(),
    )

    result = asyncio.run(
        agent.understand(
            "给我弄一下财务数据。",
            DEFAULT_SKILL_REGISTRY,
        )
    )

    assert result["query_type"] == "ambiguous_query"
    assert result["intent_trace"]["understanding_source"] == "llm_intent_gate"
    assert result["intent_trace"]["routing_state"] == "needs_clarification"
    assert result["slots"]["intent_decision"] == "clarify"


def test_llm_reject_decision_marks_request_out_of_scope() -> None:
    agent = QuestionUnderstandingAgent(
        llm_service=None,
        intent_gate=_FixedIntentGate("reject"),
        company_registry=_registry(),
    )

    result = asyncio.run(
        agent.understand(
            "帮我订一张明天去北京的机票。",
            DEFAULT_SKILL_REGISTRY,
        )
    )

    assert result["query_type"] == "ambiguous_query"
    assert result["intent_trace"]["understanding_source"] == "llm_intent_gate"
    assert result["intent_trace"]["routing_state"] == "out_of_scope"
    assert result["slots"]["intent_decision"] == "reject"


def test_llm_failure_uses_auditable_hard_action_without_embedding() -> None:
    agent = QuestionUnderstandingAgent(
        llm_service=None,
        company_registry=_registry(),
    )

    result = asyncio.run(
        agent.understand(
            "分析芯导科技经营现金流下降的原因。",
            DEFAULT_SKILL_REGISTRY,
        )
    )

    assert result["query_type"] == "analysis"
    assert result["intent_trace"]["understanding_source"] == "deterministic_fallback"
    assert result["intent_trace"]["intent_router"]["fallback"] == "deterministic_hard_signal"


def test_legacy_query_types_normalize_to_new_model() -> None:
    assert PRIMARY_INTENT_TYPES == (
        "information_extraction",
        "metric_calculation",
        "comparison",
        "analysis",
        "summarization",
    )
    assert normalize_query_type("fact_lookup") == "information_extraction"
    assert normalize_query_type("table_qa") == "information_extraction"
    assert normalize_query_type("citation_locate") == "information_extraction"
    assert normalize_query_type("multi_doc_compare") == "comparison"
    assert normalize_query_type("report_generation") == "summarization"


def test_answer_contract_preserves_output_and_citation_requirements() -> None:
    contract = _grounded_answer_scope_contract(
        {
            "company_name": "上海芯导电子科技股份有限公司",
            "years": [2025],
            "output_format": "short_report",
            "answer_requirements": {
                "need_citation": True,
                "length": "short",
            },
        }
    )

    assert "concise report" in contract
    assert "Citation requirement" in contract
    assert "keep the answer brief" in contract
