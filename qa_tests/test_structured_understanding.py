from __future__ import annotations

import asyncio

import pytest

from service.agent.company_registry import CompanyRegistry
from service.agent.controlled_agents import QuestionUnderstandingAgent
from service.agent.query_classifier import classify_query_type
from service.agent.schemas import PRIMARY_INTENT_TYPES, normalize_query_type
from service.agent.skill_registry import DEFAULT_SKILL_REGISTRY
from service.agent.structured_understanding import (
    HardSignalExtractor,
    SemanticSkillRouter,
    load_intent_router_config,
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


class _FixedRoute:
    def as_dict(self):
        return {
            "candidates": [
                {"query_type": "comparison", "score": 0.88},
                {"query_type": "information_extraction", "score": 0.75},
            ],
            "decision": "accept",
            "top_query_type": "comparison",
            "top_score": 0.88,
            "margin": 0.13,
            "provider": "test",
        }


class _FixedSemanticRouter:
    llm_fallback_enabled = False

    async def route(self, question: str):
        del question
        return _FixedRoute()


class _FakeEmbeddingService:
    provider_name = "test_embedding"

    @staticmethod
    def _vector(text: str):
        if "对比两个公司" in text or "谁的收入规模更大" in text:
            return [1.0, 0.0, 0.0, 0.0, 0.0]
        if "计算增长率" in text:
            return [0.0, 1.0, 0.0, 0.0, 0.0]
        if "分析原因" in text:
            return [0.0, 0.0, 1.0, 0.0, 0.0]
        if "总结" in text:
            return [0.0, 0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 0.0, 1.0]

    async def embed_text(self, text: str, **kwargs):
        del kwargs
        return self._vector(text)

    async def embed_texts(self, texts, **kwargs):
        del kwargs
        return [self._vector(text) for text in texts]


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


def test_embedding_router_returns_top_n_and_margin_for_five_intents() -> None:
    router = SemanticSkillRouter(
        _FakeEmbeddingService(),
        {
            "top_k": 3,
            "accept_threshold": 0.5,
            "reject_threshold": 0.2,
            "margin_threshold": 0.1,
            "prototypes": {
                "comparison": ["对比两个公司"],
                "metric_calculation": ["计算增长率"],
                "analysis": ["分析原因"],
                "summarization": ["总结文档"],
                "information_extraction": ["查询事实"],
            },
        },
    )

    route = asyncio.run(router.route("谁的收入规模更大"))

    assert route.decision == "accept"
    assert route.top_query_type == "comparison"
    assert route.candidates[0]["query_type"] == "comparison"
    assert len(route.candidates) == 3
    assert route.candidates[0]["matched_prototype_count"] == 1


def test_each_intent_has_at_least_twenty_independent_prototypes() -> None:
    config = load_intent_router_config()
    prototypes = config["prototypes"]

    assert set(prototypes) == set(PRIMARY_INTENT_TYPES)
    assert all(len(examples) >= 20 for examples in prototypes.values())
    assert all(len(examples) == len(set(examples)) for examples in prototypes.values())


def test_semantic_router_embeds_each_prototype_independently() -> None:
    class _RecordingEmbeddingService(_FakeEmbeddingService):
        def __init__(self) -> None:
            self.embedded_batches = []

        async def embed_texts(self, texts, **kwargs):
            self.embedded_batches.append(list(texts))
            return await super().embed_texts(texts, **kwargs)

    embedding_service = _RecordingEmbeddingService()
    router = SemanticSkillRouter(
        embedding_service,
        {
            "prototype_score_top_n": 2,
            "prototypes": {
                "comparison": ["对比两个公司", "谁的收入规模更大"],
                "information_extraction": ["查询事实", "提取披露数值"],
            },
        },
    )

    asyncio.run(router.route("谁的收入规模更大"))

    assert embedding_service.embedded_batches == [
        ["对比两个公司", "谁的收入规模更大", "查询事实", "提取披露数值"]
    ]
    assert router._prototype_vectors is not None
    assert len(router._prototype_vectors["comparison"]) == 2


def test_question_understanding_returns_one_intent_and_one_skill() -> None:
    agent = QuestionUnderstandingAgent(
        llm_service=None,
        semantic_router=_FixedSemanticRouter(),
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
        semantic_router=None,
        company_registry=_registry(),
    )

    result = asyncio.run(agent.understand(question, DEFAULT_SKILL_REGISTRY))

    assert result["query_type"] == query_type
    assert result["selected_skill"].skill_name == skill_name
    assert result["slots"]["__missing_required__"] == []


def test_question_understanding_keeps_compound_action_ambiguous() -> None:
    agent = QuestionUnderstandingAgent(
        llm_service=None,
        semantic_router=_FixedSemanticRouter(),
        company_registry=_registry(),
    )

    result = asyncio.run(
        agent.understand(
            "比较两家公司营收并分析差异原因。",
            DEFAULT_SKILL_REGISTRY,
        )
    )

    assert result["query_type"] == "ambiguous_query"
    assert result["intent_trace"]["routing_state"] == "ambiguous_action"


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
