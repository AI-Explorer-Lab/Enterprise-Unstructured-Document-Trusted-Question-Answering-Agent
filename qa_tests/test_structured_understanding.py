from __future__ import annotations

import asyncio

from service.agent.company_registry import CompanyRegistry
from service.agent.controlled_agents import QuestionUnderstandingAgent
from service.agent.skill_registry import DEFAULT_SKILL_REGISTRY
from service.agent.structured_understanding import (
    HardSignalExtractor,
    SemanticSkillRouter,
    query_type_from_frame,
    secondary_query_types,
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
                {"query_type": "multi_doc_compare", "score": 0.88},
                {"query_type": "table_qa", "score": 0.75},
            ],
            "decision": "accept",
            "top_query_type": "multi_doc_compare",
            "top_score": 0.88,
            "margin": 0.13,
            "provider": "test",
        }


class _FixedSemanticRouter:
    async def route(self, question: str):
        del question
        return _FixedRoute()


class _FakeEmbeddingService:
    provider_name = "test_embedding"

    @staticmethod
    def _vector(text: str):
        if "对比两个公司" in text or "谁的收入规模更大" in text:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if "财务指标" in text:
            return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        if "总结" in text:
            return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if "定位" in text:
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if "撰写" in text:
            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    async def embed_text(self, text: str, **kwargs):
        del kwargs
        return self._vector(text)

    async def embed_texts(self, texts, **kwargs):
        del kwargs
        return [self._vector(text) for text in texts]


def test_hard_signals_build_multi_dimensional_frame() -> None:
    question = "对比中芯国际和华虹半导体的营收，并给出出处，最后生成一份简短报告。"
    frame = HardSignalExtractor(_registry()).extract(question)

    assert frame["primary_action"] == "compare"
    assert frame["slots"]["companies"] == ["中芯国际", "华虹半导体"]
    assert frame["slots"]["compare_targets"] == ["中芯国际", "华虹半导体"]
    assert frame["slots"]["metrics"] == ["营业收入"]
    assert frame["requirements"]["need_citation"] is True
    assert frame["output_format"] == "short_report"
    assert query_type_from_frame(frame) == "multi_doc_compare"
    assert secondary_query_types(frame, "multi_doc_compare") == [
        "table_qa",
        "citation_locate",
        "report_generation",
    ]


def test_financial_three_statements_keep_analysis_action() -> None:
    frame = HardSignalExtractor(_registry()).extract("分析下芯导科技的财务三表")

    assert frame["primary_action"] == "analyze"
    assert frame["domain_objects"] == ["financial_three_statements"]
    assert frame["slots"]["companies"] == ["上海芯导电子科技股份有限公司"]
    assert query_type_from_frame(frame) == "table_qa"


def test_flexible_comparison_pattern_is_a_hard_action_signal() -> None:
    frame = HardSignalExtractor(_registry()).extract("中芯国际和华虹半导体谁的收入规模更大？")

    assert frame["primary_action"] == "compare"
    assert frame["slots"]["companies"] == ["中芯国际", "华虹半导体"]
    assert frame["slots"]["metrics"] == ["营业收入"]


def test_embedding_router_returns_top_n_and_margin() -> None:
    router = SemanticSkillRouter(
        _FakeEmbeddingService(),
        {
            "top_k": 3,
            "accept_threshold": 0.5,
            "reject_threshold": 0.2,
            "margin_threshold": 0.1,
            "prototypes": {
                "multi_doc_compare": ["对比两个公司"],
                "table_qa": ["财务指标"],
                "summarization": ["总结文档"],
                "citation_locate": ["定位原文"],
                "report_generation": ["撰写报告"],
                "fact_lookup": ["查询事实"],
            },
        },
    )

    route = asyncio.run(router.route("谁的收入规模更大"))

    assert route.decision == "accept"
    assert route.top_query_type == "multi_doc_compare"
    assert route.candidates[0]["query_type"] == "multi_doc_compare"
    assert len(route.candidates) == 3


def test_question_understanding_preserves_compound_requirements() -> None:
    agent = QuestionUnderstandingAgent(
        llm_service=None,
        semantic_router=_FixedSemanticRouter(),
        company_registry=_registry(),
    )

    result = asyncio.run(
        agent.understand(
            "对比中芯国际和华虹半导体的营收，并给出出处，最后生成一份简短报告。",
            DEFAULT_SKILL_REGISTRY,
            use_llm_intent_slot=False,
        )
    )

    assert result["query_type"] == "multi_doc_compare"
    assert result["secondary_intents"] == ["table_qa", "citation_locate", "report_generation"]
    assert result["need_citation"] is True
    assert result["slots"]["companies"] == ["中芯国际", "华虹半导体"]
    assert result["slots"]["metric"] == "营业收入"
    assert result["slots"]["output_format"] == "short_report"
    assert result["intent_trace"]["primary_action"] == "compare"


def test_answer_contract_preserves_short_report_and_citation_requirements() -> None:
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
