from __future__ import annotations

from service.agent.company_registry import CompanyRegistry
from service.agent.evidence_gate import EvidenceGate
from service.agent.query_planner import build_query_plan
from service.agent.retrieval_scope import resolve_retrieval_scope


def _registry() -> CompanyRegistry:
    return CompanyRegistry(
        [
            {
                "company_id": "xindao",
                "company_name": "上海芯导电子科技股份有限公司",
                "aliases": ["芯导", "芯导科技"],
            }
        ]
    )


def _xindao_scopes():
    return [
        {
            "company_id": "xindao",
            "company_name": "上海芯导电子科技股份有限公司",
            "company_aliases": ["芯导", "芯导科技"],
            "year": 2025,
        }
    ]


def test_scope_gate_refuses_company_not_present_in_collection() -> None:
    scope = resolve_retrieval_scope(
        question="对比中芯国际和华虹半导体的营收",
        query_type="comparison",
        slots={
            "companies": ["中芯国际", "华虹半导体"],
            "compare_targets": ["中芯国际", "华虹半导体"],
            "metric": "营业收入",
        },
        conversation_focus=None,
        document_scopes=_xindao_scopes(),
        company_registry=_registry(),
    )

    assert scope.should_refuse is True
    assert scope.should_clarify is False
    assert scope.refuse_reason == "unsupported_by_data"
    assert scope.unsupported_companies == ["中芯国际", "华虹半导体"]
    assert "芯导" in scope.refuse_message


def test_scope_gate_does_not_silently_replace_unknown_company_with_xindao() -> None:
    scope = resolve_retrieval_scope(
        question="中芯国际2024年营业收入是多少？",
        query_type="information_extraction",
        slots={
            "companies": ["中芯国际"],
            "years": ["2024"],
            "metric": "营业收入",
        },
        conversation_focus=None,
        document_scopes=_xindao_scopes(),
        company_registry=_registry(),
    )

    assert scope.should_refuse is True
    assert scope.company_id == ""
    assert scope.unsupported_companies == ["中芯国际"]
    assert scope.metadata_filter() == {"year": [2024]}


def test_scope_gate_accepts_xindao_and_uses_only_available_year() -> None:
    scope = resolve_retrieval_scope(
        question="分析芯导科技的财务三表",
        query_type="analysis",
        slots={
            "companies": ["上海芯导电子科技股份有限公司"],
            "domain_objects": ["financial_three_statements"],
        },
        conversation_focus=None,
        document_scopes=_xindao_scopes(),
        company_registry=_registry(),
    )

    assert scope.should_refuse is False
    assert scope.should_clarify is False
    assert scope.company_id == "xindao"
    assert scope.years == [2025]


def test_future_supported_multi_company_compare_keeps_both_companies_in_scope() -> None:
    scopes = [
        *_xindao_scopes(),
        {
            "company_id": "huahong",
            "company_name": "华虹半导体有限公司",
            "company_aliases": ["华虹半导体", "华虹"],
            "year": 2025,
        },
    ]
    scope = resolve_retrieval_scope(
        question="对比芯导科技和华虹半导体2025年的营业收入",
        query_type="comparison",
        slots={
            "companies": ["上海芯导电子科技股份有限公司", "华虹半导体"],
            "compare_targets": ["芯导科技", "华虹半导体"],
            "years": ["2025"],
            "metric": "营业收入",
        },
        conversation_focus=None,
        document_scopes=scopes,
        company_registry=_registry(),
    )

    assert scope.should_refuse is False
    assert scope.should_clarify is False
    assert scope.company_id == ""
    assert scope.years == [2025]
    assert scope.metadata_filter() == {"year": [2025]}


def test_three_statement_gate_requires_evidence_for_each_subtask() -> None:
    plan = build_query_plan("分析芯导科技的财务三表", "analysis")
    slots = {"query_plan": plan, "evidence_modes": ["table"]}
    evidence = [
        {
            "chunk_id": "balance",
            "doc_id": "xindao-2025",
            "chunk_type": "table",
            "retrieval_subtask_id": "balance_sheet",
            "score": 0.9,
        },
        {
            "chunk_id": "income",
            "doc_id": "xindao-2025",
            "chunk_type": "table",
            "retrieval_subtask_id": "income_statement",
            "score": 0.9,
        },
    ]

    first = EvidenceGate(retry_limit=1).evaluate(
        evidence,
        query_type="analysis",
        retry_count=0,
        slots=slots,
    )
    final = EvidenceGate(retry_limit=1).evaluate(
        evidence,
        query_type="analysis",
        retry_count=1,
        slots=slots,
    )

    assert first["decision"] == "retry"
    assert first["reason"] == "missing_subtask_evidence"
    assert first["missing_subtasks"] == [
        {"subtask_id": "cash_flow_statement", "display_name": "现金流量表"}
    ]
    assert final["decision"] == "refuse"
    assert final["reason"] == "missing_subtask_evidence_after_retry"


def test_three_statement_gate_answers_when_all_subtasks_are_covered() -> None:
    plan = build_query_plan("分析芯导科技的财务三表", "analysis")
    evidence = [
        {
            "chunk_id": subtask["slot"],
            "doc_id": "xindao-2025",
            "chunk_type": "table",
            "retrieval_subtask_id": subtask["slot"],
            "score": 0.9,
        }
        for subtask in plan["subtasks"]
    ]

    result = EvidenceGate(retry_limit=1).evaluate(
        evidence,
        query_type="analysis",
        retry_count=0,
        slots={"query_plan": plan, "evidence_modes": ["table"]},
    )

    assert result["decision"] == "answer"
    assert result["missing_subtasks"] == []


def test_table_gate_does_not_accept_plain_text_as_financial_table_evidence() -> None:
    result = EvidenceGate(retry_limit=0).evaluate(
        [
            {
                "chunk_id": "plain-text",
                "doc_id": "xindao-2025",
                "chunk_type": "text",
                "score": 0.9,
            }
        ],
        query_type="information_extraction",
        retry_count=0,
        slots={"metric": "营业收入", "period": "2025"},
    )

    assert result["decision"] == "refuse"
    assert result["reason"] == "missing_table_evidence_after_retry"


def test_year_comparison_gate_requires_both_periods_in_evidence() -> None:
    result = EvidenceGate(retry_limit=0).evaluate(
        [
            {
                "chunk_id": "only-one-document",
                "doc_id": "xindao-2025",
                "chunk_type": "table",
                "score": 0.9,
            }
        ],
        query_type="comparison",
        retry_count=0,
        slots={"compare_targets": ["2024", "2025"]},
    )

    assert result["decision"] == "refuse"
    assert result["reason"] == "missing_year_evidence_after_retry"


def test_year_comparison_can_use_one_table_that_contains_both_periods() -> None:
    result = EvidenceGate(retry_limit=0).evaluate(
        [
            {
                "chunk_id": "two-year-table",
                "doc_id": "xindao-2025",
                "chunk_type": "table",
                "content": "项目 2025年 2024年\n营业收入 120 100",
                "score": 0.9,
            }
        ],
        query_type="comparison",
        retry_count=0,
        slots={"compare_targets": ["2024", "2025"], "metric": "营业收入"},
    )

    assert result["decision"] == "answer"
