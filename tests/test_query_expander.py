from __future__ import annotations

from service.agent.query_expander import FIXED_QUERY_VARIANT_TOTAL, expand_queries


def test_expand_queries_returns_original_plus_three_fixed_variants() -> None:
    query = "\u8bf7\u95ee 2025 \u5e74\u8425\u4e1a\u6536\u5165\u662f\u591a\u5c11\uff1f"
    queries = expand_queries(query, "table_qa", expand_query_num=99)

    assert len(queries) == FIXED_QUERY_VARIANT_TOTAL
    assert queries[0] == query
    assert "\u8bf7\u95ee" not in queries[1]
    assert "\u6307\u6807 \u6570\u503c \u5355\u4f4d \u8868\u5934" in queries[3]


def test_citation_locate_expansion_does_not_add_page_by_default() -> None:
    query = "\u5b9a\u4f4d\u8463\u4e8b\u4f1a\u62a5\u544a\u4e2d\u7684\u98ce\u9669\u63d0\u793a\u539f\u6587"
    queries = expand_queries(query, "citation_locate")

    assert len(queries) == FIXED_QUERY_VARIANT_TOTAL
    assert "\u539f\u6587\u51fa\u5904 \u6807\u9898\u8def\u5f84 \u7ae0\u8282 \u539f\u6587\u7247\u6bb5" in queries[3]
    assert all("\u9875\u7801" not in item for item in queries)
