from __future__ import annotations

from service.retrieval.two_stage_hybrid_reranker import TwoStageHybridReranker


def test_two_stage_reranker_weights_table_quota_and_trace() -> None:
    reranker = TwoStageHybridReranker(cross_encoder_enabled=False)

    candidates = [
        {
            "chunk_id": "table-a",
            "chunk_type": "table",
            "raw_doc": "营业收入 100亿元，毛利率 22%",
            "dense_score": 0.52,
            "bm25_score": 0.95,
            "table_header_text": "指标,2024,同比",
            "table_context_text": "单位: 亿元",
            "heading_path": "第四章 财务指标",
            "doc_source": "annual_report.pdf",
            "page_idx": 4,
            "chunk_index": 2,
        },
        {
            "chunk_id": "table-b",
            "chunk_type": "table",
            "raw_doc": "净利润 18亿元，同比增长 12%",
            "dense_score": 0.40,
            "bm25_score": 0.80,
            "table_header_text": "指标,2024,同比",
            "table_context_text": "单位: 亿元",
            "heading_path": "第四章 财务指标",
            "doc_source": "annual_report.pdf",
            "page_idx": 4,
            "chunk_index": 3,
        },
        {
            "chunk_id": "text-a",
            "chunk_type": "text",
            "raw_doc": "公司在报告期内持续推进企业服务业务，收入稳定增长。",
            "dense_score": 0.92,
            "bm25_score": 0.10,
            "heading_path": "第三章 经营情况",
            "doc_source": "annual_report.pdf",
            "page_idx": 3,
            "chunk_index": 1,
        },
        {
            "chunk_id": "text-b",
            "chunk_type": "text",
            "raw_doc": "公司推进组织优化和成本管理，期间费用率下降。",
            "dense_score": 0.86,
            "bm25_score": 0.18,
            "heading_path": "第三章 经营情况",
            "doc_source": "annual_report.pdf",
            "page_idx": 3,
            "chunk_index": 2,
        },
    ]

    ranked, trace = reranker.rerank(
        query="2024年营业收入同比指标在表里是多少",
        candidates=candidates,
        top_k=3,
        query_type="table_qa",
        table_evidence_quota=2,
    )

    assert len(ranked) == 3
    assert trace["weights"]["dense_weight"] == 0.50
    assert trace["weights"]["bm25_weight"] == 0.35
    assert trace["weights"]["metadata_boost_weight"] == 0.10
    assert trace["weights"]["table_boost_weight"] == 0.05

    table_count = sum(1 for row in ranked if row["chunk_type"] == "table")
    assert table_count >= 2
    assert trace["table_evidence_selected"] >= 2

    assert ranked[0]["final_score"] >= ranked[-1]["final_score"]
    assert trace["top"]
    assert trace["cross_encoder"]["status"] == "disabled"


def test_two_stage_reranker_near_duplicate_filter_and_neighbor_supplement() -> None:
    reranker = TwoStageHybridReranker(near_duplicate_threshold=0.90, cross_encoder_enabled=False)

    candidates = [
        {
            "chunk_id": "a-main",
            "chunk_type": "text",
            "raw_doc": "产品参数包括额定功率、电压范围、支持协议和安装条件。",
            "dense_score": 0.95,
            "bm25_score": 0.70,
            "doc_source": "manual.pdf",
            "page_idx": 10,
            "chunk_index": 5,
            "heading_path": "第二节 规格参数",
        },
        {
            "chunk_id": "a-dup",
            "chunk_type": "text",
            "raw_doc": "产品参数包括额定功率、电压范围、支持协议和安装条件",
            "dense_score": 0.93,
            "bm25_score": 0.72,
            "doc_source": "manual.pdf",
            "page_idx": 10,
            "chunk_index": 6,
            "heading_path": "第二节 规格参数",
        },
        {
            "chunk_id": "b-high",
            "chunk_type": "text",
            "raw_doc": "系统支持在线升级与异常告警，兼容企业统一监控平台。",
            "dense_score": 0.92,
            "bm25_score": 0.65,
            "doc_source": "manual.pdf",
            "page_idx": 12,
            "chunk_index": 1,
            "heading_path": "第三节 系统能力",
        },
        {
            "chunk_id": "c-high",
            "chunk_type": "text",
            "raw_doc": "维护策略包含巡检周期、备件策略和故障恢复流程。",
            "dense_score": 0.91,
            "bm25_score": 0.64,
            "doc_source": "manual.pdf",
            "page_idx": 14,
            "chunk_index": 1,
            "heading_path": "第四节 运维",
        },
        {
            "chunk_id": "a-neighbor",
            "chunk_type": "text",
            "raw_doc": "补充参数: 工作温度、湿度范围与防护等级。",
            "dense_score": 0.10,
            "bm25_score": 0.05,
            "doc_source": "manual.pdf",
            "page_idx": 10,
            "chunk_index": 4,
            "heading_path": "第二节 规格参数",
        },
    ]

    ranked, trace = reranker.rerank(
        query="产品参数有哪些",
        candidates=candidates,
        top_k=3,
        query_type="fact_lookup",
    )

    ids = {row["chunk_id"] for row in ranked}
    assert "a-dup" not in ids
    assert trace["after_near_duplicate"] <= len(candidates) - 1
    assert trace["neighbor_supplemented"] >= 1


def test_two_stage_reranker_applies_cross_encoder_final_order() -> None:
    class FakeCrossEncoderScorer:
        def score(self, query, texts):
            del query
            scores = [10.0 if "关键答案" in text else 1.0 for text in texts]
            return scores, {"status": "applied", "model": "fake-cross-encoder"}

    reranker = TwoStageHybridReranker(
        cross_encoder_enabled=True,
        cross_encoder_candidate_pool=3,
        cross_encoder_scorer=FakeCrossEncoderScorer(),
    )
    candidates = [
        {
            "chunk_id": "dense-top",
            "chunk_type": "text",
            "raw_doc": "普通背景材料",
            "dense_score": 0.99,
            "bm25_score": 0.90,
        },
        {
            "chunk_id": "ce-top",
            "chunk_type": "text",
            "raw_doc": "这里包含关键答案",
            "dense_score": 0.20,
            "bm25_score": 0.10,
            "heading_path": "第一章 关键结论",
        },
    ]

    ranked, trace = reranker.rerank(
        query="关键答案是什么",
        candidates=candidates,
        top_k=1,
        query_type="fact_lookup",
    )

    assert ranked[0]["chunk_id"] == "ce-top"
    assert ranked[0]["cross_encoder_score"] == 10.0
    assert ranked[0]["light_final_score"] >= 0.0
    assert trace["cross_encoder"]["status"] == "applied"


def test_two_stage_reranker_falls_back_when_cross_encoder_unavailable() -> None:
    reranker = TwoStageHybridReranker(
        cross_encoder_enabled=True,
        cross_encoder_model="missing-model",
        cross_encoder_candidate_pool=2,
    )
    reranker._cross_encoder_load_failed = "transformers unavailable"

    ranked, trace = reranker.rerank(
        query="产品参数",
        candidates=[
            {"chunk_id": "a", "raw_doc": "产品参数", "dense_score": 0.9, "bm25_score": 0.1},
            {"chunk_id": "b", "raw_doc": "其他内容", "dense_score": 0.1, "bm25_score": 0.9},
        ],
        top_k=1,
        query_type="fact_lookup",
    )

    assert len(ranked) == 1
    assert "cross_encoder_score" not in ranked[0]
    assert trace["cross_encoder"]["status"] == "fallback"
