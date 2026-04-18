from __future__ import annotations

from service.agent.answer_generator import AnswerGenerator
from service.evaluation.ragas_evaluator import evaluate_qa_result


def _sample_evidence(chunk_type: str = "text"):
    return [
        {
            "chunk_id": "chunk_1",
            "doc_id": "doc_1",
            "doc_source": "sample.pdf",
            "collection_name": "default",
            "chunk_type": chunk_type,
            "content": "2025\u5e74\u8425\u4e1a\u6536\u5165\u4e3a123\u4ebf\u5143\u3002",
            "final_score": 0.88,
            "page_idx": 3,
            "heading_path": "\u8d22\u52a1\u6570\u636e",
        }
    ]


def test_refuse_answer_uses_readable_chinese_template():
    payload = AnswerGenerator().generate(
        question="\u6211\u8981\u67e5\u8be2\u8d22\u62a5\u4fe1\u606f",
        query_type="fact_lookup",
        evidence=[],
        decision="refuse",
        gate_reason="no_evidence_after_retry",
    )

    assert "?" not in payload["answer"]
    assert "\u672a\u68c0\u7d22\u5230\u8db3\u591f\u7684 PDF \u8bc1\u636e" in payload["answer"]
    assert "collection_name" in payload["answer"]
    assert payload["confidence"] == 0.0


def test_answer_templates_are_readable_for_core_query_types():
    generator = AnswerGenerator()
    cases = [
        ("fact_lookup", _sample_evidence("text"), "\u57fa\u4e8e PDF \u8bc1\u636e"),
        ("table_qa", _sample_evidence("table"), "\u57fa\u4e8e\u8868\u683c\u8bc1\u636e"),
        ("citation_locate", _sample_evidence("text"), "\u8bc1\u636e\u4f4d\u7f6e\u5982\u4e0b"),
        ("summarization", _sample_evidence("text"), "\u6458\u8981"),
        ("multi_doc_compare", _sample_evidence("text"), "\u591a\u6587\u6863\u5bf9\u6bd4\u5982\u4e0b"),
    ]

    for query_type, evidence, expected in cases:
        payload = generator.generate(
            question="\u6d4b\u8bd5\u95ee\u9898",
            query_type=query_type,
            evidence=evidence,
            decision="answer",
        )
        assert "?" not in payload["answer"]
        assert expected in payload["answer"]


def test_refuse_without_evidence_does_not_report_perfect_confidence():
    evaluation = evaluate_qa_result(
        question="\u6211\u8981\u67e5\u8be2\u8d22\u62a5\u4fe1\u606f",
        answer="\u672a\u68c0\u7d22\u5230\u8db3\u591f\u7684 PDF \u8bc1\u636e\uff0c\u65e0\u6cd5\u57fa\u4e8e\u6587\u6863\u53ef\u9760\u56de\u7b54\u3002",
        decision="refuse",
        citations=[],
        evidence=[],
    )

    assert evaluation["evidence_count"] == 0
    assert evaluation["confidence"] == 0.0
    assert evaluation["overall_score"] == 0.0
