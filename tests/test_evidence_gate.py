import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.agent.evidence_gate import run_evidence_gate


def _mock_evidence(chunk_type: str = "text", score: float = 0.9, doc_source: str = "doc_a.pdf"):
    return [
        {
            "chunk_id": "chunk_1",
            "doc_id": "doc_1",
            "doc_source": doc_source,
            "chunk_type": chunk_type,
            "content": "2025年收入为123亿元。",
            "final_score": score,
            "page_idx": 1,
        }
    ]


class EvidenceGateTestCase(unittest.TestCase):
    def test_clarify_on_missing_slots(self):
        result = run_evidence_gate(
            query_type="table_qa",
            evidence=_mock_evidence("table"),
            missing_slots=["metric", "period"],
            slots={},
            retry_count=0,
        )
        self.assertEqual(result["decision"], "clarify")

    def test_retry_then_refuse_on_low_scores(self):
        retry_result = run_evidence_gate(
            query_type="fact_lookup",
            evidence=_mock_evidence(score=0.1),
            missing_slots=[],
            slots={},
            retry_count=0,
            min_top_score=0.45,
            min_avg_score=0.30,
        )
        self.assertEqual(retry_result["decision"], "retry")

        refuse_result = run_evidence_gate(
            query_type="fact_lookup",
            evidence=_mock_evidence(score=0.1),
            missing_slots=[],
            slots={},
            retry_count=2,
            retry_limit=2,
            min_top_score=0.45,
            min_avg_score=0.30,
        )
        self.assertEqual(refuse_result["decision"], "refuse")

    def test_table_qa_requires_table_evidence(self):
        result = run_evidence_gate(
            query_type="table_qa",
            evidence=_mock_evidence(chunk_type="text", score=0.9),
            missing_slots=[],
            slots={"metric": "收入", "period": "2025"},
            retry_count=0,
            table_evidence_quota=1,
        )
        self.assertEqual(result["decision"], "retry")

    def test_multi_doc_compare_insufficient_docs(self):
        result = run_evidence_gate(
            query_type="multi_doc_compare",
            evidence=_mock_evidence(chunk_type="text", score=0.9, doc_source="only_one.pdf"),
            missing_slots=[],
            slots={"compare_targets": ["A", "B"]},
            retry_count=2,
            retry_limit=2,
        )
        self.assertIn(result["decision"], {"clarify", "refuse"})


if __name__ == "__main__":
    unittest.main()
