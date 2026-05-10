from __future__ import annotations

import asyncio
import unittest

from service.agent.controlled_agents import (
    EvidenceAuditAgent,
    IntentUnderstandingAgent,
    SlotFillingAgent,
    merge_audit_and_rule_gate,
)
from service.agent.skills import SummarizationSkill, TableQASkill


class StructuredIntentLLM:
    def __init__(self) -> None:
        self.structured_calls = 0
        self.complete_calls = 0

    async def structured_json(self, system_prompt, user_payload, schema, max_tokens=512):
        del system_prompt, user_payload, schema, max_tokens
        self.structured_calls += 1
        return {
            "query_type": "citation_locate",
            "matched_keyword_group": "_CITATION_KEYWORDS",
            "intent": "locate_source",
            "reason": "structured",
        }

    async def complete(self, system_prompt, user_prompt, max_tokens=512):
        del system_prompt, user_prompt, max_tokens
        self.complete_calls += 1
        return '{"query_type": "fact_lookup"}'


class CompleteSlotLLM:
    def __init__(self) -> None:
        self.complete_calls = 0

    async def complete(self, system_prompt, user_prompt, max_tokens=512):
        del system_prompt, user_prompt, max_tokens
        self.complete_calls += 1
        return (
            '{"years":["2024"],"metric":"revenue","period":"2024",'
            '"target_statement":"","compare_targets":[],"scope":""}'
        )


class StructuredAuditLLM:
    def __init__(self) -> None:
        self.structured_calls = 0

    async def structured_json(self, system_prompt, user_payload, schema, max_tokens=512):
        del system_prompt, user_payload, schema, max_tokens
        self.structured_calls += 1
        return {
            "semantic_decision": "answer",
            "missing_aspects": [],
            "evidence_coverage": "sufficient",
            "conflict_detected": False,
            "suggested_retry_query": "",
            "reason": "structured_audit",
        }


class MissingQuestionAuditLLM:
    async def structured_json(self, system_prompt, user_payload, schema, max_tokens=512):
        del system_prompt, user_payload, schema, max_tokens
        return {
            "semantic_decision": "refuse",
            "missing_aspects": ["question"],
            "evidence_coverage": "poor",
            "conflict_detected": False,
            "suggested_retry_query": "",
            "reason": "No question was provided, so semantic coverage cannot be determined.",
        }


class ControlledAgentsTestCase(unittest.TestCase):
    def test_intent_agent_maps_to_fixed_query_type_without_confidence(self):
        result = asyncio.run(IntentUnderstandingAgent().classify("Where is this sentence cited? Provide the page source."))

        self.assertEqual(result["query_type"], "citation_locate")
        self.assertEqual(result["matched_keyword_group"], "_CITATION_KEYWORDS")
        self.assertEqual(result["intent"], "locate_source")
        self.assertNotIn("confidence", result)

    def test_intent_agent_prefers_structured_json_path(self):
        llm = StructuredIntentLLM()

        result = asyncio.run(IntentUnderstandingAgent(llm).classify("Where is this sentence cited?"))

        self.assertEqual(result["query_type"], "citation_locate")
        self.assertEqual(llm.structured_calls, 1)
        self.assertEqual(llm.complete_calls, 0)

    def test_slot_agent_returns_fixed_schema_and_full_year(self):
        result = asyncio.run(SlotFillingAgent().fill("What was 2025 revenue?", "table_qa", TableQASkill))

        self.assertEqual(set(result.keys()), {"years", "metric", "period", "target_statement", "compare_targets", "scope", "table_name", "unit", "focus", "__skill_name__", "__missing_required__"})
        self.assertEqual(result["years"], ["2025"])
        self.assertEqual(result["period"], "2025")
        self.assertEqual(result["metric"], "revenue")

    def test_slot_agent_uses_skill_package_metadata_for_missing_slots(self):
        result = asyncio.run(SlotFillingAgent().fill("????2025?????", "summarization", SummarizationSkill))

        self.assertEqual(result["__skill_name__"], "SummarizationSkill")
        self.assertEqual(result["__missing_required__"], [])
    def test_slot_agent_falls_back_to_complete_when_structured_json_missing(self):
        llm = CompleteSlotLLM()

        result = asyncio.run(SlotFillingAgent(llm).fill("Revenue by year?", "table_qa"))

        self.assertIn("2024", result["years"])
        self.assertEqual(result["period"], "2024")
        self.assertGreaterEqual(llm.complete_calls, 1)

    def test_evidence_audit_can_request_retry_for_missing_table_aspect(self):
        audit = asyncio.run(
            EvidenceAuditAgent().audit(
                question="What was 2025 revenue?",
                query_type="table_qa",
                slots={"years": ["2025"], "metric": "revenue", "period": "2025"},
                selected_skill="TableQASkill",
                evidence=[{"content": "2025 net profit was 10.", "chunk_type": "text", "final_score": 0.9}],
                rerank_trace={},
            )
        )

        self.assertEqual(audit["semantic_decision"], "retry")
        self.assertIn("table_evidence", audit["missing_aspects"])
        self.assertEqual(audit["evidence_coverage"], "partial")

    def test_evidence_audit_uses_structured_json_when_available(self):
        llm = StructuredAuditLLM()

        audit = asyncio.run(
            EvidenceAuditAgent(llm).audit(
                question="What was 2025 revenue?",
                query_type="table_qa",
                slots={"years": ["2025"], "metric": "revenue", "period": "2025"},
                selected_skill="TableQASkill",
                evidence=[{"content": "2025 revenue was 10.", "chunk_type": "table", "final_score": 0.9}],
                rerank_trace={},
            )
        )

        self.assertEqual(audit["semantic_decision"], "answer")
        self.assertEqual(audit["reason"], "structured_audit")
        self.assertEqual(llm.structured_calls, 1)

    def test_evidence_audit_ignores_llm_missing_question_when_question_exists(self):
        llm = MissingQuestionAuditLLM()

        audit = asyncio.run(
            EvidenceAuditAgent(llm).audit(
                question="2025年每10股派息多少元？",
                query_type="fact_lookup",
                slots={},
                selected_skill="FactLookupSkill",
                evidence=[{"content": "2025年每10股派息2元。", "chunk_type": "text", "final_score": 0.9}],
                rerank_trace={},
            )
        )

        self.assertEqual(audit["semantic_decision"], "answer")
        self.assertEqual(audit["missing_aspects"], [])

    def test_gate_driven_decision_agent_consumes_retry_reason(self):
        result = asyncio.run(
            EvidenceAuditAgent().decide_from_gate(
                question="What was 2025 revenue?",
                query_type="table_qa",
                slots={"years": ["2025"], "metric": "revenue", "period": "2025"},
                selected_skill="TableQASkill",
                evidence=[{"content": "2025 net profit was 10.", "chunk_type": "text", "final_score": 0.9}],
                rule_gate={"decision": "retry", "reason": "missing_table_evidence", "confidence": 0.4},
                rerank_trace={},
            )
        )

        self.assertEqual(result["decision"], "retry")
        self.assertEqual(result["rule_gate"]["reason"], "missing_table_evidence")
        self.assertIn("table", result["suggested_retry_query"].lower())
        self.assertNotIn("missing_table_evidence", result["suggested_retry_query"])

    def test_gate_driven_decision_agent_prefers_audit_reason_for_refuse(self):
        llm = MissingQuestionAuditLLM()

        result = asyncio.run(
            EvidenceAuditAgent(llm).decide_from_gate(
                question="",
                query_type="fact_lookup",
                slots={},
                selected_skill="FactLookupSkill",
                evidence=[{"content": "2025年每10股派息2元。", "chunk_type": "text", "final_score": 0.9}],
                rule_gate={"decision": "answer", "reason": "evidence_passed", "confidence": 0.8},
                rerank_trace={},
            )
        )

        self.assertEqual(result["decision"], "refuse")
        self.assertIn("no question", result["reason"].lower())

    def test_merge_audit_and_rule_gate_is_conservative(self):
        merged = merge_audit_and_rule_gate(
            {"decision": "answer", "reason": "evidence_passed", "confidence": 0.8},
            {
                "semantic_decision": "retry",
                "missing_aspects": ["metric"],
                "evidence_coverage": "partial",
                "conflict_detected": False,
                "suggested_retry_query": "2025 revenue table",
                "reason": "metric missing",
            },
        )

        self.assertEqual(merged["decision"], "retry")
        self.assertEqual(merged["rule_decision"], "answer")
        self.assertEqual(merged["semantic_decision"], "retry")
        self.assertEqual(merged["suggested_retry_query"], "2025 revenue table")


if __name__ == "__main__":
    unittest.main()

