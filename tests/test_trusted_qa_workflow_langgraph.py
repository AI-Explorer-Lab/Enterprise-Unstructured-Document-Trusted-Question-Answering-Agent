from __future__ import annotations

import asyncio
import sys
import types
import unittest

# Avoid importing the real session service during workflow module import.
_session_module = types.ModuleType("service.session.session_service")
_session_module.get_session_service = lambda: None
_session_module.SessionService = object
_session_package = types.ModuleType("service.session")
_session_package.get_session_service = _session_module.get_session_service
_session_package.SessionService = _session_module.SessionService
_session_package.session_service = _session_module
sys.modules.setdefault("service.session", _session_package)
sys.modules.setdefault("service.session.session_service", _session_module)

from service.agent.trusted_qa_workflow import TrustedQAWorkflow, _LANGGRAPH_BYPASS


class TrustedQAWorkflowLangGraphTestCase(unittest.TestCase):
    @staticmethod
    def _build_workflow(enabled: bool, available: bool) -> TrustedQAWorkflow:
        workflow = TrustedQAWorkflow.__new__(TrustedQAWorkflow)
        workflow.langgraph_enabled = enabled
        workflow.langgraph_available = available
        workflow.evidence_decision = types.SimpleNamespace(retry_limit=2)
        return workflow

    def test_langgraph_initial_state_normalizes_values(self):
        workflow = self._build_workflow(enabled=True, available=True)

        state = workflow._langgraph_initial_state("q", "default", None, 0, 0, True)

        self.assertEqual(state["top_k"], 1)
        self.assertEqual(state["expand_query_num"], 1)
        self.assertEqual(state["retry_count"], 0)
        self.assertEqual(state["observations"], [])

    def test_graph_nodes_preserve_question_across_state_updates(self):
        workflow = self._build_workflow(enabled=True, available=True)

        class FakeSessionService:
            def load_session(self, session_id, collection_name="default"):
                del session_id, collection_name
                return {"session_id": "sid-1"}

        class FakeIntentAgent:
            async def classify(self, question):
                return {"query_type": "fact_lookup", "reason": question}

        class FakeSkill:
            skill_name = "FactLookupSkill"

            @staticmethod
            def package_metadata():
                return {"skill_name": "FactLookupSkill"}

        class FakeSkillRegistry:
            @staticmethod
            def select_skill(query_type):
                del query_type
                return FakeSkill()

        workflow.session_service = FakeSessionService()
        workflow.intent_agent = FakeIntentAgent()
        workflow.skill_registry = FakeSkillRegistry()

        initial = workflow._langgraph_initial_state("2025年每10股派息多少元？", "shared", None, 5, 3, True)
        after_load = asyncio.run(workflow._graph_load_session(initial))
        after_understand = asyncio.run(workflow._graph_understand_question(after_load))

        self.assertEqual(after_load["question"], "2025年每10股派息多少元？")
        self.assertEqual(after_load["collection_name"], "shared")
        self.assertEqual(after_understand["question"], "2025年每10股派息多少元？")
        self.assertEqual(after_understand["intent_trace"]["reason"], "2025年每10股派息多少元？")
        self.assertEqual(after_understand["sid"], "sid-1")

    def test_maybe_run_langgraph_skips_when_unavailable(self):
        workflow = self._build_workflow(enabled=True, available=False)
        calls = {"count": 0}

        async def fake_ask_with_langgraph(**kwargs):
            del kwargs
            calls["count"] += 1
            return {"decision": "answer"}

        workflow._ask_with_langgraph = fake_ask_with_langgraph
        result = asyncio.run(
            workflow._maybe_run_langgraph(
                question="q",
                collection_name="default",
                session_id=None,
                top_k=5,
                expand_query_num=3,
                enable_cache=True,
            )
        )

        self.assertIsNone(result)
        self.assertEqual(calls["count"], 0)

    def test_maybe_run_langgraph_fallbacks_on_runtime_error(self):
        workflow = self._build_workflow(enabled=True, available=True)

        async def fake_ask_with_langgraph(**kwargs):
            del kwargs
            raise RuntimeError("graph failed")

        workflow._ask_with_langgraph = fake_ask_with_langgraph
        result = asyncio.run(
            workflow._maybe_run_langgraph(
                question="q",
                collection_name="default",
                session_id=None,
                top_k=5,
                expand_query_num=3,
                enable_cache=True,
            )
        )

        self.assertIsNone(result)

    def test_maybe_run_langgraph_respects_bypass_context(self):
        workflow = self._build_workflow(enabled=True, available=True)
        calls = {"count": 0}

        async def fake_ask_with_langgraph(**kwargs):
            del kwargs
            calls["count"] += 1
            return {"decision": "answer"}

        workflow._ask_with_langgraph = fake_ask_with_langgraph
        token = _LANGGRAPH_BYPASS.set(True)
        try:
            result = asyncio.run(
                workflow._maybe_run_langgraph(
                    question="q",
                    collection_name="default",
                    session_id=None,
                    top_k=5,
                    expand_query_num=3,
                    enable_cache=True,
                )
            )
        finally:
            _LANGGRAPH_BYPASS.reset(token)

        self.assertIsNone(result)
        self.assertEqual(calls["count"], 0)

    def test_graph_route_after_clarify_gate_branches_to_clarify(self):
        workflow = self._build_workflow(enabled=True, available=True)

        route = workflow._graph_route_after_clarify_gate({"clarify": {"decision": "clarify"}})

        self.assertEqual(route, "clarify")

    def test_graph_route_after_clarify_gate_branches_to_retrieve(self):
        workflow = self._build_workflow(enabled=True, available=True)

        route = workflow._graph_route_after_clarify_gate({"clarify": {"decision": "answer"}})

        self.assertEqual(route, "retrieve")

    def test_graph_route_after_gate_branches_to_retry(self):
        workflow = self._build_workflow(enabled=True, available=True)

        route = workflow._graph_route_after_gate({"gate": {"decision": "retry"}, "retry_count": 0})

        self.assertEqual(route, "retry")

    def test_graph_route_after_gate_branches_to_final(self):
        workflow = self._build_workflow(enabled=True, available=True)

        route = workflow._graph_route_after_gate({"gate": {"decision": "answer"}, "retry_count": 0})

        self.assertEqual(route, "final")

    def test_graph_route_after_retry_branches_to_final_after_limit(self):
        workflow = self._build_workflow(enabled=True, available=True)

        route = workflow._graph_route_after_retry({"gate": {"decision": "retry"}, "retry_count": 2})

        self.assertEqual(route, "final")

    def test_retry_retrieval_uses_llm_expanded_queries(self):
        workflow = self._build_workflow(enabled=True, available=True)

        class FakeLLM:
            def __init__(self):
                self.calls = []

            async def expand_queries(self, question, query_type, expand_query_num):
                self.calls.append((question, query_type, expand_query_num))
                return [question, f"{question} table values"]

        class FakeRetriever:
            def __init__(self):
                self.calls = []

            async def retrieve(self, **kwargs):
                self.calls.append(kwargs)
                return {
                    "evidence": [{"content": "retry evidence", "chunk_type": "table", "final_score": 0.9}],
                    "rerank_trace": {},
                    "retrieval_trace": {"query_variants": kwargs.get("expanded_queries") or []},
                }

        class FakeEvidenceDecision:
            retry_limit = 2

            async def evaluate(self, **kwargs):
                del kwargs
                return {"decision": "refuse", "reason": "low_score"}

        fake_llm = FakeLLM()
        fake_retriever = FakeRetriever()
        workflow.llm_service = fake_llm
        workflow.retriever = fake_retriever
        workflow.evidence_decision = FakeEvidenceDecision()
        workflow.table_evidence_quota = 2

        result = asyncio.run(
            workflow._graph_retry_retrieval(
                {
                    "question": "What was 2025 revenue?",
                    "collection_name": "finance",
                    "top_k": 5,
                    "expand_query_num": 3,
                    "query_type": "table_qa",
                    "retry_count": 0,
                    "gate": {"decision": "retry", "suggested_retry_query": "2025 revenue table"},
                    "slots": {"metric": "revenue", "period": "2025"},
                    "observations": [],
                    "selected_skill": types.SimpleNamespace(skill_name="TableQASkill"),
                }
            )
        )

        self.assertEqual(fake_llm.calls, [("2025 revenue table", "table_qa", 4)])
        self.assertEqual(len(fake_retriever.calls[0]["expanded_queries"]), 4)
        self.assertEqual(fake_retriever.calls[0]["expanded_queries"][:2], ["2025 revenue table", "2025 revenue table table values"])
        self.assertIn("指标 数值 单位 表头", fake_retriever.calls[0]["expanded_queries"][-1])
        self.assertEqual(result["expanded"], fake_retriever.calls[0]["expanded_queries"])
        self.assertTrue(result["llm_expansion_used"])

    def test_ask_prefers_langgraph_response(self):
        workflow = self._build_workflow(enabled=True, available=True)

        async def fake_maybe_run_langgraph(**kwargs):
            del kwargs
            return {"decision": "answer", "answer": "ok"}

        async def fake_legacy(**kwargs):
            del kwargs
            raise AssertionError("legacy should not run")

        workflow._maybe_run_langgraph = fake_maybe_run_langgraph
        workflow._ask_legacy = fake_legacy
        result = asyncio.run(workflow.ask("hello"))

        self.assertEqual(result["decision"], "answer")
        self.assertEqual(result["answer"], "ok")

    def test_ask_falls_back_to_legacy_when_graph_returns_none(self):
        workflow = self._build_workflow(enabled=True, available=True)

        async def fake_maybe_run_langgraph(**kwargs):
            del kwargs
            return None

        async def fake_legacy(**kwargs):
            del kwargs
            return {"decision": "refuse", "answer": "legacy"}

        workflow._maybe_run_langgraph = fake_maybe_run_langgraph
        workflow._ask_legacy = fake_legacy
        result = asyncio.run(workflow.ask("hello"))

        self.assertEqual(result["decision"], "refuse")
        self.assertEqual(result["answer"], "legacy")


if __name__ == "__main__":
    unittest.main()
