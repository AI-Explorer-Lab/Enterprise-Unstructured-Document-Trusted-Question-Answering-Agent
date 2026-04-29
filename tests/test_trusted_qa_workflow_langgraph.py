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

    def test_graph_route_after_slot_fill_branches_to_clarify(self):
        workflow = self._build_workflow(enabled=True, available=True)

        route = workflow._graph_route_after_slot_fill({"clarify": {"decision": "clarify"}})

        self.assertEqual(route, "clarify")

    def test_graph_route_after_slot_fill_branches_to_retrieve(self):
        workflow = self._build_workflow(enabled=True, available=True)

        route = workflow._graph_route_after_slot_fill({"clarify": {"decision": "answer"}})

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
