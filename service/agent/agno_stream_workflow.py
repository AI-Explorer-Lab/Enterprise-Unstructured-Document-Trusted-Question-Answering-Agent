from __future__ import annotations

from importlib.metadata import version
from typing import Any, Awaitable, Callable, Dict
from uuid import uuid4

from agno.workflow import Workflow

from service.agent.agno_event_adapter import adapt_agno_event
from service.agent.agno_stream_steps import build_agno_steps


ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class AgnoWorkflowExecutionError(RuntimeError):
    pass


class AgnoStreamWorkflow:
    def __init__(self, service: Any) -> None:
        self.service = service
        self.agno_version = version("agno")

    def _workflow(self) -> tuple[Workflow, Dict[str, str]]:
        error_sink: Dict[str, str] = {}
        workflow = Workflow(
            name="trusted_financial_qa",
            steps=build_agno_steps(self.service, error_sink),
            telemetry=False,
            stream=True,
            stream_events=True,
            stream_executor_events=True,
            store_events=False,
            store_executor_outputs=True,
        )
        return workflow, error_sink

    async def run(
        self,
        *,
        question: str,
        collection_name: str,
        session_id: str | None,
        top_k: int,
        expand_query_num: int,
        enable_cache: bool,
        use_llm_intent_slot: bool,
        progress_callback: ProgressCallback | None,
    ) -> Dict[str, Any]:
        run_id = str(uuid4())
        initial_state = self.service._initial_workflow_state(
            question=question,
            collection_name=collection_name,
            session_id=session_id,
            top_k=top_k,
            expand_query_num=expand_query_num,
            enable_cache=enable_cache,
            use_llm_intent_slot=use_llm_intent_slot,
        )
        initial_state["workflow_run_id"] = run_id
        initial_state["agno_version"] = self.agno_version
        result_state: Dict[str, Any] | None = None
        error: str = ""
        workflow, error_sink = self._workflow()
        events = workflow.arun(
            input=initial_state,
            run_id=run_id,
            session_id=session_id or run_id,
            stream=True,
            stream_events=True,
        )
        async for event in events:
            progress = adapt_agno_event(event)
            if progress is not None and progress_callback is not None:
                await progress_callback(progress)
            event_name = str(getattr(event, "event", "") or type(event).__name__)
            if event_name == "WorkflowCompleted":
                content = getattr(event, "content", None)
                if isinstance(content, dict):
                    result_state = content
            elif event_name in {"StepError", "WorkflowError", "WorkflowCancelled"}:
                error = str(getattr(event, "error", "") or event_name)
        if error_sink.get("error"):
            raise AgnoWorkflowExecutionError(error_sink["error"])
        if error:
            raise AgnoWorkflowExecutionError(error)
        if not isinstance(result_state, dict):
            raise AgnoWorkflowExecutionError(
                "Agno workflow completed without dictionary state"
            )
        response = result_state.get("response")
        if not isinstance(response, dict):
            raise AgnoWorkflowExecutionError(
                "Agno workflow completed without a QA response"
            )
        return response
