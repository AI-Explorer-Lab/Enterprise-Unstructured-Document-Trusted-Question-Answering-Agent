from __future__ import annotations

import json

from fastapi.testclient import TestClient

from controller.apis import qa_controller
from main import app


def _events(text: str) -> list[tuple[str, dict]]:
    result: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        event_name = ""
        data_text = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_text = line.split(":", 1)[1].strip()
        if event_name and data_text:
            result.append((event_name, json.loads(data_text)))
    return result


def test_only_streaming_qa_route_is_registered(monkeypatch) -> None:
    async def validate(_request):
        return "collection"

    async def run(_request, _collection_name, progress_callback=None):
        if progress_callback is not None:
            await progress_callback(
                {
                    "phase": "load_session",
                    "stage": "load_session",
                    "status": "completed",
                    "session_id": "session-1",
                }
            )
        return {
            "answer": "answer",
            "decision": "answer",
            "query_type": "information_extraction",
            "confidence": 1.0,
            "session_id": "session-1",
            "citations": [],
            "evidence": [],
            "retrieval_trace": {"workflow_runner": "agno"},
        }

    async def update_metadata(*_args, **_kwargs):
        return None

    monkeypatch.setattr(qa_controller, "_validate_qa_request", validate)
    monkeypatch.setattr(qa_controller, "_run_qa", run)
    monkeypatch.setattr(
        qa_controller,
        "get_session_service",
        lambda: type(
            "SessionStub",
            (),
            {"update_session_metadata": staticmethod(update_metadata)},
        )(),
    )
    client = TestClient(app)

    removed = client.post(
        "/qa/ask",
        json={"question": "q", "collection_name": "collection"},
    )
    streamed = client.post(
        "/qa/ask/stream",
        json={
            "question": "q",
            "collection_name": "collection",
            "include_debug": True,
        },
    )

    assert removed.status_code in {404, 405}
    assert streamed.status_code == 200
    events = _events(streamed.text)
    assert [name for name, _payload in events] == ["status", "status", "final"]
    assert events[-1][1]["retrieval_trace"]["workflow_runner"] == "agno"


def test_streaming_qa_route_surfaces_execution_errors(monkeypatch) -> None:
    async def validate(_request):
        return "collection"

    async def run(_request, _collection_name, progress_callback=None):
        del progress_callback
        raise RuntimeError("agno step failed")

    monkeypatch.setattr(qa_controller, "_validate_qa_request", validate)
    monkeypatch.setattr(qa_controller, "_run_qa", run)
    client = TestClient(app)

    response = client.post(
        "/qa/ask/stream",
        json={"question": "q", "collection_name": "collection"},
    )

    assert response.status_code == 200
    events = _events(response.text)
    assert events == [
        (
            "error",
            {
                "code": "INTERNAL_ERROR",
                "message": "agno step failed",
                "error_type": "RuntimeError",
            },
        )
    ]
