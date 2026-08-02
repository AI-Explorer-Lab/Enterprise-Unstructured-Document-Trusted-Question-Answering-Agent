from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TRUSTED_QA_ENABLE_REAL_LLM"] = "0"

import httpx

from main import app


def _final_sse_payload(text: str) -> dict:
    for block in text.split("\n\n"):
        lines = block.splitlines()
        event_name = next(
            (line.split(":", 1)[1].strip() for line in lines if line.startswith("event:")),
            "",
        )
        data_text = next(
            (line.split(":", 1)[1].strip() for line in lines if line.startswith("data:")),
            "",
        )
        if event_name == "error" and data_text:
            payload = json.loads(data_text)
            raise AssertionError(payload)
        if event_name == "final" and data_text:
            payload = json.loads(data_text)
            assert isinstance(payload, dict)
            return payload
    raise AssertionError("SSE response did not contain a final event")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "acceptance.pdf"
        pdf.write_bytes(b"%PDF-1.4\n% minimal acceptance pdf\n")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://trusted-qa.local",
            timeout=240.0,
        ) as client:
            index_resp = await client.post(
                "/documents/index",
                json={
                    "pdf_path": str(pdf),
                    "collection_name": "e2e",
                    "company_id": "xindao",
                    "year": 2025,
                    "force_rebuild": True,
                },
            )
            assert index_resp.status_code == 200, index_resp.text
            ask_resp = await client.post(
                "/qa/ask/stream",
                json={
                    "question": "acceptance document parsed",
                    "collection_name": "e2e",
                    "include_debug": True,
                },
            )
        assert ask_resp.status_code == 200, ask_resp.text
        payload = {"index": index_resp.json(), "ask": _final_sse_payload(ask_resp.text)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        assert payload["ask"]["decision"] in {"answer", "refuse", "clarify"}
        assert "retrieval_trace" in payload["ask"]


if __name__ == "__main__":
    asyncio.run(main())
