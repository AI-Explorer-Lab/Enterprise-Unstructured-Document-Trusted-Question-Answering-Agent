from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Mapping

from service.agent.planner_models import ExecutionContext


def _unique(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _page_numbers(item: Mapping[str, Any]) -> List[int]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    values = [
        item.get("page_number"),
        item.get("page_idx"),
        metadata.get("page_number"),
        metadata.get("page_idx"),
        metadata.get("page_range"),
    ]
    pages: List[int] = []
    for value in values:
        for match in re.findall(r"\d+", str(value or "")):
            page = int(match)
            if page not in pages:
                pages.append(page)
    return pages


def _score(item: Mapping[str, Any]) -> float | None:
    for key in ("final_score", "confidence_score", "score", "retrieval_score"):
        try:
            value = item.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _merge_records(
    existing: Iterable[Mapping[str, Any]],
    incoming: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*existing, *incoming]:
        row = dict(item)
        fingerprint = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(row)
    return result


def update_execution_context(
    current: Mapping[str, Any] | None = None,
    *,
    evidence: Iterable[Mapping[str, Any]] | None = None,
    citations: Iterable[Mapping[str, Any]] | None = None,
    tool_name: str | None = None,
    tool_output: Any = None,
) -> Dict[str, Any]:
    context = ExecutionContext.model_validate(dict(current or {}))
    evidence_rows = [dict(item) for item in evidence or [] if isinstance(item, Mapping)]
    citation_rows = [dict(item) for item in citations or [] if isinstance(item, Mapping)]

    document_ids = _unique(
        [
            *context.document_ids,
            *(item.get("doc_id") or item.get("document_id") for item in evidence_rows),
        ]
    )
    chunk_ids = _unique(
        [*context.chunk_ids, *(item.get("chunk_id") for item in evidence_rows)]
    )
    table_ids = _unique(
        [
            *context.table_ids,
            *(
                item.get("table_id")
                or (
                    item.get("metadata", {}).get("table_id")
                    if isinstance(item.get("metadata"), Mapping)
                    else ""
                )
                for item in evidence_rows
            ),
        ]
    )
    page_numbers = list(context.page_numbers)
    for item in [*evidence_rows, *citation_rows]:
        for page in _page_numbers(item):
            if page not in page_numbers:
                page_numbers.append(page)
    merged_evidence = _merge_records(context.evidence, evidence_rows)
    merged_citations = _merge_records(context.citations, citation_rows)
    scores: List[float] = []
    for item in merged_evidence:
        value = _score(item)
        if value is not None:
            scores.append(value)

    tool_outputs = dict(context.tool_outputs)
    if tool_name:
        tool_outputs[str(tool_name)] = tool_output

    return ExecutionContext(
        document_ids=document_ids,
        chunk_ids=chunk_ids,
        page_numbers=page_numbers,
        table_ids=table_ids,
        evidence=merged_evidence,
        citations=merged_citations,
        retrieval_scores=scores,
        tool_outputs=tool_outputs,
    ).model_dump()
