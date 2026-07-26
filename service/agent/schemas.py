from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


PRIMARY_INTENT_TYPES = (
    "information_extraction",
    "metric_calculation",
    "comparison",
    "analysis",
    "summarization",
)

QUERY_TYPES = (
    *PRIMARY_INTENT_TYPES,
    "ambiguous_query",
)

PRIMARY_INTENT_TYPE_SET = set(PRIMARY_INTENT_TYPES)
QUERY_TYPE_SET = set(QUERY_TYPES)

LEGACY_QUERY_TYPE_ALIASES = {
    "fact_lookup": "information_extraction",
    "table_qa": "information_extraction",
    "citation_locate": "information_extraction",
    "report_generation": "summarization",
    "multi_doc_compare": "comparison",
}

FINAL_DECISIONS = {"answer", "clarify", "refuse"}


class IntentCandidate(BaseModel):
    intent_id: str
    score: float
    matched_prototype_count: int = 0


class IntentRoutingResult(BaseModel):
    top_intent: str | None = None
    candidates: List[IntentCandidate] = Field(default_factory=list)
    top1_score: float = 0.0
    score_margin: float = 0.0
    route_status: Literal["accepted", "ambiguous", "unknown", "disabled", "error"]
    provider: str = "unknown"


def normalize_query_type(query_type: str | None) -> str:
    value = str(query_type or "").strip().lower()
    value = LEGACY_QUERY_TYPE_ALIASES.get(value, value)
    if value in QUERY_TYPE_SET:
        return value
    return "information_extraction"
