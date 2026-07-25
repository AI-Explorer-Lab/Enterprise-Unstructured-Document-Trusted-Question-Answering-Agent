from __future__ import annotations

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


def normalize_query_type(query_type: str | None) -> str:
    value = str(query_type or "").strip().lower()
    value = LEGACY_QUERY_TYPE_ALIASES.get(value, value)
    if value in QUERY_TYPE_SET:
        return value
    return "information_extraction"
