"""Query type definitions used by classifier, skills, and response schema."""

from __future__ import annotations

from enum import Enum


class QueryType(str, Enum):
    INFORMATION_EXTRACTION = "information_extraction"
    METRIC_CALCULATION = "metric_calculation"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    AMBIGUOUS_QUERY = "ambiguous_query"


SUPPORTED_INTENT_TYPES = tuple(
    query_type.value
    for query_type in QueryType
    if query_type is not QueryType.AMBIGUOUS_QUERY
)
SUPPORTED_QUERY_TYPES = tuple(query_type.value for query_type in QueryType)
