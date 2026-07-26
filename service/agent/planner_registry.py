from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Type

from pydantic import BaseModel

from service.agent.planner_models import PLAN_MODELS, NumericCondition


ARGUMENT_TYPE_NAMES: Dict[str, str] = {
    "question": "string",
    "companies": "list[string]",
    "periods": "list[string]",
    "quarters": "list[string]",
    "half_years": "list[string]",
    "metrics": "list[string]",
    "report_types": "list[string]",
    "statement_types": "list[string]",
    "requested_pages": "list[integer]",
    "document_references": "list[string]",
    "document_name": "string",
    "numeric_conditions": "list[object]",
    "compare_targets": "list[string]",
    "scope": "string",
    "evidence_modes": "list[string]",
    "top_k": "integer",
    "need_citation": "boolean",
    "need_location": "boolean",
    "output_format": "string",
}


def valid_argument_value(name: str, value: Any) -> bool:
    expected = ARGUMENT_TYPE_NAMES.get(name)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "list[string]":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if expected == "list[integer]":
        return (
            isinstance(value, list)
            and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        )
    if expected == "list[object]":
        if not isinstance(value, list):
            return False
        try:
            for item in value:
                NumericCondition.model_validate(item)
        except Exception:
            return False
        return True
    return False


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    task_types: tuple[str, ...]
    allowed_arguments: tuple[str, ...] = tuple()
    required_arguments: tuple[str, ...] = tuple()

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "task_types": list(self.task_types),
            "allowed_arguments": list(self.allowed_arguments),
            "required_arguments": list(self.required_arguments),
            "argument_schema": {
                name: ARGUMENT_TYPE_NAMES[name]
                for name in self.allowed_arguments
                if name in ARGUMENT_TYPE_NAMES
            },
        }


_COMMON_RETRIEVAL_ARGUMENTS = (
    "companies",
    "periods",
    "quarters",
    "half_years",
    "metrics",
    "report_types",
    "statement_types",
    "requested_pages",
    "document_references",
    "document_name",
    "numeric_conditions",
    "compare_targets",
    "scope",
    "evidence_modes",
)

TOOL_DEFINITIONS: Dict[str, ToolDefinition] = {
    "query_expander": ToolDefinition("query_expander", ("search",), ("question",)),
    "parallel_hybrid_retrieval": ToolDefinition(
        "parallel_hybrid_retrieval",
        ("retrieve",),
        _COMMON_RETRIEVAL_ARGUMENTS,
    ),
    "table_prioritized_retrieval": ToolDefinition(
        "table_prioritized_retrieval",
        ("search",),
        _COMMON_RETRIEVAL_ARGUMENTS,
    ),
    "two_stage_hybrid_rerank": ToolDefinition(
        "two_stage_hybrid_rerank",
        ("search",),
        ("top_k",),
    ),
    "evidence_gate": ToolDefinition(
        "evidence_gate",
        ("locate",),
        ("need_citation", "need_location"),
    ),
    "answer_generator": ToolDefinition(
        "answer_generator",
        ("answer", "calculate", "compare", "analyze", "summarize", "generate_report"),
        ("output_format", "need_citation", "need_location"),
    ),
}

INTENT_TOOLS: Dict[str, tuple[str, ...]] = {
    "information_extraction": (
        "query_expander",
        "parallel_hybrid_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    "metric_calculation": (
        "query_expander",
        "parallel_hybrid_retrieval",
        "table_prioritized_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    "comparison": (
        "query_expander",
        "parallel_hybrid_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    "analysis": (
        "query_expander",
        "parallel_hybrid_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
    "summarization": (
        "query_expander",
        "parallel_hybrid_retrieval",
        "two_stage_hybrid_rerank",
        "evidence_gate",
        "answer_generator",
    ),
}

REQUIRED_INPUT_SLOTS: Dict[str, tuple[str, ...]] = {
    "information_extraction": tuple(),
    "metric_calculation": ("metrics",),
    "comparison": ("compare_targets",),
    "analysis": ("analysis_topic",),
    "summarization": ("summary_scope",),
}


class SchemaRegistry:
    def __init__(self, schemas: Mapping[str, Type[BaseModel]] | None = None) -> None:
        self._schemas = dict(schemas or PLAN_MODELS)

    def get(self, intent_id: str) -> Type[BaseModel] | None:
        return self._schemas.get(str(intent_id or ""))

    def required_input_slots(self, intent_id: str) -> tuple[str, ...]:
        return REQUIRED_INPUT_SLOTS.get(str(intent_id or ""), tuple())

    def intents(self) -> tuple[str, ...]:
        return tuple(self._schemas)


class ToolRegistry:
    def __init__(
        self,
        intent_tools: Mapping[str, Sequence[str]] | None = None,
        definitions: Mapping[str, ToolDefinition] | None = None,
    ) -> None:
        self._intent_tools = {
            str(intent): tuple(str(name) for name in names)
            for intent, names in (intent_tools or INTENT_TOOLS).items()
        }
        self._definitions = dict(definitions or TOOL_DEFINITIONS)

    def get_tools(self, intent_id: str) -> tuple[ToolDefinition, ...]:
        return tuple(
            self._definitions[name]
            for name in self._intent_tools.get(str(intent_id or ""), tuple())
            if name in self._definitions
        )

    def allowed_names(self, intent_id: str) -> frozenset[str]:
        return frozenset(tool.name for tool in self.get_tools(intent_id))

    def describe_tools(self, intent_id: str) -> list[Dict[str, object]]:
        return [tool.as_dict() for tool in self.get_tools(intent_id)]

    def get_definition(self, tool_name: str) -> ToolDefinition | None:
        return self._definitions.get(str(tool_name or ""))


DEFAULT_SCHEMA_REGISTRY = SchemaRegistry()
DEFAULT_TOOL_REGISTRY = ToolRegistry()
