from __future__ import annotations

from typing import Any, Dict, List, Literal, Type

from pydantic import BaseModel, ConfigDict, Field


class StrictPlannerModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutputRequirements(StrictPlannerModel):
    need_summary: bool = False
    need_location: bool = False
    need_citation: bool = True
    need_comparison: bool = False
    output_format: Literal["answer", "summary", "report", "table"] = "answer"


class NumericCondition(StrictPlannerModel):
    kind: Literal["amount", "percentage"]
    operator: Literal["eq", "gt", "gte", "lt", "lte", "approx"] = "eq"
    value: float
    unit: Literal["CNY", "percent", "percentage_point"]
    raw_text: str


class CommonInputSlots(StrictPlannerModel):
    companies: List[str] = Field(default_factory=list)
    periods: List[str] = Field(default_factory=list)
    quarters: List[str] = Field(default_factory=list)
    half_years: List[str] = Field(default_factory=list)
    report_types: List[str] = Field(default_factory=list)
    statement_types: List[str] = Field(default_factory=list)
    requested_pages: List[int] = Field(default_factory=list)
    document_references: List[str] = Field(default_factory=list)
    document_name: str | None = None
    numeric_conditions: List[NumericCondition] = Field(default_factory=list)


class InformationExtractionInputSlots(CommonInputSlots):
    metrics: List[str] = Field(default_factory=list)
    target: str | None = None


class MetricCalculationInputSlots(CommonInputSlots):
    metrics: List[str] = Field(default_factory=list)
    derived_metric: str | None = None


class ComparisonInputSlots(CommonInputSlots):
    metrics: List[str] = Field(default_factory=list)
    compare_targets: List[str] = Field(default_factory=list)
    comparison_dimension: str | None = None


class AnalysisInputSlots(CommonInputSlots):
    metrics: List[str] = Field(default_factory=list)
    analysis_topic: str | None = None
    analysis_dimension: str | None = None


class SummarizationInputSlots(CommonInputSlots):
    summary_scope: str | None = None
    focus: str | None = None


TaskType = Literal[
    "retrieve",
    "search",
    "locate",
    "answer",
    "calculate",
    "compare",
    "analyze",
    "summarize",
    "generate_report",
]


class PlanTask(StrictPlannerModel):
    task_id: str
    task_type: TaskType
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    output_key: str
    required: bool = True


class _BaseIntentPlan(StrictPlannerModel):
    missing_required_slots: List[str] = Field(default_factory=list)
    output_requirements: OutputRequirements = Field(default_factory=OutputRequirements)
    tasks: List[PlanTask] = Field(default_factory=list)


class InformationExtractionPlan(_BaseIntentPlan):
    intent: Literal["information_extraction"]
    input_slots: InformationExtractionInputSlots


class MetricCalculationPlan(_BaseIntentPlan):
    intent: Literal["metric_calculation"]
    input_slots: MetricCalculationInputSlots


class ComparisonPlan(_BaseIntentPlan):
    intent: Literal["comparison"]
    input_slots: ComparisonInputSlots


class AnalysisPlan(_BaseIntentPlan):
    intent: Literal["analysis"]
    input_slots: AnalysisInputSlots


class SummarizationPlan(_BaseIntentPlan):
    intent: Literal["summarization"]
    input_slots: SummarizationInputSlots


PLAN_MODELS: Dict[str, Type[_BaseIntentPlan]] = {
    "information_extraction": InformationExtractionPlan,
    "metric_calculation": MetricCalculationPlan,
    "comparison": ComparisonPlan,
    "analysis": AnalysisPlan,
    "summarization": SummarizationPlan,
}


class ExecutionContext(StrictPlannerModel):
    document_ids: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    page_numbers: List[int] = Field(default_factory=list)
    table_ids: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_scores: List[float] = Field(default_factory=list)
    tool_outputs: Dict[str, Any] = Field(default_factory=dict)


EXECUTION_SLOT_NAMES = frozenset(
    {
        "document_id",
        "document_ids",
        "chunk_id",
        "chunk_ids",
        "page_number",
        "page_numbers",
        "table_id",
        "table_ids",
        "evidence",
        "citations",
        "retrieval_score",
        "retrieval_scores",
        "tool_outputs",
    }
)
