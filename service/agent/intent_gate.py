from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from pydantic import BaseModel, Field

from service.agent.schemas import PRIMARY_INTENT_TYPE_SET
from utils.config_loader import PROJECT_ROOT, load_yaml_file


DEFAULT_INTENT_GATE_PATH = PROJECT_ROOT / "config" / "intent_router.yaml"
INTENT_DECISIONS = frozenset({"select", "planner", "clarify", "reject"})
INTENT_SCOPE_STATUSES = frozenset({"supported", "unclear", "out_of_scope"})

_OPERATION_TERMS: Dict[str, tuple[str, ...]] = {
    "metric_calculation": ("计算", "算出", "求出", "算一下", "计算结果"),
    "comparison": ("比较", "对比", "排序", "谁更", "哪个更"),
    "analysis": ("分析", "判断", "评价", "解释", "说明原因", "是否安全"),
    "summarization": ("总结", "概括", "梳理", "提炼", "摘要"),
    "information_extraction": (
        "原始值",
        "原值",
        "原文",
        "单独列出",
        "单独给出",
        "披露值",
        "名单",
        "页码",
    ),
}

ZERO_SHOT_INTENT_SYSTEM_PROMPT = """You are the semantic decomposition gate for a financial-report question-answering system.
Decompose the entire user request, not merely its first clause. Do not decide select or planner.

Supported primary intents:
- information_extraction: return an explicitly disclosed raw value, fact, name, passage, page, or list.
- metric_calculation: calculate a derived number, ratio, rate, difference, or change.
- comparison: compare periods, companies, business segments, documents, or values.
- analysis: explain causes, effects, risks, sustainability, meaning, or quality.
- summarization: summarize, outline, consolidate, or extract the main points.

First classify scope_status as supported, unclear, or out_of_scope. Clarify means that the requested
operation itself cannot be determined; missing company, year, metric, or report slots do not make a
clear operation unclear. Reject only clearly out-of-scope tasks.

Financial-report questions about causes, risks, effects, sustainability, business meaning, or whether
a management statement is supported by financial data are supported analysis tasks, even when the
question does not name a precise metric. Comparing reporting scopes, periods, documents, or disclosure
locations is a supported comparison task. Checking whether a calculated result agrees with a disclosed
figure is an internal disclosure-consistency check, not a separate comparison output.

requested_outputs contains only results the user explicitly expects to see in the final answer. Each
item has exactly one supported primary intent. If the user asks for two different visible results,
return two items in the user's execution order, even when the second result depends on the first.
Multiple metrics under the same task type remain the same intent type.

internal_requirements contains only process steps such as evidence_retrieval, operand_lookup,
slot_completion, citation_location, or output_formatting. These are never primary intents.

Use information_extraction only when raw disclosed values, passages, names, lists, or locations are
explicitly requested as a separately displayed final result. A lookup that only supplies operands or
evidence for a following calculation, comparison, analysis, or summary is an internal requirement.
Use summarization for condensing, organizing, outlining, or selecting main points from a broader scope.

Do not combine different operations into one requested_outputs item. For example, “calculate and judge"
must produce metric_calculation and analysis outputs; “compare and explain” must produce comparison
and analysis outputs. The program will derive select or planner from the distinct output intent types.

Do not assume any previous conversation. Output exactly one JSON object without Markdown or explanation:
{"scope_status":"supported|unclear|out_of_scope","requested_outputs":[{"intent_id":"supported primary intent","requested_output":"one non-empty user-visible result"}],"internal_requirements":["evidence_retrieval|operand_lookup|slot_completion|citation_location|output_formatting"]}

For unclear or out_of_scope, requested_outputs and internal_requirements must be empty."""


class IntentOutputSchema(BaseModel):
    intent_id: str
    requested_output: str


class LLMIntentDecompositionSchema(BaseModel):
    scope_status: str = "supported"
    requested_outputs: List[IntentOutputSchema] = Field(default_factory=list)
    internal_requirements: List[str] = Field(default_factory=list)


# Preserve the public symbol name for callers that only import the schema class.
LLMIntentDecisionSchema = LLMIntentDecompositionSchema


def load_intent_gate_config(path: str | Path | None = None) -> Dict[str, Any]:
    source = Path(path).expanduser() if path else DEFAULT_INTENT_GATE_PATH
    loaded = load_yaml_file(source)
    config = loaded.get("intent_router", loaded) if isinstance(loaded, Mapping) else {}
    return dict(config) if isinstance(config, Mapping) else {}


def _normalize_requested_outputs(values: Any) -> List[Dict[str, str]]:
    if not isinstance(values, list):
        return []
    result: List[Dict[str, str]] = []
    for value in values:
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif hasattr(value, "dict"):
            value = value.dict()
        if not isinstance(value, Mapping):
            continue
        result.append(
            {
                "intent_id": str(value.get("intent_id") or "").strip(),
                "requested_output": str(value.get("requested_output") or "").strip(),
            }
        )
    return result


def _normalize_strings(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _canonicalize_intent_id(intent_id: str, requested_output: str, question: str) -> str:
    raw = str(intent_id or "").strip().lower()
    if raw in PRIMARY_INTENT_TYPE_SET:
        if (
            raw == "information_extraction"
            and re.search(r"整理|简洁|概括|梳理", str(question or ""))
            and re.search(r"名单|主营业务|主要内容", requested_output)
        ):
            return "summarization"
        if (
            raw == "analysis"
            and re.search(r"差在哪|是否一致|对得上|对不上", str(question or ""))
            and not re.search(r"为什么|原因|风险|影响", str(question or ""))
        ):
            return "comparison"
        return raw

    aliases = (
        ("metric_calculation", ("calculate", "calculation", "calc", "metric", "ratio", "rate", "increase")),
        ("comparison", ("compare", "comparison", "disclosure_comparison", "difference", "heavier", "higher", "lower")),
        ("analysis", ("analyze", "analysis", "reason", "risk", "impact", "conclude", "judge")),
        ("summarization", ("summar", "summary", "outline", "list_", "describe_")),
        ("information_extraction", ("extract", "lookup", "find", "list")),
    )
    for canonical, prefixes in aliases:
        if any(raw == prefix or raw.startswith(prefix + "_") for prefix in prefixes):
            if (
                canonical in {"information_extraction", "summarization"}
                and re.search(r"整理|简洁|概括|梳理", str(question or ""))
                and re.search(r"名单|主营业务|主要内容", requested_output)
            ):
                return "summarization"
            return canonical

    signals = _operation_signals(requested_output)
    if len(signals) == 1:
        return signals[0]
    if "summarization" in signals and re.search(r"整理|简洁", requested_output):
        return "summarization"
    return raw


def _remove_internal_validation_outputs(
    outputs: List[Dict[str, str]],
    internal_requirements: List[str],
) -> tuple[List[Dict[str, str]], List[str]]:
    has_calculation = any(item["intent_id"] == "metric_calculation" for item in outputs)
    if not has_calculation:
        return outputs, internal_requirements
    retained: List[Dict[str, str]] = []
    requirements = list(internal_requirements)
    for item in outputs:
        text = item["requested_output"]
        is_disclosure_check = (
            item["intent_id"] == "comparison"
            and re.search(r"披露值|报表披露|等于|差几个百分点|百分点差异", text)
        )
        if is_disclosure_check:
            if "disclosure_consistency_check" not in requirements:
                requirements.append("disclosure_consistency_check")
            continue
        retained.append(item)
    return retained, requirements


def _operation_signals(text: str) -> List[str]:
    normalized = str(text or "").strip().lower()
    return [
        intent_id
        for intent_id, terms in _OPERATION_TERMS.items()
        if any(re.search(re.escape(term.lower()), normalized) for term in terms)
    ]


def _group_requested_outputs(
    outputs: List[Dict[str, str]],
) -> tuple[List[str], List[Dict[str, str]]]:
    ordered_intents: List[str] = []
    grouped: Dict[str, List[str]] = {}
    for item in outputs:
        intent_id = item["intent_id"]
        if intent_id not in ordered_intents:
            ordered_intents.append(intent_id)
            grouped[intent_id] = []
        grouped[intent_id].append(item["requested_output"])
    deliverables = [
        {
            "intent_id": intent_id,
            "requested_output": "；".join(grouped[intent_id]),
        }
        for intent_id in ordered_intents
    ]
    return ordered_intents, deliverables


class LLMIntentGate:
    """Decompose user-visible intent outputs; derive routing decisions in code."""

    def __init__(
        self,
        llm_service: Any,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.config = dict(config or load_intent_gate_config())
        self.enabled = bool(self.config.get("enabled", True))
        self.max_tokens = max(128, int(self.config.get("max_tokens", 512)))

    async def decide(self, question: str) -> Dict[str, Any]:
        if not self.enabled:
            return self._error("disabled")
        structured_json = getattr(self.llm_service, "structured_json", None)
        if not callable(structured_json):
            return self._error("llm_structured_json_unavailable")
        try:
            payload = await structured_json(
                ZERO_SHOT_INTENT_SYSTEM_PROMPT,
                {"question": str(question or "")},
                schema=LLMIntentDecompositionSchema,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            return self._error(type(exc).__name__)

        normalized = self._normalize(payload, question=question)
        if normalized.get("validation_error") != "requested_output_contains_multiple_intents":
            return normalized

        try:
            repaired = await structured_json(
                "Repair only the semantic decomposition conflict. Split every distinct user-visible "
                "operation into its own requested_outputs item. Keep operand lookup, evidence retrieval, "
                "citations, and formatting in internal_requirements. Do not output select or planner.",
                {
                    "question": str(question or ""),
                    "invalid_decomposition": payload,
                    "validation_error": normalized.get("validation_error"),
                },
                schema=LLMIntentDecompositionSchema,
                max_tokens=self.max_tokens,
            )
        except Exception:
            repaired = None
        repaired_normalized = self._normalize(repaired, question=question)
        repaired_normalized["repair_attempted"] = True
        return repaired_normalized

    def _normalize(self, payload: Any, *, question: str = "") -> Dict[str, Any]:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        elif hasattr(payload, "dict"):
            payload = payload.dict()
        if not isinstance(payload, Mapping):
            return self._error("response_is_not_object")

        raw_scope_status = str(payload.get("scope_status") or "").strip().lower()
        raw_outputs = payload.get("requested_outputs", [])
        raw_requirements = payload.get("internal_requirements", [])
        outputs = _normalize_requested_outputs(raw_outputs)
        internal_requirements = _normalize_strings(raw_requirements)
        normalized_output_count = len(outputs)
        normalized_requirement_count = len(internal_requirements)
        for item in outputs:
            item["intent_id"] = _canonicalize_intent_id(
                item["intent_id"], item["requested_output"], question
            )
        scope_status = raw_scope_status or ("supported" if outputs else "unclear")
        outputs, internal_requirements = _remove_internal_validation_outputs(
            outputs, internal_requirements
        )
        validation_error = ""
        if scope_status not in INTENT_SCOPE_STATUSES:
            validation_error = "unsupported_scope_status"
        elif not isinstance(raw_outputs, list):
            validation_error = "requested_outputs_must_be_array"
        elif normalized_output_count != len(raw_outputs):
            validation_error = "requested_outputs_must_be_objects"
        elif not isinstance(raw_requirements, list):
            validation_error = "internal_requirements_must_be_array"
        elif normalized_requirement_count != len(raw_requirements):
            validation_error = "internal_requirements_must_be_strings"
        elif scope_status == "supported" and not outputs:
            validation_error = "supported_requires_requested_outputs"
        elif scope_status in {"unclear", "out_of_scope"} and (outputs or internal_requirements):
            validation_error = "guard_scope_requires_empty_outputs"
        elif any(
            item["intent_id"] not in PRIMARY_INTENT_TYPE_SET or not item["requested_output"]
            for item in outputs
        ):
            validation_error = "requested_output_requires_supported_intent_and_text"
        else:
            semantic_conflicts = []
            for item in outputs:
                signals = _operation_signals(item["requested_output"])
                conflicting = [signal for signal in signals if signal != item["intent_id"]]
                if conflicting:
                    semantic_conflicts.append(
                        {
                            "intent_id": item["intent_id"],
                            "requested_output": item["requested_output"],
                            "conflicting_intents": conflicting,
                        }
                    )
            if semantic_conflicts:
                validation_error = "requested_output_contains_multiple_intents"

        if validation_error:
            error = self._error(validation_error)
            error["scope_status"] = scope_status
            error["requested_outputs"] = outputs
            error["internal_requirements"] = internal_requirements
            return error

        if scope_status in {"unclear", "out_of_scope"}:
            decision = "clarify" if scope_status == "unclear" else "reject"
            return {
                "valid": True,
                "decision": decision,
                "route_status": "unknown",
                "scope_status": scope_status,
                "intent_id": "",
                "sub_intents": [],
                "deliverables": [],
                "requested_outputs": [],
                "internal_requirements": [],
                "execution_intent": "",
                "top_intent": "",
                "provider": "llm",
                "strategy": "llm_decomposition_programmatic_route",
                "few_shot": False,
                "validation_error": "",
                "repair_attempted": False,
            }

        ordered_intents, deliverables = _group_requested_outputs(outputs)
        decision = "select" if len(ordered_intents) == 1 else "planner"
        execution_intent = ordered_intents[-1]
        return {
            "valid": True,
            "decision": decision,
            "route_status": "accepted",
            "scope_status": scope_status,
            "intent_id": ordered_intents[0] if decision == "select" else "",
            "sub_intents": ordered_intents if decision == "planner" else [],
            "deliverables": deliverables,
            "requested_outputs": outputs,
            "internal_requirements": internal_requirements,
            "execution_intent": execution_intent,
            "top_intent": execution_intent,
            "provider": "llm",
            "strategy": "llm_decomposition_programmatic_route",
            "few_shot": False,
            "validation_error": "",
            "repair_attempted": False,
        }

    @staticmethod
    def _error(error: str) -> Dict[str, Any]:
        return {
            "valid": False,
            "decision": "error",
            "route_status": "error",
            "scope_status": "",
            "intent_id": "",
            "sub_intents": [],
            "deliverables": [],
            "requested_outputs": [],
            "internal_requirements": [],
            "execution_intent": "",
            "top_intent": "",
            "provider": "llm",
            "strategy": "llm_decomposition_programmatic_route",
            "few_shot": False,
            "validation_error": str(error or "unknown_error"),
            "repair_attempted": False,
        }
