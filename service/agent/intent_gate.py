from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

from pydantic import BaseModel, Field

from service.agent.schemas import PRIMARY_INTENT_TYPE_SET
from utils.config_loader import PROJECT_ROOT, load_yaml_file


DEFAULT_INTENT_GATE_PATH = PROJECT_ROOT / "config" / "intent_router.yaml"
INTENT_DECISIONS = frozenset({"select", "planner", "clarify", "reject"})

ZERO_SHOT_INTENT_SYSTEM_PROMPT = """You are the intent gate for a financial-report question-answering system.
Classify the entire user request, not merely its first clause.

Supported primary intents:
- information_extraction: retrieve an explicitly disclosed value, fact, name, passage, page, or list.
- metric_calculation: calculate a derived number, ratio, rate, difference, or change.
- comparison: compare periods, companies, business segments, documents, or values.
- analysis: explain causes, effects, risks, sustainability, meaning, or quality.
- summarization: summarize, outline, consolidate, or extract the main points.

Choose exactly one decision:
- select: one primary intent is sufficient to represent the whole requested task.
- planner: the request contains two or more distinct supported operations or dependent steps.
- clarify: the request is financial-report related or referential, but lacks enough information to
  determine the requested operation without missing conversation context.
- reject: the request is outside the supported financial-report QA scope.

Do not assume any previous conversation. Output exactly one JSON object without Markdown or
explanation:
{"decision":"select|planner|clarify|reject","intent_id":"one supported primary intent or empty string","sub_intents":["zero or more supported primary intents"]}

For select, intent_id must contain exactly one supported primary intent.
For planner, sub_intents must contain at least two distinct supported primary intents in execution order.
For clarify or reject, intent_id must be empty and sub_intents must be empty."""


class LLMIntentDecisionSchema(BaseModel):
    decision: str
    intent_id: str = ""
    sub_intents: List[str] = Field(default_factory=list)


def load_intent_gate_config(path: str | Path | None = None) -> Dict[str, Any]:
    source = Path(path).expanduser() if path else DEFAULT_INTENT_GATE_PATH
    loaded = load_yaml_file(source)
    config = loaded.get("intent_router", loaded) if isinstance(loaded, Mapping) else {}
    return dict(config) if isinstance(config, Mapping) else {}


def _unique_primary_intents(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    result: List[str] = []
    for value in values:
        intent_id = str(value or "").strip()
        if intent_id in PRIMARY_INTENT_TYPE_SET and intent_id not in result:
            result.append(intent_id)
    return result


class LLMIntentGate:
    """Zero-shot LLM intent decision gate with no embedding or prototype dependency."""

    def __init__(
        self,
        llm_service: Any,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.llm_service = llm_service
        self.config = dict(config or load_intent_gate_config())
        self.enabled = bool(self.config.get("enabled", True))
        self.max_tokens = max(128, int(self.config.get("max_tokens", 320)))

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
                schema=LLMIntentDecisionSchema,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            return self._error(type(exc).__name__)
        return self._normalize(payload)

    def _normalize(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, Mapping):
            return self._error("response_is_not_object")
        decision = str(payload.get("decision") or "").strip().lower()
        intent_id = str(payload.get("intent_id") or "").strip()
        sub_intents = _unique_primary_intents(payload.get("sub_intents"))
        validation_error = ""
        if decision not in INTENT_DECISIONS:
            validation_error = "unsupported_decision"
        elif decision == "select" and intent_id not in PRIMARY_INTENT_TYPE_SET:
            validation_error = "select_requires_supported_intent"
        elif decision == "planner" and len(sub_intents) < 2:
            validation_error = "planner_requires_two_sub_intents"
        elif decision in {"clarify", "reject"} and (intent_id or sub_intents):
            validation_error = "guard_decision_requires_empty_intents"
        if validation_error:
            return self._error(validation_error)

        execution_intent = intent_id if decision == "select" else (
            sub_intents[-1] if decision == "planner" else ""
        )
        route_status = "accepted" if decision in {"select", "planner"} else "unknown"
        return {
            "valid": True,
            "decision": decision,
            "route_status": route_status,
            "intent_id": intent_id if decision == "select" else "",
            "sub_intents": sub_intents if decision == "planner" else [],
            "execution_intent": execution_intent,
            "top_intent": execution_intent,
            "provider": "llm",
            "strategy": "llm_zero_shot",
            "few_shot": False,
            "validation_error": "",
        }

    @staticmethod
    def _error(error: str) -> Dict[str, Any]:
        return {
            "valid": False,
            "decision": "error",
            "route_status": "error",
            "intent_id": "",
            "sub_intents": [],
            "execution_intent": "",
            "top_intent": "",
            "provider": "llm",
            "strategy": "llm_zero_shot",
            "few_shot": False,
            "validation_error": str(error or "unknown_error"),
        }
