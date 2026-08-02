from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from service.agent.query_planner import build_query_plan, match_domain_composite


ALLOWED_SUBTASK_QUERY_TYPES = frozenset(
    {"information_extraction", "analysis", "summarization"}
)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentDomainSubtask(_StrictModel):
    subtask_id: str
    query_type: Literal["information_extraction", "analysis", "summarization"]
    tool_name: str
    question: str
    focus_terms: List[str] = Field(default_factory=list)


class AgentDomainDecomposition(_StrictModel):
    subtasks: List[AgentDomainSubtask] = Field(default_factory=list)


@dataclass(frozen=True)
class DomainPlanValidationResult:
    valid: bool
    errors: tuple[str, ...]
    missing_objects: tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "missing_objects": list(self.missing_objects),
        }


class DomainPlanValidator:
    def __init__(self, max_subtasks: int = 10) -> None:
        self.max_subtasks = max(1, int(max_subtasks))

    def validate(
        self,
        payload: Mapping[str, Any],
        definition: Mapping[str, Any],
        original_question: str,
    ) -> DomainPlanValidationResult:
        try:
            plan = AgentDomainDecomposition.model_validate(dict(payload))
        except ValidationError as exc:
            return DomainPlanValidationResult(
                False,
                (f"schema validation failed: {exc}",),
                tuple(),
            )

        errors: List[str] = []
        required = [
            str(item.get("slot") or "")
            for item in list(definition.get("required_objects") or [])
            if isinstance(item, Mapping) and str(item.get("slot") or "")
        ]
        task_ids = [task.subtask_id for task in plan.subtasks]
        missing = tuple(item for item in required if item not in task_ids)
        extras = sorted(set(task_ids) - set(required))
        if len(task_ids) > self.max_subtasks:
            errors.append(
                f"subtask count {len(task_ids)} exceeds max_subtasks={self.max_subtasks}"
            )
        if len(task_ids) != len(set(task_ids)):
            errors.append("subtask_id values must be unique")
        if missing:
            errors.append(f"required domain objects are missing: {list(missing)}")
        if extras:
            errors.append(f"unsupported domain objects were added: {extras}")

        expected_tool = str(
            definition.get("retrieval_tool") or "parallel_hybrid_retrieval"
        )
        original_years = set(YEAR_RE.findall(original_question))
        for task in plan.subtasks:
            if task.tool_name != expected_tool:
                errors.append(
                    f"subtask {task.subtask_id} uses non-whitelisted tool {task.tool_name!r}"
                )
            if task.query_type not in ALLOWED_SUBTASK_QUERY_TYPES:
                errors.append(
                    f"subtask {task.subtask_id} has unsupported query_type {task.query_type!r}"
                )
            question = task.question.strip()
            if len(question) < 4 or len(question) > 300:
                errors.append(
                    f"subtask {task.subtask_id} question length must be between 4 and 300"
                )
            introduced_years = set(YEAR_RE.findall(question)) - original_years
            if introduced_years:
                errors.append(
                    f"subtask {task.subtask_id} introduced years not present in the request: "
                    f"{sorted(introduced_years)}"
                )

        return DomainPlanValidationResult(
            valid=not errors,
            errors=tuple(errors),
            missing_objects=missing,
        )


class DomainDecompositionPlanner:
    """Let an Agent phrase domain-specific retrieval subtasks, then validate them.

    The domain ontology, required object set, retrieval tool, global company and
    period filters remain deterministic. The Agent controls only the query and
    focus for each required object.
    """

    def __init__(
        self,
        structured_planner: Any,
        *,
        max_subtasks: int = 10,
        allow_one_repair: bool = True,
    ) -> None:
        self.structured_planner = structured_planner
        self.validator = DomainPlanValidator(max_subtasks=max_subtasks)
        self.allow_one_repair = bool(allow_one_repair)

    async def plan(self, question: str, routed_query_type: str) -> Dict[str, Any]:
        definition = match_domain_composite(question)
        if definition is None:
            return {
                "plan": build_query_plan(question, routed_query_type),
                "validation": {"valid": True, "errors": [], "missing_objects": []},
                "source": "single_query",
                "repair_attempted": False,
            }

        effective_query_type = str(routed_query_type or "").strip()
        if effective_query_type == "ambiguous_query" or not effective_query_type:
            effective_query_type = str(
                definition.get("default_query_type") or "analysis"
            )
        allowed_top_level = set(definition.get("allowed_query_types") or [])
        if allowed_top_level and effective_query_type not in allowed_top_level:
            effective_query_type = str(
                definition.get("default_query_type") or "analysis"
            )

        payload = await self._call_agent(
            question,
            effective_query_type,
            definition,
            repair=None,
        )
        source = "agent"
        repair_attempted = False
        if payload is None:
            return self._fallback(
                question,
                effective_query_type,
                source="deterministic_domain_fallback",
                repair_attempted=False,
            )

        validation = self.validator.validate(payload, definition, question)
        if not validation.valid and self.allow_one_repair:
            repair_attempted = True
            repaired = await self._call_agent(
                question,
                effective_query_type,
                definition,
                repair={
                    "validation_errors": list(validation.errors),
                    "current_plan": dict(payload),
                },
            )
            if repaired is not None:
                payload = repaired
                source = "agent_repair"
                validation = self.validator.validate(payload, definition, question)

        if not validation.valid:
            return self._fallback(
                question,
                effective_query_type,
                source="deterministic_domain_fallback_after_invalid_plan",
                repair_attempted=repair_attempted,
                rejected_validation=validation.as_dict(),
            )

        runtime_plan = self._runtime_plan(
            question,
            effective_query_type,
            definition,
            AgentDomainDecomposition.model_validate(dict(payload)),
        )
        runtime_plan["planner_trace"] = {
            "source": source,
            "validation": validation.as_dict(),
            "repair_attempted": repair_attempted,
        }
        return {
            "plan": runtime_plan,
            "validation": validation.as_dict(),
            "source": source,
            "repair_attempted": repair_attempted,
        }

    async def _call_agent(
        self,
        question: str,
        query_type: str,
        definition: Mapping[str, Any],
        repair: Mapping[str, Any] | None,
    ) -> Dict[str, Any] | None:
        structured_json = getattr(self.structured_planner, "structured_json", None)
        if not callable(structured_json):
            return None
        required_objects = [
            {
                "subtask_id": str(item.get("slot") or ""),
                "display_name": str(item.get("display_name") or item.get("slot") or ""),
                "match_terms": list(item.get("match_terms") or []),
            }
            for item in list(definition.get("required_objects") or [])
            if isinstance(item, Mapping)
        ]
        system_prompt = (
            "Create retrieval subtasks for a financial-report domain request. "
            "Generate exactly one subtask for every required domain object and no others. "
            "Adapt each question and focus_terms to the user's requested analytical focus. "
            "Do not change or invent companies, periods, or domain objects. "
            "Use only the supplied retrieval tool and allowed query types."
        )
        user_payload: Dict[str, Any] = {
            "question": question,
            "routed_query_type": query_type,
            "composite_id": definition.get("composite_id"),
            "required_objects": required_objects,
            "retrieval_tool": definition.get("retrieval_tool"),
            "allowed_subtask_query_types": sorted(ALLOWED_SUBTASK_QUERY_TYPES),
        }
        if repair:
            system_prompt += " Repair only the listed validation errors and preserve user meaning."
            user_payload["repair"] = dict(repair)
        try:
            result = await structured_json(
                system_prompt,
                user_payload,
                schema=AgentDomainDecomposition,
                max_tokens=1000,
            )
        except Exception:
            return None
        return dict(result) if isinstance(result, Mapping) else None

    def _fallback(
        self,
        question: str,
        query_type: str,
        *,
        source: str,
        repair_attempted: bool,
        rejected_validation: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        plan = build_query_plan(question, query_type)
        trace = dict(plan.get("planner_trace") or {})
        trace.update(
            {
                "source": source,
                "repair_attempted": repair_attempted,
                "rejected_validation": dict(rejected_validation or {}),
            }
        )
        plan["planner_trace"] = trace
        return {
            "plan": plan,
            "validation": trace.get("validation") or {
                "valid": True,
                "errors": [],
                "missing_objects": [],
            },
            "source": source,
            "repair_attempted": repair_attempted,
        }

    @staticmethod
    def _runtime_plan(
        question: str,
        query_type: str,
        definition: Mapping[str, Any],
        agent_plan: AgentDomainDecomposition,
    ) -> Dict[str, Any]:
        object_by_id = {
            str(item.get("slot") or ""): dict(item)
            for item in list(definition.get("required_objects") or [])
            if isinstance(item, Mapping)
        }
        subtasks: List[Dict[str, Any]] = []
        for task in agent_plan.subtasks:
            domain_object = object_by_id[task.subtask_id]
            focus_terms = [
                str(item).strip()
                for item in task.focus_terms
                if str(item).strip()
            ]
            match_terms: List[str] = []
            for item in [
                *(domain_object.get("match_terms") or []),
                *focus_terms,
            ]:
                value = str(item or "").strip()
                if value and value not in match_terms:
                    match_terms.append(value)
            subtasks.append(
                {
                    "slot": task.subtask_id,
                    "display_name": str(
                        domain_object.get("display_name") or task.subtask_id
                    ),
                    "query_type": task.query_type,
                    "tool_name": task.tool_name,
                    "question": task.question.strip(),
                    "focus_terms": focus_terms,
                    "match_terms": match_terms,
                }
            )

        display_names = [item["display_name"] for item in subtasks]
        composite = dict(definition.get("composite") or {})
        slots = dict(composite.get("slots") or {})
        slots.setdefault("metric", str(definition.get("display_name") or ""))
        slots.setdefault("period", "报告期")
        slots.setdefault("table_name", "、".join(display_names))
        slots.setdefault("focus", "summary")
        return {
            "mode": "decomposed",
            "domain": str(definition.get("domain") or ""),
            "composite_id": str(definition.get("composite_id") or ""),
            "composite_display_name": str(definition.get("display_name") or ""),
            "strategy": str(
                composite.get("strategy") or "subtask_independent_retrieval"
            ),
            "reason": str(composite.get("reason") or "Agent-planned domain decomposition."),
            "query_type": query_type,
            "question": question,
            "slots": slots,
            "subtasks": subtasks,
        }
