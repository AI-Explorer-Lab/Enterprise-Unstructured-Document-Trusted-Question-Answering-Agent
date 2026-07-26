from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from pydantic import ValidationError

from service.agent.planner_models import EXECUTION_SLOT_NAMES, PlanTask
from service.agent.planner_registry import (
    ARGUMENT_TYPE_NAMES,
    SchemaRegistry,
    ToolRegistry,
    valid_argument_value,
)


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    executable: bool
    errors: tuple[str, ...]
    missing_required_slots: tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "executable": self.executable,
            "errors": list(self.errors),
            "missing_required_slots": list(self.missing_required_slots),
        }


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def _execution_slot_paths(value: Any, path: str = "") -> List[str]:
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            next_path = f"{path}.{key}" if path else str(key)
            if str(key) in EXECUTION_SLOT_NAMES:
                found.append(next_path)
            found.extend(_execution_slot_paths(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_execution_slot_paths(item, f"{path}[{index}]"))
    return found


def _cycle_nodes(tasks: Iterable[PlanTask]) -> List[str]:
    graph = {task.task_id: list(task.depends_on) for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: List[str] = []

    def visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            cycles.append(node)
            return
        visiting.add(node)
        for dependency in graph.get(node, []):
            if dependency in graph:
                visit(dependency)
        visiting.discard(node)
        visited.add(node)

    for node in graph:
        visit(node)
    return cycles


class PlanValidator:
    def __init__(
        self,
        schema_registry: SchemaRegistry,
        tool_registry: ToolRegistry,
        max_tasks: int = 8,
    ) -> None:
        self.schema_registry = schema_registry
        self.tool_registry = tool_registry
        self.max_tasks = max(1, int(max_tasks))

    def validate(self, intent_id: str, plan_payload: Mapping[str, Any]) -> PlanValidationResult:
        errors: List[str] = []
        schema = self.schema_registry.get(intent_id)
        if schema is None:
            return PlanValidationResult(False, False, (f"unknown intent schema: {intent_id}",), tuple())
        try:
            plan = schema.model_validate(dict(plan_payload))
        except ValidationError as exc:
            return PlanValidationResult(False, False, (f"schema validation failed: {exc}",), tuple())

        if plan.intent != intent_id:
            errors.append(f"plan intent {plan.intent!r} does not match routed intent {intent_id!r}")
        tasks = list(plan.tasks)
        if len(tasks) > self.max_tasks:
            errors.append(f"task count {len(tasks)} exceeds max_tasks={self.max_tasks}")

        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            errors.append("task_id values must be unique")
        task_id_set = set(task_ids)
        for task in tasks:
            unknown_dependencies = [item for item in task.depends_on if item not in task_id_set]
            if unknown_dependencies:
                errors.append(
                    f"task {task.task_id} depends on unknown tasks: {unknown_dependencies}"
                )
            if task.task_id in task.depends_on:
                errors.append(f"task {task.task_id} cannot depend on itself")
        cycles = _cycle_nodes(tasks)
        if cycles:
            errors.append(f"cyclic task dependencies detected: {sorted(set(cycles))}")

        allowed_tools = self.tool_registry.allowed_names(intent_id)
        for task in tasks:
            if task.tool_name not in allowed_tools:
                errors.append(f"tool {task.tool_name!r} is not allowed for intent {intent_id!r}")
                continue
            definition = self.tool_registry.get_definition(task.tool_name)
            if definition is None:
                errors.append(f"tool definition missing for {task.tool_name!r}")
                continue
            if task.task_type not in definition.task_types:
                errors.append(
                    f"task {task.task_id} type {task.task_type!r} is invalid for tool {task.tool_name!r}"
                )
            unknown_arguments = sorted(set(task.arguments) - set(definition.allowed_arguments))
            if unknown_arguments:
                errors.append(
                    f"task {task.task_id} has unsupported arguments for {task.tool_name}: {unknown_arguments}"
                )
            invalid_argument_types = [
                name
                for name, value in task.arguments.items()
                if name in definition.allowed_arguments
                and (
                    name not in ARGUMENT_TYPE_NAMES
                    or not valid_argument_value(name, value)
                )
            ]
            if invalid_argument_types:
                errors.append(
                    f"task {task.task_id} has invalid argument types for {task.tool_name}: "
                    f"{sorted(invalid_argument_types)}"
                )
            missing_arguments = [
                name
                for name in definition.required_arguments
                if _is_empty(task.arguments.get(name))
            ]
            if missing_arguments:
                errors.append(
                    f"task {task.task_id} is missing required tool arguments: {missing_arguments}"
                )

        execution_paths = _execution_slot_paths(plan.input_slots.model_dump())
        for task in tasks:
            execution_paths.extend(
                _execution_slot_paths(task.arguments, f"tasks.{task.task_id}.arguments")
            )
        if execution_paths:
            errors.append(
                "planner attempted to populate execution-only slots: "
                + ", ".join(sorted(set(execution_paths)))
            )

        input_slots = plan.input_slots.model_dump()
        for task in tasks:
            for name, value in task.arguments.items():
                if name not in input_slots or _is_empty(input_slots.get(name)):
                    continue
                if value != input_slots[name]:
                    errors.append(
                        f"task {task.task_id} argument {name!r} conflicts with validated input_slots"
                    )
        required_slots = self.schema_registry.required_input_slots(intent_id)
        missing = tuple(name for name in required_slots if _is_empty(input_slots.get(name)))
        declared_missing = tuple(dict.fromkeys(plan.missing_required_slots))
        if set(missing) != set(declared_missing):
            errors.append(
                f"missing_required_slots must equal actual missing slots: expected {list(missing)}"
            )
        if not missing:
            task_types = {task.task_type for task in tasks}
            tool_names = {task.tool_name for task in tasks}
            if "retrieve" not in task_types:
                errors.append("executable plan must contain at least one retrieve task")
            if "evidence_gate" not in tool_names:
                errors.append("executable plan must contain the evidence_gate tool")
            if "answer_generator" not in tool_names:
                errors.append("executable plan must contain the answer_generator tool")

        valid = not errors
        return PlanValidationResult(
            valid=valid,
            executable=valid and not missing,
            errors=tuple(errors),
            missing_required_slots=missing,
        )
