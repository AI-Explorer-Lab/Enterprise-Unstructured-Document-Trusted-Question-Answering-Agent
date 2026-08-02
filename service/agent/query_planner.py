from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from utils.config_loader import PROJECT_ROOT, load_yaml_file
from utils.content_normalizer import normalize_whitespace


DEFAULT_DOMAIN_GLOSSARY_PATH = PROJECT_ROOT / "config" / "domain_glossary.yaml"


def build_query_plan(question: str, query_type: str = "information_extraction") -> Dict[str, Any]:
    """Build the deterministic fallback for a domain decomposition.

    Runtime requests use DomainDecompositionPlanner first. This function keeps a
    bounded fallback for model outages and remains useful to evidence-gate tests.
    The domain definition fixes required objects; it no longer fixes retrieval
    questions or focus terms.
    """
    normalized = normalize_whitespace(question, preserve_newlines=False)
    match = match_domain_composite(normalized)
    if match is None:
        return {
            "mode": "single",
            "query_type": str(query_type or "information_extraction"),
            "question": normalized,
            "subtasks": [],
        }

    domain_id = str(match.get("domain") or "")
    composite = dict(match.get("composite") or {})
    subtasks = [
        {
            "slot": str(item.get("slot") or ""),
            "display_name": str(item.get("display_name") or item.get("slot") or ""),
            "query_type": str(query_type or match.get("default_query_type") or "information_extraction"),
            "tool_name": str(match.get("retrieval_tool") or "parallel_hybrid_retrieval"),
            "question": f"{normalized}；重点检索{str(item.get('display_name') or item.get('slot') or '')}",
            "match_terms": list(item.get("match_terms") or []),
            "focus_terms": [],
        }
        for item in list(match.get("required_objects") or [])
        if isinstance(item, Mapping) and str(item.get("slot") or "")
    ]
    display_names = [str(item.get("display_name") or item.get("slot") or "") for item in subtasks]
    display_names = [item for item in display_names if item]
    composite_name = str(composite.get("display_name") or composite.get("id") or "").strip()
    metric = composite_name if len(subtasks) == len(_composite_subtasks(composite)) else "、".join(display_names)

    slots = dict(composite.get("slots") or {})
    slots.setdefault("metric", metric)
    slots.setdefault("period", "报告期")
    slots.setdefault("table_name", "、".join(display_names))
    slots.setdefault("focus", "summary")

    return {
        "mode": "decomposed",
        "domain": domain_id,
        "composite_id": str(composite.get("id") or ""),
        "composite_display_name": composite_name,
        "strategy": str(composite.get("strategy") or "subtask_independent_retrieval"),
        "reason": str(
            composite.get("reason")
            or "The question asks for multiple domain objects that should not compete in one retrieval pool."
        ),
        "query_type": str(query_type or match.get("default_query_type") or "information_extraction"),
        "question": normalized,
        "slots": slots,
        "subtasks": subtasks,
        "planner_trace": {
            "source": "deterministic_domain_fallback",
            "validation": {"valid": True, "errors": [], "missing_objects": []},
            "repair_attempted": False,
        },
    }


def is_decomposed_plan(plan: Mapping[str, Any] | None) -> bool:
    return isinstance(plan, Mapping) and str(plan.get("mode") or "") == "decomposed" and bool(plan.get("subtasks"))


@lru_cache(maxsize=1)
def load_domain_glossary(path: str | Path | None = None) -> Dict[str, Any]:
    glossary_path = Path(path).expanduser() if path else DEFAULT_DOMAIN_GLOSSARY_PATH
    loaded = load_yaml_file(glossary_path)
    if not isinstance(loaded, dict):
        return {}
    root = loaded.get("domain_glossary", loaded)
    return root if isinstance(root, dict) else {}


def match_domain_composite(question: str) -> Dict[str, Any] | None:
    if not question:
        return None

    for domain_id, domain_cfg in _iter_domains(load_domain_glossary()):
        for composite in _as_dict_list(domain_cfg.get("composites")):
            objects = _composite_subtasks(composite)
            if len(objects) < 2:
                continue

            aliases = _terms(composite.get("aliases"), composite.get("display_name"))
            if any(term in question for term in aliases):
                selected = objects
            else:
                selected = [
                    item
                    for item in objects
                    if any(term in question for term in _subtask_match_terms(item))
                ]
                min_matched = max(2, int(composite.get("min_matched_subtasks") or 2))
                if len(selected) < min_matched:
                    continue

            default_query_type = str(
                composite.get("default_query_type")
                or composite.get("query_type")
                or "analysis"
            ).strip()
            allowed_query_types = _terms(composite.get("allowed_query_types"))
            if default_query_type and default_query_type not in allowed_query_types:
                allowed_query_types.insert(0, default_query_type)
            return {
                "domain": domain_id,
                "composite": composite,
                "composite_id": str(composite.get("id") or ""),
                "display_name": str(composite.get("display_name") or composite.get("id") or ""),
                "default_query_type": default_query_type,
                "allowed_query_types": allowed_query_types,
                "retrieval_tool": str(composite.get("retrieval_tool") or "parallel_hybrid_retrieval"),
                "required_objects": selected,
            }

    return None


def _iter_domains(glossary: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    domains = glossary.get("domains")
    if isinstance(domains, Mapping):
        for domain_id, domain_cfg in domains.items():
            if isinstance(domain_cfg, Mapping):
                yield str(domain_id), domain_cfg
    elif isinstance(domains, list):
        for domain_cfg in domains:
            if isinstance(domain_cfg, Mapping):
                domain_id = str(domain_cfg.get("id") or domain_cfg.get("domain") or "")
                if domain_id:
                    yield domain_id, domain_cfg


def _composite_subtasks(composite: Mapping[str, Any]) -> List[Dict[str, Any]]:
    decomposition = composite.get("decomposition")
    if isinstance(decomposition, Mapping):
        source = decomposition.get("required_objects") or decomposition.get("subtasks")
    else:
        source = composite.get("required_objects") or composite.get("subtasks")
    return [_normalize_subtask(item) for item in _as_dict_list(source)]


def _normalize_subtask(item: Mapping[str, Any]) -> Dict[str, Any]:
    slot = str(item.get("slot") or item.get("subtask_id") or "").strip()
    display_name = str(item.get("display_name") or slot).strip()
    normalized = {
        "slot": slot,
        "display_name": display_name,
        "query_type": str(item.get("query_type") or "").strip(),
        "question": str(item.get("question") or display_name).strip(),
        "match_terms": _terms(item.get("match_terms"), display_name),
    }
    return {key: value for key, value in normalized.items() if value not in ("", [])}


def _subtask_match_terms(subtask: Mapping[str, Any]) -> List[str]:
    return _terms(subtask.get("match_terms"), subtask.get("display_name"), subtask.get("slot"))


def _terms(*values: Any) -> List[str]:
    terms: List[str] = []
    for value in values:
        if isinstance(value, str):
            candidates = [value]
        elif isinstance(value, Iterable):
            candidates = [str(item or "") for item in value]
        else:
            candidates = []
        for candidate in candidates:
            term = normalize_whitespace(candidate, preserve_newlines=False)
            if term and term not in terms:
                terms.append(term)
    return terms


def _as_dict_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
