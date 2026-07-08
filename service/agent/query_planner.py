from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from utils.config_loader import PROJECT_ROOT, load_yaml_file
from utils.content_normalizer import normalize_whitespace


DEFAULT_DOMAIN_GLOSSARY_PATH = PROJECT_ROOT / "config" / "domain_glossary.yaml"


def build_query_plan(question: str, query_type: str = "fact_lookup") -> Dict[str, Any]:
    normalized = normalize_whitespace(question, preserve_newlines=False)
    match = _match_composite(normalized)
    if match is None:
        return {
            "mode": "single",
            "query_type": str(query_type or "fact_lookup"),
            "question": normalized,
            "subtasks": [],
        }

    domain_id, composite, subtasks = match
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
        "query_type": str(composite.get("query_type") or query_type or "fact_lookup"),
        "question": normalized,
        "slots": slots,
        "subtasks": subtasks,
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


def _match_composite(question: str) -> tuple[str, Dict[str, Any], List[Dict[str, Any]]] | None:
    if not question:
        return None

    for domain_id, domain_cfg in _iter_domains(load_domain_glossary()):
        for composite in _as_dict_list(domain_cfg.get("composites")):
            subtasks = _composite_subtasks(composite)
            if len(subtasks) < 2:
                continue

            aliases = _terms(composite.get("aliases"), composite.get("display_name"))
            if any(term in question for term in aliases):
                return domain_id, composite, subtasks

            selected = [
                subtask
                for subtask in subtasks
                if any(term in question for term in _subtask_match_terms(subtask))
            ]
            min_matched = max(2, int(composite.get("min_matched_subtasks") or 2))
            if len(selected) >= min_matched:
                return domain_id, composite, selected

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
    source = decomposition.get("subtasks") if isinstance(decomposition, Mapping) else composite.get("subtasks")
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
