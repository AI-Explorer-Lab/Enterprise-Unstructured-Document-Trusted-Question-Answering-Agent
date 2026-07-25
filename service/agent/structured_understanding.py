from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from service.agent.company_registry import CompanyProfile, CompanyRegistry
from utils.config_loader import PROJECT_ROOT, load_yaml_file
from utils.content_normalizer import normalize_whitespace


DEFAULT_INTENT_ROUTER_PATH = PROJECT_ROOT / "config" / "intent_router.yaml"

ACTION_TO_QUERY_TYPE = {
    "lookup": "fact_lookup",
    "analyze": "table_qa",
    "summarize": "summarization",
    "locate": "citation_locate",
    "generate_report": "report_generation",
    "compare": "multi_doc_compare",
}

QUERY_TYPE_TO_ACTION = {
    "fact_lookup": "lookup",
    "table_qa": "lookup",
    "summarization": "summarize",
    "citation_locate": "locate",
    "report_generation": "generate_report",
    "multi_doc_compare": "compare",
}

ACTION_TERMS: Dict[str, tuple[str, ...]] = {
    "compare": ("对比", "比较", "相比", "差异", "差距", "区别", "谁更", "哪个更", "vs", "versus"),
    "analyze": ("分析", "解读", "怎么看", "说明什么", "能看出什么"),
    "summarize": ("总结", "概述", "概括", "归纳", "梳理"),
    "locate": ("定位原文", "查找原文", "原文在哪", "在哪一页", "哪一页提到", "找到原文"),
    "generate_report": ("生成报告", "撰写报告", "输出报告", "形成报告", "整理成报告"),
    "lookup": ("查一下", "查询", "告诉我", "是多少", "是什么", "多少"),
}

CITATION_REQUIREMENT_TERMS = (
    "给出出处",
    "注明出处",
    "注明来源",
    "给出来源",
    "附上来源",
    "附上页码",
    "给出页码",
    "提供依据",
    "原文依据",
    "citation",
    "source",
)

REPORT_TERMS = ("生成报告", "撰写报告", "输出报告", "形成报告", "分析报告", "简短报告", "报告形式")
SHORT_TERMS = ("简短", "简单", "简要", "精简")
TABLE_OUTPUT_TERMS = ("用表格", "表格展示", "表格形式", "整理成表格")

METRIC_ALIASES: Dict[str, tuple[str, ...]] = {
    "营业收入": ("营业收入", "营收", "收入规模"),
    "营业成本": ("营业成本",),
    "净利润": ("净利润",),
    "归母净利润": ("归母净利润", "归属于上市公司股东的净利润"),
    "经营活动产生的现金流量净额": ("经营活动产生的现金流量净额", "经营现金流", "经营活动现金流"),
    "毛利率": ("毛利率",),
    "研发投入": ("研发投入", "研发费用"),
    "货币资金": ("货币资金",),
    "应收账款": ("应收账款",),
    "基本每股收益": ("基本每股收益", "每股收益"),
}

DOMAIN_OBJECT_ALIASES: Dict[str, tuple[str, ...]] = {
    "financial_three_statements": ("财务三表", "三大财务报表", "三张财务报表", "三张报表"),
    "balance_sheet": ("合并资产负债表", "资产负债表"),
    "income_statement": ("合并利润表", "利润表"),
    "cash_flow_statement": ("合并现金流量表", "现金流量表"),
}

_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_COMPARE_CONNECTOR_RE = re.compile(r"\s*(?:和|与|及|跟|同|vs\.?|versus)\s*", re.IGNORECASE)
_CLAUSE_END_RE = re.compile(r"[，,。；;]|(?:并且|并给出|最后|同时|另外)")
_LEADING_REQUEST_RE = re.compile(
    r"^(?:请|帮我|麻烦|想要|我要|请帮我|请你)?(?:对比|比较|分析|查询|查一下|看看)?(?:一下|下)?"
)
_TRAILING_REQUEST_RE = re.compile(
    r"(?:的)?(?:营业收入|营收|收入规模|营业成本|归母净利润|净利润|毛利率|研发投入|研发费用|"
    r"经营活动产生的现金流量净额|经营现金流|现金流|财务三表|三大财务报表|三张财务报表|"
    r"资产负债表|利润表|现金流量表)(?:是多少|多少|情况|数据|指标|表现)?$"
)
_COMPANY_BEFORE_METRIC_RE = re.compile(
    r"(?P<company>[\u4e00-\u9fffA-Za-z0-9·（）()\-]{2,40}?)(?:的)?"
    r"(?:营业收入|营收|收入规模|营业成本|归母净利润|净利润|毛利率|研发投入|研发费用|"
    r"经营活动产生的现金流量净额|经营现金流|财务三表|资产负债表|利润表|现金流量表)"
)


def _clean(value: Any) -> str:
    return normalize_whitespace(str(value or ""), preserve_newlines=False)


def _unique(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = _clean(value)
        if text and text not in result:
            result.append(text)
    return result


def _source_span(text: str, source_text: str) -> List[int]:
    start = text.find(source_text)
    if start < 0:
        return []
    return [start, start + len(source_text)]


def _field_evidence(field: str, value: Any, source_text: str, text: str, method: str) -> Dict[str, Any]:
    return {
        "field": field,
        "value": value,
        "source_text": source_text,
        "source_span": _source_span(text, source_text),
        "method": method,
    }


def _primary_action(actions: Sequence[tuple[str, str, int]]) -> str:
    action_names = {item[0] for item in actions}
    if "compare" in action_names:
        return "compare"
    if "analyze" in action_names:
        return "analyze"
    if "summarize" in action_names:
        return "summarize"
    if "locate" in action_names:
        return "locate"
    if "lookup" in action_names:
        return "lookup"
    if "generate_report" in action_names:
        return "generate_report"
    return ""


def _clean_compare_target(value: str) -> str:
    text = _clean(value).strip(" ：:，,。；;？?")
    text = _LEADING_REQUEST_RE.sub("", text).strip()
    text = re.split(r"(?:谁|哪个|哪家)(?:的)?", text, maxsplit=1)[0].strip()
    text = re.sub(r"(?:19|20)\d{2}年(?:度)?", "", text).strip()
    text = _TRAILING_REQUEST_RE.sub("", text).strip(" 的：:，,。；;？?")
    return text


def _compare_targets(question: str) -> List[str]:
    first_clause = _CLAUSE_END_RE.split(question, maxsplit=1)[0]
    parts = _COMPARE_CONNECTOR_RE.split(first_clause, maxsplit=1)
    if len(parts) != 2:
        return []
    return _unique(_clean_compare_target(item) for item in parts)


def _looks_like_company(value: str, registry: CompanyRegistry) -> bool:
    text = _clean(value)
    if not text or _YEAR_RE.fullmatch(text):
        return False
    if registry.resolve_known(company_name=text) is not None:
        return True
    company_hints = ("公司", "集团", "科技", "电子", "半导体", "国际", "股份", "有限")
    if any(hint in text for hint in company_hints):
        return True
    return 4 <= len(text) <= 20 and all(not char.isdigit() for char in text)


def _company_mentions(question: str, registry: CompanyRegistry, compare_targets: Sequence[str]) -> List[str]:
    companies = [profile.company_name for profile in registry.match_all(question)]
    for target in compare_targets:
        if _looks_like_company(target, registry):
            known = registry.resolve_known(company_name=target)
            companies.append(known.company_name if known is not None else target)
    for match in _COMPANY_BEFORE_METRIC_RE.finditer(question):
        candidate = _clean_compare_target(match.group("company"))
        if _COMPARE_CONNECTOR_RE.search(candidate):
            continue
        if _looks_like_company(candidate, registry):
            known = registry.resolve_known(company_name=candidate)
            companies.append(known.company_name if known is not None else candidate)
    return _unique(companies)


class HardSignalExtractor:
    def __init__(self, company_registry: CompanyRegistry) -> None:
        self.company_registry = company_registry

    def extract(self, question: str) -> Dict[str, Any]:
        text = _clean(question)
        lowered = text.lower()
        evidence: List[Dict[str, Any]] = []

        matched_actions: List[tuple[str, str, int]] = []
        for action, terms in ACTION_TERMS.items():
            for term in terms:
                index = lowered.find(term.lower())
                if index < 0:
                    continue
                matched_actions.append((action, term, index))
                evidence.append(_field_evidence("action", action, text[index : index + len(term)], text, "action_dictionary"))
                break
        if not any(item[0] == "compare" for item in matched_actions):
            comparison_match = re.search(r"(?:谁|哪个|哪家).{0,16}更", text)
            if comparison_match:
                matched_actions.append(("compare", comparison_match.group(0), comparison_match.start()))
                evidence.append(
                    _field_evidence(
                        "action",
                        "compare",
                        comparison_match.group(0),
                        text,
                        "action_pattern",
                    )
                )
        if not any(item[0] == "generate_report" for item in matched_actions):
            report_action_match = re.search(r"(?:生成|撰写|输出|形成|整理成).{0,8}报告", text)
            if report_action_match:
                matched_actions.append(("generate_report", report_action_match.group(0), report_action_match.start()))
                evidence.append(
                    _field_evidence(
                        "action",
                        "generate_report",
                        report_action_match.group(0),
                        text,
                        "action_pattern",
                    )
                )
        matched_actions.sort(key=lambda item: item[2])
        primary_action = _primary_action(matched_actions)

        metrics: List[str] = []
        for canonical, aliases in METRIC_ALIASES.items():
            for alias in aliases:
                if alias in text:
                    metrics.append(canonical)
                    evidence.append(_field_evidence("metrics", canonical, alias, text, "financial_metric_dictionary"))
                    break

        domain_objects: List[str] = []
        for canonical, aliases in DOMAIN_OBJECT_ALIASES.items():
            for alias in aliases:
                if alias in text:
                    domain_objects.append(canonical)
                    evidence.append(_field_evidence("domain_objects", canonical, alias, text, "domain_glossary"))
                    break

        compare_targets = _compare_targets(text) if primary_action == "compare" else []
        companies = _company_mentions(text, self.company_registry, compare_targets)
        for company in companies:
            source = next((name for name in [company, *self._company_aliases(company)] if name and name in text), company)
            evidence.append(_field_evidence("companies", company, source, text, "company_registry_or_entity_pattern"))

        periods = _unique(_YEAR_RE.findall(text))
        for period in periods:
            evidence.append(_field_evidence("periods", period, period, text, "year_pattern"))

        need_citation = next((term for term in CITATION_REQUIREMENT_TERMS if term.lower() in lowered), "")
        if not need_citation:
            citation_match = re.search(r"(?:给出|提供|注明|附上|给我).{0,3}(?:出处|来源|页码|依据)", text)
            need_citation = citation_match.group(0) if citation_match else ""
        if need_citation:
            evidence.append(_field_evidence("requirements.need_citation", True, need_citation, text, "requirement_dictionary"))

        report_term = next((term for term in REPORT_TERMS if term in text), "")
        short_term = next((term for term in SHORT_TERMS if term in text), "")
        table_output_term = next((term for term in TABLE_OUTPUT_TERMS if term in text), "")
        if report_term:
            output_format = "short_report" if short_term else "report"
            evidence.append(_field_evidence("output_format", output_format, report_term, text, "output_dictionary"))
        elif table_output_term:
            output_format = "table"
            evidence.append(_field_evidence("output_format", output_format, table_output_term, text, "output_dictionary"))
        else:
            output_format = "answer"

        secondary_actions: List[str] = []
        if report_term and primary_action != "generate_report":
            secondary_actions.append("generate_report")
        if need_citation and primary_action != "locate":
            secondary_actions.append("locate_evidence")

        evidence_modes: List[str] = []
        if metrics or domain_objects:
            evidence_modes.append("table")
        if need_citation:
            evidence_modes.append("source")

        if not primary_action and (metrics or domain_objects):
            primary_action = "lookup"
        if not primary_action and text:
            primary_action = ""

        return {
            "primary_action": primary_action,
            "secondary_actions": secondary_actions,
            "domain_objects": _unique(domain_objects),
            "evidence_modes": evidence_modes,
            "output_format": output_format,
            "requirements": {
                "need_citation": bool(need_citation),
                "citation_mode": "source_for_answer" if need_citation else "",
                "length": "short" if short_term else "normal",
            },
            "slots": {
                "companies": companies,
                "periods": periods,
                "metrics": _unique(metrics),
                "compare_targets": compare_targets,
            },
            "routing_state": "ready" if primary_action else "needs_semantic_route",
            "field_evidence": evidence,
        }

    def _company_aliases(self, company_name: str) -> List[str]:
        profile = self.company_registry.resolve_known(company_name=company_name)
        if profile is None:
            return []
        return [profile.company_name, *profile.aliases]


def load_intent_router_config(path: str | Path | None = None) -> Dict[str, Any]:
    source = Path(path).expanduser() if path else DEFAULT_INTENT_ROUTER_PATH
    loaded = load_yaml_file(source)
    config = loaded.get("intent_router", loaded) if isinstance(loaded, Mapping) else {}
    return dict(config) if isinstance(config, Mapping) else {}


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(float(left[index]) * float(right[index]) for index in range(size))
    left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(size)))
    right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(size)))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class SemanticRoute:
    candidates: tuple[Dict[str, Any], ...]
    decision: str
    top_query_type: str
    top_score: float
    margin: float
    provider: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [dict(item) for item in self.candidates],
            "decision": self.decision,
            "top_query_type": self.top_query_type,
            "top_score": self.top_score,
            "margin": self.margin,
            "provider": self.provider,
        }


class SemanticSkillRouter:
    def __init__(self, embedding_service: Any, config: Mapping[str, Any] | None = None) -> None:
        self.embedding_service = embedding_service
        self.config = dict(config or load_intent_router_config())
        self.enabled = bool(self.config.get("enabled", True))
        self.top_k = max(1, int(self.config.get("top_k", 3)))
        self.accept_threshold = float(self.config.get("accept_threshold", 0.58))
        self.reject_threshold = float(self.config.get("reject_threshold", 0.32))
        self.margin_threshold = float(self.config.get("margin_threshold", 0.08))
        self.llm_fallback_enabled = bool(self.config.get("llm_fallback_enabled", True))
        raw_prototypes = self.config.get("prototypes") if isinstance(self.config.get("prototypes"), Mapping) else {}
        self.prototype_texts = {
            str(query_type): "；".join(_unique(examples if isinstance(examples, list) else [examples]))
            for query_type, examples in raw_prototypes.items()
        }
        self._prototype_vectors: Dict[str, List[float]] | None = None
        self._prototype_lock = asyncio.Lock()

    async def _load_prototype_vectors(self) -> Dict[str, List[float]]:
        if self._prototype_vectors is not None:
            return self._prototype_vectors
        async with self._prototype_lock:
            if self._prototype_vectors is not None:
                return self._prototype_vectors
            query_types = list(self.prototype_texts)
            vectors = await self.embedding_service.embed_texts(
                [self.prototype_texts[query_type] for query_type in query_types],
                use_cache=True,
                chunk_text=False,
                max_concurrency=max(1, len(query_types)),
            )
            self._prototype_vectors = {
                query_type: list(vector)
                for query_type, vector in zip(query_types, vectors)
            }
            return self._prototype_vectors

    async def route(self, question: str) -> SemanticRoute:
        if not self.enabled or not self.prototype_texts:
            return SemanticRoute(tuple(), "disabled", "", 0.0, 0.0, "disabled")

        prototype_task = asyncio.create_task(self._load_prototype_vectors())
        question_task = asyncio.create_task(
            self.embedding_service.embed_text(question, use_cache=True, chunk_text=False)
        )
        prototype_vectors, question_vector = await asyncio.gather(prototype_task, question_task)
        scored = sorted(
            (
                {
                    "query_type": query_type,
                    "score": round(_cosine_similarity(question_vector, vector), 6),
                }
                for query_type, vector in prototype_vectors.items()
            ),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        top = scored[0] if scored else {"query_type": "", "score": 0.0}
        second_score = float(scored[1]["score"]) if len(scored) > 1 else 0.0
        top_score = float(top["score"])
        margin = round(top_score - second_score, 6)
        if top_score < self.reject_threshold:
            decision = "no_match"
        elif top_score >= self.accept_threshold and margin >= self.margin_threshold:
            decision = "accept"
        else:
            decision = "ambiguous"
        return SemanticRoute(
            candidates=tuple(scored[: self.top_k]),
            decision=decision,
            top_query_type=str(top["query_type"]),
            top_score=round(top_score, 6),
            margin=margin,
            provider=str(getattr(self.embedding_service, "provider_name", "unknown")),
        )


def query_type_from_frame(frame: Mapping[str, Any], semantic_route: Mapping[str, Any] | None = None) -> str:
    action = _clean(frame.get("primary_action"))
    slots = frame.get("slots") if isinstance(frame.get("slots"), Mapping) else {}
    metrics = list(slots.get("metrics") or [])
    domain_objects = list(frame.get("domain_objects") or [])
    if action == "lookup" and (metrics or domain_objects):
        return "table_qa"
    if action == "analyze":
        return "table_qa" if (metrics or domain_objects or "table" in list(frame.get("evidence_modes") or [])) else "summarization"
    if action in ACTION_TO_QUERY_TYPE:
        return ACTION_TO_QUERY_TYPE[action]
    route = semantic_route or {}
    if route.get("decision") == "accept" and route.get("top_query_type"):
        return str(route["top_query_type"])
    return "ambiguous_query"


def secondary_query_types(frame: Mapping[str, Any], primary_query_type: str) -> List[str]:
    result: List[str] = []
    slots = frame.get("slots") if isinstance(frame.get("slots"), Mapping) else {}
    requirements = frame.get("requirements") if isinstance(frame.get("requirements"), Mapping) else {}
    if (slots.get("metrics") or frame.get("domain_objects")) and primary_query_type not in {"table_qa", "fact_lookup"}:
        result.append("table_qa")
    if requirements.get("need_citation") and primary_query_type != "citation_locate":
        result.append("citation_locate")
    if frame.get("output_format") in {"report", "short_report"} and primary_query_type != "report_generation":
        result.append("report_generation")
    return result


def merge_structured_frame(base: Mapping[str, Any], incoming: Mapping[str, Any] | None) -> Dict[str, Any]:
    result = {
        "primary_action": _clean(base.get("primary_action")),
        "secondary_actions": _unique(base.get("secondary_actions") or []),
        "domain_objects": _unique(base.get("domain_objects") or []),
        "evidence_modes": _unique(base.get("evidence_modes") or []),
        "output_format": _clean(base.get("output_format")) or "answer",
        "requirements": dict(base.get("requirements") or {}) if isinstance(base.get("requirements"), Mapping) else {},
        "slots": dict(base.get("slots") or {}) if isinstance(base.get("slots"), Mapping) else {},
        "routing_state": _clean(base.get("routing_state")) or "needs_semantic_route",
        "field_evidence": list(base.get("field_evidence") or []),
    }
    if not isinstance(incoming, Mapping):
        return result

    if not result["primary_action"] and incoming.get("primary_action"):
        result["primary_action"] = _clean(incoming.get("primary_action"))
    for key in ("secondary_actions", "domain_objects", "evidence_modes"):
        result[key] = _unique([*result.get(key, []), *(incoming.get(key) or [])])
    if result["output_format"] == "answer" and incoming.get("output_format"):
        result["output_format"] = _clean(incoming.get("output_format"))
    if isinstance(incoming.get("requirements"), Mapping):
        for key, value in incoming["requirements"].items():
            if result["requirements"].get(key) in (None, "", False):
                result["requirements"][key] = value
    if isinstance(incoming.get("slots"), Mapping):
        for key, value in incoming["slots"].items():
            current = result["slots"].get(key)
            if isinstance(value, list):
                result["slots"][key] = _unique([*(current or []), *value])
            elif current in (None, "") and value not in (None, ""):
                result["slots"][key] = value
    if incoming.get("routing_state") and result["routing_state"] == "needs_semantic_route":
        result["routing_state"] = _clean(incoming.get("routing_state"))
    return result
