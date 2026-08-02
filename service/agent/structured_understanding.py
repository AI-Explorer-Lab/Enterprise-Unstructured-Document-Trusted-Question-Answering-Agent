from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from service.agent.company_registry import CompanyProfile, CompanyRegistry
from service.agent.deterministic_slots import extract_deterministic_slots
from utils.content_normalizer import normalize_whitespace


ACTION_TO_QUERY_TYPE = {
    "extract": "information_extraction",
    "calculate": "metric_calculation",
    "compare": "comparison",
    "analyze": "analysis",
    "summarize": "summarization",
}

QUERY_TYPE_TO_ACTION = {
    "information_extraction": "extract",
    "metric_calculation": "calculate",
    "comparison": "compare",
    "analysis": "analyze",
    "summarization": "summarize",
}

ACTION_TERMS: Dict[str, tuple[str, ...]] = {
    "compare": ("对比", "比较", "相比", "差异", "差距", "区别", "谁更", "哪个更", "vs", "versus"),
    "calculate": ("计算", "算一下", "算出", "增长率", "增幅", "下降幅度", "差额", "增长了多少", "下降了多少", "增加了多少", "减少了多少"),
    "analyze": ("分析", "解读", "原因", "为什么", "影响", "怎么看", "说明什么", "意味着什么", "能看出什么", "趋势"),
    "summarize": ("总结", "概述", "概括", "归纳", "梳理"),
    "extract": (
        "查一下",
        "查询",
        "告诉我",
        "是多少",
        "是什么",
        "多少",
        "原文在哪",
        "在哪一页",
        "哪一页提到",
        "找到原文",
        "how much",
        "how long",
        "what is",
        "find",
        "extract",
    ),
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
    "扣除非经常性损益后的净利润": ("扣非净利润", "扣除非经常性损益后的净利润"),
    "经营活动产生的现金流量净额": ("经营活动产生的现金流量净额", "经营现金流", "经营活动现金流"),
    "毛利率": ("毛利率",),
    "研发投入": ("研发投入", "研发费用"),
    "货币资金": ("货币资金",),
    "应收账款": ("应收账款",),
    "资产总额": ("资产总额", "总资产"),
    "负债合计": ("负债合计", "总负债"),
    "资产负债率": ("资产负债率",),
    "流动比率": ("流动比率",),
    "速动比率": ("速动比率",),
    "净资产收益率": ("净资产收益率", "ROE"),
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
    r"^(?:请告诉我|告诉我|请|帮我|麻烦|想要|我要|请帮我|请你)?(?:对比|比较|分析|查询|查一下|看看)?(?:一下|下)?"
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


def _resolved_action_names(actions: Sequence[tuple[str, str, int]]) -> set[str]:
    action_names = {item[0] for item in actions}
    # Extraction wording often accompanies a stronger single action, for
    # example "分析……是什么" or "计算……是多少".
    if len(action_names) > 1:
        action_names.discard("extract")
    # A derived numeric answer can compare periods or companies as inputs.
    # Calculation is the terminal operation in that case.
    if "calculate" in action_names and "compare" in action_names:
        action_names.discard("compare")
    return action_names


def _primary_action(actions: Sequence[tuple[str, str, int]]) -> str:
    action_names = _resolved_action_names(actions)
    if len(action_names) == 1:
        return next(iter(action_names))
    # Multiple remaining actions are deliberately unresolved in the
    # single-intent phase.
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
    known_profiles = registry.match_all(question)
    companies = [profile.company_name for profile in known_profiles]
    for target in compare_targets:
        if _looks_like_company(target, registry):
            known = registry.resolve_known(company_name=target)
            companies.append(known.company_name if known is not None else target)
    for match in _COMPANY_BEFORE_METRIC_RE.finditer(question):
        candidate = _clean_compare_target(match.group("company"))
        if _COMPARE_CONNECTOR_RE.search(candidate):
            continue
        if _looks_like_company(candidate, registry):
            overlapping_known = next(
                (
                    profile
                    for profile in known_profiles
                    if any(
                        name and name in candidate
                        for name in [profile.company_name, *profile.aliases, profile.company_id]
                    )
                ),
                None,
            )
            if overlapping_known is not None:
                companies.append(overlapping_known.company_name)
                continue
            known = registry.resolve_known(company_name=candidate)
            companies.append(known.company_name if known is not None else candidate)
    return _unique(companies)


class HardSignalExtractor:
    def __init__(self, company_registry: CompanyRegistry) -> None:
        self.company_registry = company_registry

    def extract(
        self,
        question: str,
        *,
        reference_date: date | datetime | str | None = None,
    ) -> Dict[str, Any]:
        text = _clean(question)
        lowered = text.lower()
        evidence: List[Dict[str, Any]] = []
        deterministic = extract_deterministic_slots(
            text,
            reference_date=reference_date,
        )
        deterministic_slots = (
            deterministic.get("slots")
            if isinstance(deterministic.get("slots"), Mapping)
            else {}
        )
        evidence.extend(list(deterministic.get("field_evidence") or []))

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
        matched_actions.sort(key=lambda item: item[2])
        action_names = _resolved_action_names(matched_actions)
        has_action_conflict = len(action_names) > 1
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

        periods = _unique(deterministic_slots.get("periods") or _YEAR_RE.findall(text))
        raw_compare_targets = _compare_targets(text) if primary_action == "compare" else []
        # A connector between two explicit years describes a period comparison,
        # not two company names. Feeding those fragments into the permissive
        # company recognizer can turn domain wording such as
        # "三张财务报表的变化" into a fake company.
        company_compare_targets = raw_compare_targets if len(periods) < 2 else []
        companies = _company_mentions(text, self.company_registry, company_compare_targets)
        for company in companies:
            source = next((name for name in [company, *self._company_aliases(company)] if name and name in text), company)
            evidence.append(_field_evidence("companies", company, source, text, "company_registry_or_entity_pattern"))

        compare_targets = raw_compare_targets
        if primary_action == "compare":
            if len(companies) >= 2:
                compare_targets = list(companies)
            elif len(periods) >= 2:
                compare_targets = list(periods)
            elif len(_unique(metrics)) >= 2:
                compare_targets = _unique(metrics)

        deterministic_requirements = (
            deterministic.get("requirements")
            if isinstance(deterministic.get("requirements"), Mapping)
            else {}
        )
        need_location = bool(deterministic_requirements.get("need_location"))
        need_citation = next((term for term in CITATION_REQUIREMENT_TERMS if term.lower() in lowered), "")
        if not need_citation:
            citation_match = re.search(r"(?:给出|提供|注明|附上|给我).{0,3}(?:出处|来源|页码|依据)", text)
            need_citation = citation_match.group(0) if citation_match else ""
        if need_citation:
            evidence.append(_field_evidence("requirements.need_citation", True, need_citation, text, "requirement_dictionary"))
        need_citation_flag = bool(need_citation) or bool(
            deterministic_requirements.get("need_citation")
        )

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

        evidence_modes: List[str] = []
        if metrics or domain_objects:
            evidence_modes.append("table")
        if need_citation_flag:
            evidence_modes.append("source")

        if not primary_action and not has_action_conflict and (metrics or domain_objects):
            primary_action = "extract"

        return {
            "primary_action": primary_action,
            "domain_objects": _unique(domain_objects),
            "evidence_modes": evidence_modes,
            "output_format": output_format,
            "requirements": {
                "need_citation": need_citation_flag,
                "need_location": need_location,
                "citation_mode": "source_for_answer" if need_citation_flag else "",
                "length": "short" if short_term else "normal",
            },
            "slots": {
                "companies": companies,
                "periods": periods,
                "metrics": _unique(metrics),
                "compare_targets": compare_targets,
                "quarters": list(deterministic_slots.get("quarters") or []),
                "half_years": list(deterministic_slots.get("half_years") or []),
                "report_types": list(deterministic_slots.get("report_types") or []),
                "statement_types": list(deterministic_slots.get("statement_types") or []),
                "requested_pages": list(deterministic_slots.get("requested_pages") or []),
                "document_references": list(
                    deterministic_slots.get("document_references") or []
                ),
                "document_names": list(deterministic_slots.get("document_names") or []),
                "numeric_conditions": list(
                    deterministic_slots.get("numeric_conditions") or []
                ),
            },
            "routing_state": "ambiguous_action" if has_action_conflict else ("ready" if primary_action else "needs_semantic_route"),
            "field_evidence": evidence,
            "reference_date": deterministic.get("reference_date"),
        }

    def _company_aliases(self, company_name: str) -> List[str]:
        profile = self.company_registry.resolve_known(company_name=company_name)
        if profile is None:
            return []
        return [profile.company_name, *profile.aliases]


def query_type_from_frame(frame: Mapping[str, Any]) -> str:
    action = _clean(frame.get("primary_action"))
    return ACTION_TO_QUERY_TYPE.get(action, "ambiguous_query")


def merge_structured_frame(base: Mapping[str, Any], incoming: Mapping[str, Any] | None) -> Dict[str, Any]:
    result = {
        "primary_action": _clean(base.get("primary_action")),
        "domain_objects": _unique(base.get("domain_objects") or []),
        "evidence_modes": _unique(base.get("evidence_modes") or []),
        "output_format": _clean(base.get("output_format")) or "answer",
        "requirements": dict(base.get("requirements") or {}) if isinstance(base.get("requirements"), Mapping) else {},
        "slots": dict(base.get("slots") or {}) if isinstance(base.get("slots"), Mapping) else {},
        "routing_state": _clean(base.get("routing_state")) or "needs_semantic_route",
        "field_evidence": list(base.get("field_evidence") or []),
        "reference_date": _clean(base.get("reference_date")),
    }
    if not isinstance(incoming, Mapping):
        return result

    if not result["primary_action"] and incoming.get("primary_action"):
        result["primary_action"] = _clean(incoming.get("primary_action"))
    for key in ("domain_objects", "evidence_modes"):
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
    for item in list(incoming.get("field_evidence") or []):
        if isinstance(item, Mapping) and dict(item) not in result["field_evidence"]:
            result["field_evidence"].append(dict(item))
    if not result["reference_date"] and incoming.get("reference_date"):
        result["reference_date"] = _clean(incoming.get("reference_date"))
    return result
