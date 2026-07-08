from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping

from service.agent.company_registry import CompanyProfile, CompanyRegistry


YEAR_RE = re.compile(r"(?:19|20)\d{2}")
LATEST_TERMS = ("最新", "最近一年", "最近一期", "最新财报", "最近财报")
TREND_TERMS = ("近三年", "近两年", "历年", "这几年", "趋势", "变化", "对比", "增长情况", "逐年", "分别")
SENSITIVE_TERMS = (
    "营业收入",
    "营收",
    "净利润",
    "毛利率",
    "资产总额",
    "负债",
    "现金流",
    "研发费用",
    "非经常性损益",
    "每股收益",
    "同比",
    "增长率",
    "金额",
    "多少",
    "数值",
    "占比",
)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _as_values(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _years_from_values(values: Iterable[Any]) -> List[int]:
    years: List[int] = []
    for value in values:
        for match in YEAR_RE.findall(str(value or "")):
            year = int(match)
            if year not in years:
                years.append(year)
    return years


def _available_years(scopes: Iterable[Mapping[str, Any]], company_id: str = "") -> List[int]:
    years: List[int] = []
    for item in scopes:
        if company_id and _clean(item.get("company_id")) != company_id:
            continue
        for key in ("year", "years"):
            value = item.get(key)
            values = value if isinstance(value, list) else [value]
            for year in _years_from_values(values):
                if year not in years:
                    years.append(year)
    return sorted(years)


def _companies(scopes: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for item in scopes:
        company_id = _clean(item.get("company_id"))
        if company_id:
            result.setdefault(company_id, item)
    return result


def _trend_limit(question: str) -> int | None:
    if "近两年" in question:
        return 2
    if "近三年" in question:
        return 3
    return None


def is_year_sensitive_question(question: str, query_type: str, slots: Mapping[str, Any] | None = None) -> bool:
    text = _clean(question)
    slot_values = slots or {}
    if query_type == "table_qa":
        return True
    if _clean(slot_values.get("metric")):
        return True
    return any(term in text for term in SENSITIVE_TERMS)


@dataclass
class RetrievalScope:
    company_id: str = ""
    company_name: str = ""
    years: List[int] = field(default_factory=list)
    strict: bool = True
    source: str = "none"
    should_clarify: bool = False
    missing_slots: List[str] = field(default_factory=list)
    clarify_question: str = ""
    should_refuse: bool = False
    refuse_reason: str = ""
    refuse_message: str = ""
    unavailable_years: List[int] = field(default_factory=list)
    available_companies: List[Dict[str, Any]] = field(default_factory=list)
    available_years: List[int] = field(default_factory=list)

    def metadata_filter(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.company_id:
            payload["company_id"] = self.company_id
        if self.years:
            payload["year"] = list(self.years)
        return payload

    def as_dict(self) -> Dict[str, Any]:
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "years": list(self.years),
            "strict": self.strict,
            "source": self.source,
            "should_clarify": self.should_clarify,
            "missing_slots": list(self.missing_slots),
            "clarify_question": self.clarify_question,
            "should_refuse": self.should_refuse,
            "refuse_reason": self.refuse_reason,
            "refuse_message": self.refuse_message,
            "unavailable_years": list(self.unavailable_years),
            "available_companies": list(self.available_companies),
            "available_years": list(self.available_years),
            "metadata_filter": self.metadata_filter(),
        }


def resolve_retrieval_scope(
    *,
    question: str,
    query_type: str,
    slots: Mapping[str, Any] | None,
    conversation_focus: Mapping[str, Any] | None,
    document_scopes: Iterable[Mapping[str, Any]],
    company_registry: CompanyRegistry,
) -> RetrievalScope:
    scopes = [dict(item) for item in document_scopes if isinstance(item, Mapping)]
    focus = conversation_focus or {}
    slot_values = slots or {}
    companies = _companies(scopes)

    company: CompanyProfile | None = company_registry.match_question(question, scopes)
    source_parts: List[str] = []
    if company is not None:
        source_parts.append("explicit_company")
    if company is None:
        company = company_registry.resolve(company_name=_clean(slot_values.get("company") or slot_values.get("entity")))
        if company is not None:
            source_parts.append("slot_company")
    if company is None:
        company = company_registry.resolve(company_id=_clean(focus.get("company_id")), company_name=_clean(focus.get("company")))
        if company is not None:
            source_parts.append("conversation_context_company")
    if company is None and len(companies) == 1:
        only = next(iter(companies.values()))
        company = company_registry.resolve(company_id=_clean(only.get("company_id")), company_name=_clean(only.get("company_name")))
        if company is not None:
            source_parts.append("single_company_available")

    explicit_years = _years_from_values([question, *_as_values(slot_values.get("years")), slot_values.get("period")])
    years: List[int] = list(explicit_years)
    if years:
        source_parts.append("explicit_year")
    if not years:
        focus_years = _years_from_values([*_as_values(focus.get("years")), focus.get("period")])
        if focus_years:
            years = focus_years
            source_parts.append("conversation_context_year")

    company_id = company.company_id if company else ""
    available_years = _available_years(scopes, company_id=company_id)
    text = _clean(question)
    if not years and any(term in text for term in LATEST_TERMS) and available_years:
        years = [available_years[-1]]
        source_parts.append("latest_available")
    if not years and any(term in text for term in TREND_TERMS) and available_years:
        limit = _trend_limit(text)
        years = available_years[-limit:] if limit else available_years
        source_parts.append("trend_request")

    unavailable_years: List[int] = []
    refuse_message = ""
    if company_id and explicit_years and available_years:
        unavailable_years = [year for year in explicit_years if year not in available_years]
        if unavailable_years:
            available_hint = "、".join(str(year) for year in available_years)
            requested_hint = "、".join(str(year) for year in unavailable_years)
            company_hint = company.company_name if company else "该公司"
            refuse_message = (
                f"当前文档集中{company_hint}可用财报年份为 {available_hint}，"
                f"未找到 {requested_hint} 年报；不能使用其他年份的数据替代回答。"
            )

    missing: List[str] = []
    if len(companies) > 1 and not company_id:
        missing.append("company")
    if not years and is_year_sensitive_question(question, query_type, slot_values):
        missing.append("year")
    if not years and available_years and company_id and not missing:
        years = [available_years[-1]]
        source_parts.append("latest_default")

    company_options = [
        {"company_id": key, "company_name": _clean(value.get("company_name")) or key}
        for key, value in sorted(companies.items())
    ]
    clarify = ""
    if missing:
        pieces = []
        if "company" in missing:
            names = "、".join(item["company_name"] for item in company_options[:5])
            pieces.append(f"请先说明要查询哪家公司{f'，例如 {names}' if names else ''}")
        if "year" in missing:
            years_hint = "、".join(str(item) for item in available_years[-5:])
            pieces.append(f"请补充财报年份{f'，例如 {years_hint}' if years_hint else ''}")
        clarify = "；".join(pieces) + "。"

    return RetrievalScope(
        company_id=company_id,
        company_name=company.company_name if company else "",
        years=years,
        strict=True,
        source="+".join(source_parts) or "none",
        should_clarify=bool(missing),
        missing_slots=missing,
        clarify_question=clarify,
        should_refuse=bool(unavailable_years) and not missing,
        refuse_reason="unavailable_year" if unavailable_years and not missing else "",
        refuse_message=refuse_message if unavailable_years and not missing else "",
        unavailable_years=unavailable_years if not missing else [],
        available_companies=company_options,
        available_years=available_years,
    )
