from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping

from utils.content_normalizer import normalize_whitespace


REPORT_TYPE_ALIASES: Dict[str, tuple[str, ...]] = {
    "annual_report": ("年度报告", "年度财报", "年报", "annual report"),
    "quarterly_report": ("季度报告", "季度财报", "季报", "quarterly report"),
    "semiannual_report": ("半年度报告", "半年度财报", "半年报", "中报", "interim report"),
}

STATEMENT_TYPE_ALIASES: Dict[str, tuple[str, ...]] = {
    "balance_sheet": ("合并资产负债表", "资产负债表", "balance sheet"),
    "income_statement": ("合并利润表", "利润表", "损益表", "income statement"),
    "cash_flow_statement": ("合并现金流量表", "现金流量表", "现金流表", "cash flow statement"),
}

LOCATION_REQUIREMENT_TERMS = (
    "定位原文",
    "原文位置",
    "具体位置",
    "所在位置",
    "哪一页",
    "页码",
    "第几页",
    "原文在哪",
)

_FULL_YEAR_RE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})(?!\d)\s*(?:年(?:度)?|(?=Q[1-4]\b))?",
    re.IGNORECASE,
)
_FY_YEAR_RE = re.compile(r"\bFY\s*(?P<year>(?:19|20)\d{2})\b", re.IGNORECASE)
_SHORT_YEAR_RE = re.compile(r"(?<!\d)(?P<year>\d{2})\s*年(?:度)?")
_QUARTER_RE = re.compile(
    r"(?:(?P<year>(?:19|20)\d{2})\s*年?\s*)?"
    r"(?:第?\s*(?P<cn_quarter>[一二三四1-4])\s*季度|Q\s*(?P<q_quarter>[1-4]))",
    re.IGNORECASE,
)
_HALF_YEAR_RE = re.compile(
    r"(?:(?P<year>(?:19|20)\d{2})\s*年?\s*)?(?P<half>上半年|下半年|半年度|半年)",
)
_PAGE_RE = re.compile(r"第\s*(?P<page>\d{1,5})\s*页")
_DOCUMENT_ID_RE = re.compile(r"(?<![A-Za-z0-9])doc[_-][A-Za-z0-9_-]+", re.IGNORECASE)
_DOCUMENT_NAME_RE = re.compile(r"《(?P<name>[^《》]{2,100}(?:报告|财报))》")
_PERCENT_RE = re.compile(
    r"(?P<operator>超过|高于|大于|不少于|至少|低于|小于|不超过|至多|约|大约)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|％|个百分点)",
)
_AMOUNT_RE = re.compile(
    r"(?P<operator>超过|高于|大于|不少于|至少|低于|小于|不超过|至多|约|大约)?\s*"
    r"(?P<value>\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿]+)\s*"
    r"(?P<unit>亿元|万元|元)",
)

_QUARTER_DIGITS = {"一": "1", "二": "2", "三": "3", "四": "4"}
_RELATIVE_PERIODS = {
    "今年": 0,
    "本年": 0,
    "去年": -1,
    "上年": -1,
    "前年": -2,
}
_OPERATOR_ALIASES = {
    "超过": "gt",
    "高于": "gt",
    "大于": "gt",
    "不少于": "gte",
    "至少": "gte",
    "低于": "lt",
    "小于": "lt",
    "不超过": "lte",
    "至多": "lte",
    "约": "approx",
    "大约": "approx",
}
_AMOUNT_MULTIPLIERS = {"元": 1, "万元": 10_000, "亿元": 100_000_000}
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CN_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}


def _clean(value: Any) -> str:
    return normalize_whitespace(str(value or ""), preserve_newlines=False)


def _unique(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _reference_date(value: date | datetime | str | None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return date.today()


def _evidence(
    field: str,
    value: Any,
    source_text: str,
    text: str,
    method: str,
    **metadata: Any,
) -> Dict[str, Any]:
    start = text.find(source_text)
    payload: Dict[str, Any] = {
        "field": field,
        "value": value,
        "source_text": source_text,
        "source_span": [start, start + len(source_text)] if start >= 0 else [],
        "method": method,
    }
    payload.update({key: value for key, value in metadata.items() if value is not None})
    return payload


def _alias_matches(
    text: str,
    aliases: Mapping[str, Iterable[str]],
    field: str,
    method: str,
) -> tuple[List[str], List[Dict[str, Any]]]:
    lowered = text.lower()
    values: List[str] = []
    evidence: List[Dict[str, Any]] = []
    occupied: List[tuple[int, int]] = []
    candidates = sorted(
        (
            (canonical, alias)
            for canonical, names in aliases.items()
            for alias in names
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for canonical, alias in candidates:
        start = 0
        while True:
            index = lowered.find(alias.lower(), start)
            if index < 0:
                break
            end = index + len(alias)
            start = end
            if any(index < used_end and end > used_start for used_start, used_end in occupied):
                continue
            source_text = text[index:end]
            values.append(canonical)
            occupied.append((index, end))
            evidence.append(_evidence(field, canonical, source_text, text, method))
            break
    return _unique(values), evidence


def _parse_chinese_number(value: str) -> float | None:
    if not value:
        return None
    if all(char in _CN_DIGITS for char in value):
        return float("".join(str(_CN_DIGITS[char]) for char in value))

    total = 0
    section = 0
    number = 0
    for char in value:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _CN_SMALL_UNITS:
            unit = _CN_SMALL_UNITS[char]
            section += (number or 1) * unit
            number = 0
        elif char in _CN_LARGE_UNITS:
            section += number
            total += (section or 1) * _CN_LARGE_UNITS[char]
            section = 0
            number = 0
        else:
            return None
    return float(total + section + number)


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return _parse_chinese_number(value)


def _condition_operator(value: str | None) -> str:
    return _OPERATOR_ALIASES.get(str(value or ""), "eq")


def extract_deterministic_slots(
    question: str,
    *,
    reference_date: date | datetime | str | None = None,
) -> Dict[str, Any]:
    """Extract only bounded, auditable values from a query.

    User-requested document/page references deliberately use names that are
    distinct from executor-produced document_ids/page_numbers.
    """

    text = _clean(question)
    reference = _reference_date(reference_date)
    evidence: List[Dict[str, Any]] = []

    report_types, report_evidence = _alias_matches(
        text,
        REPORT_TYPE_ALIASES,
        "report_types",
        "report_type_dictionary",
    )
    statement_types, statement_evidence = _alias_matches(
        text,
        STATEMENT_TYPE_ALIASES,
        "statement_types",
        "statement_type_dictionary",
    )
    evidence.extend(report_evidence)
    evidence.extend(statement_evidence)

    periods: List[str] = []
    for match in _FY_YEAR_RE.finditer(text):
        value = match.group("year")
        periods.append(value)
        evidence.append(_evidence("periods", value, match.group(0), text, "fiscal_year_parser"))
    for match in _FULL_YEAR_RE.finditer(text):
        value = match.group("year")
        periods.append(value)
        evidence.append(_evidence("periods", value, match.group(0), text, "year_parser"))
    for match in _SHORT_YEAR_RE.finditer(text):
        short_year = int(match.group("year"))
        value = str(2000 + short_year if short_year <= 69 else 1900 + short_year)
        periods.append(value)
        evidence.append(_evidence("periods", value, match.group(0), text, "short_year_parser"))
    for term, offset in _RELATIVE_PERIODS.items():
        if term not in text:
            continue
        value = str(reference.year + offset)
        periods.append(value)
        evidence.append(
            _evidence(
                "periods",
                value,
                term,
                text,
                "relative_year_parser",
                reference_date=reference.isoformat(),
            )
        )

    quarters: List[str] = []
    for match in _QUARTER_RE.finditer(text):
        quarter = match.group("q_quarter") or _QUARTER_DIGITS.get(match.group("cn_quarter"), match.group("cn_quarter"))
        value = f"{match.group('year')}Q{quarter}" if match.group("year") else f"Q{quarter}"
        quarters.append(value)
        periods.append(value)
        evidence.append(_evidence("quarters", value, match.group(0), text, "quarter_parser"))

    half_years: List[str] = []
    for match in _HALF_YEAR_RE.finditer(text):
        half = "H1" if match.group("half") in {"上半年", "半年度", "半年"} else "H2"
        value = f"{match.group('year')}{half}" if match.group("year") else half
        half_years.append(value)
        periods.append(value)
        evidence.append(_evidence("half_years", value, match.group(0), text, "half_year_parser"))

    requested_pages: List[int] = []
    for match in _PAGE_RE.finditer(text):
        value = int(match.group("page"))
        requested_pages.append(value)
        evidence.append(_evidence("requested_pages", value, match.group(0), text, "page_parser"))

    document_references: List[str] = []
    for match in _DOCUMENT_ID_RE.finditer(text):
        value = match.group(0)
        document_references.append(value)
        evidence.append(_evidence("document_references", value, value, text, "document_id_parser"))

    document_names: List[str] = []
    for match in _DOCUMENT_NAME_RE.finditer(text):
        value = match.group("name").strip()
        document_names.append(value)
        evidence.append(_evidence("document_names", value, match.group(0), text, "document_name_parser"))

    numeric_conditions: List[Dict[str, Any]] = []
    for match in _PERCENT_RE.finditer(text):
        value = float(match.group("value"))
        condition = {
            "kind": "percentage",
            "operator": _condition_operator(match.group("operator")),
            "value": value,
            "unit": "percentage_point" if match.group("unit") == "个百分点" else "percent",
            "raw_text": match.group(0).strip(),
        }
        numeric_conditions.append(condition)
        evidence.append(_evidence("numeric_conditions", condition, match.group(0), text, "percentage_parser"))
    for match in _AMOUNT_RE.finditer(text):
        parsed = _number(match.group("value"))
        if parsed is None:
            continue
        unit = match.group("unit")
        condition = {
            "kind": "amount",
            "operator": _condition_operator(match.group("operator")),
            "value": parsed * _AMOUNT_MULTIPLIERS[unit],
            "unit": "CNY",
            "raw_text": match.group(0).strip(),
        }
        numeric_conditions.append(condition)
        evidence.append(_evidence("numeric_conditions", condition, match.group(0), text, "amount_parser"))

    location_term = next((term for term in LOCATION_REQUIREMENT_TERMS if term in text), "")
    if requested_pages and not location_term:
        location_term = next(
            (
                item["source_text"]
                for item in evidence
                if item.get("field") == "requested_pages"
            ),
            "",
        )
    if location_term:
        evidence.append(
            _evidence(
                "requirements.need_location",
                True,
                location_term,
                text,
                "location_requirement_dictionary",
            )
        )

    return {
        "slots": {
            "periods": _unique(periods),
            "quarters": _unique(quarters),
            "half_years": _unique(half_years),
            "report_types": report_types,
            "statement_types": statement_types,
            "requested_pages": _unique(requested_pages),
            "document_references": _unique(document_references),
            "document_names": _unique(document_names),
            "numeric_conditions": numeric_conditions,
        },
        "requirements": {
            "need_location": bool(location_term),
            "need_citation": bool(location_term),
        },
        "field_evidence": evidence,
        "reference_date": reference.isoformat(),
    }
