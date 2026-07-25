from __future__ import annotations

import re

from service.agent.schemas import normalize_query_type
from utils.content_normalizer import normalize_whitespace

_INFORMATION_EXTRACTION_KEYWORDS = {
    "中文名称",
    "中文简称",
    "法定代表人",
    "注册地址",
    "办公地址",
    "公司网址",
    "电子信箱",
    "股票简称",
    "股票代码",
    "上市板块",
    "是多少",
    "是什么",
    "多少",
    "查询",
    "查一下",
    "告诉我",
    "extract",
    "lookup",
}

_CALCULATION_KEYWORDS = {
    "计算",
    "算一下",
    "算出",
    "增长率",
    "增幅",
    "下降幅度",
    "增长了多少",
    "下降了多少",
    "增加了多少",
    "减少了多少",
    "差额",
    "calculate",
    "calculation",
}

_COMPARE_KEYWORDS = {
    "对比",
    "比较",
    "相比",
    "差异",
    "差距",
    "区别",
    "谁更",
    "哪个更",
    "哪家更",
    "versus",
    "vs",
    "compare",
}

_ANALYSIS_KEYWORDS = {
    "分析",
    "解读",
    "原因",
    "为什么",
    "影响",
    "意味着什么",
    "说明什么",
    "能看出什么",
    "趋势",
    "怎么看",
    "评价",
    "判断",
    "analyze",
    "analysis",
    "explain",
    "why",
}

_SUMMARY_KEYWORDS = {
    "总结",
    "概述",
    "摘要",
    "归纳",
    "概括",
    "梳理",
    "overview",
    "summary",
    "summarize",
}

_TABLE_KEYWORDS = {
    "表格",
    "指标",
    "数据",
    "数值",
    "同比",
    "环比",
    "毛利率",
    "营业收入",
    "营业成本",
    "净利润",
    "现金流量净额",
    "每股收益",
    "研发投入",
    "非经常性损益",
    "货币资金",
    "应收账款",
    "参数",
    "table",
    "metric",
}

_YEAR_RE = re.compile(r"(19|20)\d{2}")

_FINANCIAL_TABLE_TERMS = {
    "营业收入",
    "营业成本",
    "主营业务收入",
    "归属于上市公司股东的净利润",
    "净利润",
    "现金流量净额",
    "经营活动产生的现金流量净额",
    "基本每股收益",
    "稀释每股收益",
    "每股收益",
    "研发投入",
    "研发投入合计",
    "研发投入总额占营业收入比例",
    "非经常性损益",
    "委托他人投资或管理资产的损益",
    "货币资金",
    "应收账款",
    "账面价值",
    "期末余额",
    "经销",
    "直销",
}

_VALUE_QUERY_HINTS = {
    "多少",
    "分别",
    "合计",
    "金额",
    "余额",
    "比例",
    "同比",
    "变动比例",
    "期末",
    "年末",
    "分季度",
    "分类",
    "分解",
}

_EXPLANATION_HINTS = {
    "原因",
    "为什么",
    "影响",
    "说明",
    "意味着",
    "分析",
    "解读",
}


def _contains_any(text: str, words: set[str]) -> bool:
    return any(word in text for word in words)


def is_financial_table_query(question: str) -> bool:
    """Return whether the request needs table-shaped financial evidence.

    This is an evidence-mode decision, not an intent classification.
    """

    normalized = normalize_whitespace(question, preserve_newlines=False).lower()
    if not normalized:
        return False

    has_metric = _contains_any(normalized, _FINANCIAL_TABLE_TERMS) or _contains_any(normalized, _TABLE_KEYWORDS)
    has_value_intent = _contains_any(normalized, _VALUE_QUERY_HINTS) or bool(_YEAR_RE.search(normalized))
    asks_explanation = _contains_any(normalized, _EXPLANATION_HINTS)

    if asks_explanation and "多少" not in normalized and "分别" not in normalized:
        return False
    return has_metric and has_value_intent


def _strong_intent_signals(normalized: str) -> list[str]:
    signals: list[str] = []
    for query_type, keywords in (
        ("metric_calculation", _CALCULATION_KEYWORDS),
        ("comparison", _COMPARE_KEYWORDS),
        ("analysis", _ANALYSIS_KEYWORDS),
        ("summarization", _SUMMARY_KEYWORDS),
    ):
        if _contains_any(normalized, keywords):
            signals.append(query_type)
    # A request for a derived numeric result can mention the two scopes being
    # compared. Its terminal answer is still a calculation, not a qualitative
    # comparison.
    if "metric_calculation" in signals and "comparison" in signals:
        signals.remove("comparison")
    return signals


def classify_query_type(question: str) -> str:
    """Classify one annual-report question into exactly one primary intent.

    Compound requests with multiple strong actions are intentionally returned as
    ambiguous during the single-intent phase instead of silently dropping an
    action through fixed precedence.
    """

    normalized = normalize_whitespace(question, preserve_newlines=False).lower()
    if not normalized or len(normalized) <= 4:
        return "ambiguous_query"

    signals = _strong_intent_signals(normalized)
    if len(signals) > 1:
        return "ambiguous_query"
    if signals:
        return signals[0]

    if "这个" in normalized or "那个" in normalized:
        if not _YEAR_RE.search(normalized):
            return "ambiguous_query"

    return normalize_query_type("information_extraction")
