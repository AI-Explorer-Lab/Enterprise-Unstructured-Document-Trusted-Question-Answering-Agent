from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
SOURCE_2024 = PROJECT_ROOT / "docs/test_docs/上海芯导电子科技股份有限公司2024年年度报告.pdf"
SOURCE_2025 = PROJECT_ROOT / "docs/json_docs/MinerU_上海芯导电子科技股份有限公司财报_2025__20260411100646.json"
DATASET_PATH = OUTPUT_DIR / "recall_eval_200.jsonl"
REVIEW_PATH = OUTPUT_DIR / "recall_eval_200_review.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = str(data or "").strip()
        if value:
            self.parts.append(value)


def _html_text(value: str) -> str:
    parser = _HTMLText()
    parser.feed(str(value or ""))
    return " ".join(parser.parts)


def _block_text(block: dict[str, Any]) -> str:
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            content = str(span.get("content") or "").strip()
            if content:
                parts.append(content)
            html = str(span.get("html") or "").strip()
            if html:
                parts.append(_html_text(html))
    for child in block.get("blocks", []):
        child_text = _block_text(child)
        if child_text:
            parts.append(child_text)
    return " ".join(parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", "", text)
    return text.replace("，", ",").replace("；", ";")


def _excerpt(page_text: str, anchor: str, answer_match: str) -> str:
    compact = re.sub(r"\s+", " ", str(page_text or "")).strip()
    positions = [
        position
        for position in (compact.find(anchor), compact.find(answer_match))
        if position >= 0
    ]
    center = min(positions) if positions else 0
    start = max(0, center - 180)
    end = min(len(compact), center + 620)
    return compact[start:end].strip()


@dataclass(frozen=True)
class Fact:
    question: str
    report_year: int
    page: int
    heading_path: str
    anchor: str
    expected_answer: str
    scenario: str
    evidence_type: str = "table"
    difficulty: str = "simple"
    answer_match: str = ""
    tags: tuple[str, ...] = ()


FACTS: list[Fact] = []


def add(
    question: str,
    report_year: int,
    page: int,
    heading_path: str,
    anchor: str,
    expected_answer: str,
    scenario: str,
    evidence_type: str = "table",
    difficulty: str = "simple",
    answer_match: str = "",
    tags: tuple[str, ...] = (),
) -> None:
    FACTS.append(
        Fact(
            question=question,
            report_year=report_year,
            page=page,
            heading_path=heading_path,
            anchor=anchor,
            expected_answer=expected_answer,
            scenario=scenario,
            evidence_type=evidence_type,
            difficulty=difficulty,
            answer_match=answer_match,
            tags=tags,
        )
    )


def add_2024_facts() -> None:
    year = 2024
    basic = "第二节 公司简介和主要财务指标"
    basic_rows = [
        ("芯导科技的公司中文全称是什么？", 7, "公司的中文名称", "上海芯导电子科技股份有限公司"),
        ("芯导科技的中文简称是什么？", 7, "公司的中文简称", "芯导科技"),
        ("芯导科技的英文全称是什么？", 7, "公司的外文名称", "Shanghai Prisemi Electronics Co.,Ltd."),
        ("芯导科技的英文简称是什么？", 7, "公司的外文名称缩写", "Prisemi"),
        ("芯导科技2024年年报披露的法定代表人是谁？", 7, "公司的法定代表人", "欧新华"),
        ("芯导科技的注册地址在哪里？", 7, "公司注册地址", "中国（上海）自由贸易试验区祖冲之路2277弄7号"),
        ("芯导科技办公地址对应的两个邮政编码是什么？", 7, "公司办公地址的邮政编码", "201210；201203"),
        ("芯导科技的公司网址是什么？", 7, "公司网址", "http://www.prisemi.com"),
        ("芯导科技的投资者联系邮箱是什么？", 7, "电子信箱", "investor@prisemi.com"),
        ("芯导科技2024年年报中的董事会秘书是谁？", 7, "董事会秘书", "兰芳云"),
        ("芯导科技2024年年报中的证券事务代表是谁？", 7, "证券事务代表", "闵雨琦"),
        ("芯导科技董事会秘书的联系电话是什么？", 7, "电话", "021-60753051"),
        ("芯导科技对外披露的传真号码是什么？", 7, "传真", "021-60870156"),
        ("芯导科技披露年度报告的媒体名称是什么？", 7, "公司披露年度报告的媒体名称及网址", "上海证券报"),
        ("芯导科技披露年度报告的证券交易所网址是什么？", 7, "公司披露年度报告的证券交易所网址", "www.sse.com.cn"),
        ("芯导科技年度报告的备置地点在哪里？", 7, "公司年度报告备置地点", "公司证券部"),
        ("芯导科技股票在哪个交易所及板块上市？", 7, "股票上市交易所及板块", "上海证券交易所科创板"),
        ("芯导科技的股票代码是多少？", 7, "股票代码", "688230"),
        ("芯导科技2024年聘请的境内会计师事务所是哪家？", 7, "天职国际会计师事务所", "天职国际会计师事务所（特殊普通合伙）"),
        ("芯导科技2024年持续督导的保荐机构是哪家？", 8, "报告期内履行持续督导职责的保荐机构", "国元证券股份有限公司"),
    ]
    for question, page, anchor, answer in basic_rows:
        add(question, year, page, basic, anchor, answer, "basic_fact", tags=("company_profile",))

    finance = "第二节 公司简介和主要财务指标 > 六、近三年主要会计数据和财务指标"
    finance_rows = [
        ("芯导科技2024年的营业收入是多少元？", "营业收入", "352,941,674.80"),
        ("芯导科技2024年营业收入同比增长多少？", "营业收入", "10.15%"),
        ("芯导科技2024年归属于上市公司股东的净利润是多少元？", "归属于上市公司股东的净利润", "111,639,491.31"),
        ("芯导科技2024年归母净利润同比增长多少？", "归属于上市公司股东的净利润", "15.70%"),
        ("芯导科技2024年扣除非经常性损益后的归母净利润是多少元？", "归属于上市公司股东的扣除非经常性损益的净利润", "58,607,032.90"),
        ("芯导科技2024年扣非归母净利润同比增长多少？", "归属于上市公司股东的扣除非经常性损益的净利润", "34.50%"),
        ("芯导科技2024年经营活动产生的现金流量净额是多少元？", "经营活动产生的现金流量净额", "84,749,309.08"),
        ("芯导科技2024年经营活动现金流量净额同比增长多少？", "经营活动产生的现金流量净额", "22.68%"),
        ("芯导科技2024年末归属于上市公司股东的净资产是多少元？", "归属于上市公司股东的净资产", "2,264,139,422.25"),
        ("芯导科技2024年末归母净资产同比增长多少？", "归属于上市公司股东的净资产", "1.87%"),
        ("芯导科技2024年末总资产是多少元？", "总资产", "2,327,935,658.62"),
        ("芯导科技2024年末总资产同比增长多少？", "总资产", "2.03%"),
        ("芯导科技2024年基本每股收益是多少元？", "基本每股收益", "0.95元/股"),
        ("芯导科技2024年稀释每股收益是多少元？", "稀释每股收益", "0.95元/股"),
        ("芯导科技2024年扣非后的基本每股收益是多少元？", "扣除非经常性损益后的基本每股收益", "0.50元/股"),
        ("芯导科技2024年加权平均净资产收益率是多少？", "加权平均净资产收益率", "4.97%"),
        ("芯导科技2024年扣非后的加权平均净资产收益率是多少？", "扣除非经常性损益后的加权平均净资产收益率", "2.61%"),
        ("芯导科技2024年研发投入占营业收入的比例是多少？", "研发投入占营业收入的比例", "10.02%"),
    ]
    for question, anchor, answer in finance_rows:
        add(
            question,
            year,
            8,
            finance,
            anchor,
            answer,
            "financial_metric",
            difficulty="medium" if "同比" in question else "simple",
            answer_match=re.sub(r"(元/股|元|%)$", "", answer),
            tags=("table", "financial_metric"),
        )

    quarter = "第二节 公司简介和主要财务指标 > 八、2024年分季度主要财务数据"
    quarter_values = {
        "营业收入": ["68,696,882.19", "87,111,101.46", "98,365,572.61", "98,768,118.54"],
        "归属于上市公司股东的净利润": ["24,468,672.39", "27,745,860.66", "30,409,406.45", "29,015,551.81"],
        "经营活动产生的现金流量净额": ["3,334,567.70", "17,874,925.89", "29,297,329.90", "34,242,485.59"],
    }
    quarter_names = ["第一季度", "第二季度", "第三季度", "第四季度"]
    for metric, values in quarter_values.items():
        for quarter_name, value in zip(quarter_names, values):
            add(
                f"芯导科技2024年{quarter_name}的{metric}是多少元？",
                year,
                9,
                quarter,
                metric,
                value,
                "quarterly_metric",
                tags=("table", "quarter"),
            )

    research = "第三节 管理层讨论与分析 > 核心技术与研发进展 > 报告期内获得的研发成果"
    research_rows = [
        ("截至2024年末，芯导科技现行有效知识产权共有多少项？", "现行有效知识产权累计", "120项"),
        ("截至2024年末，芯导科技拥有多少项有效发明专利？", "发明专利", "25项"),
        ("截至2024年末，芯导科技拥有多少项有效实用新型专利？", "实用新型", "36项"),
        ("截至2024年末，芯导科技拥有多少项集成电路布图设计专有权？", "集成电路布图设计专有权", "53项"),
        ("截至2024年末，芯导科技拥有多少项商标？", "商标", "6项"),
        ("芯导科技2024年新增授权知识产权多少项？", "新增授权知识产权", "17项"),
        ("芯导科技2024年新增申请发明专利多少项？", "发明专利", "15项"),
        ("芯导科技2024年新增获得发明专利多少项？", "发明专利", "5项"),
        ("芯导科技2024年新增申请实用新型专利多少项？", "实用新型专利", "2项"),
        ("芯导科技2024年新增获得实用新型专利多少项？", "实用新型专利", "4项"),
        ("芯导科技2024年其他知识产权新增申请多少项？", "其他", "15项"),
        ("芯导科技2024年其他知识产权新增获得多少项？", "其他", "8项"),
        ("芯导科技2024年各类知识产权新增申请合计多少项？", "合计", "32项"),
        ("芯导科技2024年各类知识产权新增获得合计多少项？", "合计", "17项"),
        ("芯导科技2024年费用化研发投入是多少元？", "费用化研发投入", "35,350,211.68"),
        ("芯导科技2024年资本化研发投入是多少元？", "资本化研发投入", "0"),
        ("芯导科技2024年研发投入合计是多少元？", "研发投入合计", "35,350,211.68"),
        ("芯导科技2024年研发投入较上年度变化多少？", "研发投入合计", "-18.12%"),
    ]
    for question, anchor, answer in research_rows:
        add(
            question,
            year,
            22,
            research,
            anchor,
            answer,
            "research_fact",
            difficulty="medium" if "变化" in question else "simple",
            answer_match=answer.rstrip("项%元"),
            tags=("table", "research"),
        )

    customers = "第三节 管理层讨论与分析 > 主要销售客户及主要供应商情况"
    customer_rows = [
        ("芯导科技2024年前五名客户销售额合计是多少万元？", 36, "前五名客户销售额", "17,108.22"),
        ("芯导科技2024年前五名客户销售额占年度销售总额多少？", 36, "前五名客户销售额", "48.47%"),
        ("芯导科技2024年第一大客户的销售额是多少万元？", 36, "客户一", "8,078.49"),
        ("芯导科技2024年第一大客户销售额占年度销售总额多少？", 36, "客户一", "22.89%"),
        ("芯导科技2024年第二大客户的销售额是多少万元？", 36, "客户二", "4,535.63"),
        ("芯导科技2024年第二大客户销售额占年度销售总额多少？", 36, "客户二", "12.85%"),
        ("芯导科技2024年前五名供应商采购额合计是多少万元？", 36, "前五名供应商采购额", "11,829.23"),
        ("芯导科技2024年前五名供应商采购额占年度采购总额多少？", 36, "前五名供应商采购额", "51.53%"),
        ("芯导科技2024年第一大供应商的采购额是多少万元？", 36, "供应商一", "3,953.52"),
        ("芯导科技2024年第一大供应商采购额占年度采购总额多少？", 36, "供应商一", "17.22%"),
        ("芯导科技2024年第二大供应商的采购额是多少万元？", 36, "供应商二", "3,375.25"),
        ("芯导科技2024年第二大供应商采购额占年度采购总额多少？", 36, "供应商二", "14.70%"),
    ]
    for question, page, anchor, answer in customer_rows:
        add(
            question,
            year,
            page,
            customers,
            anchor,
            answer,
            "customer_supplier",
            answer_match=answer.rstrip("%"),
            tags=("table", "customer_supplier"),
        )

    operations = "第三节 管理层讨论与分析 > 主营业务分析"
    operation_rows = [
        ("芯导科技2024年的销售费用是多少元？", "销售费用", "7,838,515.98"),
        ("芯导科技2024年的管理费用是多少元？", "管理费用", "19,734,236.69"),
        ("芯导科技2024年的研发费用是多少元？", "研发费用", "35,350,211.68"),
        ("芯导科技2024年的财务费用是多少元？", "财务费用", "-2,713,664.55"),
        ("芯导科技2024年投资活动产生的现金流量净额是多少元？", "投资活动产生的现金流量净额", "-410,557,951.57"),
        ("芯导科技2024年筹资活动产生的现金流量净额是多少元？", "筹资活动产生的现金流量净额", "-71,251,612.22"),
        ("芯导科技2024年经营活动现金流量净额增长的主要原因是什么？", "经营活动产生的现金流量净额变动原因说明", "公司营业收入增加，收到货款增加"),
        ("芯导科技2024年投资活动现金流量净额变动的主要原因是什么？", "投资活动产生的现金流量净额变动原因说明", "使用暂时闲置资金进行现金管理"),
    ]
    for question, anchor, answer in operation_rows:
        add(
            question,
            year,
            37,
            operations,
            anchor,
            answer,
            "operating_metric" if "原因" not in question else "narrative_fact",
            evidence_type="text" if "原因" in question else "table",
            difficulty="medium" if "原因" in question else "simple",
            tags=("cash_flow",),
        )

    assets = "第三节 管理层讨论与分析 > 资产、负债情况分析"
    asset_rows = [
        ("芯导科技2024年末货币资金是多少元？", "货币资金", "54,103,148.43"),
        ("芯导科技2024年末交易性金融资产是多少元？", "交易性金融资产", "1,699,071,411.17"),
        ("芯导科技2024年末应收账款是多少元？", "应收账款", "26,598,531.32"),
        ("芯导科技2024年末存货是多少元？", "存货", "43,867,541.18"),
        ("芯导科技2024年末一年内到期的非流动资产是多少元？", "一年内到期的非流动资产", "325,559,143.84"),
        ("芯导科技2024年末长期股权投资是多少元？", "长期股权投资", "35,437,467.45"),
        ("芯导科技2024年末固定资产是多少元？", "固定资产", "135,917,111.42"),
        ("芯导科技2024年末应付账款是多少元？", "应付账款", "43,936,618.52"),
    ]
    for question, anchor, answer in asset_rows:
        add(question, year, 38, assets, anchor, answer, "balance_sheet_metric", tags=("table", "balance_sheet"))

    add(
        "芯导科技2024年末在职员工总数是多少人？",
        year,
        53,
        "第四节 公司治理 > 员工情况",
        "在职员工的数量合计",
        "118人",
        "employee_fact",
        answer_match="118",
        tags=("table", "employee"),
    )
    add(
        "芯导科技2024年末技术人员有多少人？",
        year,
        53,
        "第四节 公司治理 > 员工情况",
        "技术人员",
        "53人",
        "employee_fact",
        answer_match="53",
        tags=("table", "employee"),
    )
    add(
        "芯导科技2024年度拟每10股派发多少元现金红利？",
        year,
        54,
        "第四节 公司治理 > 利润分配预案",
        "每 10 股派发现金红利",
        "8.00元",
        "dividend_fact",
        evidence_type="text",
        answer_match="8.00",
        tags=("dividend",),
    )
    add(
        "芯导科技2024年度拟派发现金红利合计多少元？",
        year,
        54,
        "第四节 公司治理 > 利润分配预案",
        "合计拟派发现金红利",
        "94,080,000.00元",
        "dividend_fact",
        evidence_type="text",
        answer_match="94,080,000.00",
        tags=("dividend",),
    )


def add_2025_facts() -> None:
    year = 2025
    basic = "第二节 公司简介和主要财务指标"
    basic_rows = [
        ("芯导科技2025年年报中的公司中文全称是什么？", "公司的中文名称", "上海芯导电子科技股份有限公司"),
        ("芯导科技2025年年报中的公司中文简称是什么？", "公司的中文简称", "芯导科技"),
        ("芯导科技2025年年报披露的英文全称是什么？", "公司的外文名称", "Shanghai Prisemi Electronics Co., Ltd."),
        ("芯导科技2025年年报披露的英文简称是什么？", "公司的外文名称缩写", "Prisemi"),
        ("芯导科技2025年年报披露的法定代表人是谁？", "公司的法定代表人", "欧新华"),
        ("芯导科技2025年披露的注册地址在哪里？", "公司注册地址", "中国（上海）自由贸易试验区祖冲之路2277弄7号"),
        ("芯导科技2025年披露的办公地址邮政编码有哪些？", "公司办公地址的邮政编码", "201210; 201203"),
        ("芯导科技2025年年报披露的公司网址是什么？", "公司网址", "http://www.prisemi.com"),
        ("芯导科技2025年年报披露的电子信箱是什么？", "电子信箱", "investor@prisemi.com"),
        ("芯导科技2025年年报中的董事会秘书是谁？", "董事会秘书", "兰芳云"),
        ("芯导科技2025年年报中的证券事务代表是谁？", "证券事务代表", "闵雨琦"),
        ("芯导科技2025年披露的联系电话是什么？", "电话", "021-60753051"),
        ("芯导科技2025年年报显示其股票在哪个交易所及板块上市？", "股票上市交易所及板块", "上海证券交易所科创板"),
        ("芯导科技2025年年报中的股票代码是多少？", "股票代码", "688230"),
        ("芯导科技2025年聘请的境内会计师事务所是哪家？", "公司聘请的会计师事务所", "天职国际会计师事务所（特殊普通合伙）"),
        ("芯导科技2025年持续督导的保荐机构是哪家？", "报告期内履行持续督导职责的保荐机构", "国元证券股份有限公司"),
    ]
    for question, anchor, answer in basic_rows:
        add(question, year, 7, basic, anchor, answer, "basic_fact", tags=("company_profile",))

    finance = "第二节 公司简介和主要财务指标 > 六、近三年主要会计数据和财务指标"
    finance_rows = [
        ("芯导科技2025年的营业收入是多少元？", "营业收入", "393,607,502.95"),
        ("芯导科技2025年营业收入同比增长多少？", "营业收入", "11.52%"),
        ("芯导科技2025年归属于上市公司股东的净利润是多少元？", "归属于上市公司股东的净利润", "106,152,925.57"),
        ("芯导科技2025年归母净利润同比变化多少？", "归属于上市公司股东的净利润", "-4.91%"),
        ("芯导科技2025年扣除非经常性损益后的归母净利润是多少元？", "归属于上市公司股东的扣除非经常性损益的净利润", "68,886,393.66"),
        ("芯导科技2025年扣非归母净利润同比增长多少？", "归属于上市公司股东的扣除非经常性损益的净利润", "17.54%"),
        ("芯导科技2025年经营活动产生的现金流量净额是多少元？", "经营活动产生的现金流量净额", "62,787,071.96"),
        ("芯导科技2025年经营活动现金流量净额同比变化多少？", "经营活动产生的现金流量净额", "-25.91%"),
        ("芯导科技2025年末归属于上市公司股东的净资产是多少元？", "归属于上市公司股东的净资产", "2,270,493,804.73"),
        ("芯导科技2025年末归母净资产同比增长多少？", "归属于上市公司股东的净资产", "0.28%"),
        ("芯导科技2025年末总资产是多少元？", "总资产", "2,329,403,069.01"),
        ("芯导科技2025年末总资产同比增长多少？", "总资产", "0.06%"),
        ("芯导科技2025年基本每股收益是多少元？", "基本每股收益", "0.90元/股"),
        ("芯导科技2025年稀释每股收益是多少元？", "稀释每股收益", "0.90元/股"),
        ("芯导科技2025年扣非后的基本每股收益是多少元？", "扣除非经常性损益后的基本每股收益", "0.59元/股"),
        ("芯导科技2025年加权平均净资产收益率是多少？", "加权平均净资产收益率", "4.69%"),
        ("芯导科技2025年扣非后的加权平均净资产收益率是多少？", "扣除非经常性损益后的加权平均净资产收益率", "3.04%"),
        ("芯导科技2025年研发投入占营业收入的比例是多少？", "研发投入占营业收入的比例", "7.89%"),
    ]
    for question, anchor, answer in finance_rows:
        add(
            question,
            year,
            8,
            finance,
            anchor,
            answer,
            "financial_metric",
            difficulty="medium" if "同比" in question else "simple",
            answer_match=re.sub(r"(元/股|元|%)$", "", answer),
            tags=("table", "financial_metric"),
        )

    quarter = "第二节 公司简介和主要财务指标 > 八、2025年分季度主要财务数据"
    quarter_values = {
        "营业收入": ["74,262,850.66", "108,166,308.65", "108,176,534.14", "103,001,809.50"],
        "归属于上市公司股东的净利润": ["24,071,129.24", "26,127,989.84", "23,428,646.98", "32,525,159.51"],
        "经营活动产生的现金流量净额": ["-5,688,742.23", "32,273,319.04", "19,802,446.16", "16,400,048.99"],
    }
    quarter_names = ["第一季度", "第二季度", "第三季度", "第四季度"]
    for metric, values in quarter_values.items():
        for quarter_name, value in zip(quarter_names, values):
            add(
                f"芯导科技2025年{quarter_name}的{metric}是多少元？",
                year,
                9,
                quarter,
                metric,
                value,
                "quarterly_metric",
                tags=("table", "quarter"),
            )

    research = "第三节 管理层讨论与分析 > 核心技术与研发进展"
    research_rows = [
        ("芯导科技2025年新增申请发明专利多少项？", "发明专利", "10项"),
        ("芯导科技2025年新增获得发明专利多少项？", "发明专利", "13项"),
        ("截至2025年末，芯导科技累计申请发明专利多少项？", "发明专利", "82项"),
        ("截至2025年末，芯导科技累计获得发明专利多少项？", "发明专利", "38项"),
        ("芯导科技2025年新增申请实用新型专利多少项？", "实用新型专利", "2项"),
        ("芯导科技2025年新增获得实用新型专利多少项？", "实用新型专利", "2项"),
        ("截至2025年末，芯导科技累计申请实用新型专利多少项？", "实用新型专利", "56项"),
        ("截至2025年末，芯导科技累计获得实用新型专利多少项？", "实用新型专利", "38项"),
        ("芯导科技2025年各类研发成果新增申请合计多少项？", "合计", "19项"),
        ("芯导科技2025年各类研发成果新增获得合计多少项？", "合计", "25项"),
        ("截至2025年末，芯导科技各类研发成果累计申请多少项？", "合计", "223项"),
        ("截至2025年末，芯导科技各类研发成果累计获得多少项？", "合计", "141项"),
        ("芯导科技2025年费用化研发投入是多少元？", "费用化研发投入", "31,070,863.06"),
        ("芯导科技2025年研发投入合计是多少元？", "研发投入合计", "31,070,863.06"),
    ]
    for question, anchor, answer in research_rows:
        add(
            question,
            year,
            24,
            research,
            anchor,
            answer,
            "research_fact",
            answer_match=answer.rstrip("项%元"),
            tags=("table", "research"),
        )

    operations = "第三节 管理层讨论与分析 > 主营业务分析"
    operation_rows = [
        ("芯导科技2025年主营业务分析表中的营业收入是多少元？", "营业收入", "393,607,502.95"),
        ("芯导科技2025年的营业成本是多少元？", "营业成本", "264,341,326.02"),
        ("芯导科技2025年的销售费用是多少元？", "销售费用", "9,006,020.84"),
        ("芯导科技2025年的管理费用是多少元？", "管理费用", "19,299,509.16"),
        ("芯导科技2025年的财务费用是多少元？", "财务费用", "-642,779.62"),
        ("芯导科技2025年的研发费用是多少元？", "研发费用", "31,070,863.06"),
        ("芯导科技2025年经营活动现金流量净额是多少元？", "经营活动产生的现金流量净额", "62,787,071.96"),
        ("芯导科技2025年投资活动现金流量净额是多少元？", "投资活动产生的现金流量净额", "117,471,251.62"),
        ("芯导科技2025年筹资活动现金流量净额是多少元？", "筹资活动产生的现金流量净额", "-94,794,471.21"),
        ("芯导科技2025年营业成本同比增长多少？", "营业成本", "14.22%"),
    ]
    for question, anchor, answer in operation_rows:
        add(
            question,
            year,
            33,
            operations,
            anchor,
            answer,
            "operating_metric",
            difficulty="medium" if "同比" in question else "simple",
            answer_match=answer.rstrip("%"),
            tags=("table", "operating_metric"),
        )

    product = "第三节 管理层讨论与分析 > 收入和成本分析"
    product_rows = [
        ("芯导科技2025年集成电路业务营业收入是多少万元？", "集成电路", "39,360.75"),
        ("芯导科技2025年集成电路业务毛利率是多少？", "集成电路", "32.84%"),
        ("芯导科技2025年功率器件营业收入是多少万元？", "功率器件", "36,050.77"),
        ("芯导科技2025年功率器件毛利率是多少？", "功率器件", "33.13%"),
        ("芯导科技2025年TVS产品营业收入是多少万元？", "TVS", "22,154.52"),
        ("芯导科技2025年TVS产品毛利率是多少？", "TVS", "32.58%"),
        ("芯导科技2025年MOSFET产品营业收入是多少万元？", "MOSFET", "8,903.82"),
        ("芯导科技2025年MOSFET产品毛利率是多少？", "MOSFET", "34.85%"),
        ("芯导科技2025年肖特基产品营业收入是多少万元？", "肖特基", "3,668.42"),
        ("芯导科技2025年肖特基产品毛利率是多少？", "肖特基", "33.29%"),
        ("芯导科技2025年功率IC营业收入是多少万元？", "功率IC", "3,309.98"),
        ("芯导科技2025年功率IC毛利率是多少？", "功率IC", "29.75%"),
        ("芯导科技2025年中国大陆地区营业收入是多少万元？", "中国大陆", "36,650.54"),
        ("芯导科技2025年中国大陆以外地区营业收入是多少万元？", "中国大陆以外", "2,710.21"),
    ]
    for question, anchor, answer in product_rows:
        add(
            question,
            year,
            34,
            product,
            anchor,
            answer,
            "segment_metric",
            answer_match=answer.rstrip("%"),
            tags=("table", "segment"),
        )

    customer_supplier = "第三节 管理层讨论与分析 > 主要销售客户及主要供应商情况"
    customer_rows = [
        ("芯导科技2025年前五名客户销售额合计是多少万元？", 36, "公司前五名客户", "20,225.35"),
        ("芯导科技2025年前五名客户销售额占年度销售总额多少？", 36, "公司前五名客户", "51.39%"),
        ("芯导科技2025年第一大客户销售额是多少万元？", 36, "客户一", "9,521.08"),
        ("芯导科技2025年第一大客户销售额占年度销售总额多少？", 36, "客户一", "24.19%"),
        ("芯导科技2025年前五名供应商采购额合计是多少万元？", 37, "公司前五名供应商", "15,520.27"),
        ("芯导科技2025年前五名供应商采购额占年度采购总额多少？", 37, "公司前五名供应商", "57.00%"),
        ("芯导科技2025年第一大供应商采购额是多少万元？", 37, "供应商一", "4,822.09"),
        ("芯导科技2025年第一大供应商采购额占年度采购总额多少？", 37, "供应商一", "17.71%"),
    ]
    for question, page, anchor, answer in customer_rows:
        add(
            question,
            year,
            page,
            customer_supplier,
            anchor,
            answer,
            "customer_supplier",
            answer_match=answer.rstrip("%"),
            tags=("table", "customer_supplier"),
        )

    assets = "第三节 管理层讨论与分析 > 资产、负债情况分析"
    asset_rows = [
        ("芯导科技2025年末货币资金是多少元？", "货币资金", "138,659,733.35"),
        ("芯导科技2025年末交易性金融资产是多少元？", "交易性金融资产", "1,930,923,341.87"),
        ("芯导科技2025年末应收账款是多少元？", "应收账款", "25,943,313.16"),
        ("芯导科技2025年末预付款项是多少元？", "预付款项", "4,757,572.04"),
        ("芯导科技2025年末存货是多少元？", "存货", "46,601,550.73"),
        ("芯导科技2025年末长期股权投资是多少元？", "长期股权投资", "52,482,788.67"),
        ("芯导科技2025年末固定资产是多少元？", "固定资产", "127,585,336.10"),
        ("芯导科技2025年末应付账款是多少元？", "应付账款", "39,307,672.60"),
    ]
    for question, anchor, answer in asset_rows:
        add(question, year, 39, assets, anchor, answer, "balance_sheet_metric", tags=("table", "balance_sheet"))


def replace_fact(old_question: str, replacement: Fact) -> None:
    matches = [index for index, fact in enumerate(FACTS) if fact.question == old_question]
    if len(matches) != 1:
        raise AssertionError(f"replacement target must occur once: {old_question!r}; matches={matches}")
    FACTS[matches[0]] = replacement


def apply_page_coverage_replacements() -> None:
    replacements = {
        "芯导科技对外披露的传真号码是什么？": Fact(
            "芯导科技2024年计入当期损益的政府补助是多少元？",
            2024,
            10,
            "第二节 公司简介和主要财务指标 > 非经常性损益项目和金额",
            "计入当期损益的政府补助",
            "3,770,259.62元",
            "non_recurring_metric",
            answer_match="3,770,259.62",
            tags=("table", "non_recurring"),
        ),
        "芯导科技披露年度报告的媒体名称是什么？": Fact(
            "芯导科技2024年委托他人投资或管理资产取得的损益是多少元？",
            2024,
            10,
            "第二节 公司简介和主要财务指标 > 非经常性损益项目和金额",
            "委托他人投资或管理资产的损益",
            "54,641,015.45元",
            "non_recurring_metric",
            answer_match="54,641,015.45",
            tags=("table", "non_recurring"),
        ),
        "芯导科技披露年度报告的证券交易所网址是什么？": Fact(
            "芯导科技2024年采用什么经营模式进行产品研发和销售？",
            2024,
            18,
            "第三节 管理层讨论与分析 > 主要经营模式",
            "一直采用 Fabless 的经营模式",
            "Fabless经营模式",
            "business_model",
            evidence_type="text",
            answer_match="Fabless",
            tags=("narrative", "business_model"),
        ),
        "芯导科技年度报告的备置地点在哪里？": Fact(
            "芯导科技2024年主要采用什么销售模式？",
            2024,
            18,
            "第三节 管理层讨论与分析 > 主要经营模式",
            "销售模式",
            "经销为主，直销为辅",
            "business_model",
            evidence_type="text",
            tags=("narrative", "sales_mode"),
        ),
        "芯导科技2024年第一季度的经营活动产生的现金流量净额是多少元？": Fact(
            "芯导科技2024年高性能数模混合电源管理芯片项目累计投入多少万元？",
            2024,
            23,
            "第三节 管理层讨论与分析 > 在研项目情况",
            "高性能数模混合电源管理芯片开发及产业化",
            "2,952.01万元",
            "project_metric",
            answer_match="2,952.01",
            tags=("table", "research_project"),
        ),
        "芯导科技2024年第二季度的经营活动产生的现金流量净额是多少元？": Fact(
            "芯导科技2024年TVS产品收入占主营业务收入的比例是多少？",
            2024,
            30,
            "第三节 管理层讨论与分析 > 风险因素",
            "TVS 产品收入占比较高",
            "56.23%",
            "risk_fact",
            evidence_type="text",
            answer_match="56.23%",
            tags=("narrative", "risk"),
        ),
        "芯导科技2024年第三季度的经营活动产生的现金流量净额是多少元？": Fact(
            "芯导科技2024年集成电路业务的综合毛利率是多少？",
            2024,
            33,
            "第三节 管理层讨论与分析 > 收入和成本分析",
            "集成电路",
            "34.43%",
            "segment_metric",
            answer_match="34.43",
            tags=("table", "segment"),
        ),
        "芯导科技2024年第四季度的经营活动产生的现金流量净额是多少元？": Fact(
            "芯导科技2024年报告期投资额是多少元？",
            2024,
            39,
            "第三节 管理层讨论与分析 > 投资状况分析",
            "报告期投资额",
            "36,000,000.00元",
            "investment_metric",
            answer_match="36,000,000.00",
            tags=("table", "investment"),
        ),
        "芯导科技2024年第二大客户的销售额是多少万元？": Fact(
            "芯导科技无锡子公司2024年的净利润是多少万元？",
            2024,
            42,
            "第三节 管理层讨论与分析 > 主要控股参股公司分析",
            "芯导科技（无锡）有限公司",
            "-282.31万元",
            "subsidiary_metric",
            answer_match="-282.31",
            tags=("table", "subsidiary"),
        ),
        "芯导科技2024年第二大客户销售额占年度销售总额多少？": Fact(
            "审计机构认为芯导科技2024年财务报表是否公允反映了财务状况和经营成果？",
            2024,
            98,
            "第十节 财务报告 > 审计意见",
            "我们认为",
            "是，审计意见认为财务报表在所有重大方面公允反映了相关情况",
            "audit_fact",
            evidence_type="text",
            difficulty="medium",
            answer_match="公允反映",
            tags=("narrative", "audit"),
        ),
        "芯导科技2024年第二大供应商的采购额是多少万元？": Fact(
            "芯导科技2024年末现金及现金等价物余额是多少元？",
            2024,
            110,
            "第十节 财务报告 > 合并现金流量表",
            "期末现金及现金等价物余额",
            "54,103,148.43元",
            "cash_flow_statement_metric",
            answer_match="54,103,148.43",
            tags=("table", "cash_flow_statement"),
        ),
        "芯导科技2024年第二大供应商采购额占年度采购总额多少？": Fact(
            "芯导科技2024年末原材料的账面价值是多少元？",
            2024,
            158,
            "第十节 财务报告 > 存货 > 存货分类",
            "原材料",
            "17,381,261.87元",
            "inventory_metric",
            answer_match="17,381,261.87",
            tags=("table", "inventory"),
        ),
        "芯导科技2025年年报披露的电子信箱是什么？": Fact(
            "审计机构认为芯导科技2025年财务报表是否公允反映了财务状况和经营成果？",
            2025,
            96,
            "第八节 财务报告 > 审计意见",
            "我们认为",
            "是，审计意见认为财务报表在所有重大方面公允反映了相关情况",
            "audit_fact",
            evidence_type="text",
            difficulty="medium",
            answer_match="公允反映",
            tags=("narrative", "audit"),
        ),
        "芯导科技2025年披露的联系电话是什么？": Fact(
            "芯导科技2025年末流动资产合计是多少元？",
            2025,
            101,
            "第八节 财务报告 > 合并资产负债表",
            "流动资产合计",
            "2,147,087,686.04元",
            "financial_statement_metric",
            answer_match="2,147,087,686.04",
            tags=("table", "balance_sheet"),
        ),
        "芯导科技2025年年报显示其股票在哪个交易所及板块上市？": Fact(
            "芯导科技2025年末负债合计是多少元？",
            2025,
            101,
            "第八节 财务报告 > 合并资产负债表",
            "负债合计",
            "58,909,264.28元",
            "financial_statement_metric",
            answer_match="58,909,264.28",
            tags=("table", "balance_sheet"),
        ),
        "芯导科技2025年年报中的股票代码是多少？": Fact(
            "芯导科技2025年利润总额是多少元？",
            2025,
            105,
            "第八节 财务报告 > 合并利润表",
            "利润总额",
            "114,482,387.41元",
            "financial_statement_metric",
            answer_match="114,482,387.41",
            tags=("table", "income_statement"),
        ),
        "芯导科技2025年第一季度的经营活动产生的现金流量净额是多少元？": Fact(
            "芯导科技2025年销售商品、提供劳务收到的现金是多少元？",
            2025,
            108,
            "第八节 财务报告 > 合并现金流量表",
            "销售商品、提供劳务收到的现金",
            "408,745,060.66元",
            "cash_flow_statement_metric",
            answer_match="408,745,060.66",
            tags=("table", "cash_flow_statement"),
        ),
        "芯导科技2025年第二季度的经营活动产生的现金流量净额是多少元？": Fact(
            "芯导科技2025年末原材料的账面价值是多少元？",
            2025,
            157,
            "第八节 财务报告 > 存货 > 存货分类",
            "原材料",
            "17,869,473.48元",
            "inventory_metric",
            answer_match="17,869,473.48",
            tags=("table", "inventory"),
        ),
        "芯导科技2025年第三季度的经营活动产生的现金流量净额是多少元？": Fact(
            "芯导科技2025年末研发设备的账面价值是多少元？",
            2025,
            164,
            "第八节 财务报告 > 固定资产情况",
            "期末账面价值",
            "835,522.99元",
            "fixed_asset_metric",
            answer_match="835,522.99",
            tags=("table", "fixed_asset"),
        ),
        "芯导科技2025年第四季度的经营活动产生的现金流量净额是多少元？": Fact(
            "芯导科技2025年末未分配利润是多少元？",
            2025,
            178,
            "第八节 财务报告 > 未分配利润",
            "期末未分配利润",
            "301,301,829.21元",
            "equity_metric",
            answer_match="301,301,829.21",
            tags=("table", "equity"),
        ),
        "芯导科技2025年第一大客户销售额是多少万元？": Fact(
            "芯导科技2025年投资收益合计是多少元？",
            2025,
            182,
            "第八节 财务报告 > 投资收益",
            "投资收益",
            "38,075,490.56元",
            "investment_metric",
            answer_match="38,075,490.56",
            tags=("table", "investment"),
        ),
        "芯导科技2025年第一大客户销售额占年度销售总额多少？": Fact(
            "芯导科技2025年资产减值损失合计是多少元？",
            2025,
            183,
            "第八节 财务报告 > 资产减值损失",
            "资产减值损失",
            "-2,026,203.23元",
            "impairment_metric",
            answer_match="-2,026,203.23",
            tags=("table", "impairment"),
        ),
        "芯导科技2025年第一大供应商采购额是多少万元？": Fact(
            "芯导科技2025年末可随时用于支付的银行存款是多少元？",
            2025,
            188,
            "第八节 财务报告 > 现金和现金等价物的构成",
            "可随时用于支付的银行存款",
            "95,656,952.53元",
            "cash_metric",
            answer_match="95,656,952.53",
            tags=("table", "cash"),
        ),
        "芯导科技2025年第一大供应商采购额占年度采购总额多少？": Fact(
            "芯导科技2025年年报披露的最终控制方是谁？",
            2025,
            201,
            "第八节 财务报告 > 关联方及关联交易",
            "本企业最终控制方",
            "欧新华",
            "related_party_fact",
            evidence_type="text",
            tags=("narrative", "related_party"),
        ),
    }
    for old_question, replacement in replacements.items():
        replace_fact(old_question, replacement)


def _load_2024_pages() -> list[str]:
    reader = PdfReader(str(SOURCE_2024))
    return [page.extract_text() or "" for page in reader.pages]


def _load_2025_pages() -> list[str]:
    payload = json.loads(SOURCE_2025.read_text(encoding="utf-8"))
    pages: list[str] = []
    for page in payload.get("pdf_info", []):
        pages.append(" ".join(_block_text(block) for block in page.get("para_blocks", [])))
    return pages


def build() -> None:
    add_2024_facts()
    add_2025_facts()
    apply_page_coverage_replacements()
    assert len(FACTS) == 200, f"expected 200 facts, got {len(FACTS)}"
    assert sum(1 for fact in FACTS if fact.report_year == 2024) == 100
    assert sum(1 for fact in FACTS if fact.report_year == 2025) == 100

    pages_by_year = {2024: _load_2024_pages(), 2025: _load_2025_pages()}
    sources = {
        2024: {
            "source_id": "xindao-2024-annual-report",
            "path": SOURCE_2024,
            "kind": "pdf",
        },
        2025: {
            "source_id": "xindao-2025-annual-report-mineru",
            "path": SOURCE_2025,
            "kind": "mineru_json",
        },
    }
    source_hashes = {year: _sha256(value["path"]) for year, value in sources.items()}

    rows: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    failures: list[str] = []
    for index, fact in enumerate(FACTS, start=1):
        if fact.question in seen_questions:
            failures.append(f"duplicate question: {fact.question}")
        seen_questions.add(fact.question)

        pages = pages_by_year[fact.report_year]
        if fact.page < 1 or fact.page > len(pages):
            failures.append(f"{index}: page out of range: {fact.page}")
            page_text = ""
        else:
            page_text = pages[fact.page - 1]

        answer_match = fact.answer_match or fact.expected_answer
        normalized_page = _normalize(page_text)
        if _normalize(fact.anchor) not in normalized_page:
            failures.append(f"{index}: anchor not found on page {fact.page}: {fact.anchor}")
        if _normalize(answer_match) not in normalized_page:
            failures.append(f"{index}: answer not found on page {fact.page}: {answer_match}")

        source = sources[fact.report_year]
        row = {
            "query_id": f"RET-{index:03d}",
            "question": fact.question,
            "scope": {
                "company_id": "xindao",
                "company_name": "上海芯导电子科技股份有限公司",
                "report_year": fact.report_year,
            },
            "scenario": fact.scenario,
            "difficulty": fact.difficulty,
            "difficulty_tags": list(dict.fromkeys((*fact.tags, fact.evidence_type))),
            "answerable_in_corpus": True,
            "expected_answer": fact.expected_answer,
            "gold": {
                "evidence_groups": [
                    {
                        "group_id": "g1",
                        "requirement": fact.question,
                        "accepted_spans": [
                            {
                                "source_id": source["source_id"],
                                "source_path": str(source["path"].relative_to(PROJECT_ROOT)),
                                "source_sha256": source_hashes[fact.report_year],
                                "source_kind": source["kind"],
                                "page_number": fact.page,
                                "heading_path": fact.heading_path,
                                "evidence_type": fact.evidence_type,
                                "anchor_text": fact.anchor,
                                "answer_match": answer_match,
                                "relevance_grade": 2,
                                "context_excerpt": _excerpt(page_text, fact.anchor, answer_match),
                            }
                        ],
                    }
                ]
            },
            "metric_policy": {
                "primary": "evidence_recall@5",
                "eligible": ["hit@5", "evidence_recall@5", "mrr@5", "ndcg@5"],
                "gold_group_count": 1,
            },
            "split": "frozen_test",
            "annotation_status": "source_located_single_pass",
        }
        rows.append(row)

    if failures:
        raise AssertionError("\n".join(failures))

    DATASET_PATH.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    with REVIEW_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "query_id",
                "question",
                "report_year",
                "scenario",
                "difficulty",
                "expected_answer",
                "source_path",
                "page_number",
                "heading_path",
                "anchor_text",
                "annotation_status",
            ],
        )
        writer.writeheader()
        for row in rows:
            span = row["gold"]["evidence_groups"][0]["accepted_spans"][0]
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "report_year": row["scope"]["report_year"],
                    "scenario": row["scenario"],
                    "difficulty": row["difficulty"],
                    "expected_answer": row["expected_answer"],
                    "source_path": span["source_path"],
                    "page_number": span["page_number"],
                    "heading_path": span["heading_path"],
                    "anchor_text": span["anchor_text"],
                    "annotation_status": row["annotation_status"],
                }
            )

    scenario_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for row in rows:
        scenario_counts[row["scenario"]] = scenario_counts.get(row["scenario"], 0) + 1
        difficulty_counts[row["difficulty"]] = difficulty_counts.get(row["difficulty"], 0) + 1
    manifest = {
        "dataset": DATASET_PATH.name,
        "review_file": REVIEW_PATH.name,
        "dataset_version": "1.0.0-draft",
        "created_for": "QA-Agent retrieval-only offline evaluation",
        "case_count": len(rows),
        "year_counts": {"2024": 100, "2025": 100},
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "max_gold_evidence_groups_per_query": 1,
        "sources": [
            {
                "source_id": value["source_id"],
                "path": str(value["path"].relative_to(PROJECT_ROOT)),
                "kind": value["kind"],
                "sha256": source_hashes[year],
                "page_count": len(pages_by_year[year]),
            }
            for year, value in sorted(sources.items())
        ],
        "important_limitations": [
            "All 200 cases are simple or medium single-evidence-group questions.",
            "The benchmark covers one company and two report years; it does not establish cross-company generalization.",
            "The 2025 gold source is the full 219-page MinerU JSON parse. The local 12-page 2025 PDF is a report summary and is not used as the gold source.",
            "All labels are source-located but still require independent human/domain review before external metric claims.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
