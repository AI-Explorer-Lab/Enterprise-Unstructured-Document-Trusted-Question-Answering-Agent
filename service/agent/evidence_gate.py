from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence

from service.agent.controlled_agents import EvidenceAuditAgent, merge_audit_and_rule_gate


def _score(row: Dict[str, Any]) -> float:
    score, _source = _score_with_source(row)
    return score


def _score_with_source(row: Dict[str, Any]) -> tuple[float, str]:
    for key in ("light_final_score", "confidence_score", "final_score", "score", "dense_score", "bm25_score"):
        if key not in row:
            continue
        try:
            return max(0.0, min(1.0, float(row.get(key)))), key
        except Exception:
            continue
    return 0.0, ""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _metadata_value(row: Mapping[str, Any], key: str) -> Any:
    if row.get(key) not in (None, ""):
        return row.get(key)
    metadata = row.get("metadata") or row.get("metadata_json") or {}
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return None


def _filter_values(value: Any) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    for item in values:
        text = _clean(item)
        if text and text not in result:
            result.append(text)
    return result


def _matches_metadata_filter(row: Mapping[str, Any], metadata_filter: Mapping[str, Any]) -> bool:
    for key in ("company_id", "year"):
        expected = _filter_values(metadata_filter.get(key))
        if expected and _clean(_metadata_value(row, key)) not in expected:
            return False
    return True


def _years_from_value(value: Any) -> List[int]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    years: List[int] = []
    for item in values:
        for match in re.findall(r"(?:19|20)\d{2}", str(item or "")):
            year = int(match)
            if year not in years:
                years.append(year)
    return years


def _requested_years(slots: Mapping[str, Any]) -> List[int]:
    scope = slots.get("retrieval_scope")
    if isinstance(scope, Mapping):
        years = _years_from_value(scope.get("years"))
        if years:
            return years
    metadata_filter = slots.get("metadata_filter")
    if isinstance(metadata_filter, Mapping):
        years = _years_from_value(metadata_filter.get("year"))
        if years:
            return years
    return _years_from_value(slots.get("years"))


def _evidence_years(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    years: List[int] = []
    for row in rows:
        for year in _years_from_value(_metadata_value(row, "year")):
            if year not in years:
                years.append(year)
    return years


def _retry_filter_for_years(slots: Mapping[str, Any], years: Sequence[int]) -> Dict[str, Any]:
    metadata_filter = slots.get("metadata_filter")
    retry_filter = dict(metadata_filter or {}) if isinstance(metadata_filter, Mapping) else {}
    retry_filter["year"] = list(years)
    return retry_filter


def _confidence(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    values = [_score(row) for row in rows]
    top = max(values) if values else 0.0
    avg = sum(values) / max(1, len(values))
    return round(max(0.0, min(1.0, (top + avg) / 2.0)), 4)


def _score_diagnostics(
    rows: List[Dict[str, Any]],
    *,
    min_top_score: float,
    min_avg_score: float,
) -> Dict[str, Any]:
    score_pairs = [_score_with_source(row) for row in rows]
    scores = [score for score, _source in score_pairs]
    top_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / max(1, len(scores))
    top_source = ""
    if score_pairs:
        _score_value, top_source = max(score_pairs, key=lambda item: item[0])
    diagnostics: Dict[str, Any] = {
        "top_score": round(top_score, 4),
        "avg_score": round(avg_score, 4),
        "score_source": top_source,
        "score_warning": "",
    }
    if top_score < min_top_score or avg_score < min_avg_score:
        diagnostics["score_warning"] = "low_score"
    return diagnostics


class EvidenceGate:
    def __init__(
        self,
        evidence_min_docs: int = 1,
        evidence_min_top_score: float = 0.45,
        evidence_min_avg_score: float = 0.30,
        retry_limit: int = 2,
        refuse_on_low_evidence: bool = True,
    ) -> None:
        self.evidence_min_docs = max(1, int(evidence_min_docs))
        self.evidence_min_top_score = float(evidence_min_top_score)
        self.evidence_min_avg_score = float(evidence_min_avg_score)
        self.retry_limit = max(0, int(retry_limit))
        self.refuse_on_low_evidence = bool(refuse_on_low_evidence)

    def evaluate(
        self,
        evidence: List[Dict[str, Any]],
        query_type: str,
        retry_count: int = 0,
        table_evidence_quota: int = 1,
        slots: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        slots = slots or {}
        rows = [dict(item) for item in evidence]
        if not rows:
            decision = "retry" if retry_count < self.retry_limit else "refuse"
            return {
                "decision": decision,
                "reason": "no_evidence" if decision == "retry" else "no_evidence_after_retry",
                "confidence": 0.0,
            }

        diagnostics = _score_diagnostics(
            rows,
            min_top_score=self.evidence_min_top_score,
            min_avg_score=self.evidence_min_avg_score,
        )
        docs = {
            str(row.get("doc_id") or row.get("doc_source") or "")
            for row in rows
            if row.get("doc_id") or row.get("doc_source")
        }
        table_count = sum(1 for row in rows if str(row.get("chunk_type") or "") == "table")
        diagnostics.update(
            {
                "doc_count": len(docs),
                "table_evidence_count": table_count,
                "coverage_warnings": [],
            }
        )

        if query_type == "table_qa" and table_count < max(1, int(table_evidence_quota)):
            diagnostics["coverage_warnings"].append("missing_table_evidence")

        if query_type == "multi_doc_compare" and len(docs) < 2:
            diagnostics["coverage_warnings"].append("multi_doc_evidence_missing")

        requested_years = _requested_years(slots)
        if len(requested_years) > 1 and query_type in {"table_qa", "fact_lookup", "multi_doc_compare", "summarization", "report_generation"}:
            evidence_years = _evidence_years(rows)
            missing_years = [year for year in requested_years if year not in evidence_years]
            diagnostics["requested_years"] = requested_years
            diagnostics["evidence_years"] = evidence_years
            diagnostics["missing_years"] = missing_years
            if missing_years:
                diagnostics["coverage_warnings"].append("missing_year_evidence")
                decision = "retry" if retry_count < self.retry_limit else "refuse"
                reason = "missing_year_evidence" if decision == "retry" else "missing_year_evidence_after_retry"
                missing_hint = "、".join(str(year) for year in missing_years)
                retry_terms = " ".join(str(item or "").strip() for item in [slots.get("company"), slots.get("metric")] if str(item or "").strip())
                return {
                    "decision": decision,
                    "reason": reason,
                    **diagnostics,
                    "confidence": _confidence(rows),
                    "missing_years": missing_years,
                    "message": f"检索到的证据未覆盖问题要求的全部年份，缺少 {missing_hint} 年的相关证据，无法可靠完成多年份回答。",
                    "suggested_retry_query": f"{missing_hint} 年度 {retry_terms}".strip(),
                    "retry_metadata_filter": _retry_filter_for_years(slots, missing_years),
                }

        coverage_sensitive_types = {"summarization", "report_generation", "multi_doc_compare"}
        if query_type in coverage_sensitive_types and len(rows) < self.evidence_min_docs:
            diagnostics["coverage_warnings"].append("insufficient_doc_coverage")

        return {
            "decision": "answer",
            "reason": "evidence_available",
            **diagnostics,
            "confidence": _confidence(rows),
        }


class EvidenceDecisionEngine:
    def __init__(
        self,
        llm_service: Any | None = None,
        evidence_min_docs: int = 1,
        evidence_min_top_score: float = 0.45,
        evidence_min_avg_score: float = 0.30,
        retry_limit: int = 2,
        refuse_on_low_evidence: bool = True,
    ) -> None:
        self.retry_limit = max(0, int(retry_limit))
        self.rule_gate = EvidenceGate(
            evidence_min_docs=evidence_min_docs,
            evidence_min_top_score=evidence_min_top_score,
            evidence_min_avg_score=evidence_min_avg_score,
            retry_limit=retry_limit,
            refuse_on_low_evidence=refuse_on_low_evidence,
        )
        self.evidence_agent = EvidenceAuditAgent(llm_service)

    async def evaluate(
        self,
        question: str,
        query_type: str,
        slots: Mapping[str, Any] | None,
        selected_skill: str,
        evidence: Sequence[Mapping[str, Any]],
        rerank_trace: Mapping[str, Any] | None = None,
        retry_count: int = 0,
        table_evidence_quota: int = 2,
    ) -> Dict[str, Any]:
        rows = [dict(item) for item in evidence if isinstance(item, Mapping)]
        metadata_filter = (slots or {}).get("metadata_filter") if isinstance(slots, Mapping) else None
        scope_mismatches: List[Dict[str, Any]] = []
        if isinstance(metadata_filter, Mapping) and metadata_filter:
            kept_rows: List[Dict[str, Any]] = []
            for row in rows:
                if _matches_metadata_filter(row, metadata_filter):
                    kept_rows.append(row)
                else:
                    scope_mismatches.append(
                        {
                            "chunk_id": row.get("chunk_id", ""),
                            "doc_source": row.get("doc_source", ""),
                            "company_id": _metadata_value(row, "company_id"),
                            "year": _metadata_value(row, "year"),
                        }
                    )
            rows = kept_rows
        rule_gate = self.rule_gate.evaluate(
            rows,
            query_type=query_type,
            retry_count=retry_count,
            table_evidence_quota=table_evidence_quota,
            slots=dict(slots or {}),
        )
        if str(rule_gate.get("reason") or "") in {"no_evidence", "no_evidence_after_retry"}:
            rule_audit = self.evidence_agent._rule_audit(
                question=question,
                query_type=query_type,
                slots=slots,
                selected_skill=selected_skill,
                evidence=rows,
                rerank_trace=rerank_trace,
            )
        else:
            rule_audit = await self.evidence_agent.audit(
                question=question,
                query_type=query_type,
                slots=slots,
                selected_skill=selected_skill,
                evidence=rows,
                rerank_trace=rerank_trace,
            )
        merged = merge_audit_and_rule_gate(rule_gate, rule_audit)
        if scope_mismatches:
            merged["scope_mismatch_count"] = len(scope_mismatches)
            merged["scope_mismatches"] = scope_mismatches[:10]
        merged["rule_gate"] = rule_gate
        merged["evidence_audit"] = rule_audit
        if merged.get("decision") == "retry" and not str(merged.get("suggested_retry_query") or "").strip():
            merged["suggested_retry_query"] = str(rule_audit.get("suggested_retry_query") or question)
        return merged


def run_evidence_gate(
    query_type: str,
    evidence: List[Dict[str, Any]],
    slots: Dict[str, Any] | None = None,
    retry_count: int = 0,
    retry_limit: int = 2,
    min_top_score: float = 0.45,
    min_avg_score: float = 0.30,
    table_evidence_quota: int = 1,
    refuse_on_low_evidence: bool = True,
) -> Dict[str, Any]:
    gate = EvidenceGate(
        evidence_min_top_score=min_top_score,
        evidence_min_avg_score=min_avg_score,
        retry_limit=retry_limit,
        refuse_on_low_evidence=refuse_on_low_evidence,
    )
    return gate.evaluate(
        evidence=list(evidence or []),
        query_type=query_type,
        retry_count=retry_count,
        table_evidence_quota=table_evidence_quota,
        slots=slots,
    )
