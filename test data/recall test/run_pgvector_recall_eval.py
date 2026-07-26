from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy import text


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.connection import get_async_session, get_pgvector_database_url  # noqa: E402
from service.agent.query_expander import expand_queries  # noqa: E402
from service.embedding.embedding_service import EmbeddingService, build_embedding_provider_from_config  # noqa: E402
from service.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from service.retrieval.parallel_query_executor import ParallelQueryExecutor  # noqa: E402
from service.retrieval.retrieval_cache import RetrievalResultCache  # noqa: E402
from service.retrieval.runtime import get_runtime_repository  # noqa: E402
from service.retrieval.two_stage_hybrid_reranker import TwoStageHybridReranker  # noqa: E402
from utils.config_loader import get_app_config  # noqa: E402


DATASET_PATH = SCRIPT_DIR / "recall_eval_200.jsonl"
PARTIAL_PATH = SCRIPT_DIR / "pgvector_recall_results.partial.jsonl"
RESULT_JSONL_PATH = SCRIPT_DIR / "pgvector_recall_results.jsonl"
RESULT_CSV_PATH = SCRIPT_DIR / "pgvector_recall_results.csv"
FAILURE_CSV_PATH = SCRIPT_DIR / "pgvector_recall_failures.csv"
EQUIVALENT_FAILURE_CSV_PATH = SCRIPT_DIR / "pgvector_recall_equivalent_failures.csv"
SUMMARY_JSON_PATH = SCRIPT_DIR / "pgvector_recall_summary.json"
SUMMARY_CSV_PATH = SCRIPT_DIR / "pgvector_recall_summary.csv"
CORPUS_SNAPSHOT_PATH = SCRIPT_DIR / "pgvector_corpus_snapshot.json"
PAGE_METADATA_EXCEPTIONS = {"RET-020", "RET-079"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 200-case retrieval-only evaluation against pgvector.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases; 0 means all.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate corpus/gold coverage without retrieval.")
    parser.add_argument("--fresh", action="store_true", help="Ignore and replace any partial checkpoint.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_text(value: Any) -> str:
    text_value = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff.%]+", "", text_value)


def normalized_compact(value: Any) -> str:
    return normalized_text(value).replace(",", "")


def numeric_tokens(value: Any) -> list[str]:
    text_value = unicodedata.normalize("NFKC", str(value or "")).replace(",", "")
    return [token.lstrip("+").rstrip("%") for token in re.findall(r"[-+]?\d+(?:\.\d+)?%?", text_value)]


def phrase_present(needle: Any, haystack: Any) -> bool:
    needle_value = normalized_compact(needle)
    haystack_value = normalized_compact(haystack)
    return bool(needle_value and needle_value in haystack_value)


def answer_present(answer: Any, haystack: Any) -> tuple[bool, str]:
    answer_value = str(answer or "").strip()
    if not answer_value:
        return False, "empty_answer"
    if phrase_present(answer_value, haystack):
        return True, "answer_exact_normalized"

    unit_stripped = re.sub(r"(人民币)?(元|万元|亿元|个百分点|百分点|%|％)$", "", answer_value).strip()
    if unit_stripped != answer_value and len(normalized_compact(unit_stripped)) >= 2:
        if phrase_present(unit_stripped, haystack):
            return True, "answer_unit_normalized"

    expected_numbers = numeric_tokens(answer_value)
    actual_numbers = set(numeric_tokens(haystack))
    if expected_numbers and all(number in actual_numbers for number in expected_numbers):
        return True, "answer_numeric_normalized"
    return False, "answer_not_found"


def candidate_text(candidate: Mapping[str, Any]) -> str:
    return "\n".join(
        str(candidate.get(key) or "")
        for key in (
            "heading_path",
            "level1_title",
            "level2_title",
            "level3_title",
            "table_header_text",
            "table_context_text",
            "search_text",
            "raw_doc",
            "content",
        )
    )


def parse_page_range(candidate: Mapping[str, Any]) -> tuple[int | None, int | None]:
    page_range = str(candidate.get("page_range") or "").strip()
    match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", page_range)
    if match:
        start = int(match.group(1))
        end = int(match.group(2) or match.group(1))
        return min(start, end), max(start, end)
    try:
        page_idx = int(candidate.get("page_idx"))
    except (TypeError, ValueError):
        return None, None
    return page_idx, page_idx


def page_matches(candidate: Mapping[str, Any], gold_page_number: Any) -> bool:
    try:
        expected_zero_based = int(gold_page_number) - 1
    except (TypeError, ValueError):
        return False
    start, end = parse_page_range(candidate)
    return start is not None and end is not None and start <= expected_zero_based <= end


def heading_matches(gold_heading: Any, candidate: Mapping[str, Any]) -> bool:
    candidate_heading = "\n".join(
        str(candidate.get(key) or "")
        for key in ("heading_path", "level1_title", "level2_title", "level3_title")
    )
    if not candidate_heading.strip():
        return False
    segments = [
        segment.strip()
        for segment in re.split(r"[>/|]", str(gold_heading or ""))
        if len(normalized_compact(segment)) >= 4
    ]
    return any(phrase_present(segment, candidate_heading) or phrase_present(candidate_heading, segment) for segment in segments)


def match_span(candidate: Mapping[str, Any], span: Mapping[str, Any]) -> dict[str, Any]:
    text_value = candidate_text(candidate)
    page_ok = page_matches(candidate, span.get("page_number"))
    answer_ok, answer_method = answer_present(span.get("answer_match"), text_value)
    anchor_ok = phrase_present(span.get("anchor_text"), text_value)
    heading_ok = heading_matches(span.get("heading_path"), candidate)
    matched = bool(answer_ok and (anchor_ok or (page_ok and heading_ok)))
    if matched:
        if anchor_ok:
            method = f"{answer_method}+anchor+{'page' if page_ok else 'page_metadata_mismatch'}"
        else:
            method = f"{answer_method}+page+heading"
    else:
        method = "no_match"
    return {
        "matched": matched,
        "method": method,
        "page_match": page_ok,
        "answer_match": answer_ok,
        "anchor_match": anchor_ok,
        "heading_match": heading_ok,
    }


def match_case(candidate: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    for group in case.get("gold", {}).get("evidence_groups", []):
        for span in group.get("accepted_spans", []):
            detail = match_span(candidate, span)
            if detail["matched"]:
                return {**detail, "group_id": group.get("group_id"), "span": span}
    return {"matched": False, "method": "no_match"}


def query_type_for(case: Mapping[str, Any]) -> str:
    question = str(case.get("question") or "")
    if any(term in question for term in ("原因", "为什么", "影响")):
        return "analysis"
    return "information_extraction"


async def load_corpus(database_url: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    async with get_async_session(backend="pgvector", database_url=database_url) as session:
        documents = (
            await session.execute(
                text(
                    """
                    SELECT
                        d.doc_id, d.collection_name, d.doc_source, d.title, d.doc_hash,
                        d.page_count, d.indexed_at, d.metadata_json,
                        COUNT(c.id) AS chunk_count,
                        MIN(c.page_idx) AS min_page_idx,
                        MAX(c.page_idx) AS max_page_idx,
                        COUNT(*) FILTER (WHERE c.embedding IS NOT NULL) AS embedded_chunk_count,
                        COUNT(*) FILTER (WHERE c.chunk_type = 'table') AS table_chunk_count
                    FROM pdf_documents d
                    LEFT JOIN pdf_chunks c ON c.doc_id = d.doc_id
                    GROUP BY
                        d.doc_id, d.collection_name, d.doc_source, d.title, d.doc_hash,
                        d.page_count, d.indexed_at, d.metadata_json
                    ORDER BY d.indexed_at DESC
                    """
                )
            )
        ).mappings().all()
        chunks = (
            await session.execute(
                text(
                    """
                    SELECT
                        chunk_id, doc_id, collection_name, doc_source, page_idx, page_range,
                        chunk_type, chunk_index, heading_path, level1_title, level2_title,
                        level3_title, table_header_text, table_context_text, search_text,
                        content AS raw_doc, metadata_json
                    FROM pdf_chunks
                    WHERE collection_name = 'xindao'
                    ORDER BY doc_id, chunk_index
                    """
                )
            )
        ).mappings().all()
    return [dict(row) for row in documents], [dict(row) for row in chunks]


def corpus_scope_matches(chunk: Mapping[str, Any], case: Mapping[str, Any]) -> bool:
    metadata = chunk.get("metadata_json") or {}
    scope = case.get("scope") or {}
    return (
        str(metadata.get("company_id") or "") == str(scope.get("company_id") or "")
        and str(metadata.get("year") or "") == str(scope.get("report_year") or "")
    )


def preflight_gold(cases: Sequence[dict[str, Any]], chunks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        metadata = chunk.get("metadata_json") or {}
        by_scope[(str(metadata.get("company_id") or ""), str(metadata.get("year") or ""))].append(chunk)

    rows: list[dict[str, Any]] = []
    for case in cases:
        scope = case.get("scope") or {}
        candidates = by_scope[(str(scope.get("company_id") or ""), str(scope.get("report_year") or ""))]
        matches: list[dict[str, Any]] = []
        for chunk in candidates:
            detail = match_case(chunk, case)
            if detail["matched"]:
                matches.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "page_idx": chunk.get("page_idx"),
                        "page_range": chunk.get("page_range"),
                        "chunk_type": chunk.get("chunk_type"),
                        "method": detail.get("method"),
                        "page_match": detail.get("page_match"),
                    }
                )
        page_candidates: list[dict[str, Any]] = []
        if not matches:
            spans = [
                span
                for group in case.get("gold", {}).get("evidence_groups", [])
                for span in group.get("accepted_spans", [])
            ]
            for chunk in candidates:
                for span in spans:
                    if not page_matches(chunk, span.get("page_number")):
                        continue
                    detail = match_span(chunk, span)
                    page_candidates.append(
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "page_idx": chunk.get("page_idx"),
                            "page_range": chunk.get("page_range"),
                            "chunk_type": chunk.get("chunk_type"),
                            "heading_path": chunk.get("heading_path"),
                            "answer_match": detail.get("answer_match"),
                            "anchor_match": detail.get("anchor_match"),
                            "heading_match": detail.get("heading_match"),
                            "content_preview": re.sub(r"\s+", " ", str(chunk.get("raw_doc") or ""))[:500],
                        }
                    )
            answer_candidates: list[dict[str, Any]] = []
            for chunk in candidates:
                for span in spans:
                    answer_ok, answer_method = answer_present(span.get("answer_match"), candidate_text(chunk))
                    if not answer_ok:
                        continue
                    answer_candidates.append(
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "page_idx": chunk.get("page_idx"),
                            "page_range": chunk.get("page_range"),
                            "chunk_type": chunk.get("chunk_type"),
                            "heading_path": chunk.get("heading_path"),
                            "answer_method": answer_method,
                            "anchor_match": phrase_present(span.get("anchor_text"), candidate_text(chunk)),
                            "content_preview": re.sub(r"\s+", " ", str(chunk.get("raw_doc") or ""))[:500],
                        }
                    )
        else:
            answer_candidates = []
        rows.append(
            {
                "query_id": case["query_id"],
                "report_year": scope.get("report_year"),
                "gold_located_in_pgvector": bool(matches),
                "matching_chunk_count": len(matches),
                "matching_chunks": matches[:10],
                "page_candidates": page_candidates[:20],
                "off_page_answer_candidates": answer_candidates[:20],
            }
        )
    located_with_page_match_count = sum(
        1
        for row in rows
        if any(bool(match.get("page_match")) for match in row.get("matching_chunks", []))
    )
    return {
        "case_count": len(rows),
        "located_count": sum(1 for row in rows if row["gold_located_in_pgvector"]),
        "located_with_page_match_count": located_with_page_match_count,
        "page_mismatch_query_ids": [
            row["query_id"]
            for row in rows
            if row["gold_located_in_pgvector"]
            and not any(bool(match.get("page_match")) for match in row.get("matching_chunks", []))
        ],
        "missing_query_ids": [row["query_id"] for row in rows if not row["gold_located_in_pgvector"]],
        "rows": rows,
    }


def build_retriever(config: Mapping[str, Any]) -> tuple[HybridRetriever, EmbeddingService]:
    retrieval_cfg = config.get("retrieval", {})
    reranker_cfg = config.get("reranker", {})
    cache_cfg = config.get("cache", {})
    embedding_service = EmbeddingService(provider=build_embedding_provider_from_config(dict(config)))
    reranker = TwoStageHybridReranker(
        dense_weight=float(reranker_cfg.get("dense_weight", 0.50)),
        bm25_weight=float(reranker_cfg.get("bm25_weight", 0.35)),
        metadata_boost_weight=float(reranker_cfg.get("metadata_boost_weight", 0.10)),
        table_boost_weight=float(reranker_cfg.get("table_boost_weight", 0.05)),
        near_duplicate_threshold=float(reranker_cfg.get("near_duplicate_threshold", 0.90)),
        table_evidence_quota=int(retrieval_cfg.get("table_evidence_quota", 2)),
        cross_encoder_enabled=bool(reranker_cfg.get("cross_encoder_enabled", True)),
        cross_encoder_model=str(reranker_cfg.get("cross_encoder_model", "BAAI/bge-reranker-base")),
        cross_encoder_candidate_pool=int(reranker_cfg.get("cross_encoder_candidate_pool", 30)),
        cross_encoder_batch_size=int(reranker_cfg.get("cross_encoder_batch_size", 8)),
        cross_encoder_max_length=int(reranker_cfg.get("cross_encoder_max_length", 512)),
        cross_encoder_local_files_only=bool(reranker_cfg.get("cross_encoder_local_files_only", False)),
        cross_encoder_device=str(reranker_cfg.get("cross_encoder_device", "auto")),
        cross_encoder_load_on_request=bool(reranker_cfg.get("cross_encoder_load_on_request", False)),
    )
    executor = ParallelQueryExecutor(
        repository=get_runtime_repository(),
        retrieval_cache=RetrievalResultCache(
            ttl_seconds=int(cache_cfg.get("ttl_seconds", 3600)),
            max_items=int(cache_cfg.get("max_items", 5000)),
        ),
        async_embedding_builder=lambda value: embedding_service.embed_text(value, use_cache=True, chunk_text=False),
        max_concurrency=int(retrieval_cfg.get("max_concurrency", 6)),
        query_timeout_seconds=float(retrieval_cfg.get("query_timeout_seconds", 20)),
    )
    return (
        HybridRetriever(
            executor,
            reranker=reranker,
            table_evidence_quota=int(retrieval_cfg.get("table_evidence_quota", 2)),
        ),
        embedding_service,
    )


def candidate_result(
    candidate: Mapping[str, Any],
    rank: int,
    match: Mapping[str, Any],
    strict_match: bool,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": candidate.get("chunk_id"),
        "doc_id": candidate.get("doc_id"),
        "doc_source": candidate.get("doc_source"),
        "page_idx": candidate.get("page_idx"),
        "page_number": int(candidate.get("page_idx")) + 1 if candidate.get("page_idx") is not None else None,
        "page_range": candidate.get("page_range"),
        "chunk_type": candidate.get("chunk_type"),
        "heading_path": candidate.get("heading_path"),
        "source_channels": candidate.get("source_channels") or [],
        "final_score": candidate.get("final_score"),
        "rank_score": candidate.get("rank_score"),
        "confidence_score": candidate.get("confidence_score"),
        "hybrid_recall_score": candidate.get("hybrid_recall_score"),
        "cross_encoder_score": candidate.get("cross_encoder_score"),
        "score_source": candidate.get("score_source"),
        "gold_match": bool(match.get("matched")),
        "gold_strict_match": strict_match,
        "gold_match_method": match.get("method"),
        "gold_page_match": match.get("page_match"),
        "content_preview": re.sub(r"\s+", " ", str(candidate.get("raw_doc") or candidate.get("content") or ""))[:400],
    }


async def evaluate_case(retriever: HybridRetriever, case: dict[str, Any]) -> dict[str, Any]:
    query_type = query_type_for(case)
    variants = expand_queries(str(case["question"]), query_type, 4)
    scope = case["scope"]
    started = time.perf_counter()
    error = ""
    response: dict[str, Any] = {}
    try:
        response = await retriever.retrieve(
            question=str(case["question"]),
            collection_name="xindao",
            top_k=5,
            query_type=query_type,
            expand_query_num=4,
            enable_cache=False,
            expanded_queries=variants[1:],
            metadata_filter={"company_id": scope["company_id"], "year": scope["report_year"]},
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    evidence = list(response.get("evidence") or [])
    top_rows: list[dict[str, Any]] = []
    equivalent_matched_rank: int | None = None
    equivalent_matched_method = ""
    matched_rank: int | None = None
    matched_method = ""
    for rank, candidate in enumerate(evidence[:5], start=1):
        detail = match_case(candidate, case)
        strict_match = bool(
            detail.get("matched")
            and (detail.get("page_match") or case["query_id"] in PAGE_METADATA_EXCEPTIONS)
        )
        if detail.get("matched") and equivalent_matched_rank is None:
            equivalent_matched_rank = rank
            equivalent_matched_method = str(detail.get("method") or "")
        if strict_match and matched_rank is None:
            matched_rank = rank
            matched_method = str(detail.get("method") or "")
        top_rows.append(candidate_result(candidate, rank, detail, strict_match))

    retrieval_trace = response.get("retrieval_trace") or {}
    task_trace = retrieval_trace.get("task_trace") or []
    rerank_trace = response.get("rerank_trace") or {}
    cross_encoder_trace = rerank_trace.get("cross_encoder") or {}
    return {
        "query_id": case["query_id"],
        "question": case["question"],
        "report_year": scope["report_year"],
        "scenario": case["scenario"],
        "difficulty": case["difficulty"],
        "query_type": query_type,
        "hit_at_1": matched_rank == 1,
        "hit_at_3": matched_rank is not None and matched_rank <= 3,
        "hit_at_5": matched_rank is not None and matched_rank <= 5,
        "matched_rank": matched_rank,
        "matched_method": matched_method,
        "matched_page_consistent": next(
            (row.get("gold_page_match") for row in top_rows if row.get("gold_strict_match")),
            None,
        ),
        "reciprocal_rank_at_5": 1.0 / matched_rank if matched_rank else 0.0,
        "ndcg_at_5": 1.0 / math.log2(matched_rank + 1) if matched_rank else 0.0,
        "equivalent_hit_at_1": equivalent_matched_rank == 1,
        "equivalent_hit_at_3": equivalent_matched_rank is not None and equivalent_matched_rank <= 3,
        "equivalent_hit_at_5": equivalent_matched_rank is not None and equivalent_matched_rank <= 5,
        "equivalent_matched_rank": equivalent_matched_rank,
        "equivalent_matched_method": equivalent_matched_method,
        "equivalent_reciprocal_rank_at_5": 1.0 / equivalent_matched_rank if equivalent_matched_rank else 0.0,
        "equivalent_ndcg_at_5": 1.0 / math.log2(equivalent_matched_rank + 1) if equivalent_matched_rank else 0.0,
        "returned_count": len(evidence[:5]),
        "latency_ms": latency_ms,
        "error": error,
        "route_error_count": sum(1 for row in task_trace if row.get("error")),
        "route_timeout_count": sum(1 for row in task_trace if row.get("timed_out")),
        "stage1_candidate_count": retrieval_trace.get("candidate_pool_size"),
        "merged_candidate_count": retrieval_trace.get("merged_candidate_count"),
        "cross_encoder_status": cross_encoder_trace.get("status"),
        "cross_encoder_reason": cross_encoder_trace.get("reason", ""),
        "cross_encoder_model": cross_encoder_trace.get("model", ""),
        "top5": top_rows,
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def metric_row(group_type: str, group_value: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "group_type": group_type,
        "group_value": group_value,
        "count": len(rows),
        "hit_at_1_count": sum(bool(row["hit_at_1"]) for row in rows),
        "hit_at_1": round(sum(bool(row["hit_at_1"]) for row in rows) / len(rows), 6),
        "hit_at_3_count": sum(bool(row["hit_at_3"]) for row in rows),
        "hit_at_3": round(sum(bool(row["hit_at_3"]) for row in rows) / len(rows), 6),
        "hit_at_5_count": sum(bool(row["hit_at_5"]) for row in rows),
        "recall_at_5": round(sum(bool(row["hit_at_5"]) for row in rows) / len(rows), 6),
        "mrr_at_5": round(statistics.fmean(float(row["reciprocal_rank_at_5"]) for row in rows), 6),
        "ndcg_at_5": round(statistics.fmean(float(row["ndcg_at_5"]) for row in rows), 6),
        "equivalent_hit_at_5_count": sum(bool(row["equivalent_hit_at_5"]) for row in rows),
        "equivalent_recall_at_5": round(
            sum(bool(row["equivalent_hit_at_5"]) for row in rows) / len(rows),
            6,
        ),
        "latency_mean_ms": round(statistics.fmean(latencies), 2),
        "latency_p50_ms": round(percentile(latencies, 0.50), 2),
        "latency_p95_ms": round(percentile(latencies, 0.95), 2),
        "query_error_count": sum(bool(row["error"]) for row in rows),
        "route_timeout_count": sum(int(row["route_timeout_count"]) for row in rows),
    }


def build_metric_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [metric_row("overall", "all", rows)]
    for field in ("report_year", "difficulty", "scenario", "query_type"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        for value in sorted(grouped):
            output.append(metric_row(field, value, grouped[value]))
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    rows: Sequence[dict[str, Any]],
    documents: Sequence[dict[str, Any]],
    preflight: Mapping[str, Any],
    embedding_service: EmbeddingService,
    dataset_hash: str,
) -> None:
    RESULT_JSONL_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    result_fields = [
        "query_id", "question", "report_year", "scenario", "difficulty", "query_type",
        "hit_at_1", "hit_at_3", "hit_at_5", "matched_rank", "matched_method",
        "matched_page_consistent", "reciprocal_rank_at_5", "ndcg_at_5",
        "equivalent_hit_at_1", "equivalent_hit_at_3", "equivalent_hit_at_5",
        "equivalent_matched_rank", "equivalent_matched_method",
        "equivalent_reciprocal_rank_at_5", "equivalent_ndcg_at_5",
        "returned_count", "latency_ms", "error",
        "route_error_count", "route_timeout_count", "stage1_candidate_count",
        "merged_candidate_count", "cross_encoder_status", "cross_encoder_reason",
        "cross_encoder_model",
    ]
    write_csv(RESULT_CSV_PATH, rows, result_fields)
    write_csv(FAILURE_CSV_PATH, [row for row in rows if not row["hit_at_5"]], result_fields)
    write_csv(
        EQUIVALENT_FAILURE_CSV_PATH,
        [row for row in rows if not row["equivalent_hit_at_5"]],
        result_fields,
    )

    metric_rows = build_metric_rows(rows)
    write_csv(SUMMARY_CSV_PATH, metric_rows, list(metric_rows[0].keys()))
    cross_encoder_status = Counter(str(row.get("cross_encoder_status") or "") for row in rows)
    cross_encoder_reasons = Counter(str(row.get("cross_encoder_reason") or "") for row in rows if row.get("cross_encoder_reason"))
    summary = {
        "dataset_path": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
        "dataset_sha256": dataset_hash,
        "evaluated_case_count": len(rows),
        "collection_name": "xindao",
        "corpus_documents": list(documents),
        "gold_preflight": {
            "case_count": preflight["case_count"],
            "located_count": preflight["located_count"],
            "located_with_page_match_count": preflight["located_with_page_match_count"],
            "page_mismatch_query_ids": preflight["page_mismatch_query_ids"],
            "missing_query_ids": preflight["missing_query_ids"],
        },
        "retrieval_configuration": {
            "top_k": 5,
            "query_variant_total": 4,
            "cache_enabled": False,
            "metadata_filter": ["company_id", "year"],
            "embedding_provider": embedding_service.provider_name,
            "embedding_model": embedding_service.provider_model,
            "cross_encoder_status_counts": dict(cross_encoder_status),
            "cross_encoder_reason_counts": dict(cross_encoder_reasons),
        },
        "match_policy": {
            "page_number_mapping": "gold 1-based page_number -> pgvector 0-based page_idx",
            "primary_strict": (
                "answer match AND (anchor match OR heading match) AND gold page overlap; "
                "RET-020 and RET-079 are accepted page-metadata exceptions verified during corpus preflight"
            ),
            "secondary_equivalent": "answer match AND (anchor match OR (page overlap AND heading match)) on any report page",
            "normalization": "Unicode NFKC; whitespace/punctuation normalization; numeric/unit normalization",
        },
        "metrics": metric_rows,
        "generated_at_unix": time.time(),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def async_main(args: argparse.Namespace) -> None:
    cases = load_jsonl(DATASET_PATH)
    if args.limit > 0:
        cases = cases[: args.limit]
    config = get_app_config()
    database_url = get_pgvector_database_url(config)
    documents, chunks = await load_corpus(database_url)
    preflight = preflight_gold(cases, chunks)
    corpus_snapshot = {
        "documents": documents,
        "gold_preflight": preflight,
        "captured_at_unix": time.time(),
    }
    CORPUS_SNAPSHOT_PATH.write_text(
        json.dumps(corpus_snapshot, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"[preflight] documents={len(documents)} chunks={len(chunks)} "
        f"gold_located={preflight['located_count']}/{preflight['case_count']}",
        flush=True,
    )
    if preflight["missing_query_ids"]:
        print(f"[preflight] missing={','.join(preflight['missing_query_ids'])}", flush=True)
    if args.preflight_only:
        return

    existing: dict[str, dict[str, Any]] = {}
    if PARTIAL_PATH.exists() and not args.fresh:
        existing = {row["query_id"]: row for row in load_jsonl(PARTIAL_PATH)}
    if args.fresh:
        PARTIAL_PATH.write_text("", encoding="utf-8")

    retriever, embedding_service = build_retriever(config)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        row = existing.get(case["query_id"])
        if row is None:
            row = await evaluate_case(retriever, case)
            with PARTIAL_PATH.open("a", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        rows.append(row)
        print(
            f"[{index:03d}/{len(cases):03d}] {row['query_id']} "
            f"hit@5={int(row['hit_at_5'])} rank={row['matched_rank']} "
            f"latency_ms={row['latency_ms']} ce={row['cross_encoder_status']} "
            f"error={row['error'] or '-'}",
            flush=True,
        )

    write_outputs(rows, documents, preflight, embedding_service, dataset_sha256(DATASET_PATH))
    overall = build_metric_rows(rows)[0]
    print("[summary] " + json.dumps(overall, ensure_ascii=False), flush=True)


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
