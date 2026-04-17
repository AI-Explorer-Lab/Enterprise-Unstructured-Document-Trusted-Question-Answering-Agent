from __future__ import annotations

import asyncio
import threading
import time

from service.retrieval.parallel_query_executor import ParallelQueryExecutor
from service.retrieval.pgvector_repository import PgvectorRepository, deterministic_embedding
from service.retrieval.retrieval_cache import RetrievalResultCache


class SlowRepository(PgvectorRepository):
    def __init__(self, delay_seconds: float = 0.03, **kwargs):
        super().__init__(**kwargs)
        self.delay_seconds = float(delay_seconds)
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def _begin(self) -> None:
        with self._lock:
            self._active += 1
            if self._active > self.max_active:
                self.max_active = self._active

    def _end(self) -> None:
        with self._lock:
            self._active -= 1

    def dense_search(self, *args, **kwargs):
        self._begin()
        try:
            time.sleep(self.delay_seconds)
            return super().dense_search(*args, **kwargs)
        finally:
            self._end()

    def keyword_search(self, *args, **kwargs):
        self._begin()
        try:
            time.sleep(self.delay_seconds)
            return super().keyword_search(*args, **kwargs)
        finally:
            self._end()

    def table_search(self, collection_name: str, query_text: str, top_k: int):
        self._begin()
        try:
            time.sleep(self.delay_seconds)
            return PgvectorRepository.keyword_search(
                self,
                collection_name=collection_name,
                query_text=query_text,
                top_k=top_k,
                chunk_type="table",
                table_only=True,
            )
        finally:
            self._end()


def _build_chunks() -> list[dict]:
    rows = [
        {
            "chunk_id": "text-1",
            "collection_name": "finance",
            "chunk_type": "text",
            "raw_doc": "2024 年公司营业收入同比增长，主要由企业服务业务拉动。",
            "doc_source": "annual_report.pdf",
            "page_idx": 3,
            "chunk_index": 1,
            "heading_path": "第三章 经营情况",
        },
        {
            "chunk_id": "table-1",
            "collection_name": "finance",
            "chunk_type": "table",
            "raw_doc": "营业收入 100亿元；毛利率 22%",
            "doc_source": "annual_report.pdf",
            "page_idx": 4,
            "chunk_index": 2,
            "heading_path": "第四章 财务指标",
            "table_header_text": "指标,2023,2024,同比",
            "table_context_text": "单位: 亿元",
        },
        {
            "chunk_id": "table-2",
            "collection_name": "finance",
            "chunk_type": "table",
            "raw_doc": "净利润 18亿元；同比增长 12%",
            "doc_source": "annual_report.pdf",
            "page_idx": 4,
            "chunk_index": 3,
            "heading_path": "第四章 财务指标",
            "table_header_text": "指标,2023,2024,同比",
            "table_context_text": "单位: 亿元",
        },
    ]
    for row in rows:
        row["embedding"] = deterministic_embedding(row["raw_doc"])
    return rows


def test_retrieval_cache_key_ttl_and_max_items() -> None:
    clock = {"value": 100.0}
    cache = RetrievalResultCache(ttl_seconds=5, max_items=2, time_fn=lambda: clock["value"])

    key1 = cache.build_key("finance", "hash-a", "fact_lookup", 5)
    key2 = cache.build_key("finance", "hash-b", "fact_lookup", 5)
    key3 = cache.build_key("finance", "hash-c", "table_qa", 5)

    assert key1.collection_name == "finance"
    assert key1.question_hash == "hash-a"
    assert key1.query_type == "fact_lookup"
    assert key1.top_k == 5
    assert "finance" in key1.as_storage_key()
    assert "hash-a" in key1.as_storage_key()

    cache.set(key1, {"value": 1})
    cache.set(key2, {"value": 2})
    assert cache.get(key1)["value"] == 1

    clock["value"] += 10
    assert cache.get(key1) is None

    cache.set(key2, {"value": 2})
    cache.set(key3, {"value": 3})
    cache.set(key1, {"value": 4})

    assert len(cache) == 2
    assert cache.get(key2) is None
    assert cache.get(key3)["value"] == 3
    assert cache.get(key1)["value"] == 4


def test_parallel_executor_dense_bm25_table_and_cache_hit() -> None:
    repo = SlowRepository(
        backend="local_dev",
        embedding_dim=1024,
        local_chunks=_build_chunks(),
        delay_seconds=0.02,
    )
    cache = RetrievalResultCache(ttl_seconds=120, max_items=32)

    executor = ParallelQueryExecutor(
        repository=repo,
        retrieval_cache=cache,
        max_concurrency=2,
        query_timeout_seconds=1.0,
        query_expander=lambda question, n: [f"{question} 财务指标", f"{question} 表格数据"][:n],
    )

    result1 = asyncio.run(
        executor.execute(
            question="2024 营业收入同比是多少",
            collection_name="finance",
            top_k=3,
            query_type="table_qa",
            expand_query_num=2,
            enable_cache=True,
        )
    )

    trace1 = result1["retrieval_trace"]
    assert trace1["cache_hit"] is False
    assert trace1["task_count"] == 9
    assert repo.max_active <= 2

    route_set = {entry["route"] for entry in trace1["task_trace"]}
    assert {"dense", "bm25", "table"}.issubset(route_set)

    assert result1["candidates"]
    candidate_ids = {item["chunk_id"] for item in result1["candidates"]}
    assert "table-1" in candidate_ids

    result2 = asyncio.run(
        executor.execute(
            question="2024 营业收入同比是多少",
            collection_name="finance",
            top_k=3,
            query_type="table_qa",
            expand_query_num=2,
            enable_cache=True,
        )
    )
    assert result2["retrieval_trace"]["cache_hit"] is True


def test_parallel_executor_timeout_trace() -> None:
    repo = SlowRepository(
        backend="local_dev",
        embedding_dim=1024,
        local_chunks=_build_chunks(),
        delay_seconds=0.1,
    )

    executor = ParallelQueryExecutor(
        repository=repo,
        retrieval_cache=None,
        max_concurrency=1,
        query_timeout_seconds=0.02,
        query_expander=lambda question, n: [f"{question} 扩展"][:n],
    )

    result = asyncio.run(
        executor.execute(
            question="请给出营业收入",
            collection_name="finance",
            top_k=2,
            query_type="fact_lookup",
            expand_query_num=1,
            enable_cache=False,
        )
    )

    trace = result["retrieval_trace"]
    assert any(entry["timed_out"] for entry in trace["task_trace"])
