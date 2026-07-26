from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "run_pgvector_recall_eval.py"
OUTPUT_JSON = SCRIPT_DIR / "pgvector_recall_failure_diagnostics.json"
OUTPUT_CSV = SCRIPT_DIR / "pgvector_recall_failure_diagnostics.csv"


def load_eval_module() -> Any:
    spec = importlib.util.spec_from_file_location("recall_eval_runtime", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def main() -> None:
    runtime = load_eval_module()
    cases = {row["query_id"]: row for row in load_jsonl(SCRIPT_DIR / "recall_eval_200.jsonl")}
    results = load_jsonl(SCRIPT_DIR / "pgvector_recall_results.jsonl")
    failures = [row for row in results if not row["hit_at_5"]]
    retriever, _embedding_service = runtime.build_retriever(runtime.get_app_config())

    diagnostics: list[dict[str, Any]] = []
    for index, failed in enumerate(failures, start=1):
        case = cases[failed["query_id"]]
        query_type = runtime.query_type_for(case)
        variants = runtime.expand_queries(case["question"], query_type, 4)
        scope = case["scope"]
        stage1 = await retriever.parallel_executor.execute(
            question=case["question"],
            collection_name="xindao",
            top_k=40,
            query_type=query_type,
            expand_query_num=4,
            enable_cache=False,
            expanded_queries=variants[1:],
            metadata_filter={"company_id": scope["company_id"], "year": scope["report_year"]},
        )
        candidates = list(stage1.get("candidates") or [])
        stage1_gold: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            match = runtime.match_case(candidate, case)
            if match.get("matched"):
                stage1_gold.append(
                    {
                        "rank": rank,
                        "chunk_id": candidate.get("chunk_id"),
                        "page_idx": candidate.get("page_idx"),
                        "page_range": candidate.get("page_range"),
                        "chunk_type": candidate.get("chunk_type"),
                        "retrieval_score": candidate.get("retrieval_score"),
                        "dense_score": candidate.get("dense_score"),
                        "bm25_score": candidate.get("bm25_score"),
                        "source_channels": candidate.get("source_channels") or [],
                        "match_method": match.get("method"),
                    }
                )

        reranked, rerank_trace = await asyncio.to_thread(
            retriever.reranker.rerank,
            query=case["question"],
            candidates=candidates,
            top_k=40,
            query_type=query_type,
            table_evidence_quota=retriever.table_evidence_quota,
        )
        reranked_top5, rerank_top5_trace = await asyncio.to_thread(
            retriever.reranker.rerank,
            query=case["question"],
            candidates=candidates,
            top_k=5,
            query_type=query_type,
            table_evidence_quota=retriever.table_evidence_quota,
        )
        reranked_gold: list[dict[str, Any]] = []
        for rank, candidate in enumerate(reranked, start=1):
            match = runtime.match_case(candidate, case)
            if match.get("matched"):
                reranked_gold.append(
                    {
                        "rank": rank,
                        "chunk_id": candidate.get("chunk_id"),
                        "page_idx": candidate.get("page_idx"),
                        "page_range": candidate.get("page_range"),
                        "chunk_type": candidate.get("chunk_type"),
                        "cross_encoder_score": candidate.get("cross_encoder_score"),
                        "hybrid_recall_score": candidate.get("hybrid_recall_score"),
                        "source_channels": candidate.get("source_channels") or [],
                        "match_method": match.get("method"),
                    }
                )

        reranked_top5_gold_ranks = [
            rank
            for rank, candidate in enumerate(reranked_top5, start=1)
            if runtime.match_case(candidate, case).get("matched")
        ]

        if not stage1_gold:
            failure_stage = "stage1_miss"
        elif reranked_top5_gold_ranks:
            failure_stage = "non_deterministic_between_full_and_diagnostic_run"
        elif not reranked_gold:
            failure_stage = "removed_before_cross_encoder"
        elif min(item["rank"] for item in reranked_gold) > 5:
            failure_stage = "cross_encoder_ranked_below_top5"
        else:
            failure_stage = "top_k_dependent_rerank_selection"

        top5 = failed.get("top5") or []
        diagnostics.append(
            {
                "query_id": case["query_id"],
                "question": case["question"],
                "report_year": scope["report_year"],
                "scenario": case["scenario"],
                "expected_answer": case.get("expected_answer"),
                "failure_stage": failure_stage,
                "stage1_candidate_count": len(candidates),
                "stage1_best_gold_rank": min((item["rank"] for item in stage1_gold), default=None),
                "reranked_best_gold_rank": min((item["rank"] for item in reranked_gold), default=None),
                "diagnostic_top5_gold_rank": min(reranked_top5_gold_ranks, default=None),
                "stage1_gold": stage1_gold,
                "reranked_gold": reranked_gold,
                "saved_top5_chunk_ids": [item.get("chunk_id") for item in top5],
                "saved_top5_headings": [item.get("heading_path") for item in top5],
                "saved_top5_previews": [item.get("content_preview") for item in top5],
                "cross_encoder": (rerank_trace or {}).get("cross_encoder") or {},
                "diagnostic_top5_cross_encoder": (rerank_top5_trace or {}).get("cross_encoder") or {},
            }
        )
        print(
            f"[{index}/{len(failures)}] {case['query_id']} stage={failure_stage} "
            f"stage1_gold={diagnostics[-1]['stage1_best_gold_rank']} "
            f"rerank_gold={diagnostics[-1]['reranked_best_gold_rank']}",
            flush=True,
        )

    OUTPUT_JSON.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    fields = [
        "query_id",
        "question",
        "report_year",
        "scenario",
        "expected_answer",
        "failure_stage",
        "stage1_candidate_count",
        "stage1_best_gold_rank",
        "reranked_best_gold_rank",
        "diagnostic_top5_gold_rank",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(diagnostics)


if __name__ == "__main__":
    asyncio.run(main())
