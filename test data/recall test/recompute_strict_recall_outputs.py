from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "run_pgvector_recall_eval.py"


def load_eval_module() -> Any:
    spec = importlib.util.spec_from_file_location("recall_eval_runtime", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    runtime = load_eval_module()
    rows = load_jsonl(runtime.RESULT_JSONL_PATH)
    previous_summary = json.loads(runtime.SUMMARY_JSON_PATH.read_text(encoding="utf-8"))

    for row in rows:
        old_rank = row.get("matched_rank")
        row["equivalent_hit_at_1"] = old_rank == 1
        row["equivalent_hit_at_3"] = old_rank is not None and old_rank <= 3
        row["equivalent_hit_at_5"] = old_rank is not None and old_rank <= 5
        row["equivalent_matched_rank"] = old_rank
        row["equivalent_matched_method"] = row.get("matched_method") or ""
        row["equivalent_reciprocal_rank_at_5"] = 1.0 / old_rank if old_rank else 0.0
        row["equivalent_ndcg_at_5"] = (
            1.0 / runtime.math.log2(old_rank + 1) if old_rank else 0.0
        )

        strict_candidate = None
        for candidate in row.get("top5") or []:
            strict_match = bool(
                candidate.get("gold_match")
                and (
                    candidate.get("gold_page_match")
                    or row["query_id"] in runtime.PAGE_METADATA_EXCEPTIONS
                )
            )
            candidate["gold_strict_match"] = strict_match
            if strict_match and strict_candidate is None:
                strict_candidate = candidate

        strict_rank = strict_candidate.get("rank") if strict_candidate else None
        row["hit_at_1"] = strict_rank == 1
        row["hit_at_3"] = strict_rank is not None and strict_rank <= 3
        row["hit_at_5"] = strict_rank is not None and strict_rank <= 5
        row["matched_rank"] = strict_rank
        row["matched_method"] = strict_candidate.get("gold_match_method") if strict_candidate else ""
        row["matched_page_consistent"] = strict_candidate.get("gold_page_match") if strict_candidate else None
        row["reciprocal_rank_at_5"] = 1.0 / strict_rank if strict_rank else 0.0
        row["ndcg_at_5"] = 1.0 / runtime.math.log2(strict_rank + 1) if strict_rank else 0.0

    embedding = SimpleNamespace(
        provider_name=previous_summary["retrieval_configuration"]["embedding_provider"],
        provider_model=previous_summary["retrieval_configuration"]["embedding_model"],
    )
    runtime.write_outputs(
        rows=rows,
        documents=previous_summary["corpus_documents"],
        preflight=previous_summary["gold_preflight"],
        embedding_service=embedding,
        dataset_hash=previous_summary["dataset_sha256"],
    )
    overall = runtime.build_metric_rows(rows)[0]
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
