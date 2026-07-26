from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = SCRIPT_DIR / "pgvector_recall_strict_failure_analysis.json"
OUTPUT_CSV = SCRIPT_DIR / "pgvector_recall_strict_failure_analysis.csv"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    cases = {row["query_id"]: row for row in load_jsonl(SCRIPT_DIR / "recall_eval_200.jsonl")}
    results = load_jsonl(SCRIPT_DIR / "pgvector_recall_results.jsonl")
    diagnostics = {
        row["query_id"]: row
        for row in json.loads(
            (SCRIPT_DIR / "pgvector_recall_failure_diagnostics.json").read_text(encoding="utf-8")
        )
    }
    output: list[dict[str, Any]] = []
    for result in results:
        if result["hit_at_5"]:
            continue
        case = cases[result["query_id"]]
        spans = [
            span
            for group in case["gold"]["evidence_groups"]
            for span in group["accepted_spans"]
        ]
        equivalent_candidates = [
            row for row in result["top5"] if row.get("gold_match") and not row.get("gold_strict_match")
        ]
        diagnostic = diagnostics.get(result["query_id"], {})
        if result["equivalent_hit_at_5"]:
            category = "equivalent_evidence_on_other_page"
        else:
            category = str(diagnostic.get("failure_stage") or "top5_miss_not_diagnosed")
        output.append(
            {
                "query_id": result["query_id"],
                "question": result["question"],
                "report_year": result["report_year"],
                "scenario": result["scenario"],
                "difficulty": result["difficulty"],
                "expected_answer": case.get("expected_answer"),
                "failure_category": category,
                "gold_pages": ",".join(str(span["page_number"]) for span in spans),
                "equivalent_rank": result.get("equivalent_matched_rank"),
                "equivalent_retrieved_pages": ",".join(
                    str(row.get("page_number")) for row in equivalent_candidates
                ),
                "equivalent_retrieved_headings": " | ".join(
                    str(row.get("heading_path") or "") for row in equivalent_candidates
                ),
                "stage1_best_gold_rank": diagnostic.get("stage1_best_gold_rank"),
                "reranked_best_gold_rank": diagnostic.get("reranked_best_gold_rank"),
                "diagnostic_top5_gold_rank": diagnostic.get("diagnostic_top5_gold_rank"),
            }
        )

    OUTPUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(output[0].keys())
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    counts: dict[str, int] = {}
    for row in output:
        counts[row["failure_category"]] = counts.get(row["failure_category"], 0) + 1
    print(json.dumps({"strict_failure_count": len(output), "categories": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
