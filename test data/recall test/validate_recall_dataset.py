from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent
DATASET_PATH = DATA_DIR / "recall_eval_200.jsonl"
REVIEW_PATH = DATA_DIR / "recall_eval_200_review.csv"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number}: each line must be an object")
        rows.append(value)
    return rows


def validate() -> None:
    errors: list[str] = []
    require(DATASET_PATH.is_file(), f"missing {DATASET_PATH.name}", errors)
    require(REVIEW_PATH.is_file(), f"missing {REVIEW_PATH.name}", errors)
    require(MANIFEST_PATH.is_file(), f"missing {MANIFEST_PATH.name}", errors)
    if errors:
        raise AssertionError("\n".join(errors))

    rows = read_jsonl(DATASET_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    with REVIEW_PATH.open(encoding="utf-8-sig", newline="") as stream:
        review_rows = list(csv.DictReader(stream))

    require(len(rows) == 200, f"expected 200 JSONL rows, got {len(rows)}", errors)
    require(len(review_rows) == 200, f"expected 200 CSV rows, got {len(review_rows)}", errors)
    require(manifest.get("case_count") == 200, "manifest case_count must be 200", errors)

    ids = [str(row.get("query_id") or "") for row in rows]
    expected_ids = [f"RET-{index:03d}" for index in range(1, 201)]
    require(ids == expected_ids, "query IDs must be contiguous RET-001 through RET-200", errors)

    questions = [str(row.get("question") or "").strip() for row in rows]
    require(all(questions), "every row must contain a non-empty question", errors)
    require(len(set(questions)) == len(questions), "questions must be unique", errors)

    year_counts = Counter(str((row.get("scope") or {}).get("report_year")) for row in rows)
    require(year_counts == Counter({"2024": 100, "2025": 100}), f"unexpected year counts: {year_counts}", errors)

    difficulty_counts = Counter(str(row.get("difficulty") or "") for row in rows)
    require(set(difficulty_counts) <= {"simple", "medium"}, f"unsupported difficulty: {difficulty_counts}", errors)
    require(difficulty_counts.get("simple", 0) >= 180, "at least 180 cases must be simple", errors)

    for index, row in enumerate(rows, start=1):
        prefix = f"RET-{index:03d}"
        require(row.get("answerable_in_corpus") is True, f"{prefix}: must be answerable", errors)
        require(bool(str(row.get("expected_answer") or "").strip()), f"{prefix}: missing expected_answer", errors)
        require(row.get("split") == "frozen_test", f"{prefix}: split must be frozen_test", errors)
        require(
            row.get("annotation_status") == "source_located_single_pass",
            f"{prefix}: unexpected annotation_status",
            errors,
        )

        gold = row.get("gold") or {}
        groups = gold.get("evidence_groups") or []
        require(len(groups) == 1, f"{prefix}: exactly one evidence group is required", errors)
        if len(groups) != 1:
            continue
        spans = groups[0].get("accepted_spans") or []
        require(len(spans) >= 1, f"{prefix}: at least one accepted span is required", errors)
        for span in spans:
            source_path = PROJECT_ROOT / str(span.get("source_path") or "")
            require(source_path.is_file(), f"{prefix}: missing source file {source_path}", errors)
            require(int(span.get("page_number") or 0) > 0, f"{prefix}: invalid page number", errors)
            require(bool(str(span.get("heading_path") or "").strip()), f"{prefix}: missing heading path", errors)
            require(bool(str(span.get("anchor_text") or "").strip()), f"{prefix}: missing anchor text", errors)
            require(span.get("relevance_grade") == 2, f"{prefix}: gold relevance grade must be 2", errors)
            require(
                span.get("evidence_type") in {"table", "text"},
                f"{prefix}: evidence_type must be table or text",
                errors,
            )

        policy = row.get("metric_policy") or {}
        require(policy.get("primary") == "evidence_recall@5", f"{prefix}: unexpected primary metric", errors)
        require(policy.get("gold_group_count") == 1, f"{prefix}: gold_group_count must be 1", errors)

    review_ids = [str(row.get("query_id") or "") for row in review_rows]
    require(review_ids == ids, "CSV review rows must align with JSONL IDs", errors)

    for source in manifest.get("sources", []):
        path = PROJECT_ROOT / str(source.get("path") or "")
        require(path.is_file(), f"manifest source missing: {path}", errors)
        if path.is_file():
            require(
                sha256(path) == source.get("sha256"),
                f"manifest source hash drifted: {source.get('path')}",
                errors,
            )

    jsonl_hash = sha256(DATASET_PATH)
    require(re.fullmatch(r"[0-9a-f]{64}", jsonl_hash) is not None, "invalid dataset hash", errors)

    if errors:
        raise AssertionError("\n".join(errors))

    summary = {
        "status": "PASS",
        "case_count": len(rows),
        "year_counts": dict(sorted(year_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "scenario_counts": dict(sorted(Counter(row["scenario"] for row in rows).items())),
        "dataset_sha256": jsonl_hash,
        "source_hashes_verified": len(manifest.get("sources", [])),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    validate()
