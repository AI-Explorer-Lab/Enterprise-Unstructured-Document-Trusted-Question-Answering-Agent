from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.llm import get_llm_service  # noqa: E402


PRIMARY_INTENTS = (
    "information_extraction",
    "metric_calculation",
    "comparison",
    "analysis",
    "summarization",
)
DECISIONS = ("select", "planner", "clarify", "reject")
ALL_LABELS = (*PRIMARY_INTENTS, "ambiguous", "unknown", "invalid")

SYSTEM_PROMPT = """You are the intent gate for a financial-report question-answering system.
Classify the entire user request, not merely its first clause.

Supported primary intents:
- information_extraction: retrieve an explicitly disclosed value, fact, name, passage, page, or list.
- metric_calculation: calculate a derived number, ratio, rate, difference, or change.
- comparison: compare periods, companies, business segments, documents, or values.
- analysis: explain causes, effects, risks, sustainability, meaning, or quality.
- summarization: summarize, outline, consolidate, or extract the main points.

Choose exactly one decision:
- select: one primary intent is sufficient to represent the whole requested task.
- planner: the request contains two or more distinct supported operations or dependent steps.
- clarify: the request is financial-report related or referential, but lacks enough information to
  determine the requested operation without missing conversation context.
- reject: the request is outside the supported financial-report QA scope.

Do not assume any previous conversation. Output exactly one JSON object without Markdown or
explanation:
{"decision":"select|planner|clarify|reject","intent_id":"one supported primary intent or empty string","sub_intents":["zero or more supported primary intents"]}

For select, intent_id must contain exactly one supported primary intent.
For planner, sub_intents must contain at least two distinct supported primary intents in execution order.
For clarify or reject, intent_id must be empty and sub_intents must be empty."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _unique_primary_intents(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        intent = str(value or "").strip()
        if intent in PRIMARY_INTENTS and intent not in result:
            result.append(intent)
    return result


def _normalize_decision(raw_response: str) -> dict[str, Any]:
    payload = _extract_json_object(raw_response)
    if payload is None:
        return {
            "valid": False,
            "decision": "invalid",
            "intent_id": "",
            "sub_intents": [],
            "validation_error": "response_is_not_json_object",
        }
    decision = str(payload.get("decision") or "").strip().lower()
    intent_id = str(payload.get("intent_id") or "").strip()
    sub_intents = _unique_primary_intents(payload.get("sub_intents"))
    error = ""
    if decision not in DECISIONS:
        error = "unsupported_decision"
    elif decision == "select" and intent_id not in PRIMARY_INTENTS:
        error = "select_requires_supported_intent"
    elif decision == "planner" and len(sub_intents) < 2:
        error = "planner_requires_two_sub_intents"
    elif decision in {"clarify", "reject"} and (intent_id or sub_intents):
        error = "guard_decision_requires_empty_intents"
    if decision != "select":
        intent_id = ""
    if decision != "planner":
        sub_intents = []
    return {
        "valid": not error,
        "decision": decision if decision in DECISIONS else "invalid",
        "intent_id": intent_id,
        "sub_intents": sub_intents,
        "validation_error": error,
    }


def _gold_label(row: dict[str, Any]) -> str:
    return str(row.get("expected_primary_intent") or row.get("expected_route_status") or "")


def _prediction_label(normalized: dict[str, Any]) -> str:
    if not normalized["valid"]:
        return "invalid"
    decision = normalized["decision"]
    if decision == "select":
        return str(normalized["intent_id"])
    if decision == "planner":
        return "ambiguous"
    return "unknown"


async def _classify_case(
    index: int,
    row: dict[str, Any],
    llm_service: Any,
    semaphore: asyncio.Semaphore,
    retries: int,
) -> tuple[int, dict[str, Any]]:
    started = time.perf_counter()
    raw_response = ""
    attempts = 0
    async with semaphore:
        for attempt in range(max(1, retries + 1)):
            attempts = attempt + 1
            response = await llm_service.complete(
                SYSTEM_PROMPT,
                json.dumps({"question": row["question"]}, ensure_ascii=False),
                max_tokens=320,
            )
            raw_response = str(response or "").strip()
            normalized = _normalize_decision(raw_response)
            if normalized["valid"]:
                break
            if attempt < retries:
                await asyncio.sleep(min(4.0, 0.5 * (2**attempt)))
    normalized = _normalize_decision(raw_response)
    gold_label = _gold_label(row)
    prediction_label = _prediction_label(normalized)
    result = {
        "case_id": row["case_id"],
        "question": row["question"],
        "difficulty": row.get("difficulty"),
        "tags": row.get("tags") or [],
        "gold_label": gold_label,
        "gold_primary_intent": row.get("expected_primary_intent"),
        "gold_route_status": row.get("expected_route_status"),
        "raw_response": raw_response,
        "decision": normalized["decision"],
        "intent_id": normalized["intent_id"],
        "sub_intents": normalized["sub_intents"],
        "valid_response": normalized["valid"],
        "validation_error": normalized["validation_error"],
        "prediction_label": prediction_label,
        "correct": prediction_label == gold_label,
        "attempts": attempts,
        "elapsed_ms": _round((time.perf_counter() - started) * 1000),
    }
    return index, result


def _classification_metrics(
    results: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        support = sum(row["gold_label"] == label for row in results)
        tp = sum(
            row["gold_label"] == label and row["prediction_label"] == label
            for row in results
        )
        fp = sum(
            row["gold_label"] != label and row["prediction_label"] == label
            for row in results
        )
        fn = sum(
            row["gold_label"] == label and row["prediction_label"] != label
            for row in results
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
        }
    return {
        "per_class": per_class,
        "macro_precision": _round(
            sum(per_class[label]["precision"] for label in labels) / len(labels)
        ),
        "macro_recall": _round(
            sum(per_class[label]["recall"] for label in labels) / len(labels)
        ),
        "macro_f1": _round(
            sum(per_class[label]["f1"] for label in labels) / len(labels)
        ),
    }


def _confusion_matrix(
    results: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    return {
        gold: {
            predicted: sum(
                row["gold_label"] == gold and row["prediction_label"] == predicted
                for row in results
            )
            for predicted in labels
        }
        for gold in labels
        if gold != "invalid"
    }


def _summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in results if row.get("gold_primary_intent")]
    ambiguous = [row for row in results if row["gold_label"] == "ambiguous"]
    unknown = [row for row in results if row["gold_label"] == "unknown"]
    primary_correct = sum(bool(row["correct"]) for row in primary)
    return {
        "case_count": len(results),
        "valid_response_count": sum(bool(row["valid_response"]) for row in results),
        "valid_response_rate": _round(
            _safe_div(sum(bool(row["valid_response"]) for row in results), len(results))
        ),
        "overall_correct": sum(bool(row["correct"]) for row in results),
        "overall_accuracy": _round(
            _safe_div(sum(bool(row["correct"]) for row in results), len(results))
        ),
        "primary_count": len(primary),
        "primary_correct": primary_correct,
        "primary_accuracy": _round(_safe_div(primary_correct, len(primary))),
        "primary_false_guard_count": sum(
            row["prediction_label"] in {"ambiguous", "unknown", "invalid"}
            for row in primary
        ),
        "primary_false_guard_rate": _round(
            _safe_div(
                sum(
                    row["prediction_label"] in {"ambiguous", "unknown", "invalid"}
                    for row in primary
                ),
                len(primary),
            )
        ),
        "ambiguous_count": len(ambiguous),
        "ambiguous_correct": sum(bool(row["correct"]) for row in ambiguous),
        "ambiguous_recall": _round(
            _safe_div(sum(bool(row["correct"]) for row in ambiguous), len(ambiguous))
        ),
        "unknown_count": len(unknown),
        "unknown_correct": sum(bool(row["correct"]) for row in unknown),
        "unknown_recall": _round(
            _safe_div(sum(bool(row["correct"]) for row in unknown), len(unknown))
        ),
        "decision_counts": dict(Counter(row["decision"] for row in results)),
    }


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row)
            + " |"
        )
    return "\n".join(lines)


def _build_markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    summary = payload["summary"]
    metrics = payload["metrics"]
    matrix = payload["confusion_matrix"]
    results = payload["results"]
    labels = (*PRIMARY_INTENTS, "ambiguous", "unknown")
    lines = [
        "# Full LLM Intent Decision Evaluation",
        "",
        "## Metadata",
        "",
        _table(
            ["Field", "Value"],
            [
                ["Cases", metadata["case_count"]],
                ["LLM provider", metadata["llm_provider"]],
                ["LLM model", metadata["llm_model"]],
                ["Dataset SHA-256", metadata["dataset_sha256"]],
                ["Started at UTC", metadata["started_at_utc"]],
                ["Elapsed seconds", metadata["elapsed_seconds"]],
                ["LLM call attempts", metadata["llm_call_attempts"]],
            ],
        ),
        "",
        "## Summary",
        "",
        _table(
            ["Metric", "Raw count", "Result"],
            [
                [
                    "Valid structured responses",
                    f"{summary['valid_response_count']}/{summary['case_count']}",
                    _percent(summary["valid_response_rate"]),
                ],
                [
                    "Primary intent accuracy",
                    f"{summary['primary_correct']}/{summary['primary_count']}",
                    _percent(summary["primary_accuracy"]),
                ],
                [
                    "Primary false guard rate",
                    f"{summary['primary_false_guard_count']}/{summary['primary_count']}",
                    _percent(summary["primary_false_guard_rate"]),
                ],
                [
                    "Ambiguous/planner recall",
                    f"{summary['ambiguous_correct']}/{summary['ambiguous_count']}",
                    _percent(summary["ambiguous_recall"]),
                ],
                [
                    "Unknown recall (clarify or reject)",
                    f"{summary['unknown_correct']}/{summary['unknown_count']}",
                    _percent(summary["unknown_recall"]),
                ],
                [
                    "Seven-label overall accuracy",
                    f"{summary['overall_correct']}/{summary['case_count']}",
                    _percent(summary["overall_accuracy"]),
                ],
                ["Seven-label macro-F1", "-", _percent(metrics["macro_f1"])],
            ],
        ),
        "",
        "## Per-class metrics",
        "",
        _table(
            ["Label", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1"],
            [
                [
                    label,
                    metrics["per_class"][label]["support"],
                    metrics["per_class"][label]["tp"],
                    metrics["per_class"][label]["fp"],
                    metrics["per_class"][label]["fn"],
                    _percent(metrics["per_class"][label]["precision"]),
                    _percent(metrics["per_class"][label]["recall"]),
                    _percent(metrics["per_class"][label]["f1"]),
                ]
                for label in labels
            ],
        ),
        "",
        "## Seven-label confusion matrix",
        "",
        _table(
            ["Gold \\ Predicted", *labels, "invalid"],
            [
                [gold, *(matrix[gold][predicted] for predicted in (*labels, "invalid"))]
                for gold in labels
            ],
        ),
        "",
        "## Incorrect cases",
        "",
    ]
    incorrect = [row for row in results if not row["correct"]]
    if incorrect:
        lines.append(
            _table(
                [
                    "Case",
                    "Gold",
                    "Predicted",
                    "Decision",
                    "Intent",
                    "Sub-intents",
                    "Question",
                ],
                [
                    [
                        row["case_id"],
                        row["gold_label"],
                        row["prediction_label"],
                        row["decision"],
                        row["intent_id"] or "-",
                        ", ".join(row["sub_intents"]) or "-",
                        row["question"],
                    ]
                    for row in incorrect
                ],
            )
        )
    else:
        lines.append("No incorrect cases.")
    lines.append("")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("intent_recognition_eval.jsonl"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).with_name("results") / "full_llm_intent_result.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).with_name("results") / "full_llm_intent_result.md",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    rows = _read_jsonl(args.data)
    if len(rows) != 240:
        raise ValueError(f"Expected 240 frozen cases, found {len(rows)}")

    llm_service = get_llm_service()
    trace = llm_service.trace_metadata()
    if not trace.get("available"):
        raise RuntimeError(
            "Configured real LLM is unavailable: "
            + str(trace.get("last_error") or "missing provider configuration")
        )

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        asyncio.create_task(
            _classify_case(index, row, llm_service, semaphore, max(0, args.retries))
        )
        for index, row in enumerate(rows)
    ]
    ordered_results: list[dict[str, Any] | None] = [None] * len(rows)
    completed = 0
    for task in asyncio.as_completed(tasks):
        index, result = await task
        ordered_results[index] = result
        completed += 1
        if completed % 25 == 0 or completed == len(tasks):
            print(f"progress={completed}/{len(tasks)}", flush=True)
    results = [row for row in ordered_results if row is not None]
    elapsed_seconds = time.perf_counter() - started

    labels = (*PRIMARY_INTENTS, "ambiguous", "unknown")
    payload = {
        "metadata": {
            "scope": "All cases classified directly by an LLM with select/planner/clarify/reject",
            "case_count": len(results),
            "llm_provider": trace.get("provider"),
            "llm_model": trace.get("model"),
            "dataset": str(args.data.resolve()),
            "dataset_sha256": _sha256(args.data),
            "started_at_utc": started_at.isoformat(),
            "elapsed_seconds": _round(elapsed_seconds),
            "llm_call_attempts": llm_service.call_attempt_count,
            "concurrency": max(1, args.concurrency),
            "max_attempts_per_case": max(1, args.retries + 1),
        },
        "summary": _summary(results),
        "metrics": _classification_metrics(results, labels),
        "confusion_matrix": _confusion_matrix(results, (*labels, "invalid")),
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))
    print(f"json={args.out_json.resolve()}")
    print(f"markdown={args.out_md.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
