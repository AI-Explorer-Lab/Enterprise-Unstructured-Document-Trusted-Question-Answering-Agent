from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.agent.structured_understanding import (  # noqa: E402
    SemanticSkillRouter,
    load_intent_router_config,
)
from service.embedding.embedding_service import (  # noqa: E402
    EmbeddingService,
    build_embedding_provider_from_config,
)


PRIMARY_INTENTS = (
    "information_extraction",
    "metric_calculation",
    "comparison",
    "analysis",
    "summarization",
)
GUARD_LABELS = ("ambiguous", "unknown")
ALL_LABELS = (*PRIMARY_INTENTS, *GUARD_LABELS)


class StrictEmbeddingService(EmbeddingService):
    """Use the configured provider without silent deterministic fallback."""

    def __init__(self, *args: Any, max_requests_per_second: float = 4.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_requests_per_second = max(0.1, float(max_requests_per_second))
        self._rate_lock = asyncio.Lock()
        self._next_request_at = 0.0
        self.provider_request_count = 0
        self.throttle_retry_count = 0

    async def _reserve_request_slot(self) -> None:
        async with self._rate_lock:
            now = asyncio.get_running_loop().time()
            wait_seconds = max(0.0, self._next_request_at - now)
            if wait_seconds:
                await asyncio.sleep(wait_seconds)
            self._next_request_at = (
                asyncio.get_running_loop().time()
                + 1.0 / self.max_requests_per_second
            )

    async def _embed_with_provider(self, text: str) -> list[float]:
        if self.provider is None:
            raise RuntimeError("Configured external embedding provider is unavailable.")
        for attempt in range(6):
            await self._reserve_request_slot()
            self.provider_request_count += 1
            try:
                response = self.provider(text)
                if inspect.isawaitable(response):
                    response = await response
                return self._normalize_dimension(response)
            except RuntimeError as exc:
                if "Throttling.RateQuota" not in str(exc) or attempt >= 5:
                    raise
                self.throttle_retry_count += 1
                await asyncio.sleep(min(8.0, 1.0 * (2**attempt)))
        raise RuntimeError("Embedding retry loop exited unexpectedly.")

    async def embed_texts(
        self,
        texts: Sequence[str] | Iterable[str],
        use_cache: bool = True,
        chunk_text: bool = False,
        max_concurrency: int = 8,
        timeout_seconds: float | None = None,
    ) -> list[list[float]]:
        return await super().embed_texts(
            texts,
            use_cache=use_cache,
            chunk_text=chunk_text,
            max_concurrency=min(2, max(1, int(max_concurrency))),
            timeout_seconds=timeout_seconds,
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return statistics.fmean(items) if items else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _prediction_label(result: dict[str, Any]) -> str:
    if result["predicted_route_status"] == "accepted":
        return str(result["top_candidate_intent"] or "")
    return str(result["predicted_route_status"])


def _gold_label(row: dict[str, Any]) -> str:
    return str(row.get("expected_primary_intent") or row.get("expected_route_status") or "")


def _top1_projection(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in results:
        if not row.get("gold_primary_intent"):
            continue
        item = dict(row)
        item["prediction_label"] = str(row.get("top_candidate_intent") or "")
        projected.append(item)
    return projected


def _classification_metrics(
    results: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(1 for row in results if row["gold_label"] == label and row["prediction_label"] == label)
        fp = sum(1 for row in results if row["gold_label"] != label and row["prediction_label"] == label)
        fn = sum(1 for row in results if row["gold_label"] == label and row["prediction_label"] != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[label] = {
            "support": sum(1 for row in results if row["gold_label"] == label),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
        }
    return {
        "per_class": per_class,
        "macro_precision": _round(_mean(float(per_class[label]["precision"]) for label in labels)),
        "macro_recall": _round(_mean(float(per_class[label]["recall"]) for label in labels)),
        "macro_f1": _round(_mean(float(per_class[label]["f1"]) for label in labels)),
    }


def _confusion_matrix(results: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        gold: {
            predicted: sum(
                1
                for row in results
                if row["gold_label"] == gold and row["prediction_label"] == predicted
            )
            for predicted in ALL_LABELS
        }
        for gold in ALL_LABELS
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
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
    return lines


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _build_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    raw_top1_metrics = payload["raw_top1_primary_metrics"]
    primary_metrics = payload["primary_metrics"]
    guard_metrics = payload["guard_metrics"]
    metadata = payload["metadata"]
    results = payload["results"]
    lines = [
        "# Intent Recognition Evaluation Result",
        "",
        "## Evaluation metadata",
        "",
        *_markdown_table(
            ["Field", "Value"],
            [
                ["Scope", metadata["scope"]],
                ["Dataset cases", metadata["case_count"]],
                ["Embedding provider", metadata["embedding_provider"]],
                ["Embedding model", metadata["embedding_model"]],
                ["Embedding dimension", metadata["embedding_dimension"]],
                ["Prototype count", metadata["prototype_count"]],
                ["Prototype score aggregation", metadata["prototype_score_aggregation"]],
                ["Minimum intent score", metadata["min_intent_score"]],
                ["Minimum score margin", metadata["min_score_margin"]],
                ["Dataset SHA-256", metadata["dataset_sha256"]],
                ["Intent config SHA-256", metadata["intent_config_sha256"]],
                ["Started at UTC", metadata["started_at_utc"]],
                ["Elapsed seconds", metadata["elapsed_seconds"]],
            ],
        ),
        "",
        "## Summary",
        "",
        *_markdown_table(
            ["Metric", "Raw count", "Result"],
            [
                [
                    "Primary Top-1 accuracy (ignoring guard threshold)",
                    f"{summary['primary_top1_correct']}/{summary['primary_count']}",
                    _percent(summary["primary_top1_accuracy"]),
                ],
                [
                    "Primary routed accuracy (correct and accepted)",
                    f"{summary['primary_routed_correct']}/{summary['primary_count']}",
                    _percent(summary["primary_routed_accuracy"]),
                ],
                [
                    "Primary accepted coverage",
                    f"{summary['primary_accepted_count']}/{summary['primary_count']}",
                    _percent(summary["primary_accepted_coverage"]),
                ],
                [
                    "Selective accuracy among accepted primary cases",
                    f"{summary['primary_accepted_correct']}/{summary['primary_accepted_count']}",
                    _percent(summary["primary_selective_accuracy"]),
                ],
                [
                    "Primary macro-F1 after guard",
                    "-",
                    _percent(primary_metrics["macro_f1"]),
                ],
                [
                    "Seven-label overall accuracy",
                    f"{summary['overall_correct']}/{summary['case_count']}",
                    _percent(summary["overall_accuracy"]),
                ],
                [
                    "Ambiguous guard recall",
                    f"{guard_metrics['ambiguous_correct']}/{guard_metrics['ambiguous_count']}",
                    _percent(guard_metrics["ambiguous_recall"]),
                ],
                [
                    "Unknown guard recall",
                    f"{guard_metrics['unknown_correct']}/{guard_metrics['unknown_count']}",
                    _percent(guard_metrics["unknown_recall"]),
                ],
                [
                    "False guard rate on primary cases",
                    f"{guard_metrics['false_guard_count']}/{summary['primary_count']}",
                    _percent(guard_metrics["false_guard_rate"]),
                ],
            ],
        ),
        "",
        "## Raw Top-1 primary intent metrics (before guard thresholds)",
        "",
        *_markdown_table(
            ["Intent", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1"],
            [
                [
                    label,
                    raw_top1_metrics["per_class"][label]["support"],
                    raw_top1_metrics["per_class"][label]["tp"],
                    raw_top1_metrics["per_class"][label]["fp"],
                    raw_top1_metrics["per_class"][label]["fn"],
                    _percent(raw_top1_metrics["per_class"][label]["precision"]),
                    _percent(raw_top1_metrics["per_class"][label]["recall"]),
                    _percent(raw_top1_metrics["per_class"][label]["f1"]),
                ]
                for label in PRIMARY_INTENTS
            ],
        ),
        "",
        f"Raw Top-1 macro-F1: **{_percent(raw_top1_metrics['macro_f1'])}**",
        "",
        "## Raw Top-1 five-class confusion matrix",
        "",
        *_markdown_table(
            ["Gold \\ Predicted", *PRIMARY_INTENTS],
            [
                [
                    gold,
                    *(
                        payload["raw_top1_confusion_matrix"][gold][predicted]
                        for predicted in PRIMARY_INTENTS
                    ),
                ]
                for gold in PRIMARY_INTENTS
            ],
        ),
        "",
        "## Primary intent metrics after guard",
        "",
        *_markdown_table(
            ["Intent", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1"],
            [
                [
                    label,
                    primary_metrics["per_class"][label]["support"],
                    primary_metrics["per_class"][label]["tp"],
                    primary_metrics["per_class"][label]["fp"],
                    primary_metrics["per_class"][label]["fn"],
                    _percent(primary_metrics["per_class"][label]["precision"]),
                    _percent(primary_metrics["per_class"][label]["recall"]),
                    _percent(primary_metrics["per_class"][label]["f1"]),
                ]
                for label in PRIMARY_INTENTS
            ],
        ),
        "",
        "## Guard metrics",
        "",
        *_markdown_table(
            ["Gold group", "Count", "Correct status", "Recall", "Average Top-1", "Average margin"],
            [
                [
                    "ambiguous",
                    guard_metrics["ambiguous_count"],
                    guard_metrics["ambiguous_correct"],
                    _percent(guard_metrics["ambiguous_recall"]),
                    f"{guard_metrics['ambiguous_avg_top1']:.4f}",
                    f"{guard_metrics['ambiguous_avg_margin']:.4f}",
                ],
                [
                    "unknown",
                    guard_metrics["unknown_count"],
                    guard_metrics["unknown_correct"],
                    _percent(guard_metrics["unknown_recall"]),
                    f"{guard_metrics['unknown_avg_top1']:.4f}",
                    f"{guard_metrics['unknown_avg_margin']:.4f}",
                ],
            ],
        ),
        "",
        "## Seven-label confusion matrix",
        "",
        *_markdown_table(
            ["Gold \\ Predicted", *ALL_LABELS],
            [
                [gold, *(payload["confusion_matrix"][gold][predicted] for predicted in ALL_LABELS)]
                for gold in ALL_LABELS
            ],
        ),
        "",
        "## Incorrect cases",
        "",
    ]
    incorrect = [row for row in results if row["gold_label"] != row["prediction_label"]]
    if incorrect:
        lines.extend(
            _markdown_table(
                [
                    "Case",
                    "Gold",
                    "Predicted",
                    "Top candidate",
                    "Top-1",
                    "Margin",
                    "Question",
                ],
                [
                    [
                        row["case_id"],
                        row["gold_label"],
                        row["prediction_label"],
                        row["top_candidate_intent"],
                        f"{row['top1_score']:.4f}",
                        f"{row['score_margin']:.4f}",
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


async def _evaluate_case(
    row: dict[str, Any],
    router: SemanticSkillRouter,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    started = time.perf_counter()
    async with semaphore:
        route = await router.route(str(row["question"]))
    elapsed_ms = (time.perf_counter() - started) * 1000
    route_dict = route.as_dict()
    candidates = list(route_dict.get("candidates") or [])
    top_candidate = candidates[0] if candidates else {}
    result = {
        "case_id": row["case_id"],
        "question": row["question"],
        "difficulty": row["difficulty"],
        "tags": row["tags"],
        "gold_label": _gold_label(row),
        "gold_primary_intent": row.get("expected_primary_intent"),
        "gold_route_status": row["expected_route_status"],
        "predicted_route_status": route_dict.get("route_status"),
        "predicted_primary_intent": route_dict.get("top_intent") or None,
        "top_candidate_intent": top_candidate.get("intent_id") or top_candidate.get("query_type") or None,
        "top1_score": float(route_dict.get("top1_score") or 0.0),
        "score_margin": float(route_dict.get("score_margin") or 0.0),
        "candidates": candidates,
        "elapsed_ms": _round(elapsed_ms),
    }
    result["prediction_label"] = _prediction_label(result)
    result["top1_correct"] = (
        result["top_candidate_intent"] == result["gold_primary_intent"]
        if result["gold_primary_intent"]
        else None
    )
    result["final_correct"] = result["prediction_label"] == result["gold_label"]
    return result


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
        default=Path(__file__).with_name("results") / "intent_recognition_result.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).with_name("results") / "intent_recognition_result.md",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--embedding-qps", type=float, default=4.0)
    parser.add_argument(
        "--render-existing-json",
        type=Path,
        help="Recompute derived tables from an existing result JSON without calling embeddings.",
    )
    args = parser.parse_args()

    if args.render_existing_json:
        payload = json.loads(args.render_existing_json.read_text(encoding="utf-8"))
        results = list(payload.get("results") or [])
        projected = _top1_projection(results)
        payload["raw_top1_primary_metrics"] = _classification_metrics(
            projected,
            PRIMARY_INTENTS,
        )
        payload["raw_top1_confusion_matrix"] = _confusion_matrix(projected)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        args.out_md.write_text(_build_markdown(payload), encoding="utf-8")
        print(f"json={args.out_json.resolve()}")
        print(f"markdown={args.out_md.resolve()}")
        return

    rows = _read_jsonl(args.data)
    if len(rows) != 240:
        raise ValueError(f"Expected 240 frozen test cases, found {len(rows)}")

    provider = build_embedding_provider_from_config()
    if provider is None:
        raise RuntimeError("Qwen embedding provider is not configured.")
    embedding_service = StrictEmbeddingService(
        provider=provider,
        max_requests_per_second=args.embedding_qps,
    )
    router_config = load_intent_router_config()
    router = SemanticSkillRouter(embedding_service, router_config)

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results = await asyncio.gather(
        *(_evaluate_case(row, router, semaphore) for row in rows)
    )
    elapsed_seconds = time.perf_counter() - started

    primary = [row for row in results if row["gold_primary_intent"]]
    accepted_primary = [row for row in primary if row["predicted_route_status"] == "accepted"]
    ambiguous = [row for row in results if row["gold_route_status"] == "ambiguous"]
    unknown = [row for row in results if row["gold_route_status"] == "unknown"]

    primary_top1_correct = sum(bool(row["top1_correct"]) for row in primary)
    primary_routed_correct = sum(bool(row["final_correct"]) for row in primary)
    primary_accepted_correct = sum(
        row["top_candidate_intent"] == row["gold_primary_intent"]
        for row in accepted_primary
    )
    overall_correct = sum(bool(row["final_correct"]) for row in results)
    summary = {
        "case_count": len(results),
        "primary_count": len(primary),
        "primary_top1_correct": primary_top1_correct,
        "primary_top1_accuracy": _round(_safe_div(primary_top1_correct, len(primary))),
        "primary_routed_correct": primary_routed_correct,
        "primary_routed_accuracy": _round(_safe_div(primary_routed_correct, len(primary))),
        "primary_accepted_count": len(accepted_primary),
        "primary_accepted_coverage": _round(_safe_div(len(accepted_primary), len(primary))),
        "primary_accepted_correct": primary_accepted_correct,
        "primary_selective_accuracy": _round(
            _safe_div(primary_accepted_correct, len(accepted_primary))
        ),
        "overall_correct": overall_correct,
        "overall_accuracy": _round(_safe_div(overall_correct, len(results))),
    }
    guard_metrics = {
        "ambiguous_count": len(ambiguous),
        "ambiguous_correct": sum(row["predicted_route_status"] == "ambiguous" for row in ambiguous),
        "ambiguous_recall": _round(
            _safe_div(
                sum(row["predicted_route_status"] == "ambiguous" for row in ambiguous),
                len(ambiguous),
            )
        ),
        "ambiguous_avg_top1": _round(_mean(row["top1_score"] for row in ambiguous)),
        "ambiguous_avg_margin": _round(_mean(row["score_margin"] for row in ambiguous)),
        "unknown_count": len(unknown),
        "unknown_correct": sum(row["predicted_route_status"] == "unknown" for row in unknown),
        "unknown_recall": _round(
            _safe_div(
                sum(row["predicted_route_status"] == "unknown" for row in unknown),
                len(unknown),
            )
        ),
        "unknown_avg_top1": _round(_mean(row["top1_score"] for row in unknown)),
        "unknown_avg_margin": _round(_mean(row["score_margin"] for row in unknown)),
        "false_guard_count": sum(row["predicted_route_status"] != "accepted" for row in primary),
        "false_guard_rate": _round(
            _safe_div(
                sum(row["predicted_route_status"] != "accepted" for row in primary),
                len(primary),
            )
        ),
        "predicted_route_status_counts": dict(
            Counter(str(row["predicted_route_status"]) for row in results)
        ),
    }
    config_path = PROJECT_ROOT / "config" / "intent_router.yaml"
    payload = {
        "metadata": {
            "scope": "semantic_router_only_no_retrieval_no_llm_fallback",
            "case_count": len(results),
            "embedding_provider": embedding_service.provider_name,
            "embedding_model": embedding_service.provider_model,
            "embedding_dimension": embedding_service.embedding_dim,
            "prototype_count": sum(len(items) for items in router.prototype_texts.values()),
            "prototype_score_aggregation": f"mean_top_{router.prototype_score_top_n}_cosine_per_intent",
            "top_k_candidates": router.top_k,
            "min_intent_score": router.min_intent_score,
            "min_score_margin": router.min_score_margin,
            "dataset_path": str(args.data.resolve()),
            "dataset_sha256": _sha256(args.data),
            "intent_config_path": str(config_path.resolve()),
            "intent_config_sha256": _sha256(config_path),
            "started_at_utc": started_at.isoformat(),
            "elapsed_seconds": _round(elapsed_seconds),
            "concurrency": max(1, args.concurrency),
            "embedding_qps_limit": args.embedding_qps,
            "provider_request_count": embedding_service.provider_request_count,
            "throttle_retry_count": embedding_service.throttle_retry_count,
        },
        "summary": summary,
        "raw_top1_primary_metrics": _classification_metrics(
            _top1_projection(results),
            PRIMARY_INTENTS,
        ),
        "raw_top1_confusion_matrix": _confusion_matrix(
            _top1_projection(results)
        ),
        "primary_metrics": _classification_metrics(results, PRIMARY_INTENTS),
        "guard_metrics": guard_metrics,
        "confusion_matrix": _confusion_matrix(results),
        "results": results,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(json.dumps({"summary": summary, "guard_metrics": guard_metrics}, ensure_ascii=False, indent=2))
    print(f"json={args.out_json.resolve()}")
    print(f"markdown={args.out_md.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
