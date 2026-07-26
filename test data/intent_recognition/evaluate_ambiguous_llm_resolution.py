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
from typing import Any, Mapping, Sequence


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
ALL_LABELS = (*PRIMARY_INTENTS, "ambiguous", "unknown")
SYSTEM_PROMPT = (
    "Choose the best intent only from the supplied embedding candidates. "
    "Do not invent another intent and do not extract slots."
)


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


def _parse_production_contract(text: str, allowed: Sequence[str]) -> str:
    payload = _extract_json_object(text)
    intent_id = str((payload or {}).get("intent_id") or "").strip()
    return intent_id if intent_id in allowed else ""


def _parse_semantic_decision(text: str, allowed: Sequence[str]) -> str:
    production_value = _parse_production_contract(text, allowed)
    if production_value:
        return production_value
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.strip("\"'` \t\r\n。，")
    if raw in allowed:
        return raw
    matches = [intent for intent in allowed if re.search(rf"\b{re.escape(intent)}\b", raw)]
    return matches[0] if len(matches) == 1 else ""


def _allowed_candidates(row: Mapping[str, Any]) -> list[str]:
    allowed: list[str] = []
    for candidate in list(row.get("candidates") or [])[:3]:
        if not isinstance(candidate, Mapping):
            continue
        intent = str(candidate.get("query_type") or candidate.get("intent_id") or "").strip()
        if intent in PRIMARY_INTENTS and intent not in allowed:
            allowed.append(intent)
    return allowed


async def _resolve_case(
    row: dict[str, Any],
    llm_service: Any,
    semaphore: asyncio.Semaphore,
    retries: int,
) -> dict[str, Any]:
    allowed = _allowed_candidates(row)
    payload = {
        "question": row["question"],
        "candidate_intents": [
            dict(candidate)
            for candidate in list(row.get("candidates") or [])[:3]
            if isinstance(candidate, Mapping)
            and str(candidate.get("query_type") or candidate.get("intent_id") or "").strip()
            in allowed
        ],
        "conversation_context": {},
        "allowed_intent_ids": allowed,
    }
    started = time.perf_counter()
    raw_response = ""
    production_retry_response = ""
    attempts = 0
    async with semaphore:
        for attempt in range(max(1, retries + 1)):
            attempts = attempt + 1
            response = await llm_service.complete(
                SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=240,
            )
            raw_response = str(response or "").strip()
            if raw_response:
                break
            if attempt < retries:
                await asyncio.sleep(min(4.0, 0.5 * (2**attempt)))
        production_resolution = _parse_production_contract(raw_response, allowed)
        if raw_response and not production_resolution:
            attempts += 1
            retry_response = await llm_service.complete(
                SYSTEM_PROMPT,
                json.dumps(payload, ensure_ascii=False),
                max_tokens=240,
            )
            production_retry_response = str(retry_response or "").strip()
            production_resolution = _parse_production_contract(
                production_retry_response,
                allowed,
            )
    semantic_resolution = _parse_semantic_decision(raw_response, allowed)
    return {
        "case_id": row["case_id"],
        "question": row["question"],
        "difficulty": row.get("difficulty"),
        "gold_label": row["gold_label"],
        "gold_primary_intent": row.get("gold_primary_intent"),
        "embedding_candidates": list(row.get("candidates") or []),
        "allowed_intent_ids": allowed,
        "gold_in_candidates": (
            row.get("gold_primary_intent") in allowed
            if row.get("gold_primary_intent")
            else None
        ),
        "raw_response": raw_response,
        "production_retry_response": production_retry_response or None,
        "production_contract_resolution": production_resolution or None,
        "semantic_resolution": semantic_resolution or None,
        "production_contract_valid": bool(production_resolution),
        "semantic_resolution_valid": bool(semantic_resolution),
        "semantic_resolution_correct": (
            semantic_resolution == row.get("gold_primary_intent")
            if row.get("gold_primary_intent")
            else False
        ),
        "attempts": attempts,
        "elapsed_ms": _round((time.perf_counter() - started) * 1000),
    }


def _final_projection(
    embedding_results: Sequence[dict[str, Any]],
    resolutions: Mapping[str, dict[str, Any]],
    resolution_field: str,
) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for row in embedding_results:
        item = dict(row)
        if row.get("predicted_route_status") == "accepted":
            final_label = str(row.get("top_candidate_intent") or "")
            final_status = "accepted"
            source = "embedding"
        elif row.get("predicted_route_status") == "unknown":
            final_label = "unknown"
            final_status = "unknown"
            source = "embedding_unknown"
        else:
            resolution = resolutions.get(str(row["case_id"]), {})
            resolved_intent = str(resolution.get(resolution_field) or "")
            if resolved_intent in PRIMARY_INTENTS:
                final_label = resolved_intent
                final_status = "accepted"
                source = "llm_candidate_disambiguation"
            else:
                final_label = "ambiguous"
                final_status = "ambiguous"
                source = "embedding_ambiguous"
        item.update(
            {
                "final_prediction_label": final_label,
                "final_route_status": final_status,
                "final_source": source,
                "final_correct": final_label == row["gold_label"],
            }
        )
        projected.append(item)
    return projected


def _metrics(projected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in projected if row.get("gold_primary_intent")]
    accepted_primary = [row for row in primary if row["final_route_status"] == "accepted"]
    accepted_primary_correct = sum(
        row["final_prediction_label"] == row["gold_primary_intent"]
        for row in accepted_primary
    )
    ambiguous = [row for row in projected if row["gold_label"] == "ambiguous"]
    unknown = [row for row in projected if row["gold_label"] == "unknown"]
    return {
        "primary_count": len(primary),
        "primary_accepted_count": len(accepted_primary),
        "primary_accepted_coverage": _round(_safe_div(len(accepted_primary), len(primary))),
        "primary_accepted_correct": accepted_primary_correct,
        "primary_selective_accuracy": _round(
            _safe_div(accepted_primary_correct, len(accepted_primary))
        ),
        "primary_routed_accuracy": _round(
            _safe_div(accepted_primary_correct, len(primary))
        ),
        "false_guard_count": len(primary) - len(accepted_primary),
        "false_guard_rate": _round(
            _safe_div(len(primary) - len(accepted_primary), len(primary))
        ),
        "ambiguous_correct": sum(row["final_prediction_label"] == "ambiguous" for row in ambiguous),
        "ambiguous_count": len(ambiguous),
        "ambiguous_recall": _round(
            _safe_div(
                sum(row["final_prediction_label"] == "ambiguous" for row in ambiguous),
                len(ambiguous),
            )
        ),
        "unknown_correct": sum(row["final_prediction_label"] == "unknown" for row in unknown),
        "unknown_count": len(unknown),
        "unknown_recall": _round(
            _safe_div(
                sum(row["final_prediction_label"] == "unknown" for row in unknown),
                len(unknown),
            )
        ),
        "overall_correct": sum(bool(row["final_correct"]) for row in projected),
        "overall_count": len(projected),
        "overall_accuracy": _round(
            _safe_div(sum(bool(row["final_correct"]) for row in projected), len(projected))
        ),
    }


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| "
        + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row)
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def _build_markdown(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    subset = payload["ambiguous_subset"]
    production = payload["production_contract_metrics"]
    semantic = payload["semantic_resolution_metrics"]
    resolution_rows = payload["resolutions"]
    per_intent = subset["primary_per_intent"]
    return "\n".join(
        [
            "# Ambiguous Candidate LLM Resolution Evaluation",
            "",
            "## Metadata",
            "",
            _table(
                ["Field", "Value"],
                [
                    ["Input cases", metadata["input_case_count"]],
                    ["Embedding-ambiguous cases sent to LLM", metadata["ambiguous_case_count"]],
                    ["LLM provider", metadata["llm_provider"]],
                    ["LLM model", metadata["llm_model"]],
                    ["Input result SHA-256", metadata["input_sha256"]],
                    ["Started at UTC", metadata["started_at_utc"]],
                    ["Elapsed seconds", metadata["elapsed_seconds"]],
                    ["LLM call attempts", metadata["llm_call_attempts"]],
                ],
            ),
            "",
            "## Ambiguous subset",
            "",
            _table(
                ["Metric", "Count", "Result"],
                [
                    ["Primary cases", subset["primary_count"], "-"],
                    [
                        "Primary gold intent present in Top-3",
                        f"{subset['primary_gold_in_candidates']}/{subset['primary_count']}",
                        _percent(subset["primary_candidate_ceiling"]),
                    ],
                    [
                        "Production JSON contract parsed",
                        f"{subset['production_contract_valid']}/{subset['case_count']}",
                        _percent(subset["production_contract_valid_rate"]),
                    ],
                    [
                        "Semantic decision parsed (JSON or plain label)",
                        f"{subset['semantic_resolution_valid']}/{subset['case_count']}",
                        _percent(subset["semantic_resolution_valid_rate"]),
                    ],
                    [
                        "Resolved primary accuracy",
                        f"{subset['primary_semantic_correct']}/{subset['primary_count']}",
                        _percent(subset["primary_semantic_accuracy"]),
                    ],
                ],
            ),
            "",
            "## Primary ambiguous cases by intent",
            "",
            _table(
                ["Intent", "Cases", "Gold in Top-3", "LLM correct", "Accuracy"],
                [
                    [
                        intent,
                        per_intent[intent]["count"],
                        per_intent[intent]["gold_in_candidates"],
                        per_intent[intent]["correct"],
                        _percent(per_intent[intent]["accuracy"]),
                    ]
                    for intent in PRIMARY_INTENTS
                ],
            ),
            "",
            "## Full 240-case projection",
            "",
            _table(
                ["Metric", "Current production parser", "Semantic LLM decision"],
                [
                    [
                        "Primary routed accuracy",
                        _percent(production["primary_routed_accuracy"]),
                        _percent(semantic["primary_routed_accuracy"]),
                    ],
                    [
                        "Primary accepted coverage",
                        _percent(production["primary_accepted_coverage"]),
                        _percent(semantic["primary_accepted_coverage"]),
                    ],
                    [
                        "Selective accuracy among accepted primary",
                        _percent(production["primary_selective_accuracy"]),
                        _percent(semantic["primary_selective_accuracy"]),
                    ],
                    [
                        "False guard rate on primary",
                        _percent(production["false_guard_rate"]),
                        _percent(semantic["false_guard_rate"]),
                    ],
                    [
                        "Ambiguous guard recall",
                        _percent(production["ambiguous_recall"]),
                        _percent(semantic["ambiguous_recall"]),
                    ],
                    [
                        "Unknown guard recall",
                        _percent(production["unknown_recall"]),
                        _percent(semantic["unknown_recall"]),
                    ],
                    [
                        "Seven-label overall accuracy",
                        _percent(production["overall_accuracy"]),
                        _percent(semantic["overall_accuracy"]),
                    ],
                ],
            ),
            "",
            "## Incorrect or unresolved LLM cases",
            "",
            _table(
                ["Case", "Gold", "Top-3", "Raw response", "Semantic decision", "Question"],
                [
                    [
                        row["case_id"],
                        row["gold_label"],
                        ", ".join(row["allowed_intent_ids"]),
                        row["raw_response"] or "<empty>",
                        row["semantic_resolution"] or "<unresolved>",
                        row["question"],
                    ]
                    for row in resolution_rows
                    if not row["semantic_resolution_correct"]
                ],
            ),
            "",
            "Note: the production-contract column reproduces the current JSON-only parser. "
            "The semantic-decision column additionally accepts a plain allowed intent label, "
            "so it measures model judgment rather than current parser compatibility.",
            "",
        ]
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(__file__).with_name("results")
        / "intent_recognition_result_score054_margin005.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).with_name("results")
        / "ambiguous_llm_resolution_score054_margin005.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path(__file__).with_name("results")
        / "ambiguous_llm_resolution_score054_margin005.md",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()

    source = json.loads(args.input_json.read_text(encoding="utf-8"))
    embedding_results = list(source.get("results") or [])
    if len(embedding_results) != 240:
        raise ValueError(f"Expected 240 embedding results, found {len(embedding_results)}")
    ambiguous_rows = [
        row for row in embedding_results if row.get("predicted_route_status") == "ambiguous"
    ]

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
    resolutions = await asyncio.gather(
        *(
            _resolve_case(
                row,
                llm_service,
                semaphore,
                max(0, args.retries),
            )
            for row in ambiguous_rows
        )
    )
    elapsed_seconds = time.perf_counter() - started
    by_id = {str(row["case_id"]): row for row in resolutions}

    primary_resolutions = [row for row in resolutions if row.get("gold_primary_intent")]
    per_intent: dict[str, dict[str, Any]] = {}
    for intent in PRIMARY_INTENTS:
        rows = [row for row in primary_resolutions if row["gold_primary_intent"] == intent]
        correct = sum(bool(row["semantic_resolution_correct"]) for row in rows)
        per_intent[intent] = {
            "count": len(rows),
            "gold_in_candidates": sum(bool(row["gold_in_candidates"]) for row in rows),
            "correct": correct,
            "accuracy": _round(_safe_div(correct, len(rows))),
        }

    production_projection = _final_projection(
        embedding_results,
        by_id,
        "production_contract_resolution",
    )
    semantic_projection = _final_projection(
        embedding_results,
        by_id,
        "semantic_resolution",
    )
    payload = {
        "metadata": {
            "scope": "Embedding ambiguous cases resolved by bounded Top-3 LLM classification",
            "input_case_count": len(embedding_results),
            "ambiguous_case_count": len(ambiguous_rows),
            "llm_provider": trace.get("provider"),
            "llm_model": trace.get("model"),
            "input_json": str(args.input_json.resolve()),
            "input_sha256": _sha256(args.input_json),
            "started_at_utc": started_at.isoformat(),
            "elapsed_seconds": _round(elapsed_seconds),
            "llm_call_attempts": llm_service.call_attempt_count,
        },
        "ambiguous_subset": {
            "case_count": len(resolutions),
            "primary_count": len(primary_resolutions),
            "primary_gold_in_candidates": sum(
                bool(row["gold_in_candidates"]) for row in primary_resolutions
            ),
            "primary_candidate_ceiling": _round(
                _safe_div(
                    sum(bool(row["gold_in_candidates"]) for row in primary_resolutions),
                    len(primary_resolutions),
                )
            ),
            "production_contract_valid": sum(
                bool(row["production_contract_valid"]) for row in resolutions
            ),
            "production_contract_valid_rate": _round(
                _safe_div(
                    sum(bool(row["production_contract_valid"]) for row in resolutions),
                    len(resolutions),
                )
            ),
            "semantic_resolution_valid": sum(
                bool(row["semantic_resolution_valid"]) for row in resolutions
            ),
            "semantic_resolution_valid_rate": _round(
                _safe_div(
                    sum(bool(row["semantic_resolution_valid"]) for row in resolutions),
                    len(resolutions),
                )
            ),
            "primary_semantic_correct": sum(
                bool(row["semantic_resolution_correct"]) for row in primary_resolutions
            ),
            "primary_semantic_accuracy": _round(
                _safe_div(
                    sum(bool(row["semantic_resolution_correct"]) for row in primary_resolutions),
                    len(primary_resolutions),
                )
            ),
            "gold_group_counts": dict(Counter(row["gold_label"] for row in resolutions)),
            "primary_per_intent": per_intent,
        },
        "production_contract_metrics": _metrics(production_projection),
        "semantic_resolution_metrics": _metrics(semantic_projection),
        "resolutions": resolutions,
        "production_projection": production_projection,
        "semantic_projection": semantic_projection,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["ambiguous_subset"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["production_contract_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["semantic_resolution_metrics"], ensure_ascii=False, indent=2))
    print(f"json={args.out_json.resolve()}")
    print(f"markdown={args.out_md.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
