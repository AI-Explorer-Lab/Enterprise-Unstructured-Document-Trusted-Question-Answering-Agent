from __future__ import annotations

from typing import Any, Dict, Mapping


_STEP_PHASES = {
    "load_session": "load_session",
    "conversation_context": "conversation_context",
    "intent_and_planner": "intent_slot_understanding_agent",
    "clarify_gate": "clarify_gate",
    "build_clarify_response": "answer_generation",
    "retrieve_evidence": "parallel_hybrid_retrieval",
    "evaluate_evidence": "evidence_decision",
    "retry_retrieval": "retry_retrieval",
    "build_answer_response": "answer_generation",
    "persist_response": "finalize_response",
}


def _state_from_event(event: Any) -> Mapping[str, Any]:
    content = getattr(event, "content", None)
    return content if isinstance(content, Mapping) else {}


def _phase_observation(
    state: Mapping[str, Any],
    phase: str,
) -> Mapping[str, Any]:
    observations = state.get("observations")
    if not isinstance(observations, list):
        return {}
    for item in reversed(observations):
        if isinstance(item, Mapping) and item.get("phase") == phase:
            return item
    return {}


def adapt_agno_event(event: Any) -> Dict[str, Any] | None:
    """Map Agno executor events to the existing public progress schema."""

    event_name = str(getattr(event, "event", "") or type(event).__name__)
    if event_name not in {"StepStarted", "StepCompleted", "StepError"}:
        return None
    step_name = str(getattr(event, "step_name", "") or "")
    state = _state_from_event(event)
    if step_name == "response_cache_lookup":
        if event_name != "StepCompleted" or not bool(
            state.get("response_cache_hit")
        ):
            return None
        phase = "parallel_hybrid_retrieval"
    else:
        phase = _STEP_PHASES.get(step_name)
        if not phase:
            return None
    evidence = state.get("evidence") if isinstance(state.get("evidence"), list) else []
    observation = _phase_observation(state, phase)
    status = {
        "StepStarted": "running",
        "StepCompleted": "completed",
        "StepError": "failed",
    }[event_name]
    payload: Dict[str, Any] = {
        "phase": phase,
        "stage": phase,
        "status": status,
        "duration_ms": int(observation.get("duration_ms") or 0),
        "timed": "duration_ms" in observation,
        "cache_hit": bool(
            state.get("response_cache_hit") or observation.get("cache_hit")
        ),
        "cache_precheck_hit": bool(
            state.get("response_cache_hit")
            or observation.get("cache_precheck_hit")
        ),
        "query_expansion_cache_hit": bool(
            observation.get("query_expansion_cache_hit")
        ),
        "query_expansion_skipped": (
            "final_response_cache_hit"
            if bool(state.get("response_cache_hit"))
            else str(observation.get("query_expansion_skipped") or "")
        ),
        "llm_answer_cache_hit": bool(
            state.get("llm_answer_cache_hit")
            or observation.get("llm_answer_cache_hit")
        ),
        "llm_query_expansion_used": bool(
            observation.get("llm_query_expansion_used")
        ),
        "evidence_count": len(evidence),
        "session_id": str(state.get("sid") or ""),
        "workflow_run_id": str(getattr(event, "run_id", "") or ""),
    }
    error = str(getattr(event, "error", "") or "")
    if error:
        payload["error"] = error
    return payload
