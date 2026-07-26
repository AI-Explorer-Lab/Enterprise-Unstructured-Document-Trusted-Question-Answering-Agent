from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from utils.config_loader import load_yaml_file
from utils.content_normalizer import normalize_whitespace


def _clean(value: Any) -> str:
    return normalize_whitespace(str(value or ""), preserve_newlines=False)


def _unique(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = _clean(value)
        if text and text not in result:
            result.append(text)
    return result


def load_intent_router_config(path: str | Path) -> Dict[str, Any]:
    loaded = load_yaml_file(Path(path).expanduser())
    config = loaded.get("intent_router", loaded) if isinstance(loaded, Mapping) else {}
    return dict(config) if isinstance(config, Mapping) else {}


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(float(left[index]) * float(right[index]) for index in range(size))
    left_norm = math.sqrt(sum(float(left[index]) ** 2 for index in range(size)))
    right_norm = math.sqrt(sum(float(right[index]) ** 2 for index in range(size)))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class SemanticRoute:
    candidates: tuple[Dict[str, Any], ...]
    route_status: str
    top_intent: str
    top1_score: float
    score_margin: float
    provider: str

    def as_dict(self) -> Dict[str, Any]:
        candidates = [
            {
                "intent_id": str(item.get("intent_id") or item.get("query_type") or ""),
                "query_type": str(item.get("intent_id") or item.get("query_type") or ""),
                "score": float(item.get("score") or 0.0),
                "matched_prototype_count": int(item.get("matched_prototype_count") or 0),
            }
            for item in self.candidates
        ]
        return {
            "candidates": candidates,
            "route_status": self.route_status,
            "top_intent": self.top_intent or None,
            "top1_score": self.top1_score,
            "score_margin": self.score_margin,
            "provider": self.provider,
            "decision": self.route_status,
            "top_query_type": self.top_intent,
            "top_score": self.top1_score,
            "margin": self.score_margin,
        }


class SemanticSkillRouter:
    """Retired Embedding router preserved only to reproduce the historical baseline."""

    def __init__(self, embedding_service: Any, config: Mapping[str, Any]) -> None:
        self.embedding_service = embedding_service
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.top_k = max(1, int(self.config.get("top_k", 3)))
        self.prototype_score_top_n = max(
            1,
            int(self.config.get("prototype_score_top_n", 3)),
        )
        self.min_intent_score = float(self.config.get("min_intent_score", 0.72))
        self.min_score_margin = float(self.config.get("min_score_margin", 0.08))
        raw_prototypes = (
            self.config.get("prototypes")
            if isinstance(self.config.get("prototypes"), Mapping)
            else {}
        )
        self.prototype_texts = {
            str(query_type): _unique(
                examples if isinstance(examples, list) else [examples]
            )
            for query_type, examples in raw_prototypes.items()
        }
        self._prototype_vectors: Dict[str, List[List[float]]] | None = None
        self._prototype_lock = asyncio.Lock()

    async def _load_prototype_vectors(self) -> Dict[str, List[List[float]]]:
        if self._prototype_vectors is not None:
            return self._prototype_vectors
        async with self._prototype_lock:
            if self._prototype_vectors is not None:
                return self._prototype_vectors
            prototype_items = [
                (query_type, prototype)
                for query_type, prototypes in self.prototype_texts.items()
                for prototype in prototypes
            ]
            vectors = await self.embedding_service.embed_texts(
                [prototype for _, prototype in prototype_items],
                use_cache=True,
                chunk_text=False,
                max_concurrency=max(1, min(len(prototype_items), 16)),
            )
            grouped: Dict[str, List[List[float]]] = {
                query_type: [] for query_type in self.prototype_texts
            }
            for (query_type, _), vector in zip(prototype_items, vectors):
                grouped[query_type].append(list(vector))
            self._prototype_vectors = grouped
            return self._prototype_vectors

    async def route(self, question: str) -> SemanticRoute:
        if not self.enabled or not self.prototype_texts:
            return SemanticRoute(tuple(), "disabled", "", 0.0, 0.0, "disabled")

        prototype_task = asyncio.create_task(self._load_prototype_vectors())
        question_task = asyncio.create_task(
            self.embedding_service.embed_text(
                question,
                use_cache=True,
                chunk_text=False,
            )
        )
        prototype_vectors, question_vector = await asyncio.gather(
            prototype_task,
            question_task,
        )
        scored = sorted(
            (
                self._score_query_type(question_vector, query_type, vectors)
                for query_type, vectors in prototype_vectors.items()
            ),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        top = scored[0] if scored else {"query_type": "", "score": 0.0}
        second_score = float(scored[1]["score"]) if len(scored) > 1 else 0.0
        top_score = float(top["score"])
        margin = round(top_score - second_score, 6)
        if top_score < self.min_intent_score:
            route_status = "unknown"
        elif margin < self.min_score_margin:
            route_status = "ambiguous"
        else:
            route_status = "accepted"
        return SemanticRoute(
            candidates=tuple(scored[: self.top_k]),
            route_status=route_status,
            top_intent=str(top["query_type"]) if route_status != "unknown" else "",
            top1_score=round(top_score, 6),
            score_margin=margin,
            provider=str(
                getattr(self.embedding_service, "provider_name", "unknown")
            ),
        )

    def _score_query_type(
        self,
        question_vector: Sequence[float],
        query_type: str,
        vectors: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        similarities = sorted(
            (
                _cosine_similarity(question_vector, vector)
                for vector in vectors
            ),
            reverse=True,
        )
        neighbor_count = min(self.prototype_score_top_n, len(similarities))
        nearest = similarities[:neighbor_count]
        score = sum(nearest) / neighbor_count if neighbor_count else 0.0
        return {
            "intent_id": query_type,
            "query_type": query_type,
            "score": round(score, 6),
            "matched_prototype_count": neighbor_count,
        }
