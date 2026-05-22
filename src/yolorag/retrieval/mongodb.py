from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping

from yolorag.knowledge.models import SearchResult
from yolorag.knowledge.stores.base import KnowledgeStore
from yolorag.retrieval.base import Document, RetrievalResult, RetrievalTrace


DEFAULT_CANDIDATE_LIMIT = 40
CANDIDATE_MULTIPLIER = 8
DEFAULT_RERANK_ENDPOINT = "https://ai.mongodb.com/v1/rerank"
DEFAULT_RERANK_MODEL = "rerank-2.5-lite"


@dataclass(frozen=True)
class RerankedSearchResult:
    result: SearchResult
    relevance_score: float


class MongoVectorRetriever:
    def __init__(
        self,
        store: KnowledgeStore,
        *,
        filters: Mapping[str, Any] | None = None,
        reranker: MongoReranker | None = None,
        candidate_limit: int | None = None,
    ) -> None:
        if candidate_limit is not None and candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than 0.")
        self.store = store
        self.filters = dict(filters or {})
        self.reranker = reranker
        self.candidate_limit = candidate_limit

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        retrieval_started = time.perf_counter()
        candidate_limit = self._candidate_limit(top_k)
        vector_started = time.perf_counter()
        candidates = await asyncio.to_thread(
            self.store.vector_search,
            query,
            limit=candidate_limit,
            filters=self.filters or None,
        )
        vector_search_ms = _elapsed_ms(vector_started)

        if self.reranker is None:
            reranked = [
                RerankedSearchResult(result=result, relevance_score=result.score or 0.0)
                for result in candidates[:top_k]
            ]
            rerank_ms = 0
        else:
            rerank_started = time.perf_counter()
            reranked = await asyncio.to_thread(
                self.reranker.rerank,
                query,
                candidates,
                top_k,
            )
            rerank_ms = _elapsed_ms(rerank_started)

        trace = RetrievalTrace(
            provider=getattr(self.store, "provider_name", "unknown"),
            total_ms=_elapsed_ms(retrieval_started),
            vector_search_ms=vector_search_ms,
            rerank_ms=rerank_ms,
            candidate_count=len(candidates),
            returned_count=len(reranked),
            reranked=self.reranker is not None,
        )
        return [
            RetrievalResult(
                document=Document(
                    id=item.result.record.chunk_id,
                    title=item.result.record.title,
                    content=item.result.record.text,
                    metadata={
                        "url": item.result.record.url or "",
                        "source_path": item.result.record.source_path,
                        "doc_id": item.result.record.doc_id,
                        "kind": item.result.record.kind,
                        "vector_score": str(item.result.score or ""),
                        "rerank_score": str(item.relevance_score),
                    },
                ),
                score=item.relevance_score,
                reason=_reason(
                    vector_score=item.result.score,
                    rerank_score=item.relevance_score,
                    rank=index,
                    reranked=self.reranker is not None,
                ),
                trace=trace,
            )
            for index, item in enumerate(reranked, start=1)
        ]

    def _candidate_limit(self, top_k: int) -> int:
        if self.candidate_limit is not None:
            return max(top_k, self.candidate_limit)
        if self.reranker is None:
            return top_k
        return max(DEFAULT_CANDIDATE_LIMIT, top_k * CANDIDATE_MULTIPLIER)


class MongoReranker:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_RERANK_MODEL,
        endpoint: str = DEFAULT_RERANK_ENDPOINT,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> MongoReranker:
        api_key = os.getenv("YOLORAG_MONGODB_AI_API_KEY") or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing YOLORAG_MONGODB_AI_API_KEY or VOYAGE_API_KEY for MongoDB reranking.")
        return cls(
            api_key=api_key,
            model=os.getenv("YOLORAG_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        )

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[RerankedSearchResult]:
        if not candidates or top_k <= 0:
            return []

        payload = {
            "query": query,
            "documents": [_document_text(candidate) for candidate in candidates],
            "model": self.model,
            "top_k": top_k,
            "return_documents": False,
            "truncation": True,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"MongoDB rerank request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MongoDB rerank request failed: {exc}") from exc

        data = json.loads(raw)
        reranked: list[RerankedSearchResult] = []
        for item in data.get("data", []):
            index = int(item["index"])
            if index >= len(candidates):
                continue
            reranked.append(
                RerankedSearchResult(
                    result=candidates[index],
                    relevance_score=float(item["relevance_score"]),
                )
            )
        return reranked


def _document_text(candidate: SearchResult) -> str:
    record = candidate.record
    return "\n".join(
        [
            f"Title: {record.title}",
            f"URL: {record.url or ''}",
            f"Source path: {record.source_path}",
            f"Section: {' > '.join(record.headings)}",
            "",
            record.text,
        ]
    ).strip()


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _reason(
    *,
    vector_score: float | None,
    rerank_score: float,
    rank: int,
    reranked: bool,
) -> str:
    vector_text = f"{vector_score:.6f}" if vector_score is not None else "n/a"
    if not reranked:
        return f"Selected by MongoDB vector search at rank {rank} with vector score {vector_text}."
    return (
        "Selected by MongoDB vector search, then reranked by MongoDB "
        f"at rank {rank} with relevance score {rerank_score:.6f} "
        f"(vector score {vector_text})."
    )
