from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalTrace:
    provider: str
    total_ms: int
    query_embedding_ms: int = 0
    vector_search_ms: int = 0
    rerank_ms: int = 0
    candidate_count: int = 0
    returned_count: int = 0
    reranked: bool = False


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float
    reason: str
    trace: RetrievalTrace | None = None


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        ...
