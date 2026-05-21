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
class RetrievalResult:
    document: Document
    score: float
    reason: str


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        ...

