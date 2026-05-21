from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from yolorag.retrieval.base import RetrievalResult


@dataclass(frozen=True)
class ReviewResult:
    passed: bool
    confidence: float
    notes: list[str]


class Reviewer(Protocol):
    async def review(
        self,
        question: str,
        answer: str,
        context: list[RetrievalResult],
    ) -> ReviewResult:
        ...

