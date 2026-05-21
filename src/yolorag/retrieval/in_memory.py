from __future__ import annotations

import re

from yolorag.retrieval.base import Document, RetrievalResult


class InMemoryRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_terms = self._terms(query)
        results: list[RetrievalResult] = []

        for document in self.documents:
            doc_terms = self._terms(f"{document.title} {document.content}")
            overlap = query_terms & doc_terms
            if not overlap:
                continue

            score = len(overlap) / max(len(query_terms), 1)
            results.append(
                RetrievalResult(
                    document=document,
                    score=score,
                    reason=f"Matched query terms: {', '.join(sorted(overlap))}",
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def _terms(self, text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9_]+", text.lower())
            if len(term) > 2
        }

