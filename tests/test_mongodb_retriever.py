from __future__ import annotations

import asyncio
import unittest

from yolorag.knowledge.models import ChunkRecord, SearchResult
from yolorag.retrieval.mongodb import MongoVectorRetriever, RerankedSearchResult


class FakeKnowledgeStore:
    provider_name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict | None]] = []

    def vector_search(
        self,
        query: str,
        *,
        limit: int = 8,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query, limit, filters))
        return [
            _search_result(
                "chunk-1",
                title="Train",
                score=0.77,
                query_embedding_ms=123,
            ),
            _search_result(
                "chunk-2",
                title="Export",
                score=0.70,
                query_embedding_ms=456,
            ),
        ]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[RerankedSearchResult]:
        self.calls.append((query, [candidate.record.chunk_id for candidate in candidates], top_k))
        return [
            RerankedSearchResult(result=candidates[1], relevance_score=0.91),
            RerankedSearchResult(result=candidates[0], relevance_score=0.82),
        ][:top_k]


class MongoVectorRetrieverTests(unittest.TestCase):
    def test_retriever_fetches_candidates_and_uses_reranker_order(self) -> None:
        store = FakeKnowledgeStore()
        reranker = FakeReranker()
        retriever = MongoVectorRetriever(store, filters={"doc_id": "train"}, reranker=reranker)

        results = asyncio.run(retriever.retrieve("train yolo", top_k=1))

        self.assertEqual(store.calls, [("train yolo", 40, {"doc_id": "train"})])
        self.assertEqual(reranker.calls, [("train yolo", ["chunk-1", "chunk-2"], 1)])
        self.assertEqual(results[0].document.id, "chunk-2")
        self.assertEqual(results[0].document.title, "Export")
        self.assertEqual(results[0].document.metadata["source_path"], "en/modes/train.md")
        self.assertEqual(results[0].score, 0.91)
        self.assertEqual(results[0].document.metadata["rerank_score"], "0.91")
        self.assertIn("reranked by MongoDB", results[0].reason)
        self.assertIsNotNone(results[0].trace)
        assert results[0].trace is not None
        self.assertEqual(results[0].trace.query_embedding_ms, 123)
        self.assertEqual(results[0].trace.candidate_count, 2)
        self.assertEqual(results[0].trace.returned_count, 1)
        self.assertTrue(results[0].trace.reranked)

    def test_retriever_uses_configured_candidate_limit_for_reranking(self) -> None:
        store = FakeKnowledgeStore()
        reranker = FakeReranker()
        retriever = MongoVectorRetriever(store, reranker=reranker, candidate_limit=16)

        asyncio.run(retriever.retrieve("train yolo", top_k=5))

        self.assertEqual(store.calls, [("train yolo", 16, None)])

    def test_retriever_uses_top_k_as_candidate_limit_without_reranker(self) -> None:
        store = FakeKnowledgeStore()
        retriever = MongoVectorRetriever(store)

        results = asyncio.run(retriever.retrieve("train yolo", top_k=5))

        self.assertEqual(store.calls, [("train yolo", 5, None)])
        self.assertIsNotNone(results[0].trace)
        assert results[0].trace is not None
        self.assertFalse(results[0].trace.reranked)


def _search_result(
    chunk_id: str,
    title: str,
    score: float,
    query_embedding_ms: int = 0,
) -> SearchResult:
    return SearchResult(
        record=ChunkRecord(
            record_id=chunk_id,
            chunk_id=chunk_id,
            doc_id="train",
            chunk_index=0,
            source="test",
            source_path="en/modes/train.md",
            url="https://docs.ultralytics.com/modes/train/",
            title=title,
            headings=[title],
            kind="article",
            text=f"{title} YOLO models.",
            content=f"{title} YOLO models.",
            char_count=37,
            estimated_tokens=8,
            content_hash="abc",
            reference_symbols=[],
        ),
        score=score,
        provider="fake",
        query_embedding_ms=query_embedding_ms,
    )


if __name__ == "__main__":
    unittest.main()
