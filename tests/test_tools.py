from __future__ import annotations

import unittest
from asyncio import run

from yolorag.core.orchestrator import FAST_DOCS_TOP_K
from yolorag.retrieval.base import Document, RetrievalResult, RetrievalTrace
from yolorag.tools.base import ToolCallRequest
from yolorag.tools.docs_search import DocsSearchTool


class DocsSearchToolTests(unittest.TestCase):
    def test_filters_results_below_min_relevance_score(self) -> None:
        retriever = StaticRetriever(scores=[0.72, 0.2])
        tool = DocsSearchTool(retriever, min_relevance_score=0.5)

        result = run(
            tool.call(
                ToolCallRequest(
                    name="docs_search",
                    arguments={"query": "export", "top_k": 2},
                )
            )
        )

        self.assertEqual(retriever.calls, [("export", 2)])
        self.assertEqual(
            [item["id"] for item in result.output["results"]],
            ["doc-1"],
        )
        self.assertEqual(result.metadata["retrieval"]["retrieval_ms"], 42)
        self.assertEqual(result.metadata["retrieval"]["query_embedding_ms"], 7)
        self.assertEqual(result.metadata["retrieval"]["vector_search_ms"], 11)
        self.assertEqual(result.metadata["retrieval"]["rerank_ms"], 13)
        self.assertEqual(result.metadata["retrieval"]["candidate_count"], 2)
        self.assertEqual(result.metadata["retrieval"]["returned_count"], 1)
        self.assertTrue(result.metadata["retrieval"]["reranked"])

    def test_fast_docs_caps_top_k_without_truncating_content(self) -> None:
        retriever = StaticRetriever(scores=[0.9, 0.8, 0.7])
        tool = DocsSearchTool(
            retriever,
            default_top_k=FAST_DOCS_TOP_K,
            max_top_k=FAST_DOCS_TOP_K,
        )

        result = run(
            tool.call(
                ToolCallRequest(
                    name="docs_search",
                    arguments={"query": "export", "top_k": 8},
                )
            )
        )

        self.assertEqual(retriever.calls, [("export", FAST_DOCS_TOP_K)])
        self.assertEqual(result.output["results"][0]["content"], "Content 1 " * 200)


class StaticRetriever:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        self.calls.append((query, top_k))
        return [
            RetrievalResult(
                document=Document(
                    id=f"doc-{index}",
                    title=f"Doc {index}",
                    content=(f"Content {index} " * 200),
                    metadata={"url": f"https://docs.example/{index}"},
                ),
                score=score,
                reason=f"score={score}",
                trace=RetrievalTrace(
                    provider="test",
                    total_ms=42,
                    query_embedding_ms=7,
                    vector_search_ms=11,
                    rerank_ms=13,
                    candidate_count=len(self.scores),
                    returned_count=top_k,
                    reranked=True,
                ),
            )
            for index, score in enumerate(self.scores, start=1)
        ]


if __name__ == "__main__":
    unittest.main()
