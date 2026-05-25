from __future__ import annotations

import unittest
from asyncio import run

from yolorag.retrieval.base import Document, RetrievalResult
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
                    content=f"Content {index}",
                    metadata={"url": f"https://docs.example/{index}"},
                ),
                score=score,
                reason=f"score={score}",
            )
            for index, score in enumerate(self.scores, start=1)
        ]


if __name__ == "__main__":
    unittest.main()
