from __future__ import annotations

import logging
from typing import Any

from yolorag.retrieval.base import RetrievalResult, Retriever
from yolorag.tools.base import ToolCallRequest, ToolCallResult


logger = logging.getLogger(__name__)


class DocsSearchTool:
    name = "docs_search"
    description = (
        "Search the indexed Ultralytics documentation for precise product, API, "
        "installation, training, deployment, and troubleshooting context."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The documentation search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return. Defaults to 4.",
                "minimum": 1,
                "maximum": 8,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        retriever: Retriever | None,
        *,
        default_top_k: int = 4,
        max_top_k: int = 8,
        max_content_chars: int = 1400,
        min_relevance_score: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k
        self.max_content_chars = max_content_chars
        self.min_relevance_score = min_relevance_score

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        query = str(request.arguments.get("query") or "").strip()
        top_k = _bounded_int(
            request.arguments.get("top_k"),
            default=self.default_top_k,
            minimum=1,
            maximum=self.max_top_k,
        )
        if not query:
            return ToolCallResult(
                name=self.name,
                output={"results": [], "error": "Missing required query."},
                error="Missing required query.",
            )
        if self.retriever is None:
            return ToolCallResult(
                name=self.name,
                output={"results": [], "error": "Documentation search is not configured."},
                error="Documentation search is not configured.",
            )

        try:
            results = await self.retriever.retrieve(query, top_k=top_k)
        except Exception as exc:
            logger.warning("docs_search failed.", exc_info=True)
            return ToolCallResult(
                name=self.name,
                output={"results": [], "error": f"{type(exc).__name__}: {exc}"},
                error=f"{type(exc).__name__}: {exc}",
            )

        filtered_results = [
            result
            for result in results
            if _passes_relevance_threshold(result, self.min_relevance_score)
        ]

        return ToolCallResult(
            name=self.name,
            output={
                "query": query,
                "results": [
                    _result_payload(result, max_content_chars=self.max_content_chars)
                    for result in filtered_results
                ],
            },
            cost_hint="retrieval",
        )


def _result_payload(result: RetrievalResult, *, max_content_chars: int) -> dict[str, Any]:
    metadata = result.document.metadata
    return {
        "id": result.document.id,
        "title": result.document.title,
        "url": metadata.get("url", ""),
        "source_path": metadata.get("source_path", ""),
        "score": result.score,
        "reason": result.reason,
        "content": _truncate(result.document.content, max_content_chars),
    }


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _passes_relevance_threshold(
    result: RetrievalResult,
    min_relevance_score: float | None,
) -> bool:
    if min_relevance_score is None:
        return True
    return result.score >= min_relevance_score


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."
