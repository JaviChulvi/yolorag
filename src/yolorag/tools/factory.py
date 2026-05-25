from __future__ import annotations

from yolorag.retrieval.base import Retriever
from yolorag.tools.docs_search import DocsSearchTool
from yolorag.tools.mcp import MCPToolProvider
from yolorag.tools.router import DynamicToolProvider, ToolRouter


def build_tool_router(
    retriever: Retriever | None = None,
    *,
    min_relevance_score: float | None = None,
) -> ToolRouter:
    dynamic_providers: list[DynamicToolProvider] = []
    mcp_provider = MCPToolProvider.from_env()
    if mcp_provider is not None:
        dynamic_providers.append(mcp_provider)

    return ToolRouter(
        tools=[
            DocsSearchTool(
                retriever=retriever,
                min_relevance_score=min_relevance_score,
            )
        ],
        dynamic_providers=dynamic_providers,
    )
