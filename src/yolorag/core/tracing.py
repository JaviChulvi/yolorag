from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from yolorag.providers.base import ResponseMode


@dataclass(frozen=True)
class OrchestrationTrace:
    provider: str
    model: str
    mode: ResponseMode
    retrieval_used: bool
    retrieved_document_ids: list[str]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: Decimal
    pricing_source: str
    latency_ms: int
    route_reason: str
    total_ms: int = 0
    retrieval_ms: int = 0
    query_embedding_ms: int = 0
    vector_search_ms: int = 0
    rerank_ms: int = 0
    llm_ms: int = 0
    ttft_ms: int = 0
    llm_ttft_ms: int = 0
    orchestration_overhead_ms: int = 0
    retrieval_candidate_count: int = 0
    retrieval_returned_count: int = 0
    retrieval_reranked: bool = False
    retrieval_error: str | None = None
