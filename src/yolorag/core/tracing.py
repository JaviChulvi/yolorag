from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from yolorag.providers.base import ResponseMode


@dataclass(frozen=True)
class OrchestrationTrace:
    provider: str
    model: str
    mode: ResponseMode
    reasoning_budget: str
    retrieval_used: bool
    retrieved_document_ids: list[str]
    review_used: bool
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: Decimal
    pricing_source: str
    latency_ms: int
    route_reason: str
    review_confidence: float | None = None
    review_notes: list[str] = field(default_factory=list)

