from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from yolorag.usage.models import CostBreakdown, TokenUsage


ResponseMode = Literal["fast", "deep"]
Message = dict[str, str]


@dataclass(frozen=True)
class LLMRequest:
    messages: list[Message]
    model: str
    mode: ResponseMode = "fast"
    temperature: float = 0.2
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    provider: str
    model: str
    usage: TokenUsage
    cost: CostBreakdown
    latency_ms: int
    raw_response: Any
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class LLMProvider(Protocol):
    provider_name: str

    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...

