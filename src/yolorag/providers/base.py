from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from yolorag.usage.models import CostBreakdown, TokenUsage


ResponseMode = Literal["fast", "deep"]
Message = dict[str, Any]


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
    reasoning_content: str | None = None
    first_token_latency_ms: int = 0


@dataclass(frozen=True)
class LLMStreamEvent:
    content: str = ""
    usage: TokenUsage | None = None
    cost: CostBreakdown | None = None
    raw_chunk: Any = None


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a request.

    ``status`` mirrors the intended HTTP status so an API layer can surface a
    clean error (e.g. 400 for a bad model/blocked content, 502 for an upstream
    fault) rather than a generic 500.
    """

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


class LLMProvider(Protocol):
    provider_name: str

    async def complete(self, request: LLMRequest) -> LLMResponse:
        ...

    def stream_complete(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        ...
