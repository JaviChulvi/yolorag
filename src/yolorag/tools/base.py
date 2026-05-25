from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCallRequest:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


@dataclass(frozen=True)
class ToolCallResult:
    name: str
    output: Any
    cost_hint: str = "unknown"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        ...
