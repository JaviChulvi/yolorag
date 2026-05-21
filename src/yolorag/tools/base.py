from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCallRequest:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallResult:
    name: str
    output: Any
    cost_hint: str = "unknown"


class Tool(Protocol):
    name: str

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        ...

