from __future__ import annotations

import asyncio
from typing import Protocol

from yolorag.tools.base import Tool, ToolCallRequest, ToolCallResult


class DynamicToolProvider(Protocol):
    async def tools(self) -> list[Tool]:
        ...


class ToolRouter:
    def __init__(
        self,
        tools: list[Tool] | None = None,
        dynamic_providers: list[DynamicToolProvider] | None = None,
    ) -> None:
        self.tools = {tool.name: tool for tool in tools or []}
        self.dynamic_providers = list(dynamic_providers or [])

    async def available_tools(self) -> dict[str, Tool]:
        tools = dict(self.tools)
        if not self.dynamic_providers:
            return tools

        discovered = await asyncio.gather(
            *(provider.tools() for provider in self.dynamic_providers),
            return_exceptions=True,
        )
        for result in discovered:
            if isinstance(result, Exception):
                continue
            for tool in result:
                tools[tool.name] = tool
        return tools

    async def openai_schemas(self) -> list[dict]:
        return [
            _openai_tool_schema(tool)
            for tool in (await self.available_tools()).values()
        ]

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        tool = (await self.available_tools()).get(request.name)
        if tool is None:
            raise ValueError(f"Unknown tool {request.name!r}")
        return await tool.call(request)


def _openai_tool_schema(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
