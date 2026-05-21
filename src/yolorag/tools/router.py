from __future__ import annotations

from yolorag.tools.base import Tool, ToolCallRequest, ToolCallResult


class ToolRouter:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self.tools = {tool.name: tool for tool in tools or []}

    def should_use_tool(self, message: str) -> bool:
        tool_keywords = {"repo", "file", "github", "trace", "log", "run", "debug"}
        lowered = message.lower()
        return any(keyword in lowered for keyword in tool_keywords)

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        tool = self.tools.get(request.name)
        if tool is None:
            raise ValueError(f"Unknown tool {request.name!r}")
        return await tool.call(request)

