from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class PageContext(BaseModel):
    url: str | None = None
    title: str | None = None
    description: str | None = None
    path: str | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    session_id: str | None = None
    context: PageContext | None = None
    analytics: bool = True
    include_metrics: bool = False
    edit_index: int | None = None
    instructions: str | None = None
    tools: list[str] = Field(default_factory=list)
