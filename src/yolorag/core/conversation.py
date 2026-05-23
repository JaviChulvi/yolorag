from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol


ChatRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class ConversationMessageLog:
    conversation_id: str
    role: ChatRole
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    request_id: str | None = None
    message_index: int | None = None
    provider: str | None = None
    model: str | None = None
    retrieved_document_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationLogger(Protocol):
    provider_name: str

    def append_messages(self, messages: list[ConversationMessageLog]) -> None:
        """Persist completed chat messages without serving them back as context."""
