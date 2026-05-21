from __future__ import annotations

from dataclasses import dataclass, field

from yolorag.providers.base import Message


@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: str
    retrieved_document_ids: list[str] = field(default_factory=list)


@dataclass
class ConversationState:
    conversation_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    retrieved_document_ids: set[str] = field(default_factory=set)

    def recent_messages(self, limit: int = 6) -> list[Message]:
        messages: list[Message] = []
        for turn in self.turns[-limit:]:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})
        return messages

    def add_turn(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)
        self.retrieved_document_ids.update(turn.retrieved_document_ids)


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}

    def get(self, conversation_id: str) -> ConversationState:
        if conversation_id not in self._states:
            self._states[conversation_id] = ConversationState(conversation_id=conversation_id)
        return self._states[conversation_id]
