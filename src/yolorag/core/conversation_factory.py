from __future__ import annotations

from yolorag.config.settings import getenv
from yolorag.core.conversation import ConversationLogger
from yolorag.core.conversation_stores import (
    MongoConversationLogger,
    MongoConversationLoggerConfig,
    PostgresConversationLogger,
    PostgresConversationLoggerConfig,
)
from yolorag.knowledge.factory import selected_knowledge_provider


def selected_conversation_provider(provider_name: str | None = None) -> str:
    return (
        provider_name
        or getenv("YOLORAG_CONVERSATION_PROVIDER")
        or selected_knowledge_provider()
    )


def build_conversation_logger(provider_name: str | None = None) -> ConversationLogger:
    provider = selected_conversation_provider(provider_name)
    if provider == "mongodb":
        return MongoConversationLogger(MongoConversationLoggerConfig.from_env())
    if provider == "postgresql":
        return PostgresConversationLogger(PostgresConversationLoggerConfig.from_env())
    raise ValueError(f"Unsupported conversation provider {provider!r}.")
