from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from yolorag.core.conversation import ConversationMessageLog
from yolorag.knowledge.stores.mongodb import DEFAULT_MONGODB_DB
from yolorag.knowledge.stores.postgresql import DEFAULT_POSTGRES_DSN


DEFAULT_MONGODB_MESSAGES_COLLECTION = "chat_messages"
DEFAULT_POSTGRES_MESSAGES_TABLE = "chat_messages"


@dataclass(frozen=True)
class MongoConversationLoggerConfig:
    uri: str
    database: str = DEFAULT_MONGODB_DB
    collection: str = DEFAULT_MONGODB_MESSAGES_COLLECTION
    server_selection_timeout_ms: int = 5000

    @classmethod
    def from_env(cls) -> MongoConversationLoggerConfig:
        uri = os.getenv("YOLORAG_MONGODB_URI")
        if not uri:
            raise RuntimeError("Missing YOLORAG_MONGODB_URI in .env")
        return cls(
            uri=uri,
            database=os.getenv("YOLORAG_MONGODB_DB", DEFAULT_MONGODB_DB),
            collection=os.getenv(
                "YOLORAG_MONGODB_MESSAGES_COLLECTION",
                DEFAULT_MONGODB_MESSAGES_COLLECTION,
            ),
        )


class MongoConversationLogger:
    provider_name = "mongodb"

    def __init__(
        self,
        config: MongoConversationLoggerConfig,
        client: MongoClient[Any] | None = None,
    ) -> None:
        self.config = config
        self.client: MongoClient[Any] = client or MongoClient(
            config.uri,
            serverSelectionTimeoutMS=config.server_selection_timeout_ms,
        )
        self.collection: Collection[dict[str, Any]] = self.client[config.database][
            config.collection
        ]

    def append_messages(self, messages: list[ConversationMessageLog]) -> None:
        if not messages:
            return
        self.collection.insert_many([_message_to_mapping(message) for message in messages])


class TranscriptBase(DeclarativeBase):
    pass


class PostgresChatMessage(TranscriptBase):
    __tablename__ = DEFAULT_POSTGRES_MESSAGES_TABLE
    __table_args__ = (
        Index("chat_messages_conversation_idx", "conversation_id", "created_at"),
        Index("chat_messages_request_idx", "request_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    request_id: Mapped[str | None] = mapped_column(String, nullable=True)
    message_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider: Mapped[str | None] = mapped_column(String, nullable=True)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    retrieved_document_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    message_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    @classmethod
    def from_log(cls, message: ConversationMessageLog) -> PostgresChatMessage:
        return cls(
            conversation_id=message.conversation_id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            request_id=message.request_id,
            message_index=message.message_index,
            provider=message.provider,
            model=message.model,
            retrieved_document_ids=list(message.retrieved_document_ids),
            message_metadata=dict(message.metadata),
        )


@dataclass(frozen=True)
class PostgresConversationLoggerConfig:
    dsn: str = DEFAULT_POSTGRES_DSN

    @classmethod
    def from_env(cls) -> PostgresConversationLoggerConfig:
        return cls(dsn=os.getenv("YOLORAG_POSTGRES_DSN", DEFAULT_POSTGRES_DSN))


class PostgresConversationLogger:
    provider_name = "postgresql"

    def __init__(
        self,
        config: PostgresConversationLoggerConfig,
        engine: Engine | None = None,
    ) -> None:
        self.config = config
        self.engine = engine or create_engine(config.dsn)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._schema_ready = False

    def append_messages(self, messages: list[ConversationMessageLog]) -> None:
        if not messages:
            return
        self.ensure_schema()
        with self.Session() as session:
            session.add_all([PostgresChatMessage.from_log(message) for message in messages])
            session.commit()

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        TranscriptBase.metadata.create_all(self.engine)
        self._schema_ready = True


def _message_to_mapping(message: ConversationMessageLog) -> dict[str, Any]:
    return {
        "conversation_id": message.conversation_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at,
        "request_id": message.request_id,
        "message_index": message.message_index,
        "provider": message.provider,
        "model": message.model,
        "retrieved_document_ids": list(message.retrieved_document_ids),
        "metadata": dict(message.metadata),
        "inserted_at": datetime.now(UTC),
    }
