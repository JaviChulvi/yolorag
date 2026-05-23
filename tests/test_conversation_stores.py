from __future__ import annotations

import unittest

from sqlalchemy import create_engine, select

from yolorag.core.conversation import ConversationMessageLog
from yolorag.core.conversation_factory import selected_conversation_provider
from yolorag.core.conversation_stores import (
    MongoConversationLogger,
    MongoConversationLoggerConfig,
    PostgresChatMessage,
    PostgresConversationLogger,
    PostgresConversationLoggerConfig,
)


class ConversationLoggerTests(unittest.TestCase):
    def test_mongo_logger_inserts_individual_messages(self) -> None:
        client = FakeMongoClient()
        logger = MongoConversationLogger(
            MongoConversationLoggerConfig(uri="mongodb://example.test"),
            client=client,
        )

        logger.append_messages(_messages())

        documents = client.databases["yolorag"].collections["chat_messages"].documents
        self.assertEqual([document["role"] for document in documents], ["user", "assistant"])
        self.assertEqual(documents[1]["provider"], "openai")
        self.assertEqual(documents[1]["retrieved_document_ids"], ["doc-1"])

    def test_postgres_logger_inserts_individual_messages_with_sqlalchemy_model(self) -> None:
        engine = create_engine("sqlite:///:memory:")
        logger = PostgresConversationLogger(
            PostgresConversationLoggerConfig(dsn="sqlite:///:memory:"),
            engine=engine,
        )

        logger.append_messages(_messages())

        with logger.Session() as session:
            rows = session.scalars(select(PostgresChatMessage).order_by(PostgresChatMessage.id)).all()
        self.assertEqual([row.role for row in rows], ["user", "assistant"])
        self.assertEqual(rows[1].provider, "openai")
        self.assertEqual(rows[1].retrieved_document_ids, ["doc-1"])

    def test_conversation_provider_defaults_to_knowledge_provider_without_memory_option(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(
            os.environ,
            {
                "YOLORAG_KNOWLEDGE_PROVIDER": "postgresql",
            },
            clear=True,
        ):
            self.assertEqual(selected_conversation_provider(), "postgresql")


def _messages() -> list[ConversationMessageLog]:
    return [
        ConversationMessageLog(
            conversation_id="thread-1",
            role="user",
            content="first",
            request_id="request-1",
            message_index=0,
        ),
        ConversationMessageLog(
            conversation_id="thread-1",
            role="assistant",
            content="answer",
            request_id="request-1",
            message_index=1,
            provider="openai",
            model="gpt-test",
            retrieved_document_ids=["doc-1"],
        ),
    ]


class FakeMongoCollection:
    def __init__(self) -> None:
        self.documents: list[dict] = []

    def insert_many(self, documents: list[dict]) -> None:
        self.documents.extend(documents)


class FakeMongoDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, FakeMongoCollection] = {}

    def __getitem__(self, name: str) -> FakeMongoCollection:
        if name not in self.collections:
            self.collections[name] = FakeMongoCollection()
        return self.collections[name]


class FakeMongoClient:
    def __init__(self) -> None:
        self.databases: dict[str, FakeMongoDatabase] = {}

    def __getitem__(self, name: str) -> FakeMongoDatabase:
        if name not in self.databases:
            self.databases[name] = FakeMongoDatabase()
        return self.databases[name]


if __name__ == "__main__":
    unittest.main()
