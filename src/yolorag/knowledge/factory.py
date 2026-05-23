from __future__ import annotations

from yolorag.config.settings import getenv
from yolorag.knowledge.stores.base import KnowledgeStore
from yolorag.knowledge.stores.mongodb import MongoKnowledgeStore, MongoKnowledgeStoreConfig
from yolorag.knowledge.stores.postgresql import (
    PostgresKnowledgeStore,
    PostgresKnowledgeStoreConfig,
)


DEFAULT_KNOWLEDGE_PROVIDER = "mongodb"


def selected_knowledge_provider(provider_name: str | None = None) -> str:
    return provider_name or getenv("YOLORAG_KNOWLEDGE_PROVIDER", DEFAULT_KNOWLEDGE_PROVIDER)


def build_knowledge_store(provider_name: str | None = None) -> KnowledgeStore:
    provider = selected_knowledge_provider(provider_name)
    if provider == "mongodb":
        return MongoKnowledgeStore(MongoKnowledgeStoreConfig.from_env())
    if provider == "postgresql":
        return PostgresKnowledgeStore(PostgresKnowledgeStoreConfig.from_env())
    raise ValueError(f"Unsupported knowledge provider {provider!r}.")
