from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

from yolorag.knowledge.models import ChunkRecord, IngestResult, SearchResult


DEFAULT_MONGODB_DB = "yolorag"
DEFAULT_CHUNKS_COLLECTION = "docs_chunks"
DEFAULT_VECTOR_INDEX = "autoembed_index"
DEFAULT_EMBEDDING_MODEL = "voyage-4"


@dataclass(frozen=True)
class MongoKnowledgeStoreConfig:
    uri: str
    database: str = DEFAULT_MONGODB_DB
    collection: str = DEFAULT_CHUNKS_COLLECTION
    vector_index: str = DEFAULT_VECTOR_INDEX
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    text_path: str = "text"
    server_selection_timeout_ms: int = 5000

    @classmethod
    def from_env(cls) -> MongoKnowledgeStoreConfig:
        uri = os.getenv("YOLORAG_MONGODB_URI")
        if not uri:
            raise RuntimeError("Missing YOLORAG_MONGODB_URI in .env")
        return cls(
            uri=uri,
            database=os.getenv("YOLORAG_MONGODB_DB", DEFAULT_MONGODB_DB),
            collection=os.getenv(
                "YOLORAG_MONGODB_CHUNKS_COLLECTION",
                DEFAULT_CHUNKS_COLLECTION,
            ),
            vector_index=os.getenv("YOLORAG_MONGODB_VECTOR_INDEX", DEFAULT_VECTOR_INDEX),
            embedding_model=os.getenv("YOLORAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        )


class MongoKnowledgeStore:
    provider_name = "mongodb"

    def __init__(
        self,
        config: MongoKnowledgeStoreConfig,
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

    def ping(self) -> None:
        self.client.admin.command("ping")

    def search_index_names(self) -> list[str]:
        return [index["name"] for index in self.collection.list_search_indexes()]

    def ingest_chunks(
        self,
        records: Sequence[ChunkRecord],
        *,
        batch_size: int = 100,
    ) -> IngestResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        attempted = len(records)
        inserted = 0
        matched = 0
        modified = 0

        for start in range(0, attempted, batch_size):
            batch = records[start : start + batch_size]
            operations = [_upsert_operation(record) for record in batch]
            if not operations:
                continue
            result = self.collection.bulk_write(operations, ordered=False)
            inserted += result.upserted_count
            matched += result.matched_count
            modified += result.modified_count

        return IngestResult(
            attempted=attempted,
            inserted=inserted,
            matched=matched,
            modified=modified,
            provider=self.provider_name,
        )

    def vector_search(
        self,
        query: str,
        *,
        limit: int = 8,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        num_candidates = max(limit * 20, 100)
        vector_stage: dict[str, Any] = {
            "index": self.config.vector_index,
            "path": self.config.text_path,
            "query": query,
            "model": self.config.embedding_model,
            "numCandidates": num_candidates,
            "limit": limit,
        }
        if filters:
            vector_stage["filter"] = dict(filters)

        pipeline = [
            {"$vectorSearch": vector_stage},
            {
                "$project": {
                    "_id": 1,
                    "record_id": 1,
                    "chunk_id": 1,
                    "doc_id": 1,
                    "chunk_index": 1,
                    "source": 1,
                    "source_path": 1,
                    "url": 1,
                    "title": 1,
                    "headings": 1,
                    "kind": 1,
                    "text": 1,
                    "content": 1,
                    "char_count": 1,
                    "estimated_tokens": 1,
                    "content_hash": 1,
                    "reference_symbols": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        results: list[SearchResult] = []
        for document in self.collection.aggregate(pipeline):
            score = document.pop("score", None)
            results.append(
                SearchResult(
                    record=ChunkRecord.from_mapping(document),
                    score=float(score) if score is not None else None,
                    provider=self.provider_name,
                )
            )
        return results


def _upsert_operation(record: ChunkRecord) -> UpdateOne:
    now = datetime.now(UTC)
    document = record.to_mapping()
    document["_id"] = record.record_id
    document["updated_at"] = now
    return UpdateOne(
        {"_id": record.record_id},
        {
            "$set": document,
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
