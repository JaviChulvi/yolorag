from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from yolorag.knowledge.image_embeddings import DEFAULT_NOMIC_EMBEDDING_DIMENSIONS
from yolorag.knowledge.image_models import ImageEmbeddingRecord, ImageSearchResult
from yolorag.knowledge.models import IngestResult


DEFAULT_MONGODB_DB = "yolorag"
DEFAULT_IMAGE_EMBEDDINGS_COLLECTION = "image_embeddings"
DEFAULT_VECTOR_INDEX = "image_embeddings_vector_index"


def _doc_id(dataset_id: str, img_id: str) -> str:
    """Deterministic _id mirroring the Postgres composite PK (dataset_id, img_id)."""
    return f"{dataset_id}/{img_id}"


@dataclass(frozen=True)
class MongoImageEmbeddingStoreConfig:
    uri: str
    database: str = DEFAULT_MONGODB_DB
    collection: str = DEFAULT_IMAGE_EMBEDDINGS_COLLECTION
    vector_index: str = DEFAULT_VECTOR_INDEX
    embedding_dimensions: int = DEFAULT_NOMIC_EMBEDDING_DIMENSIONS
    server_selection_timeout_ms: int = 5000

    @classmethod
    def from_env(cls) -> MongoImageEmbeddingStoreConfig:
        uri = os.getenv("YOLORAG_MONGODB_URI")
        if not uri:
            raise RuntimeError("Missing YOLORAG_MONGODB_URI for MongoDB image embeddings.")
        return cls(
            uri=uri,
            database=os.getenv("YOLORAG_MONGODB_DB", DEFAULT_MONGODB_DB),
            collection=os.getenv(
                "YOLORAG_MONGODB_IMAGE_EMBEDDINGS_COLLECTION",
                DEFAULT_IMAGE_EMBEDDINGS_COLLECTION,
            ),
            vector_index=os.getenv(
                "YOLORAG_MONGODB_IMAGE_VECTOR_INDEX",
                DEFAULT_VECTOR_INDEX,
            ),
            embedding_dimensions=_env_int(
                "YOLORAG_IMAGE_EMBEDDING_DIMENSIONS",
                DEFAULT_NOMIC_EMBEDDING_DIMENSIONS,
            ),
        )


class MongoImageEmbeddingStore:
    """Persist and query Nomic image embeddings in MongoDB (Atlas / Atlas Local).

    Mirrors the Postgres store: one document per (dataset_id, img_id) with the
    128-dim embedding, a unique compound key, and a cosine $vectorSearch index.
    """

    provider_name = "mongodb-image"

    def __init__(
        self,
        config: MongoImageEmbeddingStoreConfig,
        client: MongoClient[Any] | None = None,
    ) -> None:
        if config.embedding_dimensions != DEFAULT_NOMIC_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "The MongoDB image-embedding store currently stores "
                f"{DEFAULT_NOMIC_EMBEDDING_DIMENSIONS}-dimension vectors."
            )
        self.config = config
        self.client: MongoClient[Any] = client or MongoClient(
            config.uri,
            serverSelectionTimeoutMS=config.server_selection_timeout_ms,
        )
        self.collection: Collection[dict[str, Any]] = self.client[config.database][
            config.collection
        ]
        self.last_query_embedding_ms = 0

    def ping(self) -> None:
        self.client.admin.command("ping")

    def search_index_names(self) -> list[str]:
        return [index["name"] for index in self.collection.list_search_indexes()]

    def ensure_schema(self) -> None:
        """Create the collection, unique key, and cosine vector index (idempotent)."""
        database = self.client[self.config.database]
        if self.config.collection not in database.list_collection_names():
            database.create_collection(self.config.collection)

        self.collection.create_index(
            [("dataset_id", 1), ("img_id", 1)],
            unique=True,
            name="dataset_id_img_id_unique",
        )

        # Search indexes are Atlas / Atlas Local only; don't hard-fail on community mongod.
        try:
            existing = set(self.search_index_names())
        except Exception:
            return
        if self.config.vector_index not in existing:
            self.collection.create_search_index(
                SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": self.config.embedding_dimensions,
                                "similarity": "cosine",
                            },
                            {"type": "filter", "path": "dataset_id"},
                        ]
                    },
                    name=self.config.vector_index,
                    type="vectorSearch",
                )
            )

    def upsert_embeddings(
        self,
        records: Sequence[ImageEmbeddingRecord],
        *,
        batch_size: int = 500,
    ) -> IngestResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.ensure_schema()
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

    def search(
        self,
        dataset_id: str,
        query_embedding: Sequence[float],
        *,
        limit: int = 8,
        search_ef: int | None = None,
    ) -> list[ImageSearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        num_candidates = max(int(search_ef), limit) if search_ef else max(limit * 20, 100)
        started = time.perf_counter()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.config.vector_index,
                    "path": "embedding",
                    "queryVector": [float(value) for value in query_embedding],
                    "numCandidates": num_candidates,
                    "limit": limit,
                    "filter": {"dataset_id": dataset_id},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "dataset_id": 1,
                    "img_id": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        rows = list(self.collection.aggregate(pipeline))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.last_query_embedding_ms = elapsed_ms

        return [
            ImageSearchResult(
                dataset_id=row["dataset_id"],
                img_id=row["img_id"],
                score=float(row["score"]) if row.get("score") is not None else None,
                provider=self.provider_name,
                query_embedding_ms=elapsed_ms,
            )
            for row in rows
        ]


def _upsert_operation(record: ImageEmbeddingRecord) -> UpdateOne:
    return UpdateOne(
        {"_id": _doc_id(record.dataset_id, record.img_id)},
        {
            "$set": {
                "dataset_id": record.dataset_id,
                "img_id": record.img_id,
                "embedding": [float(value) for value in record.embedding],
            }
        },
        upsert=True,
    )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return parsed
