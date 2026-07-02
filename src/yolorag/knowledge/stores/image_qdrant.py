from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from yolorag.knowledge.image_embeddings import DEFAULT_NOMIC_EMBEDDING_DIMENSIONS
from yolorag.knowledge.image_models import ImageEmbeddingRecord, ImageSearchResult
from yolorag.knowledge.models import IngestResult


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_IMAGE_EMBEDDINGS_COLLECTION = "image_embeddings"

# Qdrant point IDs must be uint or UUID (not arbitrary strings), so we derive a
# deterministic UUID from the composite key -- same (dataset_id, img_id) always maps
# to the same point, keeping upserts idempotent. The real ids stay in the payload.
_POINT_NAMESPACE = uuid.NAMESPACE_URL


def _point_id(dataset_id: str, img_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"yolorag/image_embeddings/{dataset_id}/{img_id}"))


@dataclass(frozen=True)
class QdrantImageEmbeddingStoreConfig:
    url: str = DEFAULT_QDRANT_URL
    collection: str = DEFAULT_IMAGE_EMBEDDINGS_COLLECTION
    embedding_dimensions: int = DEFAULT_NOMIC_EMBEDDING_DIMENSIONS

    @classmethod
    def from_env(cls) -> QdrantImageEmbeddingStoreConfig:
        return cls(
            url=os.getenv("YOLORAG_QDRANT_URL", DEFAULT_QDRANT_URL),
            collection=os.getenv(
                "YOLORAG_QDRANT_IMAGE_EMBEDDINGS_COLLECTION",
                DEFAULT_IMAGE_EMBEDDINGS_COLLECTION,
            ),
            embedding_dimensions=_env_int(
                "YOLORAG_IMAGE_EMBEDDING_DIMENSIONS",
                DEFAULT_NOMIC_EMBEDDING_DIMENSIONS,
            ),
        )


class QdrantImageEmbeddingStore:
    """Persist and query Nomic image embeddings in Qdrant.

    Mirrors the Postgres/Mongo stores: one point per (dataset_id, img_id) with the
    128-dim embedding, cosine distance, and a dataset_id payload index for
    dataset-scoped filtered search.
    """

    provider_name = "qdrant-image"

    def __init__(
        self,
        config: QdrantImageEmbeddingStoreConfig,
        client: QdrantClient | None = None,
    ) -> None:
        if config.embedding_dimensions != DEFAULT_NOMIC_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "The Qdrant image-embedding store currently stores "
                f"{DEFAULT_NOMIC_EMBEDDING_DIMENSIONS}-dimension vectors."
            )
        self.config = config
        self.client = client or QdrantClient(url=config.url)
        self.last_query_embedding_ms = 0

    def ping(self) -> None:
        self.client.get_collections()

    def ensure_schema(self) -> None:
        """Create the collection (128-d cosine) + dataset_id payload index (idempotent)."""
        if not self.client.collection_exists(self.config.collection):
            self.client.create_collection(
                self.config.collection,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
        try:
            self.client.create_payload_index(
                self.config.collection,
                field_name="dataset_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # index already exists

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

        for start in range(0, attempted, batch_size):
            batch = list(records[start : start + batch_size])
            if not batch:
                continue
            ids = [_point_id(record.dataset_id, record.img_id) for record in batch]
            existing = {
                point.id
                for point in self.client.retrieve(
                    self.config.collection,
                    ids=ids,
                    with_payload=False,
                    with_vectors=False,
                )
            }
            points = [
                PointStruct(
                    id=point_id,
                    vector=[float(value) for value in record.embedding],
                    payload={"dataset_id": record.dataset_id, "img_id": record.img_id},
                )
                for point_id, record in zip(ids, batch)
            ]
            self.client.upsert(self.config.collection, points=points, wait=True)
            matched += sum(1 for point_id in ids if point_id in existing)
            inserted += sum(1 for point_id in ids if point_id not in existing)

        return IngestResult(
            attempted=attempted,
            inserted=inserted,
            matched=matched,
            modified=matched,
            provider=self.provider_name,
        )

    def search(
        self,
        dataset_id: str,
        query_embedding: Sequence[float],
        *,
        limit: int = 8,
    ) -> list[ImageSearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        started = time.perf_counter()
        response = self.client.query_points(
            self.config.collection,
            query=[float(value) for value in query_embedding],
            query_filter=Filter(
                must=[FieldCondition(key="dataset_id", match=MatchValue(value=dataset_id))]
            ),
            limit=limit,
            with_payload=True,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.last_query_embedding_ms = elapsed_ms

        results: list[ImageSearchResult] = []
        for point in response.points:
            payload: dict[str, Any] = point.payload or {}
            results.append(
                ImageSearchResult(
                    dataset_id=payload.get("dataset_id"),
                    img_id=payload.get("img_id"),
                    score=float(point.score) if point.score is not None else None,
                    provider=self.provider_name,
                    query_embedding_ms=elapsed_ms,
                )
            )
        return results


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
