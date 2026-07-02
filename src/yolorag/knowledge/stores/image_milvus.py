from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

from pymilvus import DataType, MilvusClient

from yolorag.knowledge.image_embeddings import DEFAULT_NOMIC_EMBEDDING_DIMENSIONS
from yolorag.knowledge.image_models import ImageEmbeddingRecord, ImageSearchResult
from yolorag.knowledge.models import IngestResult


DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_IMAGE_EMBEDDINGS_COLLECTION = "image_embeddings"


def _pk(dataset_id: str, img_id: str) -> str:
    """VARCHAR primary key mirroring the Postgres composite PK (dataset_id, img_id)."""
    return f"{dataset_id}/{img_id}"


@dataclass(frozen=True)
class MilvusImageEmbeddingStoreConfig:
    uri: str = DEFAULT_MILVUS_URI
    collection: str = DEFAULT_IMAGE_EMBEDDINGS_COLLECTION
    embedding_dimensions: int = DEFAULT_NOMIC_EMBEDDING_DIMENSIONS

    @classmethod
    def from_env(cls) -> MilvusImageEmbeddingStoreConfig:
        return cls(
            uri=os.getenv("YOLORAG_MILVUS_URI", DEFAULT_MILVUS_URI),
            collection=os.getenv(
                "YOLORAG_MILVUS_IMAGE_EMBEDDINGS_COLLECTION",
                DEFAULT_IMAGE_EMBEDDINGS_COLLECTION,
            ),
            embedding_dimensions=_env_int(
                "YOLORAG_IMAGE_EMBEDDING_DIMENSIONS",
                DEFAULT_NOMIC_EMBEDDING_DIMENSIONS,
            ),
        )


class MilvusImageEmbeddingStore:
    """Persist and query Nomic image embeddings in Milvus.

    Mirrors the other stores: one entity per (dataset_id, img_id) keyed by a VARCHAR
    primary key, 128-dim FLOAT_VECTOR with COSINE metric, dataset_id kept as a scalar
    field for filtered search. The collection is created + loaded by ensure_schema().
    """

    provider_name = "milvus-image"

    def __init__(
        self,
        config: MilvusImageEmbeddingStoreConfig,
        client: MilvusClient | None = None,
    ) -> None:
        if config.embedding_dimensions != DEFAULT_NOMIC_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "The Milvus image-embedding store currently stores "
                f"{DEFAULT_NOMIC_EMBEDDING_DIMENSIONS}-dimension vectors."
            )
        self.config = config
        self.client = client or MilvusClient(uri=config.uri)

        self.last_query_embedding_ms = 0

    def ping(self) -> None:
        self.client.list_collections()

    def ensure_schema(self) -> None:
        """Create the collection + HNSW cosine index if missing, then load it."""
        if not self.client.has_collection(self.config.collection):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=256)
            schema.add_field("dataset_id", DataType.VARCHAR, max_length=256)
            schema.add_field("img_id", DataType.VARCHAR, max_length=128)
            schema.add_field(
                "embedding",
                DataType.FLOAT_VECTOR,
                dim=self.config.embedding_dimensions,
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 64},
            )
            self.client.create_collection(
                self.config.collection,
                schema=schema,
                index_params=index_params,
            )
        self.client.load_collection(self.config.collection)

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
            ids = [_pk(record.dataset_id, record.img_id) for record in batch]
            existing = {row["id"] for row in self.client.get(self.config.collection, ids=ids)}
            rows = [
                {
                    "id": pk,
                    "dataset_id": record.dataset_id,
                    "img_id": record.img_id,
                    "embedding": [float(value) for value in record.embedding],
                }
                for pk, record in zip(ids, batch)
            ]
            self.client.upsert(self.config.collection, data=rows)
            matched += sum(1 for pk in ids if pk in existing)
            inserted += sum(1 for pk in ids if pk not in existing)

        self.client.flush(self.config.collection)
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

        self.ensure_schema()
        started = time.perf_counter()
        response = self.client.search(
            self.config.collection,
            data=[[float(value) for value in query_embedding]],
            anns_field="embedding",
            limit=limit,
            filter=f'dataset_id == "{dataset_id}"',
            output_fields=["dataset_id", "img_id"],
            search_params={"metric_type": "COSINE"},
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.last_query_embedding_ms = elapsed_ms

        results: list[ImageSearchResult] = []
        for hit in response[0]:
            entity: dict[str, Any] = hit.get("entity", {})
            results.append(
                ImageSearchResult(
                    dataset_id=entity.get("dataset_id"),
                    img_id=entity.get("img_id"),
                    score=float(hit["distance"]) if hit.get("distance") is not None else None,
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
