from __future__ import annotations


IMAGE_STORE_PROVIDERS = ("postgresql", "mongodb", "qdrant", "milvus")


def build_image_store(provider: str, *, postgres_dsn: str | None = None):
    """Build an image-embedding store for a provider, reading connection env.

    Shared by the ingest CLI and the benchmark API so the provider list stays in
    one place. Imports are lazy so selecting one provider never requires the
    others' client libraries.
    """
    if provider == "postgresql":
        from yolorag.knowledge.stores.image_postgresql import (
            PostgresImageEmbeddingStore,
            PostgresImageEmbeddingStoreConfig,
        )

        config = PostgresImageEmbeddingStoreConfig.from_env()
        if postgres_dsn:
            config = PostgresImageEmbeddingStoreConfig(
                dsn=postgres_dsn,
                table=config.table,
                embedding_dimensions=config.embedding_dimensions,
            )
        return PostgresImageEmbeddingStore(config)
    if postgres_dsn:
        raise ValueError("--dsn is only valid for the postgresql provider.")
    if provider == "mongodb":
        from yolorag.knowledge.stores.image_mongodb import (
            MongoImageEmbeddingStore,
            MongoImageEmbeddingStoreConfig,
        )

        return MongoImageEmbeddingStore(MongoImageEmbeddingStoreConfig.from_env())
    if provider == "qdrant":
        from yolorag.knowledge.stores.image_qdrant import (
            QdrantImageEmbeddingStore,
            QdrantImageEmbeddingStoreConfig,
        )

        return QdrantImageEmbeddingStore(QdrantImageEmbeddingStoreConfig.from_env())
    if provider == "milvus":
        from yolorag.knowledge.stores.image_milvus import (
            MilvusImageEmbeddingStore,
            MilvusImageEmbeddingStoreConfig,
        )

        return MilvusImageEmbeddingStore(MilvusImageEmbeddingStoreConfig.from_env())
    raise ValueError(f"Unsupported image store provider {provider!r}.")
