from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, String, create_engine, select, text, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from yolorag.knowledge.image_embeddings import DEFAULT_NOMIC_EMBEDDING_DIMENSIONS
from yolorag.knowledge.image_models import ImageEmbeddingRecord, ImageSearchResult
from yolorag.knowledge.models import IngestResult


DEFAULT_POSTGRES_DSN = "postgresql+psycopg:///yolorag_pgvector"
DEFAULT_IMAGE_EMBEDDINGS_TABLE = "image_embeddings"


class Base(DeclarativeBase):
    pass


class ImageEmbedding(Base):
    __tablename__ = DEFAULT_IMAGE_EMBEDDINGS_TABLE
    __table_args__ = (
        Index(
            "image_embeddings_embedding_hnsw_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    img_id: Mapped[str] = mapped_column(String, primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(DEFAULT_NOMIC_EMBEDDING_DIMENSIONS),
        nullable=False,
    )


@dataclass(frozen=True)
class PostgresImageEmbeddingStoreConfig:
    dsn: str = DEFAULT_POSTGRES_DSN
    table: str = DEFAULT_IMAGE_EMBEDDINGS_TABLE
    embedding_dimensions: int = DEFAULT_NOMIC_EMBEDDING_DIMENSIONS

    @classmethod
    def from_env(cls) -> PostgresImageEmbeddingStoreConfig:
        return cls(
            dsn=os.getenv("YOLORAG_POSTGRES_DSN", DEFAULT_POSTGRES_DSN),
            table=os.getenv(
                "YOLORAG_POSTGRES_IMAGE_EMBEDDINGS_TABLE",
                DEFAULT_IMAGE_EMBEDDINGS_TABLE,
            ),
            embedding_dimensions=_env_int(
                "YOLORAG_IMAGE_EMBEDDING_DIMENSIONS",
                DEFAULT_NOMIC_EMBEDDING_DIMENSIONS,
            ),
        )


class PostgresImageEmbeddingStore:
    """Persist and query Nomic image embeddings in the `image_embeddings` table.

    Mirrors the PostgresKnowledgeStore pattern (SQLAlchemy + pgvector) but keyed by
    the composite `(dataset_id, img_id)` and storing full-precision `vector(128)`.
    """

    provider_name = "postgresql-image"

    def __init__(
        self,
        config: PostgresImageEmbeddingStoreConfig,
        engine: Engine | None = None,
    ) -> None:
        if config.table != DEFAULT_IMAGE_EMBEDDINGS_TABLE:
            raise ValueError(
                "The SQLAlchemy image-embedding model currently supports only "
                f"{DEFAULT_IMAGE_EMBEDDINGS_TABLE!r}."
            )
        if config.embedding_dimensions != DEFAULT_NOMIC_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "The SQLAlchemy image-embedding model currently stores "
                f"{DEFAULT_NOMIC_EMBEDDING_DIMENSIONS}-dimension vector embeddings."
            )
        self.config = config
        self.engine = engine or create_engine(config.dsn)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.last_query_embedding_ms = 0

    def ping(self) -> None:
        with self.Session() as session:
            session.execute(select(1))

    def ensure_schema(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(self.engine)

    def upsert_embeddings(
        self,
        records: Sequence[ImageEmbeddingRecord],
        *,
        batch_size: int = 100,
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
            with self.Session() as session:
                existing = self._existing_keys(
                    session, [(record.dataset_id, record.img_id) for record in batch]
                )
            self._upsert_batch(batch)
            matched += sum(
                1 for record in batch if (record.dataset_id, record.img_id) in existing
            )
            inserted += sum(
                1
                for record in batch
                if (record.dataset_id, record.img_id) not in existing
            )

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
        search_ef: int | None = None,
    ) -> list[ImageSearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        started = time.perf_counter()
        with self.Session() as session:
            if search_ef:
                # pgvector requires hnsw.ef_search >= k; SET LOCAL scopes it to this txn.
                # Postgres SET can't use bind params, so interpolate the validated int.
                ef_value = max(int(search_ef), limit)
                session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_value}"))
            distance = ImageEmbedding.embedding.cosine_distance(list(query_embedding))
            statement = (
                select(ImageEmbedding, (1 - distance).label("score"))
                .where(ImageEmbedding.dataset_id == dataset_id)
                .order_by(distance)
                .limit(limit)
            )
            rows = session.execute(statement).all()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.last_query_embedding_ms = elapsed_ms

        return [
            ImageSearchResult(
                dataset_id=image.dataset_id,
                img_id=image.img_id,
                score=float(score) if score is not None else None,
                provider=self.provider_name,
                query_embedding_ms=elapsed_ms,
            )
            for image, score in rows
        ]

    def _existing_keys(
        self,
        session: Session,
        keys: Sequence[tuple[str, str]],
    ) -> set[tuple[str, str]]:
        if not keys:
            return set()
        rows = session.execute(
            select(ImageEmbedding.dataset_id, ImageEmbedding.img_id).where(
                tuple_(ImageEmbedding.dataset_id, ImageEmbedding.img_id).in_(keys)
            )
        ).all()
        return {(dataset_id, img_id) for dataset_id, img_id in rows}

    def _upsert_batch(self, records: Sequence[ImageEmbeddingRecord]) -> None:
        rows = [
            {
                "dataset_id": record.dataset_id,
                "img_id": record.img_id,
                "embedding": list(record.embedding),
            }
            for record in records
        ]
        statement = insert(ImageEmbedding).values(rows)
        statement = statement.on_conflict_do_update(
            index_elements=[ImageEmbedding.dataset_id, ImageEmbedding.img_id],
            set_={"embedding": statement.excluded.embedding},
        )
        with self.Session() as session:
            session.execute(statement)
            session.commit()


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
