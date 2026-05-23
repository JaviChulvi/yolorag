from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import DateTime, Index, Integer, JSON, String, Text, create_engine, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from yolorag.knowledge.embeddings import (
    DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
    EmbeddingClient,
    OpenAIEmbeddingClient,
    OpenAIEmbeddingConfig,
)
from yolorag.knowledge.models import ChunkRecord, IngestResult, SearchResult


DEFAULT_POSTGRES_DSN = "postgresql+psycopg:///yolorag_pgvector"
DEFAULT_CHUNKS_TABLE = "docs_chunks"


class Base(DeclarativeBase):
    pass


class PostgresChunk(Base):
    __tablename__ = DEFAULT_CHUNKS_TABLE
    __table_args__ = (
        Index("docs_chunks_chunk_id_idx", "chunk_id", unique=True),
        Index("docs_chunks_doc_id_idx", "doc_id"),
        Index("docs_chunks_kind_idx", "kind"),
        Index(
            "docs_chunks_embedding_hnsw_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "halfvec_cosine_ops"},
        ),
    )

    record_id: Mapped[str] = mapped_column(String, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String, nullable=False)
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    source_path: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str | None] = mapped_column(String, nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    headings: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    estimated_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    reference_symbols: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    embedding: Mapped[list[float]] = mapped_column(
        HALFVEC(DEFAULT_OPENAI_EMBEDDING_DIMENSIONS),
        nullable=False,
    )
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    @classmethod
    def from_record(
        cls,
        record: ChunkRecord,
        *,
        embedding: Sequence[float],
        embedding_model: str,
        embedding_dimensions: int,
        updated_at: datetime,
    ) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "chunk_id": record.chunk_id,
            "doc_id": record.doc_id,
            "chunk_index": record.chunk_index,
            "source": record.source,
            "source_path": record.source_path,
            "url": record.url,
            "title": record.title,
            "headings": list(record.headings),
            "kind": record.kind,
            "text": record.text,
            "content": record.content,
            "char_count": record.char_count,
            "estimated_tokens": record.estimated_tokens,
            "content_hash": record.content_hash,
            "reference_symbols": list(record.reference_symbols),
            "embedding": list(embedding),
            "embedding_model": embedding_model,
            "embedding_dimensions": embedding_dimensions,
            "updated_at": updated_at,
        }

    def to_chunk_record(self) -> ChunkRecord:
        return ChunkRecord(
            record_id=self.record_id,
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            chunk_index=self.chunk_index,
            source=self.source,
            source_path=self.source_path,
            url=self.url,
            title=self.title,
            headings=list(self.headings or []),
            kind=self.kind,
            text=self.text,
            content=self.content,
            char_count=self.char_count,
            estimated_tokens=self.estimated_tokens,
            content_hash=self.content_hash,
            reference_symbols=list(self.reference_symbols or []),
        )


@dataclass(frozen=True)
class PostgresKnowledgeStoreConfig:
    dsn: str = DEFAULT_POSTGRES_DSN
    table: str = DEFAULT_CHUNKS_TABLE
    embedding_dimensions: int = DEFAULT_OPENAI_EMBEDDING_DIMENSIONS

    @classmethod
    def from_env(cls) -> PostgresKnowledgeStoreConfig:
        return cls(
            dsn=os.getenv("YOLORAG_POSTGRES_DSN", DEFAULT_POSTGRES_DSN),
            table=os.getenv("YOLORAG_POSTGRES_CHUNKS_TABLE", DEFAULT_CHUNKS_TABLE),
            embedding_dimensions=_env_int(
                "YOLORAG_POSTGRES_EMBEDDING_DIMENSIONS",
                DEFAULT_OPENAI_EMBEDDING_DIMENSIONS,
            ),
        )


class PostgresKnowledgeStore:
    provider_name = "postgresql"

    def __init__(
        self,
        config: PostgresKnowledgeStoreConfig,
        embedding_client: EmbeddingClient | None = None,
        engine: Engine | None = None,
    ) -> None:
        if config.table != DEFAULT_CHUNKS_TABLE:
            raise ValueError(
                "The SQLAlchemy Postgres model currently supports only "
                f"{DEFAULT_CHUNKS_TABLE!r}."
            )
        if config.embedding_dimensions != DEFAULT_OPENAI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "The SQLAlchemy Postgres model currently stores "
                f"{DEFAULT_OPENAI_EMBEDDING_DIMENSIONS}-dimension halfvec embeddings."
            )
        self.config = config
        self.embedding_client = embedding_client or OpenAIEmbeddingClient(
            OpenAIEmbeddingConfig.from_env()
        )
        if self.embedding_client.dimensions != config.embedding_dimensions:
            raise ValueError(
                "Embedding client dimensions do not match Postgres store dimensions."
            )
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

    def ingest_chunks(
        self,
        records: Sequence[ChunkRecord],
        *,
        batch_size: int = 100,
    ) -> IngestResult:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.ensure_schema()
        attempted = len(records)
        inserted = 0
        matched = 0
        modified = 0

        for start in range(0, attempted, batch_size):
            batch = list(records[start : start + batch_size])
            if not batch:
                continue
            with self.Session() as session:
                existing_ids = self._existing_record_ids(
                    session,
                    [record.record_id for record in batch],
                )
            embeddings = self.embedding_client.embed_texts([record.text for record in batch])
            self._upsert_batch(batch, embeddings)
            matched += len(existing_ids)
            modified += len(existing_ids)
            inserted += len(batch) - len(existing_ids)

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

        embedding_started = time.perf_counter()
        query_embedding = self.embedding_client.embed_texts([query])[0]
        self.last_query_embedding_ms = int((time.perf_counter() - embedding_started) * 1000)

        with self.Session() as session:
            distance = PostgresChunk.embedding.cosine_distance(query_embedding)
            statement = (
                select(PostgresChunk, (1 - distance).label("score"))
                .order_by(distance)
                .limit(limit)
            )
            for key, value in (filters or {}).items():
                statement = statement.where(_filter_expression(key, value))
            rows = session.execute(statement).all()

        return [
            SearchResult(
                record=chunk.to_chunk_record(),
                score=float(score) if score is not None else None,
                provider=self.provider_name,
            )
            for chunk, score in rows
        ]

    def _existing_record_ids(
        self,
        session: Session,
        record_ids: Sequence[str],
    ) -> set[str]:
        if not record_ids:
            return set()
        return set(
            session.scalars(
                select(PostgresChunk.record_id).where(PostgresChunk.record_id.in_(record_ids))
            )
        )

    def _upsert_batch(
        self,
        records: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(records) != len(embeddings):
            raise ValueError("records and embeddings must have the same length.")
        now = datetime.now(UTC)
        rows = [
            PostgresChunk.from_record(
                record,
                embedding=embedding,
                embedding_model=self.embedding_client.model,
                embedding_dimensions=self.config.embedding_dimensions,
                updated_at=now,
            )
            for record, embedding in zip(records, embeddings, strict=True)
        ]
        statement = insert(PostgresChunk).values(rows)
        update_fields = {
            column.name: getattr(statement.excluded, column.name)
            for column in PostgresChunk.__table__.columns
            if column.name not in {"record_id", "created_at"}
        }
        with self.Session() as session:
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=[PostgresChunk.record_id],
                    set_=update_fields,
                )
            )
            session.commit()


def _filter_expression(key: str, value: Any) -> Any:
    allowed = {"record_id", "chunk_id", "doc_id", "source", "source_path", "url", "title", "kind"}
    if key not in allowed:
        raise ValueError(f"Unsupported Postgres vector-search filter {key!r}.")
    return getattr(PostgresChunk, key) == value


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
