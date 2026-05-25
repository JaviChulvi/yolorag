from __future__ import annotations

import unittest
from datetime import UTC, datetime

from yolorag.knowledge.models import ChunkRecord
from yolorag.knowledge.stores.postgresql import (
    DEFAULT_POSTGRES_DSN,
    PostgresChunk,
    PostgresKnowledgeStoreConfig,
)


class PostgresKnowledgeStoreTests(unittest.TestCase):
    def test_config_defaults_to_local_pgvector_database(self) -> None:
        config = PostgresKnowledgeStoreConfig()

        self.assertEqual(config.dsn, DEFAULT_POSTGRES_DSN)
        self.assertEqual(config.table, "docs_chunks")
        self.assertEqual(config.embedding_dimensions, 3072)

    def test_chunk_model_round_trips_record_fields(self) -> None:
        record = _record("chunk-1")
        data = PostgresChunk.from_record(
            record,
            embedding=[0.1] * 3072,
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
            updated_at=datetime.now(UTC),
        )

        model = PostgresChunk(**data)
        restored = model.to_chunk_record()

        self.assertEqual(restored, record)
        self.assertEqual(model.embedding_model, "text-embedding-3-large")
        self.assertEqual(model.embedding_dimensions, 3072)


def _record(record_id: str) -> ChunkRecord:
    return ChunkRecord(
        record_id=record_id,
        chunk_id=record_id,
        doc_id="quickstart",
        chunk_index=0,
        source="test",
        source_path="en/quickstart.md",
        url="https://docs.ultralytics.com/quickstart/",
        title="Quickstart",
        headings=["Quickstart"],
        kind="article",
        text="Title: Quickstart\n\nTrain a YOLO model.",
        content="Train a YOLO model.",
        char_count=100,
        estimated_tokens=10,
        content_hash="abc123",
        reference_symbols=["YOLO"],
    )


if __name__ == "__main__":
    unittest.main()
