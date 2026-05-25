from __future__ import annotations

import unittest
from typing import Sequence

from yolorag.knowledge.models import ChunkRecord, IngestResult, SearchResult
from yolorag.knowledge.pipeline import ingest_records, summarize_records


class RecordingStore:
    provider_name = "recording"

    def __init__(self) -> None:
        self.records: list[ChunkRecord] = []
        self.batch_size: int | None = None

    def ingest_chunks(
        self,
        records: Sequence[ChunkRecord],
        *,
        batch_size: int = 100,
    ) -> IngestResult:
        self.records.extend(records)
        self.batch_size = batch_size
        return IngestResult(attempted=len(records), inserted=len(records), provider=self.provider_name)

    def vector_search(
        self,
        query: str,
        *,
        limit: int = 8,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        return []


class KnowledgeStoreTests(unittest.TestCase):
    def test_chunk_record_round_trips_through_mapping(self) -> None:
        record = _record("chunk-1")

        restored = ChunkRecord.from_mapping(record.to_mapping())

        self.assertEqual(restored, record)

    def test_chunk_record_accepts_setup_test_style_documents(self) -> None:
        record = ChunkRecord.from_mapping(
            {
                "_id": "setup-test",
                "doc_id": "setup-test",
                "chunk_id": "setup-test-0",
                "kind": "article",
                "source_path": "setup.md",
                "title": "Setup Test",
                "url": "https://docs.ultralytics.com/",
                "headings": ["Setup Test"],
                "text": "YOLO models can be trained with Ultralytics.",
            }
        )

        self.assertEqual(record.record_id, "setup-test")
        self.assertEqual(record.chunk_index, 0)
        self.assertEqual(record.source, "unknown")
        self.assertEqual(record.content, record.text)
        self.assertEqual(record.char_count, len(record.text))

    def test_summary_counts_records_without_provider_coupling(self) -> None:
        records = [_record("chunk-1"), _record("chunk-2", kind="reference")]

        summary = summarize_records(records)

        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.kinds, {"article": 1, "reference": 1})
        self.assertEqual(summary.estimated_tokens, 20)
        self.assertEqual(summary.total_chars, 200)

    def test_ingest_records_uses_store_interface(self) -> None:
        store = RecordingStore()
        records = [_record("chunk-1")]

        result = ingest_records(store, records, batch_size=7)

        self.assertEqual(result.provider, "recording")
        self.assertEqual(result.inserted, 1)
        self.assertEqual(store.records, records)
        self.assertEqual(store.batch_size, 7)


def _record(record_id: str, kind: str = "article") -> ChunkRecord:
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
        kind=kind,
        text="Title: Quickstart\n\nTrain a YOLO model.",
        content="Train a YOLO model.",
        char_count=100,
        estimated_tokens=10,
        content_hash="abc123",
        reference_symbols=[],
    )


if __name__ == "__main__":
    unittest.main()
