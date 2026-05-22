from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from yolorag.ingestion.docs_chunker import (
    DEFAULT_DOCS_ROOT,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    chunk_markdown_docs,
)
from yolorag.knowledge.models import ChunkRecord, IngestResult
from yolorag.knowledge.stores.base import KnowledgeStore


@dataclass(frozen=True)
class RecordsSummary:
    total: int
    kinds: dict[str, int]
    estimated_tokens: int
    total_chars: int


def build_docs_records(
    docs_root: str | Path = DEFAULT_DOCS_ROOT,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    include_reference: bool = False,
    limit: int | None = None,
    source: str = "ultralytics-docs",
) -> list[ChunkRecord]:
    chunks = chunk_markdown_docs(
        docs_root=docs_root,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        include_reference=include_reference,
    )
    if limit is not None:
        chunks = chunks[:limit]
    return [ChunkRecord.from_docs_chunk(chunk, source=source) for chunk in chunks]


def summarize_records(records: Sequence[ChunkRecord]) -> RecordsSummary:
    return RecordsSummary(
        total=len(records),
        kinds=dict(Counter(record.kind for record in records)),
        estimated_tokens=sum(record.estimated_tokens for record in records),
        total_chars=sum(record.char_count for record in records),
    )


def ingest_records(
    store: KnowledgeStore,
    records: Sequence[ChunkRecord],
    *,
    batch_size: int = 100,
) -> IngestResult:
    return store.ingest_chunks(records, batch_size=batch_size)
