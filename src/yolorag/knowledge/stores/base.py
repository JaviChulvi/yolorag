from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from yolorag.knowledge.models import ChunkRecord, IngestResult, SearchResult


class KnowledgeStore(Protocol):
    provider_name: str

    def ingest_chunks(
        self,
        records: Sequence[ChunkRecord],
        *,
        batch_size: int = 100,
    ) -> IngestResult:
        """Persist chunk records, updating existing records by stable id."""

    def vector_search(
        self,
        query: str,
        *,
        limit: int = 8,
        filters: Mapping[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return semantically similar chunks for a natural-language query."""

