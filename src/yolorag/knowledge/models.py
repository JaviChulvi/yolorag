from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from yolorag.ingestion.docs_chunker import DocsChunk


@dataclass(frozen=True)
class ChunkRecord:
    record_id: str
    chunk_id: str
    doc_id: str
    chunk_index: int
    source: str
    source_path: str
    url: str | None
    title: str
    headings: list[str]
    kind: str
    text: str
    content: str
    char_count: int
    estimated_tokens: int
    content_hash: str
    reference_symbols: list[str]

    @classmethod
    def from_docs_chunk(
        cls,
        chunk: DocsChunk,
        source: str = "ultralytics-docs",
    ) -> ChunkRecord:
        return cls(
            record_id=chunk.chunk_id,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            source=source,
            source_path=chunk.source_path,
            url=chunk.url,
            title=chunk.title,
            headings=list(chunk.headings),
            kind=chunk.kind,
            text=chunk.text,
            content=chunk.content,
            char_count=chunk.char_count,
            estimated_tokens=chunk.estimated_tokens,
            content_hash=chunk.content_hash,
            reference_symbols=list(chunk.reference_symbols),
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ChunkRecord:
        record_id = data.get("record_id") or data.get("_id") or data["chunk_id"]
        text = str(data["text"])
        return cls(
            record_id=str(record_id),
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            chunk_index=int(data.get("chunk_index", 0)),
            source=str(data.get("source", "unknown")),
            source_path=str(data["source_path"]),
            url=data.get("url"),
            title=str(data["title"]),
            headings=list(data.get("headings", [])),
            kind=str(data["kind"]),
            text=text,
            content=str(data.get("content", text)),
            char_count=int(data.get("char_count", len(text))),
            estimated_tokens=int(data.get("estimated_tokens", 0)),
            content_hash=str(data.get("content_hash", "")),
            reference_symbols=list(data.get("reference_symbols", [])),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestResult:
    attempted: int
    inserted: int = 0
    matched: int = 0
    modified: int = 0
    provider: str = "unknown"


@dataclass(frozen=True)
class SearchResult:
    record: ChunkRecord
    score: float | None
    provider: str = "unknown"
