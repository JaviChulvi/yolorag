# Knowledge Store Architecture

The RAG knowledge layer is provider-agnostic at the application boundary, with MongoDB as the first concrete implementation.

## Flow

```text
Markdown docs
  -> yolorag.ingestion.docs_chunker
  -> yolorag.knowledge.models.ChunkRecord
  -> yolorag.knowledge.stores.base.KnowledgeStore
  -> MongoDB Atlas implementation
```

The chunker does not know about databases. It only turns local Markdown files into `DocsChunk` objects.

The knowledge layer converts those chunks into stable `ChunkRecord` objects. These records are the contract between ingestion, retrieval, and storage providers.

## Interface

The provider-neutral store exposes two main methods:

```python
store.ingest_chunks(records)
store.vector_search(query, limit=8, filters=None)
```

For now, only MongoDB is implemented. A future Postgres/pgvector store should implement the same methods without changing the ingestion pipeline or chat retrieval code.

## MongoDB Mapping

MongoDB stores one document per chunk in `docs_chunks`.

```text
_id = ChunkRecord.record_id = chunk_id
text = field indexed by MongoDB Automated Embedding
kind/doc_id/source_path = filterable metadata fields
```

Rerunning ingestion uses upserts, so unchanged chunks are matched and changed chunks are updated instead of duplicated.

MongoDB Automated Embedding owns vector generation. YoloRAG stores the `text` field; MongoDB embeds it at index time and embeds query text at query time through the vector search index.

The MongoDB adapter reads the vector index name from `YOLORAG_MONGODB_VECTOR_INDEX`. For this project, the Atlas index is currently `autoembed_index`.
