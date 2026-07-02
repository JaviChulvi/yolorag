#!/usr/bin/env python3
"""Ingest pre-computed image embedding files into a vector store.

This is the cheap, replayable half of the image pipeline: it reads the
vendor-neutral ``<dataset>.jsonl`` files produced by
``generate_image_embeddings.py`` and upserts them into a store. Because the
embeddings are already on disk, you can point this at a different DB vendor
without ever re-running the Nomic ONNX pass.

Examples::

    # Ingest every embeddings/*.jsonl into Postgres (uses YOLORAG_POSTGRES_DSN)
    python -m yolorag.scripts.ingest_image_embeddings

    # A single dataset, explicit DSN
    python -m yolorag.scripts.ingest_image_embeddings embeddings/crotales.jsonl \\
        --dsn postgresql+psycopg://user:pw@localhost:5432/yolorag_pgvector
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yolorag.knowledge.image_embedding_files import (
    iter_embedding_files,
    read_embeddings,
    read_meta,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "embeddings"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest pre-computed image embeddings into a vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(DEFAULT_INPUT)],
        help=f"Embedding .jsonl files and/or directories (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--provider",
        default="postgresql",
        choices=["postgresql", "mongodb", "qdrant", "milvus"],
        help="Target store (default: postgresql). More vendors can be added here.",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Override the Postgres DSN (postgresql only; else YOLORAG_POSTGRES_DSN / default).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Upsert batch size (default: 500)."
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        print("--batch-size must be greater than 0", file=sys.stderr)
        return 2

    files = iter_embedding_files([Path(p) for p in args.inputs])
    if not files:
        print("No .jsonl embedding files found.", file=sys.stderr)
        return 1

    store = _build_store(args.provider, args.dsn)
    store.ensure_schema()

    from yolorag.knowledge.image_models import ImageEmbeddingRecord

    print(f"Ingesting {len(files)} file(s) into {store.provider_name}\n")
    total_attempted = total_inserted = total_matched = 0
    for jsonl_path in files:
        meta_path = jsonl_path.parent / f"{jsonl_path.stem}.meta.json"
        expected = read_meta(meta_path).get("count") if meta_path.exists() else None

        records = [
            ImageEmbeddingRecord(
                dataset_id=row["dataset_id"],
                img_id=row["img_id"],
                embedding=row["embedding"],
            )
            for row in read_embeddings(jsonl_path)
        ]
        if expected is not None and expected != len(records):
            print(
                f"  warning: {jsonl_path.name} has {len(records)} rows "
                f"but meta.count={expected}",
                file=sys.stderr,
            )

        result = store.upsert_embeddings(records, batch_size=args.batch_size)
        total_attempted += result.attempted
        total_inserted += result.inserted
        total_matched += result.matched
        print(
            f"  {jsonl_path.name}: attempted={result.attempted} "
            f"inserted={result.inserted} updated={result.matched}"
        )

    print(
        f"\nDone. attempted={total_attempted} inserted={total_inserted} "
        f"updated={total_matched}"
    )
    return 0


def _build_store(provider: str, dsn: str | None):
    if provider == "postgresql":
        from yolorag.knowledge.stores.image_postgresql import (
            PostgresImageEmbeddingStore,
            PostgresImageEmbeddingStoreConfig,
        )

        config = PostgresImageEmbeddingStoreConfig.from_env()
        if dsn:
            config = PostgresImageEmbeddingStoreConfig(
                dsn=dsn,
                table=config.table,
                embedding_dimensions=config.embedding_dimensions,
            )
        return PostgresImageEmbeddingStore(config)
    if provider == "mongodb":
        from yolorag.knowledge.stores.image_mongodb import (
            MongoImageEmbeddingStore,
            MongoImageEmbeddingStoreConfig,
        )

        if dsn:
            raise ValueError("Use YOLORAG_MONGODB_URI for mongodb, not --dsn.")
        return MongoImageEmbeddingStore(MongoImageEmbeddingStoreConfig.from_env())
    if provider == "qdrant":
        from yolorag.knowledge.stores.image_qdrant import (
            QdrantImageEmbeddingStore,
            QdrantImageEmbeddingStoreConfig,
        )

        if dsn:
            raise ValueError("Use YOLORAG_QDRANT_URL for qdrant, not --dsn.")
        return QdrantImageEmbeddingStore(QdrantImageEmbeddingStoreConfig.from_env())
    if provider == "milvus":
        from yolorag.knowledge.stores.image_milvus import (
            MilvusImageEmbeddingStore,
            MilvusImageEmbeddingStoreConfig,
        )

        if dsn:
            raise ValueError("Use YOLORAG_MILVUS_URI for milvus, not --dsn.")
        return MilvusImageEmbeddingStore(MilvusImageEmbeddingStoreConfig.from_env())
    raise ValueError(f"Unsupported provider {provider!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
