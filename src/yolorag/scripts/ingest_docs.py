#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from yolorag.ingestion.docs_chunker import (
    DEFAULT_DOCS_ROOT,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
)
from yolorag.knowledge.factory import build_knowledge_store
from yolorag.knowledge.pipeline import build_docs_records, ingest_records, summarize_records


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Ingest local Ultralytics docs chunks into a knowledge store."
    )
    parser.add_argument(
        "--provider",
        choices=["mongodb", "postgresql"],
        default="mongodb",
        help="Knowledge-store provider.",
    )
    parser.add_argument(
        "--docs-root",
        default=str(DEFAULT_DOCS_ROOT),
        help="Path to the Ultralytics docs folder. Defaults to ../ultralytics/docs.",
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Include docs/en/reference pages. Omitted by default for first RAG ingestion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chunks to ingest. Defaults to all selected chunks.",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write to the knowledge store. Without this, the script is a dry run.",
    )
    args = parser.parse_args()

    records = build_docs_records(
        docs_root=Path(args.docs_root),
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        include_reference=args.include_reference,
        limit=args.limit,
    )
    summary = summarize_records(records)

    print(f"Provider: {args.provider}")
    print(f"Docs root: {Path(args.docs_root).expanduser().resolve()}")
    print(f"Records prepared: {summary.total}")
    print(f"Kinds: {summary.kinds}")
    print(f"Estimated tokens: {summary.estimated_tokens}")
    print(f"Total chars: {summary.total_chars}")

    if records:
        first = records[0]
        print()
        print("First record:")
        print(f"  id: {first.record_id}")
        print(f"  title: {first.title}")
        print(f"  source_path: {first.source_path}")
        print(f"  headings: {' > '.join(first.headings)}")

    if not args.write:
        print()
        print("Dry run only. Re-run with --write to persist records.")
        return 0

    store = build_knowledge_store(args.provider)
    store.ping()
    print()
    print("Target:")
    _print_store_target(store)
    result = ingest_records(store, records, batch_size=args.batch_size)

    print()
    print("Ingestion complete:")
    print(f"  attempted: {result.attempted}")
    print(f"  inserted: {result.inserted}")
    print(f"  matched: {result.matched}")
    print(f"  modified: {result.modified}")
    return 0


def _print_store_target(store: object) -> None:
    config = getattr(store, "config", None)
    if getattr(store, "provider_name", "") == "mongodb":
        print(f"  database: {config.database}")
        print(f"  collection: {config.collection}")
        print(f"  vector_index: {config.vector_index}")
        return
    if getattr(store, "provider_name", "") == "postgresql":
        print(f"  dsn: {config.dsn}")
        print(f"  table: {config.table}")
        print(f"  embedding_dimensions: {config.embedding_dimensions}")
        return
    print(f"  provider: {getattr(store, 'provider_name', 'unknown')}")


if __name__ == "__main__":
    raise SystemExit(main())
