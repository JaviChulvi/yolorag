#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from yolorag.knowledge.factory import build_knowledge_store


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run vector search against the knowledge store.")
    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument("--provider", choices=["mongodb", "postgresql"], default="mongodb")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--kind", help="Optional kind filter, for example article or reference.")
    parser.add_argument("--doc-id", help="Optional doc_id filter.")
    parser.add_argument("--source-path", help="Optional source_path filter.")
    args = parser.parse_args()

    filters = {}
    if args.kind:
        filters["kind"] = args.kind
    if args.doc_id:
        filters["doc_id"] = args.doc_id
    if args.source_path:
        filters["source_path"] = args.source_path

    store = build_knowledge_store(args.provider)
    store.ping()
    if getattr(store, "provider_name", "") == "mongodb":
        index_names = store.search_index_names()
        if store.config.vector_index not in index_names:
            print(f"Configured vector index not found: {store.config.vector_index}")
            print(f"Available search indexes: {', '.join(index_names) if index_names else '(none)'}")
            return 4

    results = store.vector_search(
        args.query,
        limit=args.limit,
        filters=filters or None,
    )

    print(f"Provider: {args.provider}")
    if getattr(store, "provider_name", "") == "mongodb":
        print(f"Index: {store.config.vector_index}")
    if getattr(store, "provider_name", "") == "postgresql":
        print(f"Table: {store.config.table}")
        print(f"Embedding model: {store.embedding_client.model}")
        print(f"Embedding dimensions: {store.config.embedding_dimensions}")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print()

    for index, result in enumerate(results, start=1):
        record = result.record
        text = " ".join(record.text.split())
        preview = text[:500] + ("..." if len(text) > 500 else "")
        score = f"{result.score:.6f}" if result.score is not None else "n/a"
        print("=" * 100)
        print(f"{index}. score={score}")
        print(f"title: {record.title}")
        print(f"source_path: {record.source_path}")
        print(f"url: {record.url}")
        print(f"section: {' > '.join(record.headings)}")
        print("-" * 100)
        print(preview)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
