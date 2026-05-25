#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median

from yolorag.ingestion.docs_chunker import (
    DEFAULT_DOCS_ROOT,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    chunk_markdown_docs,
    has_unclosed_fence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview Markdown chunks from the local Ultralytics docs tree."
    )
    parser.add_argument(
        "--docs-root",
        default=str(DEFAULT_DOCS_ROOT),
        help="Path to the Ultralytics docs folder. Defaults to ../ultralytics/docs.",
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    parser.add_argument("--limit", type=int, default=12, help="Number of chunks to print. Use 0 for all.")
    parser.add_argument("--jsonl", action="store_true", help="Print each chunk as JSONL.")
    parser.add_argument(
        "--exclude-reference",
        action="store_true",
        help="Skip docs/en/reference pages, which are mkdocstrings/source-reference pages.",
    )
    args = parser.parse_args()

    chunks = chunk_markdown_docs(
        docs_root=Path(args.docs_root),
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        include_reference=not args.exclude_reference,
    )

    selected = chunks if args.limit == 0 else chunks[: args.limit]
    if args.jsonl:
        for chunk in selected:
            print(chunk.to_json())
        return

    print(f"Docs root: {Path(args.docs_root).expanduser().resolve()}")
    print(
        "Strategy: Markdown heading sections, paragraph packing, "
        f"max_chars={args.max_chars}, overlap_chars={args.overlap_chars}"
    )
    print(f"Chunks created: {len(chunks)}")
    print(f"Printing: {len(selected)}")
    if chunks:
        lengths = sorted(chunk.char_count for chunk in chunks)
        kinds = {}
        for chunk in chunks:
            kinds[chunk.kind] = kinds.get(chunk.kind, 0) + 1
        print(f"Kinds: {kinds}")
        print(
            "Chunk chars: "
            f"min={lengths[0]}, median={int(median(lengths))}, "
            f"p95={lengths[int(0.95 * (len(lengths) - 1))]}, max={lengths[-1]}"
        )
        print(f"Code chunks: {sum(('```' in chunk.content or '~~~' in chunk.content) for chunk in chunks)}")
        print(f"Chunks with unclosed fences: {sum(has_unclosed_fence(chunk.content) for chunk in chunks)}")
        print(f"Reference-symbol chunks: {sum(bool(chunk.reference_symbols) for chunk in chunks)}")
    print()

    for index, chunk in enumerate(selected, start=1):
        print("=" * 100)
        print(f"Chunk {index}/{len(selected)}")
        print(f"id: {chunk.chunk_id}")
        print(f"kind: {chunk.kind}")
        print(f"source: {chunk.source_path}")
        print(f"url: {chunk.url}")
        print(f"title: {chunk.title}")
        print(f"section: {' > '.join(chunk.headings)}")
        if chunk.reference_symbols:
            print(f"reference_symbols: {', '.join(chunk.reference_symbols)}")
        print(f"chars: {chunk.char_count} | estimated_tokens: {chunk.estimated_tokens}")
        print("-" * 100)
        print(chunk.text)
        print()


if __name__ == "__main__":
    main()
