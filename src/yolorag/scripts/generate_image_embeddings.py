#!/usr/bin/env python3
"""Generate Nomic image embeddings once and store them in vendor-neutral files.

The embedding pass (Nomic ONNX inference) is the expensive step, so it runs once
here and writes ``<dataset>.jsonl`` + ``<dataset>.meta.json`` per dataset (see
``yolorag.knowledge.image_embedding_files``). DB ingestion is a separate, cheap
step (``ingest_image_embeddings.py``) that replays those files into whatever
store you are evaluating -- so switching DB vendors never re-runs embeddings.

Image bytes come from the already-downloaded ``dataset_images/`` tree when present
(see ``download_dataset_images.py``); missing files fall back to the NDJSON's
signed URL. ``dataset_id`` is the dataset name; ``img_id`` is the content hash in
the image URL (e.g. ``.../i/2bcf9074....jpg`` -> ``2bcf9074...``), matching the
platform's image key.

Examples::

    # Embed every dataset under imgs/ using images in dataset_images/, write to embeddings/
    python -m yolorag.scripts.generate_image_embeddings

    # One dataset, custom locations, first 20 images only (smoke test)
    python -m yolorag.scripts.generate_image_embeddings imgs/crotales.ndjson \\
        --images-root dataset_images --out embeddings --limit 20
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from yolorag.knowledge.image_embedding_files import EmbeddingsWriter
from yolorag.scripts.download_dataset_images import (
    DEFAULT_INPUT,
    ImageTask,
    build_tasks,
    iter_ndjson_files,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGES_ROOT = REPO_ROOT / "dataset_images"
DEFAULT_OUTPUT = REPO_ROOT / "embeddings"
USER_AGENT = "yolorag-embedding-generator/1.0"


def img_id_from_url(url: str) -> str:
    """Extract the content-hash id from an image URL, ignoring query/extension."""
    stem = Path(urlparse(url).path).stem
    return stem or url


def load_image_bytes(task: ImageTask, *, timeout: float, retries: int) -> bytes | None:
    """Return image bytes from the local download, or fetch the signed URL as fallback."""
    if task.dest.exists():
        try:
            return task.dest.read_bytes()
        except OSError as exc:
            print(f"    warning: cannot read {task.dest} ({exc}); trying URL", file=sys.stderr)

    request = urllib.request.Request(task.url, headers={"User-Agent": USER_AGENT})
    last_error = ""
    for _ in range(max(1, retries)):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    print(f"    warning: failed to load {task.dest.name} ({last_error})", file=sys.stderr)
    return None


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def generate_for_dataset(
    ndjson_path: Path,
    *,
    client,
    images_root: Path,
    out_dir: Path,
    by_split: bool,
    limit: int | None,
    chunk_size: int,
    timeout: float,
    retries: int,
) -> tuple[int, int]:
    """Embed one dataset's images into ``<dataset>.jsonl``. Returns (embedded, failed)."""
    # The NDJSON stem is the dataset slug: filename-safe, matches the dataset_images/
    # folder and the platform dataset URL (the `name` field may contain spaces).
    dataset_id = ndjson_path.stem
    dataset_dir = images_root / ndjson_path.stem
    tasks = build_tasks(ndjson_path, dataset_dir, by_split)
    if limit is not None:
        tasks = tasks[:limit]

    if not tasks:
        print(f"  {ndjson_path.name}: no image records found")
        return 0, 0

    embedded = 0
    failed = 0
    print(f"  {ndjson_path.name}: dataset_id={dataset_id!r}, {len(tasks)} images")
    with EmbeddingsWriter(
        out_dir,
        dataset_id,
        model=client.model,
        dimensions=client.dimensions,
        source=ndjson_path.name,
    ) as writer:
        for chunk in _chunks(tasks, chunk_size):
            img_ids: list[str] = []
            payloads: list[bytes] = []
            for task in chunk:
                data = load_image_bytes(task, timeout=timeout, retries=retries)
                if data is None:
                    failed += 1
                    continue
                img_ids.append(img_id_from_url(task.url))
                payloads.append(data)
            if not payloads:
                continue
            vectors = client.embed_images(payloads)
            for img_id, vector in zip(img_ids, vectors):
                writer.add(img_id, vector)
            embedded += len(vectors)
            print(f"    {embedded}/{len(tasks)} embedded", flush=True)

        print(f"    wrote {writer.count} -> {writer.jsonl_path}")
    return embedded, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Nomic image embeddings into vendor-neutral JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(DEFAULT_INPUT)],
        help=f"NDJSON files and/or directories to scan for *.ndjson (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--images-root",
        default=str(DEFAULT_IMAGES_ROOT),
        help=f"Root of downloaded images (default: {DEFAULT_IMAGES_ROOT}).",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory for embedding files (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--by-split",
        action="store_true",
        help="Images were downloaded nested under train/val/test subfolders.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Embed at most N images per dataset."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Images loaded+embedded per chunk (bounds memory; default: 256).",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Per-request timeout for URL fallback."
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Attempts per image when fetching a URL."
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        print("--chunk-size must be greater than 0", file=sys.stderr)
        return 2

    ndjson_files = iter_ndjson_files([Path(p) for p in args.inputs])
    if not ndjson_files:
        print("No .ndjson files found.", file=sys.stderr)
        return 1

    # Imported lazily so `--help` works without numpy/onnxruntime/opencv installed.
    from yolorag.knowledge.image_embeddings import (
        NomicImageEmbeddingClient,
        NomicImageEmbeddingConfig,
    )

    client = NomicImageEmbeddingClient(NomicImageEmbeddingConfig.from_env())
    images_root = Path(args.images_root).expanduser()
    out_dir = Path(args.out).expanduser()
    print(
        f"Found {len(ndjson_files)} NDJSON file(s). model={client.model} "
        f"dim={client.dimensions}\nimages_root={images_root.resolve()} "
        f"out={out_dir.resolve()}\n"
    )

    total_embedded = 0
    total_failed = 0
    for ndjson_path in ndjson_files:
        embedded, failed = generate_for_dataset(
            ndjson_path,
            client=client,
            images_root=images_root,
            out_dir=out_dir,
            by_split=args.by_split,
            limit=args.limit,
            chunk_size=args.chunk_size,
            timeout=args.timeout,
            retries=args.retries,
        )
        total_embedded += embedded
        total_failed += failed

    print(f"\nDone. embedded={total_embedded} failed={total_failed}")
    return 1 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
