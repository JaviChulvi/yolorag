#!/usr/bin/env python3
"""Generate clustered synthetic image embeddings for scaled vector-DB benchmarking.

Real embeddings are only ~137 vectors, too small to stress ANN indexes. This writes
unit-norm 128-d vectors drawn from a Gaussian mixture (cluster centers + spread) in
the standard embeddings JSONL format, so the existing ingest path replays them into
every vendor identically. It also writes a held-out query set (same distribution,
NOT in the corpus) so recall@k stays honest, plus a .meta.json sidecar.

Output is regenerable from --seed, so write it to a gitignored dir (default bench_data/).

Examples::

    # 1,000,000 vectors, 256 clusters, 2,000 held-out queries
    python -m yolorag.scripts.generate_synthetic_embeddings --count 1000000

    # quick smoke set
    python -m yolorag.scripts.generate_synthetic_embeddings --count 20000 \\
        --dataset-id synthetic-20k --queries 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "bench_data"


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def _stream_vectors(
    path: Path,
    dataset_id: str,
    prefix: str,
    centers: np.ndarray,
    count: int,
    spread: float,
    rng: np.random.Generator,
    *,
    chunk: int = 20_000,
) -> int:
    dim = centers.shape[1]
    written = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        while written < count:
            n = min(chunk, count - written)
            assignments = rng.integers(0, len(centers), size=n)
            noise = rng.normal(0.0, spread, size=(n, dim)).astype(np.float32)
            vectors = _unit_rows(centers[assignments] + noise)
            lines = []
            for i in range(n):
                emb = ",".join(f"{value:.6g}" for value in vectors[i].tolist())
                img_id = f"{prefix}{written + i:09d}"
                lines.append(
                    '{"dataset_id":"%s","img_id":"%s","embedding":[%s]}'
                    % (dataset_id, img_id, emb)
                )
            handle.write("\n".join(lines))
            handle.write("\n")
            written += n
            print(f"  {path.name}: {written}/{count}", end="\r", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate clustered synthetic embeddings for scaled benchmarking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--count", type=int, default=100_000, help="Number of corpus vectors.")
    parser.add_argument("--dim", type=int, default=128, help="Embedding dimension (match Nomic: 128).")
    parser.add_argument("--clusters", type=int, default=256, help="Gaussian mixture centers.")
    parser.add_argument("--spread", type=float, default=0.15, help="Per-cluster stddev before renorm.")
    parser.add_argument("--queries", type=int, default=2000, help="Held-out query vectors to generate.")
    parser.add_argument("--dataset-id", default=None, help="Dataset id (default: synthetic-<count>).")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output dir (default: {DEFAULT_OUT}).")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed (deterministic output).")
    args = parser.parse_args()

    if args.count <= 0 or args.dim <= 0 or args.clusters <= 0:
        print("--count, --dim, --clusters must be positive.", file=sys.stderr)
        return 2

    dataset_id = args.dataset_id or f"synthetic-{_human(args.count)}"
    out_dir = Path(args.out).expanduser()
    rng = np.random.default_rng(args.seed)

    centers = _unit_rows(rng.normal(0.0, 1.0, size=(args.clusters, args.dim)).astype(np.float32))

    corpus_path = out_dir / f"{dataset_id}.jsonl"
    queries_path = out_dir / f"{dataset_id}.queries.jsonl"
    meta_path = out_dir / f"{dataset_id}.meta.json"

    print(
        f"Generating {args.count} corpus + {args.queries} query vectors "
        f"({args.clusters} clusters, dim={args.dim}) -> {out_dir}",
        file=sys.stderr,
    )
    _stream_vectors(corpus_path, dataset_id, "c", centers, args.count, args.spread, rng)
    _stream_vectors(queries_path, dataset_id, "q", centers, args.queries, args.spread, rng)

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_id": dataset_id,
                "synthetic": True,
                "model": "clustered-gaussian",
                "dimensions": args.dim,
                "count": args.count,
                "clusters": args.clusters,
                "spread": args.spread,
                "queries": args.queries,
                "seed": args.seed,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print(f"Done. corpus={corpus_path}  queries={queries_path}", file=sys.stderr)
    print(dataset_id)
    return 0


def _human(count: int) -> str:
    if count % 1_000_000 == 0:
        return f"{count // 1_000_000}m"
    if count % 1_000 == 0:
        return f"{count // 1_000}k"
    return str(count)


if __name__ == "__main__":
    raise SystemExit(main())
