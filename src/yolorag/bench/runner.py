"""Fair vector-DB benchmark: same query workload replayed against each store.

Fairness is enforced by construction (identical query set, top_k, warmup, and
concurrency across providers) and by the harness (equal container CPU/mem caps,
providers run sequentially). Recall@k vs an exact brute-force baseline catches an
ANN config that trades accuracy for speed.

Scales to large synthetic corpora: dataset metadata comes from .meta.json sidecars
(no full parse), the corpus matrix is loaded once and cached, queries come from a
held-out `<dataset>.queries.jsonl` when present, and exact recall is computed with a
chunked top-k so the N x N similarity matrix is never materialized.
"""
from __future__ import annotations

import glob
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_EMBEDDINGS_DIR = os.getenv("YOLORAG_EMBEDDINGS_DIR", "/data/embeddings")
QUERIES_SUFFIX = ".queries.jsonl"
EMBEDDINGS_SUFFIX = ".jsonl"
META_SUFFIX = ".meta.json"

# Cap the number of queries used for the (relatively expensive) exact recall baseline.
RECALL_QUERY_CAP = 500
_RECALL_CHUNK = 50_000

_CORPUS_CACHE: dict[str, "DatasetVectors"] = {}


@dataclass
class DatasetVectors:
    dataset_id: str
    img_ids: list[str]
    matrix: np.ndarray  # (N, dim) float32


@dataclass(frozen=True)
class BenchMetrics:
    provider: str
    dataset_id: str
    num_queries: int
    concurrency: int
    top_k: int
    warmup: int
    completed: int
    errors: int
    qps: float
    wall_ms: float
    latency_ms: dict[str, float]
    recall_at_k: float | None
    error_sample: str | None = None
    search_ef: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "dataset_id": self.dataset_id,
            "num_queries": self.num_queries,
            "concurrency": self.concurrency,
            "top_k": self.top_k,
            "warmup": self.warmup,
            "completed": self.completed,
            "errors": self.errors,
            "qps": self.qps,
            "wall_ms": self.wall_ms,
            "latency_ms": self.latency_ms,
            "recall_at_k": self.recall_at_k,
            "search_ef": self.search_ef,
            "error_sample": self.error_sample,
        }


def _dataset_dirs() -> list[str]:
    dirs = [os.getenv("YOLORAG_EMBEDDINGS_DIR", DEFAULT_EMBEDDINGS_DIR)]
    extra = os.getenv("YOLORAG_BENCH_DATA_DIR")
    if extra:
        dirs.append(extra)
    seen: set[str] = set()
    result: list[str] = []
    for directory in dirs:
        if directory and os.path.isdir(directory) and directory not in seen:
            seen.add(directory)
            result.append(directory)
    return result


def _corpus_path(dataset_id: str) -> str | None:
    for directory in _dataset_dirs():
        path = os.path.join(directory, f"{dataset_id}{EMBEDDINGS_SUFFIX}")
        if os.path.exists(path):
            return path
    return None


def _queries_path(dataset_id: str) -> str | None:
    for directory in _dataset_dirs():
        path = os.path.join(directory, f"{dataset_id}{QUERIES_SUFFIX}")
        if os.path.exists(path):
            return path
    return None


def list_datasets() -> dict[str, dict[str, Any]]:
    """Dataset metadata without parsing vectors: count from the .meta.json sidecar."""
    datasets: dict[str, dict[str, Any]] = {}
    for directory in _dataset_dirs():
        for path in sorted(glob.glob(os.path.join(directory, f"*{EMBEDDINGS_SUFFIX}"))):
            if path.endswith(QUERIES_SUFFIX):
                continue
            dataset_id = Path(path).name[: -len(EMBEDDINGS_SUFFIX)]
            if dataset_id in datasets:
                continue
            count: int | None = None
            meta_path = path[: -len(EMBEDDINGS_SUFFIX)] + META_SUFFIX
            if os.path.exists(meta_path):
                try:
                    count = int(json.load(open(meta_path, encoding="utf-8")).get("count"))
                except Exception:
                    count = None
            if count is None:
                with open(path, "r", encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())
            datasets[dataset_id] = {
                "dataset_id": dataset_id,
                "count": count,
                "has_queries": _queries_path(dataset_id) is not None,
            }
    return datasets


def load_corpus(dataset_id: str) -> DatasetVectors:
    """Load a corpus into a (N, dim) matrix, cached across benchmark runs."""
    if dataset_id in _CORPUS_CACHE:
        return _CORPUS_CACHE[dataset_id]
    path = _corpus_path(dataset_id)
    if path is None:
        raise ValueError(f"No corpus file for dataset {dataset_id!r}.")
    img_ids: list[str] = []
    vectors: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img_ids.append(row["img_id"])
            vectors.append(row["embedding"])
    dataset = DatasetVectors(dataset_id, img_ids, np.asarray(vectors, dtype=np.float32))
    _CORPUS_CACHE[dataset_id] = dataset
    return dataset


def _read_jsonl_vectors(path: str) -> list[tuple[str, list[float]]]:
    rows: list[tuple[str, list[float]]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.append((row["img_id"], row["embedding"]))
    return rows


def resolve_queries(
    dataset_id: str,
    num_queries: int,
    *,
    seed: int,
) -> tuple[list[tuple[str, list[float]]], bool]:
    """Return (queries, held_out). Prefer a held-out query file; else sample the corpus."""
    rng = random.Random(seed)
    queries_path = _queries_path(dataset_id)
    if queries_path is not None:
        pool = _read_jsonl_vectors(queries_path)
        if pool:
            picks = [rng.randrange(len(pool)) for _ in range(num_queries)]
            return [pool[i] for i in picks], True

    corpus = load_corpus(dataset_id)
    n = len(corpus.img_ids)
    picks = [rng.randrange(n) for _ in range(num_queries)]
    return [(corpus.img_ids[i], corpus.matrix[i].tolist()) for i in picks], False


def exact_top_k(
    corpus: DatasetVectors,
    query_matrix: np.ndarray,
    top_k: int,
    *,
    chunk: int = _RECALL_CHUNK,
) -> list[list[str]]:
    """Exact cosine top_k corpus ids per query, chunked over the corpus (no N x N)."""
    num_corpus = corpus.matrix.shape[0]
    num_queries = query_matrix.shape[0]
    k = min(top_k, num_corpus)
    best_sims = np.full((num_queries, k), -np.inf, dtype=np.float32)
    best_idx = np.zeros((num_queries, k), dtype=np.int64)

    for start in range(0, num_corpus, chunk):
        block = corpus.matrix[start : start + chunk]
        sims = query_matrix @ block.T  # (Q, b)
        take = min(k, block.shape[0])
        part_idx = np.argpartition(-sims, take - 1, axis=1)[:, :take]
        part_sims = np.take_along_axis(sims, part_idx, axis=1)
        cand_sims = np.concatenate([best_sims, part_sims], axis=1)
        cand_idx = np.concatenate([best_idx, part_idx + start], axis=1)
        sel = np.argpartition(-cand_sims, k - 1, axis=1)[:, :k]
        best_sims = np.take_along_axis(cand_sims, sel, axis=1)
        best_idx = np.take_along_axis(cand_idx, sel, axis=1)

    order = np.argsort(-best_sims, axis=1)
    final_idx = np.take_along_axis(best_idx, order, axis=1)
    return [[corpus.img_ids[j] for j in final_idx[i]] for i in range(num_queries)]


def build_recall_baseline(
    dataset_id: str,
    queries: list[tuple[str, list[float]]],
    top_k: int,
) -> dict[str, list[str]]:
    """Exact top_k for a bounded subset of queries -> {query_img_id: [corpus_ids]}."""
    corpus = load_corpus(dataset_id)
    subset = queries[: min(len(queries), RECALL_QUERY_CAP)]
    if not subset:
        return {}
    query_matrix = np.asarray([vec for _, vec in subset], dtype=np.float32)
    exact_ids = exact_top_k(corpus, query_matrix, top_k)
    return {subset[i][0]: exact_ids[i] for i in range(len(subset))}


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac)


def run_benchmark(
    store: Any,
    dataset_id: str,
    queries: list[tuple[str, list[float]]],
    *,
    concurrency: int,
    top_k: int,
    warmup: int,
    exact_by_id: dict[str, list[str]] | None = None,
    search_ef: int | None = None,
) -> BenchMetrics:
    provider = getattr(store, "provider_name", "unknown")

    for _, vec in queries[: min(warmup, len(queries))]:
        try:
            store.search(dataset_id, vec, limit=top_k, search_ef=search_ef)
        except Exception:
            pass

    def one(query: tuple[str, list[float]]) -> tuple[float, list[str] | None, str | None]:
        img_id, vec = query
        started = time.perf_counter()
        try:
            hits = store.search(dataset_id, vec, limit=top_k, search_ef=search_ef)
            latency = (time.perf_counter() - started) * 1000.0
            return latency, [h.img_id for h in hits], None
        except Exception as exc:  # noqa: BLE001 - report, don't abort the run
            latency = (time.perf_counter() - started) * 1000.0
            return latency, None, f"{type(exc).__name__}: {exc}"

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(one, queries))
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    latencies: list[float] = []
    recalls: list[float] = []
    errors = 0
    error_sample: str | None = None
    for (latency, got_ids, err), (img_id, _) in zip(outcomes, queries):
        if err is not None:
            errors += 1
            error_sample = error_sample or err
            continue
        latencies.append(latency)
        if exact_by_id is not None and got_ids is not None:
            expected = exact_by_id.get(img_id)
            if expected:
                overlap = len(set(got_ids) & set(expected))
                recalls.append(overlap / len(expected))

    latencies.sort()
    completed = len(latencies)
    qps = completed / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
    latency_ms = {
        "p50": _percentile(latencies, 0.50),
        "p90": _percentile(latencies, 0.90),
        "p95": _percentile(latencies, 0.95),
        "p99": _percentile(latencies, 0.99),
        "mean": (sum(latencies) / completed) if completed else 0.0,
        "min": latencies[0] if latencies else 0.0,
        "max": latencies[-1] if latencies else 0.0,
    }
    recall_at_k = (sum(recalls) / len(recalls)) if recalls else None

    return BenchMetrics(
        provider=provider,
        dataset_id=dataset_id,
        num_queries=len(queries),
        concurrency=concurrency,
        top_k=top_k,
        warmup=warmup,
        completed=completed,
        errors=errors,
        qps=qps,
        wall_ms=wall_ms,
        latency_ms=latency_ms,
        recall_at_k=recall_at_k,
        error_sample=error_sample,
        search_ef=search_ef,
    )


_EF_LADDER = (16, 32, 64, 128, 256, 512, 1024)


def _measure_recall(
    store: Any,
    dataset_id: str,
    queries: list[tuple[str, list[float]]],
    exact_by_id: dict[str, list[str]],
    top_k: int,
    search_ef: int,
) -> float:
    recalls: list[float] = []
    for img_id, vec in queries:
        expected = exact_by_id.get(img_id)
        if not expected:
            continue
        try:
            hits = store.search(dataset_id, vec, limit=top_k, search_ef=search_ef)
        except Exception:
            continue
        got = {h.img_id for h in hits}
        recalls.append(len(got & set(expected)) / len(expected))
    return (sum(recalls) / len(recalls)) if recalls else 0.0


def tune_search_ef(
    store: Any,
    dataset_id: str,
    queries: list[tuple[str, list[float]]],
    exact_by_id: dict[str, list[str]],
    *,
    top_k: int,
    target_recall: float,
) -> tuple[int, float]:
    """Smallest search-effort (ef) reaching target_recall on the recall subset.

    Returns (search_ef, achieved_recall). If no ef reaches the target, returns the ef
    with the best observed recall (recall is monotone-ish in ef, so that's the max ef).
    """
    ladder = sorted({top_k, *(ef for ef in _EF_LADDER if ef >= top_k)})
    best_ef, best_recall = ladder[-1], 0.0
    for ef in ladder:
        recall = _measure_recall(store, dataset_id, queries, exact_by_id, top_k, ef)
        if recall >= target_recall:
            return ef, recall
        if recall > best_recall:
            best_ef, best_recall = ef, recall
    return best_ef, best_recall
