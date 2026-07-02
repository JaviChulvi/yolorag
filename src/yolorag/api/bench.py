from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from yolorag.bench.runner import (
    DEFAULT_EMBEDDINGS_DIR,
    RECALL_QUERY_CAP,
    build_recall_baseline,
    list_datasets,
    resolve_queries,
    run_benchmark,
    tune_search_ef,
)
from yolorag.knowledge.stores.image_factory import IMAGE_STORE_PROVIDERS, build_image_store


router = APIRouter()


class BenchmarkRequest(BaseModel):
    providers: list[str] = Field(default_factory=lambda: list(IMAGE_STORE_PROVIDERS))
    dataset_id: str | None = None
    num_queries: int = Field(default=1000, ge=1, le=100_000)
    concurrency: int = Field(default=8, ge=1, le=256)
    top_k: int = Field(default=10, ge=1, le=1000)
    warmup: int = Field(default=50, ge=0, le=10_000)
    measure_recall: bool = True
    # Iso-recall: auto-tune each engine's search effort to hit this recall, then compare.
    # Set to null to use engine defaults (or search_ef for a fixed manual override).
    recall_target: float | None = Field(default=0.95, ge=0.0, le=1.0)
    search_ef: int | None = Field(default=None, ge=1)
    seed: int = 1234


def _resource_caps() -> dict[str, str]:
    return {
        "cpus": os.getenv("YOLORAG_BENCH_CPUS", "2"),
        "memory": os.getenv("YOLORAG_BENCH_MEM", "4g"),
        "note": "Per-vendor container caps (Milvus query container; etcd+MinIO are fixed overhead).",
    }


@router.get("/benchmark/meta")
def benchmark_meta() -> dict:
    datasets = list_datasets()
    return {
        "providers": list(IMAGE_STORE_PROVIDERS),
        "datasets": sorted(
            (
                {
                    "dataset_id": d["dataset_id"],
                    "count": d["count"],
                    "has_queries": d["has_queries"],
                }
                for d in datasets.values()
            ),
            key=lambda item: item["dataset_id"],
        ),
        "embeddings_dir": DEFAULT_EMBEDDINGS_DIR,
        "resource_caps": _resource_caps(),
        "defaults": {"num_queries": 1000, "concurrency": 8, "top_k": 10, "warmup": 50},
    }


@router.post("/benchmark")
def run_benchmark_endpoint(request: BenchmarkRequest) -> dict:
    datasets = list_datasets()
    if not datasets:
        raise HTTPException(
            status_code=400,
            detail=f"No embeddings found in {DEFAULT_EMBEDDINGS_DIR}. Mount embeddings/ into the backend.",
        )

    dataset_id = request.dataset_id
    if dataset_id is None:
        # Default to the largest dataset (most query pressure).
        dataset_id = max(datasets.values(), key=lambda d: d["count"])["dataset_id"]
    if dataset_id not in datasets:
        raise HTTPException(status_code=400, detail=f"Unknown dataset_id {dataset_id!r}.")

    unknown = [p for p in request.providers if p not in IMAGE_STORE_PROVIDERS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown providers: {unknown}.")

    queries, held_out = resolve_queries(dataset_id, request.num_queries, seed=request.seed)
    exact_by_id = build_recall_baseline(dataset_id, queries, request.top_k) if request.measure_recall else None
    recall_subset = queries[:RECALL_QUERY_CAP]
    iso_recall = bool(request.recall_target and exact_by_id)

    results: list[dict] = []
    # Sequential on purpose: providers must not contend for host CPU while measured.
    for provider in request.providers:
        try:
            store = build_image_store(provider)
            store.ensure_schema()
            if iso_recall:
                # Tune this engine's search effort to the shared recall target.
                search_ef, _ = tune_search_ef(
                    store,
                    dataset_id,
                    recall_subset,
                    exact_by_id,
                    top_k=request.top_k,
                    target_recall=request.recall_target,
                )
            else:
                search_ef = request.search_ef
            metrics = run_benchmark(
                store,
                dataset_id,
                queries,
                concurrency=request.concurrency,
                top_k=request.top_k,
                warmup=request.warmup,
                exact_by_id=exact_by_id,
                search_ef=search_ef,
            )
            results.append(metrics.to_dict())
        except Exception as exc:  # noqa: BLE001 - one bad vendor shouldn't kill the run
            results.append(
                {
                    "provider": provider,
                    "dataset_id": dataset_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "config": {
            "dataset_id": dataset_id,
            "dataset_count": datasets[dataset_id]["count"],
            "num_queries": request.num_queries,
            "concurrency": request.concurrency,
            "top_k": request.top_k,
            "warmup": request.warmup,
            "measure_recall": request.measure_recall,
            "held_out_queries": held_out,
            "recall_query_sample": len(recall_subset) if request.measure_recall else 0,
            "recall_target": request.recall_target if iso_recall else None,
            "iso_recall": iso_recall,
            "transport": "grpc-parity (qdrant/milvus grpc, pgvector in-process, mongo via mongot)",
            "resource_caps": _resource_caps(),
        },
        "results": results,
    }
