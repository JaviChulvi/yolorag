#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from yolorag.knowledge.stores.mongodb import MongoKnowledgeStore, MongoKnowledgeStoreConfig
from yolorag.retrieval.mongodb import MongoReranker, MongoVectorRetriever
from yolorag.runtime import build_runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = REPO_ROOT / "evals" / "profile_questions.json"
DEFAULT_RUNS_DIR = REPO_ROOT / "evals" / "runs"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = _parse_args()

    cases = _load_cases(args.questions, limit=args.limit)
    if not cases:
        raise SystemExit("No eval cases found.")

    if args.retrieval_only:
        report = asyncio.run(_run_retrieval_only(args, cases))
    else:
        report = asyncio.run(_run_full_rag(args, cases))

    output_path = args.output or _default_output_path(report["run"]["mode"], report["run"]["id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")

    _print_summary(report)
    print(f"\nSaved report: {output_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run profile-question evals and break down RAG latency by component."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--provider", choices=["openai", "deepseek"])
    parser.add_argument("--mode", choices=["fast", "deep"], default=os.getenv("YOLORAG_API_MODE", "fast"))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("YOLORAG_CHAT_VECTOR_TOP_K", "8")))
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        help=(
            "Number of vector-search candidates to send to the reranker. "
            "Defaults to YOLORAG_RERANK_CANDIDATE_LIMIT or max(40, top_k * 8)."
        ),
    )
    parser.add_argument("--limit", type=int, help="Run only the first N eval cases.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Measure Mongo vector search and reranking without calling the LLM provider.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable Mongo reranking for retrieval-only experiments.",
    )
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="Store full model answers in the JSON report instead of short previews.",
    )
    return parser.parse_args()


def _load_cases(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if limit is not None:
        cases = cases[:limit]
    return cases


async def _run_full_rag(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    if args.rerank_candidates is not None:
        os.environ["YOLORAG_RERANK_CANDIDATE_LIMIT"] = str(args.rerank_candidates)
    runtime = build_runtime(provider_name=args.provider, mode=args.mode)
    runtime.orchestrator.retrieval_top_k = args.top_k

    rows = []
    run_id = _run_id()
    for index, case in enumerate(cases, start=1):
        result = await runtime.answer(
            user_message=case["question"],
            conversation_id=f"{run_id}-{case['id']}",
        )
        source_paths = _source_paths(result.retrieved_context)
        row = {
            "id": case["id"],
            "question": case["question"],
            "tags": case.get("tags", []),
            "expected_source_paths": case.get("expected_source_paths", []),
            "expected_source_hit": _expected_source_hit(case, source_paths),
            "source_paths": source_paths,
            "source_urls": _source_urls(result.retrieved_context),
            "answer_chars": len(result.answer),
            "answer": result.answer if args.include_answers else None,
            "answer_preview": None if args.include_answers else result.answer[:500],
            "trace": asdict(result.trace),
        }
        rows.append(row)
        print(
            f"{index:02d}/{len(cases)} {case['id']} "
            f"total={result.trace.total_ms}ms retrieval={result.trace.retrieval_ms}ms "
            f"mongo={result.trace.vector_search_ms}ms rerank={result.trace.rerank_ms}ms "
            f"llm={result.trace.llm_ms}ms"
        )

    return _report(
        mode="full_rag",
        args=args,
        rows=rows,
        run_id=run_id,
    )


async def _run_retrieval_only(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    store = MongoKnowledgeStore(MongoKnowledgeStoreConfig.from_env())
    retriever = MongoVectorRetriever(
        store=store,
        reranker=None if args.no_rerank else MongoReranker.from_env(),
        candidate_limit=args.rerank_candidates,
    )

    rows = []
    run_id = _run_id()
    for index, case in enumerate(cases, start=1):
        results = await retriever.retrieve(case["question"], top_k=args.top_k)
        trace = results[0].trace if results else None
        source_paths = _source_paths(results)
        row = {
            "id": case["id"],
            "question": case["question"],
            "tags": case.get("tags", []),
            "expected_source_paths": case.get("expected_source_paths", []),
            "expected_source_hit": _expected_source_hit(case, source_paths),
            "source_paths": source_paths,
            "source_urls": _source_urls(results),
            "answer_chars": 0,
            "answer": None,
            "answer_preview": None,
            "trace": _retrieval_only_trace(trace),
        }
        rows.append(row)
        timings = row["trace"]
        print(
            f"{index:02d}/{len(cases)} {case['id']} "
            f"total={timings['total_ms']}ms mongo={timings['vector_search_ms']}ms "
            f"rerank={timings['rerank_ms']}ms candidates={timings['retrieval_candidate_count']}"
        )

    return _report(
        mode="retrieval_only",
        args=args,
        rows=rows,
        run_id=run_id,
    )


def _retrieval_only_trace(trace: Any) -> dict[str, Any]:
    if trace is None:
        return {
            "total_ms": 0,
            "retrieval_ms": 0,
            "vector_search_ms": 0,
            "rerank_ms": 0,
            "llm_ms": 0,
            "orchestration_overhead_ms": 0,
            "retrieval_candidate_count": 0,
            "retrieval_returned_count": 0,
            "retrieval_reranked": False,
        }

    return {
        "total_ms": trace.total_ms,
        "retrieval_ms": trace.total_ms,
        "vector_search_ms": trace.vector_search_ms,
        "rerank_ms": trace.rerank_ms,
        "llm_ms": 0,
        "orchestration_overhead_ms": 0,
        "retrieval_candidate_count": trace.candidate_count,
        "retrieval_returned_count": trace.returned_count,
        "retrieval_reranked": trace.reranked,
    }


def _report(
    *,
    mode: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    return {
        "run": {
            "id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "mode": mode,
            "provider": args.provider or os.getenv("YOLORAG_API_PROVIDER", "openai"),
            "response_mode": args.mode,
            "top_k": args.top_k,
            "rerank_candidate_limit": args.rerank_candidates
            or os.getenv("YOLORAG_RERANK_CANDIDATE_LIMIT"),
            "question_count": len(rows),
            "rerank_enabled": not args.no_rerank,
        },
        "summary": _summary(rows),
        "results": rows,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timing_fields = [
        "total_ms",
        "retrieval_ms",
        "vector_search_ms",
        "rerank_ms",
        "llm_ms",
        "orchestration_overhead_ms",
    ]
    timings = {
        field: _metric([row["trace"].get(field, 0) for row in rows])
        for field in timing_fields
    }
    total_ms_sum = sum(row["trace"].get("total_ms", 0) for row in rows)
    component_shares = {}
    for field in timing_fields[1:]:
        component_shares[field] = (
            sum(row["trace"].get(field, 0) for row in rows) / total_ms_sum
            if total_ms_sum
            else 0
        )

    return {
        "timings": timings,
        "component_shares": component_shares,
        "expected_source_hit_rate": (
            sum(1 for row in rows if row["expected_source_hit"]) / len(rows)
            if rows
            else 0
        ),
        "avg_retrieval_candidates": _avg(
            [row["trace"].get("retrieval_candidate_count", 0) for row in rows]
        ),
        "avg_retrieval_returned": _avg(
            [row["trace"].get("retrieval_returned_count", 0) for row in rows]
        ),
        "total_estimated_cost_usd": str(
            sum(Decimal(str(row["trace"].get("estimated_cost_usd", "0"))) for row in rows)
        ),
    }


def _metric(values: list[int]) -> dict[str, float | int]:
    clean = sorted(values)
    return {
        "avg": round(_avg(clean), 2),
        "p50": _percentile(clean, 50),
        "p95": _percentile(clean, 95),
        "max": max(clean) if clean else 0,
    }


def _avg(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0


def _percentile(values: list[int], percentile: int) -> int:
    if not values:
        return 0
    index = round((len(values) - 1) * percentile / 100)
    return values[index]


def _print_summary(report: dict[str, Any]) -> None:
    print("\nLatency Summary")
    print("component | avg ms | p50 | p95 | max | share")
    print("--- | ---: | ---: | ---: | ---: | ---:")
    shares = report["summary"]["component_shares"]
    for field, label in [
        ("total_ms", "total"),
        ("retrieval_ms", "retrieval total"),
        ("vector_search_ms", "mongodb vector"),
        ("rerank_ms", "rerank"),
        ("llm_ms", "llm"),
        ("orchestration_overhead_ms", "app overhead"),
    ]:
        metric = report["summary"]["timings"][field]
        share = "" if field == "total_ms" else f"{shares.get(field, 0) * 100:.1f}%"
        print(
            f"{label} | {metric['avg']} | {metric['p50']} | "
            f"{metric['p95']} | {metric['max']} | {share}"
        )

    hit_rate = report["summary"]["expected_source_hit_rate"] * 100
    print(f"\nExpected source hit rate: {hit_rate:.1f}%")
    print(f"Avg candidates: {report['summary']['avg_retrieval_candidates']:.1f}")
    print(f"Avg returned: {report['summary']['avg_retrieval_returned']:.1f}")
    print(f"Estimated cost: ${report['summary']['total_estimated_cost_usd']}")


def _source_paths(results: list[Any]) -> list[str]:
    return [result.document.metadata.get("source_path", "") for result in results]


def _source_urls(results: list[Any]) -> list[str]:
    urls = []
    seen = set()
    for result in results:
        url = result.document.metadata.get("url", "")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _expected_source_hit(case: dict[str, Any], source_paths: list[str]) -> bool:
    expected = set(case.get("expected_source_paths", []))
    return bool(expected.intersection(source_paths))


def _default_output_path(mode: str, run_id: str) -> Path:
    return DEFAULT_RUNS_DIR / f"{run_id}-{mode}-latency.json"


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
