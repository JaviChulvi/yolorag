from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from yolorag.core.conversation import ConversationLogger
from yolorag.core.routing import SimpleRoutePlanner
from yolorag.core.tracing import OrchestrationTrace
from yolorag.core.transcripts import (
    schedule_assistant_message_write,
    schedule_transcript_write,
    schedule_user_message_write,
)
from yolorag.providers.base import LLMProvider, LLMRequest, Message, ResponseMode
from yolorag.retrieval.base import RetrievalResult, Retriever
from yolorag.usage.models import CostBreakdown, TokenUsage


MAIN_SYSTEM_PROMPT = """\
You are YoloRAG, an assistant for Ultralytics YOLO documentation and product questions.

Answer the user's question directly and practically. Prefer the provided documentation context
when it is available, and do not mention retrieval, reranking, hidden context, internal modes,
or implementation details of this chat system.

When the docs context supports the answer, ground your response in it and include the most
relevant docs links naturally. If the context is incomplete or does not answer the question,
say so plainly before giving a cautious next step. Prefer current Ultralytics docs over older
version-specific material, and clearly label YOLOv5-specific guidance when that is all the
context supports.

For technical answers, give concrete commands or Python examples when useful. Keep the answer
concise by default, but include enough detail for the user to act.
"""

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorResult:
    answer: str
    trace: OrchestrationTrace
    retrieved_context: list[RetrievalResult]


class RAGOrchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        retriever: Retriever | None = None,
        conversation_logger: ConversationLogger | None = None,
        route_planner: SimpleRoutePlanner | None = None,
        retrieval_top_k: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.retriever = retriever
        self.conversation_logger = conversation_logger
        self.route_planner = route_planner or SimpleRoutePlanner()
        self.retrieval_top_k = retrieval_top_k

    async def answer(
        self,
        user_message: str,
        conversation_id: str = "default",
        mode: ResponseMode = "fast",
        conversation_messages: list[Message] | None = None,
        raw_user_message: str | None = None,
        request_id: str | None = None,
        user_message_index: int | None = None,
        measure_ttft: bool = False,
        persist: bool = True,
    ) -> OrchestratorResult:
        total_started = time.perf_counter()
        retrieval_query = raw_user_message or user_message
        plan = self.route_planner.plan(user_message=retrieval_query, requested_mode=mode)
        if persist:
            self._schedule_user_message_write(
                conversation_id=conversation_id,
                request_id=request_id,
                raw_user_message=raw_user_message or user_message,
                user_message_index=user_message_index,
            )
        retrieval_started = time.perf_counter()
        retrieved_context, retrieval_error = await self._retrieve_if_needed(
            user_message=retrieval_query,
            existing_document_ids=set(),
            should_retrieve=plan.should_retrieve,
            top_k=self.retrieval_top_k or plan.top_k,
            min_relevance_score=plan.min_relevance_score,
        )
        retrieval_ms = _elapsed_ms(retrieval_started)

        messages = self._request_messages(
            user_message=user_message,
            conversation_messages=conversation_messages,
            retrieved_context=retrieved_context,
        )

        pre_llm_ms = _elapsed_ms(total_started)
        response = await self.provider.complete(
            LLMRequest(
                messages=messages,
                model=self.model,
                mode=plan.mode,
                max_tokens=512 if plan.mode == "fast" else 1600,
                metadata={"stream_for_timing": measure_ttft},
            )
        )

        answer = self._append_sources(response.content, retrieved_context)
        retrieved_ids = [item.document.id for item in retrieved_context]
        retrieval_trace = retrieved_context[0].trace if retrieved_context else None
        llm_ms = response.latency_ms
        llm_ttft_ms = response.first_token_latency_ms
        ttft_ms = pre_llm_ms + llm_ttft_ms if llm_ttft_ms else 0
        total_ms = max(_elapsed_ms(total_started), retrieval_ms + llm_ms)
        orchestration_overhead_ms = max(total_ms - retrieval_ms - llm_ms, 0)

        if persist:
            self._schedule_assistant_message_write(
                conversation_id=conversation_id,
                request_id=request_id,
                assistant_message=answer,
                user_message_index=user_message_index,
                retrieved_document_ids=retrieved_ids,
                provider=response.provider,
                model=response.model,
            )

        trace = OrchestrationTrace(
            provider=response.provider,
            model=response.model,
            mode=plan.mode,
            retrieval_used=bool(retrieved_context),
            retrieved_document_ids=retrieved_ids,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            estimated_cost_usd=response.cost.total_usd,
            pricing_source=response.cost.pricing_source,
            latency_ms=response.latency_ms,
            route_reason=plan.reason,
            total_ms=total_ms,
            retrieval_ms=retrieval_ms,
            query_embedding_ms=retrieval_trace.query_embedding_ms if retrieval_trace else 0,
            vector_search_ms=retrieval_trace.vector_search_ms if retrieval_trace else 0,
            rerank_ms=retrieval_trace.rerank_ms if retrieval_trace else 0,
            llm_ms=llm_ms,
            ttft_ms=ttft_ms,
            llm_ttft_ms=llm_ttft_ms,
            orchestration_overhead_ms=orchestration_overhead_ms,
            retrieval_candidate_count=retrieval_trace.candidate_count if retrieval_trace else 0,
            retrieval_returned_count=retrieval_trace.returned_count if retrieval_trace else 0,
            retrieval_reranked=retrieval_trace.reranked if retrieval_trace else False,
            retrieval_error=retrieval_error,
        )

        return OrchestratorResult(
            answer=answer,
            trace=trace,
            retrieved_context=retrieved_context,
        )

    async def stream_answer(
        self,
        user_message: str,
        conversation_id: str = "default",
        mode: ResponseMode = "fast",
        conversation_messages: list[Message] | None = None,
        raw_user_message: str | None = None,
        request_id: str | None = None,
        user_message_index: int | None = None,
        include_metrics: bool = False,
        persist: bool = True,
    ) -> AsyncIterator[str | dict[str, Any]]:
        total_started = time.perf_counter()
        retrieval_query = raw_user_message or user_message
        plan = self.route_planner.plan(user_message=retrieval_query, requested_mode=mode)
        if persist:
            self._schedule_user_message_write(
                conversation_id=conversation_id,
                request_id=request_id,
                raw_user_message=raw_user_message or user_message,
                user_message_index=user_message_index,
            )
        retrieval_started = time.perf_counter()
        retrieved_context, _retrieval_error = await self._retrieve_if_needed(
            user_message=retrieval_query,
            existing_document_ids=set(),
            should_retrieve=plan.should_retrieve,
            top_k=self.retrieval_top_k or plan.top_k,
            min_relevance_score=plan.min_relevance_score,
        )
        retrieval_ms = _elapsed_ms(retrieval_started)
        retrieval_trace = retrieved_context[0].trace if retrieved_context else None

        messages = self._request_messages(
            user_message=user_message,
            conversation_messages=conversation_messages,
            retrieved_context=retrieved_context,
        )

        chunks = []
        usage = TokenUsage()
        cost = CostBreakdown()
        llm_started = time.perf_counter()
        llm_ttft_ms = 0
        ttft_ms = 0
        async for event in self.provider.stream_complete(
            LLMRequest(
                messages=messages,
                model=self.model,
                mode=plan.mode,
                max_tokens=512 if plan.mode == "fast" else 1600,
            )
        ):
            if event.usage is not None:
                usage = event.usage
            if event.cost is not None:
                cost = event.cost
            if not event.content:
                continue
            if not llm_ttft_ms:
                llm_ttft_ms = _elapsed_ms(llm_started)
                ttft_ms = _elapsed_ms(total_started)
            chunks.append(event.content)
            yield event.content

        llm_ms = _elapsed_ms(llm_started)
        answer = "".join(chunks)
        source_suffix = self._source_suffix(answer, retrieved_context)
        if source_suffix:
            if not ttft_ms:
                ttft_ms = _elapsed_ms(total_started)
            yield source_suffix
        final_answer = f"{answer.rstrip()}{source_suffix}" if source_suffix else answer
        retrieved_ids = [item.document.id for item in retrieved_context]
        if persist:
            self._schedule_assistant_message_write(
                conversation_id=conversation_id,
                request_id=request_id,
                assistant_message=final_answer,
                user_message_index=user_message_index,
                retrieved_document_ids=retrieved_ids,
                provider=self.provider.provider_name,
                model=self.model,
            )

        total_ms = max(_elapsed_ms(total_started), retrieval_ms + llm_ms)
        trace = OrchestrationTrace(
            provider=self.provider.provider_name,
            model=self.model,
            mode=plan.mode,
            retrieval_used=bool(retrieved_context),
            retrieved_document_ids=retrieved_ids,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_usd=cost.total_usd,
            pricing_source=cost.pricing_source,
            latency_ms=llm_ms,
            route_reason=plan.reason,
            total_ms=total_ms,
            retrieval_ms=retrieval_ms,
            query_embedding_ms=retrieval_trace.query_embedding_ms if retrieval_trace else 0,
            vector_search_ms=retrieval_trace.vector_search_ms if retrieval_trace else 0,
            rerank_ms=retrieval_trace.rerank_ms if retrieval_trace else 0,
            llm_ms=llm_ms,
            ttft_ms=ttft_ms,
            llm_ttft_ms=llm_ttft_ms,
            orchestration_overhead_ms=max(total_ms - retrieval_ms - llm_ms, 0),
            retrieval_candidate_count=retrieval_trace.candidate_count if retrieval_trace else 0,
            retrieval_returned_count=retrieval_trace.returned_count if retrieval_trace else 0,
            retrieval_reranked=retrieval_trace.reranked if retrieval_trace else False,
            retrieval_error=_retrieval_error,
        )
        if include_metrics:
            yield {"type": "metrics", "metrics": _metrics_payload(trace)}

    async def _retrieve_if_needed(
        self,
        user_message: str,
        existing_document_ids: set[str],
        should_retrieve: bool,
        top_k: int,
        min_relevance_score: float | None,
    ) -> tuple[list[RetrievalResult], str | None]:
        if not should_retrieve or self.retriever is None:
            return [], None

        try:
            results = await self.retriever.retrieve(user_message, top_k=top_k)
        except Exception as exc:
            retrieval_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Retrieval failed; continuing without retrieved context: %s",
                retrieval_error,
                exc_info=True,
            )
            return [], retrieval_error
        filtered_results = [
            result
            for result in results
            if result.document.id not in existing_document_ids
            and _passes_relevance_threshold(result, min_relevance_score)
        ]
        return filtered_results, None

    def _request_messages(
        self,
        *,
        user_message: str,
        conversation_messages: list[Message] | None,
        retrieved_context: list[RetrievalResult],
    ) -> list[Message]:
        body = (
            [dict(message) for message in conversation_messages]
            if conversation_messages
            else [{"role": "user", "content": user_message}]
        )
        context_message = (
            {"role": "system", "content": self._format_context(retrieved_context)}
            if retrieved_context
            else None
        )
        if context_message is not None:
            last_user_index = _last_user_message_index(body)
            if last_user_index is None:
                body.append(context_message)
            else:
                body.insert(last_user_index, context_message)

        return [
            {
                "role": "system",
                "content": MAIN_SYSTEM_PROMPT,
            },
            *body,
        ]

    def _schedule_user_message_write(
        self,
        *,
        conversation_id: str,
        request_id: str | None,
        raw_user_message: str,
        user_message_index: int | None,
    ) -> None:
        schedule_user_message_write(
            self.conversation_logger,
            conversation_id=conversation_id,
            request_id=request_id,
            raw_user_message=raw_user_message,
            user_message_index=user_message_index,
        )

    def _schedule_assistant_message_write(
        self,
        *,
        conversation_id: str,
        request_id: str | None,
        assistant_message: str,
        user_message_index: int | None,
        retrieved_document_ids: list[str],
        provider: str,
        model: str,
    ) -> None:
        schedule_assistant_message_write(
            self.conversation_logger,
            conversation_id=conversation_id,
            request_id=request_id,
            assistant_message=assistant_message,
            user_message_index=user_message_index,
            retrieved_document_ids=retrieved_document_ids,
            provider=provider,
            model=model,
        )

    def _schedule_transcript_write(self, messages: list) -> None:
        schedule_transcript_write(self.conversation_logger, messages)

    def _format_context(self, context: list[RetrievalResult]) -> str:
        blocks = []
        for item in context:
            metadata = item.document.metadata
            blocks.append(
                "\n".join(
                    [
                        f"Document ID: {item.document.id}",
                        f"Title: {item.document.title}",
                        f"URL: {metadata.get('url', '')}",
                        f"Source path: {metadata.get('source_path', '')}",
                        f"Selection reason: {item.reason}",
                        item.document.content,
                    ]
                )
            )
        return "Relevant retrieved context:\n\n" + "\n\n---\n\n".join(blocks)

    def _append_sources(self, answer: str, context: list[RetrievalResult]) -> str:
        source_suffix = self._source_suffix(answer, context)
        if not source_suffix:
            return answer
        return f"{answer.rstrip()}{source_suffix}"

    def _source_suffix(self, answer: str, context: list[RetrievalResult]) -> str:
        if not context or "https://docs.ultralytics.com/" in answer:
            return ""

        sources = []
        seen_urls = set()
        for item in context:
            metadata = item.document.metadata
            url = metadata.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append((item.document.title, url, metadata.get("source_path", "")))

        if not sources:
            return ""

        sources.sort(key=lambda source: source[2].startswith("en/yolov5/"))
        source_lines = [f"- [{title}]({url})" for title, url, _ in sources[:3]]
        return "\n\nSources:\n" + "\n".join(source_lines)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _passes_relevance_threshold(
    result: RetrievalResult,
    min_relevance_score: float | None,
) -> bool:
    if min_relevance_score is None:
        return True
    return result.score >= min_relevance_score


def _metrics_payload(trace: OrchestrationTrace) -> dict[str, Any]:
    return {
        "provider": trace.provider,
        "model": trace.model,
        "mode": trace.mode,
        "timings_ms": {
            "total": trace.total_ms,
            "retrieval": trace.retrieval_ms,
            "query_embedding": trace.query_embedding_ms,
            "vector_search": trace.vector_search_ms,
            "rerank": trace.rerank_ms,
            "llm": trace.llm_ms,
            "ttft": trace.ttft_ms,
            "llm_ttft": trace.llm_ttft_ms,
            "orchestration_overhead": trace.orchestration_overhead_ms,
            "wall": trace.total_ms,
        },
        "retrieval": {
            "used": trace.retrieval_used,
            "reranked": trace.retrieval_reranked,
            "candidate_count": trace.retrieval_candidate_count,
            "returned_count": trace.retrieval_returned_count,
            "document_ids": trace.retrieved_document_ids,
            "error": trace.retrieval_error,
        },
        "usage": {
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "estimated_cost_usd": float(trace.estimated_cost_usd),
            "pricing_source": trace.pricing_source,
        },
        "route_reason": trace.route_reason,
    }


def _last_user_message_index(messages: list[Message]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None
