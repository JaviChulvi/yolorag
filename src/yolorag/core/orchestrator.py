from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from yolorag.core.conversation import ConversationLogger
from yolorag.core.routing import SimpleRoutePlanner
from yolorag.core.tracing import OrchestrationTrace
from yolorag.core.transcripts import (
    schedule_assistant_message_write,
    schedule_transcript_write,
    schedule_user_message_write,
)
from yolorag.providers.base import LLMProvider, LLMRequest, LLMResponse, Message, ResponseMode
from yolorag.retrieval.base import RetrievalResult, Retriever
from yolorag.tools.base import ToolCallRequest, ToolCallResult
from yolorag.tools.router import ToolRouter
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

FAST_TOOL_SELECTION_PROMPT = """\
You are deciding whether a low-latency chat answer needs external tools before the
final answer is streamed to the user.

Use at most one tool, and only when it materially improves correctness. Prefer
docs_search for Ultralytics documentation, product, API, install, training, export,
deployment, or troubleshooting questions. Use repository or GitHub MCP tools only
when the question needs repository, issue, pull request, or source evidence that
the docs are unlikely to contain.

Keep tool queries short and specific. Do not call tools for greetings, thanks,
small talk, generic reasoning, or when the answer is already clear from the
conversation. If no tool is needed, reply exactly: NO_TOOL
"""

FAST_TOOL_TIMEOUT_SECONDS = 4.0
FAST_DOCS_TOP_K = 3
FAST_MAX_TOOL_CALLS = 1
FAST_TOOL_SELECTION_MAX_TOKENS = 96
FAST_ANSWER_MAX_TOKENS = 512

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorResult:
    answer: str
    trace: OrchestrationTrace
    retrieved_context: list[RetrievalResult]


@dataclass(frozen=True)
class FastToolPass:
    messages: list[Message]
    planning_response: LLMResponse | None
    tool_call_count: int
    tool_ms: int
    tool_error: str | None = None
    retrieval_used: bool = False
    retrieved_document_ids: list[str] = field(default_factory=list)
    retrieval_ms: int = 0
    query_embedding_ms: int = 0
    vector_search_ms: int = 0
    rerank_ms: int = 0
    retrieval_candidate_count: int = 0
    retrieval_returned_count: int = 0
    retrieval_reranked: bool = False


class RAGOrchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        retriever: Retriever | None = None,
        conversation_logger: ConversationLogger | None = None,
        route_planner: SimpleRoutePlanner | None = None,
        retrieval_top_k: int | None = None,
        tool_router: ToolRouter | None = None,
        fast_max_tool_calls: int = FAST_MAX_TOOL_CALLS,
        fast_tool_timeout_seconds: float = FAST_TOOL_TIMEOUT_SECONDS,
    ) -> None:
        self.provider = provider
        self.model = model
        self.retriever = retriever
        self.conversation_logger = conversation_logger
        self.route_planner = route_planner or SimpleRoutePlanner()
        self.retrieval_top_k = retrieval_top_k
        self.tool_router = tool_router
        self.fast_max_tool_calls = fast_max_tool_calls
        self.fast_tool_timeout_seconds = fast_tool_timeout_seconds

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
        if mode == "fast" and self.tool_router is not None:
            return await self._answer_with_fast_tools(
                user_message=user_message,
                conversation_id=conversation_id,
                conversation_messages=conversation_messages,
                raw_user_message=raw_user_message,
                request_id=request_id,
                user_message_index=user_message_index,
                measure_ttft=measure_ttft,
                persist=persist,
            )

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
        if mode == "fast" and self.tool_router is not None:
            async for event in self._stream_answer_with_fast_tools(
                user_message=user_message,
                conversation_id=conversation_id,
                conversation_messages=conversation_messages,
                raw_user_message=raw_user_message,
                request_id=request_id,
                user_message_index=user_message_index,
                include_metrics=include_metrics,
                persist=persist,
            ):
                yield event
            return

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

    async def _answer_with_fast_tools(
        self,
        *,
        user_message: str,
        conversation_id: str,
        conversation_messages: list[Message] | None,
        raw_user_message: str | None,
        request_id: str | None,
        user_message_index: int | None,
        measure_ttft: bool,
        persist: bool,
    ) -> OrchestratorResult:
        total_started = time.perf_counter()
        if persist:
            self._schedule_user_message_write(
                conversation_id=conversation_id,
                request_id=request_id,
                raw_user_message=raw_user_message or user_message,
                user_message_index=user_message_index,
            )

        base_messages = self._request_messages(
            user_message=user_message,
            conversation_messages=conversation_messages,
            retrieved_context=[],
        )
        planning_messages = self._request_messages(
            user_message=raw_user_message or user_message,
            conversation_messages=_with_latest_user_message(
                conversation_messages,
                raw_user_message,
            ),
            retrieved_context=[],
        )
        fast_tool_pass = await self._fast_tool_pass(
            base_messages,
            planning_messages=planning_messages,
        )
        pre_llm_ms = _elapsed_ms(total_started)
        response = await self.provider.complete(
            LLMRequest(
                messages=fast_tool_pass.messages,
                model=self.model,
                mode="fast",
                max_tokens=FAST_ANSWER_MAX_TOKENS,
                metadata={"stream_for_timing": measure_ttft},
            )
        )

        answer = response.content
        usage = _combine_usage(
            fast_tool_pass.planning_response.usage if fast_tool_pass.planning_response else TokenUsage(),
            response.usage,
        )
        cost = _combine_cost(
            fast_tool_pass.planning_response.cost if fast_tool_pass.planning_response else CostBreakdown(),
            response.cost,
        )
        llm_ms = _fast_llm_ms(fast_tool_pass, response.latency_ms)
        llm_ttft_ms = response.first_token_latency_ms
        ttft_ms = pre_llm_ms + llm_ttft_ms if llm_ttft_ms else 0
        total_ms = max(_elapsed_ms(total_started), fast_tool_pass.retrieval_ms + llm_ms)

        if persist:
            self._schedule_assistant_message_write(
                conversation_id=conversation_id,
                request_id=request_id,
                assistant_message=answer,
                user_message_index=user_message_index,
                retrieved_document_ids=fast_tool_pass.retrieved_document_ids,
                provider=response.provider,
                model=response.model,
            )

        trace = OrchestrationTrace(
            provider=response.provider,
            model=response.model,
            mode="fast",
            retrieval_used=fast_tool_pass.retrieval_used,
            retrieved_document_ids=fast_tool_pass.retrieved_document_ids,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_usd=cost.total_usd,
            pricing_source=cost.pricing_source,
            latency_ms=response.latency_ms,
            route_reason=_fast_tool_route_reason(fast_tool_pass),
            total_ms=total_ms,
            retrieval_ms=fast_tool_pass.retrieval_ms,
            query_embedding_ms=fast_tool_pass.query_embedding_ms,
            vector_search_ms=fast_tool_pass.vector_search_ms,
            rerank_ms=fast_tool_pass.rerank_ms,
            llm_ms=llm_ms,
            ttft_ms=ttft_ms,
            llm_ttft_ms=llm_ttft_ms,
            orchestration_overhead_ms=max(total_ms - fast_tool_pass.retrieval_ms - llm_ms, 0),
            retrieval_candidate_count=fast_tool_pass.retrieval_candidate_count,
            retrieval_returned_count=fast_tool_pass.retrieval_returned_count,
            retrieval_reranked=fast_tool_pass.retrieval_reranked,
            retrieval_error=fast_tool_pass.tool_error,
        )

        return OrchestratorResult(answer=answer, trace=trace, retrieved_context=[])

    async def _stream_answer_with_fast_tools(
        self,
        *,
        user_message: str,
        conversation_id: str,
        conversation_messages: list[Message] | None,
        raw_user_message: str | None,
        request_id: str | None,
        user_message_index: int | None,
        include_metrics: bool,
        persist: bool,
    ) -> AsyncIterator[str | dict[str, Any]]:
        total_started = time.perf_counter()
        if persist:
            self._schedule_user_message_write(
                conversation_id=conversation_id,
                request_id=request_id,
                raw_user_message=raw_user_message or user_message,
                user_message_index=user_message_index,
            )

        base_messages = self._request_messages(
            user_message=user_message,
            conversation_messages=conversation_messages,
            retrieved_context=[],
        )
        planning_messages = self._request_messages(
            user_message=raw_user_message or user_message,
            conversation_messages=_with_latest_user_message(
                conversation_messages,
                raw_user_message,
            ),
            retrieved_context=[],
        )
        fast_tool_pass = await self._fast_tool_pass(
            base_messages,
            planning_messages=planning_messages,
        )
        planning_usage = (
            fast_tool_pass.planning_response.usage
            if fast_tool_pass.planning_response
            else TokenUsage()
        )
        planning_cost = (
            fast_tool_pass.planning_response.cost
            if fast_tool_pass.planning_response
            else CostBreakdown()
        )

        chunks = []
        usage = planning_usage
        cost = planning_cost
        llm_started = time.perf_counter()
        llm_ttft_ms = 0
        ttft_ms = 0
        async for event in self.provider.stream_complete(
            LLMRequest(
                messages=fast_tool_pass.messages,
                model=self.model,
                mode="fast",
                max_tokens=FAST_ANSWER_MAX_TOKENS,
            )
        ):
            if event.usage is not None:
                usage = _combine_usage(planning_usage, event.usage)
            if event.cost is not None:
                cost = _combine_cost(planning_cost, event.cost)
            if not event.content:
                continue
            if not llm_ttft_ms:
                llm_ttft_ms = _elapsed_ms(llm_started)
                ttft_ms = _elapsed_ms(total_started)
            chunks.append(event.content)
            yield event.content

        final_llm_ms = _elapsed_ms(llm_started)
        final_answer = "".join(chunks)
        if persist:
            self._schedule_assistant_message_write(
                conversation_id=conversation_id,
                request_id=request_id,
                assistant_message=final_answer,
                user_message_index=user_message_index,
                retrieved_document_ids=fast_tool_pass.retrieved_document_ids,
                provider=self.provider.provider_name,
                model=self.model,
            )

        llm_ms = _fast_llm_ms(fast_tool_pass, final_llm_ms)
        total_ms = max(_elapsed_ms(total_started), fast_tool_pass.retrieval_ms + llm_ms)
        trace = OrchestrationTrace(
            provider=self.provider.provider_name,
            model=self.model,
            mode="fast",
            retrieval_used=fast_tool_pass.retrieval_used,
            retrieved_document_ids=fast_tool_pass.retrieved_document_ids,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_usd=cost.total_usd,
            pricing_source=cost.pricing_source,
            latency_ms=final_llm_ms,
            route_reason=_fast_tool_route_reason(fast_tool_pass),
            total_ms=total_ms,
            retrieval_ms=fast_tool_pass.retrieval_ms,
            query_embedding_ms=fast_tool_pass.query_embedding_ms,
            vector_search_ms=fast_tool_pass.vector_search_ms,
            rerank_ms=fast_tool_pass.rerank_ms,
            llm_ms=llm_ms,
            ttft_ms=ttft_ms,
            llm_ttft_ms=llm_ttft_ms,
            orchestration_overhead_ms=max(total_ms - fast_tool_pass.retrieval_ms - llm_ms, 0),
            retrieval_candidate_count=fast_tool_pass.retrieval_candidate_count,
            retrieval_returned_count=fast_tool_pass.retrieval_returned_count,
            retrieval_reranked=fast_tool_pass.retrieval_reranked,
            retrieval_error=fast_tool_pass.tool_error,
        )
        if include_metrics:
            yield {"type": "metrics", "metrics": _metrics_payload(trace)}

    async def _fast_tool_pass(
        self,
        base_messages: list[Message],
        *,
        planning_messages: list[Message] | None = None,
    ) -> FastToolPass:
        if self.tool_router is None:
            return FastToolPass(
                messages=base_messages,
                planning_response=None,
                tool_call_count=0,
                tool_ms=0,
            )

        tool_started = time.perf_counter()
        try:
            tool_schemas = await self.tool_router.openai_schemas()
        except Exception as exc:
            logger.warning("Fast tool schema discovery failed.", exc_info=True)
            return FastToolPass(
                messages=base_messages,
                planning_response=None,
                tool_call_count=0,
                tool_ms=_elapsed_ms(tool_started),
                tool_error=f"{type(exc).__name__}: {exc}",
            )
        if not tool_schemas:
            return FastToolPass(
                messages=base_messages,
                planning_response=None,
                tool_call_count=0,
                tool_ms=_elapsed_ms(tool_started),
            )

        tool_planning_messages = _with_fast_tool_prompt(planning_messages or base_messages)
        try:
            planning_response = await self.provider.complete(
                LLMRequest(
                    messages=tool_planning_messages,
                    model=self.model,
                    mode="fast",
                    temperature=0.0,
                    max_tokens=FAST_TOOL_SELECTION_MAX_TOKENS,
                    tools=tool_schemas,
                    metadata={"fast_tool_selection": True},
                )
            )
        except Exception as exc:
            logger.warning("Fast tool selection failed; streaming without tools.", exc_info=True)
            return FastToolPass(
                messages=base_messages,
                planning_response=None,
                tool_call_count=0,
                tool_ms=_elapsed_ms(tool_started),
                tool_error=f"{type(exc).__name__}: {exc}",
            )

        selected_tool_calls = planning_response.tool_calls[: self.fast_max_tool_calls]
        if not selected_tool_calls:
            return FastToolPass(
                messages=base_messages,
                planning_response=planning_response,
                tool_call_count=0,
                tool_ms=_elapsed_ms(tool_started),
            )

        final_messages = [
            *base_messages,
            _assistant_tool_message(planning_response.content, selected_tool_calls),
        ]
        tool_call_count = 0
        tool_error = None
        retrieval_used = False
        retrieved_document_ids: list[str] = []
        retrieval_ms = 0
        query_embedding_ms = 0
        vector_search_ms = 0
        rerank_ms = 0
        retrieval_candidate_count = 0
        retrieval_returned_count = 0
        retrieval_reranked = False
        for tool_call in selected_tool_calls:
            tool_name, arguments, call_id = _parse_tool_call(tool_call)
            if not tool_name:
                continue
            tool_call_count += 1
            tool_result = await self._call_fast_tool(
                ToolCallRequest(
                    name=tool_name,
                    arguments=arguments,
                    call_id=call_id,
                )
            )
            if tool_result.error and tool_error is None:
                tool_error = tool_result.error
            retrieval_metrics = _tool_retrieval_metrics(tool_result)
            if retrieval_metrics:
                retrieval_used = True
                retrieved_document_ids.extend(
                    document_id
                    for document_id in retrieval_metrics["document_ids"]
                    if document_id not in retrieved_document_ids
                )
                retrieval_ms += retrieval_metrics["retrieval_ms"]
                query_embedding_ms += retrieval_metrics["query_embedding_ms"]
                vector_search_ms += retrieval_metrics["vector_search_ms"]
                rerank_ms += retrieval_metrics["rerank_ms"]
                retrieval_candidate_count += retrieval_metrics["candidate_count"]
                retrieval_returned_count += retrieval_metrics["returned_count"]
                retrieval_reranked = retrieval_reranked or retrieval_metrics["reranked"]
            final_messages.append(_tool_result_message(call_id, tool_result))

        return FastToolPass(
            messages=final_messages,
            planning_response=planning_response,
            tool_call_count=tool_call_count,
            tool_ms=_elapsed_ms(tool_started),
            tool_error=tool_error,
            retrieval_used=retrieval_used,
            retrieved_document_ids=retrieved_document_ids,
            retrieval_ms=retrieval_ms,
            query_embedding_ms=query_embedding_ms,
            vector_search_ms=vector_search_ms,
            rerank_ms=rerank_ms,
            retrieval_candidate_count=retrieval_candidate_count,
            retrieval_returned_count=retrieval_returned_count,
            retrieval_reranked=retrieval_reranked,
        )

    async def _call_fast_tool(self, request: ToolCallRequest) -> ToolCallResult:
        try:
            if self.tool_router is None:
                raise ValueError("Fast tools are not configured.")
            return await asyncio.wait_for(
                self.tool_router.call(request),
                timeout=self.fast_tool_timeout_seconds,
            )
        except Exception as exc:
            logger.warning("Fast tool call %s failed.", request.name, exc_info=True)
            return ToolCallResult(
                name=request.name,
                output={"error": f"{type(exc).__name__}: {exc}"},
                error=f"{type(exc).__name__}: {exc}",
            )

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


def _with_latest_user_message(
    messages: list[Message] | None,
    latest_user_message: str | None,
) -> list[Message] | None:
    if messages is None or latest_user_message is None:
        return messages

    updated = [dict(message) for message in messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "user":
            updated[index]["content"] = latest_user_message
            break
    return updated


def _with_fast_tool_prompt(messages: list[Message]) -> list[Message]:
    if not messages:
        return [{"role": "system", "content": FAST_TOOL_SELECTION_PROMPT}]
    return [
        messages[0],
        {"role": "system", "content": FAST_TOOL_SELECTION_PROMPT},
        *messages[1:],
    ]


def _assistant_tool_message(content: str, tool_calls: list[dict[str, Any]]) -> Message:
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": tool_calls,
    }


def _tool_result_message(call_id: str | None, result: ToolCallResult) -> Message:
    message: Message = {
        "role": "tool",
        "content": json.dumps(result.output, ensure_ascii=False),
    }
    if call_id:
        message["tool_call_id"] = call_id
    return message


def _parse_tool_call(tool_call: dict[str, Any]) -> tuple[str | None, dict[str, Any], str | None]:
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw_arguments = function.get("arguments") or "{}"
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {}
    return name, arguments, tool_call.get("id")


def _combine_usage(first: TokenUsage, second: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        total_tokens=first.normalized_total() + second.normalized_total(),
        cached_input_tokens=first.cached_input_tokens + second.cached_input_tokens,
        cache_write_tokens=first.cache_write_tokens + second.cache_write_tokens,
        reasoning_tokens=first.reasoning_tokens + second.reasoning_tokens,
    )


def _combine_cost(first: CostBreakdown, second: CostBreakdown) -> CostBreakdown:
    pricing_source = first.pricing_source
    if second.pricing_source != first.pricing_source:
        pricing_source = (
            second.pricing_source
            if first.pricing_source == "unknown"
            else f"{first.pricing_source}+{second.pricing_source}"
        )
    return CostBreakdown(
        input_usd=first.input_usd + second.input_usd,
        output_usd=first.output_usd + second.output_usd,
        cache_read_usd=first.cache_read_usd + second.cache_read_usd,
        cache_write_usd=first.cache_write_usd + second.cache_write_usd,
        reasoning_usd=first.reasoning_usd + second.reasoning_usd,
        total_usd=first.total_usd + second.total_usd,
        pricing_source=pricing_source,
        estimated=first.estimated or second.estimated,
        unavailable_reason=first.unavailable_reason or second.unavailable_reason,
    )


def _fast_tool_route_reason(fast_tool_pass: FastToolPass) -> str:
    if fast_tool_pass.tool_call_count:
        return "Fast mode used one bounded tool pass, then streamed the final answer."
    if fast_tool_pass.tool_error:
        return "Fast mode skipped tool use after tool setup failed and streamed the final answer."
    return "Fast mode checked available tools, skipped tool use, and streamed the final answer."


def _fast_llm_ms(fast_tool_pass: FastToolPass, final_llm_ms: int) -> int:
    planning_ms = (
        fast_tool_pass.planning_response.latency_ms
        if fast_tool_pass.planning_response is not None
        else 0
    )
    return planning_ms + final_llm_ms


def _tool_retrieval_metrics(result: ToolCallResult) -> dict[str, Any] | None:
    retrieval = result.metadata.get("retrieval")
    if not isinstance(retrieval, dict) or not retrieval.get("used"):
        return None
    document_ids = retrieval.get("document_ids")
    if not isinstance(document_ids, list):
        document_ids = []
    return {
        "document_ids": [str(document_id) for document_id in document_ids],
        "retrieval_ms": _int_metric(retrieval.get("retrieval_ms")),
        "query_embedding_ms": _int_metric(retrieval.get("query_embedding_ms")),
        "vector_search_ms": _int_metric(retrieval.get("vector_search_ms")),
        "rerank_ms": _int_metric(retrieval.get("rerank_ms")),
        "candidate_count": _int_metric(retrieval.get("candidate_count")),
        "returned_count": _int_metric(retrieval.get("returned_count")),
        "reranked": bool(retrieval.get("reranked")),
    }


def _int_metric(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


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
