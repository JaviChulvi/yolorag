from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from yolorag.core.conversation import (
    ConversationTurn,
    InMemoryConversationStore,
)
from yolorag.core.routing import SimpleRoutePlanner
from yolorag.core.tracing import OrchestrationTrace
from yolorag.providers.base import LLMProvider, LLMRequest, ResponseMode
from yolorag.retrieval.base import RetrievalResult, Retriever


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
        conversation_store: InMemoryConversationStore | None = None,
        route_planner: SimpleRoutePlanner | None = None,
        force_retrieval: bool = False,
        retrieval_top_k: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.retriever = retriever
        self.conversation_store = conversation_store or InMemoryConversationStore()
        self.route_planner = route_planner or SimpleRoutePlanner()
        self.force_retrieval = force_retrieval
        self.retrieval_top_k = retrieval_top_k

    async def answer(
        self,
        user_message: str,
        conversation_id: str = "default",
        mode: ResponseMode = "fast",
    ) -> OrchestratorResult:
        total_started = time.perf_counter()
        state = self.conversation_store.get(conversation_id)
        plan = self.route_planner.plan(user_message=user_message, requested_mode=mode)
        retrieval_started = time.perf_counter()
        retrieved_context = await self._retrieve_if_needed(
            user_message=user_message,
            existing_document_ids=set() if self.force_retrieval else state.retrieved_document_ids,
            should_retrieve=plan.should_retrieve or self.force_retrieval,
            top_k=self.retrieval_top_k or plan.top_k,
        )
        retrieval_ms = _elapsed_ms(retrieval_started)

        messages = [
            {
                "role": "system",
                "content": MAIN_SYSTEM_PROMPT,
            },
            *state.recent_messages(),
        ]
        if retrieved_context:
            messages.append(
                {
                    "role": "system",
                    "content": self._format_context(retrieved_context),
                }
            )
        messages.append({"role": "user", "content": user_message})

        response = await self.provider.complete(
            LLMRequest(
                messages=messages,
                model=self.model,
                mode=plan.mode,
                max_tokens=512 if plan.mode == "fast" else 1600,
            )
        )

        answer = self._append_sources(response.content, retrieved_context)
        retrieved_ids = [item.document.id for item in retrieved_context]
        retrieval_trace = retrieved_context[0].trace if retrieved_context else None
        llm_ms = response.latency_ms
        total_ms = max(_elapsed_ms(total_started), retrieval_ms + llm_ms)
        orchestration_overhead_ms = max(total_ms - retrieval_ms - llm_ms, 0)
        state.add_turn(
            ConversationTurn(
                user_message=user_message,
                assistant_message=answer,
                retrieved_document_ids=retrieved_ids,
            )
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
            route_reason=(
                f"{plan.reason} Retrieval was forced by runtime configuration."
                if self.force_retrieval
                else plan.reason
            ),
            total_ms=total_ms,
            retrieval_ms=retrieval_ms,
            vector_search_ms=retrieval_trace.vector_search_ms if retrieval_trace else 0,
            rerank_ms=retrieval_trace.rerank_ms if retrieval_trace else 0,
            llm_ms=llm_ms,
            orchestration_overhead_ms=orchestration_overhead_ms,
            retrieval_candidate_count=retrieval_trace.candidate_count if retrieval_trace else 0,
            retrieval_returned_count=retrieval_trace.returned_count if retrieval_trace else 0,
            retrieval_reranked=retrieval_trace.reranked if retrieval_trace else False,
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
    ) -> AsyncIterator[str]:
        state = self.conversation_store.get(conversation_id)
        plan = self.route_planner.plan(user_message=user_message, requested_mode=mode)
        retrieved_context = await self._retrieve_if_needed(
            user_message=user_message,
            existing_document_ids=set() if self.force_retrieval else state.retrieved_document_ids,
            should_retrieve=plan.should_retrieve or self.force_retrieval,
            top_k=self.retrieval_top_k or plan.top_k,
        )

        messages = [
            {
                "role": "system",
                "content": MAIN_SYSTEM_PROMPT,
            },
            *state.recent_messages(),
        ]
        if retrieved_context:
            messages.append(
                {
                    "role": "system",
                    "content": self._format_context(retrieved_context),
                }
            )
        messages.append({"role": "user", "content": user_message})

        chunks = []
        async for event in self.provider.stream_complete(
            LLMRequest(
                messages=messages,
                model=self.model,
                mode=plan.mode,
                max_tokens=512 if plan.mode == "fast" else 1600,
            )
        ):
            if not event.content:
                continue
            chunks.append(event.content)
            yield event.content

        answer = "".join(chunks)
        source_suffix = self._source_suffix(answer, retrieved_context)
        if source_suffix:
            yield source_suffix
        final_answer = f"{answer.rstrip()}{source_suffix}" if source_suffix else answer
        state.add_turn(
            ConversationTurn(
                user_message=user_message,
                assistant_message=final_answer,
                retrieved_document_ids=[item.document.id for item in retrieved_context],
            )
        )

    async def _retrieve_if_needed(
        self,
        user_message: str,
        existing_document_ids: set[str],
        should_retrieve: bool,
        top_k: int,
    ) -> list[RetrievalResult]:
        if not should_retrieve or self.retriever is None:
            return []

        results = await self.retriever.retrieve(user_message, top_k=top_k)
        return [
            result
            for result in results
            if result.document.id not in existing_document_ids
        ]

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
